"""
Multi-GPU training script for Molecular Spectral VAE using HuggingFace Accelerate.
Supports mixed precision, gradient accumulation, comprehensive logging, and warmup scheduling.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, _LRScheduler

from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from omegaconf import OmegaConf

# Optional integrations
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset.all_atom_xas_dataset import XASNetDataModule
from src.model.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

logger = get_logger(__name__)


class CosineWarmupScheduler(_LRScheduler):
    """Cosine learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            # Linear warmup
            lr_factor = self._step_count / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self._step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
        
        return [
            self.min_lr + (base_lr - self.min_lr) * lr_factor 
            for base_lr in self.base_lrs
        ]


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


class TrainingMonitor:
    """Handles saving training history and plots during training."""
    
    def __init__(self, results_dir: Path, save_frequency: int = 1):
        self.results_dir = results_dir
        self.save_frequency = save_frequency
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'beta_values': [],
            'epochs': []
        }
        
    def update(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, 
               lr: float = None, beta: float = None):
        """Update training history."""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if lr is not None:
            self.history['learning_rate'].append(lr)
        if beta is not None:
            self.history['beta_values'].append(beta)
            
        # Save periodically
        if epoch % self.save_frequency == 0:
            self.save_history()
            self.plot_curves()
    
    def save_history(self):
        """Save training history to JSON."""
        with open(self.results_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
    
    def plot_curves(self):
        """Plot and save training curves."""
        if len(self.history['train_loss']) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Loss curves
        epochs = self.history['epochs']
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o', markersize=3)
        if self.history['val_loss']:
            val_epochs = epochs[:len(self.history['val_loss'])]
            axes[0].plot(val_epochs, self.history['val_loss'], label='Val Loss', marker='s', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate
        if self.history['learning_rate']:
            axes[1].plot(self.history['learning_rate'])
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
        
        # Beta values
        if self.history['beta_values']:
            axes[2].plot(self.history['beta_values'])
            axes[2].set_xlabel('Step')
            axes[2].set_ylabel('Beta')
            axes[2].set_title('Beta Schedule (KL Weight)')
            axes[2].grid(True, alpha=0.3)
        
        # Loss components if available
        axes[3].text(0.1, 0.5, f"Latest Epoch: {epochs[-1]}\n"
                              f"Train Loss: {self.history['train_loss'][-1]:.4f}\n"
                              f"Val Loss: {self.history['val_loss'][-1]:.4f}" if self.history['val_loss'] else "",
                     transform=axes[3].transAxes, fontsize=12, verticalalignment='center')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()


class XASNETTrainer:
    """Trainer class for Molecular Spectral VAE."""
    
    def __init__(self, config: OmegaConf):
        self.config = config
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create timestamped session name early
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.session_name = f"{config.name}_{self.timestamp}"
        
        # Initialize accelerator with dynamic project directory
        base_log_dir = Path(config.base_log_dir)
        project_dir = base_log_dir / self.session_name
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="tensorboard",
            project_dir=str(project_dir),
            kwargs_handlers=[ddp_kwargs]
        )
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Setup directories
        self.setup_directories()
        
        # Initialize components
        self.setup_data()
        self.setup_model()
        self.setup_optimization()
        
        # Initialize training monitor
        self.training_monitor = TrainingMonitor(
            self.results_dir, 
            save_frequency=config.eval_every
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )
    
    def setup_directories(self):
        """Create necessary directories with timestamp."""
        
        # Setup directory structure
        base_log_dir = Path(self.config.base_log_dir)
        self.output_dir = base_log_dir / self.session_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        self.results_dir = self.output_dir / "results"
        
        if self.accelerator.is_main_process:
            # Create all directories
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config
            config_dict = OmegaConf.to_container(self.config)
            config_dict['session_name'] = self.session_name
            config_dict['timestamp'] = self.timestamp
            
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            # Create a session info file
            session_info = {
                'name': self.config.name,
                'timestamp': self.timestamp,
                'session_name': self.session_name,
                'start_time': datetime.now().isoformat(),
                'command': ' '.join(sys.argv),
                'output_dir': str(self.output_dir),
                'checkpoint_dir': str(self.checkpoint_dir),
                'accelerate_config': {
                    'num_processes': self.accelerator.num_processes,
                    'mixed_precision': self.config.mixed_precision,
                    'device': str(self.accelerator.device)
                }
            }
            
            with open(self.output_dir / "session_info.json", "w") as f:
                json.dump(session_info, f, indent=2)
            
            self.accelerator.print(f"Training session: {self.session_name}")
            self.accelerator.print(f"Output directory: {self.output_dir}")
    
    def setup_data(self):
        """Setup data loaders."""
        dataset = XASNetDataModule(
            **self.config.dataset,
            seed=self.config.seed
        )
        train_dataset, val_dataset = dataset.setup('fit')
        
        self.train_loader = DataLoader(train_dataset, shuffle=True, **dataset.dataloader_kwargs)
        self.val_loader = DataLoader(val_dataset, shuffle=False, **dataset.dataloader_kwargs)
        
        self.accelerator.print(
            f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}"
        )
    
    def setup_model(self):
        """Initialize model."""
        self.model = EquiformerV2_OC20(
            **self.config.model
        )
        
        # Print model summary
        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.accelerator.print(
                f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}"
            )
    
    def setup_optimization(self):
        """Setup optimizer and scheduler."""
        # Optimizer
        self.optimizer = AdamW(
            params=self.model.parameters(),
            **self.config.optimizer
        )
        
        # Prepare for distributed training first
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        
        # Calculate actual number of training steps after prepare()
        # This accounts for the distributed sampler
        num_update_steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.config.num_epochs
        num_warmup_steps = num_update_steps_per_epoch * self.config.warmup_epochs
        
        self.accelerator.print(f"Number of training steps: {num_training_steps}")
        self.accelerator.print(f"Number of warmup steps: {num_warmup_steps}")
        self.accelerator.print(f"Steps per epoch: {num_update_steps_per_epoch}")
        
        # Learning rate scheduler
        if self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=num_training_steps * 2,
                T_mult=2,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == "cosine_warmup":
            self.scheduler = CosineWarmupScheduler(
                self.optimizer,
                warmup_steps=num_warmup_steps,
                max_steps=num_training_steps,
                min_lr=self.config.min_lr
            )
        elif self.config.scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.optimizer.learning_rate,
                total_steps=num_training_steps,
                pct_start=self.config.warmup_epochs / self.config.num_epochs,
                anneal_strategy='cos',
                final_div_factor=1e4
            )
        else:
            self.scheduler = None

        # Store training steps info
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
        self.num_training_steps = num_training_steps
    
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'total_loss': [],
        }
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f'Train Epoch {epoch}/{self.config.num_epochs}',
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                loss, loss_components = self.model(batch, is_training=True)
                   
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_val
                    )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
                
                # Collect metrics
                epoch_losses.append(loss_components['total_loss'])
                for key in epoch_metrics:
                    if key in loss_components:
                        epoch_metrics[key].append(loss_components[key])
                
                # Update progress bar
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        name: f'{value:.4f}' for name, value in loss_components.items()
                    })
                
                # Log batch metrics
                if batch_idx % self.config.log_every == 0 and self.accelerator.is_main_process:
                    self.accelerator.log({
                        "train/loss": loss_components['total_loss'],
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        **{f"train/{k}": v for k, v in loss_components.items() if k != 'total_loss'}
                    }, step=self.global_step)
                    
                    # Update training monitor with learning rate and beta
                    self.training_monitor.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        # Compute epoch averages
        avg_metrics = {
            'loss': np.mean(epoch_losses),
            **{k: np.mean(v) for k, v in epoch_metrics.items() if v}
        }
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_losses = []
        val_metrics = {
            'total_loss': [],
        }
        
        progress_bar = tqdm(
            self.val_loader, 
            desc='Validation',
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch in progress_bar:
            # Forward pass
            loss, loss_components = self.model(batch, is_training=True)
            
            # Collect metrics
            val_losses.append(loss_components['total_loss'])
            for key in val_metrics:
                if key in loss_components:
                    val_metrics[key].append(loss_components[key])
        
        # Compute averages
        avg_metrics = {
            'loss': np.mean(val_losses),
            **{k: np.mean(v) for k, v in val_metrics.items() if v}
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': OmegaConf.to_container(self.config),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_monitor.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        self.accelerator.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            self.accelerator.save(checkpoint, best_path)
        
        # Manage checkpoint limit
        self.cleanup_checkpoints()
    
    def cleanup_checkpoints(self):
        """Remove old checkpoints to maintain limit."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda x: x.stat().st_mtime
        )
        
        if len(checkpoints) > self.config.save_total_limit:
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                checkpoint.unlink()
    
    
    def load_checkpoint_for_finetuning(self, checkpoint_path: str):
        """Load checkpoint for fine-tuning (model weights only)."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load only model weights
        model_state_dict = checkpoint['model_state_dict']
        self.accelerator.unwrap_model(self.model).load_state_dict(model_state_dict)
        
        # Reset training state for fine-tuning
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.is_finetuning = True
        
        # Clear training history to start fresh
        self.training_monitor.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'beta_values': [],
            'epochs': []
        }
        
        if self.accelerator.is_main_process:
            self.accelerator.print(f"Loaded model weights for fine-tuning from: {checkpoint_path}")
            self.accelerator.print("Training state reset for fine-tuning")
            
            # Log info about the loaded checkpoint
            if 'epoch' in checkpoint:
                self.accelerator.print(f"Loaded checkpoint was from epoch: {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                self.accelerator.print(f"Loaded checkpoint best val loss: {checkpoint['best_val_loss']:.4f}")
    
    
    def load_checkpoint_for_resume(self, checkpoint_path: str):
        """Load checkpoint for resuming training (full state)."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load model weights
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_monitor.history = checkpoint['training_history']
        
        if self.accelerator.is_main_process:
            self.accelerator.print(f"Resumed training from checkpoint: {checkpoint_path}")
            self.accelerator.print(f"Starting from epoch {self.start_epoch}")
            self.accelerator.print(f"Best validation loss so far: {self.best_val_loss:.4f}")
    
    def save_training_summary(self, final_epoch: int):
        """Save training summary and final metrics."""
        if not self.accelerator.is_main_process:
            return
        
        # Calculate training duration
        end_time = datetime.now()
        session_info = json.load(open(self.output_dir / "session_info.json"))
        start_time = datetime.fromisoformat(session_info['start_time'])
        duration_seconds = (end_time - start_time).total_seconds()
        
        # Prepare summary
        summary = {
            'session_name': self.session_name,
            'final_epoch': final_epoch,
            'total_epochs_trained': final_epoch - self.start_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.training_monitor.history['train_loss'][-1] if self.training_monitor.history['train_loss'] else None,
            'final_val_loss': self.training_monitor.history['val_loss'][-1] if self.training_monitor.history['val_loss'] else None,
            'training_duration_seconds': duration_seconds,
            'training_duration_human': str(timedelta(seconds=int(duration_seconds))),
            'end_time': end_time.isoformat(),
            'early_stopped': self.early_stopping.early_stop,
            'total_steps': self.global_step,
            'config': OmegaConf.to_container(self.config)
        }
        
        # Save summary
        with open(self.results_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Final save of training history and plots
        self.training_monitor.save_history()
        self.training_monitor.plot_curves()
    
    def train(self):
        """Main training loop."""
        # Handle checkpoint loading based on config
        if hasattr(self.config, 'load_checkpoint_path') and self.config.load_checkpoint_path:
            if getattr(self.config, 'finetune', False):
                self.load_checkpoint_for_finetuning(self.config.load_checkpoint_path)
                mode_str = "Fine-tuning"
            elif getattr(self.config, 'resume', False):
                self.load_checkpoint_for_resume(self.config.load_checkpoint_path)
                mode_str = "Resuming training"
            else:
                self.accelerator.print("Warning: load_checkpoint_path specified but neither finetune nor resume is True. Ignoring checkpoint.")
                mode_str = "Starting training from scratch"
        else:
            mode_str = "Starting training from scratch"
            
        self.accelerator.print(f"{mode_str}...")
        self.accelerator.print(f"Session: {self.session_name}")
        self.accelerator.print(f"Logs directory: {self.output_dir}")
        
        try:
            for epoch in range(self.start_epoch, self.config.num_epochs + 1):
                # Training
                train_metrics = self.train_epoch(epoch)
                
                # Validation
                val_metrics = None
                if epoch % self.config.eval_every == 0:
                    val_metrics = self.validate()
                    
                    # Update training monitor
                    self.training_monitor.update(
                        epoch=epoch,
                        train_loss=train_metrics['loss'],
                        val_loss=val_metrics['loss'],
                        lr=self.optimizer.param_groups[0]['lr'],
                    )
                    
                    # Log metrics
                    if self.accelerator.is_main_process:
                        self.accelerator.log({
                            "epoch": epoch,
                            **{f"train_epoch/{k}": v for k, v in train_metrics.items()},
                            **{f"val/{k}": v for k, v in val_metrics.items()}
                        }, step=self.global_step)
                    
                    # Check for improvement
                    is_best = val_metrics['loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['loss']
                    
                    # Save checkpoint
                    if epoch % self.config.save_every == 0 or is_best:
                        self.save_checkpoint(epoch, is_best)
                    
                    # Early stopping
                    if self.early_stopping(val_metrics['loss']):
                        self.accelerator.print(f"Early stopping triggered at epoch {epoch}")
                        break
                    
                    # Print progress
                    self.accelerator.print(
                        f"Epoch {epoch}/{self.config.num_epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Best Val Loss: {self.best_val_loss:.4f}"
                    )
                else:
                    # Update training monitor without validation
                    self.training_monitor.update(
                        epoch=epoch,
                        train_loss=train_metrics['loss'],
                        lr=self.optimizer.param_groups[0]['lr'],
                    )
        
        except KeyboardInterrupt:
            self.accelerator.print("\nTraining interrupted by user")
            epoch = self.training_monitor.history['epochs'][-1] if self.training_monitor.history['epochs'] else self.start_epoch
        
        except Exception as e:
            self.accelerator.print(f"\nTraining failed with error: {str(e)}")
            raise
        
        finally:
            # Save final checkpoint and summary
            if self.accelerator.is_main_process:
                final_epoch = self.training_monitor.history['epochs'][-1] if self.training_monitor.history['epochs'] else self.start_epoch
                self.save_checkpoint(final_epoch, is_best=False)
                self.save_training_summary(final_epoch)
            
            # Finish tracking
            self.accelerator.end_training()
            self.accelerator.print("Training completed!")
            self.accelerator.print(f"Results saved to: {self.output_dir}")
            # self.accelerator.wait_for_everyone()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Molecular Spectral VAE")
    parser.add_argument("--config", type=str, default='configs/equiformer_v2_oc20.yaml', help="Path to config JSON file")
    parser.add_argument("--name", type=str, required=False, default='debug', help="Name for this training session")
    parser.add_argument("--data_dir", type=str, help="Override data directory")
    parser.add_argument("--dataset_name", type=str, help="Override dataset name")
    parser.add_argument("--base_log_dir", type=str, default="./logs", help="Base directory for logs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        raise 'conifg is not exit'
    
    # Override config with command line arguments
    config.name = args.name  # Always use the provided name
    if args.base_log_dir:
        config.base_log_dir = args.base_log_dir
    if args.data_dir:
        config.dataset.root = args.data_dir
    if args.dataset_name:
        config.dataset.dataset_name = args.dataset_name
    if args.batch_size:
        config.dataset.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.learning_rate:
        config.optimizer.learning_rate = args.learning_rate

    
    # Create trainer and start training
    trainer = XASNETTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()