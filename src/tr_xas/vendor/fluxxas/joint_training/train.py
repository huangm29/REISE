"""
Joint training: atomic-level + molecular-total XAS.
Uses original EquiformerV2 model; computes dual loss in training loop.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, _LRScheduler

from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_training.dataset import JointXASData
from src.model.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

logger = get_logger(__name__)


class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            lr_factor = self._step_count / self.warmup_steps
        else:
            progress = (self._step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * lr_factor for base_lr in self.base_lrs]


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode="min"):
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
        improved = score < (self.best_score - self.min_delta) if self.mode == "min" else score > (self.best_score + self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class TrainingMonitor:
    def __init__(self, results_dir: Path, save_frequency: int = 1):
        self.results_dir = results_dir
        self.save_frequency = save_frequency
        self.history = {
            'train_loss': [], 'train_atom_loss': [], 'train_mol_loss': [], 'train_fft_loss': [],
            'val_loss': [], 'val_atom_loss': [], 'val_mol_loss': [], 'val_fft_loss': [],
            'learning_rate': [], 'epochs': []
        }

    def update(self, epoch, train_metrics, val_metrics=None, lr=None):
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['train_atom_loss'].append(train_metrics.get('atom_loss', 0))
        self.history['train_mol_loss'].append(train_metrics.get('mol_loss', 0))
        self.history['train_fft_loss'].append(train_metrics.get('fft_loss', 0))
        if val_metrics:
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['val_atom_loss'].append(val_metrics.get('atom_loss', 0))
            self.history['val_mol_loss'].append(val_metrics.get('mol_loss', 0))
            self.history['val_fft_loss'].append(val_metrics.get('fft_loss', 0))
        if lr:
            self.history['learning_rate'].append(lr)
        if epoch % self.save_frequency == 0:
            self.save_history()
            self.plot_curves()

    def save_history(self):
        with open(self.results_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def plot_curves(self):
        if len(self.history['train_loss']) == 0:
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = self.history['epochs']
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', marker='o', markersize=3)
        if self.history['val_loss']:
            val_e = epochs[:len(self.history['val_loss'])]
            axes[0, 0].plot(val_e, self.history['val_loss'], label='Val', marker='s', markersize=3)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(epochs, self.history['train_atom_loss'], label='Train')
        if self.history['val_atom_loss']:
            axes[0, 1].plot(val_e, self.history['val_atom_loss'], label='Val')
        axes[0, 1].set_title('Atomic Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(epochs, self.history['train_mol_loss'], label='Train')
        if self.history['val_mol_loss']:
            axes[1, 0].plot(val_e, self.history['val_mol_loss'], label='Val')
        axes[1, 0].set_title('Molecular Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        if self.history['learning_rate']:
            axes[1, 1].plot(self.history['learning_rate'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()


class JointTrainer:
    def __init__(self, config: OmegaConf):
        self.config = config
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.session_name = f"{config.name}_{self.timestamp}"
        self.lambda_mol = config.get('lambda_mol', 0.5)
        self.lambda_fft = config.get('lambda_fft', 0.0)

        base_log_dir = Path(config.base_log_dir)
        project_dir = base_log_dir / self.session_name
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="tensorboard",
            project_dir=str(project_dir),
            kwargs_handlers=[ddp_kwargs]
        )
        set_seed(config.seed)
        self.setup_directories()
        self.setup_data()
        self.setup_model()
        self.setup_optimization()
        self.training_monitor = TrainingMonitor(self.results_dir, save_frequency=config.eval_every)
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )
        self.load_checkpoint_if_needed()

    def load_checkpoint_if_needed(self):
        """Load checkpoint for resume or fine-tuning."""
        ckpt_path = self.config.get('load_checkpoint_path')
        if not ckpt_path or not Path(ckpt_path).exists():
            return
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Always load model weights (unwrap first since checkpoint was saved unwrapped)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])

        if self.config.get('finetune', False):
            # Fine-tune mode: use pretrained model weights with a fresh training loop.
            # Optimizer and scheduler were already created from scratch in setup_optimization.
            if self.accelerator.is_main_process:
                self.accelerator.print(
                    f"Fine-tuning from checkpoint epoch {checkpoint.get('epoch', 'unknown')}. "
                    f"Optimizer, scheduler, and training state are reset."
                )
            return

        # Resume mode: restore full training state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Recreate scheduler for remaining epochs (do NOT load old scheduler state)
        # Old scheduler state has last_epoch=max_steps from previous run,
        # which would freeze LR at min_lr.
        resumed_epoch = checkpoint.get('epoch', 0)
        remaining_epochs = self.config.num_epochs - resumed_epoch
        steps_per_epoch = self.num_update_steps_per_epoch
        total_steps = steps_per_epoch * self.config.num_epochs
        remaining_steps = steps_per_epoch * remaining_epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs

        if self.config.scheduler_type == "cosine_warmup":
            self.scheduler = CosineWarmupScheduler(self.optimizer, warmup_steps, total_steps, min_lr=self.config.min_lr)
            # Fast-forward to current step so LR resumes correctly
            for _ in range(checkpoint.get('global_step', 0)):
                self.scheduler.step()
        elif self.config.scheduler_type == "onecycle":
            # OneCycle: total_steps must cover FULL run, but we fast-forward
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.config.optimizer.learning_rate,
                                        total_steps=total_steps,
                                        pct_start=self.config.warmup_epochs / self.config.num_epochs,
                                        anneal_strategy='cos', final_div_factor=1e4)
            for _ in range(checkpoint.get('global_step', 0)):
                self.scheduler.step()
        # For other schedulers, keep the freshly-created one

        # Restore training state
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if checkpoint.get('training_history'):
            self.training_monitor.history = checkpoint['training_history']

        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"Resumed from epoch {self.start_epoch - 1}, "
                f"global_step={self.global_step}, best_loss={self.best_val_loss:.2e}"
            )

    def setup_directories(self):
        base_log_dir = Path(self.config.base_log_dir)
        self.output_dir = base_log_dir / self.session_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        self.results_dir = self.output_dir / "results"
        if self.accelerator.is_main_process:
            for d in [self.output_dir, self.checkpoint_dir, self.tensorboard_dir, self.results_dir]:
                d.mkdir(parents=True, exist_ok=True)
            config_dict = OmegaConf.to_container(self.config)
            config_dict['session_name'] = self.session_name
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2, default=str)
            self.accelerator.print(f"Session: {self.session_name}")
            self.accelerator.print(f"Output: {self.output_dir}")

    def setup_data(self):
        cfg = self.config.dataset
        dataset_cls = JointXASData
        dataset_kwargs = dict(
            root=cfg.root,
            total_spec_path=cfg.total_spec_path,
            max_samples=cfg.get('max_samples', None),
            train_ratio=cfg.get('train_ratio', 0.9),
            seed=self.config.seed,
        )
        # Multi-element training
        if cfg.get('multi_element', False):
            from joint_training.dataset import MultiElementXASData
            dataset_cls = MultiElementXASData
            dataset_kwargs['atomic_spec_paths'] = cfg.atomic_spec_paths
        else:
            dataset_kwargs['atomic_spec_path'] = cfg.atomic_spec_path
            dataset_kwargs['atom_type'] = cfg.atom_type

        train_dataset = dataset_cls(split="train", **dataset_kwargs)
        val_dataset = dataset_cls(split="val", **dataset_kwargs)
        dl_kwargs = {
            'batch_size': cfg.batch_size,
            'num_workers': cfg.get('num_workers', 0),
            'pin_memory': cfg.get('pin_memory', False),
            'persistent_workers': cfg.get('persistent_workers', False),
        }
        self.train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
        self.val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)
        self.accelerator.print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    def setup_model(self):
        self.model = EquiformerV2_OC20(**self.config.model)
        if self.accelerator.is_main_process:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.accelerator.print(f"Params: {total:,} total, {trainable:,} trainable")

    def setup_optimization(self):
        self.optimizer = AdamW(params=self.model.parameters(), **self.config.optimizer)
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        self.accelerator.print(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}, warmup: {warmup_steps}")

        if self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=total_steps * 2, T_mult=2, eta_min=self.config.min_lr)
        elif self.config.scheduler_type == "cosine_warmup":
            self.scheduler = CosineWarmupScheduler(self.optimizer, warmup_steps, total_steps, min_lr=self.config.min_lr)
        elif self.config.scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.config.optimizer.learning_rate, total_steps=total_steps,
                                        pct_start=self.config.warmup_epochs / self.config.num_epochs, anneal_strategy='cos', final_div_factor=1e4)
        else:
            self.scheduler = None
        self.num_update_steps_per_epoch = steps_per_epoch
        self.num_training_steps = total_steps

    def fft_loss(self, pred, target):
        """Frequency-domain loss with exponential high-frequency weighting.
        
        Uses SUM reduction (not mean) so that FFT loss is naturally on the same
        scale as MSE loss (~1e9), making lambda_fft physically meaningful.
        Exponential weighting aggressively penalizes high-frequency errors
        (sharp peaks) while leaving low-frequency envelope mostly to MSE.
        """
        if pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device)
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        freqs = torch.arange(pred_fft.shape[-1], device=pred.device).float()
        # Exponential weight: freq=0 -> 1, freq=100 -> exp(4) ~ 55
        weights = torch.exp(freqs / 25.0)
        diff_real = (pred_fft.real - target_fft.real) * weights
        diff_imag = (pred_fft.imag - target_fft.imag) * weights
        # SUM reduction to match MSE scale (was MEAN, causing 3-orders gap)
        loss = (diff_real.pow(2).sum() + diff_imag.pow(2).sum()) / 2
        return loss

    def compute_loss(self, batch):
        """Compute dual loss: atomic + molecular + FFT."""
        xas_predictions = self.model(batch, is_training=False)

        # Atomic-level loss
        xas_mask = batch.xas_mask
        if xas_mask.sum() > 0:
            masked_pred = xas_predictions[xas_mask]
            masked_target = batch.y[xas_mask]
            loss_atom = F.mse_loss(masked_pred, masked_target, reduction='sum')
        else:
            loss_atom = torch.tensor(0.0, device=batch.pos.device)

        # Molecular-level loss: sum only target-element atoms
        mol_pred = scatter(xas_predictions[batch.xas_mask], batch.batch[batch.xas_mask], dim=0, reduce='sum')
        loss_mol = F.mse_loss(mol_pred, batch.y_total, reduction='sum')

        # FFT loss (on atomic-level masked predictions)
        loss_fft = torch.tensor(0.0, device=batch.pos.device)
        if self.lambda_fft > 0 and xas_mask.sum() > 0:
            loss_fft = self.fft_loss(masked_pred, masked_target)

        # Combined
        loss = loss_atom + self.lambda_mol * loss_mol + self.lambda_fft * loss_fft
        return loss, {
            'total_loss': loss.item(),
            'atom_loss': loss_atom.item(),
            'mol_loss': loss_mol.item(),
            'fft_loss': loss_fft.item(),
        }

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses, epoch_atom, epoch_mol, epoch_fft = [], [], [], []
        progress_bar = tqdm(self.train_loader, desc=f'Train {epoch}/{self.config.num_epochs}',
                            disable=not self.accelerator.is_local_main_process)
        for batch_idx, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                loss, comps = self.compute_loss(batch)
                self.accelerator.backward(loss)
                if self.config.gradient_clip_val > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                epoch_losses.append(comps['total_loss'])
                epoch_atom.append(comps['atom_loss'])
                epoch_mol.append(comps['mol_loss'])
                epoch_fft.append(comps.get('fft_loss', 0))
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        'total': f"{comps['total_loss']:.2e}",
                        'atom': f"{comps['atom_loss']:.2e}",
                        'fft': f"{comps.get('fft_loss', 0):.2e}",
                    })
                if batch_idx % self.config.log_every == 0 and self.accelerator.is_main_process:
                    self.accelerator.log({
                        "train/loss": comps['total_loss'],
                        "train/atom_loss": comps['atom_loss'],
                        "train/mol_loss": comps['mol_loss'],
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                    }, step=self.global_step)
                    self.training_monitor.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        return {
            'loss': np.mean(epoch_losses),
            'atom_loss': np.mean(epoch_atom),
            'mol_loss': np.mean(epoch_mol),
            'fft_loss': np.mean(epoch_fft),
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_losses, val_atom, val_mol, val_fft = [], [], [], []
        progress_bar = tqdm(self.val_loader, desc='Validation',
                            disable=not self.accelerator.is_local_main_process)
        for batch in progress_bar:
            loss, comps = self.compute_loss(batch)
            val_losses.append(comps['total_loss'])
            val_atom.append(comps['atom_loss'])
            val_mol.append(comps['mol_loss'])
            val_fft.append(comps.get('fft_loss', 0))
        return {
            'loss': np.mean(val_losses),
            'atom_loss': np.mean(val_atom),
            'mol_loss': np.mean(val_mol),
            'fft_loss': np.mean(val_fft),
        }

    def save_checkpoint(self, epoch, is_best=False):
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
            'training_history': self.training_monitor.history,
        }
        self.accelerator.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
        if is_best:
            self.accelerator.save(checkpoint, self.checkpoint_dir / "best_model.pt")
        self.cleanup_checkpoints()

    def cleanup_checkpoints(self):
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=lambda x: x.stat().st_mtime)
        if len(checkpoints) > self.config.save_total_limit:
            for ckpt in checkpoints[:-self.config.save_total_limit]:
                ckpt.unlink()

    def train(self):
        self.accelerator.print("Starting joint training...")
        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = None
            if epoch % self.config.eval_every == 0:
                val_metrics = self.validate()
                self.training_monitor.update(epoch, train_metrics, val_metrics,
                                             lr=self.optimizer.param_groups[0]['lr'])
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                if epoch % self.config.save_every == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
                if self.early_stopping(val_metrics['loss']):
                    self.accelerator.print(f"Early stopping at epoch {epoch}")
                    break
                self.accelerator.print(
                    f"Epoch {epoch}/{self.config.num_epochs} | "
                    f"Train: {train_metrics['loss']:.2e} (A:{train_metrics['atom_loss']:.2e} M:{train_metrics['mol_loss']:.2e}) | "
                    f"Val: {val_metrics['loss']:.2e} (A:{val_metrics['atom_loss']:.2e} M:{val_metrics['mol_loss']:.2e}) | "
                    f"Best: {self.best_val_loss:.2e}"
                )
            else:
                self.training_monitor.update(epoch, train_metrics, lr=self.optimizer.param_groups[0]['lr'])

        if self.accelerator.is_main_process:
            final_epoch = self.training_monitor.history['epochs'][-1] if self.training_monitor.history['epochs'] else self.start_epoch
            self.save_checkpoint(final_epoch, is_best=False)
            self.training_monitor.save_history()
            self.training_monitor.plot_curves()
        self.accelerator.end_training()
        self.accelerator.print("Training completed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default="joint_test")
    parser.add_argument("--base_log_dir", type=str, default="./logs")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lambda_mol", type=float)
    parser.add_argument("--lambda_fft", type=float)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--load_checkpoint", type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.name = args.name
    if args.base_log_dir:
        config.base_log_dir = args.base_log_dir
    if args.batch_size:
        config.dataset.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.learning_rate:
        config.optimizer.learning_rate = args.learning_rate
    if args.lambda_mol is not None:
        config.lambda_mol = args.lambda_mol
    if args.lambda_fft is not None:
        config.lambda_fft = args.lambda_fft
    if args.max_samples is not None:
        config.dataset.max_samples = args.max_samples
    if args.load_checkpoint:
        config.load_checkpoint_path = args.load_checkpoint

    trainer = JointTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
