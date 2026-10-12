import argparse
import torch
import torch.nn.functional as F
import json
import csv

from omegaconf import OmegaConf
from tqdm import tqdm
from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
from src.model.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from src.dataset.all_atom_xas_dataset import XASNetDataModule


class XASNetInferencer:
    def __init__(self, checkpoint_path, device):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        
        # load checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config = OmegaConf.create(self.checkpoint['config'])
        
        # Initialize model
        self.model = self._load_model()
        
        self._setup_output_directory()

    def _load_model(self):
        model = EquiformerV2_OC20(**self.config.model)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def _setup_output_directory(self):
        """Setup output directory in the checkpoint's parent log folder."""
        if self.checkpoint_path.parent.name == 'checkpoints':
            self.session_dir = self.checkpoint_path.parent.parent
        else:
            self.session_dir = self.checkpoint_path.parent
        
        checkpoint_name = self.checkpoint_path.stem
        self.base_output_dir = self.session_dir / 'evaluations' / checkpoint_name

    def set_output_directory(self, custom_name: Optional[str] = None):
        """Set the output directory for evaluation results."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if custom_name:
            self.output_dir = self.base_output_dir / f'{custom_name}_{timestamp}'
        else:
            self.output_dir = self.base_output_dir / f"eval_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation info
        eval_info = {
            'checkpoint_path': str(self.checkpoint_path),
            'checkpoint_epoch': self.checkpoint.get('epoch', 'unknown'),
            'checkpoint_step': self.checkpoint.get('global_step', 'unknown'),
            'best_val_acc': self.checkpoint.get('best_val_acc', 'unknown'),
            'evaluation_time': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'device': str(self.device),
            'model_config': OmegaConf.to_container(self.config)
        }
        
        with open(self.output_dir / 'evaluation_info.json', 'w') as f:
            json.dump(eval_info, f, indent=2)
        
        print(f"Evaluation results will be saved to: {self.output_dir}")
        
    def build_dataloader(self, root=None, dataset_name=None, batch_size=None) -> DataLoader:
        data_config = self.config.dataset
        if batch_size:
            data_config.batch_size = batch_size
        if root:
            data_config.root = root
        if dataset_name:
            data_config.dataset_name = dataset_name
        
        dataset_module = XASNetDataModule(
            **data_config,
            seed=self.config.seed,
        )
        dataset = dataset_module.setup('test')
        
        print(f'Loading {len(dataset)} samples.')
        dataloader = DataLoader(dataset, shuffle=False, **dataset_module.dataloader_kwargs)
        
        return dataloader
    
    @torch.no_grad()
    def infer(self, dataloader: DataLoader = None, atom_types: list = ['C', 'N', 'O'], only_infer: bool = False):
        """
        Inference function for multiple atom types
        
        Args:
            dataloader: DataLoader for inference data
            atom_types: List of atom types to process (e.g., ['C', 'N', 'O'])
            only_infer: If True, only compute predictions without targets/losses
        """
        if dataloader is None:
            dataloader = self.build_dataloader()
        csv_save_file = self.output_dir / 'infer_results.csv'
        
        # Create atomic number mapping
        atom_to_atomic_num = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17}
        target_atomic_nums = [atom_to_atomic_num[atom] for atom in atom_types]
            
        progress_bar = tqdm(
            dataloader, 
            desc=f'Inferring...',
        )
        
        # Initialize total losses for each atom type (only if not only_infer)
        if not only_infer:
            total_losses = {atom_type: 0.0 for atom_type in atom_types}
        
        with open(csv_save_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Create CSV header based on only_infer parameter
            header = ['SMILES']
            if only_infer:
                for atom_type in atom_types:
                    header.append(f'{atom_type}_predicted_spectrum')
            else:
                for atom_type in atom_types:
                    header.extend([f'{atom_type}_target_spectrum', f'{atom_type}_predicted_spectrum', f'{atom_type}_l1_loss'])
            writer.writerow(header)
            
            for idx, batch_data in enumerate(progress_bar, start=1):
                smiles = batch_data.smiles  # Shape: [num_samples]
                batch_data = batch_data.to(self.device)
                z = batch_data.z  # Atomic numbers
                xas_mask = batch_data.xas_mask # Shape: [num_atoms], boolean mask
                batch_idx = batch_data.batch
                xas_prediction = self.model(batch_data, is_training=False) # Shape: [num_atoms, xas_feature_dim]
                
                if not only_infer:
                    y = batch_data.y
                
                # Get number of molecules in this batch
                num_molecules = batch_idx.max().item() + 1
                
                # Initialize storage for each atom type
                atom_type_results = {}
                if not only_infer:
                    batch_losses = {atom_type: 0.0 for atom_type in atom_types}
                
                for atom_type, atomic_num in zip(atom_types, target_atomic_nums):
                    # Create mask for current atom type with XAS data
                    atom_type_mask = (z == atomic_num) & xas_mask
                    
                    if atom_type_mask.any():
                        # Get predictions for this atom type
                        atom_prediction = xas_prediction[atom_type_mask]  # Shape: [num_atom_type_atoms, xas_feature_dim]
                        atom_batch_idx = batch_idx[atom_type_mask]  # Shape: [num_atom_type_atoms]
                        
                        # Aggregate XAS spectra by molecule (sum over atoms of this type in each molecule)
                        molecular_prediction = scatter(atom_prediction, atom_batch_idx, dim=0, 
                                                    dim_size=num_molecules, reduce='sum')
                        
                        if not only_infer:
                            # Get targets and compute loss only when not in only_infer mode
                            atom_target = y[atom_type_mask]  # Shape: [num_atom_type_atoms, xas_feature_dim]
                            molecular_target = scatter(atom_target, atom_batch_idx, dim=0, 
                                                    dim_size=num_molecules, reduce='sum')
                            
                            # Calculate loss for this atom type
                            atom_l1_loss = F.l1_loss(molecular_prediction, molecular_target)
                            total_losses[atom_type] += atom_l1_loss
                            batch_losses[atom_type] = atom_l1_loss.item()
                            
                            # Calculate per-molecule L1 loss
                            per_molecule_loss = F.l1_loss(molecular_prediction, molecular_target, reduction='none').mean(dim=1)
                            
                            # Store results with targets and losses
                            atom_type_results[atom_type] = {
                                'target': molecular_target.cpu().numpy(),
                                'prediction': molecular_prediction.cpu().numpy(),
                                'loss': per_molecule_loss.cpu().numpy()
                            }
                        else:
                            # Store only predictions
                            atom_type_results[atom_type] = {
                                'prediction': molecular_prediction.cpu().numpy()
                            }
                    else:
                        # No atoms of this type with XAS data in the batch  
                        # Create zero arrays for consistency
                        if not only_infer:
                            zero_spectrum = torch.zeros(num_molecules, y.shape[1], device=self.device)  
                            atom_type_results[atom_type] = {  
                                'target': zero_spectrum.cpu().numpy(),  
                                'prediction': zero_spectrum.cpu().numpy(),  
                                'loss': torch.zeros(num_molecules).numpy()  
                            }
                        else:
                            # For only_infer mode, we need to determine spectrum size from xas_prediction
                            spectrum_size = xas_prediction.shape[1] if len(xas_prediction) > 0 else 0
                            zero_spectrum = torch.zeros(num_molecules, spectrum_size, device=self.device)
                            atom_type_results[atom_type] = {
                                'prediction': zero_spectrum.cpu().numpy()
                            }
                
                # Write data for each molecule in the batch
                for mol_idx in range(num_molecules):
                    if mol_idx < len(smiles):  # Ensure we don't exceed SMILES list
                        row = [smiles[mol_idx]]
                        
                        # Add data for each atom type
                        for atom_type in atom_types:
                            results = atom_type_results[atom_type]
                            if only_infer:
                                # Only add predicted spectrum
                                if 'prediction' in results and results['prediction'] is not None:
                                    row.append(json.dumps(results['prediction'][mol_idx].tolist()))
                                else:
                                    row.append('')
                            else:
                                # Add target, prediction, and loss
                                if results['target'] is not None:
                                    row.extend([
                                        json.dumps(results['target'][mol_idx].tolist()),  # Target spectrum as JSON
                                        json.dumps(results['prediction'][mol_idx].tolist()),  # Predicted spectrum as JSON
                                        results['loss'][mol_idx].item()  # Loss for this atom type
                                    ])
                                else:
                                    # No atoms of this type - leave cells empty
                                    row.extend(['', '', ''])
                        
                        writer.writerow(row)
                    else:
                        print(f"Warning: Molecule index {mol_idx} exceeds SMILES length {len(smiles)}. Skipping.")
                
                # Update progress bar
                if only_infer:
                    progress_bar.set_postfix({'status': 'inferring'})
                else:
                    # Update progress bar with losses for all atom types
                    loss_str = ', '.join([f'{atom}: {loss:.4f}' for atom, loss in batch_losses.items()])
                    progress_bar.set_postfix_str(loss_str)
                
        if not only_infer:
            # Print final average losses
            print("\nFinal average losses:")
            for atom_type in atom_types:
                avg_loss = total_losses[atom_type].item() / idx
                print(f'{atom_type}: {avg_loss:.6f}')
            
            overall_avg_loss = sum(total_losses.values()).item() / (len(atom_types) * idx)
            print(f'Overall average loss: {overall_avg_loss:.6f}')      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='logs/USPTO-all_atoms_2025-07-11_15-40/checkpoints/best_model.pt')
    parser.add_argument('--output_name', type=str, default='debug')
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for inference")
    
    parser.add_argument('--root', type=str, default='data/archive/dataset/fluor_pred', help="Root directory for dataset")
    parser.add_argument('--dataset_name', type=str, default='nature_ir_coordinates', help="Name of dataset to use")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for inference")
    
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--only_infer', action='store_true', help='If set, only compute predictions without targets/losses')
    args = parser.parse_args()
    
    xasnet_inferencer = XASNetInferencer(args.checkpoint, args.device)
    xasnet_inferencer.set_output_directory(args.output_name)
    
    dataloader = xasnet_inferencer.build_dataloader(
        root=args.root,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size
    )
    
    xasnet_inferencer.infer(dataloader, atom_types=['C', 'N', 'O'], only_infer=args.only_infer)
    
    