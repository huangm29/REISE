from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset, InMemoryDataset
from typing import Callable, List
from tqdm import tqdm

import pickle
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class XASNetData(InMemoryDataset):
    """Pytorch Geometric interface for XASNET datasets with XAS spectra.

    .. code-block:: python
        from src.dataset import XASNetData

        dataset = XASNetData(
            datapath="./datasets/",
            dataset_name="dataset_train_tiny",
            split="train",
        )
        sample = dataset[0]

    Args:
        datapath (str): path to existing dataset directory.
        dataset_name (str): filename of existing pickle file from datapath directory.
        split (str): type of split, must be one of ['train', 'test', 'predict'].
        xas_feature_dim (int): dimension of XAS spectral features (default: 200).
        transform (Callable): callable data transform, called on every access to element.
        pre_transform (Callable): callable data transform, called on every element during process.
    """
    pkl_suffix = ".pkl"
    
    # Mapping from element symbols to atomic numbers
    ELEMENT_TO_Z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36
    }
    
    @property
    def raw_file_names(self) -> List[str]:
        return [self.dataset_name + '_' + self.split + self.pkl_suffix]

    @property
    def processed_file_names(self) -> str:
        return f"{self.dataset_name}_{self.split}.pt"

    def __init__(
        self,
        datapath: str = "database",
        dataset_name: str = "dataset_train_tiny",
        split: str = "train",
        xas_feature_dim: int = 200,
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.dataset_name = dataset_name
        self.datapath = datapath
        self.split = split
        self.xas_feature_dim = xas_feature_dim
        
        
        self.data_all, self.slices_all = [], []
        self.offsets = [0]
        super(XASNetData, self).__init__(datapath, transform, pre_transform)

        for path in self.processed_paths:
            data, slices = torch.load(path, weights_only=False)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1])

    def len(self) -> int:
        return sum(len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all)

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(XASNetData, self).get(idx - self.offsets[data_idx])

    def _element_to_atomic_number(self, element_symbols):
        """Convert element symbols to atomic numbers."""
        return np.array([self.ELEMENT_TO_Z.get(symbol, 0) for symbol in element_symbols])

    def process(self) -> None:
        """Process pickle file containing molecular data with XAS spectra."""
        with open(self.raw_paths[0], 'rb') as f:
            data_dict = pickle.load(f)
        
        samples = []
        print(self.raw_paths[0])
        for smiles, mol_data in tqdm(data_dict.items(), desc="Processing molecules"):
            # Extract data components
            element_symbols = mol_data[0]  # Array of element symbols
            n_atoms = len(element_symbols)
            positions = mol_data[1]  # Shape: (3, n_atoms)
            
            if len(mol_data) == 3:
                xas_spectra = mol_data[2]  # Shape: (xas_feature_dim, n_target_atoms)
                # Convert XAS spectra to tensor - already has correct shape (xas_feature_dim, n_atoms)
                xas_spectra_tensor = torch.from_numpy(xas_spectra.T).float()  # Shape: (n_atoms, xas_feature_dim)
                # Create target atom mask by checking for non-zero spectra
                # An atom has XAS data if any value in its spectrum is non-zero
                target_atom_mask = torch.any(xas_spectra_tensor != 0, dim=1)  # Shape: (n_atoms,)
                # XAS targets are already in the correct format - no padding needed
                xas_targets = xas_spectra_tensor  # Shape: (n_atoms, xas_feature_dim)
            else:
                xas_targets = torch.zeros((n_atoms, self.xas_feature_dim), dtype=torch.float)  # Shape: (n_atoms, xas_feature_dim)
                target_atom_mask = torch.zeros(n_atoms, dtype=torch.bool)  # Shape: (n_atoms,)
                for idx, ele in enumerate(element_symbols):
                    if ele in ['C', 'N', 'O']:
                        target_atom_mask[idx] = True
                
            # Convert to torch tensors
            z = torch.from_numpy(self._element_to_atomic_number(element_symbols)).long()
            positions = torch.from_numpy(positions.T).float()  # Convert to (n_atoms, 3)
            
            # Create data object
            data = Data(
                z=z,
                pos=positions,
                y=xas_targets,  # XAS spectra for each atom (zeros for non-target atoms)
                xas_mask=target_atom_mask,  # Mask indicating which atoms have XAS data
                smiles=smiles,  # Store SMILES for reference
                num_nodes=n_atoms
            )
            
            samples.append(data)

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved processed dataset: {self.processed_paths[0]} with {len(samples)} samples")


class XASNetDataModule:
    """Data module for XAS datasets with specified atom type."""
    
    def __init__(
        self,
        root: str,
        dataset_name: str,
        xas_feature_dim: int = 200,
        train_size: float = 0.9,
        val_size: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.root = root
        self.dataset_name = dataset_name
        self.xas_feature_dim = xas_feature_dim
        self.sizes = [train_size, val_size]
        self.seed = seed
        self.kwargs = kwargs
        dataloader_keys = [
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
        ]
        self.dataloader_kwargs = {}
        for key in dataloader_keys:
            val = kwargs.get(key, None)
            self.dataloader_kwargs[key] = val
            if val is not None:
                del kwargs[key]
        
        
    def setup(self, stage: str) -> None:
        if stage == "fit":
            dataset = XASNetData(
                self.root, 
                self.dataset_name, 
                "train", 
                xas_feature_dim=self.xas_feature_dim,
                **self.kwargs
            )
            dataset_train, dataset_val = random_split(
                dataset, self.sizes, generator=torch.Generator().manual_seed(self.seed)
            )
            return dataset_train, dataset_val
            
        elif stage == "test":
            dataset_test = XASNetData(
                self.root, 
                self.dataset_name, 
                "test", 
                xas_feature_dim=self.xas_feature_dim,
                **self.kwargs
            )
            return dataset_test
        
        elif stage == "predict":
            dataset_predict = XASNetData(
                self.root, 
                self.dataset_name, 
                "predict", 
                xas_feature_dim=self.xas_feature_dim,
                **self.kwargs
            )
            
            return dataset_predict
            
            
if __name__ == "__main__":
    dataset = XASNetDataModule(
    root='data/archive/FG26/xas', 
    dataset_name='fg26_all_atomic', 
    batch_size=8,  # Add batch size for convenience
    num_workers=4,  # Add num_workers for faster loading
    pin_memory=True  # Add pin_memory for GPU efficiency
)
    dataset.setup('fit')
    
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()
    
    for batch in train_dataloader:
        print(batch)
        break
    