"""
Joint training dataset: molecular total spectrum + atomic-level spectrum.
Trains on a single element (C/N/O) at a time.

Strategy:
  - Load structures + atomic spectra from merged_{element}_spectra.pkl
  - Load molecular total spectra from merged_total_xas_spec_200.pkl
  - Every training molecule is guaranteed to have structure + atomic data
"""

import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class JointData(Data):
    """Custom Data class that treats y_total, element_type as graph-level (stack, not cat)."""
    def __inc__(self, key, value, *args, **kwargs):
        if key in ('y_total', 'has_atom_data', 'element_type'):
            return 0
        return super().__inc__(key, value, *args, **kwargs)
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('y_total', 'has_atom_data', 'element_type'):
            return None  # stack instead of cat
        return super().__cat_dim__(key, value, *args, **kwargs)
from typing import Callable
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class JointXASData(InMemoryDataset):
    """
    Each Data object contains:
      z: (n_atoms,) atomic numbers
      pos: (n_atoms, 3)
      y_total: (200,) molecular total spectrum for target element
      y: (n_atoms, 200) per-atom spectra (zero for non-target atoms)
      xas_mask: (n_atoms,) bool mask for target atoms
      has_atom_data: scalar bool, always True for this dataset
      smiles: str
      num_nodes: int
    """

    ELEMENT_TO_Z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
        'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
        'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
        'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33,
        'Se': 34, 'Br': 35, 'Kr': 36,
    }

    def __init__(
        self,
        root: str,
        total_spec_path: str,
        atomic_spec_path: str,
        atom_type: str = "C",
        split: str = "train",
        max_samples: int = None,
        train_ratio: float = 0.9,
        seed: int = 42,
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.total_spec_path = total_spec_path
        self.atomic_spec_path = atomic_spec_path
        self.atom_type = atom_type
        self.split = split
        self.max_samples = max_samples
        self.train_ratio = train_ratio
        self.seed = seed

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        tag = f"{self.atom_type}_{self.split}"
        if self.max_samples:
            tag += f"_n{self.max_samples}"
        return f"joint_data_{tag}.pt"

    def _element_to_z(self, symbols):
        return np.array([self.ELEMENT_TO_Z.get(s, 0) for s in symbols])

    def process(self):
        # Load total spectra (dict: smiles -> (200,) per element)
        logger.info(f"Loading total spectra: {self.total_spec_path}")
        with open(self.total_spec_path, 'rb') as f:
            total_pkg = pickle.load(f)
        smiles_list = total_pkg['smiles']
        total_specs = total_pkg[self.atom_type]  # (N_mols, 200)
        # Build fast lookup: smiles -> total spectrum
        total_lookup = {smi: total_specs[i] for i, smi in enumerate(smiles_list)}

        # Load atomic-level spectra (dict: smiles -> [elements, positions, xas])
        logger.info(f"Loading atomic spectra: {self.atomic_spec_path}")
        with open(self.atomic_spec_path, 'rb') as f:
            atomic_data = pickle.load(f)

        # Get molecule list from atomic data (guaranteed to have structure)
        all_smiles = list(atomic_data.keys())
        rng = np.random.RandomState(self.seed)
        rng.shuffle(all_smiles)

        n_train = int(len(all_smiles) * self.train_ratio)
        if self.split == "train":
            split_smiles = all_smiles[:n_train]
        else:
            split_smiles = all_smiles[n_train:]

        if self.max_samples and len(split_smiles) > self.max_samples:
            split_smiles = split_smiles[:self.max_samples]

        logger.info(
            f"Joint {self.atom_type} {self.split}: "
            f"{len(split_smiles)} molecules (max_samples={self.max_samples})"
        )

        samples = []
        missing_total = 0
        filtered_zero = 0

        for smiles in tqdm(split_smiles, desc=f"Joint {self.split}"):
            # Skip if total spectrum not found (shouldn't happen)
            if smiles not in total_lookup:
                missing_total += 1
                continue

            y_total = total_lookup[smiles]  # (200,)

            # Load atomic data
            mol_data = atomic_data[smiles]
            element_symbols = mol_data[0]
            positions = mol_data[1]       # (3, n_atoms)
            xas_spectra = mol_data[2]     # (200, n_target_atoms)

            n_atoms = len(element_symbols)
            z = torch.from_numpy(self._element_to_z(element_symbols)).long()
            positions = torch.from_numpy(positions.T).float()

            target_indices = [i for i, s in enumerate(element_symbols) if s == self.atom_type]
            if xas_spectra.shape[1] != len(target_indices):
                logger.warning(f"Shape mismatch for {smiles}, skipping")
                continue

            xas_mask = torch.zeros(n_atoms, dtype=torch.bool)
            xas_tensor = torch.from_numpy(xas_spectra.T).float()
            y = torch.zeros(n_atoms, 200, dtype=torch.float)

            n_valid = 0
            for xas_idx, atom_idx in enumerate(target_indices):
                atom_spec = xas_tensor[xas_idx]
                if torch.any(atom_spec == 0):
                    filtered_zero += 1
                    continue
                xas_mask[atom_idx] = True
                y[atom_idx] = atom_spec
                n_valid += 1
            
            if n_valid == 0:
                continue

            data = JointData(
                z=z,
                pos=positions,
                y_total=torch.from_numpy(y_total).float(),
                y=y,
                xas_mask=xas_mask,
                has_atom_data=torch.tensor(True, dtype=torch.bool),
                smiles=smiles,
                num_nodes=n_atoms,
            )
            samples.append(data)

        logger.info(
            f"Done: {len(samples)} samples, "
            f"missing_total: {missing_total}, filtered_zero: {filtered_zero}"
        )

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])


class MultiElementXASData(InMemoryDataset):
    """
    Multi-element joint training: C + N + O.
    Each Data object contains one target element per molecule.
    element_type: 0=C, 1=N, 2=O (graph-level int)
    """

    ELEMENT_TO_Z = JointXASData.ELEMENT_TO_Z
    ELEMENT_MAP = {'C': 0, 'N': 1, 'O': 2}

    def __init__(
        self,
        root: str,
        total_spec_path: str,
        atomic_spec_paths: dict,
        split: str = "train",
        max_samples: int = None,
        train_ratio: float = 0.9,
        seed: int = 42,
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.total_spec_path = total_spec_path
        self.atomic_spec_paths = atomic_spec_paths
        self.split = split
        self.max_samples = max_samples
        self.train_ratio = train_ratio
        self.seed = seed
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        tag = f"multi_{self.split}"
        if self.max_samples:
            tag += f"_n{self.max_samples}"
        return f"joint_data_{tag}.pt"

    def _element_to_z(self, symbols):
        return np.array([self.ELEMENT_TO_Z.get(s, 0) for s in symbols])

    def process(self):
        # Load total spectra (dict: smiles -> {C,N,O})
        logger.info(f"Loading total spectra: {self.total_spec_path}")
        with open(self.total_spec_path, 'rb') as f:
            total_pkg = pickle.load(f)
        smiles_list = total_pkg['smiles']
        total_specs = {
            'C': total_pkg['C'],
            'N': total_pkg['N'],
            'O': total_pkg['O'],
        }
        total_lookup = {smi: {e: total_specs[e][i] for e in total_specs} for i, smi in enumerate(smiles_list)}

        # Load atomic-level spectra for all elements
        atomic_data = {}
        for elem, path in self.atomic_spec_paths.items():
            logger.info(f"Loading atomic spectra for {elem}: {path}")
            with open(path, 'rb') as f:
                atomic_data[elem] = pickle.load(f)

        # Collect all molecules that appear in any atomic dataset
        all_smiles = set()
        for elem_data in atomic_data.values():
            all_smiles.update(elem_data.keys())
        all_smiles = sorted(list(all_smiles))
        rng = np.random.RandomState(self.seed)
        rng.shuffle(all_smiles)

        n_train = int(len(all_smiles) * self.train_ratio)
        if self.split == "train":
            split_smiles = all_smiles[:n_train]
        else:
            split_smiles = all_smiles[n_train:]

        if self.max_samples and len(split_smiles) > self.max_samples:
            split_smiles = split_smiles[:self.max_samples]

        samples = []
        missing_total = 0
        filtered_zero = {e: 0 for e in atomic_data}

        for smiles in tqdm(split_smiles, desc=f"Multi {self.split}"):
            if smiles not in total_lookup:
                missing_total += 1
                continue

            totals = total_lookup[smiles]

            # For each element present in atomic data, create a sample
            for elem, elem_data in atomic_data.items():
                if smiles not in elem_data:
                    continue

                mol_data = elem_data[smiles]
                element_symbols = mol_data[0]
                positions = mol_data[1]
                xas_spectra = mol_data[2]

                n_atoms = len(element_symbols)
                z = torch.from_numpy(self._element_to_z(element_symbols)).long()
                positions = torch.from_numpy(positions.T).float()

                target_indices = [i for i, s in enumerate(element_symbols) if s == elem]
                if xas_spectra.shape[1] != len(target_indices):
                    continue

                xas_mask = torch.zeros(n_atoms, dtype=torch.bool)
                xas_tensor = torch.from_numpy(xas_spectra.T).float()
                y = torch.zeros(n_atoms, 200, dtype=torch.float)

                n_valid = 0
                for xas_idx, atom_idx in enumerate(target_indices):
                    atom_spec = xas_tensor[xas_idx]
                    if torch.any(atom_spec == 0):
                        filtered_zero[elem] += 1
                        continue
                    xas_mask[atom_idx] = True
                    y[atom_idx] = atom_spec
                    n_valid += 1
                
                if n_valid == 0:
                    continue

                data = JointData(
                    z=z,
                    pos=positions,
                    y_total=torch.from_numpy(totals[elem]).float(),
                    y=y,
                    xas_mask=xas_mask,
                    has_atom_data=torch.tensor(True, dtype=torch.bool),
                    element_type=torch.tensor(self.ELEMENT_MAP[elem], dtype=torch.long),
                    smiles=smiles,
                    num_nodes=n_atoms,
                )
                samples.append(data)

        logger.info(
            f"Multi-element {self.split}: {len(samples)} samples, "
            f"missing_total: {missing_total}, filtered_zero: {filtered_zero}"
        )

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
