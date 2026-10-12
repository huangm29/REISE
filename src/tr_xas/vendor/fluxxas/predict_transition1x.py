"""
Predict XAS spectra for Transition1x structures using a trained multi-element model.

Supports:
  - reactant / product / transition_state structures
  - optional full reaction trajectory frames
  - parallelization by reaction index ranges (for Slurm array jobs)

Outputs a pickle file mapping reaction IDs to predicted molecular/atomic XAS spectra.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from joint_training.dataset import JointData
from src.model.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

ELEM_NAMES = {0: 'C', 1: 'N', 2: 'O'}
ELEM_MAP = {'C': 0, 'N': 1, 'O': 2}


def build_data_for_structure(atomic_numbers, positions, rxn_id, state, formula, frame_idx=None):
    """
    Build one JointData object per element (C/N/O) present in the structure.
    Returns list of Data objects.
    """
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim == 3 and positions.shape[0] == 1:
        positions = positions[0]
    n_atoms = len(atomic_numbers)

    symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
    element_symbols = np.array([symbols.get(int(z), 'X') for z in atomic_numbers])

    data_list = []
    for elem_name, elem_type in ELEM_MAP.items():
        target_indices = np.where(element_symbols == elem_name)[0]
        if len(target_indices) == 0:
            continue

        xas_mask = torch.zeros(n_atoms, dtype=torch.bool)
        xas_mask[target_indices] = True

        state_label = state if frame_idx is None else f"{state}_frame{frame_idx}"
        data = JointData(
            z=torch.from_numpy(atomic_numbers).long(),
            pos=torch.from_numpy(positions).float(),
            y_total=torch.zeros(200, dtype=torch.float),
            y=torch.zeros(n_atoms, 200, dtype=torch.float),
            xas_mask=xas_mask,
            has_atom_data=torch.tensor(True, dtype=torch.bool),
            element_type=torch.tensor(elem_type, dtype=torch.long),
            smiles=f"{formula}_{rxn_id}_{state_label}_{elem_name}",
            num_nodes=n_atoms,
        )
        # Attach metadata for result aggregation
        data.rxn_id = rxn_id
        data.state = state
        data.frame_idx = frame_idx if frame_idx is not None else -1
        data.formula = formula
        data.elem_name = elem_name
        data.elem_type = elem_type
        data_list.append(data)
    return data_list


def iter_transition1x_structures(h5_path, split='test', include_trajectory=False,
                                  traj_stride=1, rxn_start=0, rxn_end=None):
    """
    Yield (rxn_idx, rxn_id, state, frame_idx, formula, atomic_numbers, positions).
    rxn_idx is a global counter across all splits (for range-based parallelization).
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        all_rxns = []
        for split_name in [split]:
            split_group = f[split_name]
            for formula in sorted(split_group.keys()):
                formula_group = split_group[formula]
                for rxn_id in sorted(formula_group.keys()):
                    all_rxns.append((split_name, formula, rxn_id, formula_group[rxn_id]))

        total = len(all_rxns)
        rxn_end = rxn_end if rxn_end is not None else total
        selected_rxns = all_rxns[rxn_start:rxn_end]

        for rxn_idx, (split_name, formula, rxn_id, rxn) in enumerate(selected_rxns, start=rxn_start):
            atomic_numbers = rxn['atomic_numbers'][()]

            # Stationary states
            for state in ['reactant', 'product', 'transition_state']:
                if state not in rxn:
                    continue
                state_group = rxn[state]
                positions = state_group['positions'][()]
                yield (rxn_idx, rxn_id, state, None, formula, atomic_numbers, positions)

            # Trajectory frames
            if include_trajectory:
                traj_positions = rxn['positions'][()]  # (n_frames, n_atoms, 3)
                n_frames = traj_positions.shape[0]
                for frame_idx in range(0, n_frames, traj_stride):
                    yield (rxn_idx, rxn_id, 'trajectory', frame_idx, formula,
                           atomic_numbers, traj_positions[frame_idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained multi-element checkpoint")
    parser.add_argument("--h5", type=str, default="Transition1x.h5")
    parser.add_argument("--split", type=str, default="test",
                        choices=['train', 'val', 'test'])
    parser.add_argument("--output", type=str, default="transition1x_predictions.pkl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_trajectory", action='store_true',
                        help="Also predict all trajectory frames (VERY large)")
    parser.add_argument("--traj_stride", type=int, default=1,
                        help="Sample every N-th trajectory frame (only used with --include_trajectory)")
    parser.add_argument("--rxn_start", type=int, default=0,
                        help="Start reaction index for parallelization")
    parser.add_argument("--rxn_end", type=int, default=None,
                        help="End reaction index for parallelization")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of structures for quick test")
    parser.add_argument("--save_atom_spectra", action='store_true',
                        help="Also save per-atom XAS spectra (much larger output)")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    if not cfg['dataset'].get('multi_element', False):
        raise ValueError("Checkpoint is not a multi-element model.")

    print(f"Building model on {args.device}")
    model = EquiformerV2_OC20(**cfg['model'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(args.device)
    model.eval()

    print(f"Loading Transition1x structures from {args.h5} split={args.split}")
    structures = list(iter_transition1x_structures(
        args.h5, args.split,
        include_trajectory=args.include_trajectory,
        traj_stride=args.traj_stride,
        rxn_start=args.rxn_start,
        rxn_end=args.rxn_end,
    ))
    if args.max_samples:
        structures = structures[:args.max_samples]
    print(f"Total structures to predict: {len(structures)}")

    # Build all Data objects
    all_data = []
    for rxn_idx, rxn_id, state, frame_idx, formula, atomic_numbers, positions in structures:
        all_data.extend(build_data_for_structure(
            atomic_numbers, positions, rxn_id, state, formula, frame_idx
        ))
    print(f"Total (structure, element) samples: {len(all_data)}")

    if len(all_data) == 0:
        print("No data to predict. Exiting.")
        return

    from torch_geometric.data import Batch

    # Run inference
    results = defaultdict(lambda: {
        'formula': None,
        'atomic_numbers': None,
        'reactant': {},
        'product': {},
        'transition_state': {},
        'trajectory': defaultdict(dict),
    })

    def update_results_from_batch(batch, mol_pred, atom_pred=None):
        """Update results dict from a processed batch."""
        atom_mask = batch.xas_mask.cpu().numpy()
        atomic_numbers = batch.z.cpu().numpy()
        batch_vec = batch.batch.cpu().numpy()

        rxn_ids = batch.rxn_id
        states = batch.state
        frame_idxs = batch.frame_idx
        formulas = batch.formula
        elem_names = batch.elem_name

        for mol_idx in range(mol_pred.shape[0]):
            rxn_id = rxn_ids[mol_idx]
            state = states[mol_idx]
            frame_idx = int(frame_idxs[mol_idx])
            formula = formulas[mol_idx]
            elem_name = elem_names[mol_idx]

            mask = (batch_vec == mol_idx)

            res = results[rxn_id]
            res['formula'] = formula
            if res['atomic_numbers'] is None:
                res['atomic_numbers'] = atomic_numbers[mask]

            prefix = f"XAS_mol_{elem_name}"

            if state == 'trajectory':
                res['trajectory'][frame_idx][prefix] = mol_pred[mol_idx]
            else:
                res[state][prefix] = mol_pred[mol_idx]

            if args.save_atom_spectra and atom_pred is not None:
                atom_indices = np.where(atom_mask & mask)[0]
                atom_preds_this = atom_pred[atom_indices]
                atom_z_this = atomic_numbers[atom_indices]
                local_indices = atom_indices - mask.nonzero()[0][0]
                atom_prefix = f"XAS_atom_{elem_name}"
                idx_prefix = f"atom_indices_{elem_name}"
                z_prefix = f"atom_z_{elem_name}"

                if state == 'trajectory':
                    res['trajectory'][frame_idx][atom_prefix] = atom_preds_this
                    res['trajectory'][frame_idx][idx_prefix] = local_indices
                    res['trajectory'][frame_idx][z_prefix] = atom_z_this
                else:
                    res[state][atom_prefix] = atom_preds_this
                    res[state][idx_prefix] = local_indices
                    res[state][z_prefix] = atom_z_this

    def process_batch(data_chunk):
        """Process a chunk of Data objects. Raises on OOM."""
        batch = Batch.from_data_list(data_chunk).to(args.device)
        pred = model(batch, is_training=False)  # (n_atoms, 200)

        mol_pred = scatter(
            pred[batch.xas_mask],
            batch.batch[batch.xas_mask],
            dim=0,
            reduce='sum'
        ).cpu().numpy()

        atom_pred = pred.cpu().numpy() if args.save_atom_spectra else None
        update_results_from_batch(batch, mol_pred, atom_pred)
        del batch, pred, mol_pred, atom_pred

    print("Running inference...")
    n_total = len(all_data)
    n_oom_fallback = 0
    n_skipped = 0
    with torch.no_grad():
        pbar = tqdm(total=n_total, desc="Inference")
        i = 0
        while i < n_total:
            chunk_end = min(i + args.batch_size, n_total)
            chunk = all_data[i:chunk_end]

            try:
                process_batch(chunk)
                i = chunk_end
                pbar.update(len(chunk))
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                if len(chunk) == 1:
                    # Single sample OOM: skip it
                    data = chunk[0]
                    print(f"\nOOM on single sample, skipping: {data.rxn_id} {data.state} {data.elem_name}")
                    n_skipped += 1
                    i += 1
                    pbar.update(1)
                else:
                    # Try one by one
                    n_oom_fallback += 1
                    for data in chunk:
                        try:
                            process_batch([data])
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(f"\nOOM on single sample, skipping: {data.rxn_id} {data.state} {data.elem_name}")
                            n_skipped += 1
                        i += 1
                        pbar.update(1)
        pbar.close()

    print(f"Inference done. OOM fallbacks: {n_oom_fallback}, skipped samples: {n_skipped}")

    # Convert nested defaultdicts to regular dicts for pickling
    results = {k: {
        'formula': v['formula'],
        'atomic_numbers': v['atomic_numbers'],
        'reactant': dict(v['reactant']),
        'product': dict(v['product']),
        'transition_state': dict(v['transition_state']),
        'trajectory': dict({frame: dict(data) for frame, data in v['trajectory'].items()}),
    } for k, v in results.items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved predictions for {len(results)} reactions to {output_path}")

    # Print a sample summary
    first_rxn = next(iter(results))
    print(f"\nSample output for {first_rxn}:")
    print(f"  formula: {results[first_rxn]['formula']}")
    for state in ['reactant', 'product', 'transition_state']:
        keys = [k for k in results[first_rxn][state].keys() if k.startswith('XAS_mol_')]
        print(f"  {state}: {keys}")
    n_traj = len(results[first_rxn]['trajectory'])
    if n_traj > 0:
        print(f"  trajectory: {n_traj} frames with predictions")


if __name__ == "__main__":
    main()
