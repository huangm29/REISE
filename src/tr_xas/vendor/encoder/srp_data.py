"""Shared data access for spectral_reaction_path_v1.

Reads the v3.1 canonical Final-MEP dataset (read-only, one level up) and the
reaction-coordinate labels built by build_rc_labels.py (labels/).

Conventions:
  - spectra: (3, 200) float32, edges C/N/O, missing edges zero-filled with
    presence flag 0 (presence computed on raw spectra BEFORE any
    normalization, same rule as the v3.1 ts_dataset fix);
  - composition: 7-dim float32 [nH, nC, nN, nO]/N_atoms, present_C, present_N,
    present_O  (presence as 0/1);
  - every sample carries rxn_id so all evaluation/bootstrap can be grouped
    by reaction (never frame-level random splits).

s is a structural reaction coordinate (mass-weighted arc length along the
converged final MEP), not time.
"""

import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent          # v3.1 canonical pkls (read-only)
LABELS_DIR = PROJECT_ROOT / 'labels'
ELEMS = ['C', 'N', 'O']
SPEC_LEN = 200
COMP_DIM = 7


def load_reactions(split, data_dir=None):
    """Return dict rxn_id -> merged record for one split.

    record keys:
      formula, atomic_numbers (N,), composition (7,),
      reactant / product / transition_state: {'XAS_mol_E': (200,)},
      trajectory: {0..9: {'XAS_mol_E': (200,)}},
      ts_frame_idx, rc: reaction-coordinate label dict from rc_labels_*.pkl
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    with open(data_dir / f'mep_{split}.pkl', 'rb') as f:
        specs = pickle.load(f)
    with open(data_dir / f'mep_labels_{split}.pkl', 'rb') as f:
        geom = pickle.load(f)
    with open(LABELS_DIR / f'rc_labels_{split}.pkl', 'rb') as f:
        rc = pickle.load(f)
    assert set(specs) == set(geom) == set(rc), f'{split}: ID sets differ'

    out = {}
    for rxn_id in specs:
        Z = np.asarray(specs[rxn_id]['atomic_numbers'], dtype=np.int64)
        counts = np.array([(Z == z).sum() for z in (1, 6, 7, 8)], dtype=np.float32)
        comp = np.concatenate([
            counts / len(Z),
            np.array([6 in Z, 7 in Z, 8 in Z], dtype=np.float32),
        ])
        out[rxn_id] = {
            'formula': specs[rxn_id]['formula'],
            'atomic_numbers': Z,
            'composition': comp.astype(np.float32),
            'reactant': specs[rxn_id]['reactant'],
            'product': specs[rxn_id]['product'],
            'transition_state': specs[rxn_id]['transition_state'],
            'trajectory': specs[rxn_id]['trajectory'],
            'ts_frame_idx': int(geom[rxn_id]['ts_frame_idx']),
            'rc': rc[rxn_id],
        }
    return out


def frame_spectra(frame_dict):
    """(3,200) float32 spectra + (3,) presence, from a spectra dict."""
    x = np.zeros((len(ELEMS), SPEC_LEN), dtype=np.float32)
    present = np.zeros(len(ELEMS), dtype=np.float32)
    for e, elem in enumerate(ELEMS):
        key = f'XAS_mol_{elem}'
        if key in frame_dict:
            arr = np.asarray(frame_dict[key], dtype=np.float32)
            if np.abs(arr).sum() > 0:
                x[e] = arr
                present[e] = 1.0
    return x, present


def compute_norm(reactions):
    """Per-edge mean/std over all trajectory frames (train split only)."""
    sums = np.zeros(len(ELEMS))
    sq = np.zeros(len(ELEMS))
    cnt = np.zeros(len(ELEMS))
    for rec in reactions.values():
        for k in range(10):
            x, pres = frame_spectra(rec['trajectory'][k])
            for e in range(len(ELEMS)):
                if pres[e] > 0:
                    sums[e] += x[e].sum()
                    sq[e] += (x[e] ** 2).sum()
                    cnt[e] += x[e].size
    mean = sums / np.maximum(cnt, 1)
    std = np.sqrt(sq / np.maximum(cnt, 1) - mean ** 2)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(x, mean, std, present):
    """Normalize (3,200) per edge; missing edges re-zeroed afterwards."""
    y = (x - mean[:, None]) / std[:, None]
    for e in range(len(ELEMS)):
        if present[e] == 0:
            y[e] = 0.0
    return y.astype(np.float32)


def denormalize(y, mean, std, present):
    x = y * std[:, None] + mean[:, None]
    for e in range(len(ELEMS)):
        if present[e] == 0:
            x[e] = 0.0
    return x.astype(np.float32)


# ---------- spectral metrics (raw spectral units, per edge) ----------

def spectral_metrics(pred, true, present):
    """pred/true: (3,200) raw spectra. Returns dict of per-edge + macro metrics.

    MAE/RMSE in spectral units; cosine similarity; spectral angle (radians).
    Only edges with present==1 contribute.
    """
    out = {}
    maes, rmses, coss, angs = [], [], [], []
    for e, elem in enumerate(ELEMS):
        if present[e] == 0:
            continue
        p, t = pred[e], true[e]
        mae = float(np.abs(p - t).mean())
        rmse = float(np.sqrt(((p - t) ** 2).mean()))
        denom = np.linalg.norm(p) * np.linalg.norm(t)
        cos = float(np.dot(p, t) / denom) if denom > 0 else 1.0
        ang = float(np.arccos(np.clip(cos, -1.0, 1.0)))
        out[f'MAE_{elem}'] = mae
        out[f'RMSE_{elem}'] = rmse
        out[f'cos_{elem}'] = cos
        out[f'angle_{elem}'] = ang
        maes.append(mae); rmses.append(rmse); coss.append(cos); angs.append(ang)
    out['MAE'] = float(np.mean(maes))
    out['RMSE'] = float(np.mean(rmses))
    out['cos'] = float(np.mean(coss))
    out['angle'] = float(np.mean(angs))
    return out


def paired_bootstrap(diff, n_boot=10000, seed=0):
    """Reaction-level paired bootstrap 95% CI of mean(diff)."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)
