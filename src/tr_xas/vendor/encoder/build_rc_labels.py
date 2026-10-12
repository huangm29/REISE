"""Build continuous reaction-coordinate labels for the v3.1 Final-MEP dataset.

For each 10-frame final MEP ([R] + 8 internal images + [P]):

  Delta_l_k = sqrt( sum_i m_i |r_{i,k+1} - R_k r_{i,k}|^2 / sum_i m_i )
  s_k       = cumsum(Delta_l) / sum(Delta_l)          # s_0 = 0, s_9 = 1

R_k is the mass-weighted Kabsch rotation aligning frame k onto frame k+1
(weighted centroids removed first). s is a STRUCTURAL reaction coordinate
(arc length along the converged MEP), not time.

Per-frame labels:
  s, signed_delta_s = s - s_TS, abs_delta_s,
  d_TS        = mass-weighted Kabsch RMSD(frame, TS)   [Angstrom]
  delta_E_R   = E(frame) - E(R)                        [eV]
  pre_or_post = -1 (s < s_TS), 0 (TS frame), +1 (s > s_TS)
  near_TS_05  = |s - s_TS| < 0.05
  near_TS_10  = |s - s_TS| < 0.10

QC: s[0]==0, s[-1]==1, strictly increasing, s_TS == s[ts_frame_idx],
d_TS[ts] < 1e-6 A, reaction IDs identical to v3.1 canonical labels,
all values finite. frame_idx/9 is NOT used as a label (naive baseline only).

Reads v3.1 canonical data read-only from --data_dir.
Writes labels/rc_labels_{split}.pkl and rc_labels_qc.json into the project.

Usage (from spectral_reaction_path_v1/):
  python scripts/build_rc_labels.py
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent  # v3.1 canonical pkls live there
SPLITS = ['train', 'val', 'test']
EXPECTED = {'train': 9561, 'val': 225, 'test': 287}

# atomic masses (H, C, N, O are the only elements in Transition1x)
MASS = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}


def weighted_kabsch_rmsd(P, Q, m):
    """Mass-weighted Kabsch: rotate P onto Q (weighted centroids removed).

    Returns (rotation R (3,3), weighted RMSD in Angstrom).
    Minimizes sum_i m_i |R p_i - q_i|^2 / sum_i m_i.
    """
    w = m / m.sum()
    mu_p = (w[:, None] * P).sum(0)
    mu_q = (w[:, None] * Q).sum(0)
    Pc = P - mu_p
    Qc = Q - mu_q
    H = (Pc * w[:, None]).T @ Qc          # (3,3), H = sum w p q^T
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = Pc @ R.T - Qc                  # R p  (row vectors)
    rmsd = float(np.sqrt((m[:, None] * diff ** 2).sum() / m.sum()))
    return R, rmsd


def build_labels_for_split(split, data_dir):
    with open(data_dir / f'mep_labels_{split}.pkl', 'rb') as f:
        geom = pickle.load(f)
    with open(data_dir / f'mep_structures_{split}.pkl', 'rb') as f:
        structs = pickle.load(f)

    labels, fails = {}, []
    min_delta_l = np.inf
    for rxn_id in sorted(geom):
        g, st = geom[rxn_id], structs[rxn_id]
        path = np.asarray(g['path_positions'], dtype=np.float64)   # (10,N,3)
        ts = np.asarray(g['ts_positions'], dtype=np.float64)       # (N,3)
        ts_idx = int(g['ts_frame_idx'])
        en = np.asarray(st['path_energies'], dtype=np.float64)     # (10,)
        m = np.array([MASS[int(z)] for z in st['atomic_numbers']])  # (N,)

        # mass-weighted arc length between Kabsch-aligned consecutive frames
        delta_l = np.empty(9)
        for k in range(9):
            _, rmsd = weighted_kabsch_rmsd(path[k], path[k + 1], m)
            delta_l[k] = rmsd
        min_delta_l = min(min_delta_l, delta_l.min())
        if (delta_l <= 0).any():
            fails.append(f'{rxn_id}: non-positive Delta_l '
                         f'{delta_l[delta_l <= 0].tolist()}')
            continue
        s = np.concatenate([[0.0], np.cumsum(delta_l)])
        s /= s[-1]

        d_ts = np.empty(10)
        for k in range(10):
            _, d_ts[k] = weighted_kabsch_rmsd(path[k], ts, m)

        s_ts = s[ts_idx]
        signed = s - s_ts
        labels[rxn_id] = {
            'ts_frame_idx': ts_idx,
            's': s.astype(np.float32),
            's_TS': np.float32(s_ts),
            'signed_delta_s': signed.astype(np.float32),
            'abs_delta_s': np.abs(signed).astype(np.float32),
            'd_TS': d_ts.astype(np.float32),
            'delta_E_R': (en - en[0]).astype(np.float32),
            'pre_or_post': np.sign(signed).astype(np.int8),  # -1/0/+1
            'near_TS_05': (np.abs(signed) < 0.05),
            'near_TS_10': (np.abs(signed) < 0.10),
            'delta_l': delta_l.astype(np.float32),
        }
    return labels, fails, min_delta_l


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=str(DEFAULT_DATA_DIR),
                        help='v3.1 canonical dataset directory (read-only)')
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()

    out_dir = PROJECT_ROOT / 'labels'
    out_dir.mkdir(exist_ok=True)

    qc = {'passed': True, 'splits': {}, 'failures': {}}
    for split in SPLITS:
        labels, fails, min_dl = build_labels_for_split(split, data_dir)

        # ID identity with v3.1 canonical
        with open(data_dir / f'mep_labels_{split}.pkl', 'rb') as f:
            canonical_ids = set(pickle.load(f))
        if set(labels) != canonical_ids:
            fails.append('reaction IDs differ from v3.1 canonical')

        # per-reaction QC
        finite_ok = True
        for rxn_id, lab in labels.items():
            s = lab['s']
            if not (s[0] == 0.0 and abs(s[-1] - 1.0) < 1e-6):
                fails.append(f'{rxn_id}: s endpoints {s[0]}, {s[-1]}')
            if not np.all(np.diff(s) > 0):
                fails.append(f'{rxn_id}: s not strictly increasing')
            if lab['s_TS'] != s[lab['ts_frame_idx']]:
                fails.append(f'{rxn_id}: s_TS mismatch')
            if lab['d_TS'][lab['ts_frame_idx']] >= 1e-6:
                fails.append(f"{rxn_id}: d_TS[ts] = {lab['d_TS'][lab['ts_frame_idx']]}")
            for key in ('s', 'signed_delta_s', 'abs_delta_s', 'd_TS', 'delta_E_R'):
                if not np.isfinite(lab[key]).all():
                    finite_ok = False
                    fails.append(f'{rxn_id}: non-finite {key}')
        n = len(labels)
        if n != EXPECTED[split]:
            fails.append(f'count {n} != {EXPECTED[split]}')

        with open(out_dir / f'rc_labels_{split}.pkl', 'wb') as f:
            pickle.dump(labels, f)

        s_ts_all = np.array([l['s_TS'] for l in labels.values()])
        qc['splits'][split] = {
            'reactions': n,
            'min_delta_l_A': float(min_dl),
            's_TS_min_max': [float(s_ts_all.min()), float(s_ts_all.max())],
            'all_finite': finite_ok,
            'n_failures': len(fails),
        }
        qc['failures'][split] = fails[:50]
        if fails:
            qc['passed'] = False
        print(f"{split}: {n} reactions, min Delta_l {min_dl:.2e} A, "
              f"s_TS in [{s_ts_all.min():.3f}, {s_ts_all.max():.3f}], "
              f"failures {len(fails)}")

    with open(PROJECT_ROOT / 'labels' / 'rc_labels_qc.json', 'w') as f:
        json.dump(qc, f, indent=2)
    print(f"QC {'PASSED' if qc['passed'] else 'FAILED'} -> labels/rc_labels_qc.json")
    return 0 if qc['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
