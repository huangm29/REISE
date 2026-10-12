#!/usr/bin/env python
"""Task B: single-frame inverse problem on the v3.1 Final-MEP XAS dataset.

Models:
  B0  composition-only label prior (no training; per-formula train statistics,
      fallback to global train statistics for unseen formulas)
  B1  endpoint-only (X_R, X_P, composition) -> conditional-mean point predictions
  B2  snapshot-only (X_k, composition) -> binned p(s) + regressions + near-TS
  B3  endpoint-conditioned snapshot (X_R, X_k, X_P, composition) -> binned p(s),
      signed_delta_s, d_TS, delta_E_R, pre/post, near-TS   (main inverse model)

Forbidden inputs (frame index, other internal frames, MEP sequence, current
structure, current energy) are never used. delta_E_R is a label, never input.

Loss weighting (all heads, all models): every head loss has weight 1.0.
  - s (B2/B3): cross-entropy over 20 uniform bins of [0,1]
  - s (B1), abs_delta_s, signed_delta_s, d_TS, delta_E_R: MSE
  - near_TS (threshold 0.10): BCE with logits
  - pre_or_post (B3): 3-class cross-entropy
Early stopping on the sum of head losses on the val split, patience 10.

Evaluation is grouped by reaction (reaction-level split, reaction-level
bootstrap); frames are never split across groups.

Usage:
  python train_taskB.py                  # full run: B1/B2/B3 x seeds 0..4 + B0
  python train_taskB.py --models B2 --seeds 0 --max-epochs 3   # smoke test
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import srp_data
from taskB_models import B1Model, B2Model, B3Model, N_BINS

PROJECT_ROOT = srp_data.PROJECT_ROOT
CKPT_DIR = PROJECT_ROOT / 'checkpoints'
PRED_DIR = PROJECT_ROOT / 'predictions'
RESULTS_JSON = PROJECT_ROOT / 'results_taskB_5seeds.json'

BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

BATCH = 512
LR = 1e-3
PATIENCE = 10
MAX_EPOCHS = 100

REG_TARGETS = ['abs_delta_s', 'd_TS', 'delta_E_R']          # B1/B2
B3_REG_TARGETS = ['signed_delta_s', 'd_TS', 'delta_E_R']    # B3
LABEL_KEYS = ['s', 'signed_delta_s', 'abs_delta_s', 'd_TS', 'delta_E_R',
              'pre_or_post', 'near_TS_10']


# ---------------------------------------------------------------- data ------

def build_split_arrays(reactions, mean, std):
    """Reactions -> flat per-frame arrays; spectra normalized with train stats."""
    rxn_ids = sorted(reactions)
    N = len(rxn_ids)
    XK = np.zeros((N, 10, 3, srp_data.SPEC_LEN), dtype=np.float32)
    XR = np.zeros((N, 3, srp_data.SPEC_LEN), dtype=np.float32)
    XP = np.zeros((N, 3, srp_data.SPEC_LEN), dtype=np.float32)
    PK = np.zeros((N, 10, 3), dtype=np.float32)
    PR = np.zeros((N, 3), dtype=np.float32)
    PP = np.zeros((N, 3), dtype=np.float32)
    comp = np.zeros((N, srp_data.COMP_DIM), dtype=np.float32)
    labels = {k: np.zeros((N, 10), dtype=np.float32) for k in LABEL_KEYS}
    formulas, ts_idx = [], np.zeros(N, dtype=np.int64)

    for i, rid in enumerate(rxn_ids):
        rec = reactions[rid]
        comp[i] = rec['composition']
        formulas.append(rec['formula'])
        ts_idx[i] = rec['ts_frame_idx']
        for k in range(10):
            x, p = srp_data.frame_spectra(rec['trajectory'][k])
            XK[i, k] = srp_data.normalize(x, mean, std, p)
            PK[i, k] = p
        x, p = srp_data.frame_spectra(rec['reactant'])
        XR[i] = srp_data.normalize(x, mean, std, p); PR[i] = p
        x, p = srp_data.frame_spectra(rec['product'])
        XP[i] = srp_data.normalize(x, mean, std, p); PP[i] = p
        rc = rec['rc']
        for k in LABEL_KEYS:
            labels[k][i] = np.asarray(rc[k], dtype=np.float32)

    return {
        'rxn_ids': rxn_ids, 'formulas': formulas, 'ts_idx': ts_idx,
        'XK': XK, 'XR': XR, 'XP': XP, 'PK': PK, 'PR': PR, 'PP': PP,
        'comp': comp, 'labels': labels,
    }


def s_to_bin(s):
    return np.clip((s * N_BINS).astype(np.int64), 0, N_BINS - 1)


# ------------------------------------------------------------------ B0 ------

def build_b0_stats(train_arr):
    """Per-formula (and global) conditional label statistics from train."""
    formulas = np.array(train_arr['formulas'])
    stats = {}
    groups = {'__global__': np.ones(len(formulas), dtype=bool)}
    for f in sorted(set(formulas)):
        groups[f] = formulas == f
    for name, mask in groups.items():
        ent = {}
        for key in ['s', 'abs_delta_s', 'd_TS', 'delta_E_R']:
            vals = train_arr['labels'][key][mask]          # (n_rxn, 10)
            ent[key + '_mean'] = float(vals.mean())
        near = train_arr['labels']['near_TS_10'][mask]
        ent['near_rate'] = float(near.mean())
        hist = np.zeros(N_BINS, dtype=np.float64)
        np.add.at(hist, s_to_bin(train_arr['labels']['s'][mask].ravel()), 1.0)
        hist /= max(hist.sum(), 1.0)
        ent['s_hist'] = hist
        stats[name] = ent
    return stats


def b0_predict(stats, arr):
    """Per-frame B0 predictions for a split (fallback to __global__)."""
    N = len(arr['rxn_ids'])
    nf = N * 10
    out = {'p_s': np.zeros((nf, N_BINS)), 's_exp': np.zeros(nf),
           's_std': np.zeros(nf), 'abs_delta_s': np.zeros(nf),
           'd_TS': np.zeros(nf), 'delta_E_R': np.zeros(nf),
           'near_prob': np.zeros(nf)}
    for i, f in enumerate(arr['formulas']):
        ent = stats.get(f, stats['__global__'])
        sl = slice(i * 10, (i + 1) * 10)
        p = ent['s_hist']
        exp = float(p @ BIN_CENTERS)
        var = float(p @ (BIN_CENTERS ** 2) - exp ** 2)
        out['p_s'][sl] = p
        out['s_exp'][sl] = exp
        out['s_std'][sl] = np.sqrt(max(var, 0.0))
        for key in ['abs_delta_s', 'd_TS', 'delta_E_R']:
            out[key][sl] = ent[key + '_mean']
        out['near_prob'][sl] = ent['near_rate']
    return out


# ------------------------------------------------------------- training ----

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stack_inputs(arr, model_name, idx):
    """idx: flat frame indices (rxn*10+k) -> (x, pres, comp) numpy batches."""
    r, k = idx // 10, idx % 10
    if model_name == 'B1':
        x = np.stack([arr['XR'][r], arr['XP'][r]], axis=1)
        p = np.stack([arr['PR'][r], arr['PP'][r]], axis=1)
    elif model_name == 'B2':
        x = arr['XK'][r, k][:, None]
        p = arr['PK'][r, k][:, None]
    else:  # B3
        x = np.stack([arr['XR'][r], arr['XK'][r, k], arr['XP'][r]], axis=1)
        p = np.stack([arr['PR'][r], arr['PK'][r, k], arr['PP'][r]], axis=1)
    return x, p, arr['comp'][r]


def compute_loss(model_name, out, batch_y, device):
    """Sum of head losses; every head weight = 1.0 (documented in header)."""
    loss = 0.0
    if model_name == 'B1':
        loss = loss + nn.functional.mse_loss(out['s_mean'], batch_y['s'])
        for t in REG_TARGETS:
            loss = loss + nn.functional.mse_loss(out[t], batch_y[t])
    else:
        loss = loss + nn.functional.cross_entropy(
            out['s_logits'], batch_y['s_bin'])
        reg = B3_REG_TARGETS if model_name == 'B3' else REG_TARGETS
        for t in reg:
            loss = loss + nn.functional.mse_loss(out[t], batch_y[t])
        if model_name == 'B3':
            loss = loss + nn.functional.cross_entropy(
                out['prepost_logits'], batch_y['prepost_cls'])
    loss = loss + nn.functional.binary_cross_entropy_with_logits(
        out['near_logit'], batch_y['near_TS_10'])
    return loss


def label_batch(arr, idx, device):
    r, k = idx // 10, idx % 10
    y = {}
    for key in LABEL_KEYS:
        y[key] = torch.as_tensor(arr['labels'][key][r, k], device=device)
    y['s_bin'] = torch.as_tensor(s_to_bin(arr['labels']['s'][r, k]), device=device)
    y['prepost_cls'] = (y['pre_or_post'].long() + 1)
    return y


def run_epoch(model, model_name, arr, order, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    for lo in range(0, len(order), BATCH):
        idx = order[lo:lo + BATCH]
        x, p, c = stack_inputs(arr, model_name, idx)
        xb = torch.as_tensor(x, device=device)
        pb = torch.as_tensor(p, device=device)
        cb = torch.as_tensor(c, device=device)
        yb = label_batch(arr, idx, device)
        with torch.set_grad_enabled(train):
            out = model(xb, pb, cb)
            loss = compute_loss(model_name, out, yb, device)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        tot += float(loss) * len(idx)
        n += len(idx)
    return tot / max(n, 1)


def train_model(model_name, seed, train_arr, val_arr, device, max_epochs):
    seed_everything(seed)
    cls = {'B1': B1Model, 'B2': B2Model, 'B3': B3Model}[model_name]
    model = cls().to(device)
    n_params = model.count_params()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_train = len(train_arr['rxn_ids']) * 10
    val_order = np.arange(len(val_arr['rxn_ids']) * 10)
    best_val, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(max_epochs):
        rng = np.random.default_rng(seed * 1000 + ep)
        order = rng.permutation(n_train)
        tr_loss = run_epoch(model, model_name, train_arr, order, device, opt)
        val_loss = run_epoch(model, model_name, val_arr, val_order, device)
        hist.append({'epoch': ep, 'train_loss': tr_loss, 'val_loss': val_loss})
        if val_loss < best_val - 1e-5:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, n_params, hist, best_val, ep + 1


@torch.no_grad()
def predict_model(model, model_name, arr, device):
    model.eval()
    N = len(arr['rxn_ids'])
    nf = N * 10
    out = {'s_exp': np.zeros(nf), 's_std': np.zeros(nf),
           'near_prob': np.zeros(nf),
           'abs_delta_s': np.full(nf, np.nan),
           'signed_delta_s': np.full(nf, np.nan),
           'd_TS': np.zeros(nf), 'delta_E_R': np.zeros(nf),
           'prepost_pred': np.full(nf, np.nan)}
    p_s = np.zeros((nf, N_BINS)) if model_name in ('B2', 'B3') else None
    order = np.arange(nf)
    for lo in range(0, nf, 1024):
        idx = order[lo:lo + 1024]
        x, p, c = stack_inputs(arr, model_name, idx)
        o = model(torch.as_tensor(x, device=device),
                  torch.as_tensor(p, device=device),
                  torch.as_tensor(c, device=device))
        sl = slice(lo, lo + len(idx))
        if model_name == 'B1':
            out['s_exp'][sl] = o['s_mean'].cpu().numpy()
            out['abs_delta_s'][sl] = o['abs_delta_s'].cpu().numpy()
        else:
            probs = torch.softmax(o['s_logits'], dim=1).cpu().numpy()
            centers = torch.as_tensor(BIN_CENTERS, device=device)
            exp = (torch.softmax(o['s_logits'], 1) * centers).sum(1)
            var = (torch.softmax(o['s_logits'], 1) * centers ** 2).sum(1) - exp ** 2
            out['s_exp'][sl] = exp.cpu().numpy()
            out['s_std'][sl] = torch.sqrt(var.clamp(min=0)).cpu().numpy()
            p_s[sl] = probs
        for t in ['d_TS', 'delta_E_R']:
            out[t][sl] = o[t].cpu().numpy()
        out['near_prob'][sl] = torch.sigmoid(o['near_logit']).cpu().numpy()
        if model_name == 'B2':
            out['abs_delta_s'][sl] = o['abs_delta_s'].cpu().numpy()
        if model_name == 'B3':
            out['signed_delta_s'][sl] = o['signed_delta_s'].cpu().numpy()
            out['prepost_pred'][sl] = (
                o['prepost_logits'].argmax(1).cpu().numpy() - 1)
    if p_s is not None:
        out['p_s'] = p_s
    return out


# -------------------------------------------------------------- metrics ----

def ece_10bin(prob, true):
    ece = 0.0
    for lo in range(10):
        m = (prob >= lo / 10) & (prob < (lo + 1) / 10 if lo < 9 else prob <= 1.0)
        if m.sum() > 0:
            ece += m.mean() * abs(prob[m].mean() - true[m].mean())
    return float(ece)


def rxn_spearman(pred, true, n_rxn):
    """Mean per-reaction Spearman(pred, true) over the 10 frames."""
    vals = []
    for i in range(n_rxn):
        p, t = pred[i * 10:(i + 1) * 10], true[i * 10:(i + 1) * 10]
        if np.all(p == p[0]):
            vals.append(0.0)          # constant prediction -> no ordering skill
            continue
        rp, rt = rankdata(p), rankdata(t)
        rp = rp - rp.mean(); rt = rt - rt.mean()
        denom = np.sqrt((rp ** 2).sum() * (rt ** 2).sum())
        vals.append(float((rp * rt).sum() / denom) if denom > 0 else 0.0)
    return float(np.mean(vals))


def frame_metrics(pred, arr):
    """All frame-level metrics for one model's predictions on one split."""
    N = len(arr['rxn_ids'])
    y = {k: arr['labels'][k].reshape(-1) for k in LABEL_KEYS}
    m = {}
    se = pred['s_exp'] - y['s']
    m['s_MAE'] = float(np.abs(se).mean())
    m['s_RMSE'] = float(np.sqrt((se ** 2).mean()))
    m['s_spearman_rxn'] = rxn_spearman(pred['s_exp'], y['s'], N)
    for key, name in [('abs_delta_s', 'abs_delta_s_MAE'),
                      ('signed_delta_s', 'signed_delta_s_MAE')]:
        if key in pred and not np.isnan(pred[key]).all():
            m[name] = float(np.abs(pred[key] - y[key]).mean())
    m['d_TS_MAE'] = float(np.abs(pred['d_TS'] - y['d_TS']).mean())
    m['delta_E_R_MAE'] = float(np.abs(pred['delta_E_R'] - y['delta_E_R']).mean())

    near = y['near_TS_10']
    prob = pred['near_prob']
    m['near_TS_AUROC'] = float(roc_auc_score(near, prob))
    m['near_TS_AUPRC'] = float(average_precision_score(near, prob))
    pred_bin = (prob >= 0.5).astype(float)
    tp = ((pred_bin == 1) & (near == 1)).sum(); fn = ((pred_bin == 0) & (near == 1)).sum()
    tn = ((pred_bin == 0) & (near == 0)).sum(); fp = ((pred_bin == 1) & (near == 0)).sum()
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    m['near_TS_bal_acc'] = float(0.5 * (tpr + tnr))
    m['near_TS_ECE'] = ece_10bin(prob, near)

    if 'prepost_pred' in pred and not np.isnan(pred['prepost_pred']).all():
        pp, pt = pred['prepost_pred'], y['pre_or_post']
        m['prepost_acc3'] = float((pp == pt).mean())
        mask = pt != 0
        m['prepost_acc_excl_TS'] = float((pp[mask] == pt[mask]).mean())
    return m


def per_rxn_s_mae(pred, arr):
    se = np.abs(pred['s_exp'] - arr['labels']['s'].reshape(-1))
    return se.reshape(-1, 10).mean(axis=1)


def stratify(pred, arr):
    """s-MAE by formula (>=5 test reactions) and by true ts_frame_idx."""
    per_rxn = per_rxn_s_mae(pred, arr)
    formulas = np.array(arr['formulas'])
    by_formula = {}
    for f in sorted(set(formulas)):
        mask = formulas == f
        if mask.sum() >= 5:
            by_formula[f] = {'n_rxn': int(mask.sum()),
                             's_MAE': float(per_rxn[mask].mean())}
    by_ts = {}
    for t in sorted(set(arr['ts_idx'].tolist())):
        mask = arr['ts_idx'] == t
        by_ts[str(t)] = {'n_rxn': int(mask.sum()),
                         's_MAE': float(per_rxn[mask].mean())}
    return {'by_formula': by_formula, 'by_ts_frame_idx': by_ts}


def auroc_gain_bootstrap(prob_a, prob_b, near, n_rxn, n_boot=2000, seed=0):
    """Reaction-level bootstrap CI of AUROC(a) - AUROC(b)."""
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_boot)
    for b in range(n_boot):
        ridx = rng.integers(0, n_rxn, n_rxn)
        fidx = np.concatenate([np.arange(r * 10, (r + 1) * 10) for r in ridx])
        diffs[b] = roc_auc_score(near[fidx], prob_a[fidx]) - \
                   roc_auc_score(near[fidx], prob_b[fidx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean() if n_boot else 0), float(lo), float(hi)


# ------------------------------------------------------------ prediction ---

def save_predictions(path, model_name, arr, pred):
    N = len(arr['rxn_ids'])
    recs = []
    for i in range(N):
        for k in range(10):
            j = i * 10 + k
            rec = {
                'rxn_id': arr['rxn_ids'][i], 'frame': k,
                'formula': arr['formulas'][i],
                'ts_frame_idx': int(arr['ts_idx'][i]),
                's_exp': float(pred['s_exp'][j]),
                's_std': float(pred['s_std'][j]),
                'pred_abs_delta_s': float(pred.get('abs_delta_s', [np.nan])[j])
                    if 'abs_delta_s' in pred else None,
                'pred_signed_delta_s': float(pred['signed_delta_s'][j])
                    if 'signed_delta_s' in pred else None,
                'pred_d_TS': float(pred['d_TS'][j]),
                'pred_delta_E_R': float(pred['delta_E_R'][j]),
                'near_TS_prob': float(pred['near_prob'][j]),
                'pred_pre_or_post': float(pred['prepost_pred'][j])
                    if 'prepost_pred' in pred else None,
                'labels': {key: float(arr['labels'][key][i, k])
                           for key in LABEL_KEYS},
            }
            rec['p_s_bins'] = (pred['p_s'][j].tolist()
                               if 'p_s' in pred else None)
            recs.append(rec)
    with open(path, 'wb') as f:
        pickle.dump({'model': model_name, 'bin_edges': BIN_EDGES.tolist(),
                     'frames': recs}, f)


# ----------------------------------------------------------------- main ----

def agg_mean_std(vals):
    return [float(np.mean(vals)), float(np.std(vals))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['B1', 'B2', 'B3'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--out', default=str(RESULTS_JSON))
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    CKPT_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t0 = time.time()
    train_rxn = srp_data.load_reactions('train')
    mean, std = srp_data.compute_norm(train_rxn)
    splits = {'train': build_split_arrays(train_rxn, mean, std)}
    print(f'train arrays ready ({time.time()-t0:.0f}s)', flush=True)
    del train_rxn
    for sp in ['val', 'test']:
        splits[sp] = build_split_arrays(srp_data.load_reactions(sp), mean, std)
    test_arr = splits['test']
    n_test_rxn = len(test_arr['rxn_ids'])
    print(f'device={device}  test reactions={n_test_rxn}', flush=True)

    # ---- B0 (seed-independent) ----
    b0_stats = build_b0_stats(splits['train'])
    b0_pred = b0_predict(b0_stats, test_arr)
    unseen = sum(1 for f in test_arr['formulas'] if f not in b0_stats)
    print(f'B0: {unseen}/{n_test_rxn} test formulas unseen in train '
          f'(fallback to global prior)', flush=True)

    baselines = {
        'd_TS_train_mean': float(splits['train']['labels']['d_TS'].mean()),
        'd_TS_train_mean_test_MAE': float(np.abs(
            test_arr['labels']['d_TS'].reshape(-1)
            - splits['train']['labels']['d_TS'].mean()).mean()),
        'near_TS_train_rate': float(
            splits['train']['labels']['near_TS_10'].mean()),
        's_train_mean': float(splits['train']['labels']['s'].mean()),
        's_train_mean_test_MAE': float(np.abs(
            test_arr['labels']['s'].reshape(-1)
            - splits['train']['labels']['s'].mean()).mean()),
    }

    # predictions per model per seed on test
    preds = {'B0': {s: b0_pred for s in args.seeds}}
    params_report = {}
    histories = {}

    for model_name in args.models:
        preds[model_name] = {}
        for seed in args.seeds:
            t1 = time.time()
            model, n_params, hist, best_val, n_ep = train_model(
                model_name, seed, splits['train'], splits['val'], device,
                args.max_epochs)
            params_report[model_name] = n_params
            histories[f'{model_name}_seed{seed}'] = hist
            ckpt = CKPT_DIR / f'taskB_{model_name}_seed{seed}{args.tag}.pt'
            torch.save({'model': model_name, 'seed': seed,
                        'state_dict': model.state_dict(),
                        'val_loss': best_val, 'n_params': n_params}, ckpt)
            pred = predict_model(model, model_name, test_arr, device)
            preds[model_name][seed] = pred
            save_predictions(
                PRED_DIR / f'taskB_{model_name}_seed{seed}{args.tag}.pkl',
                model_name, test_arr, pred)
            met = frame_metrics(pred, test_arr)
            print(f'{model_name} seed{seed}: {n_ep} epochs, '
                  f'val_loss={best_val:.4f}, s_MAE={met["s_MAE"]:.4f}, '
                  f'spearman={met["s_spearman_rxn"]:.3f}, '
                  f'AUROC={met["near_TS_AUROC"]:.3f} '
                  f'({time.time()-t1:.0f}s)', flush=True)
            del model
            torch.cuda.empty_cache() if device == 'cuda' else None

    # B0 predictions saved once per seed name for a uniform interface
    for seed in args.seeds:
        save_predictions(
            PRED_DIR / f'taskB_B0_seed{seed}{args.tag}.pkl', 'B0',
            test_arr, b0_pred)

    # ---- per-seed + mean/std metrics ----
    per_seed, mean_std = {}, {}
    for model_name in ['B0'] + args.models:
        per_seed[model_name] = {}
        all_keys = None
        for seed in args.seeds:
            met = frame_metrics(preds[model_name][seed], test_arr)
            met['stratification'] = stratify(preds[model_name][seed], test_arr)
            per_seed[model_name][str(seed)] = met
            all_keys = [k for k in met if k != 'stratification']
        mean_std[model_name] = {
            k: agg_mean_std([per_seed[model_name][str(s)][k] for s in args.seeds])
            for k in all_keys}

    # ---- gains with reaction-level bootstrap CIs ----
    # s-MAE gains: per-reaction MAE diffs, paired bootstrap (positive = the
    # spectrum/endpoints help). near-TS AUROC gains: reaction-level bootstrap
    # of the AUROC difference over the pooled test sample.
    def s_mae_gain(hi_name, lo_name, seed):
        return per_rxn_s_mae(preds[lo_name][seed], test_arr) - \
               per_rxn_s_mae(preds[hi_name][seed], test_arr)

    gain_defs = [  # (name, better_model, worse_model, interpretation)
        ('snapshot_gain_B2_minus_B0', 'B2', 'B0',
         'MAE_B0 - MAE_B2; positive = single snapshot spectrum helps'),
        ('current_spectrum_gain_B3_minus_B1', 'B3', 'B1',
         'MAE_B1 - MAE_B3; positive = current spectrum adds over endpoints'),
        ('endpoint_conditioning_gain_B3_minus_B2', 'B3', 'B2',
         'MAE_B2 - MAE_B3; positive = endpoints add over snapshot'),
    ]
    gains = {}
    for name, hi, lo, interp in gain_defs:
        if hi not in preds or lo not in preds:
            continue
        # per-reaction diff averaged over seeds -> one paired bootstrap
        diff = np.mean([s_mae_gain(hi, lo, s) for s in args.seeds], axis=0)
        m_, lo_, hi_ = srp_data.paired_bootstrap(diff, n_boot=args.n_boot, seed=0)
        per_seed_ci = {}
        for s in args.seeds:
            pm, pl, ph = srp_data.paired_bootstrap(
                s_mae_gain(hi, lo, s), n_boot=args.n_boot, seed=s)
            per_seed_ci[str(s)] = {'mean': pm, 'ci95': [pl, ph]}
        gains[name] = {
            'definition': interp,
            's_MAE_gain': {'mean': m_, 'ci95': [lo_, hi_],
                           'per_seed': per_seed_ci},
        }
        near = test_arr['labels']['near_TS_10'].reshape(-1)
        au = []
        for s in args.seeds:
            au.append(auroc_gain_bootstrap(
                preds[hi][s]['near_prob'], preds[lo][s]['near_prob'],
                near, n_test_rxn, n_boot=2000, seed=s))
        gains[name]['near_TS_AUROC_gain'] = {
            'mean': float(np.mean([a[0] for a in au])),
            'ci95': [float(np.mean([a[1] for a in au])),
                     float(np.mean([a[2] for a in au]))],
            'per_seed': {str(s): {'mean': a[0], 'ci95': [a[1], a[2]]}
                         for s, a in zip(args.seeds, au)},
        }

    # ---- stratification aggregated across seeds ----
    strat_agg = {}
    for model_name in ['B0'] + args.models:
        by_f, by_t = {}, {}
        for s in args.seeds:
            st = per_seed[model_name][str(s)]['stratification']
            for f, v in st['by_formula'].items():
                by_f.setdefault(f, {'n_rxn': v['n_rxn'], 's_MAE': []})
                by_f[f]['s_MAE'].append(v['s_MAE'])
            for t, v in st['by_ts_frame_idx'].items():
                by_t.setdefault(t, {'n_rxn': v['n_rxn'], 's_MAE': []})
                by_t[t]['s_MAE'].append(v['s_MAE'])
        strat_agg[model_name] = {
            'by_formula': {f: {'n_rxn': v['n_rxn'],
                               's_MAE': agg_mean_std(v['s_MAE'])}
                           for f, v in sorted(by_f.items())},
            'by_ts_frame_idx': {t: {'n_rxn': v['n_rxn'],
                                    's_MAE': agg_mean_std(v['s_MAE'])}
                                for t, v in sorted(by_t.items())},
        }

    # per-formula snapshot gain (seed-averaged), to check single-formula drive
    formula_gain = {}
    if 'B2' in preds:
        diff_rxn = np.mean([s_mae_gain('B2', 'B0', s) for s in args.seeds], 0)
        formulas = np.array(test_arr['formulas'])
        for f in sorted(set(formulas)):
            mask = formulas == f
            if mask.sum() >= 5:
                formula_gain[f] = {'n_rxn': int(mask.sum()),
                                   's_MAE_gain': float(diff_rxn[mask].mean())}
    total_gain = sum(max(v['s_MAE_gain'], 0.0) * v['n_rxn']
                     for v in formula_gain.values())
    top_formula = max(formula_gain.items(),
                      key=lambda kv: kv[1]['s_MAE_gain'] * kv[1]['n_rxn']) \
        if formula_gain else (None, {'s_MAE_gain': 0, 'n_rxn': 0})
    top_share = (max(top_formula[1]['s_MAE_gain'], 0.0) * top_formula[1]['n_rxn']
                 / total_gain) if total_gain > 0 else 0.0

    # ---- verdict ----
    def g(name):
        return mean_std[name]['s_MAE'][0], mean_std[name]['s_MAE'][1]

    def ci(d):
        lo, hi = d.get('ci95', [float('nan'), float('nan')])
        return f'[{lo:.4f}, {hi:.4f}]'

    b0m, _ = g('B0')
    verdict_lines = []
    if 'B2' in mean_std:
        b2m, b2s = g('B2')
        sg = gains.get('snapshot_gain_B2_minus_B0', {})
        sm = sg.get('s_MAE_gain', {})
        verdict_lines.append(
            f"(1) Single snapshot vs label prior (B2 vs B0): s-MAE "
            f"{b2m:.4f}±{b2s:.4f} vs {b0m:.4f}; reaction-level bootstrap "
            f"gain {sm.get('mean', float('nan')):.4f} "
            f"95%CI {ci(sm)}. "
            + ("A single snapshot spectrum DOES contain TS-distance "
               "information beyond the composition prior."
               if sm.get('ci95', [0])[0] > 0 else
               "No significant snapshot information beyond the prior."))
    if 'B3' in mean_std and 'B2' in mean_std:
        b3m, b3s = g('B3')
        eg = gains.get('endpoint_conditioning_gain_B3_minus_B2', {})
        em = eg.get('s_MAE_gain', {})
        pp = mean_std['B3'].get('prepost_acc3', [float('nan')] * 2)
        ppx = mean_std['B3'].get('prepost_acc_excl_TS', [float('nan')] * 2)
        verdict_lines.append(
            f"(2) Endpoint conditioning (B3 vs B2): s-MAE {b3m:.4f}±{b3s:.4f} "
            f"vs {mean_std['B2']['s_MAE'][0]:.4f}; gain "
            f"{em.get('mean', float('nan')):.4f} 95%CI {ci(em)}. "
            f"pre/post 3-class acc {pp[0]:.3f}±{pp[1]:.3f}, excl-TS "
            f"{ppx[0]:.3f}±{ppx[1]:.3f}. "
            + ("Endpoints significantly improve the coordinate estimate."
               if em.get('ci95', [0])[0] > 0 else
               "Endpoints do not significantly improve over the snapshot."))
    if 'B3' in mean_std and 'B1' in mean_std:
        cg = gains.get('current_spectrum_gain_B3_minus_B1', {})
        cm = cg.get('s_MAE_gain', {})
        verdict_lines.append(
            f"(3) Current-spectrum gain (B3 vs B1): "
            f"{cm.get('mean', float('nan')):.4f} 95%CI {ci(cm)}. "
            + ("Reaction-level significant: the current-frame spectrum carries "
               "information beyond the endpoints."
               if cm.get('ci95', [0])[0] > 0 else
               "Not reaction-level significant."))
    verdict_lines.append(
        f"(4) Single-formula drive: largest positive per-formula snapshot "
        f"gain is {top_formula[0]} "
        f"({top_formula[1]['s_MAE_gain']:.4f} s-MAE gain over "
        f"{top_formula[1]['n_rxn']} rxns, ~{100*top_share:.0f}% of total "
        f"positive gain). "
        + ("No single formula drives the gain." if top_share < 0.25 else
           "A substantial share of the gain comes from one formula family."))

    results = {
        'task': 'B (single-frame inverse problem)',
        'loss_weighting': {
            'description': 'sum of head losses, every head weight = 1.0; '
                           'early stopping on this val sum, patience 10',
            's': 'CE over 20 uniform bins (B2/B3) / MSE (B1)',
            'regression': 'MSE per head', 'near_TS': 'BCE-with-logits',
            'pre_or_post': '3-class CE (B3)'},
        'config': {'n_bins': N_BINS, 'batch': BATCH, 'lr': LR,
                   'patience': PATIENCE, 'max_epochs': args.max_epochs,
                   'seeds': args.seeds, 'n_boot': args.n_boot,
                   'near_TS_threshold': 0.10},
        'n_params': params_report,
        'baselines': baselines,
        'b0_unseen_formulas_in_test': f'{unseen}/{n_test_rxn}',
        'per_seed': per_seed,
        'mean_std': mean_std,
        'gains': gains,
        'stratification': strat_agg,
        'formula_snapshot_gain': formula_gain,
        'verdict': '\n'.join(verdict_lines),
        'wall_time_s': time.time() - t0,
    }
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('\n' + results['verdict'], flush=True)
    print(f'\nWrote {args.out}  (total {time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
