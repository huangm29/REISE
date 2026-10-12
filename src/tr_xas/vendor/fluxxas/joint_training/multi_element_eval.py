"""Evaluate multi-element model: per-element R^2, MAE, and case plots."""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from sklearn.metrics import r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_training.dataset import MultiElementXASData
from src.model.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

ELEM_NAMES = {0: 'C', 1: 'N', 2: 'O'}


def rel_mae(pred, target):
    mae = np.mean(np.abs(pred - target))
    scale = np.mean(np.abs(target))
    return mae / (scale + 1e-8) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="logs/multi_element_eval")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_cases", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    dataset = MultiElementXASData(
        root=cfg['dataset']['root'],
        total_spec_path=cfg['dataset']['total_spec_path'],
        atomic_spec_paths=cfg['dataset']['atomic_spec_paths'],
        split="val",
        max_samples=args.max_samples,
        train_ratio=0.9,
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = EquiformerV2_OC20(**cfg['model'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Collect per-element predictions
    results = {0: [], 1: [], 2: []}  # elem_type -> list of dicts

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, is_training=False)
            mol_pred = scatter(pred[batch.xas_mask], batch.batch[batch.xas_mask], dim=0, reduce='sum').cpu().numpy()
            mol_target = batch.y_total.cpu().numpy()
            elem_types = batch.element_type.cpu().numpy()

            for mol_idx in range(len(mol_pred)):
                et = int(elem_types[mol_idx])
                mask = (batch.batch == mol_idx)
                atom_mask = batch.xas_mask[mask].cpu().numpy()
                atom_pred = pred[mask].cpu().numpy()
                atom_target = batch.y[mask].cpu().numpy()

                results[et].append({
                    'mol_pred': mol_pred[mol_idx],
                    'mol_target': mol_target[mol_idx],
                    'atom_pred': atom_pred[atom_mask],
                    'atom_target': atom_target[atom_mask],
                })

    # Per-element metrics
    summary_lines = []
    for et in [0, 1, 2]:
        name = ELEM_NAMES[et]
        data = results[et]
        if len(data) == 0:
            print(f"{name}: no samples")
            continue

        mol_preds = np.stack([d['mol_pred'] for d in data])
        mol_targets = np.stack([d['mol_target'] for d in data])
        atom_preds = np.concatenate([d['atom_pred'] for d in data])
        atom_targets = np.concatenate([d['atom_target'] for d in data])

        mol_r2 = r2_score(mol_targets.flatten(), mol_preds.flatten())
        atom_r2 = r2_score(atom_targets.flatten(), atom_preds.flatten())
        mol_rel = rel_mae(mol_preds, mol_targets)
        atom_rel = rel_mae(atom_preds, atom_targets)

        line = f"{name}: Mol R2={mol_r2:.4f} Atom R2={atom_r2:.4f} Mol RelMAE={mol_rel:.2f}% Atom RelMAE={atom_rel:.2f}% (n={len(data)})"
        print(line)
        summary_lines.append(line)

        # Case analysis: best/worst by molecular MAE
        maes = [np.mean(np.abs(d['mol_pred'] - d['mol_target'])) for d in data]
        maes = np.array(maes)
        best_idx = np.argsort(maes)[:args.n_cases]
        worst_idx = np.argsort(maes)[-args.n_cases:][::-1]

        fig, axes = plt.subplots(2, args.n_cases, figsize=(4*args.n_cases, 6))
        for col, idx in enumerate(best_idx):
            ax = axes[0, col]
            ax.plot(data[idx]['mol_target'], label='Target', alpha=0.7)
            ax.plot(data[idx]['mol_pred'], label='Pred', alpha=0.7)
            ax.set_title(f"{name} Best {col+1}\nMAE={maes[idx]:.1f}")
            ax.legend(fontsize=7)
        for col, idx in enumerate(worst_idx):
            ax = axes[1, col]
            ax.plot(data[idx]['mol_target'], label='Target', alpha=0.7)
            ax.plot(data[idx]['mol_pred'], label='Pred', alpha=0.7)
            ax.set_title(f"{name} Worst {col+1}\nMAE={maes[idx]:.1f}")
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/{name}_cases.png", dpi=200, bbox_inches='tight')
        plt.close()

    with open(f"{args.output_dir}/summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
