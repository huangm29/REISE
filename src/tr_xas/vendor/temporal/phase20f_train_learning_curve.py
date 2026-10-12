#!/usr/bin/env python3
"""Train the preregistered Phase-20 random-window TCN learning curve."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BASE = Path(os.environ["PHASE20_BASE"])
PLAN_PATH = BASE / "input/phase20f_tcn_learning_curve_plan.json"
PUBLIC_PATH = BASE / "input/blinded_xas_input/public_manifest.json"
LABELS_DIR = BASE / "minimal_training_truth"
FEATURES_DIR = BASE / "b3_seed_features"
OUT = BASE / "phase20f_tcn_learning_curve"
PREFIXES = [50, 100, 250, 500, 800, 1200]
EXPECTED_REACTION_COUNTS = [17, 43, 113, 250, 404, 617]
CURVE_SEEDS = [0, 1, 2]
FINAL_EXTRA_SEEDS = [3, 4]
N_FRAMES = 201
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1.0e-3
RANK_MARGIN = 0.5
TOPK_NEGATIVE = 10
GAUSSIAN_SIGMA_S = 0.10
GRADIENT_CLIP = 5.0
PERSISTENCE_WIDTH = 9
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20200821


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SamePadConv1d(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.pad = nn.ReplicationPad1d(dilation)
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = SamePadConv1d(channels, dilation)
        self.conv2 = SamePadConv1d(channels, dilation)
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.silu(self.norm1(self.conv1(x))))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class TemporalDetector(nn.Module):
    def __init__(self, n_features: int = 12):
        super().__init__()
        channels = 48
        self.input = nn.Conv1d(n_features, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, dilation) for dilation in (1, 2, 4, 8, 16, 32)
        ])
        self.output = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(1)


def rank_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for row_logits, row_labels in zip(logits, labels):
        positive = row_logits[row_labels > 0.5]
        negative = row_logits[row_labels <= 0.5]
        if len(positive) and len(negative):
            hard_negative = torch.topk(
                negative, k=min(TOPK_NEGATIVE, len(negative))
            ).values.mean()
            losses.append(F.relu(RANK_MARGIN + hard_negative - positive.max()))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def listwise_center_loss(logits: torch.Tensor, abs_ds: torch.Tensor) -> torch.Tensor:
    target = torch.exp(-0.5 * (abs_ds / GAUSSIAN_SIGMA_S) ** 2)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def expected_distance_loss(logits: torch.Tensor, abs_ds: torch.Tensor) -> torch.Tensor:
    probability = F.softmax(logits / 0.5, dim=1)
    return (probability * abs_ds).sum(dim=1).mean()


def batches(indices: np.ndarray, generator: np.random.Generator):
    shuffled = generator.permutation(indices)
    for start in range(0, len(shuffled), BATCH_SIZE):
        yield shuffled[start:start + BATCH_SIZE]


def scalar_features(path: Path, expected_hash: str) -> tuple[list[str], np.ndarray]:
    if sha256(path) != expected_hash:
        raise RuntimeError(f"Feature hash mismatch: {path}")
    with np.load(path, allow_pickle=True) as z:
        frame_ids = z["frame_ids"].astype(str).tolist()
        features = np.asarray(z["features"], np.float32)
    if features.shape != (5, N_FRAMES, 283):
        raise RuntimeError(f"Unexpected B3 feature shape: {path} {features.shape}")
    p_bar = features[:, :, 256:276].mean(axis=0)
    p_bar /= p_bar.sum(axis=1, keepdims=True)
    centers = np.arange(20, dtype=np.float32) / 20.0 + 0.025
    s_exp = p_bar @ centers
    variance = (p_bar * centers[None] ** 2).sum(axis=1) - s_exp ** 2
    raw = np.stack([
        s_exp,
        np.sqrt(np.maximum(variance, 0.0)),
        features[:, :, 279].mean(axis=0),
        features[:, :, 280].mean(axis=0),
        features[:, :, 281].mean(axis=0),
        features[:, :, 282].mean(axis=0),
    ], axis=1).astype(np.float32)
    difference = np.zeros_like(raw)
    difference[1:] = raw[1:] - raw[:-1]
    return frame_ids, np.concatenate([raw, difference], axis=1)


def load_units() -> tuple[list[dict], list[dict], dict]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["status"] != "FROZEN_BEFORE_PHASE20_XAS_FEATURES_OR_TCN_TRAINING":
        raise RuntimeError("Phase20F plan is not frozen")
    if plan["data_policy"]["official_test_loaded"]:
        raise RuntimeError("Invalid test policy")
    if sha256(PUBLIC_PATH) != plan["phase20d_public_manifest_sha256"]:
        raise RuntimeError("Phase20D public manifest hash mismatch")
    public = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    if public["status"] != "BLINDED_RANDOM_WINDOWS_FROZEN_BEFORE_PHASE20_XAS":
        raise RuntimeError("Unexpected public status")
    if public["official_test_loaded"] or public["train_validation_formula_overlap"]:
        raise RuntimeError("Public split contamination")

    feature_manifest_path = FEATURES_DIR / "feature_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest["status"] != "FROZEN_B3_SEED_FEATURES_COMPLETE_NO_STRUCTURAL_TRUTH":
        raise RuntimeError("B3 features are not frozen")
    if feature_manifest["public_manifest_sha256"] != sha256(PUBLIC_PATH):
        raise RuntimeError("B3/public mismatch")
    feature_index = {unit["trajectory_id"]: unit for unit in feature_manifest["units"]}

    labels_manifest_path = LABELS_DIR / "minimal_labels_manifest.json"
    labels_manifest = json.loads(labels_manifest_path.read_text(encoding="utf-8"))
    if labels_manifest["status"] != "MINIMAL_PHASE20_TRAINING_LABELS_NO_TIME_OR_TS_POSITION":
        raise RuntimeError("Unexpected label status")
    if labels_manifest["public_manifest_sha256"] != sha256(PUBLIC_PATH):
        raise RuntimeError("Labels/public mismatch")
    forbidden = {"time_fs", "source_frame_indices", "projected_s", "s_TS", "event_position"}
    if forbidden - set(labels_manifest["excluded_fields"]):
        raise RuntimeError("Label manifest does not explicitly exclude position leakage")

    public_index = {unit["trajectory_id"]: unit for unit in public["units"]}
    units = []
    for label_unit in labels_manifest["units"]:
        tid = label_unit["trajectory_id"]
        if tid not in public_index or tid not in feature_index:
            raise RuntimeError(f"Missing public/features: {tid}")
        labels_path = LABELS_DIR / label_unit["labels_npz"]
        if sha256(labels_path) != label_unit["labels_npz_sha256"]:
            raise RuntimeError(f"Label hash mismatch: {tid}")
        labels = np.load(labels_path, allow_pickle=True)
        frame_ids, x = scalar_features(
            FEATURES_DIR / feature_index[tid]["feature_npz"],
            feature_index[tid]["feature_npz_sha256"],
        )
        if frame_ids != labels["frame_ids"].astype(str).tolist():
            raise RuntimeError(f"Feature/label alignment failure: {tid}")
        public_unit = public_index[tid]
        units.append({
            "trajectory_id": tid,
            "reaction": public_unit["reaction"],
            "formula": public_unit["formula"],
            "source_split": public_unit["source_split"],
            "nested_order": int(public_unit["nested_order"]),
            "x": x,
            "labels": np.asarray(labels["near_true"], np.float32),
            "abs_ds": np.asarray(labels["abs_delta_s"], np.float32),
            "frame_ids": frame_ids,
        })

    train = [unit for unit in units if unit["source_split"] == "train"]
    validation = [unit for unit in units if unit["source_split"] == "val"]
    train_reactions = {unit["reaction"] for unit in train}
    val_reactions = {unit["reaction"] for unit in validation}
    train_formulas = {unit["formula"] for unit in train}
    val_formulas = {unit["formula"] for unit in validation}
    if (len(train), len(validation), len(train_reactions), len(val_reactions)) != (2442, 183, 617, 47):
        raise RuntimeError("Phase20 train/validation cohort mismatch")
    if train_reactions & val_reactions or train_formulas & val_formulas:
        raise RuntimeError("Train/validation leakage")
    return train, validation, {
        "plan": plan,
        "feature_manifest_sha256": sha256(feature_manifest_path),
        "labels_manifest_sha256": sha256(labels_manifest_path),
    }


def train_seed(seed: int, x: np.ndarray, y: np.ndarray, abs_ds: np.ndarray,
               mean: np.ndarray, std: np.ndarray, device: torch.device,
               tag: str) -> tuple[TemporalDetector, list[dict]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    normalized = ((x - mean) / std).astype(np.float32)
    x_t = torch.from_numpy(normalized)
    y_t = torch.from_numpy(y)
    ds_t = torch.from_numpy(abs_ds)
    positives = float(y.sum())
    negatives = float(y.size - y.sum())
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    model = TemporalDetector(x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = np.random.default_rng(seed)
    indices = np.arange(len(x))
    history = []
    model.train()
    for epoch in range(EPOCHS):
        epoch_values = []
        for selected in batches(indices, generator):
            xb = x_t[selected].to(device).transpose(1, 2)
            yb = y_t[selected].to(device)
            db = ds_t[selected].to(device)
            logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            ranking = rank_loss(logits, yb)
            center = listwise_center_loss(logits, db)
            distance = expected_distance_loss(logits, db)
            loss = bce + ranking + 0.25 * center + distance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            epoch_values.append([
                float(bce.detach().cpu()), float(ranking.detach().cpu()),
                float(center.detach().cpu()), float(distance.detach().cpu()),
                float(loss.detach().cpu()),
            ])
        values = np.asarray(epoch_values)
        history.append({
            "epoch": epoch + 1,
            "bce": float(values[:, 0].mean()),
            "ranking": float(values[:, 1].mean()),
            "center": float(values[:, 2].mean()),
            "distance": float(values[:, 3].mean()),
            "total": float(values[:, 4].mean()),
        })
        if (epoch + 1) % 20 == 0:
            print(f"TRAIN {tag} seed={seed} epoch={epoch + 1}/{EPOCHS} loss={history[-1]['total']:.6f}", flush=True)
    return model.eval(), history


def score_models(models: list[nn.Module], x: np.ndarray, mean: np.ndarray,
                 std: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    x_t = torch.from_numpy(((x - mean) / std).astype(np.float32))
    chunks = [[] for _ in models]
    with torch.no_grad():
        for start in range(0, len(x_t), 32):
            batch = x_t[start:start + 32].to(device).transpose(1, 2)
            for model_index, model in enumerate(models):
                chunks[model_index].append(torch.sigmoid(model(batch)).cpu().numpy())
    per_seed = np.stack([np.concatenate(values) for values in chunks])
    return per_seed.mean(axis=0), per_seed.std(axis=0)


def persistent9(scores: np.ndarray) -> np.ndarray:
    radius = PERSISTENCE_WIDTH // 2
    padded = np.pad(scores, ((0, 0), (radius, radius)), mode="reflect")
    kernel = np.ones(PERSISTENCE_WIDTH, np.float64) / PERSISTENCE_WIDTH
    return np.stack([
        np.convolve(row, kernel, mode="valid") for row in padded
    ]).astype(np.float32)


def summarize(units: list[dict], scores: np.ndarray) -> tuple[dict, list[dict]]:
    rows = []
    by_reaction: defaultdict[str, list] = defaultdict(list)
    for unit, unit_scores in zip(units, scores):
        selected = int(np.argmax(unit_scores))
        true_center = int(np.argmin(unit["abs_ds"]))
        row = {
            "trajectory_id": unit["trajectory_id"],
            "reaction": unit["reaction"],
            "success": bool(unit["labels"][selected] > 0.5),
            "selected_index": selected,
            "true_closest_index": true_center,
            "absolute_index_error": abs(selected - true_center),
            "selected_abs_delta_s": float(unit["abs_ds"][selected]),
            "edge_selection": bool(selected < 10 or selected >= N_FRAMES - 10),
        }
        rows.append(row)
        by_reaction[unit["reaction"]].append(row)
    per_reaction = {
        reaction: {
            "n": len(group),
            "success_rate": float(np.mean([row["success"] for row in group])),
            "median_abs_delta_s": float(np.median([row["selected_abs_delta_s"] for row in group])),
            "median_absolute_index_error": float(np.median([row["absolute_index_error"] for row in group])),
        }
        for reaction, group in sorted(by_reaction.items())
    }
    return {
        "n_reactions": len(per_reaction),
        "n_trajectories": len(rows),
        "reaction_macro_success": float(np.mean([value["success_rate"] for value in per_reaction.values()])),
        "trajectory_success": float(np.mean([row["success"] for row in rows])),
        "median_selected_abs_delta_s": float(np.median([row["selected_abs_delta_s"] for row in rows])),
        "median_absolute_index_error": float(np.median([row["absolute_index_error"] for row in rows])),
        "edge_selection_rate": float(np.mean([row["edge_selection"] for row in rows])),
        "per_reaction": per_reaction,
    }, rows


def fixed_index_scores(n_units: int, index: int) -> np.ndarray:
    scores = np.zeros((n_units, N_FRAMES), np.float32)
    scores[:, index] = 1.0
    return scores


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Output already exists: {OUT}")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(1)
    train, validation, provenance = load_units()
    OUT.mkdir()
    (OUT / "checkpoints").mkdir()
    (OUT / "validation_predictions").mkdir()

    validation_x = np.stack([unit["x"] for unit in validation])
    validation_midpoint, _ = summarize(validation, fixed_index_scores(len(validation), N_FRAMES // 2))
    validation_framewise, _ = summarize(
        validation, persistent9(np.stack([unit["x"][:, 2] for unit in validation]))
    )
    random_by_reaction: defaultdict[str, list] = defaultdict(list)
    for unit in validation:
        random_by_reaction[unit["reaction"]].append(float(unit["labels"].mean()))
    random_macro = float(np.mean([
        np.mean(values) for values in random_by_reaction.values()
    ]))
    baselines = {
        "fixed_midpoint": validation_midpoint,
        "framewise_B3v1_near_prob_persistent9": validation_framewise,
        "uniform_random_frame_expected_reaction_macro_success": random_macro,
    }
    print(json.dumps({
        "BASELINE_midpoint": validation_midpoint["reaction_macro_success"],
        "BASELINE_framewise": validation_framewise["reaction_macro_success"],
        "BASELINE_random_expected": random_macro,
    }), flush=True)

    results = {}
    per_reaction_success = {}
    validation_ids = np.asarray([unit["trajectory_id"] for unit in validation], object)
    for prefix, expected_count in zip(PREFIXES, EXPECTED_REACTION_COUNTS):
        selected_units = [unit for unit in train if unit["nested_order"] < prefix]
        selected_reactions = sorted({unit["reaction"] for unit in selected_units})
        if len(selected_reactions) != expected_count:
            raise RuntimeError(f"Prefix {prefix}: {len(selected_reactions)} != {expected_count}")
        x = np.stack([unit["x"] for unit in selected_units])
        y = np.stack([unit["labels"] for unit in selected_units])
        abs_ds = np.stack([unit["abs_ds"] for unit in selected_units])
        mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
        std = x.reshape(-1, x.shape[-1]).std(axis=0)
        std = np.where(std < 1.0e-8, 1.0, std)
        models = []
        histories = {}
        checkpoint_hashes = {}
        prefix_dir = OUT / "checkpoints" / f"prefix{prefix}"
        prefix_dir.mkdir()
        seeds = CURVE_SEEDS + (FINAL_EXTRA_SEEDS if prefix == PREFIXES[-1] else [])
        per_seed_scores = []
        for seed in seeds:
            model, history = train_seed(
                seed, x, y, abs_ds, mean, std, device, f"prefix{prefix}"
            )
            models.append(model)
            histories[str(seed)] = history
            checkpoint = prefix_dir / f"seed{seed}.pt"
            torch.save({
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "seed": seed,
                "prefix": prefix,
                "n_training_reactions": len(selected_reactions),
                "n_training_trajectories": len(selected_units),
                "training_reactions": selected_reactions,
                "normalization_mean": mean,
                "normalization_std": std,
                "plan_sha256": sha256(PLAN_PATH),
            }, checkpoint)
            checkpoint_hashes[str(seed)] = sha256(checkpoint)
            seed_score, _ = score_models([model], validation_x, mean, std, device)
            per_seed_scores.append(seed_score)

        curve_raw = np.mean(np.stack(per_seed_scores[:len(CURVE_SEEDS)]), axis=0)
        curve_persistent = persistent9(curve_raw)
        curve_summary, rows = summarize(validation, curve_persistent)
        final5_summary = None
        final5_raw = None
        if prefix == PREFIXES[-1]:
            final5_raw = np.mean(np.stack(per_seed_scores), axis=0)
            final5_summary, _ = summarize(validation, persistent9(final5_raw))
        np.savez_compressed(
            OUT / "validation_predictions" / f"prefix{prefix}.npz",
            trajectory_ids=validation_ids,
            raw_probability=curve_raw.astype(np.float32),
            persistent_probability_9fs=curve_persistent,
            final5_raw_probability=(final5_raw.astype(np.float32) if final5_raw is not None else np.empty((0,), np.float32)),
        )
        results[str(prefix)] = {
            "n_training_reactions": len(selected_reactions),
            "n_training_trajectories": len(selected_units),
            "training_reactions": selected_reactions,
            "curve_seeds": CURVE_SEEDS,
            "validation_three_seed_persistent9": curve_summary,
            "validation_five_seed_persistent9": final5_summary,
            "checkpoint_sha256": checkpoint_hashes,
            "training_histories": histories,
        }
        per_reaction_success[prefix] = np.asarray([
            curve_summary["per_reaction"][reaction]["success_rate"]
            for reaction in sorted(curve_summary["per_reaction"])
        ])
        print(json.dumps({
            "prefix": prefix,
            "train_reactions": len(selected_reactions),
            "train_trajectories": len(selected_units),
            "validation_macro": curve_summary["reaction_macro_success"],
            "validation_median_abs_ds": curve_summary["median_selected_abs_delta_s"],
            "five_seed_macro": final5_summary["reaction_macro_success"] if final5_summary else None,
        }), flush=True)
        del models, x, y, abs_ds
        torch.cuda.empty_cache()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(per_reaction_success[PREFIXES[-1]]),
                           size=(BOOTSTRAP_REPLICATES, len(per_reaction_success[PREFIXES[-1]])))
    paired = {}
    largest = per_reaction_success[PREFIXES[-1]]
    for prefix in PREFIXES[:-1]:
        difference = largest - per_reaction_success[prefix]
        bootstrap = difference[sampled].mean(axis=1)
        paired[f"1200_minus_{prefix}"] = {
            "reaction_macro_difference": float(difference.mean()),
            "reaction_cluster_bootstrap_95pct_CI": [
                float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
            ],
            "reactions_improved_equal_worse": {
                "improved": int(np.sum(difference > 0)),
                "equal": int(np.sum(difference == 0)),
                "worse": int(np.sum(difference < 0)),
            },
        }

    summary = {
        "status": "PHASE20_RANDOM_WINDOW_LEARNING_CURVE_COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "official_test_loaded": False,
        "phase20_validation_used_for_hyperparameter_selection": False,
        "plan_sha256": sha256(PLAN_PATH),
        "public_manifest_sha256": sha256(PUBLIC_PATH),
        "feature_manifest_sha256": provenance["feature_manifest_sha256"],
        "labels_manifest_sha256": provenance["labels_manifest_sha256"],
        "n_train_reactions": 617,
        "n_train_trajectories": len(train),
        "n_validation_reactions": 47,
        "n_validation_trajectories": len(validation),
        "baselines": baselines,
        "learning_curve": results,
        "paired_largest_prefix_comparisons": paired,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    summary_path = OUT / "phase20f_learning_curve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    freeze_files = [summary_path, PLAN_PATH] + sorted((OUT / "checkpoints").glob("**/*.pt"))
    freeze = {
        "status": "PHASE20F_MODELS_FROZEN_ON_OFFICIAL_TRAINVAL_WITH_TEST_UNTOUCHED",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "official_test_loaded": False,
        "files_sha256": {str(path): sha256(path) for path in freeze_files},
    }
    (OUT / "phase20f_freeze_manifest.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "midpoint_baseline": baselines["fixed_midpoint"]["reaction_macro_success"],
        "framewise_baseline": baselines["framewise_B3v1_near_prob_persistent9"]["reaction_macro_success"],
        "curve": {
            prefix: results[str(prefix)]["validation_three_seed_persistent9"]["reaction_macro_success"]
            for prefix in PREFIXES
        },
        "final_five_seed": results["1200"]["validation_five_seed_persistent9"]["reaction_macro_success"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
