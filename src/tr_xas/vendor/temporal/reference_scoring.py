#!/usr/bin/env python3
"""Truth-free frozen RC-SMP, B3v1, and Phase-20 TCN scoring for Phase 22."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


XASNET = Path("data/archive/XASNet")
V1 = XASNET / "t1x_xas_dataset_v2/spectral_reaction_path_v1"
V2H = XASNET / "t1x_xas_dataset_v2/spectral_reaction_path_v2_history"
PHASE20 = XASNET / "trajectory_inference/phase20/phase20d_20260821_130436"
TCN_DIR = PHASE20 / "phase20f_tcn_learning_curve/checkpoints/prefix1200"
TCN_FREEZE = PHASE20 / "phase20f_tcn_learning_curve/phase20f_freeze_manifest.json"
EXPECTED_TCN_FREEZE_SHA = "f9441578a2d25c23b553da81294d31badad89f9df100c2def2eed714cb30288e"
EXPECTED_PUBLIC_SHA = "51b0c80163c510acf3b0d7b404978a4a56bea14c20b4d7df045374de36adcf50"
EXPECTED_FORWARD_MANIFEST_SHA = "581a23898266e7c3266caf8d9c560ab91f78913a59f2b2afa6479a04581d7ae3"
EXPECTED_ENDPOINT_SHA = "fdb5dbf6cd8c420c1b001d0c2fd2a7583ab83897ce3529c5c8edc99325242a63"
EXPECTED_NORM_SHA = "1e16b42130884e8e44526921ad55d0606bdbd2d4794b27d833bb9756d169a739"
MIN_FREE_GIB = 10.0
S_GRID = np.linspace(0.0, 1.0, 101, dtype=np.float32)
BIN_CENTERS = np.arange(20, dtype=np.float32) / 20.0 + 0.025
PERSISTENCE_WIDTH = 9

sys.path.insert(0, str(V1 / "scripts"))
from taskA_models import PathResNet  # noqa: E402
from taskB_models import B3Model  # noqa: E402


A3_SHA = {
    0: "f57c29bf7d1d10d9528cf746610745c79110db86d2f8831c825f4e8aa7adaefd",
    1: "f4feacc5845694e2f129f91251ee97c338e919abb5f8a2dfb8e255ee76558a03",
    2: "69c0f201c1e8ee12855b6a0555e7b0900936d0c0ddf87a67982e448e865b4c16",
    3: "9cd01fa5975a6d1db7576e26a2367d47da216b7618452834db51916b8f3eaa4e",
    4: "e715e133b1fa0d87736ed0df32eba7f273ed8230995096ae1d030785ff2b4ac4",
}
B3_SHA = {
    0: "dc8168521cf6b828ef9b7f0391da764326929220d5515dbb2f8903c4fc96fd0e",
    1: "10c59a78a2a2f3a5c9655fb4a5013905ad6e09fcabb3d770fe3af2a5333a0d4c",
    2: "a3af50759ab2c9f9d1636b5aa4a5dbfa90a27c0eb394976db3533a784109bac9",
    3: "f09193378572ef41f0c82d12348a992c5754ad6764197d3c077da13ea2e48f48",
    4: "890b4a9584faba56ebc135547174f39b773cbc3c607ea0052c139b05820f54b1",
}
TCN_SHA = {
    0: "332cf96af6053f9c4a41a8e7968c441adb4f76e578757a81875234cdda121888",
    1: "fbfcfb3d2c9ba8ff77cbcf5d913aa11e727ac18a9b24aa0fe75a24cafa5ea73a",
    2: "325f5cf2406a0b32fcec0055c581632bddc7ecfd354de9248aaaa4f984f44340",
    3: "47357a82220f5b52030e6de07435ca1cd667bdef924a70be951aee1f71e00483",
    4: "dc71f337ae8ae8dea69cca965d7fb9609f9e93194b7527b904d2d552035c742d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def require_disk(path: Path) -> float:
    free = shutil.disk_usage(path).free / (1024 ** 3)
    if free < MIN_FREE_GIB:
        raise RuntimeError(f"Disk safety stop: {free:.2f} GiB < {MIN_FREE_GIB:.2f} GiB")
    return free


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


def persistent9(scores: np.ndarray) -> np.ndarray:
    radius = PERSISTENCE_WIDTH // 2
    padded = np.pad(scores, (radius, radius), mode="reflect")
    kernel = np.ones(PERSISTENCE_WIDTH, np.float64) / PERSISTENCE_WIDTH
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def cosine_residual(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = np.sum(query * reference, axis=-1)
    denominator = np.linalg.norm(query, axis=-1) * np.linalg.norm(reference, axis=-1)
    return 1.0 - numerator / np.maximum(denominator, 1.0e-12)


def make_a3_curves(
    endpoint_raw: np.ndarray,
    composition: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, str]]:
    n_groups = len(endpoint_raw)
    present = (np.abs(endpoint_raw).sum(axis=3) > 0).astype(np.float32)
    normalized = ((endpoint_raw - mean[None, None, :, None]) / std[None, None, :, None])
    normalized *= present[:, :, :, None]
    group_repeat = np.repeat(np.arange(n_groups), len(S_GRID))
    s_repeat = np.tile(S_GRID, n_groups)
    curve_sum = np.zeros((n_groups * len(S_GRID), 3, 200), np.float32)
    hashes = {}
    for seed in range(5):
        checkpoint_path = V1 / f"checkpoints/taskA_A3_seed{seed}.pt"
        actual = sha256(checkpoint_path)
        if actual != A3_SHA[seed]:
            raise RuntimeError(f"A3 checkpoint mismatch: seed{seed}")
        hashes[checkpoint_path.name] = actual
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = PathResNet()
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        for start in range(0, len(group_repeat), 2048):
            stop = min(start + 2048, len(group_repeat))
            selected = group_repeat[start:stop]
            xr = torch.from_numpy(normalized[selected, 0]).to(device)
            xp = torch.from_numpy(normalized[selected, 1]).to(device)
            comp = torch.from_numpy(composition[selected]).to(device)
            s = torch.from_numpy(s_repeat[start:stop]).to(device)
            with torch.no_grad():
                residual = model(xr, xp, comp, s)
                s_view = s[:, None, None]
                prediction = (1.0 - s_view) * xr + s_view * xp + s_view * (1.0 - s_view) * residual
            curve_sum[start:stop] += prediction.cpu().numpy()
        del model
        torch.cuda.empty_cache()
        print(f"A3_CURVE seed={seed} complete", flush=True)
    curve_norm = (curve_sum / 5.0).reshape(n_groups, len(S_GRID), 3, 200)
    curve_raw = curve_norm * std[None, None, :, None] + mean[None, None, :, None]
    curve_raw *= present[:, :1, :, None]
    return curve_raw.astype(np.float32), hashes


def load_b3_models(device: torch.device) -> tuple[list[B3Model], dict[str, str]]:
    models = []
    hashes = {}
    for seed in range(5):
        checkpoint_path = V1 / f"checkpoints/taskB_B3_seed{seed}.pt"
        actual = sha256(checkpoint_path)
        if actual != B3_SHA[seed]:
            raise RuntimeError(f"B3 checkpoint mismatch: seed{seed}")
        hashes[checkpoint_path.name] = actual
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = B3Model()
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
    return models, hashes


def load_tcn_models(device: torch.device) -> tuple[list[TemporalDetector], list[np.ndarray], list[np.ndarray], dict[str, str]]:
    if sha256(TCN_FREEZE) != EXPECTED_TCN_FREEZE_SHA:
        raise RuntimeError("Phase-20 TCN freeze-manifest mismatch")
    freeze = json.loads(TCN_FREEZE.read_text(encoding="utf-8"))
    if freeze["status"] != "PHASE20F_MODELS_FROZEN_ON_OFFICIAL_TRAINVAL_WITH_TEST_UNTOUCHED":
        raise RuntimeError("Unexpected TCN freeze status")
    if freeze["official_test_loaded"]:
        raise RuntimeError("Frozen TCN reports test contamination")
    models, means, stds, hashes = [], [], [], {}
    for seed in range(5):
        checkpoint_path = TCN_DIR / f"seed{seed}.pt"
        actual = sha256(checkpoint_path)
        if actual != TCN_SHA[seed]:
            raise RuntimeError(f"TCN checkpoint mismatch: seed{seed}")
        hashes[checkpoint_path.name] = actual
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["prefix"] != 1200 or checkpoint["n_training_reactions"] != 617:
            raise RuntimeError("Unexpected frozen TCN training cohort")
        model = TemporalDetector(12)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
        means.append(np.asarray(checkpoint["normalization_mean"], np.float32))
        stds.append(np.asarray(checkpoint["normalization_std"], np.float32))
    return models, means, stds, hashes


def score_b3(
    raw_xas: np.ndarray,
    endpoint_raw: np.ndarray,
    composition: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    models: list[B3Model],
    device: torch.device,
) -> dict[str, np.ndarray]:
    present = (np.abs(raw_xas).sum(axis=2) > 0).astype(np.float32)
    query_norm = ((raw_xas - mean[None, :, None]) / std[None, :, None]) * present[:, :, None]
    endpoint_present = (np.abs(endpoint_raw).sum(axis=2) > 0).astype(np.float32)
    endpoint_norm = ((endpoint_raw - mean[None, :, None]) / std[None, :, None]) * endpoint_present[:, :, None]
    n_frames = len(raw_xas)
    x_input = np.stack([
        np.repeat(endpoint_norm[0][None], n_frames, axis=0),
        query_norm,
        np.repeat(endpoint_norm[1][None], n_frames, axis=0),
    ], axis=1).astype(np.float32)
    presence = np.stack([
        np.repeat(endpoint_present[0][None], n_frames, axis=0),
        present,
        np.repeat(endpoint_present[1][None], n_frames, axis=0),
    ], axis=1).astype(np.float32)
    composition_batch = np.repeat(composition[None], n_frames, axis=0).astype(np.float32)
    x_t = torch.from_numpy(x_input).to(device)
    present_t = torch.from_numpy(presence).to(device)
    composition_t = torch.from_numpy(composition_batch).to(device)
    probabilities, near, d_ts, delta_e, signed = [], [], [], [], []
    with torch.no_grad():
        for model in models:
            output = model(x_t, present_t, composition_t)
            probabilities.append(torch.softmax(output["s_logits"], dim=1).cpu().numpy())
            near.append(torch.sigmoid(output["near_logit"]).cpu().numpy().reshape(-1))
            d_ts.append(output["d_TS"].cpu().numpy().reshape(-1))
            delta_e.append(output["delta_E_R"].cpu().numpy().reshape(-1))
            signed.append(output["signed_delta_s"].cpu().numpy().reshape(-1))
    p_seed = np.stack(probabilities)
    p_bar = p_seed.mean(axis=0)
    p_bar /= p_bar.sum(axis=1, keepdims=True)
    s_exp = p_bar @ BIN_CENTERS
    variance = (p_bar * BIN_CENTERS[None] ** 2).sum(axis=1) - s_exp ** 2
    result = {
        "s_exp": s_exp.astype(np.float32),
        "s_std_total": np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
        "near_prob": np.mean(near, axis=0).astype(np.float32),
        "d_TS": np.mean(d_ts, axis=0).astype(np.float32),
        "delta_E_R": np.mean(delta_e, axis=0).astype(np.float32),
        "signed_delta_s": np.mean(signed, axis=0).astype(np.float32),
    }
    raw_features = np.stack([
        result["s_exp"], result["s_std_total"], result["near_prob"],
        result["d_TS"], result["delta_E_R"], result["signed_delta_s"],
    ], axis=1).astype(np.float32)
    difference = np.zeros_like(raw_features)
    difference[1:] = raw_features[1:] - raw_features[:-1]
    result["tcn_features"] = np.concatenate([raw_features, difference], axis=1)
    return result


def score_tcn(
    features: np.ndarray,
    models: list[TemporalDetector],
    means: list[np.ndarray],
    stds: list[np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_scores = []
    with torch.no_grad():
        for model, mean, std in zip(models, means, stds):
            normalized = ((features - mean) / std).astype(np.float32)
            tensor = torch.from_numpy(normalized[None]).to(device).transpose(1, 2)
            seed_scores.append(torch.sigmoid(model(tensor))[0].cpu().numpy())
    per_seed = np.stack(seed_scores).astype(np.float32)
    raw = per_seed.mean(axis=0).astype(np.float32)
    return raw, persistent9(raw), per_seed.std(axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="phase22c_truthfree_scores")
    parser.add_argument("--limit-trajectories", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    public_path = run_dir / "blinded_xas_input/public_manifest.json"
    forward_dir = run_dir / "phase22b_forward_xas"
    forward_manifest_path = forward_dir / "forward_manifest.json"
    endpoint_path = forward_dir / "phase22_endpoint_xas.npz"
    if sha256(public_path) != EXPECTED_PUBLIC_SHA:
        raise RuntimeError("Frozen public-manifest mismatch")
    if sha256(forward_manifest_path) != EXPECTED_FORWARD_MANIFEST_SHA:
        raise RuntimeError("Frozen forward-manifest mismatch")
    if sha256(endpoint_path) != EXPECTED_ENDPOINT_SHA:
        raise RuntimeError("Frozen endpoint-XAS mismatch")
    public = json.loads(public_path.read_text(encoding="utf-8"))
    forward = json.loads(forward_manifest_path.read_text(encoding="utf-8"))
    if public["status"] != "PHASE22_OPAQUE_WINDOWS_READY_FOR_XAS":
        raise RuntimeError("Unexpected public status")
    if public["contains_reaction_identity"] or public["contains_fraction_or_time_or_structural_truth"]:
        raise RuntimeError("Forbidden truth in public input")
    if forward["status"] != "PHASE22B_FORWARD_COMPLETE_AND_FROZEN_BEFORE_TRUTH_JOIN":
        raise RuntimeError("Forward XAS is not frozen")
    if forward["private_truth_accessed"] or forward["fraction_time_or_structural_truth_accessed"]:
        raise RuntimeError("Forward provenance reports truth access")
    units = list(public["trajectories"])
    if args.limit_trajectories:
        units = units[:args.limit_trajectories]
    selected_groups = sorted({unit["group_id"] for unit in units})

    endpoint = np.load(endpoint_path, allow_pickle=True)
    all_group_ids = endpoint["group_ids"].astype(str).tolist()
    all_group_index = {group: index for index, group in enumerate(all_group_ids)}
    group_indices = np.asarray([all_group_index[group] for group in selected_groups])
    endpoint_raw = np.stack([
        np.asarray(endpoint[f"endpoint_XAS_{edge}"], np.float32)[group_indices] for edge in "CNO"
    ], axis=2)
    composition = np.asarray(endpoint["composition"], np.float32)[group_indices]
    group_to_local = {group: index for index, group in enumerate(selected_groups)}
    if endpoint_raw.shape != (len(selected_groups), 2, 3, 200):
        raise RuntimeError("Endpoint XAS shape mismatch")

    norm_path = V2H / "predictions/_train_norm_stats.npz"
    if sha256(norm_path) != EXPECTED_NORM_SHA:
        raise RuntimeError("Training normalization-statistics mismatch")
    norm = np.load(norm_path)
    mean = norm["mean"].astype(np.float32)
    std = norm["std"].astype(np.float32)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(1)

    output_dir = run_dir / args.output_name
    if output_dir.exists() and not args.resume:
        raise RuntimeError(f"Output exists without --resume: {output_dir}")
    output_dir.mkdir(exist_ok=True)
    scores_dir = output_dir / "trajectory_scores"
    scores_dir.mkdir(exist_ok=True)
    initial_free = require_disk(run_dir)
    script_sha = sha256(Path(__file__))
    launch_path = output_dir / "score_launch_manifest.json"
    if not launch_path.exists():
        launch = {
            "status": "PHASE22C_MODELS_AND_RULES_FROZEN_BEFORE_TRUTH_FREE_SCORING",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": script_sha,
            "public_manifest_sha256": EXPECTED_PUBLIC_SHA,
            "forward_manifest_sha256": EXPECTED_FORWARD_MANIFEST_SHA,
            "endpoint_xas_sha256": EXPECTED_ENDPOINT_SHA,
            "tcn_freeze_manifest_sha256": EXPECTED_TCN_FREEZE_SHA,
            "n_trajectories_requested": len(units),
            "n_groups_requested": len(selected_groups),
            "limit_trajectories": args.limit_trajectories,
            "rcsmp_rule": "argmin fixed 101-point s grid of mean C/N/O cosine residual",
            "tcn_rule": "five-seed prefix1200 ensemble followed by unchanged 9-fs persistence mean",
            "b3_rule": "five-seed B3v1 ensemble",
            "fixed_midpoint_index": 100,
            "private_truth_accessed": False,
            "reaction_fraction_time_or_structural_truth_accessed": False,
            "initial_free_space_GiB": initial_free,
        }
        atomic_json(launch_path, launch)
    else:
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        if launch["script_sha256"] != script_sha or launch["n_trajectories_requested"] != len(units):
            raise RuntimeError("Score-resume provenance mismatch")

    started = time.time()
    curves, a3_hashes = make_a3_curves(endpoint_raw, composition, mean, std, device)
    b3_models, b3_hashes = load_b3_models(device)
    tcn_models, tcn_means, tcn_stds, tcn_hashes = load_tcn_models(device)
    progress_path = output_dir / "score_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not args.resume or progress["script_sha256"] != script_sha:
            raise RuntimeError("Score progress exists without valid resume")
    else:
        progress = {"status": "RUNNING", "script_sha256": script_sha, "completed": {}}

    forward_index = {row["trajectory_id"]: row for row in forward["trajectory_outputs"]}
    for unit_index, unit in enumerate(units):
        trajectory_id = unit["trajectory_id"]
        forward_row = forward_index[trajectory_id]
        xas_path = forward_dir / forward_row["output_npz"]
        if sha256(xas_path) != forward_row["output_npz_sha256"]:
            raise RuntimeError(f"Forward trajectory hash mismatch: {trajectory_id}")
        output_path = scores_dir / f"{trajectory_id}_scores.npz"
        if trajectory_id in progress["completed"]:
            if sha256(output_path) != progress["completed"][trajectory_id]["score_npz_sha256"]:
                raise RuntimeError(f"Score-resume output mismatch: {trajectory_id}")
            continue
        with np.load(xas_path, allow_pickle=True) as data:
            frame_ids = data["frame_ids"].astype(str)
            raw_xas = np.stack([np.asarray(data[f"XAS_mol_{edge}"], np.float32) for edge in "CNO"], axis=1)
        if raw_xas.shape != (201, 3, 200) or len(set(frame_ids.tolist())) != 201:
            raise RuntimeError(f"Trajectory XAS shape/ID mismatch: {trajectory_id}")
        group_index = group_to_local[unit["group_id"]]
        b3 = score_b3(
            raw_xas, endpoint_raw[group_index], composition[group_index], mean, std, b3_models, device
        )
        tcn_raw, tcn_persistent, tcn_std = score_tcn(
            b3.pop("tcn_features"), tcn_models, tcn_means, tcn_stds, device
        )
        edge_residual = cosine_residual(raw_xas[:, None, :, :], curves[group_index][None, :, :, :])
        mean_residual = edge_residual.mean(axis=2)
        best = np.argmin(mean_residual, axis=1)
        s_hat = S_GRID[best]
        min_residual = mean_residual[np.arange(201), best]
        edge_best = edge_residual[np.arange(201), best]
        b3_persistent = persistent9(b3["near_prob"])
        values = [s_hat, min_residual, edge_best, tcn_raw, tcn_persistent, tcn_std, b3_persistent]
        values.extend(b3.values())
        if not all(np.isfinite(value).all() for value in values):
            raise RuntimeError(f"Non-finite score: {trajectory_id}")
        np.savez_compressed(
            output_path,
            frame_ids=frame_ids,
            rcsmp_s_hat=s_hat.astype(np.float32),
            rcsmp_d_manifold=min_residual.astype(np.float32),
            rcsmp_edge_residual_at_s_hat=edge_best.astype(np.float32),
            tcn_probability_raw=tcn_raw,
            tcn_probability_persistent9=tcn_persistent,
            tcn_epistemic_std=tcn_std,
            b3_near_prob_persistent9=b3_persistent,
            **{f"b3_{key}": value for key, value in b3.items()},
        )
        progress["completed"][trajectory_id] = {
            "group_id": unit["group_id"],
            "score_npz": str(output_path.relative_to(output_dir)),
            "score_npz_sha256": sha256(output_path),
            "rcsmp_progress_span_q95_minus_q05": float(np.quantile(s_hat, 0.95) - np.quantile(s_hat, 0.05)),
            "rcsmp_mean_d_manifold": float(np.mean(min_residual)),
            "tcn_max_persistent9": float(np.max(tcn_persistent)),
            "tcn_argmax_persistent9_index": int(np.argmax(tcn_persistent)),
            "b3_max_near_prob_persistent9": float(np.max(b3_persistent)),
            "b3_argmax_near_prob_persistent9_index": int(np.argmax(b3_persistent)),
            "fixed_midpoint_index": 100,
        }
        atomic_json(progress_path, progress)
        if (unit_index + 1) % 25 == 0 or unit_index + 1 == len(units):
            free = require_disk(run_dir)
            print(f"SCORE {unit_index + 1}/{len(units)} free_GiB={free:.2f}", flush=True)

    completed = progress["completed"]
    if len(completed) != len(units):
        raise RuntimeError("Final score count mismatch")
    final_free = require_disk(run_dir)
    final = {
        **launch,
        "status": "PHASE22C_ALL_SCORES_COMPLETE_AND_FROZEN_BEFORE_TRUTH_JOIN",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "n_groups": len(selected_groups),
        "n_trajectories": len(units),
        "n_frames": len(units) * 201,
        "A3_checkpoint_sha256": a3_hashes,
        "B3v1_checkpoint_sha256": b3_hashes,
        "TCN_checkpoint_sha256": tcn_hashes,
        "norm_stats_sha256": EXPECTED_NORM_SHA,
        "score_outputs": [completed[unit["trajectory_id"]] | {"trajectory_id": unit["trajectory_id"]}
                          for unit in units],
        "walltime_s": round(time.time() - started, 2),
        "final_free_space_GiB": final_free,
        "private_truth_accessed": False,
        "reaction_fraction_time_or_structural_truth_accessed": False,
    }
    final_path = output_dir / "score_manifest.json"
    atomic_json(final_path, final)
    sums_path = output_dir / "SHA256SUMS.txt"
    rows = [
        f"{sha256(launch_path)}  {launch_path.name}",
        f"{sha256(progress_path)}  {progress_path.name}",
        f"{sha256(final_path)}  {final_path.name}",
    ]
    rows.extend(
        f"{completed[unit['trajectory_id']]['score_npz_sha256']}  trajectory_scores/{unit['trajectory_id']}_scores.npz"
        for unit in units
    )
    sums_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    for path in [launch_path, progress_path, final_path, sums_path, *scores_dir.glob("*.npz")]:
        os.chmod(path, 0o444)
    print(json.dumps({
        key: final[key] for key in ("status", "n_groups", "n_trajectories", "n_frames", "walltime_s", "final_free_space_GiB")
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
