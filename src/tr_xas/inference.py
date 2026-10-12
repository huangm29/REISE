"""Frozen demo and replaceable-checkpoint encoder/TCN inference."""
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, read_json, new_output, scoring_module, sha256, write_json

FEATURE_NAMES = ["s_exp", "s_std_total", "near_prob", "d_TS", "delta_E_R", "signed_delta_s"]


def validate_input(data):
    expected = {"query": (201, 3, 200), "endpoint": (2, 3, 200), "composition": (7,), "time_fs": (201,)}
    for key, shape in expected.items():
        if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"{key} must be finite with shape {shape}")
    if not np.allclose(np.diff(data["time_fs"]), 1.0, rtol=0, atol=1e-4):
        raise ValueError("This checkpoint pipeline expects 1 fs spacing and 201 frames; resampling is not automatic")
    if not np.isclose(np.sum(data["composition"][:4]), 1.0, atol=1e-5):
        raise ValueError("First four composition entries must sum to one")


def load_custom(encoder_paths, temporal_paths, device):
    ref = scoring_module()
    encoders, temporal, means, stds = [], [], [], []
    for path in encoder_paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("model", "B3") not in ("B3", "B3-internal"):
            raise ValueError("Pipeline requires a reaction-conditioned B3 encoder")
        model = ref.B3Model()
        model.load_state_dict(state["state_dict"], strict=True)
        encoders.append(model.to(device).eval())
    for path in temporal_paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = ref.TemporalDetector(12)
        model.load_state_dict(state["state_dict"], strict=True)
        temporal.append(model.to(device).eval())
        means.append(np.asarray(state["normalization_mean"], np.float32))
        stds.append(np.asarray(state["normalization_std"], np.float32))
        if means[-1].shape != (12,) or stds[-1].shape != (12,) or not np.isfinite(means[-1]).all() or not np.isfinite(stds[-1]).all() or not (stds[-1] > 0).all():
            raise ValueError("Invalid temporal normalization")
    return ref, encoders, temporal, means, stds


def predict_arrays(data, norm, ref, encoders, temporal, means, stds, device):
    validate_input(data)
    mean, std = np.asarray(norm["mean"], np.float32), np.asarray(norm["std"], np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Invalid spectral normalization")
    result = ref.score_b3(data["query"], data["endpoint"], data["composition"], mean, std, encoders, device)
    raw, smooth, spread = ref.score_tcn(result["tcn_features"], temporal, means, stds, device)
    values = {"features": np.stack([result[key] for key in FEATURE_NAMES], axis=-1),
              "tcn_features": result["tcn_features"], "tcn_raw": raw, "tcn_smoothed": smooth,
              "tcn_seed_std": spread, "time_fs": data["time_fs"]}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("Non-finite prediction")
    if (raw < 0).any() or (raw > 1).any():
        raise ValueError("Invalid score")
    return values


def demo(assets, output, device, strict):
    # Check only external arrays/weights; source baseline hashes are in source_manifest.json.
    records = read_json(ROOT / "reference/demo_assets.json")
    for row in records:
        if Path(row["path"]).suffix in (".npz", ".pt") or row["path"] == "tcn/freeze.json":
            if sha256(assets / row["path"]) != row["sha256"]:
                raise ValueError("Demo asset mismatch: " + row["path"])
    ref = scoring_module()
    ref.V1, ref.TCN_DIR, ref.TCN_FREEZE = assets / "encoder", assets / "tcn", assets / "tcn/freeze.json"
    device = torch.device(device)
    torch.set_num_threads(2)
    encoders, _ = ref.load_b3_models(device)
    temporal, means, stds, _ = ref.load_tcn_models(device)
    norm = np.load(assets / "normalization.npz", allow_pickle=False)
    out = new_output(output, [assets])
    report = {"device": str(device), "torch": torch.__version__, "numpy": np.__version__,
              "status": "PASS", "strict_comparison_requested": strict, "cases": []}
    for fraction in (0, 50, 100):
        data = np.load(assets / f"example_fraction_{fraction:03d}.npz", allow_pickle=False)
        values = predict_arrays(data, norm, ref, encoders, temporal, means, stds, device)
        differences, passes = {}, []
        for key, atol in [("features", 1e-4), ("tcn_raw", 1e-5), ("tcn_smoothed", 1e-5), ("tcn_seed_std", 1e-5)]:
            differences[key] = float(np.abs(values[key] - data["expected_" + key]).max())
            passes.append(differences[key] <= atol)
        index = int(values["tcn_smoothed"].argmax())
        passes.append(index == int(data["expected_tcn_smoothed"].argmax()))
        exact_pass = all(passes)
        if not exact_pass:
            report["status"] = "NUMERIC_DIFFERENCE"
        report["cases"].append({"fraction": fraction / 100, "argmax_time_fs": float(data["time_fs"][index]),
                                "maximum_score": float(values["tcn_smoothed"].max()),
                                "max_abs_difference": differences, "strict_tolerances_pass": exact_pass})
        np.savez_compressed(out / f"prediction_{fraction:03d}.npz", **values)
    write_json(out / "verification.json", report)
    if strict and report["status"] != "PASS":
        raise RuntimeError("Strict frozen-reference comparison failed; see verification.json. Tolerances were not changed.")
    return report


def predict(input_path, normalization, encoder_paths, tcn_paths, output, device):
    device = torch.device(device)
    torch.set_num_threads(2)
    ref, encoders, temporal, means, stds = load_custom(encoder_paths, tcn_paths, device)
    with np.load(input_path, allow_pickle=False) as data, np.load(normalization, allow_pickle=False) as norm:
        values = predict_arrays(data, norm, ref, encoders, temporal, means, stds, device)
    out = new_output(output)
    np.savez_compressed(out / "prediction.npz", **values)
    idx = int(values["tcn_smoothed"].argmax())
    write_json(out / "summary.json", {"candidate_event_time_fs": float(values["time_fs"][idx]),
               "maximum_score": float(values["tcn_smoothed"][idx]),
               "note": "Forced argmax, not an automatic reaction/nonreaction decision or fraction estimate",
               "sha256": {str(p): sha256(p) for p in [input_path, normalization, *encoder_paths, *tcn_paths]}})
    return out

