"""Train/validate with original heads and loss; held-out evaluation is separate."""
import numpy as np
import torch
from .common import ROOT, encoder_modules, new_output, record_run, require_files, sha256, write_json, read_json


def required_files(paths, splits=("train", "val")):
    return [paths["mep"] / f"{stem}_{split}.pkl" for split in splits for stem in ("mep", "mep_labels")] + [paths["labels"] / f"rc_labels_{split}.pkl" for split in splits]


def prepare(paths, splits=("train", "val"), normalization=None):
    sd, models, train = encoder_modules()
    sd.DATA_DIR, sd.LABELS_DIR = paths["mep"], paths["labels"]
    require_files(required_files(paths, splits))
    records = {split: sd.load_reactions(split) for split in splits}
    split_manifest = read_json(ROOT / "reference/mep_split_manifest.json")
    for split, rows in records.items():
        if set(rows) != set(split_manifest[split]):
            raise ValueError("Reaction split mismatch: " + split)
    if "train" in records and "val" in records:
        if {v["formula"] for v in records["train"].values()} & {v["formula"] for v in records["val"].values()}:
            raise ValueError("Training/validation formulas overlap")
    mean, std = normalization if normalization is not None else sd.compute_norm(records["train"])
    return train, models, {split: train.build_split_arrays(rows, mean, std) for split, rows in records.items()}, mean, std


def scope_indices(arr, scope):
    idx = np.arange(len(arr["rxn_ids"]) * 10)
    if scope == "internal8":
        return idx[(idx % 10 >= 1) & (idx % 10 <= 8)]
    if scope != "all10":
        raise ValueError("frame_scope must be all10 or internal8")
    return idx


def metrics(pred, arr, indices):
    per_frame = np.abs(pred["s_exp"] - arr["labels"]["s"].reshape(-1))
    selected_rxn = indices // 10
    per_rxn = [float(per_frame[indices[selected_rxn == r]].mean()) for r in range(len(arr["rxn_ids"]))]
    out = {"s_MAE": float(per_frame[indices].mean()), "per_reaction_s_MAE": dict(zip(arr["rxn_ids"], per_rxn))}
    for key in ("signed_delta_s", "d_TS", "delta_E_R"):
        values = pred[key][indices]
        if np.isfinite(values).all():
            out[key + "_MAE"] = float(np.abs(values - arr["labels"][key].reshape(-1)[indices]).mean())
    from sklearn.metrics import roc_auc_score, average_precision_score
    true = arr["labels"]["near_TS_10"].reshape(-1)[indices]
    prob = pred["near_prob"][indices]
    if len(np.unique(true)) == 2:
        out["near_AUROC"] = float(roc_auc_score(true, prob))
        out["near_AUPRC"] = float(average_precision_score(true, prob))
    return out


def run(config, paths, output):
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly choose cpu for a development smoke test")
    torch.set_num_threads(2)
    train, models, arrays, mean, std = prepare(paths)
    out = new_output(output, [paths["mep"], paths["labels"]])
    record_run(out, config, {"mep": paths["mep"], "labels": paths["labels"]})
    write_json(out / "input_hashes.json", {str(p): sha256(p) for p in required_files(paths)})
    np.savez(out / "normalization.npz", mean=mean, std=std)
    train.BATCH = config["batch_size"]
    order_train = scope_indices(arrays["train"], config["frame_scope"])
    order_val = scope_indices(arrays["val"], config["frame_scope"])
    model_name = config["model"]
    model_class = {"B1": models.B1Model, "B2": models.B2Model, "B3": models.B3Model}[model_name]
    summary = {"split": "val", "frame_scope": config["frame_scope"], "seeds": {}}
    for seed in config["seeds"]:
        train.seed_everything(seed)
        model = model_class().to(device)
        if config.get("initial_checkpoint"):
            ckpt = torch.load(config["initial_checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["state_dict"], strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        best, best_state, bad, history = float("inf"), None, 0, []
        for epoch in range(config["max_epochs"]):
            order = np.random.default_rng(seed * 1000 + epoch).permutation(order_train)
            tr = train.run_epoch(model, model_name, arrays["train"], order, device, optimizer)
            val = train.run_epoch(model, model_name, arrays["val"], order_val, device)
            if not np.isfinite([tr, val]).all():
                raise FloatingPointError("Non-finite encoder loss")
            history.append({"epoch": epoch, "train_loss": tr, "val_loss": val})
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"encoder seed={seed} epoch={epoch + 1} val={val:.6f}", flush=True)
            if bad >= config["patience"]:
                break
        model.load_state_dict(best_state)
        checkpoint = out / f"seed{seed}.pt"
        torch.save({"state_dict": best_state, "model": model_name, "seed": seed,
                    "norm_mean": mean, "norm_std": std, "frame_scope": config["frame_scope"],
                    "val_loss": best, "config": config}, checkpoint)
        pred = train.predict_model(model, model_name, arrays["val"], device)
        train.save_predictions(out / f"validation_seed{seed}.pkl", model_name, arrays["val"], pred)
        summary["seeds"][str(seed)] = {"checkpoint_sha256": sha256(checkpoint), "history": history,
                                      "metrics": metrics(pred, arrays["val"], order_val)}
        write_json(out / "validation_summary.json", summary)
    return out


def evaluate(checkpoint, paths, output, split, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "norm_mean" not in ckpt or "norm_std" not in ckpt:
        raise ValueError("This evaluator expects a new training checkpoint containing normalization; use demo for frozen paper weights")
    train, models, arrays, _, _ = prepare(paths, (split,), (ckpt["norm_mean"], ckpt["norm_std"]))
    name = ckpt["model"]
    model = {"B1": models.B1Model, "B2": models.B2Model, "B3": models.B3Model}[name]().to(device)
    model.load_state_dict(ckpt["state_dict"])
    pred = train.predict_model(model, name, arrays[split], device)
    out = new_output(output, [paths["mep"], paths["labels"], checkpoint.parent])
    train.save_predictions(out / "predictions.pkl", name, arrays[split], pred)
    result = {"split": split, "checkpoint_sha256": sha256(checkpoint), "frame_scope": ckpt["frame_scope"],
              "metrics": metrics(pred, arrays[split], scope_indices(arrays[split], ckpt["frame_scope"]))}
    write_json(out / "summary.json", result)
    return out

