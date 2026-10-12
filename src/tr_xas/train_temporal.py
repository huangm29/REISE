"""One prefix of the original TCN training protocol, with independent output paths."""
import numpy as np
import torch
from .common import new_output, record_run, temporal_module, sha256, write_json


def run(config, paths, output):
    base = paths["phase20"]
    module = temporal_module(base)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1)
    train, validation, provenance = module.load_units()
    prefix = config["prefix"]
    if prefix not in module.PREFIXES:
        raise ValueError("Use one of the frozen nested prefixes")
    selected = [u for u in train if u["nested_order"] < prefix]
    reactions = sorted({u["reaction"] for u in selected})
    if len(reactions) != module.EXPECTED_REACTION_COUNTS[module.PREFIXES.index(prefix)]:
        raise ValueError("Training prefix cohort mismatch")
    out = new_output(output, [base])
    record_run(out, config, {"phase20": base})
    write_json(out / "input_provenance.json", provenance)
    module.EPOCHS, module.BATCH_SIZE, module.LEARNING_RATE = config["epochs"], config["batch_size"], config["learning_rate"]
    x = np.stack([u["x"] for u in selected])
    y = np.stack([u["labels"] for u in selected])
    ds = np.stack([u["abs_ds"] for u in selected])
    mean = x.reshape(-1, 12).mean(axis=0)
    std = x.reshape(-1, 12).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    val_x = np.stack([u["x"] for u in validation])
    score_list, summary = [], {"split": "val", "seeds": {}}
    for seed in config["seeds"]:
        model, history = module.train_seed(seed, x, y, ds, mean, std, device, f"prefix{prefix}")
        checkpoint = out / f"seed{seed}.pt"
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "seed": seed, "prefix": prefix, "n_training_reactions": len(reactions),
                    "n_training_trajectories": len(selected), "training_reactions": reactions,
                    "normalization_mean": mean, "normalization_std": std,
                    "plan_sha256": sha256(module.PLAN_PATH), "config": config}, checkpoint)
        raw, _ = module.score_models([model], val_x, mean, std, device)
        if not np.isfinite(raw).all():
            raise FloatingPointError("Non-finite TCN predictions")
        score_list.append(raw)
        met, rows = module.summarize(validation, module.persistent9(raw))
        summary["seeds"][str(seed)] = {"checkpoint_sha256": sha256(checkpoint), "history": history, "metrics": met}
        write_json(out / f"validation_seed{seed}.json", rows)
        write_json(out / "validation_summary.json", summary)
    raw = np.mean(np.stack(score_list), axis=0)
    smooth = module.persistent9(raw)
    summary["ensemble"], rows = module.summarize(validation, smooth)
    summary["framewise"], _ = module.summarize(validation, module.persistent9(val_x[:, :, 2]))
    np.savez_compressed(out / "validation_predictions.npz", raw=raw, smoothed=smooth,
                        trajectory_ids=np.asarray([u["trajectory_id"] for u in validation], dtype=str))
    write_json(out / "validation_rows.json", rows)
    write_json(out / "validation_summary.json", summary)
    return out

