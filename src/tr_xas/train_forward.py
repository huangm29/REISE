"""Relocate paths only; call the collected joint atomic/molecular trainer."""
import copy
import sys
from .common import ROOT, load_module, new_output, read_json, record_run, require_files, write_json


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def data_files(folder):
    return [folder / ("merged_" + name + ".pkl") for name in ("total_xas_spec_200", "carbon_spectra", "nitrogen_spectra", "oxygen_spectra")]


def resolve_config(config, paths, output):
    stage = config["stage"]
    if stage not in ("pretrain", "finetune"):
        raise ValueError("stage must be pretrain or finetune")
    cfg = merge(read_json(ROOT / config["reference_config"]), config.get("overrides", {}))
    folder = paths["fg26" if stage == "pretrain" else "tr1x"]
    total, carbon, nitrogen, oxygen = data_files(folder)
    cfg["dataset"]["total_spec_path"] = str(total)
    cfg["dataset"]["atomic_spec_paths"] = dict(zip("CNO", map(str, (carbon, nitrogen, oxygen))))
    # Use a fresh per-run graph cache: old filenames do not encode dataset identity.
    cfg["dataset"]["root"] = str(output / "graph_cache")
    cfg["base_log_dir"] = str(output / "logs")
    cfg["name"] = "fluxxas_" + stage
    cfg["load_checkpoint_path"] = str(paths["fg26_checkpoint"]) if stage == "finetune" else None
    cfg["finetune"], cfg["resume"] = stage == "finetune", False
    cfg.pop("session_name", None)
    if cfg["dataset"].get("num_workers", 0) == 0:
        cfg["dataset"]["persistent_workers"] = False
    if "learning_rate" in cfg["optimizer"]:
        raise ValueError("Forward AdamW uses optimizer.lr, not optimizer.learning_rate")
    return cfg


def run(config, paths, output):
    folder = paths["fg26" if config["stage"] == "pretrain" else "tr1x"]
    require_files(data_files(folder))
    if config["stage"] == "finetune":
        require_files([paths["fg26_checkpoint"]])
    from omegaconf import OmegaConf
    sys.path.insert(0, str(ROOT / "vendor/fluxxas"))
    trainer = load_module("tr_xas_joint_trainer", ROOT / "vendor/fluxxas/joint_training/train.py")
    out = new_output(output, [folder])
    resolved = resolve_config(config, paths, out)
    record_run(out, config, {"spectra": folder})
    write_json(out / "resolved_config.json", resolved)
    # Direct construction avoids upstream argparse overriding the output directory
    # and avoids its learning_rate/lr spelling mismatch.
    trainer.JointTrainer(OmegaConf.create(resolved)).train()
    return out
