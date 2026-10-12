import argparse
import json
from pathlib import Path
from .common import ROOT, read_json, read_paths, require_paths, require_files, sha256


def build_parser():
    p = argparse.ArgumentParser(description="TR-XAS: explicit training, scoring and offline asset checks")
    from . import __version__
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Read-only source or external-data preflight")
    check.add_argument("--paths", type=Path)
    check.add_argument("--target", choices=["sources", "encoder", "tcn", "forward", "demo"], default="sources")
    for kind in ("encoder", "tcn", "forward"):
        train = sub.add_parser("train-" + kind)
        train.add_argument("--config", required=True, type=Path)
        train.add_argument("--paths", required=True, type=Path)
        train.add_argument("--out", required=True, type=Path)
        train.add_argument("--execute", action="store_true", help="Without this flag, only validate paths/config and print the plan")
    demo = sub.add_parser("demo", help="Run three stored spectra through frozen weights")
    demo.add_argument("--assets", required=True, type=Path)
    demo.add_argument("--out", required=True, type=Path)
    demo.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    demo.add_argument("--strict", action="store_true")
    pred = sub.add_parser("predict", help="Use supplied compatible B3/TCN checkpoints on a 201-frame NPZ")
    for name in ("input", "normalization", "out"):
        pred.add_argument("--" + name, required=True, type=Path)
    pred.add_argument("--encoder", required=True, nargs="+", type=Path)
    pred.add_argument("--tcn", required=True, nargs="+", type=Path)
    pred.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ev = sub.add_parser("evaluate-encoder", help="Evaluate a new adapter checkpoint; validation is default")
    for name in ("checkpoint", "paths", "out"):
        ev.add_argument("--" + name, required=True, type=Path)
    ev.add_argument("--split", choices=["val", "test"], default="val")
    ev.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ev.add_argument("--allow-heldout-test", action="store_true")
    forward = sub.add_parser("forward", help="Geometry NPZ to molecular C/N/O spectra")
    for name in ("input", "checkpoint", "out"):
        forward.add_argument("--" + name, required=True, type=Path)
    forward.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return p


def preflight(target, paths):
    if target == "encoder":
        require_paths(paths, ["mep", "labels"])
        files = [paths["mep"] / f"{stem}_{sp}.pkl" for sp in ("train", "val") for stem in ("mep", "mep_labels")]
        files += [paths["labels"] / f"rc_labels_{sp}.pkl" for sp in ("train", "val")]
    elif target == "tcn":
        require_paths(paths, ["phase20"])
        files = [paths["phase20"] / name for name in ("input/phase20f_tcn_learning_curve_plan.json", "input/blinded_xas_input/public_manifest.json", "b3_seed_features/feature_manifest.json", "minimal_training_truth/minimal_labels_manifest.json")]
    elif target == "demo":
        require_paths(paths, ["demo"])
        files = [paths["demo"] / "normalization.npz", paths["demo"] / "example_fraction_100.npz"]
    else:
        require_paths(paths, ["tr1x"])
        files = [paths["tr1x"] / f"merged_{stem}.pkl" for stem in ("carbon_spectra", "nitrogen_spectra", "oxygen_spectra", "total_xas_spec_200")]
    require_files(files)
    return {"status": "FILES_PRESENT", "target": target, "files": [str(p) for p in files],
            "note": "Presence check; training loaders perform content, split and alignment checks"}


def validate_config(config, target):
    if target in ("encoder", "tcn"):
        if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
            raise ValueError("Provide distinct seeds")
        for key in ("batch_size", "learning_rate", "max_epochs" if target == "encoder" else "epochs"):
            if config[key] <= 0:
                raise ValueError(key + " must be positive")
        if config["device"] not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
    if target == "encoder":
        if config["model"] not in ("B1", "B2", "B3") or config["frame_scope"] not in ("all10", "internal8"):
            raise ValueError("Invalid encoder model/scope")
        if config["patience"] <= 0:
            raise ValueError("patience must be positive")
    if target == "tcn" and config["prefix"] not in [50, 100, 250, 500, 800, 1200]:
        raise ValueError("Invalid frozen training prefix")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "check":
        if args.target == "sources":
            records = read_json(ROOT / "reference/source_manifest.json")
            failures = [r["path"] for r in records if "path" in r and (not (ROOT / r["path"]).is_file() or sha256(ROOT / r["path"]) != r["sha256"])]
            print(json.dumps({"source_files": sum("path" in r for r in records), "changed_from_release_reference": failures}, indent=2))
            if failures:
                raise SystemExit(1)
        else:
            if not args.paths:
                raise ValueError("--paths is required for external data")
            print(json.dumps(preflight(args.target, read_paths(args.paths)), indent=2))
    elif args.command.startswith("train-"):
        target = args.command[6:]
        config, paths = read_json(args.config), read_paths(args.paths)
        validate_config(config, target)
        if args.out.exists():
            raise FileExistsError("Choose a new run directory: " + str(args.out))
        if target == "forward":
            from .train_forward import resolve_config, data_files
            key = "fg26" if config["stage"] == "pretrain" else "tr1x"
            require_paths(paths, [key] + (["fg26_checkpoint"] if key == "tr1x" else []))
            require_files(data_files(paths[key]))
            resolved = resolve_config(config, paths, args.out.resolve())
            print(json.dumps({"resolved_config": resolved}, indent=2))
        else:
            print(json.dumps(preflight(target, paths), indent=2))
        print(json.dumps({"execute": args.execute, "configuration": config, "out": str(args.out.resolve())}, indent=2))
        if args.execute:
            if target == "encoder":
                from .train_encoder import run
            elif target == "tcn":
                from .train_temporal import run
            else:
                from .train_forward import run
            print("Saved to", run(config, paths, args.out))
    elif args.command == "demo":
        from .inference import demo
        print(json.dumps(demo(args.assets.resolve(), args.out, args.device, args.strict), indent=2))
    elif args.command == "predict":
        from .inference import predict
        print(predict(args.input, args.normalization, args.encoder, args.tcn, args.out, args.device))
    elif args.command == "evaluate-encoder":
        if args.split == "test" and not args.allow_heldout_test:
            raise ValueError("Use validation during tuning. A locked test report requires --allow-heldout-test")
        from .train_encoder import evaluate
        print(evaluate(args.checkpoint.resolve(), read_paths(args.paths), args.out, args.split, args.device))
    elif args.command == "forward":
        from .forward import infer
        print(infer(args.input, args.checkpoint, args.out, args.device))
