from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# Runtime resources travel inside the package, including in a regular wheel.
ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def encoder_modules(training=True):
    path = ROOT / "vendor/encoder"
    sys.path.insert(0, str(path))
    import srp_data
    import taskB_models
    if training:
        import train_taskB
        return srp_data, taskB_models, train_taskB
    return srp_data, taskB_models


def temporal_module(base):
    # The original file reads this variable at import time; do not call its main.
    os.environ["PHASE20_BASE"] = str(base)
    return load_module("tr_xas_temporal_reference", ROOT / "vendor/temporal/phase20f_train_learning_curve.py")


def scoring_module():
    encoder_modules(training=False)
    return load_module("tr_xas_scoring_reference", ROOT / "vendor/temporal/reference_scoring.py")


def read_paths(path):
    config = Path(path).resolve()
    values = read_json(config)
    result = {}
    for key, value in values.items():
        if value is None or str(value).startswith("CHANGE_ME"):
            result[key] = None
        else:
            resolved = Path(value).expanduser()
            result[key] = resolved.resolve() if resolved.is_absolute() else (config.parent / resolved).resolve()
    return result


def require_paths(paths, keys):
    missing = [key for key in keys if paths.get(key) is None or not paths[key].exists()]
    if missing:
        raise FileNotFoundError("Set existing paths in your private paths file: " + ", ".join(missing))


def require_files(files):
    missing = [str(path) for path in files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def new_output(path, input_roots=()):
    out = Path(path).resolve()
    # Training must not write into a source/data tree, even through a symlink.
    for root in input_roots:
        root = Path(root).resolve()
        if out == root or root in out.parents:
            raise ValueError("Output must be outside input directories: " + str(out))
    for source in [ROOT]:
        if out == source or source in out.parents:
            raise ValueError("Output must be outside source directories")
    out.mkdir(parents=True, exist_ok=False)
    return out


def record_run(out, config, inputs):
    def git(*args):
        try:
            p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        except FileNotFoundError:
            return "unavailable"
        return p.stdout.strip() if p.returncode == 0 else "unavailable"
    sources = sorted(p for p in ROOT.rglob("*.py") if not any(part in {"runs", ".venv", "__pycache__"} for part in p.relative_to(ROOT).parts))
    write_json(out / "run.json", {
        "kind": "new_training_run_not_an_archived_paper_result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "inputs": {key: str(value) for key, value in inputs.items()},
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
    })

