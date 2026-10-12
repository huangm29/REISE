"""C/N/O geometry inference through the original graph builder and sum rule."""
import hashlib
import sys
import numpy as np
import torch
from .common import ROOT, new_output, sha256, write_json


def infer(input_path, checkpoint_path, output, device):
    sys.path.insert(0, str(ROOT / "vendor/fluxxas"))
    from predict_transition1x import build_data_for_structure
    from src.model.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
    from torch_geometric.data import Batch
    from torch_geometric.utils import scatter
    with np.load(input_path, allow_pickle=False) as data:
        z = np.asarray(data["atomic_numbers"], np.int64)
        positions = np.asarray(data["positions"], np.float64)
    if z.ndim != 1 or not len(z) or not np.isin(z, [1, 6, 7, 8]).all() or not np.isin(z, [6, 7, 8]).any():
        raise ValueError("This entry point supports H/C/N/O geometries with at least one C/N/O absorber")
    if positions.ndim == 2:
        positions = positions[None]
    if positions.ndim != 3 or positions.shape[1:] != (len(z), 3) or not len(positions) or not np.isfinite(positions).all():
        raise ValueError("positions must be finite (frames, atoms, 3) in angstrom")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = EquiformerV2_OC20(**ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    spectra = np.zeros((len(positions), 3, 200), np.float32)
    for i, pos in enumerate(positions):
        seed = int(hashlib.sha256(np.ascontiguousarray(pos, dtype=np.float64).tobytes()).hexdigest()[:8], 16) % (2 ** 31)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        graphs = build_data_for_structure(z, pos, "provided", "query", "provided", frame_idx=i)
        batch = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            atomic = model(batch, is_training=False)
            mol = scatter(atomic[batch.xas_mask], batch.batch[batch.xas_mask], dim=0, reduce="sum").cpu().numpy()
        for j, graph in enumerate(graphs):
            spectra[i, "CNO".index(graph.elem_name)] = mol[j]
    if not np.isfinite(spectra).all():
        raise FloatingPointError("Non-finite forward output")
    out = new_output(output)
    np.savez_compressed(out / "spectra.npz", spectra=spectra, presence=np.asarray([value in z for value in (6, 7, 8)], np.float32))
    write_json(out / "summary.json", {"frames": len(positions), "shape": list(spectra.shape),
               "input_sha256": sha256(input_path), "checkpoint_sha256": sha256(checkpoint_path),
               "coordinate_seed_rule": "float64 coordinates SHA256 first 8 hex modulo 2**31"})
    return out

