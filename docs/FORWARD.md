# Optional companion: FluxXAS

FluxXAS generates molecular spectra for workflows that start from geometry.
It is an optional companion to REISE. REISE accepts compatible supplied spectra
without requiring FluxXAS or its graph dependencies.

The standard installation and bundled example cover encoder → TCN. FluxXAS
source and adapters are included, but its graph dependencies and pretrained
weights are separate. The current release's local CPU verification does not
validate the FluxXAS environment.

Use an independent Linux/CUDA environment. The historical dependency snapshot is
`src/tr_xas/vendor/fluxxas/environment_portable.yaml` (Python 3.9 / PyTorch 2.6.0
/ CUDA 12.4). It is an archive record, not a tested environment lock for this
package: the installable adapter requires Python 3.10–3.12. When making a new
environment, use a supported Python version and confirm compatible graph wheels.

FluxXAS needs PyTorch Geometric, torch-scatter, torch-sparse, torch-cluster,
e3nn, PyTorch Lightning and the training dependencies imported by the original
trainer. Consult the archived environment and the
[official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
Do not assume a plain `pip install tr-xas-models` supplies these dependencies.

After preparing that environment and installing this repository:

```bash
tr-xas forward --input geometry.npz --checkpoint weights/best_model.pt --out runs/forward --device cuda
```

Input fields are `atomic_numbers (atoms,)` and `positions (frames,atoms,3)` in Å.
All frames share the atom list; only H/C/N/O and at least one C/N/O absorber are
supported. Output `spectra.npz` contains C/N/O spectra `(frames,3,200)` and
presence flags. Missing-element spectra are zero.

Use the two `forward_*.json` configurations for pretraining and fine-tuning;
paths are supplied through `paths.local.json`. No large forward-model checkpoint
or training set is bundled. FEFF training data are incomplete in the archive,
and the historical zero-spectrum filtering implementation still needs checking
before claiming full retraining reproduction. These limitations do not affect
running the included encoder/TCN example on stored spectra.
