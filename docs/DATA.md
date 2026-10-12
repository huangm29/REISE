# Data and weights

The repository includes approximately 13 MiB of example inputs and weights.
Large training datasets and FluxXAS weights are external. Put their locations
in `paths.local.json`, which is excluded from Git. Relative paths are resolved
against the directory containing that configuration file.

## Path configuration

| Key | Required directory or file |
| --- | --- |
| `mep` | `mep_{train,val,test}.pkl` and `mep_labels_{train,val,test}.pkl` |
| `labels` | `rc_labels_{train,val,test}.pkl` |
| `phase20` | The TCN layout shown below |
| `fg26` | Four `merged_*` atomic/molecular spectral pickle files |
| `tr1x` | The same four filenames for stationary Transition1x references |
| `fg26_checkpoint` | FG26 `best_model.pt`, used to initialize fine-tuning |
| `demo` | Example asset directory, containing ten checkpoints and three inputs |

```text
phase20d_20260821_130436/
  input/phase20f_tcn_learning_curve_plan.json
  input/blinded_xas_input/public_manifest.json
  b3_seed_features/feature_manifest.json
  b3_seed_features/<feature NPZ files listed in the manifest>
  minimal_training_truth/minimal_labels_manifest.json
  minimal_training_truth/<label NPZ files listed in the manifest>
```

Training requires the actual feature and label files. A results-only copy
without these files is not a valid training dataset.

## Availability

The small example in `examples/demo_assets/` is included and can be run directly.
Full training data and FluxXAS weights are not included, and no public download
URL has been provided yet. Historical source records in
`src/tr_xas/reference/assets_catalog.json` do not imply public availability.
Regenerating feature caches after changing the encoder requires the original
research workflow or a separately validated replacement.

## File formats

- **Encoder MEP pickle:** keyed by reaction ID; contains R/P/TS and ten path
  images, molecular spectra and composition. Per-frame spectra use the keys
  `XAS_mol_C/N/O`. Reaction IDs are checked against
  `src/tr_xas/reference/mep_split_manifest.json`.
- **RC label pickle:** s, signed/absolute Δs, dTS, ΔE, near-transition and
  pre/post labels. Training preserves the original reaction splits.
- **TCN feature NPZ:** `frame_ids` and `features (5,201,283)`. Label NPZ files
  contain `near_true` and `abs_delta_s`. The original loader checks frame IDs,
  hashes and training splits.
- **Forward atomic pickle:** structure key → element symbols, coordinates
  `(3,n_atoms)` and atomic spectra `(200,n_absorbers)`.
- **Forward molecular pickle:** `smiles` record key and C/N/O molecular arrays.
  Use the original seeded structure split, not a random split of atomic graphs.
- **Spectral inference NPZ:** `query (201,3,200)`, `endpoint (2,3,200)`,
  `composition (7,)` and `time_fs (201,)`. Spectra use raw intensity and C/N/O
  order. Composition contains H/C/N/O atom fractions followed by C/N/O presence.
  The time grid must have 1 fs spacing; resampling is not automatic.
- **Geometry inference NPZ:** `atomic_numbers (n_atoms,)` and
  `positions (frames,n_atoms,3)` in Å. All frames share the same H/C/N/O atom list.

Load pickle and PyTorch checkpoint files only from trusted sources, after
checking their hashes.
