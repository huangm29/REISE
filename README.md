# REISE

**Reaction Event Inference from Spectral Evolution**

REISE infers reaction progress and locates candidate reactive events from
time-resolved X-ray absorption spectra (TR-XAS). A reaction-conditioned state
encoder interprets each spectrum using reactant and product references, and a
temporal decoder uses the resulting descriptor sequence to locate events.

| Module | Input | Output |
| --- | --- | --- |
| REISE state encoder | Reactant, query and product spectra | Reaction-coordinate distribution and state descriptors |
| REISE temporal decoder (TCN) | A sequence of encoder descriptors | A candidate event time |

This release provides installable code, training configurations and a small
frozen encoder → TCN example. It is not a complete paper-figure reproduction
package. The default installation and included example run REISE on CPU.

**FluxXAS is an optional companion module** for predicting C/N/O spectra from
molecular geometries. Its source is included for workflows that need spectral
generation; it is not required to run REISE on the stored example or on
compatible supplied spectra. See [optional FluxXAS setup](docs/FORWARD.md).

The Python package and command currently retain the names `tr-xas-models` and
`tr-xas`. See [model definitions](docs/MODELS.md).

## Install

Use Python 3.10–3.12; Python 3.12 was tested for this release. From this repository:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[train]"
tr-xas --version
tr-xas check
```

The explicit CPU wheel command is for Windows/Linux. On macOS or for CUDA,
install the corresponding PyTorch 2.6.0 build from the
[official instructions](https://pytorch.org/get-started/previous-versions/#v260),
then install this project. `python -m tr_xas` is equivalent to `tr-xas`.

## Run the included example

The repository includes approximately 13 MiB of example assets: five encoder
checkpoints, five TCN checkpoints, normalization statistics and three stored
spectral inputs. No training or separate data download is needed.

```bash
tr-xas demo --assets examples/demo_assets --out runs/demo_cpu --device cpu
```

Outputs are `verification.json` and three `prediction_*.npz` files. The report
contains the candidate times, maximum scores and differences from stored CUDA
predictions. Choose a new output directory for each run.

CPU results can differ numerically from the archived CUDA reference. A report
of `NUMERIC_DIFFERENCE` means the comparison exceeded the original tolerances;
it must not be described as strict reproduction. `--strict` makes such a
difference fail the command. No tolerances have been relaxed. These examples
are stored spectra, not a geometry-to-spectrum demonstration.

The maximum is a **candidate time**, including for the 0% reactive example.
It is not a reaction/no-reaction classification or a reaction-fraction estimate.

## Use your own data

```bash
tr-xas predict --input my_spectra.npz --normalization examples/demo_assets/normalization.npz --encoder examples/demo_assets/encoder/checkpoints/taskB_B3_seed0.pt --tcn examples/demo_assets/tcn/seed0.pt --out runs/my_prediction --device cpu
```

Both checkpoint arguments accept multiple paths. The command above uses one
seed; the demo uses all five. The input requires raw C/N/O spectra with
`query (201,3,200)`, `endpoint (2,3,200)`, `composition (7,)` and
`time_fs (201,)` at 1 fs spacing. See [data formats](docs/DATA.md).

## Train and develop

Copy `configs/paths.example.json` to `paths.local.json` and fill in your data
locations. Large training data and forward-model weights are external.

```bash
tr-xas train-encoder --config configs/encoder_all10.json --paths paths.local.json --out runs/encoder_trial
tr-xas train-tcn --config configs/tcn_final.json --paths paths.local.json --out runs/tcn_trial
python -m unittest discover -s tests -v
```

Training commands validate the plan by default; add `--execute` to train.
For code changes, install with `python -m pip install -e ".[train]"`.
See [training](docs/TRAINING.md), [model definitions](docs/MODELS.md),
[optional FluxXAS setup](docs/FORWARD.md) and [verification](docs/VERIFICATION.md).

## Layout

```text
src/tr_xas/             Installed command-line interface and adapters
  vendor/encoder/      REISE state encoder
  vendor/temporal/     REISE temporal decoder
  vendor/fluxxas/      Optional geometry-to-spectrum companion
  reference/           Frozen checksums and historical validation records
  configs/             Packaged default configurations
configs/               Editable configuration examples
examples/demo_assets/  Small frozen inference example (repository/ZIP only)
tests/                 Model, training and path checks
docs/                  Usage, scope and verification
licenses/              Third-party license texts
```

All runtime source and constants are included in the wheel. Demo assets are
kept in the repository/release ZIP, outside the wheel. The `check` command
verifies installed source against this release's manifest; the same manifest
also retains original source hashes and identifies packaging changes.

## Publication status

Prepared for private repository review before public release. The project authors have
not yet selected a license for their own code or a distribution policy for the
example weights/data. Third-party terms are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). See the short
[publication checklist](docs/RELEASE.md).
