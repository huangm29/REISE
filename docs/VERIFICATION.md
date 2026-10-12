# Verification — 2026-10-12

## Current English release (0.2.1)

Version 0.2.1 updates the documentation to English and uses FluxXAS/REISE naming.
The forward source directory, import paths, configuration references and new
forward-run names use `fluxxas`. Original names remain in historical provenance
and the naming explanation. EquiformerV2 identifiers and checkpoint tensors are
unchanged.

The 0.2.1 wheel was rebuilt and installed. All 66 source records passed integrity
checks, and the renamed FluxXAS resources were found in the installed package.
All 11 tests passed without skips. The three frozen examples ran again, with
all candidate times matching the references. CPU/CUDA differences remain
explicitly reported. See [verification_0.2.1.json](verification_0.2.1.json).

## Initial packaging validation (0.2.0)

The release was built as a wheel and installed non-editably into a separate
Windows environment. Runtime resources were loaded from the installed package,
not from the original research repository. The demo was launched outside the
source directory with Python isolated mode (`-I`).

Environment: Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 1.26.4, SciPy 1.13.1,
scikit-learn 1.4.2.

| Check | Result |
| --- | --- |
| Wheel build and installation | Passed |
| Installed source/configuration integrity | 66 files matched release hashes |
| Existing unit and training smoke tests | 11 passed, 0 skipped |
| Three frozen encoder → TCN examples | Ran successfully; finite predictions |
| Custom-checkpoint prediction entry point | Ran successfully with one encoder and one TCN seed |
| Installed dependency consistency | Passed (`pip check`) |
| Candidate times compared with the CUDA archive | All three matched |
| Strict CPU/CUDA numerical comparison | Failed original tolerances; recorded as `NUMERIC_DIFFERENCE` |

| Reactive fraction in stored input | CPU candidate time (fs) | Archived candidate time (fs) |
| --- | --- | --- |
| 0% | 132 | 132 |
| 50% | 83 | 83 |
| 100% | 108 | 108 |

The largest TCN raw-score absolute difference across the three examples was
approximately `4.64e-4`; the largest smoothed-score difference was `3.24e-4`.
The original score tolerance remains `1e-5`. These results validate installation
and example execution, not strict numerical equivalence with CUDA.

The 0% input also has a forced maximum; its candidate time is not evidence of
a reaction. No new scientific performance claim is made from these examples.

Detailed differences are in [verification_cpu.json](verification_cpu.json).
The historical records in `src/tr_xas/reference/new_adapter_*` describe the
earlier 2026-09-22 adapter and are not this release's test results.

Not tested in this preparation: complete retraining, the FluxXAS graph-model
environment, CUDA on this machine, Linux/macOS, or remote GitHub Actions.
The CI workflow is provided for Windows/Linux and will run after upload.
