# Training and evaluation

Run these commands from the repository root after installing `.[train]`.
Training commands validate paths and print the plan by default; add `--execute`
to start training. Output must be a new directory outside the input data tree.

Historical references are stored in
`src/tr_xas/reference/baseline_encoder_v2_1.json` and
`src/tr_xas/reference/baseline_tcn_learning_curve.json`. Their metrics are not
identical to all later manuscript dynamics tests. Compare results using the
same split, frame scope and reference-time definition.

## REISE state encoder

```bash
tr-xas check --target encoder --paths paths.local.json
tr-xas train-encoder --config configs/encoder_all10.json --paths paths.local.json --out runs/encoder_baseline
tr-xas train-encoder --config configs/encoder_all10.json --paths paths.local.json --out runs/encoder_baseline --execute
```

For an initial environment check, copy the configuration and set `seeds` to
`[0]` and `max_epochs` to `1`. This creates a new trial, not a replacement for
the archived model. Architecture and loss definitions are in
`src/tr_xas/vendor/encoder/taskB_models.py` and
`src/tr_xas/vendor/encoder/train_taskB.py::compute_loss`.

The adapter loads train/validation data only and retains early stopping on
total validation loss. It saves checkpoints, normalization, epoch losses,
validation predictions and reaction-level metrics. `initial_checkpoint` accepts
compatible weights for fine-tuning with a new optimizer; it does not restore
optimizer state.

```bash
tr-xas evaluate-encoder --checkpoint runs/encoder_baseline/seed0.pt --paths paths.local.json --out runs/encoder_val_report --split val --device cuda
```

After fixing the configuration, use `--split test --allow-heldout-test` for a
locked test evaluation. Do not select models on the test set during tuning.

## REISE temporal decoder

```bash
tr-xas check --target tcn --paths paths.local.json
tr-xas train-tcn --config configs/tcn_final.json --paths paths.local.json --out runs/tcn_baseline
tr-xas train-tcn --config configs/tcn_final.json --paths paths.local.json --out runs/tcn_baseline --execute
```

The adapter calls the original model and training functions for one configured
prefix. Start with one seed and one epoch in a copied configuration. Model and
loss definitions are in
`src/tr_xas/vendor/temporal/phase20f_train_learning_curve.py`.

Outputs include per-seed checkpoints, ensemble validation scores, framewise
baselines, reaction-averaged localization success, selected absolute Δs,
index error and edge-selection rate. This validation uses the original
nearest-s-frame reference. Later launch-TS timing errors in the manuscript use
a different evaluation; the metrics must not be interchanged. Unified
re-evaluation of every manuscript test cohort is outside this core release.

If the TCN architecture changes, also update
`reference_scoring.py::TemporalDetector`. The architecture-parity test checks
compatibility between training and inference definitions. A changed encoder
requires new feature caches; see [MODELS.md](MODELS.md).

## FluxXAS

See [FORWARD.md](FORWARD.md) for environment requirements and verification scope.
The archived environment file is
`src/tr_xas/vendor/fluxxas/environment_portable.yaml`. Use a separate environment;
the historical snapshot is not a portable installation lock.

```bash
tr-xas train-forward --config configs/forward_pretrain.json --paths paths.local.json --out runs/fg26_trial
tr-xas train-forward --config configs/forward_finetune.json --paths paths.local.json --out runs/tr1x_trial --execute
```

`overrides` recursively updates the archived configuration. For example:

```json
{"stage":"finetune","reference_config":"vendor/fluxxas/logs/joint_multi_tr1x_finetune_v2_2026-07-24_22-18/config.json","overrides":{"num_epochs":5,"optimizer":{"lr":0.00001},"dataset":{"batch_size":8,"num_workers":0}}}
```

Use `optimizer.lr`, not `optimizer.learning_rate`. The adapter constructs the
trainer directly to avoid the original command-line spelling mismatch. Each
run has a fresh graph cache. The original trainer prunes intermediate
checkpoints beyond `save_total_limit` inside the new run and retains the best
checkpoint separately.

```bash
tr-xas forward --input my_geometry.npz --checkpoint weights/best_model.pt --out runs/forward_check --device cuda
```

This writes C/N/O molecular spectra `(frames,3,200)`. The original atomic and
molecular evaluation code is
`src/tr_xas/vendor/fluxxas/joint_training/multi_element_eval.py`; it reads paths
from checkpoint configuration. Check old checkpoint paths before using that
archived evaluator, as it may create graph caches at those paths.

**Historical implementation detail:** the collected dataset code excludes an
atomic spectrum if any point is zero, whereas one run name contains
`keepzeros`. The collected source may differ from the version used to create
historical graph caches. This rule is preserved, and exact full FG26 retraining
reproduction is not claimed. Original hashes are retained in the source manifest.

## Inference with new checkpoints

```bash
tr-xas predict --input my_spectrum.npz --normalization runs/encoder_baseline/normalization.npz --encoder runs/encoder_baseline/seed0.pt --tcn runs/tcn_baseline/seed0.pt --out runs/new_prediction --device cuda
```

Both checkpoint arguments accept multiple paths with compatible architecture
and preprocessing. Use `predict` for new weights and `demo` for the frozen
example with fixed hashes. For ensemble-averaged spectra, predict directly
from the averaged spectrum; do not average predictions from smaller ensembles.
