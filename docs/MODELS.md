# REISE models and optional FluxXAS companion

## Naming

**REISE** stands for **Reaction Event
Inference from Spectral Evolution** and combines the reaction-conditioned state
encoder with the temporal event decoder. **FluxXAS** is an optional companion
for structure-to-spectrum prediction, previously called EquiXAS in the research
archive. These names follow the current
manuscript. EquiformerV2 remains the name of the underlying graph architecture.

The Python distribution remains `tr-xas-models`, with the `tr-xas` command.
Historical checkpoint identifiers such as B3 and original source paths in
provenance records retain their original names.

## 1. REISE state encoder: R/query/P spectra → state descriptors

- Definition: `src/tr_xas/vendor/encoder/taskB_models.py`.
- Shared 1D CNN, elemental/missing-edge/role embeddings, composition MLP and
  multiple prediction heads.
- Inputs: spectra `(batch,3 roles,3 edges,200)`, presence `(batch,3,3)` and
  composition `(batch,7)`.
- B3 uses all three roles; B1 uses endpoints only; B2 uses the query only.
- Outputs: 20 s-bin logits, signed Δs, dTS, relative energy, pre/near/post
  logits and a near-transition logit.
- The 20 bins describe a single configuration's reaction coordinate. They are
  not the historical ensemble population decoder.
- `encoder_all10.json` describes the B3v1 training scheme used for downstream
  dynamics scoring. `encoder_internal8.json` excludes endpoint training images
  as a sensitivity analysis.

**Training scope and evaluation scope are distinct.** The archived 2026-09-22
Figure 3g record says `primary_scope=internal8`, but its checkpoint hashes match
the all-10 B3v1 weights used before the TCN. Evaluating eight internal images
does not imply training an internal8 model. The binding is recorded in
`src/tr_xas/reference/figure3_edge_binding.json`. Other historical panels include
separately retrained internal8 models and must be checked individually.

## 2. REISE temporal decoder: state sequences → candidate event time

- Definition/training: `src/tr_xas/vendor/temporal/phase20f_train_learning_curve.py`.
- Six scalar features, in order: expected s, total s-distribution standard
  deviation, near probability, dTS, relative energy and signed Δs. Their first
  differences produce another six channels, for 12 channels in total.
- First-frame differences are zero. Normalization uses only the selected
  training subset.
- 48 channels, kernel 3, dilations 1/2/4/8/16/32, replicate padding, GroupNorm
  and dropout 0.1.
- Balanced BCE + hard-negative ranking + 0.25 Gaussian listwise center loss +
  expected absolute Δs loss.
- Final training subset: 617 reactions and 2,442 windows. Validation: 47
  reactions and 183 windows. Each window has 201 frames with 1 fs spacing.
- Average five seeds' sigmoid scores, then apply a width-9 reflect-padded
  moving average. The maximum selects the candidate event time.
- `prefix=1200` is the frozen candidate-ranking cutoff, not a count of 1,200
  final trajectories. The manifests determine the actual cohort size.

## 3. Optional FluxXAS companion: geometry → spectra

- Architecture: `src/tr_xas/vendor/fluxxas/src/model/equiformer_v2/`.
- Joint atomic/molecular training: `src/tr_xas/vendor/fluxxas/joint_training/train.py`.
- C/N/O graph construction: `joint_training/dataset.py` and `predict_transition1x.py`.
- Each elemental edge has 200 points. Contributions from absorbing atoms of
  the same element are summed to obtain the molecular spectrum.
- Starting architecture: 6 layers, 128 sphere channels, 8 heads, lmax 6 and
  mmax 2. The loss is atomic MSE plus 0.5 times molecular MSE, retaining the
  original sum reduction.
- FG26 pretraining and Transition1x stationary-geometry fine-tuning are separate
  stages with separate configurations.
- The FEFF reference uses a separate forward checkpoint/spectral reference
  system. Its original training dataset is incomplete in the available archive;
  full FEFF retraining is not provided.

See [FORWARD.md](FORWARD.md) for the additional environment and weight requirements.

## Updating connected models

| Change | Required downstream update |
| --- | --- |
| TCN architecture/loss/optimization only | Existing frozen encoder features can be reused |
| Encoder weights or preprocessing | Rebuild training/validation features, then retrain or explicitly evaluate transfer of the TCN |
| FluxXAS weights or spectral processing | Recompute spectra and refresh encoder/TCN inputs |
| Event postprocessing or fraction calibration | Record separately from raw TCN output |

`train-tcn` uses checksum-constrained frozen caches. A changed encoder requires
new caches, manifests and experiment records. This core release does not include
a general-purpose batch cache-generation workflow; the original research
workflow extracts 283 values per seed (256 embedding values, 20 s-bin
probabilities and other heads), from which training assembles 12 scalar channels.

Individual-trajectory and averaged-spectrum inputs share the frozen encoder →
TCN inference workflow. Even a 0% reactive input has a mathematical argmax.
Reactive-fraction estimation requires separate calibration.
