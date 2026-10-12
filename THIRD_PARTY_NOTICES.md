# Third-party notices

The project authors have not selected a license for their own code or the
included example data/weights. The texts below apply to the respective upstream
contributions, not automatically to this entire project.

## EquiformerV2 and Open Catalyst

`src/tr_xas/vendor/fluxxas/src/model/equiformer_v2/` contains an adapted
EquiformerV2 implementation. The original archive does not identify a unique
upstream commit. Existing source copyright headers are preserved.

- EquiformerV2: https://github.com/atomicarchitects/equiformer_v2
- License source: https://raw.githubusercontent.com/atomicarchitects/equiformer_v2/main/LICENSE
- Local text: [licenses/EquiformerV2-MIT.txt](licenses/EquiformerV2-MIT.txt)
- Open Catalyst contributions carry Facebook/Meta copyright headers.
- License source: https://raw.githubusercontent.com/Open-Catalyst-Project/ocp/main/LICENSE.md
- Local text: [licenses/Open-Catalyst-MIT.txt](licenses/Open-Catalyst-MIT.txt)

These current upstream texts were retrieved on 2026-10-12. They document the
identified sources; the exact historical derivation of the adapted archive
still needs author confirmation.

## e3nn

The Wigner rotation implementation explicitly cites e3nn 0.4.0. `Jd.pt` is a
small rotation-constant resource, not a trained checkpoint.

- Source cited by the code: https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py
- License source: https://raw.githubusercontent.com/e3nn/e3nn/0.4.0/LICENSE
- Local text: [licenses/e3nn-MIT.txt](licenses/e3nn-MIT.txt)

## Research archive and packaging changes

The encoder, temporal model and FluxXAS adaptations were collected from the
project's local research archive. The original repository commit was
`a152136`. This release adds packaging, an installed CLI and documentation.
Historical machine-path prefixes in text files were replaced with
`data/archive/...`; they are provenance labels, not downloadable locations.
No checkpoint tensors, numerical example arrays, model layers or loss formulas
were altered.

`src/tr_xas/reference/source_manifest.json` preserves the original hashes and
records the hashes of the distributed source. `reference/demo_assets.json`
does the same for the bundled example. Its freeze-manifest checksum changes
only because historical path prefixes were replaced. This is not represented
as a byte-identical copy of every archive file.
