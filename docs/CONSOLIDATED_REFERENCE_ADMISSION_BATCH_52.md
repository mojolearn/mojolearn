# Complementary-fixture reference admission: 52 batch lanes

Complete records from two disjoint fixture scopes at source `9d64cb98b284afb3f0af94ffaf39aa127e19bbb6` supply all nine fixtures: the earlier `base` capture and a targeted capture of `ties,hashed,wide,denormal,denormal_ftz,dupes,odd,negative`. Both runs recorded every declared property. No base fixture was repeated, no partial-part records were admitted, and no hash was fabricated.

Metal and HIP agree across all eight new fixtures for each admitted lane; both GPU columns independently agree with their local CPU column. Earlier base records also agree across these columns. Strict admission checks fixture/held-out inputs, current batch revisions, enforced backend witnesses, source provenance, protocols and complete records. The complementary scopes produce 468 cells and 3,744 parts, all corroborated by Apple, AMD and CPU classes, with zero conflicts.

Original bytes of all 416 input records, SHA-256 hashes, remote paths, source witnesses and local verdicts are retained in `bench/results/identity_break/2026-09-28_admitted-batch-52/`. Its scoped `candidate-lanes.json` and admission log reproduce the merge inputs. Four execution columns reduce to three admission device classes because Arm and x86 CPU share the CPU class; no current CUDA result is claimed.

All 1,051 existing record entries remain an unchanged prefix, all unselected cells remain unchanged, and only these 52 resolved holds were removed. LayerNorm's odd-width probe, the regression-metrics wide-fixture disagreement, and physical `par-gmm` remain excluded pending independent repair evidence. No native fits were run by the admission step.

Base table SHA-256: `27878d1d4e9a991b2aaa795c8da1edf7d34c1ce5bca4a01b8323495c1640b5af`.

Promoted table SHA-256: `3872c7cbaf2e427197df8b446b52801fc903d5b27a27fc8136a581c2169e3c32`.

## Admitted lanes

- `optim-maximize`
- `resample-perm-samples`
- `resample-utils`
- `sequence-adafactor`
- `sequence-adagrad`
- `sequence-adamax`
- `sequence-autoarima`
- `sequence-croston`
- `sequence-ets`
- `sequence-garch`
- `sequence-gru`
- `sequence-lamb`
- `sequence-lion`
- `sequence-lr-schedulers`
- `sequence-lstm`
- `sequence-mlp`
- `sequence-moe`
- `sequence-nadam`
- `sequence-prophet`
- `sequence-rmsprop`
- `sequence-rnn`
- `sequence-stl`
- `sequence-theta`
- `sequence-var`
- `x-cnn-dropout2d`
- `x-cnn-gcn`
- `x-cnn-gnn-options`
- `x-cnn-sage`
- `x-decomp-als`
- `x-decomp-lstsq-rsvd`
- `x-decomp-lu`
- `x-decomp-spectral-rbf`
- `x-decomp-umap-options`
- `x-isotonic`
- `x-metrics-classification`
- `x-metrics-cluster`
- `x-metrics-ranking`
- `x-metrics-search`
- `x-metrics-splitters`
- `x-neighbors-connected-components`
- `x-neighbors-gp-cov`
- `x-neighbors-louvain`
- `x-neighbors-metrics`
- `x-neighbors-pagerank`
- `x-neighbors-svm-precomputed`
- `x-prep-label-binarizer`
- `x-prep-label-binarizer-multilabel`
- `x-prep-label-encoder`
- `x-prep-mi-discrete`
- `x-prep-multilabel-binarizer`
- `x-prep-mutual-info`
- `x-prep-priors`
