# Support and certification matrix

This file distinguishes installation support from measured numerical
certification. Portability is supplied by Mojo; validation is only what the
recorded cards establish.

## Distribution

| Platform | Distribution | Current evidence |
|---|---|---|
| Apple silicon, macOS arm64 | Published wheel path | Built at the Apple M1 ISA floor; release smoke fits run on real Apple GPU hardware |
| NVIDIA CUDA, Linux | Source build | E1/E2 result cards on H100; no released Linux wheel yet |
| AMD HIP, Linux | Source build | E1/E2 result cards on MI325X and additional unsupervised measurements on MI300X; no released Linux wheel yet |
| Other GPUs and CPUs | Compatibility report | No CPU implementation; new GPU measurements are welcome |

Exact versions, commits and result status are recorded in
[E1_RESULTS.md](E1_RESULTS.md), [E2_RESULTS.md](E2_RESULTS.md), and
[E1_RUNBOOK.md](E1_RUNBOOK.md). A check mark in source code is not a
certificate; a released result card is.

## Public capability levels

| Surface | FAST | IDENTICAL | Public status |
|---|---|---|---|
| Gradient boosting | Available | Three-vendor matrix with named refusals and residuals recorded in E1/E2 | Supported beta |
| Random forests | Available | Three-vendor E1/E2 cards for the recorded configurations | Supported beta |
| Extra Trees | Available | Three-vendor E1/E2 cards for the recorded configurations | Supported beta |
| k-means, k-NN, DBSCAN | Available | Apple, NVIDIA and AMD cards recorded and IDENTICAL: E1U cards 3/3 on both vendor columns and 80/80 E2U cells at `fe00e8a` (E3 round 8), re-verified card-by-card at `144aa5b` on 2026-08-28 -- kmeans 77 stages, knn 6, dbscan 3, identical Apple<->H100 and Apple<->MI325X | Measured cross-vendor on three columns |
| PCA, truncated SVD, OLS | Available | In the same 80/80 E2U result as the clustering row: `E3_RESULTS.md` round 8 at `fe00e8a` certifies k-means, k-NN, DBSCAN, PCA, tSVD, OLS, Ridge and logistic together, 80 cells identical on Apple<->H100 AND Apple<->MI325X (60 identical plus 20 refused with the same message on every column) | Measured cross-vendor on three columns |
| General FP32 GEMM | Available through existing specialized routes | Profile `mojolearn.identical.gemm.fp32.v1`, frozen; the identity card is bit-identical on Apple M4, NVIDIA H100 and AMD MI325X, 60 stages each, at leg 11 commit `144aa5b` (E3 round 11, judge section 7). Shapes and plans outside the card's 62-shape, eight-plan sweep have run on Apple only | Measured cross-vendor for the card's sweep; no Python surface for it is built into the released wheel |
| Isolation forest | Available | **Apple<->AMD bit-identical, 123 card stages** at `a0a0eee` (2026-08-28, its first cross-vendor run; the lane had been writing this card to a scratch path since before it was in any round). NVIDIA column pending | Measured on two columns |
| Neural-network operators (mamba, transformer) | Available, not released | **Apple<->AMD bit-identical: mamba 17 card stages, transformer 30**, at `a0a0eee`. Transformer's clause (a) passes on 262,634 cells and clause (d) -- decode == prefill -- passes under IDENTICAL and FAILS under FAST, which is the profile working as written. NVIDIA column pending; mamba's FAST arm has never been built on any vendor | Measured on two columns; NOT part of the released surface |

## Version rule

The release version and numerical-profile version are separate. A change to
reduction topology, logical partitioning, RNG position mapping, FMA policy,
FTZ seams or tie rules either proves bit-inertness against the released
profile or creates a new profile version.

This matrix is updated only from recorded evidence. New hardware columns are
welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
