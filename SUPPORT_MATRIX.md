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
| k-means, k-NN, DBSCAN | Available | Apple/AMD cards recorded; consult the unsupervised ledger for the NVIDIA status | Experimental cross-vendor certificate |
| PCA, truncated SVD, OLS | Available | Identity paths remain tied to the linalg/GEMM certificate status | Experimental identity surface |
| General FP32 GEMM | Available through existing specialized routes | Contract and implementation are under active development | Experimental; not yet a released cross-vendor claim |
| Neural-network operators | Not released | Not released | Out of scope for the current release |

## Version rule

The release version and numerical-profile version are separate. A change to
reduction topology, logical partitioning, RNG position mapping, FMA policy,
FTZ seams or tie rules either proves bit-inertness against the released
profile or creates a new profile version.

This matrix is updated only from recorded evidence. New hardware columns are
welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
