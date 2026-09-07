# Cross-vendor identity and NVIDIA comparison execution plan

Updated September 7, 2026. This is the controlling plan for the current
mojolearn work. It supersedes earlier instructions to time our FAST mode.
Do not edit the paper during this implementation and evidence campaign.

## Non-negotiable comparison layout

Every performance row names one feature, workload size, exact configuration,
incumbent implementation and timed boundary. Its three timing columns are:

| mojolearn IDENTICAL | incumbent FAST | incumbent DETERMINISTIC |
|---|---|---|
| Our only measured mode | Selected incumbent on NVIDIA | Same incumbent on NVIDIA, documented deterministic configuration |

**Never create, launch, time or include a mojolearn FAST arm in this campaign.**
The library may retain explicit FAST functionality, but it is outside this grid.
An unsupported incumbent deterministic configuration is an explicit missing
capability, not a zero time, a CPU comparison, a seeded FAST run or a failure
of cross-vendor identity. Keep the selected incumbent fixed across sizes.

The grid measures implementation cost relative to incumbents, including current
optimization gaps. It does not isolate a universal or unavoidable price of
bitwise arithmetic. Timing on NVIDIA is a demanding comparison against its
mature native libraries, not a proven universally least-favorable device.

## Matrix A: small-workload feature identity

Goal: give every public feature and supported variant an explicit evidence row
for **Apple M4 / NVIDIA / AMD**. Small bounded fixtures are sufficient for this
matrix; they do not certify arbitrary sizes or parameter combinations.

Audit historical documentation and retained artifacts before renting. Record
exact commit/source inventory, native hashes, GPU model/architecture, compiler,
input bytes, shape, numerical mode and comparison coverage. An old Metal record
with no M4 device witness must not be relabeled M4. Preserve every old certificate.

For each row, compare the same FP32 inputs and configuration across all three
vendors. Retain intermediate stage bytes where available, predictions/results,
gradients and optimizer/checkpoint state where relevant, and repeated execution
evidence. Name whether the result proves stage equality, complete state equality
or only final output equality. Floating-point tolerance is not bitwise equality.
Identical named refusals establish contract behavior, not implemented features.

Use statuses: `VERIFIED_THREE_VENDOR`, `HISTORICAL_OTHER_SOURCE`,
`TWO_VENDOR_ONLY`, `FINAL_OUTPUT_ONLY`, `UNRUN`, `DIVERGED`,
`UNSUPPORTED_CONFIGURATION`, `IMPLEMENTATION_MISSING`.
Do not turn missing evidence into a green cell. Close each executable gap in a
new immutable round; an unimplemented feature requires a port before validation.

Required coverage includes:

- Dense linear algebra: GEMM layouts, GEMV, Gram, solvers and decompositions.
- Neighbors: query/classification/regression, supported metrics, weighting,
  radius and larger-k variants; invalid and intentionally refused algorithms.
- PCA/SVD: solvers, transform/inverse and whitening as separately supported.
- Clustering/density: KMeans, DBSCAN, KDE, hierarchy, spectral and mixture APIs.
- Trees: GBDT objectives and prediction modes, symmetric/ordered behavior,
  categorical/CTR/ranking options, RF/ExtraTrees and isolation forest.
- Linear/kernel models: regression, regularization, logistic, SVC/SVR and GP.
- Time series: ARIMA filter and fitter separately; Holt-Winters and other TSA.
- UMAP: fit, transform, sparse/CSR and neighborhood-preservation quality scope.
- Mamba1/2/3: forward, backward, carried state, continuation and public APIs.
- Transformers/neural training: forward/backward, optimizer steps, MLP, tiny
  real-text LM, state serialization and actual cross-vendor checkpoint resume.
- Embeddings, metrics, loss/optimizer primitives and remaining exported APIs.

The public export inventory is the completeness checklist, not this family list.
See [feature evidence audit](CROSS_VENDOR_FEATURE_IDENTITY_AUDIT.md) for exact
existing evidence and missing cells. The old 209-configuration server-GPU matrix
must remain distinct from a new complete M4/NVIDIA/AMD API matrix.

Already closed: the fixed two-block 34,944-parameter real-text LM has complete
128-step equality on all three vendors, including final checkpoint bytes.
[Retained result](../bench/results/resume/2026-09-07-root-byte-lm-three-vendor/README.md).
Metal checkpoint continuation and the separate MLP Metal leg remain open.

## Matrix B: small, medium and large NVIDIA performance

Use exactly one chosen incumbent per matched workload. Start with NVIDIA cuML
for matching classical estimators and UMAP, NVIDIA cuBLAS/cuSOLVER through a
recorded adapter for linear algebra, PyTorch CUDA for supported neural workloads,
and the best algorithmically matched CUDA tree library. Symmetric CatBoost,
XGBoost histogram trees and LightGBM CUDA are different workloads unless their
configuration and quality targets are explicitly reconciled. Do not select the
slowest opponent or change opponents after seeing a ratio.

Predeclare three size points per scalable workload. Start small, then medium,
then large only after correctness and memory admission. Size labels are relative
to each algorithm's scaling, not a universal row count. Include odd/non-tile
dimensions and tie/zero/edge fixtures in identity testing separately.

Initial runnable adapter candidates (subject to API limits and memory checks):

| Workload | Small | Medium | Large | Incumbent |
|---|---|---|---|---|
| GEMV square dimension | 512 | 2048 | 8192 | PyTorch CUDA / cuBLAS FP32 |
| NT product rows, inner/output64 | 1024 | 16384 | 262144 | PyTorch CUDA / cuBLAS FP32 |
| Gram observations, features32 | 4096 | 65536 | 1048576 | PyTorch CUDA / cuBLAS FP32 |
| kNN index/query/features, k10 | 4096/64/32 | 65536/128/32 | 262144/256/32 | cuML brute-force |
| UMAP | existing256 | existing1024 | adapter extension required | cuML, matched graph/epochs |
| Symmetric trees | existing1024x8 | parameterized adapter required | parameterized adapter required | CatBoost GPU |
| Fixed tiny LM training | current certified shape | larger shape not implemented | larger shape not implemented | matched PyTorch CUDA model |

Do not call the current UMAP1024 or fixed tiny-LM fixture a large-workload result.
Add size parameters/adapters for all remaining features before launching those
rows. Large-workload identity beyond the small certification matrix requires
its own evidence and cannot be inferred from a successful timing run.

Match input bytes, FP32 precision/math settings, layouts, objective, algorithm,
iterations, initialization and output requirements. For stochastic algorithms,
matching quality targets may be necessary; report the difference explicitly.
Disable TF32 in the matched full-FP32 comparison. Any reduced-precision optimized
comparison belongs to a separately declared quality-matched experiment.

Use the same physical NVIDIA GPU, frozen software and timing boundary for all
columns. Synchronize GPU completion, retain warmups and at least seven timed
rounds, rotate order, report median plus dispersion and raw samples. Report
transfer-inclusive public API timings separately from device-resident kernels.
Do not compare a host-array API to an incumbent kernel-only call.

Retain all failures, accuracy gates, input hashes, actual loaded CUDA library
paths, mode flags, allocation estimates and peak observations. A timeout/OOM is
a named outcome, not permission to silently shrink only one arm. Preflight
quadratic allocations; large jobs remain under remote RSS/VRAM/deadline guards.

Deterministic incumbent policy must be implementation/version specific:

- PyTorch: strict deterministic algorithms (no warn-only fallback), CUDA workspace
  configuration before initialization, cuDNN deterministic controls and recorded
  precision flags. [Official reproducibility guidance](https://docs.pytorch.org/docs/2.9/notes/randomness.html).
- cuBLAS: pin toolkit/architecture, streams, workspace and math settings to the
  documented guarantee. [NVIDIA reproducibility scope](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility).
- CatBoost GPU training is documented as nondeterministic; its deterministic
  NVIDIA training cell is unsupported. [Official GPU documentation](https://catboost.ai/docs/en/features/training-on-gpu).
- LightGBM's `deterministic` parameter is CPU-only; do not label CUDA training
  deterministic by setting it. [Official parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html).
- cuML: inspect each pinned algorithm's actual guarantee. A seed alone is not
  sufficient. Record `DETERMINISTIC_SUPPORT_UNVERIFIED` until established;
  distinguish that from documented unsupported behavior.

## Rental and execution order

Root alone runs builds, tests, comparisons, measurements, rentals and publishing.
Agents may only inspect/write source and documents; **never tests or measurements**.
One model/build/measurement job at a time, two CPU cores/threads by default,
never more than three. NVIDIA on RunPod; AMD on DigitalOcean; local Apple M4.

Do not start many parallel RunPods. First reuse one NVIDIA and one AMD session
serially for the bounded missing identity rows. Then run the size sweep on a
single selected NVIDIA architecture. A second NVIDIA architecture (e.g.4090
sm89 plus H100 sm90) and a different physical machine provide explicit extra
coverage and release qualification; they do not substitute for AMD or Metal.
Stage these serially. Every rental has a fixed expiry, independent teardown,
fetch reserve and deletion/absence verification. No unattended renewal.

The M4 user-authorized tiny-job policy removes the fixed free-reserve minimum
but retains pressure/swap/compression/CPU/RSS/deadline and cleanup enforcement.
It does not authorize large local performance sweeps or unbounded workloads.

Execution checklist:

- [x] Record this corrected three-column plan and keep our FAST out of new runs.
- [x] Change Python selector and binding-build defaults to IDENTICAL in source.
- [x] Author IDENTICAL-only tiny-LM wheel payload/build/installed-check support.
- [ ] Root validate integrated file fixtures and fix all failures.
- [ ] Finish export-by-export old-evidence audit and small three-vendor gap list.
- [ ] Freeze candidate source, manifests, dependencies and deterministic policies.
- [ ] Build and qualify final tiny-LM-inclusive0.6.1 wheel on actual sm89/sm90/gfx942;
  preserve legacy0.6.0 evidence. Three sets138extensions,25installed jobs perarch.
- [ ] Execute missing small identity rows on NVIDIA/AMD/M4 and retain a new matrix.
- [ ] Implement missing scalable adapters and fixed per-feature quality gates.
- [ ] Run NVIDIA small/medium/large rows, ourIDENTICAL versus incumbentfast/deterministic.
- [ ] Generate reviewable grids from retained records; never manually mark cells green.
- [ ] Publish0.6.1 after exact final-byte admission and verify public file hashes.

Publication of a default mode or API does not turn every feature into a certified
configuration. Each matrix cell carries its actual evidence boundary.
