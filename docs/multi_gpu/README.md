# Ordered multi-GPU training

This is an opt-in implementation. It does **not** enable every mojolearn
estimator, and it does not establish blanket cross-vendor bitwise identity.
No local tests or model executions were used during this implementation.
The cloud gates passed on two RunPod RTX 4090s and two H100s using the same
frozen source snapshot. Gradients and model states matched bit for bit for
the tested configurations. Training data came from Cloudflare R2 and was
verified against the dataset manifest. See the
[initial cloud evidence](../../bench/results/multi_gpu/2026-09-14/README.md).
The later concurrent-byte, boosting and classical changes have separate
[two-H100 evidence](../../bench/results/multi_gpu/2026-09-14/continued-h100/README.md);
they have not received a new same-source cross-architecture qualification.
[Main integration and IsolationForest receipts](../../bench/results/multi_gpu/2026-09-14/integration-h100/README.md)
record the merged Gram default and the subsequent whole-tree driver on two H100s.
[SVM and GP receipts](../../bench/results/multi_gpu/2026-09-14/svm-gp-h100/README.md)
qualify their subsequent kernel-row paths on two H100s.

## Available paths

| Surface | Partition | Numerical contract |
| --- | --- | --- |
| Pooled byte language model | Decoder layers and their model/optimizer state; embedding/head on the first device | Original layer kernels, ordered logical gradient sums and atomic owned AdamW updates |
| Offloaded byte language model replay | One decoder layer at a time on one GPU, canonical state on the host | Same ordered logical gradients and staged atomic AdamW updates |
| Byte language model | Logical microbatches across resident device replicas | Copy gradient 0; left-fold gradients 1..K-1; disjoint AdamW updates and parameter broadcast |
| SmallMLPTrainer | Frozen microbatch snapshots on concurrent GPU workers | Same ordered sum; original global clipping and disjoint host-staged optimizer updates |
| SambaStack (Mamba/attention blocks) | Frozen microbatch snapshots on concurrent GPU workers | Same ordered sum; explicit logical dropout stream/offsets |
| RandomForest classifier/regressor | Whole trees over full replicated data | Original global tree IDs, seed and quantiles; original prediction order |
| ExtraTrees classifier/regressor | Whole trees over full replicated data | Original global tree IDs and full-data label quantization; original prediction order |
| IsolationForest | Whole tree ranges during fit and score-time rebuild | Original global RNG IDs, tree-local node indices, score order and contamination threshold |
| KMeans | Whole row tiles during distance/assignment | Original per-row feature/centroid arithmetic; original initialization, full-data updates and convergence |
| GradientBoosting, greedy symmetric/depthwise/lossguide | Whole packed feature groups during histogram construction | Original row-reduction geometry and global scale; disjoint histogram-column copies |
| ARIMA / ExponentialSmoothing | Independent series assigned to workers | Original per-series initialization, solver and likelihood arithmetic |
| LinearRegression / Ridge / covariance PCA / TruncatedSVD | Original Gram chunks; wider v1 output rows; minimum-norm OLS row Gram | Original contraction, final fold and root solver order |
| LogisticRegression | QN gradient feature columns | Original per-cell row reduction; unchanged root objective, line search and optimizer |
| Lasso / ElasticNet | Whole FP32-v1 dot leaves during cyclic coordinate descent | Original balanced tree; unchanged coordinate and convergence order |
| SVC / SVR | Linear/RBF kernel output rows during fit and prediction | Original per-cell FP32-v1 dot and RBF epilogue; original root working-set and update order |
| GaussianProcessRegressor | Covariance and cross-covariance output rows | Original postfix expression and feature order; global WhiteKernel diagonal; root factorization/solve/variance |
| StandardScaler / MinMaxScaler | Independent feature columns, fit and transforms | Original row chunks and final per-column fold; output bytes copied into column order |

The gates exercise particular configurations, not all parameter combinations.
Samba fixtures include a Mamba layer without dropout and a mixed Mamba/attention
stack with dropout 0.1. Existing model admission restrictions still apply.

## Byte LM and one-GPU replay

```python
from mojolearn.parallel_training import ParallelByteLanguageModelTrainer

initial = existing_trainer.state_dict()
with ParallelByteLanguageModelTrainer(
    initial, devices=(0, 1), logical_shards=8,
) as trainer:
    result = trainer.train_step(microbatches)  # exactly 8, in logical order
    checkpoint = trainer.checkpoint()
    gradients = trainer.export_gradients()

# Exactly the same logical microbatch shape, count and order on one device.
with ParallelByteLanguageModelTrainer.from_checkpoint(
    checkpoint, devices=(0,),
) as replay:
    replay.train_step(next_microbatches)
```

Each microbatch is `(shape.batch, shape.length + 1)` int32 token IDs. The
starting model, moments, optimizer configuration and data bytes must agree.
The logical shard count belongs to the checkpoint; physical GPU count does
not. The caller owns data scheduling and must supply the next logical window.
Checkpoints here are Python dictionaries with portable array bytes, not a new
on-disk serialization format. Export before `close()`, which releases state.

The first gradient is copied unchanged. Each later add is
`ftz(fma(1, ftz(total), ftz(shard)))` in FP32, with logical shards visited in
ascending order. Shard loss gradients are means over their own microbatch;
the aggregate is their **sum**, not their average. No division by K occurs.
Losses are returned as a vector in logical order. This is intentionally a
separate contract from `training.accumulate_grads`' existing balanced tree.

K=1 retains the original step. K>1 is intended to match the defined one-GPU
accumulation replay, not a single GEMM over a concatenated K-times-larger
batch. Cross-vendor equality additionally requires the underlying model
kernels to agree for the selected shapes and build arms.

The byte-LM driver schedules gradient bodies concurrently in waves, with one
context and trainer owned by each task. It joins the entire wave before the
root folds gradients in logical order. Two-H100 replay gates pass with this
schedule; throughput and eight-device scaling have not been measured.
Builds with process-global step-phase counters refuse concurrent devices
because those counters are not thread-safe.
It holds two additional flat FP32 gradient buffers on the root device.
Device-to-device copies use `DeviceBuffer.enqueue_copy_to`; the source stream
is synchronized before any destination reads. The implementation uses no
collective reduction. See the [Modular copy API](https://max.modular.com/api/mojo/max/gpu/host/device_context/DeviceBuffer/).

## MLP and Samba

```python
from mojolearn.parallel_training import ParallelNeuralTrainer

with ParallelNeuralTrainer(model, devices=(0, 1), logical_shards=3) as trainer:
    trainer.train_step([(x0, y0), (x1, y1), (x2, y2)])
    checkpoint = trainer.checkpoint()

with ParallelNeuralTrainer.from_checkpoint(checkpoint, devices=(0,)) as replay:
    replay.train_step(next_shards)
```

The supplied model is exclusively owned while wrapped. Workers receive a
frozen snapshot and compute gradients concurrently. A cooperative worker
partitions gradient columns and optimizer ranges across the selected GPUs,
retains the original complete-registry clipping operation on the first GPU,
and publishes the owner state after every owner succeeds. Parameters, moments and RNG state are restored on a failed step.
Samba requires `accumulation_steps=1` because `logical_shards` owns the new
accumulation contract. Every logical shard has its own loss normalization;
ignored-target semantics remain the existing Samba semantics.

The subprocess transport is host-staged and byte-preserving. It is not NCCL,
not a distributed autograd system, and not a multi-node service. Workers are
started with device visibility set before importing the GPU runtime. Their
private pickle channel only accepts data from their parent process.

## Forests and KMeans

```python
from mojolearn.parallel_ensemble import fit_forest
from mojolearn.parallel_classical import fit_kmeans

fit_forest(forest, X, y, devices=(0, 1), trees_per_shard=16)
fit_kmeans(kmeans, X, devices=(0, 1), sample_weight=weights)
```

Additional entries:

```python
from mojolearn.parallel_ensemble import (
    fit_boosting, fit_isolation_forest, score_isolation_forest,
)
from mojolearn.parallel_classical import (
    fit_arima, fit_exponential_smoothing, fit_gram_estimator, fit_logistic,
    fit_coordinate_descent, fit_svm, predict_svm,
    fit_gaussian_process, predict_gaussian_process,
)
from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler

fit_boosting(boosting, X, y, devices=(0, 1), sample_weight=weights)
fit_isolation_forest(isolation, X, devices=(0, 1))
anomaly_scores = score_isolation_forest(isolation, queries, devices=(0, 1))
fit_arima(arima, series, devices=(0, 1), series_per_shard=2)
fit_exponential_smoothing(holtwinters, devices=(0, 1), series_per_shard=2)
fit_gram_estimator(ridge, X, y, devices=(0, 1))
fit_logistic(logistic, X, labels, devices=(0, 1))
fit_coordinate_descent(lasso, X, y, devices=(0, 1))
fit_svm(svc, X, labels, devices=(0, 1))
predicted = predict_svm(svc, queries, devices=(0, 1))
fit_gaussian_process(gp, X, y, devices=(0, 1))
mean, std = predict_gaussian_process(gp, queries, devices=(0, 1), return_std=True)
fit_scaler(scaler, X, devices=(0, 1), columns_per_shard=16)
scaled = transform_scaler(scaler, X, devices=(0, 1), columns_per_shard=16)
```

Boosting supports greedy symmetric, depthwise and lossguide searchers and
pointwise symmetric search, including classifier/regressor adapters. Separate
drivers cover OrderedRMSE and ExperimentalTwoLevelFeatureFreq; see below.
The root still owns the full index and histogram. Per-level allocation and
staging overhead may outweigh computation savings.

ARIMA workers receive only their assigned series; scaler workers receive only
their assigned columns. These partitions reduce the GPU memory required per
worker. The host holds the complete data and result, and a single shard still
must fit on one GPU. ARIMA prediction currently uses the existing single-GPU
methods; scalers provide an explicit distributed transform entry. No run beyond
one GPU's memory capacity has been qualified. These new paths have two-H100
equality evidence, not new AMD/Apple or NVIDIA cross-architecture qualification.

IsolationForest builds tree ranges concurrently, then copies node buffers into
global tree order. Every worker receives the full training data; build scratch
is partitioned, and the complete forest is assembled on the root. Its existing
Python estimator rebuilds trees on each scoring call, so use the explicit
`score_isolation_forest` entry to distribute that rebuild too. Scoring itself
keeps the original root reduction and contamination threshold. Diagnostic
trace capture is refused by the distributed path.

SVC/SVR distribute output rows of their linear/RBF kernel matrices, including
prediction kernels. The complete data, working-set state and assembled kernel
tile remain on the root. One-row operations use only one device. Concurrent
kernels refuse builds with process-global GEMM phase counters enabled.

Gaussian processes distribute covariance rows, preserving global row indices
for WhiteKernel's structural training diagonal. Cross-covariance retains its
zero WhiteKernel contribution even for identical query/training coordinates.
The kernel expression keeps its original postfix order. Full covariance,
Cholesky, solves, likelihood and prediction variance remain on the root. Only
the existing fixed-hyperparameter surface is supported; sabotage probes are
refused by the distributed entry. Returned standard deviations also publish
the existing variance-clamp diagnostics on the supplied estimator.

Gram, logistic and coordinate-descent paths retain full root data/solver state.
The coordinate-descent partition applies to automatic dot scheduling; explicit
native plan probes still execute the requested original plan. Dots of at most
128 rows have one indivisible leaf and stay on the root. Per-call allocation
and transport are not throughput-qualified.

These return and update the supplied estimator only after the full fit
succeeds. Forest shards use the complete dataset, never independent subsets
whose models are averaged afterwards. Trees return in global ID order and
use the existing predictor. Choose a tree range size that amortizes repeated
full-data transport while leaving enough ranges to occupy the GPUs.

KMeans runs in one isolated worker with all selected GPUs visible. Its native
assignment path partitions aligned row tiles and copies results back to their
original positions. No floating-point per-device centroid sums are combined.
It currently allocates and stages per assignment call, so transport overhead
can outweigh any parallel work. Small inputs may contain fewer row tiles than
selected GPUs. The internal `MOJOLEARN_KMEANS_DEVICE_COUNT` switch belongs to
this worker; users should use the Python entry rather than changing process
visibility or environment settings around concurrent fits.

## Remaining rollout

Here, rollout means completing implementation and cloud qualification, not a
backward-compatibility or staged-release process. Alpha API changes are allowed.
The [coverage inventory](COVERAGE.md) distinguishes estimator support from
memory capacity. MLP/Samba retain complete host model replicas; their GPU operations and
several classical root paths still impose individual-device memory limits. Byte-LM layer pooling removes that model-wide
GPU requirement; an individual layer and embedding/head must fit their owners. Independent-series and scaler-column workers
receive only their partitions, but beyond-single-GPU capacity is not qualified.

The user's requested order is neural training, forests/ExtraTrees, then
boosting and classical estimators. The following remain unimplemented:

- Wide full-PCA transpose QR and larger Gram configurations need further
  partitions and qualification; the root eigensolver state still requires one GPU.
- Broader neighbor/density/graph configurations, resident reference and graph
  pooling, and native-only surfaces need additional partitions and qualification.
- Full Samba model partitioning, resident staging reuse, additional admitted fixtures,
  eight physical GPUs, AMD/Apple cross-vendor evidence,
  injected lost-device recovery, and throughput/cost measurements.

Unsupported estimators are not silently routed through independent subset
fits or claimed to be multi-GPU capable.

## Cloud gates

Build the matching IDENTICAL extensions on the rented host. Stage the corpus
with `tools/stage_from_r2.sh` in strict mode. The gate scripts require `--cloud`
and a RunPod environment marker and should only be invoked on the pod:

- `tools/byte_lm_parallel_check.py`: K1, odd logical K replay, replica equality,
  invalid-shard atomicity, group post-update rollback and checkpoint migration.
- `training/checks/ordered_gradient_check.mojo`: cancellation, zero and FTZ seams.
- `tools/parallel_training_check.py --lane mlp|samba|forest`.
- `tools/parallel_kmeans_check.py`: feature widths 7/8/32 and all three init modes.
- `tools/parallel_boosting_check.py`: greedy tree policies, mixed packing widths,
  weighted inputs, classification/regression and failed-fit publication.
- `tools/parallel_arima_check.py`: independent-series fits, forecasts and rollback.
- `tools/parallel_preprocessing_check.py`: column fits, transforms, inverse and refusal.
- `tools/parallel_gram_check.py`: OLS/Ridge/covariance PCA/SVD fitted state and outputs.
- `training/checks/gram_parallel_check.mojo`: every original Gram partial and final result.
- `tools/parallel_logistic_check.py`: binary/multiclass QN and failed-fit publication.
- `tools/parallel_holtwinters_check.py`: additive/multiplicative fits and forecasts.
- `tools/parallel_solver_check.py`: Lasso/ElasticNet fits and predictions.
- `training/checks/dot_parallel_check.mojo`: distributed dots against the FP32-v1 oracle.
- `tools/parallel_iforest_check.py`: bootstrap/features/contamination, scores, labels and failed-fit publication.
- `training/checks/iforest_parallel_check.mojo`: all eight native tree buffers.
- `tools/parallel_svm_check.py`: SVC/SVR fitted state, predictions and failed-fit publication.
- `training/checks/svm_parallel_check.mojo`: linear/RBF kernel cells across feature-leaf boundaries.
- `tools/parallel_gp_check.py`: GP factors, duals, likelihood, mean/std and clamp diagnostics.
- `training/checks/gp_parallel_check.mojo`: covariance cells and structural diagonal indexing.
- Existing `tools/byte_lm_session_check.py --run` for the refactored step.

Every report's scope is limited to the hardware, inputs and configurations
actually executed. A passing same-vendor replay gate is not a three-vendor card.

### Neighbors and density queries

`ParallelQueries` in `mojolearn.parallel_neighbors` accepts fitted
`NearestNeighbors`, `RadiusNeighbors`, `KNeighborsClassifier`,
`KNeighborsRegressor`, and `KernelDensity` instances in IDENTICAL mode.
These estimators store reference data during fit; GPU work occurs at query time.

```python
from mojolearn.parallel_neighbors import ParallelQueries
with ParallelQueries(model, devices=(0, 1), rows_per_shard=128) as queries:
    result = queries.query(X_query, method="kneighbors")
```

Admitted methods are `kneighbors`, `radius_neighbors`, `predict`,
`predict_proba`, or `score_samples`, as appropriate to the estimator. Query
keywords pass to the original method. Radius queries with `X=None` retain self
edges. Whole query rows retain the original complete reference traversal,
selection, voting and density reduction; results are joined by copying bytes
in input order. Ragged radius lists and multi-target probabilities retain their
original structure. Workers persist until `close()` or context exit. Calls on
one driver must be serialized. Each call sends the current fitted estimator.

`last_shards_` records actual row ranges, devices and native query diagnostics
only after a successful call. The fitted estimator's diagnostics are untouched.
The full reference index is replicated; this is query-work partitioning, not
pooled index memory. Density `score` is deliberately outside this API; the
original host summation can consume the complete ordered `score_samples` output.
The two-H100 cloud gate passed 64 cases, including duplicate references, ragged
radius results, self edges, weighted/multi-target predictions, all six density
kernels, failed worker waves and restart. Receipts are in
`bench/results/multi_gpu/2026-09-14/neighbors-h100/`. This is not new
cross-vendor or pooled-index capacity qualification.


### DBSCAN neighborhood rows

`mojolearn.parallel_classical.fit_dbscan(model, X, devices=(0, 1),
sample_weight=weights)` distributes the original brute-force L2/L1 or RBC L2
neighborhood rows. Brute adjacency and degrees, and RBC count/fill/bounded
one-pass CSR arrays, are assembled in their original row order using integer
offsets. Weighted degrees keep the original root reduction; core points,
connectivity propagation and label merges retain the original batch order.
Every fit publishes its fitted state only after successful completion.

Cloud checks on two H100s passed 18 full fits (including signed/zero weights),
12 complete neighborhood-array cases, and nine existing native gate groups
on both one and two devices. The native groups cover both RBC loop arms,
small memory budgets, batching and weighted-oracle/sabotage checks. Receipts:
`bench/results/multi_gpu/2026-09-14/dbscan-h100/`.

The complete reference data/index, batch graph and label state remain on the
root. Per-call contexts also replicate the reference index. This is distributed
neighborhood computation; it establishes neither pooled-index capacity nor a
speedup or new cross-vendor qualification.

### Reference-sharded nearest neighbors

`ReferenceShardedNeighbors` in `mojolearn.parallel_neighbors_reference` accepts
fitted brute/auto `NearestNeighbors`, `KNeighborsClassifier`, and
`KNeighborsRegressor` in IDENTICAL mode:

```python
from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
with ReferenceShardedNeighbors(model, devices=(0, 1),
                               reference_rows_per_shard=1_000_000) as queries:
    distances, indices = queries.kneighbors(X_query)
```

Each worker receives only its reference shard and query rows. The driver stages
one device wave at a time, merges native distance-bit/global-index keys, and
repeats the native host ordering without recomputing distances. Predictions use
the original vote half. `predict` and `predict_proba` are admitted as appropriate.
The fitted host reference and complete target table remain; this is a streamed
capacity path, not an all-resident VRAM pool. RBC and radius are outside it.

Two-H100 checks passed 57 cases and a 96 GiB logical index, using sixteen 6 GiB
shards and an exact GPU oracle whose nearest reference is the final global row.
The original 64 whole-query receipts retain their exact hashes. The 96 GiB run
predates the final signed-zero/NaN host-order correction; the final 57-case and
2 GiB checks cover that correction. Exact sources, binaries and logs are in
`bench/results/multi_gpu/2026-09-14/reference-graph-h100/`.

### Hierarchy, spectral clustering and UMAP

`mojolearn.parallel_graph.fit_graph(model, X, devices=(0, 1))` supports
`AgglomerativeClustering`, `SpectralClustering`, and `UMAP` in IDENTICAL mode.
It distributes native distance/neighbor rows while retaining original norm
bytes and global graph, merge, eigensolver and optimizer order. Spectral's
KMeans assignments use the existing row driver. `transform_umap(model, X,
devices=(0, 1))` distributes the transform's neighbor search while preserving
the complete transform's optimizer/RNG order. Fit publication is atomic.

Fifteen two-H100 public cases pass complete fitted-state/output comparisons;
six native cases compare every distance and selected index bit. Full root
reference, graph and solver state remain. This establishes compute partitioning
for these paths, not pooled graph capacity or new cross-vendor identity.


### Pointwise and ordered boosting

`fit_boosting` also supports `use_pointwise_searcher=True`, and accepts the
public `GradientBoostingClassifier` and `GradientBoostingRegressor` adapters
with their label/probability and evaluation-set contracts. Whole packed feature
groups are assigned to devices (32 binary, eight half-byte, four one-byte
features per group). Each group keeps the original complete document fold,
prefix scan and sibling subtraction. The root gathers disjoint interleaved
weight/target pairs before the original global split selection. No cross-shard
histogram summation is introduced. A policy uses at most one device per group.
The opt-in private-document-slots experiment is refused by this driver.

```python
from mojolearn.parallel_ensemble import (
    fit_boosting, fit_ordered_rmse, fit_feature_freq,
)
fit_boosting(model, X, y, devices=(0, 1), sample_weight=weights)
fit_ordered_rmse(ordered_model, X, y, permutation=order, devices=(0, 1))
fit_feature_freq(feature_freq_model, X, y, devices=(0, 1))
```

`fit_ordered_rmse` retains the supplied permutation, growing-prefix
approximations and ordered leaf updates. `fit_feature_freq` retains categorical
candidate generation and both sequential level transitions; only its existing
greedy histogram work is distributed. Every driver publishes fitted state only
after successful completion.

Two-H100 evidence covers nine pointwise fits with identical histogram dumps and
trace records, six full/partial native histogram comparisons (including one-hot
features), 16 OrderedRMSE fits and traces, four two-level categorical fits, and
four adapter fits. Sixteen previous greedy fixtures retain their exact receipt
hashes. The initial interleaved-pair gather defect and its failing traces are
preserved alongside the corrected results in
`bench/results/multi_gpu/2026-09-14/pointwise-h100/`.

These paths retain full root data, histogram and model state; pointwise workers
currently clone the full compressed index and histogram before gathering only
owned bins. They do not establish pooled model capacity, performance scaling,
or new cross-vendor identity. Wider configurations remain to be qualified.


### Wider Gram matrices and minimum-norm OLS

`fit_gram_estimator` now also admits more than 128 features and wide
`LinearRegression` (more columns than rows). Existing 1..128-feature Gram
chunks keep their previous schedule. Wider column Gram matrices partition
output rows on the existing FP32-v1 contraction profile; each row retains all
contraction terms. Wide OLS instead partitions its original sequential row-Gram
kernel, then runs the existing minimum-norm eigensolver and updates unchanged.
No cross-device floating-point sum is introduced. Requests beyond the new
driver's signed 32-bit copy/output indexing are refused before staging.

Two-H100 qualification passes 21 raw-matrix comparisons (including 513-feature
outputs), indexing-limit refusals, 18 full-model cases through 257 features,
and all 20 previous Gram cases with exactly unchanged receipt hashes. The
full-model cases cover OLS with/without intercept, weighted tall OLS, Ridge,
whitened covariance PCA and TruncatedSVD. Receipts and exact source archives:
`bench/results/multi_gpu/2026-09-14/gram-outputs-h100/`.

Full root data, eigensolver matrices and model state remain. This is compute
partitioning, not pooled model/state capacity or new cross-vendor qualification.


### Tall full PCA

`fit_gram_estimator` also accepts tall `PCA(svd_solver="full")`. It assigns
original TSQR panels to devices, gathers each panel's R and destroyed input in
the original positions, then runs the original stacked-R factorization on the
root. Panel boundaries, lane reductions and the one-panel path remain unchanged.
Wide full PCA's transpose-QR route is still refused by this driver.

Two-H100 gates compare complete native QR state for five shapes (1 through 64
panels) and eight public fitted-state/transform/inverse-transform cases with
and without whitening. Evidence: `bench/results/multi_gpu/2026-09-14/qr-h100/`.
Root input and solver state remain resident; this does not establish pooled
capacity, performance scaling or new cross-vendor identity.


### Byte-LM optimizer-state pooling

`ParallelByteLanguageModelTrainer` owns AdamW moment buffers and rollback
copies in disjoint contiguous parameter ranges by default. Each GPU computes
the original microbatch gradient; the existing ordered full-gradient sum stays
unchanged. All ranges are snapshotted before any update. Each owner runs the
original elementwise AdamW kernel, then broadcasts its updated parameter slice.
On failure the group restores every owned range before rebuilding replicas.

`optimizer_ownership()` reports actual native ranges and allocated moment,
rollback and gradient-reduction bytes. Across K devices moments and rollback
total 20 bytes per parameter (reduction scratch adds eight),
compared with 20*K for complete optimizer replicas. Parameters, full gradients,
model weight copies and activations remain replicated; this is optimizer-state
pooling, not full pooled model capacity. `pool_optimizer=False` retains the
replicated optimizer path for direct checks. Portable checkpoints contain full
canonical state and can reopen with a different physical device count.

The byte-LM optimizer admits AdamW without clipping. SmallMLP/Samba use the shared host-staged optimizer path below, which preserves
their global clipping and per-tensor optimizer contracts.


### Shared neural optimizer pooling

`ParallelNeuralTrainer` uses a cooperative update worker across its selected
devices. The first selected GPU retains the original ordered gradient sum and,
when enabled, the original complete-registry global-norm clip. Disjoint
parameter/moment ranges then use the existing SGD, Adam or AdamW step with
clipping disabled because it has already run. SGD ranges retain each original
tensor's momentum flag, including when a tensor spans multiple devices.

The update is host staged: each GPU allocates only its parameter, gradient and
moment range during that phase. The full-gradient clipping allocation is freed
before those updates. Full host arrays stage all results, and caller state is
published only after every worker succeeds. This is not persistent optimizer
residency or pooled model weights/activations. SmallMLP and Samba gradient
workers still need a complete host model.

The native entry selects this path with `MOJOLEARN_OPTIMIZER_DEVICE_COUNT`;
cooperative workers set it to their selected device count, and one device
retains the original entry. `tools/parallel_optimizer_check.py` compares all
state, clipped gradients and norm/coefficients for SGD (including dampening
and Nesterov), Adam and AdamW; `tools/parallel_training_check.py` exercises
full MLP and Samba training, including clipping and attention/dropout.


### Distributed byte-LM gradient-sum scratch

The pooled byte-LM path also owns its two gradient-reduction scratch buffers
in the same parameter ranges as optimizer state. Each range copies logical
gradient zero and left-folds the remaining logical gradients with the original
FP32 add kernel. Different ranges never participate in a floating-point sum
with each other. Disjoint completed ranges are copied into each replica's
full gradient before the original scan and pooled optimizer update.

Reduction scratch totals eight bytes per parameter across the group, divided
among owners instead of concentrated on the first GPU. On two GPUs this
halves the first GPU's reduction-scratch allocation. Model parameters, full
gradient replicas and activations still require further memory partitioning.
The replicated optimizer comparison path retains its original root reduction.


### H100 / RTX 5090 replay evidence

A build from the exact frozen H100 source tree passes all 16 corresponding
receipt groups on two RTX 5090s: pooled neural optimizers, byte-LM replay,
MLP/Samba (including clipping and attention/dropout), boosting, wider Gram
and tall full PCA. Complete structured receipts match, excluding only prose
scope; raw pointwise histogram dump bytes and pointwise/OrderedRMSE trace
records are also compared directly. Evidence and exact source identities:
`bench/results/multi_gpu/2026-09-14/rtx5090-pooled-replay/`.

Both GPU types are NVIDIA. These fixtures establish neither new AMD/Apple
qualification nor throughput scaling or full pooled-model capacity.

The frozen-source record precedes the distributed gradient-scratch change.
That change separately passes RTX 5090 ownership, 2/3/5/8-logical-shard replay
and recovery gates, retaining the earlier output hashes; its evidence is in
`bench/results/multi_gpu/2026-09-14/byte-gradient-pool-rtx5090/`.


### Layer-owned byte-LM model training

`mojolearn.model_pool_training.PooledByteLanguageModelTrainer` places decoder
layers on separate GPUs. Canonical parameters, AdamW moments, gradients and
rollback state are partitioned; no GPU owns a complete model replica. Layer
kernel weight copies and saved forward/backward stages stay with their owner.
The first selected device owns the embedding and language-model head.

```python
from mojolearn.model_pool_training import PooledByteLanguageModelTrainer

with PooledByteLanguageModelTrainer(state, devices=(0, 1), logical_shards=3) as trainer:
    result = trainer.train_step([microbatch0, microbatch1, microbatch2])
    checkpoint = trainer.checkpoint()
    ownership = trainer.model_ownership()
```

The initial schedule visits layers sequentially and copies activations and
cotangents across owners. Each existing kernel retains its microbatch shape;
logical gradients use the original fixed-order FP32 sum. This is capacity
pooling, with no throughput claim. A layer and the embedding/head must each
fit their owner. The current admission permits up to one more GPU than decoder layers
(the extra owner holds embedding/head), with at most 64 GPUs. Logical microbatch count is
independent of GPU count, so two GPUs can jointly train one microbatch.

All owners snapshot before any update; a failed update restores every chunk.
The canonical checkpoint retains `mojolearn.parallel-byte-lm.v1` and can move
between this trainer and `ParallelByteLanguageModelTrainer` without changing
the logical shard order. `model_ownership()` reports actual canonical buffer
allocations, excluding kernel weight copies, activations and workspaces.

Qualification and capacity receipts live in
`bench/results/multi_gpu/2026-09-14/byte-model-pool-rtx5090/`. The new model
path has its own same-source H100/RTX 5090 comparison: nine complete
state/gradient/loss receipt groups match. Its 958,746,624-parameter capacity
fixture completes a step on two RTX 5090s while the same driver runs out of
memory on one; resident use is 21766/21732 MiB. These are NVIDIA architecture
checks, with no AMD/Apple qualification or throughput claim. This does not extend model pooling to MLP/Samba
or eliminate the remaining root allocations in classical/tree estimators.

### One-GPU replay with host offload

`mojolearn.offload_training.OffloadedByteLanguageModelTrainer` replays the same
ordered checkpoint while keeping canonical parameters, moments, gradient sums
and saved layer inputs in host memory. It loads one decoder layer at a time;
backward reconstructs that layer's original forward stages from the saved
input and unchanged weights. Embedding/head buffers remain on the selected GPU.
All arithmetic, including each sequential gradient add and AdamW update, uses
the existing GPU kernels. Host work copies bytes and stages the transaction.

```python
from mojolearn.offload_training import OffloadedByteLanguageModelTrainer

with OffloadedByteLanguageModelTrainer.from_checkpoint(
    checkpoint, devices=(0,),
) as replay:
    replay.train_step(next_microbatches)
    continued = replay.checkpoint()
```

Exactly one physical device is required. Logical microbatch count, order,
shape, optimizer configuration and corpus bytes must agree with the pooled
run. One decoder layer with its training buffers, or one optimizer chunk, plus
the embedding/head must fit the GPU, and host
memory must hold the full state, gradient sums, saved inputs and transaction
copies. This is a capacity/replay path with extra transfers and recomputation;
no throughput claim is made. It does not offload a single oversized layer.

Updates run one canonical chunk at a time and stage their results in separate
host arrays. The trainer publishes the new state only after all chunks pass
the existing device scans. Failure leaves the previous state intact; failure
to publish a successful native result restores the retained host snapshot.
The checkpoint contract is shared with both resident byte-LM trainers.

Cloud gates are `tools/byte_lm_offload_check.py`,
`training/checks/byte_lm_offload_check.mojo` and
`tools/byte_lm_offload_capacity_check.py`. Qualification receipts are in
`bench/results/multi_gpu/2026-09-14/byte-offload-h100/`.

The earlier parallel-driver baseline also has a separate
[AMD/NVIDIA/Apple identity record](../../bench/results/identity_break/2026-09-14_136-lanes/README.md):
sixteen parallel lanes on two MI300X GPUs and two H100s match single-device
AMD/Apple replay for 144 training cells. That frozen record predates the new
layer-pooling/offload implementations and does not qualify their added paths.

The new offload path passes twelve H100 shape/logical-count groups, including
eight logical shards, plus native and injected-failure checks. Its
958,746,624-parameter fixture matches two-GPU pooled P/M/V, flags, gradients
and losses for two consecutive updates of three logical microbatches each.
Observed GPU-memory samples peak at 2621 MiB on the offload device versus
21551 MiB per pooled H100. Full state resides in host RAM. RunPod had no
RTX5090 stock for this leg, so actual RTX5090 and AMD/Apple offload execution
remain unqualified. These measurements establish capacity and exact replay
for the recorded fixtures, with no throughput or all-estimator pooling claim.


### Pooled neural gradient buffers

MLP/Samba's cooperative update worker partitions accumulation columns across
its selected GPUs. Each owner receives every microbatch for its assigned
columns in the original order and calls the existing accumulation kernel.
Column ownership changes storage and launch geometry; it does not combine
per-device partial sums. The original per-column balanced tree is unchanged.
`ParallelNeuralTrainer` still invokes that kernel as a pair add at each step
of its defined logical left fold. Standalone accumulation retains its own
balanced-tree and token-alignment contract.

The result stays in host staging until all GPU owners join successfully.
Native admission scans the complete input before starting workers, and a
failed worker cannot publish another worker's completed columns. Each GPU
allocates only its local stacked inputs and reduction scratch; the host still
holds full input/output arrays. The original global norm clip remains on the
first device. This is another pooled allocation in the neural training path,
not a complete resident Samba model or a throughput claim.
