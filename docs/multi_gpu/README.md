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

## Available paths

| Surface | Partition | Numerical contract |
| --- | --- | --- |
| Byte language model | Logical microbatches across resident device replicas | Copy gradient 0; left-fold gradients 1..K-1; one update per replica |
| SmallMLPTrainer | Frozen microbatch snapshots on concurrent GPU workers | Same ordered sum; one update on the first selected GPU |
| SambaStack (Mamba/attention blocks) | Frozen microbatch snapshots on concurrent GPU workers | Same ordered sum; explicit logical dropout stream/offsets |
| RandomForest classifier/regressor | Whole trees over full replicated data | Original global tree IDs, seed and quantiles; original prediction order |
| ExtraTrees classifier/regressor | Whole trees over full replicated data | Original global tree IDs and full-data label quantization; original prediction order |
| KMeans | Whole row tiles during distance/assignment | Original per-row feature/centroid arithmetic; original initialization, full-data updates and convergence |
| GradientBoosting, greedy symmetric/depthwise/lossguide | Whole packed feature groups during histogram construction | Original row-reduction geometry and global scale; disjoint histogram-column copies |
| ARIMA / ExponentialSmoothing | Independent series assigned to workers | Original per-series initialization, solver and likelihood arithmetic |
| LinearRegression / Ridge / covariance PCA / TruncatedSVD | Original 128 Gram chunks at 1..128 features | Original chunk partials copied to global positions; unchanged final fold and root solver |
| LogisticRegression | QN gradient feature columns | Original per-cell row reduction; unchanged root objective, line search and optimizer |
| Lasso / ElasticNet | Whole FP32-v1 dot leaves during cyclic coordinate descent | Original balanced tree; unchanged coordinate and convergence order |
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
frozen snapshot and compute gradients concurrently; the first selected GPU
reduces and applies one update, then publishes the owner state. Parameters, moments and RNG state are restored on a failed step.
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
from mojolearn.parallel_ensemble import fit_boosting
from mojolearn.parallel_classical import (
    fit_arima, fit_exponential_smoothing, fit_gram_estimator, fit_logistic,
    fit_coordinate_descent,
)
from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler

fit_boosting(boosting, X, y, devices=(0, 1), sample_weight=weights)
fit_arima(arima, series, devices=(0, 1), series_per_shard=2)
fit_exponential_smoothing(holtwinters, devices=(0, 1), series_per_shard=2)
fit_gram_estimator(ridge, X, y, devices=(0, 1))
fit_logistic(logistic, X, labels, devices=(0, 1))
fit_coordinate_descent(lasso, X, y, devices=(0, 1))
fit_scaler(scaler, X, devices=(0, 1), columns_per_shard=16)
scaled = transform_scaler(scaler, X, devices=(0, 1), columns_per_shard=16)
```

Boosting currently supports the greedy symmetric, depthwise and lossguide
searchers, including the classifier/regressor aliases. It refuses the
pointwise searcher and the separate OrderedRMSE/ExperimentalTwoLevelFeatureFreq
classes. Histogram shards own compressed feature columns and local histogram
columns, but the root still owns the full index and histogram. Per-level
allocation and staging overhead may outweigh computation savings.

ARIMA workers receive only their assigned series; scaler workers receive only
their assigned columns. These partitions reduce the GPU memory required per
worker. The host holds the complete data and result, and a single shard still
must fit on one GPU. ARIMA prediction currently uses the existing single-GPU
methods; scalers provide an explicit distributed transform entry. No run beyond
one GPU's memory capacity has been qualified. These new paths have two-H100
equality evidence, not new AMD/Apple or NVIDIA cross-architecture qualification.

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
memory capacity. Neural state and several classical root paths still require
one GPU to hold complete state. Independent-series and scaler-column workers
receive only their partitions, but beyond-single-GPU capacity is not qualified.

The user's requested order is neural training, forests/ExtraTrees, then
boosting and classical estimators. The following remain unimplemented:

- OrderedRMSE, pointwise boosting and categorical two-level feature search: boosting rounds
  depend on preceding predictions, so the forest tree-range driver is invalid.
  A dedicated feature/histogram partition must preserve quantization, global
  scales, row order, split tie breaks, leaf estimates and categorical state.
- Wider/full-solver Gram paths and SVM:
  distribute the appropriate matrix or objective work without changing its
  reduction tree or solver trajectory.
- Neighbors/density, graph/manifold methods, mixture models,
  IsolationForest and other classical surfaces: each needs its
  own partition and qualification. Some have little training work to split.
- Resident staging reuse, larger models,
  eight physical GPUs, H100/5090 replay, AMD/Apple cross-vendor evidence,
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
- Existing `tools/byte_lm_session_check.py --run` for the refactored step.

Every report's scope is limited to the hardware, inputs and configurations
actually executed. A passing same-vendor replay gate is not a three-vendor card.
