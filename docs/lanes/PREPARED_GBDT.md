# Reusable native numeric GBDT datasets

`gbdt.prepared.prepare_numeric_dataset` prepares a dense numeric pool once.
Its `PreparedNumericDataset.fit` method reuses the quantization grid,
compressed device index, labels, weights and device context for fresh fits.
It supports SymmetricTree, Depthwise and Lossguide with RMSE, Logloss and
CrossEntropy in FAST, IDENTICAL and DETERMINISTIC builds.

```mojo
from max.gpu.host import DeviceContext
from gbdt.prepared import prepare_numeric_dataset

var ctx = DeviceContext()
# x_colmajor and y are List[Float32]; x is feature-major.
var pool = prepare_numeric_dataset(
    ctx, x_colmajor, y, n_rows, n_features,
    border_count=32, random_seed=UInt64(42),
)
var first = pool.fit(
    grow_policy="Lossguide", max_leaves=31, n_estimators=100,
    learning_rate=Float32(0.1), random_seed=UInt64(42),
)
var second = pool.fit(
    grow_policy="Lossguide", max_leaves=15, n_estimators=100,
    learning_rate=Float32(0.05), min_split_gain=Float64(0.1),
)
```

Both results are ordinary `TrainedModel` values, usable with existing
prediction and model serialization. The second fit starts a fresh ensemble;
it does not continue the first model.

Preparation uses the exact grid-building body used by `train()`, including
sample selection and NaN handling. Inputs are copied into owned storage.
Changing the caller's arrays afterward cannot change the pool. Treat its
fields as read-only and run fits on one pool sequentially.

The preparation seed fixes the sampled quantization grid. A later fit seed
changes training randomness while retaining that grid. To compare against
ordinary `train()` with sampled borders, use matching preparation options
and seed. Rebuild the pool when changing rows, labels, weights, border count
or NaN policy. Never reuse a training pool across validation folds that have
different training rows.

The API currently serves native Mojo callers. It does not expose Python
handles, categorical/target-dependent CTR pools, evaluation pools, class
weights, or every training objective. Per-fit histogram and leaf-estimation
workspaces are still allocated by the training driver; the pool eliminates
repeated data preparation and input-buffer allocation, not all allocations.

`bootstrap_type` uses native `BOOTSTRAP_KERNEL_*` constants. Its parameter is
Bayesian temperature, Bernoulli subsample, or Poisson rate (lambda), rather
than a cross-library option alias. Inputs are checked before GPU launches.

Validation command:

```sh
python3 tools/gbdt_prepared_check.py --out bench/results/prepared-gbdt
```

The gate compares complete serialized models, predictions and Float64 loss
bits against ordinary fits across policies, losses, weighted/unweighted
rows, sampled/full grids, NaNs and a constant feature. It also checks repeated
fits, caller mutation, invalid Poisson inputs and an alternating-order
repeated-fit benchmark. Report one-time preparation separately from reuse.
