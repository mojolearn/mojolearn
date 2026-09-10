# GPU regression errors and mode-aware scoring

This slice adds `mojolearn.metrics.mean_squared_error`,
`mean_absolute_error` and `root_mean_squared_error`. It implements the first,
bounded A1 stage of [the pipeline plan](GPU_PIPELINE_PLAN.md).

```python
import numpy as np
from mojolearn import metrics

y = np.array([3, -0.5, 2, 7], dtype=np.float32)
prediction = np.array([2.5, 0, 2, 8], dtype=np.float32)
loss = metrics.mean_squared_error(y, prediction, numeric_mode="identical")
assert loss == 0.375
```

## Public contract

Inputs are finite, nonempty Float32 arrays with equal lengths, shaped `(n,)`
or `(n, 1)`. Strided/read-only arrays are supported through contiguous host
preparation. Other dtypes require an explicit cast. `sample_weight` must be
None and `multioutput` must be `"uniform_average"`; unsupported options raise
before launching. This is single-output support, including column vectors.
The result is a Python float holding a Float32 scalar.

The names and definitions follow the scikit-learn
[MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html),
[MAE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
and [RMSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html)
contracts. This is a new Mojo implementation over existing pinned reduction
primitives; matching a metric definition does not promise sklearn result bits.
Weighted and multiple-output semantics remain planned.

Residuals, squares, accumulation, division and RMSE's square root execute on
the GPU in Float32. The reduction uses source-defined chunks and a fixed
final fold. The final fold currently uses one GPU thread for `ceil(n/256)`
partials; throughput is unmeasured. Host code validates/prepares inputs, transfers data and reads back
the scalar; it does not compute these error reductions. Each call currently
owns a device context and uploads both arrays. This is not a resident buffer API.

The arithmetic contract is intentionally explicit: subtraction, squaring or
the sum can overflow to positive infinity even when a wider/scaled computation
would return a finite result. RMSE can therefore overflow when the exact RMSE
fits in Float32. IDENTICAL flushes subnormal arithmetic using the existing
numeric policy. There is no Float64 or compensated accumulator in this slice.

## Mode dispatch and sklearn scoring

New functions, `accuracy_score` and `r2_score` accept `numeric_mode=None`.
An explicit mode selects that artifact without changing the global default.
None resolves the current library default at every call. All metrics now use
the shared backend loader instead of the obsolete single-artifact cache.
The loader checks vendor provenance; the metric wrapper also checks compiled
mode readback when available (`metrics_numeric_mode`, or the historical
`umap_numeric_mode` export on the same extension).

RF/Extra Trees' default `score` passes its estimator mode to accuracy or R².
To use these new losses in sklearn search, select the MojoLearn function
explicitly:

```python
from sklearn.metrics import make_scorer

scorer = make_scorer(metrics.mean_squared_error,
                     greater_is_better=False, numeric_mode="identical")
```

The sklearn string `"neg_mean_squared_error"` selects sklearn's metric.
Using an external scorer or transformer does not qualify the complete
workflow as IDENTICAL. See [the pipeline qualification contract](GPU_PIPELINE_PLAN.md).

## Qualification

All three modes pass native checks and 261 public metric checks on Apple M4.

Native numeric checks and public binding tests cover hand-calculated cases,
independent high-precision references, chunk boundaries, noncontiguous inputs,
overflow/subnormal behavior and interleaved mode selection. Run the complete native/public gate with:

```sh
MOJOLEARN_PYTHON="$PWD/.pixi/envs/default/bin/python" pixi run check-regression-errors
```

The task holds the shared build lock. Evidence is recorded in
[the result directory](../../bench/results/regression_errors_2026-09-10/RESULTS.md).
CUDA/HIP qualification and installed-wheel cross-vendor comparison remain
pending. Historical identity evidence for older metrics does not certify
these new kernels.
