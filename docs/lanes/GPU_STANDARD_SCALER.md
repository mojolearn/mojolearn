# GPU StandardScaler: bounded dense Float32 standardization

`mojolearn.StandardScaler` learns per-feature statistics and standardizes
finite dense Float32 arrays on the GPU. It complements
[MinMaxScaler](GPU_MINMAX_SCALER.md); scaling remains optional for tree models.
The next useful pipeline step is the
[B2 GBDT adapter/protocol audit](GBDT_SKLEARN_ADAPTER_PLAN.md), not a claim
that arbitrary sklearn pipelines are already supported.

```python
import numpy as np
from mojolearn import StandardScaler

X = np.array([[1, 4], [3, 4], [5, 4]], dtype=np.float32)
scaler = StandardScaler(numeric_mode="identical")
scaled = scaler.fit_transform(X)
restored = scaler.inverse_transform(scaled)
```

## Contract

Inputs are nonempty, finite, dense two-dimensional Float32 arrays. Other
dtypes require an explicit cast. Sparse inputs, NaNs, infinities, sample
weights and incremental `partial_fit` are outside this implementation.
Fit, transform, fit_transform and inverse_transform use the GPU; no CPU
backend is added. The element count `n * d` must fit in Int32.
`copy=True` is required at construction; transform/inverse accept `copy=None`
or `True`. In-place transformation is refused. `with_mean` and `with_std`
accept bool values. Returned transforms are copied Float32 arrays.

Learned attributes are Float32 vectors: `mean_` whenever either flag is
enabled, and `var_`/`scale_` when `with_std=True`. Disabled statistics are
`None`; with both flags false all three are `None`. Fit also records
`n_features_in_` and `n_samples_seen_`. Nonfinite statistics, negative variance,
nonpositive scales and nonfinite transform outputs are refused. User-mutated
mean/scale arrays must remain finite contiguous Float32 feature vectors.

The default transform centers features and divides by their learned scale.
Inverse transformation reverses those operations subject to Float32 rounding.
`with_mean` and `with_std` select centering and scaling independently.
Population variance uses denominator n, not the sample-variance n-1 rule.

## Source references and numerical scope

Behavior references are the actual
[sklearn 1.8 StandardScaler source](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/preprocessing/_data.py)
and its
[incremental mean/variance helper](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/utils/extmath.py),
alongside cuML commit
[00094f7e third-party preprocessing](https://github.com/rapidsai/cuml/blob/00094f7e4e4b5da3a968d193a4da6085fa38f11b/python/cuml/cuml/_thirdparty/sklearn/preprocessing/_data.py)
and its corresponding `utils/extmath.py`. The native header also cites
[cuML v26.08.00](https://github.com/rapidsai/cuml/tree/v26.08.00/python/cuml/cuml/_thirdparty/sklearn).
The native header records the
implementation-specific `STD-1`, `STD-2` and `STD-3` policies.

sklearn accumulates moments in Float64, applies a correction to its centered
sum of squares and detects indistinguishable constant features using a
Float64 error bound involving mean, variance and sample count. The inspected
cuML scaler uses its older exact-zero scale helper. A Float32 GPU policy is
not automatically equivalent to either implementation. This is an independent
bounded implementation, not a literal source implementation or broad sklearn/cuML
numerical-parity claim.

The GPU implementation uses a separate centered variance pass:

- `STD-1`: Float32 sums in 256-value slabs, followed by ascending GPU
  chunk folds and division by n. IDENTICAL uses a fixed logical slab tree;
  the other modes use the existing mode-specific block reduction. Squared residuals are centered around the
  computed mean. There is no Float64 accumulator or sklearn residual-sum
  correction, and no cancellation-prone difference of squared moments.
- `STD-2`: zero population variance uses scale one, following the inspected
  cuML exact-zero convention. This does not use sklearn's Float64 error bound
  or MinMaxScaler's ten-epsilon rule. Exact constant inputs are detected
  during the mean pass so accumulation rounding does not invent variance.
  Equality is evaluated after the mode's input FTZ policy; constant columns
  retain the first effective value as mean. Signed zeros compare equal,
  and IDENTICAL can treat flushed subnormal values as constant zero.
- `STD-3`: IDENTICAL applies its operand/result FTZ policy, pinned multiply
  and division, and portable square root. Nonfinite statistics or transform
  results are refused.

When `with_std=False`, the variance pass is skipped. When both flags are
false, statistics are skipped and the transform returns an exact bit copy,
including subnormal values. With statistics enabled, fit uses two Float32
scratch arrays of shape `(ceil(n / 256), d)` for partial sums and constant
markers. The variance pass reuses that scratch. Per-feature finalization
scans chunks serially; elementwise transforms run in parallel. This
O(ceil(n/256) * d) scratch cost and the serial chunk folds remain scaling
considerations, not measured speed claims.

## Modes and qualification

FAST, DETERMINISTIC and IDENTICAL select compiled arithmetic policies.
`numeric_mode=None` resolves the process default at fit. Fit records
`numeric_mode_`, `with_mean_` and `with_std_`; transforms use those captured
values even after the default or constructor attributes change. Pickling
retains learned arrays and the fitted mode. The bounded shared scaler
protocol supports get/set parameters, cloning, fitted checks and transformer
tags; `set_params` clears fitted state.
The mode name alone does not qualify a new scaler across devices. Host
validation, packing, uploads and returned arrays also mean this is not a
fully resident GPU training/scoring pipeline.

Local M4 build/smoke checks passed in FAST, DETERMINISTIC and IDENTICAL:
all three preprocessing builds passed native StandardScaler/MinMaxScaler ABI
smokes and numeric-mode readback. The final public StandardScaler smoke passed
48 fit configurations spanning all four flag combinations and all three
modes. Three small StandardScaler → RandomForest → GPU R² pipelines each
returned 1.0. These are integration fixtures, not generalization results.
The existing MinMaxScaler smoke passed 309 checks, and both scaler protocol
smokes passed. See the [recorded evidence](../../bench/results/standard_scaler_2026-09-10/RESULTS.md)
for commands, artifacts and scope.

All runs exited successfully, but runtime diagnostics remain unresolved:
the final StandardScaler run emitted three CoreAnalytics context-leak
diagnostics; its retained initial 36-configuration run emitted two, and the
MinMaxScaler run emitted one. Their origin and memory impact are unknown;
these passes do not establish memory-lifetime correctness.

Broader adversarial numerical checks, cross-device comparisons, full-wheel
qualification and throughput measurements remain pending. No speed or
cross-vendor bit-identity claim is made; earlier scaler, metric or forest
evidence does not extend automatically.
