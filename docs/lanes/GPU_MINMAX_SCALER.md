# GPU MinMaxScaler: bounded dense Float32 preprocessing

`mojolearn.MinMaxScaler` learns per-feature bounds and applies an affine
transformation on the GPU. This is optional preprocessing for tree models;
it does not change a forest's training algorithm. `StandardScaler` remains
the next preprocessing slice.

```python
import numpy as np
from mojolearn import MinMaxScaler

X = np.array([[1, 5], [3, 5], [5, 5]], dtype=np.float32)
scaler = MinMaxScaler(feature_range=(0, 1), numeric_mode="identical")
scaled = scaler.fit_transform(X)
restored = scaler.inverse_transform(scaled)
```

## Contract and source references

The bounded surface accepts finite, nonempty dense two-dimensional Float32
arrays. Other dtypes require an explicit cast. Sparse inputs, NaNs and
infinities are refused. It provides fit, transform, fit_transform and
inverse_transform with copied Float32 outputs. This is a GPU implementation;
there is no new CPU backend. The array element count must fit in Int32.

`feature_range=(lower, upper)` must contain finite numeric endpoints that
remain strictly ordered after Float32 conversion. `copy=True` is required;
`copy=False` is refused. `clip=True` bounds forward transforms to the fitted
feature range; inverse transforms cannot recover values removed by clipping.
Weights and `partial_fit` are explicitly refused. Fitted nonfinite statistics,
nonpositive scales and nonfinite transform outputs are refused rather than
returned as usable results.

For each feature, the scaler learns `data_min_`, `data_max_`, `data_range_`,
`scale_` and `min_`. It records `n_features_in_` and `n_samples_seen_`.
This implementation follows sklearn's near-constant rule: `data_range_`
below `10 * Float32 epsilon` uses denominator one for `scale_`. It preserves
the measured range in `data_range_`. For an exactly constant column, observed
training values map to the lower bound subject to Float32 rounding; a
near-constant column is not stretched to fill the entire feature range.
The affine operation is `X * scale_ + min_`; inverse transformation subtracts
`min_` and divides by `scale_`. Without clipping, unseen values can transform
outside the requested feature range.

The behavior reference is
[sklearn 1.8 MinMaxScaler source](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/preprocessing/_data.py).
The local cuML source was also inspected at commit
[00094f7e, third-party preprocessing](https://github.com/rapidsai/cuml/blob/00094f7e4e4b5da3a968d193a4da6085fa38f11b/python/cuml/cuml/_thirdparty/sklearn/preprocessing/_data.py).
The kernel lane also inspected cuML v26.08.00 at
[265b9da6](https://github.com/rapidsai/cuml/blob/265b9da6a0e75dbef071a3168398b993a5ff6f0e/python/cuml/cuml/_thirdparty/sklearn/preprocessing/_data.py),
which retains the exact-zero helper; its line ranges appear in the native header.
Their constant-feature policies differ: sklearn's vector helper replaces
ranges below ten times the dtype epsilon with one, whereas that cuML source
replaces only exact zero. Both sources support NaN-aware statistics and
incremental updates beyond this bounded implementation. These references
define behavior to compare; they do not establish a literal source implementation or
full sklearn/cuML parity. The native header and
native header record local
`MINMAX-1` (near-constant rule), `MINMAX-2` (integer extrema) and `MINMAX-3`
(Float32/FTZ and finite-input/output limits) deviations.

## Modes and qualification

`numeric_mode=None` resolves the process default at fit; explicit FAST,
DETERMINISTIC or IDENTICAL selects that artifact. Fit records `numeric_mode_`,
`feature_range_` and `clip_`; subsequent transforms use that fitted state even
if the process default changes. Pickling retains the resolved mode.
`get_params`/`set_params`, fitted-state checks and transformer tags provide a
bounded sklearn protocol. Changing parameters through `set_params` clears
fitted state; this does not claim support for every sklearn meta-estimator.

Fit reads row-major input in 256-row chunks per feature. Shared-memory
integer min/max trees reduce each chunk; one thread per feature then scans
its chunk extrema. Sortable integer keys preserve subnormal extrema and select
negative zero for the minimum and positive zero for the maximum when both
occur. This signed-zero rule is explicit rather than dependent on arrival
order or a vendor's floating min/max instruction.

Learned `data_min_`/`data_max_` preserve those input bits. Subsequent range,
scale, offset and transform calculations follow the mode's Float32 arithmetic
policy, including IDENTICAL flushing subnormal operands/results. Therefore
`data_range_` need not equal an unflushed host subtraction of the stored
extrema for subnormal inputs. Multiplication uses `identical_mul` and division
uses `identical_div`; IDENTICAL forward transform pins the rounded multiply
before the add. FAST and DETERMINISTIC retain their existing compiler arithmetic
policies.

Fit scratch uses two UInt32 arrays of shape `(ceil(n / 256), d)`, or
O(ceil(n/256) * d) storage, in addition to input and returned statistics.
Per-feature finalization scans chunk extrema serially; transform/inverse
operations are elementwise GPU kernels. Native host entry points validate
inputs and refuse nonfinite results or nonpositive scales after download.
Host packing, uploads and returned arrays mean this is not a fully resident
GPU pipeline. Neither this allocation plan nor reuse of existing arithmetic
primitives is a measured speed improvement.

All three mode builds and their native launch checks passed on the Apple M4.
The public GPU smoke passed 309 checks, and the sklearn clone/tags/parameter
smoke passed. One public-run CoreAnalytics context diagnostic remains unresolved.
Cross-vendor numerical qualification, broad adversarial checks and throughput
measurements remain pending. Neither the IDENTICAL mode name nor earlier
metrics/forest evidence certifies this new scaler across devices. No speed
or cross-device bit-identity claim is made.

The focused task is `pixi run check-minmax-scaler`; it builds the preprocessing
extension and runs `checks/minmax_scaler_smoke.py`. See the
[local build/smoke record](../../bench/results/minmax_scaler_2026-09-10/RESULTS.md)
for completed modes and results. This does not replace broader qualification.
