# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The exogenous regressors across the Python boundary (lane/arima-exog,
2026-09-15): the caller's `(batch_size, n_rows, n_exog)` C-order array to
the filter's layout, with the non-finite refusal by name.

HOST ONLY, no arithmetic: a permuting copy. Shared by the GPU entries
(`arima/estimator.mojo`) and the host bindings
(`bindings/_mojolearn_arima_host.mojo`, `bindings/arima_host_predict.mojo`),
so every door reads the same cells in the same order and refuses the same
value with the same sentence.

DEVIATION 996: THE LAYOUT. cuML's `exog` is `(n_obs, n_exog * batch_size)`
in Fortran order with each series' regressors in adjacent columns
(`arima.pyx:163-166`), which is the flat layout its strided-batched gemm
reads, `[bid*n_exog*nobs + i*nobs + t]` (`batched_kalman.cu:933-947`). This
package's `y` is `(batch_size, n_obs)` (the `ARIMA` class says why), so
`exog` is `(batch_size, n_obs, n_exog)`: series, then time, then regressor,
statsmodels' `(n_obs, k_exog)` with a batch axis in front. The filter keeps
theirs, and this file is the one permutation between the two.

DEVIATION 997: A NON-FINITE EXOGENOUS VALUE IS REFUSED BY NAME. Theirs
checks `exog` with `ensure_all_finite=False` (`arima.pyx:344-351`,
`:705-712`) and has no missing-value arm for it: a NaN regressor goes
through the gemm into `obs_intercept` and makes the prediction, the
innovation and the log-likelihood NaN with the vendor's payload, silently.
A non-finite `y` is refused by name here (`_refuse_non_finite`); the
regressors take the same rule, naming the series, row and regressor.
"""
from std.math import isfinite

from bindings.hostptr import read_f32


def exog_filter_layout(
    address: Int, batch_size: Int, n_rows: Int, n_exog: Int, name: String
) raises -> List[Float32]:
    """`batch_size * n_rows * n_exog` float32 read from `address` as
    `(batch_size, n_rows, n_exog)` C order, returned as `[b*n_exog*n_rows +
    i*n_rows + t]`. A list of one zero when there is nothing to read
    (`n_exog == 0` or `n_rows == 0`), and `address` is then not read."""
    var n = batch_size * n_rows * n_exog
    if n <= 0:
        return List[Float32](length=1, fill=Float32(0.0))
    var src = read_f32(address, n)
    var out = List[Float32](length=n, fill=Float32(0.0))
    for b in range(batch_size):
        for t in range(n_rows):
            for i in range(n_exog):
                var v = src[(b * n_rows + t) * n_exog + i]
                if not isfinite(v):
                    raise Error(
                        "ARIMA: " + name + " contains a non-finite value at series "
                        + String(b) + ", row " + String(t) + ", regressor " + String(i)
                        + "; exogenous regressors have no missing-value arm and a"
                        + " non-finite one is refused by name (DEVIATION 997)"
                    )
                out[b * n_exog * n_rows + i * n_rows + t] = v
    return out^
