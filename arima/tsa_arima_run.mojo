# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN body of `bindings/tsa_arima_native.mojo` (wave 8 run split): the ARIMA
information-criterion penalties, moved unchanged (the unused Python context
argument is dropped)."""

from x_metrics.epilogue import portable_log_c


def _penalties(order: List[Int], big_n: Int, n_obs: Int) raises -> Tuple[Float64, Float64]:
    """`_ic_penalties`: 2N, and log(T) N with the pinned log (T = n_obs after
    differencing)."""
    var t = n_obs - (order[1] + order[6] * order[4])
    var n_par = Float64(big_n)
    var log_t = Float64(0.0)
    if t > 0:
        # the pinned log: portable_log_c is packaging/portable_math.c `mojolearn_log`
        # operation for operation (what `_portable_math.log` returns)
        log_t = portable_log_c(Float64(t))
    return (2.0 * n_par, log_t * n_par)
