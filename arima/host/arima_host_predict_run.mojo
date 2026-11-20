# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN of `bindings/arima_host_predict.mojo` (wave 8 run split): the host
ARIMA predict and forecast bodies the doors ran with the GIL released, typed
Mojo only. Imports no Python (tools/python_free_dirs.sh). The statements are
the former GIL-released blocks unchanged, except that the block's final
`written = ...` assignment is this function's `return` (lane
rs-sequence-tokenizer, 2026-11-12)."""
from arima.host.arima_oracle import arima_host_refuse_unrestated
from arima.host.arima_host_entries import (
    _forecast_into,
    _order,
    _predict_in_sample_into,
    _refuse_shape,
)


def arima_predict_run(
    y_address: Int, exog_address: Int, exog_fut_address: Int, params_address: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int, n_obs: Int, start: Int, end: Int,
    p: Int, d: Int, q: Int, P: Int, D: Int, Q: Int, s: Int, k: Int, n_exog: Int,
) raises -> Int:
    # `arima_predict_ptr_host` then `_predict_into`
    # (`arima/estimator.mojo:502-506`, `:422-433`), in their order.
    var order = _order(p, d, q, P, D, Q, s, k, n_exog)
    _refuse_shape(batch_size, n_obs, "arima_predict")
    if start < 0 or end <= start:
        raise Error(
            "arima_predict: need 0 <= start < end (start=" + String(start)
            + ", end=" + String(end) + ")"
        )
    if start > n_obs:
        raise Error(
            "arima_predict: there can't be a gap between the data and the"
            " prediction (start=" + String(start) + ", n_obs="
            + String(n_obs) + ")"
        )
    arima_host_refuse_unrestated(order, "arima_predict")
    if start < n_obs:
        return _predict_in_sample_into(
            y_address, exog_address, exog_fut_address, params_address, op,
            batch_size, n_obs, start, end, order,
        )
    else:
        return _forecast_into(
            y_address, exog_address, exog_fut_address, params_address, op,
            batch_size, n_obs, end - n_obs, order,
        )


def arima_forecast_run(
    y_address: Int, exog_address: Int, exog_fut_address: Int, params_address: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int, n_obs: Int, n_steps: Int,
    p: Int, d: Int, q: Int, P: Int, D: Int, Q: Int, s: Int, k: Int, n_exog: Int,
) raises -> Int:
    # `arima_forecast_ptr_host` then `_predict_into`
    # (`arima/estimator.mojo:536-545`, `:422-433`), in their order.
    if n_steps < 1:
        raise Error(
            "arima_forecast: n_steps must be >= 1 (n_steps="
            + String(n_steps) + ")"
        )
    var order = _order(p, d, q, P, D, Q, s, k, n_exog)
    _refuse_shape(batch_size, n_obs, "arima_forecast")
    arima_host_refuse_unrestated(order, "arima_forecast")
    return _forecast_into(
        y_address, exog_address, exog_fut_address, params_address, op,
        batch_size, n_obs, n_steps, order,
    )
