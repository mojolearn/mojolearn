# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The typed ARIMA host entries behind the Python doors
`bindings/arima_host_predict.mojo` and `bindings/_mojolearn_arima_host.mojo`
(lane thin-trees-metrics, 2026-10-21: moved word for word out of those
shells, so a Python-free caller reaches the same source; no Python here).

`_order`, `_refuse_shape`, `_forecast_into` and `_predict_in_sample_into`
are `bindings/arima_host_predict.mojo`'s, unchanged. `arima_host_fit_into`
is the body `arima_fit_binding` ran with the GIL released, after
`_refuse_method`, unchanged.
"""
from std.memory import bitcast

from arima.exog_layout import exog_filter_layout
from core.hostptr import read_f32
from arima.impl.arima_ic import arima_ic_bits
from arima.host.arima_oracle import (
    ArimaHostOrder,
    arima_host_fit,
    arima_host_forecast,
    arima_host_predict,
    arima_host_refuse_unrestated,
)
from arima.impl.tsa.arima_common import ARIMAOrder, validate_order


def _order(
    p: Int, d: Int, q: Int, P: Int, D: Int, Q: Int, s: Int, k: Int, n_exog: Int
) raises -> ArimaHostOrder:
    """`arima/estimator.mojo::_order` (`:150-162`): the device's
    `validate_order` on the nine integers, then the host order."""
    var order = ARIMAOrder(p, d, q, P, D, Q, s, k, n_exog)
    validate_order(order)
    return ArimaHostOrder(p, d, q, P, D, Q, s, k, n_exog)


def _refuse_shape(batch_size: Int, n_obs: Int, who: String) raises:
    """`arima/estimator.mojo::_refuse_shape` (`:165-175`), its words."""
    if batch_size < 1:
        raise Error(
            who + ": batch_size must be >= 1 (batch_size=" + String(batch_size) + ")"
        )
    if n_obs < 2:
        raise Error(
            who + ": n_obs must be at least 2 (n_obs=" + String(n_obs) + ")"
        )


def _forecast_into(
    y_address: Int,
    exog_address: Int,
    exog_fut_address: Int,
    params_address: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    n_steps: Int,
    order: ArimaHostOrder,
) raises -> Int:
    """The body both prediction entries share once `start == n_obs`, as
    `_predict_into` is shared on the device."""
    var N = order.complexity()
    var y = read_f32(y_address, batch_size * n_obs)
    var pr = read_f32(params_address, N * batch_size)
    var exog = exog_filter_layout(exog_address, batch_size, n_obs, order.n_exog, "exog")
    var fut = exog_filter_layout(
        exog_fut_address, batch_size, n_steps, order.n_exog, "exog (future values)"
    )
    var fc = arima_host_forecast(y, exog, fut, pr, batch_size, n_obs, n_steps, order)
    if len(fc) != n_steps * batch_size:
        raise Error(
            "arima_forecast: the host oracle returned a forecast of an"
            " unexpected length; nothing written"
        )
    for i in range(n_steps * batch_size):
        op.unsafe_store(i, fc[i])
    return n_steps * batch_size


def _predict_in_sample_into(
    y_address: Int,
    exog_address: Int,
    exog_fut_address: Int,
    params_address: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    start: Int,
    end: Int,
    order: ArimaHostOrder,
) raises -> Int:
    """`start < n_obs`: `arima_host_predict`, the in-sample block and any
    forecast after it, written series major."""
    var N = order.complexity()
    var y = read_f32(y_address, batch_size * n_obs)
    var pr = read_f32(params_address, N * batch_size)
    var ld = end - start
    var num_steps = end - n_obs if end > n_obs else 0
    var exog = exog_filter_layout(exog_address, batch_size, n_obs, order.n_exog, "exog")
    var fut = exog_filter_layout(
        exog_fut_address, batch_size, num_steps, order.n_exog, "exog (future values)"
    )
    var out = arima_host_predict(y, exog, fut, pr, batch_size, n_obs, start, end, order)
    if len(out) != ld * batch_size:
        raise Error(
            "arima_predict: the host oracle returned a prediction of an"
            " unexpected length; nothing written"
        )
    for i in range(ld * batch_size):
        op.unsafe_store(i, out[i])
    return ld * batch_size


def arima_host_fit_into(
    y_address: Int,
    exog_address: Int,
    pp: MutPointer[Float32, MutUntrackedOrigin],
    xp: MutPointer[Float32, MutUntrackedOrigin],
    x0p: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float32, MutUntrackedOrigin],
    fp: MutPointer[Int32, MutUntrackedOrigin],
    batch_size: Int,
    n_obs: Int,
    p: Int,
    d: Int,
    q: Int,
    P: Int,
    D: Int,
    Q: Int,
    s: Int,
    k: Int,
    n_exog: Int,
    max_iterations: Int,
    ic_address: Int,
    pen_aic: Float64,
    pen_bic: Float64,
) raises -> Int:
    """`arima_fit_ptr_host` (`arima/estimator.mojo:361-369`) after the
    method check, in its order: validate, fit, check the lengths, write the
    params, x, x0, stats and flags, and the AIC/BIC words when
    `ic_address != 0`. Returns `N * batch_size`."""
    var order = _order(p, d, q, P, D, Q, s, k, n_exog)
    _refuse_shape(batch_size, n_obs, "arima_fit")
    if max_iterations < 1:
        raise Error(
            "arima_fit: max_iterations must be >= 1 (max_iterations="
            + String(max_iterations) + ")"
        )
    arima_host_refuse_unrestated(order, "arima_fit")
    var N = order.complexity()
    var y = read_f32(y_address, batch_size * n_obs)
    var exog = exog_filter_layout(exog_address, batch_size, n_obs, order.n_exog, "exog")
    var r = arima_host_fit(y, exog, batch_size, n_obs, order, max_iterations)
    if (
        len(r.t_x) != N * batch_size
        or len(r.x) != N * batch_size
        or len(r.x0) != N * batch_size
        or len(r.loglike) != batch_size
        or len(r.fx) != batch_size
        or len(r.n_iter) != batch_size
        or len(r.retcode) != batch_size
    ):
        raise Error(
            "arima_fit: the host oracle returned arrays of an unexpected"
            " length; nothing written"
        )
    for i in range(N * batch_size):
        pp.unsafe_store(i, r.t_x[i])
        xp.unsafe_store(i, r.x[i])
        x0p.unsafe_store(i, r.x0[i])
    for b in range(batch_size):
        sp.unsafe_store(b, r.loglike[b])
        sp.unsafe_store(batch_size + b, r.fx[b])
        fp.unsafe_store(b, r.n_iter[b])
        fp.unsafe_store(batch_size + b, r.retcode[b])
    # lane cpu3-seq: AIC then BIC, the device kernel's arithmetic
    # (`arima/impl/arima_ic.mojo::arima_ic_bits`) on the host column's
    # log-likelihood (infeasible series already -inf)
    if ic_address != 0:
        var icp = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=ic_address)
        var pa = bitcast[DType.uint64](pen_aic)
        var pb = bitcast[DType.uint64](pen_bic)
        for b in range(batch_size):
            icp.unsafe_store(b, arima_ic_bits(r.loglike[b], pa))
            icp.unsafe_store(batch_size + b, arima_ic_bits(r.loglike[b], pb))
    return N * batch_size
