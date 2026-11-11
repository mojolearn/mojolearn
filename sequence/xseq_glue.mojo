# SPDX-License-Identifier: Apache-2.0
"""The forecaster operations of the x_sequence glue entry (lane
calltable-xseq, 2026-11-11), over any executor. Request layouts:
`core/xseq_entry_table.mojo`.

These are the bodies the former Python entries (`theta`, `croston`, `ets`,
`garch`, `prophet_*` of `_mojolearn_x_sequence[_host]`) ran after their
`py_ints`/`py_f64s` conversion, with the same argument checks in the same
words; the sequence owner reaches them through the native table now. The
device image overrides ETS(A, A|Ad, N), GARCH and the Prophet fit with its
one-block-per-series team bodies (`sequence/xseq_glue_device.mojo`), as its
Python entries did. No kernel, fold order or route changed.
"""
from sequence.exec_trait import Exec
from sequence.ops_impl import (
    theta_impl, theta_forecast_impl, croston_impl, croston_forecast_impl, ets_impl, garch_impl,
    prophet_fit_impl, prophet_predict_impl, prophet_predict_days_impl,
)
from sequence.prophet_prep import prophet_changepoints, prophet_days, prophet_features
from core.xseq_entry_table import (
    XS_THETA, XS_THETA_FORECAST, XS_CROSTON, XS_CROSTON_FORECAST, XS_ETS, XS_GARCH, XS_PROPHET_DAYS,
    XS_PROPHET_FEATURES, XS_PROPHET_CHANGEPOINTS, XS_PROPHET_FIT, XS_PROPHET_PREDICT, XS_PROPHET_PREDICT_DAYS,
)


def xseq_prophet_days_run(addrs: List[Int], ip: List[Int]) raises -> Int:
    """addrs = [src, days (N) float64 out, info (3) float64 out]; ip = [N,
    kind]. Returns the status (0 ok, 1 a NaN, 2 not ascending)."""
    if len(addrs) != 3 or len(ip) != 2:
        raise Error("prophet_days: requires 3 addresses and 2 integer parameters")
    return prophet_days(addrs, ip[0], ip[1])


def xseq_prophet_features_run[E: Exec](mut ex: E, addrs: List[Int], ip: List[Int], fp: List[Float64]) raises -> Int:
    """addrs = [days (N) float64, periods (ns) float64, t (N) float32 out,
    frac (N, max(ns, 1)) float32 out]; ip = [N, ns]; fp = [start, t_scale].
    Returns 0."""
    if len(addrs) != 4 or len(ip) != 2 or len(fp) != 2:
        raise Error("prophet_features: requires 4 addresses, 2 integer and 2 float parameters")
    return prophet_features(ex, addrs, ip[0], ip[1], fp[0], fp[1])


def xseq_prophet_changepoints_run(addrs: List[Int], ip: List[Int], fp: List[Float64]) raises -> Int:
    """addrs = [days (N) float64, user (n_user) float64 days, out (max(S, 1))
    float32]; ip = [N, n_user (-1: none given), hist, n_cp]; fp = [start,
    t_scale]. Returns S (>= 1), or -1 when a caller's changepoint falls
    outside the data (or is NaN)."""
    if len(addrs) != 3 or len(ip) != 4 or len(fp) != 2:
        raise Error("prophet_changepoints: requires 3 addresses, 4 integer and 2 float parameters")
    return prophet_changepoints(addrs, ip[0], ip[1], ip[2], ip[3], fp[0], fp[1])


def xseq_glue_run[E: Exec](mut ex: E, op: Int, addrs: List[Int], ip: List[Int], fp: List[Float64]) raises -> Int:
    """One forecaster operation on `ex` (the host image's every operation;
    the device image's every operation but its team overrides)."""
    if op == XS_THETA:
        return theta_impl(ex, addrs, ip, fp)
    if op == XS_THETA_FORECAST:
        return theta_forecast_impl(ex, addrs, ip)
    if op == XS_CROSTON:
        return croston_impl(ex, addrs, ip)
    if op == XS_CROSTON_FORECAST:
        return croston_forecast_impl(ex, addrs, ip)
    if op == XS_ETS:
        return ets_impl(ex, addrs, ip, fp)
    if op == XS_GARCH:
        return garch_impl(ex, addrs, ip)
    if op == XS_PROPHET_DAYS:
        return xseq_prophet_days_run(addrs, ip)
    if op == XS_PROPHET_FEATURES:
        return xseq_prophet_features_run(ex, addrs, ip, fp)
    if op == XS_PROPHET_CHANGEPOINTS:
        return xseq_prophet_changepoints_run(addrs, ip, fp)
    if op == XS_PROPHET_FIT:
        return prophet_fit_impl(ex, addrs, ip, fp)
    if op == XS_PROPHET_PREDICT:
        return prophet_predict_impl(ex, addrs, ip)
    if op == XS_PROPHET_PREDICT_DAYS:
        return prophet_predict_days_impl(ex, addrs, ip, fp)
    raise Error("x_sequence glue: unknown operation " + String(op))
