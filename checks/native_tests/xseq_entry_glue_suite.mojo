# SPDX-License-Identifier: Apache-2.0
"""The sequence table's x_sequence glue entry and the metrics R^2 entry,
reached natively (lane calltable-xseq, 2026-11-11).

Ordered preconditions first (they share the one process table): callers
refuse before publication, then the publications `_mojolearn_x_sequence_host`
and `_mojolearn_metrics_host` make at module init, then a tier-key mismatch,
a schema mismatch and an unknown operation refuse. Then one case per glue
operation and per R^2 operation through the published pointer: literal
closed forms where the body has one (constant Theta, Croston's flat
forecast, Prophet's day conversion, a perfect R^2), word-for-word parity
with the same body called directly otherwise, and for the operations whose
fixtures need a whole fitted model, the body's own refusal text carried
through the entry's sink unchanged. No Python participates.

The cases are independent units and run across the native host worker
pool. Run in a fresh process (the first check needs unpublished tables).
"""
from core.entry_table import entry_ref_call, entry_ref_same
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.native_descriptor_call import native_descriptor_call
from core.xseq_entry_table import (
    XseqEntryKey, xseq_glue_entry, xseq_glue_ref, xseq_table_entry, XSEQ_ENTRY_GLUE, XSEQ_GLUE_ABI, XS_OP_COUNT,
    XS_THETA, XS_THETA_FORECAST, XS_CROSTON, XS_CROSTON_FORECAST, XS_ETS, XS_GARCH, XS_PROPHET_DAYS,
    XS_PROPHET_FEATURES, XS_PROPHET_CHANGEPOINTS, XS_PROPHET_FIT, XS_PROPHET_PREDICT, XS_PROPHET_PREDICT_DAYS,
    XS_VAR_FIT, XS_VAR_FORECAST, XS_STL,
)
from core.metrics_entry_table import MetricsEntryKey, metrics_r2_entry, MR_R2, MR_R2_WEIGHTED, MR_OP_COUNT
from sequence.exec import HostExec
from sequence.xseq_glue import xseq_glue_run
from sequence.xseq_glue_host import publish_x_sequence_host
from metrics.host.r2_entry_host import publish_metrics_r2_host, _r2_value, _r2_weighted_value

comptime XSEQ_GLUE_CASES = 18


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> XseqEntryKey:
    return XseqEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _mkey() -> MetricsEntryKey:
    return MetricsEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _glue(op: Int, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    return entry_ref_call(xseq_glue_ref(_key()), op, addrs, ints, floats)


def _direct(op: Int, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    var ex = HostExec()
    return xseq_glue_run(ex, op, addrs, ints, floats)


def _f32(mut l: List[Float32]) -> Int:
    return Int(l.unsafe_ptr())


def _f64(mut l: List[Float64]) -> Int:
    return Int(l.unsafe_ptr())


def _same_f32(a: List[Float32], b: List[Float32], label: String) raises:
    require(len(a) == len(b), label + ": lengths differ")
    for i in range(len(a)):  # small-loop(a handful of fixture words)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), label + ": word " + String(i) + " differs")


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def _refusal_parity(op: Int, label: String) raises:
    """An empty request: the table route refuses with the body's own text."""
    var none = List[Int]()
    var direct = String("")
    try:
        _ = _direct(op, none, none, List[Float64]())
    except problem:
        direct = String(problem)
    require(label in direct, label + ": the direct body did not refuse as expected: " + direct)
    var routed = String("")
    try:
        _ = _glue(op, none, none, List[Float64]())
    except problem:
        routed = String(problem)
    require(direct in routed, label + ": the table route refused differently: " + routed)


def preconditions_unpublished() raises:
    var none = List[Int]()
    try:
        _ = _glue(XS_THETA, none, none, List[Float64]())
        raise Error("an unpublished sequence table did not refuse")
    except problem:
        _refuses(problem, "provider is not published")
    try:
        _ = native_descriptor_call(metrics_r2_entry(_mkey()), MR_R2, none, none, List[Float64]())
        raise Error("an unpublished metrics R^2 table did not refuse")
    except problem:
        _refuses(problem, "provider is not published")


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: the table resolves once into a ref that a
    connection owner reuses (the same pointer and layout word every time),
    and a layout mismatch refuses at resolution, never at a call."""
    var first = xseq_glue_ref(_key())
    var second = xseq_glue_ref(_key())
    require(entry_ref_same(first, second), "x_sequence glue: a second resolution returned another entry")
    try:
        _ = xseq_table_entry(_key(), XSEQ_ENTRY_GLUE, String(XSEQ_GLUE_ABI), XS_OP_COUNT + 1)
        raise Error("x_sequence glue: a layout mismatch did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
        require("ABI schema differs" in String(problem), "x_sequence glue: unexpected refusal: " + String(problem))


def preconditions_published() raises:
    ref_resolution()
    var none = List[Int]()
    try:
        _ = xseq_glue_entry(XseqEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0))
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = xseq_table_entry(_key(), XSEQ_ENTRY_GLUE, String("mojolearn.sequence.other.v1"), XS_OP_COUNT)
        raise Error("a mismatched schema did not refuse")
    except problem:
        _refuses(problem, "ABI schema differs")
    try:
        _ = _glue(XS_OP_COUNT, none, none, List[Float64]())
        raise Error("an unknown glue operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")
    try:
        _ = native_descriptor_call(metrics_r2_entry(_mkey()), MR_OP_COUNT, none, none, List[Float64]())
        raise Error("an unknown R^2 operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")


def _theta_constant(mut y: List[Float32], mut forecast: List[Float32], mut info: List[Float32], h: Int) raises -> Int:
    # One constant series (20) of 8 points, model 0, fixed alpha 1 / theta 2:
    # its forecast is the level (checks/native_tests/theta_state_cases.mojo).
    var addrs: List[Int] = [_f32(y), _f32(forecast), _f32(info)]
    var ints: List[Int] = [1, 8, h, 1, 0, 0, 7]
    var floats: List[Float64] = [20.0, 1.0, 2.0]
    return _glue(XS_THETA, addrs, ints, floats)


def case(unit: Int) raises:
    if unit == 0:
        var y = List[Float32](length=8, fill=20.0)
        var forecast = List[Float32](length=3, fill=0.0)
        var info = List[Float32](length=8, fill=0.0)
        require(_theta_constant(y, forecast, info, 3) == 3, "theta result count")
        for i in range(3):  # small-loop(fixture words)
            require(forecast[i] == 20.0, "theta constant forecast")
        _ = y
        _ = info
    elif unit == 1:
        var y = List[Float32](length=8, fill=20.0)
        var forecast = List[Float32](length=3, fill=0.0)
        var info = List[Float32](length=8, fill=0.0)
        _ = _theta_constant(y, forecast, info, 3)
        var tail = List[Float32](length=5, fill=0.0)
        var addrs: List[Int] = [_f32(y), _f32(info), _f32(tail)]
        var ints: List[Int] = [1, 8, 5, 1]
        require(_glue(XS_THETA_FORECAST, addrs, ints, List[Float64]()) == 5, "theta_forecast result count")
        for i in range(5):  # small-loop(fixture words)
            require(tail[i] == 20.0, "theta_forecast constant tail")
        _ = y
        _ = info
        _ = forecast
    elif unit == 2:
        var y: List[Float32] = [0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 4.0]
        var routed = List[Float32](length=1, fill=0.0)
        var direct = List[Float32](length=1, fill=0.0)
        var ints: List[Int] = [1, 10, 0]
        require(_glue(XS_CROSTON, [_f32(y), _f32(routed)], ints, List[Float64]()) == 1, "croston result count")
        require(_direct(XS_CROSTON, [_f32(y), _f32(direct)], ints, List[Float64]()) == 1, "croston direct count")
        _same_f32(routed, direct, "croston")
        _ = y
    elif unit == 3:
        var mean: List[Float32] = [2.5, -1.0]
        var forecast = List[Float32](length=6, fill=0.0)
        var ints: List[Int] = [2, 3]
        require(_glue(XS_CROSTON_FORECAST, [_f32(mean), _f32(forecast)], ints, List[Float64]()) == 6,
                "croston_forecast result count")
        for i in range(6):  # small-loop(fixture words)
            require(forecast[i] == (Float32(2.5) if i < 3 else Float32(-1.0)), "croston_forecast flat rows")
        _ = mean
    elif unit == 4:
        var src: List[Float64] = [0.0, 1.0, 3.0]
        var days = List[Float64](length=3, fill=-1.0)
        var info = List[Float64](length=3, fill=-1.0)
        var ints: List[Int] = [3, 0]
        require(_glue(XS_PROPHET_DAYS, [_f64(src), _f64(days), _f64(info)], ints, List[Float64]()) == 0,
                "prophet_days status")
        require(days[0] == 0.0 and days[1] == 1.0 and days[2] == 3.0, "prophet_days copy")
        require(info[0] == 0.0 and info[1] == 3.0 and info[2] == 1.0, "prophet_days info")
        var unsorted: List[Float64] = [2.0, 1.0, 3.0]
        require(_glue(XS_PROPHET_DAYS, [_f64(unsorted), _f64(days), _f64(info)], ints, List[Float64]()) == 2,
                "prophet_days unsorted status")
        _ = src
        _ = unsorted
    elif unit == 5:
        var days: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        var periods: List[Float64] = [7.0]
        var t_routed = List[Float32](length=6, fill=0.0)
        var f_routed = List[Float32](length=6, fill=0.0)
        var t_direct = List[Float32](length=6, fill=0.0)
        var f_direct = List[Float32](length=6, fill=0.0)
        var ints: List[Int] = [6, 1]
        var floats: List[Float64] = [0.0, 5.0]
        _ = _glue(XS_PROPHET_FEATURES, [_f64(days), _f64(periods), _f32(t_routed), _f32(f_routed)], ints, floats)
        _ = _direct(XS_PROPHET_FEATURES, [_f64(days), _f64(periods), _f32(t_direct), _f32(f_direct)], ints, floats)
        _same_f32(t_routed, t_direct, "prophet_features t")
        _same_f32(f_routed, f_direct, "prophet_features frac")
        require(t_routed[5] == 1.0, "prophet_features scaled end")
        _ = days
        _ = periods
    elif unit == 6:
        var days: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        var user: List[Float64] = [0.0]
        var routed = List[Float32](length=3, fill=0.0)
        var direct = List[Float32](length=3, fill=0.0)
        var ints: List[Int] = [10, -1, 8, 3]
        var floats: List[Float64] = [0.0, 9.0]
        var s1 = _glue(XS_PROPHET_CHANGEPOINTS, [_f64(days), _f64(user), _f32(routed)], ints, floats)
        var s2 = _direct(XS_PROPHET_CHANGEPOINTS, [_f64(days), _f64(user), _f32(direct)], ints, floats)
        require(s1 == s2 and s1 >= 1, "prophet_changepoints count")
        _same_f32(routed, direct, "prophet_changepoints")
        _ = days
        _ = user
    elif unit == 7:
        _refusal_parity(XS_ETS, "ets: requires")
    elif unit == 8:
        _refusal_parity(XS_GARCH, "garch: requires")
    elif unit == 9:
        _refusal_parity(XS_PROPHET_FIT, "prophet_fit: requires")
    elif unit == 10:
        _refusal_parity(XS_PROPHET_PREDICT, "prophet_predict: requires")
    elif unit == 11:
        _refusal_parity(XS_PROPHET_PREDICT_DAYS, "prophet_predict_days: requires")
    elif unit == 12:
        _refusal_parity(XS_VAR_FIT, "var_fit: requires")
    elif unit == 13:
        _refusal_parity(XS_VAR_FORECAST, "var_forecast: requires")
    elif unit == 14:
        _refusal_parity(XS_STL, "stl: requires")
    elif unit == 15:
        var y: List[Float32] = [1.0, 2.0, 3.0, 4.0]
        var out = List[Float64](length=1, fill=-9.0)
        var ints: List[Int] = [4]
        var addrs: List[Int] = [_f32(y), _f32(y), _f64(out)]
        _ = native_descriptor_call(metrics_r2_entry(_mkey()), MR_R2, addrs, ints, List[Float64]())
        require(out[0] == Float64(_r2_value(_f32(y), _f32(y), 4)), "r2 parity")
        require(out[0] == 1.0, "r2 of a perfect prediction")
        _ = y
    elif unit == 16:
        var y: List[Float32] = [1.0, 2.0, 3.0, 4.0]
        var yh: List[Float32] = [1.5, 2.0, 2.5, 4.0]
        var w: List[Float32] = [1.0, 2.0, 1.0, 0.5]
        var out = List[Float64](length=1, fill=-9.0)
        var ints: List[Int] = [4]
        var addrs: List[Int] = [_f32(y), _f32(yh), _f32(w), _f64(out)]
        _ = native_descriptor_call(metrics_r2_entry(_mkey()), MR_R2_WEIGHTED, addrs, ints, List[Float64]())
        require(out[0] == Float64(_r2_weighted_value(_f32(y), _f32(yh), _f32(w), 4)), "weighted r2 parity")
        _ = y
        _ = yh
        _ = w
    elif unit == 17:
        var y: List[Float32] = [1.0]
        var out = List[Float64](length=1, fill=0.0)
        var ints: List[Int] = [1]
        try:
            _ = native_descriptor_call(metrics_r2_entry(_mkey()), MR_R2, [_f32(y), _f32(y)], ints, List[Float64]())
            raise Error("a short R^2 request did not refuse")
        except problem:
            _refuses(problem, "invalid request spans")
        _ = y
        _ = out
    else:
        raise Error("unknown case " + String(unit))


def run_checks() raises:
    preconditions_unpublished()
    # The publications _mojolearn_x_sequence_host and _mojolearn_metrics_host make at module init.
    publish_x_sequence_host()
    publish_metrics_r2_host()
    preconditions_published()
    var failures = List[Int](length=XSEQ_GLUE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL xseq glue entry case", unit, String(problem))
    host_parallelize(task, XSEQ_GLUE_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "xseq glue entry cases failed: " + String(failed))
    print("PASS xseq glue entry", XSEQ_GLUE_CASES)
