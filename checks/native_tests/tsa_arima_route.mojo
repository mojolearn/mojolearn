# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The ARIMA operations table and the device route, on the host.

The route (core/device_route.mojo `DeviceRoute`, resolved by
core/arima_run.mojo) is what an ARIMA or AutoARIMA estimator resolves once
for its `device`: it is refused before its table is published and for a
vendor no image published; it reports the CPU words of the host entry; the
fit through the route writes the bytes of the direct host fit; the forecast
and the in-sample prediction through the ARIMA image's route and through the
forecast inference image's route write the same bytes; the CPU entries refuse
the resident, criterion and grouped-search operations by name; the forecast
inference image's entry refuses the fit by name; an unresolved route's entry
refuses instead of calling anything.

The resident owner's release through the ARIMA table is not covered here: no
CPU image carries the resident operations (they are the GPU image's), and
the owner is a Python-registered type. It is covered at the Python boundary
(python/mojolearn/tests/test_tsa_resident_state.py) on a GPU installation.

Run in a fresh standalone process: the missing-table assertion and the
publications are ordered because they share the production tables. After
that the independent fixtures (two orders) run across the native host worker
pool. No Python module participates.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from arima.arima_ops_forecast_host import publish_arima_ops_forecast_host
from arima.arima_ops_host import publish_arima_ops_host
from arima.host.arima_host_entries import arima_host_fit_into
from arima.host.mojolearn_arima_host_run import ARIMA_METHOD_MLE
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.arima_entry_table import (
    ARIMA_OP_FIT, ARIMA_OP_FLAGS, ARIMA_OP_FORECAST, ARIMA_OPERATION_COUNT, ARIMA_TABLE_ARIMA,
    ARIMA_TABLE_FORECAST, arima_ops_check_spans,
)
from core.arima_run import (
    arima_fast_search_mode_run, arima_fit_run, arima_forecast_run, arima_ic_run, arima_ndiffs_run,
    arima_predict_run, arima_resident_count_run, arima_resident_prepare_run, arima_route_fits,
    arima_route_has_ic, arima_route_resident, arima_route_resolve, arima_route_searches,
    arima_search_switches_run,
)
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.hostptr import f32_ptr, i32_ptr

comptime TSA_ARIMA_ROUTE_CASES = 2
comptime _N = 64
comptime _B = 2
comptime _H = 5
comptime _ITER = 25


def span_rules() raises:
    var word = List[Int](length=1, fill=0)
    var result = Int(word.unsafe_ptr())
    var eight: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8]
    var thirteen: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    var two: List[Float64] = [0.5, 0.25]
    arima_ops_check_spans(ARIMA_OP_FLAGS, 0, 0, 0, 0, 0, 0, result)
    arima_ops_check_spans(ARIMA_OP_FIT, Int(eight.unsafe_ptr()), 8, Int(thirteen.unsafe_ptr()), 13,
        Int(two.unsafe_ptr()), 2, result)
    arima_ops_check_spans(ARIMA_OP_FORECAST, Int(eight.unsafe_ptr()), 5, Int(thirteen.unsafe_ptr()), 12, 0, 0, result)
    var refused = 0
    try:
        arima_ops_check_spans(ARIMA_OP_FIT, Int(eight.unsafe_ptr()), 7, Int(thirteen.unsafe_ptr()), 13,
            Int(two.unsafe_ptr()), 2, result)
    except:
        refused += 1
    try:
        arima_ops_check_spans(ARIMA_OP_FORECAST, Int(eight.unsafe_ptr()), 5, Int(thirteen.unsafe_ptr()), 13, 0, 0, result)
    except:
        refused += 1
    try:
        arima_ops_check_spans(ARIMA_OPERATION_COUNT, 0, 0, 0, 0, 0, 0, result)
    except:
        refused += 1
    try:
        arima_ops_check_spans(ARIMA_OP_FLAGS, 0, 0, 0, 0, 0, 0, 0)
    except:
        refused += 1
    _ = eight
    _ = thirteen
    _ = two
    _ = word
    require(refused == 4, "arima ops span rule accepted a malformed request")
    print("PASS arima ops span rules")


def missing_before_publication() raises:
    var refused = 0
    try:
        _ = arima_route_resolve(DEVICE_CPU, ARIMA_TABLE_ARIMA, Int(GLOBAL_NUMERIC_MODE), 0)
    except:
        refused += 1
    try:
        _ = arima_route_resolve(DEVICE_CPU, ARIMA_TABLE_FORECAST, Int(GLOBAL_NUMERIC_MODE), 0)
    except:
        refused += 1
    require(refused == 2, "an ARIMA route resolved before its table was published")
    # An unresolved route's entry refuses and writes nothing.
    var idle = DeviceRoute(DEVICE_CPU)
    var sink = List[Float32](length=8, fill=-37)
    var order: List[Int] = [1, 0, 1, 0, 0, 0, 0]
    var called = False
    try:
        _ = arima_forecast_run(idle, Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()),
            Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()), 1, 4, 1, order, 1, 0)
        called = True
    except:
        pass
    require(not called and sink[0] == -37 and not arima_route_fits(idle), "an unresolved ARIMA route called an entry")
    _ = sink
    _ = order
    print("PASS arima route refused before publication")


def series() -> List[Float32]:
    # Two short stationary-looking series: a decaying oscillation around a
    # level plus a small fixed pattern, different per series.
    var data = List[Float32](length=_B * _N, fill=0)
    for s in range(_B):  # small-loop(fixture construction, two short series)
        var previous = Float32(0.5) + Float32(s)
        for t in range(_N):
            var shock = Float32(((t * 7 + s * 3) % 11) - 5) * Float32(0.1)
            var value = Float32(0.6) * previous + shock + Float32(1 + s)
            data[s * _N + t] = value
            previous = value
    return data^


def same_bytes(left: List[Float32], right: List[Float32]) -> Bool:
    if len(left) != len(right):
        return False
    var lp = left.unsafe_ptr().bitcast[UInt32]()
    var rp = right.unsafe_ptr().bitcast[UInt32]()
    var same = True
    for i in range(len(left)):  # small-loop(fixture comparison words)
        if lp[i] != rp[i]:
            same = False
    _ = left
    _ = right
    return same


def same_words(left: List[Float64], right: List[Float64]) -> Bool:
    if len(left) != len(right):
        return False
    var lp = left.unsafe_ptr().bitcast[UInt64]()
    var rp = right.unsafe_ptr().bitcast[UInt64]()
    var same = True
    for i in range(len(left)):  # small-loop(fixture comparison words)
        if lp[i] != rp[i]:
            same = False
    _ = left
    _ = right
    return same


def fixture(differenced: Bool) raises:
    var route = arima_route_resolve(DEVICE_CPU, ARIMA_TABLE_ARIMA, Int(GLOBAL_NUMERIC_MODE), 0)
    var inference = arima_route_resolve(DEVICE_CPU, ARIMA_TABLE_FORECAST, Int(GLOBAL_NUMERIC_MODE), 0)
    # [p, d, q, P, D, Q, s] and the intercept: ARIMA(1,0,1) with a constant,
    # ARIMA(0,1,1) without.
    var order: List[Int] = [1, 0, 1, 0, 0, 0, 0]
    var k = 1
    if differenced:
        order[0] = 0
        order[1] = 1
        k = 0
    var big_n = order[0] + order[2] + order[3] + order[5] + k + 1
    var total = _B * big_n
    var data = series()
    var no_exog = List[Float32](length=1, fill=0)
    var params = List[Float32](length=total, fill=0)
    var x = List[Float32](length=total, fill=0)
    var x0 = List[Float32](length=total, fill=0)
    var stats = List[Float32](length=2 * _B, fill=0)
    var flags = List[Int32](length=2 * _B, fill=0)
    var ics = List[Float64](length=2 * _B, fill=0)
    var direct_params = List[Float32](length=total, fill=0)
    var direct_x = List[Float32](length=total, fill=0)
    var direct_x0 = List[Float32](length=total, fill=0)
    var direct_stats = List[Float32](length=2 * _B, fill=0)
    var direct_flags = List[Int32](length=2 * _B, fill=0)
    var direct_ics = List[Float64](length=2 * _B, fill=0)
    var pen_aic = Float64(2 * big_n)
    var pen_bic = Float64(big_n) * 4.0
    var written = arima_fit_run(route, Int(data.unsafe_ptr()), Int(no_exog.unsafe_ptr()), Int(params.unsafe_ptr()),
        Int(x.unsafe_ptr()), Int(x0.unsafe_ptr()), Int(stats.unsafe_ptr()), Int(flags.unsafe_ptr()),
        Int(ics.unsafe_ptr()), _B, _N, order, k, 0, ARIMA_METHOD_MLE, _ITER, pen_aic, pen_bic)
    var direct = arima_host_fit_into(Int(data.unsafe_ptr()), Int(no_exog.unsafe_ptr()),
        f32_ptr(Int(direct_params.unsafe_ptr())), f32_ptr(Int(direct_x.unsafe_ptr())),
        f32_ptr(Int(direct_x0.unsafe_ptr())), f32_ptr(Int(direct_stats.unsafe_ptr())),
        i32_ptr(Int(direct_flags.unsafe_ptr())), _B, _N, order[0], order[1], order[2], order[3], order[4],
        order[5], order[6], k, 0, _ITER, Int(direct_ics.unsafe_ptr()), pen_aic, pen_bic)
    require(written == total and direct == total, "ARIMA fit through the route returned a different parameter count")
    require(same_bytes(params, direct_params) and same_bytes(x, direct_x) and same_bytes(x0, direct_x0)
        and same_bytes(stats, direct_stats), "ARIMA fit through the route wrote different bytes")
    require(same_words(ics, direct_ics), "ARIMA fit through the route wrote different criteria")
    var fc = List[Float32](length=_H * _B, fill=0)
    var fc_inference = List[Float32](length=_H * _B, fill=0)
    _ = arima_forecast_run(route, Int(data.unsafe_ptr()), Int(no_exog.unsafe_ptr()), Int(no_exog.unsafe_ptr()),
        Int(params.unsafe_ptr()), Int(fc.unsafe_ptr()), _B, _N, _H, order, k, 0)
    _ = arima_forecast_run(inference, Int(data.unsafe_ptr()), Int(no_exog.unsafe_ptr()), Int(no_exog.unsafe_ptr()),
        Int(params.unsafe_ptr()), Int(fc_inference.unsafe_ptr()), _B, _N, _H, order, k, 0)
    require(same_bytes(fc, fc_inference), "ARIMA forecast through the two CPU routes wrote different bytes")
    var start = _N - 8
    var steps = 8 + _H
    var inside = List[Float32](length=steps * _B, fill=0)
    var inside_inference = List[Float32](length=steps * _B, fill=0)
    _ = arima_predict_run(route, Int(data.unsafe_ptr()), Int(no_exog.unsafe_ptr()), Int(no_exog.unsafe_ptr()),
        Int(params.unsafe_ptr()), Int(inside.unsafe_ptr()), _B, _N, start, start + steps, order, k, 0)
    _ = arima_predict_run(inference, Int(data.unsafe_ptr()), Int(no_exog.unsafe_ptr()), Int(no_exog.unsafe_ptr()),
        Int(params.unsafe_ptr()), Int(inside_inference.unsafe_ptr()), _B, _N, start, start + steps, order, k, 0)
    require(same_bytes(inside, inside_inference), "ARIMA prediction through the two CPU routes wrote different bytes")
    _ = data
    _ = no_exog
    _ = params
    _ = x
    _ = x0
    _ = stats
    _ = flags
    _ = ics
    _ = direct_params
    _ = direct_x
    _ = direct_x0
    _ = direct_stats
    _ = direct_flags
    _ = direct_ics
    _ = fc
    _ = fc_inference
    _ = inside
    _ = inside_inference
    _ = order


def published_words() raises:
    var route = arima_route_resolve(DEVICE_CPU, ARIMA_TABLE_ARIMA, Int(GLOBAL_NUMERIC_MODE), 0)
    require(route.vendor == 0 and route.mode == Int(GLOBAL_NUMERIC_MODE) and arima_route_fits(route)
        and not arima_route_resident(route) and not arima_route_searches(route) and not arima_route_has_ic(route),
        "the CPU ARIMA route reports different words")
    var inference = arima_route_resolve(DEVICE_CPU, ARIMA_TABLE_FORECAST, Int(GLOBAL_NUMERIC_MODE), 0)
    require(inference.vendor == 0 and not arima_route_fits(inference) and not arima_route_resident(inference)
        and not arima_route_searches(inference), "the forecast inference ARIMA route reports different words")
    var refused = 0
    try:
        _ = arima_route_resolve(DEVICE_GPU, ARIMA_TABLE_ARIMA, Int(GLOBAL_NUMERIC_MODE), 2)
    except:
        refused += 1
    var sink = List[Float32](length=16, fill=-37)
    var words = List[Int](length=11, fill=1)
    var wide = List[Float64](length=4, fill=-37)
    var order: List[Int] = [1, 0, 1, 0, 0, 0, 0]
    try:
        _ = arima_resident_prepare_run(route, Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()), words)
    except:
        refused += 1
    try:
        _ = arima_ic_run(route, Int(sink.unsafe_ptr()), Int(wide.unsafe_ptr()), 1, 2.0, 4.0)
    except:
        refused += 1
    try:
        _ = arima_ndiffs_run(route, Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()), 1, 8, 2)
    except:
        refused += 1
    var flags = List[Int32](length=4, fill=-37)
    try:
        _ = arima_fit_run(inference, Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()),
            Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()), Int(flags.unsafe_ptr()),
            Int(wide.unsafe_ptr()), 1, 8, order, 1, 0, ARIMA_METHOD_MLE, 1, 2.0, 4.0)
    except:
        refused += 1
    require(refused == 5 and sink[0] == -37 and wide[0] == -37 and flags[0] == -37,
        "a foreign vendor, a CPU resident, criterion or search operation, or the inference fit was not refused")
    require(arima_resident_count_run(route) == 0 and arima_resident_count_run(inference) == 0,
        "a CPU ARIMA binding reports resident handles")
    require(arima_fast_search_mode_run(route) == 0 and arima_search_switches_run(route) == 0,
        "a CPU ARIMA binding reports a fast search arm or a grouped search")
    _ = sink
    _ = words
    _ = wide
    _ = flags
    _ = order
    print("PASS arima published route words and refusals")


def _route_fixture(unit: Int) raises:
    fixture(unit == 1)


def test_unpublished_then_published_route() raises:
    span_rules()
    missing_before_publication()
    # The same native publications as _mojolearn_arima_host and
    # _mojolearn_forecast_host initialization, with no Python import.
    publish_arima_ops_host()
    publish_arima_ops_forecast_host()
    # A repeated initialization publishes the same endpoints and is accepted.
    publish_arima_ops_host()
    published_words()
    check_cases("tsa_arima_route", TSA_ARIMA_ROUTE_CASES, _route_fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
