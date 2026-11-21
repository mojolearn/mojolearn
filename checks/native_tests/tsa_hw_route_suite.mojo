# SPDX-License-Identifier: Apache-2.0
"""The Holt-Winters operations table and the device route, on the host.

The route (core/device_route.mojo `DeviceRoute`, resolved by core/tsa_hw_run.mojo) is what an ExponentialSmoothing estimator
resolves once for its `device`: it is refused before its table is published
and for a vendor no image published; it reports the CPU words of the host
entry; the fit, the forecast and the in-sample prediction through the route
write the bytes of the direct host calls; the CPU entries refuse the resident
operations by name; the forecast inference image's entry refuses the fit by
name; an unresolved route's entry refuses instead of calling anything.

Run in a fresh standalone process: the missing-table assertion and the
publications are ordered because they share the production tables. After
that the independent fixtures (additive, multiplicative) run across the
native host worker pool. No Python module participates.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, i32_ptr
from core.device_route import (
    DeviceRoute, DEVICE_CPU, DEVICE_GPU, DEVICE_INSTALL, device_code, device_installed, device_vendor_word,
)
from core.mojolearn_tsa_host_impl import holtwinters_fit_host_into
from core.tsa_entry_table import (
    TSA_OP_FLAGS, TSA_OP_HW_FIT, TSA_OP_HW_FORECAST, TSA_OP_HW_QUERY_RESIDENT, TSA_OP_HW_RESIDENT_PREPARE,
    TSA_OPERATION_COUNT, TSA_TABLE_FORECAST, TSA_TABLE_TSA, tsa_ops_check_spans,
)
from core.tsa_hw_run import (
    tsa_hw_fit_run, tsa_hw_forecast_run, tsa_hw_predict_run, tsa_hw_resident_count_run,
    tsa_hw_resident_prepare_run, tsa_hw_query_resident_run, tsa_hw_route_fits, tsa_hw_route_resident,
    tsa_hw_route_resolve,
)
from bindings.holtwinters_host_predict_run import holtwinters_forecast_run, holtwinters_predict_run
from holtwinters.impl.tsa.holtwinters_params import seasonal_name
from tsa.tsa_ops_forecast_host import publish_tsa_hw_ops_forecast_host
from tsa.tsa_ops_host import publish_tsa_hw_ops_host

comptime TSA_HW_ROUTE_CASES = 2
comptime _N = 24
comptime _B = 2
comptime _FREQ = 4
comptime _START = 2
comptime _H = 5


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def span_rules() raises:
    var word = List[Int](length=1, fill=0)
    var result = Int(word.unsafe_ptr())
    var four: List[Int] = [1, 2, 3, 4]
    var six: List[Int] = [1, 2, 3, 4, 5, 6]
    var one: List[Float64] = [0.5]
    tsa_ops_check_spans(TSA_OP_FLAGS, 0, 0, 0, 0, 0, 0, result)
    tsa_ops_check_spans(TSA_OP_HW_FIT, Int(four.unsafe_ptr()), 4, Int(six.unsafe_ptr()), 6,
        Int(one.unsafe_ptr()), 1, result)
    var refused = 0
    try:
        tsa_ops_check_spans(TSA_OP_HW_FIT, Int(four.unsafe_ptr()), 3, Int(six.unsafe_ptr()), 6,
            Int(one.unsafe_ptr()), 1, result)
    except:
        refused += 1
    try:
        tsa_ops_check_spans(TSA_OP_HW_FORECAST, Int(four.unsafe_ptr()), 2, Int(six.unsafe_ptr()), 6, 0, 0, result)
    except:
        refused += 1
    try:
        tsa_ops_check_spans(TSA_OPERATION_COUNT, 0, 0, 0, 0, 0, 0, result)
    except:
        refused += 1
    try:
        tsa_ops_check_spans(TSA_OP_FLAGS, 0, 0, 0, 0, 0, 0, 0)
    except:
        refused += 1
    _ = four
    _ = six
    _ = one
    _ = word
    require(refused == 4, "tsa ops span rule accepted a malformed request")
    print("PASS tsa ops span rules")


def device_rules() raises:
    require(device_code(String("gpu")) == DEVICE_GPU and device_code(String("cpu")) == DEVICE_CPU
        and device_code(String("GPU")) == -1, "device words differ")
    require(device_installed(DEVICE_INSTALL, 0) == DEVICE_CPU
        and device_installed(DEVICE_INSTALL, 2) == DEVICE_GPU
        and device_installed(DEVICE_CPU, 2) == DEVICE_CPU, "installed device rule differs")
    var refused = False
    try:
        _ = device_installed(DEVICE_GPU, 0)
    except:
        refused = True
    require(refused, "an explicit gpu on a CPU-only installation was not refused")
    require(device_vendor_word(0) == "cpu" and device_vendor_word(2) == "cuda", "vendor words differ")
    print("PASS tsa device rules")


def missing_before_publication() raises:
    var refused = 0
    try:
        _ = tsa_hw_route_resolve(DEVICE_CPU, TSA_TABLE_TSA, Int(GLOBAL_NUMERIC_MODE), 0)
    except:
        refused += 1
    try:
        _ = tsa_hw_route_resolve(DEVICE_CPU, TSA_TABLE_FORECAST, 1, 0)
    except:
        refused += 1
    require(refused == 2, "a route resolved before its table was published")
    # An unresolved route's entry refuses and writes nothing.
    var idle = DeviceRoute(DEVICE_CPU)
    var sink = List[Float32](length=4, fill=-37)
    var called = False
    try:
        _ = tsa_hw_forecast_run(idle, Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()), _N, 1, _FREQ, 1, 0)
        called = True
    except:
        pass
    require(not called and sink[0] == -37 and not tsa_hw_route_fits(idle), "an unresolved route called an entry")
    _ = sink
    print("PASS tsa route refused before publication")


def series(multiplicative: Bool) -> List[Float32]:
    var data = List[Float32](length=_B * _N, fill=0)
    for s in range(_B):  # small-loop(fixture construction, two short series)
        for t in range(_N):
            var season = Float32((t % _FREQ) + 1)
            var level = Float32(10 + s) + Float32(t) * Float32(0.25)
            if multiplicative:
                data[s * _N + t] = level * (Float32(1) + season * Float32(0.05))
            else:
                data[s * _N + t] = level + season
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


def fixture(multiplicative: Bool) raises:
    var route = tsa_hw_route_resolve(DEVICE_CPU, TSA_TABLE_TSA, Int(GLOBAL_NUMERIC_MODE), 0)
    var inference = tsa_hw_route_resolve(DEVICE_CPU, TSA_TABLE_FORECAST, 1, 0)
    var seasonal = 1 if multiplicative else 0
    var cl = (_N - _FREQ) * _B
    var data = series(multiplicative)
    var comps = List[Float32](length=3 * cl, fill=0)
    var stats = List[Float32](length=4 * _B, fill=0)
    var flags = List[Int32](length=2 * _B, fill=0)
    var direct_comps = List[Float32](length=3 * cl, fill=0)
    var direct_stats = List[Float32](length=4 * _B, fill=0)
    var direct_flags = List[Int32](length=2 * _B, fill=0)
    var written = tsa_hw_fit_run(route, Int(data.unsafe_ptr()), Int(comps.unsafe_ptr()), Int(stats.unsafe_ptr()),
        Int(flags.unsafe_ptr()), _N, _B, _FREQ, _START, 1, seasonal, Float64(Float32(2.24e-3)))
    var direct = holtwinters_fit_host_into(Int(data.unsafe_ptr()), _N, _B, _FREQ, _START, Float32(2.24e-3), 1,
        seasonal_name(seasonal), f32_ptr(Int(direct_comps.unsafe_ptr())), f32_ptr(Int(direct_stats.unsafe_ptr())),
        i32_ptr(Int(direct_flags.unsafe_ptr())))
    require(written == cl and direct == cl, "fit through the route returned a different components length")
    require(same_bytes(comps, direct_comps) and same_bytes(stats, direct_stats),
        "fit through the route wrote different bytes")
    var fc = List[Float32](length=_H * _B, fill=0)
    var fc_inference = List[Float32](length=_H * _B, fill=0)
    var fc_direct = List[Float32](length=_H * _B, fill=0)
    _ = tsa_hw_forecast_run(route, Int(comps.unsafe_ptr()), Int(fc.unsafe_ptr()), _N, _B, _FREQ, _H, seasonal)
    _ = tsa_hw_forecast_run(inference, Int(comps.unsafe_ptr()), Int(fc_inference.unsafe_ptr()), _N, _B, _FREQ, _H, seasonal)
    _ = holtwinters_forecast_run(f32_ptr(Int(comps.unsafe_ptr())), f32_ptr(Int(fc_direct.unsafe_ptr())), _N, _B, _FREQ, _H,
        seasonal_name(seasonal))
    require(same_bytes(fc, fc_direct) and same_bytes(fc_inference, fc_direct),
        "forecast through the route wrote different bytes")
    var steps = _N - _FREQ - 2
    var inside = List[Float32](length=steps * _B, fill=0)
    var inside_direct = List[Float32](length=steps * _B, fill=0)
    _ = tsa_hw_predict_run(route, Int(comps.unsafe_ptr()), Int(inside.unsafe_ptr()), _N, _B, _FREQ, _FREQ, _FREQ + steps, seasonal)
    _ = holtwinters_predict_run(f32_ptr(Int(comps.unsafe_ptr())), f32_ptr(Int(inside_direct.unsafe_ptr())), _N, _B, _FREQ,
        _FREQ, _FREQ + steps, seasonal_name(seasonal))
    require(same_bytes(inside, inside_direct), "in-sample prediction through the route wrote different bytes")
    _ = data
    _ = comps
    _ = stats
    _ = flags
    _ = direct_comps
    _ = direct_stats
    _ = direct_flags
    _ = fc
    _ = fc_inference
    _ = fc_direct
    _ = inside
    _ = inside_direct


def published_words() raises:
    var route = tsa_hw_route_resolve(DEVICE_CPU, TSA_TABLE_TSA, Int(GLOBAL_NUMERIC_MODE), 0)
    require(route.vendor == 0 and route.mode == Int(GLOBAL_NUMERIC_MODE) and tsa_hw_route_fits(route)
        and not tsa_hw_route_resident(route), "the CPU route reports different words")
    var inference = tsa_hw_route_resolve(DEVICE_CPU, TSA_TABLE_FORECAST, 1, 0)
    require(inference.vendor == 0 and not tsa_hw_route_fits(inference) and not tsa_hw_route_resident(inference),
        "the forecast inference route reports different words")
    var refused = 0
    try:
        _ = tsa_hw_route_resolve(DEVICE_GPU, TSA_TABLE_TSA, Int(GLOBAL_NUMERIC_MODE), 2)
    except:
        refused += 1
    var sink = List[Float32](length=8, fill=-37)
    try:
        _ = tsa_hw_resident_prepare_run(route, Int(sink.unsafe_ptr()), _N, 1, _FREQ)
    except:
        refused += 1
    try:
        _ = tsa_hw_query_resident_run(route, 1, Int(sink.unsafe_ptr()), 0, 1, 0)
    except:
        refused += 1
    var stats = List[Float32](length=8, fill=-37)
    var flags = List[Int32](length=4, fill=-37)
    try:
        _ = tsa_hw_fit_run(inference, Int(sink.unsafe_ptr()), Int(sink.unsafe_ptr()), Int(stats.unsafe_ptr()),
            Int(flags.unsafe_ptr()), _N, 1, _FREQ, _START, 1, 0, 2.24e-3)
    except:
        refused += 1
    require(refused == 4 and sink[0] == -37 and stats[0] == -37,
        "a foreign vendor, a CPU resident operation or the inference fit was not refused")
    require(tsa_hw_resident_count_run(route) == 0 and tsa_hw_resident_count_run(inference) == 0,
        "a CPU binding reports resident handles")
    _ = sink
    _ = stats
    _ = flags
    print("PASS tsa published route words and refusals")


def run_checks() raises:
    span_rules()
    device_rules()
    missing_before_publication()
    # The same native publications as _mojolearn_tsa_host and
    # _mojolearn_forecast_host initialization, with no Python import.
    publish_tsa_hw_ops_host()
    publish_tsa_hw_ops_forecast_host()
    # A repeated initialization publishes the same endpoints and is accepted.
    publish_tsa_hw_ops_host()
    published_words()
    var failures = List[Int](length=TSA_HW_ROUTE_CASES, fill=0)
    var marks = failures.unsafe_ptr()
    def task(unit: Int) {imm marks}:
        try:
            fixture(unit == 1)
        except problem:
            marks[unit] = 1
            print("FAIL tsa route fixture", unit, String(problem))
    host_parallelize(task, TSA_HW_ROUTE_CASES)
    var failed = 0
    for failure in failures:  # collect independent fixture status words
        failed += failure
    _ = failures
    require(failed == 0, "Holt-Winters fit/forecast/predict through the route failed")
    print("PASS tsa route fit/forecast/predict", TSA_HW_ROUTE_CASES)
