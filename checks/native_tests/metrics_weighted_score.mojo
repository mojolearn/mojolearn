# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Weighted accuracy and weighted R^2 against their definitions (ported
from python/mojolearn/tests/test_weighted_score.py
`test_weighted_accuracy_and_r2_match_the_definition`).

The original draw (`np.random.default_rng(5)`, n = 4099: three-class labels
and predictions that agree with probability 0.7, uniform weights in
[0.25, 4) with every eleventh weight zero, a normal target and a noisy
prediction) is NumPy PCG64 output, stored as raw words in
`fixtures/metrics/weighted-*` by `tools/tsa_metrics_native_fixtures.py`.

    WEIGHTED_UNIT 0  accuracy == sum(w [y_true == y_pred]) / sum(w) in
                     Float64, to 2e-6 relative
    WEIGHTED_UNIT 1  R^2 == 1 - sum(w (y - y_hat)^2) / sum(w (y - avg)^2)
                     in Float64 (the residual squared in float32, as NumPy
                     computes it), to 1e-5 relative; a repeat is bitwise
    WEIGHTED_UNIT 2  a constant target predicted exactly scores 1.0
    WEIGHTED_UNIT 3  a constant target predicted off by one scores 0.0

The refusals, the normalize=False counts and the source checks stay in the
Python file. Importable; no `main()`.

`test_weighted_count_and_errors` (lane py2mojo-D, batch 5, 2026-11-22):

    units 0-1  test_weighted_score.py `test_normalize_false_sums_correct_sample_weights`: the
               accuracy_count program the door builds for normalize=False with weights
               (bindings/mcls_classify_door.mojo `_accuracy` / `mc_sums`, then `mc_count_tail` of
               bindings/mcls_classify_run.mojo): y_true [0, 1, 1], weights [0.5, 2.0, 4.0] (dyadic),
               y_pred [0, 1, 0] counts exactly 2.5 and y_pred = y_true exactly 6.5
    units 2-10 test_regression_metrics.py `test_supported_options_reach_expansion`: the `_mean_error`
               program (bindings/metrics_reg_door.mojo, typed `_Reg` of bindings/metrics_reg_run.mojo)
               of mean_squared_error, mean_absolute_error and root_mean_squared_error x
               {sample_weight ones, multioutput raw_values, multioutput None} over y [1, 2],
               p [1.5, 1]: the weighted binary64 definition to rel 1e-6, raw_values one value

Both compositions repeat the door's stage order line for line and run on the column's metrics program
entry. The not-called plain binding and the returned shape stay in the Python files.
"""
from std.memory import bitcast
from checks.native_tests.tsa_metrics_fixture_words import METRICS_FIXTURE_DIR, fixture_f32, fixture_i32
from metrics.host.metrics_oracle import host_weighted_accuracy_ptr
from metrics.host.r2_entry_host import _r2_weighted_value
from metrics.estimator import accuracy_score_weighted_host
from metrics.r2_entry_device import _r2_weighted_value as _r2_weighted_value_gpu
from std.testing import TestSuite, assert_true as require
from checks.native_tests.testsuite_cases import check_cases
from std.sys.compile import is_defined
from std.math import sqrt
from bindings.arena_prog_run import MetricArenaOwner, MetricProgramInput, metrics_plan_execute
from bindings.mcls_classify_run import McSums, mc_count_tail
from bindings.mcls_common_run import MC_NONE
from bindings.metrics_reg_run import _Reg, reg_raw_values
from core.metrics_program_plan import MetricProgramPlan, metric_term
from core.native_descriptor_abi import NativeDescriptorCall
from x_metrics.descriptor_host import metrics_program_host_v1
from x_metrics.descriptor_device import metrics_program_device_v1
from checks.native_tests.testsuite_serial import check_cases_serial


comptime WEIGHTED_N = 4099
comptime WEIGHTED_UNITS = 4


trait WeightedCalls:
    @staticmethod
    def accuracy(y_true: List[Int32], y_pred: List[Int32], w: List[Float32], n: Int) raises -> Float32:
        ...

    @staticmethod
    def r2(y: List[Float32], y_hat: List[Float32], w: List[Float32], n: Int) raises -> Float32:
        ...


def isclose(a: Float64, b: Float64, rel: Float64) -> Bool:
    """`math.isclose(a, b, rel_tol=rel)`."""
    return abs(a - b) <= rel * max(abs(a), abs(b))


def weighted_unit[K: WeightedCalls](unit: Int) raises:
    var w = fixture_f32(METRICS_FIXTURE_DIR + "weighted-w.f32", WEIGHTED_N)
    if unit == 0:
        var yt = fixture_i32(METRICS_FIXTURE_DIR + "weighted-yt.i32", WEIGHTED_N)
        var yp = fixture_i32(METRICS_FIXTURE_DIR + "weighted-yp.i32", WEIGHTED_N)
        var hit = Float64(0)
        var total = Float64(0)
        for i in range(WEIGHTED_N):  # small-loop(Float64 reference fold over one fixture)
            total += Float64(w[i])
            if yt[i] == yp[i]:
                hit += Float64(w[i])
        var want = hit / total
        var got = Float64(K.accuracy(yt, yp, w, WEIGHTED_N))
        require(isclose(got, want, 2e-6), "weighted accuracy " + String(got) + " != " + String(want))
    elif unit == 1:
        var y = fixture_f32(METRICS_FIXTURE_DIR + "weighted-y.f32", WEIGHTED_N)
        var yh = fixture_f32(METRICS_FIXTURE_DIR + "weighted-yh.f32", WEIGHTED_N)
        var swy = Float64(0)
        var sw = Float64(0)
        for i in range(WEIGHTED_N):  # small-loop(Float64 reference fold over one fixture)
            swy += Float64(y[i]) * Float64(w[i])
            sw += Float64(w[i])
        var avg = swy / sw
        var sse = Float64(0)
        var ssto = Float64(0)
        for i in range(WEIGHTED_N):  # small-loop(Float64 reference fold over one fixture)
            var d1 = y[i] - yh[i]
            sse += Float64(w[i]) * Float64(d1 * d1)
            var d2 = Float64(y[i]) - avg
            ssto += Float64(w[i]) * d2 * d2
        var want = 1.0 - sse / ssto
        var first = K.r2(y, yh, w, WEIGHTED_N)
        require(isclose(Float64(first), want, 1e-5), "weighted r2 " + String(first) + " != " + String(want))
        var again = K.r2(y, yh, w, WEIGHTED_N)
        require(bitcast[DType.uint32](again) == bitcast[DType.uint32](first), "weighted r2 repeat moved")
    else:
        var c = List[Float32](length=8, fill=Float32(2.0))
        var ones = List[Float32](length=8, fill=Float32(1.0))
        if unit == 2:
            require(K.r2(c, c, ones, 8) == 1.0, "constant target predicted exactly does not score 1")
        else:
            var off = List[Float32](length=8, fill=Float32(3.0))
            require(K.r2(c, off, ones, 8) == 0.0, "constant target predicted off by one does not score 0")


# ---- The host column (was metrics_weighted_score_cpu_suite.mojo) ----
# Weighted accuracy and R^2 on the host column: the metrics host
# binding's bodies (`host_weighted_accuracy_ptr`, `_r2_weighted_value`).
#
# The four units are independent and run across host cores.
struct HostWeighted(WeightedCalls):
    @staticmethod
    def accuracy(y_true: List[Int32], y_pred: List[Int32], w: List[Float32], n: Int) raises -> Float32:
        return host_weighted_accuracy_ptr(
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(y_true.unsafe_ptr())),
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(y_pred.unsafe_ptr())),
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(w.unsafe_ptr())), n)

    @staticmethod
    def r2(y: List[Float32], y_hat: List[Float32], w: List[Float32], n: Int) raises -> Float32:
        return _r2_weighted_value(Int(y.unsafe_ptr()), Int(y_hat.unsafe_ptr()), Int(w.unsafe_ptr()), n)


# ---- The device column (was metrics_weighted_score_gpu_suite.mojo) ----
# Weighted accuracy and R^2 on the device column: the metrics device
# binding's bodies (`accuracy_score_weighted_host`, `_r2_weighted_value`).
#
# The units share the process's one metrics device context, so they run in
# order; each reduction keeps its own chunked device parallelism.
struct DeviceWeighted(WeightedCalls):
    @staticmethod
    def accuracy(y_true: List[Int32], y_pred: List[Int32], w: List[Float32], n: Int) raises -> Float32:
        return accuracy_score_weighted_host(y_true, y_pred, w, n)

    @staticmethod
    def r2(y: List[Float32], y_hat: List[Float32], w: List[Float32], n: Int) raises -> Float32:
        return _r2_weighted_value_gpu(Int(y.unsafe_ptr()), Int(y_hat.unsafe_ptr()), Int(w.unsafe_ptr()), n)


def _host_unit(unit: Int) raises:
    weighted_unit[HostWeighted](unit)


def _cpu_cases() raises:
    check_cases("metrics_weighted_score", WEIGHTED_UNITS, _host_unit)


def _gpu_cases() raises:
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():  # discovery instantiates every module function
        var failed = String("")
        var count = 0
        for unit in range(WEIGHTED_UNITS):  # driver-loop: over the units on the one device context
            try:
                weighted_unit[DeviceWeighted](unit)
            except error:
                count += 1
                failed += "\n      metrics_weighted_score row " + String(unit) + ": " + String(error)
        if count > 0:
            raise Error("metrics_weighted_score: " + String(count) + " of " + String(WEIGHTED_UNITS) + " cases failed" + failed)


def test_cases() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        _cpu_cases()
    else:
        _gpu_cases()


# ---- test_weighted_count_and_errors ----

comptime OPTION_UNITS = 11
comptime _RE_DIV = 1  # bindings/metrics_reg_run.mojo
comptime _RE_ROOT = 2
comptime _NONE = -1


def run_with_pins(entry: NativeDescriptorCall, plan: MetricProgramPlan, pins: List[Int], sizes: List[Int]) raises -> MetricArenaOwner:
    """The program's records: the constant words, and the caller buffers `pins` (addresses) in input order."""
    var records = List[MetricProgramInput]()
    var pin = 0
    for r in range(len(plan.input_offsets)):  # driver-loop: over the program's input records
        if plan.input_consts[r] >= 0:
            records.append(MetricProgramInput(plan.input_offsets[r], plan.const_address(r),
                                              len(plan.consts[plan.input_consts[r]])))
        else:
            require(pin < len(pins), "program: more caller inputs than the " + String(len(pins)) + " given")
            records.append(MetricProgramInput(plan.input_offsets[r], pins[pin], sizes[pin]))
            pin += 1
    require(pin == len(pins), "program: " + String(pin) + " caller inputs, " + String(len(pins)) + " given")
    var arena = metrics_plan_execute(entry, plan, records, False, False)
    _ = records^
    return arena^


def weighted_count(entry: NativeDescriptorCall, yt: List[Int32], yp: List[Int32], w: List[Float32],
                   labels: Int) raises -> Float64:
    """accuracy_score(normalize=False, sample_weight=w): `mc_sums` (weighted) then `mc_count_tail`."""
    var n = len(yt)
    var plan = MetricProgramPlan()
    var s = McSums()
    s.n = n
    s.labels = labels
    s.weighted = True
    # bindings/mcls_classify_door.mojo `mc_sums`, in its order
    var a = plan.input(n)
    var b = plan.input(n)
    var match_off = plan.scratch(n)
    plan.stage_op("pair_key", n, [a, b, match_off, labels, 1])
    var w_off = plan.input(n)
    s.match_off = match_off
    s.w_off = w_off
    var keys: List[Int] = [match_off, a, b]
    for g in range(3):  # driver-loop: over the three count groupings
        var got = plan.group(keys[g], n, labels, MC_NONE, 1, w_off, 1)
        s.groups.append(got[0])
        s.groups.append(got[1])
    var zero = plan.scratch(n)
    plan.stage_op("pair_key", n, [a, a, zero, 1, 2])
    var tot = plan.group(zero, n, 1, MC_NONE, 1, w_off, 1)
    s.groups.append(tot[0])
    s.groups.append(tot[1])
    var out = mc_count_tail(plan, s, labels)
    var pins: List[Int] = [Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), Int(w.unsafe_ptr())]
    var sizes: List[Int] = [n, n, n]
    var arena = run_with_pins(entry, plan, pins, sizes)
    require(arena.i32(out) == 0, "weighted count: the weight total flag is set")
    var value = arena.f64(out + 2)
    _ = arena^
    return value


def mean_error(entry: NativeDescriptorCall, kind: String, root: Bool, weighted: Bool, mo: String,
               y: List[Float32], p: List[Float32], w: List[Float32]) raises -> Float64:
    """`_mean_error` over (n, 1) targets: one value (raw_values: its one output)."""
    var n = len(y)
    var r = _Reg(n, 1, weighted, mo, List[Float64]())
    r.program(0.0)
    var mode = _RE_DIV | (_RE_ROOT if root else 0)
    var y_off = r.y_off
    var p_off = r.p_off
    var mean = r.stage_mean(metric_term(kind), mode, y_off, p_off, False)
    var out = r.average(mean, False, _NONE)
    var pins: List[Int] = [Int(y.unsafe_ptr()), Int(p.unsafe_ptr())]
    var sizes: List[Int] = [n, n]
    if weighted:
        pins.append(Int(w.unsafe_ptr()))
        sizes.append(n)
    var arena = run_with_pins(entry, r.plan, pins, sizes)
    var value: Float64
    if mo == "raw_values":
        var raw = reg_raw_values(r, out, arena)
        require(len(raw) == 1, "raw_values: " + String(len(raw)) + " outputs, expected 1")
        value = raw[0]
    else:
        value = arena.f64(out)
    _ = arena^
    return value


def option_unit(entry: NativeDescriptorCall, unit: Int) raises:
    if unit < 2:
        var yt: List[Int32] = [0, 1, 1]
        var yp: List[Int32] = [0, 1, 0]
        if unit == 1:
            yp = yt.copy()
        var w: List[Float32] = [0.5, 2.0, 4.0]
        var got = weighted_count(entry, yt, yp, w, 2)
        var want = 2.5 if unit == 0 else 6.5
        require(got == want, "accuracy_score(normalize=False, sample_weight) " + String(got) + " != " + String(want))
        return
    var name = (unit - 2) // 3
    var option = (unit - 2) % 3
    var y: List[Float32] = [1.0, 2.0]
    var p: List[Float32] = [1.5, 1.0]
    var w: List[Float32] = [1.0, 1.0]
    var sw = Float64(0)
    var sq = Float64(0)
    var ab = Float64(0)
    for i in range(2):  # small-loop(the two-row binary64 definition)
        var e = Float64(p[i] - y[i])
        sw += Float64(w[i])
        sq += Float64(w[i]) * e * e
        ab += Float64(w[i]) * abs(e)
    var want = sq / sw
    var kind = String("sq")
    var label = String("mean_squared_error")
    if name == 1:
        want = ab / sw
        kind = "abs"
        label = "mean_absolute_error"
    elif name == 2:
        want = sqrt(sq / sw)
        label = "root_mean_squared_error"
    var mo = String("raw_values") if option == 1 else String("uniform_average")
    var got = mean_error(entry, kind, name == 2, option == 0, mo, y, p, w)
    require(abs(got - want) <= max(1e-6 * abs(want), 1e-12),
            label + " option " + String(option) + ": " + String(got) + " expected " + String(want))


def _host_option_unit(unit: Int) raises:
    option_unit(metrics_program_host_v1, unit)


def test_weighted_count_and_errors() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        check_cases("weighted_count_and_errors", OPTION_UNITS, _host_option_unit)
    else:
        # ordered programs on the one shared device context
        def ordered(unit: Int) raises:
            option_unit(metrics_program_device_v1, unit)
        check_cases_serial("weighted_count_and_errors", OPTION_UNITS, ordered)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
