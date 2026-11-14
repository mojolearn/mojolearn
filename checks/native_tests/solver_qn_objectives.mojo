# SPDX-License-Identifier: Apache-2.0
"""LinearSVC and LinearSVR fit their documented objectives (test_qn_objectives_surface.py).

The original random workload (NumPy seed 0, 400x5 and its 64-row draw,
recorded under fixtures/solver_qn_objectives/) runs through the typed QN
transport the public doors call (core/linear_svc_native_api.mojo) with the
public defaults the doors pass: max_iter 1000, linesearch 100, memory 5,
intercept on, penalty normalized, gradient tolerance 1e-4 and change
tolerance 1e-5, strengths 1/C on the l2 (or l1) slot. The reported objective
is recomputed in Float64 from the packed coefficients, as the Python test did:

  0-1  LinearSVC hinge / squared hinge, C=2: objective to 1e-4 relative,
       labels are the sign of the decision, accuracy > 0.9
  2-5  LinearSVR {epsilon_insensitive, squared} x {l1, l2}, C=4, epsilon 0.05:
       objective to 1e-4 relative, R^2 > 0.9
  6    LinearSVR epsilon 0 and 0.5 (l2, defaults) give different coefficients
  7-8  unit sample weights reproduce the unweighted LinearSVC / LinearSVR
       coefficients (rtol 1e-4, atol 1e-5) on the 64-row draw
  9    LinearSVC one-vs-rest (max_iter 200) on three classes: machine 2 is the
       binary fit of class 2 against the rest, word for word; the decision is
       X coef + b to 1e-4; the fused labels are the decision argmax

The QNRegressor unit-weight case and the scikit-learn/Ridge comparisons stay
in Python with the public label, shape and refusal checks.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.linear_svc_native_api import (
    LinearSvcDriver,
    LinearSvcOptions,
    LinearSvcFitRequest,
    LinearSvcFitPlan,
    linear_svc_pack,
)
from core.logistic_native_api import (
    QnFitRequest,
    QnQueryRequest,
    QN_NATIVE_DECISION,
    QN_NATIVE_BINARY,
    QN_NATIVE_MULTICLASS,
    QN_NATIVE_MULTICLASS_AVAILABLE,
)
from core.qn_single_fit import QnSingleFitPlan
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from glm.host.linear_svc_native_host import linear_svc_host_v1
from glm.linear_svc_native_device import linear_svc_device_v1
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime QN_OBJECTIVE_FIXTURES = 10
comptime QN_ROWS = 400
comptime QN_SMALL_ROWS = 64
comptime QN_FEATURES = 5
comptime LOSS_HINGE = 3
comptime LOSS_SQUARED_HINGE = 4
comptime LOSS_EPSILON = 5
comptime LOSS_SQUARED_EPSILON = 6


def recorded_f32(name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/solver_qn_objectives/" + name, "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == count * 4, "QN objectives fixture extent changed")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def word(index: Int) {imm bp, imm rp}:
            var at = index * 4
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(word, count)
        _ = bytes^
    return result^


def recorded_codes(name: String, count: Int) raises -> List[Int32]:
    var words = recorded_f32(name, count)
    var result = List[Int32](unsafe_uninit_length=count)
    var wp = words.unsafe_ptr()
    var rp = result.unsafe_ptr()
    def code(index: Int) {imm wp, imm rp}:
        rp.unsafe_store(index, bitcast[DType.int32](wp[index]))
    host_parallelize(code, count)
    _ = words^
    return result^


def total(values: List[Float64]) -> Float64:
    var sum = Float64(0)
    for value in values:  # one ordered reduction of independent row terms
        sum += value
    return sum


@fieldwise_init
struct SvcFit(Movable):
    var w: List[Float32]
    var coefficients: List[Float32]
    var biases: List[Float32]
    var objective: Float64


def fit_svc(driver: LinearSvcDriver, x: List[Float32], codes: List[Int32], weights: List[Float32],
            n: Int, first: Int, fits: Int, max_iter: Int, loss: Int, c: Float64) raises -> SvcFit:
    var d = QN_FEATURES
    var options = LinearSvcOptions(n, d, first, fits, max_iter, 100, 5, True, True, loss, 0.0, 1.0 / c, 1e-4, 1e-5)
    var plan = LinearSvcFitPlan(fits, d + 1)
    plan.storage[0].prepare_info(fits)
    driver.fit(LinearSvcFitRequest(Int(x.unsafe_ptr()), Int(codes.unsafe_ptr()),
        Int(weights.unsafe_ptr()) if len(weights) > 0 else 0, options^), plan)
    var w = List[Float32](length=fits * (d + 1), fill=0)
    var coefficients = List[Float32](length=fits * d, fill=0)
    var biases = List[Float32](length=fits, fill=0)
    linear_svc_pack(plan.storage[0], fits, d, True, Int(w.unsafe_ptr()),
        Int(coefficients.unsafe_ptr()), Int(biases.unsafe_ptr()))
    var objective = Float64(plan.storage[0].info[0])
    _ = x
    _ = codes
    _ = weights
    _ = plan^
    return SvcFit(w^, coefficients^, biases^, objective)


def fit_svr(driver: LinearSvcDriver, x: List[Float32], y: List[Float32], weights: List[Float32], n: Int,
            loss: Int, l1_penalty: Bool, c: Float64, epsilon: Float64) raises -> Tuple[List[Float32], Float64]:
    var d = QN_FEATURES
    var coefficients = List[Float32](length=d + 1, fill=0)
    var plan = QnSingleFitPlan()
    var request = QnFitRequest(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(coefficients.unsafe_ptr()), 0,
        Int(weights.unsafe_ptr()) if len(weights) > 0 else 0, 0, n, d, 1, 1000, 100, 5, True, True, loss,
        1.0 / c if l1_penalty else 0.0, 0.0 if l1_penalty else 1.0 / c, 1e-4, 1e-5, epsilon)
    plan.run(driver.qn(), request^)
    var objective = plan.objective()
    _ = x
    _ = y
    _ = weights
    _ = plan^
    return (coefficients^, objective)


def linear_scores(x: List[Float32], coefficients: List[Float32], bias: Float64, n: Int) raises -> List[Float64]:
    """z = X coef + b in Float64, the test's `_z`."""
    var result = List[Float64](length=n, fill=0)
    var xp = x.unsafe_ptr()
    var cp = coefficients.unsafe_ptr()
    var rp = result.unsafe_ptr()
    def row(index: Int) {imm xp, imm cp, imm rp, imm bias}:
        var z = bias
        for f in range(QN_FEATURES):  # one row's dot product
            z += Float64(xp[index * QN_FEATURES + f]) * Float64(cp[f])
        rp.unsafe_store(index, z)
    host_parallelize(row, n)
    _ = x
    _ = coefficients
    return result^


def penalty(coefficients: List[Float32], l1_penalty: Bool) -> Float64:
    var sum = Float64(0)
    for f in range(QN_FEATURES):  # five coefficients
        var value = Float64(coefficients[f])
        sum += abs(value) if l1_penalty else 0.5 * value * value
    return sum


def close_objective(reported: Float64, recomputed: Float64, name: String) raises:
    require(abs(reported - recomputed) < 1e-4 * max(1.0, recomputed),
        name + ": reported objective " + String(reported) + " is not the documented " + String(recomputed))


def all_close(actual: List[Float32], expected: List[Float32], name: String) raises:
    require(len(actual) == len(expected), name + " extents differ")
    for i in range(len(actual)):  # the five coefficients, assert_allclose(rtol=1e-4, atol=1e-5)
        var e = Float64(expected[i])
        require(abs(Float64(actual[i]) - e) <= 1e-5 + 1e-4 * abs(e), name + " differs at " + String(i))


def svc_objective_case(driver: LinearSvcDriver, x: List[Float32], yr: List[Float32], loss: Int) raises:
    var n = QN_ROWS
    var codes = List[Int32](unsafe_uninit_length=n)
    var yp = yr.unsafe_ptr()
    var cp = codes.unsafe_ptr()
    def label(index: Int) {imm yp, imm cp}:
        cp.unsafe_store(index, Int32(1) if yp[index] > 0.5 else Int32(0))
    host_parallelize(label, n)
    var none = List[Float32]()
    var fit = fit_svc(driver, x, codes, none, n, 1, 1, 1000, loss, 2.0)
    var z = linear_scores(x, fit.coefficients, Float64(fit.biases[0]), n)
    var terms = List[Float64](length=n, fill=0)
    var zp = z.unsafe_ptr()
    var tp = terms.unsafe_ptr()
    def term(index: Int) {imm zp, imm tp, imm cp, imm loss}:
        var t = max(Float64(0), 1.0 - (2.0 * Float64(cp[index]) - 1.0) * zp[index])
        tp.unsafe_store(index, t if loss == LOSS_HINGE else t * t)
    host_parallelize(term, n)
    # lz.mean() + 0.5 * (1 / C) / n * coef.coef with C = 2; penalty(l2) is half coef.coef
    var objective = total(terms) / Float64(n) + 0.5 * (1.0 / 2.0) / Float64(n) * (2.0 * penalty(fit.coefficients, False))
    close_objective(fit.objective, objective, "LinearSVC loss " + String(loss))
    var scores = List[Float32](length=n, fill=0)
    var labels = List[Int64](length=n, fill=-9)
    driver.qn().query(QN_NATIVE_DECISION, QnQueryRequest(Int(x.unsafe_ptr()), Int(fit.w.unsafe_ptr()),
        Int(scores.unsafe_ptr()), 0, n, QN_FEATURES, 1, True, False))
    driver.qn().query(QN_NATIVE_BINARY, QnQueryRequest(Int(x.unsafe_ptr()), Int(fit.w.unsafe_ptr()),
        Int(labels.unsafe_ptr()), 0, n, QN_FEATURES, 1, True, False))
    var flags = List[Int](length=n, fill=0)
    var hits = List[Int](length=n, fill=0)
    var sp = scores.unsafe_ptr()
    var lp = labels.unsafe_ptr()
    var fp = flags.unsafe_ptr()
    var hp = hits.unsafe_ptr()
    def check(index: Int) {imm sp, imm lp, imm fp, imm hp, imm cp}:
        var sign = Int64(1) if sp[index] > 0 else Int64(0)
        fp.unsafe_store(index, Int(lp[index] != sign))
        hp.unsafe_store(index, Int(sign == Int64(cp[index])))
    host_parallelize(check, n)
    var wrong = 0
    var correct = 0
    for i in range(n):  # two tallies of independent row verdicts
        wrong += flags[i]
        correct += hits[i]
    require(wrong == 0, "LinearSVC labels are not the sign of the decision")
    require(Float64(correct) / Float64(n) > 0.9, "LinearSVC accuracy " + String(correct) + "/400 is not above 0.9")
    _ = codes^
    _ = scores^
    _ = labels^
    _ = z^
    _ = terms^


def svr_objective_case(driver: LinearSvcDriver, x: List[Float32], yr: List[Float32], loss: Int, l1_penalty: Bool) raises:
    var n = QN_ROWS
    var none = List[Float32]()
    var fit = fit_svr(driver, x, yr, none, n, loss, l1_penalty, 4.0, 0.05)
    var z = linear_scores(x, fit[0], Float64(fit[0][QN_FEATURES]), n)
    var terms = List[Float64](length=n, fill=0)
    var residuals = List[Float64](length=n, fill=0)
    var zp = z.unsafe_ptr()
    var yp = yr.unsafe_ptr()
    var tp = terms.unsafe_ptr()
    var rp = residuals.unsafe_ptr()
    def term(index: Int) {imm zp, imm yp, imm tp, imm rp, imm loss}:
        var r = Float64(yp[index]) - zp[index]
        var outside = max(Float64(0), abs(r) - 0.05)
        tp.unsafe_store(index, outside if loss == LOSS_EPSILON else outside * outside)
    host_parallelize(term, n)
    var objective = total(terms) / Float64(n) + (1.0 / 4.0) / Float64(n) * penalty(fit[0], l1_penalty)
    close_objective(fit[1], objective, "LinearSVR loss " + String(loss) + (" l1" if l1_penalty else " l2"))
    var predictions = List[Float32](length=n, fill=0)
    driver.qn().query(QN_NATIVE_DECISION, QnQueryRequest(Int(x.unsafe_ptr()), Int(fit[0].unsafe_ptr()),
        Int(predictions.unsafe_ptr()), 0, n, QN_FEATURES, 1, True, False))
    var mean = Float64(0)
    for i in range(n):  # one ordered reduction for the target mean
        mean += Float64(yr[i])
    mean /= Float64(n)
    var deviations = List[Float64](length=n, fill=0)
    var pp = predictions.unsafe_ptr()
    var dp = deviations.unsafe_ptr()
    def squares(index: Int) {imm pp, imm yp, imm rp, imm dp, imm mean}:
        var r = Float64(yp[index]) - Float64(pp[index])
        rp.unsafe_store(index, r * r)
        var c = Float64(yp[index]) - mean
        dp.unsafe_store(index, c * c)
    host_parallelize(squares, n)
    var r2 = 1.0 - total(residuals) / total(deviations)
    require(r2 > 0.9, "LinearSVR R^2 " + String(r2) + " is not above 0.9")
    _ = predictions^
    _ = deviations^
    _ = residuals^
    _ = terms^
    _ = z^


def one_vs_rest_case(driver: LinearSvcDriver, x: List[Float32]) raises:
    var n = QN_ROWS
    var codes = recorded_codes("y3-400.i32", n)
    var rest = List[Int32](unsafe_uninit_length=n)
    var cp = codes.unsafe_ptr()
    var bp = rest.unsafe_ptr()
    def binary(index: Int) {imm cp, imm bp}:
        bp.unsafe_store(index, Int32(1) if cp[index] == Int32(2) else Int32(0))
    host_parallelize(binary, n)
    var none = List[Float32]()
    var multi = fit_svc(driver, x, codes, none, n, 0, 3, 200, LOSS_SQUARED_HINGE, 1.0)
    var one = fit_svc(driver, x, rest, none, n, 1, 1, 200, LOSS_SQUARED_HINGE, 1.0)
    for f in range(QN_FEATURES):  # machine 2's five coefficient words
        # np.array_equal in the original: value equality of each coefficient
        require(multi.coefficients[2 * QN_FEATURES + f] == one.coefficients[f],
            "one-vs-rest machine 2 is not the binary fit of class 2, feature " + String(f))
    var scores = List[Float32](length=n * 3, fill=0)
    var labels = List[Int64](length=n, fill=-9)
    driver.qn().query(QN_NATIVE_DECISION, QnQueryRequest(Int(x.unsafe_ptr()), Int(multi.w.unsafe_ptr()),
        Int(scores.unsafe_ptr()), 0, n, QN_FEATURES, 3, True, False))
    # The fused multiclass label query exists where the provider says so
    # (linear_svc_native_cases.mojo applies the same gate).
    var fused = driver.metadata(QN_NATIVE_MULTICLASS_AVAILABLE) != 0
    if fused:
        driver.qn().query(QN_NATIVE_MULTICLASS, QnQueryRequest(Int(x.unsafe_ptr()), Int(multi.w.unsafe_ptr()),
            Int(labels.unsafe_ptr()), 0, n, QN_FEATURES, 3, True, False))
    var flags = List[Int](length=n, fill=0)
    var xp = x.unsafe_ptr()
    var sp = scores.unsafe_ptr()
    var lp = labels.unsafe_ptr()
    var mp = multi.coefficients.unsafe_ptr()
    var ip = multi.biases.unsafe_ptr()
    var fp = flags.unsafe_ptr()
    def check(row: Int) {imm xp, imm sp, imm lp, imm mp, imm ip, imm fp, imm fused}:
        var bad = False
        var best = 0
        for machine in range(3):  # the three machines of one row
            var z = Float64(ip[machine])
            for f in range(QN_FEATURES):  # one dot product
                z += Float64(xp[row * QN_FEATURES + f]) * Float64(mp[machine * QN_FEATURES + f])
            bad = bad or abs(Float64(sp[row * 3 + machine]) - z) >= 1e-4
            if sp[row * 3 + machine] > sp[row * 3 + best]:
                best = machine
        bad = bad or (fused and lp[row] != Int64(best))
        fp.unsafe_store(row, Int(bad))
    host_parallelize(check, n)
    var failed = 0
    for value in flags:  # independent row verdicts
        failed += value
    require(failed == 0, "one-vs-rest decision or argmax label differs on " + String(failed) + " rows")
    _ = codes^
    _ = rest^
    _ = scores^
    _ = labels^


def fixture(driver: LinearSvcDriver, unit: Int) raises:
    if unit <= 5 or unit == 9:
        var x = recorded_f32("x-400x5.f32", QN_ROWS * QN_FEATURES)
        var yr = recorded_f32("yr-400.f32", QN_ROWS)
        if unit <= 1:
            svc_objective_case(driver, x, yr, LOSS_HINGE if unit == 0 else LOSS_SQUARED_HINGE)
        elif unit <= 5:
            svr_objective_case(driver, x, yr, LOSS_EPSILON if unit <= 3 else LOSS_SQUARED_EPSILON, unit % 2 == 0)
        else:
            one_vs_rest_case(driver, x)
    elif unit == 6:
        var x = recorded_f32("x-400x5.f32", QN_ROWS * QN_FEATURES)
        var yr = recorded_f32("yr-400.f32", QN_ROWS)
        var none = List[Float32]()
        var a = fit_svr(driver, x, yr, none, QN_ROWS, LOSS_EPSILON, False, 1.0, 0.0)
        var b = fit_svr(driver, x, yr, none, QN_ROWS, LOSS_EPSILON, False, 1.0, 0.5)
        var same = True
        for f in range(QN_FEATURES):  # coef_ words of the two fits
            same = same and bitcast[DType.uint32](a[0][f]) == bitcast[DType.uint32](b[0][f])
        require(not same, "LinearSVR epsilon 0 and 0.5 gave the same coefficients: epsilon was not read")
    else:
        var x = recorded_f32("x-400x5.f32", QN_SMALL_ROWS * QN_FEATURES)
        var yr = recorded_f32("yr-64.f32", QN_SMALL_ROWS)
        var none = List[Float32]()
        var ones = List[Float32](length=QN_SMALL_ROWS, fill=1)
        if unit == 7:
            var codes = List[Int32](unsafe_uninit_length=QN_SMALL_ROWS)
            var yp = yr.unsafe_ptr()
            var cp = codes.unsafe_ptr()
            def label(index: Int) {imm yp, imm cp}:
                cp.unsafe_store(index, Int32(1) if yp[index] > 0.5 else Int32(0))
            host_parallelize(label, QN_SMALL_ROWS)
            var a = fit_svc(driver, x, codes, none, QN_SMALL_ROWS, 1, 1, 1000, LOSS_SQUARED_HINGE, 1.0)
            var b = fit_svc(driver, x, codes, ones, QN_SMALL_ROWS, 1, 1, 1000, LOSS_SQUARED_HINGE, 1.0)
            all_close(b.coefficients, a.coefficients, "LinearSVC unit weights")
        else:
            var a = fit_svr(driver, x, yr, none, QN_SMALL_ROWS, LOSS_EPSILON, True, 1.0, 0.0)
            var b = fit_svr(driver, x, yr, ones, QN_SMALL_ROWS, LOSS_EPSILON, True, 1.0, 0.0)
            var ca = List[Float32](length=QN_FEATURES, fill=0)
            var cb = List[Float32](length=QN_FEATURES, fill=0)
            for f in range(QN_FEATURES):  # coef_ is the first five words
                ca[f] = a[0][f]
                cb[f] = b[0][f]
            all_close(cb, ca, "LinearSVR unit weights")


# ---- The host column (was solver_qn_objectives_cpu_suite.mojo) ----
# LinearSVC/LinearSVR documented objectives on the host QN provider; the ten
# independent fixtures run across host cores.
def _run_cpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(QN_OBJECTIVE_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            fixture(LinearSvcDriver(linear_svc_host_v1), unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, QN_OBJECTIVE_FIXTURES)


# ---- The device column (was solver_qn_objectives_gpu_suite.mojo) ----
# LinearSVC/LinearSVR documented objectives on the device QN provider.
#
# The ten fixtures run in order: they submit to the shared family
# DeviceContext, and concurrent submission from several host threads to one
# context is a separately recorded open failure. The QN kernels keep their own
# parallelism; the Float64 objective cells use host cores.
def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(QN_OBJECTIVE_FIXTURES)
    for unit in range(QN_OBJECTIVE_FIXTURES):
        try:
            fixture(LinearSvcDriver(linear_svc_device_v1), unit)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("solver_qn_objectives", cpu=True, gpu=True, serial="", run=run_cases)
