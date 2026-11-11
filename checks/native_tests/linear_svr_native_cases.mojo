# SPDX-License-Identifier: Apache-2.0
"""The existing SVR QN losses against independent one-coordinate optima.

For balanced +/-1 targets, t=1-epsilon. Squared loss minimizes
(t-w)^2 + l2*w^2/2 + l1*abs(w). The weighted-intercept case reduces to
1.5*(t-w)^2 + l2*w^2/2 with b=(t-w)/2. Absolute loss with l2=2 has
w=1/2 while it remains outside the epsilon tube. No solver computes an oracle.
"""
from core.host_parallel import host_parallelize
from core.logistic_native_api import QnFitRequest, QnQueryRequest, QN_NATIVE_DECISION
from core.linear_svc_native_api import LinearSvcDriver
from core.qn_single_fit import QnSingleFitPlan

comptime LINEAR_SVR_NATIVE_CASES = 6


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def close(actual: Float64, expected: Float64) -> Bool:
    return Bool(abs(actual - expected) < 0.0002)


def fixture(driver: LinearSvcDriver, unit: Int) raises:
    var n = 400 if unit == 0 else 8
    var d = 5 if unit == 0 else 2
    var intercept = unit == 4
    var source = List[Float32](length=n * d, fill=0)
    var target = List[Float32](length=n, fill=0)
    var weights = List[Float32](length=n, fill=1)
    var coefficients = List[Float32](length=d + Int(intercept), fill=0)
    var predictions = List[Float32](length=n, fill=0)
    var xp = source.unsafe_ptr()
    var yp = target.unsafe_ptr()
    var sp = weights.unsafe_ptr()
    def fill(row: Int) {imm xp, imm yp, imm sp, imm d, imm unit}:
        var sign = Float32(-1 if row % 2 == 0 else 1)
        xp[row * d] = sign
        yp[row] = sign
        if unit == 4:
            sp[row] = Float32(1 if row % 2 == 0 else 3)
    host_parallelize(fill, n)
    var plan = QnSingleFitPlan()
    var original_info = Int(plan.info.unsafe_ptr())
    var moved = plan^
    require(Int(moved.info.unsafe_ptr()) == original_info, "single-target metadata move copied storage")
    var l1 = Float64(0.5 if unit == 3 else 0)
    var l2 = Float64(0 if unit == 3 else (2 if unit == 2 else 0.5))
    var epsilon = Float64(-0.1 if unit == 5 else 0.1)
    var request = QnFitRequest(Int(xp), Int(yp), Int(coefficients.unsafe_ptr()), 0,
        Int(sp) if unit == 4 else 0, 0, n, d, 1, 200, 50, 5, intercept, unit == 0,
        5 if unit == 2 else 6, l1, l2, 0.000001, 0.000000001, epsilon)
    if unit == 5:
        var refused = False
        try:
            moved.run(driver.qn(), request)
        except problem:
            require(String(problem).startswith("qn_fit: svr_eps must be finite and non-negative"),
                "epsilon refusal text changed")
            refused = True
        require(refused, "negative epsilon accepted")
        _ = source
        _ = target
        _ = coefficients
        return
    moved.run(driver.qn(), request)
    var t = 1 - epsilon
    var strength = l2 / Float64(n) if unit == 0 else l2
    var expected_w = (2 * t - l1) / (2 + strength)
    var expected_bias = Float64(0)
    var expected_fx = (t - expected_w) * (t - expected_w) + strength * expected_w * expected_w / 2 + l1 * expected_w
    if unit == 2:
        expected_w = 0.5
        expected_fx = t - 0.25
    elif unit == 4:
        expected_w = 6 * t / 7
        expected_bias = t / 14
        expected_fx = 3 * t * t / 14
    require(close(Float64(coefficients[0]), expected_w), "SVR fitted coefficient differs from analytic optimum")
    require(close(moved.objective(), expected_fx), "SVR objective differs from analytic optimum")
    require(moved.iterations >= 0 and moved.iterations <= 200, "SVR iteration metadata outside solver bound")
    if intercept:
        require(close(Float64(coefficients[d]), expected_bias), "SVR weighted intercept differs from analytic optimum")
    driver.qn().query(QN_NATIVE_DECISION, QnQueryRequest(Int(xp), Int(coefficients.unsafe_ptr()),
        Int(predictions.unsafe_ptr()), 0, n, d, 1, intercept, False))
    var unused = List[Int](length=d, fill=0)
    var up = unused.unsafe_ptr()
    var cp = coefficients.unsafe_ptr()
    def check_column(column: Int) {imm cp, imm up}:
        if column > 0:
            up[column] = Int(not close(Float64(cp[column]), 0))
    host_parallelize(check_column, d)
    var unused_failed = 0
    for flag in unused:  # metadata: independent coordinate check statuses
        unused_failed += flag
    require(unused_failed == 0, "SVR unused coefficient changed")
    var pp = predictions.unsafe_ptr()
    var failures = List[Int](length=n, fill=0)
    var fp = failures.unsafe_ptr()
    def check(row: Int) {imm pp, imm fp, imm expected_w, imm expected_bias}:
        var sign = Float64(-1 if row % 2 == 0 else 1)
        fp[row] = Int(not close(Float64(pp[row]), sign * expected_w + expected_bias))
    host_parallelize(check, n)
    var failed = 0
    for flag in failures:  # metadata: independent row check statuses
        failed += flag
    require(failed == 0, "SVR predictions differ from analytic optimum")
    _ = moved
    _ = source
    _ = target
    _ = weights
    _ = coefficients
    _ = predictions
