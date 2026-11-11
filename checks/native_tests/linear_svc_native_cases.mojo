# SPDX-License-Identifier: Apache-2.0
"""Existing QN solvers against closed-form hinge/squared-hinge objectives.

Binary unit points reduce the objective to A*(1-w)^2 + lambda*w^2/2.
The unequal-weight intercept case has A=3/2 and b=(1-w)/2. Three
one-hot classes give one independent coordinate per class and the OVR
solution +/-2/(2+3*lambda). These analytic references do not reuse the
solver or its kernels. The original random400x5 Python workload is retained.
"""
from core.host_parallel import host_parallelize
from core.linear_svc_native_api import (
    LinearSvcDriver, LinearSvcOptions, LinearSvcFitRequest, LinearSvcFitPlan,
    LINEAR_SVC_OVR, linear_svc_pack,
)
from core.logistic_native_api import (
    QnQueryRequest, QN_NATIVE_DECISION, QN_NATIVE_BINARY,
    QN_NATIVE_MULTICLASS, QN_NATIVE_MULTICLASS_AVAILABLE,
)
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime LINEAR_SVC_PROVIDER_CASES = 8
comptime LINEAR_SVC_CPU_CASES = 10


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def close(actual: Float32, expected: Float64) -> Bool:
    return Bool(abs(Float64(actual) - expected) < 0.0002)


def pack_fixture(unit: Int) raises:
    var intercept = unit == 8
    var fits = 3 if intercept else 2
    var width = 3 if intercept else 2
    var plan = LinearSvcFitPlan(fits, width)
    if intercept:
        plan.storage[0].coefficients = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    else:
        plan.storage[0].coefficients = [1.0, 2.0, 3.0, 4.0]
    var address = plan.address()
    var moved = plan^
    require(moved.address() == address, "native OVR owner address survives movement")
    var w = List[Float32](length=fits * width, fill=-9)
    var coefficients = List[Float32](length=fits * 2, fill=-9)
    var biases = List[Float32](length=fits, fill=7)
    linear_svc_pack(moved.storage[0], fits, 2, intercept, Int(w.unsafe_ptr()),
        Int(coefficients.unsafe_ptr()), Int(biases.unsafe_ptr()))
    if intercept:
        require(w[0] == 1 and w[1] == 4 and w[2] == 7 and w[3] == 2 and w[4] == 5
            and w[5] == 8 and w[6] == 3 and w[7] == 6 and w[8] == 9, "column-major public _w copy")
        require(coefficients[0] == 1 and coefficients[1] == 2 and coefficients[2] == 4
            and coefficients[3] == 5 and coefficients[4] == 7 and coefficients[5] == 8,
            "row-major public coefficients omit intercept")
        require(biases[0] == 3 and biases[1] == 6 and biases[2] == 9, "public intercept copy")
    else:
        require(w[0] == 1 and w[1] == 3 and w[2] == 2 and w[3] == 4, "no-intercept transpose")
        require(coefficients[0] == 1 and coefficients[1] == 2 and coefficients[2] == 3
            and coefficients[3] == 4, "no-intercept coefficient copy")
        require(biases[0] == 7 and biases[1] == 7, "disabled intercept preserves destination initialization")
    _ = w
    _ = coefficients
    _ = biases
    _ = moved


def fixture(driver: LinearSvcDriver, unit: Int) raises:
    if unit >= 8:
        pack_fixture(unit)
        return
    if unit == 7:
        var a: Array[Int, 4] = [0, 0, 0, 0]
        var p: Array[Int, 10] = [4, 2, 1, 1, 10, 10, 5, 1, 1, 4]
        var f: Array[Float64, 4] = [0, 1, 0.0001, 0.00001]
        var refused = False
        try:
            _ = native_descriptor_call_fixed(driver.entry, LINEAR_SVC_OVR, a, p, f)
        except problem:
            require(String(problem) == "LinearSVC native ABI: invalid OVR request", "null owner refusal before solver")
            refused = True
        require(refused, "null OVR owner accepted")
        return
    var multi = unit == 6
    var n = 399 if multi else (400 if unit == 0 else 8)
    var d = 5 if unit == 0 or multi else 2
    var fits = 3 if multi else 1
    var intercept = unit == 0 or unit == 4
    var source = List[Float32](length=n * d, fill=0)
    var codes = List[Int32](length=n, fill=0)
    var sample_weights = List[Float32](length=n, fill=1)
    var xp = source.unsafe_ptr()
    var cp = codes.unsafe_ptr()
    var sp = sample_weights.unsafe_ptr()
    def fill(row: Int) {imm xp, imm cp, imm sp, imm d, imm multi, imm unit}:
        if multi:
            cp[row] = Int32(row % 3)
            xp[row * d + row % 3] = 1
        else:
            cp[row] = Int32(row % 2)
            xp[row * d] = Float32(-1.0 if row % 2 == 0 else 1.0)
        if unit == 4:
            sp[row] = Float32(1.0 if row % 2 == 0 else 3.0)
        elif unit == 5:
            sp[row] = 2
    host_parallelize(fill, n)
    var l1 = Float64(0.5) if unit == 3 else Float64(0)
    var l2 = Float64(0) if unit == 3 else (Float64(2) if unit == 2 else Float64(0.5))
    var options = LinearSvcOptions(n, d, 0 if multi else 1, fits, 200, 50, 5,
        intercept, unit == 0, 3 if unit == 2 else 4, l1, l2, 0.000001, 0.000000001)
    var plan = LinearSvcFitPlan(fits, d + Int(intercept))
    plan.storage[0].prepare_info(fits)
    driver.fit(LinearSvcFitRequest(Int(xp), Int(cp), Int(sp) if unit == 4 or unit == 5 else 0, options), plan)
    var expected_w = Float64(0.8)
    var expected_bias = Float64(0)
    var expected_objective = Float64(0.2)
    if unit == 0:
        var penalty = Float64(0.5) / Float64(n)
        expected_w = 2 / (2 + penalty)
        expected_objective = penalty / (2 + penalty)
    elif unit == 2:
        expected_w = 0.5
        expected_objective = 0.75
    elif unit == 3:
        expected_w = 0.75
        expected_objective = 0.4375
    elif unit == 4:
        expected_w = Float64(6) / 7
        expected_bias = Float64(1) / 14
        expected_objective = Float64(3) / 14
    elif unit == 5:
        expected_w = Float64(8) / 9
        expected_objective = Float64(2) / 9
    elif multi:
        expected_w = Float64(4) / 7
        expected_objective = Float64(3) / 7
    require(plan.iterations() > 0 and plan.iterations() <= 200, "solver iteration result range")
    for fit in range(fits):  # metadata: one objective per model target
        require(close(plan.storage[0].info[3 * fit], expected_objective), "analytic fitted objective")
    var w = List[Float32](length=fits * (d + Int(intercept)), fill=0)
    var coefficients = List[Float32](length=fits * d, fill=0)
    var biases = List[Float32](length=fits, fill=0)
    linear_svc_pack(plan.storage[0], fits, d, intercept, Int(w.unsafe_ptr()),
        Int(coefficients.unsafe_ptr()), Int(biases.unsafe_ptr()))
    var wp = coefficients.unsafe_ptr()
    var bp = biases.unsafe_ptr()
    var coefficient_flags = List[Int](length=fits * d, fill=0)
    var cfp = coefficient_flags.unsafe_ptr()
    def check_coefficient(index: Int) {imm wp, imm bp, imm cfp, imm d, imm multi, imm expected_w, imm expected_bias}:
        var fit = index // d
        var feature = index % d
        var expected = Float64(0)
        if multi and feature < 3:
            expected = expected_w if feature == fit else -expected_w
        elif not multi and feature == 0:
            expected = expected_w
        if not close(wp[index], expected) or not close(bp[fit], expected_bias):
            cfp[index] = 1
    host_parallelize(check_coefficient, fits * d)
    for flag in coefficient_flags:  # metadata: independent coefficient verdicts
        require(flag == 0, "analytic coefficient/intercept optimum")
    var scores = List[Float32](length=n * fits, fill=0)
    driver.qn().query(QN_NATIVE_DECISION, QnQueryRequest(Int(xp), Int(w.unsafe_ptr()),
        Int(scores.unsafe_ptr()), 0, n, d, fits, intercept, False))
    var dp = scores.unsafe_ptr()
    var row_flags = List[Int](length=n, fill=0)
    var rfp = row_flags.unsafe_ptr()
    def check_scores(row: Int) {imm dp, imm rfp, imm multi, imm expected_w, imm expected_bias}:
        if multi:
            var label = row % 3
            if (not close(dp[row * 3 + label], expected_w)
                or not close(dp[row * 3 + (label + 1) % 3], -expected_w)
                or not close(dp[row * 3 + (label + 2) % 3], -expected_w)):
                rfp[row] = 1
        elif not close(dp[row], (-expected_w if row % 2 == 0 else expected_w) + expected_bias):
            rfp[row] = 1
    host_parallelize(check_scores, n)
    for flag in row_flags:  # metadata: independent query verdicts
        require(flag == 0, "analytic QN decision scores")
    if not multi or driver.metadata(QN_NATIVE_MULTICLASS_AVAILABLE) != 0:
        var predicted = List[Int64](length=n, fill=-9)
        driver.qn().query(QN_NATIVE_MULTICLASS if multi else QN_NATIVE_BINARY,
            QnQueryRequest(Int(xp), Int(w.unsafe_ptr()), Int(predicted.unsafe_ptr()), 0,
                n, d, fits, intercept, False))
        var pred = predicted.unsafe_ptr()
        def check_labels(row: Int) {imm pred, imm rfp, imm multi}:
            rfp[row] = Int(pred[row] != Int64(row % (3 if multi else 2)))
        host_parallelize(check_labels, n)
        for flag in row_flags:  # metadata: independent fused-label verdicts
            require(flag == 0, "analytic fused QN labels")
        _ = predicted
    _ = source
    _ = codes
    _ = sample_weights
    _ = scores
    _ = w
    _ = coefficients
    _ = biases
    _ = plan
