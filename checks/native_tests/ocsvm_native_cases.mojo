# SPDX-License-Identifier: Apache-2.0
"""Original OneClassSVM workloads and independent kernel/KKT references.

Each reference row is independent and parallel. Coordinate and dual sums inside
one reference row are inherently ordered reductions, evaluated in Float64;
they do not call a production kernel or another fitted model. Original Python
scikit-learn comparisons remain retained until the orchestrator accepts these
additional native contracts. No timing or cross-device identity is asserted.
"""
from std.math import exp, tanh
from core.host_parallel import host_parallelize
from core.labels_native_host import labels_host_fsum
from core.neighbors_native_driver import XnDriver
from core.ocsvm_resident_api import OcResidentDriver
from core.ocsvm_plan import (OcMatrix, OcWeights, OcFitState, oc_f32_address, oc_i32_address,
    oc_take_rows, oc_take_cols, oc_kernel, oc_variance, oc_kernel_product,
    oc_fused_labels, oc_product, oc_decision, oc_sign)
from checks.native_tests.ocsvm_native_inputs import oc_training, oc_query

comptime OCSVM_NATIVE_CASES = 12


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _sum(values: List[Float64]) raises -> Float64:
    var dst = List[Float64](length=1, fill=0)
    require(labels_host_fsum(Int(values.unsafe_ptr()), len(values), Int(dst.unsafe_ptr())) == 0,
        "reference summation refused")
    _ = values
    return dst[0]


def _kernel(lhs: Int, row: Int, rhs: Int, column: Int, features: Int,
    kernel: Int, degree: Int, gamma: Float64, coef0: Float64) -> Float64:
    var xp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=lhs)
    var yp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=rhs)
    if kernel == 4:
        return Float64(xp[row * features + column])
    var dot = Float64(0)
    var squared = Float64(0)
    for coordinate in range(features):  # one independent reference kernel's coordinate reductions
        var x = Float64(xp[row * features + coordinate])
        var y = Float64(yp[column * features + coordinate])
        dot += x * y
        var delta = x - y
        squared += delta * delta
    if kernel == 0:
        return dot
    if kernel == 2:
        return exp(-gamma * squared)
    if kernel == 3:
        return tanh(gamma * dot + coef0)
    var value = Float64(1)
    for power in range(degree):  # fixed scalar polynomial exponent, independent of sample count
        value *= gamma * dot + coef0
    return value


def fixture(driver: XnDriver, resident_driver: OcResidentDriver, unit: Int, resident: Bool = False) raises:
    var original = unit >= 6 and unit <= 10
    var n = 150 if original else (4 if unit == 5 else 2)
    var d = 6 if original else 2
    var kernel = 2
    if unit == 1 or unit == 7:
        kernel = 0
    elif unit == 2 or unit == 8:
        kernel = 1
    elif unit == 3 or unit == 9:
        kernel = 3
    elif unit == 4:
        kernel = 4
    var source = oc_training() if original else List[Float32](length=n * d, fill=0)
    var sample_weights = List[Float32](length=n, fill=1)
    var xp = source.unsafe_ptr()
    var wp = sample_weights.unsafe_ptr()
    def prepare(row: Int) {imm xp, imm wp, imm n, imm d, imm unit, imm original, imm kernel}:
        if unit == 7:
            for column in range(d):  # this independent row's fixed coordinates
                xp[row * d + column] += 2
        elif not original:
            if kernel == 4:
                xp[row * d + row] = 1
            elif unit == 5:
                xp[row * d] = Float32(-5 if row == 0 else (-1 if row == 1 else (1 if row == 2 else 5)))
            else:
                xp[row * d] = Float32(-1 if row == 0 else 1)
        if unit == 5:
            wp[row] = Float32(1 if row == 1 or row == 2 else 0)
        elif unit == 10:
            wp[row] = Float32(0 if row == 7 else Float32(1 + row % 3) * Float32(0.5))
        elif unit == 11:
            wp[row] = 0
    host_parallelize(prepare, n)
    var weights = OcWeights()
    if unit == 5 or unit == 10 or unit == 11:
        if unit == 11:
            var refused = False
            try:
                weights.compact(driver, Int(wp), n)
            except problem:
                require(String(problem).startswith("Invalid input - all samples have zero or negative weights."),
                    "OneClassSVM all-zero refusal changed")
                refused = True
            require(refused, "OneClassSVM accepted all-zero sample weights")
            _ = source
            _ = sample_weights
            return
        weights.compact(driver, Int(wp), n)
        require(weights.count == (2 if unit == 5 else 149), "positive compaction count changed")
        if unit == 5:
            require(weights.rows[0] == 1 and weights.rows[1] == 2, "positive row order changed")
    else:
        weights.unit(n)
    var m = weights.count
    var training = OcMatrix(n, d, Int(xp))
    if weights.selected:
        training = oc_take_rows(driver, Int(xp), n, d, oc_i32_address(weights.rows), m)
    var gamma = Float64(0.1 if original else (1 if kernel == 1 else 0.5))
    if unit == 6 or unit == 10:
        var variance = oc_variance(driver, Int(xp), n * d)
        gamma = Float64(Float32(1 / (Float64(d) * variance) if variance != 0 else 1))
    var degree = 3 if original else 2
    var coef0 = Float64(0.5 if unit == 2 else 0)
    var nu = Float64(0.3 if original else 0.5)
    var tolerance = Float64(0.00001 if unit == 8 else (0.001 if original else 0.000001))
    var gram = OcMatrix(0, 0)
    if kernel == 4:
        gram = OcMatrix(n, d, Int(xp))
    elif not resident:
        gram = oc_kernel(driver, training.address(), training.address(), m, m, d, kernel, degree, gamma, coef0)
    var fitted = OcFitState(m)
    var alpha_address = oc_f32_address(fitted.alpha)
    fitted.initialize(driver, oc_f32_address(weights.values), nu)
    require(oc_f32_address(fitted.alpha) == alpha_address, "alpha initialization replaced native storage")
    fitted.prepare_metadata()
    if resident and kernel != 4:
        fitted.solve_resident(resident_driver, training.address(), oc_f32_address(weights.values), d,
            kernel, degree, 10000000, False, gamma, coef0, tolerance)
    else:
        fitted.solve_gram(driver, gram.address(), oc_f32_address(weights.values), 10000000, tolerance)
    var rho = Float64(fitted.info[0])
    var alpha = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=alpha_address)
    var cv = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=oc_f32_address(weights.values))
    var alpha_wide = List[Float64](length=m, fill=0)
    var cv_wide = List[Float64](length=m, fill=0)
    var failures = List[Int](length=m, fill=0)
    var ap = alpha_wide.unsafe_ptr()
    var cp = cv_wide.unsafe_ptr()
    var fp = failures.unsafe_ptr()
    var training_address = training.address()
    def check_kkt(row: Int) {imm alpha, imm cv, imm ap, imm cp, imm fp, imm training_address,
        imm m, imm d, imm kernel, imm degree, imm gamma, imm coef0, imm rho, imm tolerance}:
        var score = Float64(0)
        var magnitude = Float64(0)
        for column in range(m):  # one independent reference row's dual reduction
            var term = Float64(alpha[column]) * _kernel(training_address, row, training_address,
                column, d, kernel, degree, gamma, coef0)
            score += term
            magnitude += abs(term)
        var a = Float64(alpha[row])
        var bound = Float64(cv[row])
        ap[row] = a
        cp[row] = bound
        # Float32 production kernels may differ from this Float64 reference.
        # The bound scales with absolute terms, never a fitted reference model.
        var error = 20 * tolerance + 0.00005 * magnitude + 0.0002
        var delta = score - rho
        var violation = Bool(a < -0.000002 or a > bound + 0.000002)
        if a <= 0.000002:
            violation = violation or delta < -error
        elif a >= bound - 0.000002:
            violation = violation or delta > error
        else:
            violation = violation or abs(delta) > error
        fp[row] = Int(violation)
    host_parallelize(check_kkt, m)
    var failed = 0
    for flag in failures:  # independent KKT status words
        failed += flag
    require(failed == 0, "OneClassSVM constrained optimum/KKT conditions failed")
    var total_bounds = _sum(cv_wide)
    require(abs(_sum(alpha_wide) - nu * total_bounds) < max(Float64(0.0002), 0.000002 * total_bounds),
        "OneClassSVM alpha sum no longer equals nu times total bounds")
    if not original:
        require(abs(Float64(alpha[0]) - 0.5) < 0.0002 and abs(Float64(alpha[1]) - 0.5) < 0.0002,
            "symmetric OneClassSVM dual differs from analytic optimum")
        var expected_rho = Float64(0)
        if kernel == 2:
            expected_rho = (1 + exp(Float64(-2))) / 2
        elif kernel == 1:
            expected_rho = 1.25
        elif kernel == 4:
            expected_rho = 0.5
        require(abs(rho - expected_rho) < 0.0002, "OneClassSVM rho differs from analytic optimum")
    fitted.compact_support(driver)
    var ns = fitted.supports
    require(ns > 0 and ns <= m, "OneClassSVM support count outside fitted bound")
    var mapped = List[Int32](length=ns, fill=0)
    var mp = mapped.unsafe_ptr()
    var sp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=oc_i32_address(fitted.support_rows))
    var rp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=oc_i32_address(weights.rows)) if weights.selected else sp
    var selected = weights.selected
    def map_support(index: Int) {imm mp, imm sp, imm rp, imm selected}:
        mp[index] = rp[Int(sp[index])] if selected else sp[index]
    host_parallelize(map_support, ns)
    var supports = OcMatrix(0, 0)
    if kernel != 4:
        supports = oc_take_rows(driver, Int(xp), n, d, Int(mp), ns)
    var nq = (40 if unit == 10 else n) if original else 4
    var queries = oc_query() if unit == 10 else (source.copy() if original else List[Float32](length=nq * d, fill=0))
    var qp = queries.unsafe_ptr()
    if not original:
        def fill_query(row: Int) {imm qp, imm d, imm kernel}:
            if kernel == 4:
                qp[row * d] = Float32(0.2 if row == 0 else (1.2 if row == 1 else (1 if row == 2 else -0.2)))
                qp[row * d + 1] = Float32(0.2 if row == 0 else (1 if row == 2 else 0))
            else:
                qp[row * d] = Float32(-2 if row == 0 else (-0.5 if row == 1 else (0.5 if row == 2 else 2)))
        host_parallelize(fill_query, nq)
    var cross = oc_take_cols(driver, Int(qp), nq, n, Int(mp), ns) if kernel == 4 else oc_kernel(
        driver, Int(qp), supports.address(), nq, ns, d, kernel, degree, gamma, coef0)
    var scores = List[Float32](length=nq, fill=0)
    var fused = List[Float32](length=nq, fill=0)
    var decisions = List[Float32](length=nq, fill=0)
    var labels = List[Int32](length=nq, fill=0)
    var fused_labels = List[Int32](length=nq, fill=0)
    oc_product(driver, cross.address(), oc_f32_address(fitted.support_values), Int(scores.unsafe_ptr()), nq, ns)
    oc_decision(driver, Int(scores.unsafe_ptr()), Int(decisions.unsafe_ptr()), nq, rho)
    oc_sign(driver, Int(decisions.unsafe_ptr()), Int(labels.unsafe_ptr()), nq)
    if kernel != 4:
        oc_kernel_product(driver, Int(qp), supports.address(), oc_f32_address(fitted.support_values),
            Int(fused.unsafe_ptr()), nq, ns, d, kernel, degree, gamma, coef0)
        oc_fused_labels(driver, Int(qp), supports.address(), oc_f32_address(fitted.support_values),
            Int(fused_labels.unsafe_ptr()), nq, ns, d, kernel, degree, gamma, coef0, rho)
    var query_flags = List[Int](length=nq, fill=0)
    var qfp = query_flags.unsafe_ptr()
    var scorep = scores.unsafe_ptr()
    var fusedp = fused.unsafe_ptr()
    var dp = decisions.unsafe_ptr()
    var lp = labels.unsafe_ptr()
    var flp = fused_labels.unsafe_ptr()
    var query_address = Int(qp)
    def check_query(row: Int) {imm qfp, imm scorep, imm fusedp, imm dp, imm lp, imm flp,
        imm query_address, imm training_address, imm alpha, imm m, imm d, imm kernel,
        imm degree, imm gamma, imm coef0, imm rho, imm original}:
        var expected = Float64(0)
        var magnitude = Float64(0)
        for column in range(m):  # one independent query row's reference dual reduction
            var term = Float64(alpha[column]) * _kernel(query_address, row, training_address,
                column, d, kernel, degree, gamma, coef0)
            expected += term
            magnitude += abs(term)
        var error = 0.00005 * magnitude + 0.0002 if original else Float64(0.0002)
        var bad = Bool(abs(Float64(scorep[row]) - expected) > error)
        bad = bad or abs(Float64(dp[row]) - (expected - rho)) > error
        if kernel != 4:
            bad = bad or abs(Float64(fusedp[row]) - expected) > error
        if abs(expected - rho) > error:
            var expected_label = Int32(1 if expected > rho else -1)
            bad = bad or lp[row] != expected_label
            if kernel != 4:
                bad = bad or flp[row] != expected_label
        qfp[row] = Int(bad)
    host_parallelize(check_query, nq)
    var query_failed = 0
    for flag in query_flags:  # independent query-reference status words
        query_failed += flag
    require(query_failed == 0, "OneClassSVM full/fused scores or labels differ from independent kernels")
    _ = source
    _ = sample_weights
    _ = weights
    _ = training
    _ = gram
    _ = fitted
    _ = alpha_wide
    _ = cv_wide
    _ = mapped
    _ = supports
    _ = queries
    _ = cross
    _ = scores
    _ = fused
    _ = decisions
    _ = labels
    _ = fused_labels
