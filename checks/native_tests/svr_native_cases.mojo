# SPDX-License-Identifier: Apache-2.0
"""Actual SVR providers, independent two-point optima and the 150x5 surface workload.

The planted workload, salts, noise and quality bounds come from
test_svr_surface.arm_planted. Analytic cases use a symmetric two-point
dual: alpha=min(C,(1-epsilon)/(Kself-Kcross)), bias=0. These fixtures do
not fit another model to manufacture expected predictions.
"""
from std.math import exp, tanh
from core.host_parallel import host_parallelize
from core.labels_native_host import labels_host_fsum
from core.svr_native_api import SvrNativeDriver, SvrFitPlan, SvrFitRequest, SvrQueryRequest

comptime SVR_NATIVE_CASES = 12
comptime SVR_RESIDENT_CASES = 9


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _unit(index: Int, salt: UInt64) -> Float32:
    var word = UInt64(index + 1) * UInt64(0x9E3779B97F4A7C15) + salt
    word = (word ^ (word >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    word = (word ^ (word >> 27)) * UInt64(0x94D049BB133111EB)
    word = word ^ (word >> 31)
    return Float32(Int(word >> 43) - 1048576) / Float32(1048576)


def _sum(values: List[Float64]) raises -> Float64:
    var total = List[Float64](length=1, fill=0)
    require(labels_host_fsum(Int(values.unsafe_ptr()), len(values), Int(total.unsafe_ptr())) == 0,
        "independent reference sum refused")
    _ = values
    return total[0]


def fixture(driver: SvrNativeDriver, unit: Int, resident: Bool = False) raises:
    var planted = unit == 7 or unit == 8
    var n = 150 if planted else 2
    var d = 5 if planted else 2
    var kernel = 2 if unit == 8 else (unit if unit < 5 else 0)
    var gamma = Float64(1 if kernel == 1 else 0.5)
    var coef0 = Float64(0.5 if kernel == 1 else 0)
    var c = Float64(100 if planted else (0.5 if unit == 4 else 2))
    var epsilon = Float64(0.001 if planted else 0.1)
    var tolerance = Float64(0.001 if unit == 8 else (0.0001 if planted else 0.000001))
    var source = List[Float32](length=n * d, fill=0)
    var target = List[Float32](length=n, fill=0)
    var bounds = List[Float32](length=n, fill=0.2)
    var xp = source.unsafe_ptr()
    var yp = target.unsafe_ptr()
    def fill(row: Int) {imm xp, imm yp, imm d, imm planted, imm kernel}:
        if planted:
            var weights: Array[Float32, 5] = [0.75, -0.5, 0.25, -1.25, 0.5]
            var value = Float32(0)
            for column in range(5):  # fixed five-coordinate independent reference dot
                var x = _unit(row * d + column, UInt64(0x51D3))
                xp[row * d + column] = x
                value += x * weights[column]
            yp[row] = value + Float32(0.125) + _unit(row, UInt64(0x9E11)) * Float32(0.06)
        else:
            var sign = Float32(-1 if row == 0 else 1)
            yp[row] = sign
            if kernel == 4:
                xp[row * d + row] = 1
            else:
                xp[row * d] = sign
    host_parallelize(fill, n)
    var plan = SvrFitPlan(n, d)
    var first_address = Int(plan.dual.unsafe_ptr())
    var fitted = plan^
    require(Int(fitted.dual.unsafe_ptr()) == first_address, "SVR fit scratch move copied storage")
    var request = SvrFitRequest(Int(xp), Int(yp), Int(fitted.dual.unsafe_ptr()),
        Int(fitted.support.unsafe_ptr()), Int(fitted.vectors.unsafe_ptr()), Int(fitted.info.unsafe_ptr()),
        Int(bounds.unsafe_ptr()) if unit == 5 else 0, 0 if unit == 9 else n, d,
        kernel, -1 if planted else 100000, 1000, 2, 0, 0, 0, driver.device() and unit != 11, False, unit == 11,
        gamma, c, epsilon, tolerance, coef0)
    if unit == 9:
        var refused = False
        try:
            _ = driver.fit(request)
        except problem:
            require(String(problem).startswith("svr_fit: n_rows and n_features must both be positive"),
                "SVR zero-row refusal changed")
            refused = True
        require(refused, "SVR zero rows accepted")
        _ = source
        _ = target
        _ = fitted
        return
    if unit != 6 and unit != 10:
        fitted.count = driver.fit(request)
        require(fitted.count > 0 and fitted.count <= n, "SVR support count outside original bound")
        require(Int(fitted.info[1]) == fitted.count, "SVR count/result metadata disagree")
        require(fitted.info[2] >= 1, "SVR iteration count missing")
        if driver.device():
            require(fitted.info[3] == 0 and fitted.info[4] == 0, "SVR finite/range status changed")
    var count = fitted.count
    var query = List[Float32]()
    var query_address = Int(xp)
    var query_width = d
    if kernel == 4:
        query = List[Float32](length=n * count, fill=0)
        var qp = query.unsafe_ptr()
        # The fitted owner is named after the synchronous join and query.
        var sp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=Int(fitted.support.unsafe_ptr()))
        def cross(cell: Int) {imm qp, imm sp, imm count}:
            qp[cell] = Float32(Int(sp[cell % count]) == cell // count)
        host_parallelize(cross, n * count)
        query_address = Int(qp)
        query_width = count
    var prediction = List[Float32](length=n, fill=-999)
    var status = List[Int32](length=2, fill=-1)
    var lease = driver.lease_open() if resident and kernel != 4 else 0
    var query_request = SvrQueryRequest(query_address,
        Int(fitted.dual.unsafe_ptr()) if count else 0, Int(fitted.vectors.unsafe_ptr()) if count else 0,
        Int(prediction.unsafe_ptr()), Int(status.unsafe_ptr()), n, query_width,
        -1 if unit == 10 else count, kernel, 2, lease, 0, 0, 0, 0,
        1.25 if unit == 6 else fitted.info[0], gamma, 256, coef0)
    try:
        if unit == 10:
            var refused = False
            try:
                _ = driver.query(query_request, resident)
            except problem:
                require(String(problem).startswith("svr_predict: n_support cannot be negative"),
                    "SVR negative support refusal changed")
                refused = True
            require(refused, "SVR negative support accepted")
        else:
            _ = driver.query(query_request, resident)
            if resident:
                require(status[0] == 0 and status[1] == 0, "SVR resident query status changed")
                # A second query exercises the same native resident lease.
                _ = driver.query(query_request, True)
            var prediction_ptr = prediction.unsafe_ptr()
            var failures = List[Int](length=n, fill=0)
            var failure_ptr = failures.unsafe_ptr()
            var squared = List[Float64](length=n, fill=0)
            var targets_wide = List[Float64](length=n, fill=0)
            var sqp = squared.unsafe_ptr()
            var twp = targets_wide.unsafe_ptr()
            var expected_scale = Float64(0.4 if unit == 5 else (0.5 if kernel == 4 else 0.9))
            def check(row: Int) {imm prediction_ptr, imm failure_ptr, imm sqp, imm twp, imm yp, imm unit, imm planted, imm expected_scale}:
                var expected = Float64(1.25 if unit == 6 else (Float64(-1 if row == 0 else 1) * expected_scale))
                var delta = Float64(prediction_ptr[row]) - (Float64(yp[row]) if planted else expected)
                failure_ptr[row] = Int(abs(delta) >= (0.24 if unit == 7 else 0.0002)) if not planted or unit == 7 else 0
                sqp[row] = delta * delta
                twp[row] = Float64(yp[row])
            host_parallelize(check, n)
            var failed = 0
            for flag in failures:  # independent fixture status words
                failed += flag
            require(failed == 0, "SVR prediction differs from independent reference")
            if planted:
                var center = _sum(targets_wide) / Float64(n)
                var variance = List[Float64](length=n, fill=0)
                var vp = variance.unsafe_ptr()
                def centered(row: Int) {imm vp, imm twp, imm center}:
                    var delta = twp[row] - center
                    vp[row] = delta * delta
                host_parallelize(centered, n)
                var score = 1 - _sum(squared) / _sum(variance)
                require(score > (0.99 if unit == 7 else 0.9), "SVR original planted quality bound failed")
                if unit == 7:
                    require(abs(fitted.info[0] - 0.125) < 0.08, "SVR planted intercept changed")
                _ = variance
            else:
                if unit != 6:
                    var difference = Float64(2)
                    if kernel == 2:
                        difference = 1 - exp(Float64(-2))
                    elif kernel == 3:
                        difference = 2 * tanh(Float64(0.5))
                    elif kernel == 4:
                        difference = 1
                    var expected_alpha = min(Float64(0.2 if unit == 5 else c), 0.9 / difference)
                    require(count == 2, "symmetric SVR support count changed")
                    require(abs(Float64(fitted.dual[0]) + expected_alpha) < 0.0002,
                        "SVR negative dual differs from analytic optimum")
                    require(abs(Float64(fitted.dual[1]) - expected_alpha) < 0.0002,
                        "SVR positive dual differs from analytic optimum")
            _ = targets_wide
            _ = squared
    finally:
        if lease > 0:
            driver.lease_close(lease)
        _ = source
        _ = target
        _ = bounds
        _ = fitted
        _ = query
        _ = prediction
        _ = status
