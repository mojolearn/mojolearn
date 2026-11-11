# SPDX-License-Identifier: Apache-2.0
"""Actual GP graphs over original workloads and independent references.

This first tranche covers posterior/covariance/sampling and singular fits.
Normalization, gradients, optimizer and classifier graphs have separate pending
coverage; no existing Python numerical assertions are removed yet. GPU wrappers
order graphs on the one physical GPU, retaining parallelism inside every kernel.
"""
from std.math import isfinite, sqrt
from core.host_parallel import host_parallelize
from core.gp_native_api import GPDriver, GPRFitRequest, GPRQueryRequest, GPRCovarianceRequest, GPRSampleRequest, GP_CAPABILITIES, GP_CAP_FIT, GP_CAP_POINTER, GP_CAP_RESIDENT, GP_ROUTE_LIST, GP_ROUTE_POINTER, GP_ROUTE_RESIDENT
from checks.native_tests.gp_state_suite import require
from checks.native_tests.gp_native_data import input_values, reference_values, kernel_storage, guards, guards64, failures_clear, row_sums, GP_GUARD, GP_GUARD64

comptime GP_NATIVE_BASE_FIXTURES = 6
comptime GP_NATIVE_CPU_FIXTURES = 7
comptime GP_NATIVE_GPU_FIXTURES = 8


def unchanged(values: List[Float32], original: List[Float32], message: String) raises:
    require(len(values) == len(original), message + " extent")
    var failures = List[Int](length=len(values), fill=0)
    var vp = values.unsafe_ptr()
    var op = original.unsafe_ptr()
    var fp = failures.unsafe_ptr()
    def cell(index: Int) {imm vp, imm op, imm fp}:
        fp.unsafe_store(index, Int(vp[index] != op[index]))
    host_parallelize(cell, len(values))
    failures_clear(failures, message)
    _ = values
    _ = original


def posterior_oracle(name: String, targets: List[Float32], mean: List[Float32], variance: List[Float32],
                     stddev: List[Float32], clamped: List[Int32], covariance: List[Float32],
                     query_rows: Int, clamp_count: Int, recovery: Bool) raises:
    var expected_mean = reference_values(name, "mean", query_rows)
    var expected_covariance = reference_values(name, "covariance", query_rows * query_rows)
    var failures = List[Int](length=query_rows * query_rows, fill=0)
    var counts = List[Float32](length=query_rows, fill=0)
    var fp = failures.unsafe_ptr()
    var cp = covariance.unsafe_ptr() + 1
    var ep = expected_covariance.unsafe_ptr()
    var mp = mean.unsafe_ptr() + 1
    var vp = variance.unsafe_ptr() + 1
    var sp = stddev.unsafe_ptr() + 1
    var kp = clamped.unsafe_ptr() + 1
    var em = expected_mean.unsafe_ptr()
    var tp = targets.unsafe_ptr()
    var np = counts.unsafe_ptr()
    def cell(index: Int) {imm fp, imm cp, imm ep, imm mp, imm vp, imm sp, imm kp, imm em, imm tp, imm np, imm query_rows, imm recovery}:
        var row = index // query_rows
        var column = index % query_rows
        var value = Float64(cp[index])
        var okay = isfinite(value) and cp[index] == cp[column * query_rows + row]
        # Existing external covariance tolerance, now also against independent
        # equations on the original diagonal/symmetry workload (not old bits).
        okay = okay and abs(value - ep[index]) <= Float64(2e-3)
        if column == 0:
            var m = Float64(mp[row])
            var v = Float64(vp[row])
            var s = Float64(sp[row])
            okay = okay and isfinite(m) and isfinite(v) and isfinite(s) and s >= 0 and v >= 0
            if recovery:
                okay = okay and abs(m - Float64(tp[row])) < Float64(0.00006103515625)
            else:
                okay = okay and abs(m - em[row]) <= Float64(2e-3) + Float64(2e-4) * abs(em[row])
            var diagonal_std = sqrt(max(Float64(cp[row * query_rows + row]), Float64(0)))
            okay = okay and abs(diagonal_std - s) <= Float64(1e-4) + Float64(1e-3) * abs(s)
            okay = okay and (kp[row] == Int32(0) or kp[row] == Int32(1))
            np.unsafe_store(row, Float32(kp[row]))
        fp.unsafe_store(index, Int(not okay))
    host_parallelize(cell, query_rows * query_rows)
    failures_clear(failures, "GP independent posterior, original recovery/symmetry/diagonal bounds")
    var zero = List[Float64](length=1, fill=0)
    var count = row_sums(counts, 0, 1, query_rows, zero, False)
    require(Int(count[0]) == clamp_count, "GP original clamp flags/count consistency")
    _ = targets
    _ = mean
    _ = variance
    _ = stddev
    _ = clamped
    _ = covariance
    _ = expected_mean^
    _ = expected_covariance^


def sample_oracle(samples: List[Float32], mean: List[Float32], stddev: List[Float32],
                  rows: Int, count: Int, moments: Bool) raises:
    var failures = List[Int](length=rows * count, fill=0)
    var sp = samples.unsafe_ptr() + 1
    var fp = failures.unsafe_ptr()
    def finite_cell(index: Int) {imm sp, imm fp}:
        fp.unsafe_store(index, Int(not isfinite(sp[index])))
    host_parallelize(finite_cell, rows * count)
    failures_clear(failures, "GP samples finite")
    if moments:
        var zero = List[Float64](length=rows, fill=0)
        var sums = row_sums(samples, 1, rows, count, zero, False)
        var means = List[Float64](length=rows, fill=0)
        var sump = sums.unsafe_ptr()
        var emp = means.unsafe_ptr()
        def mean_cell(row: Int) {imm sump, imm emp, imm count}:
            emp.unsafe_store(row, sump[row] / Float64(count))
        host_parallelize(mean_cell, rows)
        var squares = row_sums(samples, 1, rows, count, means, True)
        var sqp = squares.unsafe_ptr()
        var mp = mean.unsafe_ptr() + 1
        var stp = stddev.unsafe_ptr() + 1
        def moment_cell(row: Int) {imm fp, imm emp, imm sqp, imm mp, imm stp, imm count}:
            var expected_std = Float64(stp[row])
            var empirical_std = sqrt(sqp[row] / Float64(count))
            var okay = expected_std > 0 and abs(emp[row] - Float64(mp[row])) < Float64(6) * expected_std / sqrt(Float64(count)) + Float64(1e-4)
            okay = okay and abs(empirical_std - expected_std) / expected_std < Float64(0.08)
            fp.unsafe_store(row, Int(not okay))
        host_parallelize(moment_cell, rows)
        failures_clear(failures, "GP original4000-draw mean6SE+1e-4 and std8percent bounds")
        _ = sums^
        _ = means^
        _ = squares^
    _ = samples
    _ = mean
    _ = stddev


def fitted_case(driver: GPDriver, query_driver: GPDriver, unit: Int, route: Int) raises:
    var name = "surface16"
    var rows = 16
    var features = 2
    var query_rows = 16
    var kind = 0
    if unit == 1:
        name = "cov60"
        rows = 60
        features = 3
        query_rows = 20
        kind = 1
    elif unit == 2:
        name = "parallel64"
        rows = 64
        features = 4
        query_rows = 32
        kind = 1
    elif unit == 3 or unit == 4:
        name = "sample_shape96" if unit == 3 else "sample_moments96"
        rows = 96
        features = 3
        query_rows = 24 if unit == 3 else 6
        kind = 1
    var source = input_values(name, "source", rows * features)
    var target = input_values(name, "target", rows)
    var query = input_values(name, "query", query_rows * features)
    var source_before = source.copy()
    var target_before = target.copy()
    var query_before = query.copy()
    var kernel = kernel_storage(kind)
    var factor = List[Float32](length=rows * rows + 2, fill=GP_GUARD)
    var dual = List[Float32](length=rows + 2, fill=GP_GUARD)
    var scalars = List[Float64](length=7, fill=GP_GUARD64)
    var mean = List[Float32](length=query_rows + 2, fill=GP_GUARD)
    var variance = List[Float32](length=query_rows + 2, fill=GP_GUARD)
    var stddev = List[Float32](length=query_rows + 2, fill=GP_GUARD)
    var clamped = List[Int32](length=query_rows + 2, fill=-31)
    var covariance = List[Float32](length=query_rows * query_rows + 2, fill=GP_GUARD)
    var covariance_mean = List[Float32](length=query_rows + 2, fill=GP_GUARD)
    var mean_only = List[Float32](length=query_rows + 2, fill=GP_GUARD)
    var lease = -1
    try:
        require((driver.metadata(GP_CAPABILITIES) & GP_CAP_FIT) != 0, "GP full provider fit capability")
        if route == GP_ROUTE_POINTER:
            require((driver.metadata(GP_CAPABILITIES) & GP_CAP_POINTER) != 0, "GP selected pointer capability")
        if route == GP_ROUTE_RESIDENT:
            require((query_driver.metadata(GP_CAPABILITIES) & GP_CAP_RESIDENT) != 0, "GP selected resident capability")
            lease = query_driver.lease_open()
        var status = driver.gpr_fit(GPRFitRequest(Int(source.unsafe_ptr()), Int(target.unsafe_ptr()),
            Int(factor.unsafe_ptr() + 1), Int(dual.unsafe_ptr() + 1), Int(scalars.unsafe_ptr() + 1),
            rows, features, kernel.view(), 0.00000095367431640625, route == GP_ROUTE_POINTER))
        require(status == 0 and scalars[1] == 0.0 and scalars[2] > 0 and isfinite(scalars[5]),
            "GP original full fit succeeds with finite likelihood")
        var request = GPRQueryRequest(Int(source.unsafe_ptr()), Int(factor.unsafe_ptr() + 1), Int(dual.unsafe_ptr() + 1),
            Int(query.unsafe_ptr()), Int(mean.unsafe_ptr() + 1), Int(variance.unsafe_ptr() + 1),
            Int(stddev.unsafe_ptr() + 1), Int(clamped.unsafe_ptr() + 1), rows, features, query_rows, kernel.view(),
            status, True, False, 1.0, 0.0, route, lease, -1, 0, -1)
        var result = query_driver.gpr_query(request)
        require(result.finite == 0 and result.range_status == 0, "GP admitted query status words")
        var cov_status = driver.gpr_covariance(GPRCovarianceRequest(Int(source.unsafe_ptr()), Int(factor.unsafe_ptr() + 1),
            Int(dual.unsafe_ptr() + 1), Int(query.unsafe_ptr()), Int(covariance_mean.unsafe_ptr() + 1),
            Int(covariance.unsafe_ptr() + 1), rows, features, query_rows, kernel.view(), status, False, 1.0, 0.0))
        require(cov_status == query_rows, "GP covariance return row count")
        posterior_oracle(name, target, mean, variance, stddev, clamped, covariance, query_rows, result.value, unit == 0)
        request.mean = Int(mean_only.unsafe_ptr() + 1)
        request.want_std = False
        request.variance = 0
        request.stddev = 0
        request.clamped = 0
        _ = query_driver.gpr_query(request)
        # Independent numerical agreement, not a same-bits/repeated-fit campaign.
        var expected = reference_values(name, "mean", query_rows)
        var failures = List[Int](length=query_rows, fill=0)
        var fp = failures.unsafe_ptr()
        var ep = expected.unsafe_ptr()
        var mp = mean_only.unsafe_ptr() + 1
        var cp = covariance_mean.unsafe_ptr() + 1
        def mean_cell(row: Int) {imm fp, imm ep, imm mp, imm cp}:
            var bound = Float64(2e-3) + Float64(2e-4) * abs(ep[row])
            fp.unsafe_store(row, Int(not (abs(Float64(mp[row]) - ep[row]) <= bound and abs(Float64(cp[row]) - ep[row]) <= bound)))
        host_parallelize(mean_cell, query_rows)
        failures_clear(failures, "GP mean-only/covariance output independent mean oracle")
        if unit == 3 or unit == 4:
            var count = 5 if unit == 3 else 4000
            var samples = List[Float32](length=query_rows * count + 2, fill=GP_GUARD)
            try:
                var written = driver.gpr_sample(GPRSampleRequest(Int(source.unsafe_ptr()), Int(factor.unsafe_ptr() + 1),
                    Int(dual.unsafe_ptr() + 1), Int(query.unsafe_ptr()), Int(samples.unsafe_ptr() + 1),
                    rows, features, query_rows, kernel.view(), status, count, 1 if unit == 3 else 3, 0, False, 1.0, 0.0))
                require(written == count, "GP sample return draw count")
                sample_oracle(samples, mean, stddev, query_rows, count, unit == 4)
                guards(samples, "GP sample output guards")
            finally:
                _ = samples^
        guards(factor, "GP factor output guards")
        guards(dual, "GP dual output guards")
        guards64(scalars, "GP fit scalar output guards")
        guards(mean, "GP mean output guards")
        guards(variance, "GP variance output guards")
        guards(stddev, "GP std output guards")
        require(clamped[0] == Int32(-31) and clamped[len(clamped) - 1] == Int32(-31), "GP clamp output guards")
        guards(covariance, "GP covariance output guards")
        guards(covariance_mean, "GP covariance mean output guards")
        guards(mean_only, "GP mean-only output guards")
        unchanged(source, source_before, "GP borrowed training source unchanged")
        unchanged(target, target_before, "GP borrowed target unchanged")
        unchanged(query, query_before, "GP borrowed query unchanged")
        _ = expected^
    finally:
        if lease >= 0:
            query_driver.lease_close(lease)
        _ = source^
        _ = target^
        _ = query^
        _ = kernel^
        _ = factor^
        _ = dual^
        _ = scalars^
        _ = mean^
        _ = variance^
        _ = stddev^
        _ = clamped^
        _ = covariance^
        _ = covariance_mean^
        _ = mean_only^


def singular_case(driver: GPDriver) raises:
    var source = input_values("surface16", "source", 32)
    var target = input_values("surface16", "target", 16)
    source[2] = source[0]
    source[3] = source[1]
    var kernel = kernel_storage(7)
    var factor = List[Float32](length=258, fill=GP_GUARD)
    var dual = List[Float32](length=18, fill=GP_GUARD)
    var scalars = List[Float64](length=7, fill=GP_GUARD64)
    try:
        var status = driver.gpr_fit(GPRFitRequest(Int(source.unsafe_ptr()), Int(target.unsafe_ptr()),
            Int(factor.unsafe_ptr() + 1), Int(dual.unsafe_ptr() + 1), Int(scalars.unsafe_ptr() + 1),
            16, 2, kernel.view(), 0.0, False))
        require(status != 0 and Int(scalars[1]) == status, "GP original duplicate-row alpha0 fit reports factorization failure")
        guards(factor, "GP failed-fit factor guards")
        guards(dual, "GP failed-fit dual guards")
        guards64(scalars, "GP failed-fit scalar guards")
    finally:
        _ = source^
        _ = target^
        _ = kernel^
        _ = factor^
        _ = dual^
        _ = scalars^


def fixture(driver: GPDriver, query_driver: GPDriver, unit: Int, host: Bool) raises:
    if unit == 5:
        singular_case(driver)
    elif unit == 6 and host:
        require(query_driver.metadata(GP_CAPABILITIES) == 0, "GP inference-only image capability mask")
        fitted_case(driver, query_driver, 0, GP_ROUTE_LIST)
    else:
        var route = GP_ROUTE_LIST if unit < GP_NATIVE_BASE_FIXTURES else (GP_ROUTE_POINTER if unit == 6 else GP_ROUTE_RESIDENT)
        fitted_case(driver, query_driver, unit if unit < GP_NATIVE_BASE_FIXTURES else 0, route)
