# SPDX-License-Identifier: Apache-2.0
"""Actual scaler/preparation/GP graphs with independent target statistics.

The three full graphs retain the original 60x3/96x3 input bytes, kernel,
alpha and query counts. Analytic missing-value cases exercise the private
normalizer only: they do not expand the GP public fit admission contract.
"""
from std.math import isfinite, isnan, sqrt
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.scaler_native_api import ScalerDriver, PrepProgramDriver
from core.gp_native_normalization import GPRNormalizeRequest, gp_normalize_targets
from core.gp_native_api import GPDriver, GPRFitRequest, GPRQueryRequest, GPRCovarianceRequest, GP_ROUTE_LIST
from checks.native_tests.gp_state_suite import require
from checks.native_tests.gp_native_cases import posterior_oracle, unchanged
from checks.native_tests.gp_native_data import input_values, reference_values, kernel_storage, guards, guards64, failures_clear, GP_GUARD, GP_GUARD64

comptime GP_NORMALIZATION_FIXTURES = 13


def normalized_graph(gp: GPDriver, scaler: ScalerDriver, program: PrepProgramDriver, unit: Int, direct: Bool) raises:
    var name = "cov60norm" if unit == 0 else ("normalize96" if unit == 1 else "constant96")
    var rows = 60 if unit == 0 else 96
    var queries = 20 if unit == 0 else 24
    var source = input_values(name, "source", rows * 3)
    var target = input_values(name, "target", rows)
    var query = input_values(name, "query", queries * 3)
    var source_before = source.copy()
    var target_before = target.copy()
    var query_before = query.copy()
    var expected_stats = reference_values(name, "statistics", 5)
    var expected_covariance = reference_values(name, "covariance", queries * queries)
    var expected_mean = reference_values(name, "mean", queries)
    var kernel = kernel_storage(1)
    var normalized = List[Float32](length=rows + 2, fill=GP_GUARD)
    var factor = List[Float32](length=rows * rows + 2, fill=GP_GUARD)
    var dual = List[Float32](length=rows + 2, fill=GP_GUARD)
    var scalars = List[Float64](length=7, fill=GP_GUARD64)
    var mean = List[Float32](length=queries + 2, fill=GP_GUARD)
    var variance = List[Float32](length=queries + 2, fill=GP_GUARD)
    var stddev = List[Float32](length=queries + 2, fill=GP_GUARD)
    var clamped = List[Int32](length=queries + 2, fill=-31)
    var covariance = List[Float32](length=queries * queries + 2, fill=GP_GUARD)
    var covariance_mean = List[Float32](length=queries + 2, fill=GP_GUARD)
    try:
        var norm = gp_normalize_targets(scaler, program,
            GPRNormalizeRequest(Int(target.unsafe_ptr()), Int(normalized.unsafe_ptr() + 1), rows, direct))
        require(abs(norm.mean - expected_stats[0]) < Float64(1e-3), "GP original target mean tolerance")
        require(abs(norm.stddev - expected_stats[1]) <= Float64(1e-4) * max(abs(expected_stats[1]), Float64(1)),
            "GP independent target population standard deviation")
        if unit == 2:
            require(norm.mean == 7.0 and norm.stddev == 1.0, "GP original constant mean7/scale1")
        var normalization_failures = List[Int](length=rows, fill=0)
        var nfp = normalization_failures.unsafe_ptr()
        var np = normalized.unsafe_ptr() + 1
        var tp = target.unsafe_ptr()
        var expected_location = expected_stats[0]
        var expected_scale = expected_stats[1]
        def normalized_cell(row: Int) {imm nfp, imm np, imm tp, imm expected_location, imm expected_scale}:
            var expected = (Float64(tp[row]) - expected_location) / expected_scale
            nfp.unsafe_store(row, Int(abs(Float64(np[row]) - expected) > Float64(2e-5)))
        host_parallelize(normalized_cell, rows)
        failures_clear(normalization_failures, "GP normalized targets independent element oracle")
        var status = gp.gpr_fit(GPRFitRequest(Int(source.unsafe_ptr()), Int(normalized.unsafe_ptr() + 1),
            Int(factor.unsafe_ptr() + 1), Int(dual.unsafe_ptr() + 1), Int(scalars.unsafe_ptr() + 1),
            rows, 3, kernel.view(), 0.00000095367431640625, False))
        require(status == 0 and scalars[1] == 0 and isfinite(scalars[5]), "GP normalized native fit succeeds")
        var result = gp.gpr_query(GPRQueryRequest(Int(source.unsafe_ptr()), Int(factor.unsafe_ptr() + 1),
            Int(dual.unsafe_ptr() + 1), Int(query.unsafe_ptr()), Int(mean.unsafe_ptr() + 1),
            Int(variance.unsafe_ptr() + 1), Int(stddev.unsafe_ptr() + 1), Int(clamped.unsafe_ptr() + 1),
            rows, 3, queries, kernel.view(), status, True, True, norm.stddev, norm.mean, GP_ROUTE_LIST, -1, -1, 0, -1))
        require(result.finite == 0 and result.range_status == 0, "GP normalized query native status")
        require(gp.gpr_covariance(GPRCovarianceRequest(Int(source.unsafe_ptr()), Int(factor.unsafe_ptr() + 1),
            Int(dual.unsafe_ptr() + 1), Int(query.unsafe_ptr()), Int(covariance_mean.unsafe_ptr() + 1),
            Int(covariance.unsafe_ptr() + 1), rows, 3, queries, kernel.view(), status, True, norm.stddev, norm.mean)) == queries,
            "GP normalized covariance row count")
        posterior_oracle(name, target, mean, variance, stddev, clamped, covariance, queries, result.value, False)
        var failures = List[Int](length=queries, fill=0)
        var fp = failures.unsafe_ptr()
        var sp = stddev.unsafe_ptr() + 1
        var cp = expected_covariance.unsafe_ptr()
        var mp = covariance_mean.unsafe_ptr() + 1
        var ep = expected_mean.unsafe_ptr()
        def prediction_cell(row: Int) {imm fp, imm sp, imm cp, imm mp, imm ep, imm queries}:
            var expected_std = sqrt(max(cp[row * queries + row], Float64(0)))
            var okay = abs(Float64(sp[row]) - expected_std) <= Float64(2e-3) + Float64(2e-3) * abs(expected_std)
            okay = okay and abs(Float64(mp[row]) - ep[row]) <= Float64(2e-3) + Float64(2e-4) * abs(ep[row])
            fp.unsafe_store(row, Int(not okay))
        host_parallelize(prediction_cell, queries)
        failures_clear(failures, "GP original normalized prediction tolerances and covariance mean")
        guards(normalized, "GP normalized target guards")
        guards(factor, "GP normalized fit factor guards")
        guards(dual, "GP normalized dual guards")
        guards64(scalars, "GP normalized fit scalar guards")
        guards(mean, "GP normalized mean guards")
        guards(variance, "GP normalized variance guards")
        guards(stddev, "GP normalized std guards")
        guards(covariance, "GP normalized covariance guards")
        guards(covariance_mean, "GP normalized covariance mean guards")
        require(clamped[0] == Int32(-31) and clamped[queries + 1] == Int32(-31), "GP normalized clamp guards")
        unchanged(source, source_before, "GP normalized borrowed source unchanged")
        unchanged(target, target_before, "GP normalized borrowed target unchanged")
        unchanged(query, query_before, "GP normalized borrowed query unchanged")
    finally:
        _ = source^
        _ = target^
        _ = query^
        _ = expected_stats^
        _ = expected_covariance^
        _ = expected_mean^
        _ = kernel^
        _ = normalized^
        _ = factor^
        _ = dual^
        _ = scalars^
        _ = mean^
        _ = variance^
        _ = stddev^
        _ = clamped^
        _ = covariance^
        _ = covariance_mean^


def analytic_case(scaler: ScalerDriver, program: PrepProgramDriver, unit: Int, direct: Bool) raises:
    var nan = bitcast[DType.float32](UInt32(0x7fc00021))
    var source: List[Float32] = [-3.0, -1.0, 1.0, 3.0]
    var expected_mean = Float64(0)
    var expected_scale = sqrt(Float64(5))
    if unit == 1:
        source = [7.0, 7.0, 7.0, 7.0]
        expected_mean = 7
        expected_scale = 1
    elif unit == 2:
        source = [-3.0, nan, 1.0, nan, 5.0]
        expected_mean = 1
        expected_scale = sqrt(Float64(32) / Float64(3))
    elif unit == 3:
        source = [nan, nan, nan]
    var before = source.copy()
    var destination = List[Float32](length=len(source) + 2, fill=GP_GUARD)
    try:
        var result = gp_normalize_targets(scaler, program,
            GPRNormalizeRequest(Int(source.unsafe_ptr()), Int(destination.unsafe_ptr() + 1), len(source), direct))
        if unit == 3:
            require(isnan(result.mean) and isnan(result.stddev), "GP all-missing statistics remain missing")
        else:
            require(abs(result.mean - expected_mean) < Float64(1e-6) and abs(result.stddev - expected_scale) < Float64(1e-6),
                "GP analytic finite/constant/observed-only population moments")
        var failures = List[Int](length=len(source), fill=0)
        var fp = failures.unsafe_ptr()
        var sp = source.unsafe_ptr()
        var bp = before.unsafe_ptr()
        var dp = destination.unsafe_ptr() + 1
        def cell(row: Int) {imm fp, imm sp, imm bp, imm dp, imm expected_mean, imm expected_scale}:
            var okay = bitcast[DType.uint32](sp[row]) == bitcast[DType.uint32](bp[row])
            if isnan(sp[row]):
                okay = okay and isnan(dp[row])
            else:
                var expected = (Float64(sp[row]) - expected_mean) / expected_scale
                okay = okay and abs(Float64(dp[row]) - expected) < Float64(1e-6)
            fp.unsafe_store(row, Int(not okay))
        host_parallelize(cell, len(source))
        failures_clear(failures, "GP missing positions restored and input words preserved")
        guards(destination, "GP analytic normalization guards")
    finally:
        _ = source^
        _ = before^
        _ = destination^


def refusal_case(scaler: ScalerDriver, program: PrepProgramDriver, unit: Int, direct: Bool) raises:
    var source: List[Float32] = [1.0, 2.0, 3.0]
    if unit >= 4:
        source[1] = bitcast[DType.float32](UInt32(0x7f800000) if unit == 4 else UInt32(0xff800000))
    var destination = List[Float32](length=5, fill=GP_GUARD)
    var rows = 0 if unit == 0 else (-1 if unit == 1 else 3)
    var source_address = 0 if unit == 2 else Int(source.unsafe_ptr())
    var destination_address = 0 if unit == 3 else Int(destination.unsafe_ptr() + 1)
    var expected = "Scaler input must contain at least one sample and feature" if unit < 2 else (
        "GP normalization: nonempty target spans require storage" if unit < 4 else "Scaler input contains infinity; only NaN may be missing")
    var refused = False
    try:
        _ = gp_normalize_targets(scaler, program, GPRNormalizeRequest(source_address, destination_address, rows, direct))
    except error:
        require(String(error) == expected, "GP normalization precise refusal")
        refused = True
    require(refused, "GP normalization invalid span/infinite value refused")
    require(destination[1] == GP_GUARD and destination[2] == GP_GUARD and destination[3] == GP_GUARD,
        "GP refused normalization leaves destination unpublished")
    guards(destination, "GP refused normalization guards")
    _ = source^
    _ = destination^


def normalization_fixture(gp: GPDriver, scaler: ScalerDriver, program: PrepProgramDriver, unit: Int, direct: Bool) raises:
    if unit < 3:
        normalized_graph(gp, scaler, program, unit, direct)
    elif unit < 7:
        analytic_case(scaler, program, unit - 3, direct)
    else:
        refusal_case(scaler, program, unit - 7, direct)
