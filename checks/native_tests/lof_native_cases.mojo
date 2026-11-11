# SPDX-License-Identifier: Apache-2.0
"""Actual LOF provider graphs with independent mathematical references.

Original 60x5/25x5 seeded inputs are preserved. Saved Float64 NumPy full-sort
references use the mathematical density/LOF definitions. Three 4x2 cases also
use hand-derived rational distances/densities/scores across all routes.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.neighbors_native_driver import XnDriver
from core.lof_plan import LofPlan, LofLabelsPlan, lof_offset, lof_f32_address
from checks.native_tests.pca_native_cases import require, unchanged

comptime LOF_NATIVE_FIXTURES = 20


def source_values(name: String, count: Int) raises -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/lof/" + name + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 4 * count, "LOF source extent")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def cell(index: Int) {imm bp, imm rp}:
            var at = 4 * index
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            rp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(cell, count)
        _ = bytes^
    return result^


def reference_values(name: String, count: Int) raises -> List[Float64]:
    var result = List[Float64](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/lof/" + name + ".f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 8 * count, "LOF reference extent")
        var bp = bytes.unsafe_ptr()
        var rp = result.unsafe_ptr()
        def cell(index: Int) {imm bp, imm rp}:
            var bits = UInt64(0)
            for byte in range(8):  # Fixed-width file decoding, not data work.
                bits |= UInt64(bp[index * 8 + byte]) << UInt64(byte * 8)
            rp.unsafe_store(index, bitcast[DType.float64](bits))
        host_parallelize(cell, count)
        _ = bytes^
    return result^


def close_values(actual: List[Float32], expected: List[Float64], name: String) raises:
    var count = len(expected)
    require(len(actual) == count, name + " extent")
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = actual.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm ep}:
        fp.unsafe_store(index, Int(not (abs(Float64(ap[index]) - ep[index])
            <= Float64(2e-6) + Float64(2e-6) * abs(ep[index]))))
    host_parallelize(cell, count)
    for failed in failures:
        require(failed == 0, name + " independent oracle tolerance")
    _ = actual
    _ = expected


def labels(driver: XnDriver, address: Int, expected_scores: List[Float64],
           expected_threshold: Float64, native_threshold: Float64, mode: Int) raises:
    var count = len(expected_scores)
    var plan = LofLabelsPlan(count)
    var result = List[Int64](length=count + 2, fill=Int64(0x123456789))
    try:
        plan.classify(driver, address, mode, native_threshold)
        plan.write_i64(Int(result.unsafe_ptr()) + 8)
        require(result[0] == Int64(0x123456789) and result[count + 1] == Int64(0x123456789), "LOF label output guards")
        var failures = List[Int](length=count, fill=0)
        var fp = failures.unsafe_ptr()
        var rp = result.unsafe_ptr() + 1
        var ep = expected_scores.unsafe_ptr()
        def cell(index: Int) {imm fp, imm rp, imm ep, imm expected_threshold}:
            fp.unsafe_store(index, Int(rp[index] != Int64(-1 if ep[index] < expected_threshold else 1)))
        host_parallelize(cell, count)
        for failed in failures:
            require(failed == 0, "LOF labels agree with independent score threshold")
    finally:
        _ = result^
        _ = plan^
        _ = expected_scores


def decision_reference(scores: List[Float64], offset: Float64) -> List[Float64]:
    var count = len(scores)
    var result = List[Float64](unsafe_uninit_length=count)
    var rp = result.unsafe_ptr()
    var sp = scores.unsafe_ptr()
    def cell(index: Int) {imm rp, imm sp, imm offset}:
        rp.unsafe_store(index, sp[index] - offset)
    host_parallelize(cell, count)
    _ = scores
    return result^


def original_case(driver: XnDriver, unit: Int) raises:
    var novelty = unit >= 12
    var neighbors = 6 if novelty else (1 if unit < 4 else (7 if unit < 8 else 59))
    var contamination_index = unit % 4
    var contamination = Float64(0.1 if contamination_index == 1 else (0.25 if contamination_index == 2 else 0.5))
    var automatic = contamination_index == 0
    if novelty:
        automatic = unit == 12
        contamination = 0.2
    var route = unit % 3
    var source = source_values("fit", 300)
    var original = source.copy()
    var fitted = LofPlan(60, 60, 5, neighbors, True, route == 2, route == 1)
    var prefix = String("fit-k") + String(neighbors)
    var reference_distances = reference_values(prefix + "-distances", 60 * neighbors)
    var reference_density = reference_values(prefix + "-density", 60)
    var reference_scores = reference_values(prefix + "-scores", 60)
    var reference_offsets = reference_values("query-offsets" if novelty else prefix + "-offsets", 2 if novelty else 4)
    try:
        require(fitted.query_neighbors(driver, Int(source.unsafe_ptr()), Int(source.unsafe_ptr())) == 0, "LOF finite fit admission")
        fitted.query_density(driver, lof_f32_address(fitted.distances), 60)
        fitted.query_score(driver, lof_f32_address(fitted.densities), 60)
        close_values(fitted.distances, reference_distances, "LOF original neighbor distances")
        close_values(fitted.densities, reference_density, "LOF original reachability density")
        close_values(fitted.scores, reference_scores, "LOF original outlier scores")
        var offset = Float64(-1.5)
        if not automatic:
            offset = lof_offset(driver, lof_f32_address(fitted.scores), 60, contamination)
        var expected_offset = reference_offsets[unit - 12 if novelty else contamination_index]
        require(abs(offset - expected_offset) <= Float64(2e-6) + Float64(2e-6) * abs(expected_offset), "LOF independent percentile offset")
        if not novelty:
            labels(driver, lof_f32_address(fitted.scores), reference_scores, expected_offset, offset, 0)
        else:
            var query = source_values("query", 125)
            var before = query.copy()
            var request = LofPlan(25, 60, 5, 6, False, route == 2, route == 1)
            var query_distances = reference_values("query-distances", 150)
            var query_density = reference_values("query-density", 25)
            var query_scores = reference_values("query-scores", 25)
            try:
                require(request.query_neighbors(driver, Int(query.unsafe_ptr()), Int(source.unsafe_ptr())) == 0, "LOF finite novelty admission")
                request.query_density(driver, lof_f32_address(fitted.distances), 60)
                request.query_score(driver, lof_f32_address(fitted.densities), 60)
                request.query_decision(driver, -offset)
                close_values(request.distances, query_distances, "LOF original novelty distances")
                close_values(request.densities, query_density, "LOF original novelty densities")
                close_values(request.scores, query_scores, "LOF original novelty scores")
                var expected_decision = decision_reference(query_scores, expected_offset)
                close_values(request.decisions, expected_decision, "LOF original novelty decisions")
                labels(driver, lof_f32_address(request.decisions), query_scores, expected_offset, 0, 1)
                unchanged(query, before, "LOF query input")
            finally:
                _ = query^
                _ = request^
        unchanged(source, original, "LOF fit source")
    finally:
        _ = fitted^
        _ = source^


def analytic_case(driver: XnDriver, route: Int) raises:
    # Exactly the public mutation fixture. Distances and rational LOF values
    # follow from points at 0,1,2,9 on a line; no product output is an oracle.
    var source: List[Float32] = [0, 0, 1, 0, 2, 0, 9, 0]
    var query: List[Float32] = [0.5, 0, 8, 0]
    var fitted = LofPlan(4, 4, 2, 2, True, route == 2, route == 1)
    var request = LofPlan(2, 4, 2, 2, False, route == 2, route == 1)
    var distances: List[Float64] = [1, 2, 1, 1, 1, 2, 7, 8]
    var densities: List[Float64] = [2.0 / 3.0, 0.5, 2.0 / 3.0, 2.0 / 15.0]
    var scores: List[Float64] = [-0.875, -4.0 / 3.0, -0.875, -4.375]
    var query_scores: List[Float64] = [-0.875, -2.8]
    var query_distances: List[Float64] = [0.5, 0.5, 1, 6]
    var query_densities: List[Float64] = [2.0 / 3.0, 1.0 / 7.0]
    try:
        require(fitted.query_neighbors(driver, Int(source.unsafe_ptr()), Int(source.unsafe_ptr())) == 0, "LOF analytic fit status")
        fitted.query_density(driver, lof_f32_address(fitted.distances), 4)
        fitted.query_score(driver, lof_f32_address(fitted.densities), 4)
        close_values(fitted.distances, distances, "LOF analytic distances")
        close_values(fitted.densities, densities, "LOF analytic densities")
        close_values(fitted.scores, scores, "LOF analytic scores")
        var offset = lof_offset(driver, lof_f32_address(fitted.scores), 4, 0.25)
        require(abs(offset + Float64(67.0 / 32.0)) <= Float64(2e-6), "LOF analytic quartile")
        labels(driver, lof_f32_address(fitted.scores), scores, -67.0 / 32.0, offset, 0)
        require(request.query_neighbors(driver, Int(query.unsafe_ptr()), Int(source.unsafe_ptr())) == 0, "LOF analytic query status")
        request.query_density(driver, lof_f32_address(fitted.distances), 4)
        request.query_score(driver, lof_f32_address(fitted.densities), 4)
        request.query_decision(driver, -offset)
        close_values(request.distances, query_distances, "LOF analytic query distances")
        close_values(request.densities, query_densities, "LOF analytic query densities")
        close_values(request.scores, query_scores, "LOF analytic query scores")
        var decisions = decision_reference(query_scores, -67.0 / 32.0)
        close_values(request.decisions, decisions, "LOF analytic query decisions")
        labels(driver, lof_f32_address(request.decisions), query_scores, -67.0 / 32.0, 0, 1)
    finally:
        _ = request^
        _ = fitted^
        _ = source^
        _ = query^


def finite_case(driver: XnDriver, route: Int) raises:
    var source: List[Float32] = [0, 0, 1, 0, 2, 0, 9, 0]
    source[0] = bitcast[DType.float32](UInt32(0x7FC00042))
    var before = source.copy()
    var plan = LofPlan(4, 4, 2, 2, True, route == 2, route == 1)
    try:
        require(plan.query_neighbors(driver, Int(source.unsafe_ptr()), Int(source.unsafe_ptr())) == 1, "LOF deferred NaN status across route")
        unchanged(source, before, "LOF nonfinite source")
    finally:
        _ = plan^
        _ = source^


def fixture(driver: XnDriver, unit: Int) raises:
    if unit < 14:
        original_case(driver, unit)
    elif unit < 17:
        analytic_case(driver, unit - 14)
    else:
        finite_case(driver, unit - 17)
