# SPDX-License-Identifier: Apache-2.0
"""Original GP theta precision/FTZ domain through the typed provider.

The adjacent binary64 inputs deliberately differ. Their agreement checks
Float32 argument admission, not repeated fits or cross-vendor identity.
Independent stdlib exp references preserve the original two-ULP bound.
"""
from std.memory import bitcast
from std.math import isfinite
from core.host_parallel import host_parallelize
from core.gp_native_api import GPDriver, GPThetaRequest
from checks.native_tests.gp_state_suite import require
from checks.native_tests.gp_native_data import reference_values, guards64, failures_clear, GP_GUARD64

comptime GP_THETA_FIXTURES = 2
comptime GP_THETA_WORDS = 4097


def source_words(values: List[Float64], original: List[Float64]) raises:
    require(len(values) == len(original), "GP theta source extent remains unchanged")
    var failures = List[Int](length=len(values), fill=0)
    var fp = failures.unsafe_ptr()
    var vp = values.unsafe_ptr()
    var op = original.unsafe_ptr()
    def word(index: Int) {imm fp, imm vp, imm op}:
        fp.unsafe_store(index, Int(bitcast[DType.uint64](vp[index]) != bitcast[DType.uint64](op[index])))
    host_parallelize(word, len(values))
    failures_clear(failures, "GP theta borrowed binary64 words unchanged")
    _ = values
    _ = original


def grid_case(driver: GPDriver) raises:
    var theta = reference_values("theta_policy", "base", GP_THETA_WORDS)
    var lower = reference_values("theta_policy", "lower", GP_THETA_WORDS)
    var upper = reference_values("theta_policy", "upper", GP_THETA_WORDS)
    var independent_exp = reference_values("theta_policy", "exp", GP_THETA_WORDS)
    var theta_before = theta.copy()
    var lower_before = lower.copy()
    var upper_before = upper.copy()
    var actual = List[Float64](length=GP_THETA_WORDS + 2, fill=GP_GUARD64)
    var below = List[Float64](length=GP_THETA_WORDS + 2, fill=GP_GUARD64)
    var above = List[Float64](length=GP_THETA_WORDS + 2, fill=GP_GUARD64)
    try:
        require(theta[0] == -64.0 and theta[2048] == 0.0 and theta[4096] == 64.0,
            "GP original theta grid endpoints and center")
        require(bitcast[DType.uint64](lower[2048]) == UInt64(0x8000000000000001)
            and bitcast[DType.uint64](upper[2048]) == UInt64(1),
            "GP original signed binary64 neighbors of zero remain present")
        require(driver.theta(GPThetaRequest(Int(theta.unsafe_ptr()), Int(actual.unsafe_ptr() + 1), GP_THETA_WORDS)) == GP_THETA_WORDS,
            "GP theta original4097 output count")
        require(driver.theta(GPThetaRequest(Int(lower.unsafe_ptr()), Int(below.unsafe_ptr() + 1), GP_THETA_WORDS)) == GP_THETA_WORDS,
            "GP theta lower-adjacent output count")
        require(driver.theta(GPThetaRequest(Int(upper.unsafe_ptr()), Int(above.unsafe_ptr() + 1), GP_THETA_WORDS)) == GP_THETA_WORDS,
            "GP theta upper-adjacent output count")
        var failures = List[Int](length=GP_THETA_WORDS, fill=0)
        var fp = failures.unsafe_ptr()
        var ap = actual.unsafe_ptr() + 1
        var bp = below.unsafe_ptr() + 1
        var up = above.unsafe_ptr() + 1
        var ep = independent_exp.unsafe_ptr()
        def cell(index: Int) {imm fp, imm ap, imm bp, imm up, imm ep}:
            var word = bitcast[DType.uint32](Float32(ap[index]))
            var reference = bitcast[DType.uint32](Float32(ep[index]))
            var okay = isfinite(ap[index]) and ap[index] > Float64(0)
            okay = okay and abs(Int(word) - Int(reference)) <= 2
            okay = okay and bitcast[DType.uint32](Float32(bp[index])) == word
            okay = okay and bitcast[DType.uint32](Float32(up[index])) == word
            fp.unsafe_store(index, Int(not okay))
        host_parallelize(cell, GP_THETA_WORDS)
        failures_clear(failures, "GP original finite-positive, stdlib-exp2ULP and adjacent-input Float32 precision contracts")
        guards64(actual, "GP theta base output guards")
        guards64(below, "GP theta lower output guards")
        guards64(above, "GP theta upper output guards")
        source_words(theta, theta_before)
        source_words(lower, lower_before)
        source_words(upper, upper_before)
    finally:
        _ = theta^
        _ = lower^
        _ = upper^
        _ = independent_exp^
        _ = theta_before^
        _ = lower_before^
        _ = upper_before^
        _ = actual^
        _ = below^
        _ = above^


def ftz_case(driver: GPDriver) raises:
    var theta: List[Float64] = [-100.0, -90.0, -88.0, 0.0]
    var original = theta.copy()
    var expected: List[Float64] = [0.0, 0.0, 0.0, 1.0]
    var actual = List[Float64](length=6, fill=GP_GUARD64)
    try:
        require(driver.theta(GPThetaRequest(Int(theta.unsafe_ptr()), Int(actual.unsafe_ptr() + 1), 4)) == 4,
            "GP original FTZ output count")
        var failures = List[Int](length=4, fill=0)
        var fp = failures.unsafe_ptr()
        var ap = actual.unsafe_ptr() + 1
        var ep = expected.unsafe_ptr()
        def cell(index: Int) {imm fp, imm ap, imm ep}:
            # Preserve the original list equality (numeric zero, not a newly
            # imposed signed-zero bit convention).
            fp.unsafe_store(index, Int(ap[index] != ep[index]))
        host_parallelize(cell, 4)
        failures_clear(failures, "GP original underflow/subnormal FTZ and expzero values")
        guards64(actual, "GP theta FTZ output guards")
        source_words(theta, original)
    finally:
        _ = theta^
        _ = original^
        _ = expected^
        _ = actual^


def theta_fixture(driver: GPDriver, unit: Int) raises:
    if unit == 0:
        grid_case(driver)
    else:
        ftz_case(driver)
