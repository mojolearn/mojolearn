# SPDX-License-Identifier: Apache-2.0
"""Independent observed-label accuracy oracles over the actual COUNT program.

The six analytic graphs prescribe the sorted union and every remapped row.
The seventh graph reuses the original 48x3 binary GP workload and compares
the composed score with an independent parallel equality count. These are
supplementary composition checks; public label identity remains a Python seam.
"""
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorCall
from core.labels_score_keys import ScoreLabelKey, SCORE_INT, SCORE_BIGINT, SCORE_UTF8, SCORE_UCS4
from core.gp_native_score_labels import GPScorePair, gp_score_integer_labels, gp_score_key_labels, gp_score_pair
from core.gp_native_accuracy import GPCAccuracyProgram
from core.gp_native_api import GPDriver, GPCFitRequest, GPCQueryRequest
from checks.native_tests.gp_state_suite import require
from checks.native_tests.gp_native_data import input_values, input_labels, failures_clear, GP_GUARD, GP_GUARD64
from checks.native_tests.gp_native_classifier_cases import classifier_kernel

comptime GPC_SCORE_NATIVE_CASES = 14


def prescribed_codes(actual: List[Int32], expected: List[Int32]) raises:
    require(len(actual) == len(expected), "prescribed metric code extent")
    var failures = List[Int](length=len(actual), fill=0)
    var a = actual.unsafe_ptr()
    var e = expected.unsafe_ptr()
    var f = failures.unsafe_ptr()
    def cell(row: Int) {imm a, imm e, imm f}:
        f[row] = Int(a[row] != e[row])
    host_parallelize(cell, len(actual))
    failures_clear(failures, "independently prescribed observed-union codes")
    _ = actual
    _ = expected
    _ = failures


def numeric_pair(unit: Int) raises -> GPScorePair:
    var truth: List[Int64] = [-9, 2, -9, 50, 2, 17]
    var prediction: List[Int64] = [-9, 50, 2, 50, 17, 999]
    if unit == 1:
        truth = [7, 7, 7, 7]
        prediction = [7, 7, 7, 7]
    elif unit == 2:
        truth = [11, 11, 11, 11]
        prediction = [7, 7, 7, 7]
    elif unit == 3:
        truth = [-9223372036854775807 - 1, -4294967296, 0, 9223372036854775807,
            -9223372036854775807 - 1, 4294967296]
        prediction = [-9223372036854775807 - 1, 0, 0, 9223372036854775807, 4294967296, 4294967296]
    var true_codes = gp_score_integer_labels(Int(truth.unsafe_ptr()), len(truth))
    var pred_codes = gp_score_integer_labels(Int(prediction.unsafe_ptr()), len(prediction))
    var result = gp_score_pair(true_codes, pred_codes)
    # The result owns remapped rows after both source arrays and local class
    # encoders die. Mutating originals also catches accidental aliasing.
    truth[0] = 12345
    prediction[0] = -12345
    _ = truth
    _ = prediction
    _ = true_codes
    _ = pred_codes
    return result^


def text_pair() raises -> GPScorePair:
    var a = String("a")
    var z = String("z")
    var astral = String("😀")
    var points: List[UInt32] = [0x61, 0x7A, 0xD800, 0x1F600]
    var ka = ScoreLabelKey(SCORE_UTF8, 0, Int(a.unsafe_ptr()), a.byte_length(), False)
    var kz = ScoreLabelKey(SCORE_UTF8, 0, Int(z.unsafe_ptr()), z.byte_length(), False)
    var ke = ScoreLabelKey(SCORE_UTF8, 0, Int(astral.unsafe_ptr()), astral.byte_length(), False)
    var pa = ScoreLabelKey(SCORE_UCS4, 0, Int(points.unsafe_ptr()), 1, False)
    var pz = ScoreLabelKey(SCORE_UCS4, 0, Int(points.unsafe_ptr() + 1), 1, False)
    var ps = ScoreLabelKey(SCORE_UCS4, 0, Int(points.unsafe_ptr() + 2), 1, False)
    var pe = ScoreLabelKey(SCORE_UCS4, 0, Int(points.unsafe_ptr() + 3), 1, False)
    var truth: List[ScoreLabelKey] = [kz, ke, ka, pe, ps]
    var prediction: List[ScoreLabelKey] = [pz, pa, ka, ke, pe]
    var true_codes = gp_score_key_labels(truth)
    var pred_codes = gp_score_key_labels(prediction)
    var result = gp_score_pair(true_codes, pred_codes)
    # Key byte/point owners live through encoding and remapping. Returned
    # codes must remain independent after all borrowed text storage dies.
    _ = a
    _ = z
    _ = astral
    _ = points
    _ = truth
    _ = prediction
    _ = true_codes
    _ = pred_codes
    return result^


def bigint_pair() raises -> GPScorePair:
    var magnitude = String("40000000000000000000000")  # 2**90, outside signed Int64
    var big = ScoreLabelKey(SCORE_BIGINT, 0, Int(magnitude.unsafe_ptr()), magnitude.byte_length(), False)
    var negative = big.copy()
    negative.negative = True
    var zero = ScoreLabelKey(SCORE_INT, 0, 0, 0, False)
    var one = ScoreLabelKey(SCORE_INT, 1, 0, 0, False)
    var truth: List[ScoreLabelKey] = [big, negative, zero, big, one, negative]
    var prediction: List[ScoreLabelKey] = [big, zero, zero, one, one, negative]
    var true_codes = gp_score_key_labels(truth)
    var pred_codes = gp_score_key_labels(prediction)
    var result = gp_score_pair(true_codes, pred_codes)
    _ = magnitude
    _ = truth
    _ = prediction
    _ = true_codes
    _ = pred_codes
    return result^


def selected_pair(unit: Int) raises -> GPScorePair:
    if unit == 4:
        return text_pair()
    if unit == 5:
        return bigint_pair()
    return numeric_pair(unit)


def owned_program(unit: Int) raises -> GPCAccuracyProgram:
    var pair = selected_pair(unit)
    var expected_truth: List[Int32] = [0, 1, 0, 3, 1, 2]
    var expected_prediction: List[Int32] = [0, 3, 1, 3, 2, 4]
    var classes = 5
    if unit == 1:
        expected_truth = [0, 0, 0, 0]
        expected_prediction = [0, 0, 0, 0]
        classes = 1
    elif unit == 2:
        expected_truth = [1, 1, 1, 1]
        expected_prediction = [0, 0, 0, 0]
        classes = 2
    elif unit == 3:
        expected_truth = [0, 1, 2, 4, 0, 3]
        expected_prediction = [0, 2, 2, 4, 3, 3]
    elif unit == 4:
        expected_truth = [1, 3, 0, 3, 2]
        expected_prediction = [1, 0, 0, 3, 3]
        classes = 4
    elif unit == 5:
        expected_truth = [3, 0, 1, 3, 2, 0]
        expected_prediction = [3, 1, 1, 2, 2, 0]
        classes = 4
    require(pair.rows == len(expected_truth) and pair.classes == classes, "observed metric union shape")
    prescribed_codes(pair.truth, expected_truth)
    prescribed_codes(pair.prediction, expected_prediction)
    var result = GPCAccuracyProgram(pair.rows, pair.classes,
        Int(pair.truth.unsafe_ptr()), Int(pair.prediction.unsafe_ptr()))
    require(len(result.inputs) == 3 and result.inputs[0] == 0
        and result.inputs[1] == Int32(2 * pair.rows) and result.inputs[2] == -1,
        "COUNT input record retains both exact Int32 spans")
    require(len(result.outputs) > 4, "COUNT retains its complete group/count output ranges")
    pair.truth[0] = -77
    pair.prediction[0] = -88
    _ = pair
    _ = expected_truth
    _ = expected_prediction
    return result^


def analytic_case(metrics: NativeDescriptorCall, unit: Int, ranges: Bool) raises:
    var program = owned_program(unit)
    var address = Int(program.arena.unsafe_ptr())
    var moved = program^
    require(Int(moved.arena.unsafe_ptr()) == address, "whole COUNT program move retains native arena")
    var expected = Float64(1) / 3
    if unit == 1:
        expected = 1
    elif unit == 2:
        expected = 0
    elif unit == 3 or unit == 5:
        expected = Float64(2) / 3
    elif unit == 4:
        expected = Float64(3) / 5
    require(abs(moved.run(metrics, ranges) - expected) <= Float64(1e-12),
        "COUNT accuracy matches prescribed independent label equalities")
    _ = moved


def original_binary_case(driver: GPDriver, metrics: NativeDescriptorCall, ranges: Bool) raises:
    var source = input_values("classifier48", "source", 144)
    var query = input_values("classifier48", "query", 30)
    var codes = input_labels("classifier48", "labels2", 48)
    var kernel = classifier_kernel(False)
    var targets = List[Float32](length=50, fill=GP_GUARD)
    var factor = List[Float32](length=2306, fill=GP_GUARD)
    var pi = List[Float32](length=50, fill=GP_GUARD)
    var wsr = List[Float32](length=50, fill=GP_GUARD)
    var scalars = List[Float64](length=5, fill=GP_GUARD64)
    var mean = List[Float32](length=12, fill=GP_GUARD)
    var variance = List[Float32](length=12, fill=GP_GUARD)
    var probability = List[Float64](length=12, fill=GP_GUARD64)
    var prediction = List[Int64](length=12, fill=-31)
    var truth = List[Int64](length=10, fill=0)
    var hits = List[Int](length=10, fill=0)
    try:
        var iterations = driver.gpc_fit(GPCFitRequest(Int(source.unsafe_ptr()), Int(codes.unsafe_ptr()),
            Int(factor.unsafe_ptr() + 1), Int(pi.unsafe_ptr() + 1), Int(wsr.unsafe_ptr() + 1),
            Int(scalars.unsafe_ptr() + 1), 48, 3, kernel.view(), 100, True, 1, Int(targets.unsafe_ptr() + 1)))
        require(iterations >= 1 and iterations <= 100, "original GPC fit iteration range")
        require(driver.gpc_query(GPCQueryRequest(Int(source.unsafe_ptr()), Int(targets.unsafe_ptr() + 1),
            Int(pi.unsafe_ptr() + 1), Int(wsr.unsafe_ptr() + 1), Int(factor.unsafe_ptr() + 1),
            Int(query.unsafe_ptr()), Int(mean.unsafe_ptr() + 1), Int(variance.unsafe_ptr() + 1),
            Int(probability.unsafe_ptr() + 1), 48, 3, 10, kernel.view(), False, 1,
            Int(prediction.unsafe_ptr() + 1))) == 0, "original GPC prediction status")
        var cp = codes.unsafe_ptr()
        var pp = prediction.unsafe_ptr() + 1
        var tp = truth.unsafe_ptr()
        var hp = hits.unsafe_ptr()
        def equality(row: Int) {imm cp, imm pp, imm tp, imm hp}:
            tp[row] = Int64(cp[row])
            hp[row] = Int(pp[row] == Int64(cp[row]))
        host_parallelize(equality, 10)
        # Ten independent row predicates above; this is only oracle reduction.
        var matched = 0
        for hit in hits:
            matched += hit
        var actual_truth = gp_score_integer_labels(Int(truth.unsafe_ptr()), 10)
        var actual_prediction = gp_score_integer_labels(Int(prediction.unsafe_ptr() + 1), 10)
        var pair = gp_score_pair(actual_truth, actual_prediction)
        var program = GPCAccuracyProgram(10, pair.classes,
            Int(pair.truth.unsafe_ptr()), Int(pair.prediction.unsafe_ptr()))
        require(abs(program.run(metrics, ranges) - Float64(matched) / 10) <= Float64(1e-6),
            "original workload score equals independent prediction equality fraction")
        require(prediction[0] == -31 and prediction[11] == -31, "prediction output canaries")
        require(factor[0] == GP_GUARD and factor[2305] == GP_GUARD
            and targets[0] == GP_GUARD and targets[49] == GP_GUARD
            and pi[0] == GP_GUARD and pi[49] == GP_GUARD
            and wsr[0] == GP_GUARD and wsr[49] == GP_GUARD, "fit output canaries")
        require(scalars[0] == GP_GUARD64 and scalars[4] == GP_GUARD64
            and mean[0] == GP_GUARD and mean[11] == GP_GUARD
            and variance[0] == GP_GUARD and variance[11] == GP_GUARD
            and probability[0] == GP_GUARD64 and probability[11] == GP_GUARD64, "query and scalar canaries")
        _ = program
        _ = pair
        _ = actual_truth
        _ = actual_prediction
    finally:
        _ = source
        _ = query
        _ = codes
        _ = kernel
        _ = targets
        _ = factor
        _ = pi
        _ = wsr
        _ = scalars
        _ = mean
        _ = variance
        _ = probability
        _ = prediction
        _ = truth
        _ = hits


def empty_case() raises:
    var no_labels: List[ScoreLabelKey] = []
    var left = gp_score_key_labels(no_labels)
    var right = gp_score_key_labels(no_labels)
    var pair = gp_score_pair(left, right)
    require(pair.rows == 0 and pair.classes == 0 and len(pair.truth) == 0
        and len(pair.prediction) == 0, "empty native observed-label union owns empty codes")
    _ = no_labels
    _ = left
    _ = right
    _ = pair


def fixture(driver: GPDriver, metrics: NativeDescriptorCall, unit: Int) raises:
    if unit < 12:
        analytic_case(metrics, unit // 2, (unit & 1) != 0)
    else:
        original_binary_case(driver, metrics, (unit & 1) != 0)
