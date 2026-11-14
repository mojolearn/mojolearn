# SPDX-License-Identifier: Apache-2.0
"""NearestCentroid probabilities through the native descriptor's softmax.

Ported from `python/mojolearn/tests/test_nearest_centroid_native_capability.py`
(deleted): 36 four-feature rows in three classes (class c offset by +c, labels
`i % 3`), 11 queries, `shrink_threshold` None (0) and 0.2. The probabilities
(`NcQueryPlan.probabilities`, the descriptor's float32 `softmax` export) are
the float64 softmax of the decision scores within rtol 2e-5 / atol 2e-6, every
row sums to 1 within 2e-6, and the most probable class is the predicted one.
The rows are standard normal (Box-Muller over a SplitMix64 stream regenerated
here; the relations hold for any data). Priors are the class proportions
(1/3 each), as the public fit publishes them.
"""
from std.math import cos, exp, log, sqrt
from core.nearest_centroid_plan import NcDriver, NcFitPlan, NcQueryPlan, NcLabelCodes
from core.native_descriptor_abi import NativeDescriptorCall

comptime NC_CAPABILITY_FIXTURES = 2


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _next(mut state: UInt64) -> UInt64:
    """SplitMix64."""
    state += UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def normal_rows(seed: Int, count: Int) -> List[Float32]:
    var state = UInt64(seed) * UInt64(0x2545F4914F6CDD1D) + UInt64(1)
    var values = List[Float32](length=count, fill=0)
    for i in range(count):
        var u1 = (Float64(Int(_next(state) >> 11)) + 1.0) * (1.0 / 9007199254740992.0)
        var u2 = Float64(Int(_next(state) >> 11)) * (1.0 / 9007199254740992.0)
        values[i] = Float32(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2))
    return values^


def fixture(entry: NativeDescriptorCall, unit: Int) raises:
    comptime N = 36
    comptime F = 4
    comptime C = 3
    comptime Q = 11
    var threshold = 0.0 if unit == 0 else 0.2
    var x = normal_rows(518, N * F)
    var labels = List[Int32](length=N, fill=0)
    for i in range(N):
        labels[i] = Int32(i % C)
        for f in range(F):
            x[i * F + f] += Float32(i % C)
    var q = normal_rows(519, Q * F)
    var stats = List[Float32](length=F, fill=0)
    var centers = List[Float32](length=C * F, fill=0)
    var deviations = List[Float32](length=C * F, fill=0)
    var driver = NcDriver(entry)
    var plan = NcFitPlan(N, F, C, True, False)
    plan.count(driver, Int(labels.unsafe_ptr()))
    require(plan.statistics(driver, Int(x.unsafe_ptr()), Int(labels.unsafe_ptr()), Int(stats.unsafe_ptr())) == 0,
        "finite fit status")
    var summary = plan.median_std(driver, Int(stats.unsafe_ptr()))
    plan.shrink(driver, Int(x.unsafe_ptr()), Int(stats.unsafe_ptr()), Int(centers.unsafe_ptr()),
        Int(deviations.unsafe_ptr()), summary[0], threshold)
    var priors = List[Float32](length=C, fill=Float32(1.0 / 3.0))
    var request = NcQueryPlan(Int(q.unsafe_ptr()), Int(centers.unsafe_ptr()), Int(stats.unsafe_ptr()),
        Int(priors.unsafe_ptr()), Q, F, C)
    var scores = List[Float32](length=Q * C, fill=0)
    var probabilities = List[Float32](length=Q * C, fill=-1)
    request.decision(driver, Int(scores.unsafe_ptr()))
    request.probabilities(driver, Int(probabilities.unsafe_ptr()), 0)
    var codes = NcLabelCodes(Q)
    request.distance_labels(driver, codes.address(), True)
    var far = 0
    var unnormalized = 0
    var argmax_moved = 0
    for row in range(Q):
        var top = Float64(scores[row * C])
        for c in range(1, C):
            top = max(top, Float64(scores[row * C + c]))
        var total = 0.0
        for c in range(C):
            total += exp(Float64(scores[row * C + c]) - top)
        var row_sum = 0.0
        var best = 0
        for c in range(C):
            var expected = exp(Float64(scores[row * C + c]) - top) / total
            var got = Float64(probabilities[row * C + c])
            if abs(got - expected) > 2e-6 + 2e-5 * abs(expected):
                far += 1
            row_sum += got
            if probabilities[row * C + c] > probabilities[row * C + best]:
                best = c
        if abs(row_sum - 1.0) > 2e-6:
            unnormalized += 1
        if Int32(best) != codes.values[row]:
            argmax_moved += 1
    var tag = "shrink " + ("none" if unit == 0 else "0.2") + ": "
    require(far == 0, tag + String(far) + " probabilities differ from the softmax of the decision scores")
    require(unnormalized == 0, tag + String(unnormalized) + " probability rows do not sum to 1")
    require(argmax_moved == 0, tag + String(argmax_moved) + " rows' most probable class is not the prediction")
    _ = x^
    _ = labels^
    _ = q^
    _ = stats^
    _ = centers^
    _ = deviations^
    _ = plan^
    _ = priors^
    _ = scores^
    _ = probabilities^
    _ = codes^
