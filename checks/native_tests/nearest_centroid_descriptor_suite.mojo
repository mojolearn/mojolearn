# SPDX-License-Identifier: Apache-2.0
"""Bounded fit/query plan fixtures using the selected real native descriptor.

Eight metric/split/shrink combinations and three empty-query contracts run
independently across host workers.
Closed-form symmetric-data oracles do not derive expected values from kernels.
"""
from std.math import sqrt
from core.host_parallel import host_parallelize
from core.nearest_centroid_plan import NcDriver, NcFitPlan, NcQueryPlan, NcLabelCodes, nc_binary_decision, nc_scratch_address
from core.native_descriptor_abi import NativeDescriptorCall
from checks.native_tests.nearest_centroid_plan_suite import require


def fixture(entry: NativeDescriptorCall, unit: Int) raises:
    var euclidean = unit % 2 == 0
    var split = (unit // 2) % 2 != 0
    var threshold = 0.0 if unit < 4 else 0.3
    var x: List[Float32] = [-3.0, -1.0, -3.0, 1.0, -1.0, -1.0, -1.0, 1.0,
                            1.0, -1.0, 1.0, 1.0, 3.0, -1.0, 3.0, 1.0]
    var labels: List[Int32] = [0, 0, 0, 0, 1, 1, 1, 1]
    var stats = List[Float32](length=2, fill=0)
    var centers = List[Float32](length=4, fill=0)
    var deviations = List[Float32](length=4, fill=0)
    var driver = NcDriver(entry)
    var plan = NcFitPlan(8, 2, 2, euclidean, split)
    require(len(plan.centroids) == 0 and len(plan.dataset_centroid) == 0, "centroid scratch allocated before priors")
    plan.count(driver, Int(labels.unsafe_ptr()))
    require(plan.counts[0] == 4 and plan.counts[1] == 4, "class counts")
    require(plan.wide_counts[0] == 4 and plan.wide_counts[1] == 4, "wide class counts")
    require(plan.statistics(driver, Int(x.unsafe_ptr()), Int(labels.unsafe_ptr()), Int(stats.unsafe_ptr())) == 0,
            "finite fixture status")
    var summary = plan.median_std(driver, Int(stats.unsafe_ptr()))
    require(not summary[1] and not plan.constant(driver, Int(x.unsafe_ptr())), "variance metadata")
    require(abs(summary[0] - sqrt(4.0 / 3.0)) < 2e-6, "pooled standard deviation")
    plan.shrink(driver, Int(x.unsafe_ptr()), Int(stats.unsafe_ptr()), Int(centers.unsafe_ptr()),
                Int(deviations.unsafe_ptr()), summary[0], threshold)
    var expected = 2.0 - threshold * sqrt(2.0 / 3.0)
    require(abs(Float64(centers[0]) + expected) < 2e-6 and abs(Float64(centers[2]) - expected) < 2e-6,
            "closed-form class centroid/shrinkage")
    require(centers[1] == 0 and centers[3] == 0, "zero-center feature")
    var q: List[Float32] = [-2.0, 0.0, 0.0, 0.0, 2.0, 0.0]
    var priors: List[Float32] = [0.5, 0.5]
    var request = NcQueryPlan(Int(q.unsafe_ptr()), Int(centers.unsafe_ptr()), Int(stats.unsafe_ptr()),
                             Int(priors.unsafe_ptr()), 3, 2, 2)
    var codes = NcLabelCodes(3)
    codes.values[0] = -1
    codes.values[1] = -1
    codes.values[2] = -1
    request.distance_labels(driver, codes.address(), euclidean)
    require(codes.values[0] == 0 and codes.values[1] == 0 and codes.values[2] == 1, "distance labels and first tie")
    require(request.fused(driver, "nc_labels", codes.address(), 0 if euclidean else 1) == 0,
            "fused distance finite status")
    require(codes.values[0] == 0 and codes.values[1] == 0 and codes.values[2] == 1, "fused distance labels")
    request.score_labels(driver, codes.address())
    require(codes.values[0] == 0 and codes.values[1] == 0 and codes.values[2] == 1, "score labels and first tie")
    var scores = List[Float32](length=6, fill=0)
    var differences = List[Float32](length=3, fill=0)
    request.decision(driver, Int(scores.unsafe_ptr()))
    nc_binary_decision(driver, Int(scores.unsafe_ptr()), Int(differences.unsafe_ptr()), 3)
    require(differences[0] < 0 and differences[1] == 0 and differences[2] > 0, "binary score orientation")
    request.probabilities(driver, Int(scores.unsafe_ptr()), 0)
    require(abs(Float64(scores[2]) - 0.5) < 1e-6 and abs(Float64(scores[3]) - 0.5) < 1e-6,
            "equal posterior at midpoint")
    require(scores[0] > scores[1] and scores[4] < scores[5], "posterior class orientation")
    require(request.fused(driver, "nc_proba", Int(scores.unsafe_ptr()), 1) == 0, "fused log probabilities")
    require(abs(Float64(scores[2]) + 0.6931471805599453) < 2e-6, "log posterior midpoint")
    _ = x^
    _ = labels^
    _ = q^
    _ = priors^
    _ = stats^
    _ = centers^
    _ = deviations^
    _ = plan^
    _ = codes^


def empty_case(entry: NativeDescriptorCall, unit: Int) raises:
    var words = List[Float32](length=4, fill=1)
    var codes = NcLabelCodes(1)
    codes.values[0] = -1
    var empty_codes = NcLabelCodes(0)
    require(empty_codes.address() == 0, "empty code owner must retain null Array address")
    var empty = List[Float32]()
    require(nc_scratch_address(empty) == 0, "empty scratch must retain null Array address")
    var address = Int(words.unsafe_ptr())
    # Input addresses remain non-null, as an empty NumPy view may be; the
    # private empty output scratch must retain the old descriptor refusal.
    var request = NcQueryPlan(address, address, address, address, 0, 2, 2)
    var driver = NcDriver(entry)
    var refused = False
    try:
        if unit == 0:
            request.distance_labels(driver, codes.address(), True)
        elif unit == 1:
            request.score_labels(driver, codes.address())
        else:
            request.probabilities(driver, address, 0)
    except error:
        refused = String(error) == "x_neighbors: null buffer address"
    require(refused and codes.values[0] == -1, "empty query changed refusal or wrote output")
    _ = words^
    _ = codes^
    _ = empty_codes^


def run_checks(entry: NativeDescriptorCall) raises:
    var failures = List[Int](length=11, fill=0)
    var fp = failures.unsafe_ptr()
    def run(unit: Int) {imm fp, imm entry}:
        try:
            if unit < 8:
                fixture(entry, unit)
            else:
                empty_case(entry, unit - 8)
        except error:
            fp[unit] = 1
            print("FAIL nearest_centroid_native fixture", unit, String(error))
    host_parallelize(run, 11)
    var total = 0
    for failed in failures:
        total += failed
    require(total == 0, "nearest_centroid_native failed " + String(total) + " fixtures")
    print("PASS nearest_centroid_native: 11 native fit/query/refusal fixtures")
