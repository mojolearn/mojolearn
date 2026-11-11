# SPDX-License-Identifier: Apache-2.0
"""Bounded fit/query plan fixtures using the real native CPU descriptor.

Eight independent metric/split/shrink combinations run across host workers.
Closed-form symmetric-data oracles do not derive expected values from kernels.
"""
from std.math import sqrt
from core.host_parallel import host_parallelize
from core.nearest_centroid_plan import NcDriver, NcFitPlan, NcQueryPlan, nc_binary_decision
from x_neighbors.native_descriptor_host import xn_native_v1
from checks.native_tests.nearest_centroid_plan_suite import require


def fixture(unit: Int) raises:
    var euclidean = unit % 2 == 0
    var split = (unit // 2) % 2 != 0
    var threshold = 0.0 if unit < 4 else 0.3
    var x: List[Float32] = [-3.0, -1.0, -3.0, 1.0, -1.0, -1.0, -1.0, 1.0,
                            1.0, -1.0, 1.0, 1.0, 3.0, -1.0, 3.0, 1.0]
    var labels: List[Int32] = [0, 0, 0, 0, 1, 1, 1, 1]
    var stats = List[Float32](length=2, fill=0)
    var centers = List[Float32](length=4, fill=0)
    var deviations = List[Float32](length=4, fill=0)
    var driver = NcDriver(xn_native_v1)
    var plan = NcFitPlan(8, 2, 2, euclidean, split)
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
    var codes = List[Int32](length=3, fill=-1)
    request.distance_labels(driver, Int(codes.unsafe_ptr()), euclidean)
    require(codes[0] == 0 and codes[1] == 0 and codes[2] == 1, "distance labels and first tie")
    require(request.fused(driver, "nc_labels", Int(codes.unsafe_ptr()), 0 if euclidean else 1) == 0,
            "fused distance finite status")
    require(codes[0] == 0 and codes[1] == 0 and codes[2] == 1, "fused distance labels")
    request.score_labels(driver, Int(codes.unsafe_ptr()))
    require(codes[0] == 0 and codes[1] == 0 and codes[2] == 1, "score labels and first tie")
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


def main() raises:
    var failures = List[Int](length=8, fill=0)
    var fp = failures.unsafe_ptr()
    def run(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp[unit] = 1
            print("FAIL nearest_centroid_native_cpu fixture", unit, String(error))
    host_parallelize(run, 8)
    var total = 0
    for failed in failures:
        total += failed
    require(total == 0, "nearest_centroid_native_cpu failed " + String(total) + " fixtures")
    print("PASS nearest_centroid_native_cpu: 8 native fit/query fixtures")
