# SPDX-License-Identifier: Apache-2.0
"""N07 real forest streamed-replica route, unchanged existing fixed-point bins.
No GPU timing. Independent histogram tally, small-replica fallback and actual
weighted fit/zero-weight/failure contracts precede any speed qualification.
"""
from max.gpu.host import DeviceContext
from ensemble.checks.rf_perf_candidates_check import Fixture, run_histogram_arm, expected_histogram
from ensemble.checks.sample_weight_check import arm_a_zero_weight_drop, arm_b_validation, arm_c_double_counting, arm_d_weighted_bins_train
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import IDN_RF_STREAM_REPLICAS, forest_stream_replica_count


def main() raises:
    var ctx = DeviceContext()
    var fx = Fixture(ctx)
    var expected = expected_histogram(fx)
    var control = run_histogram_arm[TILE=4](ctx, fx, False, False)
    var candidate = run_histogram_arm[TILE=4, SMEM_COPIES=4](ctx, fx, False, False)
    var fallback = run_histogram_arm[SMEM_COPIES=4, SMEM_SLOTS=48](ctx, fx, False, False)
    for cell in range(len(expected)):
        if control[cell] != expected[cell] or candidate[cell] != expected[cell] or fallback[cell] != expected[cell]:
            raise Error("N07 independent exact histogram mismatch cell" + String(cell))
    var before = forest_stream_replica_count()
    var fails = arm_a_zero_weight_drop(ctx)
    fails += arm_b_validation(ctx)
    fails += arm_c_double_counting(ctx)
    fails += arm_d_weighted_bins_train(ctx)
    var after = forest_stream_replica_count()
    if fails != 0:
        raise Error("N07 weighted forest contract failed")
    comptime if IDN_RF_STREAM_REPLICAS:
        if after <= before:
            raise Error("N07 actual production streamed-replica route not reached")
    else:
        if before != 0 or after != 0:
            raise Error("N07 baseline unexpectedly enabled stream route")
    _ = fx^
    print("N07_PRODUCTION_PASS exact_histograms weighted_fits failure_contract route_hits", after - before)
