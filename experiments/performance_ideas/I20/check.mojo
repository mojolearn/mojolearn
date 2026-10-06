# SPDX-License-Identifier: Apache-2.0
"""Actual query/reference tiles and fused versus materialized log-kernel
paths share the original logical fold. All six kernels x three supported
unexpanded metrics x extreme bandwidths x weighted/unweighted samples.
A separate current dispatch/host-fold gate covers versioned chunk LSE."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from kde.checks.kde_check import _train_fixture, _query_fixture, _weight_fixture, _scores_by_path, _all_kernels, check_kde_tiled_equals_staged
from kde.impl.neighbors.kernel_density import DIST_L2_SQRT_UNEXPANDED, DIST_L1, DIST_LINF

def main() raises:
    var ctx = DeviceContext()
    var train = _train_fixture(131,7,29)
    var query = _query_fixture(train,131,9,7,37)
    var weights = _weight_fixture(131,41)
    var empty = List[Float32]()
    var cases = 0
    for h in [Float32(0.0001),Float32(0.25),Float32(10000)]:
        for metric in [DIST_L2_SQRT_UNEXPANDED,DIST_L1,DIST_LINF]:
            for kernel in _all_kernels():
                for weighted in range(2):
                    var w = weights.copy() if weighted!=0 else empty.copy()
                    var reference = _scores_by_path(ctx,train,query,w,weighted!=0,131,9,7,h,kernel,metric,0,0,0)
                    for path in [2,4,5]:
                        var actual = _scores_by_path(ctx,train,query,w,weighted!=0,131,9,7,h,kernel,metric,path,32,67,16,128)
                        for i in range(9):
                            if bitcast[DType.uint32](actual[i])!=bitcast[DType.uint32](reference[i]):
                                raise Error("I20 tiled LSE changed density bits")
                    cases+=1
    check_kde_tiled_equals_staged()
    print("I20 PASS bandwidth_metric_kernel_weight_cases=",cases,"independent_schedules=3")
