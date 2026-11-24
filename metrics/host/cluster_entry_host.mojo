# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the cluster label-pair metrics table
(core/metrics_cluster_table.mojo, lane mf-cluster5): the host bodies of
bindings/_mojolearn_metrics_host.mojo's `rand_score`, `adjusted_rand_score`,
`mutual_info_score`, `homogeneity_score`, `completeness_score`,
`v_measure_score`, `fowlkes_mallows_score` and
`homogeneity_completeness_v_measure`, moved here unchanged (the same reads,
checks, refusal words and oracle calls, in their order) so the table entry
and the binding's Python exports run ONE body. Called with the GIL released;
no Python.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import i32_ptr, read_i32
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.metrics_cluster_table import (
    CLUSTER_OP_RAND, CLUSTER_OP_ADJUSTED_RAND, CLUSTER_OP_MUTUAL_INFO, CLUSTER_OP_HOMOGENEITY,
    CLUSTER_OP_COMPLETENESS, CLUSTER_OP_V_MEASURE, CLUSTER_OP_FOWLKES_MALLOWS, CLUSTER_OP_HCV,
    ClusterWordPtr, ClusterFloatPtr, cluster_ops_check_spans,
)
from metrics.host.metrics_oracle import (
    host_adjusted_rand_score, host_fowlkes_mallows, host_homogeneity_score, host_mutual_info_ptr, host_v_measure,
)
from metrics.host.classification_oracle import host_rand_score
from metrics.label_pair_admission import metrics_check_pair as _check_pair, metrics_check_range as _check_range
from metrics.mojolearn_metrics_host_run import homogeneity_completeness_v_measure_run


def metrics_cluster_host_run(operation: Int, yt_addr: Int, yp_addr: Int, n: Int, lower_word: Int, upper_word: Int,
                             beta: Float64, dst: ClusterFloatPtr) raises -> Int:
    """The host body of one operation (module note); answers the float64
    words written at `dst`."""
    var lower = Int32(lower_word)
    var upper = Int32(upper_word)
    if operation == CLUSTER_OP_RAND:
        var yt = read_i32(yt_addr, max(0, n))
        var yp = read_i32(yp_addr, max(0, n))
        _check_pair(yt, yp, n)
        dst[0] = host_rand_score(yt, yp, n)
        return 1
    if operation == CLUSTER_OP_ADJUSTED_RAND:
        var yt = read_i32(yt_addr, n)
        var yp = read_i32(yp_addr, n)
        _check_pair(yt, yp, n)
        dst[0] = host_adjusted_rand_score(yt, yp, n)
        return 1
    if operation == CLUSTER_OP_MUTUAL_INFO:
        if n <= 0:
            raise Error("metrics: n must be positive, got " + String(n))
        _check_range(lower, upper)
        dst[0] = host_mutual_info_ptr(i32_ptr(yt_addr), i32_ptr(yp_addr), n, lower, upper)
        return 1
    var yt = read_i32(yt_addr, n)
    var yp = read_i32(yp_addr, n)
    if operation == CLUSTER_OP_FOWLKES_MALLOWS:
        # the body admits its inputs, as the device body does (lane r3-small)
        dst[0] = host_fowlkes_mallows(yt, yp, n, lower, upper)
        return 1
    if operation == CLUSTER_OP_HCV:
        var hcv = homogeneity_completeness_v_measure_run(yt, yp, n, lower, upper, beta)
        dst[0] = hcv[0]
        dst[1] = hcv[1]
        dst[2] = hcv[2]
        return 3
    _check_pair(yt, yp, n)
    _check_range(lower, upper)
    if operation == CLUSTER_OP_HOMOGENEITY:
        dst[0] = host_homogeneity_score(yt, yp, n, lower, upper)
    elif operation == CLUSTER_OP_COMPLETENESS:
        dst[0] = host_homogeneity_score(yp, yt, n, lower, upper)
    elif operation == CLUSTER_OP_V_MEASURE:
        dst[0] = host_v_measure(yt, yp, n, lower, upper, beta)
    else:
        raise Error("metrics cluster ops: unknown operation " + String(operation))
    return 1


def metrics_cluster_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        cluster_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = ClusterWordPtr(unsafe_from_address=aa)
        var p = ClusterWordPtr(unsafe_from_address=pp)
        var f = ClusterFloatPtr(unsafe_from_address=ff)
        var r = ClusterWordPtr(unsafe_from_address=result)
        r[] = metrics_cluster_host_run(operation, a[0], a[1], p[0], p[1], p[2], f[0],
                                       ClusterFloatPtr(unsafe_from_address=a[2]))
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct MetricsClusterHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(metrics_cluster_host_v1)


comptime _OPS_SLOT = ("MojolearnMetricsClusterHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                      else "MojolearnMetricsClusterHostFastV1")
# The publisher identity of this image's table (its entry list's address).
comptime METRICS_CLUSTER_HOST_API = SharedSlot[MetricsClusterHostAPI, _OPS_SLOT]
