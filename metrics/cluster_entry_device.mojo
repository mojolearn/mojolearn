# SPDX-License-Identifier: Apache-2.0
"""The device column of the cluster label-pair metrics table
(core/metrics_cluster_table.mojo, lane mf-cluster5): the bodies of
bindings/_mojolearn_metrics.mojo's `rand_score`, `adjusted_rand_score`,
`mutual_info_score`, `homogeneity_score`, `completeness_score`,
`v_measure_score`, `fowlkes_mallows_score` and
`homogeneity_completeness_v_measure`, moved here unchanged (the same label
loads and the same metrics/estimator.mojo device entries, on the image's one
process-lifetime context) so the table entry and the binding's Python
exports run ONE body. Called with the GIL released; no Python.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import read_i32
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.metrics_cluster_table import (
    CLUSTER_OP_RAND, CLUSTER_OP_ADJUSTED_RAND, CLUSTER_OP_MUTUAL_INFO, CLUSTER_OP_HOMOGENEITY,
    CLUSTER_OP_COMPLETENESS, CLUSTER_OP_V_MEASURE, CLUSTER_OP_FOWLKES_MALLOWS, CLUSTER_OP_HCV,
    ClusterWordPtr, ClusterFloatPtr, cluster_ops_check_spans,
)
from metrics.estimator import (
    adjusted_rand_score_host, completeness_score_host, fowlkes_mallows_score_host, homogeneity_score_host,
    mutual_info_score_host, rand_score_host, v_measure_score_host, homogeneity_completeness_v_measure_host,
)


def metrics_cluster_device_run(operation: Int, yt_addr: Int, yp_addr: Int, n: Int, lower_word: Int,
                               upper_word: Int, beta: Float64, dst: ClusterFloatPtr) raises -> Int:
    """The device body of one operation (module note); answers the float64
    words written at `dst`."""
    var lower = Int32(lower_word)
    var upper = Int32(upper_word)
    # `_load_i32` (metrics/mojolearn_metrics_run.mojo): the labels copied once
    var yt = read_i32(yt_addr, max(0, n))
    var yp = read_i32(yp_addr, max(0, n))
    if operation == CLUSTER_OP_RAND:
        dst[0] = rand_score_host(yt, yp, n)
        return 1
    if operation == CLUSTER_OP_ADJUSTED_RAND:
        dst[0] = adjusted_rand_score_host(yt, yp, n)
        return 1
    if operation == CLUSTER_OP_MUTUAL_INFO:
        dst[0] = mutual_info_score_host(yt, yp, n, lower, upper)
        return 1
    if operation == CLUSTER_OP_FOWLKES_MALLOWS:
        dst[0] = fowlkes_mallows_score_host(yt, yp, n, lower, upper)
        return 1
    if operation == CLUSTER_OP_HOMOGENEITY:
        dst[0] = homogeneity_score_host(yt, yp, n, lower, upper)
        return 1
    if operation == CLUSTER_OP_COMPLETENESS:
        dst[0] = completeness_score_host(yt, yp, n, lower, upper)
        return 1
    if operation == CLUSTER_OP_V_MEASURE:
        dst[0] = v_measure_score_host(yt, yp, n, lower, upper, beta)
        return 1
    if operation == CLUSTER_OP_HCV:
        var h = Float64(0.0)
        var c = Float64(0.0)
        var v = Float64(0.0)
        homogeneity_completeness_v_measure_host(yt, yp, n, lower, upper, beta, h, c, v)
        dst[0] = h
        dst[1] = c
        dst[2] = v
        return 3
    raise Error("metrics cluster ops: unknown operation " + String(operation))


def metrics_cluster_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        cluster_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = ClusterWordPtr(unsafe_from_address=aa)
        var p = ClusterWordPtr(unsafe_from_address=pp)
        var f = ClusterFloatPtr(unsafe_from_address=ff)
        var r = ClusterWordPtr(unsafe_from_address=result)
        r[] = metrics_cluster_device_run(operation, a[0], a[1], p[0], p[1], p[2], f[0],
                                         ClusterFloatPtr(unsafe_from_address=a[2]))
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct MetricsClusterDeviceAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(metrics_cluster_device_v1)


comptime _OPS_SLOT = ("MojolearnMetricsClusterDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                      else "MojolearnMetricsClusterDeviceFastV1")
# The publisher identity of this image's table (its entry list's address).
comptime METRICS_CLUSTER_DEVICE_API = SharedSlot[MetricsClusterDeviceAPI, _OPS_SLOT]
