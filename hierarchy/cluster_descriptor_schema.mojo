# SPDX-License-Identifier: Apache-2.0
# Private transport to the existing x_cluster agglo_entry numerical owner.
# All spans are borrowed for one call; no Python object enters this ABI.
comptime HIERARCHY_CLUSTER_ABI = "mojolearn.hierarchy.cluster.v1"
comptime HIERARCHY_CLUSTER_OPERATIONS = 3
comptime HIERARCHY_CLUSTER_FIT = 0
comptime HIERARCHY_CLUSTER_MODE = 1
comptime HIERARCHY_CLUSTER_CAPABILITIES = 2
comptime HIERARCHY_CLUSTER_CAP_FIT = 1
comptime HIERARCHY_CLUSTER_CAP_DEVICE = 2


def hierarchy_cluster_validate_call(operation: Int, na: Int, np: Int, nf: Int,
                                    result: Int) raises:
    if operation < 0 or operation >= HIERARCHY_CLUSTER_OPERATIONS:
        raise Error("hierarchy cluster provider: unknown operation")
    if result == 0:
        raise Error("hierarchy cluster provider: null result")
    if operation == HIERARCHY_CLUSTER_FIT:
        # A: f32 X, f32 connectivity, i32 children, i32 labels, f32 distances,
        #    f64 scalars[2] (connected components, output clusters).
        # I: X count, connectivity count, n, d, linkage, metric, n_edges,
        #    n_merges, edge_mode, cut, use_threshold. F: Minkowski p, threshold.
        if na != 6 or np != 11 or nf != 2:
            raise Error("hierarchy cluster provider: fit requires six buffers, eleven integers and two scalars")
    elif na != 0 or np != 0 or nf != 0:
        raise Error("hierarchy cluster provider: metadata takes no operands")


def hierarchy_cluster_validate_buffers(a: List[Int], p: List[Int]) raises:
    var n = p[2]
    var d = p[3]
    var edges = p[6]
    var merges = p[7]
    if n < 2 or d < 0 or merges < 0 or merges >= n:
        raise Error("hierarchy cluster provider: invalid tree dimensions")
    # Byte-span overflow guards, not workload routing thresholds. The existing
    # algorithm owns an n*n float32 distance matrix on either execution column.
    if n > (9223372036854775807 // 4) // n or (d > 0 and n > (9223372036854775807 // 4) // d):
        raise Error("hierarchy cluster provider: matrix byte span overflows native size")
    if p[0] != n * d or p[1] < 0 or p[1] > 9223372036854775807 // 4:
        raise Error("hierarchy cluster provider: inconsistent input spans")
    if p[4] < 0 or p[4] > 3 or p[5] < -1 or p[5] > 5:
        raise Error("hierarchy cluster provider: invalid linkage or metric")
    if p[5] == 5 and n != d:
        raise Error("hierarchy cluster provider: precomputed distances must be square")
    if p[8] < 0 or p[8] > 2 or p[9] < 0 or p[9] > n or (p[10] != 0 and p[10] != 1):
        raise Error("hierarchy cluster provider: invalid connectivity or cut mode")
    if edges < 0:
        if edges != -1 or p[1] != 0:
            raise Error("hierarchy cluster provider: unconstrained tree has connectivity storage")
    elif p[8] == 1:
        if edges != n * n or p[1] != edges:
            raise Error("hierarchy cluster provider: invalid dense connectivity span")
    else:
        var width = 3 if p[8] == 2 else 2
        if edges > (9223372036854775807 // 4) // width or p[1] != edges * width:
            raise Error("hierarchy cluster provider: invalid sparse connectivity span")
    if (p[0] > 0 and a[0] == 0) or (p[1] > 0 and a[1] == 0):
        raise Error("hierarchy cluster provider: null input buffer")
    if a[3] == 0 or a[5] == 0 or (merges > 0 and (a[2] == 0 or a[4] == 0)):
        raise Error("hierarchy cluster provider: null output buffer")
