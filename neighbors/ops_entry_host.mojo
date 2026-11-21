# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the neighbors operations table
(core/neighbors_entry_table.mojo): the host k-NN classifier and regressor
bodies (core/core_host_impl.mojo) the CPU binding's Python exports
(bindings/_mojolearn_core_host.mojo `knn_classify`, `knn_regress`) call, the
two radius passes, the host search, the host ball-cover search and the two
votes on merged neighbors.
The host binding has no resident index, so the resident operations and the
fold probe are refused here, as the absent exports were. The table entry is
called with the GIL already released by its caller.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, i32_ptr, u32_ptr
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.neighbors_entry_table import (
    NB_OP_CLASSIFY, NB_OP_REGRESS, NB_CLASSIFY_FIXED, nb_ops_check_spans, nb_class_counts,
    NB_OP_RADIUS_COUNT, NB_OP_RADIUS_FILL,
    NB_OP_SEARCH, NB_OP_RBC_SEARCH, NB_OP_CLASSIFY_NEIGHBORS, NB_OP_REGRESS_NEIGHBORS,
    NB_CLASSIFY_NEIGHBORS_FIXED,
)
from core.core_host_impl import (
    knn_classify_host, knn_regress_host, radius_neighbors_count_host, radius_neighbors_fill_host,
    knn_search_host, rbc_knn_search_host, knn_classify_neighbors_host, knn_regress_neighbors_host,
)

#: The query tile the host entries report: the value of
#: bindings/_mojolearn_core_host.mojo `KNN_HOST_QUERY_TILE` (one query at a time).
comptime KNN_HOST_QUERY_TILE = 1


def nb_ops_host_run(operation: Int, a: MutPointer[Int, MutUntrackedOrigin], p: MutPointer[Int, MutUntrackedOrigin],
                    np: Int, f: MutPointer[Float64, MutUntrackedOrigin]) raises -> Int:
    if operation == NB_OP_CLASSIFY:
        var n_classes = nb_class_counts(p, NB_CLASSIFY_FIXED, np, p[5])
        # The query tile word (p[4]) is unread on the host, as in the export.
        knn_classify_host(
            a[0], a[1], a[2], i32_ptr(a[3]), f32_ptr(a[4]), i32_ptr(a[5]),
            p[0], p[1], p[2], p[3], p[5], p[6] != 0, n_classes, p[7], Float32(f[0]), p[8],
        )
        return KNN_HOST_QUERY_TILE
    if operation == NB_OP_REGRESS:
        knn_regress_host(
            a[0], a[1], a[2], f32_ptr(a[3]), p[0], p[1], p[2], p[3], p[5], p[6], Float32(f[0]), p[7],
        )
        return KNN_HOST_QUERY_TILE
    if operation == NB_OP_RADIUS_COUNT:
        # The export's body (bindings/_mojolearn_core_host.mojo
        # radius_neighbors_count_binding). The host pass returns no finite
        # word: the radius RUN scans a borrowed query on the host first.
        return radius_neighbors_count_host(a[0], a[1], i32_ptr(a[2]), p[0], p[1], p[2], Float32(f[0]),
                                           p[3], Float32(f[1]))
    if operation == NB_OP_RADIUS_FILL:
        # The export's body (radius_neighbors_fill_binding): the id and
        # distance addresses are read only for a non-empty result.
        var capacity = p[3]
        return radius_neighbors_fill_host(a[0], a[1], i32_ptr(a[2]), a[3] if capacity > 0 else 0,
                                          a[4] if capacity > 0 else 0, p[0], p[1], p[2], Float32(f[0]),
                                          capacity, p[4] != 0, p[5], Float32(f[1]))
    if operation == NB_OP_SEARCH:
        # The export's body (bindings/_mojolearn_core_host.mojo
        # knn_search_binding): the query tile word (p[5]) is unread on the host.
        knn_search_host(a[0], a[1], f32_ptr(a[2]), u32_ptr(a[3]), p[0], p[1], p[2], p[3], p[4] != 0,
                        p[6], Float32(f[0]))
        return KNN_HOST_QUERY_TILE
    if operation == NB_OP_RBC_SEARCH:
        # The export's body (rbc_knn_search_binding): the host computes every
        # candidate distance, and reports that count.
        rbc_knn_search_host(a[0], a[1], i32_ptr(a[2]), f32_ptr(a[3]), p[0], p[1], p[2], p[3], p[4],
                            Float32(f[0]))
        return p[1] * p[0]
    if operation == NB_OP_CLASSIFY_NEIGHBORS:
        # The export's body (knn_classify_neighbors_binding).
        var n_classes = nb_class_counts(p, NB_CLASSIFY_NEIGHBORS_FIXED, np, p[3])
        knn_classify_neighbors_host(a[0], a[1], a[2], i32_ptr(a[3]), f32_ptr(a[4]), i32_ptr(a[5]),
                                    p[0], p[1], p[2], p[3], p[4] != 0, n_classes, p[5])
        return 0
    if operation == NB_OP_REGRESS_NEIGHBORS:
        # The export's body (knn_regress_neighbors_binding).
        knn_regress_neighbors_host(a[0], a[1], a[2], f32_ptr(a[3]), p[0], p[1], p[2], p[3], p[4])
        return 0
    raise Error("neighbors ops: operation " + String(operation)
                + " needs a resident index, which the CPU column does not keep")


def neighbors_ops_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        nb_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        r[] = nb_ops_host_run(operation, a, p, np, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NeighborsOpsHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(neighbors_ops_host_v1)


comptime _OPS_SLOT = "MojolearnNeighborsOpsHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeighborsOpsHostFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime NEIGHBORS_OPS_HOST_API = SharedSlot[NeighborsOpsHostAPI, _OPS_SLOT]
