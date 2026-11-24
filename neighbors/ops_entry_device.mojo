# SPDX-License-Identifier: Apache-2.0
"""The device column of the neighbors operations table
(core/neighbors_entry_table.mojo): the k-NN bodies the GPU base binding's
Python exports (bindings/_mojolearn.mojo `knn_index_prepare`,
`knn_index_release`, `knn_fold_ready`, `knn_classify`, `knn_regress`,
`knn_classify_resident`, `knn_regress_resident`) call, with the same checks,
the same kernels and the same words. The table entry is called with the GIL
already released by its caller.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, i32_ptr, u32_ptr
from core.neural_context import process_ctx
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.neighbors_entry_table import (
    NB_OP_INDEX_PREPARE, NB_OP_INDEX_RELEASE, NB_OP_FOLD_READY, NB_OP_INDEX_PREPARE_FOLD, NB_OP_CLASSIFY, NB_OP_REGRESS,
    NB_OP_CLASSIFY_RESIDENT, NB_OP_REGRESS_RESIDENT, NB_CLASSIFY_FIXED, NB_CLASSIFY_RESIDENT_FIXED,
    NB_OP_RADIUS_COUNT, NB_OP_RADIUS_FILL,
    NB_OP_SEARCH, NB_OP_SEARCH_RESIDENT, NB_OP_RBC_SEARCH, NB_OP_CLASSIFY_NEIGHBORS, NB_OP_REGRESS_NEIGHBORS,
    NB_CLASSIFY_NEIGHBORS_FIXED,
    nb_ops_check_spans, nb_class_counts, nb_status_word, nb_fold_words,
)
from neighbors.impl.multi_gpu import neighbors_device_plan
from neighbors.resident_index import (
    knn_fold_ready, knn_index_classify, knn_index_prepare, knn_index_prepare_fold, knn_index_regress, knn_index_release,
    knn_index_search,
)
from neighbors.impl.detail.knn_brute_force import KNN_METHOD_AUTO
from neighbors.estimator import (
    knn_classifier_predict, knn_regressor_predict, radius_neighbors_count, radius_neighbors_fill,
    knn_search, rbc_knn_search, knn_classifier_from_neighbors, knn_regressor_from_neighbors,
)

#: The base binding's ONE process-lifetime DeviceContext: the same slot name
#: bindings/_mojolearn.mojo `_DEVCTX_SLOT` names, so the table entry and the
#: Python exports share the one context.
comptime NEIGHBORS_DEVCTX_SLOT = "MojolearnCoreContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnCoreContextFastV3"

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def _f32(address: Int) raises -> _F32P:
    # The exports' `_f32_ptr` refusal, word for word.
    if address == 0:
        raise Error("mojolearn: null buffer address")
    return f32_ptr(address)


def _i32(address: Int) raises -> _I32P:
    # The exports' `_i32_ptr` refusal, word for word.
    if address == 0:
        raise Error("mojolearn: null int32 buffer address")
    return i32_ptr(address)


def _resident_queries(index: _F32P, address: Int, has_fold: Int) raises -> _F32P:
    # A fold's queries are gathered on the device from the store; the owner
    # passes no host query address (0) for it. The Python exports refused
    # that 0 as a null buffer, so a fold never reached the device gather
    # through them; here the index pointer stands in, never read as queries.
    if has_fold != 0 and address == 0:
        return index
    return _f32(address)


def nb_ops_device_run(operation: Int, a: MutPointer[Int, MutUntrackedOrigin], p: MutPointer[Int, MutUntrackedOrigin],
                      np: Int, f: MutPointer[Float64, MutUntrackedOrigin]) raises -> Int:
    if operation == NB_OP_INDEX_PREPARE:
        var ni = p[0]
        var nf = p[1]
        if ni <= 0 or nf <= 0:
            raise Error("knn_index_prepare: n_index and n_features must be positive")
        var status = a[1]
        if status < 0 or status % 4 != 0:
            raise Error("knn_index_prepare: the status address must be a positive int32-aligned address")
        return knn_index_prepare(_f32(a[0]), ni, nf, status)
    if operation == NB_OP_INDEX_RELEASE:
        knn_index_release(p[0])
        return 0
    if operation == NB_OP_FOLD_READY:
        return 1 if knn_fold_ready() else 0
    if operation == NB_OP_INDEX_PREPARE_FOLD:
        # Lane vdf-scope-cluster2a: the index from a device fold (no host words).
        var status = nb_status_word(a[0])
        var fold = nb_fold_words(1, p[0], p[1], p[2], status)
        return knn_index_prepare_fold(fold[0], fold[1], fold[2], p[3], p[4], status)
    if operation == NB_OP_CLASSIFY:
        var n_classes = nb_class_counts(p, NB_CLASSIFY_FIXED, np, p[5])
        var ip = _f32(a[0])
        var qp = _f32(a[1])
        var yp = _i32(a[2])
        var lp = _i32(a[3])
        var pp = _f32(a[4])
        var up = _i32(a[5])
        # The entry reads the neighbors device plan once and passes it down.
        var dplan = neighbors_device_plan()
        var ctx = process_ctx[NEIGHBORS_DEVCTX_SLOT]()
        var used = knn_classifier_predict(
            dplan, ctx, ip, p[0], qp, p[1], p[2], p[3], yp, p[5], n_classes, lp, pp, up,
            p[6] != 0, p[4], p[7], Float32(f[0]), p[8],
        )
        ctx.synchronize()
        return used
    if operation == NB_OP_REGRESS:
        var ip = _f32(a[0])
        var qp = _f32(a[1])
        var yp = _f32(a[2])
        var op = _f32(a[3])
        var ctx = process_ctx[NEIGHBORS_DEVCTX_SLOT]()
        var used = knn_regressor_predict(
            ctx, ip, p[0], qp, p[1], p[2], p[3], yp, p[5], op, p[4], p[6], Float32(f[0]), p[7]
        )
        ctx.synchronize()
        return used
    if operation == NB_OP_CLASSIFY_RESIDENT:
        var n_classes = nb_class_counts(p, NB_CLASSIFY_RESIDENT_FIXED, np, p[6])
        var ip = _f32(a[0])
        var qp = _resident_queries(ip, a[1], p[10])
        var yp = _i32(a[2])
        var lp = _i32(a[3])
        var pp = _f32(a[4])
        var up = _i32(a[5])
        var status = nb_status_word(a[6])
        var fold = nb_fold_words(p[10], p[11], p[12], p[13], status)
        return knn_index_classify(
            p[0], ip, p[1], qp, p[2], p[3], p[4], yp, p[6], n_classes, lp, pp, up,
            p[7] != 0, p[5], p[8], Float32(f[0]), p[9], status, fold[0], fold[1], fold[2],
        )
    if operation == NB_OP_RADIUS_COUNT:
        # The export's body (bindings/_mojolearn.mojo radius_neighbors_count_binding).
        var ip = _f32(a[0])
        var qp = _f32(a[1])
        var ap = _i32(a[2])
        var status = a[3]
        if status < 0 or status % 4 != 0:
            raise Error("radius_neighbors_count: the status address must be a positive int32-aligned address")
        var ctx = process_ctx[NEIGHBORS_DEVCTX_SLOT]()
        var nnz = radius_neighbors_count(ctx, ip, p[0], qp, p[1], p[2], Float32(f[0]), ap, p[3], Float32(f[1]), status)
        ctx.synchronize()
        return nnz
    if operation == NB_OP_RADIUS_FILL:
        # The export's body (bindings/_mojolearn.mojo radius_neighbors_fill_binding).
        var ip = _f32(a[0])
        var qp = _f32(a[1])
        var ap = _i32(a[2])
        var xp = _i32(a[3])
        var dp = _f32(a[4])
        var ctx = process_ctx[NEIGHBORS_DEVCTX_SLOT]()
        var nnz = radius_neighbors_fill(ctx, ip, p[0], qp, p[1], p[2], Float32(f[0]), ap, xp, dp, p[3],
                                        p[4] != 0, p[5], Float32(f[1]))
        ctx.synchronize()
        return nnz
    if operation == NB_OP_SEARCH:
        # The export's body (bindings/_mojolearn.mojo knn_search_binding).
        var ip = _f32(a[0])
        var qp = _f32(a[1])
        var dp = _f32(a[2])
        var xp = u32_ptr(a[3])
        var ctx = process_ctx[NEIGHBORS_DEVCTX_SLOT]()
        var used = knn_search(ctx, ip, p[0], qp, p[1], p[2], p[3], dp, xp, p[4] != 0, p[5], KNN_METHOD_AUTO,
                              p[6], Float32(f[0]), True)
        ctx.synchronize()
        return used
    if operation == NB_OP_SEARCH_RESIDENT:
        # The export's body (knn_search_resident_binding); a fold is taken
        # by the resident classify / regress operations only.
        var ip = _f32(a[0])
        var qp = _f32(a[1])
        var dp = _f32(a[2])
        var xp = u32_ptr(a[3])
        var status = nb_status_word(a[4])
        return knn_index_search(p[0], ip, p[1], qp, p[2], p[3], p[4], dp, xp, p[5] != 0, p[6], KNN_METHOD_AUTO,
                                p[7], Float32(f[0]), True, status)
    if operation == NB_OP_RBC_SEARCH:
        # The export's body (rbc_knn_search_binding).
        var ip = _f32(a[0])
        var qp = _f32(a[1])
        var xp = _i32(a[2])
        var dp = _f32(a[3])
        var ctx = process_ctx[NEIGHBORS_DEVCTX_SLOT]()
        var n_dists = rbc_knn_search(ctx, ip, p[0], qp, p[1], p[2], p[3], xp, dp, p[4], Float32(f[0]))
        ctx.synchronize()
        return n_dists
    if operation == NB_OP_CLASSIFY_NEIGHBORS:
        # The export's body (knn_classify_neighbors_binding).
        var n_classes = nb_class_counts(p, NB_CLASSIFY_NEIGHBORS_FIXED, np, p[3])
        var dp = _f32(a[0])
        var xp = u32_ptr(a[1])
        var yp = _i32(a[2])
        var lp = _i32(a[3])
        var pp = _f32(a[4])
        var up = _i32(a[5])
        var ctx = process_ctx[NEIGHBORS_DEVCTX_SLOT]()
        knn_classifier_from_neighbors(ctx, dp, xp, p[0], p[1], p[2], yp, p[3], n_classes, lp, pp, up,
                                      p[4] != 0, p[5])
        ctx.synchronize()
        return 0
    if operation == NB_OP_REGRESS_NEIGHBORS:
        # The export's body (knn_regress_neighbors_binding).
        var dp = _f32(a[0])
        var xp = u32_ptr(a[1])
        var yp = _f32(a[2])
        var op = _f32(a[3])
        var ctx = process_ctx[NEIGHBORS_DEVCTX_SLOT]()
        knn_regressor_from_neighbors(ctx, dp, xp, p[0], p[1], p[2], yp, p[3], op, p[4])
        ctx.synchronize()
        return 0
    # NB_OP_REGRESS_RESIDENT (nb_ops_check_spans refused any other id)
    var ip = _f32(a[0])
    var qp = _resident_queries(ip, a[1], p[9])
    var yp = _f32(a[2])
    var op = _f32(a[3])
    var status = nb_status_word(a[4])
    var fold = nb_fold_words(p[9], p[10], p[11], p[12], status)
    return knn_index_regress(
        p[0], ip, p[1], qp, p[2], p[3], p[4], yp, p[6], op, p[5], p[7], Float32(f[0]), p[8],
        status, fold[0], fold[1], fold[2],
    )


def neighbors_ops_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        nb_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        r[] = nb_ops_device_run(operation, a, p, np, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NeighborsOpsDeviceAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(neighbors_ops_device_v1)


comptime _OPS_SLOT = "MojolearnNeighborsOpsDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeighborsOpsDeviceFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime NEIGHBORS_OPS_DEVICE_API = SharedSlot[NeighborsOpsDeviceAPI, _OPS_SLOT]
