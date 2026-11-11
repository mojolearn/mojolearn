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
from core.hostptr import f32_ptr, i32_ptr
from core.neural_context import process_ctx
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.neighbors_entry_table import (
    NB_OP_INDEX_PREPARE, NB_OP_INDEX_RELEASE, NB_OP_FOLD_READY, NB_OP_CLASSIFY, NB_OP_REGRESS,
    NB_OP_CLASSIFY_RESIDENT, NB_OP_REGRESS_RESIDENT, NB_CLASSIFY_FIXED, NB_CLASSIFY_RESIDENT_FIXED,
    nb_ops_check_spans, nb_class_counts, nb_status_word, nb_fold_words,
)
from neighbors.impl.multi_gpu import neighbors_device_plan
from neighbors.resident_index import (
    knn_fold_ready, knn_index_classify, knn_index_prepare, knn_index_regress, knn_index_release,
)
from neighbors.estimator import knn_classifier_predict, knn_regressor_predict

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
        var qp = _f32(a[1])
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
    # NB_OP_REGRESS_RESIDENT (nb_ops_check_spans refused any other id)
    var ip = _f32(a[0])
    var qp = _f32(a[1])
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
