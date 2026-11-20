# SPDX-License-Identifier: Apache-2.0
"""The device column of the PCA owned-fit table
(core/decomp_entry_table_pca.mojo): the body of the GPU estimators binding's
Python export `pca_fit_owned` (bindings/_mojolearn_estimators.mojo), which
now UNPACKs its arguments, releases the GIL and calls this same body: the
same pointers, the same owner (decomposition/impl/pca_owner.mojo) and the
same words, so the bits are unchanged. The table entry is called with the
GIL already released by its caller; nothing here touches Python.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, f64_ptr
from core.neural_context import process_ctx
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.decomp_entry_table_pca import pca_ops_check_spans, pca_status_word
from decomposition.impl.pca_owner import pca_fit_owned

#: The estimators binding's ONE process-lifetime DeviceContext: the slot name
#: bindings/_mojolearn_estimators.mojo `_DEVCTX_SLOT` names, so the table
#: entry and the Python exports share the one context.
comptime PCA_DEVCTX_SLOT = "MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnEstimatorsContextFastV3"


def pca_fit_owned_device_run(a: List[Int], p: List[Int], fraction: Float64) raises -> Tuple[Int, Float64]:
    """`pca_fit_owned`: a x, components, mean, explained, ratio, singular,
    out (0: none), noise (unused here), status (0: no finite test); p n_rows,
    n_features, n_components, dense, keep mode, whiten. Returns (kept count,
    noise variance); a flagged X returns (0, 0.0) with its status written."""
    var xp = f32_ptr(a[0])
    var cp = f32_ptr(a[1])
    var mp = f32_ptr(a[2])
    var ep = f32_ptr(a[3])
    var rp = f32_ptr(a[4])
    var sp = f32_ptr(a[5])
    var oa = a[6]
    var op = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=oa)  # read only when oa != 0
    var word_addr = pca_status_word(a[8])
    var ctx = process_ctx[PCA_DEVCTX_SLOT]()
    return pca_fit_owned(
        ctx, xp, cp, mp, ep, rp, sp, p[0], p[1], p[2], p[3] != 0, p[4], fraction, op, oa != 0, p[5] != 0,
        word_addr,
    )


def pca_ops_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        pca_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var pa = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var pw = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var pf = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var a = List[Int](capacity=na)
        for i in range(na):  # small-loop(no: nine buffer addresses): metadata, not data
            a.append(pa[i])
        var p = List[Int](capacity=np)
        for i in range(np):  # small-loop(no: six settings words): metadata, not data
            p.append(pw[i])
        var noise_out = f64_ptr(a[7])
        var fitted = pca_fit_owned_device_run(a, p, pf[0])
        noise_out[] = fitted[1]
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        r[] = fitted[0]
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct PcaOpsDeviceAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(pca_ops_device_v1)


comptime _OPS_SLOT = "MojolearnPcaOwnedOpsDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnPcaOwnedOpsDeviceFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime PCA_OPS_DEVICE_API = SharedSlot[PcaOpsDeviceAPI, _OPS_SLOT]
