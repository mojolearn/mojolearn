# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the PCA owned-fit table
(core/decomp_entry_table_pca.mojo): the body of the host estimators binding's
Python export `pca_fit_owned` (bindings/_mojolearn_estimators_host.mojo),
which now UNPACKs its arguments, releases the GIL and calls this same body
(core/estimators_host_impl.mojo `pca_fit_owned_host_run` behind the parallel
host finite scan): the same words, so the bits are unchanged. The table entry
is called with the GIL already released by its caller; nothing here touches
Python.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, f64_ptr
from core.host_finite_scan import host_finite_status
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.decomp_entry_table_pca import pca_ops_check_spans, pca_status_word
from core.estimators_host_impl import pca_fit_owned_host_run


def pca_fit_owned_host_run_words(a: List[Int], p: List[Int], fraction: Float64) raises -> Tuple[Int, Float64]:
    """`pca_fit_owned` on the host, over the device body's words. With a
    status address, X's finite status word is written there first (one
    parallel host scan) and a flagged X is not fit: (0, 0.0)."""
    var x_address = a[0]
    var cp = f32_ptr(a[1])
    var mp = f32_ptr(a[2])
    var ep = f32_ptr(a[3])
    var rp = f32_ptr(a[4])
    var sp = f32_ptr(a[5])
    var oa = a[6]
    var op = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=oa)  # read only when oa != 0
    var nr = p[0]
    var nf = p[1]
    var status_address = pca_status_word(a[8])
    var status = 0
    if status_address != 0 and nr >= 1 and nf >= 1:
        # parallel: the deferred finite test of X (SIMD scan across the host's cores)
        status = Int(host_finite_status(x_address, nr * nf))
    if status_address != 0:
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=status_address).unsafe_store(0, Int32(status))
    if status != 0:
        return (0, Float64(0.0))
    return pca_fit_owned_host_run(
        x_address, cp, mp, ep, rp, sp, nr, nf, p[2], p[3] != 0, p[4], fraction, op, oa != 0, p[5] != 0
    )


def pca_ops_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
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
        var fitted = pca_fit_owned_host_run_words(a, p, pf[0])
        noise_out[] = fitted[1]
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        r[] = fitted[0]
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct PcaOpsHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(pca_ops_host_v1)


comptime _OPS_SLOT = "MojolearnPcaOwnedOpsHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnPcaOwnedOpsHostFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime PCA_OPS_HOST_API = SharedSlot[PcaOpsHostAPI, _OPS_SLOT]
