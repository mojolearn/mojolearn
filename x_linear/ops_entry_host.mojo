# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the x_linear operations table
(core/linear_entry_table_ext.mojo) and the native bodies the CPU binding's
Python exports (bindings/_mojolearn_x_linear_host.mojo) share with it.

The same x_linear/ source as before (`x_linear_fit_host`, the decision
hosts, `class_prep_host`, `spearman_sign_host`); the table entry is called
with the GIL already released by its caller.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_finite_scan import host_all_finite_f32
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.linear_entry_table_ext import (
    XL_OP_FLAGS, XL_OP_FIT, XL_OP_FIT_FOLD, XL_OP_DECISION, XL_OP_DECISION_CODES,
    XL_OP_DECISION_ROWS, XL_OP_HUBER_OUT, XL_OP_CLASS_PREP, XL_OP_SPEARMAN,
    XL_FIT_HEADER, xl_ops_check_spans, xl_ops_int32, xl_ops_float32,
)
from x_linear.class_prep import class_prep_host
from x_linear.ops import FP, IP
from x_linear.ridgecv import ridge_best_alpha
from x_linear.spearman import spearman_sign_host
from x_linear.x_linear_host_impl import (
    x_linear_fit_host, x_linear_decision_host, x_linear_decision_codes_host, x_linear_decision_rows_host,
    x_linear_decision_huber_out_host,
)


def xl_x_status_host(xa: Int, count: Int, status: Int):
    """The deferred finite test of X on the CPU column (lane a1b; the
    status word of core/admitted_finite_status.mojo, by the parallel SIMD
    scan of core/host_finite_scan.mojo, without that module's device
    import): the word at host address `status` (0: the door tested X)."""
    if status != 0:
        var finite = count <= 0 or host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=xa), count)
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=status)[0] = Int32(0) if finite else Int32(1)


def xl_class_prep_host_run(ca: Int, sa: Int, wa: Int, oa: Int, n: Int, k: Int, balanced: Bool,
                           weighted: Bool) raises -> Int:
    """The GPU column's class prep on the CPU (x_linear/class_prep.mojo):
    the same contract, the same words."""
    if n <= 0 or k < 1 or ca == 0 or wa == 0:
        raise Error("x_linear class prep: n > 0, k > 0, codes and class weights required")
    if weighted and sa == 0:
        raise Error("x_linear class prep: weighted counts need sample weights")
    return class_prep_host(
        IP(unsafe_from_address=ca), FP(unsafe_from_address=sa if sa != 0 else wa), sa != 0,
        FP(unsafe_from_address=wa), n, k, balanced, weighted, FP(unsafe_from_address=oa if oa != 0 else wa),
        oa != 0)


def xl_spearman_host_run(xa: Int, ya: Int, n: Int) raises -> Int:
    if n <= 0 or xa == 0 or ya == 0:
        raise Error("x_linear spearman: n > 0 and two buffers required")
    return spearman_sign_host(FP(unsafe_from_address=xa), FP(unsafe_from_address=ya), n)


def xl_best_alpha_host_run(scores_addr: Int, n: Int) -> Int:
    """RidgeCV's best alpha index over the n mean CV scores; -1 if all NaN."""
    if n <= 0:
        return -1
    return ridge_best_alpha(FP(unsafe_from_address=scores_addr), n)


def x_linear_ops_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        xl_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var f = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == XL_OP_FLAGS:
            # The CPU column has no device fold store.
            r[] = 0
        elif operation == XL_OP_FIT:
            var dims: List[Int] = [p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]]
            var ipl = xl_ops_int32(pp, XL_FIT_HEADER, p[8])
            var fpl = xl_ops_float32(ff, p[9])
            r[] = x_linear_fit_host(p[0], a[0], a[1], dims, ipl, fpl, a[2])
            _ = dims^
            _ = ipl^
            _ = fpl^
        elif operation == XL_OP_FIT_FOLD:
            raise Error("x_linear_fit_fold: only an identical GPU build reads a device fold")
        elif operation == XL_OP_DECISION:
            xl_x_status_host(a[0], p[0] * p[1], p[4])
            r[] = x_linear_decision_host(a[0], a[1], p[0], p[1], p[2], p[3], a[2])
        elif operation == XL_OP_DECISION_CODES:
            xl_x_status_host(a[0], p[0] * p[1], p[7])
            r[] = x_linear_decision_codes_host(a[0], a[1], p[0], p[1], p[2], p[3], p[4], p[5], p[6], a[2])
        elif operation == XL_OP_DECISION_ROWS:
            xl_x_status_host(a[0], p[0] * p[1], p[5])
            r[] = x_linear_decision_rows_host(a[0], a[1], p[0], p[1], p[2], p[3], p[4], a[2])
        elif operation == XL_OP_HUBER_OUT:
            r[] = x_linear_decision_huber_out_host(a[0], a[1], p[0], p[1], p[2], a[2], f[0], a[3])
        elif operation == XL_OP_CLASS_PREP:
            r[] = xl_class_prep_host_run(a[0], a[1], a[2], a[3], p[0], p[1], p[2] != 0, p[3] != 0)
        elif operation == XL_OP_SPEARMAN:
            r[] = xl_spearman_host_run(a[0], a[1], p[0])
        else:
            r[] = xl_best_alpha_host_run(a[0], p[0])
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XLinearOpsHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(x_linear_ops_host_v1)


comptime _OPS_SLOT = "MojolearnXLinearOpsHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXLinearOpsHostFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime XLINEAR_OPS_HOST_API = SharedSlot[XLinearOpsHostAPI, _OPS_SLOT]
