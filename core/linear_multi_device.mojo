# SPDX-License-Identifier: Apache-2.0
"""The multi-output Ridge runs of the GPU estimators binding
(`bindings/_mojolearn_estimators.mojo`), its one `abi("C")` entry in the
ridge multi table (core/linear_entry_table_multi.mojo), and its publication
(lane calltable-base-helpers, 2026-11-11).

Each run is the body of the Python export of the same name after its
arguments were unpacked (the exports now unpack and call these runs): the
same kernels, the same synchronization and the same words. No Python object
crosses; the entry is called with the GIL released by its caller.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.hostptr import f32_ptr, f64_ptr
from core.linear_entry_table import LinearEntryKey
from core.linear_entry_table_multi import (
    RIDGE_MULTI_ABI, RM_OP_COUNT, RM_COL_SUMS, RM_MEANS_FINISH, RM_CENTER, RM_FIT, RM_PREDICT,
    ridge_multi_table_publish, ridge_multi_check_spans,
)
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_context import process_ctx
from core.shared_slot import SharedSlot
from glm.impl.center_device import col_sums_device, center_device
from glm.impl.lm_finish import lm_means_finish
from glm.impl.ridge_multi import MULTIOUT_RIDGE, ridge_fit_multi_host, ridge_predict_multi_host

#: The estimators binding's process-lifetime DeviceContext slot
#: (`bindings/_mojolearn_estimators.mojo::_DEVCTX_SLOT`): the same name, so
#: the same DeviceContext.
comptime ESTIMATORS_DEVCTX_SLOT = (
    "MojolearnEstimatorsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojolearnEstimatorsContextFastV3"
)


def lm_col_sums_run(xa: Int, oa: Int, nr: Int, nc: Int) raises:
    """The exact, correctly rounded float64 sum of every column of a float32
    [rows, cols] matrix, on the device."""
    var ctx = process_ctx[ESTIMATORS_DEVCTX_SLOT]()
    col_sums_device(ctx, xa, oa, nr, nc)


def lm_means_finish_run(sums: Int, mean64: Int, mean32: Int, nc: Int, nr: Int, weighted: Bool,
                        total: Float64) raises:
    """The column means from `lm_col_sums`' exact sums, binary64 and their
    float32 roundings."""
    if nc <= 0:
        return
    lm_means_finish(f64_ptr(sums), f64_ptr(mean64), f32_ptr(mean32), nc, nr, weighted, total)


def lm_center_run(xa: Int, ma: Int, oa: Int, nr: Int, nc: Int) raises:
    """out = x - mu per column in float32 on the device, subnormals
    flushed."""
    var ctx = process_ctx[ESTIMATORS_DEVCTX_SLOT]()
    center_device(ctx, xa, ma, oa, nr, nc)


def ridge_fit_multi_run(x: Int, y: Int, coef: Int, ymean: Int, nr: Int, nf: Int, nt: Int, alpha: Float32,
                        center: Bool, intercepts: Bool, xmean: Int, icpt: Int) raises:
    """Ridge on every column of Y in one program; with `intercepts` (and
    nt, nf > 0) icpt[j] = ymean[j] - xmean . coef[j, :] in the same device
    session."""
    var xp = f32_ptr(x)
    var yp = f32_ptr(y)
    var wp = f32_ptr(coef)
    var mp = f32_ptr(ymean)
    var want_icpt = intercepts and nt > 0 and nf > 0
    var xm = f32_ptr(xmean) if want_icpt else wp
    var ic = f32_ptr(icpt) if want_icpt else wp
    var ctx = process_ctx[ESTIMATORS_DEVCTX_SLOT]()
    ridge_fit_multi_host(ctx, xp, yp, wp, mp, nr, nf, nt, alpha, center, want_icpt, xm, ic)
    ctx.synchronize()
    _ = ctx^


def ridge_predict_multi_run(x: Int, coef: Int, icpt: Int, dst: Int, nr: Int, nf: Int, nt: Int) raises:
    """out (n_rows x n_targets, row-major) = X coef^T + icpt."""
    var xp = f32_ptr(x)
    var cp = f32_ptr(coef)
    var ip = f32_ptr(icpt)
    var op = f32_ptr(dst)
    var ctx = process_ctx[ESTIMATORS_DEVCTX_SLOT]()
    ridge_predict_multi_host(ctx, xp, cp, ip, op, nr, nf, nt)
    ctx.synchronize()


def _w(span: Int, index: Int) -> Int:
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=span)[index]


def _f(span: Int, index: Int) -> Float64:
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=span)[index]


def ridge_multi_dispatch(op: Int, aa: Int, pp: Int, ff: Int) raises:
    """One ridge multi operation by its `RM_*` code over admitted spans."""
    if op == RM_COL_SUMS:
        lm_col_sums_run(_w(aa, 0), _w(aa, 1), _w(pp, 0), _w(pp, 1))
        return
    if op == RM_MEANS_FINISH:
        lm_means_finish_run(_w(aa, 0), _w(aa, 1), _w(aa, 2), _w(pp, 0), _w(pp, 1), _w(pp, 2) != 0, _f(ff, 0))
        return
    if op == RM_CENTER:
        lm_center_run(_w(aa, 0), _w(aa, 1), _w(aa, 2), _w(pp, 0), _w(pp, 1))
        return
    if op == RM_FIT:
        ridge_fit_multi_run(_w(aa, 0), _w(aa, 1), _w(aa, 2), _w(aa, 3), _w(pp, 0), _w(pp, 1), _w(pp, 2),
                            Float32(_f(ff, 0)), _w(pp, 3) != 0, _w(pp, 4) != 0, _w(aa, 4), _w(aa, 5))
        return
    if op == RM_PREDICT:
        ridge_predict_multi_run(_w(aa, 0), _w(aa, 1), _w(aa, 2), _w(aa, 3), _w(pp, 0), _w(pp, 1), _w(pp, 2))
        return
    raise Error("ridge multi: unknown operation " + String(op))


def ridge_multi_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The ridge multi entry (`RIDGE_MULTI_ABI`): addresses aa[0:na], Int
    words pp[0:np], Float64 words ff[0:nf]. Writes 0 to result; a refusal
    goes to sink."""
    try:
        ridge_multi_check_spans(operation, aa, na, pp, np, ff, nf, result)
        ridge_multi_dispatch(operation, aa, pp, ff)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct RidgeMultiOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the ridge multi table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime _RIDGE_MULTI_OWNER_NAME = "MojolearnRidgeMultiOwnerV1"
comptime RIDGE_MULTI_OWNER = SharedSlot[RidgeMultiOwner, _RIDGE_MULTI_OWNER_NAME]


def publish_ridge_multi() raises:
    """Publish the ridge multi entry at the estimators binding's init, only
    when the build carries the one-program ridge (`MULTIOUT_RIDGE`) and has a
    GPU vendor; otherwise nothing is published and callers fall back to one
    member per column."""
    comptime if MULTIOUT_RIDGE:
        var word = (1 if COMPILED_VENDOR == "metal" else 2 if COMPILED_VENDOR == "cuda"
                    else 3 if COMPILED_VENDOR == "hip" else -1)
        if word < 1:
            return
        var owner = RIDGE_MULTI_OWNER.get_or_create_ptr()
        var key = LinearEntryKey(Int(GLOBAL_NUMERIC_MODE), word)
        ridge_multi_table_publish(key, Int(owner), [String(RIDGE_MULTI_ABI)], [RM_OP_COUNT],
                                  [ridge_multi_device_v1])
