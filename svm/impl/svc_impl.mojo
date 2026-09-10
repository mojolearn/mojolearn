# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`svcFit` / `svcPredict`: the C-SVC entry points, dense FP32.

FOLLOWS `cuml/cpp/src/svm/svc_impl.cuh` + `svc.cu` at cuML v26.08.00:
`svcFitX` (dense), `svcPredictX` (dense, dense support), `applyPrediction`,
`computeBatchDecisionFunction`, and the `SVC` class's `fit` / `predict` /
`decisionFunction`. NOT implemented: `svcFitSparse` / `svcPredictSparse` and the
CSR arms, the PRECOMPUTED arm, multiclass (their own `ASSERT(model.n_classes
== 2, "Only binary classification is implemented at the moment")` is kept
as a raise), `svmFreeBuffers` (host lists).

    raft::label::getUniquelabels   -> host sort + unique of the labels
                                     (control plane; the device sort they
                                     use is thrust's, the labels are tiny)
    raft::label::getOvrlabels      -> ovr_labels_kernel: y = +1 where label
                                     == unique_labels[1] (the LARGER label),
                                     else -1 -- `getOvrlabels(..., idx=1)`
    raft::linalg::gemv (predict)   -> decision_kernel, one thread per query
                                     row, ascending over the support vectors
                                     (DEVIATION 634's rule again; theirs is
                                     cuBLAS)
    applyPrediction                -> the same kernel's epilogue:
                                     `val + b < 0 ? labels[0] : labels[1]`
                                     or `val + b`

The training matrix is ROW-MAJOR here (theirs `layout_f_contiguous`); a
layout, not an arithmetic, and.
"""

from std.builtin.sort import sort
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add
from svm.checks.device_select import SEL_TPB, read_f32, upload_f32
from svm.impl.distance.kernel_matrices import (
    kernel_op,
    kernel_workspace_floats,
    row_norms_l2sq,
)
from svm.impl.smosolver import SmoSolver, SmoTrace
from svm.impl.svm_parameter import (
    KERNEL_LINEAR,
    KERNEL_RBF,
    KernelParams,
    SvmModel,
    SvmParameter,
    check_finite_list,
    check_rung1_scope,
)


def _grid(n: Int) -> Int:
    return (n + SEL_TPB - 1) // SEL_TPB


def ovr_labels_kernel(
    y_out: MutPointer[Float32, MutAnyOrigin],
    labels: MutPointer[Float32, MutAnyOrigin],
    positive: Float32,
    n_in: Int32,
):
    """`getOvrlabels`: `y == y_unique[idx] ? +1 : -1` at `idx = 1`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var v = labels.unsafe_load(i)
        y_out.unsafe_store(i, Float32(1.0) if v == positive else Float32(-1.0))


def decision_kernel(
    preds: MutPointer[Float32, MutAnyOrigin],
    tile: MutPointer[Float32, MutAnyOrigin],
    dual_coefs: MutPointer[Float32, MutAnyOrigin],
    n_support_in: Int32,
    batch_in: Int32,
    b: Float32,
    label0: Float32,
    label1: Float32,
    predict_class_in: Int32,
):
    """`computeBatchDecisionFunction` + `applyPrediction` for one batch:
    `val = sum_j K[i, j] * dual_coefs[j]` ascending over `j`, then `val +
    b` (decision function) or the label lookup. `tile` is `[batch x
    n_support]` row-major."""
    var ns = Int(n_support_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(batch_in):
        var acc = Float32(0.0)
        for j in range(ns):
            acc = ftz(
                identical_mul_add(
                    tile.unsafe_load(i * ns + j), dual_coefs.unsafe_load(j), acc
                )
            )
        var val = ftz(acc + b)
        if predict_class_in != 0:
            preds.unsafe_store(i, label0 if val < Float32(0.0) else label1)
        else:
            preds.unsafe_store(i, val)


def unique_labels_sorted(labels: List[Float32]) raises -> List[Float32]:
    """`raft::label::getUniquelabels`: sorted distinct values."""
    var keys = List[Float64]()
    for v in labels:
        keys.append(Float64(v))
    sort(keys)
    var out = List[Float32]()
    for i in range(len(keys)):
        if i == 0 or keys[i] != keys[i - 1]:
            out.append(Float32(keys[i]))
    return out^


def svc_fit(
    ctx: DeviceContext,
    x_host: List[Float32],
    labels_host: List[Float32],
    n_rows: Int,
    n_cols: Int,
    param: SvmParameter,
    kp: KernelParams,
    mut card: IdentityTrace,
    mut trace: SmoTrace,
    has_sample_weight: Bool = False,
    kernel_tile_byte_limit: Int = 1 << 30,
    block_solve_threads: Int = 0,
    record_iterations: Bool = False,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = 0.0,
) raises -> SvmModel:
    """`svcFit(handle, input, n_rows, n_cols, labels, param, kernel_params,
    model, sample_weight)`. Returns the model (host); `trace` receives the
    iteration record (empty unless `record_iterations`)."""
    if n_cols <= 0:
        raise Error("Parameter n_cols: number of columns cannot be less than one")
    if n_rows <= 0:
        raise Error("Parameter n_rows: number of rows cannot be less than one")
    if len(x_host) != n_rows * n_cols or len(labels_host) != n_rows:
        raise Error("svc_fit: x / labels sizes do not match n_rows x n_cols")
    check_rung1_scope(param, kp, has_sample_weight)
    # DEVIATION 636 (row 39, FACT 2): no NaN/inf may enter; see svm_parameter.mojo
    check_finite_list(x_host, "X")
    check_finite_list(labels_host, "labels")

    var model = SvmModel()
    model.unique_labels = unique_labels_sorted(labels_host)
    model.n_classes = len(model.unique_labels)
    if model.n_classes != 2:
        raise Error(
            "Only binary classification is implemented at the moment (got "
            + String(model.n_classes) + " classes)"
        )
    card.header(
        "svm.svc_fit n_rows=" + String(n_rows) + " n_cols=" + String(n_cols)
        + " kernel=" + String(kp.kernel) + " C=" + String(param.C)
        + " tol=" + String(param.tol)
    )

    var x = upload_f32(ctx, x_host)
    var labels = upload_f32(ctx, labels_host)
    var y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.synchronize()
    ctx.enqueue_function[ovr_labels_kernel](
        y.unsafe_ptr(), labels.unsafe_ptr(), model.unique_labels[1], Int32(n_rows),
        grid_dim=_grid(n_rows), block_dim=SEL_TPB,
    )
    ctx.synchronize()
    card.record_device[DType.float32](ctx, "svm.input.x", x, n_rows * n_cols)
    card.record_device[DType.float32](ctx, "svm.input.y", y, n_rows)

    var smo = SmoSolver(
        ctx, param, kp, n_rows, n_cols, kernel_tile_byte_limit,
        block_solve_threads, record_iterations, scratch_pad, scratch_poison,
    )
    smo.solve(ctx, x, y, model, card, param.max_iter, param.max_outer_iter)
    model.n_cols = n_cols
    ctx.synchronize()
    trace = smo.trace^
    smo.trace = SmoTrace()
    _ = x^
    _ = labels^
    _ = y^
    return model^



# ---------------------------------------------------------------------------
# DEVIATION 2493 (2026-09-10): THE FUSED FAST DECISION PASS
# ---------------------------------------------------------------------------
# `svc_predict` below is upstream's shape: a `[batch x n_support]` kernel
# tile per batch (`kernel_op`), then `decision_kernel` folding each row of
# the tile against the dual coefficients, batches sized by `cache_size`.
# The fold reads the tile with a stride of `n_support` between neighbouring
# threads, so every batch's 4 x batch x n_support bytes are written once
# and read back uncoalesced; at 100,000 rows against 39,349 support
# vectors that is 75 batches and about 15 GB of tile traffic for a
# 783 ms predict on the M4.
#
# FAST folds the decision inside the kernel evaluation instead: one thread
# per query row with its features in registers, the support vectors
# streamed through shared memory with their squared norms and dual
# coefficients, `acc += dual_j * K(x_i, sv_j)` ascending in j, then the
# `+ b` and the label epilogue. No tile, no batch loop, one launch. Linear
# and RBF; register rows up to `SVC_FUSED_KREG` features (wider inputs keep
# the tiled path). IDENTICAL and DETERMINISTIC keep the tiled path with its
# recorded `svm.predict.*` card. FAST only; the FAST arm is free to move.
comptime SVC_FUSED_TPB = 128
comptime SVC_FUSED_TILE_FLOATS = 3072
comptime SVC_FUSED_KREG = 64


def svc_fused_decision_kernel[KPAD: Int](
    preds: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    sv: MutPointer[Float32, MutAnyOrigin],
    dual: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_support_in: Int32,
    k_in: Int32,
    is_rbf_in: Int32,
    gain: Float32,
    b: Float32,
    label0: Float32,
    label1: Float32,
    predict_class_in: Int32,
):
    from std.math import exp

    comptime ROWS = SVC_FUSED_TILE_FLOATS // KPAD
    var n_rows = Int(n_rows_in)
    var n_support = Int(n_support_in)
    var k = Int(k_in)
    var is_rbf = Int(is_rbf_in) != 0
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * SVC_FUSED_TPB + tid
    var active = i < n_rows

    var tile = stack_allocation[
        SVC_FUSED_TILE_FLOATS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var tile_dual = stack_allocation[
        ROWS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tile_norm = stack_allocation[
        ROWS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var xreg = stack_allocation[KPAD, Scalar[DType.float32]]()
    var nx = Float32(0.0)
    comptime for f in range(KPAD):
        var v = Float32(0.0)
        if active and f < k:
            v = x.unsafe_load(i * k + f)
        xreg.unsafe_store(f, v)
        nx += v * v

    var acc = Float32(0.0)
    var j0 = 0
    while j0 < n_support:
        var rows_here = n_support - j0
        if rows_here > ROWS:
            rows_here = ROWS
        barrier()
        var total = rows_here * KPAD
        var idx = tid
        while idx < total:
            var r = idx // KPAD
            var f = idx - r * KPAD
            var v = Float32(0.0)
            if f < k:
                v = sv.unsafe_load((j0 + r) * k + f)
            tile.unsafe_store(idx, v)
            idx += SVC_FUSED_TPB
        idx = tid
        while idx < rows_here:
            tile_dual.unsafe_store(idx, dual.unsafe_load(j0 + idx))
            # The support row's squared norm, formed here from the tile
            # (the row is in shared memory already; one thread per row).
            var nsv = Float32(0.0)
            comptime for f in range(KPAD):
                var t = sv.unsafe_load((j0 + idx) * k + f) if f < k else Float32(0.0)
                nsv += t * t
            tile_norm.unsafe_store(idx, nsv)
            idx += SVC_FUSED_TPB
        barrier()
        if active:
            for r in range(rows_here):
                var tb = r * KPAD
                var dot = Float32(0.0)
                comptime for f in range(KPAD):
                    dot += xreg.unsafe_load(f) * tile.unsafe_load(tb + f)
                var kv: Float32
                if is_rbf:
                    kv = exp(-gain * (nx + tile_norm.unsafe_load(r) - dot * Float32(2.0)))
                else:
                    kv = dot
                acc += tile_dual.unsafe_load(r) * kv
        j0 += ROWS
    if active:
        var val = acc + b
        if predict_class_in != 0:
            preds.unsafe_store(i, label0 if val < Float32(0.0) else label1)
        else:
            preds.unsafe_store(i, val)


def _svc_fused_launch[KPAD: Int](
    ctx: DeviceContext,
    mut preds: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut sv: DeviceBuffer[DType.float32],
    mut dual: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_support: Int,
    k: Int,
    is_rbf: Bool,
    gain: Float32,
    b: Float32,
    label0: Float32,
    label1: Float32,
    predict_class: Bool,
) raises:
    ctx.enqueue_function[svc_fused_decision_kernel[KPAD]](
        preds.unsafe_ptr(), x.unsafe_ptr(), sv.unsafe_ptr(), dual.unsafe_ptr(),
        Int32(n_rows), Int32(n_support), Int32(k), Int32(1 if is_rbf else 0),
        gain, b, label0, label1, Int32(1 if predict_class else 0),
        grid_dim=((n_rows + SVC_FUSED_TPB - 1) // SVC_FUSED_TPB, 1, 1),
        block_dim=(SVC_FUSED_TPB, 1, 1),
    )


def svc_fused_decision(
    ctx: DeviceContext,
    mut preds: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut sv: DeviceBuffer[DType.float32],
    mut dual: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_support: Int,
    k: Int,
    is_rbf: Bool,
    gain: Float32,
    b: Float32,
    label0: Float32,
    label1: Float32,
    predict_class: Bool,
) raises -> Bool:
    """DEVIATION 2493's dispatch: True when the fused pass served the
    prediction, False when k is too wide for a register row."""
    var kpad = ((k + 3) // 4) * 4
    if kpad == 4:
        _svc_fused_launch[4](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 8:
        _svc_fused_launch[8](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 12:
        _svc_fused_launch[12](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 16:
        _svc_fused_launch[16](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 20:
        _svc_fused_launch[20](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 24:
        _svc_fused_launch[24](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 28:
        _svc_fused_launch[28](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 32:
        _svc_fused_launch[32](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 36:
        _svc_fused_launch[36](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 40:
        _svc_fused_launch[40](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 44:
        _svc_fused_launch[44](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 48:
        _svc_fused_launch[48](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 52:
        _svc_fused_launch[52](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 56:
        _svc_fused_launch[56](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 60:
        _svc_fused_launch[60](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    elif kpad == 64:
        _svc_fused_launch[64](ctx, preds, x, sv, dual, n_rows, n_support, k, is_rbf, gain, b, label0, label1, predict_class)
    else:
        return False
    return True


def svc_predict(
    ctx: DeviceContext,
    model: SvmModel,
    x_host: List[Float32],
    n_rows: Int,
    n_cols: Int,
    kp: KernelParams,
    buffer_size_mib: Float64,
    predict_class: Bool,
    mut card: IdentityTrace,
) raises -> List[Float32]:
    """`svcPredict(handle, input, n_rows, n_cols, kernel_params, model,
    preds, buffer_size, predict_class)`, dense input and dense support.
    `buffer_size` is `param.cache_size` at their `SVC::predict` call site;
    here it is a parameter, and it is the n_rows BATCH knob the launch-
    invariance gate turns."""
    if n_cols != model.n_cols:
        raise Error("Parameter n_cols: shall be the same that was used for fitting")
    if len(x_host) != n_rows * n_cols:
        raise Error("svc_predict: x size does not match n_rows x n_cols")
    if n_rows <= 0:
        return List[Float32]()
    check_finite_list(x_host, "predict X")  # DEVIATION 636
    var n_support = model.n_support
    var n_batch = n_rows
    var buffer_bytes = buffer_size_mib * 1024.0 * 1024.0
    if n_support > 0 and Float64(n_batch * n_support * 4) > buffer_bytes:
        n_batch = Int(buffer_bytes / Float64(n_support * 4))
        if n_batch < 1:
            n_batch = 1

    var preds = List[Float32]()
    if n_support == 0:
        # cudaMemsetAsync(y, 0): the decision function is `0 + b`.
        var v = ftz(Float32(0.0) + model.b)
        for _ in range(n_rows):
            if predict_class:
                preds.append(
                    model.unique_labels[0] if v < Float32(0.0) else model.unique_labels[1]
                )
            else:
                preds.append(v)
        return preds^

    var x = upload_f32(ctx, x_host)
    var sv = upload_f32(ctx, model.support_matrix)
    var dual = upload_f32(ctx, model.dual_coefs)
    var d_preds = ctx.enqueue_create_buffer[DType.float32](n_rows)
    # DEVIATION 2493: FAST with no card recording folds the decision inside
    # the kernel evaluation; no tile, no batch loop.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
        if not card.enabled and n_cols <= SVC_FUSED_KREG and (
            kp.kernel == KERNEL_RBF or kp.kernel == KERNEL_LINEAR
        ):
            ctx.synchronize()
            if svc_fused_decision(
                ctx, d_preds, x, sv, dual, n_rows, n_support, n_cols,
                kp.kernel == KERNEL_RBF, Float32(kp.gamma), model.b,
                model.unique_labels[0], model.unique_labels[1], predict_class,
            ):
                ctx.synchronize()
                preds = read_f32(ctx, d_preds, n_rows)
                for i in range(n_rows):
                    if preds[i] != preds[i]:
                        raise Error(
                            "svm predict: NaN decision value at row " + String(i)
                            + " (floating point overflow; DEVIATION 637)"
                        )
                _ = x^
                _ = sv^
                _ = dual^
                _ = d_preds^
                return preds^
    var K = ctx.enqueue_create_buffer[DType.float32](n_batch * n_support)
    var nl2 = n_rows if kp.kernel == KERNEL_RBF else 1
    var l2_input = ctx.enqueue_create_buffer[DType.float32](nl2)
    var l2_support = ctx.enqueue_create_buffer[DType.float32](
        n_support if kp.kernel == KERNEL_RBF else 1
    )
    var wsf = kernel_workspace_floats(n_batch, n_support, n_cols)
    if wsf < 1:
        wsf = 1
    var ws = ctx.enqueue_create_buffer[DType.float32](wsf)
    ctx.synchronize()
    if kp.kernel == KERNEL_RBF:
        row_norms_l2sq(ctx, l2_input, x, n_rows, n_cols)
        row_norms_l2sq(ctx, l2_support, sv, n_support, n_cols)
        ctx.synchronize()
    var ptag = String("svm.predict.class.") if predict_class else String("svm.predict.decision.")
    card.record_device[DType.float32](ctx, ptag + "x", x, n_rows * n_cols)

    var i = 0
    var nb = n_batch
    while i < n_rows:
        if i + nb >= n_rows:
            nb = n_rows - i
        var xb = x.create_sub_buffer[DType.float32](i * n_cols, nb * n_cols)
        var l2b = l2_input.create_sub_buffer[DType.float32](
            i if kp.kernel == KERNEL_RBF else 0,
            nb if kp.kernel == KERNEL_RBF else 1,
        )
        kernel_op(ctx, kp, K, xb, sv, nb, n_support, n_cols, l2b, l2_support, ws)
        var pb = d_preds.create_sub_buffer[DType.float32](i, nb)
        ctx.enqueue_function[decision_kernel](
            pb.unsafe_ptr(), K.unsafe_ptr(), dual.unsafe_ptr(),
            Int32(n_support), Int32(nb), model.b,
            model.unique_labels[0], model.unique_labels[1],
            Int32(1) if predict_class else Int32(0),
            grid_dim=_grid(nb), block_dim=SEL_TPB,
        )
        ctx.synchronize()
        _ = xb^
        _ = l2b^
        _ = pb^
        i += nb
    ctx.synchronize()
    preds = read_f32(ctx, d_preds, n_rows)
    # DEVIATION 637 (row 39, FACT 2): a NaN decision value (float overflow
    # of `sum K * dual` with finite inputs) carries a vendor payload and
    # must not be recorded; raise before the record, as the solver does.
    for i in range(n_rows):
        if preds[i] != preds[i]:
            raise Error(
                "svm predict: NaN decision value at row " + String(i)
                + " (floating point overflow; DEVIATION 637)"
            )
    card.record_device[DType.float32](ctx, ptag + "out", d_preds, n_rows)
    _ = x^
    _ = sv^
    _ = dual^
    _ = K^
    _ = l2_input^
    _ = l2_support^
    _ = ws^
    _ = d_preds^
    return preds^
