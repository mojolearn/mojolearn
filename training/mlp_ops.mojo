# SPDX-License-Identifier: Apache-2.0
"""Bounded IDENTICAL FP32 GPU operations for the public small MLP.

C-row-major inputs; rows 1..256 and columns 1..64. Bias addition and
ascending row sums use checks.numerics' pinned FMA with multiplier one,
flushing operands/results at each seam. No atomics or vendor reductions.
ReLU returns positive values, otherwise +0; its derivative is zero at zero.
This is new surface arithmetic awaiting root-run numerical qualification.
Host pointers are borrowed for a synchronous call and never retained.

`mlp_train_step_host` (lane/apple-mlp-fused, 2026-09-30) is the whole
training step as ONE call: the same GEMMs, the same `_mlp_kernel` launches,
the same cross-entropy and the same optimizer step the Python trainer made
one binding call at a time, on the same operands in the same order, with the
intermediates kept on the device. Same cells, same order, same bits; what
changes is the count of host round trips a step pays (about forty
synchronizes across twelve calls before, seven in one call now), which is
the whole cost of this cell on a Metal box.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from gemm.checks.gemm_identical import (
    _fast_vendor_gemm,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_NN, OP_NT, OP_TN
from training.checks.loss_oracle import (
    CeConfig,
    IGNORE_INDEX_DEFAULT,
    REDUCTION_MEAN,
)
from core.device_scan import device_first_nonfinite
from training.checks.optimizer import (
    OPT_RECORD_INTERMEDIATES,
    SAB_CHUNKS,
    identical_optimizer_step,
    identical_optimizer_workspace_floats,
)
from training.checks.optimizer_oracle import OPT_ADAMW, OptimizerConfig
from training.estimator import (
    _refuse_hyperparameters,
    identical_ce_admit_call,
    identical_ce_loss_resident,
)

#: The public small MLP's fixed architecture, 8 -> 16 -> 3, and the flat
#: parameter layout `[w1 (16 x 8), b1 (16), w2 (3 x 16), b2 (3)]` the
#: optimizer steps over (`python/mojolearn/_mlp_impl.py::_SHAPES`, in that
#: order; the offsets are the registry `_offsets_for` builds from it).
comptime MLP_IN = 8
comptime MLP_HID = 16
comptime MLP_OUT = 3
comptime MLP_W1 = MLP_HID * MLP_IN
comptime MLP_B1 = MLP_HID
comptime MLP_W2 = MLP_OUT * MLP_HID
comptime MLP_B2 = MLP_OUT
comptime MLP_OFF_B1 = MLP_W1
comptime MLP_OFF_W2 = MLP_W1 + MLP_B1
comptime MLP_OFF_B2 = MLP_OFF_W2 + MLP_W2
comptime MLP_TOTAL = MLP_OFF_B2 + MLP_B2
comptime MLP_TENSORS = 4
#: `mlp_train_step_host` modes: the forward alone (logits), the forward and
#: the backward (loss, logits, the four gradients), or the whole step (the
#: AdamW update applied to the weights, `m`, `v` and the flags in place).
comptime MLP_STEP_FORWARD = 0
comptime MLP_STEP_GRADS = 1
comptime MLP_STEP_TRAIN = 2


def mlp_validate_shape(rows: Int, cols: Int) raises:
    comptime if GLOBAL_NUMERIC_MODE > NUMERIC_IDENTICAL:  # NUMERIC_DETERMINISTIC (2)
        raise Error("small MLP operations: no DETERMINISTIC tier (FAST or IDENTICAL)")
    if rows < 1 or rows > 256 or cols < 1 or cols > 64:
        raise Error("small MLP operations require rows 1..256 and cols 1..64")


def _finite(values: MutPointer[Float32, MutUntrackedOrigin], count: Int) raises:
    for i in range(count):
        if not isfinite(values.unsafe_load(i)):
            raise Error("small MLP operation has nonfinite input or output")


def _add(left: Float32, right: Float32) -> Float32:
    # Existing pinned FP32 arithmetic: fma(1, left, right) is a single
    # rounded addition. Both operands and the result use the IDENTICAL FTZ.
    return ftz(identical_mul_add(Float32(1), ftz(left), ftz(right)))


def _mlp_kernel(
    source: MutPointer[Float32, MutAnyOrigin],
    other: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    rows_arg: Int32, cols_arg: Int32, operation_arg: Int32,
):
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var operation = Int(operation_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if operation == 3:
        if i >= cols:
            return
        var value = Float32(0)
        # One GPU lane per column. The fold order does not depend on GPU
        # width, warp size, launch partition or scheduling.
        for row in range(rows):
            value = _add(value, source.unsafe_load(row * cols + i))
        output.unsafe_store(i, value)
        return
    if i >= rows * cols:
        return
    if operation == 2:
        var activation = source.unsafe_load(i)
        var value = Float32(0)
        if activation > Float32(0):
            value = ftz(other.unsafe_load(i))
        output.unsafe_store(i, value)
        return
    var value = _add(source.unsafe_load(i), other.unsafe_load(i % cols))
    if operation == 1 and value <= Float32(0):
        value = Float32(0)
    output.unsafe_store(i, value)


def _mlp_host(
    ctx: DeviceContext,
    input_ptr: MutPointer[Float32, MutUntrackedOrigin],
    other_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, cols: Int, operation: Int,
) raises -> Int:
    mlp_validate_shape(rows, cols)
    if operation < 0 or operation > 3:
        raise Error("invalid small MLP operation")
    var count = rows * cols
    var out_count = cols if operation == 3 else count
    var other_count = count if operation == 2 else cols
    if operation == 3:
        other_count = 1  # unused kernel operand; borrowed input placeholder
    _finite(input_ptr, count)
    if operation != 3:
        _finite(other_ptr, other_count)
    var source = ctx.enqueue_create_buffer[DType.float32](count)
    var other = ctx.enqueue_create_buffer[DType.float32](other_count)
    var output = ctx.enqueue_create_buffer[DType.float32](out_count)
    ctx.enqueue_copy(dst_buf=source, src_ptr=input_ptr)
    if operation != 3:
        ctx.enqueue_copy(dst_buf=other, src_ptr=other_ptr)
    # ONE wait per call, not three (lane/apple-mlp-fused, 2026-09-30). The
    # uploads, the launch and the download are queued on one in-order
    # context, so the waits that stood between them ordered nothing (the
    # argument of DEVIATION 2721; `samba_linear_forward_host` is the
    # precedent). The host pointers outlive the call, so the uploads may
    # read them whenever the queue reaches them. Same launch, same bits.
    _launch_mlp(
        ctx, source.unsafe_ptr(), other.unsafe_ptr(), output.unsafe_ptr(),
        rows, cols, operation,
    )
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=output)
    ctx.synchronize()
    # Explicit ownership after the drain prevents last-use buffer teardown
    # while a queued kernel or copy still references borrowed memory.
    _ = output^
    _ = other^
    _ = source^
    _finite(out_ptr, out_count)
    return out_count


def mlp_bias_activation_host(
    ctx: DeviceContext,
    input_ptr: MutPointer[Float32, MutUntrackedOrigin],
    bias_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, cols: Int, relu_flag: Int,
) raises -> Int:
    mlp_validate_shape(rows, cols)
    if relu_flag != 0 and relu_flag != 1:
        raise Error("mlp_bias_activation relu_flag must be 0 or 1")
    return _mlp_host(ctx, input_ptr, bias_ptr, out_ptr, rows, cols, relu_flag)


def mlp_relu_backward_host(
    ctx: DeviceContext,
    activation_ptr: MutPointer[Float32, MutUntrackedOrigin],
    incoming_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, cols: Int,
) raises -> Int:
    return _mlp_host(ctx, activation_ptr, incoming_ptr, out_ptr, rows, cols, 2)


def mlp_sum_rows_host(
    ctx: DeviceContext,
    input_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, cols: Int,
) raises -> Int:
    return _mlp_host(ctx, input_ptr, input_ptr, out_ptr, rows, cols, 3)


def _launch_mlp[
    source_origin: MutOrigin, other_origin: MutOrigin, output_origin: MutOrigin
](
    ctx: DeviceContext,
    source: MutPointer[Float32, source_origin],
    other: MutPointer[Float32, other_origin],
    output: MutPointer[Float32, output_origin],
    rows: Int, cols: Int, operation: Int,
) raises:
    """`_mlp_kernel` at the geometry every caller uses: 128 threads a block,
    one thread per output cell (per column for the row sum)."""
    var out_count = cols if operation == 3 else rows * cols
    ctx.enqueue_function[_mlp_kernel](
        source, other, output, Int32(rows), Int32(cols), Int32(operation),
        grid_dim=((out_count + 127) // 128, 1, 1), block_dim=(128, 1, 1),
    )


def _gemm(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int, n: Int, k: Int, op: Int,
) raises:
    """`identical_gemm`'s dispatch (the linalg binding's `matmul`, which the
    per-operation trainer calls) without its own wait: under FAST the vendor
    kernel first (DEVIATION 1876), else the profile's plan for the shape
    with the caller's workspace, which `identical_gemm_workspace_max_floats`
    sized for every shape this file runs. The caller waits once at the end;
    the buffers are the caller's and outlive that wait."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        if _fast_vendor_gemm(ctx, c, a, b, m, n, k, op):
            return
    identical_gemm_into(ctx, c, a, b, ws, m, n, k, op)


def _max_ws(mut ws_n: Int, m: Int, n: Int, k: Int):
    var w = identical_gemm_workspace_max_floats(m, n, k)
    if w > ws_n:
        ws_n = w


def mlp_train_step_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    y_ptr: MutPointer[Int32, MutUntrackedOrigin],
    w1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m_ptr: MutPointer[Float32, MutUntrackedOrigin],
    v_ptr: MutPointer[Float32, MutUntrackedOrigin],
    flags_ptr: MutPointer[Int32, MutUntrackedOrigin],
    loss_ptr: MutPointer[Float32, MutUntrackedOrigin],
    logits_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dw1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    db1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dw2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    db2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dx_ptr: MutPointer[Float32, MutUntrackedOrigin],
    info_ptr: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int,
    mode: Int,
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    want_input_grad: Int,
) raises -> Int:
    """The small MLP's forward, backward and AdamW step as ONE device call
    (lane/apple-mlp-fused, 2026-09-30). Returns `rows * 3`, the logits
    written.

    THE SAME CALLS THE PYTHON TRAINER MADE, IN THE SAME ORDER. Each line
    below names the `SmallMLPTrainer` call it stands for; the kernels, the
    operands, the shapes and the profile dispatch (`choose_gemm_plan` on the
    same (m, n, k)) are the per-operation path's, so every cell is the same
    arithmetic in the same order and the bits do not move. What moves is
    the transport: the twelve calls paid about forty synchronizes and about
    sixty buffer allocations a step; this one pays seven waits (the logits
    download for the loss's host refusal scan, the loss's own two, its
    transport out, the optimizer's refusal read-back and its wait, and the
    final one) and about forty allocations. On a Metal box, where a host
    round trip is a few hundred microseconds, that is the whole step.

        forward     pre1 = x . w1^T (OP_NT), act = relu(pre1 + b1),
                    pre2 = act . w2^T (OP_NT), logits = pre2 + b2
        loss        mean cross-entropy over classes 0..2 with its gradient
                    (`identical_ce_loss_resident`, after `ce_refuse_inputs`'s
                    refusals, the logits' non-finite scan on the device)
        backward    dw2 = dlogits^T . act (OP_TN), db2 = sum_rows(dlogits),
                    incoming = dlogits . w2 (OP_NN), dhidden = relu'(act)
                    incoming, dw1 = dhidden^T . x (OP_TN), db1 =
                    sum_rows(dhidden), and dx = dhidden . w1 (OP_NN) when
                    `want_input_grad`
        step        `identical_optimizer_step` (AdamW, no clip) over the
                    flat `[w1, b1, w2, b2]` registry, the weights, `m`, `v`
                    and the flags written back in place

    `mode` is `MLP_STEP_FORWARD` (the logits alone; `y` is unread),
    `MLP_STEP_GRADS` (the loss, the logits and the four gradients; the
    weights are read only) or `MLP_STEP_TRAIN` (all of it; `t` is the
    optimizer's ONE-BASED step number, the first step of a run is 1).

    THE REFUSALS the per-operation path raised are raised here: a shape
    outside the bound, a non-finite input, weight, state or output, a target
    outside 0..2, a non-finite loss, a non-finite hyperparameter (the
    optimizer's own `_refuse_hyperparameters`), and the certified entries'
    own (contract 8a on the device). The wording differs; the outcome, an
    error before any state is published, is the same, and the Python
    trainer re-validates the state it receives either way.

    THE BUFFERS. `x` is `rows x 8`, `y` is `rows` int32 classes, the four
    weights are `16 x 8`, `16`, `3 x 16`, `3` (read, and WRITTEN IN PLACE
    under `MLP_STEP_TRAIN`), `m` and `v` are 195 each (read and written
    under `MLP_STEP_TRAIN`; may be any one-element buffer otherwise), the
    flags are 4 int32 (Adam leaves them unchanged, written back as read),
    `loss` is one float, `logits` is `rows x 3`, the gradients are the
    weights' shapes, `dx` is `rows x 8` when `want_input_grad` (else any
    one-element buffer), `info` is the optimizer's three floats, all +0.0
    here because the clip never runs.
    """
    mlp_validate_shape(rows, MLP_HID)
    if mode < MLP_STEP_FORWARD or mode > MLP_STEP_TRAIN:
        raise Error("small MLP train step: mode must be 0 (forward), 1 (grads) or 2 (train)")
    if want_input_grad != 0 and want_input_grad != 1:
        raise Error("small MLP train step: want_input_grad must be 0 or 1")
    if mode == MLP_STEP_TRAIN and t < 1:
        raise Error("small MLP train step: t is ONE-BASED; the first step of a run is t = 1")
    _finite(x_ptr, rows * MLP_IN)
    _finite(w1_ptr, MLP_W1)
    _finite(b1_ptr, MLP_B1)
    _finite(w2_ptr, MLP_W2)
    _finite(b2_ptr, MLP_B2)
    if mode != MLP_STEP_FORWARD:
        for i in range(rows):
            var yi = Int(y_ptr.unsafe_load(i))
            if yi < 0 or yi >= MLP_OUT:
                raise Error("small MLP train step: targets must be classes 0..2")
    var cfg_opt = OptimizerConfig(
        OPT_ADAMW, lr, beta1, beta2, eps, weight_decay,
        Float32(0.0), Float32(0.0), False, Float32(0.0),
    )
    if mode == MLP_STEP_TRAIN:
        _refuse_hyperparameters(cfg_opt)
        _finite(m_ptr, MLP_TOTAL)
        _finite(v_ptr, MLP_TOTAL)

    # One workspace for every GEMM of the step: the max of the certified
    # sizer over the shapes, and the GEMMs run one after another on the one
    # in-order context. A fresh (unzeroed) buffer per GEMM is what the
    # per-operation path handed each one, so no plan reads what a previous
    # GEMM left there.
    var ws_n = 1
    _max_ws(ws_n, rows, MLP_HID, MLP_IN)
    _max_ws(ws_n, rows, MLP_OUT, MLP_HID)
    if mode != MLP_STEP_FORWARD:
        _max_ws(ws_n, MLP_OUT, MLP_HID, rows)
        _max_ws(ws_n, rows, MLP_HID, MLP_OUT)
        _max_ws(ws_n, MLP_HID, MLP_IN, rows)
        if want_input_grad != 0:
            _max_ws(ws_n, rows, MLP_IN, MLP_HID)

    # ---- Transport in: the batch, the targets, the flat parameters (the
    # four weights land in their registry slots through sub-buffer views,
    # the pattern of sequence/exec_device.mojo). Nothing waits here.
    var x_d = ctx.enqueue_create_buffer[DType.float32](rows * MLP_IN)
    var y_d = ctx.enqueue_create_buffer[DType.int32](rows)
    var p_d = ctx.enqueue_create_buffer[DType.float32](MLP_TOTAL)
    var w1_v = p_d.create_sub_buffer[DType.float32](0, MLP_W1)
    var b1_v = p_d.create_sub_buffer[DType.float32](MLP_OFF_B1, MLP_B1)
    var w2_v = p_d.create_sub_buffer[DType.float32](MLP_OFF_W2, MLP_W2)
    var b2_v = p_d.create_sub_buffer[DType.float32](MLP_OFF_B2, MLP_B2)
    ctx.enqueue_copy(dst_buf=x_d, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=y_d, src_ptr=y_ptr)
    ctx.enqueue_copy(dst_buf=w1_v, src_ptr=w1_ptr)
    ctx.enqueue_copy(dst_buf=b1_v, src_ptr=b1_ptr)
    ctx.enqueue_copy(dst_buf=w2_v, src_ptr=w2_ptr)
    ctx.enqueue_copy(dst_buf=b2_v, src_ptr=b2_ptr)
    var ws = ctx.enqueue_create_buffer[DType.float32](ws_n)
    var pre1 = ctx.enqueue_create_buffer[DType.float32](rows * MLP_HID)
    var act = ctx.enqueue_create_buffer[DType.float32](rows * MLP_HID)
    var pre2 = ctx.enqueue_create_buffer[DType.float32](rows * MLP_OUT)
    var logits = ctx.enqueue_create_buffer[DType.float32](rows * MLP_OUT)

    # ---- Forward: `_forward` (two `_matmul(..., transpose_b=True)`, two
    # `_bias`).
    _gemm(ctx, pre1, x_d, w1_v, ws, rows, MLP_HID, MLP_IN, OP_NT)
    _launch_mlp(
        ctx, pre1.unsafe_ptr(), p_d.unsafe_ptr() + MLP_OFF_B1, act.unsafe_ptr(),
        rows, MLP_HID, 1,
    )
    _gemm(ctx, pre2, act, w2_v, ws, rows, MLP_OUT, MLP_HID, OP_NT)
    _launch_mlp(
        ctx, pre2.unsafe_ptr(), p_d.unsafe_ptr() + MLP_OFF_B2, logits.unsafe_ptr(),
        rows, MLP_OUT, 0,
    )
    # The loss's refusal scan of the logits runs on the device (one scan
    # launch and its partials read back; cpu-gpu-cleanup n-train-mamba), and
    # the logits come down once, as a result, behind it on the same queue.
    if device_first_nonfinite(ctx, logits, rows * MLP_OUT) >= 0:
        raise Error("small MLP operation has nonfinite input or output")
    ctx.enqueue_copy(dst_ptr=logits_ptr, src_buf=logits)
    info_ptr.unsafe_store(0, Float32(0.0))
    info_ptr.unsafe_store(1, Float32(0.0))
    info_ptr.unsafe_store(2, Float32(0.0))
    if mode == MLP_STEP_FORWARD:
        ctx.synchronize()
        _ = x_d^
        _ = y_d^
        _ = w1_v^
        _ = b1_v^
        _ = w2_v^
        _ = b2_v^
        _ = p_d^
        _ = ws^
        _ = pre1^
        _ = act^
        _ = pre2^
        _ = logits^
        return rows * MLP_OUT

    # ---- Loss: `_training_impl.cross_entropy(logits, y, reduction='mean',
    # return_grad=True)`, which is `ce_loss_binding` -> `identical_ce_loss_host`:
    # the admit, the refusals, `ce_count`, then the resident half with
    # `dlogits` left on the device. The refusals ran above: the logits' scan on
    # the device, the shape by construction (`rows x 3` against `rows`), the
    # targets as classes 0..2. So no target equals the ignore index and
    # `ce_count` is `rows`, exactly.
    identical_ce_admit_call(REDUCTION_MEAN, 1, rows)
    var cfg = CeConfig(MLP_OUT, IGNORE_INDEX_DEFAULT, REDUCTION_MEAN, Float32(0.0), 0)
    var count = rows
    var h_row = List[Float32](capacity=rows)
    for _ in range(rows):
        h_row.append(Float32(0.0))
    var dlogits = ctx.enqueue_create_buffer[DType.float32](rows * MLP_OUT)
    identical_ce_loss_resident(
        ctx, loss_ptr, h_row.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        dlogits, logits, y_d, rows, count, REDUCTION_MEAN, 1, cfg,
    )
    _ = h_row^
    if not isfinite(loss_ptr.unsafe_load(0)):
        raise Error("small MLP train step: loss is not finite")

    # ---- Backward: `_gradient`'s six calls in its order, the gradients
    # landing in their registry slots of one flat buffer, which is what
    # `_pack` built for the optimizer.
    var g_d = ctx.enqueue_create_buffer[DType.float32](MLP_TOTAL)
    var dw1_v = g_d.create_sub_buffer[DType.float32](0, MLP_W1)
    var db1_v = g_d.create_sub_buffer[DType.float32](MLP_OFF_B1, MLP_B1)
    var dw2_v = g_d.create_sub_buffer[DType.float32](MLP_OFF_W2, MLP_W2)
    var db2_v = g_d.create_sub_buffer[DType.float32](MLP_OFF_B2, MLP_B2)
    var incoming = ctx.enqueue_create_buffer[DType.float32](rows * MLP_HID)
    var dhidden = ctx.enqueue_create_buffer[DType.float32](rows * MLP_HID)
    _gemm(ctx, dw2_v, dlogits, act, ws, MLP_OUT, MLP_HID, rows, OP_TN)
    _launch_mlp(
        # operation 3 never reads `other`; `incoming` is a distinct placeholder
        ctx, dlogits.unsafe_ptr(), incoming.unsafe_ptr(), g_d.unsafe_ptr() + MLP_OFF_B2,
        rows, MLP_OUT, 3,
    )
    _gemm(ctx, incoming, dlogits, w2_v, ws, rows, MLP_HID, MLP_OUT, OP_NN)
    _launch_mlp(
        ctx, act.unsafe_ptr(), incoming.unsafe_ptr(), dhidden.unsafe_ptr(),
        rows, MLP_HID, 2,
    )
    _gemm(ctx, dw1_v, dhidden, x_d, ws, MLP_HID, MLP_IN, rows, OP_TN)
    _launch_mlp(
        # operation 3 never reads `other`; `incoming` is a distinct placeholder
        ctx, dhidden.unsafe_ptr(), incoming.unsafe_ptr(), g_d.unsafe_ptr() + MLP_OFF_B1,
        rows, MLP_HID, 3,
    )
    var dx_n = 1
    if want_input_grad != 0:
        dx_n = rows * MLP_IN
    var dx_d = ctx.enqueue_create_buffer[DType.float32](dx_n)
    if want_input_grad != 0:
        _gemm(ctx, dx_d, dhidden, w1_v, ws, rows, MLP_IN, MLP_HID, OP_NN)
        ctx.enqueue_copy(dst_ptr=dx_ptr, src_buf=dx_d)
    # The gradients are a result whatever the mode; the optimizer (no clip)
    # never writes them, so their download may sit ahead of it in the queue.
    ctx.enqueue_copy(dst_ptr=dw1_ptr, src_buf=dw1_v)
    ctx.enqueue_copy(dst_ptr=db1_ptr, src_buf=db1_v)
    ctx.enqueue_copy(dst_ptr=dw2_ptr, src_buf=dw2_v)
    ctx.enqueue_copy(dst_ptr=db2_ptr, src_buf=db2_v)

    if mode == MLP_STEP_TRAIN:
        # ---- Step: `AdamW.step(grads)` -> `optimizer_step_binding` ->
        # `identical_optimizer_step_host`, with the flat parameters and
        # gradients already on the device. The buffers are the host
        # wrapper's, sized as it sizes them.
        var m_d = ctx.enqueue_create_buffer[DType.float32](MLP_TOTAL)
        var v_d = ctx.enqueue_create_buffer[DType.float32](MLP_TOTAL)
        ctx.enqueue_copy(dst_buf=m_d, src_ptr=m_ptr)
        ctx.enqueue_copy(dst_buf=v_d, src_ptr=v_ptr)
        var record_n = 1
        comptime if OPT_RECORD_INTERMEDIATES:
            record_n = MLP_TOTAL
        var denom_out = ctx.enqueue_create_buffer[DType.float32](record_n)
        var q_out = ctx.enqueue_create_buffer[DType.float32](record_n)
        var sumsq = ctx.enqueue_create_buffer[DType.float32](MLP_TENSORS)
        var norms = ctx.enqueue_create_buffer[DType.float32](MLP_TENSORS)
        var total_cell = ctx.enqueue_create_buffer[DType.float32](1)
        var out2 = ctx.enqueue_create_buffer[DType.float32](2)
        var offsets = List[Int]()
        offsets.append(0)
        offsets.append(MLP_OFF_B1)
        offsets.append(MLP_OFF_W2)
        offsets.append(MLP_OFF_B2)
        offsets.append(MLP_TOTAL)
        var ws_opt = ctx.enqueue_create_buffer[DType.float32](
            identical_optimizer_workspace_floats(offsets)
        )
        var sab_partials = ctx.enqueue_create_buffer[DType.float32](SAB_CHUNKS)
        var buf_initialized = List[Bool]()
        for j in range(MLP_TENSORS):
            buf_initialized.append(flags_ptr.unsafe_load(j) != Int32(0))
        identical_optimizer_step(
            ctx, p_d, g_d, m_d, v_d, denom_out, q_out, sumsq, norms,
            total_cell, out2, ws_opt, sab_partials, buf_initialized, offsets,
            cfg_opt, t,
        )
        # ---- Transport out: the new weights into the caller's four arrays
        # (what `_unpack_into` did on the host), the moments, the flags.
        ctx.enqueue_copy(dst_ptr=w1_ptr, src_buf=w1_v)
        ctx.enqueue_copy(dst_ptr=b1_ptr, src_buf=b1_v)
        ctx.enqueue_copy(dst_ptr=w2_ptr, src_buf=w2_v)
        ctx.enqueue_copy(dst_ptr=b2_ptr, src_buf=b2_v)
        ctx.enqueue_copy(dst_ptr=m_ptr, src_buf=m_d)
        ctx.enqueue_copy(dst_ptr=v_ptr, src_buf=v_d)
        ctx.synchronize()
        for j in range(MLP_TENSORS):
            var flag = Int32(0)
            if buf_initialized[j]:
                flag = Int32(1)
            flags_ptr.unsafe_store(j, flag)
        _ = m_d^
        _ = v_d^
        _ = denom_out^
        _ = q_out^
        _ = sumsq^
        _ = norms^
        _ = total_cell^
        _ = out2^
        _ = ws_opt^
        _ = sab_partials^
    else:
        ctx.synchronize()

    # The per-operation path checked every gradient as it came back
    # (`_gradient`'s `_array(..., ' gradient')` and `_mlp_host`'s `_finite`).
    _finite(dw1_ptr, MLP_W1)
    _finite(db1_ptr, MLP_B1)
    _finite(dw2_ptr, MLP_W2)
    _finite(db2_ptr, MLP_B2)
    if want_input_grad != 0:
        _finite(dx_ptr, rows * MLP_IN)

    # Explicit ownership after the drain (the file's rule above).
    _ = x_d^
    _ = y_d^
    _ = w1_v^
    _ = b1_v^
    _ = w2_v^
    _ = b2_v^
    _ = p_d^
    _ = ws^
    _ = pre1^
    _ = act^
    _ = pre2^
    _ = logits^
    _ = dlogits^
    _ = dw1_v^
    _ = db1_v^
    _ = dw2_v^
    _ = db2_v^
    _ = g_d^
    _ = incoming^
    _ = dhidden^
    _ = dx_d^
    return rows * MLP_OUT
