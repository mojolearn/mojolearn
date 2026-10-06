# SPDX-License-Identifier: Apache-2.0
"""Apple FAST candidates for the small MLP trainer (lane afn-mlp, 2026-10-03).

Every launch and every host entry here compiles ONLY under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` and
behind its own `-D MOJOLEARN_AFN_MLP_<NAME>` define (default OFF); the
IDENTICAL build and every other vendor compile `training/mlp_ops.mojo`'s
path unchanged.

THE STEP main's `mlp_train_step_host` pays on Apple (counted in
docs/apple-fast/notes/neural-mlp.md): 6 uploads, about 40 allocations,
2 vendor GEMMs + 2 pinned OP_TN GEMMs (each several launches) + 1 vendor
OP_NN GEMM, 4 `_mlp_kernel` launches, the logits' non-finite scan (a
launch, a readback and a wait), the cross-entropy's own launches and three
waits, the optimizer's clip-free step (launches, a readback, a wait), and
the final wait: about 20 launches and 7 waits for a 256 x 8 batch whose
arithmetic is ~100k FMAs. On Metal every launch costs ~20 us host time
plus ~0.25 us per live buffer, and a wait with a readback ~180 us, so the
step is transport, not arithmetic.

MOJOLEARN_AFN_MLP_FUSED_STEP   `mlp_fused_kernel`: forward (8x16 GEMV +
    bias + ReLU, 16x3 + bias), softmax cross-entropy and the backward
    (dlogits, ReLU mask, dhidden, dx) for ONE row per thread, the weights
    in threadgroup memory, each block of MLP_FT_ROWS rows folding its
    gradient partials (dw1, db1, dw2, db2, the mean loss) in threadgroup
    memory into one 196-float slot; `mlp_fold_adam_kernel` sums the
    (at most MLP_FT_MAX_BLOCKS) block slots in a free order and applies AdamW in the same
    thread. TWO launches, ONE wait, TWO allocations (one float arena with
    sub-buffer views, one int32 buffer for y) per step. f32 everywhere;
    the fold order is the free one FAST allows (per-block serial over the
    rows, then over the blocks).
MOJOLEARN_AFN_MLP_RESIDENT     the session (`mlp_resident_open_host` ..
    `mlp_resident_close_host`): parameters, gradients, the moments, a
    shadow of the three, the loss cells, the block slots and the batch
    staging live in ONE arena for the length of the trainer; a step
    uploads x and y, runs the two launches, downloads the loss, the logits
    and the gradients and waits once. The weights never travel per step
    (`mlp_resident_download_host` brings them to the host when the trainer
    is asked for its state). A step that fails the host's finiteness check
    restores p, m and v from the shadow on the device (one launch) before
    raising, so the published state is the pre-step one, as the
    transactional trainer promises.
MOJOLEARN_AFN_MLP_MULTISTEP    `k` minibatches per host call on the
    session: x and y for all k steps uploaded once, sliced per step on the
    device, 2k launches in one command stream, one wait. Implies RESIDENT.
MOJOLEARN_AFN_MLP_ALL          the three together.
"""
from std.atomic import Atomic, Ordering
from std.ffi import _Global
from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp, isfinite, log, sqrt
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from core.device_scan import device_first_nonfinite
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime _AFN_APPLE_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
comptime MLP_AFN_ALL = _AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MLP_ALL"]()
# T07 AFN26 alias: not tested in this campaign; default remains disabled.
comptime MLP_FUSED_STEP = _AFN_APPLE_FAST and (
    is_defined["MOJOLEARN_AFN_MLP_FUSED_STEP"]() or MLP_AFN_ALL
    or is_defined["MOJOLEARN_AFN26_MLP_FUSED_STEP"]()
)
# T08 AFN26 alias: not tested in this campaign; multistep implies resident.
comptime MLP_MULTISTEP = _AFN_APPLE_FAST and (
    is_defined["MOJOLEARN_AFN_MLP_MULTISTEP"]() or MLP_AFN_ALL
    or is_defined["MOJOLEARN_AFN26_MLP_MULTISTEP"]()
)
# T08 AFN26 alias: not tested in this campaign; resident is a separate arm.
comptime MLP_RESIDENT = _AFN_APPLE_FAST and (
    is_defined["MOJOLEARN_AFN_MLP_RESIDENT"]() or MLP_MULTISTEP
    or is_defined["MOJOLEARN_AFN26_MLP_RESIDENT"]()
)

#: The architecture, 8 -> 16 -> 3, and the flat registry `[w1, b1, w2, b2]`
#: (the same constants `training/mlp_ops.mojo` states; restated here so
#: that file can import this one without a cycle).
comptime FT_IN = 8
comptime FT_HID = 16
comptime FT_OUT = 3
comptime FT_W1 = FT_HID * FT_IN
comptime FT_OFF_B1 = FT_W1
comptime FT_OFF_W2 = FT_W1 + FT_HID
comptime FT_OFF_B2 = FT_OFF_W2 + FT_OUT * FT_HID
comptime FT_TOTAL = FT_OFF_B2 + FT_OUT
#: one block slot: the 195 gradient cells and the block's share of the loss
comptime FT_SLOT = FT_TOTAL + 1
# T12: not tested. One thread owns a row; 32 reduces per-block activation
# scratch and increases partials, while 128 reduces partials and increases
# shared storage. These fixed hardware/work alternatives do not dispatch on
# batch dimensions. Neither enables fused or resident execution by itself.
comptime MLP_FT_ROWS = (
    32 if _AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN26_MLP_ROWS32"]()
    else 128 if _AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN26_MLP_ROWS128"]()
    else 64
)
# Public per-step row capacity from mlp_ops; not tested with alternate tiles.
comptime MLP_FT_MAX_ROWS = 256
# not tested: resident steps reuse this slot region serially, so capacity is
# derived from one public step's row cap, not the multistep transport capacity.
comptime MLP_FT_MAX_BLOCKS = (MLP_FT_MAX_ROWS + MLP_FT_ROWS - 1) // MLP_FT_ROWS
#: `mlp_fold_adam_kernel`'s block
comptime FT_FOLD_TPB = 64

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _fp(mut buf: DeviceBuffer[DType.float32]) -> FP:
    return buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _ip(mut buf: DeviceBuffer[DType.int32]) -> IP:
    return buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def mlp_fused_kernel(
    x: FP,
    y: IP,
    p: FP,
    logits: FP,
    dx: FP,
    slots: FP,
    rows_in: Int32,
    inv_rows: Float32,
    want_dx: Int32,
    want_grads: Int32,
):
    """One row per thread, MLP_FT_ROWS rows per block. Forward, loss and
    backward in registers; the block's gradient partials folded in
    threadgroup memory and written to `slots[block * FT_SLOT ..]`."""
    var wsh = stack_allocation[FT_TOTAL, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_x = stack_allocation[MLP_FT_ROWS * FT_IN, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_act = stack_allocation[MLP_FT_ROWS * FT_HID, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_dh = stack_allocation[MLP_FT_ROWS * FT_HID, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_dl = stack_allocation[MLP_FT_ROWS * FT_OUT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_nll = stack_allocation[MLP_FT_ROWS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var rows = Int(rows_in)
    var t = tid
    while t < FT_TOTAL:
        wsh[t] = p.unsafe_load(t)
        t += MLP_FT_ROWS
    barrier()

    var r = Int(block_idx.x) * MLP_FT_ROWS + tid
    var valid = r < rows
    var grads = want_grads != Int32(0)
    var xr = InlineArray[Float32, FT_IN](fill=Float32(0))
    var act = InlineArray[Float32, FT_HID](fill=Float32(0))
    var lg = InlineArray[Float32, FT_OUT](fill=Float32(0))
    var dl = InlineArray[Float32, FT_OUT](fill=Float32(0))
    var dh = InlineArray[Float32, FT_HID](fill=Float32(0))
    var nll = Float32(0)
    if valid:
        comptime for i in range(FT_IN):
            xr[i] = x.unsafe_load(r * FT_IN + i)
        comptime for h in range(FT_HID):
            var a = wsh[FT_OFF_B1 + h]
            comptime for i in range(FT_IN):
                a += wsh[h * FT_IN + i] * xr[i]
            act[h] = a if a > Float32(0) else Float32(0)
        comptime for o in range(FT_OUT):
            var l = wsh[FT_OFF_B2 + o]
            comptime for h in range(FT_HID):
                l += wsh[FT_OFF_W2 + o * FT_HID + h] * act[h]
            lg[o] = l
            logits.unsafe_store(r * FT_OUT + o, l)
        if grads:
            var yi = Int(y.unsafe_load(r))
            var m = lg[0]
            comptime for o in range(1, FT_OUT):
                if lg[o] > m:
                    m = lg[o]
            var s = Float32(0)
            var e = InlineArray[Float32, FT_OUT](fill=Float32(0))
            comptime for o in range(FT_OUT):
                e[o] = exp(lg[o] - m)
                s += e[o]
            var ls = log(s)
            comptime for o in range(FT_OUT):
                var pr = e[o] / s
                if o == yi:
                    pr -= Float32(1)
                    nll = (ls - (lg[o] - m)) * inv_rows
                dl[o] = pr * inv_rows
            comptime for h in range(FT_HID):
                var g = Float32(0)
                if act[h] > Float32(0):
                    comptime for o in range(FT_OUT):
                        g += dl[o] * wsh[FT_OFF_W2 + o * FT_HID + h]
                dh[h] = g
            if want_dx != Int32(0):
                comptime for i in range(FT_IN):
                    var d = Float32(0)
                    comptime for h in range(FT_HID):
                        d += dh[h] * wsh[h * FT_IN + i]
                    dx.unsafe_store(r * FT_IN + i, d)
    if grads:
        # an out-of-range row stages zeros, so the fold reads a full tile
        comptime for i in range(FT_IN):
            s_x[tid * FT_IN + i] = xr[i]
        comptime for h in range(FT_HID):
            s_act[tid * FT_HID + h] = act[h]
            s_dh[tid * FT_HID + h] = dh[h]
        comptime for o in range(FT_OUT):
            s_dl[tid * FT_OUT + o] = dl[o]
        s_nll[tid] = nll
    barrier()
    if not grads:
        return
    var base = Int(block_idx.x) * FT_SLOT
    var cell = tid
    while cell < FT_SLOT:
        var acc = Float32(0)
        if cell < FT_OFF_B1:
            var h = cell // FT_IN
            var i = cell - h * FT_IN
            for rr in range(MLP_FT_ROWS):
                acc += s_dh[rr * FT_HID + h] * s_x[rr * FT_IN + i]
        elif cell < FT_OFF_W2:
            var h = cell - FT_OFF_B1
            for rr in range(MLP_FT_ROWS):
                acc += s_dh[rr * FT_HID + h]
        elif cell < FT_OFF_B2:
            var c = cell - FT_OFF_W2
            var o = c // FT_HID
            var h = c - o * FT_HID
            for rr in range(MLP_FT_ROWS):
                acc += s_dl[rr * FT_OUT + o] * s_act[rr * FT_HID + h]
        elif cell < FT_TOTAL:
            var o = cell - FT_OFF_B2
            for rr in range(MLP_FT_ROWS):
                acc += s_dl[rr * FT_OUT + o]
        else:
            for rr in range(MLP_FT_ROWS):
                acc += s_nll[rr]
        slots.unsafe_store(base + cell, acc)
        cell += MLP_FT_ROWS


def mlp_fold_adam_kernel(
    slots: FP,
    blocks_in: Int32,
    g_out: FP,
    loss_out: FP,
    p: FP,
    m_state: FP,
    v_state: FP,
    do_update: Int32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    c1: Float32,
    c2: Float32,
    step_size: Float32,
    rt_bc2: Float32,
    decay_mul: Float32,
):
    """Cell `i`: the gradient as the sum of the block slots (the loss for
    cell 195), then torch's AdamW on that cell when `do_update`:
    p *= 1 - lr wd; m = b1 m + (1 - b1) g; v = b2 v + (1 - b2) g^2;
    p -= (lr / bc1) * m / (sqrt(v) / sqrt(bc2) + eps). The same seams the
    IDENTICAL kernel spells (`training/checks/optimizer.mojo`), unpinned."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= FT_SLOT:
        return
    var blocks = Int(blocks_in)
    var g = Float32(0)
    for b in range(blocks):
        g += slots.unsafe_load(b * FT_SLOT + i)
    if i == FT_TOTAL:
        loss_out.unsafe_store(0, g)
        return
    g_out.unsafe_store(i, g)
    if do_update == Int32(0):
        return
    var pv = p.unsafe_load(i) * decay_mul
    var mv = beta1 * m_state.unsafe_load(i) + c1 * g
    var vv = beta2 * v_state.unsafe_load(i) + c2 * (g * g)
    var dn = sqrt(vv) / rt_bc2 + eps
    pv = pv - step_size * (mv / dn)
    p.unsafe_store(i, pv)
    m_state.unsafe_store(i, mv)
    v_state.unsafe_store(i, vv)


def mlp_copy_kernel(dst: FP, src: FP, n_in: Int32):
    """`dst[i] = src[i]` (the session's shadow and its restore)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(i, src.unsafe_load(i))


@fieldwise_init
struct _AdamScalars(Copyable, Movable):
    var c1: Float32
    var c2: Float32
    var step_size: Float32
    var rt_bc2: Float32
    var decay_mul: Float32


def _pow_f32(base: Float32, e_in: Int) -> Float32:
    var acc = Float32(1)
    var b = base
    var e = e_in
    while e > 0:
        if (e & 1) != 0:
            acc = acc * b
        b = b * b
        e = e >> 1
    return acc


def _adam_scalars(
    t: Int, lr: Float32, beta1: Float32, beta2: Float32, weight_decay: Float32
) -> _AdamScalars:
    """Contract 7.1's host scalars (`step_scalars`), unpinned."""
    var bc1 = Float32(1) - _pow_f32(beta1, t)
    var bc2 = Float32(1) - _pow_f32(beta2, t)
    return _AdamScalars(
        Float32(1) - beta1, Float32(1) - beta2, lr / bc1, sqrt(bc2),
        Float32(1) - lr * weight_decay,
    )


def _blocks_for(rows: Int) -> Int:
    return (rows + MLP_FT_ROWS - 1) // MLP_FT_ROWS


def _launch_fused(
    ctx: DeviceContext, x: FP, y: IP, p: FP, logits: FP, dx: FP, slots: FP,
    rows: Int, want_dx: Int, want_grads: Int,
) raises:
    ctx.enqueue_function[mlp_fused_kernel](
        x, y, p, logits, dx, slots, Int32(rows), Float32(1) / Float32(rows),
        Int32(want_dx), Int32(want_grads),
        grid_dim=(_blocks_for(rows), 1, 1), block_dim=(MLP_FT_ROWS, 1, 1),
    )


def _launch_fold_adam(
    ctx: DeviceContext, slots: FP, blocks: Int, g: FP, loss: FP, p: FP, m: FP, v: FP,
    do_update: Int, sc: _AdamScalars, beta1: Float32, beta2: Float32, eps: Float32,
) raises:
    ctx.enqueue_function[mlp_fold_adam_kernel](
        slots, Int32(blocks), g, loss, p, m, v, Int32(do_update),
        beta1, beta2, eps, sc.c1, sc.c2, sc.step_size, sc.rt_bc2, sc.decay_mul,
        grid_dim=((FT_SLOT + FT_FOLD_TPB - 1) // FT_FOLD_TPB, 1, 1),
        block_dim=(FT_FOLD_TPB, 1, 1),
    )


def mlp_flag_clear_kernel(flag: IP):
    """`flag[0] = 0` (the resident step's nonfinite word)."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        flag.unsafe_store(0, Int32(0))


def mlp_nonfinite_kernel(flag: IP, src: FP, n_in: Int32):
    """`flag[0] = 1` when any of `src[0:n]` is nonfinite (every writer
    stores the same 1, so the race is benign)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in) and not isfinite(src.unsafe_load(i)):
        flag.unsafe_store(0, Int32(1))


def _launch_nonfinite(ctx: DeviceContext, flag: IP, src: FP, n: Int) raises:
    ctx.enqueue_function[mlp_nonfinite_kernel](
        flag, src, Int32(n),
        grid_dim=((n + FT_FOLD_TPB - 1) // FT_FOLD_TPB, 1, 1), block_dim=(FT_FOLD_TPB, 1, 1),
    )


def _launch_copy(ctx: DeviceContext, dst: FP, src: FP, n: Int) raises:
    ctx.enqueue_function[mlp_copy_kernel](
        dst, src, Int32(n),
        grid_dim=((n + FT_FOLD_TPB - 1) // FT_FOLD_TPB, 1, 1), block_dim=(FT_FOLD_TPB, 1, 1),
    )


def _finite_or_raise(values: MutPointer[Float32, MutUntrackedOrigin], count: Int) raises:
    for i in range(count):
        if not isfinite(values.unsafe_load(i)):
            raise Error("small MLP operation has nonfinite input or output")


# ---------------------------------------------------------------------------
# MOJOLEARN_AFN_MLP_FUSED_STEP: the one-call step, every buffer per call.
# ---------------------------------------------------------------------------


def mlp_fast_step_host(
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
    """`mlp_train_step_host`'s contract (same arguments, same modes, same
    outputs, the refusals already raised by the caller) as two launches
    and one wait. Returns `rows * 3`."""
    comptime if not MLP_FUSED_STEP:
        raise Error("mlp_fast_step_host: compiled without MOJOLEARN_AFN_MLP_FUSED_STEP")
    else:
        var train = mode == 2
        var grads = mode != 0
        var blocks = _blocks_for(rows)
        var dx_n = rows * FT_IN if want_input_grad != 0 else 1
        # the arena: [x | p | g | m | v | logits | dx | loss | slots]
        var off_x = 0
        var off_p = off_x + rows * FT_IN
        var off_g = off_p + FT_TOTAL
        var off_m = off_g + FT_TOTAL
        var off_v = off_m + FT_TOTAL
        var off_lg = off_v + FT_TOTAL
        var off_dx = off_lg + rows * FT_OUT
        var off_loss = off_dx + dx_n
        var off_slots = off_loss + 1
        var total = off_slots + blocks * FT_SLOT
        var arena = ctx.enqueue_create_buffer[DType.float32](total)
        var y_d = ctx.enqueue_create_buffer[DType.int32](rows)
        var x_v = arena.create_sub_buffer[DType.float32](off_x, rows * FT_IN)
        var w1_v = arena.create_sub_buffer[DType.float32](off_p, FT_W1)
        var b1_v = arena.create_sub_buffer[DType.float32](off_p + FT_OFF_B1, FT_HID)
        var w2_v = arena.create_sub_buffer[DType.float32](off_p + FT_OFF_W2, FT_OUT * FT_HID)
        var b2_v = arena.create_sub_buffer[DType.float32](off_p + FT_OFF_B2, FT_OUT)
        var dw1_v = arena.create_sub_buffer[DType.float32](off_g, FT_W1)
        var db1_v = arena.create_sub_buffer[DType.float32](off_g + FT_OFF_B1, FT_HID)
        var dw2_v = arena.create_sub_buffer[DType.float32](off_g + FT_OFF_W2, FT_OUT * FT_HID)
        var db2_v = arena.create_sub_buffer[DType.float32](off_g + FT_OFF_B2, FT_OUT)
        var m_v = arena.create_sub_buffer[DType.float32](off_m, FT_TOTAL)
        var v_v = arena.create_sub_buffer[DType.float32](off_v, FT_TOTAL)
        var lg_v = arena.create_sub_buffer[DType.float32](off_lg, rows * FT_OUT)
        var dx_v = arena.create_sub_buffer[DType.float32](off_dx, dx_n)
        var loss_v = arena.create_sub_buffer[DType.float32](off_loss, 1)
        ctx.enqueue_copy(dst_buf=x_v, src_ptr=x_ptr)
        ctx.enqueue_copy(dst_buf=y_d, src_ptr=y_ptr)
        ctx.enqueue_copy(dst_buf=w1_v, src_ptr=w1_ptr)
        ctx.enqueue_copy(dst_buf=b1_v, src_ptr=b1_ptr)
        ctx.enqueue_copy(dst_buf=w2_v, src_ptr=w2_ptr)
        ctx.enqueue_copy(dst_buf=b2_v, src_ptr=b2_ptr)
        if train:
            ctx.enqueue_copy(dst_buf=m_v, src_ptr=m_ptr)
            ctx.enqueue_copy(dst_buf=v_v, src_ptr=v_ptr)
        var base = _fp(arena)
        _launch_fused(
            ctx, base + off_x, _ip(y_d), base + off_p, base + off_lg, base + off_dx,
            base + off_slots, rows, want_input_grad, 1 if grads else 0,
        )
        if grads:
            var sc = _adam_scalars(t if train else 1, lr, beta1, beta2, weight_decay)
            _launch_fold_adam(
                ctx, base + off_slots, blocks, base + off_g, base + off_loss,
                base + off_p, base + off_m, base + off_v, 1 if train else 0, sc,
                beta1, beta2, eps,
            )
        # cpu2-l11-neural (2026-10-04): the refusal of a non-finite logit,
        # loss, gradient or dx is a device scan here, before any result or
        # state is downloaded; the Python trainer no longer walks them.
        if device_first_nonfinite(ctx, lg_v, rows * FT_OUT) >= 0:
            raise Error("small MLP operation has nonfinite input or output")
        if grads:
            var g_all = arena.create_sub_buffer[DType.float32](off_g, FT_TOTAL)
            var bad_g = device_first_nonfinite(ctx, g_all, FT_TOTAL)
            _ = g_all^
            if bad_g >= 0 or device_first_nonfinite(ctx, loss_v, 1) >= 0:
                raise Error("small MLP operation has nonfinite input or output")
            if want_input_grad != 0 and device_first_nonfinite(ctx, dx_v, dx_n) >= 0:
                raise Error("small MLP operation has nonfinite input or output")
        ctx.enqueue_copy(dst_ptr=logits_ptr, src_buf=lg_v)
        if grads:
            ctx.enqueue_copy(dst_ptr=loss_ptr, src_buf=loss_v)
            ctx.enqueue_copy(dst_ptr=dw1_ptr, src_buf=dw1_v)
            ctx.enqueue_copy(dst_ptr=db1_ptr, src_buf=db1_v)
            ctx.enqueue_copy(dst_ptr=dw2_ptr, src_buf=dw2_v)
            ctx.enqueue_copy(dst_ptr=db2_ptr, src_buf=db2_v)
            if want_input_grad != 0:
                ctx.enqueue_copy(dst_ptr=dx_ptr, src_buf=dx_v)
        if train:
            ctx.enqueue_copy(dst_ptr=w1_ptr, src_buf=w1_v)
            ctx.enqueue_copy(dst_ptr=b1_ptr, src_buf=b1_v)
            ctx.enqueue_copy(dst_ptr=w2_ptr, src_buf=w2_v)
            ctx.enqueue_copy(dst_ptr=b2_ptr, src_buf=b2_v)
            ctx.enqueue_copy(dst_ptr=m_ptr, src_buf=m_v)
            ctx.enqueue_copy(dst_ptr=v_ptr, src_buf=v_v)
        ctx.synchronize()
        info_ptr.unsafe_store(0, Float32(0))
        info_ptr.unsafe_store(1, Float32(0))
        info_ptr.unsafe_store(2, Float32(0))
        for j in range(4):
            var flag = Int32(0)
            if flags_ptr.unsafe_load(j) != Int32(0):
                flag = Int32(1)
            flags_ptr.unsafe_store(j, flag)
        # Explicit ownership after the wait (the rule of mlp_ops.mojo).
        _ = x_v^
        _ = w1_v^
        _ = b1_v^
        _ = w2_v^
        _ = b2_v^
        _ = dw1_v^
        _ = db1_v^
        _ = dw2_v^
        _ = db2_v^
        _ = m_v^
        _ = v_v^
        _ = lg_v^
        _ = dx_v^
        _ = loss_v^
        _ = y_d^
        _ = arena^
        _finite_or_raise(logits_ptr, rows * FT_OUT)
        if grads:
            if not isfinite(loss_ptr.unsafe_load(0)):
                raise Error("small MLP train step: loss is not finite")
            _finite_or_raise(dw1_ptr, FT_W1)
            _finite_or_raise(db1_ptr, FT_HID)
            _finite_or_raise(dw2_ptr, FT_OUT * FT_HID)
            _finite_or_raise(db2_ptr, FT_OUT)
            if want_input_grad != 0:
                _finite_or_raise(dx_ptr, rows * FT_IN)
        return rows * FT_OUT


# ---------------------------------------------------------------------------
# MOJOLEARN_AFN_MLP_RESIDENT / _MULTISTEP: the session.
# ---------------------------------------------------------------------------

#: the resident arena: [p | g | m | v | shadow(p, g, m, v) | losses(cap_k) |
#: slots | x(cap_rows * 8) | logits(cap_rows * 3) | dx(cap_rows * 8)]
comptime RS_OFF_P = 0
comptime RS_OFF_G = FT_TOTAL
comptime RS_OFF_M = 2 * FT_TOTAL
comptime RS_OFF_V = 3 * FT_TOTAL
comptime RS_OFF_SHADOW = 4 * FT_TOTAL
comptime RS_STATE = 4 * FT_TOTAL
comptime RS_OFF_LOSS = RS_OFF_SHADOW + RS_STATE


struct _MlpSession(Defaultable, Movable):
    var arena: List[DeviceBuffer[DType.float32]]
    var ybuf: List[DeviceBuffer[DType.int32]]
    var cap_rows: List[Int]
    var cap_k: List[Int]
    var live: List[Bool]

    def __init__(out self):
        self.arena = List[DeviceBuffer[DType.float32]]()
        self.ybuf = List[DeviceBuffer[DType.int32]]()
        self.cap_rows = List[Int]()
        self.cap_k = List[Int]()
        self.live = List[Bool]()


comptime _MLP_SESSIONS = _Global[
    StorageType=_MlpSession, name="MojoAfnMlpSessionsFast", init_fn=_MlpSession.__init__
]


def _rs_offsets(cap_rows: Int, cap_k: Int) -> List[Int]:
    """[slots, x, logits, dx, total] offsets for a session's capacity."""
    var off_slots = RS_OFF_LOSS + cap_k
    var off_x = off_slots + MLP_FT_MAX_BLOCKS * FT_SLOT
    var off_lg = off_x + cap_rows * FT_IN
    var off_dx = off_lg + cap_rows * FT_OUT
    var total = off_dx + cap_rows * FT_IN
    var out = List[Int]()
    out.append(off_slots)
    out.append(off_x)
    out.append(off_lg)
    out.append(off_dx)
    out.append(total)
    return out^


def _rs_check(h: Int) raises:
    var s = _MLP_SESSIONS.get_or_create_ptr()
    if h < 0 or h >= len(s[].live) or not s[].live[h]:
        raise Error("small MLP resident session: unknown handle " + String(h))


def mlp_resident_open_host(
    ctx: DeviceContext,
    w1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m_ptr: MutPointer[Float32, MutUntrackedOrigin],
    v_ptr: MutPointer[Float32, MutUntrackedOrigin],
    cap_k: Int,
) raises -> Int:
    """Open a session holding the weights and moments; `cap_k` minibatches
    of up to 256 rows may be stepped per call. Returns the handle."""
    comptime if not MLP_RESIDENT:
        raise Error("mlp_resident_open_host: compiled without MOJOLEARN_AFN_MLP_RESIDENT")
    else:
        if cap_k < 1:
            raise Error("small MLP resident session: cap_k must be >= 1")
        var cap_rows = MLP_FT_MAX_ROWS * cap_k
        var offs = _rs_offsets(cap_rows, cap_k)
        var arena = ctx.enqueue_create_buffer[DType.float32](offs[4])
        # ybuf[cap_rows] is the step's nonfinite flag word.
        var ybuf = ctx.enqueue_create_buffer[DType.int32](cap_rows + 1)
        _rs_upload(ctx, arena, w1_ptr, b1_ptr, w2_ptr, b2_ptr, m_ptr, v_ptr)
        ctx.synchronize()
        var s = _MLP_SESSIONS.get_or_create_ptr()
        var h = -1
        for j in range(len(s[].live)):
            if not s[].live[j] and h < 0:
                h = j
        if h < 0:
            s[].arena.append(arena^)
            s[].ybuf.append(ybuf^)
            s[].cap_rows.append(cap_rows)
            s[].cap_k.append(cap_k)
            s[].live.append(True)
            h = len(s[].live) - 1
        else:
            s[].arena[h] = arena^
            s[].ybuf[h] = ybuf^
            s[].cap_rows[h] = cap_rows
            s[].cap_k[h] = cap_k
            s[].live[h] = True
        return h


def _rs_upload(
    ctx: DeviceContext,
    mut arena: DeviceBuffer[DType.float32],
    w1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m_ptr: MutPointer[Float32, MutUntrackedOrigin],
    v_ptr: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var w1_v = arena.create_sub_buffer[DType.float32](RS_OFF_P, FT_W1)
    var b1_v = arena.create_sub_buffer[DType.float32](RS_OFF_P + FT_OFF_B1, FT_HID)
    var w2_v = arena.create_sub_buffer[DType.float32](RS_OFF_P + FT_OFF_W2, FT_OUT * FT_HID)
    var b2_v = arena.create_sub_buffer[DType.float32](RS_OFF_P + FT_OFF_B2, FT_OUT)
    var m_v = arena.create_sub_buffer[DType.float32](RS_OFF_M, FT_TOTAL)
    var v_v = arena.create_sub_buffer[DType.float32](RS_OFF_V, FT_TOTAL)
    ctx.enqueue_copy(dst_buf=w1_v, src_ptr=w1_ptr)
    ctx.enqueue_copy(dst_buf=b1_v, src_ptr=b1_ptr)
    ctx.enqueue_copy(dst_buf=w2_v, src_ptr=w2_ptr)
    ctx.enqueue_copy(dst_buf=b2_v, src_ptr=b2_ptr)
    ctx.enqueue_copy(dst_buf=m_v, src_ptr=m_ptr)
    ctx.enqueue_copy(dst_buf=v_v, src_ptr=v_ptr)
    ctx.synchronize()
    _ = w1_v^
    _ = b1_v^
    _ = w2_v^
    _ = b2_v^
    _ = m_v^
    _ = v_v^


def mlp_resident_upload_host(
    ctx: DeviceContext,
    h: Int,
    w1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m_ptr: MutPointer[Float32, MutUntrackedOrigin],
    v_ptr: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """Replace the session's weights and moments (`load_state_dict`)."""
    comptime if not MLP_RESIDENT:
        raise Error("mlp_resident_upload_host: compiled without MOJOLEARN_AFN_MLP_RESIDENT")
    else:
        _rs_check(h)
        var s = _MLP_SESSIONS.get_or_create_ptr()
        _rs_upload(ctx, s[].arena[h], w1_ptr, b1_ptr, w2_ptr, b2_ptr, m_ptr, v_ptr)


def mlp_resident_download_host(
    ctx: DeviceContext,
    h: Int,
    w1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m_ptr: MutPointer[Float32, MutUntrackedOrigin],
    v_ptr: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """The session's weights and moments to the host (`state_dict`)."""
    comptime if not MLP_RESIDENT:
        raise Error("mlp_resident_download_host: compiled without MOJOLEARN_AFN_MLP_RESIDENT")
    else:
        _rs_check(h)
        var s = _MLP_SESSIONS.get_or_create_ptr()
        var w1_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_P, FT_W1)
        var b1_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_P + FT_OFF_B1, FT_HID)
        var w2_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_P + FT_OFF_W2, FT_OUT * FT_HID)
        var b2_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_P + FT_OFF_B2, FT_OUT)
        var m_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_M, FT_TOTAL)
        var v_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_V, FT_TOTAL)
        ctx.enqueue_copy(dst_ptr=w1_ptr, src_buf=w1_v)
        ctx.enqueue_copy(dst_ptr=b1_ptr, src_buf=b1_v)
        ctx.enqueue_copy(dst_ptr=w2_ptr, src_buf=w2_v)
        ctx.enqueue_copy(dst_ptr=b2_ptr, src_buf=b2_v)
        ctx.enqueue_copy(dst_ptr=m_ptr, src_buf=m_v)
        ctx.enqueue_copy(dst_ptr=v_ptr, src_buf=v_v)
        ctx.synchronize()
        _ = w1_v^
        _ = b1_v^
        _ = w2_v^
        _ = b2_v^
        _ = m_v^
        _ = v_v^


def mlp_resident_close_host(ctx: DeviceContext, h: Int) raises:
    """Release the session's arena (its views are dead after the wait every
    entry ended with); a one-cell buffer holds the slot."""
    comptime if not MLP_RESIDENT:
        raise Error("mlp_resident_close_host: compiled without MOJOLEARN_AFN_MLP_RESIDENT")
    else:
        var s = _MLP_SESSIONS.get_or_create_ptr()
        if h < 0 or h >= len(s[].live) or not s[].live[h]:
            return
        s[].live[h] = False
        s[].arena[h] = ctx.enqueue_create_buffer[DType.float32](1)
        s[].ybuf[h] = ctx.enqueue_create_buffer[DType.int32](1)
        s[].cap_rows[h] = 0
        s[].cap_k[h] = 0
        ctx.synchronize()


def mlp_resident_step_host(
    ctx: DeviceContext,
    h: Int,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    y_ptr: MutPointer[Int32, MutUntrackedOrigin],
    losses_ptr: MutPointer[Float32, MutUntrackedOrigin],
    logits_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dw1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    db1_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dw2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    db2_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dx_ptr: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int,
    k: Int,
    mode: Int,
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    want_input_grad: Int,
) raises -> Int:
    """`k` steps of `rows` rows each on session `h` (`k > 1` needs
    MOJOLEARN_AFN_MLP_MULTISTEP): `x` is `k * rows x 8`, `y` is `k * rows`,
    `losses` receives `k` mean losses, `logits`, the four gradients and
    `dx` are the LAST step's. `mode` 0 (forward, k must be 1), 1 (grads,
    k must be 1, the state unchanged) or 2 (train; step `t` is the first
    step's one-based number). A nonfinite result restores p, m and v from
    the shadow taken before the first step and raises. Returns
    `rows * 3`."""
    comptime if not MLP_RESIDENT:
        raise Error("mlp_resident_step_host: compiled without MOJOLEARN_AFN_MLP_RESIDENT")
    else:
        _rs_check(h)
        if rows < 1 or rows > MLP_FT_MAX_ROWS:
            raise Error("small MLP operations require rows 1..256 and cols 1..64")
        if mode < 0 or mode > 2:
            raise Error("small MLP train step: mode must be 0 (forward), 1 (grads) or 2 (train)")
        if k < 1:
            raise Error("small MLP resident step: k must be >= 1")
        if k > 1:
            comptime if not MLP_MULTISTEP:
                raise Error("small MLP resident step: k > 1 needs -D MOJOLEARN_AFN_MLP_MULTISTEP")
            if mode != 2:
                raise Error("small MLP resident step: k > 1 is for mode 2 (train) only")
        if mode == 2 and t < 1:
            raise Error("small MLP train step: t is ONE-BASED; the first step of a run is t = 1")
        if want_input_grad != 0 and want_input_grad != 1:
            raise Error("small MLP train step: want_input_grad must be 0 or 1")
        var s = _MLP_SESSIONS.get_or_create_ptr()
        if k > s[].cap_k[h] or k * rows > s[].cap_rows[h]:
            raise Error("small MLP resident step: k exceeds the session's capacity")
        _finite_or_raise(x_ptr, k * rows * FT_IN)
        if mode != 0:
            for i in range(k * rows):
                var yi = Int(y_ptr.unsafe_load(i))
                if yi < 0 or yi >= FT_OUT:
                    raise Error("small MLP train step: targets must be classes 0..2")
        var offs = _rs_offsets(s[].cap_rows[h], s[].cap_k[h])
        var off_slots = offs[0]
        var off_x = offs[1]
        var off_lg = offs[2]
        var off_dx = offs[3]
        var train = mode == 2
        var grads = mode != 0
        var blocks = _blocks_for(rows)
        var base = _fp(s[].arena[h])
        var yb = _ip(s[].ybuf[h])
        var x_v = s[].arena[h].create_sub_buffer[DType.float32](off_x, k * rows * FT_IN)
        var y_v = s[].ybuf[h].create_sub_buffer[DType.int32](0, k * rows)
        var lg_v = s[].arena[h].create_sub_buffer[DType.float32](off_lg, rows * FT_OUT)
        var dx_n = rows * FT_IN if want_input_grad != 0 else 1
        var dx_v = s[].arena[h].create_sub_buffer[DType.float32](off_dx, dx_n)
        var loss_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_LOSS, k)
        var dw1_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_G, FT_W1)
        var db1_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_G + FT_OFF_B1, FT_HID)
        var dw2_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_G + FT_OFF_W2, FT_OUT * FT_HID)
        var db2_v = s[].arena[h].create_sub_buffer[DType.float32](RS_OFF_G + FT_OFF_B2, FT_OUT)
        if train:
            _launch_copy(ctx, base + RS_OFF_SHADOW, base + RS_OFF_P, RS_STATE)
        ctx.enqueue_copy(dst_buf=x_v, src_ptr=x_ptr)
        if grads:
            ctx.enqueue_copy(dst_buf=y_v, src_ptr=y_ptr)
        for step in range(k):
            _launch_fused(
                ctx, base + off_x + step * rows * FT_IN, yb + step * rows, base + RS_OFF_P,
                base + off_lg, base + off_dx, base + off_slots, rows, want_input_grad,
                1 if grads else 0,
            )
            if grads:
                var sc = _adam_scalars(t + step if train else 1, lr, beta1, beta2, weight_decay)
                _launch_fold_adam(
                    ctx, base + off_slots, blocks, base + RS_OFF_G, base + RS_OFF_LOSS + step,
                    base + RS_OFF_P, base + RS_OFF_M, base + RS_OFF_V, 1 if train else 0, sc,
                    beta1, beta2, eps,
                )
        # The nonfinite check runs on the device: one flag word comes back.
        var flag = yb + s[].cap_rows[h]
        var flag_v = s[].ybuf[h].create_sub_buffer[DType.int32](s[].cap_rows[h], 1)
        ctx.enqueue_function[mlp_flag_clear_kernel](flag, grid_dim=(1, 1, 1), block_dim=(1, 1, 1))
        _launch_nonfinite(ctx, flag, base + off_lg, rows * FT_OUT)
        if grads:
            _launch_nonfinite(ctx, flag, base + RS_OFF_LOSS, k)
            _launch_nonfinite(ctx, flag, base + RS_OFF_G, FT_TOTAL)
            if want_input_grad != 0:
                _launch_nonfinite(ctx, flag, base + off_dx, rows * FT_IN)
        var flag_host = List[Int32](length=1, fill=Int32(0))
        ctx.enqueue_copy(dst_ptr=flag_host.unsafe_ptr(), src_buf=flag_v)
        ctx.enqueue_copy(dst_ptr=logits_ptr, src_buf=lg_v)
        if grads:
            ctx.enqueue_copy(dst_ptr=losses_ptr, src_buf=loss_v)
            ctx.enqueue_copy(dst_ptr=dw1_ptr, src_buf=dw1_v)
            ctx.enqueue_copy(dst_ptr=db1_ptr, src_buf=db1_v)
            ctx.enqueue_copy(dst_ptr=dw2_ptr, src_buf=dw2_v)
            ctx.enqueue_copy(dst_ptr=db2_ptr, src_buf=db2_v)
            if want_input_grad != 0:
                ctx.enqueue_copy(dst_ptr=dx_ptr, src_buf=dx_v)
        ctx.synchronize()
        _ = x_v^
        _ = y_v^
        _ = lg_v^
        _ = dx_v^
        _ = loss_v^
        _ = dw1_v^
        _ = db1_v^
        _ = dw2_v^
        _ = db2_v^
        _ = flag_v^
        var ok = flag_host[0] == 0
        if not ok:
            if train:
                _launch_copy(ctx, base + RS_OFF_P, base + RS_OFF_SHADOW, RS_STATE)
                ctx.synchronize()
            raise Error("small MLP operation has nonfinite input or output")
        return rows * FT_OUT
