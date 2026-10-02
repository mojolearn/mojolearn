# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer transport for the ops a Python-driven stack needs beside the
blocks: embedding forward/backward, RMSNorm forward/backward, the LM head
GEMM forward/backward, and the balanced-tree gradient accumulate of
optimizer contract clause 9.2.

NO NEW ARITHMETIC. Embedding is `embedding/checks/embedding_identical.mojo`,
RMSNorm is the llama forward kernel and `transformer_backward.mojo`'s
`bwd_rms_norm`, the head is `gemm/checks` at `OP_NT`, and the accumulate is
the contract's own `ftz(ftz(x) + ftz(y))` node over pairs in ascending
microbatch index. Every buffer lives for one call; no pointer is retained.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_mul_add
from embedding.checks.embedding_identical import (
    identical_embedding_backward_into,
    identical_embedding_forward_into,
)
from embedding.checks.embedding_oracle import EmbConfig
from gemm.checks.gemm_backward import (
    identical_gemm_backward_a_into,
    identical_gemm_backward_a_workspace_max_floats,
    identical_gemm_backward_b_into,
    identical_gemm_backward_b_workspace_max_floats,
)
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_NT
from training.checks.optimizer_oracle import microbatch_split_is_identical
from training.checks.loss_oracle import CeConfig, ce_count, ce_refuse_inputs
from training.estimator import identical_ce_admit_call, identical_ce_loss_resident
from transformer.checks.transformer_backward import bwd_rms_norm
from transformer.impl.llama.modeling_llama import (
    llama_rms_norm,
)


comptime SAMBA_TPB = 256


def _grid(n: Int) -> Int:
    var g = (n + SAMBA_TPB - 1) // SAMBA_TPB
    if g < 1:
        return 1
    return g


def _refuse_nonfinite(
    name: String, ptr: MutPointer[Float32, MutUntrackedOrigin], n: Int
) raises:
    for i in range(n):
        if not isfinite(ptr.unsafe_load(i)):
            raise Error(
                "mojolearn samba ops: non-finite " + name + " at flat index "
                + String(i)
            )


def _upload_f32(
    ctx: DeviceContext, ptr: MutPointer[Float32, MutUntrackedOrigin], n: Int
) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=ptr)
    return buf^


def _upload_i32(
    ctx: DeviceContext, ptr: MutPointer[Int32, MutUntrackedOrigin], n: Int
) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=ptr)
    return buf^


# ONE WAIT PER OP (lane/neural-apple, 2026-09-28): each op below enqueues its
# uploads, its kernels and its downloads on one in-order `ctx` and waits once,
# before the host reads the result (the caller's host arrays and every device
# buffer stay alive until that wait). The waits between those steps were
# host round trips only; on Apple each costs a command-buffer commit and
# completion. No kernel, operand or order changes.

# ===========================================================================
# EMBEDDING
# ===========================================================================


def samba_embedding_forward_host(
    ctx: DeviceContext,
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_positions: Int,
    vocab: Int,
    width: Int,
) raises -> Int:
    """`y[n_positions, width] = w[ids]`. Returns `n_positions * width`."""
    if n_positions < 1 or vocab < 1 or width < 1:
        raise Error("mojolearn samba ops: embedding shape must be positive")
    _refuse_nonfinite("embedding weight", w_ptr, vocab * width)
    var cells = n_positions * width
    var w = _upload_f32(ctx, w_ptr, vocab * width)
    var ids = _upload_i32(ctx, ids_ptr, n_positions)
    var y = ctx.enqueue_create_buffer[DType.float32](cells)
    var cfg = EmbConfig.llama(vocab, width)
    identical_embedding_forward_into(ctx, y, w, ids, n_positions, cfg)
    ctx.enqueue_copy(dst_ptr=y_ptr, src_buf=y)
    ctx.synchronize()
    _ = w^
    _ = ids^
    _ = y^
    return cells


def samba_embedding_backward_host(
    ctx: DeviceContext,
    dw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dy_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_positions: Int,
    vocab: Int,
    width: Int,
) raises -> Int:
    """`dw[vocab, width]`, a FRESH gradient (no accumulate, no padding row).
    Returns `vocab * width`."""
    if n_positions < 1 or vocab < 1 or width < 1:
        raise Error("mojolearn samba ops: embedding shape must be positive")
    _refuse_nonfinite("embedding upstream gradient", dy_ptr, n_positions * width)
    var cells = vocab * width
    var dy = _upload_f32(ctx, dy_ptr, n_positions * width)
    var ids = _upload_i32(ctx, ids_ptr, n_positions)
    var dw = ctx.enqueue_create_buffer[DType.float32](cells)
    var counts = ctx.enqueue_create_buffer[DType.int32](vocab)
    var run_begin = ctx.enqueue_create_buffer[DType.int32](vocab + 1)
    var perm = ctx.enqueue_create_buffer[DType.int32](n_positions)
    var cfg = EmbConfig.llama(vocab, width)
    identical_embedding_backward_into(
        ctx, dw, dy, ids, counts, run_begin, perm, n_positions, cfg
    )
    ctx.enqueue_copy(dst_ptr=dw_ptr, src_buf=dw)
    ctx.synchronize()
    _ = dy^
    _ = ids^
    _ = dw^
    _ = counts^
    _ = run_begin^
    _ = perm^
    return cells


# ===========================================================================
# RMSNORM
# ===========================================================================


def samba_rms_norm_forward_host(
    ctx: DeviceContext,
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m: Int,
    dm: Int,
    eps: Float32,
) raises -> Int:
    """`y = w * x * rsqrt(mean(x^2) + eps)` over `m` rows. Returns `m * dm`."""
    if m < 1 or dm < 1:
        raise Error("mojolearn samba ops: rms_norm shape must be positive")
    if not isfinite(eps) or eps < Float32(0.0):
        raise Error("mojolearn samba ops: rms_norm eps must be finite and >= 0")
    _refuse_nonfinite("rms_norm input", x_ptr, m * dm)
    _refuse_nonfinite("rms_norm weight", w_ptr, dm)
    var cells = m * dm
    var x = _upload_f32(ctx, x_ptr, cells)
    var w = _upload_f32(ctx, w_ptr, dm)
    var y = ctx.enqueue_create_buffer[DType.float32](cells)
    var sumsq = ctx.enqueue_create_buffer[DType.float32](m)
    llama_rms_norm(ctx, sumsq, y, x, w, m, dm, eps)
    ctx.enqueue_copy(dst_ptr=y_ptr, src_buf=y)
    ctx.synchronize()
    _ = x^
    _ = w^
    _ = y^
    _ = sumsq^
    return cells


def samba_rms_norm_backward_host(
    ctx: DeviceContext,
    dx_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dy_ptr: MutPointer[Float32, MutUntrackedOrigin],
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m: Int,
    dm: Int,
    eps: Float32,
) raises -> Int:
    """`dx[m, dm]` and `dw[dm]` for the forward above. The forward's row
    sum of squares is RECOMPUTED here by the same kernel, so the backward
    needs nothing cached. Returns `m * dm`."""
    if m < 1 or dm < 1:
        raise Error("mojolearn samba ops: rms_norm shape must be positive")
    if not isfinite(eps) or eps < Float32(0.0):
        raise Error("mojolearn samba ops: rms_norm eps must be finite and >= 0")
    var cells = m * dm
    _refuse_nonfinite("rms_norm input", x_ptr, cells)
    _refuse_nonfinite("rms_norm weight", w_ptr, dm)
    _refuse_nonfinite("rms_norm upstream gradient", dy_ptr, cells)
    var x = _upload_f32(ctx, x_ptr, cells)
    var w = _upload_f32(ctx, w_ptr, dm)
    var dy = _upload_f32(ctx, dy_ptr, cells)
    var y = ctx.enqueue_create_buffer[DType.float32](cells)
    var sumsq = ctx.enqueue_create_buffer[DType.float32](m)
    var dot_out = ctx.enqueue_create_buffer[DType.float32](m)
    var dx = ctx.enqueue_create_buffer[DType.float32](cells)
    var dw = ctx.enqueue_create_buffer[DType.float32](dm)
    var dh = ctx.enqueue_create_buffer[DType.float32](cells)
    var dprod = ctx.enqueue_create_buffer[DType.float32](cells)
    var rstd = ctx.enqueue_create_buffer[DType.float32](m)
    var dvcoef = ctx.enqueue_create_buffer[DType.float32](m)
    var ones = ctx.enqueue_create_buffer[DType.float32](m)
    var h_ones = ctx.enqueue_create_host_buffer[DType.float32](m)
    ctx.synchronize()
    for i in range(m):
        h_ones.unsafe_ptr().unsafe_store(i, Float32(1.0))
    ctx.enqueue_copy(dst_buf=ones, src_ptr=h_ones.unsafe_ptr())
    llama_rms_norm(ctx, sumsq, y, x, w, m, dm, eps)
    bwd_rms_norm[0](
        ctx, dot_out, dx, dw, dh, dprod, rstd, dvcoef, ones, dy, x, w, sumsq,
        dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), dx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), False, m, dm, eps,
    )
    ctx.enqueue_copy(dst_ptr=dx_ptr, src_buf=dx)
    ctx.enqueue_copy(dst_ptr=dw_ptr, src_buf=dw)
    ctx.synchronize()
    _ = h_ones^
    _ = x^
    _ = w^
    _ = dy^
    _ = y^
    _ = sumsq^
    _ = dot_out^
    _ = dx^
    _ = dw^
    _ = dh^
    _ = dprod^
    _ = rstd^
    _ = dvcoef^
    _ = ones^
    return cells


# ===========================================================================
# THE LM HEAD: C[m, n] = A[m, k] . W[n, k]^T  (OP_NT, torch's Linear)
# ===========================================================================


def samba_linear_forward_host(
    ctx: DeviceContext,
    c_ptr: MutPointer[Float32, MutUntrackedOrigin],
    a_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
) raises -> Int:
    """Returns `m * n`."""
    if m < 1 or n < 1 or k < 1:
        raise Error("mojolearn samba ops: linear shape must be positive")
    _refuse_nonfinite("linear input", a_ptr, m * k)
    _refuse_nonfinite("linear weight", w_ptr, n * k)
    var a = _upload_f32(ctx, a_ptr, m * k)
    var w = _upload_f32(ctx, w_ptr, n * k)
    var c = ctx.enqueue_create_buffer[DType.float32](m * n)
    var ws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(m, n, k)
    )
    identical_gemm_into(ctx, c, a, w, ws, m, n, k, OP_NT)
    ctx.enqueue_copy(dst_ptr=c_ptr, src_buf=c)
    ctx.synchronize()
    _ = a^
    _ = w^
    _ = c^
    _ = ws^
    return m * n


def samba_linear_backward_host(
    ctx: DeviceContext,
    da_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dc_ptr: MutPointer[Float32, MutUntrackedOrigin],
    a_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
) raises -> Int:
    """`da[m, k] = dc . W` and `dw[n, k] = dc^T . A`. The weight gradient's
    contraction is over `m`, the TOKEN count, which is what clause 9.2 is
    about. Returns `m * k`."""
    if m < 1 or n < 1 or k < 1:
        raise Error("mojolearn samba ops: linear shape must be positive")
    _refuse_nonfinite("linear input", a_ptr, m * k)
    _refuse_nonfinite("linear weight", w_ptr, n * k)
    _refuse_nonfinite("linear upstream gradient", dc_ptr, m * n)
    var a = _upload_f32(ctx, a_ptr, m * k)
    var w = _upload_f32(ctx, w_ptr, n * k)
    var dc = _upload_f32(ctx, dc_ptr, m * n)
    var da = ctx.enqueue_create_buffer[DType.float32](m * k)
    var dw = ctx.enqueue_create_buffer[DType.float32](n * k)
    var ws_a = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_backward_a_workspace_max_floats(OP_NT, m, n, k)
    )
    var ws_b = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_backward_b_workspace_max_floats(OP_NT, m, n, k)
    )
    identical_gemm_backward_a_into(ctx, da, dc, w, ws_a, m, n, k, OP_NT)
    identical_gemm_backward_b_into(ctx, dw, dc, a, ws_b, m, n, k, OP_NT)
    ctx.enqueue_copy(dst_ptr=da_ptr, src_buf=da)
    ctx.enqueue_copy(dst_ptr=dw_ptr, src_buf=dw)
    ctx.synchronize()
    _ = a^
    _ = w^
    _ = dc^
    _ = da^
    _ = dw^
    _ = ws_a^
    _ = ws_b^
    return m * k


# ===========================================================================
# THE HEAD, THE LOSS AND THE HEAD BACKWARD IN ONE CALL (lane/py-lm, 2026-09-28)
# ===========================================================================


def samba_head_loss_host(
    ctx: DeviceContext,
    loss_ptr: MutPointer[Float32, MutUntrackedOrigin],
    row_ptr: MutPointer[Float32, MutUntrackedOrigin],
    da_ptr: MutPointer[Float32, MutUntrackedOrigin],
    dw_ptr: MutPointer[Float32, MutUntrackedOrigin],
    a_ptr: MutPointer[Float32, MutUntrackedOrigin],
    w_ptr: MutPointer[Float32, MutUntrackedOrigin],
    targets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
    ignore_index: Int,
    reduction: Int,
    num_items: Int,
    label_smoothing: Float32,
) raises -> Int:
    """`samba_linear_forward_host`, then `identical_ce_loss_host` with a
    gradient, then `samba_linear_backward_host`, as ONE call: the calls
    `SambaStack.loss_and_grads` made one by one, the same kernels on the
    same operands in the same order, with the logits and their gradient
    kept on the device between them. Before, `logits` [m, n] crossed the
    bus four times per microbatch (down, up, `dlogits` down, up); here the
    logits come down ONCE, for the loss's own host refusal scan
    (`ce_refuse_inputs`, the copy `identical_ce_loss_host` also pays), and
    `dlogits` never leaves the device. Returns `count`.

    THE REFUSALS, in the order the three calls raised them: the head's
    nonfinite input and weight, the loss's call shape, its shape, logits
    and targets refusals. The head backward's own scans of its input and
    weight re-read the same unchanged bytes the forward already admitted.
    Its scan of the upstream gradient is not restated: `dlogits` is the
    softmax-minus-target weights (each in [-1, 1] for finite admitted
    logits) divided by `ce_divisor`, which is never below 1 (a zero MEAN
    count and a negative `num_items` are refused), so it is finite
    whenever the loss admitted its inputs."""
    if m < 1 or n < 1 or k < 1:
        raise Error("mojolearn samba ops: linear shape must be positive")
    _refuse_nonfinite("linear input", a_ptr, m * k)
    _refuse_nonfinite("linear weight", w_ptr, n * k)
    var cells = m * n
    var a = _upload_f32(ctx, a_ptr, m * k)
    var w = _upload_f32(ctx, w_ptr, n * k)
    var c = ctx.enqueue_create_buffer[DType.float32](cells)
    var ws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(m, n, k)
    )
    identical_gemm_into(ctx, c, a, w, ws, m, n, k, OP_NT)
    var h_c = ctx.enqueue_create_host_buffer[DType.float32](cells)
    ctx.enqueue_copy(dst_ptr=h_c.unsafe_ptr(), src_buf=c)
    ctx.synchronize()
    _ = ws^

    identical_ce_admit_call(reduction, 1, m)
    var cfg = CeConfig(n, ignore_index, reduction, label_smoothing, num_items)
    var h_logits = List[Float32](capacity=cells)
    var hp = h_c.unsafe_ptr()
    for i in range(cells):
        h_logits.append(hp.unsafe_load(i))
    var h_targets = List[Int32](capacity=m)
    for i in range(m):
        h_targets.append(targets_ptr.unsafe_load(i))
    _ = ce_refuse_inputs(h_logits, h_targets, cfg)
    var count = ce_count(h_targets, ignore_index)
    _ = h_logits^
    _ = h_targets^
    _ = h_c^

    var targets = _upload_i32(ctx, targets_ptr, m)
    ctx.synchronize()
    var dc = ctx.enqueue_create_buffer[DType.float32](cells)
    identical_ce_loss_resident(
        ctx, loss_ptr, row_ptr, dc, c, targets, m, count, reduction, 1, cfg,
    )

    var da = ctx.enqueue_create_buffer[DType.float32](m * k)
    var dw = ctx.enqueue_create_buffer[DType.float32](n * k)
    var ws_a = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_backward_a_workspace_max_floats(OP_NT, m, n, k)
    )
    var ws_b = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_backward_b_workspace_max_floats(OP_NT, m, n, k)
    )
    identical_gemm_backward_a_into(ctx, da, dc, w, ws_a, m, n, k, OP_NT)
    identical_gemm_backward_b_into(ctx, dw, dc, a, ws_b, m, n, k, OP_NT)
    ctx.enqueue_copy(dst_ptr=da_ptr, src_buf=da)
    ctx.enqueue_copy(dst_ptr=dw_ptr, src_buf=dw)
    ctx.synchronize()
    _ = a^
    _ = w^
    _ = c^
    _ = targets^
    _ = dc^
    _ = da^
    _ = dw^
    _ = ws_a^
    _ = ws_b^
    return count


# ===========================================================================
# THE BALANCED-TREE ACCUMULATE (optimizer contract clause 9.2, condition 5)
# ===========================================================================


def samba_tree_level_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    pairs_in: Int32,
):
    """`dst[j] = ftz(ftz(src[2j]) + ftz(src[2j+1]))` for `pairs` pieces of
    `n` floats each; one thread per output cell, no fold across threads."""
    var n = Int(n_in)
    var pairs = Int(pairs_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n * pairs:
        return
    var j = i // n
    var e = i - j * n
    var left = ftz(src.unsafe_load((2 * j) * n + e))
    var right = ftz(src.unsafe_load((2 * j + 1) * n + e))
    dst.unsafe_store(i, ftz(identical_mul_add(Float32(1.0), left, right)))


def samba_validate_accumulation(n: Int, a: Int, t_tokens: Int) raises:
    """Shared shape/alignment admission before any accumulation allocation."""
    if n < 1:
        raise Error("mojolearn samba ops: accumulate n must be at least 1")
    if a < 1:
        raise Error("mojolearn samba ops: accumulation_steps must be >= 1")
    var q = a
    while q > 1:
        if q % 2 != 0:
            raise Error(
                "mojolearn samba ops: accumulation_steps must be a POWER OF"
                " TWO (optimizer contract clause 9.2 condition 4), got "
                + String(a)
            )
        q = q // 2
    if t_tokens == 0:
        raise Error("mojolearn samba ops: t_tokens must be >= 1 or -1")
    if t_tokens > 0 and not microbatch_split_is_identical(t_tokens, a):
        raise Error(
            "mojolearn samba ops: MISALIGNED microbatch split, T = "
            + String(t_tokens) + " tokens at A = " + String(a)
            + " does not satisfy optimizer contract clause 9.2 (leaf size,"
            " T mod L, A divides P, A a power of two); this accumulation"
            " would be a different numerical experiment from the unsplit step"
        )


def samba_ftz_copy_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`dst[i] = ftz(src[i])`, one thread per cell: the `A == 1` accumulate
    (the tree of one piece is its seam)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, ftz(src.unsafe_load(i)))


def samba_accumulate_buffer(
    ctx: DeviceContext,
    var src: DeviceBuffer[DType.float32],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    a: Int,
) raises -> Int:
    """The balanced-tree accumulate over `a` consecutive blocks of `n` floats
    already resident in `src` (ascending microbatch index), written to the
    host `out_ptr`. Every level is one grid-wide launch; `A == 1` is one ftz
    copy on the device. The caller validated the shape. Returns `n`."""
    var dst = ctx.enqueue_create_buffer[DType.float32](n * max(1, a // 2))
    if a == 1:
        ctx.enqueue_function[samba_ftz_copy_kernel](
            dst.unsafe_ptr(), src.unsafe_ptr(), Int32(n),
            grid_dim=(_grid(n), 1, 1),
            block_dim=(SAMBA_TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=dst)
        ctx.synchronize()
        _ = src^
        _ = dst^
        return n
    var pieces = a
    var from_src = True
    while pieces > 1:
        var pairs = pieces // 2
        if from_src:
            ctx.enqueue_function[samba_tree_level_kernel](
                dst.unsafe_ptr(), src.unsafe_ptr(), Int32(n), Int32(pairs),
                grid_dim=(_grid(n * pairs), 1, 1),
                block_dim=(SAMBA_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[samba_tree_level_kernel](
                src.unsafe_ptr(), dst.unsafe_ptr(), Int32(n), Int32(pairs),
                grid_dim=(_grid(n * pairs), 1, 1),
                block_dim=(SAMBA_TPB, 1, 1),
            )
        from_src = not from_src
        pieces = pairs
    if from_src:
        var view = src.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=view)
        ctx.synchronize()
        _ = view
    else:
        var view = dst.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=view)
        ctx.synchronize()
        _ = view
    _ = src^
    _ = dst^
    return n


def samba_accumulate_host(
    ctx: DeviceContext,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    parts_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    a: Int,
    t_tokens: Int,
) raises -> Int:
    """`out[n] = tree(parts[0], ..., parts[a-1])`, `parts` being `a`
    consecutive blocks of `n` floats in ascending microbatch index, on the
    device (`samba_accumulate_buffer`); `host` names the caller's buffers.
    `t_tokens >= 1` asks for clause 9.2's alignment predicate and refuses a
    misaligned split BY NAME; `t_tokens < 0` makes no alignment claim (the
    residual or tied-weight pair add). Returns `n`."""
    samba_validate_accumulation(n, a, t_tokens)
    _refuse_nonfinite("accumulate parts", parts_ptr, n * a)
    var src = _upload_f32(ctx, parts_ptr, n * a)
    return samba_accumulate_buffer(ctx, src^, out_ptr, n, a)
