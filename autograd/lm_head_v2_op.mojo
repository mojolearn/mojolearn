# SPDX-License-Identifier: Apache-2.0
"""The chunked LM head (`LanguageModelConfig(chunked_lm_head_v2=True)`) as one
executor op with its backward rule (lane m2-neural-leftovers, 2026-11-09).

    loss = lm_head_v2(hidden (rows, width), weight (vocab, width), codes (rows, 1))
           OPS_LM_HEAD_V2

is the mean next-token cross-entropy of `hidden @ weight^T` against the class
codes, computed without the (rows, vocab) logits: per row, the maximum, the
denominator and the row loss over vocabulary panels
(training/chunked_lm_head_v2.mojo `chunked_lm_head_v2_row_body`), then the
fixed leaf-and-tree mean of the row losses. The recorded op is one tape
custom node (`LM_HEAD_V2_TAG`) over [hidden, weight]; its VJP
(`lm_head_v2_backward`, reached through autograd/rules_step_ops.mojo
StepOpRules) recomputes each logit from the saved row statistics: dHidden one
row owner per thread, dWeight one vocabulary owner per thread, both folds
ascending, each scaled by the upstream gradient. Memory: three row vectors and
the int32 codes, no logits.

One algorithm on every column: the device kernels and the host arm call the
same @always_inline bodies (`chunked_lm_head_v2_row_body`,
`chunked_lm_head_v2_dhidden_body`, `chunked_lm_head_v2_dweight_body`); the
host arm splits rows (and vocabulary entries) across cores, the fold inside
each owner is unchanged. The codes are class indices as float values (the
cross-entropy op's convention); a code that names no vocabulary entry (the
trainer's -100 targets) contributes `log(denominator)`, as the trainer's
chunked head always did.
"""
from std.gpu import block_dim, block_idx, thread_idx
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from checks.numerics import ftz, identical_div, identical_mul
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from autograd.block_step_ops import block_buffer_view
from training.chunked_lm_head_v2 import (
    LM_HEAD_V2_TPB, LM_HEAD_V2_LOSS_LEAF,
    chunked_lm_head_v2_rows_kernel, chunked_lm_head_v2_dhidden_kernel, chunked_lm_head_v2_dweight_kernel,
    chunked_lm_head_v2_row_body, chunked_lm_head_v2_dhidden_body, chunked_lm_head_v2_dweight_body,
    _enqueue_loss_total,
)

#: The session op code (autograd/ops_api.mojo) and the tape custom tag.
comptime OPS_LM_HEAD_V2 = 87
comptime LM_HEAD_V2_TAG = 109

comptime _F = MutPointer[Float32, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _fp(p: FP) -> _F:
    return _F(unsafe_from_address=Int(p))


@always_inline
def _ip(p: FP) -> _I:
    return _I(unsafe_from_address=Int(p))


def lm_head_v2_check(rows: Int, vocab: Int, width: Int) raises:
    if rows < 1 or vocab < 2 or width < 1:
        raise Error("ValueError: ops: lm_head_v2 needs rows >= 1, vocab >= 2 and width >= 1")
    if rows > 2147483647 // width or vocab > 2147483647 // width:
        raise Error("ValueError: ops: lm_head_v2 exceeds signed 32-bit kernel indexing")


# ---------------------------------------------------------------- device
def _codes_kernel(targets: _I, codes: _F, rows_in: Int32):
    """targets[i] = the int32 class code of codes[i] (a float class value)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(rows_in):
        return
    targets.unsafe_store(i, codes.unsafe_load(i).cast[DType.int32]())


def _scale_kernel(dst: _F, upstream: _F, n_in: Int32):
    """dst[i] *= upstream[0] (the VJP's chain factor; 1 at a loss root)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, ftz(identical_mul(dst.unsafe_load(i), upstream.unsafe_load(0))))


def _grid(n: Int) -> Int:
    return (n + LM_HEAD_V2_TPB - 1) // LM_HEAD_V2_TPB


def lm_head_v2_forward_device(
    ex: DeviceExec, hidden: FP, weight: FP, codes: FP, targets: FP,
    maxima: FP, denom: FP, row_loss: FP, loss: FP, rows: Int, vocab: Int, width: Int,
) raises:
    """loss[0] = the mean row loss; the row statistics stay for the VJP."""
    var ctx = ex.ctx.copy()
    ctx.enqueue_function[_codes_kernel](_ip(targets), _fp(codes), Int32(rows),
                                        grid_dim=_grid(rows), block_dim=LM_HEAD_V2_TPB)
    ctx.enqueue_function[chunked_lm_head_v2_rows_kernel](
        _fp(maxima), _fp(denom), _fp(row_loss), _fp(hidden), _fp(weight), _ip(targets),
        Int32(rows), Int32(vocab), Int32(width), grid_dim=_grid(rows), block_dim=LM_HEAD_V2_TPB,
    )
    var lb = block_buffer_view(ex, loss, 1)
    var rb = block_buffer_view(ex, row_loss, rows)
    _enqueue_loss_total(ctx, lb, rb, rows)
    _ = lb^
    _ = rb^


def lm_head_v2_backward_device(
    ex: DeviceExec, d_hidden: FP, d_weight: FP, hidden: FP, weight: FP, targets: FP,
    maxima: FP, denom: FP, upstream: FP, rows: Int, vocab: Int, width: Int,
) raises:
    """d_hidden (rows, width) and d_weight (vocab, width), each written whole
    (a null destination is skipped), scaled by upstream[0]."""
    var ctx = ex.ctx.copy()
    if Int(d_hidden) != 0:
        ctx.enqueue_function[chunked_lm_head_v2_dhidden_kernel](
            _fp(d_hidden), _fp(hidden), _fp(weight), _ip(targets), _fp(maxima), _fp(denom),
            Int32(rows), Int32(vocab), Int32(width), grid_dim=_grid(rows), block_dim=LM_HEAD_V2_TPB,
        )
        ctx.enqueue_function[_scale_kernel](_fp(d_hidden), _fp(upstream), Int32(rows * width),
                                            grid_dim=_grid(rows * width), block_dim=LM_HEAD_V2_TPB)
    if Int(d_weight) != 0:
        ctx.enqueue_function[chunked_lm_head_v2_dweight_kernel](
            _fp(d_weight), _fp(hidden), _fp(weight), _ip(targets), _fp(maxima), _fp(denom),
            Int32(rows), Int32(vocab), Int32(width), grid_dim=_grid(vocab), block_dim=LM_HEAD_V2_TPB,
        )
        ctx.enqueue_function[_scale_kernel](_fp(d_weight), _fp(upstream), Int32(vocab * width),
                                            grid_dim=_grid(vocab * width), block_dim=LM_HEAD_V2_TPB)


# ---------------------------------------------------------------- host
def _host_loss_total(loss: _F, row_loss: _F, rows: Int):
    """`_enqueue_loss_total`'s fold on the host: leaves of LM_HEAD_V2_LOSS_LEAF
    rows ascending from +0.0, then the balanced tree over the leaf partials in
    ascending order (an odd tail carried), then `ftz(total / rows)`."""
    var leaves = (rows + LM_HEAD_V2_LOSS_LEAF - 1) // LM_HEAD_V2_LOSS_LEAF
    var level = List[Float32](length=leaves, fill=Float32(0))
    for j in range(leaves):  # small-loop(leaves: one partial per 256 rows): the fixed fold, not data-parallel work
        var acc = Float32(0.0)
        for row in range(j * LM_HEAD_V2_LOSS_LEAF, min((j + 1) * LM_HEAD_V2_LOSS_LEAF, rows)):  # small-loop(a leaf's rows, ascending): the fixed fold
            acc = ftz(acc + row_loss.unsafe_load(row))
        level[j] = acc
    if leaves == 1:
        loss.unsafe_store(0, ftz(identical_div(level[0], Float32(rows))))
        return
    var width = leaves
    while width > 1:  # small-loop(tree levels): the fixed fold
        var out_width = (width + 1) // 2
        var next = List[Float32](length=out_width, fill=Float32(0))
        for q in range(out_width):  # small-loop(a tree level's nodes): the fixed fold
            if 2 * q + 1 < width:
                next[q] = ftz(ftz(level[2 * q]) + ftz(level[2 * q + 1]))
            else:
                next[q] = level[2 * q]
        level = next^
        width = out_width
    loss.unsafe_store(0, ftz(identical_div(ftz(level[0]), Float32(rows))))


def lm_head_v2_forward_host(
    hidden: FP, weight: FP, codes: FP, targets: FP,
    maxima: FP, denom: FP, row_loss: FP, loss: FP, rows: Int, vocab: Int, width: Int,
) raises:
    """The device forward's words on the host, rows split across cores."""
    var h = _fp(hidden)
    var w = _fp(weight)
    var c = _fp(codes)
    var t = _ip(targets)
    var mx = _fp(maxima)
    var dn = _fp(denom)
    var rl = _fp(row_loss)
    var tasks = host_predict_task_count_work(rows, rows * vocab * width)
    var chunk = host_predict_chunk(rows, tasks)

    def _rows(k: Int) {imm h, imm w, imm c, imm t, imm mx, imm dn, imm rl, imm chunk, imm rows, imm vocab, imm width}:
        var lo = k * chunk
        for row in range(lo, min(lo + chunk, rows)):
            t.unsafe_store(row, c.unsafe_load(row).cast[DType.int32]())
            chunked_lm_head_v2_row_body(row, mx, dn, rl, h, w, t, vocab, width)

    host_parallelize(_rows, tasks)
    _host_loss_total(_fp(loss), rl, rows)


def lm_head_v2_backward_host(
    d_hidden: FP, d_weight: FP, hidden: FP, weight: FP, targets: FP,
    maxima: FP, denom: FP, upstream: FP, rows: Int, vocab: Int, width: Int,
) raises:
    """The device VJP's words on the host: row owners, then vocabulary
    owners, each split across cores; then the upstream factor."""
    var h = _fp(hidden)
    var w = _fp(weight)
    var t = _ip(targets)
    var mx = _fp(maxima)
    var dn = _fp(denom)
    var up = _fp(upstream).unsafe_load(0)
    if Int(d_hidden) != 0:
        var dh = _fp(d_hidden)
        var tasks = host_predict_task_count_work(rows, rows * vocab * width)
        var chunk = host_predict_chunk(rows, tasks)

        def _rows(k: Int) {imm dh, imm h, imm w, imm t, imm mx, imm dn, imm up, imm chunk, imm rows, imm vocab, imm width}:
            var lo = k * chunk
            for row in range(lo, min(lo + chunk, rows)):
                chunked_lm_head_v2_dhidden_body(row, dh, h, w, t, mx, dn, rows, vocab, width)
                for f in range(width):
                    var cell = row * width + f
                    dh.unsafe_store(cell, ftz(identical_mul(dh.unsafe_load(cell), up)))

        host_parallelize(_rows, tasks)
    if Int(d_weight) != 0:
        var dw = _fp(d_weight)
        var vtasks = host_predict_task_count_work(vocab, rows * vocab * width)
        var vchunk = host_predict_chunk(vocab, vtasks)

        def _tokens(k: Int) {imm dw, imm h, imm w, imm t, imm mx, imm dn, imm up, imm vchunk, imm rows, imm vocab, imm width}:
            var lo = k * vchunk
            for token in range(lo, min(lo + vchunk, vocab)):
                chunked_lm_head_v2_dweight_body(token, dw, h, w, t, mx, dn, rows, width)
                for f in range(width):
                    var cell = token * width + f
                    dw.unsafe_store(cell, ftz(identical_mul(dw.unsafe_load(cell), up)))

        host_parallelize(_tokens, vtasks)
