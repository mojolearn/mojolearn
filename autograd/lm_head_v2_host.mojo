# SPDX-License-Identifier: Apache-2.0
"""The host arm of OPS_LM_HEAD_V2 (autograd/lm_head_v2_op.mojo): the device
op's words on the CPU column. A host module: only the host executor
(sequence/exec.mojo `HostExec.lm_head_v2_host`) imports it.

The forward and both VJP passes call the device kernels' own @always_inline
bodies (training/chunked_lm_head_v2.mojo), one row (or vocabulary entry) per
iteration, rows split into contiguous tasks across cores; the fold inside
each owner is the kernel's, so the task count never changes a word.
"""
from sequence.ops import FP
from checks.numerics import ftz, identical_div, identical_mul
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from training.chunked_lm_head_v2 import (
    LM_HEAD_V2_LOSS_LEAF,
    chunked_lm_head_v2_row_body, chunked_lm_head_v2_dhidden_body, chunked_lm_head_v2_dweight_body,
)

comptime _F = MutPointer[Float32, MutAnyOrigin]
comptime _I = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _fp(p: FP) -> _F:
    return _F(unsafe_from_address=Int(p))


@always_inline
def _ip(p: FP) -> _I:
    return _I(unsafe_from_address=Int(p))


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


def lm_head_v2_host_run(backward: Bool, bufs: List[FP], rows: Int, vocab: Int, width: Int) raises:
    """`Exec.lm_head_v2_host`: forward bufs [hidden, weight, codes, targets,
    maxima, denom, row_loss, loss]; backward bufs [d_hidden, d_weight,
    hidden, weight, targets, maxima, denom, upstream]."""
    if len(bufs) != 8:
        raise Error("ops: lm_head_v2 host arm takes eight buffers")
    if backward:
        lm_head_v2_backward_host(bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6], bufs[7],
                                 rows, vocab, width)
    else:
        lm_head_v2_forward_host(bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6], bufs[7],
                                rows, vocab, width)
