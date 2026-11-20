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
from training.chunked_lm_head_v2 import LM_HEAD_V2_LOSS_LEAF
from training.chunked_lm_head_gemm_panel_host import lmhead_gp_host_forward_into, lmhead_gp_host_backward_into

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
        var nxt = List[Float32](length=out_width, fill=Float32(0))
        for q in range(out_width):  # small-loop(a tree level's nodes): the fixed fold
            if 2 * q + 1 < width:
                nxt[q] = ftz(ftz(level[2 * q]) + ftz(level[2 * q + 1]))
            else:
                nxt[q] = level[2 * q]
        level = nxt^
        width = out_width
    loss.unsafe_store(0, ftz(identical_div(ftz(level[0]), Float32(rows))))


# The GEMM-panel profile (the only profile since 2026-11-20): the op's host
# arm runs the host column of the public door's arm
# (training/chunked_lm_head_gemm_panel_host.mojo), the words the device op
# computes under the same define (autograd/lm_head_v2_op.mojo). The op's own
# parts stay: the float class codes, +0.0 as the shifted logit of a code that
# names no vocabulary entry, and the upstream factor.
def _forward_gp_host(
    hidden: FP, weight: FP, codes: FP, targets: FP,
    maxima: FP, denom: FP, row_loss: FP, loss: FP, rows: Int, vocab: Int, width: Int,
) raises:
    var c = _fp(codes)
    var t = _ip(targets)
    var rl = _fp(row_loss)
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)

    def _rows(k: Int) {imm c, imm t, imm rl, imm chunk, imm rows}:
        var lo = k * chunk
        for row in range(lo, min(lo + chunk, rows)):
            t.unsafe_store(row, c.unsafe_load(row).cast[DType.int32]())
            rl.unsafe_store(row, Float32(0.0))

    host_parallelize(_rows, tasks)
    lmhead_gp_host_forward_into(hidden, weight, t, rows, vocab, width, _fp(maxima), _fp(denom), rl)
    _host_loss_total(_fp(loss), rl, rows)


def _scale_host(values: _F, up: Float32, count: Int):
    """`_scale_kernel` on the host: values[i] *= upstream, split across cores."""
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)

    def _cells(k: Int) {imm values, imm up, imm chunk, imm count}:
        var lo = k * chunk
        for i in range(lo, min(lo + chunk, count)):
            values.unsafe_store(i, ftz(identical_mul(values.unsafe_load(i), up)))

    host_parallelize(_cells, tasks)


def _backward_gp_host(
    d_hidden: FP, d_weight: FP, hidden: FP, weight: FP, targets: FP,
    maxima: FP, denom: FP, upstream: FP, rows: Int, vocab: Int, width: Int,
) raises:
    var want_dh = Int(d_hidden) != 0
    var want_dw = Int(d_weight) != 0
    if not want_dh and not want_dw:
        return
    var up = _fp(upstream).unsafe_load(0)
    lmhead_gp_host_backward_into(hidden, weight, _ip(targets), rows, vocab, width, _fp(maxima), _fp(denom),
                                 _fp(d_hidden), d_weight, want_dh, want_dw)
    if want_dh:
        _scale_host(_fp(d_hidden), up, rows * width)
    if want_dw:
        _scale_host(_fp(d_weight), up, vocab * width)


def lm_head_v2_forward_host(
    hidden: FP, weight: FP, codes: FP, targets: FP,
    maxima: FP, denom: FP, row_loss: FP, loss: FP, rows: Int, vocab: Int, width: Int,
) raises:
    """The device forward's words on the host, rows split across cores."""
    _forward_gp_host(hidden, weight, codes, targets, maxima, denom, row_loss, loss, rows, vocab, width)


def lm_head_v2_backward_host(
    d_hidden: FP, d_weight: FP, hidden: FP, weight: FP, targets: FP,
    maxima: FP, denom: FP, upstream: FP, rows: Int, vocab: Int, width: Int,
) raises:
    """The device VJP's words on the host: row owners, then vocabulary
    owners, each split across cores; then the upstream factor."""
    _backward_gp_host(d_hidden, d_weight, hidden, weight, targets, maxima, denom, upstream, rows, vocab, width)


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
