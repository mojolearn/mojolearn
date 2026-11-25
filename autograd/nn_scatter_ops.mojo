# SPDX-License-Identifier: Apache-2.0
# THE KERNEL BODIES OF SCATTER_ADD AND THE SEGMENT REDUCTIONS (lane nn-blocks-scatter, 2026-11-25).
#
# One `@always_inline` body per code, shared by HostExec and DeviceExec through sequence/dispatch.mojo, so the host
# column and every GPU run the same words. Codes OP_SCT_FIRST..OP_SCT_LAST (6800..6819, the range the nn-blocks brief
# assigns this lane). The drivers, the tape records and the backward rules are autograd/nn_scatter.mojo.
#
# NO FLOAT ATOMICS. Every reduction here reads the RUNS of the embedding backward (autograd/embedding_ops.mojo
# embedding_runs_into: for each target row v its source positions in ASCENDING position order at
# perm[run_begin[v] .. run_begin[v + 1]), built by a counting scan or a bitonic sort of (id, position) pairs, exact
# integer work). The sum and the mean fold a row with the embedding's own fold (OP_EMB_FOLD, op_emb_fold:
# acc = ftz(ftz(acc) + ftz(src)) from +0.0, or from the carried value for scatter_add), one thread per output cell,
# so the order of every addition is fixed by the data alone: the same words on the host and on every vendor, in any
# thread order. The bodies below add what the embedding does not have: the flat target key of a scatter, the mean's
# division by the row count, the max with its first argmax, and the VJPs of the mean and the max.
#
# Ids are int32 row numbers held as their bit patterns in 4-byte executor words, read with bitcast, never through
# ld (ftz would zero a small id's subnormal pattern), as autograd/embedding_ops.mojo does.
from std.memory import bitcast
from sequence.ops import FP, Args
from checks.numerics import ftz, identical_div

comptime OP_SCT_KEYS = 6800
comptime OP_SCT_MEAN = 6801
comptime OP_SCT_MAX = 6802
comptime OP_SCT_MAX_VJP = 6803
comptime OP_SCT_MEAN_VJP = 6804
comptime OP_SCT_FIRST = 6800
comptime OP_SCT_LAST = 6819


@always_inline
def _sct_ld_i(p: FP, i: Int) -> Int:
    """The int32 held as bits in word i of p."""
    return Int(bitcast[DType.int32](p.unsafe_load(i)))


@always_inline
def _sct_st_i(p: FP, i: Int, v: Int):
    """Word i of p holds int32 v as its bit pattern."""
    p.unsafe_store(i, bitcast[DType.float32](Int32(v)))


@always_inline
def op_sct_keys(t: Int, a: Args):
    """The flat output cell of source element t of a scatter_add along `dim`. The tensors are seen as
    (outer, extent, inner): src and index (O, K, I), x and the output (O, V, I). key[t] = (o V + index[t]) I + i for
    t = (o K + k) I + i. p0 index (int32 words, every value refused outside [0, V) by the caller), p1 keys (int32
    words); i0 K, i1 I, i2 V. For one output cell (o, v, i) the sources are the (o, k, i) with index v, so ascending t
    is ascending k: the runs fold them in ascending k."""
    var inner = a.i1
    var ki = a.i0 * inner
    var o = t // ki
    var rem = t - o * ki
    var i = rem - (rem // inner) * inner
    _sct_st_i(a.p1, t, (o * a.i2 + _sct_ld_i(a.p0, t)) * inner + i)


@always_inline
def op_sct_mean(t: Int, a: Args):
    """segment_mean's division, in place over the folded sums. Cell t = (v, j): out = ftz(sum / count) with count
    the row's run length; an empty row keeps the fold's +0.0 (mean = sum / max(count, 1)). p0 the sums (S, d), p1
    run_begin (S + 1 words); i0 d."""
    var v = t // a.i0
    var count = _sct_ld_i(a.p1, v + 1) - _sct_ld_i(a.p1, v)
    if count > 0:
        a.p0.unsafe_store(t, ftz(identical_div(ftz(a.p0.unsafe_load(t)), Float32(count))))


@always_inline
def op_sct_max(t: Int, a: Args):
    """segment_max. Cell t = (v, j): the largest ftz(data[pos, j]) over the row's positions in ascending order, the
    FIRST position holding it kept (a later equal value does not replace it; a NaN replaces a number and the first
    NaN stays, so a NaN propagates as torch.max does). An empty row stores +0.0 and argmax -1. p0 data (N, d), p1
    perm, p2 run_begin (S + 1), p3 out (S, d), p4 argmax (S, d int32 words: the source row, -1 for none); i0 d."""
    var width = a.i0
    var v = t // width
    var j = t - v * width
    var lo = _sct_ld_i(a.p2, v)
    var hi = _sct_ld_i(a.p2, v + 1)
    if lo >= hi:
        a.p3.unsafe_store(t, Float32(0.0))
        _sct_st_i(a.p4, t, -1)
        return
    var pos = _sct_ld_i(a.p1, lo)
    var best = ftz(a.p0.unsafe_load(pos * width + j))
    var at = pos
    for r in range(lo + 1, hi):
        var p = _sct_ld_i(a.p1, r)
        var w = ftz(a.p0.unsafe_load(p * width + j))
        if w > best or (w != w and best == best):
            best = w
            at = p
    a.p3.unsafe_store(t, best)
    _sct_st_i(a.p4, t, at)


@always_inline
def op_sct_max_vjp(t: Int, a: Args):
    """segment_max's VJP. Data cell t = (pos, j): the upstream gradient of cell (ids[pos], j) when pos is that cell's
    argmax, else +0.0. Every data cell is written once (no accumulation, no atomics). p0 ids (N int32 words), p1
    argmax (S, d words), p2 upstream (S, d), p3 the data gradient (N, d); i0 d."""
    var width = a.i0
    var pos = t // width
    var j = t - pos * width
    var cell = _sct_ld_i(a.p0, pos) * width + j
    if _sct_ld_i(a.p1, cell) == pos:
        a.p3.unsafe_store(t, ftz(a.p2.unsafe_load(cell)))
    else:
        a.p3.unsafe_store(t, Float32(0.0))


@always_inline
def op_sct_mean_vjp(t: Int, a: Args):
    """segment_mean's VJP. Data cell t = (pos, j): ftz(upstream[ids[pos], j] / count) with count the length of
    row ids[pos] (at least 1: pos is in it). p0 ids (N int32 words), p1 run_begin (S + 1), p2 upstream (S, d),
    p3 the data gradient (N, d); i0 d."""
    var width = a.i0
    var pos = t // width
    var j = t - pos * width
    var v = _sct_ld_i(a.p0, pos)
    var count = _sct_ld_i(a.p1, v + 1) - _sct_ld_i(a.p1, v)
    a.p3.unsafe_store(t, ftz(identical_div(ftz(a.p2.unsafe_load(v * width + j)), Float32(count))))


@always_inline
def apply_nn_scatter[OP: Int](t: Int, a: Args):
    comptime if OP == OP_SCT_KEYS:
        op_sct_keys(t, a)
    elif OP == OP_SCT_MEAN:
        op_sct_mean(t, a)
    elif OP == OP_SCT_MAX:
        op_sct_max(t, a)
    elif OP == OP_SCT_MAX_VJP:
        op_sct_max_vjp(t, a)
    elif OP == OP_SCT_MEAN_VJP:
        op_sct_mean_vjp(t, a)
