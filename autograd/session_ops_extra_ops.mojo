# SPDX-License-Identifier: Apache-2.0
# THE KERNEL BODIES OF TOPK AND LOGSUMEXP (sub-lane session-ops-extra, 2026-11-25).
#
# One `@always_inline` body per code, shared by HostExec and DeviceExec through sequence/dispatch.mojo, so the host
# column and every GPU run the same words. Codes OP_SOX_FIRST..OP_SOX_LAST (6980..6999, the range the session-ops
# brief assigns this sub-lane). The drivers, the tape records and the backward rules are autograd/session_ops_extra.mojo.
# Only what no existing body does is here: the order key of a top-k line and its read-out, and the logsumexp row with
# its VJP. The sort itself is the embedding's bitonic stage (OP_EMB_SORT_STAGE, autograd/embedding_ops.mojo), the
# gather is OP_SCT_KEYS + OP_EMB_GATHER and the gather's backward is OP_EMB_FOLD.
#
# TOPK KEYS. Each line of x (every position of the other axes) is laid out on its own block of Lp words (Lp the power
# of two >= the axis length V), one (key, value) pair a word pair: value = the position j along the axis, key = an
# int32 whose ASCENDING order is the order the top-k wants, so the embedding's ascending (key, value) network sorts it
# and equal keys keep ascending positions: THE LOWEST INDEX WINS A TIE, on every vendor (exact integers, no float
# compare, no atomics). For largest=True the key is ~ord(v), for largest=False ord(v), where ord is the monotone int32
# image of the float (ftz first; -0.0 is +0.0, so the two zeros tie and the lower index wins; every NaN is one value
# larger than +inf, as torch.topk orders it). Padding words (j >= V) take INT32_MAX and a value above every real
# position, so they sort after every real element.
#
# The network's last merge sorts a block ascending when its block number is even and descending when it is odd (the
# bitonic network's direction bit `i & Lp`), so the read-out takes slot r of an even line and slot Lp - 1 - r of an
# odd one; the pairs of every stage stay inside one aligned block (partner distance < merge size <= Lp).
#
# LOGSUMEXP. Row t of a (rows, cols) matrix: m the row maximum and log_total = log(sum exp(x - m)) EXACTLY as
# log-softmax computes them (autograd/ops.mojo op_ag_softmax_fwd; under ROW_COOP the row_first_max / row_expsum32
# fold32 of autograd/row_ops.mojo), result = ftz(m + log_total): the log-softmax's own max-shift fold, so a row of
# large scores stays finite. SETTLED: a row whose maximum is infinite returns that maximum (+inf, or -inf for a row
# of only -inf) and NaN when the row holds a NaN; its VJP is 0 for every element (the shift is undefined there).
from std.memory import bitcast
from sequence.ops import FP, Args, ld, st, add, sub, mul
from checks.numerics import ftz, identical_exp, identical_log
from autograd.row_ops import ROW_COOP, row_first_max, row_expsum32

comptime OP_SOX_TOPK_PACK = 6980
comptime OP_SOX_TOPK_OUT = 6981
comptime OP_SOX_LSE = 6982
comptime OP_SOX_LSE_VJP = 6983
comptime OP_SOX_FIRST = 6980
comptime OP_SOX_LAST = 6999

comptime _SOX_I32_MAX = 2147483647
comptime _SOX_I32_MIN = -2147483648


@always_inline
def _sox_ld_i(p: FP, i: Int) -> Int:
    """The int32 held as bits in word i of p."""
    return Int(bitcast[DType.int32](p.unsafe_load(i)))


@always_inline
def _sox_st_i(p: FP, i: Int, v: Int):
    """Word i of p holds int32 v as its bit pattern."""
    p.unsafe_store(i, bitcast[DType.float32](Int32(v)))


@always_inline
def _sox_infinite(v: Float32) -> Bool:
    """True for +inf and -inf (inf - inf is NaN; a finite v gives 0; a NaN is not infinite)."""
    return v == v and not (v - v == Float32(0.0))


@always_inline
def sox_order_key(v: Float32, largest: Bool, zero_equal: Bool = True) -> Int:
    """The int32 sort key of value v: ascending keys are descending values for `largest`, ascending values
    otherwise. v is flushed first; -0.0 and +0.0 are one key; every NaN is one key above +inf (first for
    `largest`, last otherwise, before the padding)."""
    var w = ftz(v)
    if w != w:
        return _SOX_I32_MIN if largest else _SOX_I32_MAX
    if zero_equal and w == Float32(0.0):
        w = Float32(0.0)
    var s = bitcast[DType.int32](w)
    var o = s if s >= Int32(0) else s ^ Int32(0x7FFFFFFF)
    return Int(~o) if largest else Int(o)


@always_inline
def op_sox_topk_pack(t: Int, a: Args):
    """Word t of the padded line layout: line l = t // Lp, slot j = t % Lp; x seen as (O, V, I), line l = o I + i.
    keys[t] = the order key of x[(o V + j) I + i] for j < V, else INT32_MAX; values[t] = j. p0 x, p1 keys (int32
    words), p2 values (int32 words); i0 V, i1 I, i2 Lp, i3 1 for largest, i4 1 (0 only in a sabotage build: -0.0
    kept below +0.0)."""
    var lp = a.i2
    var line = t // lp
    var j = t - line * lp
    var key = _SOX_I32_MAX
    if j < a.i0:
        var inner = a.i1
        var o = line // inner
        var i = line - o * inner
        key = sox_order_key(a.p0.unsafe_load((o * a.i0 + j) * inner + i), a.i3 != 0, a.i4 != 0)
    _sox_st_i(a.p1, t, key)
    _sox_st_i(a.p2, t, j)


@always_inline
def op_sox_topk_out(t: Int, a: Args):
    """Output word t = (o, r, i) of the (O, k, I) indices: the position of rank r of line o I + i, read from the
    sorted values (slot r of an even line, Lp - 1 - r of an odd one: the network's direction bit). p0 the sorted
    values, p1 the indices (int32 words); i0 k, i1 I, i2 Lp."""
    var inner = a.i1
    var ki = a.i0 * inner
    var o = t // ki
    var rem = t - o * ki
    var r = rem // inner
    var i = rem - r * inner
    var line = o * inner + i
    var slot = r if (line & 1) == 0 else a.i2 - 1 - r
    _sox_st_i(a.p1, t, _sox_ld_i(a.p0, line * a.i2 + slot))


@always_inline
def sox_lse_row(p: FP, base: Int, cols: Int, settle: Bool = True) -> Float32:
    """log(sum_j exp(x_j)) of the row at p[base .. base + cols): the log-softmax's maximum and log of the shifted
    exponential sum (its fold, its words), then ftz(m + log_total); an infinite maximum is settled (see the file
    header)."""
    var m: Float32
    var log_total: Float32
    comptime if ROW_COOP:
        m = ld(p, base + row_first_max(p, base, cols))
    else:
        m = ld(p, base)
        for k in range(1, cols):
            m = max(m, ld(p, base + k))
    if settle and _sox_infinite(m):
        # an infinite maximum: the row's +inf (or -inf for a row of only -inf), NaN when a NaN is in the row
        for k in range(cols):
            var w = ld(p, base + k)
            if w != w:
                return w
        return m
    comptime if ROW_COOP:
        log_total = ftz(identical_log(row_expsum32(p, base, cols, m)))
    else:
        var total = Float32(0)
        for k in range(cols):
            total = add(total, ftz(identical_exp(sub(ld(p, base + k), m))))
        log_total = ftz(identical_log(total))
    return add(m, log_total)


@always_inline
def op_sox_lse(t: Int, a: Args):
    """Row t: out[t] = logsumexp of x's row t. p0 x (rows, cols), p1 out (rows); i0 cols, i1 1 (0 only in a
    sabotage build: an infinite maximum not settled)."""
    st(a.p1, t, sox_lse_row(a.p0, t * a.i0, a.i0, a.i1 != 0))


@always_inline
def op_sox_lse_vjp(t: Int, a: Args):
    """Element t = (r, c): dx = g[r] * exp(x - out[r]), 0 when out[r] is not finite. p0 x (rows, cols), p1 out
    (rows), p2 the upstream gradient (rows), p3 dx (rows, cols); i0 cols. Every word written once."""
    var r = t // a.i0
    var lse = ld(a.p1, r)
    if not (lse == lse) or _sox_infinite(lse):
        st(a.p3, t, Float32(0.0))
        return
    st(a.p3, t, mul(ld(a.p2, r), ftz(identical_exp(sub(ld(a.p0, t), lse)))))


@always_inline
def apply_session_ops_extra[OP: Int](t: Int, a: Args):
    comptime if OP == OP_SOX_TOPK_PACK:
        op_sox_topk_pack(t, a)
    elif OP == OP_SOX_TOPK_OUT:
        op_sox_topk_out(t, a)
    elif OP == OP_SOX_LSE:
        op_sox_lse(t, a)
    elif OP == OP_SOX_LSE_VJP:
        op_sox_lse_vjp(t, a)
