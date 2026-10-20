# SPDX-License-Identifier: Apache-2.0
# RANK-N BROADCASTS, AXIS REDUCTIONS AND MAX (lane ag-rankn-fold,
# docs/plans/AUTOGRAD_RANKN.md 2.1, 2.2): the element bodies of the rank-N
# binary ops with a middle-axis broadcast, the axis-set sums and means, and
# the order-free max. Shared by HostExec and DeviceExec through
# sequence/dispatch.mojo, so the host column and every GPU run the same
# bodies and the same order.
#
# Codes OP_RNF_FIRST..OP_RNF_LAST (4200..4219) are this family;
# sequence/dispatch.mojo routes and fences them. 4209..4219 are free.
#
# THE ORDER (plan 2.0 (F), the virtual-matrix fold). A sum over an axis set
# A orders the kept axes row-major (they index the outputs, O of them) and
# the reduced axes row-major (they index the terms, R of them); the sum is
# the tape's blocked fold v2 over the virtual matrix [O, R] along
# AXIS_COLS: leaves of AG_FOLD_LEAF = 64 consecutive terms added ascending
# from +0.0, the leaf partial of (output o, leaf b) at o * nb + b
# (nb = ceil(R / 64), _ag_reduce_part_v2's AXIS_COLS layout), then the
# existing OP_AGF_LEVEL and OP_AG_REDUCE_FOLD launches (autograd/fold.mojo
# blocked_fold_finish with rows = O, cols = R). A fold of at most 64 terms
# an output is one launch that writes the finished output (v2's one-launch
# form, the same bits). Only the leaf launches below are new; the term of
# (o, r) is the source word at the coordinates decoded from (o, r).
#
# THE DESCRIPTOR (int32 bit patterns in tape storage, read with rn_word,
# never ld; bound to Args.p11):
#   word 0  nk: kept axis records (outermost first)
#   word 1  nr: reduced axis records (outermost first), after the kept ones
#   word 2  base offset of operand x in its buffer
#   word 3  base offset of operand y in its buffer (binary ops)
#   then nk + nr records of RNF_REC = 4 words, one an axis:
#     [extent, output stride, x stride, column 3]
#   The output stride is the output's row-major stride of that axis (0
#   where the output does not carry it: a reduced axis of a reduction's
#   VJP); x stride reads operand x's buffer (0 on a broadcast axis); column
#   3 is operand y's stride (binary ops) or x's logical row-major stride
#   (max: where its gradient word sits).
# A linear index decodes over a record list innermost (last record) first:
# coordinate c = rem % extent, rem //= extent, each offset += c * stride.
# A per-thread loop over at most 8 records (the tensor rank): shape
# metadata, no data, and no host loop.
#
# Max (plan 2.2) is order-free: the combine of two (value, index) pairs
# takes the larger value, a NaN wins over any number, equal values (and two
# NaNs) take the lower index. The result is then the same for any fold
# structure; the launches still follow (F)'s leaves and fan-in 64 so the
# parallelism is the fold's. The index is r, the row-major position over
# the reduced axes (for one axis: the position along it); within one output
# a lower r is a lower linear source index.
from std.memory import bitcast
from sequence.ops import Args, FP, ld, st, add, mul
from autograd.ops import _bin_value, div, ADD, SUBTRACT, MULTIPLY, DIVIDE, AG_FOLD_LEAF, ag_part_finish
from autograd.rankn import rn_word
from checks.numerics import ftz

comptime OP_RN_BIN_FWD = 4200
comptime OP_RN_BIN_VJP = 4201
comptime OP_RN_BIN_VJP_PART = 4202
comptime OP_RN_REDUCE_PART = 4203
comptime OP_RN_BCAST_VJP = 4204
comptime OP_RN_MAX_PART = 4205
comptime OP_RN_MAX_LEVEL = 4206
comptime OP_RN_MAX_FOLD = 4207
comptime OP_RN_MAX_VJP = 4208
comptime OP_RNF_FIRST = 4200
comptime OP_RNF_LAST = 4219

comptime RNF_HEAD = 4
comptime RNF_REC = 4


@always_inline
def _rec(p: FP, k: Int, col: Int) -> Int:
    return rn_word(p, RNF_HEAD + k * RNF_REC + col)


@always_inline
def _ld_bits(p: FP, i: Int) -> Int:
    # an int32 pattern (an index), never through ld: ftz would zero it
    return Int(bitcast[DType.int32](p.unsafe_load(i)))


@always_inline
def _st_bits(p: FP, i: Int, v: Int):
    p.unsafe_store(i, bitcast[DType.float32](Int32(v)))


@always_inline
def _ld_y(a: Args, i: Int) -> Float32:
    # i10 = 1: y is x's buffer (bound once, p1 unbound). A branch, never a
    # pointer select (the Metal rule, autograd/ops.mojo _ld_right).
    if a.i10 == 1:
        return ld(a.p0, i)
    return ld(a.p1, i)


@always_inline
def _decode3(p: FP, first: Int, count: Int, index: Int) -> Tuple[Int, Int, Int]:
    """(output, x, y) offsets of `index` decoded over records first ..
    first + count - 1 (innermost last), columns 1, 2, 3."""
    var rem = index
    var o = 0
    var x = 0
    var y = 0
    var k = first + count - 1
    while k >= first:  # small-loop(count <= 8: tensor rank, shape metadata)
        var e = _rec(p, k, 0)
        var c = rem % e
        rem = rem // e
        o += c * _rec(p, k, 1)
        x += c * _rec(p, k, 2)
        y += c * _rec(p, k, 3)
        k -= 1
    return (o, x, y)


@always_inline
def _vjp_term(a: Args, e: Int, oo: Int, xo: Int, yo: Int) -> Float32:
    """The binary VJP of operand e (i8) at one output element: op_ag_bin_vjp's
    arithmetic (autograd/ops.mojo _bin_vjp_at) with the operands read at
    their strided offsets. p0 x, p1 y (unbound when i10 = 1), p2 the
    forward output z (DIVIDE, operand 1), p3 upstream; i1 kind."""
    var op = a.i1
    var g = ld(a.p3, oo)
    if op == SUBTRACT and e == 1:
        g = -g
    elif op == MULTIPLY:
        if e == 0:
            g = mul(g, _ld_y(a, yo))
        else:
            g = mul(g, ld(a.p0, xo))
    elif op == DIVIDE:
        var y = _ld_y(a, yo)
        if e == 0:
            g = div(g, y)
        else:
            g = -div(mul(g, ld(a.p2, oo)), y)
    return g


# ---------------------------------------------------------------- bodies
@always_inline
def op_rn_bin_fwd(t: Int, a: Args):
    """z[t] = x op y at output element t. p0 x, p1 y (unbound when i10 = 1),
    p2 z, p11 descriptor (nk = output rank, nr = 0); i1 kind (ADD ..
    DIVIDE). OP_AG_BIN_FWD's value (_bin_value on ld words), so the same
    word per element."""
    var nk = rn_word(a.p11, 0)
    var d = _decode3(a.p11, 0, nk, t)
    st(a.p2, t, _bin_value(a.i1, ld(a.p0, rn_word(a.p11, 2) + d[1]), _ld_y(a, rn_word(a.p11, 3) + d[2])))


@always_inline
def op_rn_bin_vjp(t: Int, a: Args):
    """The VJP of an operand of the output's shape at output element t,
    written to p4[t] (the operand's gradient word t). op_rn_bin_fwd's
    slots and descriptor plus p3 upstream, p4 destination, i8 operand."""
    var nk = rn_word(a.p11, 0)
    var d = _decode3(a.p11, 0, nk, t)
    st(a.p4, t, _vjp_term(a, a.i8, t, rn_word(a.p11, 2) + d[1], rn_word(a.p11, 3) + d[2]))


@always_inline
def op_rn_bin_vjp_part(t: Int, a: Args):
    """Leaf of order (F) over a broadcast operand's broadcast axes. The
    descriptor's kept records are the operand's axes (O = its word count),
    the reduced records its broadcast axes (R terms). Thread t = (output
    o = t // nb, leaf b = t % nb), nb = i9; R = i6. Terms ftz(_vjp_term)
    added ascending from +0.0 (the stored-then-loaded VJP word, as
    op_agk_bin_vjp_part folds). i11 = 1 (nb = 1): p5[t] = +0.0 + the chain,
    the finished gradient word; else the partial at p4[t]."""
    var nb = a.i9
    var R = a.i6
    var nk = rn_word(a.p11, 0)
    var nr = rn_word(a.p11, 1)
    var o = t // nb
    var lo = (t % nb) * AG_FOLD_LEAF
    var hi = min(lo + AG_FOLD_LEAF, R)
    var base = _decode3(a.p11, 0, nk, o)
    var xb = rn_word(a.p11, 2) + base[1]
    var yb = rn_word(a.p11, 3) + base[2]
    var s = Float32(0)
    for r in range(lo, hi):  # the declared 64-term leaf chain
        var d = _decode3(a.p11, nk, nr, r)
        s = add(s, ftz(_vjp_term(a, a.i8, base[0] + d[0], xb + d[1], yb + d[2])))
    if a.i11 == 1:
        st(a.p5, t, add(Float32(0), s))
    else:
        st(a.p4, t, s)


@always_inline
def op_rn_reduce_part(t: Int, a: Args):
    """Leaf of order (F) for a sum or mean over an axis set. p0 source, p1
    partials, p2 destination, p11 descriptor (kept records: the source's
    kept axes, whose row-major index is the output index; reduced records:
    the reduced axes; column 2 the source strides, word 2 its base offset).
    Thread t = (o = t // nb, leaf b = t % nb), nb = i9, R = i6. The chain
    ends in ag_part_finish (autograd/ops.mojo): i4 = 1 (nb = 1) writes
    (+0.0 + s) * f0 to p2[t], else the partial s to p1[t]; i5 = i7 = 0."""
    var nb = a.i9
    var R = a.i6
    var nk = rn_word(a.p11, 0)
    var nr = rn_word(a.p11, 1)
    var o = t // nb
    var lo = (t % nb) * AG_FOLD_LEAF
    var hi = min(lo + AG_FOLD_LEAF, R)
    var xb = rn_word(a.p11, 2) + _decode3(a.p11, 0, nk, o)[1]
    var s = Float32(0)
    for r in range(lo, hi):  # the declared 64-term leaf chain
        s = add(s, ld(a.p0, xb + _decode3(a.p11, nk, nr, r)[1]))
    ag_part_finish(t, a, s)


@always_inline
def op_rn_bcast_vjp(t: Int, a: Args):
    """A sum's or mean's VJP: p1[t] = upstream at t's output coordinates
    times f0 (op_ag_reduce_vjp's rule). p0 upstream (output words), p1
    destination (the source's gradient, its row-major layout), p11
    descriptor: nk = the source's rank, one record an axis in order, column
    1 the output stride (0 on a reduced axis). Thread t = source word t."""
    var nk = rn_word(a.p11, 0)
    st(a.p1, t, mul(ld(a.p0, _decode3(a.p11, 0, nk, t)[0]), a.f0))


@always_inline
def _better(v: Float32, i: Int, bv: Float32, bi: Int) -> Bool:
    """Whether (v, i) beats (bv, bi) in max's order-free combine: a NaN wins
    over a number, the larger value wins, a tie (or two NaNs) takes the
    lower index."""
    var vn = v != v
    var bn = bv != bv
    if vn and bn:
        return i < bi
    if vn:
        return True
    if bn:
        return False
    if v > bv:
        return True
    if v == bv:
        return i < bi
    return False


@always_inline
def op_rn_max_part(t: Int, a: Args):
    """Max leaf: op_rn_reduce_part's slots, mapping and descriptor, with the
    combine of _better over (word, r). Partials: values p1[t], indices p4[t]
    (int32 patterns). i4 = 1 (nb = 1): the finished max to p2[t] and its
    index r to p3[t]."""
    var nb = a.i9
    var R = a.i6
    var nk = rn_word(a.p11, 0)
    var nr = rn_word(a.p11, 1)
    var o = t // nb
    var lo = (t % nb) * AG_FOLD_LEAF
    var hi = min(lo + AG_FOLD_LEAF, R)
    var xb = rn_word(a.p11, 2) + _decode3(a.p11, 0, nk, o)[1]
    var bv = ld(a.p0, xb + _decode3(a.p11, nk, nr, lo)[1])
    var bi = lo
    for r in range(lo + 1, hi):  # the leaf's terms, ascending
        var v = ld(a.p0, xb + _decode3(a.p11, nk, nr, r)[1])
        if _better(v, r, bv, bi):
            bv = v
            bi = r
    if a.i4 == 1:
        st(a.p2, t, bv)
        _st_bits(a.p3, t, bi)
    else:
        st(a.p1, t, bv)
        _st_bits(a.p4, t, bi)


@always_inline
def op_rn_max_level(t: Int, a: Args):
    """One level of max's tree, in place (op_agf_level's AXIS_COLS geometry):
    p1 value partials, p4 index partials; i9 nb (leaf partials an output),
    i5 S (stride of the live partials), i6 P (live partials an output).
    Thread t = (o = t // G, group j = t % G), G = ceil(P / 64): live
    partials j * 64 .. min(j * 64 + 64, P) - 1 combine into the first."""
    var nb = a.i9
    var S = a.i5
    var P = a.i6
    var groups = (P + AG_FOLD_LEAF - 1) // AG_FOLD_LEAF
    var o = t // groups
    var lo = (t % groups) * AG_FOLD_LEAF
    var n = min(AG_FOLD_LEAF, P - lo)
    var base = o * nb + lo * S
    var bv = ld(a.p1, base)
    var bi = _ld_bits(a.p4, base)
    for k in range(1, n):  # the group's partials, ascending
        var at = base + k * S
        var v = ld(a.p1, at)
        var i = _ld_bits(a.p4, at)
        if _better(v, i, bv, bi):
            bv = v
            bi = i
    st(a.p1, base, bv)
    _st_bits(a.p4, base, bi)


@always_inline
def op_rn_max_fold(t: Int, a: Args):
    """Max's last level: output t combines its i6 live partials at stride i5
    (p1 values, p4 indices, from o * i9) into p2[t] (the max) and p3[t]
    (its index r, an int32 pattern)."""
    var nb = a.i9
    var S = a.i5
    var P = a.i6
    var base = t * nb
    var bv = ld(a.p1, base)
    var bi = _ld_bits(a.p4, base)
    for k in range(1, P):  # the live partials, ascending
        var at = base + k * S
        var v = ld(a.p1, at)
        var i = _ld_bits(a.p4, at)
        if _better(v, i, bv, bi):
            bv = v
            bi = i
    st(a.p2, t, bv)
    _st_bits(a.p3, t, bi)


@always_inline
def op_rn_max_vjp(t: Int, a: Args):
    """Max's VJP: thread t = output t writes the upstream word p0[t] into the
    source's gradient p1 at its argmax (p3[t] = r): the source's logical
    index from (t over the kept records, r over the reduced records),
    column 3 (the source's row-major strides). The outputs partition the
    source, so no two threads write one word; the tape zero-fills p1."""
    var nk = rn_word(a.p11, 0)
    var nr = rn_word(a.p11, 1)
    var i = _decode3(a.p11, 0, nk, t)[2] + _decode3(a.p11, nk, nr, _ld_bits(a.p3, t))[2]
    st(a.p1, i, ld(a.p0, t))


@always_inline
def apply_rankn_fold[OP: Int](t: Int, a: Args):
    """The OP_RNF_FIRST..OP_RNF_LAST family (sequence/dispatch.mojo)."""
    comptime if OP == OP_RN_BIN_FWD:
        op_rn_bin_fwd(t, a)
    elif OP == OP_RN_BIN_VJP:
        op_rn_bin_vjp(t, a)
    elif OP == OP_RN_BIN_VJP_PART:
        op_rn_bin_vjp_part(t, a)
    elif OP == OP_RN_REDUCE_PART:
        op_rn_reduce_part(t, a)
    elif OP == OP_RN_BCAST_VJP:
        op_rn_bcast_vjp(t, a)
    elif OP == OP_RN_MAX_PART:
        op_rn_max_part(t, a)
    elif OP == OP_RN_MAX_LEVEL:
        op_rn_max_level(t, a)
    elif OP == OP_RN_MAX_FOLD:
        op_rn_max_fold(t, a)
    elif OP == OP_RN_MAX_VJP:
        op_rn_max_vjp(t, a)
