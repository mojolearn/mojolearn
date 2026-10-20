# SPDX-License-Identifier: Apache-2.0
# Rank-N broadcasting binary ops, axis-set reductions, max and softmax along
# any axis on the tape (lane ag-rankn-fold, docs/plans/AUTOGRAD_RANKN.md 2.1,
# 2.2, 2.7). Every entry routes to the existing 2-D method whenever that
# method computes the same mathematics (rank 2, a same-shape or
# trailing-dims broadcast, a last / leading / trailing-block / all-axes
# sum), so those calls keep their launches and words; only the remaining
# cases take the new bodies of autograd/rankn_fold_ops.mojo, recorded as
# CUSTOM nodes with the tags below and differentiated by RankNFoldRules.
#
# Tags 136..151 (the plan's fold block):
#   RN_TAG_BINARY + op (137 ADD, 138 SUBTRACT, 139 MULTIPLY, 140 DIVIDE):
#     a middle-axis broadcast. saved: p0 x's buffer, p1 y's (unbound when
#     i10 = 1: one buffer), p11 the forward descriptor, p9 / p8 x's / y's
#     reduction descriptor, p7 / p6 their fold partials (R > 64 only); i0
#     output words, i1 op, i2 / i3 x's O / R, i4 / i5 y's, i6 / i7 1 when x /
#     y has the output's shape.
#   RN_TAG_SUM 141 / RN_TAG_MEAN 142: saved p11 the VJP descriptor, i0 the
#     source's words, f0 the factor (1, or identical_div(1, R)).
#   RN_TAG_MAX 143: saved p10 the argmax (int32 patterns of r, one an
#     output), p11 the forward descriptor, i0 the output words.
# Operands are read through Tape.operand_nd (buffer, base offset, strides),
# so a pending permute feeds these ops with no copy; gradients are written
# in each operand's own row-major layout.
#
# Host loops here run over tensor axes (at most RN_MAX_RANK = 8) to build
# descriptors, over node inputs, and over fold tree levels; none over rows
# or elements.
from std.collections import InlineArray
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr
from checks.numerics import identical_div
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.ops import ADD, SUBTRACT, MULTIPLY, DIVIDE, AXIS_ALL, AXIS_ROWS, AXIS_COLS, AG_FOLD_LEAF
from autograd.fold import blocked_fold_finish, fold_partials
from autograd.rankn import ShapeN, RN_MAX_RANK, rn_broadcast, rn_normalize_axis
from autograd.rankn_fold_ops import (
    OP_RN_BIN_FWD, OP_RN_BIN_VJP, OP_RN_BIN_VJP_PART, OP_RN_REDUCE_PART, OP_RN_BCAST_VJP,
    OP_RN_MAX_PART, OP_RN_MAX_LEVEL, OP_RN_MAX_FOLD, OP_RN_MAX_VJP,
)

comptime RNF_TAG_FIRST = 136
comptime RNF_TAG_LAST = 151
comptime RN_TAG_BINARY = 136
comptime RN_TAG_SUM = 141
comptime RN_TAG_MEAN = 142
comptime RN_TAG_MAX = 143


# ---------------------------------------------------------------- shapes
def _ones(ndim: Int) -> ShapeN:
    return ShapeN(ndim, InlineArray[Int, RN_MAX_RANK](fill=1))


def _aligned_dim(s: ShapeN, out_ndim: Int, j: Int) -> Int:
    """s's extent on output axis j (s right-aligned to out_ndim axes; 1 on
    an axis s does not have)."""
    var k = j - (out_ndim - s.ndim)
    if k < 0:
        return 1
    return s.d[k]


def _aligned_stride(s: ShapeN, strides: InlineArray[Int, RN_MAX_RANK], so: ShapeN, j: Int) -> Int:
    """The stride that reads s (right-aligned) on output axis j: 0 on an
    axis s lacks or broadcasts."""
    var k = j - (so.ndim - s.ndim)
    if k < 0 or s.d[k] != so.d[j]:
        return 0
    return strides[k]


def _row_major(s: ShapeN) -> InlineArray[Int, RN_MAX_RANK]:
    var r = InlineArray[Int, RN_MAX_RANK](fill=0)
    var acc = 1
    var k = s.ndim - 1
    while k >= 0:  # small-loop(ndim <= 8: tensor rank, shape metadata)
        r[k] = acc
        acc *= s.d[k]
        k -= 1
    return r^


def _as_shape[E: Exec](mut tape: Tape, mut ex: E, t: Tensor, s: ShapeN) raises -> Tensor:
    """t with shape s: t itself when it already has it, else a free view."""
    if tape.shape_of(t).same(s):
        return t
    return tape.view_nd(ex, t, s)


def _matrix_bcast(big: ShapeN, small: ShapeN) -> Bool:
    """small's matrix view is (1, 1), (1, cols) or (rows, 1) of big's: the
    trailing-dims broadcast the 2-D binary kernel computes (given that
    small broadcasts to big under the rank-N rule)."""
    var r = small.rows()
    var c = small.cols()
    if r == 1 and c == 1:
        return True
    if r == 1 and c == big.cols():
        return True
    return r == big.rows() and c == 1


def _two_d_route(sx: ShapeN, sy: ShapeN, so: ShapeN) -> Bool:
    if sx.ndim <= 2 and sy.ndim <= 2:
        return True
    if sx.same(sy):
        return True
    if so.same(sx) and _matrix_bcast(sx, sy):
        return True
    return so.same(sy) and _matrix_bcast(sy, sx)


def _push_rec(mut w: List[Int], extent: Int, out_stride: Int, s2: Int, s3: Int):
    w.append(extent)
    w.append(out_stride)
    w.append(s2)
    w.append(s3)


def _blocks(R: Int) -> Int:
    return (R + AG_FOLD_LEAF - 1) // AG_FOLD_LEAF


# ---------------------------------------------------------------- binary
def rankn_binary[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, y: Tensor, op: Int) raises -> Tensor:
    """x op y (op ADD, SUBTRACT, MULTIPLY or DIVIDE from autograd.ops) with
    NumPy broadcasting over any rank. Same shapes, rank 2 and trailing-dims
    broadcasts run the tape's binary op; a middle-axis broadcast runs
    OP_RN_BIN_FWD (the same word per element) and records RN_TAG_BINARY."""
    if op != ADD and op != SUBTRACT and op != MULTIPLY and op != DIVIDE:
        raise Error("rankn_binary takes ADD, SUBTRACT, MULTIPLY or DIVIDE")
    var sx = tape.shape_of(x)
    var sy = tape.shape_of(y)
    var so = rn_broadcast(sx, sy)
    if _two_d_route(sx, sy, so):
        var r = tape._binary(ex, x, y, op)
        return _as_shape(tape, ex, r, so)
    var xo = tape.operand_nd(ex, x)
    var yo = tape.operand_nd(ex, y)
    var xbuf = xo[0]
    var ybuf = yo[0]
    var xs = xo[3].copy()
    var ys = yo[3].copy()
    var count = so.count()
    var orm = _row_major(so)
    # forward descriptor: every output axis kept, in order
    var w = List[Int]()
    w.append(so.ndim)
    w.append(0)
    w.append(xo[1])
    w.append(yo[1])
    for j in range(so.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        _push_rec(w, so.d[j], orm[j], _aligned_stride(sx, xs, so, j), _aligned_stride(sy, ys, so, j))
    var desc = tape.descriptor(ex, w)
    var output = tape.reserve(ex, count)
    var a = Args()
    a.p0 = xbuf
    if Int(ybuf) == Int(xbuf):
        a.i10 = 1
    else:
        a.p1 = ybuf
    a.p2 = output
    a.p11 = desc
    a.i1 = op
    ex.launch[OP_RN_BIN_FWD](a, count)
    var saved = Args()
    saved.p0 = a.p0
    saved.p1 = a.p1
    saved.i10 = a.i10
    saved.p11 = desc
    saved.i0 = count
    saved.i1 = op
    saved.i6 = 1 if sx.same(so) else 0
    saved.i7 = 1 if sy.same(so) else 0
    if tape.is_recording():
        for e in range(2):  # driver-loop: over node inputs (the two operands)
            var s = sx if e == 0 else sy
            if s.same(so):
                continue
            # kept: the operand's axes (its row-major index is its gradient
            # word); reduced: the axes it broadcasts over
            var kept = List[Int]()
            var red = List[Int]()
            for j in range(so.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
                if _aligned_dim(s, so.ndim, j) == so.d[j]:
                    kept.append(j)
                else:
                    red.append(j)
            var pw = List[Int]()
            pw.append(len(kept))
            pw.append(len(red))
            pw.append(xo[1])
            pw.append(yo[1])
            for k in range(len(kept)):  # small-loop(ndim <= 8: tensor rank, shape metadata)
                var j = kept[k]
                _push_rec(pw, so.d[j], orm[j], _aligned_stride(sx, xs, so, j), _aligned_stride(sy, ys, so, j))
            for k in range(len(red)):  # small-loop(ndim <= 8: tensor rank, shape metadata)
                var j = red[k]
                _push_rec(pw, so.d[j], orm[j], _aligned_stride(sx, xs, so, j), _aligned_stride(sy, ys, so, j))
            var O = s.count()
            var R = count // O
            var pdesc = tape.descriptor(ex, pw)
            var partials = dummy_ptr()
            if _blocks(R) > 1:
                partials = tape.reserve(ex, fold_partials(O, R, AXIS_COLS))
            if e == 0:
                saved.p9 = pdesc
                saved.p7 = partials
                saved.i2 = O
                saved.i3 = R
            else:
                saved.p8 = pdesc
                saved.p6 = partials
                saved.i4 = O
                saved.i5 = R
    return tape.custom_nd(ex, [x, y], output, so, RN_TAG_BINARY + op, saved)


def _binary_vjp[E: Exec](
    mut ex: E, saved: Args, output: FP, upstream: FP, gradients: List[FP],
) raises:
    var count = saved.i0
    for e in range(2):  # driver-loop: over node inputs (the two operands)
        var a = Args()
        a.p0 = saved.p0
        if saved.i10 == 1:
            a.i10 = 1
        else:
            a.p1 = saved.p1
        a.p2 = output
        a.p3 = upstream
        a.i1 = saved.i1
        a.i8 = e
        var same = saved.i6 if e == 0 else saved.i7
        if same == 1:
            a.p11 = saved.p11
            a.p4 = gradients[e]
            ex.launch[OP_RN_BIN_VJP](a, count)
            continue
        var O = saved.i2 if e == 0 else saved.i4
        var R = saved.i3 if e == 0 else saved.i5
        var nb = _blocks(R)
        a.p11 = saved.p9 if e == 0 else saved.p8
        a.i6 = R
        a.i9 = nb
        if nb == 1:
            a.i11 = 1
            a.p5 = gradients[e]
            ex.launch[OP_RN_BIN_VJP_PART](a, O)
        else:
            var partials = saved.p7 if e == 0 else saved.p6
            a.p4 = partials
            ex.launch[OP_RN_BIN_VJP_PART](a, O * nb)
            blocked_fold_finish(ex, gradients[e], partials, O, R, AXIS_COLS, Float32(1))


# ---------------------------------------------------------------- reductions
def _axis_flags(s: ShapeN, axes: List[Int]) raises -> InlineArray[Bool, RN_MAX_RANK]:
    """Which axes of s are reduced; an empty list reduces every axis."""
    var red = InlineArray[Bool, RN_MAX_RANK](fill=False)
    if len(axes) == 0:
        for k in range(s.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
            red[k] = True
        return red
    for k in range(len(axes)):  # small-loop(len(axes) <= 8: axis list, shape metadata)
        var ax = rn_normalize_axis(axes[k], s.ndim)
        if red[ax]:
            raise Error("reduction axes must be distinct")
        red[ax] = True
    return red


def _reduce_desc(
    s: ShapeN, red: InlineArray[Bool, RN_MAX_RANK], base: Int, strides: InlineArray[Int, RN_MAX_RANK],
) -> List[Int]:
    """The (F) descriptor of a reduction of s: kept records then reduced
    records, columns [extent, 0, read stride, logical row-major stride]."""
    var lrm = _row_major(s)
    var nk = 0
    for k in range(s.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        if not red[k]:
            nk += 1
    var w = List[Int]()
    w.append(nk)
    w.append(s.ndim - nk)
    w.append(base)
    w.append(0)
    for k in range(s.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        if not red[k]:
            _push_rec(w, s.d[k], 0, strides[k], lrm[k])
    for k in range(s.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        if red[k]:
            _push_rec(w, s.d[k], 0, strides[k], lrm[k])
    return w^


def _kept_shape(s: ShapeN, red: InlineArray[Bool, RN_MAX_RANK]) -> ShapeN:
    var so = s
    for k in range(s.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        if red[k]:
            so.d[k] = 1
    return so


def rankn_sum[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, axes: List[Int], mean: Bool = False,
) raises -> Tensor:
    """The sum (mean when `mean`) of x over `axes` (an empty list: every
    axis), keeping the reduced axes with extent 1. All axes, the last axis,
    every leading axis and a trailing block run the tape's sum on the
    matrix view (its words); other axis sets run the (F) fold of
    autograd/rankn_fold_ops.mojo and record RN_TAG_SUM / RN_TAG_MEAN."""
    var sx = tape.shape_of(x)
    var red = _axis_flags(sx, axes)
    var so = _kept_shape(sx, red)
    var n = sx.ndim
    var nred = 0
    var first = n
    for k in range(n):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        if red[k]:
            nred += 1
            if first == n:
                first = k
    var trailing = nred > 0 and first + nred == n
    var axis = -2
    if nred == n:
        axis = AXIS_ALL
    elif n >= 2 and nred == n - 1 and not red[n - 1]:
        axis = AXIS_ROWS
    elif trailing:
        axis = AXIS_COLS
    if axis != -2:
        var src = x
        if axis == AXIS_COLS and not (nred == 1 and red[n - 1]):
            var O = so.count()
            src = tape.view_nd(ex, x, ShapeN.matrix(O, sx.count() // O))
        var r = tape.mean(ex, src, axis) if mean else tape.sum(ex, src, axis)
        return _as_shape(tape, ex, r, so)
    var xo = tape.operand_nd(ex, x)
    var O = so.count()
    var R = sx.count() // O
    var desc = tape.descriptor(ex, _reduce_desc(sx, red, xo[1], xo[3]))
    var output = tape.reserve(ex, O)
    var factor = identical_div(Float32(1), Float32(R)) if mean else Float32(1)
    var nb = _blocks(R)
    var a = Args()
    a.p0 = xo[0]
    a.p2 = output
    a.p11 = desc
    a.i3 = AG_FOLD_LEAF
    a.i6 = R
    a.i9 = nb
    a.f0 = factor
    if nb == 1:
        a.i4 = 1
        ex.launch[OP_RN_REDUCE_PART](a, O)
    else:
        var partials = tape.reserve(ex, fold_partials(O, R, AXIS_COLS))
        a.p1 = partials
        ex.launch[OP_RN_REDUCE_PART](a, O * nb)
        blocked_fold_finish(ex, output, partials, O, R, AXIS_COLS, factor)
    var saved = Args()
    saved.i0 = sx.count()
    saved.f0 = factor
    if tape.is_recording():
        var orm = _row_major(so)
        var w = List[Int]()
        w.append(n)
        w.append(0)
        w.append(0)
        w.append(0)
        for k in range(n):  # small-loop(ndim <= 8: tensor rank, shape metadata)
            _push_rec(w, sx.d[k], 0 if red[k] else orm[k], 0, 0)
        saved.p11 = tape.descriptor(ex, w)
    return tape.custom_nd(ex, [x], output, so, RN_TAG_MEAN if mean else RN_TAG_SUM, saved)


def rankn_mean[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, axes: List[Int]) raises -> Tensor:
    """The mean of x over `axes` (rankn_sum with mean)."""
    return rankn_sum(tape, ex, x, axes, True)


def rankn_max_argmax[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, axes: List[Int],
) raises -> Tuple[Tensor, FP]:
    """(the max of x over `axes`, keepdims; its argmax): a NaN wins, ties
    take the first position. The argmax buffer holds one int32 pattern an
    output, the row-major position over the reduced axes (for one axis, the
    index along it), valid until the next reset. The gradient flows to that
    position (RN_TAG_MAX)."""
    var sx = tape.shape_of(x)
    var red = _axis_flags(sx, axes)
    var so = _kept_shape(sx, red)
    var xo = tape.operand_nd(ex, x)
    var O = so.count()
    var R = sx.count() // O
    var desc = tape.descriptor(ex, _reduce_desc(sx, red, xo[1], xo[3]))
    var output = tape.reserve(ex, O)
    var arg = tape.reserve(ex, O)
    var nb = _blocks(R)
    var a = Args()
    a.p0 = xo[0]
    a.p2 = output
    a.p3 = arg
    a.p11 = desc
    a.i6 = R
    a.i9 = nb
    if nb == 1:
        a.i4 = 1
        ex.launch[OP_RN_MAX_PART](a, O)
    else:
        var pv = tape.reserve(ex, O * nb)
        var pi = tape.reserve(ex, O * nb)
        a.p1 = pv
        a.p4 = pi
        ex.launch[OP_RN_MAX_PART](a, O * nb)
        var live = nb
        var stride = 1
        # driver-loop: over fold tree levels (ceil(log64(R)) - 1 of them)
        while live > AG_FOLD_LEAF:
            var groups = (live + AG_FOLD_LEAF - 1) // AG_FOLD_LEAF
            var g = Args()
            g.p1 = pv
            g.p4 = pi
            g.i9 = nb
            g.i5 = stride
            g.i6 = live
            ex.launch[OP_RN_MAX_LEVEL](g, groups * O)
            live = groups
            stride *= AG_FOLD_LEAF
        var f = Args()
        f.p1 = pv
        f.p4 = pi
        f.p2 = output
        f.p3 = arg
        f.i9 = nb
        f.i5 = stride
        f.i6 = live
        ex.launch[OP_RN_MAX_FOLD](f, O)
    var saved = Args()
    saved.p10 = arg
    saved.p11 = desc
    saved.i0 = O
    var t = tape.custom_nd(ex, [x], output, so, RN_TAG_MAX, saved)
    return (t, arg)


def rankn_max[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, axes: List[Int]) raises -> Tensor:
    """The max of x over `axes` (rankn_max_argmax without the argmax)."""
    var r = rankn_max_argmax(tape, ex, x, axes)
    return r[0]


# ---------------------------------------------------------------- softmax
def rankn_softmax[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, axis: Int = -1, log: Bool = False,
) raises -> Tensor:
    """softmax (log_softmax when `log`) of x along `axis`. The last axis is
    the tape's softmax on the matrix view; another axis is permute (a view)
    -> the tape's softmax (its words on the permuted matrix) -> the inverse
    permute (a view)."""
    var sx = tape.shape_of(x)
    var n = sx.ndim
    if n <= 1:
        var r = tape.log_softmax(ex, x) if log else tape.softmax(ex, x)
        return _as_shape(tape, ex, r, sx)
    var ax = rn_normalize_axis(axis, n)
    if ax == n - 1:
        var r = tape.log_softmax(ex, x) if log else tape.softmax(ex, x)
        return _as_shape(tape, ex, r, sx)
    var perm = List[Int]()
    for k in range(n):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        if k != ax:
            perm.append(k)
    perm.append(ax)
    var inv = List[Int](length=n, fill=0)
    for k in range(n):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        inv[perm[k]] = k
    var p = tape.permute(ex, x, perm)
    var sp = tape.shape_of(p)
    var s = tape.log_softmax(ex, p) if log else tape.softmax(ex, p)
    s = _as_shape(tape, ex, s, sp)
    return tape.permute(ex, s, inv)


# ---------------------------------------------------------------- rules
@fieldwise_init
struct RankNFoldRules(BackwardRules, ImplicitlyCopyable, Movable):
    """The backward of the rank-N fold lane's CUSTOM tags (136..151)."""

    def supports(self, tag: Int) -> Bool:
        return (tag >= RN_TAG_BINARY + ADD and tag <= RN_TAG_BINARY + DIVIDE) or tag == RN_TAG_SUM or tag == RN_TAG_MEAN or tag == RN_TAG_MAX

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # binary and sum VJPs write every destination word; max writes one
        # word an output into a zero-filled gradient
        return tag == RN_TAG_MAX

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag >= RN_TAG_BINARY + ADD and tag <= RN_TAG_BINARY + DIVIDE:
            _binary_vjp(ex, saved, output, upstream, gradients)
        elif tag == RN_TAG_SUM or tag == RN_TAG_MEAN:
            var a = Args()
            a.p0 = upstream
            a.p1 = gradients[0]
            a.p11 = saved.p11
            a.f0 = saved.f0
            ex.launch[OP_RN_BCAST_VJP](a, saved.i0)
        elif tag == RN_TAG_MAX:
            var a = Args()
            a.p0 = upstream
            a.p1 = gradients[0]
            a.p3 = saved.p10
            a.p11 = saved.p11
            ex.launch[OP_RN_MAX_VJP](a, saved.i0)
        else:
            raise Error("rank-N fold rules: unknown tag")
