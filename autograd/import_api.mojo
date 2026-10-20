# SPDX-License-Identifier: Apache-2.0
"""The import family on the tape (lane ag-import-kernels, docs/plans/
MODEL_IMPORT.md section 3.3): the tape calls of the elementwise kinds an
imported graph needs (autograd/import_ops.mojo), `expand` through the
rank-N strided copy, `argmax` over the fold lane's max positions, constant
`pad`, and `ImportRules`, their backward.

Every entry runs one launch of a body of autograd/import_ops.mojo (or of
the rank-N core's OP_RN_COPY_STRIDED for expand, or the fold lane's max for
argmax) and records one CUSTOM node with a tag of IMP_TAG_FIRST ..
IMP_TAG_LAST (192..207, above autograd/rankn.mojo RN_TAG_END), so an
inference session gets the forward words and a recording session can
differentiate the same composition. Operands are read through
Tape.operand_nd (buffer, base offset, strides), so a pending permute feeds
the binary kinds and expand with no copy; the unary kinds read the dense
value (Tape.value_of).

Tags:
  IMP_TAG_UNARY 192: saved i0 words, i1 kind, f0 / f1 its parameters.
  IMP_TAG_BINARY 193: saved p0 a's buffer, p1 b's (unbound when i9 = 1),
    p2 c's (WHERE; unbound when i10 or i11 = 1), p11 the forward
    descriptor, p10 the scratch of a broadcast operand's VJP terms, p9 / p8
    a's / b's reduction descriptor, p7 / p6 their fold partials (R > 64
    only); i0 output words, i1 kind, i2 / i3 a's O / R, i4 / i5 b's, i6 /
    i7 1 when a / b has the output's shape.
  IMP_TAG_EXPAND 194: saved p11 the reduction descriptor over the upstream,
    p7 the fold partials (R > 64 only); i2 O (the source's words), i3 R.
  IMP_TAG_ARGMAX 195: saved i1 the output form; no gradient.
  IMP_TAG_PAD 196: saved p11 the VJP descriptor, i0 the source's words.

A broadcast operand's gradient (binary kinds, expand) is the sum of its
VJP terms over the broadcast axes: the terms are stored once (ftz), then
OP_RN_REDUCE_PART's leaves and the blocked fold (autograd/fold.mojo
blocked_fold_finish) add them in order (F) of autograd/rankn_fold_ops.mojo.

Host loops here run over tensor axes (at most RN_MAX_RANK = 8) to build
descriptors and over node inputs; none over rows or elements."""
from std.collections import InlineArray
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.ops import AXIS_COLS, AG_FOLD_LEAF
from autograd.fold import blocked_fold_finish, fold_partials
from autograd.rankn import ShapeN, RN_MAX_RANK, rn_broadcast, rn_normalize_axis, rn_desc_view
from autograd.rankn_ops import rn_copy_strided
from autograd.rankn_fold import rankn_max_argmax
from autograd.rankn_fold_ops import OP_RN_REDUCE_PART
from autograd.rankn_api import RankNRules, rn_shape
from autograd.import_ops import (
    OP_IMP_UNARY_FWD, OP_IMP_UNARY_VJP, OP_IMP_BIN_FWD, OP_IMP_BIN_VJP, OP_IMP_ARGMAX,
    OP_IMP_PAD, OP_IMP_PAD_VJP, IMP_WHERE, IMP_UNARY_LAST, IMP_BINARY_LAST,
    IMP_ARGMAX_INT32, IMP_ARGMAX_FLOAT, unary_has_gradient, binary_has_gradient,
)

comptime IMP_TAG_FIRST = 192
comptime IMP_TAG_LAST = 207
comptime IMP_TAG_UNARY = 192
comptime IMP_TAG_BINARY = 193
comptime IMP_TAG_EXPAND = 194
comptime IMP_TAG_ARGMAX = 195
comptime IMP_TAG_PAD = 196


# ---------------------------------------------------------------- shapes
def _row_major(s: ShapeN) -> InlineArray[Int, RN_MAX_RANK]:
    var r = InlineArray[Int, RN_MAX_RANK](fill=0)
    var acc = 1
    var k = s.ndim - 1
    while k >= 0:  # small-loop(ndim <= 8: tensor rank, shape metadata)
        r[k] = acc
        acc *= s.d[k]
        k -= 1
    return r^


def _aligned_dim(s: ShapeN, so: ShapeN, j: Int) -> Int:
    """s's extent on output axis j (s right-aligned to so; 1 on an axis s
    does not have)."""
    var k = j - (so.ndim - s.ndim)
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


def _blocks(R: Int) -> Int:
    return (R + AG_FOLD_LEAF - 1) // AG_FOLD_LEAF


def _push(mut w: List[Int], c0: Int, c1: Int, c2: Int, c3: Int):
    w.append(c0)
    w.append(c1)
    w.append(c2)
    w.append(c3)


def _reduce_words(s: ShapeN, so: ShapeN) -> List[Int]:
    """The OP_RN_REDUCE_PART descriptor (autograd/rankn_fold_ops.mojo) of
    a broadcast operand's gradient: a dense output-shaped source (base 0),
    kept records = the output axes the operand carries (its row-major
    index is its gradient word), reduced records = the axes it broadcasts
    over; column 2 the source's row-major strides."""
    var orm = _row_major(so)
    var kept = List[Int]()
    var red = List[Int]()
    for j in range(so.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        if _aligned_dim(s, so, j) == so.d[j]:
            kept.append(j)
        else:
            red.append(j)
    var w = List[Int]()
    _push(w, len(kept), len(red), 0, 0)
    for k in range(len(kept)):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        var j = kept[k]
        _push(w, so.d[j], 0, orm[j], 0)
    for k in range(len(red)):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        var j = red[k]
        _push(w, so.d[j], 0, orm[j], 0)
    return w^


def _fold_into[E: Exec](
    mut ex: E, src: FP, dst: FP, desc: FP, partials: FP, O: Int, R: Int,
) raises:
    """dst[o] = the sum of src's terms over the reduced records of `desc`,
    order (F): OP_RN_REDUCE_PART leaves of 64 ascending terms, then the
    blocked fold when an output has more than one leaf."""
    var nb = _blocks(R)
    var a = Args()
    a.p0 = src
    a.p2 = dst
    a.p11 = desc
    a.i3 = AG_FOLD_LEAF
    a.i6 = R
    a.i9 = nb
    a.f0 = Float32(1)
    if nb == 1:
        a.i4 = 1
        ex.launch[OP_RN_REDUCE_PART](a, O)
    else:
        a.p1 = partials
        ex.launch[OP_RN_REDUCE_PART](a, O * nb)
        blocked_fold_finish(ex, dst, partials, O, R, AXIS_COLS, Float32(1))


# ---------------------------------------------------------------- unary
def imp_unary[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, kind: Int, f0: Float32 = 0, f1: Float32 = 0,
) raises -> Tensor:
    """kind(x) elementwise (a unary kind of autograd/import_ops.mojo, its
    parameters in f0 and f1): one OP_IMP_UNARY_FWD launch, x's shape."""
    if kind < 1 or kind > IMP_UNARY_LAST:
        raise Error("import: unknown unary kind " + String(kind))
    var s = tape.shape_of(x)
    var n = s.count()
    var src = tape.value_of(ex, x)
    var output = tape.reserve(ex, n)
    var a = Args()
    a.p0 = src
    a.p1 = output
    a.i1 = kind
    a.f0 = f0
    a.f1 = f1
    ex.launch[OP_IMP_UNARY_FWD](a, n)
    var saved = Args()
    saved.i0 = n
    saved.i1 = kind
    saved.f0 = f0
    saved.f1 = f1
    return tape.custom_nd(ex, [x], output, s, IMP_TAG_UNARY, saved)


# ---------------------------------------------------------------- binary and where
def _imp_bin_nd[E: Exec](mut tape: Tape, mut ex: E, ops: List[Tensor], kind: Int) raises -> Tensor:
    """The broadcasting binary kinds over operands ops[0] (a), ops[1] (b)
    and, for WHERE, ops[2] (the condition c)."""
    var three = kind == IMP_WHERE
    var sa = tape.shape_of(ops[0])
    var sb = tape.shape_of(ops[1])
    var sc = sa
    var so = rn_broadcast(sa, sb)
    if three:
        sc = tape.shape_of(ops[2])
        so = rn_broadcast(so, sc)
    var ao = tape.operand_nd(ex, ops[0])
    var bo = tape.operand_nd(ex, ops[1])
    var abuf = ao[0]
    var bbuf = bo[0]
    var cbuf = abuf
    var cbase = 0
    var as_ = ao[3].copy()
    var bs = bo[3].copy()
    var cs = ao[3].copy()
    if three:
        var co = tape.operand_nd(ex, ops[2])
        cbuf = co[0]
        cbase = co[1]
        cs = co[3].copy()
    var count = so.count()
    var orm = _row_major(so)
    var w = List[Int]()
    w.append(so.ndim)
    w.append(0)
    w.append(ao[1])
    w.append(bo[1])
    w.append(cbase)
    for j in range(so.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        w.append(so.d[j])
        w.append(orm[j])
        w.append(_aligned_stride(sa, as_, so, j))
        w.append(_aligned_stride(sb, bs, so, j))
        w.append(_aligned_stride(sc, cs, so, j) if three else 0)
    var desc = tape.descriptor(ex, w)
    var output = tape.reserve(ex, count)
    var a = Args()
    a.p0 = abuf
    if Int(bbuf) == Int(abuf):
        a.i9 = 1
    else:
        a.p1 = bbuf
    if three:
        if Int(cbuf) == Int(abuf):
            a.i10 = 1
        elif Int(cbuf) == Int(bbuf):
            a.i11 = 1
        else:
            a.p2 = cbuf
    a.p3 = output
    a.p11 = desc
    a.i1 = kind
    ex.launch[OP_IMP_BIN_FWD](a, count)
    var saved = Args()
    saved.p0 = a.p0
    saved.p1 = a.p1
    saved.p2 = a.p2
    saved.i9 = a.i9
    saved.i10 = a.i10
    saved.i11 = a.i11
    saved.p11 = desc
    saved.i0 = count
    saved.i1 = kind
    saved.i6 = 1 if sa.same(so) else 0
    saved.i7 = 1 if sb.same(so) else 0
    if tape.is_recording() and binary_has_gradient(kind):
        var scratch = dummy_ptr()
        if saved.i6 == 0 or saved.i7 == 0:
            scratch = tape.reserve(ex, count)
        saved.p10 = scratch
        for e in range(2):  # driver-loop: over node inputs (the two differentiable operands)
            var s = sa if e == 0 else sb
            if s.same(so):
                continue
            var O = s.count()
            var R = count // O
            var rdesc = tape.descriptor(ex, _reduce_words(s, so))
            var partials = dummy_ptr()
            if _blocks(R) > 1:
                partials = tape.reserve(ex, fold_partials(O, R, AXIS_COLS))
            if e == 0:
                saved.p9 = rdesc
                saved.p7 = partials
                saved.i2 = O
                saved.i3 = R
            else:
                saved.p8 = rdesc
                saved.p6 = partials
                saved.i4 = O
                saved.i5 = R
    return tape.custom_nd(ex, ops, output, so, IMP_TAG_BINARY, saved)


def imp_binary[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, y: Tensor, kind: Int) raises -> Tensor:
    """x kind y with NumPy broadcasting over any rank (a binary kind of
    autograd/import_ops.mojo other than WHERE): one OP_IMP_BIN_FWD launch."""
    if kind < 1 or kind > IMP_BINARY_LAST or kind == IMP_WHERE:
        raise Error("import: unknown binary kind " + String(kind))
    var ops = List[Tensor]()
    ops.append(x)
    ops.append(y)
    return _imp_bin_nd(tape, ex, ops, kind)


def imp_where[E: Exec](mut tape: Tape, mut ex: E, cond: Tensor, x: Tensor, y: Tensor) raises -> Tensor:
    """x where cond is nonzero, else y, the three broadcast together as in
    NumPy: one OP_IMP_BIN_FWD launch. The gradient flows to x and y, none to
    the condition."""
    var ops = List[Tensor]()
    ops.append(x)
    ops.append(y)
    ops.append(cond)
    return _imp_bin_nd(tape, ex, ops, IMP_WHERE)


def _binary_vjp[E: Exec](mut ex: E, saved: Args, output: FP, upstream: FP, gradients: List[FP]) raises:
    var count = saved.i0
    for e in range(2):  # driver-loop: over node inputs (the two differentiable operands)
        var a = Args()
        a.p0 = saved.p0
        a.p1 = saved.p1
        a.p2 = saved.p2
        a.i9 = saved.i9
        a.i10 = saved.i10
        a.i11 = saved.i11
        a.p3 = output
        a.p5 = upstream
        a.p11 = saved.p11
        a.i1 = saved.i1
        a.i8 = e
        var same = saved.i6 if e == 0 else saved.i7
        if same == 1:
            a.p4 = gradients[e]
            ex.launch[OP_IMP_BIN_VJP](a, count)
            continue
        a.p4 = saved.p10
        ex.launch[OP_IMP_BIN_VJP](a, count)
        var O = saved.i2 if e == 0 else saved.i4
        var R = saved.i3 if e == 0 else saved.i5
        var rdesc = saved.p9 if e == 0 else saved.p8
        var partials = saved.p7 if e == 0 else saved.p6
        _fold_into(ex, saved.p10, gradients[e], rdesc, partials, O, R)


# ---------------------------------------------------------------- expand
def imp_expand[E: Exec](mut tape: Tape, mut ex: E, x: Tensor, dims: List[Int]) raises -> Tensor:
    """x broadcast to the NumPy broadcast of its shape and `dims` (ONNX
    Expand): one OP_RN_COPY_STRIDED launch with zero strides on the
    broadcast axes, the source words copied. Its gradient sums the upstream
    over the broadcast axes (order (F))."""
    var sx = tape.shape_of(x)
    var target = rn_shape(dims, "expand")
    var so = rn_broadcast(sx, target)
    if so.same(sx):
        return tape.view_nd(ex, x, so)
    var xo = tape.operand_nd(ex, x)
    var xs = xo[3].copy()
    var st = InlineArray[Int, RN_MAX_RANK](fill=0)
    for j in range(so.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        st[j] = _aligned_stride(sx, xs, so, j)
    var words = List[Int]()
    rn_desc_view(words, so, st)
    var desc = tape.descriptor(ex, words)
    var count = so.count()
    var output = tape.reserve(ex, count)
    rn_copy_strided(ex, output, xo[0], xo[1], desc, 0, count)
    var saved = Args()
    var O = sx.count()
    var R = count // O
    saved.i2 = O
    saved.i3 = R
    if tape.is_recording():
        saved.p11 = tape.descriptor(ex, _reduce_words(sx, so))
        if _blocks(R) > 1:
            saved.p7 = tape.reserve(ex, fold_partials(O, R, AXIS_COLS))
    return tape.custom_nd(ex, [x], output, so, IMP_TAG_EXPAND, saved)


# ---------------------------------------------------------------- argmax
def imp_argmax[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, axis: Int, keepdims: Bool, form: Int,
) raises -> Tensor:
    """The position of the largest value along `axis` (a NaN wins, ties take
    the first position: the fold lane's max), as int32 words in float
    storage (form IMP_ARGMAX_INT32) or as float32 values
    (IMP_ARGMAX_FLOAT). keepdims keeps the axis with length 1. No
    gradient."""
    if form != IMP_ARGMAX_INT32 and form != IMP_ARGMAX_FLOAT:
        raise Error("import: argmax form must be 0 (int32 words) or 1 (float values)")
    var sx = tape.shape_of(x)
    if sx.ndim < 1:
        raise Error("argmax takes a tensor of at least one dimension")
    var ax = rn_normalize_axis(axis, sx.ndim)
    var axes = List[Int]()
    axes.append(ax)
    var r = rankn_max_argmax(tape, ex, x, axes)
    var sk = tape.shape_of(r[0])
    var O = sk.count()
    var output = tape.reserve(ex, O)
    var a = Args()
    a.p0 = r[1]
    a.p1 = output
    a.i1 = form
    ex.launch[OP_IMP_ARGMAX](a, O)
    var saved = Args()
    saved.i1 = form
    var t = tape.custom_nd(ex, [x], output, sk, IMP_TAG_ARGMAX, saved)
    if keepdims:
        return t
    var dims = List[Int]()
    for k in range(sx.ndim):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        if k != ax:
            dims.append(sx.d[k])
    return tape.view_nd(ex, t, rn_shape(dims, "argmax"))


# ---------------------------------------------------------------- pad
def imp_pad[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, pads: List[Int], value: Float32,
) raises -> Tensor:
    """x padded with `value` (ONNX Pad, constant mode): pads[k] words before
    axis k and pads[ndim + k] after it; a negative pad crops. One
    OP_IMP_PAD launch; the gradient is the upstream narrowed back."""
    var sx = tape.shape_of(x)
    var nd = sx.ndim
    if nd < 1:
        raise Error("pad takes a tensor of at least one dimension")
    if len(pads) != 2 * nd:
        raise Error("pad takes two pads (before and after) for each of the tensor's "
                    + String(nd) + " axes")
    var dims = List[Int]()
    for k in range(nd):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        var e = sx.d[k] + pads[k] + pads[nd + k]
        if e < 1:
            raise Error("pad: axis " + String(k) + " would have no elements")
        dims.append(e)
    var so = rn_shape(dims, "pad")
    var xo = tape.operand_nd(ex, x)
    var xs = xo[3].copy()
    var w = List[Int]()
    w.append(nd)
    w.append(xo[1])
    for k in range(nd):  # small-loop(ndim <= 8: tensor rank, shape metadata)
        _push(w, so.d[k], pads[k], sx.d[k], xs[k])
    var desc = tape.descriptor(ex, w)
    var count = so.count()
    var output = tape.reserve(ex, count)
    var a = Args()
    a.p0 = xo[0]
    a.p1 = output
    a.p11 = desc
    a.f0 = value
    ex.launch[OP_IMP_PAD](a, count)
    var saved = Args()
    saved.i0 = sx.count()
    if tape.is_recording():
        var orm = _row_major(so)
        var v = List[Int]()
        v.append(nd)
        v.append(0)
        for k in range(nd):  # small-loop(ndim <= 8: tensor rank, shape metadata)
            _push(v, sx.d[k], pads[k], orm[k], so.d[k])
        saved.p11 = tape.descriptor(ex, v)
    return tape.custom_nd(ex, [x], output, so, IMP_TAG_PAD, saved)


# ---------------------------------------------------------------- rules
@fieldwise_init
struct ImportRules(BackwardRules):
    """RankNRules plus the import family's tags (192..207)."""

    def supports(self, tag: Int) -> Bool:
        if tag >= IMP_TAG_FIRST and tag <= IMP_TAG_LAST:
            return (tag == IMP_TAG_UNARY or tag == IMP_TAG_BINARY or tag == IMP_TAG_EXPAND
                    or tag == IMP_TAG_ARGMAX or tag == IMP_TAG_PAD)
        return RankNRules().supports(tag)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag >= IMP_TAG_FIRST and tag <= IMP_TAG_LAST:
            # the VJP bodies and the fold write every destination word; the
            # kinds without a gradient, argmax and where's condition write
            # none, so those destinations are zero-filled
            if tag == IMP_TAG_UNARY:
                return not unary_has_gradient(saved.i1)
            if tag == IMP_TAG_BINARY:
                return edge == 2 or not binary_has_gradient(saved.i1)
            return tag == IMP_TAG_ARGMAX
        return RankNRules().zeroed_destination(tag, saved, edge)

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag >= IMP_TAG_FIRST and tag <= IMP_TAG_LAST:
            return False
        return RankNRules().writes_total(tag, saved, edge)

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == IMP_TAG_UNARY:
            if not unary_has_gradient(saved.i1):
                return
            var a = Args()
            a.p0 = inputs[0]
            a.p1 = output
            a.p2 = upstream
            a.p3 = gradients[0]
            a.i1 = saved.i1
            a.f0 = saved.f0
            a.f1 = saved.f1
            ex.launch[OP_IMP_UNARY_VJP](a, saved.i0)
        elif tag == IMP_TAG_BINARY:
            if not binary_has_gradient(saved.i1):
                return
            _binary_vjp(ex, saved, output, upstream, gradients)
        elif tag == IMP_TAG_EXPAND:
            _fold_into(ex, upstream, gradients[0], saved.p11, saved.p7, saved.i2, saved.i3)
        elif tag == IMP_TAG_ARGMAX:
            return
        elif tag == IMP_TAG_PAD:
            var a = Args()
            a.p0 = upstream
            a.p1 = gradients[0]
            a.p11 = saved.p11
            ex.launch[OP_IMP_PAD_VJP](a, saved.i0)
        elif tag >= IMP_TAG_FIRST and tag <= IMP_TAG_LAST:
            raise Error("import rules: unknown tag")
        else:
            RankNRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
