# SPDX-License-Identifier: Apache-2.0
# RANK-N LAYOUT OPS (lane ag-rankn-layout, docs/plans/AUTOGRAD_RANKN.md
# sections 2.5 and 2.6): concat, narrow, split and take along any axis of a
# tape tensor, and RankNLayoutRules, the backward of the two custom tags.
#
#   concat   CUSTOM tag RN_CONCAT (160). Forward: one OP_RN_SLAB_COPY per
#            operand (each operand read in place through operand_nd's
#            strides, so a pending permute costs no copy) into its slab of
#            the output. Backward: one slab copy per operand from the
#            upstream gradient into the operand's gradient; every word is
#            written once, so the destination is not zero-filled.
#   narrow   CUSTOM tag RN_NARROW (161). The slab is copied into the node's
#            value at record time (one OP_RN_SLAB_COPY: a pending narrow
#            view needs the tape's generalised _pending / _materialize_at /
#            operand_nd to know this node's saved layout, which the core
#            lane owns; the words are the same either way, a copy moves
#            bits). Backward: the operand's gradient destination is zero
#            filled, then the upstream is copied into its slab; a second
#            consumer goes through the tape's scratch + OP_AG_ACCUMULATE as
#            any second consumer does (adding +0.0 outside the slab is that
#            word under ftz).
#   split    a list of narrows.
#   take     a composition, no kernel: permute the axis to the front (a
#            view), materialise it as the [dim, rest] matrix, Tape.embedding
#            (its executor id check, its gather and its run-sorted
#            scatter-add VJP: the embedding profile's words), then views
#            back. Its words are the embedding's.
#
# Every rank-2 caller of the existing tape ops is untouched: these are new
# entry points only. Descriptors are built on the host from at most 8 axes
# (shape metadata, no data) and uploaded once per recorded node.
from autograd.tape import Tape, Tensor, IdsHandle, BackwardRules
from autograd.rankn import RN_MAX_RANK, ShapeN, rn_normalize_axis, rn_desc_shape, rn_desc_strides
from autograd.rankn_layout_ops import (
    OP_RN_SLAB_COPY, RN_CONCAT, RN_NARROW, RNL_REC_WORDS,
)
from sequence.exec_trait import Exec
from sequence.ops import FP, Args


# ---------------------------------------------------------------- shapes
def rn_shape_of_dims(dims: List[Int]) raises -> ShapeN:
    """The ShapeN with these dims, row-major (unused slots 1)."""
    if len(dims) > RN_MAX_RANK:
        raise Error("tensors have at most " + String(RN_MAX_RANK) + " dimensions")
    var d = InlineArray[Int, RN_MAX_RANK](fill=1)
    for k in range(len(dims)):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        if dims[k] < 1:
            raise Error("tensor dimensions must be positive")
        d[k] = dims[k]
    return ShapeN(len(dims), d)


def _dense_strides(s: ShapeN) -> InlineArray[Int, RN_MAX_RANK]:
    """Row-major strides of s (0 on an unused axis)."""
    var r = InlineArray[Int, RN_MAX_RANK](fill=0)
    for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        r[k] = s.stride(k)
    return r


def _put_record(
    mut words: List[Int], src_base: Int, dst_base: Int, shape: ShapeN,
    src: InlineArray[Int, RN_MAX_RANK], dst: InlineArray[Int, RN_MAX_RANK],
) raises:
    """Append one slab-copy record (autograd/rankn_layout_ops.mojo RNL_REC_*)."""
    var at = len(words)
    words.append(src_base)
    words.append(dst_base)
    rn_desc_shape(words, shape)
    rn_desc_strides(words, src)
    rn_desc_strides(words, dst)
    if len(words) - at != RNL_REC_WORDS:
        raise Error("rank-N layout: descriptor record has the wrong length")


def _slab_copy[E: Exec](mut ex: E, src: FP, dst: FP, desc: FP, record: Int, count: Int) raises:
    """One OP_RN_SLAB_COPY launch: record `record` of `desc`, `count` threads."""
    var a = Args()
    a.p0 = src
    a.p1 = dst
    a.p11 = desc
    a.i0 = record * RNL_REC_WORDS
    ex.launch[OP_RN_SLAB_COPY](a, count)


# ---------------------------------------------------------------- concat
def rankn_concat[E: Exec](mut tape: Tape, mut ex: E, xs: List[Tensor], axis: Int) raises -> Tensor:
    """The tensors of xs joined along `axis`.

    Every tensor has the same rank and the same dims except along `axis`;
    the result's dim there is their sum. Its gradient gives each input the
    slab of the output gradient it filled.
    """
    if len(xs) < 1:
        raise Error("concat: at least one tensor is needed")
    var first = tape.shape_of(xs[0])
    if first.ndim < 1:
        raise Error("concat: tensors need at least one dimension")
    var a = rn_normalize_axis(axis, first.ndim)
    var total = 0
    for j in range(len(xs)):  # driver-loop: over concat operands
        var s = tape.shape_of(xs[j])
        if s.ndim != first.ndim:
            raise Error("concat: every tensor must have the same number of dimensions")
        for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            if k != a and s.d[k] != first.d[k]:
                raise Error("concat: dimensions differ outside the concat axis")
        total += s.d[a]
    var out_dims = List[Int]()
    for k in range(first.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        out_dims.append(total if k == a else first.d[k])
    var out_shape = rn_shape_of_dims(out_dims)
    var out_strides = _dense_strides(out_shape)
    var output = tape.reserve(ex, out_shape.count())
    # records 0 .. n-1: the forward copies; n .. 2n-1: the backward copies
    var words = List[Int]()
    var sources = List[FP]()
    var counts = List[Int]()
    var offset = 0
    for j in range(len(xs)):  # driver-loop: over concat operands
        var op = tape.operand_nd(ex, xs[j])
        var s = op[2]
        sources.append(op[0])
        counts.append(s.count())
        _put_record(words, op[1], offset * out_shape.stride(a), s, op[3].copy(), out_strides.copy())
        offset += s.d[a]
    offset = 0
    for j in range(len(xs)):  # driver-loop: over concat operands
        var s = tape.shape_of(xs[j])
        _put_record(words, offset * out_shape.stride(a), 0, s, out_strides.copy(), _dense_strides(s))
        offset += s.d[a]
    var desc = tape.descriptor(ex, words)
    for j in range(len(xs)):  # driver-loop: over concat operands
        _slab_copy(ex, sources[j], output, desc, j, counts[j])
    var saved = Args()
    saved.p11 = desc
    saved.i0 = len(xs)
    saved.i1 = a
    return tape.custom_nd(ex, xs, output, out_shape, RN_CONCAT, saved)


# ---------------------------------------------------------------- narrow / split
def rankn_narrow[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, axis: Int, start: Int, length: Int,
) raises -> Tensor:
    """The `length` slices of x from `start` along `axis`.

    Its gradient places the output gradient in that slab of x's gradient
    and zeros elsewhere.
    """
    var op = tape.operand_nd(ex, x)
    var s = op[2]
    if s.ndim < 1:
        raise Error("narrow: the tensor needs at least one dimension")
    var a = rn_normalize_axis(axis, s.ndim)
    if length < 1 or start < 0 or start > s.d[a] - length:
        raise Error("narrow: the slab [start, start + length) must lie inside the axis")
    var dims = List[Int]()
    for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        dims.append(length if k == a else s.d[k])
    var out_shape = rn_shape_of_dims(dims)
    var out_strides = _dense_strides(out_shape)
    var src_strides = op[3].copy()
    var words = List[Int]()
    # record 0: the slab of the source (read in place through its strides)
    _put_record(words, op[1] + start * src_strides[a], 0, out_shape, src_strides.copy(), out_strides.copy())
    # record 1: the upstream into the slab of x's dense row-major gradient
    _put_record(words, 0, start * s.stride(a), out_shape, out_strides.copy(), _dense_strides(s))
    var desc = tape.descriptor(ex, words)
    var output = tape.reserve(ex, out_shape.count())
    _slab_copy(ex, op[0], output, desc, 0, out_shape.count())
    var saved = Args()
    saved.p0 = op[0]
    saved.p11 = desc
    saved.i1 = out_shape.count()
    saved.i2 = a
    saved.i3 = start
    saved.i4 = length
    saved.i9 = 0  # copied: not a pending view
    return tape.custom_nd(ex, [x], output, out_shape, RN_NARROW, saved)


def rankn_split[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, sizes: List[Int], axis: Int,
) raises -> List[Tensor]:
    """x cut along `axis` into consecutive pieces of the given sizes, which
    must add up to x's dim there."""
    var s = tape.shape_of(x)
    if s.ndim < 1:
        raise Error("split: the tensor needs at least one dimension")
    var a = rn_normalize_axis(axis, s.ndim)
    var covered = 0
    for j in range(len(sizes)):  # driver-loop: over split pieces
        covered += sizes[j]
    if len(sizes) < 1 or covered != s.d[a]:
        raise Error("split: the sizes must add up to the dimension along the axis")
    var pieces = List[Tensor]()
    var start = 0
    for j in range(len(sizes)):  # driver-loop: over split pieces
        pieces.append(rankn_narrow(tape, ex, x, a, start, sizes[j]))
        start += sizes[j]
    return pieces^


# ---------------------------------------------------------------- take
def rankn_take[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, ids: IdsHandle, axis: Int,
) raises -> Tensor:
    """The slices of x at the ids along `axis`: that dim becomes the id count.

    An id outside [0, x's dim along axis) is refused before any slice is
    read. The gradient of x sums each slice's gradients in ascending id
    position.
    """
    var s = tape.shape_of(x)
    if s.ndim < 1:
        raise Error("take: the tensor needs at least one dimension")
    var a = rn_normalize_axis(axis, s.ndim)
    var dim = s.d[a]
    var rest = s.count() // dim
    var n = tape.ids_count(ids)
    var src = x
    if a != 0:
        # the axis to the front: [a, 0, .., a-1, a+1, ..]
        var perm = List[Int]()
        perm.append(a)
        for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            if k != a:
                perm.append(k)
        src = tape.permute(ex, x, perm)
        tape.materialize(ex, src)
    var table = src
    if s.ndim != 2:
        table = tape.view_nd(ex, src, ShapeN.matrix(dim, rest))
    var e = tape.embedding(ex, table, ids)
    var y = e
    if s.ndim != 2:
        var dims = List[Int]()
        dims.append(n)
        for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            if k != a:
                dims.append(s.d[k])
        y = tape.view_nd(ex, e, rn_shape_of_dims(dims))
    if a != 0:
        # the front axis back to `a`: [1, .., a, 0, a+1, ..]
        var inverse = List[Int]()
        for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            if k < a:
                inverse.append(k + 1)
            elif k == a:
                inverse.append(0)
            else:
                inverse.append(k)
        y = tape.permute(ex, y, inverse)
    return y


# ---------------------------------------------------------------- backward
@fieldwise_init
struct RankNLayoutRules(BackwardRules):
    """The backward of concat (RN_CONCAT) and narrow (RN_NARROW)."""

    def supports(self, tag: Int) -> Bool:
        return tag == RN_CONCAT or tag == RN_NARROW

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # concat writes every word of every operand's destination; narrow
        # writes only its slab, so the rest must be zeros
        return tag == RN_NARROW

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == RN_CONCAT:
            var n = saved.i0
            for j in range(len(gradients)):  # driver-loop: over concat operands
                _slab_copy(ex, upstream, gradients[j], saved.p11, n + j, rows[j] * cols[j])
        elif tag == RN_NARROW:
            _slab_copy(ex, upstream, gradients[0], saved.p11, 1, saved.i1)
        else:
            raise Error("rank-N layout: unknown custom tag")
