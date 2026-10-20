# SPDX-License-Identifier: Apache-2.0
"""Rank-N glue shared by the operation session (autograd/ops_api.mojo) and
the model trainer (autograd/model.mojo): shape metadata helpers over at
most RN_MAX_RANK axes, the free-view helpers, the along-an-axis softmax
composition, and `RankNRules`, the one rule set that delegates each custom
tag block to its lane's rules (lane ag-rankn-python, docs/plans/
AUTOGRAD_RANKN.md section 6, lane 6).

No kernel lives here and no word is computed here: every helper is shape
metadata (`# small-loop(ndim: ...)`) or a composition of tape methods and
the rank-N lanes' entries (autograd/rankn_fold.mojo, rankn_bmm.mojo,
rankn_layout.mojo, rankn_conv.mojo), so a composition built from these
has the words of the ops it calls.

Tag blocks (plan section 7): 136..151 fold, 152..159 bmm, 160..175 layout,
176..191 conv; every other tag goes to NeuralRules (LAYER_NORM), so a graph
without rank-N nodes backpropagates exactly as with NeuralRules."""
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.layers import NeuralRules
from autograd.rankn import RN_MAX_RANK, ShapeN, rn_normalize_axis
from autograd.rankn_fold import RankNFoldRules
from autograd.rankn_bmm import RankNBmmRules
from autograd.rankn_layout import RankNLayoutRules
from autograd.rules_conv2d import Conv2dRules

comptime RN_TAG_FOLD_FIRST = 136
comptime RN_TAG_FOLD_LAST = 151
comptime RN_TAG_BMM_FIRST = 152
comptime RN_TAG_BMM_LAST = 159
comptime RN_TAG_LAYOUT_FIRST = 160
comptime RN_TAG_LAYOUT_LAST = 175
comptime RN_TAG_CONV_FIRST = 176
comptime RN_TAG_CONV_LAST = 191

# Operands of one concat node (plan 2.5: p0..p9 operands, p10 output, p11
# the descriptor); a longer list chains concat nodes.
comptime RN_CONCAT_MAX = 10


@fieldwise_init
struct RankNRules(BackwardRules):
    """NeuralRules plus every rank-N lane's custom rules, by tag block."""

    def supports(self, tag: Int) -> Bool:
        if tag >= RN_TAG_FOLD_FIRST and tag <= RN_TAG_FOLD_LAST:
            return RankNFoldRules().supports(tag)
        if tag >= RN_TAG_BMM_FIRST and tag <= RN_TAG_BMM_LAST:
            return RankNBmmRules().supports(tag)
        if tag >= RN_TAG_LAYOUT_FIRST and tag <= RN_TAG_LAYOUT_LAST:
            return RankNLayoutRules().supports(tag)
        if tag >= RN_TAG_CONV_FIRST and tag <= RN_TAG_CONV_LAST:
            return Conv2dRules().supports(tag)
        return NeuralRules().supports(tag)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag >= RN_TAG_FOLD_FIRST and tag <= RN_TAG_FOLD_LAST:
            return RankNFoldRules().zeroed_destination(tag, saved, edge)
        if tag >= RN_TAG_BMM_FIRST and tag <= RN_TAG_BMM_LAST:
            return RankNBmmRules().zeroed_destination(tag, saved, edge)
        if tag >= RN_TAG_LAYOUT_FIRST and tag <= RN_TAG_LAYOUT_LAST:
            return RankNLayoutRules().zeroed_destination(tag, saved, edge)
        if tag >= RN_TAG_CONV_FIRST and tag <= RN_TAG_CONV_LAST:
            return Conv2dRules().zeroed_destination(tag, saved, edge)
        return NeuralRules().zeroed_destination(tag, saved, edge)

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag >= RN_TAG_FOLD_FIRST and tag <= RN_TAG_FOLD_LAST:
            return RankNFoldRules().writes_total(tag, saved, edge)
        if tag >= RN_TAG_BMM_FIRST and tag <= RN_TAG_BMM_LAST:
            return RankNBmmRules().writes_total(tag, saved, edge)
        if tag >= RN_TAG_LAYOUT_FIRST and tag <= RN_TAG_LAYOUT_LAST:
            return RankNLayoutRules().writes_total(tag, saved, edge)
        if tag >= RN_TAG_CONV_FIRST and tag <= RN_TAG_CONV_LAST:
            return Conv2dRules().writes_total(tag, saved, edge)
        return NeuralRules().writes_total(tag, saved, edge)

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag >= RN_TAG_FOLD_FIRST and tag <= RN_TAG_FOLD_LAST:
            RankNFoldRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
        elif tag >= RN_TAG_BMM_FIRST and tag <= RN_TAG_BMM_LAST:
            RankNBmmRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
        elif tag >= RN_TAG_LAYOUT_FIRST and tag <= RN_TAG_LAYOUT_LAST:
            RankNLayoutRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
        elif tag >= RN_TAG_CONV_FIRST and tag <= RN_TAG_CONV_LAST:
            Conv2dRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
        else:
            NeuralRules().apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)


# ---------------------------------------------------------------- shape metadata
def rn_shape(dims: List[Int], what: String) raises -> ShapeN:
    """The ShapeN of `dims` (0 .. RN_MAX_RANK positive dims, row-major)."""
    if len(dims) > RN_MAX_RANK:
        raise Error(what + ": tensors have at most " + String(RN_MAX_RANK) + " dimensions")
    var d = InlineArray[Int, RN_MAX_RANK](fill=1)
    for k in range(len(dims)):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        if dims[k] < 1:
            raise Error(what + ": every dimension must be positive")
        d[k] = dims[k]
    return ShapeN(len(dims), d)


def rn_dims(s: ShapeN) -> List[Int]:
    """The dims of `s` as a list (empty for a scalar of rank 0)."""
    var out = List[Int](capacity=s.ndim)
    for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        out.append(s.d[k])
    return out^


def rn_with_batch(batch: Int, sample: ShapeN, what: String) raises -> ShapeN:
    """(batch,) + sample: a model's per-sample shape with its batch axis."""
    var dims = List[Int](capacity=sample.ndim + 1)
    dims.append(batch)
    for k in range(sample.ndim):  # small-loop(ndim: tensor rank, at most 7): shape metadata, no data
        dims.append(sample.d[k])
    return rn_shape(dims, what)


def rn_resolve(s: ShapeN, dims: List[Int], what: String) raises -> ShapeN:
    """`dims` (positive, at most one -1 inferred) as a shape holding the
    same element count as `s`."""
    var known = 1
    var infer = -1
    for k in range(len(dims)):  # small-loop(ndim: target rank, at most 8): shape metadata, no data
        if dims[k] == -1:
            if infer != -1:
                raise Error(what + ": at most one dimension can be -1")
            infer = k
        elif dims[k] < 1:
            raise Error(what + ": dimensions must be positive or -1")
        else:
            known *= dims[k]
    var total = s.count()
    var out = dims.copy()
    if infer != -1:
        if total % known != 0:
            raise Error(what + ": cannot infer the -1 dimension from " + String(total) + " elements")
        out[infer] = total // known
    elif known != total:
        raise Error(what + ": the new shape holds " + String(known) + " elements, the tensor "
                    + String(total))
    return rn_shape(out, what)


def rn_axis_mask(ndim: Int, axes: List[Int], what: String) raises -> List[Bool]:
    """One flag per axis: True for the axes listed (negative axes count
    from the end); a repeated axis is refused."""
    var mask = List[Bool](length=ndim, fill=False)
    for k in range(len(axes)):  # small-loop(axes: at most 8): shape metadata, no data
        var a = rn_normalize_axis(axes[k], ndim)
        if mask[a]:
            raise Error(what + ": axis " + String(axes[k]) + " is listed twice")
        mask[a] = True
    return mask^


def rn_mask_axes(mask: List[Bool]) -> List[Int]:
    """The ascending axes whose flag is set."""
    var out = List[Int]()
    for k in range(len(mask)):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        if mask[k]:
            out.append(k)
    return out^


def rn_reduced(s: ShapeN, mask: List[Bool], keepdims: Bool, what: String) raises -> ShapeN:
    """`s` with each flagged axis set to 1 (keepdims) or removed."""
    var dims = List[Int]()
    for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        if not mask[k]:
            dims.append(s.d[k])
        elif keepdims:
            dims.append(1)
    return rn_shape(dims, what)


def rn_check_permutation(axes: List[Int], ndim: Int, what: String) raises -> List[Int]:
    """`axes` normalized (negative axes count from the end), refused unless
    it lists every axis of a rank-`ndim` tensor once."""
    if len(axes) != ndim:
        raise Error(what + ": axes must list each of the tensor's " + String(ndim) + " axes once")
    var seen = List[Bool](length=ndim, fill=False)
    var out = List[Int](capacity=ndim)
    for k in range(ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        var a = rn_normalize_axis(axes[k], ndim)
        if seen[a]:
            raise Error(what + ": axes must list each of the tensor's " + String(ndim) + " axes once")
        seen[a] = True
        out.append(a)
    return out^


def rn_inverse(axes: List[Int]) -> List[Int]:
    """The inverse permutation of `axes`."""
    var inv = List[Int](length=len(axes), fill=0)
    for k in range(len(axes)):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        inv[axes[k]] = k
    return inv^


# ---------------------------------------------------------------- views and compositions
def rn_as_shape[E: Exec](mut tape: Tape, mut ex: E, t: Tensor, s: ShapeN) raises -> Tensor:
    """`t` itself when its shape is `s`, else the free view of it with
    shape `s` (the same element count): no launch, no new word."""
    if tape.shape_of(t).same(s):
        return t
    return tape.view_nd(ex, t, s)


def rn_softmax_along[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, axis: Int, log: Bool,
) raises -> Tensor:
    """softmax (or log_softmax) of `x` along `axis`. The last axis is the
    tape's row softmax on the matrix view; any other axis is permute (a
    view) -> the row softmax (the dense reader materialises the permuted
    matrix once) -> permute back (a view): plan 2.7, the 2-D words on the
    permuted matrix."""
    var s = tape.shape_of(x)
    var nd = s.ndim if s.ndim > 0 else 1
    var a = rn_normalize_axis(axis, nd)
    if a == nd - 1:
        var y = tape.log_softmax(ex, x) if log else tape.softmax(ex, x)
        return rn_as_shape(tape, ex, y, s)
    var p = List[Int](capacity=nd)
    for k in range(nd):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        if k != a:
            p.append(k)
    p.append(a)
    var moved = tape.permute(ex, x, p)
    var z = tape.log_softmax(ex, moved) if log else tape.softmax(ex, moved)
    z = rn_as_shape(tape, ex, z, tape.shape_of(moved))
    return tape.permute(ex, z, rn_inverse(p))
