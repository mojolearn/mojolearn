# SPDX-License-Identifier: Apache-2.0
"""WEIGHT IMPORT (lane ag-weight-import, 2026-10-19): PyTorch state_dict
tensors and Hugging Face safetensors tensors into a Sequential model's flat
parameter vector (`model_import_params`) or into a session's uploaded
tensors (`session_load_weights`).

Python only matches names: it hands, per source tensor, one record of
WI_RECORD integers (the target, the slot, a layout code, the tensor's
address and byte offset, its dtype, item size, rank, dims and strides) and
the tensor's name for the refusals. Everything else is here: the dtype
check, the shape check against the model description, and the copy. The
copy is core/cast_in.mojo's CastIn view of the source (a transpose is the
view with its two strides swapped) through autograd/cast_ops.mojo
upload_cast: one upload of the tensor's bytes and one OP_CAST_F32 launch
on the executor, which widens float16 (exact bit expansion), bfloat16 (the
exact 16-bit shift) and float64 (round to nearest even) to float32 and
writes the destination row-major. Integer statements only, so the imported
words are the same on Apple, NVIDIA, AMD and the host column. No new
kernel and no new opcode.

Record (WI_RECORD words per source tensor):
  [0] target: the model's layer index, or the session tensor handle
  [1] slot: WI_SLOT_WEIGHT or WI_SLOT_BIAS (models; 0 for sessions)
  [2] layout: WI_LAYOUT_TORCH (a linear weight stored (out, in), as torch
      and Hugging Face store it), WI_LAYOUT_NATIVE (stored (in, out), as
      params_), WI_LAYOUT_AS_IS or WI_LAYOUT_TRANSPOSE (sessions)
  [3] address of the buffer, [4] byte offset of element 0 in it
  [5] NumPy dtype number (float16 23, float32 11, float64 12) or
      WI_DTYPE_BF16, [6] item size
  [7] rank (1..WI_MAX_RANK), [8..11] dims (0 past the rank)
  [12] stride unit: WI_STRIDE_C (C order, strides unused), WI_STRIDE_BYTES
      or WI_STRIDE_ITEMS, [13..16] strides (signed)

Model parameter layout (autograd/model.mojo ModelSpec): a linear layer's
weight (in, out) row-major, then its bias (out); a layer norm's weight,
then its bias (width each); a conv2d layer's weight (out_channels,
in_channels, k, k) row-major (torch's own layout), then its bias."""
from sequence.exec_trait import Exec
from sequence.ops import FP
from core.cast_in import CAST_KIND_BFLOAT, CAST_KIND_FLOAT, CastIn, cast_in_kind
from autograd.cast_ops import upload_cast
from autograd.model import LAYER_CONV2D, LAYER_LAYER_NORM, LAYER_LINEAR, SPEC2_ARGS, ModelSpec, conv_in_per_group

#: the dtype code of a bfloat16 source (NumPy has no number for it)
comptime WI_DTYPE_BF16 = 4096
comptime WI_RECORD = 17
comptime WI_MAX_RANK = 4
comptime WI_SLOT_WEIGHT = 0
comptime WI_SLOT_BIAS = 1
comptime WI_LAYOUT_TORCH = 0
comptime WI_LAYOUT_NATIVE = 1
comptime WI_LAYOUT_AS_IS = 2
comptime WI_LAYOUT_TRANSPOSE = 3
comptime WI_STRIDE_C = 0
comptime WI_STRIDE_BYTES = 1
comptime WI_STRIDE_ITEMS = 2


struct WeightSource(Copyable, Movable):
    """One source tensor: element 0's address, element kind and size, dims
    and byte strides."""
    var name: String
    var addr: Int
    var kind: Int
    var itemsize: Int
    var dims: List[Int]
    var strides: List[Int]

    def __init__(
        out self, var name: String, addr: Int, kind: Int, itemsize: Int, var dims: List[Int], var strides: List[Int],
    ):
        self.name = name^
        self.addr = addr
        self.kind = kind
        self.itemsize = itemsize
        self.dims = dims^
        self.strides = strides^


def _slot_name(slot: Int) -> String:
    if slot == WI_SLOT_BIAS:
        return String("bias")
    return String("weight")


def _q(name: String) -> String:
    return "'" + name + "'"


def wi_shape_str(dims: List[Int]) -> String:
    var s = String("(")
    for k in range(len(dims)):  # small-loop(dims: a tensor rank, at most 8): message text, no data
        if k > 0:
            s += ", "
        s += String(dims[k])
    if len(dims) == 1:
        s += ","
    return s + ")"


def wi_source(recs: List[Int], at: Int, name: String) raises -> WeightSource:
    """Record `at` of `recs` as a checked WeightSource."""
    var base = at * WI_RECORD
    var addr = recs[base + 3]
    var off = recs[base + 4]
    if addr == 0:
        raise Error(_q(name) + ": null buffer address")
    if off < 0:
        raise Error(_q(name) + ": negative byte offset")
    var dt = recs[base + 5]
    var isz = recs[base + 6]
    var kind: Int
    if dt == WI_DTYPE_BF16:
        if isz != 2:
            raise Error(_q(name) + ": a bfloat16 tensor has 2-byte items")
        kind = CAST_KIND_BFLOAT
    else:
        kind = cast_in_kind(dt, isz, _q(name))
        if kind != CAST_KIND_FLOAT:
            raise Error(_q(name) + ": a weight must be float16, bfloat16, float32 or float64")
    var nd = recs[base + 7]
    if nd < 1 or nd > WI_MAX_RANK:
        raise Error(_q(name) + ": a weight has 1 to " + String(WI_MAX_RANK) + " dimensions")
    var dims = List[Int](capacity=nd)
    for k in range(nd):  # small-loop(nd: the tensor rank, at most 4): record parsing, no data
        var d = recs[base + 8 + k]
        if d < 1:
            raise Error(_q(name) + ": the tensor is empty")
        dims.append(d)
    var unit = recs[base + 12]
    var strides = List[Int](length=nd, fill=0)
    if unit == WI_STRIDE_C:
        var step = isz
        for j in range(nd):  # small-loop(nd: the tensor rank, at most 4): record parsing, no data
            var k = nd - 1 - j
            strides[k] = step
            step *= dims[k]
    elif unit == WI_STRIDE_BYTES or unit == WI_STRIDE_ITEMS:
        var mul = 1 if unit == WI_STRIDE_BYTES else isz
        for k in range(nd):  # small-loop(nd: the tensor rank, at most 4): record parsing, no data
            strides[k] = recs[base + 13 + k] * mul
    else:
        raise Error(_q(name) + ": unknown stride unit " + String(unit))
    return WeightSource(name.copy(), addr + off, kind, isz, dims^, strides^)


def wi_matrix(src: WeightSource, transpose: Bool) raises -> CastIn:
    """The source as the CastIn of its row-major (rows, cols) matrix: the
    last axis is the columns and the leading axes collapse into the rows
    (they must be laid out as one: C order, any strides on the last two
    axes); a rank-1 tensor is one column. `transpose` (rank 2 only) views
    the (a, b) source as its (b, a) transpose: the two strides swap."""
    var nd = len(src.dims)
    var rows: Int
    var cols: Int
    var s0: Int
    var s1: Int
    if nd == 1:
        rows = src.dims[0]
        cols = 1
        s0 = src.strides[0]
        s1 = src.itemsize
    else:
        cols = src.dims[nd - 1]
        s1 = src.strides[nd - 1]
        # the leading axes of more than one element (a size-1 axis has no step)
        var ld = List[Int]()
        var ls = List[Int]()
        for k in range(nd - 1):  # small-loop(nd: the tensor rank, at most 4): shape metadata, no data
            if src.dims[k] != 1:
                ld.append(src.dims[k])
                ls.append(src.strides[k])
        if len(ld) == 0:
            rows = 1
            s0 = cols * src.itemsize
        else:
            var last = len(ld) - 1
            rows = ld[last]
            s0 = ls[last]
            for j in range(last):  # small-loop(last: the leading axes, at most 3): shape metadata, no data
                var k = last - 1 - j
                if ls[k] != ld[k + 1] * ls[k + 1]:
                    raise Error(_q(src.name) + ": the tensor's leading axes are not laid out in C order; "
                                + "pass a contiguous tensor")
                rows *= ld[k]
    var c: CastIn
    if transpose:
        if nd != 2:
            raise Error(_q(src.name) + ": a transposed load takes a 2-D tensor")
        c = CastIn(src.addr, src.kind, src.itemsize, cols, rows, s1, s0)
    else:
        c = CastIn(src.addr, src.kind, src.itemsize, rows, cols, s0, s1)
    c.check(_q(src.name))
    return c


def _has_bias(spec: ModelSpec, i: Int) -> Bool:
    var k = spec.kinds[i]
    if k == LAYER_LINEAR:
        return spec.arg1[i] == 1
    if k == LAYER_CONV2D:
        return spec.xargs[SPEC2_ARGS * i + 4] == 1
    return k == LAYER_LAYER_NORM


def _has_params(spec: ModelSpec, i: Int) -> Bool:
    var k = spec.kinds[i]
    return k == LAYER_LINEAR or k == LAYER_LAYER_NORM or k == LAYER_CONV2D


def _layer_kind(spec: ModelSpec, i: Int) -> String:
    var k = spec.kinds[i]
    if k == LAYER_LINEAR:
        return "linear"
    if k == LAYER_LAYER_NORM:
        return "layer_norm"
    return "conv2d"


def _expected(spec: ModelSpec, i: Int, slot: Int, layout: Int) -> List[Int]:
    """The source shape layer i's slot takes in `layout`."""
    var out = List[Int]()
    var k = spec.kinds[i]
    if k == LAYER_LINEAR:
        var fi = spec.in_width[i]
        var fo = spec.arg0[i]
        if slot == WI_SLOT_BIAS:
            out.append(fo)
        elif layout == WI_LAYOUT_NATIVE:
            out.append(fi)
            out.append(fo)
        else:
            out.append(fo)
            out.append(fi)
    elif k == LAYER_LAYER_NORM:
        out.append(spec.in_width[i])
    else:
        var oc = spec.arg0[i]
        out.append(oc)
        if slot == WI_SLOT_WEIGHT:
            var ks = spec.arg1[i]
            out.append(conv_in_per_group(spec, i))
            out.append(ks)
            out.append(ks)
    return out^


def _weight_count(spec: ModelSpec, i: Int) -> Int:
    var k = spec.kinds[i]
    if k == LAYER_LINEAR:
        return spec.in_width[i] * spec.arg0[i]
    if k == LAYER_LAYER_NORM:
        return spec.in_width[i]
    var ks = spec.arg1[i]
    return spec.arg0[i] * conv_in_per_group(spec, i) * ks * ks


def _same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for k in range(len(a)):  # small-loop(a: a tensor rank, at most 8): shape metadata, no data
        if a[k] != b[k]:
            return False
    return True


def model_import_params[E: Exec](
    mut ex: E, spec: ModelSpec, recs: List[Int], names: List[String], P: FP,
) raises:
    """P (spec.n_params host floats) = the model's parameters from the
    source tensors: every record is checked (layer, slot, dtype, shape,
    layout) and every parameter of the model must be covered exactly once
    before anything is written; then one upload_cast per tensor into one
    executor buffer and one download into P."""
    var n = len(names)
    if n < 1 or len(recs) != n * WI_RECORD:
        raise Error("load_state_dict: the import takes " + String(WI_RECORD) + " integers per source tensor")
    var nl = spec.n_layers()
    var seen = List[Bool](length=2 * nl, fill=False)
    var views = List[CastIn]()
    var dsts = List[Int]()
    for r in range(n):  # driver-loop: over source tensors
        var base = r * WI_RECORD
        var i = recs[base]
        var slot = recs[base + 1]
        var layout = recs[base + 2]
        var name = names[r]
        if i < 0 or i >= nl or not _has_params(spec, i):
            raise Error("load_state_dict: " + _q(name) + " names layer " + String(i)
                        + ", which has no parameters")
        if slot != WI_SLOT_WEIGHT and slot != WI_SLOT_BIAS:
            raise Error("load_state_dict: " + _q(name) + ": unknown parameter slot " + String(slot))
        if slot == WI_SLOT_BIAS and not _has_bias(spec, i):
            raise Error("load_state_dict: " + _q(name) + ": layer " + String(i) + " (" + _layer_kind(spec, i)
                        + ") has no bias")
        if layout != WI_LAYOUT_TORCH and layout != WI_LAYOUT_NATIVE:
            raise Error("load_state_dict: " + _q(name) + ": unknown layout " + String(layout))
        if seen[2 * i + slot]:
            raise Error("load_state_dict: " + _q(name) + ": layer " + String(i)
                        + "'s " + _slot_name(slot) + " is given twice")
        seen[2 * i + slot] = True
        var src = wi_source(recs, r, name)
        var want = _expected(spec, i, slot, layout)
        if not _same(src.dims, want):
            raise Error("load_state_dict: " + _q(name) + " has shape " + wi_shape_str(src.dims) + "; layer "
                        + String(i) + " (" + _layer_kind(spec, i) + ") takes a "
                        + _slot_name(slot) + " of shape " + wi_shape_str(want))
        # torch stores a linear weight (out, in): its transpose is params_' (in, out)
        var transpose = spec.kinds[i] == LAYER_LINEAR and slot == WI_SLOT_WEIGHT and layout == WI_LAYOUT_TORCH
        views.append(wi_matrix(src, transpose))
        var off = spec.param_off[i]
        if slot == WI_SLOT_BIAS:
            off += _weight_count(spec, i)
        dsts.append(off)
    for i in range(nl):  # driver-loop: over layers
        if not _has_params(spec, i):
            continue
        if not seen[2 * i]:
            raise Error("load_state_dict: no source tensor for layer " + String(i) + "'s ("
                        + _layer_kind(spec, i) + ") weight")
        if _has_bias(spec, i) and not seen[2 * i + 1]:
            raise Error("load_state_dict: no source tensor for layer " + String(i) + "'s ("
                        + _layer_kind(spec, i) + ") bias")
    var D = ex.alloc(spec.n_params)
    for r in range(len(views)):  # driver-loop: over source tensors
        upload_cast(ex, D + dsts[r], views[r])
    ex.download(P, D, spec.n_params)
    ex.free(D)


def _squeezed(dims: List[Int]) -> List[Int]:
    var out = List[Int]()
    for k in range(len(dims)):  # small-loop(dims: a tensor rank, at most 8): shape metadata, no data
        if dims[k] != 1:
            out.append(dims[k])
    return out^


def session_load_check(src: WeightSource, layout: Int, shape: List[Int]) raises -> CastIn:
    """The CastIn that writes `src` into a session tensor of `shape`:
    WI_LAYOUT_AS_IS takes the same dims (size-1 axes aside: a (n,) bias
    fills a (1, n) tensor), WI_LAYOUT_TRANSPOSE an (a, b) source into a
    (b, a) tensor (a torch linear weight into a session weight)."""
    if layout == WI_LAYOUT_TRANSPOSE:
        if len(src.dims) != 2 or len(shape) != 2 or src.dims[0] != shape[1] or src.dims[1] != shape[0]:
            raise Error("load_weights: " + _q(src.name) + " has shape " + wi_shape_str(src.dims)
                        + "; transposed it does not fill a tensor of shape " + wi_shape_str(shape))
        return wi_matrix(src, True)
    if layout != WI_LAYOUT_AS_IS:
        raise Error("load_weights: " + _q(src.name) + ": unknown layout " + String(layout))
    if not _same(_squeezed(src.dims), _squeezed(shape)):
        raise Error("load_weights: " + _q(src.name) + " has shape " + wi_shape_str(src.dims)
                    + "; the tensor has shape " + wi_shape_str(shape))
    return wi_matrix(src, False)
