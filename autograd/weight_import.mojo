# SPDX-License-Identifier: Apache-2.0
"""WEIGHT IMPORT (lane ag-weight-import, 2026-10-19): PyTorch state_dict
tensors and Hugging Face safetensors tensors into a Sequential model's flat
parameter vector (`model_import_params`) or into a session's uploaded
tensors (`session_load_weights`).

Python only matches names: it hands, per source tensor, one record of
integers (the target, the slot, a layout code, the tensor's
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

Record (wi_record_words(rank) = WI_RECORD_HEAD + 2 * rank words per source
tensor: the record takes the tensor's own rank, there is no rank cap; the
records of one call are laid end to end and wi_record_starts walks them):
  [0] target: the model's layer index, or the session tensor handle
  [1] slot: WI_SLOT_WEIGHT or WI_SLOT_BIAS (models; 0 for sessions)
  [2] layout: WI_LAYOUT_TORCH (a linear weight stored (out, in), as torch
      and Hugging Face store it), WI_LAYOUT_NATIVE (stored (in, out), as
      params_), WI_LAYOUT_AS_IS or WI_LAYOUT_TRANSPOSE (sessions)
  [3] address of the buffer, [4] byte offset of element 0 in it
  [5] NumPy dtype number (float16 23, float32 11, float64 12) or
      WI_DTYPE_BF16, [6] item size
  [7] rank r (1 or more), [8 .. 8 + r) dims
  [8 + r] stride unit: WI_STRIDE_C (C order, strides unused, still r words),
      WI_STRIDE_BYTES or WI_STRIDE_ITEMS, [9 + r .. 9 + 2r) strides (signed)

Model parameter layout (autograd/model.mojo ModelSpec): a linear layer's
weight (in, out) row-major, then its bias (out); a layer norm's weight,
then its bias (width each); a conv2d layer's weight (out_channels,
in_channels, k, k) row-major (torch's own layout), then its bias."""
from sequence.exec_trait import Exec
from sequence.ops import FP
from core.cast_in import CAST_KIND_BFLOAT, CAST_KIND_FLOAT, CastIn, cast_in_kind
from autograd.cast_ops import upload_cast
from autograd.model_blocks import block_param_sizes, block_tensor_dims, block_tensor_transposed, is_block
# nn-attention begin
from autograd.nn_attention import nn_attention_torch_blocks, nn_attention_is
from autograd.model import SPEC2_VERSION
# nn-attention end
from autograd.model import LAYER_CONV1D, LAYER_CONV2D, LAYER_LAYER_NORM, LAYER_LINEAR, SPEC2_ARGS, ModelSpec, conv_in_per_group

#: the dtype code of a bfloat16 source (NumPy has no number for it)
comptime WI_DTYPE_BF16 = 4096
#: the words of a record besides its dims and strides (fields 0..7 and the stride unit)
comptime WI_RECORD_HEAD = 9
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
    for k in range(len(dims)):  # small-loop(dims: a tensor rank): message text, no data
        if k > 0:
            s += ", "
        s += String(dims[k])
    if len(dims) == 1:
        s += ","
    return s + ")"


def wi_record_words(rank: Int) -> Int:
    """The words of one import record of a tensor of `rank` dimensions."""
    return WI_RECORD_HEAD + 2 * rank


def wi_record_starts(recs: List[Int], n: Int, what: String) raises -> List[Int]:
    """The first word of each of the `n` records laid end to end in `recs`
    (each record is as long as its own rank makes it); every word of `recs`
    belongs to exactly one record."""
    var starts = List[Int](capacity=n)
    var at = 0
    for r in range(n):  # driver-loop: over source tensors (metadata words, no data)
        if at + 8 > len(recs):
            raise Error(what + ": the import records end inside record " + String(r))
        var nd = recs[at + 7]
        if nd < 1:
            raise Error(what + ": record " + String(r) + " has rank " + String(nd) + "; a weight has 1 or more dimensions")
        starts.append(at)
        at += wi_record_words(nd)
    if at != len(recs):
        raise Error(what + ": the import takes " + String(at) + " integers for " + String(n)
                    + " source tensors; " + String(len(recs)) + " were given")
    return starts^


def wi_source(recs: List[Int], base: Int, name: String) raises -> WeightSource:
    """The record that starts at word `base` of `recs` as a checked WeightSource."""
    if base < 0 or base + 8 > len(recs):
        raise Error(_q(name) + ": the import record runs past its words")
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
    if nd < 1:
        raise Error(_q(name) + ": a weight has 1 or more dimensions")
    if base + wi_record_words(nd) > len(recs):
        raise Error(_q(name) + ": the import record runs past its words")
    var dims = List[Int](capacity=nd)
    for k in range(nd):  # small-loop(nd: the tensor rank): record parsing, no data
        var d = recs[base + 8 + k]
        if d < 1:
            raise Error(_q(name) + ": the tensor is empty")
        dims.append(d)
    var unit = recs[base + 8 + nd]
    var strides = List[Int](length=nd, fill=0)
    if unit == WI_STRIDE_C:
        var step = isz
        for j in range(nd):  # small-loop(nd: the tensor rank): record parsing, no data
            var k = nd - 1 - j
            strides[k] = step
            step *= dims[k]
    elif unit == WI_STRIDE_BYTES or unit == WI_STRIDE_ITEMS:
        var mul = 1 if unit == WI_STRIDE_BYTES else isz
        for k in range(nd):  # small-loop(nd: the tensor rank): record parsing, no data
            strides[k] = recs[base + 9 + nd + k] * mul
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
        for k in range(nd - 1):  # small-loop(nd: the tensor rank): shape metadata, no data
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
            for j in range(last):  # small-loop(last: the leading axes of the tensor): shape metadata, no data
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
    if k == LAYER_CONV2D or k == LAYER_CONV1D:
        return spec.xargs[SPEC2_ARGS * i + 4] == 1
    return k == LAYER_LAYER_NORM


def _slots(spec: ModelSpec, i: Int) -> Int:
    """How many parameter tensors layer i has: weight (and bias), or a block layer's tensors."""
    var k = spec.kinds[i]
    if is_block(k):
        return len(block_param_sizes(k, spec.xargs, i, spec.in_shape[i]))
    if not _has_params(spec, i):
        return 0
    return 2 if _has_bias(spec, i) else 1


def _slot_offset(spec: ModelSpec, i: Int, slot: Int) -> Int:
    var off = spec.param_off[i]
    var k = spec.kinds[i]
    if is_block(k):
        var sizes = block_param_sizes(k, spec.xargs, i, spec.in_shape[i])
        for j in range(slot):  # small-loop(slot < 4: the block's earlier tensors)
            off += sizes[j]
        return off
    if slot == WI_SLOT_BIAS:
        off += _weight_count(spec, i)
    return off


def _has_params(spec: ModelSpec, i: Int) -> Bool:
    var k = spec.kinds[i]
    if is_block(k):
        return len(block_param_sizes(k, spec.xargs, i, spec.in_shape[i])) > 0
    return k == LAYER_LINEAR or k == LAYER_LAYER_NORM or k == LAYER_CONV2D or k == LAYER_CONV1D


def _layer_kind(spec: ModelSpec, i: Int) -> String:
    var k = spec.kinds[i]
    if k == LAYER_LINEAR:
        return "linear"
    if k == LAYER_LAYER_NORM:
        return "layer_norm"
    if k == LAYER_CONV1D:
        return "conv1d"
    return "conv2d"


def _expected(spec: ModelSpec, i: Int, slot: Int, layout: Int) -> List[Int]:
    """The source shape layer i's slot takes in `layout`."""
    var out = List[Int]()
    var k = spec.kinds[i]
    if is_block(k):
        return block_tensor_dims(k, spec.xargs, i, spec.in_shape[i], slot, layout == WI_LAYOUT_TORCH)
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
            if k == LAYER_CONV2D:
                out.append(ks)
    return out^


def _weight_count(spec: ModelSpec, i: Int) -> Int:
    var k = spec.kinds[i]
    if k == LAYER_LINEAR:
        return spec.in_width[i] * spec.arg0[i]
    if k == LAYER_LAYER_NORM:
        return spec.in_width[i]
    var ks = spec.arg1[i]
    if k == LAYER_CONV1D:
        return spec.arg0[i] * conv_in_per_group(spec, i) * ks
    return spec.arg0[i] * conv_in_per_group(spec, i) * ks * ks


def _same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for k in range(len(a)):  # small-loop(a: a tensor rank): shape metadata, no data
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
    if n < 1:
        raise Error("load_state_dict: the import takes at least one source tensor")
    var starts = wi_record_starts(recs, n, "load_state_dict")
    var nl = spec.n_layers()
    var seen = List[Bool](length=4 * nl, fill=False)
    var views = List[CastIn]()
    var dsts = List[Int]()
    for r in range(n):  # driver-loop: over source tensors
        var base = starts[r]
        var i = recs[base]
        var slot = recs[base + 1]
        var layout = recs[base + 2]
        var name = names[r]
        if i < 0 or i >= nl or not _has_params(spec, i):
            raise Error("load_state_dict: " + _q(name) + " names layer " + String(i)
                        + ", which has no parameters")
        if slot < 0 or slot >= _slots(spec, i):
            raise Error("load_state_dict: " + _q(name) + ": unknown parameter slot " + String(slot))
        if slot == WI_SLOT_BIAS and not is_block(spec.kinds[i]) and not _has_bias(spec, i):
            raise Error("load_state_dict: " + _q(name) + ": layer " + String(i) + " (" + _layer_kind(spec, i)
                        + ") has no bias")
        if layout != WI_LAYOUT_TORCH and layout != WI_LAYOUT_NATIVE:
            raise Error("load_state_dict: " + _q(name) + ": unknown layout " + String(layout))
        if seen[4 * i + slot]:
            raise Error("load_state_dict: " + _q(name) + ": layer " + String(i)
                        + "'s " + _slot_name(slot) + " is given twice")
        seen[4 * i + slot] = True
        var src = wi_source(recs, base, name)
        var want = _expected(spec, i, slot, layout)
        if not _same(src.dims, want):
            raise Error("load_state_dict: " + _q(name) + " has shape " + wi_shape_str(src.dims) + "; layer "
                        + String(i) + " (" + _layer_kind(spec, i) + ") takes a "
                        + _slot_name(slot) + " of shape " + wi_shape_str(want))
        # torch stores a linear weight (out, in): its transpose is params_' (in, out)
        var transpose = spec.kinds[i] == LAYER_LINEAR and slot == WI_SLOT_WEIGHT and layout == WI_LAYOUT_TORCH
        if is_block(spec.kinds[i]):
            transpose = layout == WI_LAYOUT_TORCH and block_tensor_transposed(spec.kinds[i], slot)
        views.append(wi_matrix(src, transpose))
        dsts.append(_slot_offset(spec, i, slot))
        # nn-attention begin: PyTorch's stacked in_proj_weight (3E, E) onto params_' (in, out) blocks
        if layout == WI_LAYOUT_TORCH and spec.version == SPEC2_VERSION and nn_attention_is(spec.kinds[i]):
            var nb = nn_attention_torch_blocks(spec.kinds[i], spec.xargs, i, spec.in_shape[i], slot)
            if len(nb) > 0:
                _ = views.pop()
                _ = dsts.pop()
                for q in range(len(nb) // 3):  # small-loop(at most 2 blocks of one tensor)
                    var sd: List[Int] = [nb[3 * q + 1], src.dims[1]]
                    var ss: List[Int] = [src.strides[0], src.strides[1]]
                    var sub = WeightSource(src.name, src.addr + nb[3 * q] * src.strides[0], src.kind, src.itemsize,
                                           sd^, ss^)
                    views.append(wi_matrix(sub, True))
                    dsts.append(_slot_offset(spec, i, slot) + nb[3 * q + 2])
        # nn-attention end
    for i in range(nl):  # driver-loop: over layers
        if not _has_params(spec, i):
            continue
        if is_block(spec.kinds[i]):
            for j in range(_slots(spec, i)):  # small-loop(at most four tensors of a block layer)
                if not seen[4 * i + j]:
                    raise Error("load_state_dict: no source tensor for layer " + String(i) + "'s tensor "
                                + String(j))
            continue
        if not seen[4 * i]:
            raise Error("load_state_dict: no source tensor for layer " + String(i) + "'s ("
                        + _layer_kind(spec, i) + ") weight")
        if _has_bias(spec, i) and not seen[4 * i + 1]:
            raise Error("load_state_dict: no source tensor for layer " + String(i) + "'s ("
                        + _layer_kind(spec, i) + ") bias")
    var D = ex.alloc(spec.n_params)
    for r in range(len(views)):  # driver-loop: over source tensors
        upload_cast(ex, D + dsts[r], views[r])
    ex.download(P, D, spec.n_params)
    ex.free(D)


def _squeezed(dims: List[Int]) -> List[Int]:
    var out = List[Int]()
    for k in range(len(dims)):  # small-loop(dims: a tensor rank): shape metadata, no data
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


# WEIGHT EXPORT (lane api-torch round 2, 2026-11-21): the inverse of
# `model_import_params`, for Sequential.state_dict / parameters /
# named_parameters. Each parameter tensor of the model leaves in PyTorch's
# layout (`nn.Sequential` names; a linear weight (out, in)), at the same
# offset it has in params_, so one float32 buffer of spec.n_params holds them
# all. The copy is the import's own machinery reversed: one upload of
# params_, then per linear weight one upload_cast of its (in, out) block
# viewed as its (out, in) transpose (the two strides swapped), then one
# download. float32 to float32: the words are moved, never rounded, so the
# exported values are params_' bits on every vendor and the host column.
# No new kernel and no new opcode.
#
# Shape record (WE_SHAPE_HEAD + rank words per tensor, the tensor's own rank,
# no rank cap; laid end to end in state_dict order: layers in order, the
# weight then the bias):
#   [0] layer index, [1] slot (WI_SLOT_WEIGHT / WI_SLOT_BIAS), [2] offset in
#   the buffer (floats), [3] rank r, [4 .. 4 + r) dims
comptime WE_SHAPE_HEAD = 4


def model_export_shape_words(spec: ModelSpec) -> Int:
    """The words `model_export_params` writes into its shape result."""
    var words = 0
    for i in range(spec.n_layers()):  # driver-loop: over layers
        if not _has_params(spec, i):
            continue
        for slot in range(_slots(spec, i)):  # small-loop(at most 4 tensors of a layer): metadata, no data
            words += WE_SHAPE_HEAD + len(_expected(spec, i, slot, WI_LAYOUT_TORCH))
    return words


def model_export_tensors(spec: ModelSpec) -> Int:
    """How many parameter tensors `model_export_params` writes."""
    var count = 0
    for i in range(spec.n_layers()):  # driver-loop: over layers
        count += _slots(spec, i)
    return count


def model_export_params[E: Exec](
    mut ex: E, spec: ModelSpec, params: Int, dst: Int, shapes: Int, capacity: Int,
) raises -> Int:
    """dst (spec.n_params host floats) = the model's parameters at `params`
    in PyTorch's layouts; `shapes` (capacity words) = one shape record per
    tensor (WE_SHAPE_HEAD + its rank words). Returns the tensor count, or
    -(the words needed) with nothing written when `capacity` is too small
    (the caller asks again with that many)."""
    var n = model_export_tensors(spec)
    if params == 0 or dst == 0 or shapes == 0:
        raise Error("state_dict: null buffer address")
    var needed = model_export_shape_words(spec)
    if needed > capacity:
        return -needed
    var rec = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=shapes)
    var D = ex.alloc(spec.n_params)
    ex.upload(D, FP(unsafe_from_address=params), spec.n_params)
    var at = 0
    for i in range(spec.n_layers()):  # driver-loop: over layers
        if not _has_params(spec, i):
            continue
        for slot in range(_slots(spec, i)):  # small-loop(at most 4 tensors of a layer): metadata, no data
            var off = _slot_offset(spec, i, slot)
            if is_block(spec.kinds[i]) and block_tensor_transposed(spec.kinds[i], slot):
                # params_ holds the projection (in, out); torch's (out, in) is its transpose
                var bd = block_tensor_dims(spec.kinds[i], spec.xargs, i, spec.in_shape[i], slot, False)
                var bview = CastIn(params + 4 * off, CAST_KIND_FLOAT, 4, bd[1], bd[0], 4, 4 * bd[1])
                bview.check("state_dict: layer " + String(i) + " tensor " + String(slot))
                upload_cast(ex, D + off, bview)
            if spec.kinds[i] == LAYER_LINEAR and slot == WI_SLOT_WEIGHT:
                # params_ holds (in, out) row-major: element (o, k) of the
                # (out, in) view is at k * out + o.
                var fi = spec.in_width[i]
                var fo = spec.arg0[i]
                var view = CastIn(params + 4 * off, CAST_KIND_FLOAT, 4, fo, fi, 4, 4 * fo)
                view.check("state_dict: layer " + String(i) + " (linear) weight")
                upload_cast(ex, D + off, view)
            # nn-attention begin: params_' (in, out) blocks of in_proj -> PyTorch's stacked (3E, E) rows
            var nb = List[Int]()
            if spec.version == SPEC2_VERSION and nn_attention_is(spec.kinds[i]):
                nb = nn_attention_torch_blocks(spec.kinds[i], spec.xargs, i, spec.in_shape[i], slot)
            for q in range(len(nb) // 3):  # small-loop(at most 2 blocks of one tensor)
                var ne = spec.in_shape[i].d[1]
                var nr = nb[3 * q + 1]
                var nview = CastIn(params + 4 * (off + nb[3 * q + 2]), CAST_KIND_FLOAT, 4, nr, ne, 4, 4 * nr)
                nview.check("state_dict: layer " + String(i) + " in_proj_weight")
                upload_cast(ex, D + off + nb[3 * q] * ne, nview)
            # nn-attention end
            var dims = _expected(spec, i, slot, WI_LAYOUT_TORCH)
            rec[at] = i
            rec[at + 1] = slot
            rec[at + 2] = off
            rec[at + 3] = len(dims)
            for k in range(len(dims)):  # small-loop(dims: the tensor rank): metadata, no data
                rec[at + WE_SHAPE_HEAD + k] = dims[k]
            at += WE_SHAPE_HEAD + len(dims)
    ex.download(FP(unsafe_from_address=dst), D, spec.n_params)
    ex.free(D)
    return n
