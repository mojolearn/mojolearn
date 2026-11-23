# SPDX-License-Identifier: Apache-2.0
"""Classic attention layers of a Sequential description (lane nn-attention, 2026-11-23).

  code  layer                 arguments                                 parameters (params_ order)
  67    multihead_attention   a0 num_heads, a1 flags (1 bias, 2 causal, in_proj_weight, out_proj.weight,
                              4 key_padding_mask), a2 key/value source   in_proj_bias, out_proj.bias
                              (-2 self, -1 the model input, else an     (PyTorch's names; its layouts in
                              earlier layer), a3 embed_dim, a4 kdim      state_dict, see below)
                              (0: embed_dim; the value width too);
                              f dropout of the attention weights
  72    positional_encoding   a0 max_len, a1 d_model; f dropout         none

PyTorch's ``nn.MultiheadAttention`` (batch_first=True): q = x Wq^T + bq, k = m Wk^T + bk, v = m Wv^T + bv
(m the key/value source, x itself for self-attention), per head softmax(q k^T / sqrt(head_dim) + mask) v,
then the output projection. ``positional_encoding`` is the sinusoidal encoding of PyTorch's transformer
tutorial: x + PE[:length], PE[p, 2j] = sin(p / 10000^(2j / d)), PE[p, 2j + 1] = cos(...), then dropout.

params_ holds the projections in the tape's Linear layout (in, out), so each projection is ONE tape.linear
node with its bias in the GEMM's epilogue: self-attention's in_proj is (E, 3E), the columns [q | k | v]
(its transpose is PyTorch's stacked in_proj_weight (3E, E)), one GEMM for q, k and v; cross-attention's is
Wq (E, E) then [Wk | Wv] (kdim, 2E), one GEMM for the queries and one for the keys and values. state_dict
and load_state_dict give PyTorch's (3E, E) (autograd/weight_import.mojo, nn-attention groups); with kdim
other than E it is one vector of params_' words (PyTorch keeps three tensors then).

The heads are split by strided views of the projection output (nn_view: no copy; the batched matmul reads
them in place through their strides; the gradient lands in the projection's gradient through one slab
copy each) and joined by one copy (the context's permuted view read densely by the output GEMM).

One body per operation: the GEMMs and their epilogues are tape.linear's, the two attention products the
rank-N batched matmul's, the softmax and its VJP the tape softmax's bodies called inside one row kernel that
also applies the scale and the masks (autograd/nn_attention_ops.mojo), the dropout the tape's. No mask
tensor and no positional table exist: the causal and key-padding masks are index comparisons inside the
softmax row (the key padding reads the batch's lengths vector), and the encoding's sinusoids are computed
where they are added, with the rotary table's expressions.

Launches, one layer, one training step (forward / backward; dropout p > 0 adds 1 / 1):
                                       before (composed ops)        now
  self-attention, no mask              13 / about 26                6 / 12
  self-attention, causal or padding    17 / about 30                6 / 12
  self-attention, causal and padding   20 / about 33                6 / 12
  cross-attention (any mask)           13 to 20 / 26 to 33          7 / 14
  now forward: in-projection GEMM (cross: 2), scores bmm, masked softmax, context bmm, head-merge copy,
  output GEMM. Backward: the in-projection's dW and dX GEMMs with db from the dW launch (cross: 2 each),
  3 view slab copies into the projection's gradient (cross: 3), 2 + 2 bmm, the softmax VJP, the merge
  copy's VJP, the output GEMM's 2.
  positional_encoding: before 4 / 1 (upload, table, interleave, add), now 1 / 1 (the add with the encoding
  computed in place; its VJP copies the upstream).

The encoder and decoder layers (PyTorch's TransformerEncoderLayer / TransformerDecoderLayer) and stacks are
containers written into the description by the Sequential door (bindings/sequential_description_pack.mojo,
lane nn-attention group): multihead_attention, dropout, residual, layer_norm and linear layers.
"""
from std.collections import Array
from std.sys.compile import is_defined
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY
from checks.numerics import ftz, identical_div, identical_sqrt
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_api import rn_shape
from autograd.rankn_bmm import rankn_matmul
from autograd.rankn_layout import _put_record, _dense_strides, _slab_copy
from autograd.nn_attention_ops import (
    OP_NN_ATTN_SOFTMAX, OP_NN_ATTN_SOFTMAX_VJP, OP_NN_PE_ADD, NN_OP_CAUSAL, NN_OP_KEY_PADDING,
)

comptime LAYER_NN_MHA = 67
comptime LAYER_NN_POS_ENC = 72
comptime NN_ATTN_FIRST = 67
comptime NN_ATTN_LAST = 72

#: multihead_attention a1 flags
comptime NN_MHA_BIAS = 1
comptime NN_MHA_CAUSAL = 2
comptime NN_MHA_KEY_PADDING = 4
#: a2: the key/value source of self-attention, and a decoder-stack template's memory placeholder
comptime NN_MHA_SELF = -2
comptime NN_MHA_MEMORY_UNSET = -3

#: the largest length the identical cos / sin domain covers (graph_rotary's bound: position * frequency <
#: 8192, the frequency of pair 0 is 1)
comptime NN_PE_MAX_LEN = 8192
#: backward tags: the masked softmax, the positional encoding, and the head view (a rank-N tag of the layout
#: family's free range 162..175, so the tape keeps it as a pending strided view)
comptime NN_TAG_SOFTMAX = 6710
comptime NN_TAG_PE = 6711
comptime NN_TAG_VIEW = 174

# Fault injection (verify sabotage arms, test builds only): the scores without 1 / sqrt(head_dim); and
# (autograd/nn_attention_ops.mojo) the encoding with sin and cos swapped. Both must make
# checks/native_tests/nn_attention.mojo FAIL.
comptime NN_SABOTAGE_SCALE = is_defined["MOJOLEARN_NN_ATTENTION_SABOTAGE_SCALE"]()

comptime _ARGS = 6


def nn_attention_is(code: Int) -> Bool:
    return code == LAYER_NN_MHA or code == LAYER_NN_POS_ENC


def _x(xargs: List[Int], i: Int, k: Int) -> Int:
    return xargs[_ARGS * i + k]


def _sh(dims: List[Int]) raises -> ShapeN:
    return rn_shape(dims, "model")


def nn_mha_kdim(xargs: List[Int], i: Int) -> Int:
    var kd = _x(xargs, i, 4)
    return _x(xargs, i, 3) if kd == 0 else kd


def nn_attention_source(code: Int, xargs: List[Int], i: Int) -> Int:
    """The earlier output layer i reads besides its input (-1 the model input), or NN_MHA_SELF (none)."""
    if code == LAYER_NN_MHA:
        return _x(xargs, i, 2)
    return NN_MHA_SELF


def nn_attention_reads(code: Int, xargs: List[Int], i: Int, j: Int) -> Bool:
    return code == LAYER_NN_MHA and _x(xargs, i, 2) == j


# ---------------------------------------------------------------- description
def nn_attention_shape(code: Int, xargs: List[Int], i: Int, f: Float32, cur: ShapeN, name: String) raises -> ShapeN:
    """Validate layer i on its per-sample input shape `cur` (the key/value source is checked by
    nn_attention_check_source); its output shape, `cur`."""
    if not (f >= Float32(0) and f < Float32(1)):
        raise Error(name + ": dropout must be in [0, 1)")
    if cur.ndim != 2:
        raise Error(name + ": the input must be (length, features) samples")
    var d = cur.d[1]
    if code == LAYER_NN_MHA:
        var h = _x(xargs, i, 0)
        var flags = _x(xargs, i, 1)
        var e = _x(xargs, i, 3)
        if e != d:
            raise Error(name + " (MultiheadAttention): embed_dim is " + String(e) + " and the input has "
                        + String(d) + " features")
        if h < 1 or d % h != 0:
            raise Error(name + " (MultiheadAttention): num_heads must divide embed_dim")
        if flags < 0 or flags > 7:
            raise Error(name + " (MultiheadAttention): unknown flags " + String(flags))
        if _x(xargs, i, 4) < 0:
            raise Error(name + " (MultiheadAttention): kdim must be positive")
        return cur
    if code == LAYER_NN_POS_ENC:
        var ml = _x(xargs, i, 0)
        var dm = _x(xargs, i, 1)
        if dm != 0 and dm != d:
            raise Error(name + " (PositionalEncoding): d_model is " + String(dm) + " and the input has "
                        + String(d) + " features")
        if d % 2 != 0:
            raise Error(name + " (PositionalEncoding): d_model must be even")
        if ml < 1 or ml > NN_PE_MAX_LEN:
            raise Error(name + " (PositionalEncoding): max_len must be from 1 to " + String(NN_PE_MAX_LEN))
        if cur.d[0] > ml:
            raise Error(name + " (PositionalEncoding): the length " + String(cur.d[0]) + " is longer than max_len "
                        + String(ml))
        return cur
    raise Error(name + ": unknown layer code " + String(code))


def nn_attention_check_source(
    code: Int, xargs: List[Int], i: Int, index: Int, cur: ShapeN, sample: ShapeN, src_shape: ShapeN, name: String,
) raises:
    """The key/value source of multihead_attention record i of `xargs`, the model's layer `index` (the source's
    shape `src_shape`; `cur` for self-attention):
    an earlier layer or the model input, of kdim features; a causal mask needs as many keys as queries; a key
    padding mask needs keys of the model input's length (the lengths are the input's)."""
    if code != LAYER_NN_MHA:
        return
    var src = _x(xargs, i, 2)
    if src == NN_MHA_MEMORY_UNSET:
        raise Error(name + " (MultiheadAttention): a transformer_decoder_layer's memory is not set; pass memory=")
    if src < NN_MHA_SELF or src >= index:
        raise Error(name + " (MultiheadAttention): the key/value source must be -1 (the model input) or the "
                    + "index of an earlier layer")
    if cur.ndim == 2 and _x(xargs, i, 3) != cur.d[1]:
        raise Error(name + " (MultiheadAttention): embed_dim is " + String(_x(xargs, i, 3)) + " and the input has "
                    + String(cur.d[1]) + " features")
    var kd = nn_mha_kdim(xargs, i)
    if src == NN_MHA_SELF and kd != cur.d[1]:
        raise Error(name + " (MultiheadAttention): self-attention takes kdim = embed_dim")
    if src_shape.ndim != 2:
        raise Error(name + " (MultiheadAttention): the key/value source must give (length, features) samples")
    if src_shape.d[1] != kd:
        raise Error(name + " (MultiheadAttention): the key/value source has " + String(src_shape.d[1])
                    + " features and kdim is " + String(kd))
    var flags = _x(xargs, i, 1)
    if (flags & NN_MHA_CAUSAL) != 0 and src_shape.d[0] != cur.d[0]:
        raise Error(name + " (MultiheadAttention): a causal mask needs as many keys as queries")
    if (flags & NN_MHA_KEY_PADDING) != 0 and src_shape.d[0] != sample.d[0]:
        raise Error(name + " (MultiheadAttention): key_padding_mask reads the fit and predict lengths, which "
                    + "count the model input's positions; the keys have " + String(src_shape.d[0])
                    + " positions and the input " + String(sample.d[0]))


def nn_attention_key_padding_length(code: Int, xargs: List[Int], i: Int, sample: ShapeN) -> Int:
    """The padded length a layer's key padding mask reads (the model input's length), 0 for none."""
    if code == LAYER_NN_MHA and (_x(xargs, i, 1) & NN_MHA_KEY_PADDING) != 0:
        return sample.d[0]
    return 0


def nn_attention_param_sizes(code: Int, xargs: List[Int], i: Int, cur: ShapeN) -> List[Int]:
    var out = List[Int]()
    if code == LAYER_NN_MHA:
        var e = cur.d[1]
        var kd = nn_mha_kdim(xargs, i)
        out.append(e * (e + 2 * kd))  # in_proj: self (E, 3E); cross Wq (E, E) then [Wk | Wv] (kdim, 2E)
        out.append(e * e)             # out_proj.weight (E, E)
        out.append(3 * e)             # in_proj_bias [bq | bk | bv]
        out.append(e)                 # out_proj.bias
    return out^


def nn_attention_param_names(code: Int) -> List[String]:
    var out = List[String]()
    if code == LAYER_NN_MHA:
        out.append(".in_proj_weight")
        out.append(".out_proj.weight")
        out.append(".in_proj_bias")
        out.append(".out_proj.bias")
    return out^


def nn_attention_init_bounds(code: Int, xargs: List[Int], i: Int, cur: ShapeN) -> List[Float32]:
    """PyTorch's MultiheadAttention init: in_proj_weight xavier_uniform over the stacked (3E, E) tensor
    (bound sqrt(6 / 4E), kept with kdim), out_proj.weight Linear's kaiming_uniform(a = sqrt 5), bound
    1 / sqrt(E), both biases 0 (block_init_bounds' convention: -1 fills 0)."""
    var out = List[Float32]()
    if code == LAYER_NN_MHA:
        var e = Float32(cur.d[1])
        out.append(ftz(identical_sqrt(ftz(identical_div(Float32(6), ftz(Float32(4) * e))))))
        out.append(ftz(identical_div(Float32(1), identical_sqrt(e))))
        out.append(Float32(-1))
        out.append(Float32(-1))
    return out^


def nn_attention_tensor_dims(code: Int, xargs: List[Int], i: Int, cur: ShapeN, slot: Int, torch_layout: Bool) -> List[Int]:
    """The shape of tensor `slot`: in_proj_weight PyTorch's (3E, E) or params_' (E, 3E) for self-attention
    (cross-attention params_' is one vector, Wq then [Wk | Wv]; with kdim = E PyTorch's is (3E, E));
    out_proj.weight (E, E) both ways (the transpose: block_tensor_transposed); in_proj_bias (3E); out_proj.bias
    (E). With kdim other than E in_proj_weight is one vector of params_' words in both layouts."""
    var out = List[Int]()
    if code == LAYER_NN_MHA:
        var e = cur.d[1]
        var kd = nn_mha_kdim(xargs, i)
        var self_attn = _x(xargs, i, 2) == NN_MHA_SELF
        if slot == 0:
            if kd == e and torch_layout:
                out.append(3 * e)
                out.append(e)
            elif self_attn:
                out.append(e)
                out.append(3 * e)
            else:
                out.append(e * (e + 2 * kd))
        elif slot == 1:
            out.append(e)
            out.append(e)
        elif slot == 2:
            out.append(3 * e)
        else:
            out.append(e)
    return out^


def nn_attention_torch_blocks(code: Int, xargs: List[Int], i: Int, cur: ShapeN, slot: Int) -> List[Int]:
    """How PyTorch's (3E, E) in_proj_weight maps onto params_' words: (first PyTorch row, rows, params_ offset
    from the tensor's) per block, each block the transpose of a params_ (in, out) matrix of `rows` columns;
    empty when the tensor is not transposed by blocks (another slot, or kdim other than E)."""
    var out = List[Int]()
    if code != LAYER_NN_MHA or slot != 0:
        return out^
    var e = cur.d[1]
    if nn_mha_kdim(xargs, i) != e:
        return out^
    if _x(xargs, i, 2) == NN_MHA_SELF:
        out.append(0)
        out.append(3 * e)
        out.append(0)
    else:
        out.append(0)
        out.append(e)
        out.append(0)
        out.append(e)
        out.append(2 * e)
        out.append(e * e)
    return out^


def nn_attention_state(code: Int, xargs: List[Int], i: Int) -> Tuple[Int, Int]:
    """(offset from the layer's parameter offset, words) of words the optimizer never updates: without bias
    the two bias tensors (held at zero, read by no op), as PyTorch's bias=False has none."""
    if code == LAYER_NN_MHA and (_x(xargs, i, 1) & NN_MHA_BIAS) == 0:
        var e = _x(xargs, i, 3)
        var kd = nn_mha_kdim(xargs, i)
        return (e * (e + 2 * kd) + e * e, 4 * e)
    return (0, 0)


# ---------------------------------------------------------------- forward
def _p[E: Exec](mut tape: Tape, mut ex: E, P: FP, Gr: FP, off: Int, rows: Int, cols: Int, train: Bool) raises -> Tensor:
    if train:
        return tape.variable_into(ex, P + off, Gr + off, rows, cols)
    return tape.variable(ex, P + off, rows, cols, False)


def nn_view[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, base: Int, dims: List[Int], strides: List[Int],
) raises -> Tensor:
    """A strided view of the dense tensor x without a copy: element c is x[base + sum_k c_k strides[k]]
    (a head of a projection output). Its gradient is one slab copy of the upstream into those words of x's
    gradient (NN_TAG_VIEW; the views of one tensor read disjoint words)."""
    var op = tape.operand_nd(ex, x)
    var xs = op[2]
    var dense = xs.strides()
    for k in range(xs.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        if op[3][k] != dense[k] and xs.d[k] != 1:
            raise Error("nn_view: the source must be dense")
    if op[1] != 0:
        raise Error("nn_view: the source must be dense")
    var shape = _sh(dims)
    var st = Array[Int, RN_MAX_RANK](fill=0)
    for k in range(len(strides)):  # small-loop(rank: at most 4): shape metadata, no data
        st[k] = strides[k]
    var saved = tape.strided_view_args(ex, op[0], base, shape, st)
    var words = List[Int]()
    _put_record(words, 0, base, shape, _dense_strides(shape), st)
    saved.p1 = tape.descriptor(ex, words)
    saved.i11 = shape.count()
    var output = tape.reserve(ex, shape.count())
    return tape.custom_nd(ex, [x], output, shape, NN_TAG_VIEW, saved)


def nn_attn_softmax[E: Exec](
    mut tape: Tape, mut ex: E, scores: Tensor, B: Int, H: Int, T: Int, S: Int, scale: Float32, flags: Int,
    lengths: Optional[FP],
) raises -> Tensor:
    """softmax(scale * scores + mask) over the last axis of scores (B, H, T, S), one launch; the causal mask
    (flags & 2) and the key padding mask (flags & 4, the batch's lengths) are index comparisons in the row."""
    var a = Args()
    a.p0 = tape.value_of(ex, scores)
    var output = tape.reserve(ex, B * H * T * S)
    a.p1 = output
    var f = flags & NN_OP_CAUSAL
    if (flags & NN_OP_KEY_PADDING) != 0 and lengths:
        a.p2 = lengths.value()
        f |= NN_OP_KEY_PADDING
    a.i0 = S
    a.i1 = T
    a.i2 = H
    a.i3 = f
    a.f0 = scale
    ex.launch[OP_NN_ATTN_SOFTMAX](a, B * H * T)
    var s4: List[Int] = [B, H, T, S]
    return tape.custom_nd(ex, [scores], output, _sh(s4), NN_TAG_SOFTMAX, a)


def nn_mha_forward[E: Exec](
    mut tape: Tape, mut ex: E, xargs: List[Int], i: Int, f: Float32, cur: ShapeN, P: FP, Gr: FP, off: Int,
    h: Tensor, memory: Tensor, train: Bool, seed: UInt64, counter: Int, lengths: Optional[FP],
) raises -> Tensor:
    """multihead_attention layer i: queries from h (B, T, E), keys and values from `memory` (B, S, kdim;
    h itself for self-attention)."""
    var T = cur.d[0]
    var Ed = cur.d[1]
    var H = _x(xargs, i, 0)
    var flags = _x(xargs, i, 1)
    var kd = nn_mha_kdim(xargs, i)
    var hd = Ed // H
    var B = tape.rows(h) * tape.cols(h) // cur.count()
    var self_attn = _x(xargs, i, 2) == NN_MHA_SELF
    var bias = (flags & NN_MHA_BIAS) != 0
    var o_wo = off + Ed * (Ed + 2 * kd)
    var o_b = o_wo + Ed * Ed
    var wo = _p(tape, ex, P, Gr, o_wo, Ed, Ed, train)
    var bo: Optional[Tensor] = None
    if bias:
        bo = Optional[Tensor](_p(tape, ex, P, Gr, o_b + 3 * Ed, 1, Ed, train))
    var S = T
    var q: Tensor
    var kt: Tensor
    var v: Tensor
    if self_attn:
        # one GEMM: (B T, E) @ (E, 3E) + [bq | bk | bv], the heads as views of its (B, T, 3E) words
        var w = _p(tape, ex, P, Gr, off, Ed, 3 * Ed, train)
        var b: Optional[Tensor] = None
        if bias:
            b = Optional[Tensor](_p(tape, ex, P, Gr, o_b, 1, 3 * Ed, train))
        var qkv = tape.linear(ex, h, w, b, 0)
        var row = 3 * Ed
        var hq: List[Int] = [B, H, T, hd]
        var sq: List[Int] = [T * row, hd, row, 1]
        var hk: List[Int] = [B, H, hd, T]
        var sk: List[Int] = [T * row, hd, 1, row]
        q = nn_view(tape, ex, qkv, 0, hq, sq)
        kt = nn_view(tape, ex, qkv, Ed, hk, sk)
        v = nn_view(tape, ex, qkv, 2 * Ed, hq, sq)
    else:
        var ms = tape.shape_of(memory)
        S = ms.d[ms.ndim - 2]
        var wq = _p(tape, ex, P, Gr, off, Ed, Ed, train)
        var wkv = _p(tape, ex, P, Gr, off + Ed * Ed, kd, 2 * Ed, train)
        var bq: Optional[Tensor] = None
        var bkv: Optional[Tensor] = None
        if bias:
            bq = Optional[Tensor](_p(tape, ex, P, Gr, o_b, 1, Ed, train))
            bkv = Optional[Tensor](_p(tape, ex, P, Gr, o_b + Ed, 1, 2 * Ed, train))
        var qp = tape.linear(ex, h, wq, bq, 0)
        var kv = tape.linear(ex, memory, wkv, bkv, 0)
        var hq: List[Int] = [B, H, T, hd]
        var sq: List[Int] = [T * Ed, hd, Ed, 1]
        var hk: List[Int] = [B, H, hd, S]
        var sk: List[Int] = [S * 2 * Ed, hd, 1, 2 * Ed]
        var hv: List[Int] = [B, H, S, hd]
        var sv: List[Int] = [S * 2 * Ed, hd, 2 * Ed, 1]
        q = nn_view(tape, ex, qp, 0, hq, sq)
        kt = nn_view(tape, ex, kv, 0, hk, sk)
        v = nn_view(tape, ex, kv, Ed, hv, sv)
    var scores = rankn_matmul(tape, ex, q, kt)    # (B, H, T, S)
    var scale = ftz(identical_div(Float32(1), identical_sqrt(Float32(hd))))
    comptime if NN_SABOTAGE_SCALE:
        scale = Float32(1)
    var probs = nn_attn_softmax(tape, ex, scores, B, H, T, S, scale, flags, lengths)
    if train and f > Float32(0):
        var s4: List[Int] = [B, H, T, S]
        probs = tape.view_nd(ex, tape.dropout(ex, probs, f, seed, counter), _sh(s4))
    var ctx = rankn_matmul(tape, ex, probs, v)    # (B, H, T, hd)
    var cp: List[Int] = [0, 2, 1, 3]
    var c2: List[Int] = [B * T, Ed]
    ctx = tape.view_nd(ex, tape.permute(ex, ctx, cp), _sh(c2))   # the heads joined: one copy
    var y = tape.linear(ex, ctx, wo, bo, 0)
    var y3: List[Int] = [B, T, Ed]
    return tape.view_nd(ex, y, _sh(y3))


def nn_pos_enc_forward[E: Exec](
    mut tape: Tape, mut ex: E, f: Float32, cur: ShapeN, h: Tensor, train: Bool, seed: UInt64, counter: Int,
) raises -> Tensor:
    """h (B, T, d) + PE[:T] in one launch, then dropout (training, p > 0)."""
    var T = cur.d[0]
    var d = cur.d[1]
    var n = tape.rows(h) * tape.cols(h)
    var B = n // cur.count()
    var a = Args()
    a.p0 = tape.value_of(ex, h)
    var output = tape.reserve(ex, n)
    a.p1 = output
    a.i0 = d
    a.i1 = T
    ex.launch[OP_NN_PE_ADD](a, n)
    var y3: List[Int] = [B, T, d]
    var y = tape.custom_nd(ex, [h], output, _sh(y3), NN_TAG_PE, a)
    if train and f > Float32(0):
        y = tape.view_nd(ex, tape.dropout(ex, y, f, seed, counter), _sh(y3))
    return y


def nn_attention_forward[E: Exec](
    mut tape: Tape, mut ex: E, code: Int, xargs: List[Int], i: Int, f: Float32, cur: ShapeN, P: FP, Gr: FP,
    off: Int, h: Tensor, memory: Tensor, train: Bool, seed: UInt64, counter: Int, lengths: Optional[FP],
) raises -> Tensor:
    """Record layer i (a code of this file) on the tape; `memory` is the key/value source's output (h for
    self-attention and for positional_encoding); `counter` the dropout mask counter of this layer and step."""
    if code == LAYER_NN_MHA:
        return nn_mha_forward(tape, ex, xargs, i, f, cur, P, Gr, off, h, memory, train, seed, counter, lengths)
    if code == LAYER_NN_POS_ENC:
        return nn_pos_enc_forward(tape, ex, f, cur, h, train, seed, counter)
    raise Error("model: unknown attention layer code " + String(code))


# ---------------------------------------------------------------- backward
def nn_attention_rule(tag: Int) -> Bool:
    return tag == NN_TAG_SOFTMAX or tag == NN_TAG_PE or tag == NN_TAG_VIEW


@fieldwise_init
struct NNAttentionRules(BackwardRules, ImplicitlyCopyable, Movable):
    """The backward of the masked softmax (one VJP launch), the positional encoding (the upstream copied)
    and the head views (one slab copy into the source's gradient, the rest zero)."""

    def supports(self, tag: Int) -> Bool:
        return nn_attention_rule(tag)

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return tag == NN_TAG_VIEW

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == NN_TAG_VIEW:
            _slab_copy(ex, upstream, gradients[0], saved.p1, 0, saved.i11)
        elif tag == NN_TAG_SOFTMAX:
            var a = Args()
            a.p0 = output
            a.p1 = upstream
            a.p2 = gradients[0]
            a.i0 = 0
            a.i1 = saved.i0
            a.f0 = saved.f0
            ex.launch[OP_NN_ATTN_SOFTMAX_VJP](a, rows[0] * cols[0] // saved.i0)
        elif tag == NN_TAG_PE:
            var c = Args()
            c.p0 = upstream
            c.p1 = gradients[0]
            ex.launch[OP_COPY](c, rows[0] * cols[0])
        else:
            raise Error("nn-attention: unknown backward tag " + String(tag))
