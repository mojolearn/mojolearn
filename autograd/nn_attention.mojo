# SPDX-License-Identifier: Apache-2.0
"""Classic attention layers of a Sequential description (lane nn-attention, 2026-11-23).

  code  layer                 arguments                                 parameters (params_ order)
  67    multihead_attention   a0 num_heads, a1 flags (1 bias, 2 causal, in_proj_weight, out_proj.weight,
                              4 key_padding_mask), a2 key/value source   in_proj_bias, out_proj.bias
                              (-2 self, -1 the model input, else an     (PyTorch's names and layouts)
                              earlier layer), a3 embed_dim, a4 kdim
                              (0: embed_dim; the value width too);
                              f dropout of the attention weights
  72    positional_encoding   a0 max_len, a1 d_model; f dropout         none

PyTorch's ``nn.MultiheadAttention`` (batch_first=True): q = x Wq^T + bq, k = m Wk^T + bk, v = m Wv^T + bv
(m the key/value source, x itself for self-attention), per head softmax(q k^T / sqrt(head_dim) + mask) v,
then the output projection. The weights are held in PyTorch's (out, in) layout, so ``in_proj_weight`` is
PyTorch's stacked (3 embed_dim, embed_dim) tensor and its row slices are the three projections.
``positional_encoding`` is the sinusoidal encoding of PyTorch's transformer tutorial ("Attention Is All You
Need"): x + PE[:length], PE[p, 2j] = sin(p / 10000^(2j / d)), PE[p, 2j + 1] = cos(...), then dropout.

One body per operation: no math is written in this file. Every value comes from an existing tape op whose
host and device kernels call one body each: the projections are the transformer projection op
(graph_transformer_linear, the neural GEMM cell the fused Transformer's panels run), the bias add, scale,
mask select and softmax are the tape's own elementwise and row ops, the two attention products the rank-N
batched matmul, the dropout the tape's dropout (the same mask function as the dropout layer), and the
sinusoids are the rotary op's table body (OP_TFP_ROPE_TABLE, identical_cos / identical_sin of position *
frequency, the frequencies by graph_rotary's formula). The backward is the existing rules of those ops
(ModelRules), so this file adds no backward rule.

Launches (B batch rows, one layer, training step):
  multihead_attention forward: 3 projection GEMMs + 3 bias adds (bias) + scores matmul + scale + causal or
    padding mask (iota, compare, fill, select: 4; both masks 7) + softmax + dropout (p > 0) + context matmul
    + context copy (the permuted view's one copy) + output GEMM + bias add: 12 to 20.
  backward: about two launches per forward launch (the GEMM rules' dX and dW, the matmuls' two operands).
  positional_encoding forward: 1 frequency upload (head_dim / 2 constants, as graph_rotary) + 1 table + 1
    interleave copy + 1 add (+ dropout); backward: the add's pass-through.

The encoder and decoder layers (PyTorch's TransformerEncoderLayer / TransformerDecoderLayer) and stacks are
containers written into the description by the Sequential door (bindings/sequential_description_pack.mojo,
lane nn-attention group): multihead_attention, dropout, residual, layer_norm and linear layers; no fused
kernel and no code here.
"""
from std.sys.compile import is_defined
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_SEQ_IOTA, OP_FILL
from checks.numerics import ftz, identical_div, identical_sqrt, portable_powf
from autograd.tape import Tape, Tensor
from autograd.ops import ADD
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_shape
from autograd.rankn_bmm import rankn_matmul
from autograd.rankn_fold import rankn_binary
from autograd.rankn_layout import rankn_concat
from autograd.transformer_primitives import graph_transformer_linear
from autograd.transformer_primitives_ops import OP_TFP_ROPE_TABLE
from autograd.import_api import imp_binary, imp_where
from autograd.import_ops import IMP_GT, IMP_GE, IMP_OR

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

#: the score at a masked position before the row softmax; the value of autograd/model_blocks.mojo
#: ATTENTION_MASKED (not imported: model_blocks imports this module): exp(MASKED - max) is +0.0 for every
#: finite row max, without an infinity in the saved scores
comptime NN_ATTN_MASKED = Float32(-1.0e30)
#: the sinusoid base (PyTorch's tutorial and the paper) and the largest length the identical cos / sin
#: domain covers (graph_rotary's bound: position * frequency < 8192, the frequency of pair 0 is 1)
comptime NN_PE_BASE = Float32(10000.0)
comptime NN_PE_MAX_LEN = 8192

# Fault injection (verify sabotage arms, test builds only): the scores without 1 / sqrt(head_dim); the
# encoding with sin and cos swapped. Both must make checks/native_tests/nn_attention.mojo FAIL.
comptime NN_SABOTAGE_SCALE = is_defined["MOJOLEARN_NN_ATTENTION_SABOTAGE_SCALE"]()
comptime NN_SABOTAGE_PE = is_defined["MOJOLEARN_NN_ATTENTION_SABOTAGE_PE_SWAP"]()

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
        out.append(e * (e + 2 * kd))  # in_proj_weight: Wq (E, E), Wk (E, kdim), Wv (E, kdim), PyTorch's (out, in)
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


def nn_attention_tensor_dims(code: Int, xargs: List[Int], i: Int, cur: ShapeN, slot: Int) -> List[Int]:
    """The shape of tensor `slot` (PyTorch's and params_' layouts are the same): in_proj_weight (3E, E)
    (one vector [Wq | Wk | Wv] of E (E + 2 kdim) values when kdim differs, as PyTorch keeps three tensors
    then), out_proj.weight (E, E), in_proj_bias (3E), out_proj.bias (E)."""
    var out = List[Int]()
    if code == LAYER_NN_MHA:
        var e = cur.d[1]
        var kd = nn_mha_kdim(xargs, i)
        if slot == 0:
            if kd == e:
                out.append(3 * e)
                out.append(e)
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


def _constant[E: Exec](mut tape: Tape, mut ex: E, value: Float32) raises -> Tensor:
    var p = tape.reserve(ex, 1)
    var a = Args()
    a.p0 = p
    a.f0 = value
    ex.launch[OP_FILL](a, 1)
    return tape.variable(ex, p, 1, 1, False)


def _positions[E: Exec](mut tape: Tape, mut ex: E, n: Int) raises -> FP:
    var pos = tape.reserve(ex, n)
    var a = Args()
    a.p0 = pos
    ex.launch[OP_SEQ_IOTA](a, n)
    return pos


def _project[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, w: Tensor, b: Optional[Tensor], dims: List[Int],
) raises -> Tensor:
    """x W^T (+ b), W in PyTorch's (out, in) layout, viewed as `dims`."""
    var y = graph_transformer_linear(tape, ex, x, w)
    if b:
        y = rankn_binary(tape, ex, y, b.value(), ADD)
    return tape.view_nd(ex, y, _sh(dims))


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
    var ms = tape.shape_of(memory)
    var S = ms.d[ms.ndim - 2]
    var bias = (flags & NN_MHA_BIAS) != 0
    var o_wk = off + Ed * Ed
    var o_wv = o_wk + Ed * kd
    var o_wo = o_wv + Ed * kd
    var o_b = o_wo + Ed * Ed
    var wq = _p(tape, ex, P, Gr, off, Ed, Ed, train)
    var wk = _p(tape, ex, P, Gr, o_wk, Ed, kd, train)
    var wv = _p(tape, ex, P, Gr, o_wv, Ed, kd, train)
    var wo = _p(tape, ex, P, Gr, o_wo, Ed, Ed, train)
    var bq: Optional[Tensor] = None
    var bk: Optional[Tensor] = None
    var bv: Optional[Tensor] = None
    var bo: Optional[Tensor] = None
    if bias:
        bq = Optional[Tensor](_p(tape, ex, P, Gr, o_b, 1, Ed, train))
        bk = Optional[Tensor](_p(tape, ex, P, Gr, o_b + Ed, 1, Ed, train))
        bv = Optional[Tensor](_p(tape, ex, P, Gr, o_b + 2 * Ed, 1, Ed, train))
        bo = Optional[Tensor](_p(tape, ex, P, Gr, o_b + 3 * Ed, 1, Ed, train))
    var q4: List[Int] = [B, T, H, hd]
    var k4: List[Int] = [B, S, H, hd]
    var q = _project(tape, ex, h, wq, bq, q4)
    var k = _project(tape, ex, memory, wk, bk, k4)
    var v = _project(tape, ex, memory, wv, bv, k4)
    var qp: List[Int] = [0, 2, 1, 3]   # (B, H, T, hd)
    var kp: List[Int] = [0, 2, 3, 1]   # (B, H, hd, S)
    var qh = tape.permute(ex, q, qp)
    var kt = tape.permute(ex, k, kp)
    var vh = tape.permute(ex, v, qp)
    var scores = rankn_matmul(tape, ex, qh, kt)
    comptime if not NN_SABOTAGE_SCALE:
        scores = tape.scale(ex, scores, ftz(identical_div(Float32(1), identical_sqrt(Float32(hd)))))
    var masked = (flags & (NN_MHA_CAUSAL | NN_MHA_KEY_PADDING)) != 0
    if masked:
        var cond: Optional[Tensor] = None
        if (flags & NN_MHA_CAUSAL) != 0:
            # (T, S): key j after query i (S == T, checked at parse)
            var pos = _positions(tape, ex, T)
            var col = tape.variable(ex, pos, T, 1, False)
            var row = tape.variable(ex, pos, 1, T, False)
            cond = Optional[Tensor](imp_binary(tape, ex, row, col, IMP_GT))
        if (flags & NN_MHA_KEY_PADDING) != 0 and lengths:
            # (B, 1, 1, S): key position at or past the row's length (PyTorch's key_padding_mask True)
            var kpos = _positions(tape, ex, S)
            var krow = tape.variable(ex, kpos, 1, S, False)
            var lens = tape.variable(ex, lengths.value(), B, 1, False)
            var pad = imp_binary(tape, ex, krow, lens, IMP_GE)
            var p4: List[Int] = [B, 1, 1, S]
            pad = tape.view_nd(ex, pad, _sh(p4))
            if cond:
                cond = Optional[Tensor](imp_binary(tape, ex, cond.value(), pad, IMP_OR))
            else:
                cond = Optional[Tensor](pad)
        if cond:
            scores = imp_where(tape, ex, cond.value(), _constant(tape, ex, NN_ATTN_MASKED), scores)
    var s4: List[Int] = [B, H, T, S]
    var probs = tape.view_nd(ex, tape.softmax(ex, scores), _sh(s4))
    if train and f > Float32(0):
        probs = tape.view_nd(ex, tape.dropout(ex, probs, f, seed, counter), _sh(s4))
    var ctx = rankn_matmul(tape, ex, probs, vh)   # (B, H, T, hd)
    ctx = tape.permute(ex, ctx, qp)               # (B, T, H, hd)
    var c3: List[Int] = [B, T, Ed]
    ctx = tape.view_nd(ex, ctx, _sh(c3))
    return _project(tape, ex, ctx, wo, bo, c3)


def nn_pos_enc_forward[E: Exec](
    mut tape: Tape, mut ex: E, f: Float32, cur: ShapeN, h: Tensor, train: Bool, seed: UInt64, counter: Int,
) raises -> Tensor:
    """h (B, T, d) + PE[:T], then dropout (training, p > 0)."""
    var T = cur.d[0]
    var d = cur.d[1]
    var half = d // 2
    # the frequencies by graph_rotary's formula (autograd/transformer_primitives.mojo; head_dim d, theta 10000)
    var inv = List[Float32]()
    for j in range(half):  # cpu-route: native sinusoid frequency constants, graph_rotary's shared host contract S6
        var exponent = ftz(identical_div(Float32(2 * j), Float32(d)))
        inv.append(ftz(identical_div(Float32(1), portable_powf(NN_PE_BASE, exponent))))
    var ip = tape.reserve(ex, half)
    ex.upload(ip, FP(unsafe_from_address=Int(inv.unsafe_ptr())), half)
    _ = inv^  # the host upload source stays alive through the transfer
    var cosp = tape.reserve(ex, T * half)
    var sinp = tape.reserve(ex, T * half)
    var a = Args()
    a.p0 = ip
    a.p1 = cosp
    a.p2 = sinp
    a.i0 = half
    a.i1 = 0
    ex.launch[OP_TFP_ROPE_TABLE](a, T * half)
    var t3: List[Int] = [T, half, 1]
    var sv = tape.view_nd(ex, tape.variable(ex, sinp, T, half, False), _sh(t3))
    var cv = tape.view_nd(ex, tape.variable(ex, cosp, T, half, False), _sh(t3))
    var parts = List[Tensor]()
    comptime if NN_SABOTAGE_PE:
        parts.append(cv)
        parts.append(sv)
    else:
        parts.append(sv)
        parts.append(cv)
    var t2: List[Int] = [T, d]
    var pe = tape.view_nd(ex, rankn_concat(tape, ex, parts, 2), _sh(t2))   # (T, half, 2) -> interleaved (T, d)
    var y = rankn_binary(tape, ex, h, pe, ADD)
    if train and f > Float32(0):
        var B = tape.rows(h) * tape.cols(h) // cur.count()
        var y3: List[Int] = [B, T, d]
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
