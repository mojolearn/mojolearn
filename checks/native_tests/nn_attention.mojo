# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Classic attention in Sequential (lane nn-attention, 2026-11-23): multihead_attention (67) and
positional_encoding (72) of autograd/nn_attention.mojo, and the Transformer encoder and decoder layers the
Sequential door writes from them (bindings/sequential_description_pack.mojo, its nn-attention group: the same
module lists are written here by hand).

test_nn_attention_layers, per case, through the model interpreter (autograd/model.mojo, ModelRules):
  - the forward against a float64 interpreter of the same description written here (projections in
    PyTorch's (out, in) layout, per-head softmax(q k^T / sqrt(head_dim) + mask) v, the output projection,
    layer_norm, the feed-forward network, residuals, the sinusoidal encoding), tolerance 1e-4 (1 + |y|);
  - every parameter gradient against central finite differences of the interpreter's own loss sum(seed * y)
    (h = 1/128, tolerance 3e-3 (1 + |fd|); a parameter outside it is probed again centrally at h/4 and
    one-sided at h/16 on both sides, a ReLU kink inside the wider step bending the central difference);
  - one `NN_ATTENTION <case> digest` line over the output and gradient words.
  Cases: self-attention with bias; causal without bias (the bias words stay out of every op: their
  gradient is 0); cross-attention to an earlier layer of another length and width (kdim 5, 3 keys); the key
  padding mask from per-row lengths; attention-weight dropout in training (the finite differences run the
  same masks); the positional encoding; an encoder layer post-norm (relu) and pre-norm (gelu); a decoder
  layer (causal self-attention, cross-attention to an encoder memory).
test_nn_attention_copy: an encoder-decoder (embedding, positional encoding, one encoder layer, one causal
  decoder layer attending to it, lm_head; token cross-entropy) learns to copy 5-token sequences: held-out
  token accuracy >= 0.95, every loss, parameter and prediction finite.
Columns: the CPU column runs on HostExec. The device column runs each layer case on DeviceExec AND HostExec
and requires the same words (forward values and every gradient), and runs the copy fit on DeviceExec.
Fault injection: -D MOJOLEARN_NN_ATTENTION_SABOTAGE_SCALE (scores without 1 / sqrt(head_dim)) and
-D MOJOLEARN_NN_ATTENTION_SABOTAGE_PE_SWAP (sin and cos swapped) must make test_nn_attention_layers FAIL.
"""
from std.math import exp, sqrt, cos, sin, log, erf
from std.memory import bitcast
from std.testing import TestSuite
from autograd.tape import Tape, Tensor
from autograd.model import (
    ModelSpec, parse_model, _model_forward_prepared, _ModelForwardTopology, FIT_STATE_WORDS,
    LOSS_TOKEN_CROSS_ENTROPY, TARGET_DENSE, OUT_RAW, model_predict,
)
from autograd.model_blocks import ModelRules, LAYER_EMBEDDING, LAYER_NARROW, LAYER_GELU, LAYER_LM_HEAD
from autograd.nn_attention import LAYER_NN_MHA, LAYER_NN_POS_ENC
from autograd.model_execution import model_float_words, model_fit_native
from core.cast_in import CAST_KIND_SIGNED, CastIn
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, OPT_ADAM, dummy_ptr
from checks.native_tests.training_sequential_common import (
    SeqModel, L_LINEAR, L_RELU, L_LAYER_NORM, L_DROPOUT, L_RESIDUAL, L_FLATTEN,
)
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD

comptime L_BRANCH = 19
comptime _H = Float64(1.0) / Float64(128)
comptime _SELF = -2


def _fail(message: String) raises:
    print("FAIL NN_ATTENTION", message)
    raise Error(message)


def _mix(mut digest: UInt64, values: List[Float32]):
    for i in range(len(values)):  # small-loop(check words)
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def _same_words(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": host and device word counts differ")
    for i in range(len(a)):  # small-loop(check words)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": host and device differ at word " + String(i) + " (" + String(a[i]) + " vs "
                  + String(b[i]) + ")")


def _vals(n: Int, step: Int, modulus: Int, denom: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(n: check data)
        v[i] = Float32((i * step) % modulus - modulus // 2) / Float32(denom)
    return v^


struct _Case(Copyable, Movable):
    var name: String
    var model: SeqModel
    var rows: Int
    var features: Int
    var x: List[Float32]
    var lengths: List[Float32]
    var train: Bool

    def __init__(out self, name: String, var model: SeqModel, rows: Int, features: Int, var x: List[Float32],
                 var lengths: List[Float32], train: Bool = False):
        self.name = name
        self.model = model^
        self.rows = rows
        self.features = features
        self.x = x^
        self.lengths = lengths^
        self.train = train

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.model = SeqModel()
        self.model.codes = copy.model.codes.copy()
        self.model.args = copy.model.args.copy()
        self.model.floats = copy.model.floats.copy()
        self.model.sample = copy.model.sample.copy()
        self.rows = copy.rows
        self.features = copy.features
        self.x = copy.x.copy()
        self.lengths = copy.lengths.copy()
        self.train = copy.train


def _model(sample: List[Int]) -> SeqModel:
    var m = SeqModel()
    m.sample = sample.copy()
    return m^


def _mha(mut m: SeqModel, heads: Int, flags: Int, source: Int, embed: Int, kdim: Int = 0, p: Float64 = 0.0):
    m.layer(LAYER_NN_MHA, heads, flags, source, embed, kdim, 0, f=p)


def _ffn(mut m: SeqModel, d: Int, dff: Int, gelu: Bool):
    m.layer(L_LINEAR, dff, 1)
    if gelu:
        m.layer(LAYER_GELU, 0)
    else:
        m.layer(L_RELU)
    m.layer(L_DROPOUT, f=0.0)
    m.layer(L_LINEAR, d, 1)
    m.layer(L_DROPOUT, f=0.0)


def _encoder_layer(mut m: SeqModel, d: Int, heads: Int, dff: Int, norm_first: Bool, gelu: Bool, flags: Int = 1):
    """PyTorch's TransformerEncoderLayer as the door writes it (dropout 0, eps 1e-5)."""
    var entry = len(m.codes) - 1
    if norm_first:
        m.layer(L_LAYER_NORM, f=1e-5)
        _mha(m, heads, flags, _SELF, d)
        m.layer(L_DROPOUT, f=0.0)
        m.layer(L_RESIDUAL, entry)
        var r1 = len(m.codes) - 1
        m.layer(L_LAYER_NORM, f=1e-5)
        _ffn(m, d, dff, gelu)
        m.layer(L_RESIDUAL, r1)
    else:
        _mha(m, heads, flags, _SELF, d)
        m.layer(L_DROPOUT, f=0.0)
        m.layer(L_RESIDUAL, entry)
        m.layer(L_LAYER_NORM, f=1e-5)
        var n1 = len(m.codes) - 1
        _ffn(m, d, dff, gelu)
        m.layer(L_RESIDUAL, n1)
        m.layer(L_LAYER_NORM, f=1e-5)


def _decoder_layer(mut m: SeqModel, d: Int, heads: Int, dff: Int, memory: Int):
    """PyTorch's TransformerDecoderLayer (post-norm, causal self-attention) as the door writes it."""
    var entry = len(m.codes) - 1
    _mha(m, heads, 3, _SELF, d)
    m.layer(L_DROPOUT, f=0.0)
    m.layer(L_RESIDUAL, entry)
    m.layer(L_LAYER_NORM, f=1e-5)
    var n1 = len(m.codes) - 1
    _mha(m, heads, 1, memory, d)
    m.layer(L_DROPOUT, f=0.0)
    m.layer(L_RESIDUAL, n1)
    m.layer(L_LAYER_NORM, f=1e-5)
    var n2 = len(m.codes) - 1
    _ffn(m, d, dff, False)
    m.layer(L_RESIDUAL, n2)
    m.layer(L_LAYER_NORM, f=1e-5)


def _head(mut m: SeqModel, units: Int = 2):
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, units, 1)


def _cases() -> List[_Case]:
    var out = List[_Case]()
    var none = List[Float32]()
    var x24 = _vals(48, 13, 37, 32)   # 2 rows of (4, 6)
    # self-attention with bias: (4, 6) -> linear(8) -> mha(2 heads) -> flatten -> linear(2)
    var a = _model([4, 6])
    a.layer(L_LINEAR, 8, 1)
    _mha(a, 2, 1, _SELF, 8)
    _head(a)
    out.append(_Case("mha_self_bias", a^, 2, 24, x24.copy(), none.copy()))
    var c = _model([4, 6])
    c.layer(L_LINEAR, 8, 1)
    _mha(c, 4, 2, _SELF, 8)
    _head(c)
    out.append(_Case("mha_causal_nobias", c^, 2, 24, x24.copy(), none.copy()))
    # cross-attention: layer 0 linear(8) (queries), 1 branch(-1), 2 linear(5), 3 narrow to 3 positions (the
    # keys and values: S 3, kdim 5), 4 branch(0), 5 mha(source 3)
    var x5 = _model([4, 6])
    x5.layer(L_LINEAR, 8, 1)
    x5.layer(L_BRANCH, -1)
    x5.layer(L_LINEAR, 5, 1)
    x5.layer(LAYER_NARROW, 0, 0, 3)
    x5.layer(L_BRANCH, 0)
    _mha(x5, 2, 1, 3, 8, 5)
    _head(x5)
    out.append(_Case("mha_cross", x5^, 2, 24, x24.copy(), none.copy()))
    var kp = _model([4, 6])
    kp.layer(L_LINEAR, 8, 1)
    _mha(kp, 2, 5, _SELF, 8)
    _head(kp)
    var lens: List[Float32] = [4.0, 2.0]
    out.append(_Case("mha_key_padding", kp^, 2, 24, x24.copy(), lens^))
    var dr = _model([4, 6])
    dr.layer(L_LINEAR, 8, 1)
    _mha(dr, 2, 1, _SELF, 8, 0, 0.25)
    _head(dr)
    out.append(_Case("mha_dropout_train", dr^, 2, 24, x24.copy(), none.copy(), True))
    var pe = _model([5, 6])
    pe.layer(L_LINEAR, 6, 1)
    pe.layer(LAYER_NN_POS_ENC, 5000, 6, f=0.0)
    _head(pe)
    out.append(_Case("positional_encoding", pe^, 2, 30, _vals(60, 7, 31, 16), none.copy()))
    var e1 = _model([4, 6])
    e1.layer(L_LINEAR, 8, 1)
    _encoder_layer(e1, 8, 2, 16, False, False)
    _head(e1)
    out.append(_Case("encoder_post_relu", e1^, 2, 24, x24.copy(), none.copy()))
    var e2 = _model([4, 6])
    e2.layer(L_LINEAR, 8, 1)
    _encoder_layer(e2, 8, 2, 16, True, True)
    _head(e2)
    out.append(_Case("encoder_pre_gelu", e2^, 2, 24, x24.copy(), none.copy()))
    # decoder: 0 linear(8), 1 narrow to 3 positions (the memory), 2 branch(0), then the decoder layer
    var d1 = _model([4, 6])
    d1.layer(L_LINEAR, 8, 1)
    d1.layer(LAYER_NARROW, 0, 0, 3)
    d1.layer(L_BRANCH, 0)
    _decoder_layer(d1, 8, 2, 16, 1)
    _head(d1)
    out.append(_Case("decoder_post", d1^, 2, 24, x24.copy(), none.copy()))
    return out^


def _params(n: Int) -> List[Float32]:
    # multiples of 1/64 in [-0.36, 0.36], moved off that lattice by up to 1/500 so that no ReLU input of the
    # encoder and decoder cases lands exactly on the kink (a lattice of weights and inputs made one exactly 0.0,
    # where the gradient is one-sided and the central difference half of it)
    var v = _vals(n, 17, 47, 64)
    for i in range(n):  # small-loop(n: check data)
        v[i] += Float32((i * 7) % 13 - 6) / Float32(3000)
    return v^


def _run[E: Exec](mut ex: E, c: _Case, spec: ModelSpec, params: List[Float32], grad: Bool,
                  mut grads: List[Float32]) raises -> List[Float32]:
    var tape = Tape()
    var holder = Tape()
    var pt = holder.tensor(ex, params, 1, len(params))
    var zeros = List[Float32](length=len(params), fill=0)
    var gt = holder.tensor(ex, zeros, 1, len(params))
    var x = tape.tensor(ex, c.x, c.rows, c.features)
    var lens = Optional[FP]()
    var lp = dummy_ptr()
    if len(c.lengths) > 0:
        lp = ex.alloc(len(c.lengths))
        ex.upload(lp, FP(unsafe_from_address=Int(c.lengths.unsafe_ptr())), len(c.lengths))
        lens = Optional[FP](lp)
    var topology = _ModelForwardTopology(spec)
    var y = _model_forward_prepared(tape, ex, spec, holder.value(pt), holder.value(gt), x, grad or c.train,
                                    UInt64(7), 3, topology, lens)
    var yv = tape.values(ex, y)
    if grad:
        var seed = _vals(len(yv), 3, 11, 16)
        var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
        tape.backward_with_rules(ex, y, st, ModelRules())
        grads = holder.values(ex, gt)
    if len(c.lengths) > 0:
        ex.free(lp)
    _ = c.lengths
    return yv^


def _loss[E: Exec](mut ex: E, c: _Case, spec: ModelSpec, params: List[Float32]) raises -> Float64:
    var g = List[Float32]()
    var yv = _run(ex, c, spec, params, False, g)
    var seed = _vals(len(yv), 3, 11, 16)
    var s = Float64(0)
    for i in range(len(yv)):  # small-loop(check loss)
        s += Float64(seed[i]) * Float64(yv[i])
    return s


# ---------------------------------------------------------------- float64 interpreter of the descriptions
struct _M(Copyable, Movable):
    """One sample's tensor: (rows, cols) float64, row-major."""
    var rows: Int
    var cols: Int
    var v: List[Float64]

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.v = List[Float64](length=rows * cols, fill=0)


def _lin64(x: _M, p: List[Float32], off: Int, fo: Int, bias: Bool) -> _M:
    """params_' Linear: weight (in, out) row-major, then bias."""
    var y = _M(x.rows, fo)
    for r in range(x.rows):  # small-loop(oracle)
        for o in range(fo):  # small-loop(oracle)
            var s = Float64(0)
            for k in range(x.cols):  # small-loop(oracle)
                s += x.v[r * x.cols + k] * Float64(p[off + k * fo + o])
            if bias:
                s += Float64(p[off + x.cols * fo + o])
            y.v[r * fo + o] = s
    return y^


def _cols64(x: _M, p: List[Float32], w_off: Int, width: Int, col0: Int, fo: Int, b_off: Int) -> _M:
    """x @ W[:, col0 : col0 + fo] + b, W (in, width) row-major at w_off (the tape's Linear layout), b at
    b_off (-1: none)."""
    var y = _M(x.rows, fo)
    for r in range(x.rows):  # small-loop(oracle)
        for o in range(fo):  # small-loop(oracle)
            var s = Float64(0)
            for k in range(x.cols):  # small-loop(oracle)
                s += x.v[r * x.cols + k] * Float64(p[w_off + k * width + col0 + o])
            if b_off >= 0:
                s += Float64(p[b_off + o])
            y.v[r * fo + o] = s
    return y^


def _mha64(xq: _M, xkv: _M, p: List[Float32], off: Int, heads: Int, flags: Int, length: Int, self_attn: Bool) -> _M:
    """params_' layout: self in_proj (E, 3E) columns [q | k | v]; cross Wq (E, E) then [Wk | Wv] (kdim, 2E);
    out_proj (E, E); in_proj_bias [bq | bk | bv]; out_proj.bias. All (in, out)."""
    var e = xq.cols
    var kd = xkv.cols
    var hd = e // heads
    var bias = (flags & 1) != 0
    var o_wo = off + e * (e + 2 * kd)
    var o_b = o_wo + e * e
    var q: _M
    var k: _M
    var v: _M
    if self_attn:
        q = _cols64(xq, p, off, 3 * e, 0, e, o_b if bias else -1)
        k = _cols64(xkv, p, off, 3 * e, e, e, o_b + e if bias else -1)
        v = _cols64(xkv, p, off, 3 * e, 2 * e, e, o_b + 2 * e if bias else -1)
    else:
        q = _cols64(xq, p, off, e, 0, e, o_b if bias else -1)
        k = _cols64(xkv, p, off + e * e, 2 * e, 0, e, o_b + e if bias else -1)
        v = _cols64(xkv, p, off + e * e, 2 * e, e, e, o_b + 2 * e if bias else -1)
    var T = xq.rows
    var S = xkv.rows
    var ctx = _M(T, e)
    for h in range(heads):  # small-loop(oracle)
        for i in range(T):  # small-loop(oracle)
            var sc = List[Float64](length=S, fill=0)
            var ok = List[Bool](length=S, fill=True)
            var mx = Float64(-1e300)
            for j in range(S):  # small-loop(oracle)
                if ((flags & 2) != 0 and j > i) or ((flags & 4) != 0 and j >= length):
                    ok[j] = False
                    continue
                var s = Float64(0)
                for d in range(hd):  # small-loop(oracle)
                    s += q.v[i * e + h * hd + d] * k.v[j * e + h * hd + d]
                sc[j] = s / sqrt(Float64(hd))
                mx = max(mx, sc[j])
            var den = Float64(0)
            for j in range(S):  # small-loop(oracle)
                if ok[j]:
                    sc[j] = exp(sc[j] - mx)
                    den += sc[j]
            for j in range(S):  # small-loop(oracle)
                if ok[j]:
                    for d in range(hd):  # small-loop(oracle)
                        ctx.v[i * e + h * hd + d] += sc[j] / den * v.v[j * e + h * hd + d]
    return _cols64(ctx, p, o_wo, e, 0, e, o_b + 3 * e if bias else -1)


def _ln64(x: _M, p: List[Float32], off: Int, eps: Float64) -> _M:
    var y = _M(x.rows, x.cols)
    var d = x.cols
    for r in range(x.rows):  # small-loop(oracle)
        var mean = Float64(0)
        for k in range(d):  # small-loop(oracle)
            mean += x.v[r * d + k]
        mean /= Float64(d)
        var var_ = Float64(0)
        for k in range(d):  # small-loop(oracle)
            var t = x.v[r * d + k] - mean
            var_ += t * t
        var_ /= Float64(d)
        var rs = 1.0 / sqrt(var_ + eps)
        for k in range(d):  # small-loop(oracle)
            y.v[r * d + k] = (x.v[r * d + k] - mean) * rs * Float64(p[off + k]) + Float64(p[off + d + k])
    return y^


def _oracle(c: _Case, spec: ModelSpec, p: List[Float32]) raises -> List[Float64]:
    """The description of case c in float64 (dropout layers p 0; the dropout case has no oracle)."""
    var codes = c.model.codes.copy()
    var margs = c.model.args.copy()
    var floats = c.model.floats.copy()
    var T = c.model.sample[0]
    var F = c.model.sample[1]
    var y = List[Float64]()
    for b in range(c.rows):  # small-loop(oracle: samples)
        var x0 = _M(T, F)
        for k in range(T * F):  # small-loop(oracle)
            x0.v[k] = Float64(c.x[b * T * F + k])
        var length = T if len(c.lengths) == 0 else Int(c.lengths[b])
        var outs = List[_M]()
        var h = x0.copy()
        for i in range(len(codes)):  # small-loop(oracle: layers)
            var k = codes[i]
            var a0 = margs[6 * i]
            var a1 = margs[6 * i + 1]
            var a2 = margs[6 * i + 2]
            var off = spec.param_off[i]
            var f = floats[i]
            if k == L_LINEAR:
                h = _lin64(h, p, off, a0, a1 == 1)
            elif k == L_RELU:
                for q in range(len(h.v)):  # small-loop(oracle)
                    h.v[q] = max(h.v[q], 0.0)
            elif k == LAYER_GELU:
                for q in range(len(h.v)):  # small-loop(oracle)
                    h.v[q] = 0.5 * h.v[q] * (1.0 + erf(h.v[q] / sqrt(2.0)))
            elif k == L_DROPOUT:
                pass
            elif k == L_LAYER_NORM:
                h = _ln64(h, p, off, f)
            elif k == L_RESIDUAL:
                var src = x0.copy() if a0 == -1 else outs[a0].copy()
                for q in range(len(h.v)):  # small-loop(oracle)
                    h.v[q] += src.v[q]
            elif k == L_BRANCH:
                h = x0.copy() if a0 == -1 else outs[a0].copy()
            elif k == LAYER_NARROW:
                if a0 != 0:
                    raise Error("oracle: narrow on axis 0 only")
                var n = _M(a2, h.cols)
                for q in range(a2 * h.cols):  # small-loop(oracle)
                    n.v[q] = h.v[a1 * h.cols + q]
                h = n^
            elif k == L_FLATTEN:
                h.cols = h.rows * h.cols
                h.rows = 1
            elif k == LAYER_NN_MHA:
                var kv = h.copy() if a2 == _SELF else (x0.copy() if a2 == -1 else outs[a2].copy())
                h = _mha64(h, kv, p, off, a0, a1, length, a2 == _SELF)
            elif k == LAYER_NN_POS_ENC:
                var d = h.cols
                for t in range(h.rows):  # small-loop(oracle)
                    for j in range(d // 2):  # small-loop(oracle)
                        var ang = Float64(t) / (10000.0 ** (Float64(2 * j) / Float64(d)))
                        h.v[t * d + 2 * j] += sin(ang)
                        h.v[t * d + 2 * j + 1] += cos(ang)
            else:
                raise Error("oracle: layer code " + String(k))
            outs.append(h.copy())
        for q in range(len(h.v)):  # small-loop(oracle)
            y.append(h.v[q])
    return y^


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": length " + String(len(got)) + " expected " + String(len(want)))
    for i in range(len(got)):  # small-loop(check words)
        var d = abs(Float64(got[i]) - want[i])
        if not (d <= tol * (1.0 + abs(want[i]))):
            _fail(name + ": element " + String(i) + " got " + String(got[i]) + " expected " + String(want[i]))


def _fd[E: Exec](mut ex: E, c: _Case, spec: ModelSpec, params: List[Float32], i: Int, h: Float64) raises -> Float64:
    var up = params.copy()
    up[i] = Float32(Float64(params[i]) + h)
    var down = params.copy()
    down[i] = Float32(Float64(params[i]) - h)
    return (_loss(ex, c, spec, up) - _loss(ex, c, spec, down)) / (Float64(up[i]) - Float64(down[i]))


def run_case[E: Exec](mut ex: E, c: _Case, finite_differences: Bool) raises -> List[Float32]:
    var spec = parse_model(c.model.program(c.features), model_float_words(c.model.table()))
    var params = _params(spec.n_params)
    var grads = List[Float32]()
    var yv = _run(ex, c, spec, params, True, grads)
    if not c.train:
        _close(c.name + " forward", yv, _oracle(c, spec, params), 1e-4)
    if finite_differences:
        var fd = List[Float64](length=len(params), fill=0)
        for i in range(len(params)):  # small-loop(every parameter probed)
            fd[i] = _fd(ex, c, spec, params, i, _H)
            # a ReLU kink inside [p - h, p + h] bends the central difference: such a parameter is probed
            # again with a quarter of the step, and the gradient must match that one
            var d = abs(Float64(grads[i]) - fd[i])
            if not (d <= 3e-3 * (1.0 + abs(fd[i]))):
                # a ReLU input inside [p - h, p + h] bends the central difference: probe again at h/4 and,
                # one-sided, at h/16 (the side away from the kink is the derivative the tape takes)
                var probes = List[Float64]()
                probes.append(_fd(ex, c, spec, params, i, _H / 4.0))
                var base = _loss(ex, c, spec, params)
                var up = params.copy()
                up[i] = Float32(Float64(params[i]) + _H / 16.0)
                var down = params.copy()
                down[i] = Float32(Float64(params[i]) - _H / 16.0)
                probes.append((_loss(ex, c, spec, up) - base) / (Float64(up[i]) - Float64(params[i])))
                probes.append((base - _loss(ex, c, spec, down)) / (Float64(params[i]) - Float64(down[i])))
                var best = fd[i]
                for k in range(len(probes)):  # small-loop(3 probes)
                    if abs(Float64(grads[i]) - probes[k]) < abs(Float64(grads[i]) - best):
                        best = probes[k]
                print("NN_ATTENTION", c.name, "re-probe", i, "grad", grads[i], "fd h", fd[i], "h/4", probes[0],
                      "right h/16", probes[1], "left h/16", probes[2])
                fd[i] = best
        _close(c.name + " gradient", grads, fd, 3e-3)
    var words = yv.copy()
    for i in range(len(grads)):  # small-loop(check words)
        words.append(grads[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_ATTENTION", c.name, "params", spec.n_params, "digest", digest)
    return words^


def test_nn_attention_layers() raises:
    var cases = _cases()
    for k in range(len(cases)):  # driver-loop: over the layer cases
        var host = HostExec()
        var hw = run_case(host, cases[k], True)
        comptime if not CPU_COLUMN_BUILD:
            var dev = DeviceExec()
            var dw = run_case(dev, cases[k], False)
            _same_words(cases[k].name, hw, dw)
    print("PASS NN_ATTENTION layers", len(cases))


# ---------------------------------------------------------------- the copy task
comptime _CP_L = 5          # tokens per sequence
comptime _CP_V = 6          # 0 the start token, 1..5 the symbols
comptime _CP_D = 16
comptime _CP_TRAIN = 256
comptime _CP_TEST = 128
comptime _CP_EPOCHS = 40


def _copy_model() -> SeqModel:
    """Input (2 L,) token ids [source | start, source[:-1]]; target the source."""
    var m = _model([2 * _CP_L])
    m.layer(LAYER_EMBEDDING, _CP_V, _CP_D)
    m.layer(LAYER_NARROW, 0, 0, _CP_L)
    m.layer(LAYER_NN_POS_ENC, 5000, _CP_D, f=0.0)
    _encoder_layer(m, _CP_D, 2, 32, False, False)
    var memory = len(m.codes) - 1
    m.layer(L_BRANCH, 0)
    m.layer(LAYER_NARROW, 0, _CP_L, _CP_L)
    m.layer(LAYER_NN_POS_ENC, 5000, _CP_D, f=0.0)
    _decoder_layer(m, _CP_D, 2, 32, memory)
    m.layer(LAYER_LM_HEAD, _CP_V)
    return m^


def _copy_data(rows: Int, salt: Int, mut x: List[Float32], mut y: List[Int32]):
    var state = UInt64(0x9E3779B97F4A7C15) ^ UInt64(salt)
    for r in range(rows):  # small-loop(fixture rows)
        var src = List[Int]()
        for _ in range(_CP_L):  # small-loop(fixture tokens)
            state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            src.append(1 + Int((state >> 33) % UInt64(_CP_V - 1)))
        for t in range(_CP_L):  # small-loop(fixture tokens)
            x.append(Float32(src[t]))
        x.append(Float32(0))
        for t in range(_CP_L - 1):  # small-loop(fixture tokens)
            x.append(Float32(src[t]))
        for t in range(_CP_L):  # small-loop(fixture tokens)
            y.append(Int32(src[t]))
        _ = r


def _copy_fit[E: Exec](mut ex: E) raises -> Tuple[Float64, List[Float32]]:
    """Fit the copy model (adam, lr 0.01, batch 32, seed 0); (held-out token accuracy, loss curve)."""
    var m = _copy_model()
    var features = 2 * _CP_L
    var prog = m.program(features)
    var table = m.table()
    var spec = parse_model(prog, model_float_words(table))
    var x = List[Float32]()
    var y = List[Int32]()
    _copy_data(_CP_TRAIN, 1, x, y)
    var params = List[Float32](length=spec.n_params, fill=Float32(0))
    var mm = List[Float32](length=spec.n_params, fill=Float32(0))
    var vv = List[Float32](length=spec.n_params, fill=Float32(0))
    var state = List[Int64](length=FIT_STATE_WORDS, fill=Int64(0))
    var curve = List[Float32](length=_CP_EPOCHS, fill=Float32(0))
    var ip: List[Int] = [
        _CP_TRAIN, LOSS_TOKEN_CROSS_ENTROPY, TARGET_DENSE, OPT_ADAM, 0, 0, 1, 32, _CP_EPOCHS, 1, 0, 0, 0, 1, 0,
    ]
    var fp: List[Float64] = [0.01, 0.9, 0.999, 1e-8, 0.0, 0.0, 1e-4, 0.1, 1.0]
    for i in range(len(table)):  # driver-loop: over layers
        fp.append(table[i])
    var addrs: List[Int] = [
        0, 0, Int(params.unsafe_ptr()), Int(curve.unsafe_ptr()), Int(mm.unsafe_ptr()), Int(vv.unsafe_ptr()),
        Int(state.unsafe_ptr()), 0,
    ]
    var sources: List[CastIn] = [
        CastIn.f32_c(Int(x.unsafe_ptr()), _CP_TRAIN, features),
        CastIn(Int(y.unsafe_ptr()), CAST_KIND_SIGNED, 4, _CP_TRAIN, _CP_L, _CP_L * 4, 4),
    ]
    _ = model_fit_native(ex, addrs, sources, prog, ip, fp)
    ex.sync()
    _ = x^
    _ = y^
    _ = mm^
    _ = vv^
    _ = state^
    for i in range(len(params)):  # small-loop(check words)
        if params[i] != params[i] or abs(params[i]) > Float32(1e6):
            _fail("copy fit: parameter " + String(i) + " is not finite")
    for i in range(len(curve)):  # small-loop(check words)
        if curve[i] != curve[i] or abs(curve[i]) > Float32(1e6):
            _fail("copy fit: loss of epoch " + String(i) + " is not finite")
    var xt = List[Float32]()
    var yt = List[Int32]()
    _copy_data(_CP_TEST, 2, xt, yt)
    var out = List[Float32](length=_CP_TEST * _CP_L * _CP_V, fill=Float32(-7.0))
    model_predict(ex, spec, CastIn.f32_c(Int(xt.unsafe_ptr()), _CP_TEST, features), _CP_TEST,
                  FP(unsafe_from_address=Int(params.unsafe_ptr())), FP(unsafe_from_address=Int(out.unsafe_ptr())),
                  4096, OUT_RAW, 0, 0, 0, Optional[FP]())
    ex.sync()
    _ = xt^
    var right = 0
    for t in range(_CP_TEST * _CP_L):  # small-loop(check tokens)
        var best = 0
        for v in range(_CP_V):  # small-loop(check logits)
            var z = out[t * _CP_V + v]
            if z != z or abs(z) > Float32(1e6):
                _fail("copy fit: prediction " + String(t) + " is not finite")
            if z > out[t * _CP_V + best]:
                best = v
        if Int32(best) == yt[t]:
            right += 1
    _ = params^
    return (Float64(right) / Float64(_CP_TEST * _CP_L), curve^)


def test_nn_attention_copy() raises:
    var host = HostExec()
    var r = _copy_fit(host)
    var acc = r[0]
    var curve = r[1].copy()
    print("NN_ATTENTION copy host loss", curve[0], "->", curve[len(curve) - 1], "held-out token accuracy", acc)
    if not (acc >= 0.95):
        _fail("copy fit (host): held-out token accuracy " + String(acc) + " is below 0.95")
    comptime if not CPU_COLUMN_BUILD:
        var dev = DeviceExec()
        var rd = _copy_fit(dev)
        print("NN_ATTENTION copy device held-out token accuracy", rd[0])
        if not (rd[0] >= 0.95):
            _fail("copy fit (device): held-out token accuracy " + String(rd[0]) + " is below 0.95")
    print("PASS NN_ATTENTION copy")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
