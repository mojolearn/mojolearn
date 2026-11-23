# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The neural building blocks of Sequential (lane neural-blocks, 2026-11-22).

Each block layer of autograd/model_blocks.mojo (silu, gelu erf and tanh, softplus, rms_norm, rotary,
multiply / the gated container, narrow after a padded depthwise conv1d (Mamba's causal conv), embedding,
attention: multi-head, grouped-query, rotary, causal, sliding window and non-causal; rnn, lstm, gru;
selective_scan, Mamba-1's core as a generic op, lane neural-blocks-2, at state sizes 4 and 8; ssd_scan,
Mamba-2's SSD core as a generic op, lane neural-blocks-3, at one and two heads; trapezoid_scan, Mamba-3's
core as a generic op, lane neural-blocks-3-m3, at 1 and 2 heads; lm_head, the LM head's logits, lane
neural-blocks-3-lm) is
checked through the model interpreter (autograd/model.mojo model_forward, the version-2 backward with
ModelRules), inside a small model `linear -> block -> flatten -> linear`:
  - every parameter gradient against central finite differences of the interpreter's own forward loss
    sum(seed * y) (h = 1/128, tolerance 3e-3 (1 + |fd|));
  - the attention forward against a float64 oracle written here (projections, rotary, mask, softmax);
  - the rms_norm forward against a float64 oracle;
  - the selective_scan forward against a float64 oracle of the reference recurrence;
  - the ssd_scan forward against a float64 oracle of the plain (unchunked) Mamba-2 recurrence;
  - the trapezoid_scan forward against a float64 oracle of the trapezoid recurrence;
  - the lm_head logits against a float64 oracle; and (test_lm_head_token_loss) the token cross-entropy as
    the fit records it (model_token_loss: the forward stops before the head, the fused head and loss
    autograd/lm_head_v2_graph.mojo run on its input): the loss against a float64 oracle AND against the
    cross-entropy of the model's own predicted logits, the gradients of the head weight, of the earlier
    layers and of the head's input against central finite differences of the recorded loss, the digest
    pinned (`-D MOJOLEARN_LMHEAD_SABOTAGE_LEAF_DESCENDING` must make it FAIL);
  - one `NEURAL_BLOCKS <case> digest` line over the output and gradient words.
Columns: the CPU column runs on HostExec. The device column runs each case on DeviceExec AND HostExec and
requires the same words (forward values and every gradient): the host-column-versus-device bytes check.
"""
from std.math import exp, sqrt, cos, sin, log, tanh
from std.memory import bitcast
from std.testing import TestSuite
from autograd.tape import Tape, Tensor
from autograd.model import ModelSpec, parse_model, model_forward
from autograd.model_blocks import (
    ModelRules, LAYER_SILU, LAYER_GELU, LAYER_SOFTPLUS, LAYER_RMS_NORM, LAYER_ROTARY, LAYER_MULTIPLY,
    LAYER_NARROW, LAYER_EMBEDDING, LAYER_ATTENTION, LAYER_RECURRENT, LAYER_SELECTIVE_SCAN,
    LAYER_SSD_SCAN,
    LAYER_TRAPEZOID_SCAN,  # Mamba-3 (lane neural-blocks-3-m3)
)
# lm_head (lane neural-blocks-3-lm)
from autograd.model_blocks import LAYER_LM_HEAD
from autograd.model import model_token_loss
from autograd.model_execution import model_float_words
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.training_sequential_common import SeqModel, L_LINEAR, L_FLATTEN, L_PERMUTE
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from autograd.model_verify import model_input_id_range, verify_fill

comptime L_CONV1D = 15
comptime L_BRANCH = 19
comptime _H = Float64(1.0) / Float64(128)


def _fail(message: String) raises:
    print("FAIL NEURAL_BLOCKS", message)
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

    def __init__(out self, name: String, var model: SeqModel, rows: Int, features: Int, var x: List[Float32]):
        self.name = name
        self.model = model^
        self.rows = rows
        self.features = features
        self.x = x^

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


def _model(sample: List[Int]) -> SeqModel:
    var m = SeqModel()
    m.sample = sample.copy()
    return m^


def _head(mut m: SeqModel, units: Int = 2):
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, units, 1)


def _cases() -> List[_Case]:
    var out = List[_Case]()
    var acts: List[Int] = [LAYER_SILU, LAYER_GELU, LAYER_GELU, LAYER_SOFTPLUS]
    var names: List[String] = ["silu", "gelu", "gelu_tanh", "softplus"]
    for k in range(4):  # small-loop(4 activations)
        var m = _model([3, 4])
        m.layer(L_LINEAR, 6, 1)
        m.layer(acts[k], 1 if k == 2 else 0)
        _head(m)
        out.append(_Case(names[k], m^, 3, 12, _vals(36, 7, 23, 16)))
    var r = _model([3, 4])
    r.layer(L_LINEAR, 8, 1)
    r.layer(LAYER_RMS_NORM, f=1e-6)
    _head(r)
    out.append(_Case("rms_norm", r^, 3, 12, _vals(36, 5, 19, 16)))
    var ro = _model([5, 4])
    ro.layer(L_LINEAR, 8, 1)
    ro.layer(LAYER_ROTARY, 4, f=10000.0)
    _head(ro)
    out.append(_Case("rotary", ro^, 2, 20, _vals(40, 3, 17, 16)))
    # gated: layer 0 linear, 1 linear (gate), 2 silu, 3 branch(0), 4 linear (up), 5 multiply(2)
    var g = _model([3, 4])
    g.layer(L_LINEAR, 6, 1)
    g.layer(L_LINEAR, 5, 0)
    g.layer(LAYER_SILU)
    g.layer(L_BRANCH, 0)
    g.layer(L_LINEAR, 5, 0)
    g.layer(LAYER_MULTIPLY, 2)
    _head(g)
    out.append(_Case("gated", g^, 3, 12, _vals(36, 11, 29, 16)))
    # Mamba's causal depthwise conv: (L 6, C 4) -> permute (C, L) -> conv1d k 3, padding 2, groups 4 -> the
    # first 6 positions
    var c = _model([6, 4])
    c.layer(L_LINEAR, 4, 1)
    c.layer(L_PERMUTE, 1, 0, -1, -1, -1, -1)
    c.layer(L_CONV1D, 4, 3, 1, 2, 1, 4)
    c.layer(LAYER_NARROW, 1, 0, 6)
    _head(c)
    out.append(_Case("causal_conv_narrow", c^, 2, 24, _vals(48, 7, 31, 16)))
    var e = _model([5])
    e.layer(LAYER_EMBEDDING, 7, 4)
    _head(e)
    var ids = List[Float32]()
    for i in range(15):  # small-loop(15 ids)
        ids.append(Float32((i * 3) % 7))
    out.append(_Case("embedding", e^, 3, 5, ids^))
    # attention: (heads, kv_heads, rotary, window, causal)
    var att: List[List[Int]] = [[2, 1, 1, 0, 1], [2, 2, 0, 0, 1], [2, 2, 1, 3, 1], [2, 2, 0, 0, 0]]
    var att_names: List[String] = ["attention_gqa_rotary", "attention_mha", "attention_window3", "attention_full"]
    for k in range(len(att)):  # small-loop(4 attention forms)
        var a = _model([5, 8])
        a.layer(L_LINEAR, 8, 1)
        a.layer(LAYER_ATTENTION, att[k][0], att[k][1], att[k][2], att[k][3], att[k][4], 0, f=10000.0)
        _head(a)
        out.append(_Case(att_names[k], a^, 2, 40, _vals(80, 13, 37, 32)))
    var cell_names: List[String] = ["rnn", "lstm", "gru"]
    for k in range(3):  # small-loop(3 cells)
        var rr = _model([4, 3])
        rr.layer(L_LINEAR, 3, 1)
        rr.layer(LAYER_RECURRENT, 5, k)
        _head(rr)
        out.append(_Case(cell_names[k], rr^, 2, 12, _vals(24, 5, 13, 8)))
    # selective_scan: (L 6, 4) -> linear(2 dim + 2 state) = [u | delta | B | C], dim 3 -> scan -> (6, 3)
    var states: List[Int] = [4, 8]
    for k in range(2):  # small-loop(2 state sizes)
        var ss = _model([6, 4])
        ss.layer(L_LINEAR, 6 + 2 * states[k], 1)
        ss.layer(LAYER_SELECTIVE_SCAN, states[k])
        _head(ss)
        out.append(_Case("selective_scan_n" + String(states[k]), ss^, 2, 24, _vals(48, 7, 31, 16)))
    # Mamba-2 (lane neural-blocks-3) ssd_scan: (L, 2) -> linear(65 heads + 256) -> softplus (dt > 0) -> scan
    var heads: List[Int] = [1, 2]
    var lens: List[Int] = [6, 3]
    for k in range(2):  # small-loop(2 head counts)
        var sd = _model([lens[k], 2])
        sd.layer(L_LINEAR, 65 * heads[k] + 256, 1)
        sd.layer(LAYER_SOFTPLUS, 0)
        sd.layer(LAYER_SSD_SCAN, heads[k])
        _head(sd)
        out.append(_Case("ssd_scan_h" + String(heads[k]), sd^, 2, 2 * lens[k], _vals(4 * lens[k], 7, 31, 16)))
    # Mamba-3 (lane neural-blocks-3-m3) trapezoid_scan: (L 6, 4) -> linear(heads (P + 2 N + 3)) =
    # [v | k | q | dt | A | trap] -> scan -> (6, heads P); (heads, headdim, state) = (2, 2, 3) and (1, 3, 4)
    var tshape: List[List[Int]] = [[2, 2, 3], [1, 3, 4]]
    for k in range(2):  # small-loop(2 head layouts)
        var ts = _model([6, 4])
        ts.layer(L_LINEAR, tshape[k][0] * (tshape[k][1] + 2 * tshape[k][2] + 3), 1)
        ts.layer(LAYER_TRAPEZOID_SCAN, tshape[k][0], tshape[k][2])
        _head(ts)
        out.append(_Case("trapezoid_scan_h" + String(tshape[k][0]), ts^, 2, 24, _vals(48, 5, 29, 16)))
    # lm_head (lane neural-blocks-3-lm): (L 5, 4) -> linear(6) -> lm_head(11) logits (5, 11) -> flatten -> linear
    var lh = _model([5, 4])
    lh.layer(L_LINEAR, 6, 1)
    lh.layer(LAYER_LM_HEAD, 11)
    _head(lh)
    out.append(_Case("lm_head_logits", lh^, 2, 20, _vals(40, 7, 31, 16)))
    return out^


def _params(n: Int) -> List[Float32]:
    # multiples of 1/64 in [-0.36, 0.36]
    return _vals(n, 17, 47, 64)


def _run[E: Exec](mut ex: E, c: _Case, spec: ModelSpec, params: List[Float32], grad: Bool,
                  mut grads: List[Float32]) raises -> List[Float32]:
    var tape = Tape()
    var holder = Tape()
    var pt = holder.tensor(ex, params, 1, len(params))
    var zeros = List[Float32](length=len(params), fill=0)
    var gt = holder.tensor(ex, zeros, 1, len(params))
    var x = tape.tensor(ex, c.x, c.rows, c.features)
    var y = model_forward(tape, ex, spec, holder.value(pt), holder.value(gt), x, grad, UInt64(0), 0)
    var yv = tape.values(ex, y)
    if grad:
        var seed = _vals(len(yv), 3, 11, 16)
        var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
        tape.backward_with_rules(ex, y, st, ModelRules())
        grads = holder.values(ex, gt)
    return yv^


def _loss[E: Exec](mut ex: E, c: _Case, spec: ModelSpec, params: List[Float32]) raises -> Float64:
    var g = List[Float32]()
    var yv = _run(ex, c, spec, params, False, g)
    var seed = _vals(len(yv), 3, 11, 16)
    var s = Float64(0)
    for i in range(len(yv)):  # small-loop(check loss)
        s += Float64(seed[i]) * Float64(yv[i])
    return s


# ---------------------------------------------------------------- float64 oracles
def _linear64(x: List[Float64], rows: Int, fi: Int, w: List[Float32], off: Int, fo: Int, bias: Bool) -> List[Float64]:
    var y = List[Float64](length=rows * fo, fill=0)
    for r in range(rows):  # small-loop(oracle)
        for o in range(fo):  # small-loop(oracle)
            var s = Float64(0)
            for k in range(fi):  # small-loop(oracle)
                s += x[r * fi + k] * Float64(w[off + k * fo + o])
            if bias:
                s += Float64(w[off + fi * fo + o])
            y[r * fo + o] = s
    return y^


def _rotary64(mut v: List[Float64], rows: Int, T: Int, width: Int, hd: Int, theta: Float64):
    var half = hd // 2
    for r in range(rows):  # small-loop(oracle)
        var t = r % T
        for h0 in range(0, width, hd):  # small-loop(oracle)
            for j in range(half):  # small-loop(oracle)
                var freq = 1.0 / (theta ** (Float64(2 * j) / Float64(hd)))
                var ang = Float64(t) * freq
                var a = v[r * width + h0 + j]
                var b = v[r * width + h0 + j + half]
                v[r * width + h0 + j] = a * cos(ang) - b * sin(ang)
                v[r * width + h0 + j + half] = b * cos(ang) + a * sin(ang)


def _attention_oracle(c: _Case, spec: ModelSpec, p: List[Float32]) -> List[Float64]:
    """The model `linear(8) -> attention -> flatten -> linear(2)` in float64."""
    var T = 5
    var D = 8
    var B = c.rows
    var a = c.model.args.copy()
    var H = a[6]
    var Hk = a[7]
    var rot = a[8] == 1
    var window = a[9]
    var causal = a[10] == 1
    var hd = D // H
    var G = H // Hk
    var x = List[Float64]()
    for i in range(len(c.x)):  # small-loop(oracle)
        x.append(Float64(c.x[i]))
    var h = _linear64(x, B * T, 8, p, 0, D, True)
    var off = spec.param_off[1]
    var q = _linear64(h, B * T, D, p, off, H * hd, False)
    var k = _linear64(h, B * T, D, p, off + D * H * hd, Hk * hd, False)
    var v = _linear64(h, B * T, D, p, off + D * H * hd + D * Hk * hd, Hk * hd, False)
    if rot:
        _rotary64(q, B * T, T, H * hd, hd, 10000.0)
        _rotary64(k, B * T, T, Hk * hd, hd, 10000.0)
    var ctx = List[Float64](length=B * T * H * hd, fill=0)
    for b in range(B):  # small-loop(oracle)
        for hh in range(H):  # small-loop(oracle)
            var kv = hh // G
            for i in range(T):  # small-loop(oracle)
                var sc = List[Float64](length=T, fill=0)
                var ok = List[Bool](length=T, fill=True)
                var mx = Float64(-1e300)
                for j in range(T):  # small-loop(oracle)
                    if causal and (j > i or (window > 0 and i - j >= window)):
                        ok[j] = False
                        continue
                    var s = Float64(0)
                    for d in range(hd):  # small-loop(oracle)
                        s += q[(b * T + i) * H * hd + hh * hd + d] * k[(b * T + j) * Hk * hd + kv * hd + d]
                    sc[j] = s / sqrt(Float64(hd))
                    mx = max(mx, sc[j])
                var den = Float64(0)
                for j in range(T):  # small-loop(oracle)
                    if ok[j]:
                        sc[j] = exp(sc[j] - mx)
                        den += sc[j]
                for j in range(T):  # small-loop(oracle)
                    if ok[j]:
                        for d in range(hd):  # small-loop(oracle)
                            ctx[(b * T + i) * H * hd + hh * hd + d] += sc[j] / den * v[(b * T + j) * Hk * hd + kv * hd + d]
    var o = _linear64(ctx, B * T, H * hd, p, off + D * H * hd + 2 * D * Hk * hd, D, False)
    return _linear64(o, B, T * D, p, spec.param_off[3], 2, True)


def _rms_oracle(c: _Case, spec: ModelSpec, p: List[Float32]) -> List[Float64]:
    var x = List[Float64]()
    for i in range(len(c.x)):  # small-loop(oracle)
        x.append(Float64(c.x[i]))
    var h = _linear64(x, c.rows * 3, 4, p, 0, 8, True)
    var off = spec.param_off[1]
    for r in range(c.rows * 3):  # small-loop(oracle)
        var ss = Float64(0)
        for k in range(8):  # small-loop(oracle)
            ss += h[r * 8 + k] * h[r * 8 + k]
        var rs = 1.0 / sqrt(ss / 8.0 + 1e-6)
        for k in range(8):  # small-loop(oracle)
            h[r * 8 + k] = h[r * 8 + k] * rs * Float64(p[off + k])
    return _linear64(h, c.rows, 24, p, spec.param_off[3], 2, True)


def _scan_oracle(c: _Case, spec: ModelSpec, p: List[Float32]) -> List[Float64]:
    """`linear -> selective_scan -> flatten -> linear(2)` in float64: h[t] = exp(delta A) h[t-1] + delta B u,
    y = C . h + u D, A = -exp(A_log)."""
    var L = 6
    var dim = 3
    var N = c.model.args[6]
    var W = 2 * dim + 2 * N
    var x = List[Float64]()
    for i in range(len(c.x)):  # small-loop(oracle)
        x.append(Float64(c.x[i]))
    var h = _linear64(x, c.rows * L, 4, p, 0, W, True)
    var off = spec.param_off[1]
    var y = List[Float64](length=c.rows * L * dim, fill=0)
    for b in range(c.rows):  # small-loop(oracle)
        for d in range(dim):  # small-loop(oracle)
            var st = List[Float64](length=N, fill=0)
            for t in range(L):  # small-loop(oracle)
                var r = (b * L + t) * W
                var uv = h[r + d]
                var dl = h[r + dim + d]
                var acc = Float64(0)
                for n in range(N):  # small-loop(oracle)
                    var av = -exp(Float64(p[off + d * N + n]))
                    st[n] = exp(dl * av) * st[n] + dl * h[r + 2 * dim + n] * uv
                    acc += h[r + 2 * dim + N + n] * st[n]
                y[(b * L + t) * dim + d] = acc + uv * Float64(p[off + dim * N + d])
    return _linear64(y, c.rows, L * dim, p, spec.param_off[3], 2, True)


def _ssd_oracle(c: _Case, spec: ModelSpec, p: List[Float32]) -> List[Float64]:
    """`linear -> softplus -> ssd_scan -> flatten -> linear(2)` in float64, the plain recurrence per head:
    h[t] = exp(dt A) h[t-1] + dt B[t] x[t]^T, y = C[t] . h[t] + D x, A = -exp(A_log) (no chunks)."""
    var H = c.model.args[12]
    var L = c.model.sample[0]
    var P = 64
    var N = 128
    var W = 65 * H + 256
    var x = List[Float64]()
    for i in range(len(c.x)):  # small-loop(oracle)
        x.append(Float64(c.x[i]))
    var h = _linear64(x, c.rows * L, 2, p, 0, W, True)
    for i in range(len(h)):  # small-loop(oracle)
        h[i] = log(1.0 + exp(h[i]))
    var off = spec.param_off[2]
    var y = List[Float64](length=c.rows * L * H * P, fill=0)
    for b in range(c.rows):  # small-loop(oracle)
        for hh in range(H):  # small-loop(oracle)
            var av = -exp(Float64(p[off + hh]))
            var dv = Float64(p[off + H + hh])
            for q in range(P):  # small-loop(oracle)
                var st = List[Float64](length=N, fill=0)
                for t in range(L):  # small-loop(oracle)
                    var r = (b * L + t) * W
                    var xv = h[r + hh * P + q]
                    var dt = h[r + H * P + 2 * N + hh]
                    var acc = Float64(0)
                    for n in range(N):  # small-loop(oracle)
                        st[n] = exp(dt * av) * st[n] + dt * h[r + H * P + n] * xv
                        acc += h[r + H * P + N + n] * st[n]
                    y[(b * L + t) * H * P + hh * P + q] = acc + dv * xv
    return _linear64(y, c.rows, L * H * P, p, spec.param_off[4], 2, True)
def _trap_oracle(c: _Case, spec: ModelSpec, p: List[Float32]) -> List[Float64]:
    """Mamba-3 (lane neural-blocks-3-m3): `linear -> trapezoid_scan -> flatten -> linear(2)` in float64:
    alpha = exp(A dt), sigma = sigmoid(trap), h[t] = alpha h[t-1] + dt (1 - sigma) alpha k[t-1] v[t-1]
    + dt sigma k[t] v[t], y = q . h + D v."""
    var L = 6
    var H = c.model.args[6]
    var N = c.model.args[7]
    var spec_w = c.model.args[0]
    var P = (spec_w - H * (2 * N + 3)) // H
    var W = spec_w
    var x = List[Float64]()
    for i in range(len(c.x)):  # small-loop(oracle)
        x.append(Float64(c.x[i]))
    var h = _linear64(x, c.rows * L, 4, p, 0, W, True)
    var off = spec.param_off[1]
    var HP = H * P
    var HN = H * N
    var y = List[Float64](length=c.rows * L * HP, fill=0)
    for b in range(c.rows):  # small-loop(oracle)
        for hh in range(H):  # small-loop(oracle)
            for pp in range(P):  # small-loop(oracle)
                var st = List[Float64](length=N, fill=0)
                for t in range(L):  # small-loop(oracle)
                    var r = (b * L + t) * W
                    var vv = h[r + hh * P + pp]
                    var dt = h[r + HP + 2 * HN + hh]
                    var av = h[r + HP + 2 * HN + H + hh]
                    var sg = 1.0 / (1.0 + exp(-h[r + HP + 2 * HN + 2 * H + hh]))
                    var al = exp(av * dt)
                    var be = dt * (1.0 - sg) * al
                    var ga = dt * sg
                    var acc = Float64(0)
                    for n in range(N):  # small-loop(oracle)
                        var drive = ga * h[r + HP + hh * N + n] * vv
                        if t > 0:
                            var rp = r - W
                            drive += be * h[rp + HP + hh * N + n] * h[rp + hh * P + pp]
                        st[n] = al * st[n] + drive
                        acc += h[r + HP + HN + hh * N + n] * st[n]
                    y[(b * L + t) * HP + hh * P + pp] = acc + Float64(p[off + hh]) * vv
    return _linear64(y, c.rows, L * HP, p, spec.param_off[3], 2, True)
# lm_head (lane neural-blocks-3-lm)
def _lmhead_oracle(c: _Case, spec: ModelSpec, p: List[Float32]) -> List[Float64]:
    """`linear(6) -> lm_head(11) -> flatten -> linear(2)` in float64; the head weight is (vocab, features)."""
    var x = List[Float64]()
    for i in range(len(c.x)):  # small-loop(oracle)
        x.append(Float64(c.x[i]))
    var h = _linear64(x, c.rows * 5, 4, p, 0, 6, True)
    var off = spec.param_off[1]
    var z = List[Float64](length=c.rows * 5 * 11, fill=0)
    for r in range(c.rows * 5):  # small-loop(oracle)
        for v in range(11):  # small-loop(oracle)
            var acc = Float64(0)
            for k in range(6):  # small-loop(oracle)
                acc += h[r * 6 + k] * Float64(p[off + v * 6 + k])
            z[r * 11 + v] = acc
    return _linear64(z, c.rows, 55, p, spec.param_off[3], 2, True)


def _ce64(z: List[Float64], codes: List[Float32], rows: Int, vocab: Int) -> Float64:
    """The mean token cross-entropy of float64 logits (rows, vocab)."""
    var total = Float64(0)
    for r in range(rows):  # small-loop(oracle)
        var mx = z[r * vocab]
        for v in range(vocab):  # small-loop(oracle)
            mx = max(mx, z[r * vocab + v])
        var den = Float64(0)
        for v in range(vocab):  # small-loop(oracle)
            den += exp(z[r * vocab + v] - mx)
        total += log(den) + mx - z[r * vocab + Int(codes[r])]
    return total / Float64(rows)
# end lm_head


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": length " + String(len(got)) + " expected " + String(len(want)))
    for i in range(len(got)):  # small-loop(check words)
        var d = abs(Float64(got[i]) - want[i])
        if not (d <= tol * (1.0 + abs(want[i]))):
            _fail(name + ": element " + String(i) + " got " + String(got[i]) + " expected " + String(want[i]))


def _pinned(name: String) -> UInt64:
    """This version's recorded output-and-gradient digest of a case (0: none recorded). The selective scan's
    words are the Mamba-1 block bodies' (autograd/kernels/mamba1_scan_body.mojo), so its sabotage arms
    (-D MOJOLEARN_MAMBA_SABOTAGE_S9_UNFUSED, ..._S5_EXP2, ..._BWD_S9B_UNFUSED, ...) change these words on
    every column and the check FAILS (the arm is caught). Recorded on the CPU column (cpc1, lane
    neural-blocks-2, 2026-11-22); the device column must give the same words (test_neural_blocks). Bits
    change between versions: a product change records the new digest here in the same commit."""
    if name == "selective_scan_n4":
        return UInt64(12107700610124185639)
    if name == "selective_scan_n8":
        return UInt64(3944575162351859353)
    # Mamba-3 (lane neural-blocks-3-m3): the trapezoid scan's words are the Mamba-3 trapezoid bodies'
    # (autograd/kernels/mamba3_trapezoid_body.mojo); -D MOJOLEARN_MAMBA3_SABOTAGE_TRAP_LEFT_ONLY moves them.
    # Recorded on the CPU column (cpc2, 2026-11-22).
    # Mamba-2 (lane neural-blocks-3): the SSD scan's words are the Mamba-2 block bodies'
    # (autograd/kernels/mamba2_ssd_body.mojo); -D MOJOLEARN_MAMBA2_SABOTAGE_SEGSUM_DESCENDING moves them.
    if name == "ssd_scan_h1":
        return UInt64(13223243358280786975)
    if name == "ssd_scan_h2":
        return UInt64(11601189536576961029)
    if name == "trapezoid_scan_h2":
        return UInt64(9596372206100812605)
    if name == "trapezoid_scan_h1":
        return UInt64(5371293045682281016)
    # lm_head (lane neural-blocks-3-lm): recorded on the CPU column (cpc2, 2026-11-22)
    if name == "lm_head_logits":
        return UInt64(17656890029989354755)
    return UInt64(0)


def run_case[E: Exec](mut ex: E, c: _Case, finite_differences: Bool) raises -> List[Float32]:
    var spec = parse_model(c.model.program(c.features), model_float_words(c.model.table()))
    var params = _params(spec.n_params)
    # rms_norm's weight away from zero (the interpreter's init fills it with 1; the check uses its own values)
    var grads = List[Float32]()
    var yv = _run(ex, c, spec, params, True, grads)
    if c.name.startswith("attention"):
        _close(c.name + " forward", yv, _attention_oracle(c, spec, params), 1e-4)
    elif c.name == "rms_norm":
        _close(c.name + " forward", yv, _rms_oracle(c, spec, params), 1e-4)
    elif c.name.startswith("selective_scan"):
        _close(c.name + " forward", yv, _scan_oracle(c, spec, params), 1e-4)
    elif c.name.startswith("ssd_scan"):
        _close(c.name + " forward", yv, _ssd_oracle(c, spec, params), 1e-4)
    elif c.name.startswith("trapezoid_scan"):  # Mamba-3 (lane neural-blocks-3-m3)
        _close(c.name + " forward", yv, _trap_oracle(c, spec, params), 1e-4)
    elif c.name == "lm_head_logits":
        _close(c.name + " forward", yv, _lmhead_oracle(c, spec, params), 1e-4)
    if finite_differences:
        var fd = List[Float64](length=len(params), fill=0)
        for i in range(len(params)):  # small-loop(every parameter probed)
            var up = params.copy()
            up[i] = Float32(Float64(params[i]) + _H)
            var down = params.copy()
            down[i] = Float32(Float64(params[i]) - _H)
            fd[i] = (_loss(ex, c, spec, up) - _loss(ex, c, spec, down)) / (2.0 * _H)
        _close(c.name + " gradient", grads, fd, 3e-3)
    var words = yv.copy()
    for i in range(len(grads)):  # small-loop(check words)
        words.append(grads[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NEURAL_BLOCKS", c.name, "params", spec.n_params, "digest", digest)
    var pinned = _pinned(c.name)
    if pinned != 0 and digest != pinned:
        _fail(c.name + ": digest " + String(digest) + " is not this version's recorded " + String(pinned)
              + " (a changed fold order or contraction in the layer's body: a sabotage arm, or a product change"
              + " whose new words must be recorded here)")
    return words^


def test_neural_blocks() raises:
    var cases = _cases()
    for k in range(len(cases)):  # driver-loop: over the block cases
        var host = HostExec()
        var hw = run_case(host, cases[k], True)
        comptime if not CPU_COLUMN_BUILD:
            var dev = DeviceExec()
            var dw = run_case(dev, cases[k], False)
            _same_words(cases[k].name, hw, dw)
    print("PASS NEURAL_BLOCKS blocks", len(cases))


# ---------------------------------------------------------------- lm_head token loss (lane neural-blocks-3-lm)
comptime _TOK_L = 5
comptime _TOK_F = 4
comptime _TOK_V = 11
comptime _TOK_B = 3


def _token_model(k: Int) -> SeqModel:
    """0: lm_head(11) alone on (5, 4) samples (its input is the model input); 1: linear(6) -> silu -> lm_head(11)."""
    var m = _model([_TOK_L, _TOK_F])
    if k == 1:
        m.layer(L_LINEAR, 6, 1)
        m.layer(LAYER_SILU)
    m.layer(LAYER_LM_HEAD, _TOK_V)
    return m^


def _token_codes() -> List[Float32]:
    var codes = List[Float32]()
    for i in range(_TOK_B * _TOK_L):  # small-loop(15 tokens)
        codes.append(Float32((i * 4 + 1) % _TOK_V))
    return codes^


def _token_run[E: Exec](mut ex: E, spec: ModelSpec, params: List[Float32], xv: List[Float32], grad: Bool,
                        mut gw: List[Float32], mut gx: List[Float32]) raises -> Float32:
    """The recorded token loss; with grad, its gradients of the parameters (gw) and of the input (gx)."""
    var tape = Tape()
    var holder = Tape()
    var pt = holder.tensor(ex, params, 1, len(params))
    var zeros = List[Float32](length=len(params), fill=0)
    var gt = holder.tensor(ex, zeros, 1, len(params))
    var x = tape.tensor(ex, xv, _TOK_B, _TOK_L * _TOK_F, grad)
    var codes = tape.tensor(ex, _token_codes(), _TOK_B, _TOK_L)
    var loss = model_token_loss(tape, ex, spec, holder.value(pt), holder.value(gt), x, codes, grad, UInt64(0), 0)
    var lv = tape.values(ex, loss)
    if grad:
        tape.backward_scalar_with_rules(ex, loss, ModelRules())
        gw = holder.values(ex, gt)
        gx = tape.gradients(ex, x)
    return lv[0]


def _token_logits[E: Exec](mut ex: E, spec: ModelSpec, params: List[Float32], xv: List[Float32]) raises -> List[Float32]:
    """The model's predicted logits (the whole forward, the head's projection op)."""
    var tape = Tape()
    var holder = Tape()
    var pt = holder.tensor(ex, params, 1, len(params))
    var x = tape.tensor(ex, xv, _TOK_B, _TOK_L * _TOK_F)
    var y = model_forward(tape, ex, spec, holder.value(pt), holder.value(pt), x, False, UInt64(0), 0)
    return tape.values(ex, y)


def _token_oracle(k: Int, spec: ModelSpec, p: List[Float32], xv: List[Float32]) -> Float64:
    var rows = _TOK_B * _TOK_L
    var h = List[Float64]()
    for i in range(len(xv)):  # small-loop(oracle)
        h.append(Float64(xv[i]))
    var width = _TOK_F
    if k == 1:
        h = _linear64(h, rows, _TOK_F, p, 0, 6, True)
        for i in range(len(h)):  # small-loop(oracle)
            h[i] = h[i] / (1.0 + exp(-h[i]))
        width = 6
    var off = spec.param_off[spec.n_layers() - 1]
    var z = List[Float64](length=rows * _TOK_V, fill=0)
    for r in range(rows):  # small-loop(oracle)
        for v in range(_TOK_V):  # small-loop(oracle)
            var acc = Float64(0)
            for j in range(width):  # small-loop(oracle)
                acc += h[r * width + j] * Float64(p[off + v * width + j])
            z[r * _TOK_V + v] = acc
    return _ce64(z, _token_codes(), rows, _TOK_V)


def _token_pinned(k: Int) -> UInt64:
    """This version's recorded token-loss digest (loss, parameter and input gradients) of model k (0: none
    recorded). The denominators fold in the lm_head op's leaf body (training/chunked_lm_head_panel_math.mojo),
    so its sabotage arm changes these words on every column. Bits change between versions: a product change
    records the new digest here in the same commit. Recorded on the CPU column (cpc2, 2026-11-22)."""
    if k == 0:
        return UInt64(11108790232435158226)
    if k == 1:
        return UInt64(12471558340238927144)
    return UInt64(0)


def run_token_case[E: Exec](mut ex: E, k: Int, finite_differences: Bool) raises -> List[Float32]:
    var name = "lm_head_token_loss_" + String(k)
    var m = _token_model(k)
    var spec = parse_model(m.program(_TOK_L * _TOK_F), model_float_words(m.table()))
    var params = _params(spec.n_params)
    var xv = _vals(_TOK_B * _TOK_L * _TOK_F, 7, 31, 16)
    var gw = List[Float32]()
    var gx = List[Float32]()
    var lv = _token_run(ex, spec, params, xv, True, gw, gx)
    var want = _token_oracle(k, spec, params, xv)
    if not (abs(Float64(lv) - want) <= 1e-5 * (1.0 + abs(want))):
        _fail(name + ": loss " + String(lv) + " expected " + String(want))
    # the fused loss against the cross-entropy of the model's own predicted logits (one GEMM cell, two paths)
    var zl = _token_logits(ex, spec, params, xv)
    var z = List[Float64]()
    for i in range(len(zl)):  # small-loop(check words)
        z.append(Float64(zl[i]))
    var from_logits = _ce64(z, _token_codes(), _TOK_B * _TOK_L, _TOK_V)
    if not (abs(Float64(lv) - from_logits) <= 1e-5 * (1.0 + abs(from_logits))):
        _fail(name + ": fused loss " + String(lv) + " differs from the predicted logits' " + String(from_logits))
    if finite_differences:
        var none_w = List[Float32]()
        var none_x = List[Float32]()
        var fdw = List[Float64](length=len(params), fill=0)
        for i in range(len(params)):  # small-loop(every parameter probed)
            var up = params.copy()
            up[i] = Float32(Float64(params[i]) + _H)
            var down = params.copy()
            down[i] = Float32(Float64(params[i]) - _H)
            fdw[i] = (Float64(_token_run(ex, spec, up, xv, False, none_w, none_x))
                      - Float64(_token_run(ex, spec, down, xv, False, none_w, none_x))) / (2.0 * _H)
        _close(name + " parameter gradient", gw, fdw, 3e-3)
        var fdx = List[Float64](length=len(xv), fill=0)
        for i in range(len(xv)):  # small-loop(every input probed)
            var up = xv.copy()
            up[i] = Float32(Float64(xv[i]) + _H)
            var down = xv.copy()
            down[i] = Float32(Float64(xv[i]) - _H)
            fdx[i] = (Float64(_token_run(ex, spec, params, up, False, none_w, none_x))
                      - Float64(_token_run(ex, spec, params, down, False, none_w, none_x))) / (2.0 * _H)
        _close(name + " input gradient", gx, fdx, 3e-3)
    var words = List[Float32]()
    words.append(lv)
    for i in range(len(gw)):  # small-loop(check words)
        words.append(gw[i])
    for i in range(len(gx)):  # small-loop(check words)
        words.append(gx[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NEURAL_BLOCKS", name, "params", spec.n_params, "loss", lv, "digest", digest)
    var pinned = _token_pinned(k)
    if pinned != 0 and digest != pinned:
        _fail(name + ": digest " + String(digest) + " is not this version's recorded " + String(pinned)
              + " (a changed fold order in the lm_head op's bodies: a sabotage arm, or a product change whose new"
              + " words must be recorded here)")
    return words^


def test_lm_head_token_loss() raises:
    for k in range(2):  # driver-loop: over the two token models
        var host = HostExec()
        var hw = run_token_case(host, k, True)
        comptime if not CPU_COLUMN_BUILD:
            var dev = DeviceExec()
            var dw = run_token_case(dev, k, False)
            _same_words("lm_head_token_loss_" + String(k), hw, dw)
    print("PASS NEURAL_BLOCKS lm_head token loss 2")
# end lm_head


def test_embedding_seeded_probe() raises:
    """The seeded probe of a model whose first layer reads ids (lane nn-fix-3, 2026-11-23: the save's verify
    report fed floats in [-1, 1) to an embedding, whose ids came out -1, so an embedding model could not be
    saved): model_input_id_range names the embedding's num_embeddings (0 for a model that reads values),
    verify_fill writes seeded ids in [0, n) as float32, the same words on every call, and the embedding
    model's forward runs on them."""
    var e = _model([5])
    e.layer(LAYER_EMBEDDING, 7, 4)
    _head(e)
    var spec = e.spec(5)
    var n = model_input_id_range(spec)
    if n != 7:
        _fail("embedding probe: the id range is " + String(n) + ", not num_embeddings 7")
    var v = _model([3, 4])
    v.layer(L_LINEAR, 6, 1)
    _head(v)
    if model_input_id_range(v.spec(12)) != 0:
        _fail("embedding probe: a model that reads values has an id range")
    comptime ROWS = 16
    var ids = List[Float32](length=ROWS * 5, fill=-1)
    var again = List[Float32](length=ROWS * 5, fill=-1)
    _ = verify_fill(Int(ids.unsafe_ptr()), len(ids), 0, 0, n)
    _ = verify_fill(Int(again.unsafe_ptr()), len(again), 0, 0, n)
    var seen = List[Bool](length=n, fill=False)
    for i in range(len(ids)):  # small-loop(the probe's values)
        var x = ids[i]
        if not (x >= 0 and x < Float32(n)) or Float32(Int(x)) != x:
            _fail("embedding probe: value " + String(i) + " is " + String(x) + ", not an id in [0, 7)")
        if bitcast[DType.uint32](x) != bitcast[DType.uint32](again[i]):
            _fail("embedding probe: value " + String(i) + " differs between two fills")
        seen[Int(x)] = True
    var distinct = 0
    for k in range(n):  # small-loop(n: ids)
        if seen[k]:
            distinct += 1
    if distinct < 4:
        _fail("embedding probe: only " + String(distinct) + " of 7 ids drawn in 80 values")
    var c = _Case("embedding_probe", e^, ROWS, 5, ids.copy())
    var params = _params(spec.n_params)
    var grads = List[Float32]()
    var host = HostExec()
    var y = _run(host, c, spec, params, False, grads)
    for i in range(len(y)):  # small-loop(the probe's outputs)
        if not (y[i] == y[i]):
            _fail("embedding probe: output " + String(i) + " is NaN")
    _ = ids^
    _ = again^
    print("PASS NEURAL_BLOCKS embedding seeded probe", distinct, "ids", len(y), "outputs")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
