# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The neural building blocks of Sequential (lane neural-blocks, 2026-11-22).

Each block layer of autograd/model_blocks.mojo (silu, gelu erf and tanh, softplus, rms_norm, rotary,
multiply / the gated container, narrow after a padded depthwise conv1d (Mamba's causal conv), embedding,
attention: multi-head, grouped-query, rotary, causal, sliding window and non-causal; rnn, lstm, gru) is
checked through the model interpreter (autograd/model.mojo model_forward, the version-2 backward with
ModelRules), inside a small model `linear -> block -> flatten -> linear`:
  - every parameter gradient against central finite differences of the interpreter's own forward loss
    sum(seed * y) (h = 1/128, tolerance 3e-3 (1 + |fd|));
  - the attention forward against a float64 oracle written here (projections, rotary, mask, softmax);
  - the rms_norm forward against a float64 oracle;
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
    LAYER_NARROW, LAYER_EMBEDDING, LAYER_ATTENTION, LAYER_RECURRENT,
)
from autograd.model_execution import model_float_words
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.training_sequential_common import SeqModel, L_LINEAR, L_FLATTEN, L_PERMUTE
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD

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


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": length " + String(len(got)) + " expected " + String(len(want)))
    for i in range(len(got)):  # small-loop(check words)
        var d = abs(Float64(got[i]) - want[i])
        if not (d <= tol * (1.0 + abs(want[i]))):
            _fail(name + ": element " + String(i) + " got " + String(got[i]) + " expected " + String(want[i]))


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
