# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The activation layers of Sequential and its identity layer (lane nn-activations, 2026-11-23).

Layers 47..53 of autograd/model_blocks.mojo (autograd/nn_activations.mojo: identity, leaky_relu, elu, prelu shared
and per channel, mish, hardswish, hardsigmoid), whose element bodies are checks/numerics.mojo's, are checked
through the model interpreter (autograd/model.mojo model_forward, the version-2 backward with ModelRules) in the
model `activation -> flatten -> linear(2)` on (3, 4) samples:
  - the forward against a float64 oracle written here (PyTorch's formulas: relu6(x + 3) / 6 for the hard forms,
    log(1 + exp(x)) inside mish), tolerance 1e-5 (1 + |y|);
  - the gradient of every parameter (the head, PReLU's slope) AND of every input against central differences of
    the float64 oracle's loss sum(seed * y), h = 1/128, tolerance 1e-4 (1 + |fd|). The inputs are odd multiples
    of 1/16, so no input lies within h of a kink (0, and -3 / 3 of the hard forms);
  - identity: its words (outputs and every gradient) are the words of the same model without it, bit for bit;
  - one `NN_ACTIVATIONS <case> digest` line over the output and gradient words, pinned below.
Fits: each layer inside `linear(16) -> act -> linear(16) -> act -> linear(2)` (identity: `relu -> identity` in the
second slot; prelu: per feature, then shared) learns a disk (x0^2 + x1^2 < 1.4 on standard-normal points) with
Adam: the loss falls, the loss curve, parameters and probabilities are finite, and the held-out accuracy reaches
0.9.
Fault injection: `-D MOJOLEARN_NN_ACT_SABOTAGE` (checks/numerics.mojo) must make this check FAIL.
Columns: the CPU column runs every case and fit on HostExec, spread across host cores. The device column runs each
on DeviceExec AND HostExec and requires the same words: the host-column-versus-device bytes check.
"""
from std.math import exp, log, tanh
from std.memory import bitcast
from std.testing import TestSuite
from autograd.tape import Tape, Tensor
from autograd.model import ModelSpec, parse_model, model_forward, LOSS_CROSS_ENTROPY, TARGET_CODES, OUT_ARGMAX
from autograd.model import OUT_SOFTMAX
from autograd.model_blocks import ModelRules
from autograd.model_execution import model_float_words
from autograd.nn_activations import (
    LAYER_IDENTITY, LAYER_LEAKY_RELU, LAYER_ELU, LAYER_PRELU, LAYER_MISH, LAYER_HARDSWISH, LAYER_HARDSIGMOID,
)
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.core_ops_common import normal
from checks.native_tests.training_sequential_common import SeqModel, codes_as_words, L_LINEAR, L_FLATTEN, L_RELU
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize

comptime _H = Float64(1.0) / Float64(128)
comptime _ROWS = 3
comptime _C = 3
comptime _L = 4
comptime _W = _C * _L


def _fail(message: String) raises:
    print("FAIL NN_ACTIVATIONS", message)
    raise Error(message)


def _mix(mut digest: UInt64, values: List[Float32]):
    for i in range(len(values)):  # small-loop(check words)
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def _same_words(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": word counts differ (" + String(len(a)) + " vs " + String(len(b)) + ")")
    for i in range(len(a)):  # small-loop(check words)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _vals(n: Int, step: Int, modulus: Int, denom: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(n: check data)
        v[i] = Float32((i * step) % modulus - modulus // 2) / Float32(denom)
    return v^


def _inputs() -> List[Float32]:
    """Odd multiples of 1/16 in [-4.4, 4.4]: on both sides of every kink, never within 1/16 of one."""
    var v = List[Float32](length=_ROWS * _W, fill=0)
    for i in range(_ROWS * _W):  # small-loop(check data)
        v[i] = Float32(2 * ((i * 29) % 71 - 35) + 1) / Float32(16)
    return v^


struct _Case(Copyable, Movable):
    var name: String
    var code: Int
    var a0: Int
    var a1: Int
    var f: Float64

    def __init__(out self, name: String, code: Int, a0: Int = 0, a1: Int = 0, f: Float64 = 0.0):
        self.name = name
        self.code = code
        self.a0 = a0
        self.a1 = a1
        self.f = f

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.code = copy.code
        self.a0 = copy.a0
        self.a1 = copy.a1
        self.f = copy.f


def _bits(v: Float32) -> Int:
    return Int(bitcast[DType.int32](v))


def _cases() -> List[_Case]:
    var out = List[_Case]()
    out.append(_Case("identity", LAYER_IDENTITY))
    out.append(_Case("leaky_relu", LAYER_LEAKY_RELU, f=0.1))
    out.append(_Case("elu", LAYER_ELU, f=1.3))
    out.append(_Case("prelu_shared", LAYER_PRELU, 1, _bits(Float32(0.25))))
    out.append(_Case("prelu_channels", LAYER_PRELU, _C, _bits(Float32(0.1))))
    out.append(_Case("mish", LAYER_MISH))
    out.append(_Case("hardswish", LAYER_HARDSWISH))
    out.append(_Case("hardsigmoid", LAYER_HARDSIGMOID))
    return out^


def _model(c: _Case, with_activation: Bool) -> SeqModel:
    var m = SeqModel()
    m.sample = [_C, _L]
    if with_activation:
        m.layer(c.code, c.a0, c.a1, f=c.f)
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)
    return m^


def _params(n: Int) -> List[Float32]:
    # multiples of 1/64 in [-0.36, 0.36]
    return _vals(n, 17, 47, 64)


def _run[E: Exec](mut ex: E, spec: ModelSpec, params: List[Float32], xv: List[Float32],
                  mut gw: List[Float32], mut gx: List[Float32]) raises -> List[Float32]:
    """The outputs; the gradients of sum(seed * y) for the parameters (gw) and the inputs (gx)."""
    var tape = Tape()
    var holder = Tape()
    var pt = holder.tensor(ex, params, 1, len(params))
    var zeros = List[Float32](length=len(params), fill=0)
    var gt = holder.tensor(ex, zeros, 1, len(params))
    var x = tape.tensor(ex, xv, _ROWS, _W, True)
    var y = model_forward(tape, ex, spec, holder.value(pt), holder.value(gt), x, True, UInt64(0), 0)
    var yv = tape.values(ex, y)
    var seed = _vals(len(yv), 3, 11, 16)
    var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
    tape.backward_with_rules(ex, y, st, ModelRules())
    gw = holder.values(ex, gt)
    gx = tape.gradients(ex, x)
    return yv^


# ---------------------------------------------------------------- float64 oracle
def _act64(c: _Case, x: Float64, channel: Int, p: List[Float64]) -> Float64:
    if c.code == LAYER_LEAKY_RELU:
        return x if x >= 0 else c.f * x
    if c.code == LAYER_ELU:
        return x if x >= 0 else c.f * (exp(x) - 1.0)
    if c.code == LAYER_PRELU:
        var w = p[0] if c.a0 == 1 else p[channel]
        return x if x >= 0 else w * x
    if c.code == LAYER_MISH:
        return x * tanh(log(1.0 + exp(x)))
    var r6 = min(max(x + 3.0, 0.0), 6.0) / 6.0
    if c.code == LAYER_HARDSWISH:
        return x * r6
    if c.code == LAYER_HARDSIGMOID:
        return r6
    return x  # identity


def _oracle(c: _Case, spec: ModelSpec, p: List[Float64], x: List[Float64]) -> List[Float64]:
    var off = spec.param_off[spec.n_layers() - 1]
    var slope = List[Float64]()
    if c.code == LAYER_PRELU:
        for k in range(c.a0):  # small-loop(oracle)
            slope.append(p[spec.param_off[0] + k])
    var y = List[Float64](length=_ROWS * 2, fill=0)
    for r in range(_ROWS):  # small-loop(oracle)
        for o in range(2):  # small-loop(oracle)
            var s = p[off + _W * 2 + o]
            for k in range(_W):  # small-loop(oracle)
                s += _act64(c, x[r * _W + k], k // _L, slope) * p[off + k * 2 + o]
            y[r * 2 + o] = s
    return y^


def _loss64(c: _Case, spec: ModelSpec, p: List[Float64], x: List[Float64]) -> Float64:
    var y = _oracle(c, spec, p, x)
    var seed = _vals(len(y), 3, 11, 16)
    var s = Float64(0)
    for i in range(len(y)):  # small-loop(oracle)
        s += Float64(seed[i]) * y[i]
    return s


def _f64(v: List[Float32]) -> List[Float64]:
    var out = List[Float64](length=len(v), fill=0)
    for i in range(len(v)):  # small-loop(oracle)
        out[i] = Float64(v[i])
    return out^


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": length " + String(len(got)) + " expected " + String(len(want)))
    for i in range(len(got)):  # small-loop(check words)
        var d = abs(Float64(got[i]) - want[i])
        if not (d <= tol * (1.0 + abs(want[i]))):
            _fail(name + ": element " + String(i) + " got " + String(got[i]) + " expected " + String(want[i]))


def _pinned(name: String) -> UInt64:
    """This version's recorded output-and-gradient digest of a case (0: none recorded). Recorded on the CPU
    column; the device column must give the same words. Bits change between versions: a product change records
    the new digest here in the same commit. Recorded on the CPU column (cpc1, lane nn-activations, 2026-11-23)."""
    if name == "identity":
        return UInt64(14541554893642753075)
    if name == "leaky_relu":
        return UInt64(1003246152945393296)
    if name == "elu":
        return UInt64(16297280965698229233)
    if name == "prelu_shared":
        return UInt64(5928345840216340137)
    if name == "prelu_channels":
        return UInt64(7705402478610619409)
    if name == "mish":
        return UInt64(2956685032833292905)
    if name == "hardswish":
        return UInt64(1359718371854466513)
    if name == "hardsigmoid":
        return UInt64(10447624012318339352)
    return UInt64(0)


def run_case[E: Exec](mut ex: E, c: _Case, oracle: Bool) raises -> List[Float32]:
    var m = _model(c, True)
    var spec = parse_model(m.program(_W), model_float_words(m.table()))
    var params = _params(spec.n_params)
    if c.code == LAYER_PRELU:
        # the slope's init is the layer's (0.25 / 0.1); the check uses its own values, one per channel
        for k in range(c.a0):  # small-loop(at most 3 slopes)
            params[spec.param_off[0] + k] = Float32(k + 1) / Float32(8)
    var xv = _inputs()
    var gw = List[Float32]()
    var gx = List[Float32]()
    var yv = _run(ex, spec, params, xv, gw, gx)
    if oracle:
        var p64 = _f64(params)
        var x64 = _f64(xv)
        _close(c.name + " forward", yv, _oracle(c, spec, p64, x64), 1e-5)
        var fdw = List[Float64](length=len(params), fill=0)
        for i in range(len(params)):  # small-loop(every parameter probed)
            var up = p64.copy()
            up[i] += _H
            var down = p64.copy()
            down[i] -= _H
            fdw[i] = (_loss64(c, spec, up, x64) - _loss64(c, spec, down, x64)) / (2.0 * _H)
        _close(c.name + " parameter gradient", gw, fdw, 1e-4)
        var fdx = List[Float64](length=len(xv), fill=0)
        for i in range(len(xv)):  # small-loop(every input probed)
            var up = x64.copy()
            up[i] += _H
            var down = x64.copy()
            down[i] -= _H
            fdx[i] = (_loss64(c, spec, p64, up) - _loss64(c, spec, p64, down)) / (2.0 * _H)
        _close(c.name + " input gradient", gx, fdx, 1e-4)
        if c.code == LAYER_IDENTITY:
            # identity changes nothing: the same words as the model without it (same parameters)
            var bare = _model(c, False)
            var bspec = parse_model(bare.program(_W), model_float_words(bare.table()))
            var bw = List[Float32]()
            var bx = List[Float32]()
            var by = _run(ex, bspec, params, xv, bw, bx)
            _same_words("identity outputs vs no layer", yv, by)
            _same_words("identity parameter gradient vs no layer", gw, bw)
            _same_words("identity input gradient vs no layer", gx, bx)
    var words = yv.copy()
    for i in range(len(gw)):  # small-loop(check words)
        words.append(gw[i])
    for i in range(len(gx)):  # small-loop(check words)
        words.append(gx[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_ACTIVATIONS", c.name, "params", spec.n_params, "digest", digest)
    var pinned = _pinned(c.name)
    if pinned != 0 and digest != pinned:
        _fail(c.name + ": digest " + String(digest) + " is not this version's recorded " + String(pinned)
              + " (a changed activation body: a sabotage arm, or a product change whose new words must be recorded"
              + " here)")
    return words^


# ---------------------------------------------------------------- fits
comptime _TRAIN = 512
comptime _TEST = 256
comptime _EPOCHS = 40


struct _Disk(Movable):
    var x: List[Float32]
    var codes: List[Int]
    var y: List[Float32]

    def __init__(out self, rows: Int, salt: Int):
        """Standard-normal points in the plane; class 1 inside the disk x0^2 + x1^2 < 1.4 (about half)."""
        self.x = normal(rows * 2, salt)
        self.codes = List[Int](length=rows, fill=0)
        for r in range(rows):  # small-loop(fixture data)
            var a = Float64(self.x[2 * r])
            var b = Float64(self.x[2 * r + 1])
            self.codes[r] = 1 if a * a + b * b < 1.4 else 0
        self.y = codes_as_words(self.codes)


def _fit_model(c: _Case) -> SeqModel:
    var m = SeqModel()
    m.sample = [2]
    m.layer(L_LINEAR, 16, 1)
    if c.code == LAYER_IDENTITY:
        m.layer(L_RELU)
    elif c.code == LAYER_PRELU:
        m.layer(LAYER_PRELU, 16, c.a1)
    else:
        m.layer(c.code, c.a0, c.a1, f=c.f)
    m.layer(L_LINEAR, 16, 1)
    m.layer(c.code, 1 if c.code == LAYER_PRELU else c.a0, c.a1, f=c.f)
    m.layer(L_LINEAR, 2, 1)
    return m^


def _finite(name: String, what: String, v: List[Float32]) raises:
    for i in range(len(v)):  # small-loop(check words)
        var w = bitcast[DType.uint32](v[i]) & UInt32(0x7F800000)
        if w == UInt32(0x7F800000):
            _fail(name + " fit: " + what + " " + String(i) + " is not finite (" + String(v[i]) + ")")


def run_fit[E: Exec](mut ex: E, c: _Case) raises -> List[Float32]:
    """Fit the disk with case c's layer; returns the fitted parameters, then the held-out probabilities."""
    var train = _Disk(_TRAIN, 21)
    var test = _Disk(_TEST, 22)
    var m = _fit_model(c)
    m.compile(LOSS_CROSS_ENTROPY, "adam", 0.03)
    _ = m.fit(ex, train.x, _TRAIN, 2, train.y, 1, TARGET_CODES, _EPOCHS, 32, -1, 5)
    var curve = m.losses()
    _finite(c.name, "loss of epoch", curve)
    _finite(c.name, "parameter", m.params)
    if not (curve[len(curve) - 1] < curve[0]):
        _fail(c.name + " fit: the loss did not fall (" + String(curve[0]) + " -> " + String(curve[len(curve) - 1])
              + ")")
    var proba = m.predict(ex, test.x, _TEST, 2, OUT_SOFTMAX)
    _finite(c.name, "probability", proba)
    var pred = m.predict(ex, test.x, _TEST, 2, OUT_ARGMAX)
    var right = 0
    for i in range(_TEST):  # small-loop(check accuracy)
        if Int(pred[i]) == test.codes[i]:
            right += 1
    var acc = Float64(right) / Float64(_TEST)
    print("NN_ACTIVATIONS fit", c.name, "loss", curve[0], "->", curve[len(curve) - 1], "held-out accuracy", acc)
    if acc < 0.9:
        _fail(c.name + " fit: held-out accuracy " + String(acc) + " is below 0.9")
    var words = m.params.copy()
    for i in range(len(proba)):  # small-loop(predictions)
        words.append(proba[i])
    _ = train^
    _ = test^
    return words^


# ---------------------------------------------------------------- tests
def test_nn_activations() raises:
    # Independent cases: the host column spreads them across host cores (one HostExec each).
    var cases = _cases()
    var n = len(cases)
    var failed = List[Int](length=n, fill=0)
    var host_words = List[List[Float32]](length=n, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    var cp = cases.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp, imm cp}:
        try:
            var host = HostExec()
            wp[unit] = run_case(host, cp[unit], True)
        except error:
            print("FAIL NN_ACTIVATIONS case", unit, error)
            fp[unit] = 1
    host_parallelize(task, n)
    for u in range(n):  # small-loop(n: failure flags)
        if failed[u] != 0:
            _fail("case " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for k in range(n):  # driver-loop: over the cases on the one device
            var dev = DeviceExec()
            var dw = run_case(dev, cases[k], False)
            _same_words(cases[k].name, host_words[k], dw)
    _ = failed^
    _ = host_words^
    _ = cases^
    print("PASS NN_ACTIVATIONS cases", n)


def test_nn_activation_fits() raises:
    # one fit per layer (the per-channel prelu case stands for prelu), across host cores
    var every = _cases()
    var fits = List[_Case]()
    for k in range(len(every)):  # small-loop(8 cases)
        if every[k].name != "prelu_shared":
            fits.append(every[k].copy())
    var n = len(fits)
    var failed = List[Int](length=n, fill=0)
    var host_words = List[List[Float32]](length=n, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    var cp = fits.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp, imm cp}:
        try:
            var host = HostExec()
            wp[unit] = run_fit(host, cp[unit])
        except error:
            print("FAIL NN_ACTIVATIONS fit", unit, error)
            fp[unit] = 1
    host_parallelize(task, n)
    for u in range(n):  # small-loop(n: failure flags)
        if failed[u] != 0:
            _fail("fit " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for k in range(n):  # driver-loop: over the fits on the one device
            var dev = DeviceExec()
            var dw = run_fit(dev, fits[k])
            _same_words(fits[k].name + " fit", host_words[k], dw)
    _ = failed^
    _ = host_words^
    _ = fits^
    _ = every^
    print("PASS NN_ACTIVATIONS fits", n)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
