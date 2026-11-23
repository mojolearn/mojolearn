# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Group and instance normalization (lane nn-norms, 2026-11-23; autograd/nn_norms.mojo, the Sequential layers
autograd/model_blocks.mojo LAYER_GROUP_NORM = 54 and LAYER_INSTANCE_NORM = 55).

  - op cases (graph_group_norm, graph_instance_norm on the tape): the forward against a float64 oracle written
    here elementwise (PyTorch's group_norm / instance_norm formulas, no product helper), the running update
    included; every gradient (x, weight, bias) against central differences of the oracle's loss sum(seed * y),
    every element probed. Group norm with 1 < G < C (1-D and 2-D), G == C (the instance path), G == 1 without
    affine on (C,) samples; instance norm 1-D affine, 2-D affine with tracked statistics (training: the update),
    1-D tracked statistics in eval (the running statistics normalize).
  - the state is not the optimizer's: an instance norm (affine, tracked) fitted with AdamW (weight decay 0.5) for
    two one-batch epochs holds PyTorch's running statistics, its weight trained, its predictions the eval forward
    of the fitted words; a checkpoint of it loads back word for word (the names of the optional tensors).
  - a ResNet-1D with group norm after every convolution, and a 1-D CNN with instance norm, fitted on a separable
    synthetic two-class sequence task: the loss falls, every loss, parameter and prediction is finite, held-out
    accuracy at least 0.9.
Columns: the CPU column runs every unit on HostExec across host cores; the device column runs each unit on
DeviceExec as well and requires the host column's words. Sabotage arms (test defines, autograd/nn_norms.mojo):
MOJOLEARN_NN_NORMS_SABOTAGE_GROUP (each channel normalized alone instead of its group) and
MOJOLEARN_NN_NORMS_SABOTAGE_RUNNING (the running update keeps the first sample's instead of the batch average).
"""
from std.math import sin, sqrt, isfinite
from std.memory import bitcast
from std.testing import TestSuite
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.nn_norms import graph_group_norm, graph_instance_norm
from autograd.model_blocks import ModelRules
from autograd.model import LOSS_CROSS_ENTROPY, TARGET_CODES, OUT_ARGMAX, OUT_RAW
from sequence.exec_trait import Exec
from sequence.ops import FP
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.core_ops_common import normal
from checks.native_tests.training_sequential_common import SeqModel, codes_as_words, L_RELU, L_RESIDUAL, L_FLATTEN, L_LINEAR
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize


comptime L_CONV1D = 15
comptime L_ADAPTIVE_AVGPOOL1D = 18
comptime L_BRANCH = 19
comptime L_GROUP_NORM = 54
comptime L_INSTANCE_NORM = 55

comptime _EPS = Float64(1e-5)
comptime _MOM = Float64(0.25)


def _fail(message: String) raises:
    print("FAIL NN_NORMS", message)
    raise Error(message)


def _sh(dims: List[Int]) -> ShapeN:
    var d = Array[Int, RN_MAX_RANK](fill=1)
    for k in range(len(dims)):  # small-loop(len(dims): a tensor rank)
        d[k] = dims[k]
    return ShapeN(len(dims), d)


def _dyadic(n: Int, step: Int, modulus: Int, denom: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(n: check data)
        v[i] = Float32((i * step) % modulus - modulus // 2) / Float32(denom)
    return v^


def _mix(mut digest: UInt64, values: List[Float32]):
    for i in range(len(values)):  # small-loop(len(values): check digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": " + String(len(got)) + " values, " + String(len(want)) + " expected")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        var d = abs(Float64(got[i]) - want[i])
        if not isfinite(got[i]) or d > tol * (1.0 + abs(want[i])):
            _fail(name + ": index " + String(i) + " got " + String(got[i]) + " want " + String(want[i]))


def _same_words(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": host and device lengths differ")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": host and device words differ at " + String(i) + " (" + String(a[i]) + " vs "
                  + String(b[i]) + ")")


def _finite(name: String, v: List[Float32]) raises:
    for i in range(len(v)):  # small-loop(len(v): check finiteness)
        if not isfinite(v[i]):
            _fail(name + ": index " + String(i) + " is not finite (" + String(v[i]) + ")")


def _seed(count: Int) -> List[Float32]:
    return _dyadic(count, 3, 11, 16)


# ---------------------------------------------------------------- op cases
# kind 0: group norm (groups G); kind 1: instance norm (G = C) with `track` and `train`
struct _NormCase(Copyable, Movable):
    var name: String
    var kind: Int
    var n: Int
    var c: Int
    var groups: Int
    var dims: List[Int]  # the per-sample axes after the channel
    var affine: Bool
    var track: Bool
    var train: Bool

    def __init__(out self, name: String, kind: Int, n: Int, c: Int, groups: Int, dims: List[Int], affine: Bool,
                 track: Bool, train: Bool):
        self.name = name
        self.kind = kind
        self.n = n
        self.c = c
        self.groups = groups
        self.dims = dims.copy()
        self.affine = affine
        self.track = track
        self.train = train

    def rest(self) -> Int:
        var s = 1
        for k in range(len(self.dims)):  # small-loop(at most 2 axes)
            s *= self.dims[k]
        return s


def _cases() -> List[_NormCase]:
    var out = List[_NormCase]()
    out.append(_NormCase("group_norm 1d G2 of C4", 0, 3, 4, 2, [5], True, False, True))
    out.append(_NormCase("group_norm 2d G3 of C6", 0, 2, 6, 3, [3, 2], True, False, True))
    out.append(_NormCase("group_norm G == C", 0, 2, 3, 3, [4], True, False, True))
    out.append(_NormCase("group_norm G1 (C,) no affine", 0, 4, 5, 1, List[Int](), False, False, True))
    out.append(_NormCase("instance_norm1d affine", 1, 3, 2, 2, [6], True, False, True))
    out.append(_NormCase("instance_norm2d affine tracked train", 1, 2, 3, 3, [2, 3], True, True, True))
    out.append(_NormCase("instance_norm1d tracked eval", 1, 3, 2, 2, [4], False, True, False))
    return out^


def _vals(c: _NormCase) -> List[List[Float32]]:
    """x, weight, bias, running [mean C | var C]."""
    var out = List[List[Float32]]()
    out.append(_dyadic(c.n * c.c * c.rest(), 5, 17, 16))
    var g = List[Float32](length=c.c, fill=0)
    var b = List[Float32](length=c.c, fill=0)
    var r = List[Float32](length=2 * c.c, fill=0)
    for k in range(c.c):  # small-loop(channels)
        g[k] = Float32(0.75) + Float32(k) / Float32(4)
        b[k] = Float32(k) / Float32(8) - Float32(0.125)
        r[k] = Float32(k) / Float32(16) - Float32(0.0625)
        r[c.c + k] = Float32(0.5) + Float32(k) / Float32(4)
    out.append(g^)
    out.append(b^)
    out.append(r^)
    return out^


def _oracle(c: _NormCase, x: List[Float64], w: List[Float64], bb: List[Float64], run: List[Float32],
            mut running: List[Float64]) -> List[Float64]:
    """The forward in float64 (PyTorch's formulas) and the running statistics after the call."""
    var N = c.n
    var C = c.c
    var R = c.rest()
    var G = c.groups if c.kind == 0 else C
    var cg = C // G
    var L = cg * R
    var out = List[Float64](length=N * C * R, fill=0)
    running = List[Float64](length=2 * C, fill=0)
    for k in range(2 * C):  # small-loop(running words)
        running[k] = Float64(run[k])
    var eval_running = c.kind == 1 and c.track and not c.train
    var sum_mean = List[Float64](length=C, fill=0)
    var sum_unb = List[Float64](length=C, fill=0)
    for n in range(N):  # small-loop(check oracle)
        for g in range(G):
            var base = (n * C + g * cg) * R
            var mean = Float64(0)
            var var_b = Float64(0)
            if eval_running:
                mean = Float64(run[g])
                var_b = Float64(run[C + g])
            else:
                for k in range(L):
                    mean += x[base + k]
                mean /= Float64(L)
                for k in range(L):
                    var d = x[base + k] - mean
                    var_b += d * d
                var_b /= Float64(L)
                if c.kind == 1:
                    sum_mean[g] += mean
                    sum_unb[g] += var_b * Float64(L) / Float64(L - 1)
            var inv = 1.0 / sqrt(var_b + _EPS)
            for k in range(L):
                var ch = g * cg + k // R
                var v = (x[base + k] - mean) * inv
                if c.affine:
                    v = v * w[ch] + bb[ch]
                out[base + k] = v
    if c.kind == 1 and c.track and c.train:
        for ch in range(C):  # small-loop(channels)
            running[ch] = (1.0 - _MOM) * Float64(run[ch]) + _MOM * sum_mean[ch] / Float64(N)
            running[C + ch] = (1.0 - _MOM) * Float64(run[C + ch]) + _MOM * sum_unb[ch] / Float64(N)
    return out^


def _as64(v: List[Float32]) -> List[Float64]:
    var o = List[Float64](length=len(v), fill=0)
    for i in range(len(v)):  # small-loop(len(v))
        o[i] = Float64(v[i])
    return o^


def _oracle_loss(c: _NormCase, vals: List[List[Float32]], seed: List[Float32], which: Int, i: Int,
                 h: Float64) -> Float64:
    """sum(seed * y) of the float64 forward with input `which`'s element i moved by h."""
    var ins = List[List[Float64]]()
    for a in range(3):  # small-loop(3 inputs)
        ins.append(_as64(vals[a]))
    ins[which][i] += h
    var running = List[Float64]()
    var y = _oracle(c, ins[0], ins[1], ins[2], vals[3], running)
    var total = Float64(0)
    for k in range(len(y)):  # small-loop(check loss)
        total += Float64(seed[k]) * y[k]
    return total


def run_norm_case[E: Exec](mut ex: E, c: _NormCase) raises -> List[Float32]:
    """Forward, running statistics and every gradient checked; returns the words compared."""
    var vals = _vals(c)
    var tape = Tape()
    var xd = List[Int]()
    xd.append(c.n)
    xd.append(c.c)
    for k in range(len(c.dims)):  # small-loop(at most 2 axes)
        xd.append(c.dims[k])
    var x = tape.tensor_nd(ex, vals[0], _sh(xd), True)
    var w = tape.tensor(ex, vals[1], 1, c.c, True)
    var b = tape.tensor(ex, vals[2], 1, c.c, True)
    var run = tape.tensor(ex, vals[3], 1, 2 * c.c, False)
    var wo: Optional[Tensor] = None
    var bo: Optional[Tensor] = None
    if c.affine:
        wo = w
        bo = b
    var y: Tensor
    if c.kind == 0:
        y = graph_group_norm(tape, ex, x, c.n, c.c, c.rest(), c.groups, wo, bo, Float32(_EPS))
    else:
        var ro: Optional[FP] = None
        if c.track:
            ro = tape.value(run)
        y = graph_instance_norm(tape, ex, x, c.n, c.c, c.rest(), wo, bo, ro, Float32(_MOM), Float32(_EPS), c.train)
    var yv = tape.values(ex, y)
    var running = List[Float64]()
    var want = _oracle(c, _as64(vals[0]), _as64(vals[1]), _as64(vals[2]), vals[3], running)
    _close(c.name + " forward", yv, want, Float64(2e-5))
    var rv = tape.values(ex, run)
    _close(c.name + " running statistics", rv, running, Float64(1e-6))
    var seed = _seed(len(yv))
    var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
    tape.backward_with_rules(ex, y, st, ModelRules())
    var words = yv.copy()
    for i in range(len(rv)):  # small-loop(running words)
        words.append(rv[i])
    var ins: List[Tensor] = [x, w, b]
    var n_in = 3 if c.affine else 1
    comptime h = Float64(1e-5)
    for a in range(n_in):  # small-loop(at most 3 inputs)
        var g = tape.gradients(ex, ins[a])
        var fd = List[Float64](length=len(g), fill=0)
        for i in range(len(g)):  # small-loop(len(g): every element probed)
            fd[i] = (_oracle_loss(c, vals, seed, a, i, h) - _oracle_loss(c, vals, seed, a, i, -h)) / (2.0 * h)
        _close(c.name + " gradient " + String(a), g, fd, Float64(2e-4))
        for i in range(len(g)):  # small-loop(len(g))
            words.append(g[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_NORMS", c.name, "digest", digest)
    return words^


# ---------------------------------------------------------------- state and checkpoint
comptime _ST_C = 2
comptime _ST_L = 8
comptime _ST_N = 12


def _momentum_bits(m: Float32) -> Int:
    return Int(bitcast[DType.int32](m))


def run_instance_state[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    """The tracked statistics are state, not parameters: instance_norm1d(affine, tracked) then a linear head,
    fitted with AdamW (lr 0.05, weight decay 0.5), batch = every row, no shuffle, two epochs. Its running
    statistics must be PyTorch's update applied twice to the batch's average instance statistics (fixed: the
    input does not change), within 1e-6, where a decayed vector would be 0.975^2 of it; its weight must have
    moved; its predictions must be the eval forward (the running statistics) of the fitted words. A checkpoint
    of the fitted model loads back with the same parameter words."""
    var cols = _ST_C * _ST_L
    var x = _dyadic(_ST_N * cols, 7, 23, 8)
    var y = _dyadic(_ST_N, 5, 9, 4)
    var m = SeqModel()
    m.sample = [_ST_C, _ST_L]
    m.layer(L_INSTANCE_NORM, _ST_C, _momentum_bits(Float32(_MOM)), 1, 1, 1, 0, _EPS)
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 1, 1)
    m.compile(0, "adamw", 0.05, 0.5)
    _ = m.fit(ex, x, _ST_N, cols, y, 1, 0, 2, _ST_N, 0, 5)
    var p = m.params.copy()
    var C = _ST_C
    var S = _ST_L
    for ch in range(C):  # small-loop(channels)
        var mean_avg = Float64(0)
        var unb_avg = Float64(0)
        for n in range(_ST_N):
            var mean = Float64(0)
            for s in range(S):
                mean += Float64(x[(n * C + ch) * S + s])
            mean /= Float64(S)
            var sq = Float64(0)
            for s in range(S):
                var d = Float64(x[(n * C + ch) * S + s]) - mean
                sq += d * d
            mean_avg += mean
            unb_avg += sq / Float64(S - 1)
        mean_avg /= Float64(_ST_N)
        unb_avg /= Float64(_ST_N)
        var rm = Float64(0)
        var rvar = Float64(1)
        for _ in range(2):  # small-loop(2 epochs of one batch)
            rm = (1.0 - _MOM) * rm + _MOM * mean_avg
            rvar = (1.0 - _MOM) * rvar + _MOM * unb_avg
        var gm: List[Float32] = [p[2 * C + ch]]
        var wm: List[Float64] = [rm]
        _close("state: running_mean " + String(ch), gm, wm, Float64(1e-6))
        var gv: List[Float32] = [p[3 * C + ch]]
        var wv: List[Float64] = [rvar]
        _close("state: running_var " + String(ch), gv, wv, Float64(1e-6))
        if p[ch] == Float32(1):
            _fail("state: the instance norm's weight did not train")
    var pred = m.predict(ex, x, _ST_N, cols, OUT_RAW)
    var lin = 4 * C
    var want = List[Float64](length=_ST_N, fill=0)
    for n in range(_ST_N):  # small-loop(check oracle)
        var acc = Float64(p[lin + cols])
        for ch in range(C):
            var inv = 1.0 / sqrt(Float64(p[3 * C + ch]) + _EPS)
            for s in range(S):
                var xh = (Float64(x[(n * C + ch) * S + s]) - Float64(p[2 * C + ch])) * inv
                acc += (xh * Float64(p[ch]) + Float64(p[C + ch])) * Float64(p[lin + ch * S + s])
        want[n] = acc
    _close("state: eval-mode predictions", pred, want, Float64(1e-4))
    # the checkpoint's tensor names follow the layer's options (weight, bias, running_mean, running_var)
    var file = String("nn_norms_state_") + String(unit) + ".ckpt"
    _ = m.save(file, cols, False)
    var back = SeqModel()
    back.sample = [_ST_C, _ST_L]
    back.layer(L_INSTANCE_NORM, _ST_C, _momentum_bits(Float32(_MOM)), 1, 1, 1, 0, _EPS)
    back.layer(L_FLATTEN)
    back.layer(L_LINEAR, 1, 1)
    back.compile(0, "adamw", 0.05, 0.5)
    back.load(file, cols)
    _same_words("state: checkpoint round trip", p, back.params)
    var again = back.predict(ex, x, _ST_N, cols, OUT_RAW)
    _same_words("state: predictions after the checkpoint", pred, again)
    print("NN_NORMS instance-norm state: running statistics kept from the optimizer, eval predictions, checkpoint")
    var words = p.copy()
    for i in range(len(pred)):  # small-loop(predictions)
        words.append(pred[i])
    return words^


# ---------------------------------------------------------------- fits
comptime _CH = 2
comptime _LEN = 32
comptime _TRAIN = 96
comptime _TEST = 64
comptime _EPOCHS = 6


struct _Seq(Movable):
    var x: List[Float32]
    var y: List[Float32]
    var codes: List[Int]
    var rows: Int

    def __init__(out self, rows: Int, salt: Int):
        """checks/native_tests/sequence_blocks.mojo's task: class 1 oscillates five times over the length in
        channel 0, class 0 twice; channel 1 is a slow ramp shared by both; 0.3 standard-normal noise; random
        phase."""
        var noise = normal(rows * _CH * _LEN, salt)
        var phase = normal(rows, salt + 1)
        self.x = List[Float32](length=rows * _CH * _LEN, fill=0)
        self.codes = List[Int](length=rows, fill=0)
        self.rows = rows
        for r in range(rows):  # small-loop(fixture data)
            var cls = (r * 7 + salt) % 2
            self.codes[r] = cls
            var f = Float64(5 if cls == 1 else 2)
            for t in range(_LEN):
                var arg = 6.283185307179586 * f * Float64(t) / Float64(_LEN) + Float64(phase[r])
                self.x[(r * _CH) * _LEN + t] = Float32(sin(arg)) + Float32(0.3) * noise[(r * _CH) * _LEN + t]
                self.x[(r * _CH + 1) * _LEN + t] = Float32(Float64(t) / Float64(_LEN)) + Float32(0.3) * noise[(r * _CH + 1) * _LEN + t]
        self.y = codes_as_words(self.codes)


def _gn(mut m: SeqModel, c: Int):
    m.layer(L_GROUP_NORM, 4, c, 1, 0, 0, 0, 1e-5)


def _resnet_gn(mut m: SeqModel):
    # a ResNet-1D block with group norm (4 groups) after every convolution (conv, gn, relu), 1x1 shortcut
    m.layer(L_CONV1D, 8, 7, 1, 3, 0, 1)     # 0
    _gn(m, 8)                                # 1
    m.layer(L_RELU)                          # 2  block input
    m.layer(L_CONV1D, 12, 5, 1, 2, 0, 1)    # 3  body
    _gn(m, 12)                               # 4
    m.layer(L_RELU)                          # 5
    m.layer(L_CONV1D, 12, 3, 1, 1, 0, 1)    # 6
    _gn(m, 12)                               # 7  body end
    m.layer(L_BRANCH, 2)                     # 8  back to the block input
    m.layer(L_CONV1D, 12, 1, 1, 0, 0, 1)    # 9  1x1 shortcut
    _gn(m, 12)                               # 10
    m.layer(L_RESIDUAL, 7)                   # 11 shortcut + body
    m.layer(L_RELU)                          # 12
    m.layer(L_ADAPTIVE_AVGPOOL1D, 1)         # 13
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)


def _cnn_in(mut m: SeqModel):
    # a 1-D CNN with instance norm (affine, PyTorch's default untracked statistics) after the first convolution.
    # (The first cpc1 run had a second instance norm after the second convolution: held-out accuracy 0.59. Each
    # channel's amplitude, the frequency response of the convolution before it, is what the norm removes, so a
    # norm right before the pooled head leaves the head only the shape of a sinusoid; the second convolution
    # recovers the frequency from the normalized signal.)
    m.layer(L_CONV1D, 8, 5, 1, 2, 0, 1)
    m.layer(L_INSTANCE_NORM, 8, _momentum_bits(Float32(0.1)), 1, 1, 0, 0, 1e-5)
    m.layer(L_RELU)
    m.layer(L_CONV1D, 8, 3, 1, 1, 1, 1)
    m.layer(L_RELU)
    m.layer(L_ADAPTIVE_AVGPOOL1D, 1)
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)


def run_fit[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """Fit `which` (0 ResNet-1D with group norm, 1 CNN with instance norm); returns params then logits."""
    var train = _Seq(_TRAIN, 11)
    var test = _Seq(_TEST, 12)
    var m = SeqModel()
    m.sample = [_CH, _LEN]
    var name: String
    if which == 0:
        _resnet_gn(m)
        name = "resnet1d-gn"
    else:
        _cnn_in(m)
        name = "cnn1d-in"
    m.compile(LOSS_CROSS_ENTROPY, "adam", 0.01)
    var cols = _CH * _LEN
    _ = m.fit(ex, train.x, _TRAIN, cols, train.y, 1, TARGET_CODES, _EPOCHS, 32, -1, 3)
    var curve = m.losses()
    _finite(name + " loss curve", curve)
    _finite(name + " parameters", m.params)
    if not (curve[len(curve) - 1] < curve[0]):
        _fail(name + ": the loss did not fall (" + String(curve[0]) + " -> " + String(curve[len(curve) - 1]) + ")")
    var logits = m.predict(ex, test.x, _TEST, cols, OUT_RAW)
    _finite(name + " predictions", logits)
    var pred = m.predict(ex, test.x, _TEST, cols, OUT_ARGMAX)
    var right = 0
    for i in range(_TEST):  # small-loop(check accuracy)
        if Int(pred[i]) == test.codes[i]:
            right += 1
    var acc = Float64(right) / Float64(_TEST)
    print("NN_NORMS fit", name, "loss", curve[0], "->", curve[len(curve) - 1], "held-out accuracy", acc)
    if acc < 0.9:
        _fail(name + ": held-out accuracy " + String(acc) + " is below 0.9")
    var words = m.params.copy()
    for i in range(len(logits)):  # small-loop(predictions)
        words.append(logits[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_NORMS fit", name, "digest", digest)
    _ = train^
    _ = test^
    return words^


# ---------------------------------------------------------------- tests
def test_nn_norms_ops() raises:
    # Independent cases and the state unit: the host column spreads them across host cores (one HostExec each).
    var cases = _cases()
    var nc = len(cases)
    var n = nc + 1
    var failed = List[Int](length=n, fill=0)
    var host_words = List[List[Float32]](length=n, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    var cp = cases.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp, imm cp, imm nc}:
        try:
            var host = HostExec()
            if unit < nc:
                wp[unit] = run_norm_case(host, cp[unit])
            else:
                wp[unit] = run_instance_state(host, unit)
        except error:
            print("FAIL NN_NORMS unit", unit, error)
            fp[unit] = 1
    host_parallelize(task, n)
    for u in range(n):  # small-loop(n: failure flags)
        if failed[u] != 0:
            _fail("norm unit " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for k in range(n):  # driver-loop: over the norm units on the one device
            var dev = DeviceExec()
            var dw = run_norm_case(dev, cases[k]) if k < nc else run_instance_state(dev, n + k)
            _same_words("norm unit " + String(k), host_words[k], dw)
    _ = failed^
    _ = host_words^
    _ = cases^
    print("PASS NN_NORMS ops", n)


def test_nn_norms_fits() raises:
    comptime F = 2
    var failed = List[Int](length=F, fill=0)
    var host_words = List[List[Float32]](length=F, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = run_fit(host, unit)
        except error:
            print("FAIL NN_NORMS fit", unit, error)
            fp[unit] = 1
    host_parallelize(task, F)
    for u in range(F):  # small-loop(F: failure flags)
        if failed[u] != 0:
            _fail("fit " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for which in range(F):  # driver-loop: over the fits on the one device
            var dev = DeviceExec()
            var dw = run_fit(dev, which)
            _same_words("fit " + String(which), host_words[which], dw)
    _ = failed^
    _ = host_words^
    print("PASS NN_NORMS fits", F)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
