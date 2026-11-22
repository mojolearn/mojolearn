# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The 1-D sequence blocks of Sequential (lane seq-blocks, 2026-11-22).

Blocks on the tape (autograd/rankn_conv.mojo rankn_conv1d, rankn_max_pool1d,
rankn_avg_pool1d; tape.mean over the length; autograd/rankn_layout.mojo
rankn_concat on the channel axis), each:
  - the forward against a float64 oracle written here elementwise (no product helper);
  - every gradient (x, and weight and bias for conv1d) against central finite
    differences of the product's own forward loss sum(seed * y), every element
    probed. The fixtures are dyadic (multiples of 1/16 and 1/8, h = 1/64), so the
    conv, the pools and the concat are exact in binary32 and the differences equal
    the gradients; the mean and the average pool divide by the window (1/13 and 1/3
    round) and get 1e-5 (forward) and 1e-4 (gradients);
  - one `SEQ_BLOCKS <case> digest` line over every compared word.
Baselines through the model interpreter (autograd/model.mojo LAYER_CONV1D ...
LAYER_CONCAT, the flat form the residual_block and parallel containers write):
a 1D-CNN, a ResNet-1D block with a 1x1 shortcut, and an InceptionTime module with
a residual shortcut, each fitted on a separable synthetic two-class sequence task
(class 1: a faster oscillation in channel 0 than class 0, with noise): the loss
falls and the held-out accuracy reaches 0.9.
Columns: the CPU column runs everything on HostExec. The device column runs every
block and every fit on DeviceExec AND on HostExec and requires the same words
(values, gradients, fitted parameters, predictions): the host-column-versus-device
bytes check. The GRU / LSTM baseline over variable-length sequences waits for the
recurrent `lengths` block (not built in this lane).
"""
from std.math import sin
from std.memory import bitcast
from std.testing import TestSuite
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_api import RankNRules
from autograd.rankn_conv import rankn_conv1d, rankn_max_pool1d, rankn_avg_pool1d
from autograd.rankn_layout import rankn_concat
from autograd.model import LOSS_CROSS_ENTROPY, TARGET_CODES, OUT_ARGMAX
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.core_ops_common import normal
from checks.native_tests.training_sequential_common import SeqModel, codes_as_words, L_RELU, L_RESIDUAL, L_FLATTEN, L_LINEAR
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD


comptime L_CONV1D = 15
comptime L_MAXPOOL1D = 16
comptime L_AVGPOOL1D = 17
comptime L_ADAPTIVE_AVGPOOL1D = 18
comptime L_BRANCH = 19
comptime L_CONCAT = 20
comptime UNUSED = -2

comptime _H = Float64(1.0) / Float64(64)


def _fail(message: String) raises:
    print("FAIL SEQ_BLOCKS", message)
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


def _distinct(n: Int) -> List[Float32]:
    # distinct multiples of 1/16 (a permutation of 0..n-1), so no max-pool window ties within 2h
    var v = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(n: check data)
        v[i] = Float32((i * 37) % n - n // 2) / Float32(16)
    return v^


def _mix(mut digest: UInt64, values: List[Float32]):
    for i in range(len(values)):  # small-loop(len(values): check digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": " + String(len(got)) + " values, " + String(len(want)) + " expected")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        var d = abs(Float64(got[i]) - want[i])
        if d > tol * (1.0 + abs(want[i])):
            _fail(name + ": index " + String(i) + " got " + String(got[i]) + " want " + String(want[i]))


def _same_words(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": host and device lengths differ")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": host and device words differ at " + String(i) + " (" + String(a[i]) + " vs "
                  + String(b[i]) + ")")


# ---------------------------------------------------------------- block cases
# case geometry: x (N, C, L); op 0 conv1d (OC, K, S, P, G, bias), 1 max_pool1d (K, S, P),
# 2 avg_pool1d (K, S, P), 3 mean over the length, 4 concat of x and a second input (N, C2, L)
struct _Case(Copyable, Movable):
    var name: String
    var op: Int
    var n: Int
    var c: Int
    var l: Int
    var g: List[Int]

    def __init__(out self, name: String, op: Int, n: Int, c: Int, l: Int, g: List[Int]):
        self.name = name
        self.op = op
        self.n = n
        self.c = c
        self.l = l
        self.g = g.copy()


def _cases() -> List[_Case]:
    var out = List[_Case]()
    out.append(_Case("conv1d s1 p1 bias", 0, 2, 3, 13, [4, 3, 1, 1, 1, 1]))
    out.append(_Case("conv1d s2 p0", 0, 2, 3, 13, [4, 5, 2, 0, 1, 0]))
    out.append(_Case("conv1d groups3 bias", 0, 2, 3, 13, [6, 3, 1, 1, 3, 1]))
    out.append(_Case("conv1d k1 bottleneck", 0, 2, 3, 13, [5, 1, 1, 0, 1, 1]))
    out.append(_Case("max_pool1d k3 s1 p1", 1, 2, 3, 13, [3, 1, 1]))
    out.append(_Case("max_pool1d k2 s2", 1, 2, 3, 13, [2, 2, 0]))
    out.append(_Case("avg_pool1d k3 s2 p1", 2, 2, 3, 13, [3, 2, 1]))
    out.append(_Case("mean over length", 3, 2, 3, 13, List[Int]()))
    out.append(_Case("concat channels", 4, 2, 3, 13, [2]))
    return out^


def _in_counts(c: _Case) -> List[Int]:
    """Element counts of the inputs: x, then (conv1d) weight and bias, (concat) the second input."""
    var out = List[Int]()
    out.append(c.n * c.c * c.l)
    if c.op == 0:
        out.append(c.g[0] * (c.c // c.g[4]) * c.g[1])
        if c.g[5] == 1:
            out.append(c.g[0])
    elif c.op == 4:
        out.append(c.n * c.g[0] * c.l)
    return out^


def _inputs(c: _Case) -> List[List[Float32]]:
    var counts = _in_counts(c)
    var out = List[List[Float32]]()
    if c.op == 1:
        out.append(_distinct(counts[0]))
    else:
        out.append(_dyadic(counts[0], 5, 17, 16))
    for k in range(1, len(counts)):  # small-loop(at most 3 inputs)
        out.append(_dyadic(counts[k], 7 + 2 * k, 13, 8))
    return out^


def _forward[E: Exec](mut ex: E, mut tape: Tape, c: _Case, vals: List[List[Float32]], grad: Bool,
                      mut ins: List[Tensor]) raises -> Tensor:
    ins = List[Tensor]()
    var x = tape.tensor_nd(ex, vals[0], _sh([c.n, c.c, c.l]), grad)
    ins.append(x)
    if c.op == 0:
        var oc = c.g[0]
        var k = c.g[1]
        var w = tape.tensor_nd(ex, vals[1], _sh([oc, c.c // c.g[4], k]), grad)
        ins.append(w)
        var bias = Optional[Tensor]()
        if c.g[5] == 1:
            var b = tape.tensor_nd(ex, vals[2], _sh([oc]), grad)
            ins.append(b)
            bias = Optional[Tensor](b)
        return rankn_conv1d(tape, ex, x, w, bias, c.g[2], c.g[3], c.g[4])
    if c.op == 1:
        return rankn_max_pool1d(tape, ex, x, c.g[0], c.g[1], c.g[2])
    if c.op == 2:
        return rankn_avg_pool1d(tape, ex, x, c.g[0], c.g[1], c.g[2])
    if c.op == 3:
        return tape.mean(ex, x, 1)
    var x2 = tape.tensor_nd(ex, vals[1], _sh([c.n, c.g[0], c.l]), grad)
    ins.append(x2)
    var parts = List[Tensor]()
    parts.append(x)
    parts.append(x2)
    return rankn_concat(tape, ex, parts, 1)


def _oracle(c: _Case, vals: List[List[Float32]]) -> List[Float64]:
    """The forward in float64, elementwise."""
    var N = c.n
    var C = c.c
    var L = c.l
    var x = vals[0].copy()
    var out = List[Float64]()
    if c.op == 0:
        var OC = c.g[0]; var K = c.g[1]; var S = c.g[2]; var P = c.g[3]; var G = c.g[4]
        var Cg = C // G
        var OCg = OC // G
        var OL = (L + 2 * P - K) // S + 1
        for n in range(N):  # small-loop(check oracle)
            for oc in range(OC):
                for o in range(OL):
                    var acc = Float64(vals[2][oc]) if c.g[5] == 1 else Float64(0)
                    for cl in range(Cg):
                        var ch = (oc // OCg) * Cg + cl
                        for k in range(K):
                            var t = o * S - P + k
                            if t >= 0 and t < L:
                                acc += Float64(x[(n * C + ch) * L + t]) * Float64(vals[1][(oc * Cg + cl) * K + k])
                    out.append(acc)
    elif c.op == 1 or c.op == 2:
        var K = c.g[0]; var S = c.g[1]; var P = c.g[2]
        var OL = (L + 2 * P - K) // S + 1
        for r in range(N * C):  # small-loop(check oracle)
            for o in range(OL):
                var acc = Float64(0)
                var first = True
                for k in range(K):
                    var t = o * S - P + k
                    if t < 0 or t >= L:
                        continue
                    var v = Float64(x[r * L + t])
                    if c.op == 1:
                        if first or v > acc:
                            acc = v
                        first = False
                    else:
                        acc += v
                out.append(acc if c.op == 1 else acc / Float64(K))
    elif c.op == 3:
        for r in range(N * C):  # small-loop(check oracle)
            var acc = Float64(0)
            for t in range(L):
                acc += Float64(x[r * L + t])
            out.append(acc / Float64(L))
    else:
        var C2 = c.g[0]
        for n in range(N):  # small-loop(check oracle)
            for ch in range(C + C2):
                for t in range(L):
                    if ch < C:
                        out.append(Float64(x[(n * C + ch) * L + t]))
                    else:
                        out.append(Float64(vals[1][(n * C2 + ch - C) * L + t]))
    return out^


def _seed(count: Int) -> List[Float32]:
    return _dyadic(count, 3, 11, 16)


def _loss[E: Exec](mut ex: E, c: _Case, vals: List[List[Float32]], seed: List[Float32]) raises -> Float64:
    var tape = Tape()
    var ins = List[Tensor]()
    var y = _forward(ex, tape, c, vals, False, ins)
    var yv = tape.values(ex, y)
    var s = Float64(0)
    for i in range(len(yv)):  # small-loop(check loss)
        s += Float64(seed[i]) * Float64(yv[i])
    return s


def run_block_case[E: Exec](mut ex: E, c: _Case) raises -> List[Float32]:
    """Forward and every gradient checked; returns the words compared (values, then gradients)."""
    var vals = _inputs(c)
    var tape = Tape()
    var ins = List[Tensor]()
    var y = _forward(ex, tape, c, vals, True, ins)
    var yv = tape.values(ex, y)
    # the mean and the average pool divide by the window (1/13, 1/3 round); the rest is exact
    var tol = Float64(1e-5) if (c.op == 2 or c.op == 3) else Float64(1e-9)
    _close(c.name + " forward", yv, _oracle(c, vals), tol)
    var seed = _seed(len(yv))
    var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
    tape.backward_with_rules(ex, y, st, RankNRules())
    var words = yv.copy()
    for a in range(len(ins)):  # small-loop(at most 3 inputs)
        var g = tape.gradients(ex, ins[a])
        var fd = List[Float64](length=len(g), fill=0)
        for i in range(len(g)):  # small-loop(len(g): every element probed)
            var up = vals.copy()
            up[a][i] = Float32(Float64(vals[a][i]) + _H)
            var down = vals.copy()
            down[a][i] = Float32(Float64(vals[a][i]) - _H)
            fd[i] = (_loss(ex, c, up, seed) - _loss(ex, c, down, seed)) / (2.0 * _H)
        _close(c.name + " gradient " + String(a), g, fd, Float64(1e-4) if (c.op == 2 or c.op == 3) else Float64(1e-9))
        for i in range(len(g)):  # small-loop(len(g))
            words.append(g[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("SEQ_BLOCKS", c.name, "digest", digest)
    return words^


# ---------------------------------------------------------------- baselines
comptime _CH = 2
comptime _LEN = 32
comptime _TRAIN = 192
comptime _TEST = 64


struct _Seq(Movable):
    var x: List[Float32]
    var y: List[Float32]
    var codes: List[Int]
    var rows: Int

    def __init__(out self, rows: Int, salt: Int):
        """Class 1 oscillates five times over the length in channel 0, class 0 twice; channel 1
        is a slow ramp shared by both; every value has 0.3 standard-normal noise. Random phase."""
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


def _cnn(mut m: SeqModel):
    m.layer(L_CONV1D, 8, 5, 1, 2, 1, 1)
    m.layer(L_RELU)
    m.layer(L_MAXPOOL1D, 2, 2, 0)
    m.layer(L_CONV1D, 8, 3, 1, 1, 1, 1)
    m.layer(L_RELU)
    m.layer(L_ADAPTIVE_AVGPOOL1D, 1)
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)


def _resnet(mut m: SeqModel):
    # stem, then residual_block(body = conv5 relu conv3, shortcut = conv1) as the container writes it
    m.layer(L_CONV1D, 8, 7, 1, 3, 1, 1)     # 0
    m.layer(L_RELU)                          # 1  block input
    m.layer(L_CONV1D, 12, 5, 1, 2, 1, 1)    # 2  body
    m.layer(L_RELU)                          # 3
    m.layer(L_CONV1D, 12, 3, 1, 1, 1, 1)    # 4  body end
    m.layer(L_BRANCH, 1)                     # 5  back to the block input
    m.layer(L_CONV1D, 12, 1, 1, 0, 1, 1)    # 6  1x1 shortcut
    m.layer(L_RESIDUAL, 4)                   # 7  shortcut + body
    m.layer(L_RELU)                          # 8
    m.layer(L_ADAPTIVE_AVGPOOL1D, 1)         # 9
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)


def _inception(mut m: SeqModel):
    # parallel(bottleneck conv9, bottleneck conv5, maxpool3 conv1) on the input, as the container
    # writes it, then a residual 1x1 shortcut from the model input
    m.layer(L_CONV1D, 4, 1, 1, 0, 0, 1)                        # 0  bottleneck
    m.layer(L_CONV1D, 6, 9, 1, 4, 0, 1)                        # 1  branch a end
    m.layer(L_BRANCH, 0)                                       # 2
    m.layer(L_CONV1D, 6, 5, 1, 2, 0, 1)                        # 3  branch b end
    m.layer(L_BRANCH, -1)                                      # 4  the model input
    m.layer(L_MAXPOOL1D, 3, 1, 1)                              # 5
    m.layer(L_CONV1D, 6, 1, 1, 0, 0, 1)                        # 6  branch c end
    m.layer(L_CONCAT, 1, 3, 6, UNUSED, UNUSED, UNUSED)         # 7  (18, L)
    m.layer(L_RELU)                                            # 8
    m.layer(L_BRANCH, -1)                                      # 9
    m.layer(L_CONV1D, 18, 1, 1, 0, 1, 1)                       # 10 shortcut
    m.layer(L_RESIDUAL, 8)                                     # 11
    m.layer(L_RELU)                                            # 12
    m.layer(L_ADAPTIVE_AVGPOOL1D, 1)                           # 13
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)


def run_baseline[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """Fit baseline `which` (0 CNN, 1 ResNet-1D, 2 InceptionTime); returns params then predictions."""
    var train = _Seq(_TRAIN, 11)
    var test = _Seq(_TEST, 12)
    var m = SeqModel()
    m.sample = [_CH, _LEN]
    var name: String
    if which == 0:
        _cnn(m)
        name = "cnn1d"
    elif which == 1:
        _resnet(m)
        name = "resnet1d"
    else:
        _inception(m)
        name = "inceptiontime"
    m.compile(LOSS_CROSS_ENTROPY, "adam", 0.01)
    var cols = _CH * _LEN
    _ = m.fit(ex, train.x, _TRAIN, cols, train.y, 1, TARGET_CODES, 20, 32, -1, 3)
    var curve = m.losses()
    if not (curve[len(curve) - 1] < curve[0]):
        _fail(name + ": the loss did not fall (" + String(curve[0]) + " -> " + String(curve[len(curve) - 1]) + ")")
    var pred = m.predict(ex, test.x, _TEST, cols, OUT_ARGMAX)
    var right = 0
    for i in range(_TEST):  # small-loop(check accuracy)
        if Int(pred[i]) == test.codes[i]:
            right += 1
    var acc = Float64(right) / Float64(_TEST)
    print("SEQ_BLOCKS baseline", name, "loss", curve[0], "->", curve[len(curve) - 1], "held-out accuracy", acc)
    if acc < 0.9:
        _fail(name + ": held-out accuracy " + String(acc) + " is below 0.9")
    var words = m.params.copy()
    for i in range(len(pred)):  # small-loop(predictions)
        words.append(pred[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("SEQ_BLOCKS baseline", name, "digest", digest)
    _ = train^
    _ = test^
    return words^


# ---------------------------------------------------------------- tests
def test_sequence_blocks() raises:
    var cases = _cases()
    for k in range(len(cases)):  # driver-loop: over the block cases
        var host = HostExec()
        var hw = run_block_case(host, cases[k])
        comptime if not CPU_COLUMN_BUILD:
            var dev = DeviceExec()
            var dw = run_block_case(dev, cases[k])
            _same_words(cases[k].name, hw, dw)
    print("PASS SEQ_BLOCKS blocks", len(cases))


def test_sequence_baselines() raises:
    for which in range(3):  # driver-loop: over the three composed baselines
        var host = HostExec()
        var hw = run_baseline(host, which)
        comptime if not CPU_COLUMN_BUILD:
            var dev = DeviceExec()
            var dw = run_baseline(dev, which)
            _same_words("baseline " + String(which), hw, dw)
    print("PASS SEQ_BLOCKS baselines 3")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
