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
bytes check. The GRU / LSTM baselines over variable-length sequences are
checks/native_tests/sequence_lengths.mojo.

Batch norm (lane seq-blocks-3; autograd/rankn_conv.mojo rankn_batch_norm, the
Sequential layer autograd/model_blocks.mojo LAYER_BATCH_NORM):
  - training and eval forwards of a 1-D (N, C, L) and a 2-D (N, C, H, W) case against
    a float64 oracle, the running update included; every gradient (x, weight, bias)
    against central differences of the float64 oracle's loss sum(seed * y);
  - the state is not the optimizer's: a model whose first layer is the batch norm,
    fitted with AdamW (weight decay 0.5) for two one-batch epochs, holds exactly the
    running statistics PyTorch's update gives (the optimizer would have decayed them),
    its weight trained, and its predictions are the eval-mode forward of the fitted
    words (the running statistics, not the batch's);
  - a ResNet-1D and an InceptionTime with batch norm after every convolution reach
    held-out accuracy 0.9 with a falling loss.
"""
from std.math import sin
from std.memory import bitcast
from std.testing import TestSuite
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_api import RankNRules
from autograd.rankn_conv import rankn_conv1d, rankn_max_pool1d, rankn_avg_pool1d, rankn_batch_norm
from std.math import sqrt
from autograd.rankn_layout import rankn_concat
from autograd.model import LOSS_CROSS_ENTROPY, TARGET_CODES, OUT_ARGMAX, OUT_RAW
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.core_ops_common import normal
from checks.native_tests.training_sequential_common import SeqModel, codes_as_words, L_RELU, L_RESIDUAL, L_FLATTEN, L_LINEAR
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize


comptime L_CONV1D = 15
comptime L_MAXPOOL1D = 16
comptime L_AVGPOOL1D = 17
comptime L_ADAPTIVE_AVGPOOL1D = 18
comptime L_BRANCH = 19
comptime L_CONCAT = 20
comptime L_BATCH_NORM = 46
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
comptime _TRAIN = 96
comptime _TEST = 64
# 96 rows x 6 epochs: the first cpc1 run (192 rows x 20 epochs) reached held-out accuracy 1.0 on all three
# baselines but took 5 hours of host column; this keeps the task far above the 0.9 bar at a sixth of the steps.
comptime _EPOCHS = 6


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


def _momentum_bits(m: Float32) -> Int:
    return Int(bitcast[DType.int32](m))


def _bn(mut m: SeqModel, c: Int):
    m.layer(L_BATCH_NORM, c, _momentum_bits(Float32(0.1)), 1, 0, 0, 0, 1e-5)


def _resnet_bn(mut m: SeqModel):
    # the ResNet-1D above with a batch norm after every convolution (conv, bn, relu)
    m.layer(L_CONV1D, 8, 7, 1, 3, 0, 1)     # 0
    _bn(m, 8)                                # 1
    m.layer(L_RELU)                          # 2  block input
    m.layer(L_CONV1D, 12, 5, 1, 2, 0, 1)    # 3  body
    _bn(m, 12)                               # 4
    m.layer(L_RELU)                          # 5
    m.layer(L_CONV1D, 12, 3, 1, 1, 0, 1)    # 6
    _bn(m, 12)                               # 7  body end
    m.layer(L_BRANCH, 2)                     # 8  back to the block input
    m.layer(L_CONV1D, 12, 1, 1, 0, 0, 1)    # 9  1x1 shortcut
    _bn(m, 12)                               # 10
    m.layer(L_RESIDUAL, 7)                   # 11 shortcut + body
    m.layer(L_RELU)                          # 12
    m.layer(L_ADAPTIVE_AVGPOOL1D, 1)         # 13
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)


def _inception_bn(mut m: SeqModel):
    # the InceptionTime module above with the batch norm of the original: after the concat and on the shortcut
    m.layer(L_CONV1D, 4, 1, 1, 0, 0, 1)                        # 0  bottleneck
    m.layer(L_CONV1D, 6, 9, 1, 4, 0, 1)                        # 1  branch a end
    m.layer(L_BRANCH, 0)                                       # 2
    m.layer(L_CONV1D, 6, 5, 1, 2, 0, 1)                        # 3  branch b end
    m.layer(L_BRANCH, -1)                                      # 4  the model input
    m.layer(L_MAXPOOL1D, 3, 1, 1)                              # 5
    m.layer(L_CONV1D, 6, 1, 1, 0, 0, 1)                        # 6  branch c end
    m.layer(L_CONCAT, 1, 3, 6, UNUSED, UNUSED, UNUSED)         # 7  (18, L)
    _bn(m, 18)                                                 # 8
    m.layer(L_RELU)                                            # 9
    m.layer(L_BRANCH, -1)                                      # 10
    m.layer(L_CONV1D, 18, 1, 1, 0, 0, 1)                       # 11 shortcut
    _bn(m, 18)                                                 # 12
    m.layer(L_RESIDUAL, 9)                                     # 13
    m.layer(L_RELU)                                            # 14
    m.layer(L_ADAPTIVE_AVGPOOL1D, 1)                           # 15
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)


def run_baseline[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """Fit baseline `which` (0 CNN, 1 ResNet-1D, 2 InceptionTime, 3 ResNet-1D with batch norm, 4 InceptionTime
    with batch norm); returns params then predictions."""
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
    elif which == 2:
        _inception(m)
        name = "inceptiontime"
    elif which == 3:
        _resnet_bn(m)
        name = "resnet1d-bn"
    else:
        _inception_bn(m)
        name = "inceptiontime-bn"
    m.compile(LOSS_CROSS_ENTROPY, "adam", 0.01)
    var cols = _CH * _LEN
    _ = m.fit(ex, train.x, _TRAIN, cols, train.y, 1, TARGET_CODES, _EPOCHS, 32, -1, 3)
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
            wp[unit] = run_block_case(host, cp[unit])
        except error:
            print("FAIL SEQ_BLOCKS block", unit, error)
            fp[unit] = 1
    host_parallelize(task, n)
    for u in range(n):  # small-loop(n: failure flags)
        if failed[u] != 0:
            _fail("block case " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for k in range(n):  # driver-loop: over the block cases on the one device
            var dev = DeviceExec()
            var dw = run_block_case(dev, cases[k])
            _same_words(cases[k].name, host_words[k], dw)
    _ = failed^
    _ = host_words^
    _ = cases^
    print("PASS SEQ_BLOCKS blocks", n)


def test_sequence_baselines() raises:
    # The three fits are independent: on the host column each runs on its own HostExec across host
    # cores (failure flags per unit); the device column then runs them in order on the one device.
    comptime B = 5
    var failed = List[Int](length=B, fill=0)
    var host_words = List[List[Float32]](length=B, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = run_baseline(host, unit)
        except error:
            print("FAIL SEQ_BLOCKS baseline", unit, error)
            fp[unit] = 1
    host_parallelize(task, B)
    for u in range(B):  # small-loop(B: failure flags)
        if failed[u] != 0:
            _fail("baseline " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for which in range(B):  # driver-loop: over the baselines on the one device
            var dev = DeviceExec()
            var dw = run_baseline(dev, which)
            _same_words("baseline " + String(which), host_words[which], dw)
    _ = failed^
    _ = host_words^
    print("PASS SEQ_BLOCKS baselines 5")


# ---------------------------------------------------------------- batch norm
# case: x (N, C, S) with S the product of the spatial axes (2-D: H W); train or eval
struct _BnCase(Copyable, Movable):
    var name: String
    var n: Int
    var c: Int
    var dims: List[Int]  # the per-sample axes after the channel
    var train: Bool

    def __init__(out self, name: String, n: Int, c: Int, dims: List[Int], train: Bool):
        self.name = name
        self.n = n
        self.c = c
        self.dims = dims.copy()
        self.train = train

    def spatial(self) -> Int:
        var s = 1
        for k in range(len(self.dims)):  # small-loop(at most 2 axes)
            s *= self.dims[k]
        return s


def _bn_cases() -> List[_BnCase]:
    var out = List[_BnCase]()
    out.append(_BnCase("batch_norm1d train", 4, 3, [5], True))
    out.append(_BnCase("batch_norm2d train", 2, 2, [3, 4], True))
    out.append(_BnCase("batch_norm1d eval", 3, 3, [4], False))
    out.append(_BnCase("batch_norm1d (N, C) train", 6, 4, List[Int](), True))
    return out^


comptime _BN_EPS = Float64(1e-5)
comptime _BN_MOM = Float64(0.25)


def _bn_vals(c: _BnCase) -> List[List[Float32]]:
    """x, gamma, beta, running [mean C | var C]."""
    var out = List[List[Float32]]()
    out.append(_dyadic(c.n * c.c * c.spatial(), 5, 17, 16))
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


def _bn_oracle(c: _BnCase, vals: List[List[Float32]], mut running: List[Float64]) -> List[Float64]:
    """The forward in float64 (and, training, the running update into `running`)."""
    var N = c.n
    var C = c.c
    var S = c.spatial()
    var out = List[Float64](length=N * C * S, fill=0)
    running = List[Float64](length=2 * C, fill=0)
    for ch in range(C):  # small-loop(check oracle)
        var mean = Float64(0)
        var var_b = Float64(0)
        if c.train:
            for n in range(N):
                for s in range(S):
                    mean += Float64(vals[0][(n * C + ch) * S + s])
            mean /= Float64(N * S)
            for n in range(N):
                for s in range(S):
                    var d = Float64(vals[0][(n * C + ch) * S + s]) - mean
                    var_b += d * d
            var_b /= Float64(N * S)
            var unb = var_b * Float64(N * S) / Float64(N * S - 1)
            running[ch] = (1.0 - _BN_MOM) * Float64(vals[3][ch]) + _BN_MOM * mean
            running[C + ch] = (1.0 - _BN_MOM) * Float64(vals[3][C + ch]) + _BN_MOM * unb
        else:
            mean = Float64(vals[3][ch])
            var_b = Float64(vals[3][C + ch])
            running[ch] = mean
            running[C + ch] = var_b
        var inv = 1.0 / sqrt(var_b + _BN_EPS)
        for n in range(N):
            for s in range(S):
                var i = (n * C + ch) * S + s
                out[i] = (Float64(vals[0][i]) - mean) * inv * Float64(vals[1][ch]) + Float64(vals[2][ch])
    return out^


def _bn_oracle_loss(c: _BnCase, vals: List[List[Float32]], seed: List[Float32], which: Int, i: Int,
                    h: Float64) -> Float64:
    """sum(seed * y) of the float64 forward with input `which`'s element i moved by h."""
    var x = List[List[Float64]]()
    for a in range(3):  # small-loop(3 inputs)
        var v = List[Float64](length=len(vals[a]), fill=0)
        for k in range(len(vals[a])):
            v[k] = Float64(vals[a][k])
        x.append(v^)
    x[which][i] += h
    var N = c.n
    var C = c.c
    var S = c.spatial()
    var total = Float64(0)
    for ch in range(C):  # small-loop(check oracle)
        var mean = Float64(vals[3][ch])
        var var_b = Float64(vals[3][C + ch])
        if c.train:
            mean = 0
            for n in range(N):
                for s in range(S):
                    mean += x[0][(n * C + ch) * S + s]
            mean /= Float64(N * S)
            var_b = 0
            for n in range(N):
                for s in range(S):
                    var d = x[0][(n * C + ch) * S + s] - mean
                    var_b += d * d
            var_b /= Float64(N * S)
        var inv = 1.0 / sqrt(var_b + _BN_EPS)
        for n in range(N):
            for s in range(S):
                var k = (n * C + ch) * S + s
                total += Float64(seed[k]) * ((x[0][k] - mean) * inv * x[1][ch] + x[2][ch])
    return total


def run_bn_case[E: Exec](mut ex: E, c: _BnCase) raises -> List[Float32]:
    """Forward, running update and every gradient checked; returns the words compared."""
    var vals = _bn_vals(c)
    var tape = Tape()
    var xd = List[Int]()
    xd.append(c.n)
    xd.append(c.c)
    for k in range(len(c.dims)):  # small-loop(at most 2 axes)
        xd.append(c.dims[k])
    var x = tape.tensor_nd(ex, vals[0], _sh(xd), True)
    var cd: List[Int] = [c.c]
    var gamma = tape.tensor_nd(ex, vals[1], _sh(cd), True)
    var beta = tape.tensor_nd(ex, vals[2], _sh(cd), True)
    var run = tape.tensor(ex, vals[3], 1, 2 * c.c, False)
    # the layer views the sample as (N, C, S), as the model interpreter does
    var x3 = tape.view_nd(ex, x, _sh([c.n, c.c, c.spatial()]))
    var y = rankn_batch_norm(tape, ex, x3, gamma, beta, tape.value(run), Float32(_BN_EPS), Float32(_BN_MOM), c.train)
    var yv = tape.values(ex, y)
    var running = List[Float64]()
    var want = _bn_oracle(c, vals, running)
    _close(c.name + " forward", yv, want, Float64(2e-5))
    var rv = tape.values(ex, run)
    _close(c.name + " running statistics", rv, running, Float64(1e-6))
    var seed = _seed(len(yv))
    var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
    tape.backward_with_rules(ex, y, st, RankNRules())
    var words = yv.copy()
    for i in range(len(rv)):  # small-loop(running words)
        words.append(rv[i])
    var ins: List[Tensor] = [x, gamma, beta]
    comptime h = Float64(1e-5)
    for a in range(3):  # small-loop(3 inputs)
        var g = tape.gradients(ex, ins[a])
        var fd = List[Float64](length=len(g), fill=0)
        for i in range(len(g)):  # small-loop(len(g): every element probed)
            fd[i] = (_bn_oracle_loss(c, vals, seed, a, i, h) - _bn_oracle_loss(c, vals, seed, a, i, -h)) / (2.0 * h)
        _close(c.name + " gradient " + String(a), g, fd, Float64(2e-4))
        for i in range(len(g)):  # small-loop(len(g))
            words.append(g[i])
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("SEQ_BLOCKS", c.name, "digest", digest)
    return words^


comptime _ST_C = 2
comptime _ST_L = 8
comptime _ST_N = 12


def run_bn_state[E: Exec](mut ex: E) raises -> List[Float32]:
    """The running statistics are state, not parameters: a model whose first layer is the batch norm, fitted with
    AdamW (lr 0.05, weight decay 0.5), batch = every row, no shuffle, two epochs. Its running statistics must be
    PyTorch's update applied twice to the input's (fixed) batch statistics, word for word within 1e-6, where a
    decayed vector would be 0.975^2 of it; its weight must have moved; its predictions must be the eval forward
    of the fitted words."""
    var cols = _ST_C * _ST_L
    var x = _dyadic(_ST_N * cols, 7, 23, 8)
    var y = _dyadic(_ST_N, 5, 9, 4)
    var m = SeqModel()
    m.sample = [_ST_C, _ST_L]
    m.layer(L_BATCH_NORM, _ST_C, _momentum_bits(Float32(_BN_MOM)), 1, 0, 0, 0, _BN_EPS)
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 1, 1)
    m.compile(0, "adamw", 0.05, 0.5)
    _ = m.fit(ex, x, _ST_N, cols, y, 1, 0, 2, _ST_N, 0, 5)
    var p = m.params.copy()
    var C = _ST_C
    var S = _ST_L
    for ch in range(C):  # small-loop(channels)
        var mean = Float64(0)
        for n in range(_ST_N):
            for s in range(S):
                mean += Float64(x[(n * C + ch) * S + s])
        mean /= Float64(_ST_N * S)
        var sq = Float64(0)
        for n in range(_ST_N):
            for s in range(S):
                var d = Float64(x[(n * C + ch) * S + s]) - mean
                sq += d * d
        var unb = sq / Float64(_ST_N * S - 1)
        var rm = Float64(0)
        var rvar = Float64(1)
        for _ in range(2):  # small-loop(2 epochs of one batch)
            rm = (1.0 - _BN_MOM) * rm + _BN_MOM * mean
            rvar = (1.0 - _BN_MOM) * rvar + _BN_MOM * unb
        var gm: List[Float32] = [p[2 * C + ch]]
        var wm: List[Float64] = [rm]
        _close("state: running_mean " + String(ch), gm, wm, Float64(1e-6))
        var gv: List[Float32] = [p[3 * C + ch]]
        var wv: List[Float64] = [rvar]
        _close("state: running_var " + String(ch), gv, wv, Float64(1e-6))
        if p[ch] == Float32(1):
            _fail("state: the batch norm's weight did not train")
    var pred = m.predict(ex, x, _ST_N, cols, OUT_RAW)
    var lin = 4 * C
    var want = List[Float64](length=_ST_N, fill=0)
    for n in range(_ST_N):  # small-loop(check oracle)
        var acc = Float64(p[lin + cols])
        for ch in range(C):
            var inv = 1.0 / sqrt(Float64(p[3 * C + ch]) + _BN_EPS)
            for s in range(S):
                var xh = (Float64(x[(n * C + ch) * S + s]) - Float64(p[2 * C + ch])) * inv
                acc += (xh * Float64(p[ch]) + Float64(p[C + ch])) * Float64(p[lin + ch * S + s])
        want[n] = acc
    _close("state: eval-mode predictions", pred, want, Float64(1e-4))
    print("SEQ_BLOCKS batch-norm state: running statistics restored around every optimizer step, eval predictions")
    var words = p.copy()
    for i in range(len(pred)):  # small-loop(predictions)
        words.append(pred[i])
    return words^


def test_sequence_batch_norm() raises:
    var cases = _bn_cases()
    var n = len(cases) + 1
    var failed = List[Int](length=n, fill=0)
    var host_words = List[List[Float32]](length=n, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    var cp = cases.unsafe_ptr()
    var nc = len(cases)
    def task(unit: Int) {imm fp, imm wp, imm cp, imm nc}:
        try:
            var host = HostExec()
            if unit < nc:
                wp[unit] = run_bn_case(host, cp[unit])
            else:
                wp[unit] = run_bn_state(host)
        except error:
            print("FAIL SEQ_BLOCKS batch norm", unit, error)
            fp[unit] = 1
    host_parallelize(task, n)
    for u in range(n):  # small-loop(n: failure flags)
        if failed[u] != 0:
            _fail("batch norm unit " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for k in range(n):  # driver-loop: over the batch-norm units on the one device
            var dev = DeviceExec()
            var dw = run_bn_case(dev, cases[k]) if k < nc else run_bn_state(dev)
            _same_words("batch norm unit " + String(k), host_words[k], dw)
    _ = failed^
    _ = host_words^
    _ = cases^
    print("PASS SEQ_BLOCKS batch norm", n)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
