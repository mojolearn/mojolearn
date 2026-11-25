# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Sequential's pools as Session ops (sub-lane session-ops-pool, 2026-11-25; session codes 140..148,
autograd/session_ops_pool.mojo), through the call the Python door makes (`OpsSession.op(code, [x], ints, [])`).

test_session_ops_pool_cases, per case (24 cases: max / avg pool 1-D, 2-D with PyTorch's options, 3-D, adaptive avg
1-D / 2-D / 3-D and adaptive max 1-D / 2-D; the smallest shapes that reach padding, stride, dilation, ceil_mode,
count_include_pad, divisor_override, uneven adaptive windows and an output size kept with None):
  - the forward against a float64 reference written here from PyTorch's definitions (window or adaptive window,
    padding never chosen by a max, the average's divisor); max exact, average within 1e-6 relative;
  - the gradient of x of the loss sum(seed * y) from the recording session's scalar backward against the
    reference's closed form (the seed at the first maximum, or seed / divisor over the window) and, where the case
    has no tie, against central finite differences of the Session op's own forward loss, every element probed
    (inputs distinct multiples of 1/16, h = 1/64, so no window's maximum changes: max 1e-9, average 1e-4);
  - the seeded backward (Session.vjp with the same seed) gives the scalar backward's words;
  - where Sequential has the layer, the Session op's forward and gradient words equal Sequential's forward function
    run on a bare tape with the same input and seed (LayerRules over ImportRules, the rules the session uses):
    rankn_max_pool1d / rankn_avg_pool1d (layers 16, 17), rankn_max_pool2d / rankn_avg_pool2d (13, 14; the option
    codes 142 / 143 with plain words give the old codes' words), rankn_max_pool3d / rankn_avg_pool3d /
    rankn_adaptive_avg_pool3d (74..76), Sequential 18's dispatch for adaptive_avg_pool1d (the tape mean, an equal
    window avg pool, else nn_adaptive_avg_pool1d), nn_adaptive_avg_pool2d (58), nn_adaptive_max_pool1d / 2d (60, 59);
  - two tie cases (max_pool2d with options, adaptive_max_pool2d): the first maximum in window order takes the
    gradient (reference words), Session = Sequential words.
test_session_ops_pool_refusals: the refusals in words (rank, padding over half the window, a window that does not
fit, a non-positive window, an adaptive 3-D size that neither is 1 nor divides, the rank word, the word count).

Columns: the CPU column runs every case on HostExec. The device column runs every case on DeviceExec AND HostExec and
requires the same words (forward, gradient, Sequential's words). Cases run in order: each opens sessions in the one
process session table (ops_session_open), as nn_qkv's cases do.
"""
from std.memory import bitcast
from std.testing import TestSuite
from autograd.ops_api import OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN
from autograd.import_api import ImportRules
from autograd.session_layer_rules import LayerRules
from autograd.rankn_conv import rankn_max_pool1d, rankn_avg_pool1d, rankn_max_pool2d, rankn_avg_pool2d
from autograd.nn_conv3d import Dhw, rankn_max_pool3d, rankn_avg_pool3d, rankn_adaptive_avg_pool3d
from autograd.nn_conv import (
    nn_adaptive_avg_pool1d, nn_adaptive_avg_pool2d, nn_adaptive_max_pool1d, nn_adaptive_max_pool2d,
)
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.ops_higher_common import tensor, total, values, gradient, same_words, op2
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD, check_cases_serial, require

comptime _MUL = 4
comptime _H = Float64(1.0) / Float64(64)

# the Sequential forward function a case compares with (0: none, the option has no Sequential layer)
comptime SEQ_NONE = 0
comptime SEQ_MAX1D = 1
comptime SEQ_AVG1D = 2
comptime SEQ_MAX2D = 3
comptime SEQ_AVG2D = 4
comptime SEQ_MAX3D = 5
comptime SEQ_AVG3D = 6
comptime SEQ_ADAPT3D = 7
comptime SEQ_ADAPT_AVG1D = 8
comptime SEQ_ADAPT_AVG2D = 9
comptime SEQ_ADAPT_MAX1D = 10
comptime SEQ_ADAPT_MAX2D = 11


struct _Case(Movable):
    var name: String
    var code: Int
    var ints: List[Int]
    var xdims: List[Int]
    var adaptive: Bool
    var is_max: Bool
    # (depth, height, width) of the reference; 1-D and 2-D inputs have leading axes of size 1
    var k: List[Int]
    var s: List[Int]
    var p: List[Int]
    var d: List[Int]
    var o: List[Int]
    var ceil: Bool
    var cip: Bool
    var div: Int
    var seq: Int
    var ties: Bool

    def __init__(out self, name: String, code: Int, var ints: List[Int], var xdims: List[Int], is_max: Bool,
                 seq: Int):
        self.name = name
        self.code = code
        self.ints = ints^
        self.xdims = xdims^
        self.adaptive = False
        self.is_max = is_max
        self.k = [1, 1, 1]
        self.s = [1, 1, 1]
        self.p = [0, 0, 0]
        self.d = [1, 1, 1]
        self.o = [1, 1, 1]
        self.ceil = False
        self.cip = True
        self.div = 0
        self.seq = seq
        self.ties = False

    def spatial(self) -> List[Int]:
        """(D, H, W) of x, leading axes of size 1 for a 1-D or 2-D input."""
        var r = len(self.xdims)
        var out: List[Int] = [1, 1, 1]
        for a in range(r - 2):  # small-loop(at most three spatial axes)
            out[3 - (r - 2) + a] = self.xdims[2 + a]
        return out^

    def out3(self) -> List[Int]:
        var sp = self.spatial()
        var out = List[Int]()
        for a in range(3):  # small-loop(three axes)
            if self.adaptive:
                out.append(self.o[a])
            else:
                out.append(_out_size(sp[a], self.k[a], self.p[a], self.s[a], self.d[a], self.ceil))
        return out^

    def odims(self) -> List[Int]:
        var r = len(self.xdims)
        var o = self.out3()
        var out: List[Int] = [self.xdims[0], self.xdims[1]]
        for a in range(r - 2):  # small-loop(at most three spatial axes)
            out.append(o[3 - (r - 2) + a])
        return out^

    def xcount(self) -> Int:
        var n = 1
        for v in self.xdims:  # small-loop(at most five dims)
            n *= v
        return n


def _out_size(size: Int, k: Int, p: Int, s: Int, dil: Int, ceil: Bool) -> Int:
    """PyTorch's pooling_output_shape (aten/src/ATen/native/Pool.h), written again here."""
    var num = size + 2 * p - dil * (k - 1) - 1 + ((s - 1) if ceil else 0)
    var o = num // s + 1
    if ceil and (o - 1) * s >= size + p:
        o -= 1
    return o


def _win(name: String, code: Int, var ints: List[Int], var xdims: List[Int], is_max: Bool, seq: Int,
         var k: List[Int], var s: List[Int], var p: List[Int], var d: List[Int], ceil: Bool, cip: Bool,
         div: Int) -> _Case:
    var c = _Case(name, code, ints^, xdims^, is_max, seq)
    c.k = k^
    c.s = s^
    c.p = p^
    c.d = d^
    c.ceil = ceil
    c.cip = cip
    c.div = div
    return c^


def _adapt(name: String, code: Int, var ints: List[Int], var xdims: List[Int], is_max: Bool, seq: Int,
           var o: List[Int]) -> _Case:
    var c = _Case(name, code, ints^, xdims^, is_max, seq)
    c.adaptive = True
    c.o = o^
    return c^


comptime POOL_CASES = 24


def _case(which: Int) -> _Case:
    # 1-D: x (2, 2, 7)
    if which == 0:
        return _win("max_pool1d", 140, [3, 2, 1, 1, 0], [2, 2, 7], True, SEQ_MAX1D,
                    [1, 1, 3], [1, 1, 2], [0, 0, 1], [1, 1, 1], False, True, 0)
    if which == 1:
        return _win("max_pool1d-dilation-ceil", 140, [2, 2, 1, 2, 1], [2, 2, 7], True, SEQ_NONE,
                    [1, 1, 2], [1, 1, 2], [0, 0, 1], [1, 1, 2], True, True, 0)
    if which == 2:
        return _win("avg_pool1d", 141, [3, 2, 1, 0, 1], [2, 2, 7], False, SEQ_AVG1D,
                    [1, 1, 3], [1, 1, 2], [0, 0, 1], [1, 1, 1], False, True, 0)
    if which == 3:
        return _win("avg_pool1d-ceil-exclude-pad", 141, [3, 2, 1, 1, 0], [2, 2, 7], False, SEQ_NONE,
                    [1, 1, 3], [1, 1, 2], [0, 0, 1], [1, 1, 1], True, False, 0)
    # 2-D: x (1, 2, 5, 6)
    if which == 4 or which == 21:
        var c = _win("max_pool2d-options" if which == 4 else "max_pool2d-options-ties", 142,
                     [3, 2, 2, 1, 1, 1, 1, 2, 1], [1, 2, 5, 6], True, SEQ_NONE,
                     [1, 3, 2], [1, 2, 1], [0, 1, 1], [1, 1, 2], True, True, 0)
        c.ties = which == 21
        return c^
    if which == 5:
        return _win("max_pool2d-plain-words", 142, [2, 2, 2, 2, 0, 0, 1, 1, 0], [1, 2, 5, 6], True, SEQ_MAX2D,
                    [1, 2, 2], [1, 2, 2], [0, 0, 0], [1, 1, 1], False, True, 0)
    if which == 6:
        return _win("avg_pool2d-ceil-exclude-pad", 143, [3, 3, 2, 2, 1, 1, 1, 0, 0], [1, 2, 5, 6], False, SEQ_NONE,
                    [1, 3, 3], [1, 2, 2], [0, 1, 1], [1, 1, 1], True, False, 0)
    if which == 7:
        return _win("avg_pool2d-divisor", 143, [2, 3, 1, 2, 1, 1, 0, 1, 4], [1, 2, 5, 6], False, SEQ_NONE,
                    [1, 2, 3], [1, 1, 2], [0, 1, 1], [1, 1, 1], False, True, 4)
    if which == 8:
        return _win("avg_pool2d-plain-words", 143, [2, 2, 2, 2, 0, 0, 0, 1, 0], [1, 2, 5, 6], False, SEQ_AVG2D,
                    [1, 2, 2], [1, 2, 2], [0, 0, 0], [1, 1, 1], False, True, 0)
    # 3-D: x (1, 2, 3, 4, 4)
    if which == 9:
        return _win("max_pool3d", 144, [2, 2, 2, 1, 2, 2, 1, 1, 0, 1, 1, 2, 1], [1, 2, 3, 4, 4], True, SEQ_MAX3D,
                    [2, 2, 2], [1, 2, 2], [1, 1, 0], [1, 1, 2], True, True, 0)
    if which == 10:
        return _win("avg_pool3d-exclude-pad", 145, [2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0], [1, 2, 3, 4, 4], False,
                    SEQ_AVG3D, [2, 2, 2], [2, 2, 2], [1, 1, 1], [1, 1, 1], False, False, 0)
    if which == 11:
        return _win("avg_pool3d-divisor-ceil", 145, [2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 3], [1, 2, 3, 4, 4], False,
                    SEQ_AVG3D, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], True, True, 3)
    # adaptive 3-D: x (1, 2, 2, 4, 4)
    if which == 12:
        return _adapt("adaptive_avg_pool3d-global", 146, [1, 1, 1], [1, 2, 2, 4, 4], False, SEQ_ADAPT3D, [1, 1, 1])
    if which == 13:
        # depth None (word 0): the input's 2
        return _adapt("adaptive_avg_pool3d-tiles-keep", 146, [0, 2, 2], [1, 2, 2, 4, 4], False, SEQ_ADAPT3D,
                      [2, 2, 2])
    # adaptive 1-D
    if which == 14:
        return _adapt("adaptive_avg_pool1d-uneven", 147, [1, 3, 0], [2, 2, 7], False, SEQ_ADAPT_AVG1D, [1, 1, 3])
    if which == 15:
        return _adapt("adaptive_avg_pool1d-global", 147, [1, 1, 0], [2, 2, 7], False, SEQ_ADAPT_AVG1D, [1, 1, 1])
    if which == 16:
        return _adapt("adaptive_avg_pool1d-divides", 147, [1, 3, 0], [2, 2, 6], False, SEQ_ADAPT_AVG1D, [1, 1, 3])
    # adaptive 2-D: x (1, 2, 5, 7)
    if which == 17:
        return _adapt("adaptive_avg_pool2d", 147, [2, 3, 4], [1, 2, 5, 7], False, SEQ_ADAPT_AVG2D, [1, 3, 4])
    if which == 18:
        return _adapt("adaptive_avg_pool2d-keep", 147, [2, 0, 2], [1, 2, 5, 7], False, SEQ_ADAPT_AVG2D, [1, 5, 2])
    if which == 19:
        return _adapt("adaptive_max_pool1d", 148, [1, 3, 0], [2, 2, 7], True, SEQ_ADAPT_MAX1D, [1, 1, 3])
    if which == 20 or which == 22:
        var c = _adapt("adaptive_max_pool2d" if which == 20 else "adaptive_max_pool2d-ties", 148, [2, 2, 3],
                       [1, 2, 5, 7], True, SEQ_ADAPT_MAX2D, [1, 2, 3])
        c.ties = which == 22
        return c^
    # which == 23: a 3-D max pool with ties
    var c = _win("max_pool3d-ties", 144, [2, 2, 2, 1, 2, 2, 1, 1, 0, 1, 1, 2, 1], [1, 2, 3, 4, 4], True, SEQ_MAX3D,
                 [2, 2, 2], [1, 2, 2], [1, 1, 0], [1, 1, 2], True, True, 0)
    c.ties = True
    return c^


def _x(c: _Case) -> List[Float32]:
    """Distinct multiples of 1/16 (count <= 101: 37 is a unit mod 101), or with ties a few repeated levels."""
    var n = c.xcount()
    var out = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(fixture)
        var v = (i * 37) % 101
        if c.ties:
            v = v % 5
            out[i] = Float32(v - 2) / Float32(16)
        else:
            out[i] = Float32(v - 50) / Float32(16)
    return out^


def _seed(count: Int) -> List[Float32]:
    var out = List[Float32](length=count, fill=0)
    for j in range(count):  # small-loop(fixture)
        out[j] = Float32(((j * 13) % 17) - 8) / Float32(8)
    return out^


struct _Ref(Movable):
    var out: List[Float64]
    var grad: List[Float64]

    def __init__(out self, n_out: Int, n_in: Int):
        self.out = List[Float64](length=n_out, fill=0)
        self.grad = List[Float64](length=n_in, fill=0)


def _reference(c: _Case, x: List[Float32], seed: List[Float32]) -> _Ref:
    """The forward and the gradient of sum(seed * y) in float64 from PyTorch's definitions."""
    var sp = c.spatial()
    var D = sp[0]
    var H = sp[1]
    var W = sp[2]
    var o = c.out3()
    var NC = c.xdims[0] * c.xdims[1]
    var r = _Ref(NC * o[0] * o[1] * o[2], len(x))
    var at = 0
    for nc in range(NC):  # small-loop(check reference)
        var base = nc * D * H * W
        for od in range(o[0]):
            for oh in range(o[1]):
                for ow in range(o[2]):
                    var lo = List[Int](length=3, fill=0)
                    var hi = List[Int](length=3, fill=0)
                    var step: List[Int] = [1, 1, 1]
                    var oo: List[Int] = [od, oh, ow]
                    for a in range(3):  # small-loop(three axes)
                        if c.adaptive:
                            lo[a] = (oo[a] * sp[a]) // o[a]
                            hi[a] = ((oo[a] + 1) * sp[a] + o[a] - 1) // o[a]
                        else:
                            lo[a] = oo[a] * c.s[a] - c.p[a]
                            hi[a] = lo[a] + (c.k[a] - 1) * c.d[a] + 1
                            step[a] = c.d[a]
                    var best = Float64(0)
                    var bi = -1
                    var acc = Float64(0)
                    var z = lo[0]
                    while z < hi[0]:
                        var h = lo[1]
                        while h < hi[1]:
                            var w = lo[2]
                            while w < hi[2]:
                                if z >= 0 and z < D and h >= 0 and h < H and w >= 0 and w < W:
                                    var i = base + (z * H + h) * W + w
                                    var v = Float64(x[i])
                                    if bi < 0 or v > best:
                                        best = v
                                        bi = i
                                    acc += v
                                w += step[2]
                            h += step[1]
                        z += step[0]
                    var g = Float64(seed[at])
                    if c.is_max:
                        r.out[at] = best
                        r.grad[bi] += g
                    else:
                        var size: Int
                        if c.adaptive:
                            size = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
                        elif c.div > 0:
                            size = c.div
                        else:
                            # the window clipped to the padded input, then to the input unless count_include_pad
                            var e0 = min(lo[0] + c.k[0], D + c.p[0])
                            var e1 = min(lo[1] + c.k[1], H + c.p[1])
                            var e2 = min(lo[2] + c.k[2], W + c.p[2])
                            size = (e0 - lo[0]) * (e1 - lo[1]) * (e2 - lo[2])
                            if not c.cip:
                                size = ((min(e0, D) - max(lo[0], 0)) * (min(e1, H) - max(lo[1], 0))
                                        * (min(e2, W) - max(lo[2], 0)))
                        r.out[at] = acc / Float64(size)
                        var z2 = max(lo[0], 0)
                        while z2 < min(hi[0], D):
                            var h2 = max(lo[1], 0)
                            while h2 < min(hi[1], H):
                                var w2 = max(lo[2], 0)
                                while w2 < min(hi[2], W):
                                    r.grad[base + (z2 * H + h2) * W + w2] += g / Float64(size)
                                    w2 += 1
                                h2 += 1
                            z2 += 1
                    at += 1
    return r^


def _seq_forward[E: Exec](mut tape: Tape, mut ex: E, c: _Case, x: Tensor) raises -> Tensor:
    """Sequential's forward function for the case's layer, called as the model interpreter calls it."""
    if c.seq == SEQ_MAX1D:
        return rankn_max_pool1d(tape, ex, x, c.k[2], c.s[2], c.p[2])
    if c.seq == SEQ_AVG1D:
        return rankn_avg_pool1d(tape, ex, x, c.k[2], c.s[2], c.p[2])
    if c.seq == SEQ_MAX2D:
        return rankn_max_pool2d(tape, ex, x, c.k[2], c.s[2])
    if c.seq == SEQ_AVG2D:
        return rankn_avg_pool2d(tape, ex, x, c.k[2], c.s[2])
    if c.seq == SEQ_MAX3D:
        return rankn_max_pool3d(tape, ex, x, Dhw(c.k[0], c.k[1], c.k[2]), Dhw(c.s[0], c.s[1], c.s[2]),
                                Dhw(c.p[0], c.p[1], c.p[2]), Dhw(c.d[0], c.d[1], c.d[2]), c.ceil)
    if c.seq == SEQ_AVG3D:
        return rankn_avg_pool3d(tape, ex, x, Dhw(c.k[0], c.k[1], c.k[2]), Dhw(c.s[0], c.s[1], c.s[2]),
                                Dhw(c.p[0], c.p[1], c.p[2]), c.ceil, c.cip, c.div)
    if c.seq == SEQ_ADAPT3D:
        return rankn_adaptive_avg_pool3d(tape, ex, x, Dhw(c.o[0], c.o[1], c.o[2]))
    if c.seq == SEQ_ADAPT_AVG1D:
        # autograd/model.mojo LAYER_ADAPTIVE_AVGPOOL1D, and nn_adaptive_avg_pool1d for a length it does not divide
        var length = c.xdims[2]
        var ol = c.o[2]
        if ol == 1:
            return tape.mean(ex, x, 1)
        if length % ol == 0:
            return rankn_avg_pool1d(tape, ex, x, length // ol, length // ol, 0)
        return nn_adaptive_avg_pool1d(tape, ex, x, ol)
    if c.seq == SEQ_ADAPT_AVG2D:
        return nn_adaptive_avg_pool2d(tape, ex, x, c.o[1], c.o[2])
    if c.seq == SEQ_ADAPT_MAX1D:
        return nn_adaptive_max_pool1d(tape, ex, x, c.o[2])
    return nn_adaptive_max_pool2d(tape, ex, x, c.o[1], c.o[2])


struct _Run(Movable):
    var out: List[Float32]
    var grad: List[Float32]
    var seq_out: List[Float32]
    var seq_grad: List[Float32]

    def __init__(out self):
        self.out = List[Float32]()
        self.grad = List[Float32]()
        self.seq_out = List[Float32]()
        self.seq_grad = List[Float32]()


def _pool[E: Exec & Movable & Deinitable](mut s: OpsSession[E], c: _Case, hx: Int) raises -> Int:
    var hs: List[Int] = [hx]
    return s.op(c.code, hs, c.ints, List[Float32]())


def _fd_loss[E: Exec & Movable & Deinitable](mut s: OpsSession[E], c: _Case, xv: List[Float32],
                                             seed: List[Float32]) raises -> Float64:
    """sum(seed * y) in float64 of the Session op's own forward on xv (an inference session)."""
    var hx = tensor(s, xv, c.xdims)
    var y = values(s, _pool(s, c, hx), c.odims())
    s.reset()
    var total64 = Float64(0)
    for i in range(len(y)):  # small-loop(check loss)
        total64 += Float64(seed[i]) * Float64(y[i])
    return total64


def _run_case[E: Exec & Movable & Deinitable, KIND: Int](var plain: E, var recording: E, mut ex: E,
                                                         which: Int) raises -> _Run:
    var c = _case(which)
    var xv = _x(c)
    var odims = c.odims()
    var oc = 1
    for v in odims:  # small-loop(at most five dims)
        oc *= v
    var seed = _seed(oc)
    var r = _Run()
    var sid = ops_session_open[E, KIND](plain^, False)
    var rid = ops_session_open[E, KIND](recording^, True)
    try:
        var s = ops_session[E, KIND](rid)
        var hx = tensor(s[], xv, c.xdims, True)
        var hseed = tensor(s[], seed, odims)
        var y = _pool(s[], c, hx)
        r.out = values(s[], y, odims)
        var loss = total(s[], op2(s[], _MUL, y, hseed))
        s[].backward(loss, False)
        r.grad = gradient(s[], hx, c.xdims)
        s[].reset()
        # the seeded backward (Session.vjp) of the same call with the seed itself: the scalar backward's words
        var y2 = _pool(s[], c, hx)
        var wrt: List[Int] = [hx]
        var g = s[].vjp(y2, wrt, hseed, False)
        same_words(values(s[], g[0], c.xdims), r.grad, c.name + ": vjp gradient differs from backward's")
        s[].reset()
        # the reference
        var want = _reference(c, xv, seed)
        require(len(want.out) == len(r.out), c.name + ": output extent")
        for i in range(len(r.out)):  # small-loop(check comparison)
            var e = want.out[i]
            var tol = Float64(0) if c.is_max else Float64(1e-6) * abs(e) + Float64(1e-6)
            require(abs(Float64(r.out[i]) - e) <= tol, c.name + ": output " + String(i) + " is " + String(r.out[i])
                    + ", the float64 reference " + String(e))
        for i in range(len(r.grad)):  # small-loop(check comparison)
            var e = want.grad[i]
            var tol = Float64(0) if c.is_max else Float64(1e-5) * abs(e) + Float64(1e-5)
            require(abs(Float64(r.grad[i]) - e) <= tol, c.name + ": gradient " + String(i) + " is "
                    + String(r.grad[i]) + ", the reference " + String(e))
        # central differences of the op's own forward loss, every element probed (no tie: no maximum changes)
        if not c.ties:
            var p = ops_session[E, KIND](sid)
            for i in range(len(xv)):  # small-loop(len(xv): every element probed)
                var up = xv.copy()
                up[i] = Float32(Float64(xv[i]) + _H)
                var down = xv.copy()
                down[i] = Float32(Float64(xv[i]) - _H)
                var fd = (_fd_loss(p[], c, up, seed) - _fd_loss(p[], c, down, seed)) / (2.0 * _H)
                var tol = Float64(1e-9) if c.is_max else Float64(1e-4)
                require(abs(Float64(r.grad[i]) - fd) <= tol, c.name + ": gradient " + String(i) + " is "
                        + String(r.grad[i]) + ", central differences " + String(fd))
        ops_session_close[E, KIND](rid)
        ops_session_close[E, KIND](sid)
    except error:
        ops_session_close[E, KIND](rid)
        ops_session_close[E, KIND](sid)
        raise error
    if c.seq != SEQ_NONE:
        # Sequential's forward function on a bare tape, the same input and seed, the session's rules
        var tape = Tape()
        var x = tape.tensor_nd(ex, xv, ShapeN.of(c.xdims), True)
        var ys = _seq_forward(tape, ex, c, x)
        r.seq_out = tape.values(ex, ys)
        var st = tape.tensor(ex, seed, tape.rows(ys), tape.cols(ys))
        tape.backward_with_rules(ex, ys, st, LayerRules[ImportRules](ImportRules()))
        r.seq_grad = tape.gradients(ex, x)
        tape.release(ex)
        same_words(r.seq_out, r.out, c.name + ": the Session op's forward differs from Sequential's layer")
        same_words(r.seq_grad, r.grad, c.name + ": the Session op's gradient differs from Sequential's layer")
    return r^


def test_session_ops_pool_cases() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_case(which: Int) raises:
            var ex = HostExec()
            _ = _run_case[HostExec, OPS_HOST](HostExec(), HostExec(), ex, which)
        check_cases_serial("session_ops_pool", POOL_CASES, host_case)
    else:
        def device_case(which: Int) raises:
            var dex = DeviceExec()
            var dev = _run_case[DeviceExec, OPS_DEVICE](DeviceExec(), DeviceExec(), dex, which)
            var hex = HostExec()
            var host = _run_case[HostExec, OPS_HOST](HostExec(), HostExec(), hex, which)
            var tag = "session_ops_pool case " + String(which)
            same_words(dev.out, host.out, tag + ": device and host forward")
            same_words(dev.grad, host.grad, tag + ": device and host gradient")
            same_words(dev.seq_out, host.seq_out, tag + ": device and host Sequential forward")
            same_words(dev.seq_grad, host.seq_grad, tag + ": device and host Sequential gradient")
        check_cases_serial("session_ops_pool", POOL_CASES, device_case)


def _refused[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, hx: Int, ints: List[Int],
                                             words: String) raises:
    var raised = False
    try:
        var hs: List[Int] = [hx]
        _ = s.op(code, hs, ints, List[Float32]())
    except problem:
        raised = True
        require(String(problem).find(words) >= 0, "pool code " + String(code) + " refused with '" + String(problem)
                + "', not '" + words + "'")
    require(raised, "pool code " + String(code) + " did not refuse: " + words)


def _refusals[E: Exec & Movable & Deinitable, KIND: Int](sid: Int) raises:
    var s = ops_session[E, KIND](sid)
    var d3: List[Int] = [1, 1, 7]
    var d4: List[Int] = [1, 1, 4, 4]
    var d5: List[Int] = [1, 1, 2, 4, 4]
    var x3 = tensor(s[], List[Float32](length=7, fill=1), d3)
    var x4 = tensor(s[], List[Float32](length=16, fill=1), d4)
    var x5 = tensor(s[], List[Float32](length=32, fill=1), d5)
    _refused(s[], 140, x4, [2, 2, 0, 1, 0], "max_pool1d: x must be (N, C, L), got a tensor of 4 dimensions")
    _refused(s[], 140, x3, [2, 1, 2, 1, 0],
             "max_pool1d: padding must be non-negative and at most half of kernel_size (padding 2, kernel_size 2)")
    _refused(s[], 141, x3, [9, 1, 0, 0, 1], "avg_pool1d: the window does not fit the padded input")
    _refused(s[], 142, x4, [0, 2, 1, 1, 0, 0, 1, 1, 0], "max_pool2d: kernel_size, stride and dilation must be positive")
    _refused(s[], 143, x4, [2, 2, 2, 2, 0, 0, 0, 1, -1], "avg_pool2d: divisor_override must be a positive integer")
    _refused(s[], 146, x5, [3, 3, 3], "adaptive_avg_pool3d: the output sizes must be 1 or divide the input's")
    _refused(s[], 147, x3, [3, 1, 0], "the rank word must be 1 or 2")
    _refused(s[], 148, x4, [1, 2, 0], "adaptive_max_pool1d: x must be (N, C, L)")
    _refused(s[], 144, x5, [2, 2, 2, 2, 2], "takes 1 tensors, 13 integers")
    s[].reset()


def test_session_ops_pool_refusals() raises:
    comptime if CPU_COLUMN_BUILD:
        var sid = ops_session_open[HostExec, OPS_HOST](HostExec(), False)
        try:
            _refusals[HostExec, OPS_HOST](sid)
        except error:
            ops_session_close[HostExec, OPS_HOST](sid)
            raise error
        ops_session_close[HostExec, OPS_HOST](sid)
    else:
        var sid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), False)
        try:
            _refusals[DeviceExec, OPS_DEVICE](sid)
        except error:
            ops_session_close[DeviceExec, OPS_DEVICE](sid)
            raise error
        ops_session_close[DeviceExec, OPS_DEVICE](sid)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
