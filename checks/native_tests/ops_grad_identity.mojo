# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""Forward and backward of the public session ops, device words against host
words, at shapes that reach the tuned routes.

Each case builds the same recorded graph twice in one process, on
`OpsSession[HostExec]` and on `OpsSession[DeviceExec]`: inputs from a fixed
hash, `y = op(inputs)`, `loss = sum(y * y)` (a seed that differs per element),
one backward. The op's output and the gradient of every input must be the same
float32 words on both. An op with no backward rule must refuse the backward
with the same text on both (the forward is still compared). No tolerance, no
stored reference: the host column is the reference, as the contract states it.

The earlier coverage of these ops' VJPs stopped at 16 input elements and at
products of 64 x 8 x 6, below every tuned route. Here:
  products (`matmul`, `linear`): one thread a cell (K < 4), the 16 x 16 tile,
    the 64 x 64 micro-tile (both edges >= 64), split-K (a small output with a
    long contraction), both thin tiles, a weight gradient whose contraction is
    600 rows, a batched rank-3 product, and `linear` with bias under each
    activation: the VJPs put the long axis in the contraction, so the
    backward crosses route rules the forward of the same case does not;
  element-wise and reduction ops at 70 x 67 (4,690 elements, 19 blocks);
  losses at 300 rows; layer and RMS norm at 130 x 70;
  conv2d 3 x 3 with stride, padding and groups, both pools;
  the shape ops (permute, both reshapes, concat, split, pad, transpose, expand).

What a sabotage changes here: any device kernel or VJP rule whose fold order,
flush or operand order differs from the host arm's (the op's output or a
gradient differs in at least one word at these sizes); a route rule that
host and device evaluate differently (a product's words differ wherever the
two orders round differently); a backward that reads a stale or reused
buffer on one column.

Also: rotary, the transformer's own linear and softmax, and a seeded
permutation with the rows gathered through it (forward only: a gather
carries no gradient).

Not covered (their arguments are integer id descriptors or a mode word this
file does not build): embedding, take, index, dropout, any/all and the
2-D-only `sum(axis)` / `mean(axis)` spellings. OPS_LM_HEAD_V2 has its own
check (ops_lm_head_v2); the block-step and generation ops are not public
session ops.

One device queue (DeviceExec shares the sequence context), so the cases run
one after another; a case's host work is a few milliseconds and uses the host
pool inside its kernels. The sessions do their own transfers; this file
queues no copy.
"""
from autograd.ops_api import (
    OpsSession,
    OPS_ABS, OPS_ADD, OPS_ARGMAX, OPS_AVGPOOL2D, OPS_BCE_LOGITS, OPS_CEIL, OPS_CLIP, OPS_COMPARE, OPS_CONCAT,
    OPS_CONV2D, OPS_COPY, OPS_COS, OPS_CROSS_ENTROPY, OPS_CUMSUM, OPS_DETACH, OPS_DIV, OPS_ELU, OPS_ERF, OPS_EXP,
    OPS_EXPAND, OPS_FLOOR, OPS_GELU, OPS_HARD_SIGMOID, OPS_HARD_SWISH, OPS_HUBER, OPS_LAYER_NORM, OPS_LEAKY_RELU,
    OPS_LINEAR, OPS_LOG, OPS_LOGICAL, OPS_LOG_SOFTMAX, OPS_LOG_SOFTMAX_N, OPS_MAE, OPS_MATMUL, OPS_MAX, OPS_MAXIMUM,
    OPS_MAXPOOL2D, OPS_MEAN_N, OPS_MINIMUM, OPS_MSE, OPS_MUL, OPS_NARROW, OPS_PAD, OPS_PERMUTE, OPS_POW,
    OPS_RECIPROCAL, OPS_RELU, OPS_RESHAPE, OPS_RESHAPE_N, OPS_RMS_NORM, OPS_ROUND, OPS_SCALE, OPS_SIGMOID, OPS_SIGN,
    OPS_SILU, OPS_SIN, OPS_SOFTMAX, OPS_SOFTMAX_N, OPS_SOFTPLUS, OPS_SQRT, OPS_SUB, OPS_SUM_N, OPS_TANH,
    OPS_TRANSPOSE, OPS_WHERE, OPS_GATHER_ROWS, OPS_PERMUTATION, OPS_ROTARY, OPS_TRANSFORMER_LINEAR,
    OPS_TRANSFORMER_SOFTMAX,
)
from core.hostptr import f32_ptr
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.ops_higher_common import tensor, gradient
from checks.native_tests.core_ops_common import uniform
from std.testing import TestSuite
from checks.native_tests.testsuite_serial import check_cases_serial
from checks.native_tests.core_ops_common import require


comptime OGI_MATMUL = 8
comptime OGI_LINEAR = 4
comptime OGI_UNARY = 28
comptime OGI_BINARY = 8
comptime OGI_SELECT = 3
comptime OGI_REDUCE = 10
comptime OGI_LOSS = 5
comptime OGI_NORM = 2
comptime OGI_CONV = 4
comptime OGI_SHAPE = 8
comptime OGI_EXTRA = 4
comptime OGI_CASES = (OGI_MATMUL + OGI_LINEAR + OGI_UNARY + OGI_BINARY + OGI_SELECT + OGI_REDUCE + OGI_LOSS
                      + OGI_NORM + OGI_CONV + OGI_SHAPE + OGI_EXTRA)

# the element-wise shape: 4,690 elements
comptime _ER = 70
comptime _EC = 67


def _d2(a: Int, b: Int) -> List[Int]:
    var d: List[Int] = [a, b]
    return d^


def _d3(a: Int, b: Int, c: Int) -> List[Int]:
    var d: List[Int] = [a, b, c]
    return d^


def _d4(a: Int, b: Int, c: Int, e: Int) -> List[Int]:
    var d: List[Int] = [a, b, c, e]
    return d^


def _count(dims: List[Int]) -> Int:
    var n = 1
    for d in dims:
        n *= d
    return n


def _leaf[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], dims: List[Int], seed: Int, low: Float64, high: Float64, grad: Bool = True
) raises -> Int:
    """An uploaded tensor of hashed values in [low, high)."""
    var words = uniform(_count(dims), seed, low, high)
    return tensor(s, words, dims, grad)


def _words[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int) raises -> List[Float32]:
    """Tensor h's float32 words, whatever its shape."""
    var n = s.shape_nd(h).count()
    var words = List[Float32](length=n, fill=0)
    s.download(h, f32_ptr(Int(words.unsafe_ptr())))
    return words^


def _op[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], code: Int, hs: List[Int], ints: List[Int], fs: List[Float32]
) raises -> Int:
    return s.op(code, hs, ints, fs)


def _op1[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int) raises -> Int:
    var hs: List[Int] = [a]
    return s.op(code, hs, List[Int](), List[Float32]())


def _op2[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, b: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    return s.op(code, hs, List[Int](), List[Float32]())


def _finish[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], y: Int, xs: List[Int], xdims: List[List[Int]], mut refusal: String
) raises -> List[List[Float32]]:
    """[y's words, the gradient of each xs[i]] after `sum(y * y).backward()`.
    A backward the tape refuses leaves its text in `refusal` and only y's
    words in the result."""
    var out = List[List[Float32]]()
    out.append(_words(s, y))
    var hs: List[Int] = [_op2(s, OPS_MUL, y, y)]
    var all: List[Int] = [-1]
    var loss = s.op(OPS_SUM_N, hs, all, List[Float32]())
    try:
        s.backward(loss, False)
    except error:
        refusal = String(error)
        return out^
    for i in range(len(xs)):
        out.append(gradient(s, xs[i], xdims[i]))
    return out^


def _forward[E: Exec & Movable & Deinitable](mut s: OpsSession[E], y: Int) raises -> List[List[Float32]]:
    """A forward-only op: its words."""
    var out = List[List[Float32]]()
    out.append(_words(s, y))
    return out^


def _one(dims: List[Int]) -> List[List[Int]]:
    var out = List[List[Int]]()
    out.append(dims.copy())
    return out^


def _two(a: List[Int], b: List[Int]) -> List[List[Int]]:
    var out = List[List[Int]]()
    out.append(a.copy())
    out.append(b.copy())
    return out^


def _three(a: List[Int], b: List[Int], c: List[Int]) -> List[List[Int]]:
    var out = List[List[Int]]()
    out.append(a.copy())
    out.append(b.copy())
    out.append(c.copy())
    return out^


# ---------------------------------------------------------------- products

def _matmul[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    # (M, K) @ (K, N); the VJPs are (M, N) x (N, K) and (K, M) x (M, N)
    var m = 9
    var kk = 3
    var n = 5            # 0: one thread a cell (K < 4)
    if k == 1:
        m = 40
        kk = 24
        n = 36           # the 16 x 16 tile
    elif k == 2:
        m = 128
        kk = 96
        n = 80           # the 64 x 64 micro-tile, forward and both VJPs
    elif k == 3:
        m = 8
        kk = 300
        n = 6            # split-K: a small output over a long contraction
    elif k == 4:
        m = 200
        kk = 24
        n = 4            # thin along N
    elif k == 5:
        m = 4
        kk = 24
        n = 200          # thin along M
    elif k == 6:
        m = 600
        kk = 70
        n = 66           # the weight gradient contracts 600 rows
    var a = _leaf(s, _d2(m, kk), 1000 + k, -1.0, 1.0)
    var b = _leaf(s, _d2(kk, n), 1100 + k, -1.0, 1.0)
    var xs: List[Int] = [a, b]
    return _finish(s, _op2(s, OPS_MATMUL, a, b), xs, _two(_d2(m, kk), _d2(kk, n)), refusal)


def _batched[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut refusal: String) raises -> List[List[Float32]]:
    var a = _leaf(s, _d3(3, 70, 66), 1007, -1.0, 1.0)
    var b = _leaf(s, _d3(3, 66, 68), 1107, -1.0, 1.0)
    var xs: List[Int] = [a, b]
    return _finish(s, _op2(s, OPS_MATMUL, a, b), xs, _two(_d3(3, 70, 66), _d3(3, 66, 68)), refusal)


def _linear[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    # x (rows, in), W (in, out), b (1, out); activation 0 none, 1 relu, 2 sigmoid, 3 tanh
    var x = _leaf(s, _d2(130, 70), 1200 + k, -1.0, 1.0)
    var w = _leaf(s, _d2(70, 66), 1300 + k, -0.3, 0.3)
    var ints: List[Int] = [k]
    if k == 2:
        # no bias
        var hs: List[Int] = [x, w]
        var xs: List[Int] = [x, w]
        return _finish(s, s.op(OPS_LINEAR, hs, ints, List[Float32]()), xs, _two(_d2(130, 70), _d2(70, 66)), refusal)
    var b = _leaf(s, _d2(1, 66), 1400 + k, -0.5, 0.5)
    var hs: List[Int] = [x, w, b]
    var xs: List[Int] = [x, w, b]
    return _finish(s, s.op(OPS_LINEAR, hs, ints, List[Float32]()), xs,
                   _three(_d2(130, 70), _d2(70, 66), _d2(1, 66)), refusal)


# ---------------------------------------------------------------- element-wise

def _unary[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    var code = OPS_RELU
    var low = -2.0
    var high = 2.0
    var ints = List[Int]()
    var fs = List[Float32]()
    var differentiable = True
    if k == 1:
        code = OPS_SIGMOID
    elif k == 2:
        code = OPS_TANH
    elif k == 3:
        code = OPS_EXP
    elif k == 4:
        code = OPS_LOG
        low = 0.5
        high = 3.0
    elif k == 5:
        code = OPS_SQRT
        low = 0.5
        high = 3.0
    elif k == 6:
        code = OPS_SCALE
        fs.append(Float32(1.7))
    elif k == 7:
        code = OPS_ERF
    elif k == 8:
        code = OPS_GELU
        ints.append(0)
    elif k == 9:
        code = OPS_GELU
        ints.append(1)
    elif k == 10:
        code = OPS_SOFTPLUS
    elif k == 11:
        code = OPS_SILU
    elif k == 12:
        code = OPS_LEAKY_RELU
        fs.append(Float32(0.1))
    elif k == 13:
        code = OPS_ELU
        fs.append(Float32(1.0))
    elif k == 14:
        code = OPS_HARD_SIGMOID
        fs.append(Float32(0.2))
        fs.append(Float32(0.5))
    elif k == 15:
        code = OPS_HARD_SWISH
    elif k == 16:
        code = OPS_CLIP
        fs.append(Float32(-0.5))
        fs.append(Float32(0.5))
    elif k == 17:
        code = OPS_ABS
    elif k == 18:
        code = OPS_SIN
    elif k == 19:
        code = OPS_COS
    elif k == 20:
        code = OPS_RECIPROCAL
        low = 0.5
        high = 3.0
    elif k == 21:
        code = OPS_POW
        fs.append(Float32(1.5))
        low = 0.5
        high = 3.0
    elif k == 22:
        code = OPS_FLOOR
        differentiable = False
    elif k == 23:
        code = OPS_CEIL
        differentiable = False
    elif k == 24:
        code = OPS_ROUND
        differentiable = False
    elif k == 25:
        code = OPS_SIGN
        differentiable = False
    elif k == 26:
        code = OPS_DETACH
        differentiable = False
    elif k == 27:
        code = OPS_COPY
        differentiable = False
    var x = _leaf(s, _d2(_ER, _EC), 2000 + k, low, high, differentiable)
    var hs: List[Int] = [x]
    var y = s.op(code, hs, ints, fs)
    if not differentiable:
        return _forward(s, y)
    return _finish(s, y, hs, _one(_d2(_ER, _EC)), refusal)


def _binary[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    var code = OPS_ADD
    var a_low = -2.0
    var b_low = -2.0
    var b_rows = _ER
    if k == 1:
        code = OPS_SUB
    elif k == 2:
        code = OPS_MUL
    elif k == 3:
        code = OPS_DIV
        b_low = 0.5
    elif k == 4:
        code = OPS_POW
        a_low = 0.5
    elif k == 5:
        code = OPS_MINIMUM
    elif k == 6:
        code = OPS_MAXIMUM
    elif k == 7:
        # broadcast: (70, 67) + (1, 67)
        b_rows = 1
    var a = _leaf(s, _d2(_ER, _EC), 3000 + k, a_low, 3.0)
    var b = _leaf(s, _d2(b_rows, _EC), 3100 + k, b_low, 3.0)
    var xs: List[Int] = [a, b]
    return _finish(s, _op2(s, code, a, b), xs, _two(_d2(_ER, _EC), _d2(b_rows, _EC)), refusal)


def _select[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    var a = _leaf(s, _d2(_ER, _EC), 3200 + k, -2.0, 2.0)
    var b = _leaf(s, _d2(_ER, _EC), 3300 + k, -2.0, 2.0)
    if k == 2:
        # argmax along the columns, as float values (no gradient)
        var one: List[Int] = [a]
        var arg: List[Int] = [1, 0, 1]
        return _forward(s, s.op(OPS_ARGMAX, one, arg, List[Float32]()))
    var pair: List[Int] = [a, b]
    var gt: List[Int] = [1]
    var cond = s.op(OPS_COMPARE, pair, gt, List[Float32]())
    if k == 1:
        # where(a > b and a < c, a, b): compare, logical_and
        var c = _leaf(s, _d2(_ER, _EC), 3400, 0.0, 1.0, False)
        var ac: List[Int] = [a, c]
        var lt: List[Int] = [2]
        var below = s.op(OPS_COMPARE, ac, lt, List[Float32]())
        var both: List[Int] = [cond, below]
        var conj: List[Int] = [1]
        cond = s.op(OPS_LOGICAL, both, conj, List[Float32]())
    var hs: List[Int] = [cond, a, b]
    var y = s.op(OPS_WHERE, hs, List[Int](), List[Float32]())
    return _finish(s, y, pair, _two(_d2(_ER, _EC), _d2(_ER, _EC)), refusal)


def _reduce[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    var dims = _d2(_ER, _EC)
    if k == 6 or k == 7:
        dims = _d3(6, 40, 30)
    var x = _leaf(s, dims, 4000 + k, -2.0, 2.0)
    var hs: List[Int] = [x]
    var ints = List[Int]()
    var code = OPS_SUM_N
    if k == 0:
        ints.append(-1)                 # every element
    elif k == 1:
        ints.append(1)                  # keepdims, along the columns
        ints.append(1)
    elif k == 2:
        code = OPS_MEAN_N
        ints.append(-1)
        ints.append(0)
    elif k == 3:
        code = OPS_MAX
        ints.append(-1)
        ints.append(1)
    elif k == 4:
        code = OPS_SOFTMAX
    elif k == 5:
        code = OPS_LOG_SOFTMAX
    elif k == 6:
        code = OPS_SOFTMAX_N
        ints.append(1)
    elif k == 7:
        code = OPS_LOG_SOFTMAX_N
        ints.append(-1)
    elif k == 8:
        code = OPS_CUMSUM
        ints.append(1)
        ints.append(0)
        ints.append(0)
    else:
        code = OPS_CUMSUM               # exclusive, reversed, down the rows
        ints.append(0)
        ints.append(1)
        ints.append(1)
    return _finish(s, s.op(code, hs, ints, List[Float32]()), hs, _one(dims), refusal)


def _loss[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    var rows = 300
    var classes = 70
    var mean: List[Int] = [0]
    if k == 4:
        # cross entropy: logits (rows, classes), one float class code per row
        var z = _leaf(s, _d2(rows, classes), 5004, -2.0, 2.0)
        var codes = List[Float32](length=rows, fill=0)
        for i in range(rows):
            codes[i] = Float32((i * 37 + 11) % classes)
        var c = tensor(s, codes, _d2(rows, 1), False)
        var hs: List[Int] = [z, c]
        var xs: List[Int] = [z]
        return _finish(s, s.op(OPS_CROSS_ENTROPY, hs, mean, List[Float32]()), xs, _one(_d2(rows, classes)), refusal)
    var code = OPS_MSE
    var fs = List[Float32]()
    var t_low = -2.0
    var t_high = 2.0
    if k == 1:
        code = OPS_MAE
    elif k == 2:
        code = OPS_HUBER
        fs.append(Float32(0.5))
    elif k == 3:
        code = OPS_BCE_LOGITS
        t_low = 0.0
        t_high = 1.0
    var p = _leaf(s, _d2(rows, 16), 5000 + k, -2.0, 2.0)
    var t = _leaf(s, _d2(rows, 16), 5100 + k, t_low, t_high, False)
    var hs: List[Int] = [p, t]
    var xs: List[Int] = [p]
    return _finish(s, s.op(code, hs, mean, fs), xs, _one(_d2(rows, 16)), refusal)


def _norm[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    var x = _leaf(s, _d2(130, 70), 6000 + k, -2.0, 2.0)
    var w = _leaf(s, _d2(1, 70), 6100 + k, 0.5, 1.5)
    if k == 1:
        var hs: List[Int] = [x, w]
        var eps: List[Float32] = [Float32(0.000001)]
        return _finish(s, s.op(OPS_RMS_NORM, hs, List[Int](), eps), hs, _two(_d2(130, 70), _d2(1, 70)), refusal)
    var b = _leaf(s, _d2(1, 70), 6200, -0.5, 0.5)
    var hs: List[Int] = [x, w, b]
    var eps: List[Float32] = [Float32(0.00001)]
    return _finish(s, s.op(OPS_LAYER_NORM, hs, List[Int](), eps), hs,
                   _three(_d2(130, 70), _d2(1, 70), _d2(1, 70)), refusal)


def _conv[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    var xd = _d4(2, 6, 17, 19)
    var x = _leaf(s, xd, 7000 + k, -1.0, 1.0)
    if k >= 2:
        # max pool 2 x 2 stride 2; average pool 3 x 3 stride 2
        var hs: List[Int] = [x]
        var ints: List[Int] = [2, 2]
        if k == 3:
            ints = [3, 2]
        return _finish(s, s.op(OPS_MAXPOOL2D if k == 2 else OPS_AVGPOOL2D, hs, ints, List[Float32]()), hs,
                       _one(xd), refusal)
    # 0: 3 x 3, stride 1, padding 1, one group, with bias; 1: stride 2, padding 1, two groups, no bias
    var wd = _d4(8, 6, 3, 3)
    if k == 1:
        wd = _d4(8, 3, 3, 3)
    var w = _leaf(s, wd, 7100 + k, -0.3, 0.3)
    if k == 1:
        var hs: List[Int] = [x, w]
        var ints: List[Int] = [2, 1, 2]
        return _finish(s, s.op(OPS_CONV2D, hs, ints, List[Float32]()), hs, _two(xd, wd), refusal)
    var b = _leaf(s, _d2(1, 8), 7200, -0.5, 0.5)
    var hs: List[Int] = [x, w, b]
    var ints: List[Int] = [1, 1]
    return _finish(s, s.op(OPS_CONV2D, hs, ints, List[Float32]()), hs, _three(xd, wd, _d2(1, 8)), refusal)


def _shape[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    if k == 0:
        var d = _d3(5, 30, 28)
        var x = _leaf(s, d, 8000, -2.0, 2.0)
        var hs: List[Int] = [x]
        var ints: List[Int] = [0, 2, 0, 1]          # permute axes (2, 0, 1)
        return _finish(s, s.op(OPS_PERMUTE, hs, ints, List[Float32]()), hs, _one(d), refusal)
    if k == 1:
        var d = _d3(5, 30, 28)
        var x = _leaf(s, d, 8001, -2.0, 2.0)
        var hs: List[Int] = [x]
        var ints: List[Int] = [0, 30, -1]           # reshape to (30, 140)
        return _finish(s, s.op(OPS_RESHAPE_N, hs, ints, List[Float32]()), hs, _one(d), refusal)
    if k == 2:
        var a = _leaf(s, _d2(_ER, _EC), 8002, -2.0, 2.0)
        var b = _leaf(s, _d2(_ER, 33), 8102, -2.0, 2.0)
        var hs: List[Int] = [a, b]
        var ints: List[Int] = [1]                   # concat along the columns
        return _finish(s, s.op(OPS_CONCAT, hs, ints, List[Float32]()), hs, _two(_d2(_ER, _EC), _d2(_ER, 33)), refusal)
    var cols = _EC
    if k == 3:
        cols = 66
    var x = _leaf(s, _d2(_ER, cols), 8000 + k, -2.0, 2.0)
    var hs: List[Int] = [x]
    var ints = List[Int]()
    var fs = List[Float32]()
    var code = OPS_TRANSPOSE
    if k == 3:
        code = OPS_NARROW                           # split along the columns in two, the second piece
        ints = [1, 1, 1, 2]
    elif k == 4:
        code = OPS_PAD                              # before (1, 2), after (3, 0), value 0.5
        ints = [1, 2, 3, 0]
        fs.append(Float32(0.5))
    elif k == 5:
        code = OPS_RESHAPE                          # the 2-D reshape to (67, 70)
        ints = [_EC, _ER]
    # k == 6: transpose, no words
    return _finish(s, s.op(code, hs, ints, fs), hs, _one(_d2(_ER, cols)), refusal)


def _expand[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut refusal: String) raises -> List[List[Float32]]:
    var x = _leaf(s, _d2(1, _EC), 8006, -2.0, 2.0)
    var hs: List[Int] = [x]
    var ints: List[Int] = [_ER, _EC]
    return _finish(s, s.op(OPS_EXPAND, hs, ints, List[Float32]()), hs, _one(_d2(1, _EC)), refusal)


def _extra[E: Exec & Movable & Deinitable](mut s: OpsSession[E], k: Int, mut refusal: String) raises -> List[List[Float32]]:
    if k == 0:
        # rotary position embedding: head_dim 16 over a width of 64, positions from 3, theta 10000
        var x = _leaf(s, _d2(_ER, 64), 9000, -2.0, 2.0)
        var hs: List[Int] = [x]
        var ints: List[Int] = [16, 3]
        var theta: List[Float32] = [Float32(10000.0)]
        return _finish(s, s.op(OPS_ROTARY, hs, ints, theta), hs, _one(_d2(_ER, 64)), refusal)
    if k == 1:
        # the transformer's own projection (its GEMM fold): a square weight, so either orientation fits
        var x = _leaf(s, _d2(130, 70), 9001, -1.0, 1.0)
        var w = _leaf(s, _d2(70, 70), 9101, -0.3, 0.3)
        var hs: List[Int] = [x, w]
        return _finish(s, _op2(s, OPS_TRANSFORMER_LINEAR, x, w), hs, _two(_d2(130, 70), _d2(70, 70)), refusal)
    if k == 2:
        var x = _leaf(s, _d2(_ER, _EC), 9002, -2.0, 2.0)
        var hs: List[Int] = [x]
        return _finish(s, _op1(s, OPS_TRANSFORMER_SOFTMAX, x), hs, _one(_d2(_ER, _EC)), refusal)
    # a seeded permutation of the rows and the rows gathered through it (no gradient through a gather)
    var x = _leaf(s, _d2(_ER, _EC), 9003, -2.0, 2.0, False)
    var order: List[Int] = [_ER, 7, 0, 0]              # rows, seed low, seed high, epoch
    var perm = s.op(OPS_PERMUTATION, List[Int](), order, List[Float32]())
    var hs: List[Int] = [x, perm]
    var span: List[Int] = [0, 0]                        # from the first index, every index
    return _forward(s, s.op(OPS_GATHER_ROWS, hs, span, List[Float32]()))


# ---------------------------------------------------------------- the table

def _case_name(which: Int) -> String:
    var k = which
    if k < OGI_MATMUL:
        return "matmul shape " + String(k)
    k -= OGI_MATMUL
    if k < OGI_LINEAR:
        return "linear activation " + String(k)
    k -= OGI_LINEAR
    if k < OGI_UNARY:
        return "element-wise op " + String(k)
    k -= OGI_UNARY
    if k < OGI_BINARY:
        return "binary op " + String(k)
    k -= OGI_BINARY
    if k < OGI_SELECT:
        return "where / compare / argmax " + String(k)
    k -= OGI_SELECT
    if k < OGI_REDUCE:
        return "reduction / softmax / cumsum " + String(k)
    k -= OGI_REDUCE
    if k < OGI_LOSS:
        return "loss " + String(k)
    k -= OGI_LOSS
    if k < OGI_NORM:
        return "norm " + String(k)
    k -= OGI_NORM
    if k < OGI_CONV:
        return "conv / pool " + String(k)
    k -= OGI_CONV
    if k < OGI_SHAPE:
        return "shape op " + String(k)
    k -= OGI_SHAPE
    return "rotary / transformer linear / transformer softmax / permutation + gather_rows " + String(k)


def _case[E: Exec & Movable & Deinitable](mut s: OpsSession[E], which: Int, mut refusal: String) raises -> List[List[Float32]]:
    var k = which
    if k < OGI_MATMUL:
        if k == OGI_MATMUL - 1:
            return _batched(s, refusal)
        return _matmul(s, k, refusal)
    k -= OGI_MATMUL
    if k < OGI_LINEAR:
        return _linear(s, k, refusal)
    k -= OGI_LINEAR
    if k < OGI_UNARY:
        return _unary(s, k, refusal)
    k -= OGI_UNARY
    if k < OGI_BINARY:
        return _binary(s, k, refusal)
    k -= OGI_BINARY
    if k < OGI_SELECT:
        return _select(s, k, refusal)
    k -= OGI_SELECT
    if k < OGI_REDUCE:
        return _reduce(s, k, refusal)
    k -= OGI_REDUCE
    if k < OGI_LOSS:
        return _loss(s, k, refusal)
    k -= OGI_LOSS
    if k < OGI_NORM:
        return _norm(s, k, refusal)
    k -= OGI_NORM
    if k < OGI_CONV:
        return _conv(s, k, refusal)
    k -= OGI_CONV
    if k < OGI_SHAPE:
        if k == OGI_SHAPE - 1:
            return _expand(s, refusal)
        return _shape(s, k, refusal)
    return _extra(s, k - OGI_SHAPE, refusal)


def _host_words(which: Int, mut refusal: String) raises -> List[List[Float32]]:
    var s = OpsSession[HostExec](HostExec(), True)
    try:
        var words = _case(s, which, refusal)
        s.close()
        return words^
    except error:
        s.close()
        raise error


def _device_words(which: Int, mut refusal: String) raises -> List[List[Float32]]:
    var s = OpsSession[DeviceExec](DeviceExec(), True)
    try:
        var words = _case(s, which, refusal)
        s.close()
        return words^
    except error:
        s.close()
        raise error


def _compare(which: Int) raises:
    var name = _case_name(which)
    var host_refusal = String("")
    var device_refusal = String("")
    var host = _host_words(which, host_refusal)
    var device = _device_words(which, device_refusal)
    require(host_refusal == device_refusal,
            name + ": the backward was refused differently: host `" + host_refusal + "`, device `" + device_refusal + "`")
    require(len(host) == len(device), name + ": " + String(len(host)) + " host results, " + String(len(device)) + " device results")
    for part in range(len(host)):
        var what = name + ", output"
        if part > 0:
            what = name + ", gradient of input " + String(part - 1)
        require(len(host[part]) == len(device[part]), what + ": extent " + String(len(host[part])) + " vs " + String(len(device[part])))
        for i in range(len(host[part])):
            if host[part][i].to_bits() != device[part][i].to_bits():
                raise Error(what + ": word " + String(i) + " of " + String(len(host[part])) + " is "
                            + String(device[part][i].to_bits()) + " on the device, " + String(host[part][i].to_bits())
                            + " on the host (bits)")


# The device column only: the op cases in order, the device sessions share one queue.
def test_grad_identity() raises:
    check_cases_serial("grad_identity", OGI_CASES, _compare)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
