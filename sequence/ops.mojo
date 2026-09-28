# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SEQUENCE LANE'S OPERATIONS: one scalar body per output element, the
same source compiled into a GPU kernel (`sequence/exec_device.mojo`, one
thread per element) and into a host loop (`sequence/exec.mojo::HostExec`,
ascending element order). Nothing here knows which one runs it.

IDENTICAL by construction (IDENTITY_PATHS.md "The rule"):
  * every reduction is ONE thread's ascending loop (GEMM over k, column sums
    over rows, the loss over its elements); no atomics, no tree whose shape
    depends on the launch;
  * every product is `identical_mul`, every product-plus-sum is
    `identical_mul_add` (the contraction pin), every quotient
    `identical_div`, every transcendental the `identical_*` seam;
  * every value written is passed through `ftz` (the denormal pin), and every
    operand read from a caller's buffer too;
  * no computed NaN reaches an output: the softmax subtracts the row max, the
    log reads a sum >= 1.
"""
from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_sigmoid,
    identical_sqrt,
    identical_tanh,
)

from std.sys.compile import is_defined

comptime FP = MutPointer[Float32, MutUntrackedOrigin]

#: The CPU identity gate's negative control (-D MOJOLEARN_HOST_SABOTAGE=1,
#: host binding only): the GEMM reduction runs k DESCENDING, so every trained
#: model this binary returns differs from the device's.
comptime SEQUENCE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

# ------------------------------------------------------------------ op codes
comptime OP_GEMM = 1
comptime OP_BIAS = 2
comptime OP_COLSUM = 3
comptime OP_CELL_FWD = 4
comptime OP_CELL_BWD = 5
comptime OP_GATHER_SEQ = 6
comptime OP_GATHER_ROWS = 7
comptime OP_MSE = 8
comptime OP_CE = 9
comptime OP_SUM = 10
comptime OP_OPT = 11
comptime OP_FILL = 12
comptime OP_COPY = 13
comptime OP_SEQ_OUT = 14
comptime OP_SOFTMAX = 15
comptime OP_STL = 16
comptime OP_VAR_DESIGN = 17
comptime OP_COLSCALE = 18
comptime OP_CHOLSOLVE = 19
comptime OP_ROWSCALE = 20
comptime OP_VAR_FORECAST = 21
comptime OP_SUB = 22
comptime OP_SCALE = 23
comptime OP_ACT = 24
comptime OP_ACT_BWD = 25
comptime OP_MLP_ROWLOSS = 26
comptime OP_SUMSQ = 27
comptime OP_MLP_BLOSS = 28
comptime OP_L2GRAD = 29
comptime OP_DIVS = 30
comptime OP_AF_ALPHA = 31
comptime OP_AF_ROW = 32
comptime OP_AF_COL = 33
comptime OP_AF_RMEAN = 34
comptime OP_AF_UPDATE_MAT = 35
comptime OP_AF_VEC = 36
comptime OP_AF_DENOM = 37
comptime OP_AF_APPLY = 38
comptime OP_SEG_SUMSQ = 39
comptime OP_LAMB_UPD = 40
comptime OP_LAMB_RATIO = 41
comptime OP_LAMB_APPLY = 42
comptime OP_LN_FWD = 43
comptime OP_LN_BWD_X = 44
comptime OP_LN_BWD_W = 45
comptime OP_THETA = 46
comptime OP_CROSTON = 47
comptime OP_ETS = 48
comptime OP_GARCH = 49
comptime OP_PROPHET_FEATURES = 50
comptime OP_PROPHET_FIT = 51
comptime OP_PROPHET_PREDICT = 52
comptime OP_MOE_ROUTE = 53
comptime OP_MOE_HIDDEN = 54
comptime OP_MOE_OUT = 55
comptime OP_ETS_LIK = 56
comptime OP_ETS_INIT = 57
comptime OP_CELL_FWD_H = 58
comptime OP_CELL_BWD_H = 59
comptime OP_GEMM_EPI = 60
comptime OP_COLSUM_DIV = 61
comptime OP_GEMM_EPI_TAIL = 62

# ------------------------------------------------------------------ cells
comptime CELL_RNN_TANH = 0
comptime CELL_RNN_RELU = 1
comptime CELL_LSTM = 2
comptime CELL_GRU = 3

# ------------------------------------------------------------------ optimizers
comptime OPT_SGD = 0
comptime OPT_ADAM = 1
comptime OPT_ADAMW = 2
comptime OPT_RMSPROP = 3
comptime OPT_ADAGRAD = 4
comptime OPT_SK_ADAM = 5
comptime OPT_SK_SGD = 6
comptime OPT_LION = 7
comptime OPT_ADAMAX = 8
comptime OPT_NADAM = 9


def gates_of(cell: Int) -> Int:
    if cell == CELL_LSTM:
        return 4
    if cell == CELL_GRU:
        return 3
    return 1


def dummy_ptr() -> FP:
    """A placeholder for an unused pointer slot; never dereferenced."""
    return FP(unsafe_from_address=64)


@fieldwise_init
struct Args(ImplicitlyCopyable, Movable):
    """The argument slots every operation reads: twelve buffers, twelve
    integers, eight floats. An unused slot holds a placeholder."""

    var p0: FP
    var p1: FP
    var p2: FP
    var p3: FP
    var p4: FP
    var p5: FP
    var p6: FP
    var p7: FP
    var p8: FP
    var p9: FP
    var p10: FP
    var p11: FP
    var i0: Int
    var i1: Int
    var i2: Int
    var i3: Int
    var i4: Int
    var i5: Int
    var i6: Int
    var i7: Int
    var i8: Int
    var i9: Int
    var i10: Int
    var i11: Int
    var f0: Float32
    var f1: Float32
    var f2: Float32
    var f3: Float32
    var f4: Float32
    var f5: Float32
    var f6: Float32
    var f7: Float32

    def __init__(out self):
        var d = dummy_ptr()
        self.p0 = d
        self.p1 = d
        self.p2 = d
        self.p3 = d
        self.p4 = d
        self.p5 = d
        self.p6 = d
        self.p7 = d
        self.p8 = d
        self.p9 = d
        self.p10 = d
        self.p11 = d
        self.i0 = 0
        self.i1 = 0
        self.i2 = 0
        self.i3 = 0
        self.i4 = 0
        self.i5 = 0
        self.i6 = 0
        self.i7 = 0
        self.i8 = 0
        self.i9 = 0
        self.i10 = 0
        self.i11 = 0
        self.f0 = 0.0
        self.f1 = 0.0
        self.f2 = 0.0
        self.f3 = 0.0
        self.f4 = 0.0
        self.f5 = 0.0
        self.f6 = 0.0
        self.f7 = 0.0


@always_inline
def ld(p: FP, i: Int) -> Float32:
    return ftz(p.unsafe_load(i))


@always_inline
def st(p: FP, i: Int, v: Float32):
    p.unsafe_store(i, ftz(v))


@always_inline
def mul(a: Float32, b: Float32) -> Float32:
    return ftz(identical_mul(a, b))


@always_inline
def fma3(a: Float32, b: Float32, c: Float32) -> Float32:
    return ftz(identical_mul_add(a, b, c))


@always_inline
def add(a: Float32, b: Float32) -> Float32:
    return ftz(a + b)


@always_inline
def sub(a: Float32, b: Float32) -> Float32:
    return ftz(a - b)


comptime SUMSQ_STAGE = 64
comptime GEMM_STAGE = 16
comptime COLSUM_STAGE = 32
comptime SUMSQ_VEC = 4


@always_inline
def sumsq_fold(p: FP, start: Int, n: Int, stride: Int) -> Float32:
    """sum_{k < n} p[start + k stride]^2, k ascending, one fma per term (the
    lane's one-thread reduction). The values are LOADED SUMSQ_STAGE at a time
    before they are folded (contiguous runs as 16-byte vector loads once the
    address is 16-byte aligned), so a GPU thread keeps many loads in flight
    instead of waiting out each one; the fold itself is the same chain of
    fmas in the same order, so the bits are those of the plain loop."""
    var acc = Float32(0.0)
    var k = 0
    if stride == 1:
        # scalar head up to a 16-byte boundary
        while k < n and (Int(p + (start + k)) & 15) != 0:
            var w = ld(p, start + k)
            acc = fma3(w, w, acc)
            k += 1
        while k + SUMSQ_STAGE <= n:
            var v = SIMD[DType.float32, SUMSQ_STAGE]()
            comptime for j in range(SUMSQ_STAGE // SUMSQ_VEC):
                var q = (p + (start + k + j * SUMSQ_VEC)).load[width=SUMSQ_VEC, alignment=16]()
                comptime for i in range(SUMSQ_VEC):
                    v[j * SUMSQ_VEC + i] = ftz(q[i])
            comptime for j in range(SUMSQ_STAGE):
                acc = fma3(v[j], v[j], acc)
            k += SUMSQ_STAGE
    else:
        while k + SUMSQ_STAGE <= n:
            var v = SIMD[DType.float32, SUMSQ_STAGE]()
            comptime for j in range(SUMSQ_STAGE):
                v[j] = ld(p, start + (k + j) * stride)
            comptime for j in range(SUMSQ_STAGE):
                acc = fma3(v[j], v[j], acc)
            k += SUMSQ_STAGE
    while k < n:
        var w = ld(p, start + k * stride)
        acc = fma3(w, w, acc)
        k += 1
    return acc


@always_inline
def sigm(x: Float32) -> Float32:
    return ftz(identical_sigmoid(x))


@always_inline
def tanh_(x: Float32) -> Float32:
    return ftz(identical_tanh(x))


# ------------------------------------------------------------------ bodies
@always_inline
def gemm_dot(pa: FP, abase: Int, sak: Int, pb: FP, bbase: Int, sbk: Int, K: Int, acc0: Float32) -> Float32:
    """THE GEMM REDUCTION: acc0 + sum_k pa[abase + k sak] pb[k sbk + bbase],
    k ascending, one fused multiply-add per term. `op_gemm` and the fused
    recurrent steps (`op_cell_fwd_h`, `op_cell_bwd_h`) all fold through it, so
    one seam (5500) holds every GEMM of the lane."""
    var acc = acc0
    comptime if SEQUENCE_HOST_SABOTAGE:
        var k = K - 1
        while k >= 0:
            acc = fma3(ld(pa, abase + k * sak), ld(pb, k * sbk + bbase), acc)
            k -= 1
    else:
        # GEMM_STAGE terms are LOADED before they are folded (a thread keeps
        # that many loads in flight instead of waiting out each one); the
        # fold is the same chain of fmas in the same order.
        var k = 0
        while k + GEMM_STAGE <= K:
            var va = SIMD[DType.float32, GEMM_STAGE]()
            var vb = SIMD[DType.float32, GEMM_STAGE]()
            comptime for i in range(GEMM_STAGE):
                va[i] = ld(pa, abase + (k + i) * sak)
                vb[i] = ld(pb, (k + i) * sbk + bbase)
            comptime for i in range(GEMM_STAGE):
                acc = fma3(va[i], vb[i], acc)
            k += GEMM_STAGE
        while k < K:
            acc = fma3(ld(pa, abase + k * sak), ld(pb, k * sbk + bbase), acc)
            k += 1
    return acc


def op_gemm(t: Int, a: Args):
    """C[m, n] (row stride i8) = (i7 ? C[m, n] : 0) + sum_k A(m, k) B(k, n),
    k ascending, one fused multiply-add per term. A(m, k) = p0[m*i3 + k*i4],
    B(k, n) = p1[k*i5 + n*i6]; i0=M, i1=N, i2=K."""
    var n_cols = a.i1
    var m = t // n_cols
    var n = t - m * n_cols
    var ci = m * a.i8 + n
    var acc = Float32(0.0)
    if a.i7 != 0:
        acc = ld(a.p2, ci)
    var abase = m * a.i3
    var bbase = n * a.i6
    acc = gemm_dot(a.p0, abase, a.i4, a.p1, bbase, a.i5, a.i2, acc)
    st(a.p2, ci, acc)


def op_bias(t: Int, a: Args):
    """Y[r, c] = X[r, c] + b[c]; X row stride i2, Y row stride i3, i1 = C."""
    var r = t // a.i1
    var c = t - r * a.i1
    st(a.p2, r * a.i3 + c, add(ld(a.p0, r * a.i2 + c), ld(a.p1, c)))


def op_colsum(t: Int, a: Args):
    """out[c] = (i3 ? out[c] : 0) + sum_r X[r*i2 + c], r ascending; i0 = R."""
    var acc = Float32(0.0)
    if a.i3 != 0:
        acc = ld(a.p1, t)
    # staged loads, the same adds in the same order (as gemm_dot)
    var r = 0
    while r + COLSUM_STAGE <= a.i0:
        var v = SIMD[DType.float32, COLSUM_STAGE]()
        comptime for i in range(COLSUM_STAGE):
            v[i] = ld(a.p0, (r + i) * a.i2 + t)
        comptime for i in range(COLSUM_STAGE):
            acc = add(acc, v[i])
        r += COLSUM_STAGE
    while r < a.i0:
        acc = add(acc, ld(a.p0, r * a.i2 + t))
        r += 1
    st(a.p1, t, acc)


def op_cell_fwd(t: Int, a: Args):
    """One time step of the cell for element t = b*H + u.
    p0 GX [B, G*H] (input projection + b_ih), p1 GH [B, G*H] (h @ W_hh^T +
    b_hh), p2 ACT out [B, G*H], p3 h_prev, p4 c_prev, p5 h out, p6 c out;
    i0 cell, i2 H."""
    var cell = a.i0
    var H = a.i2
    var G = gates_of(cell)
    var b = t // H
    var u = t - b * H
    var row = b * G * H
    if cell == CELL_RNN_TANH:
        var h = tanh_(add(ld(a.p0, row + u), ld(a.p1, row + u)))
        st(a.p2, row + u, h)
        st(a.p5, t, h)
    elif cell == CELL_RNN_RELU:
        var v = add(ld(a.p0, row + u), ld(a.p1, row + u))
        var h = v if v > Float32(0.0) else Float32(0.0)
        st(a.p2, row + u, h)
        st(a.p5, t, h)
    elif cell == CELL_LSTM:
        var ig = sigm(add(ld(a.p0, row + u), ld(a.p1, row + u)))
        var fg = sigm(add(ld(a.p0, row + H + u), ld(a.p1, row + H + u)))
        var gg = tanh_(add(ld(a.p0, row + 2 * H + u), ld(a.p1, row + 2 * H + u)))
        var og = sigm(add(ld(a.p0, row + 3 * H + u), ld(a.p1, row + 3 * H + u)))
        var c = fma3(fg, ld(a.p4, t), mul(ig, gg))
        var h = mul(og, tanh_(c))
        st(a.p2, row + u, ig)
        st(a.p2, row + H + u, fg)
        st(a.p2, row + 2 * H + u, gg)
        st(a.p2, row + 3 * H + u, og)
        st(a.p6, t, c)
        st(a.p5, t, h)
    else:
        # GRU, PyTorch's gate order (r, z, n) and its update
        # h' = n + z * (h - n)
        var r = sigm(add(ld(a.p0, row + u), ld(a.p1, row + u)))
        var z = sigm(add(ld(a.p0, row + H + u), ld(a.p1, row + H + u)))
        var n = tanh_(fma3(r, ld(a.p1, row + 2 * H + u), ld(a.p0, row + 2 * H + u)))
        var h = fma3(z, sub(ld(a.p3, t), n), n)
        st(a.p2, row + u, r)
        st(a.p2, row + H + u, z)
        st(a.p2, row + 2 * H + u, n)
        st(a.p5, t, h)


def op_cell_bwd(t: Int, a: Args):
    """The step's backward for element t = b*H + u.
    p0 ACT [B, G*H], p1 GH [B, G*H], p2 h_prev, p3 c_prev, p4 c_t,
    p5 dh (recurrent, in), p6 dHout[t] (from above, in), p7 dc (in/out),
    p8 dGX out, p9 dGH out, p10 dh_direct out; i0 cell, i2 H."""
    var cell = a.i0
    var H = a.i2
    var G = gates_of(cell)
    var b = t // H
    var u = t - b * H
    var row = b * G * H
    var dh = add(ld(a.p5, t), ld(a.p6, t))
    if cell == CELL_RNN_TANH:
        var h = ld(a.p0, row + u)
        var da = mul(dh, sub(Float32(1.0), mul(h, h)))
        st(a.p8, row + u, da)
        st(a.p9, row + u, da)
        st(a.p10, t, Float32(0.0))
    elif cell == CELL_RNN_RELU:
        var h = ld(a.p0, row + u)
        var da = dh if h > Float32(0.0) else Float32(0.0)
        st(a.p8, row + u, da)
        st(a.p9, row + u, da)
        st(a.p10, t, Float32(0.0))
    elif cell == CELL_LSTM:
        var ig = ld(a.p0, row + u)
        var fg = ld(a.p0, row + H + u)
        var gg = ld(a.p0, row + 2 * H + u)
        var og = ld(a.p0, row + 3 * H + u)
        var tc = tanh_(ld(a.p4, t))
        var dc = fma3(mul(dh, og), sub(Float32(1.0), mul(tc, tc)), ld(a.p7, t))
        var dog = mul(dh, tc)
        var dig = mul(dc, gg)
        var dgg = mul(dc, ig)
        var dfg = mul(dc, ld(a.p3, t))
        var dai = mul(mul(dig, ig), sub(Float32(1.0), ig))
        var daf = mul(mul(dfg, fg), sub(Float32(1.0), fg))
        var dag = mul(dgg, sub(Float32(1.0), mul(gg, gg)))
        var dao = mul(mul(dog, og), sub(Float32(1.0), og))
        st(a.p8, row + u, dai)
        st(a.p8, row + H + u, daf)
        st(a.p8, row + 2 * H + u, dag)
        st(a.p8, row + 3 * H + u, dao)
        st(a.p9, row + u, dai)
        st(a.p9, row + H + u, daf)
        st(a.p9, row + 2 * H + u, dag)
        st(a.p9, row + 3 * H + u, dao)
        st(a.p7, t, mul(dc, fg))
        st(a.p10, t, Float32(0.0))
    else:
        var r = ld(a.p0, row + u)
        var z = ld(a.p0, row + H + u)
        var n = ld(a.p0, row + 2 * H + u)
        var ghn = ld(a.p1, row + 2 * H + u)
        var dz = mul(dh, sub(ld(a.p2, t), n))
        var dn = mul(dh, sub(Float32(1.0), z))
        var dan = mul(dn, sub(Float32(1.0), mul(n, n)))
        var dar = mul(mul(mul(dan, ghn), r), sub(Float32(1.0), r))
        var daz = mul(mul(dz, z), sub(Float32(1.0), z))
        st(a.p8, row + u, dar)
        st(a.p8, row + H + u, daz)
        st(a.p8, row + 2 * H + u, dan)
        st(a.p9, row + u, dar)
        st(a.p9, row + H + u, daz)
        st(a.p9, row + 2 * H + u, mul(dan, r))
        st(a.p10, t, mul(dh, z))


def op_cell_fwd_h(t: Int, a: Args):
    """ONE LAUNCH PER TIME STEP (Apple speed, 2026-09-28): the step's hidden
    projection, its bias and the cell, which were three launches (`op_gemm`
    of h_prev @ W_hh^T into GH, `op_bias` of b_hh, `op_cell_fwd`). Element
    t = b*H + u computes GH[b, g*H + u] for its own G gates with the same
    fold (`gemm_dot`, k ascending from 0) and the same bias add, stores them
    where the GEMM did, then runs `op_cell_fwd` verbatim, which reads only
    those G columns. Same arithmetic in the same order: the same bits.
    Args as `op_cell_fwd`, plus p7 W_hh [G*H, H], p8 b_hh [G*H]."""
    var H = a.i2
    var GH = gates_of(a.i0) * H
    var b = t // H
    var u = t - b * H
    for g in range(gates_of(a.i0)):
        var n = g * H + u
        var acc = gemm_dot(a.p3, b * H, 1, a.p7, n * H, 1, H, Float32(0.0))
        st(a.p1, b * GH + n, add(ftz(acc), ld(a.p8, n)))
    op_cell_fwd(t, a)


def op_cell_bwd_h(t: Int, a: Args):
    """ONE LAUNCH PER TIME STEP of the backward: the previous (later) step's
    recurrent GEMM dh[b, u] += sum_k dGH_{s+1}[b, k] W_hh[k, u] (which was a
    launch of its own after every `op_cell_bwd`) folded into the start of
    this step's cell backward. Element t = b*H + u owns dh[t]: it folds the
    same terms in the same order (`gemm_dot`, k ascending, from the direct
    part the later step stored), stores the sum where the GEMM did, then runs
    `op_cell_bwd` verbatim. Args as `op_cell_bwd`, plus p11 W_hh, i3 = 1 when
    a later step exists (0 at s = T - 1, where dh is the zero fill), i4 the
    offset of dGH_{s+1} from p9 (B*G*H)."""
    if a.i3 != 0:
        var H = a.i2
        var GH = gates_of(a.i0) * H
        var b = t // H
        var u = t - b * H
        var acc = gemm_dot(a.p9 + a.i4, b * GH, 1, a.p11, u, H, GH, ld(a.p5, t))
        st(a.p5, t, acc)
    op_cell_bwd(t, a)


def op_gather_seq(t: Int, a: Args):
    """out[s, b, d] = X[idx[i3 + b], s, d]: batch-first rows to time-major.
    i0 T, i1 B, i2 D."""
    var T = a.i0
    var B = a.i1
    var D = a.i2
    var s = t // (B * D)
    var rem = t - s * B * D
    var b = rem // D
    var d = rem - b * D
    var row = Int(a.p1.unsafe_load(a.i3 + b))
    a.p2.unsafe_store(t, ld(a.p0, (row * T + s) * D + d))


def op_gather_rows(t: Int, a: Args):
    """out[b, o] = Y[idx[i2 + b], o]; i1 O."""
    var b = t // a.i1
    var o = t - b * a.i1
    var row = Int(a.p1.unsafe_load(a.i2 + b))
    a.p2.unsafe_store(t, ld(a.p0, row * a.i1 + o))


def op_mse(t: Int, a: Args):
    """grad = (yhat - y) * f0, sq = (yhat - y)^2."""
    var d = sub(ld(a.p0, t), ld(a.p1, t))
    st(a.p2, t, mul(d, a.f0))
    st(a.p3, t, mul(d, d))


def op_ce(t: Int, a: Args):
    """Softmax cross-entropy of row t: p0 logits [B, C], p1 labels [B] (as
    float), p2 grad [B, C] = (softmax - onehot) * f0, p3 row loss [B].
    The max is subtracted first, so no exp overflows and the log reads a sum
    >= 1; the max is the value, so a tie does not matter."""
    var C = a.i1
    var base = t * C
    var m = ld(a.p0, base)
    for c in range(1, C):
        var v = ld(a.p0, base + c)
        if v > m:
            m = v
    var s = Float32(0.0)
    for c in range(C):
        s = add(s, ftz(identical_exp(sub(ld(a.p0, base + c), m))))
    var ls = ftz(identical_log(s))
    var y = Int(a.p1.unsafe_load(t))
    st(a.p3, t, sub(ls, sub(ld(a.p0, base + y), m)))
    for c in range(C):
        var p = ftz(identical_div(ftz(identical_exp(sub(ld(a.p0, base + c), m))), s))
        if c == y:
            p = sub(p, Float32(1.0))
        st(a.p2, base + c, mul(p, a.f0))


def op_sum(t: Int, a: Args):
    """p1[i0] = (sum_{k < i1} p0[k], k ascending) * f0; one thread."""
    var acc = Float32(0.0)
    for k in range(a.i1):
        acc = add(acc, ld(a.p0, k))
    st(a.p1, a.i0, mul(acc, a.f0))


@always_inline
def lerp(s: Float32, e: Float32, w: Float32) -> Float32:
    """torch.lerp: s + w (e - s) for w < 0.5, else e - (e - s)(1 - w)
    (DEVIATION 5507: the two-branch form, one body for Adamax, NAdam and
    Adafactor)."""
    if w < Float32(0.5):
        return fma3(w, sub(e, s), s)
    return sub(e, mul(sub(e, s), sub(Float32(1.0), w)))


def op_opt(t: Int, a: Args):
    """One optimizer update of parameter t, PyTorch's rules.
    p0 param, p1 grad, p2 s1, p3 s2, p4 s3; i0 kind, i1 step (1-based),
    i2 flags (bit0 nesterov / centered, bit1 maximize-free momentum on);
    f0 lr, f1 beta1 / momentum / lr_decay, f2 beta2 / alpha, f3 eps,
    f4 weight_decay, f5 host scalar A, f6 host scalar B, f7 dampening /
    rmsprop momentum."""
    var kind = a.i0
    var p = ld(a.p0, t)
    var g = ld(a.p1, t)
    var lr = a.f0
    var wd = a.f4
    if kind == OPT_SGD:
        # torch.optim.SGD: d_p = g + wd p; buf = mu buf + (1 - damp) d_p
        # (buf = d_p on the first step); nesterov d_p = d_p + mu buf
        if wd != Float32(0.0):
            g = fma3(wd, p, g)
        var mu = a.f1
        if mu != Float32(0.0):
            var buf: Float32
            if a.i1 == 1:
                buf = g
            else:
                buf = fma3(mu, ld(a.p2, t), mul(sub(Float32(1.0), a.f7), g))
            st(a.p2, t, buf)
            if (a.i2 & 1) != 0:
                g = fma3(mu, buf, g)
            else:
                g = buf
        st(a.p0, t, fma3(-lr, g, p))
    elif kind == OPT_ADAM or kind == OPT_ADAMW:
        # f5 = lr / (1 - beta1^t), f6 = sqrt(1 - beta2^t), both host scalars
        if kind == OPT_ADAMW:
            p = mul(p, sub(Float32(1.0), mul(lr, wd)))
        elif wd != Float32(0.0):
            g = fma3(wd, p, g)
        var b1 = a.f1
        var b2 = a.f2
        var m = fma3(b1, ld(a.p2, t), mul(sub(Float32(1.0), b1), g))
        var v = fma3(b2, ld(a.p3, t), mul(mul(sub(Float32(1.0), b2), g), g))
        st(a.p2, t, m)
        st(a.p3, t, v)
        var denom = add(ftz(identical_div(ftz(identical_sqrt(v)), a.f6)), a.f3)
        st(a.p0, t, fma3(-a.f5, ftz(identical_div(m, denom)), p))
    elif kind == OPT_RMSPROP:
        # torch.optim.RMSprop: v = alpha v + (1 - alpha) g^2; centered:
        # gavg = alpha gavg + (1 - alpha) g, avg = sqrt(v - gavg^2) + eps,
        # else sqrt(v) + eps; momentum mu: buf = mu buf + g / avg,
        # p -= lr buf; else p -= lr g / avg
        if wd != Float32(0.0):
            g = fma3(wd, p, g)
        var alpha = a.f2
        var v = fma3(alpha, ld(a.p3, t), mul(mul(sub(Float32(1.0), alpha), g), g))
        st(a.p3, t, v)
        var avg: Float32
        if (a.i2 & 1) != 0:
            var ga = fma3(alpha, ld(a.p4, t), mul(sub(Float32(1.0), alpha), g))
            st(a.p4, t, ga)
            var var_ = sub(v, mul(ga, ga))
            if var_ < Float32(0.0):
                var_ = Float32(0.0)
            avg = add(ftz(identical_sqrt(var_)), a.f3)
        else:
            avg = add(ftz(identical_sqrt(v)), a.f3)
        var mu = a.f7
        if mu > Float32(0.0):
            var buf = fma3(mu, ld(a.p2, t), ftz(identical_div(g, avg)))
            st(a.p2, t, buf)
            st(a.p0, t, fma3(-lr, buf, p))
        else:
            st(a.p0, t, fma3(-lr, ftz(identical_div(g, avg)), p))
    elif kind == OPT_SK_ADAM:
        # sklearn AdamOptimizer: m, v as Adam; p += -lr_t m / (sqrt(v) + eps),
        # lr_t = lr sqrt(1 - b2^t) / (1 - b1^t) a host scalar (f5)
        var b1 = a.f1
        var b2 = a.f2
        var m = fma3(b1, ld(a.p2, t), mul(sub(Float32(1.0), b1), g))
        var v = fma3(b2, ld(a.p3, t), mul(sub(Float32(1.0), b2), mul(g, g)))
        st(a.p2, t, m)
        st(a.p3, t, v)
        var den = add(ftz(identical_sqrt(v)), a.f3)
        st(a.p0, t, add(p, ftz(identical_div(mul(-a.f5, m), den))))
    elif kind == OPT_SK_SGD:
        # sklearn SGDOptimizer: vel = mu vel - lr g; nesterov: the step is
        # mu vel - lr g with the new vel, else vel; p += step
        var mu = a.f1
        var lg = mul(lr, g)
        var vel = fma3(mu, ld(a.p2, t), -lg)
        st(a.p2, t, vel)
        var step = vel
        if (a.i2 & 1) != 0:
            step = fma3(mu, vel, -lg)
        st(a.p0, t, add(p, step))
    elif kind == OPT_LION:
        # Lion (Chen et al. 2023; lion-pytorch): p *= 1 - lr wd;
        # p -= lr sign(b1 m + (1 - b1) g); m = b2 m + (1 - b2) g
        var b1 = a.f1
        var b2 = a.f2
        var pd = mul(p, sub(Float32(1.0), mul(lr, wd)))
        var m = ld(a.p2, t)
        var c = fma3(b1, m, mul(sub(Float32(1.0), b1), g))
        var u = Float32(0.0)
        if c > Float32(0.0):
            u = Float32(1.0)
        elif c < Float32(0.0):
            u = Float32(-1.0)
        st(a.p0, t, fma3(-lr, u, pd))
        st(a.p2, t, fma3(b2, m, mul(sub(Float32(1.0), b2), g)))
    elif kind == OPT_ADAMAX:
        # torch Adamax: m.lerp_(g, 1 - b1); u = max(b2 u, |g| + eps);
        # p += -clr m / u, clr = lr / (1 - b1^t) (f5)
        if wd != Float32(0.0):
            g = fma3(wd, p, g)
        var m = lerp(ld(a.p2, t), g, sub(Float32(1.0), a.f1))
        var ub = mul(a.f2, ld(a.p3, t))
        var ga = add(abs(g), a.f3)
        var u = ub if ub > ga else ga
        st(a.p2, t, m)
        st(a.p3, t, u)
        st(a.p0, t, fma3(-a.f5, ftz(identical_div(m, u)), p))
    elif kind == OPT_NADAM:
        # torch NAdam: m.lerp_(g, 1 - b1); v = b2 v + (1 - b2) g g;
        # den = sqrt(v / bc2) + eps; p += c1 g / den; p += c2 m / den
        # (f5 bc2, f6 c1, f7 c2; flags bit0 decoupled weight decay)
        if wd != Float32(0.0):
            if (a.i2 & 1) != 0:
                p = mul(p, sub(Float32(1.0), mul(lr, wd)))
            else:
                g = fma3(wd, p, g)
        var m = lerp(ld(a.p2, t), g, sub(Float32(1.0), a.f1))
        var v = fma3(a.f2, ld(a.p3, t), mul(mul(sub(Float32(1.0), a.f2), g), g))
        st(a.p2, t, m)
        st(a.p3, t, v)
        var den = add(ftz(identical_sqrt(ftz(identical_div(v, a.f5)))), a.f3)
        var p1 = fma3(a.f6, ftz(identical_div(g, den)), p)
        st(a.p0, t, fma3(a.f7, ftz(identical_div(m, den)), p1))
    elif kind == OPT_ADAGRAD:
        # torch.optim.Adagrad: clr = lr / (1 + (t - 1) lr_decay) (f5, host);
        # sum += g^2; p -= clr g / (sqrt(sum) + eps)
        if wd != Float32(0.0):
            g = fma3(wd, p, g)
        var s = fma3(g, g, ld(a.p3, t))
        st(a.p3, t, s)
        var std = add(ftz(identical_sqrt(s)), a.f3)
        st(a.p0, t, fma3(-a.f5, ftz(identical_div(g, std)), p))


def op_fill(t: Int, a: Args):
    a.p0.unsafe_store(t, a.f0)


def op_copy(t: Int, a: Args):
    a.p1.unsafe_store(t, a.p0.unsafe_load(t))


def op_seq_out(t: Int, a: Args):
    """out[(i3 + b), s, h] = src[s, b, h]: time-major back to batch-first.
    i0 T, i1 B, i2 H."""
    var T = a.i0
    var B = a.i1
    var H = a.i2
    var b = t // (T * H)
    var rem = t - b * T * H
    var s = rem // H
    var h = rem - s * H
    a.p1.unsafe_store(((a.i3 + b) * T + s) * H + h, a.p0.unsafe_load((s * B + b) * H + h))


def op_softmax(t: Int, a: Args):
    """p1[t, :] = softmax(p0[t, :]) over i1 columns, the row max first."""
    var C = a.i1
    var base = t * C
    var m = ld(a.p0, base)
    for c in range(1, C):
        var v = ld(a.p0, base + c)
        if v > m:
            m = v
    var s = Float32(0.0)
    for c in range(C):
        s = add(s, ftz(identical_exp(sub(ld(a.p0, base + c), m))))
    for c in range(C):
        st(a.p1, base + c, ftz(identical_div(ftz(identical_exp(sub(ld(a.p0, base + c), m))), s)))
