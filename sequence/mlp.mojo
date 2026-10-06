# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MULTI-LAYER PERCEPTRON, scikit-learn's (`sklearn/neural_network/
_multilayer_perceptron.py`: `_forward_pass`, `_backprop`, `_fit_stochastic`,
`_update_no_improvement_count`; `_stochastic_optimizers.py`: `AdamOptimizer`,
`SGDOptimizer`; `_base.py`: the activations, their derivatives and the three
losses), trained on the GPU with the lane's operations. The element bodies
here and the epoch loop (`mlp_fit`) are one source for the GPU and the CPU.

The arithmetic follows the reference's statements: coef grads
(A^T delta + alpha W) / n, intercept grads mean(delta), the loss
(mean squared / 2 or the clipped log loss) plus 0.5 alpha sum(W^2) / n,
Adam's lr_t = lr sqrt(1 - b2^t) / (1 - b1^t), SGD's velocity and Nesterov
step, invscaling lr = lr0 / (t + 1)^power_t with t the samples seen, and the
adaptive division by 5. Shuffling is ours: a splitmix64 Fisher-Yates per
epoch on the host (the reference's is numpy's RandomState, which is not
restated).
"""
from sequence.ops import FP, Args, add, fma3, ld, mul, op_bias, op_colsum, op_gemm, st, sub, sumsq_fold
from checks.numerics import ftz, identical_div, identical_exp, identical_log, identical_sigmoid, identical_tanh
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.adafactor import AF_NORM_BLOCK, af_fold_parts
from std.sys.compile import is_defined

#: nr-small (2026-10-04, review item 7: "MLP fit has one-thread folds per
#: batch"). IDENTICAL, every column and the host together (the epoch loop
#: and these bodies are one source for the GPU and the CPU):
#: MLP_BLOCKED_FOLDS: ||W_l||^2 per layer is THE BLOCKED ORDER of
#:   `sequence/adafactor.mojo` (blocks of AF_NORM_BLOCK consecutive weights,
#:   each one ascending fma chain from zero, `op_mlp_l2part`, then the block
#:   partials added ascending from zero, `op_mlp_l2fold`), and the batch's
#:   row-loss sum is blocks of MLP_ROW_BLOCK rows added ascending, then the
#:   block sums ascending (`op_mlp_rowpart`, `op_mlp_bloss`). A layer of at
#:   most AF_NORM_BLOCK weights and a batch of at most MLP_ROW_BLOCK rows is
#:   one block: the old chain, the old bits; larger ones get new bits.
#:   -D MOJOLEARN_IDN_MLP_BLOCKED_FOLDS_OFF.
#: MLP_EPOCH_DEV (roadmap D13): the epoch order is `op_mlp_perm` on the
#:   executor (the x_cnn Feistel permutation, integers only) instead of a
#:   host Fisher-Yates walk and an N-word upload per epoch, and the epoch
#:   loss is `op_mlp_epoch_loss` (float32: sum of batch loss times batch
#:   rows, ascending fmas, divided by N) instead of a float64 fold on the
#:   host of every batch loss; one word comes back per epoch for the
#:   stopping rule. New shuffle order and new curve bits on every column.
#:   -D MOJOLEARN_IDN_MLP_EPOCH_DEV_OFF.
#: Both also turn off under MOJOLEARN_IDN_ALL_OFF.
comptime _MLP_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime MLP_BLOCKED_FOLDS = _MLP_IDN and not is_defined["MOJOLEARN_IDN_MLP_BLOCKED_FOLDS_OFF"]()
comptime MLP_EPOCH_DEV = _MLP_IDN and not is_defined["MOJOLEARN_IDN_MLP_EPOCH_DEV_OFF"]()
# NI52 scheduling-only subarm: leave the exact epoch Feistel mapping intact,
# but construct even the no-shuffle identity order on the selected executor.
# Avoid the otherwise unused host N-row permutation initialization. Default
# OFF, includes host executor; no convergence/identity/performance evidence yet.
comptime MLP_DEVICE_EPOCH_ORDER = MLP_EPOCH_DEV and is_defined[
    "MOJOLEARN_IDN_MLP_DEVICE_EPOCH_ORDER"
]()
comptime MLP_L2_BLOCK = AF_NORM_BLOCK
comptime MLP_ROW_BLOCK = 256

comptime ACT_IDENTITY = 0
comptime ACT_LOGISTIC = 1
comptime ACT_TANH = 2
comptime ACT_RELU = 3
comptime ACT_SOFTMAX = 4

comptime EPI_BIAS_ACT = 1
comptime EPI_L2GRAD = 2
comptime EPI_ACT_BWD = 3

comptime LOSS_SQUARED = 0
comptime LOSS_BINARY_LOG = 1
comptime LOSS_LOG = 2


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def op_act(t: Int, a: Args):
    """p0[t] = act(p0[t]); i0 activation (not softmax)."""
    var z = ld(a.p0, t)
    var k = a.i0
    if k == ACT_LOGISTIC:
        z = ftz(identical_sigmoid(z))
    elif k == ACT_TANH:
        z = ftz(identical_tanh(z))
    elif k == ACT_RELU:
        z = z if z > Float32(0.0) else Float32(0.0)
    st(a.p0, t, z)


def op_act_bwd(t: Int, a: Args):
    """p0[t] (delta) *= act'(p1[t]) written through the activation's output,
    the reference's inplace_*_derivative; i0 activation."""
    var d = ld(a.p0, t)
    var z = ld(a.p1, t)
    var k = a.i0
    if k == ACT_LOGISTIC:
        d = mul(mul(d, z), sub(Float32(1.0), z))
    elif k == ACT_TANH:
        d = mul(d, sub(Float32(1.0), mul(z, z)))
    elif k == ACT_RELU:
        if z == Float32(0.0):
            d = Float32(0.0)
    st(a.p0, t, d)


@always_inline
def _clip(p: Float32) -> Float32:
    comptime EPS: Float32 = 1.1920928955078125e-07
    if p < EPS:
        return EPS
    if p > Float32(1.0) - EPS:
        return Float32(1.0) - EPS
    return p


def op_mlp_rowloss(t: Int, a: Args):
    """Row t: p0 prediction [B, O], p1 target [B, O], p2 delta = p - y,
    p3 row loss; i0 loss, i1 O. Squared: sum (y - p)^2. Log: -sum
    xlogy(y, clip(p)) (+ xlogy(1 - y, 1 - clip(p)) for the binary loss)."""
    var O = a.i1
    var base = t * O
    var acc = Float32(0.0)
    for o in range(O):
        var p = ld(a.p0, base + o)
        var y = ld(a.p1, base + o)
        st(a.p2, base + o, sub(p, y))
        if a.i0 == LOSS_SQUARED:
            var d = sub(y, p)
            acc = fma3(d, d, acc)
        else:
            var pc = _clip(p)
            if y != Float32(0.0):
                acc = fma3(y, ftz(identical_log(pc)), acc)
            if a.i0 == LOSS_BINARY_LOG:
                var ny = sub(Float32(1.0), y)
                if ny != Float32(0.0):
                    acc = fma3(ny, ftz(identical_log(sub(Float32(1.0), pc))), acc)
    st(a.p3, t, acc if a.i0 == LOSS_SQUARED else -acc)


def op_sumsq(t: Int, a: Args):
    """One thread: p1[i0] = sum_{k < i1} p0[k]^2, k ascending. With i2 = L > 0
    (apple2: every layer in one launch, L <= 4) thread t folds layer t:
    p1[t] = sum of p0[i(4+2t) .. + i(5+2t)]^2."""
    if a.i2 > 0:
        var off = a.i4
        var cnt = a.i5
        if t == 1:
            off = a.i6
            cnt = a.i7
        elif t == 2:
            off = a.i8
            cnt = a.i9
        elif t == 3:
            off = a.i10
            cnt = a.i11
        st(a.p1, t, sumsq_fold(a.p0, off, cnt, 1))
        return
    st(a.p1, a.i0, sumsq_fold(a.p0, 0, a.i1, 1))


def op_mlp_bloss(t: Int, a: Args):
    """One thread: p2[i0] = data + reg. data = sum(p0[0:B]) / (B O) / 2
    (squared) or / B (log); reg = 0.5 alpha sum(p1[0:L]) / B.
    i1 B, i2 O, i3 L, i4 loss; f0 = 0.5 alpha. i5 > 0 (MLP_BLOCKED_FOLDS):
    sum(p0[0:B]) is the i5 block sums at p3 added ascending."""
    var B = a.i1
    var s = Float32(0.0)
    if a.i5 > 0:
        # MLP_BLOCKED_FOLDS: the i5 block sums of op_mlp_rowpart (p3)
        s = af_fold_parts(a.p3, a.i5)
    else:
        for k in range(B):
            s = add(s, ld(a.p0, k))
    var data: Float32
    if a.i4 == LOSS_SQUARED:
        data = div(div(s, Float32(B * a.i2)), Float32(2.0))
    else:
        data = div(s, Float32(B))
    var q = Float32(0.0)
    for l in range(a.i3):
        q = add(q, ld(a.p1, l))
    st(a.p2, a.i0, add(data, div(mul(a.f0, q), Float32(B))))


def op_l2grad(t: Int, a: Args):
    """p0[t] = (p0[t] + alpha p1[t]) / n; f0 alpha, f1 n."""
    st(a.p0, t, div(fma3(a.f0, ld(a.p1, t), ld(a.p0, t)), a.f1))


def op_divs(t: Int, a: Args):
    """p0[t] = p0[t] / f0."""
    st(a.p0, t, div(ld(a.p0, t), a.f0))


def op_gemm_epi(t: Int, a: Args):
    """ONE LAUNCH FOR A GEMM AND ITS ELEMENTWISE FOLLOWER (Apple speed,
    2026-09-28; each was a launch of its own): `op_gemm` for element t, then,
    on the element it just stored, the follower's body verbatim:
    i9 = EPI_BIAS_ACT: `op_bias` (p3 the bias) then `op_act` when i10 != 0;
    i9 = EPI_L2GRAD: `op_l2grad` (p3 the weights, f0 alpha, f1 n);
    i9 = EPI_ACT_BWD: `op_act_bwd` (p3 the activations, i10 the activation).
    The same values in the same order: the same bits. C must be dense
    (i8 = i1): the followers index it by t."""
    op_gemm(t, a)
    gemm_epi_tail(t, a)


@always_inline
def gemm_epi_tail(t: Int, a: Args):
    """`op_gemm_epi`'s follower alone (the host runs the GEMM on its own
    kernel, `sequence/host_gemm.mojo`, then this)."""
    var e = Args()
    e.p0 = a.p2
    e.p1 = a.p3
    if a.i9 == EPI_BIAS_ACT:
        e.p2 = a.p2
        e.i1 = a.i1
        e.i2 = a.i8
        e.i3 = a.i8
        op_bias(t, e)
        if a.i10 != 0:
            e.i0 = a.i10
            op_act(t, e)
    elif a.i9 == EPI_L2GRAD:
        e.f0 = a.f0
        e.f1 = a.f1
        op_l2grad(t, e)
    elif a.i9 == EPI_ACT_BWD:
        e.i0 = a.i10
        op_act_bwd(t, e)


def op_gemm_epi_tail(t: Int, a: Args):
    gemm_epi_tail(t, a)


def op_colsum_div(t: Int, a: Args):
    """`op_colsum` then `op_divs` of the same element by f0: one launch."""
    op_colsum(t, a)
    var e = Args()
    e.p0 = a.p1
    e.f0 = a.f0
    op_divs(t, e)


# ------------------------------------------------------------------ host RNG
struct SplitMix(Movable):
    """splitmix64: the lane's host shuffle generator, integer only."""

    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> UInt64:
        self.s += UInt64(0x9E3779B97F4A7C15)
        var z = self.s
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        return z ^ (z >> 31)

    def below(mut self, n: Int) -> Int:
        """Uniform in [0, n) by rejection (no modulo bias)."""
        var un = UInt64(n)
        var threshold = (UInt64(0) - un) % un
        while True:
            var r = self.next()
            if r >= threshold:
                return Int(r % un)


def fisher_yates(mut rng: SplitMix, mut perm: List[Float32]):
    var n = len(perm)
    var i = n - 1
    while i > 0:
        var j = rng.below(i + 1)
        var t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
        i -= 1


# ------------------------------------------------------------------ nr-small blocked folds
@always_inline
def _layer_oc(a: Args, l: Int) -> Tuple[Int, Int]:
    """Layer l's (offset, count) in op_sumsq's packing (i4..i11, l < 4)."""
    if l == 0:
        return (a.i4, a.i5)
    if l == 1:
        return (a.i6, a.i7)
    if l == 2:
        return (a.i8, a.i9)
    return (a.i10, a.i11)


def op_mlp_l2part(t: Int, a: Args):
    """MLP_BLOCKED_FOLDS. The blocks of i2 = L <= 4 layers (op_sumsq's
    (offset, count) pairs in i4..i11) concatenated in layer order, i1 = B
    weights a block: thread t is block t, p1[t] = sum of the block's
    p0[off + lo + k]^2, k ascending from zero (`sumsq_fold`)."""
    var B = a.i1
    var rest = t
    for l in range(a.i2):
        var oc = _layer_oc(a, l)
        var nb = (oc[1] + B - 1) // B
        if rest < nb:
            var lo = rest * B
            st(a.p1, t, sumsq_fold(a.p0, oc[0] + lo, min(B, oc[1] - lo), 1))
            return
        rest -= nb


def op_mlp_l2fold(t: Int, a: Args):
    """MLP_BLOCKED_FOLDS. i2 = L > 0 (packed, thread t = layer t): p1[t] =
    layer t's block partials of op_mlp_l2part (p0, layers concatenated,
    i1 = B) added ascending from zero. i2 == 0 (one layer, one thread):
    p1[i0] = p0[0:i1] added ascending from zero."""
    if a.i2 > 0:
        var B = a.i1
        var start = 0
        for l in range(t):
            start += (_layer_oc(a, l)[1] + B - 1) // B
        var nb = (_layer_oc(a, t)[1] + B - 1) // B
        st(a.p1, t, af_fold_parts(a.p0 + start, nb))
        return
    st(a.p1, a.i0, af_fold_parts(a.p0, a.i1))


def op_mlp_rowpart(t: Int, a: Args):
    """MLP_BLOCKED_FOLDS. Block t of i1 = B rows of the i0 row losses:
    p1[t] = p0[t B + k] added ascending from zero (op_mlp_bloss's chain on
    the block)."""
    var B = a.i1
    var lo = t * B
    var hi = min(lo + B, a.i0)
    var s = Float32(0.0)
    for k in range(lo, hi):
        s = add(s, ld(a.p0, k))
    st(a.p1, t, s)


# ------------------------------------------------------------------ nr-small device epoch (D13)
def mlp_epoch_key(seed: UInt64, epoch: Int) -> UInt64:
    """Epoch `epoch`'s permutation key: splitmix64's output at state
    seed + (epoch + 1) * golden (x_cnn/ops.mojo `epoch_key`, the same words)."""
    var z = seed + UInt64(epoch + 1) * UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def mlp_perm_args(n: Int, key: UInt64, dst: FP) -> Args:
    """`op_mlp_perm`'s arguments for an epoch over n rows (1 <= n < 2^24,
    the rows travel as float32 words, as the host order always did)."""
    var bits = 1
    while (1 << bits) < n:
        bits += 1
    var a = Args()
    a.p0 = dst
    a.i0 = n
    a.i1 = (bits + 1) // 2
    for w in range(4):
        var word = Int((key >> UInt64(16 * w)) & UInt64(0xFFFF))
        if w == 0:
            a.i2 = word
        elif w == 1:
            a.i3 = word
        elif w == 2:
            a.i4 = word
        else:
            a.i5 = word
    return a


@always_inline
def _mlp_perm_mix(x: UInt32) -> UInt32:
    """murmur3's 32-bit finalizer (x_cnn/ops.mojo `_perm_mix`)."""
    var z = x
    z = (z ^ (z >> 16)) * UInt32(0x85EBCA6B)
    z = (z ^ (z >> 13)) * UInt32(0xC2B2AE35)
    return z ^ (z >> 16)


def op_mlp_perm(t: Int, a: Args):
    """p0[t] = the row at position t of the epoch's order: six Feistel rounds
    on the two i1-bit halves of t, cycle-walked into [0, i0) (x_cnn/ops.mojo
    `epoch_rows_at`, the same function); i2..i5 the 64-bit key as 16-bit
    words, low first. Integers only: the same row on every column."""
    comptime if MLP_DEVICE_EPOCH_ORDER:
        if a.i6 != 0:
            st(a.p0, t, Float32(t))
            return
    var n = UInt32(a.i0)
    var h = UInt32(a.i1)
    var mask = (UInt32(1) << UInt32(h)) - UInt32(1)
    var k0 = UInt32(a.i2) | (UInt32(a.i3) << UInt32(16))
    var k1 = UInt32(a.i4) | (UInt32(a.i5) << UInt32(16))
    var x = UInt32(t)
    while True:
        var l = (x >> h) & mask
        var r = x & mask
        for rd in range(6):
            var rk = _mlp_perm_mix(k0 + UInt32(rd) * UInt32(0x9E3779B9)) ^ k1
            var f = _mlp_perm_mix(r + rk) & mask
            var tmp = l ^ f
            l = r
            r = tmp
        x = (l << h) | r
        if x < n:
            break
    a.p0.unsafe_store(t, Float32(Int(x)))


def op_mlp_epoch_loss(t: Int, a: Args):
    """One thread: p1[i3] = (sum over batches bi ascending of p0[bi] * B_bi,
    one fma each from zero) / i2, B_bi = min(i1, i2 - bi i1); i0 batches,
    i1 batch rows, i2 = N rows."""
    var acc = Float32(0.0)
    for bi in range(a.i0):
        var off = bi * a.i1
        var B = a.i1 if off + a.i1 <= a.i2 else a.i2 - off
        acc = fma3(ld(a.p0, bi), Float32(B), acc)
    st(a.p1, a.i3, div(acc, Float32(a.i2)))
