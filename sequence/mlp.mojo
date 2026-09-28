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
    """One thread: p1[i0] = sum_{k < i1} p0[k]^2, k ascending."""
    st(a.p1, a.i0, sumsq_fold(a.p0, 0, a.i1, 1))


def op_mlp_bloss(t: Int, a: Args):
    """One thread: p2[i0] = data + reg. data = sum(p0[0:B]) / (B O) / 2
    (squared) or / B (log); reg = 0.5 alpha sum(p1[0:L]) / B.
    i1 B, i2 O, i3 L, i4 loss; f0 = 0.5 alpha."""
    var B = a.i1
    var s = Float32(0.0)
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
