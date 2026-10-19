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
from sequence.ops import FP, Args, add, fma3, gemm_dot, ld, mul, op_bias, op_colsum, op_gemm, st, sub, sumsq_fold
from checks.numerics import ftz, identical_div, identical_exp, identical_log, identical_sigmoid, identical_tanh
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.adafactor import AF_NORM_BLOCK, af_fold_parts
from autograd.ops import ag_fold_block
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
#: MLP_LOSSBLOCK (lane ag-pass3-lossblock, 2026-10-19; the loss-section
#:   comment block below): the per-batch seed, row-loss and L2 sums in two
#:   launches with the tape's fold rule, and the epoch loss as a blocked fold.
#:   Default on wherever MLP_BLOCKED_FOLDS is; bit version mlp-lossblock-1
#:   (new reported-loss bits). -D MOJOLEARN_MLP_LOSSBLOCK_OFF restores the
#:   MLP_BLOCKED_FOLDS kernels (op_mlp_rowloss, op_mlp_l2part, op_mlp_l2fold,
#:   op_mlp_rowpart, op_mlp_bloss, op_mlp_epoch_loss) for the A/B.
comptime MLP_LOSSBLOCK = MLP_BLOCKED_FOLDS and not is_defined["MOJOLEARN_MLP_LOSSBLOCK_OFF"]()
#: MLP_LB_ROWS (lane ag-pass5-loss, 2026-10-19; PASS5_IDEAS idea 2): the
#:   loss block's row losses one thread per row (B threads in op_mlp_lb_part,
#:   each storing its row's loss term) and the row-block sums by many threads
#:   in the fold launch, in place of ceil(B / R) part threads that each chained
#:   R rows of software logs. Same bits by construction (the loss-section
#:   comment block below says why). Default on wherever MLP_LOSSBLOCK is;
#:   -D MOJOLEARN_MLP_LOSSBLOCK_ROWS_OFF restores the pass-3 row-block threads
#:   for the A/B.
comptime MLP_LB_ROWS = MLP_LOSSBLOCK and not is_defined["MOJOLEARN_MLP_LOSSBLOCK_ROWS_OFF"]()
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


# ------------------------------------------------------------------ the loss block
#: THE PER-BATCH LOSS AND L2 BLOCK (lane ag-pass3-lossblock, 2026-10-19).
#: Shared by the hand trainer and every tape arm (sequence/mlp_fit.mojo runs
#: it between the forward and the backward of each batch). Measured on AMD
#: MI325X, HIGGS 1M x 28, (256,) b512, 1954 batches, MOJOLEARN_AG_PROFILE=2:
#: codes 26/28/90/91/92 are ~360 ms of a 1218 ms profiled tape step.
#:
#: Before (MLP_BLOCKED_FOLDS, -D MOJOLEARN_MLP_LOSSBLOCK_OFF keeps it), per
#: batch, L <= 4 layers: 5 launches (L > 4: 2 L + 3).
#:   OP_MLP_ROWLOSS (26)  B threads; chain O loss terms a row.
#:   OP_MLP_L2PART  (90)  one thread per AF_NORM_BLOCK = 4096 weights; chain
#:                        up to 4096 fmas (HIGGS: 3 threads, chains 4096,
#:                        3072, 256; 0.1 ms a launch, latency-bound).
#:   OP_MLP_L2FOLD  (91)  L threads; chain ceil(count / 4096) adds.
#:   OP_MLP_ROWPART (92)  one thread per MLP_ROW_BLOCK = 256 rows; chain up
#:                        to 256 adds (HIGGS: 2 threads).
#:   OP_MLP_BLOSS   (28)  1 thread; chain ceil(B / 256) + L adds.
#:   OP_MLP_EPOCH_LOSS (94), once an epoch: 1 thread; chain n_batches fmas
#:                        (HIGGS: 1954).
#: After (MLP_LOSSBLOCK, the default under MLP_BLOCKED_FOLDS), per batch,
#: L <= 4 layers: 2 launches (L > 4: 2 ceil(L / 4)). Every block length is
#: the tape's fold rule, autograd/ops.mojo ag_fold_block (R = the smallest
#: power of two >= 64 with R R >= extent): R and the block count both grow
#: as sqrt(extent), so no chain is long and no launch is a few threads.
#:   OP_MLP_LB_PART (1600) B O seed threads (delta = p - y, one sub each),
#:                        ceil(B / R(B)) row-block threads (chain R(B) rows
#:                        of O loss terms, the rows' losses added ascending
#:                        from +0.0), and per layer ceil(c / R(c)) weight-
#:                        block threads (chain R(c) fmas, sumsq_fold).
#:                        HIGGS: 512 + 8 (64 rows) + 56 (128 weights) + 4 (64).
#:   OP_MLP_LB_FOLD (1601) 1 thread; chain sum over layers of their block
#:                        counts + the row-block count + L adds (HIGGS:
#:                        56 + 4 + 8 + 2 = 70).
#:   Once an epoch: OP_MLP_LB_EPOCH_PART (1602), ceil(n / R(n)) threads over
#:   the n batch losses (chain R(n) fmas, HIGGS 64), and OP_MLP_LB_EPOCH_FOLD
#:   (1603), 1 thread (chain ceil(n / R(n)) adds, HIGGS 31).
#: BIT VERSION "mlp-lossblock-1": the blocked order of ||W_l||^2, of the
#:   batch's row-loss sum and of the epoch loss changes, so the REPORTED loss
#:   (the curve, loss_) gets new bits on every column together (host, Metal,
#:   NVIDIA, AMD) for any layer of more than 64 weights, any batch of more
#:   than 64 rows or any fit of more than 64 batches. The gradients do not
#:   read these sums (alpha W is OP_L2GRAD's), so the parameters move only if
#:   n_iter does (the stopping rule reads the curve). HIGGS recipe, old
#:   digests hand 2c2662ecb00cceed and tape aee3b17ef66dc060; the new ones
#:   are recorded by the orchestrator. A layer, batch and fit of at most 64
#:   each is one block: the old chain, the old bits (the 48 fits of
#:   autograd/checks/mlp_cases.mojo keep theirs).
#: Why the L2 term stays per batch: scikit-learn's reported loss adds
#:   0.5 alpha ||W||^2 / B to each batch's loss with the weights that batch
#:   saw, and the epoch loss weights each batch by B / N, so the epoch's L2
#:   part is 0.5 alpha / N times the sum over batches of ||W_b||^2: a
#:   once-an-epoch norm would report a different number. The curve feeds
#:   tol / n_iter_no_change every epoch, so the term is always consumed; it
#:   is made cheap instead (2 launches, short chains, many threads).
#: MLP_LB_ROWS (the default; -D MOJOLEARN_MLP_LOSSBLOCK_ROWS_OFF keeps the
#: layout above), per batch, L <= 4 layers: 3 launches (L > 4: 2 ceil(L / 4)
#: + 1). The pass-3 part launch's critical path was one row-block thread's
#: chain of R(B) rows of O loss terms, two software logs a term under
#: IDENTICAL (HIGGS: 128 dependent logs, 42 us a batch on AMD, 23.5 us on
#: NVIDIA); the seed threads idled after one subtract.
#:   OP_MLP_LB_PART (1600) i3 = 1: B O seed threads, then B row threads
#:                        (rowloss[row] = mlp_row_loss(row), chain O terms),
#:                        then the weight blocks as above. i3 = 2 (the data
#:                        sum comes from -D MOJOLEARN_AUTOGRAD_LOSS): no row
#:                        threads. HIGGS: 512 + 512 + 56 + 4.
#:   OP_MLP_LB_FOLD (1601) i11 = 2, once per group of four layers: ceil(B /
#:                        R(B)) row-block threads on the first group (rowparts
#:                        [k] = rowloss[k R : min((k + 1) R, B)] added
#:                        ascending from +0.0, af_fold_parts) and one thread
#:                        per layer of the group (l2[l] = its block partials
#:                        added ascending, the pass-3 fold's sum). HIGGS: 8
#:                        threads of 64 adds + 2 threads of 56 and 4.
#:   OP_MLP_LB_FOLD (1601) i11 = 3, once: 1 thread, the batch loss from the
#:                        row-block partials and l2, the pass-3 last-group
#:                        arithmetic (chain ceil(B / R) + L adds; HIGGS 10).
#: BITS: the same as MLP_LB_ROWS off, every column, every shape. A row's loss
#:   is mlp_row_loss unchanged; its value is an ftz'd fma3 result or that
#:   value negated, so the st (ftz) / ld (ftz) round trip through rowloss
#:   returns the same word. Each row block adds the same rows ascending from
#:   +0.0 with `add` (af_fold_parts is that chain with its loads staged), the
#:   blocks are mlp_lb_block(B) as before, and the final fold, the per-layer
#:   sums and the batch-loss arithmetic are the pass-3 fold's verbatim. No bit
#:   version: digests (hand, composed, fused, loss curves) are unchanged.
#:   autograd/checks/lossblock_cases.mojo runs both layouts on every case and
#:   fails on any differing word.


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


# ------------------------------------------------------------------ the tape's in-register epilogue
# Lane ag-pass3-impl (AMD phase-2 profile: the pass-2 tiled OP_AGK_GEMM_EPI
# cost 2x the plain tiled OP_GEMM per launch on MI325X, 636 vs 316 ms for the
# same five GEMMs a batch). The follower ran AFTER the store: st(C), then an
# Args record and op_bias / op_act / op_l2grad / op_act_bwd re-loading the
# word from C and storing it again (a second pass over the output and two
# Args records a thread). Here the follower's arithmetic is applied to the
# accumulator in registers and the cell is stored once. SAME BITS: the
# reloaded word was ld(st(acc)) = ftz(ftz(acc)) = acc (fma3 already flushes),
# and every other operand and operation is the follower's, in its order.
# The bodies below are op_act's and op_act_bwd's verbatim (the hand path's
# own ops are left as they are).
@always_inline
def act_value(z_in: Float32, k: Int) -> Float32:
    """op_act's body on a value: act(z), activation code k (not softmax)."""
    var z = z_in
    if k == ACT_LOGISTIC:
        z = ftz(identical_sigmoid(z))
    elif k == ACT_TANH:
        z = ftz(identical_tanh(z))
    elif k == ACT_RELU:
        z = z if z > Float32(0.0) else Float32(0.0)
    return z


@always_inline
def act_bwd_value(d_in: Float32, z: Float32, k: Int) -> Float32:
    """op_act_bwd's body on values: delta times act'(z), z the activation."""
    var d = d_in
    if k == ACT_LOGISTIC:
        d = mul(mul(d, z), sub(Float32(1.0), z))
    elif k == ACT_TANH:
        d = mul(d, sub(Float32(1.0), mul(z, z)))
    elif k == ACT_RELU:
        if z == Float32(0.0):
            d = Float32(0.0)
    return d


@always_inline
def gemm_epi_value(
    acc: Float32, ci: Int, n: Int, epi: Int, epi_arg: Int, p3: FP, f0: Float32, f1: Float32,
) -> Float32:
    """gemm_epi_tail's result for dense cell ci (column n) computed from the
    GEMM's accumulator: EPI_BIAS_ACT act(acc + p3[n]) (epi_arg the activation,
    0 none); EPI_L2GRAD (acc + f0 p3[ci]) / f1; EPI_ACT_BWD acc act'(p3[ci]);
    otherwise acc."""
    if epi == EPI_BIAS_ACT:
        var z = add(acc, ld(p3, n))
        if epi_arg != 0:
            z = act_value(z, epi_arg)
        return z
    if epi == EPI_L2GRAD:
        return div(fma3(f0, ld(p3, ci), acc), f1)
    if epi == EPI_ACT_BWD:
        return act_bwd_value(acc, ld(p3, ci), epi_arg)
    return acc


@always_inline
def op_agk_gemm_epi(t: Int, a: Args):
    """OP_AGK_GEMM_EPI's one-thread body (the launch where the tile is off:
    K under GT_MIN_K, a non-dense C, Apple, FAST; under
    MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1 also an edge under GT_MIN_EDGE;
    sequence/gemm_tiled.mojo seq_gemm_route): op_gemm's chain for cell t, then
    gemm_epi_value, one store. The same word as op_gemm_epi. Dense C
    (i8 = i1), no accumulate (i7 = 0), as the tape launches it."""
    var n_cols = a.i1
    var m = t // n_cols
    var n = t - m * n_cols
    var ci = m * a.i8 + n
    var acc = Float32(0.0)
    if a.i7 != 0:
        acc = ld(a.p2, ci)
    acc = gemm_dot(a.p0, m * a.i3, a.i4, a.p1, n * a.i6, a.i5, a.i2, acc)
    st(a.p2, ci, gemm_epi_value(acc, ci, n, a.i9, a.i10, a.p3, a.f0, a.f1))


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


# ------------------------------------------------------------------ pass-3 loss block (MLP_LOSSBLOCK)
# The kernels of the loss-section comment block above (lane
# ag-pass3-lossblock). Block lengths are mlp_lb_block = autograd/ops.mojo
# ag_fold_block, the tape's fold rule, so a row-loss sum here and the tape's
# blocked_fold (autograd/fold.mojo, AXIS_ALL) over the same B row losses add
# the same terms in the same blocks and the same order.
@always_inline
def mlp_lb_block(extent: Int) -> Int:
    """Elements per block of a loss-block fold over `extent` terms: the
    tape's rule, ag_fold_block (the smallest power of two R >= 64 with
    R * R >= extent)."""
    return ag_fold_block(max(extent, 1))


@always_inline
def mlp_lb_blocks(extent: Int) -> Int:
    """The block count of a loss-block fold over `extent` terms (0 for none)."""
    if extent <= 0:
        return 0
    var r = mlp_lb_block(extent)
    return (extent + r - 1) // r


@always_inline
def _lb_count(a: Args, g: Int) -> Int:
    """op_mlp_lb_fold: the group's layer g weight count (i6..i9, 0 = absent)."""
    if g == 0:
        return a.i6
    if g == 1:
        return a.i7
    if g == 2:
        return a.i8
    return a.i9


@always_inline
def mlp_row_loss(p: FP, y: FP, base: Int, O: Int, kind: Int) -> Float32:
    """One row's loss term, op_mlp_rowloss's chain verbatim (the same terms in
    the same order from +0.0): squared sum (y - p)^2, log -sum xlogy(y,
    clip(p)) (+ xlogy(1 - y, 1 - clip(p)) for the binary loss)."""
    var acc = Float32(0.0)
    for o in range(O):
        var pv = ld(p, base + o)
        var yv = ld(y, base + o)
        if kind == LOSS_SQUARED:
            var d = sub(yv, pv)
            acc = fma3(d, d, acc)
        else:
            var pc = _clip(pv)
            if yv != Float32(0.0):
                acc = fma3(yv, ftz(identical_log(pc)), acc)
            if kind == LOSS_BINARY_LOG:
                var ny = sub(Float32(1.0), yv)
                if ny != Float32(0.0):
                    acc = fma3(ny, ftz(identical_log(sub(Float32(1.0), pc))), acc)
    return acc if kind == LOSS_SQUARED else -acc


@always_inline
def op_mlp_lb_part(t: Int, a: Args):
    """MLP_LOSSBLOCK, the first launch of a batch's loss block. p0 the output
    [B, O], p1 the target [B, O], p2 the seed (delta = p - y), p3 the row
    block partials, p4 the parameters, p5 this launch's L2 block partials.
    i0 loss, i1 O, i2 B (0: a layer-only launch, no seed or row threads),
    i4..i11 up to four layers' (offset, count) as op_sumsq packs them
    (count 0 = absent). Threads, in order:
      [0, B O): p2[t] = p0[t] - p1[t];
      then ceil(B / R(B)) row blocks: p3[k] = the row losses of rows
        [k R, min((k + 1) R, B)) added ascending from +0.0, R = mlp_lb_block(B);
      then each layer's ceil(c / R(c)) weight blocks, layers in order:
        p5[q] = the block's squared weights, one fma each ascending from
        +0.0 (sumsq_fold), R = mlp_lb_block(c)."""
    var O = a.i1
    var B = a.i2
    var ne = B * O
    if t < ne:
        st(a.p2, t, sub(ld(a.p0, t), ld(a.p1, t)))
        return
    var nrb = mlp_lb_blocks(B)
    var r = t - ne
    if r < nrb:
        var R = mlp_lb_block(B)
        var lo = r * R
        var hi = min(lo + R, B)
        var s = Float32(0.0)
        for row in range(lo, hi):
            s = add(s, mlp_row_loss(a.p0, a.p1, row * O, O, a.i0))
        st(a.p3, r, s)
        return
    var q = r - nrb
    var rest = q
    for l in range(4):
        var oc = _layer_oc(a, l)
        var nb = mlp_lb_blocks(oc[1])
        if rest < nb:
            var R = mlp_lb_block(oc[1])
            var lo = rest * R
            st(a.p5, q, sumsq_fold(a.p4, oc[0] + lo, min(R, oc[1] - lo), 1))
            return
        rest -= nb


@always_inline
def op_mlp_lb_fold(t: Int, a: Args):
    """MLP_LOSSBLOCK, the second launch: one thread. p0 the data partials
    (i5 of them: the row blocks, or the one data sum of -D
    MOJOLEARN_AUTOGRAD_LOSS), p1 this group's L2 block partials, p2 the batch
    losses, p3 the L per-layer ||W||^2. i0 batch index, i1 B, i2 O, i3 L,
    i4 loss, i6..i9 the group's layer counts (0 = absent), i10 the group's
    first layer, i11 = 1 on the last group; f0 = 0.5 alpha.
    p3[i10 + g] = layer g's block partials added ascending from +0.0; on the
    last group p2[i0] = data + f0 sum(p3[0:L]) / B, data = (the i5 partials
    added ascending from +0.0) / (B O) / 2 (squared) or / B (log):
    op_mlp_bloss's arithmetic on the new sums."""
    var start = 0
    for g in range(4):
        var c = _lb_count(a, g)
        if c <= 0:
            break
        var nb = mlp_lb_blocks(c)
        st(a.p3, a.i10 + g, af_fold_parts(a.p1 + start, nb))
        start += nb
    if a.i11 == 0:
        return
    var B = a.i1
    var s = af_fold_parts(a.p0, a.i5)
    var data: Float32
    if a.i4 == LOSS_SQUARED:
        data = div(div(s, Float32(B * a.i2)), Float32(2.0))
    else:
        data = div(s, Float32(B))
    var qs = Float32(0.0)
    for l in range(a.i3):
        qs = add(qs, ld(a.p3, l))
    st(a.p2, a.i0, add(data, div(mul(a.f0, qs), Float32(B))))


@always_inline
def op_mlp_lb_epoch_part(t: Int, a: Args):
    """MLP_LOSSBLOCK, the epoch loss's first launch: thread t = block t of the
    i0 batch losses p0, R = i4 = mlp_lb_block(i0): p1[t] = sum over bi in
    [t R, min((t + 1) R, i0)) ascending of p0[bi] * B_bi, one fma each from
    +0.0, B_bi = min(i1, i2 - bi i1); i1 batch rows, i2 N."""
    var R = a.i4
    var lo = t * R
    var hi = min(lo + R, a.i0)
    var acc = Float32(0.0)
    for bi in range(lo, hi):
        var off = bi * a.i1
        var B = a.i1 if off + a.i1 <= a.i2 else a.i2 - off
        acc = fma3(ld(a.p0, bi), Float32(B), acc)
    st(a.p1, t, acc)


@always_inline
def op_mlp_lb_epoch_fold(t: Int, a: Args):
    """MLP_LOSSBLOCK, the epoch loss's second launch: one thread. p2[i3] =
    (the i0 partials p0 added ascending from +0.0) / i2 (N)."""
    st(a.p2, a.i3, div(af_fold_parts(a.p0, a.i0), Float32(a.i2)))
