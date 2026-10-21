# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The tape trainers' one optimizer set (lane ag-optim, 2026-10-19).
#
# What it is: the recurrent trainer's OptConfig / opt_scalars / opt_step
# (sequence/recurrent.mojo) generalised to every kind the OP_OPT element
# body knows (sequence/ops.mojo::op_opt: torch SGD, Adam, AdamW, RMSprop,
# Adagrad, Lion, Adamax, NAdam and sklearn's Adam and SGD, the two the MLP
# trainer uses), with the moment buffers owned by the executor, a
# global-norm gradient clip and host learning-rate schedules.
#
# Why it does not import sequence.recurrent: rnn_fit calls this module under
# -D MOJOLEARN_AUTOGRAD_OPTIM, so sequence.recurrent imports it; this module
# imports only sequence.ops, sequence.exec_trait, checks.numerics,
# autograd.ops and autograd.composed_ops (sequence.ops and sequence.mlp). recurrent.mojo keeps its opt_scalars / opt_step for the A/B's
# arm A; once the define is promoted they become this module's functions.
#
# Identity: the update is ONE OP_OPT launch over the flat parameter vector
# (the same element body every sequence trainer already runs) with the
# same host scalars computed the same way, so a trainer that moves onto
# this module issues the same launch with the same argument words and gets
# the same bits. The clip, when on, is a blocked fold with a fixed order
# (block length ag_fold_block(n), a function of n alone): one launch of
# block partials of g^2 (one fused multiply-add per term, ascending), one
# launch that adds the partials ascending from +0.0 and forms the
# coefficient on the device, then one elementwise scale launch.
from sequence.exec_trait import Exec
from sequence.ops import (
    FP,
    Args,
    dummy_ptr,
    OP_FILL,
    OP_OPT,
    OPT_ADAGRAD,
    OPT_ADAM,
    OPT_ADAMAX,
    OPT_ADAMW,
    OPT_LION,
    OPT_NADAM,
    OPT_RMSPROP,
    OPT_SGD,
    OPT_SK_ADAM,
    OPT_SK_SGD,
    add,
    fma3,
    ld,
    mul,
    op_opt,
    st,
)
from std.memory import bitcast
from autograd.ops import ag_fold_block
from autograd.composed_ops import OP_AGC_OPT_FINISH, AGC_FINISH_TENSORS, agc_set_bound
from checks.numerics import ftz, identical_cos, identical_div, identical_mul, identical_pow64, identical_sqrt
from std.sys.compile import is_defined

# Lane ag-pass5-composed (2026-10-19, PASS5_IDEAS idea 5), default on: a
# trainer that finishes its raw gradients with OP_L2GRAD (weights: (g +
# alpha w) / divisor) and OP_DIVS (biases: g / divisor) before the step
# hands those ranges to launch_opt_finish / step_lr_finish instead, which
# run the finish and the update in ONE launch (autograd/composed_ops.mojo
# OP_AGC_OPT_FINISH). Bits: identical by construction (the same two bodies
# on the same words, stored, then op_opt on the stored word; the gradient
# vector ends holding the finished words, as before). Launches: 2 a
# parameter tensor pair fewer each step (the composed tape MLP: 4 a batch
# for one hidden layer). -D MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF keeps the
# separate finish launches (arm A).
comptime AUTOGRAD_OPT_FUSE = not is_defined["MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF"]()

# Operation codes of this family, a range of their own clear of the tape's
# 1000..1010 (autograd/ops.mojo); sequence/dispatch.mojo routes the range
# and asserts it overlaps no other family.
comptime OP_AGO_SUMSQ_PART = 1110
comptime OP_AGO_CLIP_COEF = 1111
comptime OP_AGO_CLIP_SCALE = 1112
comptime OP_AGO_FIRST = OP_AGO_SUMSQ_PART
comptime OP_AGO_LAST = OP_AGO_CLIP_SCALE

# Lane ag-pass6-host (2026-10-19, PASS6_IDEAS idea H(2)), default on: the
# clip's coefficient and scale run inside the update launch. After the
# OP_AGO_SUMSQ_PART partials, OP_AGH_OPT_CLIP's thread t adds the nparts
# partials ascending from +0.0 (OP_AGO_CLIP_COEF's chain, word for word),
# forms the same coefficient, stores coef * g_t (OP_AGO_CLIP_SCALE's word)
# and runs op_opt on it; thread 0 also stores [total_norm, coef] in the
# cell, as CLIP_COEF did. Bits: identical by construction (the same
# operations on the same words in the same order; each thread redoes the
# nparts ~ sqrt(n) adds). Launches: 2 fewer a step with a clip (CLIP_COEF
# and CLIP_SCALE). -D MOJOLEARN_AUTOGRAD_OPT_CLIP_FUSE_OFF keeps the three
# clip launches before OP_OPT (arm A).
comptime AUTOGRAD_OPT_CLIP_FUSE = not is_defined["MOJOLEARN_AUTOGRAD_OPT_CLIP_FUSE_OFF"]()

# The pass-6 host lane's family (2400..; sequence/dispatch.mojo routes the
# range and asserts it overlaps no other family).
comptime OP_AGH_OPT_CLIP = 2400
comptime OP_AGH_FIRST = OP_AGH_OPT_CLIP
comptime OP_AGH_LAST = OP_AGH_OPT_CLIP

# torch.nn.utils.clip_grad_norm_: coef = max_norm / (total_norm + 1e-6)
# (the certified LM clip's clip_eps, training/checks/optimizer_contract.mojo).
comptime CLIP_EPS = Float32(1.0e-6)

# Learning-rate schedule kinds (LrSchedule.kind).
comptime SCHED_CONSTANT = 0
comptime SCHED_STEP = 1
comptime SCHED_COSINE = 2


# ------------------------------------------------------------------ element bodies
@always_inline
def op_ago_sumsq_part(t: Int, a: Args):
    """p0 gradient (i0 elements), p1 partials; i1 block length B. Thread t =
    block t: p1[t] = sum over k in [t B, min((t + 1) B, n)) of g_k^2, k
    ascending from +0.0, one fused multiply-add per term."""
    var lo = t * a.i1
    var hi = min(lo + a.i1, a.i0)
    var s = Float32(0)
    for k in range(lo, hi):
        var g = ld(a.p0, k)
        s = fma3(g, g, s)
    st(a.p1, t, s)


@always_inline
def op_ago_clip_coef(t: Int, a: Args):
    """One thread. p0 partials (i0 of them), p1 cell [total_norm, coef];
    f0 max_norm. The partials are added ascending from +0.0; total_norm =
    sqrt(sum); coef = min(1, max_norm / (total_norm + 1e-6))."""
    var s = Float32(0)
    for k in range(a.i0):
        s = add(s, ld(a.p0, k))
    var norm = ftz(identical_sqrt(s))
    var coef = ftz(identical_div(a.f0, add(norm, CLIP_EPS)))
    if not (coef < Float32(1.0)):
        coef = Float32(1.0)
    st(a.p1, 0, norm)
    st(a.p1, 1, coef)


@always_inline
def op_ago_clip_scale(t: Int, a: Args):
    """p0 gradient, p1 cell [total_norm, coef]. g_t <- coef g_t, every
    element (no skip at coef = 1: the multiply's flushes are part of the
    result, as in the certified clip)."""
    st(a.p0, t, mul(ld(a.p1, 1), ld(a.p0, t)))


@always_inline
def _clip_norm_coef(parts: FP, nparts: Int, max_norm: Float32) -> Tuple[Float32, Float32]:
    """(total_norm, coef) from the partials: op_ago_clip_coef's chain."""
    var s = Float32(0)
    for k in range(nparts):
        s = add(s, ld(parts, k))
    var norm = ftz(identical_sqrt(s))
    var coef = ftz(identical_div(max_norm, add(norm, CLIP_EPS)))
    if not (coef < Float32(1.0)):
        coef = Float32(1.0)
    return (norm, coef)


@always_inline
def op_agh_opt_clip(t: Int, a: Args):
    """The clip's coefficient and scale, then op_opt, on parameter t.

    op_opt's slots (p0 params, p1 grads, p2..p4 state, i0 kind, i1 step,
    i2 flags, f0..f7) plus: p5 the sum-of-squares partials, p6 the cell
    [total_norm, coef], i3 the partial count, i4 the bit pattern of
    max_norm (every float slot is op_opt's). Thread 0 writes the cell."""
    var nc = _clip_norm_coef(a.p5, a.i3, bitcast[DType.float32](UInt32(a.i4)))
    if t == 0:
        st(a.p6, 0, nc[0])
        st(a.p6, 1, nc[1])
    st(a.p1, t, mul(nc[1], ld(a.p1, t)))
    op_opt(t, a)


@always_inline
def apply_optim_fused[OP: Int](t: Int, a: Args):
    comptime if OP == OP_AGH_OPT_CLIP:
        op_agh_opt_clip(t, a)


@always_inline
def apply_optim[OP: Int](t: Int, a: Args):
    comptime if OP == OP_AGO_SUMSQ_PART:
        op_ago_sumsq_part(t, a)
    elif OP == OP_AGO_CLIP_COEF:
        op_ago_clip_coef(t, a)
    elif OP == OP_AGO_CLIP_SCALE:
        op_ago_clip_scale(t, a)


# ------------------------------------------------------------------ configuration
@fieldwise_init
struct OptimizerConfig(ImplicitlyCopyable, Movable):
    """One parameter group's optimizer.

    kind: an OPT_* code of sequence/ops.mojo. lr: the base learning rate.
    momentum: SGD / sklearn SGD momentum, RMSprop momentum. beta1, beta2:
    the Adam family's and Lion's betas. eps: the denominator's epsilon.
    weight_decay: coupled (g += wd p) or, for AdamW, Lion and NAdam with
    `decoupled`, decoupled (p *= 1 - lr wd). nesterov: SGD's Nesterov
    step. clip: the global gradient-norm bound, <= 0 for no clip.
    dampening: SGD's. alpha: RMSprop's smoothing constant. lr_decay:
    Adagrad's. centered: RMSprop's. decoupled: NAdam's decoupled decay.
    momentum_decay: NAdam's."""

    var kind: Int
    var lr: Float32
    var momentum: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var nesterov: Bool
    var clip: Float32
    var dampening: Float32
    var alpha: Float32
    var lr_decay: Float32
    var centered: Bool
    var decoupled: Bool
    var momentum_decay: Float32

    def flags(self) -> Int:
        """OP_OPT's i2: bit 0 is Nesterov (SGD kinds), centered (RMSprop) or
        decoupled decay (NAdam)."""
        var k = self.kind
        if k == OPT_SGD or k == OPT_SK_SGD:
            return 1 if self.nesterov else 0
        if k == OPT_RMSPROP:
            return 1 if self.centered else 0
        if k == OPT_NADAM:
            return 1 if self.decoupled else 0
        return 0

    def f1(self) -> Float32:
        """OP_OPT's f1: momentum (SGD kinds), lr_decay (Adagrad), beta1."""
        var k = self.kind
        if k == OPT_SGD or k == OPT_SK_SGD:
            return self.momentum
        if k == OPT_ADAGRAD:
            return self.lr_decay
        if k == OPT_RMSPROP:
            return Float32(0.0)
        return self.beta1

    def f2(self) -> Float32:
        """OP_OPT's f2: alpha (RMSprop), else beta2."""
        if self.kind == OPT_RMSPROP:
            return self.alpha
        return self.beta2

    def f7(self) -> Float32:
        """OP_OPT's f7 before the host scalars: dampening (SGD), momentum
        (RMSprop), momentum_decay (NAdam)."""
        var k = self.kind
        if k == OPT_SGD:
            return self.dampening
        if k == OPT_RMSPROP:
            return self.momentum
        if k == OPT_NADAM:
            return self.momentum_decay
        return Float32(0.0)


def optimizer_config(kind: Int, lr: Float32) -> OptimizerConfig:
    """A configuration with every coefficient at its neutral value (no
    momentum, no decay, no clip); the caller sets the fields its kind
    reads. betas default to torch's (0.9, 0.999), eps to 1e-8, alpha to 0.99."""
    return OptimizerConfig(
        kind, lr, Float32(0.0), Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.0),
        False, Float32(0.0), Float32(0.0), Float32(0.99), Float32(0.0), False, False, Float32(0.004),
    )


def optimizer_config_packed(
    kind: Int, flags: Int, f1: Float32, f2: Float32, eps: Float32, wd: Float32, f7: Float32, lr: Float32,
) -> OptimizerConfig:
    """The configuration of sequence/recurrent.mojo's packed OptConfig
    (kind, flags, f1, f2, eps, wd, f7): each packed word lands in the
    field that packs back to it, so flags(), f1(), f2() and f7() return
    the packed words (RMSprop's f1, which OP_OPT does not read, packs as
    0)."""
    var c = optimizer_config(kind, lr)
    c.eps = eps
    c.weight_decay = wd
    var bit = (flags & 1) != 0
    if kind == OPT_SGD or kind == OPT_SK_SGD:
        c.momentum = f1
        c.nesterov = bit
        c.dampening = f7
        c.beta2 = f2
    elif kind == OPT_RMSPROP:
        c.alpha = f2
        c.momentum = f7
        c.centered = bit
    elif kind == OPT_ADAGRAD:
        c.lr_decay = f1
        c.beta2 = f2
    else:
        c.beta1 = f1
        c.beta2 = f2
        if kind == OPT_NADAM:
            c.momentum_decay = f7
            c.decoupled = bit
    return c


def optimizer_slots(cfg: OptimizerConfig) -> Tuple[Bool, Bool, Bool]:
    """Which of the three state buffers OP_OPT reads or writes for this
    configuration (sequence/pyapi.mojo::opt_slots, plus the sklearn kinds)."""
    var k = cfg.kind
    if k == OPT_SGD:
        return (cfg.momentum != Float32(0.0), False, False)
    if k == OPT_SK_SGD:
        return (True, False, False)
    if k == OPT_RMSPROP:
        return (cfg.momentum > Float32(0.0), True, cfg.centered)
    if k == OPT_ADAGRAD:
        return (False, True, False)
    if k == OPT_LION:
        return (True, False, False)
    # Adam, AdamW, Adamax, NAdam, sklearn Adam
    return (True, True, False)


# ------------------------------------------------------------------ learning-rate schedules
@fieldwise_init
struct LrSchedule(ImplicitlyCopyable, Movable):
    """kind: SCHED_CONSTANT, SCHED_STEP (lr gamma^floor((t - 1) / step_size))
    or SCHED_COSINE (min_lr + (lr - min_lr) (1 + cos(pi min(t - 1, total) /
    total)) / 2). warmup > 0 multiplies the scheduled rate by t / warmup
    for t <= warmup (linear warmup). The rate is a per-step host scalar,
    not data: it is computed here, in Mojo, and travels in the launch's
    arguments."""

    var kind: Int
    var total: Int
    var warmup: Int
    var step_size: Int
    var gamma: Float32
    var min_lr: Float32


def constant_schedule() -> LrSchedule:
    return LrSchedule(SCHED_CONSTANT, 1, 0, 1, Float32(1.0), Float32(0.0))


def scheduled_lr(s: LrSchedule, base: Float32, t: Int) -> Float32:
    """The learning rate of one-based step t."""
    var lr = base
    if s.kind == SCHED_STEP and s.step_size > 0:
        var k = (t - 1) // s.step_size
        lr = ftz(identical_mul(base, Float32(identical_pow64(Float64(s.gamma), Float64(k)))))
    elif s.kind == SCHED_COSINE and s.total > 0:
        var done = min(max(t - 1, 0), s.total)
        var frac = ftz(identical_div(Float32(done), Float32(s.total)))
        var c = ftz(identical_cos(ftz(identical_mul(Float32(3.14159265358979), frac))))
        var half = ftz(identical_mul(Float32(0.5), ftz(base - s.min_lr)))
        lr = ftz(s.min_lr + ftz(identical_mul(half, ftz(Float32(1.0) + c))))
    if s.warmup > 0 and t <= s.warmup:
        lr = ftz(identical_div(ftz(identical_mul(lr, Float32(t))), Float32(s.warmup)))
    return lr


# ------------------------------------------------------------------ state
struct OptimizerState(Movable):
    """The moment buffers on the executor (allocated once, sized by the
    slots the kind uses; an unused slot is a one-float buffer the element
    body never dereferences), the clip's partials and cell, and the host
    scalars' running state: beta1^t and beta2^t as running products (one
    identical_mul a step, so the step's bias correction is a function of
    t, the trainer running every step in order) and NAdam's mu product.
    `release` returns every buffer through Exec.free."""

    var s1: FP
    var s2: FP
    var s3: FP
    var parts: FP
    var cell: FP
    var n: Int
    var block: Int
    var nparts: Int
    var pw1: Float32
    var pw2: Float32
    var mu_prod: Float32

    def __init__(out self):
        """An empty state that owns nothing (no allocation): a placeholder a
        trainer replaces with a real state when its optimizer seam is on.
        Never stepped and never released."""
        var d = dummy_ptr()
        self.s1 = d
        self.s2 = d
        self.s3 = d
        self.parts = d
        self.cell = d
        self.n = 0
        self.block = 1
        self.nparts = 0
        self.pw1 = Float32(1.0)
        self.pw2 = Float32(1.0)
        self.mu_prod = Float32(1.0)

    def __init__[E: Exec](out self, mut ex: E, cfg: OptimizerConfig, n: Int, init_acc: Float32 = Float32(0.0)) raises:
        """`init_acc` fills the second state buffer (Adagrad's
        initial_accumulator_value), as rnn_fit does."""
        var used = optimizer_slots(cfg)
        var n2 = n if used[1] else 1
        var acquired = List[FP]()
        var block = ag_fold_block(n)
        var nparts = (n + block - 1) // block
        var clip = cfg.clip > Float32(0.0)
        try:
            var s2 = ex.alloc(n2)
            acquired.append(s2)
            if init_acc != Float32(0.0):
                var f = Args()
                f.p0 = s2
                f.f0 = init_acc
                ex.launch[OP_FILL](f, n2)
            var s1 = ex.alloc(n if used[0] else 1)
            acquired.append(s1)
            var s3 = ex.alloc(n if used[2] else 1)
            acquired.append(s3)
            var parts = ex.alloc(nparts if clip else 1)
            acquired.append(parts)
            var cell = ex.alloc(2)
            acquired.append(cell)
            self.s1 = s1
            self.s2 = s2
            self.s3 = s3
            self.parts = parts
            self.cell = cell
        except error:
            for buffer in acquired:  # driver-loop: over acquired optimizer state buffers
                ex.free(buffer)
            raise error
        self.n = n
        self.block = block
        self.nparts = nparts
        self.pw1 = Float32(1.0)
        self.pw2 = Float32(1.0)
        self.mu_prod = Float32(1.0)

    def release[E: Exec](mut self, mut ex: E) raises:
        ex.free(self.s1)
        ex.free(self.s2)
        ex.free(self.s3)
        ex.free(self.parts)
        ex.free(self.cell)


def optimizer_scalars(
    cfg: OptimizerConfig, mut st: OptimizerState, t: Int, lr: Float32,
) -> Tuple[Float32, Float32, Float32]:
    """(f5, f6, f7) of step t (one-based), advancing the running state.
    The torch kinds are sequence/recurrent.mojo::opt_scalars word for word;
    sklearn Adam is sequence/mlp_fit.mojo's lr_t = lr sqrt(1 - b2^t) /
    (1 - b1^t), word for word."""
    var f5 = Float32(0.0)
    var f6 = Float32(0.0)
    var f7 = cfg.f7()
    var b1 = cfg.f1()
    var b2 = cfg.f2()
    var k = cfg.kind
    if k == OPT_ADAM or k == OPT_ADAMW or k == OPT_ADAMAX or k == OPT_NADAM or k == OPT_SK_ADAM:
        st.pw1 = ftz(identical_mul(st.pw1, b1))
        st.pw2 = ftz(identical_mul(st.pw2, b2))
    if k == OPT_ADAM or k == OPT_ADAMW:
        f5 = ftz(identical_div(lr, Float32(1.0) - st.pw1))
        f6 = ftz(identical_sqrt(Float32(1.0) - st.pw2))
    elif k == OPT_SK_ADAM:
        f5 = ftz(identical_div(ftz(identical_mul(lr, ftz(identical_sqrt(Float32(1.0) - st.pw2)))),
                               Float32(1.0) - st.pw1))
    elif k == OPT_ADAMAX:
        f5 = ftz(identical_div(lr, Float32(1.0) - st.pw1))
    elif k == OPT_NADAM:
        # torch _single_tensor_nadam (see sequence/recurrent.mojo::opt_scalars)
        var md = Float64(f7)
        var b164 = Float64(b1)
        var mu = b164 * (1.0 - 0.5 * identical_pow64(0.96, Float64(t) * md))
        var mu_next = b164 * (1.0 - 0.5 * identical_pow64(0.96, Float64(t + 1) * md))
        st.mu_prod = ftz(identical_mul(st.mu_prod, Float32(mu)))
        var mp = Float64(st.mu_prod)
        var lr64 = Float64(lr)
        f5 = Float32(1.0) - st.pw2
        f6 = Float32(-lr64 * (1.0 - mu) / (1.0 - mp))
        f7 = Float32(-lr64 * mu_next / (1.0 - mp * mu_next))
    elif k == OPT_ADAGRAD:
        f5 = ftz(identical_div(lr, Float32(1.0) + ftz(identical_mul(Float32(t - 1), b1))))
    return (f5, f6, f7)


# ------------------------------------------------------------------ the step
def clip_grad_norm[E: Exec](mut ex: E, max_norm: Float32, mut state: OptimizerState, grads: FP, n: Int) raises:
    """grads *= min(1, max_norm / (||grads|| + 1e-6)) over the flat vector:
    a two-launch blocked fold for the norm (state.cell[0]) and the
    coefficient (state.cell[1]), then one elementwise scale. Nothing is
    read back to the host."""
    if n != state.n:
        raise Error("autograd optim: the clip's vector length differs from the state's")
    var p = Args()
    p.p0 = grads
    p.p1 = state.parts
    p.i0 = n
    p.i1 = state.block
    ex.launch[OP_AGO_SUMSQ_PART](p, state.nparts)
    var c = Args()
    c.p0 = state.parts
    c.p1 = state.cell
    c.i0 = state.nparts
    c.f0 = max_norm
    ex.launch[OP_AGO_CLIP_COEF](c, 1)
    var s = Args()
    s.p0 = grads
    s.p1 = state.cell
    ex.launch[OP_AGO_CLIP_SCALE](s, n)


def step_lr[E: Exec](
    mut ex: E, cfg: OptimizerConfig, mut state: OptimizerState, params: FP, grads: FP, n: Int, t: Int, lr: Float32,
) raises:
    """One update of the n parameters at one-based step t with learning
    rate lr: the clip (cfg.clip > 0), then ONE OP_OPT launch. The trainer
    calls it every step in order (the running bias-correction products)."""
    if n != state.n:
        raise Error("autograd optim: the parameter count differs from the state's")
    var fuse_clip = AUTOGRAD_OPT_CLIP_FUSE and cfg.clip > Float32(0.0)
    if cfg.clip > Float32(0.0):
        if fuse_clip:
            # the partials only; the coefficient and scale run in the update
            var p = Args()
            p.p0 = grads
            p.p1 = state.parts
            p.i0 = n
            p.i1 = state.block
            ex.launch[OP_AGO_SUMSQ_PART](p, state.nparts)
        else:
            clip_grad_norm(ex, cfg.clip, state, grads, n)
    var a = Args()
    a.p0 = params
    a.p1 = grads
    a.p2 = state.s1
    a.p3 = state.s2
    a.p4 = state.s3
    a.i0 = cfg.kind
    a.i1 = t
    a.i2 = cfg.flags()
    a.f0 = lr
    a.f1 = cfg.f1()
    a.f2 = cfg.f2()
    a.f3 = cfg.eps
    a.f4 = cfg.weight_decay
    var sc = optimizer_scalars(cfg, state, t, lr)
    a.f5 = sc[0]
    a.f6 = sc[1]
    a.f7 = sc[2]
    if fuse_clip:
        a.p5 = state.parts
        a.p6 = state.cell
        a.i3 = state.nparts
        a.i4 = Int(bitcast[DType.uint32](cfg.clip))
        ex.launch[OP_AGH_OPT_CLIP](a, n)
        return
    ex.launch[OP_OPT](a, n)


def step[E: Exec](
    mut ex: E, cfg: OptimizerConfig, mut state: OptimizerState, params: FP, grads: FP, n: Int, t: Int,
) raises:
    """`step_lr` at the configuration's own learning rate."""
    step_lr(ex, cfg, state, params, grads, n, t, cfg.lr)


# ------------------------------------------------------------------ fused gradient finish
@fieldwise_init
struct GradFinish(Copyable, Movable):
    """The finish a trainer applies to its raw gradient vector before the
    step, by parameter tensor: tensor l's weights run from the previous
    tensor's bias end (0 for the first) to bias_start[l] and take
    (g + alpha w) / divisor; its biases run to bias_end[l] and take
    g / divisor. The tensors tile [0, bias_end[last]) in order."""

    var bias_start: List[Int]
    var bias_end: List[Int]
    var alpha: Float32
    var divisor: Float32


def launch_opt_finish[E: Exec](
    mut ex: E, o: Args, fin: GradFinish, n: Int, shift_s1: Bool, shift_s2: Bool, shift_s3: Bool,
) raises:
    """The finish and the OP_OPT update `o` (op_opt's arguments over the n
    parameters) as OP_AGC_OPT_FINISH launches, one per group of up to
    AGC_FINISH_TENSORS tensors. A group's launch covers its own range of
    the vectors: the parameter and gradient pointers, and each state
    pointer whose shift flag is set (a buffer of n words), start at the
    group's first word. Only the sklearn kinds (OPT_SK_ADAM, OPT_SK_SGD)."""
    if o.i0 != OPT_SK_ADAM and o.i0 != OPT_SK_SGD:
        raise Error("autograd optim: the fused gradient finish takes the sklearn Adam or SGD step")
    var L = len(fin.bias_start)
    if L < 1 or len(fin.bias_end) != L or fin.bias_end[L - 1] != n:
        raise Error("autograd optim: the gradient finish's tensors must tile the parameter vector")
    var first = 0
    var base = 0
    while first < L:  # driver-loop: over groups of parameter tensors
        var count = min(AGC_FINISH_TENSORS, L - first)
        var a = o
        a.p0 = o.p0 + base
        a.p1 = o.p1 + base
        if shift_s1:
            a.p2 = o.p2 + base
        if shift_s2:
            a.p3 = o.p3 + base
        if shift_s3:
            a.p4 = o.p4 + base
        a.i3 = count
        for j in range(count):  # driver-loop: over the group's tensors
            agc_set_bound(a, 2 * j, fin.bias_start[first + j] - base)
            agc_set_bound(a, 2 * j + 1, fin.bias_end[first + j] - base)
        a.f6 = fin.alpha
        a.f7 = fin.divisor
        var end = fin.bias_end[first + count - 1]
        ex.launch[OP_AGC_OPT_FINISH](a, end - base)
        base = end
        first += count


def step_lr_finish[E: Exec](
    mut ex: E, cfg: OptimizerConfig, mut state: OptimizerState, params: FP, grads: FP, n: Int, t: Int,
    lr: Float32, fin: GradFinish,
) raises:
    """`step_lr` on raw gradients that `fin` finishes in the same launch
    (launch_opt_finish): the same words as finishing them first and then
    calling step_lr. No clip (a clip reads the finished gradients)."""
    if n != state.n:
        raise Error("autograd optim: the parameter count differs from the state's")
    if cfg.clip > Float32(0.0):
        raise Error("autograd optim: the fused gradient finish does not clip")
    var a = Args()
    a.p0 = params
    a.p1 = grads
    a.p2 = state.s1
    a.p3 = state.s2
    a.p4 = state.s3
    a.i0 = cfg.kind
    a.i1 = t
    a.i2 = cfg.flags()
    a.f0 = lr
    a.f1 = cfg.f1()
    a.f2 = cfg.f2()
    a.f3 = cfg.eps
    a.f4 = cfg.weight_decay
    var sc = optimizer_scalars(cfg, state, t, lr)
    a.f5 = sc[0]
    var used = optimizer_slots(cfg)
    launch_opt_finish(ex, a, fin, n, used[0], used[1], used[2])
