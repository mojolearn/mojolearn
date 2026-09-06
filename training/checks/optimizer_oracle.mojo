# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IDENTICAL FP32 optimizer step, written out, on the host. This file's own `opt_refuse_bad_inputs` is now what the device entry point calls, so both sides fail with the same name (DEVIATION 1496)."""

from gemm.checks.gemm_oracle import OP_NT, contract_leaf_size, gemm_oracle
from checks.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)



comptime OPT_SGD = 0
comptime OPT_ADAM = 1
comptime OPT_ADAMW = 2

comptime CLIP_EPS_BITS: UInt32 = 0x358637BD

comptime ONE_BITS: UInt32 = 0x3F800000

comptime QNAN_BITS: UInt32 = 0x7FC00000

comptime INF_BITS: UInt32 = 0x7F800000


def clip_eps() -> Float32:
    """The profile's `CLIP_EPS`, from its bit pattern rather than from a decimal literal (`[[mojo-string-float-roundtrip]]`)."""
    from std.memory import bitcast

    return bitcast[DType.float32](CLIP_EPS_BITS)




def refuse_nonfinite(name: String, values: List[Float32]) raises:
    """Row 39. A NaN or infinity in a gradient, a parameter or a state is REFUSED BY NAME before any recorded stage."""
    from std.memory import bitcast

    for i in range(len(values)):
        var au = bitcast[DType.uint32](values[i]) & UInt32(0x7FFFFFFF)
        if au > INF_BITS:
            raise Error(
                String("optimizer: NaN in ")
                + name
                + String(" at flat index ")
                + String(i)
                + String(" REFUSED (row 39: NaN payloads are vendor-shaped")
                + String("; no stage may record one)")
            )
        if au == INF_BITS:
            raise Error(
                String("optimizer: infinity in ")
                + name
                + String(" at flat index ")
                + String(i)
                + String(" REFUSED (row 39)")
            )


def refuse_nonfinite_scalar(name: String, v: Float32) raises:
    """`refuse_nonfinite` for one value."""
    var one = List[Float32]()
    one.append(v)
    refuse_nonfinite(name, one)




def pow_int_f32(base: Float32, t: Int) -> Float32:
    """DEVIATION 1171. What it refuses is `b_t = b_{t-1} * beta`, which is what an implementation reaches for because it is one multiply per step."""
    from std.memory import bitcast

    var acc = bitcast[DType.float32](ONE_BITS)
    var b = base
    var e = t
    while e > 0:
        if (e & 1) != 0:
            acc = ftz(identical_mul(acc, b))
        b = ftz(identical_mul(b, b))
        e = e >> 1
    return acc




@fieldwise_init
struct OptimizerConfig(Copyable, Movable):
    """One parameter group's configuration. `max_norm <= 0.0` means the gradient-norm clip is OFF, and that is the configuration contract 11(c)'s parameter-count-invariance gate must run under -- with clipping ON, one parameter's update depends on every other parameter in the model, by the reference's own semantics and not by a defect (contract 3.5)."""

    var kind: Int
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var momentum: Float32
    var dampening: Float32
    var nesterov: Bool
    var max_norm: Float32


@fieldwise_init
struct StepScalars(Copyable, Movable):
    """The host scalars of contract 7.1, computed ONCE per step per parameter group."""

    var b1t: Float32
    var b2t: Float32
    var bc1: Float32
    var bc2: Float32
    var step_size: Float32
    var rt_bc2: Float32
    var c1: Float32
    var c2: Float32
    var decay_mul: Float32
    var neg_lr: Float32
    var c_damp: Float32


def step_scalars(cfg: OptimizerConfig, t: Int) -> StepScalars:
    """Contract 7.1, line for line."""
    from std.memory import bitcast

    var one = bitcast[DType.float32](ONE_BITS)

    var b1t = pow_int_f32(cfg.beta1, t)
    var b2t = pow_int_f32(cfg.beta2, t)
    var bc1 = ftz(one - b1t)
    var bc2 = ftz(one - b2t)
    var step_size = ftz(identical_div(cfg.lr, bc1))
    var rt_bc2 = ftz(identical_sqrt(bc2))
    var c1 = ftz(one - cfg.beta1)
    var c2 = ftz(one - cfg.beta2)
    var decay_mul = ftz(one - ftz(identical_mul(cfg.lr, cfg.weight_decay)))
    var neg_lr = -cfg.lr
    var c_damp = ftz(one - cfg.dampening)

    return StepScalars(
        b1t,
        b2t,
        bc1,
        bc2,
        step_size,
        rt_bc2,
        c1,
        c2,
        decay_mul,
        neg_lr,
        c_damp,
    )




def microbatch_split_is_identical(t_tokens: Int, a: Int) -> Bool:
    """Contract clause 9.2. The cross-microbatch combination must be the v1 BALANCED TREE over the `a` pieces in ascending microbatch index, `ftz(ftz(x) + ftz(y))` at every node -- **not a running serial sum.** Conditions 3 and 4 together are what make each microbatch's leaf range a COMPLETE SUBTREE."""
    if a <= 0 or t_tokens <= 0:
        return False
    if t_tokens % a != 0:
        return False
    var leaf_full = contract_leaf_size(t_tokens)
    var leaf_piece = contract_leaf_size(t_tokens // a)
    if leaf_full != leaf_piece:
        return False
    if leaf_full <= 0:
        return False
    if t_tokens % leaf_full != 0:
        return False
    var p_count = t_tokens // leaf_full
    if p_count % a != 0:
        return False
    var q = a
    while q > 1:
        if q % 2 != 0:
            return False
        q = q // 2
    return True




def _slice(xs: List[Float32], begin: Int, count: Int) -> List[Float32]:
    """One tensor's elements as their own `List`, because `gemm_oracle` takes a whole operand."""
    var out = List[Float32]()
    for i in range(count):
        out.append(xs[begin + i])
    return out^


def clip_tensor_sumsq_oracle(
    grads: List[Float32], begin: Int, count: Int
) -> Float32:
    """DEVIATION 1178, contract clause 3.2. There `P == 1`, the tree has no arithmetic node, and the v1 answer IS the serial ascending chain, so a hand-written serial fold passes."""
    var g = _slice(grads, begin, count)
    var out = gemm_oracle(g, g, OP_NT, 1, 1, count)
    if len(out) == 0:
        return Float32(0.0)
    return out[0]


def clip_coefficient(total_norm: Float32, max_norm: Float32) raises -> Float32:
    """Contract clauses 3.4a and 3.4b. **This is the ONLY compare-select in the whole profile** (contract 8c), and it is on a value `refuse_nonfinite_scalar` has already established is finite."""
    from std.memory import bitcast

    refuse_nonfinite_scalar(String("clip.total_norm"), total_norm)
    var one = bitcast[DType.float32](ONE_BITS)
    var denom = ftz(total_norm + clip_eps())
    var coef = ftz(identical_div(max_norm, denom))
    if coef < one:
        return coef
    return one


def clip_grad_norm_oracle(
    mut grads: List[Float32],
    offsets: List[Int],
    max_norm: Float32,
    mut sumsq_out: List[Float32],
    mut norm_out: List[Float32],
    mut total_out: List[Float32],
) raises -> Float32:
    """DEVIATIONS 1178, 1179, 1180. **`j` IS the `param_id`**, and the ascending order of `j` is the cross-tensor summation order."""
    var j_count = len(offsets) - 1
    if j_count <= 0:
        return Float32(0.0)

    for j in range(j_count):
        var begin = offsets[j]
        var count = offsets[j + 1] - begin
        var s = ftz(clip_tensor_sumsq_oracle(grads, begin, count))
        sumsq_out.append(s)
        norm_out.append(ftz(identical_sqrt(s)))

    var norms_copy = _slice(norm_out, 0, len(norm_out))
    var tot = gemm_oracle(norms_copy, norms_copy, OP_NT, 1, 1, j_count)
    var total_sumsq = Float32(0.0)
    if len(tot) > 0:
        total_sumsq = ftz(tot[0])
    var total_norm = ftz(identical_sqrt(total_sumsq))

    var coef = clip_coefficient(total_norm, max_norm)

    total_out.append(total_sumsq)
    total_out.append(total_norm)
    total_out.append(coef)

    for i in range(len(grads)):
        grads[i] = ftz(identical_mul(coef, ftz(grads[i])))

    return coef




@fieldwise_init
struct AdamElement(Copyable, Movable):
    """One element's results, including the two intermediates the card records (`adam.denom` and `adam.q`)."""

    var p: Float32
    var m: Float32
    var v: Float32
    var denom: Float32
    var q: Float32


def adam_element_oracle(
    p_in: Float32,
    g_in: Float32,
    m_in: Float32,
    v_in: Float32,
    cfg: OptimizerConfig,
    sc: StepScalars,
) -> AdamElement:
    """Contract 7.2, seams O1 through O14, in order. The fixture must be BUILT to separate, which means `step_size * q` and `p` must be within a few binades of each other with the product's tail nonzero."""
    var g = ftz(g_in)  # O1
    var p = ftz(p_in)  # O2
    var mp = ftz(m_in)  # O3
    var vp = ftz(v_in)  # O3

    if cfg.weight_decay != Float32(0.0):
        if cfg.kind == OPT_ADAMW:
            p = ftz(identical_mul(sc.decay_mul, p))
        else:
            g = ftz(identical_mul_add(cfg.weight_decay, p, g))

    var ms = ftz(identical_mul(cfg.beta1, mp))  # O5, PRODUCT
    var m = ftz(identical_mul_add(sc.c1, g, ms))  # O6, FUSED
    var g2 = ftz(identical_mul(g, g))  # O7, PRODUCT
    var vs = ftz(identical_mul(cfg.beta2, vp))  # O8, PRODUCT
    var v = ftz(identical_mul_add(sc.c2, g2, vs))  # O9, FUSED

    var s = ftz(identical_sqrt(v))  # O10
    var sd = ftz(identical_div(s, sc.rt_bc2))  # O11
    var dn = ftz(sd + cfg.eps)  # O12, eps OUTSIDE the sqrt
    var q = ftz(identical_div(m, dn))  # O13, a TRUE divide
    var p_out = ftz(identical_mul_add(-sc.step_size, q, p))  # O14, FUSED

    return AdamElement(p_out, m, v, dn, q)




@fieldwise_init
struct SgdElement(Copyable, Movable):
    """One element's results."""

    var p: Float32
    var buf: Float32
    var direction: Float32


def sgd_element_oracle(
    p_in: Float32,
    g_in: Float32,
    buf_in: Float32,
    buf_initialized: Bool,
    cfg: OptimizerConfig,
    sc: StepScalars,
) -> SgdElement:
    """Contract 7.3, seams S1 through S5. *What makes a sabotage of this inert*: at `dampening = 0.0`, `c_damp` is exactly `1.0` and `identical_mul(1.0, g)` returns `g` for every finite `g`, **so the default configuration cannot see this clause at all.** The fixture must set `dampening != 0` and must compare at `t = 1`, then at `t = 2` and beyond to show the divergence persists rather than washing out."""
    var g = ftz(g_in)  # S1
    var p = ftz(p_in)  # S2

    if cfg.weight_decay != Float32(0.0):
        g = ftz(identical_mul_add(cfg.weight_decay, p, g))  # S3, FUSED

    var b = ftz(buf_in)
    if cfg.momentum != Float32(0.0):
        if not buf_initialized:
            b = g
        else:
            var bs = ftz(identical_mul(cfg.momentum, b))  # PRODUCT
            b = ftz(identical_mul_add(sc.c_damp, g, bs))  # FUSED
        if cfg.nesterov:
            g = ftz(identical_mul_add(cfg.momentum, b, g))  # 7.3c, FUSED
        else:
            g = b

    var p_out = ftz(identical_mul_add(sc.neg_lr, g, p))  # S5, FUSED
    return SgdElement(p_out, b, g)




struct OptimizerStages(Movable):
    """Every recorded stage of one step, in the card's order (contract section 10)."""

    var clip_sumsq: List[Float32]  # [J]
    var clip_norm: List[Float32]  # [J]
    var clip_total: List[Float32]  # [3] total_sumsq, total_norm, coef
    var clip_grad: List[Float32]  # [sum N_j], the rescaled gradient
    var sched: List[Float32]  # [7] see `sched_field_name`
    var adam_m: List[Float32]  # [sum N_j]
    var adam_v: List[Float32]
    var adam_denom: List[Float32]
    var adam_q: List[Float32]
    var sgd_buf: List[Float32]
    var sgd_dir: List[Float32]
    var param_out: List[Float32]

    def __init__(out self):
        self.clip_sumsq = List[Float32]()
        self.clip_norm = List[Float32]()
        self.clip_total = List[Float32]()
        self.clip_grad = List[Float32]()
        self.sched = List[Float32]()
        self.adam_m = List[Float32]()
        self.adam_v = List[Float32]()
        self.adam_denom = List[Float32]()
        self.adam_q = List[Float32]()
        self.sgd_buf = List[Float32]()
        self.sgd_dir = List[Float32]()
        self.param_out = List[Float32]()


def sched_field_name(i: Int) -> String:
    """The `sched` stage's field order, so a card reader does not have to count."""
    if i == 0:
        return String("sched.pow1")
    if i == 1:
        return String("sched.pow2")
    if i == 2:
        return String("sched.bc1")
    if i == 3:
        return String("sched.bc2")
    if i == 4:
        return String("sched.step_size")
    if i == 5:
        return String("sched.rt_bc2")
    if i == 6:
        return String("sched.decay_mul")
    return String("sched.?")


def optimizer_step_oracle(
    mut param: List[Float32],
    mut grad: List[Float32],
    mut m_state: List[Float32],
    mut v_state: List[Float32],
    mut buf_initialized: List[Bool],
    offsets: List[Int],
    cfg: OptimizerConfig,
    t: Int,
) raises -> OptimizerStages:
    """**THE NORMATIVE ANSWER of `mojolearn.identical.optimizer.fp32.v1`.** One step, in order --- refuse, clip, host scalars, then the elementwise update. offsets[j+1]` being tensor `j`; `j` IS the `param_id` and its ascending order is the cross-tensor summation order of clause 3.3."""
    var stages = OptimizerStages()
    var j_count = len(offsets) - 1
    if j_count <= 0:
        return stages^

    refuse_nonfinite(String("input.param"), param)
    refuse_nonfinite(String("input.grad"), grad)
    refuse_nonfinite(String("state.m"), m_state)
    refuse_nonfinite(String("state.v"), v_state)

    if cfg.max_norm > Float32(0.0):
        var sq = List[Float32]()
        var nm = List[Float32]()
        var tt = List[Float32]()
        _ = clip_grad_norm_oracle(grad, offsets, cfg.max_norm, sq, nm, tt)
        for i in range(len(grad)):
            stages.clip_grad.append(grad[i])
        stages.clip_sumsq = sq^
        stages.clip_norm = nm^
        stages.clip_total = tt^

    var sc = step_scalars(cfg, t)
    stages.sched.append(sc.b1t)
    stages.sched.append(sc.b2t)
    stages.sched.append(sc.bc1)
    stages.sched.append(sc.bc2)
    stages.sched.append(sc.step_size)
    stages.sched.append(sc.rt_bc2)
    stages.sched.append(sc.decay_mul)

    var n_total = len(param)
    if cfg.kind == OPT_SGD:
        for j in range(j_count):
            var begin = offsets[j]
            var end = offsets[j + 1]
            var was_init = buf_initialized[j]
            for i in range(begin, end):
                var e = sgd_element_oracle(
                    param[i], grad[i], m_state[i], was_init, cfg, sc
                )
                param[i] = e.p
                m_state[i] = e.buf
                stages.sgd_buf.append(e.buf)
                stages.sgd_dir.append(e.direction)
                stages.param_out.append(e.p)
            if cfg.momentum != Float32(0.0):
                buf_initialized[j] = True
    else:
        for i in range(n_total):
            var e = adam_element_oracle(
                param[i], grad[i], m_state[i], v_state[i], cfg, sc
            )
            param[i] = e.p
            m_state[i] = e.m
            v_state[i] = e.v
            stages.adam_m.append(e.m)
            stages.adam_v.append(e.v)
            stages.adam_denom.append(e.denom)
            stages.adam_q.append(e.q)
            stages.param_out.append(e.p)

    return stages^
