# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IDENTICAL FP32 optimizer step, written out, on the host. The shared contract (ids, constants, refusals, configuration, step scalars) is in `training/checks/optimizer_contract.mojo`. This file's own `opt_refuse_bad_inputs` is now what the device entry point calls, so both sides fail with the same name (DEVIATION 1496)."""

from gemm.contract import OP_NT, contract_leaf_size
from gemm.checks.gemm_oracle import gemm_oracle
from gemm.host.gemm_host_rows import gemm_host_rows
from checks.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from training.checks.optimizer_contract import (
    CLIP_EPS_BITS,
    INF_BITS,
    ONE_BITS,
    OPT_ADAM,
    OPT_ADAMW,
    OPT_SGD,
    QNAN_BITS,
    OptimizerConfig,
    StepScalars,
    clip_eps,
    microbatch_split_is_identical,
    opt_nonfinite_message,
    pow_int_f32,
    refuse_nonfinite,
    refuse_nonfinite_scalar,
    step_scalars,
)


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
    var out = gemm_host_rows(g, g, OP_NT, 1, 1, count)
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
    var tot = gemm_host_rows(norms_copy, norms_copy, OP_NT, 1, 1, j_count)
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
