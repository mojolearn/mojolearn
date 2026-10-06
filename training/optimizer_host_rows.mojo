# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The optimizer oracle's Adam update, elements over host tasks (lane
neural-pass6) and lanes (lane neural-pass8).

`optimizer_step_oracle` (`training/checks/optimizer_oracle.mojo`) is THE
normative answer: refuse, clip, host scalars, then `adam_element_oracle` per
element, appending five stage lists as it goes. The byte LM host training step
ran it over 20.4 M parameters on one core: 850 ms of a 2.4 s step, a third of
the wall, while every other core waited.

This file is the same update in a different SCHEDULE only. With no clipping
(the byte LM's validator admits none, `byte_validate_optimizer`), an element's
new parameter and moments are a pure function of that element's four inputs
and the step's scalars, so the elements run on host tasks through
`host_parallelize` (the calling thread's floating-point environment on every
task, DEVIATION 5900), written in place, no stage list.

LANES (lane neural-pass8). Inside a task, `HOST_FW` elements advance together
through `adam_lanes`: `adam_element_oracle`'s fourteen statements with each
scalar seam replaced by its lane twin that equals it on every bit pattern:
`ftz` -> `ftz_lanes`, `identical_mul` (the fenced product) -> `pinned_mul_lanes`,
`identical_mul_add` (one fma) -> `identical_mul_add_simd`, `identical_sqrt`
(`portable_sqrtf`) -> `sqrt_lanes`, `identical_div` (`portable_divf`) ->
`div_lanes`, and a plain `+` on lanes. The scalar routine's three divides and
software square root were the per-core cost (71 ms a step on a 64-core host
after the task split). `training/checks/host_lanes_sqrt_check.mojo` compares
`sqrt_lanes` with the scalar over all 2^32 bit patterns and `adam_lanes`
with `adam_element_oracle` on sampled quadruples; the tail elements of a
task take `adam_element_oracle` itself. The refusals are the oracle's own
calls in the oracle's order, so a non-finite input raises the oracle's
message with the oracle's index. Clipping and SGD are refused here; the
caller runs the oracle for them.
"""

from std.math import min

from checks.numerics import identical_mul_add_simd
from core.host_lanes import (
    F32V,
    HOST_FW,
    div_lanes,
    ftz_lanes,
    lanes_are_identical,
    pinned_mul_lanes,
    sqrt_lanes,
)
from core.host_tasks import host_row_tasks
from core.host_parallel import host_parallelize
from training.checks.optimizer_contract import (
    OPT_ADAMW,
    OPT_SGD,
    OptimizerConfig,
    StepScalars,
    refuse_nonfinite,
    step_scalars,
)
from training.checks.optimizer_oracle import adam_element_oracle

#: Scalar operations per element, for the task split only (a schedule knob).
comptime ADAM_ELEMENT_WORK = 24


@always_inline
def adam_lanes(p_in: F32V, g_in: F32V, m_in: F32V, v_in: F32V,
               cfg: OptimizerConfig, sc: StepScalars) -> Tuple[F32V, F32V, F32V]:
    """`adam_element_oracle` on every lane: `(p_out, m, v)`. Contract 7.2,
    seams O1 through O14, in the oracle's order (module note)."""
    var g = ftz_lanes(g_in)  # O1
    var p = ftz_lanes(p_in)  # O2
    var mp = ftz_lanes(m_in)  # O3
    var vp = ftz_lanes(v_in)  # O3
    if cfg.weight_decay != Float32(0.0):
        if cfg.kind == OPT_ADAMW:
            p = ftz_lanes(pinned_mul_lanes(F32V(sc.decay_mul), p))
        else:
            g = ftz_lanes(identical_mul_add_simd[HOST_FW](F32V(cfg.weight_decay), p, g))
    var ms = ftz_lanes(pinned_mul_lanes(F32V(cfg.beta1), mp))  # O5, PRODUCT
    var m = ftz_lanes(identical_mul_add_simd[HOST_FW](F32V(sc.c1), g, ms))  # O6, FUSED
    var g2 = ftz_lanes(pinned_mul_lanes(g, g))  # O7, PRODUCT
    var vs = ftz_lanes(pinned_mul_lanes(F32V(cfg.beta2), vp))  # O8, PRODUCT
    var v = ftz_lanes(identical_mul_add_simd[HOST_FW](F32V(sc.c2), g2, vs))  # O9, FUSED
    var s = ftz_lanes(sqrt_lanes(v))  # O10
    var sd = ftz_lanes(div_lanes(s, F32V(sc.rt_bc2)))  # O11
    var dn = ftz_lanes(sd + F32V(cfg.eps))  # O12, eps OUTSIDE the sqrt
    var q = ftz_lanes(div_lanes(m, dn))  # O13, a TRUE divide
    var p_out = ftz_lanes(identical_mul_add_simd[HOST_FW](F32V(-sc.step_size), q, p))  # O14, FUSED
    return (p_out, m, v)


def adam_host_rows(
    mut param: List[Float32],
    grad: List[Float32],
    mut m_state: List[Float32],
    mut v_state: List[Float32],
    cfg: OptimizerConfig,
    t: Int,
) raises:
    """`optimizer_step_oracle`'s `param_out`, `adam_m` and `adam_v`, written
    into `param`, `m_state` and `v_state` (module note)."""
    if cfg.kind == OPT_SGD:
        raise Error("adam_host_rows: SGD takes the oracle path")
    if cfg.max_norm > Float32(0.0):
        raise Error("adam_host_rows: gradient clipping takes the oracle path")
    var n = len(param)
    if len(grad) != n or len(m_state) != n or len(v_state) != n:
        raise Error("adam_host_rows: param, grad, m and v must have one length")
    refuse_nonfinite(String("input.param"), param)
    refuse_nonfinite(String("input.grad"), grad)
    refuse_nonfinite(String("state.m"), m_state)
    refuse_nonfinite(String("state.v"), v_state)
    var sc = step_scalars(cfg, t)
    var tasks = host_row_tasks(n, ADAM_ELEMENT_WORK)
    var chunk = (n + tasks - 1) // tasks
    chunk = ((chunk + HOST_FW - 1) // HOST_FW) * HOST_FW
    var pp = param.unsafe_ptr()
    var gp = grad.unsafe_ptr()
    var mpp = m_state.unsafe_ptr()
    var vpp = v_state.unsafe_ptr()
    def _elements(task: Int) {imm pp, imm gp, imm mpp, imm vpp, imm cfg, imm sc, imm n, imm chunk}:
        var i = task * chunk
        var hi = min((task + 1) * chunk, n)
        comptime if lanes_are_identical:
            while i + HOST_FW <= hi:
                var out = adam_lanes(
                    pp.unsafe_load[width=HOST_FW](i), gp.unsafe_load[width=HOST_FW](i),
                    mpp.unsafe_load[width=HOST_FW](i), vpp.unsafe_load[width=HOST_FW](i), cfg, sc)
                pp.unsafe_store(i, out[0])
                mpp.unsafe_store(i, out[1])
                vpp.unsafe_store(i, out[2])
                i += HOST_FW
        while i < hi:
            var e = adam_element_oracle(pp.unsafe_load(i), gp.unsafe_load(i), mpp.unsafe_load(i),
                                        vpp.unsafe_load(i), cfg, sc)
            pp.unsafe_store(i, e.p)
            mpp.unsafe_store(i, e.m)
            vpp.unsafe_store(i, e.v)
            i += 1
    if tasks <= 1:
        _elements(0)
    else:
        host_parallelize(_elements, tasks)
