# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The optimizer oracle's Adam update, elements over host tasks (lane neural-pass6).

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
task, DEVIATION 5900), each through `adam_element_oracle` itself, written in
place, no stage list. The refusals are the oracle's own calls in the oracle's
order, so a non-finite input raises the oracle's message with the oracle's
index. Clipping and SGD are refused here; the caller runs the oracle for them.
"""

from std.math import min

from core.host_lanes import host_row_tasks
from core.host_parallel import host_parallelize
from training.checks.optimizer_oracle import (
    OPT_SGD,
    OptimizerConfig,
    adam_element_oracle,
    refuse_nonfinite,
    step_scalars,
)

#: Scalar operations per element, for the task split only (a schedule knob).
comptime ADAM_ELEMENT_WORK = 24


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
    def _elements(task: Int) {mut param, imm grad, mut m_state, mut v_state, imm cfg, imm sc, imm n, imm chunk}:
        for i in range(task * chunk, min((task + 1) * chunk, n)):
            var e = adam_element_oracle(param[i], grad[i], m_state[i], v_state[i], cfg, sc)
            param[i] = e.p
            m_state[i] = e.m
            v_state[i] = e.v
    if tasks <= 1:
        _elements(0)
    else:
        host_parallelize(_elements, tasks)
