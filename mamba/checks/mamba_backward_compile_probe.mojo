# SPDX-License-Identifier: Apache-2.0
# THE COMPILE PROBE FOR THE MAMBA BACKWARD. IT GATES COMPILATION AND NOTHING
# ELSE -- it asserts no value, compares no bits and runs no kernel, so it is
# NOT one of the plan's gates MB1-MB10 and its passing says nothing about
# whether any gradient is correct.
#
# WHAT IT DOES ESTABLISH, and it was worth establishing: on 2026-09-03
# `mamba/checks/mamba_backward.mojo` went through a compiler for the first
# time since it was written, UNMODIFIED, in all four configurations -- the
# clean build and each of the three sabotage arms. The plan's section 10 item
# 1 predicted "the five-element `Tuple` returns and the `comptime if` early
# returns are the two things most likely to need adjustment"; neither needed
# any. Every construct in that file already had a compiling twin in
# `gemm/checks/gemm_backward.mojo`.
#
# THE SABOTAGE BUILDS ARE NOT OPTIONAL. A `comptime if` branch that is not
# taken is not reliably elaborated, so the clean build type-checks only half
# the routing calls in that file. Build all four or the probe has covered
# less than it appears to.
# `mamba/checks/mamba_backward.mojo` is a library module with no `main()`, so
# `mojo build` on it fails with "module does not define a `main` function"
# before it ever type checks anything (pixi.toml records that exact failure
# for 21 of 22 check files). This driver exists solely to give the compiler an
# entry point that pulls the module in.
#
# `_force_elaborate` is NEVER CALLED. It is a top-level function of this
# module, so it is elaborated, and its body is the only thing that puts the
# three device launchers at a real call site with real `DeviceBuffer`
# arguments. It constructs no device and needs no GPU.

from max.gpu.host import DeviceBuffer, DeviceContext

from mamba.checks.mamba_fixture import MambaDims
from mamba.checks.mamba_backward import (
    PROJ_COUNT,
    PROJ_IN,
    RED_COUNT,
    RED_D,
    mamba_backward_declarations,
    mamba_backward_inert_sabotages,
    mamba_backward_ones_floats,
    mamba_backward_proj_a_call,
    mamba_backward_proj_a_into,
    mamba_backward_proj_a_workspace_max_floats,
    mamba_backward_proj_b_call,
    mamba_backward_proj_b_into,
    mamba_backward_proj_b_workspace_max_floats,
    mamba_backward_proj_call_name,
    mamba_backward_reduce_into,
    mamba_backward_reduce_workspace_max_floats,
    mamba_backward_sabotage_name,
    mamba_backward_workspace_max_floats,
    mamba_proj_forward_call,
    mamba_proj_name,
    mamba_reduction_name,
    mamba_reduction_needs_preproduct,
    mamba_reduction_tap,
    mamba_reduction_width,
    t2_h_checkpoint_floats,
    t3_dh_floats,
)


def _force_elaborate(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut c: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    dims: MambaDims,
    m: Int,
) raises:
    """NEVER CALLED. Exists to type check the three launcher call sites."""
    mamba_backward_proj_a_into(ctx, a, b, c, ws, PROJ_IN, dims, m)
    mamba_backward_proj_b_into(ctx, a, b, c, ws, PROJ_IN, dims, m)
    mamba_backward_reduce_into(ctx, a, b, c, ws, RED_D, dims, m)


def main() raises:
    var dims = MambaDims.of(64)
    var m = 96

    print(mamba_backward_sabotage_name())
    print(mamba_backward_inert_sabotages())
    print(mamba_backward_declarations())
    print("ones", mamba_backward_ones_floats(m))
    print("ws", mamba_backward_workspace_max_floats(dims, m))
    print("t2", t2_h_checkpoint_floats(2, 48, dims))
    print("t3", t3_dh_floats(2, 48, dims))

    for i in range(PROJ_COUNT):
        var fwd = mamba_proj_forward_call(i, dims, m)
        var ca = mamba_backward_proj_a_call(i, dims, m)
        var cb = mamba_backward_proj_b_call(i, dims, m)
        print(
            mamba_proj_name(i),
            "fwd",
            fwd[0],
            fwd[1],
            fwd[2],
            fwd[3],
        )
        print("  dA ", mamba_backward_proj_call_name(i, ca))
        print("  dB ", mamba_backward_proj_call_name(i, cb))
        print(
            "  ws ",
            mamba_backward_proj_a_workspace_max_floats(i, dims, m),
            mamba_backward_proj_b_workspace_max_floats(i, dims, m),
        )

    for j in range(RED_COUNT):
        print(
            mamba_reduction_name(j),
            mamba_reduction_width(j, dims),
            mamba_reduction_tap(j),
            mamba_reduction_needs_preproduct(j),
            mamba_backward_reduce_workspace_max_floats(j, dims, m),
        )
