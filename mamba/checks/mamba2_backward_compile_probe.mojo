# SPDX-License-Identifier: Apache-2.0
# THE COMPILE PROBE FOR THE MAMBA-2 BACKWARD. IT GATES COMPILATION AND NOTHING
# ELSE -- it asserts no value, compares no bits and runs no kernel, so it is
# not a gate and its passing says nothing about whether any gradient is
# correct. NOTHING HERE HAS BEEN COMPILED OR RUN.
#
# `mamba/checks/mamba2_backward.mojo` is a library module with no `main()`, so
# `mojo build` on it fails with "module does not define a `main` function"
# before it ever type checks anything (pixi.toml records that exact failure
# for 21 of 22 check files). This driver exists solely to give the compiler an
# entry point that pulls the module in. It is the exact shape of
# `mamba_backward_compile_probe.mojo`, which is the file that put the Mamba-1
# routing layer through a compiler for the first time.
#
# THE SABOTAGE BUILDS ARE NOT OPTIONAL. A `comptime if` branch that is not
# taken is not reliably elaborated, so the clean build type-checks only half
# the routing calls. Build all FIVE configurations -- clean plus each of the
# four arms -- or the probe has covered less than it appears to:
#
#   MOJOLEARN_MAMBA2_SABOTAGE_BWD_PROJ_SWAP
#   MOJOLEARN_MAMBA2_SABOTAGE_BWD_WS_FROM_FORWARD
#   MOJOLEARN_MAMBA2_SABOTAGE_BWD_DD_TOKENS_FIRST
#   MOJOLEARN_MAMBA2_SABOTAGE_BWD_CONV_TAP_SLOT
#
# `_force_elaborate` is NEVER CALLED. It is a top-level function of this
# module, so it is elaborated, and its body is the only thing that puts the
# three device launchers at a real call site with real `DeviceBuffer`
# arguments. It constructs no device and needs no GPU.

from max.gpu.host import DeviceBuffer, DeviceContext

from mamba.checks.mamba2_fixture import Mamba2Dims
from mamba.checks.mamba2_backward import (
    CELL2_COUNT,
    PROJ2_COUNT,
    PROJ2_IN,
    RED2_COUNT,
    RED2_D,
    TOPO2_COUNT,
    mamba2_backward_cell_k,
    mamba2_backward_cell_leaves,
    mamba2_backward_cell_name,
    mamba2_backward_declarations,
    mamba2_backward_inert_sabotages,
    mamba2_backward_ones_floats,
    mamba2_backward_proj_a_call,
    mamba2_backward_proj_a_into,
    mamba2_backward_proj_a_workspace_max_floats,
    mamba2_backward_proj_b_call,
    mamba2_backward_proj_b_into,
    mamba2_backward_proj_b_workspace_max_floats,
    mamba2_backward_proj_call_name,
    mamba2_backward_reduce_into,
    mamba2_backward_reduce_workspace_max_floats,
    mamba2_backward_sabotage_name,
    mamba2_backward_state_floats,
    mamba2_backward_topology_site,
    mamba2_backward_workspace_max_floats,
    mamba2_dd_inner_fold_call,
    mamba2_proj_forward_call,
    mamba2_proj_name,
    mamba2_reduction_name,
    mamba2_reduction_needs_inner_p_fold,
    mamba2_reduction_needs_preproduct,
    mamba2_reduction_tap,
    mamba2_reduction_width,
)


def _force_elaborate(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut c: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    dims: Mamba2Dims,
    m: Int,
) raises:
    """NEVER CALLED. Exists to type check the three launcher call sites."""
    mamba2_backward_proj_a_into(ctx, a, b, c, ws, PROJ2_IN, dims, m)
    mamba2_backward_proj_b_into(ctx, a, b, c, ws, PROJ2_IN, dims, m)
    mamba2_backward_reduce_into(ctx, a, b, c, ws, RED2_D, dims, m)


def main() raises:
    var dims = Mamba2Dims.of(64)
    var m = 96

    print(mamba2_backward_sabotage_name())
    print(mamba2_backward_inert_sabotages())
    print(mamba2_backward_declarations())
    print("ones", mamba2_backward_ones_floats(m))
    print("ws", mamba2_backward_workspace_max_floats(dims, m))
    print("state", mamba2_backward_state_floats(2, 770, dims))

    var inner = mamba2_dd_inner_fold_call(dims)
    print("dD inner", inner[0], inner[1], inner[2], inner[3])

    for i in range(PROJ2_COUNT):
        var fwd = mamba2_proj_forward_call(i, dims, m)
        var ca = mamba2_backward_proj_a_call(i, dims, m)
        var cb = mamba2_backward_proj_b_call(i, dims, m)
        print(mamba2_proj_name(i), "fwd", fwd[0], fwd[1], fwd[2], fwd[3])
        print("  dA ", mamba2_backward_proj_call_name(i, ca))
        print("  dB ", mamba2_backward_proj_call_name(i, cb))
        print(
            "  ws ",
            mamba2_backward_proj_a_workspace_max_floats(i, dims, m),
            mamba2_backward_proj_b_workspace_max_floats(i, dims, m),
        )

    for j in range(RED2_COUNT):
        print(
            mamba2_reduction_name(j),
            mamba2_reduction_width(j, dims),
            mamba2_reduction_tap(j),
            mamba2_reduction_needs_preproduct(j),
            mamba2_reduction_needs_inner_p_fold(j),
            mamba2_backward_reduce_workspace_max_floats(j, dims, m),
        )

    for c in range(CELL2_COUNT):
        var lf = mamba2_backward_cell_leaves(c)
        print(
            mamba2_backward_cell_name(c),
            "k",
            mamba2_backward_cell_k(c),
            "leaf",
            lf[0],
            "count",
            lf[1],
        )

    for t in range(TOPO2_COUNT):
        print(mamba2_backward_topology_site(t))
