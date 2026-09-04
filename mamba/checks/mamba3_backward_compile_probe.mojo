# SPDX-License-Identifier: Apache-2.0
# THE COMPILE PROBE FOR THE MAMBA-3 BACKWARD. IT GATES COMPILATION AND NOTHING
# ELSE -- it asserts no value, compares no bits and runs no kernel, so it is
# not a gate and its passing says nothing about whether any gradient is
# correct. NOTHING HERE HAS BEEN COMPILED OR RUN.
#
# `mamba/checks/mamba3_backward.mojo` is a library module with no `main()`, so
# `mojo build` on it fails with "module does not define a `main` function"
# before it ever type checks anything. This driver exists solely to give the
# compiler an entry point that pulls the module in, and it is the exact shape
# of `mamba_backward_compile_probe.mojo`.
#
# THE SABOTAGE BUILDS ARE NOT OPTIONAL. A `comptime if` branch that is not
# taken is not reliably elaborated. Build all FIVE configurations -- clean
# plus each of the four arms:
#
#   MOJOLEARN_MAMBA3_SABOTAGE_BWD_PROJ_SWAP
#   MOJOLEARN_MAMBA3_SABOTAGE_BWD_WS_FROM_FORWARD
#   MOJOLEARN_MAMBA3_SABOTAGE_BWD_ROTATE_HALF_SPLIT
#   MOJOLEARN_MAMBA3_SABOTAGE_BWD_BIAS_WIDTH_HEADS_ONLY
#
# `_force_elaborate` is NEVER CALLED. It is a top-level function of this
# module, so it is elaborated, and its body is the only thing that puts the
# three device launchers at a real call site with real `DeviceBuffer`
# arguments. It constructs no device and needs no GPU.

from max.gpu.host import DeviceBuffer, DeviceContext

from mamba.checks.mamba3_fixture import Mamba3Dims
from mamba.checks.mamba3_backward import (
    CELL3_COUNT,
    PROJ3_COUNT,
    PROJ3_IN,
    RED3_COUNT,
    RED3_D,
    TOPO3_COUNT,
    mamba3_backward_cell_k,
    mamba3_backward_cell_leaves,
    mamba3_backward_cell_name,
    mamba3_backward_declarations,
    mamba3_backward_inert_sabotages,
    mamba3_backward_ones_floats,
    mamba3_backward_proj_a_call,
    mamba3_backward_proj_a_into,
    mamba3_backward_proj_a_workspace_max_floats,
    mamba3_backward_proj_b_call,
    mamba3_backward_proj_b_into,
    mamba3_backward_proj_b_workspace_max_floats,
    mamba3_backward_proj_call_name,
    mamba3_backward_reduce_into,
    mamba3_backward_reduce_workspace_max_floats,
    mamba3_backward_sabotage_name,
    mamba3_backward_state_floats,
    mamba3_backward_theta_chain_floats,
    mamba3_backward_topology_site,
    mamba3_backward_workspace_max_floats,
    mamba3_proj_forward_call,
    mamba3_proj_name,
    mamba3_reduction_name,
    mamba3_reduction_needs_preproduct,
    mamba3_reduction_width,
    mamba3_rope_pair_count,
    mamba3_rope_pair_indices,
    mamba3_rope_pair_is_rotated,
)
from mamba.impl.mamba_ssm.modules.mamba3_backward import (
    mamba3_backward_block_norm_into,
    mamba3_backward_pack_in_proj_into,
)


def _force_elaborate(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut c: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    mut p0: DeviceBuffer[DType.float32],
    mut p1: DeviceBuffer[DType.float32],
    mut p2: DeviceBuffer[DType.float32],
    mut p3: DeviceBuffer[DType.float32],
    mut p4: DeviceBuffer[DType.float32],
    mut p5: DeviceBuffer[DType.float32],
    mut p6: DeviceBuffer[DType.float32],
    mut p7: DeviceBuffer[DType.float32],
    mut p8: DeviceBuffer[DType.float32],
    dims: Mamba3Dims,
    m: Int,
) raises:
    """NEVER CALLED. Exists to type check the three launcher call sites."""
    mamba3_backward_proj_a_into(ctx, a, b, c, ws, PROJ3_IN, dims, m)
    mamba3_backward_proj_b_into(ctx, a, b, c, ws, PROJ3_IN, dims, m)
    mamba3_backward_reduce_into(ctx, a, b, c, ws, RED3_D, dims, m)
    mamba3_backward_pack_in_proj_into(
        ctx, p0, p1, p2, p3, p4, p5, p6, p7, p8, m, dims
    )
    mamba3_backward_block_norm_into(
        ctx, p0, p1, p2, p3, p4, p5, p6, m, dims
    )


def main() raises:
    var dims = Mamba3Dims.of(64)
    var m = 96

    print(mamba3_backward_sabotage_name())
    print(mamba3_backward_inert_sabotages())
    print(mamba3_backward_declarations())
    print("ones", mamba3_backward_ones_floats(m))
    print("ws", mamba3_backward_workspace_max_floats(dims, m))
    print("state", mamba3_backward_state_floats(2, 257, dims))
    print("theta", mamba3_backward_theta_chain_floats(2, 257, dims))
    print("pairs", mamba3_rope_pair_count())

    for p in range(mamba3_rope_pair_count()):
        var idx = mamba3_rope_pair_indices(p)
        if p == 0 or p == 31 or p == 32 or p == 63:
            print(
                "pair",
                p,
                idx[0],
                idx[1],
                "rotated",
                mamba3_rope_pair_is_rotated(p),
            )

    for i in range(PROJ3_COUNT):
        var fwd = mamba3_proj_forward_call(i, dims, m)
        var ca = mamba3_backward_proj_a_call(i, dims, m)
        var cb = mamba3_backward_proj_b_call(i, dims, m)
        print(mamba3_proj_name(i), "fwd", fwd[0], fwd[1], fwd[2], fwd[3])
        print("  dA ", mamba3_backward_proj_call_name(i, ca))
        print("  dB ", mamba3_backward_proj_call_name(i, cb))
        print(
            "  ws ",
            mamba3_backward_proj_a_workspace_max_floats(i, dims, m),
            mamba3_backward_proj_b_workspace_max_floats(i, dims, m),
        )

    for j in range(RED3_COUNT):
        print(
            mamba3_reduction_name(j),
            mamba3_reduction_width(j, dims),
            mamba3_reduction_needs_preproduct(j),
            mamba3_backward_reduce_workspace_max_floats(j, dims, m),
        )

    for c in range(CELL3_COUNT):
        var lf = mamba3_backward_cell_leaves(c)
        print(
            mamba3_backward_cell_name(c),
            "k",
            mamba3_backward_cell_k(c),
            "leaf",
            lf[0],
            "count",
            lf[1],
        )

    for t in range(TOPO3_COUNT):
        print(mamba3_backward_topology_site(t))
