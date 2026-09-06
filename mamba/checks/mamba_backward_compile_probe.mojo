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
# SINCE 2026-09-03 IT ALSO COVERS THE ARITHMETIC. The two kernel files
# `mamba/impl/mamba_ssm/ops/selective_scan_backward.mojo` and
# `mamba/impl/transformers/models/mamba/modeling_mamba_backward.mojo` are
# pulled in through `_force_elaborate_kernels`, so a syntax or type error in
# any of the fourteen new launchers or the ten new kernels fails this build.
# **THAT IS STILL NOT A RESULT ABOUT GRADIENTS.** No value is asserted, no
# bits are compared, no kernel is launched, and gates MB1-MB10 do not exist.
#
# THE SABOTAGE BUILDS ARE NOT OPTIONAL, and there are now nineteen arms rather
# than three. A `comptime if` branch that is not taken is not reliably
# elaborated, so the clean build type-checks only the profile half of every
# arm. Build all twenty configurations (clean plus each arm) or the probe has
# covered less than it appears to; the arm lists are in the three files'
# headers and `pixi.toml`'s `build-mamba-backward-probe` comment.
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
    TOPO_COUNT,
    mamba_backward_declarations,
    mamba_backward_kernel_sites,
    mamba_backward_topology_site,
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

# THE ARITHMETIC, imported from the two kernel files DIRECTLY rather than
# through `mamba_backward`'s re-exports, so that this probe fails on a broken
# kernel file even if the routing layer's import list drifts.
from mamba.impl.mamba_ssm.ops.selective_scan_backward import (
    bwd_da_partial_floats,
    bwd_dh_floats,
    bwd_h_checkpoint_floats,
    mamba_backward_scan_sabotage_name,
    mamba_bwd_da_into,
    mamba_bwd_da_log_into,
    mamba_bwd_dbc_into,
    mamba_bwd_param_fold_into,
    selective_scan_bwd_scan_into,
    selective_scan_checkpoint_fn,
)
from mamba.impl.transformers.models.mamba.modeling_mamba_backward import (
    mamba_backward_block_sabotage_name,
    mamba_bwd_concat_p_into,
    mamba_bwd_concat_xp_into,
    mamba_bwd_conv_tap_product_into,
    mamba_bwd_ddtp_into,
    mamba_bwd_dhin_into,
    mamba_bwd_du_join_into,
    mamba_bwd_gate_into,
    mamba_bwd_norm_into,
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


def _force_elaborate_kernels(
    ctx: DeviceContext,
    mut q0: DeviceBuffer[DType.float32],
    mut q1: DeviceBuffer[DType.float32],
    mut q2: DeviceBuffer[DType.float32],
    mut q3: DeviceBuffer[DType.float32],
    mut q4: DeviceBuffer[DType.float32],
    mut q5: DeviceBuffer[DType.float32],
    mut q6: DeviceBuffer[DType.float32],
    mut q7: DeviceBuffer[DType.float32],
    mut q8: DeviceBuffer[DType.float32],
    mut q9: DeviceBuffer[DType.float32],
    mut q10: DeviceBuffer[DType.float32],
    dims: MambaDims,
    b: Int,
    l: Int,
    m: Int,
) raises:
    """NEVER CALLED. Exists to put every ARITHMETIC launcher of the two new
    kernel files at a real call site with real `DeviceBuffer` arguments.

    The eleven buffers are deliberately anonymous and reused across calls.
    This function type checks SIGNATURES and elaborates KERNEL BODIES; it is
    not a backward pass and the argument order it uses is not a plan. A driver
    that copied it would compute nonsense. **AND IT IS NOT A GATE**: an
    elaborated kernel is a kernel the compiler accepted, not a kernel that
    computes a gradient, and gates MB1 through MB10 remain unbuilt.

    THE SABOTAGE BUILDS ARE STILL NOT OPTIONAL, and there are now nineteen
    arms rather than three. A `comptime if` branch that is not taken is not
    reliably elaborated, so the clean build type checks only the profile half
    of every arm below. The arm list is in the two kernel files' headers.
    """
    selective_scan_checkpoint_fn(
        ctx, q0, q1, q2, q3, q4, q5, b, l, dims.d_inner
    )
    selective_scan_bwd_scan_into(
        ctx, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, b, l, dims.d_inner
    )
    mamba_bwd_dbc_into(ctx, q0, q1, q2, q3, q4, q5, b, l, dims.d_inner)
    mamba_bwd_da_into(
        ctx, q0, q1, q2, q3, q4, q5, q6, q7, b, l, dims.d_inner
    )
    mamba_bwd_param_fold_into(ctx, q0, q1, b, dims.d_inner)
    mamba_bwd_da_log_into(ctx, q0, q1, q2, dims.d_inner)

    mamba_bwd_gate_into(
        ctx, q0, q1, q2, q3, q4, q5, q6, q7, q8, m, dims.d_inner
    )
    mamba_bwd_ddtp_into(ctx, q0, q1, q2, q3, m, dims.d_inner)
    mamba_bwd_du_join_into(ctx, q0, q1, q2, q3, q4, q5, m, dims.d_inner)
    mamba_bwd_dhin_into(ctx, q0, q1, q2, b, l, dims.d_inner)
    mamba_bwd_conv_tap_product_into(
        ctx, q0, q1, q2, q3, b, l, dims.d_inner, 0
    )
    mamba_bwd_concat_xp_into(ctx, q0, q1, q2, q3, m, dims)
    mamba_bwd_concat_p_into(ctx, q0, q1, q2, m, dims.d_inner)
    mamba_bwd_norm_into(
        ctx, q0, q1, q2, q3, q4, q5, q6, q7, q8, m, dims.d_model
    )


def main() raises:
    var dims = MambaDims.of(64)
    var m = 96

    print(mamba_backward_sabotage_name())
    print(mamba_backward_scan_sabotage_name())
    print(mamba_backward_block_sabotage_name())
    print(mamba_backward_inert_sabotages())
    print(mamba_backward_declarations())
    print(mamba_backward_kernel_sites())
    for i in range(TOPO_COUNT):
        print("site", i, mamba_backward_topology_site(i))
    print("ones", mamba_backward_ones_floats(m))
    print("ws", mamba_backward_workspace_max_floats(dims, m))
    print("t2", t2_h_checkpoint_floats(2, 48, dims))
    print("t3", t3_dh_floats(2, 48, dims))

    # THE SAME TWO NUMBERS, FROM THE KERNEL FILE'S OWN HELPERS. They are
    # spelled twice on purpose -- once in `MambaDims` terms in the routing
    # layer, once in `dim` terms beside the kernel that allocates against
    # them -- and a gate must assert they AGREE, because two homes for a
    # buffer size is two places for a size to drift and an out-of-bounds
    # device write returns plausible bits.
    print("t2 kernel-side", bwd_h_checkpoint_floats(2, 48, dims.d_inner))
    print("t3 kernel-side", bwd_dh_floats(2, 48, dims.d_inner))
    print("t5 partial", bwd_da_partial_floats(2, dims.d_inner))

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
