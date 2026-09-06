# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Routing, workspace sizing, and topology declarations for Mamba-1 backward."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined

from gemm.checks.gemm_oracle import OP_NT
from gemm.checks.gemm_identical import identical_gemm_workspace_max_floats
from gemm.checks.gemm_backward import (
    gemm_backward_a_call,
    gemm_backward_b_call,
    gemm_backward_call_name,
    identical_gemm_backward_a_into,
    identical_gemm_backward_a_workspace_max_floats,
    identical_gemm_backward_b_into,
    identical_gemm_backward_b_workspace_max_floats,
    identical_gemm_backward_bias_into,
    identical_gemm_backward_bias_ones_floats,
    identical_gemm_backward_bias_workspace_max_floats,
)
from mamba.checks.mamba_fixture import D_CONV, D_STATE, MambaDims

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



comptime SAB_BWD_PROJ_SWAP = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_PROJ_SWAP"
]()
comptime SAB_BWD_WS_FROM_FORWARD = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_WS_FROM_FORWARD"
]()
comptime SAB_BWD_CONV_TAP_SLOT = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_CONV_TAP_SLOT"
]()

comptime ANY_BWD_SABOTAGE = (
    SAB_BWD_PROJ_SWAP or SAB_BWD_WS_FROM_FORWARD or SAB_BWD_CONV_TAP_SLOT
)


def mamba_backward_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for a check's banner. A check must print THIS, plus `gemm_backward_sabotage_name()`, plus `mamba_block_sabotage_name()` and `mamba_scan_sabotage_name()`, because a backward binary can carry a forward sabotage and a gemm sabotage as well and a banner naming only one of the four mislabels the run."""
    comptime if SAB_BWD_PROJ_SWAP:
        return String("BWD_PROJ_SWAP")
    comptime if SAB_BWD_WS_FROM_FORWARD:
        return String("BWD_WS_FROM_FORWARD")
    comptime if SAB_BWD_CONV_TAP_SLOT:
        return String("BWD_CONV_TAP_SLOT")
    return String("none")


def mamba_backward_inert_sabotages() -> String:
    """The arms a gate must NOT count as covered by this lane, with the reason."""
    return (
        String("SAB_BWD_OPERAND_ORDER: INERT here. All four Mamba-1")
        + " projections are OP_NT, so every one of the eight backward calls"
        + " puts dC on the LEFT and the side flag is never exercised.\n"
        + "SAB_BWD_UNTRANSPOSED: PARTIAL here. It moves the dA and dB shapes"
        + " for OP_NT, so it does bite, but it cannot distinguish OP_NN from"
        + " OP_TN routing because this lane calls neither.\n"
        + "SAB_BWD_BIAS_AXIS: covered, but only at M != W. At a shape where"
        + " the token count equals the reduced width it returns a vector of"
        + " the right length and the wrong contents.\n"
        + "SAB_FOLD_STRIDE and SAB_LEAF_ROTATE: NEWLY REACHABLE, through T3."
        + " mamba_bwd_dbc_kernel imports gemm_identical's own _fold_push and"
        + " _leaf_at, so both arms bite inside a Mamba entry point for the"
        + " first time. Only at d_inner > 128, where the partition has more"
        + " than one leaf; below that P == 1 and both are inert.\n"
        + "SAB_LEAF_READS_LAUNCH, SAB_PAD_PLUS_ZERO and SAB_FOLD_SERIAL:"
        + " NOT reachable through T3. They live in the bodies of gemm's own"
        + " kernels, not in the four helpers T3 imports, so running them"
        + " against this lane measures nothing. They ARE reachable through"
        + " the four dB routings, whose k' is the token count and therefore"
        + " exceeds 128 at every real shape."
    )



comptime PROJ_IN = 0
comptime PROJ_X = 1
comptime PROJ_DT = 2
comptime PROJ_OUT = 3

comptime PROJ_COUNT = 4


def mamba_proj_name(which: Int) -> String:
    if which == PROJ_IN:
        return String("in_proj")
    if which == PROJ_X:
        return String("x_proj")
    if which == PROJ_DT:
        return String("dt_proj")
    if which == PROJ_OUT:
        return String("out_proj")
    return String("?")


def mamba_proj_forward_call(
    which: Int, dims: MambaDims, m: Int
) -> Tuple[Int, Int, Int, Int]:
    """`(op, m, n, k)` of the FORWARD projection, from the block's dimensions. It reads `which`, the three dimensions and `m`, and nothing else, so a backward call at any shape gets a deterministic plan for the same reason a forward one does."""
    var dm = dims.d_model
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    if which == PROJ_IN:
        return (OP_NT, m, 2 * di, dm)
    if which == PROJ_X:
        return (OP_NT, m, xr, di)
    if which == PROJ_DT:
        return (OP_NT, m, di, r)
    return (OP_NT, m, dm, di)


def mamba_backward_proj_a_call(
    which: Int, dims: MambaDims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the ACTIVATION gradient of `which`."""
    var fwd = mamba_proj_forward_call(which, dims, m)
    comptime if SAB_BWD_PROJ_SWAP:
        return gemm_backward_b_call(fwd[0], fwd[1], fwd[2], fwd[3])
    return gemm_backward_a_call(fwd[0], fwd[1], fwd[2], fwd[3])


def mamba_backward_proj_b_call(
    which: Int, dims: MambaDims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the WEIGHT gradient of `which`. The forward is batch invariant because gemm contract 6.1 forbids the leaf size from depending on `m`; the weight gradient contracts over the batch, so the batch arrives as `k`, where section 6 of the same contract REQUIRES the leaf size to depend on it."""
    var fwd = mamba_proj_forward_call(which, dims, m)
    comptime if SAB_BWD_PROJ_SWAP:
        return gemm_backward_a_call(fwd[0], fwd[1], fwd[2], fwd[3])
    return gemm_backward_b_call(fwd[0], fwd[1], fwd[2], fwd[3])


def mamba_backward_proj_call_name(
    which: Int, call: Tuple[Int, Int, Int, Int, Int]
) -> String:
    """`"out_proj NN(dC,W) 64x32x8"`, for a gate's per-case banner."""
    return mamba_proj_name(which) + " " + gemm_backward_call_name(call)


def mamba_backward_proj_a_into(
    ctx: DeviceContext,
    mut d_input: DeviceBuffer[DType.float32],
    mut d_output: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    which: Int,
    dims: MambaDims,
    m: Int,
) raises:
    """The ACTIVATION gradient of projection `which`, into `d_input`. `ws` must hold at least `mamba_backward_proj_a_workspace_max_floats(which, dims, m)` floats."""
    var fwd = mamba_proj_forward_call(which, dims, m)
    comptime if SAB_BWD_PROJ_SWAP:
        identical_gemm_backward_b_into(
            ctx, d_input, d_output, weight, ws, fwd[1], fwd[2], fwd[3], fwd[0]
        )
        return
    identical_gemm_backward_a_into(
        ctx, d_input, d_output, weight, ws, fwd[1], fwd[2], fwd[3], fwd[0]
    )


def mamba_backward_proj_b_into(
    ctx: DeviceContext,
    mut d_weight: DeviceBuffer[DType.float32],
    mut d_output: DeviceBuffer[DType.float32],
    mut input_stage: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    which: Int,
    dims: MambaDims,
    m: Int,
) raises:
    """The WEIGHT gradient of projection `which`, into `d_weight`. ASYNCHRONOUS, caller-owned buffers."""
    var fwd = mamba_proj_forward_call(which, dims, m)
    comptime if SAB_BWD_PROJ_SWAP:
        identical_gemm_backward_a_into(
            ctx,
            d_weight,
            d_output,
            input_stage,
            ws,
            fwd[1],
            fwd[2],
            fwd[3],
            fwd[0],
        )
        return
    identical_gemm_backward_b_into(
        ctx,
        d_weight,
        d_output,
        input_stage,
        ws,
        fwd[1],
        fwd[2],
        fwd[3],
        fwd[0],
    )



comptime RED_D = 0
comptime RED_DT_BIAS = 1
comptime RED_CONV_BIAS = 2
comptime RED_CONV_W_TAP0 = 3
comptime RED_CONV_W_TAP1 = 4
comptime RED_CONV_W_TAP2 = 5
comptime RED_CONV_W_TAP3 = 6
comptime RED_NORM_W = 7

comptime RED_COUNT = 8


def mamba_reduction_name(which: Int) -> String:
    if which == RED_D:
        return String("dD")
    if which == RED_DT_BIAS:
        return String("d.dt_proj.bias")
    if which == RED_CONV_BIAS:
        return String("d.conv1d.bias")
    if which == RED_NORM_W:
        return String("d.norm.weight")
    if which >= RED_CONV_W_TAP0 and which <= RED_CONV_W_TAP3:
        return (
            String("d.conv1d.weight.tap")
            + String(which - RED_CONV_W_TAP0)
        )
    return String("?")


def mamba_reduction_width(which: Int, dims: MambaDims) -> Int:
    """`W`, the length of the reduced vector."""
    if which == RED_NORM_W:
        return dims.d_model
    return dims.d_inner


def mamba_reduction_tap(which: Int) -> Int:
    """The conv tap index `k` a `RED_CONV_W_*` id writes, or `-1`. Gate MB1 must assert the fixture is asymmetric before counting this arm as covered."""
    if which < RED_CONV_W_TAP0 or which > RED_CONV_W_TAP3:
        return -1
    var k = which - RED_CONV_W_TAP0
    comptime if SAB_BWD_CONV_TAP_SLOT:
        return (D_CONV - 1) - k
    return k


def mamba_reduction_needs_preproduct(which: Int) -> Bool:
    """True where `src` must be a Hadamard product the caller materialized."""
    return which != RED_DT_BIAS and which != RED_CONV_BIAS


def mamba_backward_reduce_into(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    mut ones: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    which: Int,
    dims: MambaDims,
    m: Int,
) raises:
    """`out[w] = sum over the M rows of src[M, W]`, as a v1 `OP_NN` at `(1, W, M)`. For a `RED_CONV_W_*` id the caller writes into `out[d * D_CONV + mamba_reduction_tap(which)]` for each `d`, since the conv weight is `[d_inner, D_CONV]` row-major, and this launcher writes a CONTIGUOUS `[W]` vector."""
    identical_gemm_backward_bias_into(
        ctx, out, src, ones, ws, m, mamba_reduction_width(which, dims)
    )


def mamba_backward_ones_floats(m: Int) -> Int:
    """How long the shared ones vector is."""
    return identical_gemm_backward_bias_ones_floats(m)




def mamba_backward_proj_a_workspace_max_floats(
    which: Int, dims: MambaDims, m: Int
) -> Int:
    var fwd = mamba_proj_forward_call(which, dims, m)
    comptime if SAB_BWD_WS_FROM_FORWARD:
        return identical_gemm_workspace_max_floats(fwd[1], fwd[2], fwd[3])
    return identical_gemm_backward_a_workspace_max_floats(
        fwd[0], fwd[1], fwd[2], fwd[3]
    )


def mamba_backward_proj_b_workspace_max_floats(
    which: Int, dims: MambaDims, m: Int
) -> Int:
    """The `dB` sizing, and the one most likely to surprise, because `k'` is the token count and a plan chosen at one batch size is not the plan chosen at another."""
    var fwd = mamba_proj_forward_call(which, dims, m)
    comptime if SAB_BWD_WS_FROM_FORWARD:
        return identical_gemm_workspace_max_floats(fwd[1], fwd[2], fwd[3])
    return identical_gemm_backward_b_workspace_max_floats(
        fwd[0], fwd[1], fwd[2], fwd[3]
    )


def mamba_backward_reduce_workspace_max_floats(
    which: Int, dims: MambaDims, m: Int
) -> Int:
    return identical_gemm_backward_bias_workspace_max_floats(
        m, mamba_reduction_width(which, dims)
    )


def mamba_backward_workspace_max_floats(dims: MambaDims, m: Int) -> Int:
    """ONE number a backward pass can allocate once and reuse for all fourteen routed calls at this shape."""
    var w = 1
    for i in range(PROJ_COUNT):
        var wa = mamba_backward_proj_a_workspace_max_floats(i, dims, m)
        if wa > w:
            w = wa
        var wb = mamba_backward_proj_b_workspace_max_floats(i, dims, m)
        if wb > w:
            w = wb
    for i in range(RED_COUNT):
        var wr = mamba_backward_reduce_workspace_max_floats(i, dims, m)
        if wr > w:
            w = wr
    return w



comptime T1_DIRECTION_IS_DESCENDING_IN_T = True
comptime T1_SEED_IS_POSITIVE_ZERO = True
comptime T1_IS_FUSED = True
comptime T1_DA_INDEX_OFFSET = 1

comptime T2_H_IS_CHECKPOINTED = True
comptime T2_H_SUBTRACTION_REFUSED = True


def t2_h_checkpoint_floats(b: Int, l: Int, dims: MambaDims) -> Int:
    """`B * d_inner * (L + 1) * D_STATE`, the checkpoint's size."""
    return b * dims.d_inner * (l + 1) * D_STATE


comptime T3_DIRECTION_IS_ASCENDING_IN_D = True
comptime T3_SEED_IS_POSITIVE_ZERO = True
comptime T3_IS_FUSED = True
comptime T3_DB_PREFORMS_DELTA_TIMES_U = True


def t3_dh_floats(b: Int, l: Int, dims: MambaDims) -> Int:
    """`B * d_inner * L * D_STATE`, the materialized `dh`. The declared spelling is GATE-SCALE; a blocked variant is phase K and its tiling must be `contract_partition(d_inner)`'s leaf boundary, never a VRAM budget, or the answer becomes a function of the machine."""
    return b * dims.d_inner * l * D_STATE


comptime T4_DIRECTION_IS_DESCENDING_IN_T = True
comptime T4_SEED_IS_POSITIVE_ZERO = True
comptime T4_IS_FUSED = True
comptime T4_ROUTED_TO_GEMM = False

comptime T5_USES_PRIVATE_SLOTS = True
comptime T5_DIRECTION_IS_ASCENDING_IN_B = True
comptime T5_SEED_IS_POSITIVE_ZERO = True
comptime T5_ATOMIC_REFUSED = True

comptime T6_JOIN_ORDER_D_THEN_SCAN_THEN_XPROJ = True

comptime T7_RSTD3_IS_LEFT_ASSOCIATED = True
comptime T7_DIVIDES_BY_DMODEL_ONCE_PER_ROW = True
comptime T7_FINAL_SUBTRACT_IS_UNFUSED = True

comptime T8_DRES_IS_THE_LEFT_OPERAND = True

comptime S14B_USES_SIGMOID_MULTIPLY = True
comptime S14B_GUARD_THRESHOLD_IS_20 = True

comptime RECOMPUTE_USES_THE_FORWARD_SPELLING = True


comptime TOPO_T1 = 0
comptime TOPO_T2 = 1
comptime TOPO_T3 = 2
comptime TOPO_T4 = 3
comptime TOPO_T5 = 4
comptime TOPO_T6 = 5
comptime TOPO_T7 = 6
comptime TOPO_T8 = 7
comptime TOPO_S14B = 8
comptime TOPO_COUNT = 9


def mamba_backward_topology_site(which: Int) -> String:
    """`"T1 file::function"`, the kernel that implements one declaration."""
    if which == TOPO_T1:
        return String(
            "T1 mamba/impl/mamba_ssm/ops/selective_scan_backward.mojo"
            "::selective_scan_bwd_scan_kernel"
        )
    if which == TOPO_T2:
        return String(
            "T2 mamba/impl/mamba_ssm/ops/selective_scan_backward.mojo"
            "::selective_scan_checkpoint_kernel (producer),"
            " ::selective_scan_bwd_scan_kernel (reader)"
        )
    if which == TOPO_T3:
        return String(
            "T3 mamba/impl/mamba_ssm/ops/selective_scan_backward.mojo"
            "::mamba_bwd_dbc_kernel"
        )
    if which == TOPO_T4:
        return String(
            "T4 mamba/impl/mamba_ssm/ops/selective_scan_backward.mojo"
            "::mamba_bwd_da_partial_kernel"
        )
    if which == TOPO_T5:
        return String(
            "T5 mamba/impl/mamba_ssm/ops/selective_scan_backward.mojo"
            "::mamba_bwd_param_fold_kernel"
        )
    if which == TOPO_T6:
        return String(
            "T6 mamba/impl/transformers/models/mamba/"
            "modeling_mamba_backward.mojo::mamba_bwd_du_join_kernel"
        )
    if which == TOPO_T7:
        return String(
            "T7 mamba/impl/transformers/models/mamba/"
            "modeling_mamba_backward.mojo::mamba_bwd_norm_kernel"
        )
    if which == TOPO_T8:
        return String(
            "T8 mamba/impl/transformers/models/mamba/"
            "modeling_mamba_backward.mojo::mamba_bwd_norm_kernel (the final"
            " store; THE ABSORPTION SITE)"
        )
    if which == TOPO_S14B:
        return String(
            "S14b mamba/impl/transformers/models/mamba/"
            "modeling_mamba_backward.mojo::mamba_bwd_ddtp_kernel"
        )
    return String("?")


def mamba_backward_kernel_sites() -> String:
    """Every declaration beside the function that implements it, one block a gate can print and diff across builds and vendors."""
    var s = String("mojolearn.identical.mamba1.bwd.fp32.v1 KERNEL SITES\n")
    for i in range(TOPO_COUNT):
        s = s + mamba_backward_topology_site(i) + "\n"
    return s


def mamba_backward_declarations() -> String:
    """Every declared choice, as one printable block a gate can diff."""
    var s = String("mojolearn.identical.mamba1.bwd.fp32.v1 DECLARATIONS\n")
    s = s + "T1 dh recurrence: t DESCENDING, seed +0.0, FUSED, da offset "
    s = s + String(T1_DA_INDEX_OFFSET) + "\n"
    s = s + "T2 h[t-1]: CHECKPOINTED; upstream subtraction REFUSED\n"
    s = s + "T3 dB/dC over d_inner: d ASCENDING, seed +0.0, FUSED,"
    s = s + " gemm v1 at (1, " + String(D_STATE) + ", d_inner);"
    s = s + " dB pre-forms delta*u\n"
    s = s + "T4 dA over t: t DESCENDING, seed +0.0, FUSED, NOT routed\n"
    s = s + "T5 parameter fold over b: PRIVATE SLOTS, b ASCENDING,"
    s = s + " seed +0.0, atomics REFUSED\n"
    s = s + "T6 du join: D-skip, then scan, then x_proj; three flushed adds\n"
    s = s + "T7 RMSNorm bwd: rstd^3 left-associated, one div per row,"
    s = s + " final subtract UNFUSED\n"
    s = s + "T8 dx join: dres LEFT, one flushed add (ABSORPTION SITE)\n"
    s = s + "S14b softplus': MULTIPLY by identical_sigmoid, guard <= 20\n"
    s = s + "recompute: forward spelling always\n"
    s = s + "sabotage routing: " + mamba_backward_sabotage_name() + "\n"
    s = s + "sabotage bwd scan: " + mamba_backward_scan_sabotage_name() + "\n"
    s = s + (
        "sabotage bwd block: " + mamba_backward_block_sabotage_name() + "\n"
    )
    return s
