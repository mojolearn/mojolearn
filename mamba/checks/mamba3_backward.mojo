# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Routing, workspace sizing, and topology declarations for Mamba-3 backward."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined

from gemm.checks.gemm_oracle import (
    OP_NN,
    OP_NT,
    contract_leaf_size,
    leaf_count,
)
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
from mamba.checks.mamba3_fixture import (
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    Mamba3Dims,
)



comptime SAB_BWD3_PROJ_SWAP = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_BWD_PROJ_SWAP"
]()
comptime SAB_BWD3_WS_FROM_FORWARD = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_BWD_WS_FROM_FORWARD"
]()
comptime SAB_BWD3_ROTATE_HALF_SPLIT = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_BWD_ROTATE_HALF_SPLIT"
]()
comptime SAB_BWD3_BIAS_WIDTH_HEADS_ONLY = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_BWD_BIAS_WIDTH_HEADS_ONLY"
]()

comptime ANY_BWD3_SABOTAGE = (
    SAB_BWD3_PROJ_SWAP
    or SAB_BWD3_WS_FROM_FORWARD
    or SAB_BWD3_ROTATE_HALF_SPLIT
    or SAB_BWD3_BIAS_WIDTH_HEADS_ONLY
)


def mamba3_backward_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for a check's banner. A check must print THIS, plus `gemm_backward_sabotage_name()`, plus the forward block's and the SISO core's own names, because a backward binary can carry a forward sabotage and a gemm sabotage as well."""
    comptime if SAB_BWD3_PROJ_SWAP:
        return String("BWD3_PROJ_SWAP")
    comptime if SAB_BWD3_WS_FROM_FORWARD:
        return String("BWD3_WS_FROM_FORWARD")
    comptime if SAB_BWD3_ROTATE_HALF_SPLIT:
        return String("BWD3_ROTATE_HALF_SPLIT")
    comptime if SAB_BWD3_BIAS_WIDTH_HEADS_ONLY:
        return String("BWD3_BIAS_WIDTH_HEADS_ONLY")
    return String("none")


def mamba3_backward_inert_sabotages() -> String:
    """The arms a gate must NOT count as covered by this lane, with the reason."""
    return (
        String("SAB_BWD_OPERAND_ORDER: INERT here. Both projections are")
        + " OP_NT, so both backward calls put dC on the LEFT.\n"
        + "SAB_BWD_UNTRANSPOSED: PARTIAL, as in the sibling lanes.\n"
        + "SAB_BWD3_ROTATE_HALF_SPLIT on pairs >= num_rope_angles: VACUOUS"
        + " BY CONSTRUCTION. rope_fraction = 0.5 leaves pairs 32..63"
        + " unrotated and DEVIATION 828 records that the computed and the"
        + " structural spellings coincide bit for bit, because"
        + " cos(+0.0) = 1.0 and sin(+0.0) = +0.0 exactly. Any gate that"
        + " expects an arm to fire there must declare it VACUOUS IN"
        + " ADVANCE, the way the forward declared FOLD_SERIAL_ZERO_SEED"
        + " vacuous.\n"
        + "Every rotation arm at theta == 0: INERT. sin is exactly +0.0"
        + " there and the transpose is one character in two branches, so a"
        + " sign error produces a plausible correctly-shaped wrong gradient"
        + " that is bit identical on three vendors. The fixture must plant"
        + " NONZERO angles, the same requirement DIAG_INCLUDE_SUBTRACT has"
        + " on the forward side.\n"
        + "DEVIATION 1371's join arms, NINE sites: INERT on any fixture"
        + " where one leg dominates the others.\n"
        + "DEVIATION 1372's shifted leg: INERT on any fixture where dt and"
        + " sigma(trap) are UNIFORM across tokens.\n"
        + "DEVIATION 1368's clamp mask: VACUOUS unless dd_A < -9999 is"
        + " PLANTED. That is the only region where S5's A_floor bound"
        + " binds, and the forward's own A_FLOOR_UNCLAMPED arm carries the"
        + " same requirement.\n"
        + "The gemm fold arms: reachable through exactly TWO doors here,"
        + " the dB weight gradients at k' = M and DEVIATION 1364's dscale"
        + " cell at k = P*N. Every other contraction in this backward has"
        + " k <= 128 and is ONE LEAF."
    )



comptime PROJ3_IN = 0
comptime PROJ3_OUT = 1

comptime PROJ3_COUNT = 2


def mamba3_proj_name(which: Int) -> String:
    if which == PROJ3_IN:
        return String("in_proj")
    if which == PROJ3_OUT:
        return String("out_proj")
    return String("?")


def mamba3_proj_forward_call(
    which: Int, dims: Mamba3Dims, m: Int
) -> Tuple[Int, Int, Int, Int]:
    """`(op, m, n, k)` of the FORWARD projection, from the block's dimensions."""
    var dm = dims.d_model
    var di = dims.d_inner
    if which == PROJ3_IN:
        return (OP_NT, m, dims.d_in_proj(), dm)
    return (OP_NT, m, dm, di)


def mamba3_backward_proj_a_call(
    which: Int, dims: Mamba3Dims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the ACTIVATION gradient of `which`."""
    var fwd = mamba3_proj_forward_call(which, dims, m)
    comptime if SAB_BWD3_PROJ_SWAP:
        return gemm_backward_b_call(fwd[0], fwd[1], fwd[2], fwd[3])
    return gemm_backward_a_call(fwd[0], fwd[1], fwd[2], fwd[3])


def mamba3_backward_proj_b_call(
    which: Int, dims: Mamba3Dims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the WEIGHT gradient of `which`."""
    var fwd = mamba3_proj_forward_call(which, dims, m)
    comptime if SAB_BWD3_PROJ_SWAP:
        return gemm_backward_a_call(fwd[0], fwd[1], fwd[2], fwd[3])
    return gemm_backward_b_call(fwd[0], fwd[1], fwd[2], fwd[3])


def mamba3_backward_proj_call_name(
    which: Int, call: Tuple[Int, Int, Int, Int, Int]
) -> String:
    """`"out_proj NN(dC,W) 64x32x8"`, for a gate's per-case banner."""
    return mamba3_proj_name(which) + " " + gemm_backward_call_name(call)


def mamba3_backward_proj_a_into(
    ctx: DeviceContext,
    mut d_input: DeviceBuffer[DType.float32],
    mut d_output: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    which: Int,
    dims: Mamba3Dims,
    m: Int,
) raises:
    """The ACTIVATION gradient of projection `which`, into `d_input`. ASYNCHRONOUS, caller-owned buffers, `[[mojo-buffer-freed-at-last-use]]`."""
    var fwd = mamba3_proj_forward_call(which, dims, m)
    comptime if SAB_BWD3_PROJ_SWAP:
        identical_gemm_backward_b_into(
            ctx, d_input, d_output, weight, ws, fwd[1], fwd[2], fwd[3], fwd[0]
        )
        return
    identical_gemm_backward_a_into(
        ctx, d_input, d_output, weight, ws, fwd[1], fwd[2], fwd[3], fwd[0]
    )


def mamba3_backward_proj_b_into(
    ctx: DeviceContext,
    mut d_weight: DeviceBuffer[DType.float32],
    mut d_output: DeviceBuffer[DType.float32],
    mut input_stage: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    which: Int,
    dims: Mamba3Dims,
    m: Int,
) raises:
    """The WEIGHT gradient of projection `which`, into `d_weight`. ASYNCHRONOUS, caller-owned buffers."""
    var fwd = mamba3_proj_forward_call(which, dims, m)
    comptime if SAB_BWD3_PROJ_SWAP:
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



comptime RED3_D = 0
comptime RED3_DT_BIAS = 1
comptime RED3_BNORM_W = 2
comptime RED3_CNORM_W = 3
comptime RED3_B_BIAS = 4
comptime RED3_C_BIAS = 5
comptime RED3_NORM_W = 6

comptime RED3_COUNT = 7


def mamba3_reduction_name(which: Int) -> String:
    if which == RED3_D:
        return String("dD")
    if which == RED3_DT_BIAS:
        return String("d.dt_bias")
    if which == RED3_BNORM_W:
        return String("d.B_norm.weight")
    if which == RED3_CNORM_W:
        return String("d.C_norm.weight")
    if which == RED3_B_BIAS:
        return String("d.B_bias")
    if which == RED3_C_BIAS:
        return String("d.C_bias")
    if which == RED3_NORM_W:
        return String("d.block_norm.weight")
    return String("?")


def mamba3_bias_reduction_width(dims: Mamba3Dims) -> Int:
    """`H * N`, the width of one `[H, N]` bias plane reduced over tokens."""
    comptime if SAB_BWD3_BIAS_WIDTH_HEADS_ONLY:
        return dims.nheads
    return dims.nheads * M3_D_STATE


def mamba3_reduction_width(which: Int, dims: Mamba3Dims) -> Int:
    """`W`, the length of the reduced vector."""
    if which == RED3_NORM_W:
        return dims.d_model
    if which == RED3_BNORM_W or which == RED3_CNORM_W:
        return M3_D_STATE
    if which == RED3_B_BIAS or which == RED3_C_BIAS:
        return mamba3_bias_reduction_width(dims)
    return dims.nheads


def mamba3_reduction_needs_preproduct(which: Int) -> Bool:
    """True where `src` must be a Hadamard product the caller materialized."""
    return (
        which == RED3_BNORM_W
        or which == RED3_CNORM_W
        or which == RED3_NORM_W
    )


def mamba3_backward_reduce_into(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    mut ones: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    which: Int,
    dims: Mamba3Dims,
    m: Int,
) raises:
    """`out[w] = sum over the M rows of src[M, W]`, as a v1 `OP_NN` at `(1, W, M)`. ASYNCHRONOUS, caller-owned buffers, INCLUDING `ones` and `src`."""
    identical_gemm_backward_bias_into(
        ctx, out, src, ones, ws, m, mamba3_reduction_width(which, dims)
    )


def mamba3_backward_ones_floats(m: Int) -> Int:
    """How long the shared ones vector is."""
    return identical_gemm_backward_bias_ones_floats(m)




def mamba3_rope_pair_count() -> Int:
    """`N / 2 = 64` interleaved pairs per head, of which the first `num_rope_angles = 32` are rotated."""
    return M3_D_STATE // 2


def mamba3_rope_pair_is_rotated(pair: Int) -> Bool:
    """`pair < num_rope_angles`. **A SABOTAGE ARM AGAINST THOSE PAIRS IS INERT BY CONSTRUCTION** and must be declared VACUOUS in advance: DEVIATION 828 records that the computed and structural spellings coincide bit for bit because `cos(+0.0) = 1.0` and `sin(+0.0) = +0.0` exactly, and the backward inherits that coincidence unchanged."""
    return pair < M3_NUM_ROPE_ANGLES


def mamba3_rope_pair_indices(pair: Int) -> Tuple[Int, Int]:
    """The two component indices of `pair`, INTERLEAVED: `(2p, 2p+1)`. It is also not gated: that file records "COMPILED AND RUN 2026-09-03, NOT YET GATED" and that its check REFUSED TO CERTIFY."""
    comptime if SAB_BWD3_ROTATE_HALF_SPLIT:
        return (pair, pair + (M3_D_STATE // 2))
    return (2 * pair, 2 * pair + 1)



comptime CELL3_DCOEF = 0  #: D10, `dcoef[i,h] = sum_p dY * v`; serves dD too
comptime CELL3_DE17 = 1  #: D15, `de17[i,h] = sum_p dY_pre * dot17`
comptime CELL3_DQ_ROT_17 = 2  #: D16, `dq_rot = sum_p d_dot17 * h`
comptime CELL3_DH_IN_17 = 3  #: D17, `dh_in = sum_i d_dot17 * q_rot`
comptime CELL3_DM = 4  #: D18, `dM[i,j,h] = sum_p dY_pre * v[j]`
comptime CELL3_DV_16 = 5  #: D19, TRANSPOSED STRICT mask
comptime CELL3_DQ_ROT_16 = 6  #: D23, `dq_rot = sum_j ds * k_scaled`
comptime CELL3_DK_SCALED_16 = 7  #: D24, TRANSPOSED STRICT mask
comptime CELL3_DSCALE_C = 8  #: D26, the ONLY multi-leaf cell in this backward
comptime CELL3_DK_SCALED_20 = 9  #: D29, `sum_p dincr * v_dec`
comptime CELL3_DV_DEC = 10  #: D30, `sum_n dincr * k_scaled`
comptime CELL3_D_DREV = 11  #: D32, `sum_p dv_dec * v`
comptime CELL3_DSCALE_15 = 12  #: D38, `dscale[i,h] = sum_n dk_scaled * k_rot`
comptime CELL3_DDT_10 = 13  #: D71, `ddt_10[t,h] = sum_r d_inc * a`

comptime CELL3_COUNT = 14


def mamba3_backward_cell_name(which: Int) -> String:
    if which == CELL3_DCOEF:
        return String("dcoef")
    if which == CELL3_DE17:
        return String("de17")
    if which == CELL3_DQ_ROT_17:
        return String("dq_rot.S17")
    if which == CELL3_DH_IN_17:
        return String("dh_in.S17")
    if which == CELL3_DM:
        return String("dM")
    if which == CELL3_DV_16:
        return String("dv.S16.transposed")
    if which == CELL3_DQ_ROT_16:
        return String("dq_rot.S16")
    if which == CELL3_DK_SCALED_16:
        return String("dk_scaled.S16.transposed")
    if which == CELL3_DSCALE_C:
        return String("dscale_c")
    if which == CELL3_DK_SCALED_20:
        return String("dk_scaled.S20")
    if which == CELL3_DV_DEC:
        return String("dv_decayed.S20")
    if which == CELL3_D_DREV:
        return String("d_drev.S20")
    if which == CELL3_DSCALE_15:
        return String("dscale.S15")
    if which == CELL3_DDT_10:
        return String("ddt.S10")
    return String("?")


def mamba3_backward_cell_k(which: Int) -> Int:
    """The CONTRACTED LENGTH of one backward cell."""
    if which == CELL3_DCOEF:
        return M3_HEADDIM
    if which == CELL3_DE17:
        return M3_HEADDIM
    if which == CELL3_DQ_ROT_17:
        return M3_HEADDIM
    if which == CELL3_DH_IN_17:
        return M3_CHUNK_SIZE
    if which == CELL3_DM:
        return M3_HEADDIM
    if which == CELL3_DV_16:
        return M3_CHUNK_SIZE
    if which == CELL3_DQ_ROT_16:
        return M3_CHUNK_SIZE
    if which == CELL3_DK_SCALED_16:
        return M3_CHUNK_SIZE
    if which == CELL3_DSCALE_C:
        return M3_HEADDIM * M3_D_STATE
    if which == CELL3_DK_SCALED_20:
        return M3_HEADDIM
    if which == CELL3_DV_DEC:
        return M3_D_STATE
    if which == CELL3_D_DREV:
        return M3_HEADDIM
    if which == CELL3_DSCALE_15:
        return M3_D_STATE
    if which == CELL3_DDT_10:
        return M3_NUM_ROPE_ANGLES
    return 0


def mamba3_backward_cell_leaves(which: Int) -> Tuple[Int, Int]:
    """`(leaf_size, leaf_count)` for one cell, from `contract_leaf_size`."""
    var k = mamba3_backward_cell_k(which)
    var el = contract_leaf_size(k)
    return (el, leaf_count(k, el))




def mamba3_backward_proj_a_workspace_max_floats(
    which: Int, dims: Mamba3Dims, m: Int
) -> Int:
    var fwd = mamba3_proj_forward_call(which, dims, m)
    comptime if SAB_BWD3_WS_FROM_FORWARD:
        return identical_gemm_workspace_max_floats(fwd[1], fwd[2], fwd[3])
    return identical_gemm_backward_a_workspace_max_floats(
        fwd[0], fwd[1], fwd[2], fwd[3]
    )


def mamba3_backward_proj_b_workspace_max_floats(
    which: Int, dims: Mamba3Dims, m: Int
) -> Int:
    """The `dB` sizing, and the one most likely to surprise, because `k'` is the token count and a plan chosen at one batch size is not the plan chosen at another."""
    var fwd = mamba3_proj_forward_call(which, dims, m)
    comptime if SAB_BWD3_WS_FROM_FORWARD:
        return identical_gemm_workspace_max_floats(fwd[1], fwd[2], fwd[3])
    return identical_gemm_backward_b_workspace_max_floats(
        fwd[0], fwd[1], fwd[2], fwd[3]
    )


def mamba3_backward_reduce_workspace_max_floats(
    which: Int, dims: Mamba3Dims, m: Int
) -> Int:
    return identical_gemm_backward_bias_workspace_max_floats(
        m, mamba3_reduction_width(which, dims)
    )


def mamba3_backward_workspace_max_floats(dims: Mamba3Dims, m: Int) -> Int:
    """ONE number a backward pass can allocate once and reuse for all eleven routed calls at this shape."""
    var w = 1
    for i in range(PROJ3_COUNT):
        var wa = mamba3_backward_proj_a_workspace_max_floats(i, dims, m)
        if wa > w:
            w = wa
        var wb = mamba3_backward_proj_b_workspace_max_floats(i, dims, m)
        if wb > w:
            w = wb
    for i in range(RED3_COUNT):
        var wr = mamba3_backward_reduce_workspace_max_floats(i, dims, m)
        if wr > w:
            w = wr
    return w


def mamba3_backward_state_floats(b: Int, l: Int, dims: Mamba3Dims) -> Int:
    """`B * C * H * P * N`, the incoming-per-chunk state the backward reads. Mamba-1's T2 is deleted here for Mamba-2's reason: the quantity upstream's `h[t] - dbu[t]` recovery fights to obtain is materialized at chunk boundaries by construction, so there is nothing to refuse and nothing to price."""
    var c = (l + M3_CHUNK_SIZE - 1) // M3_CHUNK_SIZE
    if l <= 0:
        c = 0
    return b * c * dims.nheads * M3_HEADDIM * M3_D_STATE


def mamba3_backward_theta_chain_floats(
    b: Int, l: Int, dims: Mamba3Dims
) -> Int:
    """`B * H * L * R`, the angle-gradient chain DEVIATION 1367 declares."""
    if l <= 0:
        return 0
    return b * dims.nheads * l * M3_NUM_ROPE_ANGLES



comptime B3T1_DIRECTION_IS_DESCENDING_IN_C = True
comptime B3T1_SEED_IS_POSITIVE_ZERO = True
comptime B3T1_IS_FUSED = True
comptime B3T1_SCALE_INDEX_OFFSET = 1
comptime B3T1_WITNESSED_FROM_L = 129

comptime B3T3_DIRECTION_IS_ASCENDING_IN_H = True
comptime B3T3_SEED_IS_POSITIVE_ZERO = True
comptime B3T3_ADD_IS_UNFUSED = True
comptime B3T3_SITES = 2
comptime B3T3_SITS_AFTER_THE_ROTATION_TRANSPOSE = True
comptime B3T3_H_FOLD_IS_ONE_LEAF_BELOW_DMODEL = 4096
comptime B3T3_IS_VACUOUS_AT_DMODEL_32 = True

comptime B3_INHERITS_MAMBA2_BACKWARD_1345_TO_1349 = True
comptime B3_CLAMP_BOUND_BINDS_STRUCTURALLY = True
comptime B3_CLAMP_CHECK_ARM_IS_STILL_OWED = True

comptime M3_1_INNER_FTZ_IS_THE_BARRIER = True
comptime M3_1_IS_UNFUSED = True
comptime M3_1_TENSORS = 2
comptime M3_1_EXPRESSIONS_PER_PAIR = 4
comptime M3_1_TRIG_IS_THE_PORTABLE_PAIR = True

comptime M3_2_USES_THE_COLLAPSED_SPELLING = True
comptime M3_2_INNER_FTZ_IS_THE_BARRIER = True
comptime M3_2_REQUIRES_ROT_Q_AND_ROT_K_AS_BUFFERS = True

comptime M3_3_DIRECTION_IS_DESCENDING_IN_T = True
comptime M3_3_SEED_IS_POSITIVE_ZERO = True
comptime M3_3_ADD_IS_UNFUSED = True
comptime M3_3_COEFFICIENT_IS_STRUCTURALLY_ONE = True
comptime M3_3_HAS_NO_CHUNK_RESET = True
comptime M3_3_NO_DECODE_BACKWARD_EXISTS = True
comptime M3_3_UPSTREAM_ATOMIC_REFUSED = True

comptime M3_4_IS_A_GATHER_NOT_A_SCATTER = True
comptime M3_4_BOUNDARY_TERMS_ARE_STRUCTURAL_ZERO = True
comptime M3_4_ARM_NEEDS_NONUNIFORM_DT_AND_TRAP = True

comptime M3_5_USES_THE_SQUARED_FORWARD_VALUE = True
comptime M3_5_BRANCHES_ON_THE_INPUT_SIGN = True
comptime M3_5_BRANCHLESS_RESPELLING_REFUSED = True

comptime M3_MOD2PI_JACOBIAN_IS_A_STRUCTURAL_COPY = True

comptime B3_JOIN_RULE_IS_ASCENDING_SEAM_ORDER = True
comptime B3_JOIN_SITES = 9
comptime B3_JOIN_ADDS_ARE_UNFUSED = True
comptime B3_DDT_JOIN_IS_FOUR_WAY = True
comptime B3_DV_JOIN_IS_THREE_WAY = True
comptime B3_JOIN_ARMS_PREDICTED_INERT_IF_ONE_LEG_DOMINATES = True

comptime B3T7_RSTD3_IS_LEFT_ASSOCIATED = True
comptime B3T7_DIVIDES_BY_ROW_WIDTH_ONCE_PER_ROW = True
comptime B3T7_FINAL_SUBTRACT_IS_UNFUSED = True
comptime B3T7_SITES = 3
comptime B3T7_RECONCILIATION_WITH_TRANSFORMER_LANE_IS_OWED = True

comptime B3T8_DRES_IS_THE_LEFT_OPERAND = True

comptime B3_T2_HAS_NO_SITE = True
comptime B3_T4_HAS_NO_SITE = True
comptime B3_T5_HAS_NO_SITE = True
comptime B3_TWO_AXIS_DD_HAS_NO_SITE = True

comptime B3_SOFTPLUS_D_USES_SIGMOID_MULTIPLY = True
comptime B3_SOFTPLUS_D_GUARD_THRESHOLD_IS_20 = True
comptime B3_EXP_DERIVATIVE_REUSES_THE_FORWARD_VALUE = True
comptime B3_TANH_D_IS_ONE_MINUS_TANH_SQUARED = True
comptime B3_TANH_D_ASSOCIATION_IS_LEFT_TO_RIGHT = True
comptime B3_SIGMOID_D_REUSES_THE_CARDED_SIGMA = True
comptime B3_RECOMPUTE_USES_THE_FORWARD_SPELLING = True

comptime B3_UPSTREAM_AUTOTUNED_CHUNK_SIZE_REFUSED = True
comptime B3_UPSTREAM_ATOMICS_REFUSED = True
comptime B3_NO_SECOND_REFERENCE_EXISTS = True



comptime TOPO3_T1_CHUNK = 0
comptime TOPO3_T3_HEAD = 1
comptime TOPO3_INHERITED_M2 = 2
comptime TOPO3_M3_1_ROT = 3
comptime TOPO3_M3_2_DTHETA = 4
comptime TOPO3_M3_3_CHAIN = 5
comptime TOPO3_M3_4_SHIFT = 6
comptime TOPO3_M3_5_HEAVY = 7
comptime TOPO3_MOD2PI = 8
comptime TOPO3_JOIN = 9
comptime TOPO3_T7_NORM = 10
comptime TOPO3_T8_ABSORB = 11

comptime TOPO3_COUNT = 12


def mamba3_backward_topology_name(which: Int) -> String:
    if which == TOPO3_T1_CHUNK:
        return String("T1@chunks 1362 reverse chunk recurrence")
    if which == TOPO3_T3_HEAD:
        return String("T3 1363 head fold at H, after the rotation")
    if which == TOPO3_INHERITED_M2:
        return String("1364 mamba2 backward 1345-1349, inherited")
    if which == TOPO3_M3_1_ROT:
        return String("M3-1 1365 transposed rotation, inner ftz barrier")
    if which == TOPO3_M3_2_DTHETA:
        return String("M3-2 1366 dtheta, collapsed spelling")
    if which == TOPO3_M3_3_CHAIN:
        return String("M3-3 1367 whole-sequence reverse angle chain")
    if which == TOPO3_M3_4_SHIFT:
        return String("M3-4 1368 trapezoid shifted leg, a gather")
    if which == TOPO3_M3_5_HEAVY:
        return String("M3-5 1369 heavy_tail derivative")
    if which == TOPO3_MOD2PI:
        return String("1370 mod2pi Jacobian, a structural copy")
    if which == TOPO3_JOIN:
        return String("JOIN 1371 ascending seam order, 9 sites")
    if which == TOPO3_T7_NORM:
        return String("T7 1372 RMSNorm backward, 3 sites")
    if which == TOPO3_T8_ABSORB:
        return String("T8 1373 residual absorption")
    return String("?")


def mamba3_backward_topology_site(which: Int) -> String:
    """Where the kernel for one declaration WILL live, and that it does not."""
    var core = (
        which == TOPO3_T1_CHUNK
        or which == TOPO3_INHERITED_M2
        or which == TOPO3_M3_1_ROT
        or which == TOPO3_M3_2_DTHETA
    )
    if core:
        return (
            mamba3_backward_topology_name(which)
            + " -- NOT WRITTEN; mamba/impl/mamba_ssm/ops/"
            + "mamba3_siso_backward.mojo (does not exist)"
        )
    return (
        mamba3_backward_topology_name(which)
        + " -- NOT WRITTEN; mamba/impl/mamba_ssm/modules/"
        + "mamba3_backward.mojo (does not exist)"
    )


def mamba3_backward_declarations() -> String:
    """Every declared choice, as one printable block a gate can diff."""
    var s = String(
        "mojolearn.identical.mamba3.siso.bwd.fp32.v1 DECLARATIONS\n"
    )
    s = s + "1362 T1@chunks: c DESCENDING, seed +0.0, FUSED, scale offset "
    s = s + String(B3T1_SCALE_INDEX_OFFSET) + "; witnessed from L = "
    s = s + String(B3T1_WITNESSED_FROM_L) + "\n"
    s = s + "1363 T3 head fold: h ASCENDING, seed +0.0, UNFUSED, "
    s = s + String(B3T3_SITES) + " sites at length H, AFTER the rotation"
    s = s + " transpose; ONE LEAF below d_model "
    s = s + String(B3T3_H_FOLD_IS_ONE_LEAF_BELOW_DMODEL) + "\n"
    s = s + "1364 INHERITS mamba2 bwd 1345-1349 unchanged; the S5 clamp"
    s = s + " bound BINDS here and needs a planted dd_A < -9999\n"
    s = s + "1365 rotation transpose: inner ftz barrier, UNFUSED, "
    s = s + String(M3_1_TENSORS) + " tensors x "
    s = s + String(M3_1_EXPRESSIONS_PER_PAIR) + " expressions per pair;"
    s = s + " interleaved (2p, 2p+1) pairing\n"
    s = s + "1366 dtheta: the COLLAPSED spelling off carded rot.q/rot.k,"
    s = s + " upstream's six-product form is the required-RED arm\n"
    s = s + "1367 theta chain: t DESCENDING over the WHOLE SEQUENCE, seed"
    s = s + " +0.0, UNFUSED, coefficient structurally 1; NO DECODE BACKWARD\n"
    s = s + "1368 trapezoid shifted leg: a GATHER, boundary terms +0.0\n"
    s = s + "1369 heavy_tail': branch on dd_A's SIGN; left branch is"
    s = s + " pinned_mul(hv, hv) off the carded A.out\n"
    s = s + "1370 mod2pi': a STRUCTURAL COPY, no float operation\n"
    s = s + "1371 join rule: ASCENDING SEAM ORDER, left-associated, "
    s = s + String(B3_JOIN_SITES) + " sites (ddt FOUR-way, dv THREE-way),"
    s = s + " all predicted INERT if one leg dominates\n"
    s = s + "1372 T7 RMSNorm bwd: " + String(B3T7_SITES) + " sites;"
    s = s + " transformer-lane reconciliation OWED\n"
    s = s + "1373 T8 dx join: dres LEFT, one flushed add (ABSORPTION SITE)\n"
    s = s + "1374 deletions: T2, T4, T5 and mamba2's two-axis dD have NO"
    s = s + " SITE in this profile\n"
    s = s + "1078 softplus': MULTIPLY by identical_sigmoid, guard <= 20\n"
    s = s + "exp', tanh', sigmoid': the forward's rounded values reused\n"
    s = s + "REFUSED from upstream: autotuned CHUNK_SIZE, five atomics,"
    s = s + " the PTX cos/sin recompute\n"
    s = s + "sabotage: " + mamba3_backward_sabotage_name() + "\n"
    return s
