# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Routing, workspace sizing, and topology declarations for Mamba-2 backward."""

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
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    Mamba2Dims,
)



comptime SAB_BWD2_PROJ_SWAP = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_BWD_PROJ_SWAP"
]()
comptime SAB_BWD2_WS_FROM_FORWARD = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_BWD_WS_FROM_FORWARD"
]()
comptime SAB_BWD2_DD_TOKENS_FIRST = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_BWD_DD_TOKENS_FIRST"
]()
comptime SAB_BWD2_CONV_TAP_SLOT = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_BWD_CONV_TAP_SLOT"
]()

comptime ANY_BWD2_SABOTAGE = (
    SAB_BWD2_PROJ_SWAP
    or SAB_BWD2_WS_FROM_FORWARD
    or SAB_BWD2_DD_TOKENS_FIRST
    or SAB_BWD2_CONV_TAP_SLOT
)


def mamba2_backward_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for a check's banner. A check must print THIS, plus `gemm_backward_sabotage_name()`, plus the forward block's and the SSD core's own sabotage names, because a backward binary can carry a forward sabotage and a gemm sabotage as well and a banner naming only one of them mislabels the run."""
    comptime if SAB_BWD2_PROJ_SWAP:
        return String("BWD2_PROJ_SWAP")
    comptime if SAB_BWD2_WS_FROM_FORWARD:
        return String("BWD2_WS_FROM_FORWARD")
    comptime if SAB_BWD2_DD_TOKENS_FIRST:
        return String("BWD2_DD_TOKENS_FIRST")
    comptime if SAB_BWD2_CONV_TAP_SLOT:
        return String("BWD2_CONV_TAP_SLOT")
    return String("none")


def mamba2_backward_inert_sabotages() -> String:
    """The arms a gate must NOT count as covered by this lane, with the reason."""
    return (
        String("SAB_BWD_OPERAND_ORDER: INERT here. Both Mamba-2 projections")
        + " are OP_NT, so both backward calls put dC on the LEFT and the"
        + " side flag is never exercised.\n"
        + "SAB_BWD_UNTRANSPOSED: PARTIAL here. It moves the dA and dB shapes"
        + " for OP_NT, so it bites, but it cannot distinguish OP_NN from"
        + " OP_TN routing because this lane calls neither at a projection.\n"
        + "SAB_BWD_BIAS_AXIS: covered, but only at M != W. At a shape where"
        + " the token count equals a reduced width it returns a vector of"
        + " the right length and the wrong contents. Ten reductions here,"
        + " widths H, CD, d_inner and d_model; the gate must name which"
        + " widths its fixture separates from M.\n"
        + "FOLD_SERIAL_ZERO_SEED and the other gemm fold arms: REACHABLE"
        + " here for the first time in this profile through TWO doors, the"
        + " dB weight gradients at k' = M and DEVIATION 1345's dscale cell"
        + " at k = P*N = 8192, which is 64 leaves. The forward reached the"
        + " tree only at k = Q = 256, which is two.\n"
        + "SAB_BWD2_DD_TOKENS_FIRST: NEAR-INERT unless the per-token dD"
        + " contributions are planted at comparable magnitude.\n"
        + "Every arm named in DEVIATION 1350's join rule: INERT on any"
        + " fixture where one leg dominates the others. Five sites."
    )



comptime PROJ2_IN = 0
comptime PROJ2_OUT = 1

comptime PROJ2_COUNT = 2


def mamba2_proj_name(which: Int) -> String:
    if which == PROJ2_IN:
        return String("in_proj")
    if which == PROJ2_OUT:
        return String("out_proj")
    return String("?")


def mamba2_proj_forward_call(
    which: Int, dims: Mamba2Dims, m: Int
) -> Tuple[Int, Int, Int, Int]:
    """`(op, m, n, k)` of the FORWARD projection, from the block's dimensions."""
    var dm = dims.d_model
    var di = dims.d_inner
    if which == PROJ2_IN:
        return (OP_NT, m, dims.d_in_proj(), dm)
    return (OP_NT, m, dm, di)


def mamba2_backward_proj_a_call(
    which: Int, dims: Mamba2Dims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the ACTIVATION gradient of `which`."""
    var fwd = mamba2_proj_forward_call(which, dims, m)
    comptime if SAB_BWD2_PROJ_SWAP:
        return gemm_backward_b_call(fwd[0], fwd[1], fwd[2], fwd[3])
    return gemm_backward_a_call(fwd[0], fwd[1], fwd[2], fwd[3])


def mamba2_backward_proj_b_call(
    which: Int, dims: Mamba2Dims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the WEIGHT gradient of `which`. The forward is batch invariant because gemm contract 6.1 forbids the leaf size from depending on `m`; the weight gradient contracts over the batch, so the batch arrives as `k`, where section 6 of the same contract REQUIRES the leaf size to depend on it."""
    var fwd = mamba2_proj_forward_call(which, dims, m)
    comptime if SAB_BWD2_PROJ_SWAP:
        return gemm_backward_a_call(fwd[0], fwd[1], fwd[2], fwd[3])
    return gemm_backward_b_call(fwd[0], fwd[1], fwd[2], fwd[3])


def mamba2_backward_proj_call_name(
    which: Int, call: Tuple[Int, Int, Int, Int, Int]
) -> String:
    """`"out_proj NN(dC,W) 64x32x8"`, for a gate's per-case banner."""
    return mamba2_proj_name(which) + " " + gemm_backward_call_name(call)


def mamba2_backward_proj_a_into(
    ctx: DeviceContext,
    mut d_input: DeviceBuffer[DType.float32],
    mut d_output: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    which: Int,
    dims: Mamba2Dims,
    m: Int,
) raises:
    """The ACTIVATION gradient of projection `which`, into `d_input`. `ws` must hold at least `mamba2_backward_proj_a_workspace_max_floats(which, dims, m)` floats."""
    var fwd = mamba2_proj_forward_call(which, dims, m)
    comptime if SAB_BWD2_PROJ_SWAP:
        identical_gemm_backward_b_into(
            ctx, d_input, d_output, weight, ws, fwd[1], fwd[2], fwd[3], fwd[0]
        )
        return
    identical_gemm_backward_a_into(
        ctx, d_input, d_output, weight, ws, fwd[1], fwd[2], fwd[3], fwd[0]
    )


def mamba2_backward_proj_b_into(
    ctx: DeviceContext,
    mut d_weight: DeviceBuffer[DType.float32],
    mut d_output: DeviceBuffer[DType.float32],
    mut input_stage: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    which: Int,
    dims: Mamba2Dims,
    m: Int,
) raises:
    """The WEIGHT gradient of projection `which`, into `d_weight`. ASYNCHRONOUS, caller-owned buffers."""
    var fwd = mamba2_proj_forward_call(which, dims, m)
    comptime if SAB_BWD2_PROJ_SWAP:
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



comptime RED2_D = 0
comptime RED2_DT_BIAS = 1
comptime RED2_A_LOG = 2
comptime RED2_CONV_BIAS = 3
comptime RED2_CONV_W_TAP0 = 4
comptime RED2_CONV_W_TAP1 = 5
comptime RED2_CONV_W_TAP2 = 6
comptime RED2_CONV_W_TAP3 = 7
comptime RED2_NORM_W = 8
comptime RED2_GNORM_W = 9

comptime RED2_COUNT = 10


def mamba2_reduction_name(which: Int) -> String:
    if which == RED2_D:
        return String("dD")
    if which == RED2_DT_BIAS:
        return String("d.dt_bias")
    if which == RED2_A_LOG:
        return String("d.A_log.pre")
    if which == RED2_CONV_BIAS:
        return String("d.conv1d.bias")
    if which == RED2_NORM_W:
        return String("d.norm.weight")
    if which == RED2_GNORM_W:
        return String("d.gated_norm.weight")
    if which >= RED2_CONV_W_TAP0 and which <= RED2_CONV_W_TAP3:
        return String("d.conv1d.weight.tap") + String(
            which - RED2_CONV_W_TAP0
        )
    return String("?")


def mamba2_reduction_width(which: Int, dims: Mamba2Dims) -> Int:
    """`W`, the length of the reduced vector."""
    if which == RED2_NORM_W:
        return dims.d_model
    if which == RED2_GNORM_W:
        return dims.d_inner
    if which == RED2_CONV_BIAS or (
        which >= RED2_CONV_W_TAP0 and which <= RED2_CONV_W_TAP3
    ):
        return dims.conv_dim()
    return dims.nheads


def mamba2_reduction_tap(which: Int) -> Int:
    """The conv tap index `k` a `RED2_CONV_W_*` id writes, or `-1`. The gate must assert the fixture is asymmetric before counting this arm as covered."""
    if which < RED2_CONV_W_TAP0 or which > RED2_CONV_W_TAP3:
        return -1
    var k = which - RED2_CONV_W_TAP0
    comptime if SAB_BWD2_CONV_TAP_SLOT:
        return (M2_D_CONV - 1) - k
    return k


def mamba2_reduction_needs_preproduct(which: Int) -> Bool:
    """True where `src` must be a Hadamard product the caller materialized."""
    return which != RED2_DT_BIAS and which != RED2_CONV_BIAS


def mamba2_reduction_needs_inner_p_fold(which: Int) -> Bool:
    """True only for `RED2_D`."""
    return which == RED2_D


def mamba2_dd_inner_fold_call(dims: Mamba2Dims) -> Tuple[Int, Int, Int, Int]:
    """`(op, m, n, k)` of `dD`'s INNER fold, the one over `headdim`. `OP_NN` at `(1, 1, P)` per `(token, head)`: `P = 64`, one leaf, one serial ascending chain seeded `+0.0` with a fused multiply-add per term, which is gemm v1's own leaf."""
    return (OP_NN, 1, 1, M2_HEADDIM)


def mamba2_backward_reduce_into(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    mut ones: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    which: Int,
    dims: Mamba2Dims,
    m: Int,
) raises:
    """`out[w] = sum over the M rows of src[M, W]`, as a v1 `OP_NN` at `(1, W, M)`. For a `RED2_CONV_W_*` id the caller writes into `out[d * M2_D_CONV + mamba2_reduction_tap(which)]` for each `d`, since the conv weight is `[CD, D_CONV]` row-major and this launcher writes a CONTIGUOUS `[W]` vector."""
    identical_gemm_backward_bias_into(
        ctx, out, src, ones, ws, m, mamba2_reduction_width(which, dims)
    )


def mamba2_backward_ones_floats(m: Int) -> Int:
    """How long the shared ones vector is."""
    return identical_gemm_backward_bias_ones_floats(m)



comptime CELL2_DE18 = 0  #: C20, `de18[i,h] = sum_p dY_off * dot18`
comptime CELL2_DH_PREV = 1  #: C22, `dh_prev = sum_i d_dot18 * C`
comptime CELL2_DBDEC = 2  #: C27, `dBdec = sum_p dcstate * X_d`
comptime CELL2_DXD_16 = 3  #: C28, `dX_d_16 = sum_n dcstate * B_decay`
comptime CELL2_DDECAY = 4  #: C29, `ddecay = sum_n dBdec * B`
comptime CELL2_DM = 5  #: C33, `dM[i,j,h] = sum_p dY_diag * X_d[j]`
comptime CELL2_DXD_14 = 6  #: C34, the TRANSPOSED mask, DEVIATION 1346
comptime CELL2_DC_12 = 7  #: C38, `dC_12[i,n] = sum_j dG * B[j,n]`
comptime CELL2_DB_12 = 8  #: C39, `dB_12[j,n] = sum_i dG * C[i,n]`
comptime CELL2_DDT_X = 9  #: C44, `ddt_x[l,h] = sum_p dX_d * x`
comptime CELL2_DSCALE = 10  #: C24, DEVIATION 1345, the longest crossing fold
comptime CELL2_DD_INNER = 11  #: DEVIATION 1344's inner fold over headdim

comptime CELL2_COUNT = 12


def mamba2_backward_cell_name(which: Int) -> String:
    if which == CELL2_DE18:
        return String("de18")
    if which == CELL2_DH_PREV:
        return String("dh_prev")
    if which == CELL2_DBDEC:
        return String("dB_decay")
    if which == CELL2_DXD_16:
        return String("dX_d.S16")
    if which == CELL2_DDECAY:
        return String("ddecay")
    if which == CELL2_DM:
        return String("dM")
    if which == CELL2_DXD_14:
        return String("dX_d.S14.transposed")
    if which == CELL2_DC_12:
        return String("dC.S12")
    if which == CELL2_DB_12:
        return String("dB.S12")
    if which == CELL2_DDT_X:
        return String("ddt.x")
    if which == CELL2_DSCALE:
        return String("dscale_c")
    if which == CELL2_DD_INNER:
        return String("dD.inner")
    return String("?")


def mamba2_backward_cell_k(which: Int) -> Int:
    """The CONTRACTED LENGTH of one intra-chunk backward cell."""
    if which == CELL2_DE18:
        return M2_HEADDIM
    if which == CELL2_DH_PREV:
        return M2_CHUNK_SIZE
    if which == CELL2_DBDEC:
        return M2_HEADDIM
    if which == CELL2_DXD_16:
        return M2_D_STATE
    if which == CELL2_DDECAY:
        return M2_D_STATE
    if which == CELL2_DM:
        return M2_HEADDIM
    if which == CELL2_DXD_14:
        return M2_CHUNK_SIZE
    if which == CELL2_DC_12:
        return M2_CHUNK_SIZE
    if which == CELL2_DB_12:
        return M2_CHUNK_SIZE
    if which == CELL2_DDT_X:
        return M2_HEADDIM
    if which == CELL2_DSCALE:
        return M2_HEADDIM * M2_D_STATE
    if which == CELL2_DD_INNER:
        return M2_HEADDIM
    return 0


def mamba2_backward_cell_leaves(which: Int) -> Tuple[Int, Int]:
    """`(leaf_size, leaf_count)` for one cell, from `contract_leaf_size`. 128 -> one leaf, one serial ascending chain k = 256 -> two leaves of 128, ONE fold level k = 8192 -> 64 leaves of 128, SIX fold levels so `CELL2_DSCALE` is the first and only place either the Mamba-2 forward or its backward exercises a balanced tree deeper than one level."""
    var k = mamba2_backward_cell_k(which)
    var el = contract_leaf_size(k)
    return (el, leaf_count(k, el))




def mamba2_backward_proj_a_workspace_max_floats(
    which: Int, dims: Mamba2Dims, m: Int
) -> Int:
    var fwd = mamba2_proj_forward_call(which, dims, m)
    comptime if SAB_BWD2_WS_FROM_FORWARD:
        return identical_gemm_workspace_max_floats(fwd[1], fwd[2], fwd[3])
    return identical_gemm_backward_a_workspace_max_floats(
        fwd[0], fwd[1], fwd[2], fwd[3]
    )


def mamba2_backward_proj_b_workspace_max_floats(
    which: Int, dims: Mamba2Dims, m: Int
) -> Int:
    """The `dB` sizing, and the one most likely to surprise, because `k'` is the token count and a plan chosen at one batch size is not the plan chosen at another."""
    var fwd = mamba2_proj_forward_call(which, dims, m)
    comptime if SAB_BWD2_WS_FROM_FORWARD:
        return identical_gemm_workspace_max_floats(fwd[1], fwd[2], fwd[3])
    return identical_gemm_backward_b_workspace_max_floats(
        fwd[0], fwd[1], fwd[2], fwd[3]
    )


def mamba2_backward_reduce_workspace_max_floats(
    which: Int, dims: Mamba2Dims, m: Int
) -> Int:
    return identical_gemm_backward_bias_workspace_max_floats(
        m, mamba2_reduction_width(which, dims)
    )


def mamba2_backward_workspace_max_floats(dims: Mamba2Dims, m: Int) -> Int:
    """ONE number a backward pass can allocate once and reuse for all fourteen routed calls at this shape."""
    var w = 1
    for i in range(PROJ2_COUNT):
        var wa = mamba2_backward_proj_a_workspace_max_floats(i, dims, m)
        if wa > w:
            w = wa
        var wb = mamba2_backward_proj_b_workspace_max_floats(i, dims, m)
        if wb > w:
            w = wb
    for i in range(RED2_COUNT):
        var wr = mamba2_backward_reduce_workspace_max_floats(i, dims, m)
        if wr > w:
            w = wr
    return w


def mamba2_backward_state_floats(b: Int, l: Int, dims: Mamba2Dims) -> Int:
    """`B * C * H * P * N`, the incoming-per-chunk state the backward reads. There is nothing to refuse and nothing to price: this is `M * di * N / Q` entries per batch rather than `M * di * N`, a factor of `Q = 256` fewer."""
    var c = (l + M2_CHUNK_SIZE - 1) // M2_CHUNK_SIZE
    if l <= 0:
        c = 0
    return b * c * dims.nheads * M2_HEADDIM * M2_D_STATE



comptime B2T1_DIRECTION_IS_DESCENDING_IN_C = True
comptime B2T1_SEED_IS_POSITIVE_ZERO = True
comptime B2T1_IS_FUSED = True
comptime B2T1_SCALE_INDEX_OFFSET = 1
comptime B2T1_STEPS_ARE_CHUNKS_NOT_TOKENS = True

comptime B2T3_DIRECTION_IS_ASCENDING = True
comptime B2T3_SEED_IS_POSITIVE_ZERO = True
comptime B2T3_IS_FUSED = True
comptime B2T3_SITES = 3
comptime B2T3_H_FOLD_IS_ONE_LEAF_BELOW_DMODEL = 4096
comptime B2T3_H_SITES_ARE_VACUOUS_AT_DMODEL_32 = True
comptime B2T3_DI_TREE_NEEDS_DMODEL_AT_LEAST = 128

comptime M2_1_FOLDS_HEADDIM_FIRST = True
comptime M2_1_BOTH_LEGS_ARE_ROUTED = True

comptime M2_2_IS_ROUTED_TO_GEMM = True
comptime M2_2_LINEARIZATION_IS_P_MAJOR = True

comptime M2_3_FOLD_IS_FULL_Q_WITH_STRUCTURAL_ZEROS = True
comptime M2_3_TRANSPOSE_ARM_NEEDS_ASYMMETRIC_M = True

comptime M2_4_IS_PREFIX_THEN_SUFFIX = True
comptime M2_4_INNER_J_IS_ASCENDING = True
comptime M2_4_OUTER_I_IS_ASCENDING = True
comptime M2_4_SEEDS_ARE_POSITIVE_ZERO = True
comptime M2_4_ADDS_ARE_UNFUSED = True
comptime M2_4_DIRECT_RECTANGLE_REFUSED = True
comptime M2_4_SUMMED_AREA_DIFFERENCE_REFUSED = True
comptime M2_4_UPSTREAM_HAS_TWO_SPELLINGS = True

comptime M2_5_DIRECTION_IS_DESCENDING_IN_I = True
comptime M2_5_SEED_IS_POSITIVE_ZERO = True
comptime M2_5_ADD_IS_UNFUSED = True
comptime M2_5_RESETS_AT_EVERY_CHUNK_BOUNDARY = True
comptime M2_NO_DESCENDING_CONTRACTION_ANYWHERE = True

comptime M2_6_MASK_USES_TOTAL_ORDER_KEYS = True
comptime M2_6_BOUNDARY_IS_INCLUSIVE = True
comptime M2_6_PREDICATE_READS_PRE_CLAMP = True
comptime M2_6_NEGATIVE_ZERO_IS_CLAMPED_BY_TOTAL_ORDER = True

comptime B2_JOIN_RULE_IS_ASCENDING_SEAM_ORDER = True
comptime B2_JOIN_SITES = 5
comptime B2_JOIN_ADDS_ARE_UNFUSED = True
comptime B2_JOIN_ARMS_PREDICTED_INERT_IF_ONE_LEG_DOMINATES = True

comptime B2T7_RSTD3_IS_LEFT_ASSOCIATED = True
comptime B2T7_DIVIDES_BY_ROW_WIDTH_ONCE_PER_ROW = True
comptime B2T7_FINAL_SUBTRACT_IS_UNFUSED = True
comptime B2T7_SITES = 2
comptime B2T7_RECONCILIATION_WITH_TRANSFORMER_LANE_IS_OWED = True

comptime B2T8_DRES_IS_THE_LEFT_OPERAND = True

comptime B2_T2_HAS_NO_SITE = True
comptime B2_T4_HAS_NO_SITE = True
comptime B2_T5_HAS_NO_SITE = True
comptime B2_MAMBA1_T5_SITE_LIST_IS_QUESTIONED = True

comptime B2_SOFTPLUS_D_USES_SIGMOID_MULTIPLY = True
comptime B2_SOFTPLUS_D_GUARD_THRESHOLD_IS_20 = True
comptime B2_EXP_DERIVATIVE_REUSES_THE_FORWARD_VALUE = True
comptime B2_RECOMPUTE_USES_THE_FORWARD_SPELLING = True

comptime B2_UPSTREAM_SM_COUNT_SPLIT_REFUSED = True
comptime B2_UPSTREAM_ATOMICS_REFUSED = True
comptime B2_UPSTREAM_DETERMINISM_IS_AN_ENV_VAR = True

comptime B2_CONV_BACKWARD_IS_INFERRED_NOT_READ = True



comptime TOPO2_T1_CHUNK = 0
comptime TOPO2_T3_SHARED = 1
comptime TOPO2_M2_1_DD = 2
comptime TOPO2_M2_2_DSCALE = 3
comptime TOPO2_M2_3_TRANSPOSED = 4
comptime TOPO2_M2_4_SEGSUM = 5
comptime TOPO2_M2_5_CUMSUM = 6
comptime TOPO2_M2_6_CLAMP = 7
comptime TOPO2_JOIN = 8
comptime TOPO2_T7_NORM = 9
comptime TOPO2_T8_ABSORB = 10

comptime TOPO2_COUNT = 11


def mamba2_backward_topology_name(which: Int) -> String:
    if which == TOPO2_T1_CHUNK:
        return String("T1@chunks 1342 reverse chunk recurrence")
    if which == TOPO2_T3_SHARED:
        return String("T3 1343 shared-channel fold, di and H")
    if which == TOPO2_M2_1_DD:
        return String("M2-1 1344 dD, headdim then tokens")
    if which == TOPO2_M2_2_DSCALE:
        return String("M2-2 1345 dscale_c at k = P*N")
    if which == TOPO2_M2_3_TRANSPOSED:
        return String("M2-3 1346 transposed mask at full Q")
    if which == TOPO2_M2_4_SEGSUM:
        return String("M2-4 1347 segsum rectangle, prefix then suffix")
    if which == TOPO2_M2_5_CUMSUM:
        return String("M2-5 1348 in-chunk suffix cumsum")
    if which == TOPO2_M2_6_CLAMP:
        return String("M2-6 1349 clamp derivative, total-order mask")
    if which == TOPO2_JOIN:
        return String("JOIN 1350 ascending seam order, 5 sites")
    if which == TOPO2_T7_NORM:
        return String("T7 1351 RMSNorm backward, 2 sites")
    if which == TOPO2_T8_ABSORB:
        return String("T8 1352 residual absorption")
    return String("?")


def mamba2_backward_topology_site(which: Int) -> String:
    """Where the kernel for one declaration WILL live, and that it does not."""
    if which == TOPO2_M2_6_CLAMP or which == TOPO2_JOIN:
        return (
            mamba2_backward_topology_name(which)
            + " -- NOT WRITTEN; mamba/impl/mamba_ssm/modules/"
            + "mamba2_backward.mojo (does not exist)"
        )
    if which == TOPO2_T7_NORM or which == TOPO2_T8_ABSORB:
        return (
            mamba2_backward_topology_name(which)
            + " -- NOT WRITTEN; mamba/impl/mamba_ssm/modules/"
            + "mamba2_backward.mojo (does not exist)"
        )
    return (
        mamba2_backward_topology_name(which)
        + " -- NOT WRITTEN; mamba/impl/mamba_ssm/modules/"
        + "ssd_minimal_backward.mojo (does not exist)"
    )


def mamba2_backward_declarations() -> String:
    """Every declared choice, as one printable block a gate can diff."""
    var s = String("mojolearn.identical.mamba2.bwd.fp32.v1 DECLARATIONS\n")
    s = s + "1342 T1@chunks: c DESCENDING, seed +0.0, FUSED, scale offset "
    s = s + String(B2T1_SCALE_INDEX_OFFSET)
    s = s + "; UNWITNESSABLE below L = 513\n"
    s = s + "1343 T3 shared-channel: ASCENDING, seed +0.0, FUSED, "
    s = s + String(B2T3_SITES) + " sites at lengths d_inner and H;"
    s = s + " H fold is ONE LEAF below d_model "
    s = s + String(B2T3_H_FOLD_IS_ONE_LEAF_BELOW_DMODEL)
    s = s + "; d_inner tree needs d_model >= "
    s = s + String(B2T3_DI_TREE_NEEDS_DMODEL_AT_LEAST) + "\n"
    s = s + "1344 dD: headdim FIRST then tokens, BOTH LEGS ROUTED\n"
    s = s + "1345 dscale_c: ROUTED at k = "
    s = s + String(M2_HEADDIM * M2_D_STATE)
    s = s + ", linearization p-MAJOR\n"
    s = s + "1346 dX_d transposed: FULL Q with structural zeros, never a"
    s = s + " shortened chain\n"
    s = s + "1347 segsum bwd: a RECTANGLE; prefix j ASCENDING then suffix"
    s = s + " i ASCENDING, seeds +0.0, adds UNFUSED\n"
    s = s + "1348 in-chunk cumsum bwd: i DESCENDING, seed +0.0, UNFUSED,"
    s = s + " resets at every chunk boundary\n"
    s = s + "1349 clamp': TOTAL-ORDER key mask on the PRE-clamp value,"
    s = s + " boundary INCLUSIVE\n"
    s = s + "1350 join rule: ASCENDING SEAM ORDER, left-associated, "
    s = s + String(B2_JOIN_SITES) + " sites, all predicted INERT if one leg"
    s = s + " dominates\n"
    s = s + "1351 T7 RMSNorm bwd: rstd^3 left-associated, one div per row,"
    s = s + " final subtract UNFUSED, " + String(B2T7_SITES) + " sites;"
    s = s + " transformer-lane reconciliation OWED\n"
    s = s + "1352 T8 dx join: dres LEFT, one flushed add (ABSORPTION SITE)\n"
    s = s + "1353 deletions: T2, T4 and T5 have NO SITE in this profile\n"
    s = s + "1078 softplus': MULTIPLY by identical_sigmoid, guard <= 20\n"
    s = s + "exp': the forward's rounded value, 4 sites; recompute uses the"
    s = s + " forward spelling\n"
    s = s + "NO DESCENDING CONTRACTION ANYWHERE; two descending walks only\n"
    s = s + "sabotage: " + mamba2_backward_sabotage_name() + "\n"
    return s
