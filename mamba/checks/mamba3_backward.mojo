# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The backward pass of one Mamba-3 (SISO) block, ROUTING AND DECLARATIONS ONLY.

**THIS FILE CONTAINS NO FLOATING-POINT ARITHMETIC.** Not one float multiply,
not one float add, not one `ftz`, not one `identical_mul_add`, not one
`pinned_mul`, not one `portable_cosf`, and no kernel of its own. It does
contain INTEGER shape and PAIR-INDEX arithmetic, because a routing layer is
made of index and dimension bookkeeping and pretending otherwise would be a
slogan rather than a claim. The falsifiable form is that **no expression below
has a floating-point type**. Every buffer is a
`DeviceBuffer[DType.float32]` and nothing is ever loaded out of one, so the
token `Float32` occurs in this file only inside prose. Grepping this file for
`Float32` and finding only prose is the whole test. It does two things and
neither touches a float.

1.  It ROUTES the parts of the Mamba-3 backward that are already somebody
    else's certified arithmetic. **Eleven calls**: four are the gradients of
    the block's two `nn.Linear` projections, which are
    `gemm/checks/gemm_backward.mojo`'s six-row table entered at this block's
    shapes, and seven are token-axis parameter reductions spelled as
    `mojolearn.identical.gemm.fp32.v1` `OP_NN` products against a vector of
    ones, DEVIATION 851's trick reused unchanged. Mamba-1 routes fourteen and
    Mamba-2 routes fourteen; Mamba-3 routes eleven because it has no conv
    (`IDENTICAL_MAMBA3_CONTRACT.md` section 0), and the five conv reductions
    go with it.
2.  It DECLARES, and does not implement, every fold, recurrence and rotation
    topology the Mamba-3 backward needs that nothing in this repository owns.
    A declaration here is a name, a direction, a seed, a fusion decision, an
    index set and the statement of what would falsify it. **There is
    deliberately no kernel**, because a kernel written before its gate is a
    belief with a compile step.

THIS PROFILE INHERITS AND DOES NOT RE-ANSWER
---------------------------------------------
`mamba/IDENTICAL_MAMBA3_CONTRACT.md` opens by making Mamba-2 its baseline:
"wherever a numerical question recurs from Mamba-2 (or Mamba-1 through it),
this profile INHERITS the answer unchanged and does not re-answer it".
**The same rule governs the backward and this file obeys it literally.**
`mamba/checks/mamba2_backward.mojo`'s DEVIATIONS 1342-1353 are inherited as a
block by DEVIATION 1364 below, and the fifteen numbers this file spends buy
only what Mamba-3 ADDS. The two Mamba backward lanes may not answer the same
question twice any more than the two forward lanes may.

THE PROFILE IS NOT AMENDED AND MAY NOT BE
------------------------------------------
`mamba/IDENTICAL_MAMBA3_CONTRACT.md` is CONSUMED here and never edited.
Nothing in this file changes a seam decision in its section 4, a constant in
its section 3 (`CHUNK_SIZE = 64`, `rope_fraction = 0.5`, `A_floor` included)
or the stage list in its section 7, so nothing here is a
`mojolearn.identical.mamba3.siso.fp32.v1` v2. What that contract's section 10
says about itself, "no training/backward", stays true of v1. The backward is a
SEPARATE profile, `mojolearn.identical.mamba3.siso.bwd.fp32.v1`, which
consumes the forward one and adds clauses of its own, and it does not exist
yet.

THE PROPORTION, RE-COUNTED RATHER THAN INHERITED
-------------------------------------------------
`mamba/BACKWARD_SCOPE_M2_M3.md` section 4 counts 98 operations, split
8 copies / 5 (a) / 22 (b) / 63 (c), with 30 of the (c) rows called c-TOPOLOGY
and eight of those called NEW topologies. That document says of itself that
not one number in it was measured. Reading the contracts against it moves ten
rows and deletes one. This file's own count, under the scope document's
counting rule so the two are comparable:

    copies                                    8
    (a)  existing primitive, no new order      5
    (b)  routable to gemm v1                  25
    (c)  genuinely new arithmetic             59
         of which  c-elementwise              37
                   c-inherited order           3
                   c-TOPOLOGY                 19
                                        -------
                                              97

The moves: `dscale_c` and the two transposed strict-mask contractions are
routed cells (DEVIATION 1364 inherits Mamba-2's argument for all three); the
clamp derivative, the heavy-tail derivative, the two rotation transposes, the
two `dtheta` terms and the trapezoid's shifted leg are ELEMENTWISE, not folds
(they contract nothing); and the parameter fold over the batch has no site
(DEVIATION 1374).

**The finding that survives the recount, and it is the one to carry.**
**Mamba-3 introduces exactly ONE new fold topology over Mamba-2: DEVIATION
1367's whole-sequence reverse angle chain.** Everything else Mamba-3 adds
over Mamba-2 is per-token elementwise arithmetic with a data-dependent
parameter, which has no gemm shape and no fold shape either. The scope
document's "eight new topologies" counts eight new DECISIONS, which is right,
and eight new fold structures, which is not. The routed fraction is 26
percent, still the lowest of the three models (Mamba-1 31, Mamba-2 34), and
the reason is unchanged: elementwise work does not route.

WHAT THIS FILE DOES NOT BUY
----------------------------
Not identical training, not an identical model, not identical gradients for
one block. Fifty nine operations have no kernel. And two structural things
this profile can never buy, both stated here rather than discovered later:
**there is no decode backward for Mamba-3** (DEVIATION 1367), and the
`Input_States` continuation is no more bit-equal backward than it is forward
(`IDENTICAL_MAMBA3_CONTRACT.md` section 5, claim 2).

**Nothing in this file has been compiled or run.**

DEVIATIONS 1360-1374 SPENT, 1375-1379 RESERVED. The band was verified free by
a repository-wide grep before it was opened; the pattern that reproduces the
check is

    grep -rhoE "DEVIATIONS? 13[4-9][0-9]" . | sort -u

which must print nothing but this file's band and Mamba-2's.

`[[mojo-buffer-freed-at-last-use]]`: every launcher here is an `_into` form.
It enqueues and returns. The CALLER owns every buffer, including the ones
vector and every pre-product, and must keep each alive past its own
`ctx.synchronize()`.
"""

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


# ===========================================================================
# THE SABOTAGE SWITCHES
# ===========================================================================
# OFF in every build that does not name them, exactly as the forward file's
# ten are. None of them touches arithmetic, because this file has none. They
# move WHICH certified product is computed, WHICH INDEX PAIR is rotated and
# WHERE the result is written, which is the only thing here that can be wrong.
#
# Build with, e.g.:
#     tools/with_identical_mode.sh pixi run mojo build \
#         -D MOJOLEARN_MAMBA3_SABOTAGE_BWD_ROTATE_HALF_SPLIT=1 \
#         -I . mamba/checks/mamba3_backward_compile_probe.mojo \
#         -o /tmp/mojolearn_m3b_probe

#: The `dA` and `dB` routings for the two projections are SWAPPED.
comptime SAB_BWD3_PROJ_SWAP = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_BWD_PROJ_SWAP"
]()
#: The workspace is sized from the FORWARD shape rather than from the backward
#: call's shape. Right at small shapes because allocations have slack, whole
#: `+0.0` regions at large ones.
comptime SAB_BWD3_WS_FROM_FORWARD = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_BWD_WS_FROM_FORWARD"
]()
#: The backward rotation pairs HALVES `(i, i + N/2)` instead of the SISO
#: interleaved `(2i, 2i+1)`. This is the backward twin of the forward's
#: `ROTATE_HALF_SPLIT` arm (`IDENTICAL_MAMBA3_CONTRACT.md` section 8f) and of
#: the trap the transformer lane's RoPE backward walks into, which rotates
#: halves because that is HuggingFace's checkpoint layout
#: (`transformer/checks/transformer_backward.mojo`:1023-1026). **The pairing
#: is INDEX ARITHMETIC, so this file owns it and can be sabotaged here**; the
#: rotation's floats are not this file's.
comptime SAB_BWD3_ROTATE_HALF_SPLIT = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_BWD_ROTATE_HALF_SPLIT"
]()
#: `dq_bias` and `dk_bias` are reduced at width `H` instead of `H * N`. The
#: biases are `[H, N]` (`mamba3.py`:121-122, `Mamba3Weights.b_bias`), token
#: major and contiguous, so the correct reduction is one ones-vector call at
#: `W = H * N`. Reducing at `W = H` produces a vector of the wrong length,
#: which a shape check catches, and at `N == 1` it would not; `N` is 128 here
#: so this arm is REACHABLE and not inert.
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
    """Which sabotage this binary compiled with, for a check's banner.

    A check must print THIS, plus `gemm_backward_sabotage_name()`, plus the
    forward block's and the SISO core's own names, because a backward binary
    can carry a forward sabotage and a gemm sabotage as well.
    """
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
    """The arms a gate must NOT count as covered by this lane, with the
    reason. Each entry is a PREDICTION until a gate turns it into a
    measurement, and three of them are VACUOUS BY CONSTRUCTION rather than
    merely unlikely to fire.
    """
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


# ===========================================================================
# THE TWO PROJECTIONS
# ===========================================================================
# Contract seam S4 and section 2. `dt`, `A`, `trap` and the angle rates are
# COLUMNS of `in_proj.out` (the 8-way split, mamba3.py:106-107), not separate
# `nn.Linear` calls. There is no conv and no `x_proj`.

#: `norm.out [M, d_model] . W_in [d_in_proj, d_model]^T` -> `[M, d_in_proj]`,
#: columns `z | x | B | C | dd_dt | dd_A | trap | angle`.
comptime PROJ3_IN = 0
#: `gate.out [M, d_inner] . W_out [d_model, d_inner]^T` -> `[M, d_model]`.
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
    """`(op, m, n, k)` of the FORWARD projection, from the block's dimensions.

    THE ONE DOOR. **THE MAMBA-3 FORWARD PROJECTION SHAPE HAS EXACTLY ONE
    PRODUCER, THIS FUNCTION**, and every routing, launcher and sizing helper
    below calls it, so changing which product a Mamba-3 backward computes
    means editing one pure, host-side, device-free function.

    Both rows are `OP_NT`, contract seam S4. `m` is the token count `M = B*L`.
    `in_proj`'s `n` is `2*d_inner + 2*G*N + 3*nheads + num_rope_angles`, which
    `Mamba3Dims.d_in_proj()` owns; this file does not re-derive it, because
    the column layout is the forward's and a second derivation is a second
    opinion.

    DEVIATION 1360. Contract section 4 seam S4; gemm contract 0.1 and 2.
    """
    var dm = dims.d_model
    var di = dims.d_inner
    if which == PROJ3_IN:
        return (OP_NT, m, dims.d_in_proj(), dm)
    return (OP_NT, m, dm, di)


def mamba3_backward_proj_a_call(
    which: Int, dims: Mamba3Dims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the ACTIVATION gradient of `which`.

    A pass-through to `gemm_backward_a_call` so the routing table has ONE
    definition in this repository. Always `OP_NN(dC, W) at (m, k, n)` with
    `dC` LEFT, because both forward rows are `OP_NT`.

    DEVIATION 1360.
    """
    var fwd = mamba3_proj_forward_call(which, dims, m)
    comptime if SAB_BWD3_PROJ_SWAP:
        return gemm_backward_b_call(fwd[0], fwd[1], fwd[2], fwd[3])
    return gemm_backward_a_call(fwd[0], fwd[1], fwd[2], fwd[3])


def mamba3_backward_proj_b_call(
    which: Int, dims: Mamba3Dims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the WEIGHT gradient of `which`.

    Always `OP_TN(dC, A) at (n, k, m)` with `dC` LEFT. **`k'` IS `m`, THE
    TOKEN COUNT, IN BOTH ROWS**, so a microbatched Mamba-3 training step is
    not bit equal to an unsplit one for `dW_in` or `dW_out`, on any vendor.
    Seven more parameter gradients in this block have the same property and
    are not matmuls.

    DEVIATION 1360. Gemm contract 0.1, 2, 3, 6 and 6.1.
    """
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
    """The ACTIVATION gradient of projection `which`, into `d_input`.

    `d_output` is `[M, n]`, `weight` the FORWARD weight unchanged and
    untransposed, `d_input` receives `[M, k]`, `ws` at least
    `mamba3_backward_proj_a_workspace_max_floats(which, dims, m)` floats.
    Everything numerical happens inside `identical_gemm_backward_a_into`.

    ASYNCHRONOUS, caller-owned buffers, `[[mojo-buffer-freed-at-last-use]]`.
    DEVIATION 1360.
    """
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
    """The WEIGHT gradient of projection `which`, into `d_weight`.

    `input_stage` is the FORWARD input to this projection, `norm.out` or
    `gate.out`, unchanged. `dA` and `dB` write DISJOINT buffers and read the
    same `d_output`, so they need no barrier between them and may share one
    `ws` on one context.

    ASYNCHRONOUS, caller-owned buffers. DEVIATION 1360.
    """
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


# ===========================================================================
# THE SEVEN TOKEN-AXIS REDUCTIONS
# ===========================================================================
# Each is `sum over the M token rows of a [M, W] matrix` giving `[W]`, which
# is `identical_gemm_backward_bias_into(ctx, out, src, ones, ws, M, W)` and
# therefore an `OP_NN` v1 GEMM at `(1, W, M)` under DEVIATION 851's argument.
#
# SEVEN, NOT NINE. `BACKWARD_SCOPE_M2_M3.md` section 9 item 2 gives Mamba-3
# nine ones-vector sites; its own operation table (section 4.1) lists seven,
# D11, D54, D55, D60, D65, D84 and D93, and seven is what the parameter list
# supports: `Mamba3Weights` carries nine parameters, of which `w_in` and
# `w_out` are gemm weight gradients and the other seven are these. The
# discrepancy is recorded rather than silently corrected.

#: `dD[h] = sum over tokens of dcoef[t,h]`, `W = H`. `src` is `dcoef`, which
#: is `sum over p of dY * v` and is a CONTRACTION OUTPUT, not a pre-product.
#: **Mamba-3 HAS NO TWO-AXIS `dD` FOLD, where Mamba-2 does**, and one forward
#: seam decision is the whole reason: S18 spells `t = ftz(D[h] + qk_gamma_i)`
#: and multiplies that ONE value by `v`, so a single coefficient gradient
#: serves BOTH `dD` and `dqk_gamma` and the `headdim` fold has to exist for
#: the second consumer anyway. Mamba-2's DEVIATION 1344 has no counterpart
#: here.
comptime RED3_D = 0
#: `d(dt_bias)[h] = sum over tokens of ddd_dt[t,h]`, `W = H`. Seam S6's bias
#: add. `src` is the stage directly; no pre-product.
comptime RED3_DT_BIAS = 1
#: `d(B_norm.weight)[n] = sum over tokens of dB_head[t,n] * inner_B[t,n]`,
#: `W = N`. Seam S21. `src` is the pre-product. `inner_B` is the normalized
#: value BEFORE the weight multiply; recovering it by dividing `bcnorm.B` by
#: the weight is a division and a different answer.
comptime RED3_BNORM_W = 2
#: `d(C_norm.weight)[n]`, the same at `W = N`.
comptime RED3_CNORM_W = 3
#: `d(B_bias)[h,n] = sum over tokens of dk_pre[t,h,n]`, `W = H * N`. The bias
#: is `[H, N]` (`mamba3.py`:121-122, `mimo_rank = 1` squeezed out; the fixture
#: records `M3_TID_B_BIAS ... [H, N]`), token-major and contiguous, so the
#: whole `[H, N]` plane is ONE ones-vector call at `W = H*N`, not `H` calls.
#: **This answers `BACKWARD_SCOPE_M2_M3.md` section 11 item 4**, which could
#: not determine the bias shape and therefore could not name the reduction
#: axis: the shape is `[H, N]` and the axis is TOKENS ONLY.
comptime RED3_B_BIAS = 4
#: `d(C_bias)[h,n]`, the same at `W = H * N`.
comptime RED3_C_BIAS = 5
#: `d(block norm.weight)[j] = sum over tokens of dnrm[t,j] * inner[t,j]`,
#: `W = d_model`. Seams S1-S3. `src` is the pre-product.
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
    """`H * N`, the width of one `[H, N]` bias plane reduced over tokens.

    ONE ones-vector call per bias, not `H` calls, because the bias plane is
    token-major and contiguous. `SAB_BWD3_BIAS_WIDTH_HEADS_ONLY` returns `H`
    instead, which is a vector of the wrong length and therefore an arm a
    shape assertion can catch at `N = 128`.
    """
    comptime if SAB_BWD3_BIAS_WIDTH_HEADS_ONLY:
        return dims.nheads
    return dims.nheads * M3_D_STATE


def mamba3_reduction_width(which: Int, dims: Mamba3Dims) -> Int:
    """`W`, the length of the reduced vector.

    Four distinct widths: `H` for the two per-head parameters, `N` for the two
    B/C norm weights, `H*N` for the two biases and `d_model` for the block
    norm. `SAB_BWD_BIAS_AXIS` separated at one width is not separated at the
    others.
    """
    if which == RED3_NORM_W:
        return dims.d_model
    if which == RED3_BNORM_W or which == RED3_CNORM_W:
        return M3_D_STATE
    if which == RED3_B_BIAS or which == RED3_C_BIAS:
        return mamba3_bias_reduction_width(dims)
    return dims.nheads


def mamba3_reduction_needs_preproduct(which: Int) -> Bool:
    """True where `src` must be a Hadamard product the caller materialized.

    Three of the seven, all three of them RMSNorm weight gradients. `RED3_D`
    takes a contraction output, and `RED3_DT_BIAS`, `RED3_B_BIAS` and
    `RED3_C_BIAS` take a gradient stage directly. A caller that passes a raw
    stage where a pre-product belongs computes a plausible column sum of the
    wrong quantity and no shape check can see it, which is why the gate for
    this is a finite-difference arm and not an assertion here.
    """
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
    """`out[w] = sum over the M rows of src[M, W]`, as a v1 `OP_NN` at
    `(1, W, M)`.

    `ones` holds `M` entries of EXACTLY one
    (`identical_gemm_backward_bias_ones_floats(m)`). A wrong value there is a
    wrong gradient with no symptom.

      - `m == 0` gives `k' == 0`, so every entry of `out` is a stored `+0.0`,
        which gemm contract section 8 requires WRITTEN rather than skipped.
      - `W == 0` gives an empty output and nothing is written.

    **DEVIATION 1361.** `k'` is `M`, so these seven are as microbatch
    sensitive as the two weight matmuls and they are the ones a reader is most
    likely to assume are exempt, because they look like reductions rather than
    matmuls.

    ASYNCHRONOUS, caller-owned buffers, INCLUDING `ones` and `src`.
    """
    identical_gemm_backward_bias_into(
        ctx, out, src, ones, ws, m, mamba3_reduction_width(which, dims)
    )


def mamba3_backward_ones_floats(m: Int) -> Int:
    """How long the shared ones vector is. One per token count, allocated once
    and reused by all seven reductions."""
    return identical_gemm_backward_bias_ones_floats(m)


# ===========================================================================
# THE ROTATION'S PAIR INDEX
# ===========================================================================
# The rotation's ARITHMETIC is DEVIATION 1365's declaration and is not here.
# Its PAIRING is integer index bookkeeping, so it is here, and it is worth
# being here because the pairing is where the transformer lane's already
# written RoPE backward becomes unusable rather than reusable.


def mamba3_rope_pair_count() -> Int:
    """`N / 2 = 64` interleaved pairs per head, of which the first
    `num_rope_angles = 32` are rotated."""
    return M3_D_STATE // 2


def mamba3_rope_pair_is_rotated(pair: Int) -> Bool:
    """`pair < num_rope_angles`. `rope_fraction = 0.5` (contract section 3),
    so pairs 32..63 are UNROTATED and their backward is a STRUCTURAL COPY.

    **A SABOTAGE ARM AGAINST THOSE PAIRS IS INERT BY CONSTRUCTION** and must
    be declared VACUOUS in advance: DEVIATION 828 records that the computed
    and structural spellings coincide bit for bit because `cos(+0.0) = 1.0`
    and `sin(+0.0) = +0.0` exactly, and the backward inherits that coincidence
    unchanged.
    """
    return pair < M3_NUM_ROPE_ANGLES


def mamba3_rope_pair_indices(pair: Int) -> Tuple[Int, Int]:
    """The two component indices of `pair`, INTERLEAVED: `(2p, 2p+1)`.

    Contract seam S13 pins the interleaved SISO pairing; `mamba3.py`:360-363
    records the half-split `(i, i + N/2)` as the MIMO permutation, and
    `ROTATE_HALF_SPLIT` is the forward's arm against it. The backward carries
    the same arm because the transpose of a rotation is a rotation on the SAME
    pair, so a backward that pairs halves is wrong in exactly the way the
    forward would be.

    **THIS IS WHY `transformer/checks/transformer_backward.mojo`:996's
    `bwd_rope_kernel` CANNOT BE CALLED FROM HERE.** It is a real, written,
    transposed-rotation backward with an explicit refusal of the fma, and it
    rotates HALVES (`:1023-1026`) because that is HuggingFace's checkpoint
    layout. Its IDIOM transfers, its PAIRING does not, and it has no angle
    term at all (DEVIATION 1366) because there `cos`/`sin` are a precomputed
    table at absolute positions. It is also not gated: that file records
    "COMPILED AND RUN 2026-09-03, NOT YET GATED" and that its check REFUSED TO
    CERTIFY.

    DEVIATION 1365.
    """
    comptime if SAB_BWD3_ROTATE_HALF_SPLIT:
        return (pair, pair + (M3_D_STATE // 2))
    return (2 * pair, 2 * pair + 1)


# ===========================================================================
# THE INTRA-CHUNK BACKWARD CELL TABLE
# ===========================================================================
# SHAPES ONLY. No launcher, deliberately: gemm v1's entry point is per cell
# and the BATCHED driver over `(b, chunk, head)` is the builders' execution
# plan, which mamba2 DEVIATION 784 (inherited by this profile's S4) forbids
# from touching the arithmetic.

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
    """The CONTRACTED LENGTH of one backward cell.

    Built from `P = 64`, `N = 128`, `Q = 64` and `R = 32`, every one of which
    contract section 3 freezes. **Not one is a function of `B`, `L`, the
    launch or the vendor.**

    The transformer lane's largest finding does NOT bite here and the reason
    is worth stating where the shapes are.
    `transformer/checks/transformer_backward.mojo`:1294-1304, DEVIATION 1402:
    "A DERIVATIVE SWAPS WHICH AXIS IS CONTRACTED", and there the axis it swaps
    into is the KV axis, a SEQUENCE LENGTH, so `contract_partition` builds one
    tree at `S = 257` and a different one at `S = 200`. Here the axis a
    derivative swaps into is `Q = 64`, a profile constant, so
    `contract_leaf_size(Q)` builds the same one leaf at every `L`. **Chunking
    bought that**, and it is the strongest structural argument in favour of
    this backward over Mamba-1's.
    """
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
    """`(leaf_size, leaf_count)` for one cell, from `contract_leaf_size`.

    **THIRTEEN OF THE FOURTEEN CELLS ARE ONE LEAF**, because `Q = 64`,
    `P = 64`, `N = 128` and `R = 32` are all at or below
    `CONTRACT_K_LEAF_MIN = 128`. The exception is `CELL3_DSCALE_C` at
    `k = P*N = 8192`, which is 64 leaves of 128 and six fold levels.

    **SO THE ENTIRE MAMBA-3 BACKWARD HAS EXACTLY ONE MULTI-LEAF CONTRACTION
    OUTSIDE THE TWO WEIGHT GRADIENTS**, and a gate that fires `SAB_NODE_ORDER`
    or `SAB_FOLD_STRIDE` anywhere else in this backward has fired it against a
    single serial chain and measured nothing. Mamba-2 at `Q = 256` has three
    two-leaf cells beside its `dscale`; Mamba-3 at `Q = 64` has none.
    """
    var k = mamba3_backward_cell_k(which)
    var el = contract_leaf_size(k)
    return (el, leaf_count(k, el))


# ===========================================================================
# WORKSPACE SIZING
# ===========================================================================


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
    """The `dB` sizing, and the one most likely to surprise, because `k'` is
    the token count and a plan chosen at one batch size is not the plan chosen
    at another."""
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
    """ONE number a backward pass can allocate once and reuse for all eleven
    routed calls at this shape.

    Safe across the eleven because they are enqueued on ONE `DeviceContext`
    and MAX runs them in order. **Not safe across two contexts or two
    streams**, and a second stream would be a change to this function's
    contract.

    It does NOT include the memory the declarations below need, and it does
    not include the intra-chunk cells' workspaces, since this file provides no
    launcher for them.

    DEVIATION 1360.
    """
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
    """`B * C * H * P * N`, the incoming-per-chunk state the backward reads.

    Mamba-1's T2 is deleted here for Mamba-2's reason: the quantity upstream's
    `h[t] - dbu[t]` recovery fights to obtain is materialized at chunk
    boundaries by construction, so there is nothing to refuse and nothing to
    price. `C = ceil(l / Q)` at `Q = 64`, so the saving over a per-token
    checkpoint is a factor of 64 rather than Mamba-2's 256.
    """
    var c = (l + M3_CHUNK_SIZE - 1) // M3_CHUNK_SIZE
    if l <= 0:
        c = 0
    return b * c * dims.nheads * M3_HEADDIM * M3_D_STATE


def mamba3_backward_theta_chain_floats(
    b: Int, l: Int, dims: Mamba3Dims
) -> Int:
    """`B * H * L * R`, the angle-gradient chain DEVIATION 1367 declares.

    **THIS IS THE ONE TERM IN EITHER SIBLING'S BACKWARD THAT IS LINEAR IN THE
    SEQUENCE LENGTH WITH NO CHUNK DIVISOR**, because Mamba-3's angle state
    crosses chunk boundaries by construction (seam S10, `theta_-1` is the
    incoming angle state) where Mamba-2's cumsum resets at every boundary.
    Every other activation term the backward holds is either per token at a
    head width or per chunk.
    """
    if l <= 0:
        return 0
    return b * dims.nheads * l * M3_NUM_ROPE_ANGLES


# ===========================================================================
# THE DECLARATIONS
# ===========================================================================
# NOT KERNELS. Each block below is the pinned decision for one topology or one
# order, in the forward contract's own vocabulary, stated so that whoever
# writes the kernel is transcribing rather than deciding, and so that a gate
# can print `mamba3_backward_declarations()` and diff it across vendors and
# across builds.
#
# A declaration is not a measurement. Every one is a PREDICTION and none has
# run.

# ---------------------------------------------------------------------------
# INHERITED, at a new axis: the reverse recurrence. DEVIATION 1362.
# ---------------------------------------------------------------------------
#: Forward seam S20 is `h = ftz(fma(exp(da_cs_last), h, increment))`, serial
#: over chunks, cited by the contract as "mamba2 S17's fused answer to the
#: identical decay-state-plus-increment question". Its backward is the same
#: first-order linear recurrence run backwards: DESCENDING in `c` from
#: `C - 1`, seeded `+0.0` at `c = C`, ONE rounding, FUSED, scale index `c + 1`.
comptime B3T1_DIRECTION_IS_DESCENDING_IN_C = True
comptime B3T1_SEED_IS_POSITIVE_ZERO = True
comptime B3T1_IS_FUSED = True
comptime B3T1_SCALE_INDEX_OFFSET = 1
#: `C = ceil(L / 64)`, so the recurrence runs at `L > 64` and takes two or
#: more steps at `L > 128`. **The off-by-one is bitwise inert at `C == 1` and
#: whenever every `scale_c` is equal across chunks.** The forward's shape set
#: (contract section 8g, `L` in {1, 4, 63, 64, 65, 129, 257}) contains
#: `L = 129` and `L = 257`, which do witness it — a strictly better position
#: than Mamba-2's, whose `Q = 256` puts the same declaration out of reach
#: below `L = 513`.
comptime B3T1_WITNESSED_FROM_L = 129

# ---------------------------------------------------------------------------
# INHERITED, and WORSE IN KIND: the head fold. DEVIATION 1363.
# ---------------------------------------------------------------------------
#: T3, the shared-channel contraction, TWO sites: `dC_head[t,n] = sum_h
#: dq_pre[t,h,n]` and `dB_head[t,n] = sum_h dk_pre[t,h,n]`, both at length
#: `H`. Serial ASCENDING in `h` from `+0.0`. **UNFUSED, because there is no
#: multiply**: the forward's head broadcast of `B` and `C` is a pure copy, so
#: the backward is a pure sum and a builder must not manufacture an `fma`
#: against a literal one.
comptime B3T3_DIRECTION_IS_ASCENDING_IN_H = True
comptime B3T3_SEED_IS_POSITIVE_ZERO = True
comptime B3T3_ADD_IS_UNFUSED = True
comptime B3T3_SITES = 2
#: **THE FOLD CANNOT BE DONE EARLY AND THE CONTRACT ALREADY SAYS WHY.**
#: Section 3's `ngroups` row does not merely restate Mamba-2's: it adds
#: "B/C shared across heads by COPY, but rotation is PER HEAD (dt is
#: per-head), so post-rotation K/Q DIFFER per head — the broadcast is a copy,
#: what follows it is arithmetic". Each head applies its own rotation at its
#: own `theta[t,h,r]`, so `dq_pre[t,h,n]` and `dk_pre[t,h,n]` differ per head
#: and the sum over heads can only happen AFTER every head's rotation has been
#: transposed. The fold therefore sits BETWEEN DEVIATION 1365's rotation
#: backward and seam S21's B/C RMSNorm backward, and each of its terms is the
#: output of a three-rounding rotation. Mamba-1's version of this seam sums
#: values that were read directly; this one sums values that were rounded
#: three times each.
comptime B3T3_SITS_AFTER_THE_ROTATION_TRANSPOSE = True
#: Same partition finding as Mamba-2's DEVIATION 1343: `contract_leaf_size(k)`
#: is `k` for every `k <= 128` and `H = d_model / 32`, so the head fold is ONE
#: LEAF at every `d_model <= 4096`, and at the gate shape `d_model = 32` it is
#: a SINGLE TERM, which makes every direction and partition arm against it
#: INERT BY CONSTRUCTION. A gate must assert its arm's predicted term count.
comptime B3T3_H_FOLD_IS_ONE_LEAF_BELOW_DMODEL = 4096
comptime B3T3_IS_VACUOUS_AT_DMODEL_32 = True

# ---------------------------------------------------------------------------
# INHERITED AS A BLOCK: Mamba-2's backward topologies. DEVIATION 1364.
# ---------------------------------------------------------------------------
#: `mamba/checks/mamba2_backward.mojo`'s DEVIATIONS 1345 through 1349 apply
#: here UNCHANGED and are not re-derived, exactly as the forward contract
#: inherits mamba2's answers rather than re-answering them. Named so a reader
#: can see which questions this profile did NOT reopen:
#:
#:   1345  `dscale_c` ROUTED as a v1 cell at `k = P*N`, linearization pinned
#:         `p`-MAJOR. Site: `CELL3_DSCALE_C`.
#:   1346  the transposed masked contraction runs the FULL chunk length with
#:         structural zeros, never a shortened chain whose `k` depends on the
#:         output index. TWO sites here (`CELL3_DV_16`, `CELL3_DK_SCALED_16`)
#:         and the mask is STRICT (`i > j`), not `>=`, because DEVIATION 830
#:         moves the diagonal out of the attention matrix entirely.
#:   1347  the segsum backward is a RECTANGLE, spelled prefix-`j`-ascending
#:         then suffix-`i`-ascending, with the direct fold and the summed-area
#:         difference both refused.
#:   1348  the in-chunk cumsum backward is a suffix sum, `i` DESCENDING, seed
#:         `+0.0`, UNFUSED, resetting at every chunk boundary.
#:   1349  the clamp derivative is a TOTAL-ORDER-key mask on the PRE-clamp
#:         value with an INCLUSIVE boundary.
comptime B3_INHERITS_MAMBA2_BACKWARD_1345_TO_1349 = True
#: **ONE INHERITANCE CHANGES CHARACTER AND IT IS THE CLAMP.** In Mamba-2 the
#: clamp is `dt_limit`, whose default `(0.0, +inf)` cannot move a bit. In
#: Mamba-3 it is seam S5's `clamp(max = -A_floor)` and the bound BINDS
#: structurally: `A_floor = 1e-4` and the bound is active whenever
#: `dd_A < -9999`. So the mask is REACHABLE here and VACUOUS unless the
#: fixture plants that region, exactly as the forward's `A_FLOOR_UNCLAMPED`
#: arm requires a planted `dd_A = -20000`.
comptime B3_CLAMP_BOUND_BINDS_STRUCTURALLY = True
#: A documentation defect found on the way, recorded here because this file
#: may not edit that one: `IDENTICAL_MAMBA3_CONTRACT.md`:150 says
#: `identical_clamp` "EXISTS ... this profile is the SECOND consumer", while
#: the same contract's section 2a stale-sentence note and the mamba2 STILL
#: OWED block still list the `check-portable-nn` arm for it as owed. The
#: honest state is EXISTS, ARM OWED.
comptime B3_CLAMP_CHECK_ARM_IS_STILL_OWED = True

# ---------------------------------------------------------------------------
# NEW: the transposed rotation. DEVIATION 1365. TWO tensors, eight sites.
# ---------------------------------------------------------------------------
#: The rotation matrix is orthogonal, so the exact derivative with respect to
#: the input is its TRANSPOSE, which is the rotation by the negated angle:
#:
#:     dx0 = ftz(ftz(pinned_mul(d0, c)) + ftz(pinned_mul(d1, s)))
#:     dx1 = ftz(ftz(pinned_mul(d1, c)) - ftz(pinned_mul(d0, s)))
#:
#: **THE INNER `ftz` IS THE THING THAT MAKES IT UNFUSED**, and this is not a
#: style note, it is the repeat of a REAL BUG the forward paid for on
#: 2026-09-03. Seam S13 combined two products as `a*b - c*d` over
#: `pinned_mul`, and `pinned_mul(a,b)` is `identical_mul_add(a, b, -0.0)`,
#: mathematically `a*b`, so a backend may simplify it and then re-contract the
#: difference into ONE fma where the row asks for three roundings. That row
#: had said UNFUSED for the whole of its life and nothing enforced it. It cost
#: four stages of the AMD card (`rot.k`, `kscale.out`, `ssd.h_last`,
#: `ssd.k_last`), and **AMD was the column that was RIGHT**: its device
#: produced the unfused answer while Apple and NVIDIA contracted and every
#: host oracle contracted with them, so the majority agreed with itself and
#: the honest column was logged as the divergence. `ftz`'s
#: bitcast-compare-select gives the `fmul` a second use and hands the
#: add-or-subtract a `select`, which no backend can fuse.
comptime M3_1_INNER_FTZ_IS_THE_BARRIER = True
comptime M3_1_IS_UNFUSED = True
#: **TWO POINTS THAT ARE NOT A RESTATEMENT OF THE FORWARD'S FINDING.**
#: First, the PLUS form is as exposed as the minus form and is harder to
#: catch: the forward bug surfaced only because three vendors split two to
#: one, and nothing guarantees a three-column split next time. **The barrier
#: is spelled because the contract asks for it, not because a column
#: disagreed.** Second, the site count is LARGER than the forward's: `q` and
#: `k` both rotate and the backward runs each, so every rotated pair carries
#: four sum-or-difference-of-two-products expressions where the forward
#: carries two.
comptime M3_1_TENSORS = 2
comptime M3_1_EXPRESSIONS_PER_PAIR = 4
#: `cos` and `sin` are RECOMPUTED with `portable_cosf` and `portable_sinf`,
#: never with anything algebraically equal. `theta` is a carded stage
#: (`angle.theta`), so the recompute is exact by construction, in the shape of
#: the transformer lane's DEVIATION 1420. Upstream recomputes with the PTX
#: `cos.approx`/`sin.approx` (`mamba3_siso_bwd.py`:980-981, :1128-1129), which
#: DEVIATION 828 already refuses on the forward side and which is refused
#: again here for the identical reason.
comptime M3_1_TRIG_IS_THE_PORTABLE_PAIR = True

# ---------------------------------------------------------------------------
# NEW: the angle half of the rotation. DEVIATION 1366.
# ---------------------------------------------------------------------------
#: `theta` in Mamba-3 is DATA DEPENDENT: it comes from `in_proj` through
#: `angle_raw`, `identical_tanh`, `pi` and `dt`, accumulated serially. So the
#: rotation's derivative has a SECOND term that the transformer lane's RoPE
#: backward does not have at all, because there `cos`/`sin` are a precomputed
#: table at absolute positions. It is easy to write a rotation backward that
#: omits this term entirely and it will look correct.
#:
#:     dtheta = d0 * (-x0*s - x1*c) + d1 * (x0*c - x1*s)
#:
#: **TWO SPELLINGS, AND CHOOSING BETWEEN THEM IS THE FIRST DECISION A MAMBA-3
#: BACKWARD MAKES.** Upstream's (`mamba3_siso_bwd.py`:1044-1046) is six
#: products and three sum-or-difference combinations, all contractable, and it
#: recomputes the rotation from `x0`, `x1`, `c`, `s`. The COLLAPSED one
#: observes that `(x0*c - x1*s)` IS the forward's `ro0` and `-(x0*s + x1*c)`
#: IS `-ro1`, so
#:
#:     dtheta = ftz(ftz(pinned_mul(d1, ro0)) - ftz(pinned_mul(d0, ro1)))
#:
#: two products, one difference, ONE contraction site.
#: **PINNED: THE COLLAPSED SPELLING.** It is available because the forward
#: CARDS the rotated values pre-scale (`rot.q [M,H,N]` and
#: `rot.k [M,H,N] (PRE-scale)`, contract section 7) and seam S15 confirms the
#: carried k state is the pre-scale value. Reading a carded forward value is
#: strictly stronger than recomputing it with the same function, which is what
#: `IDENTICAL_BACKWARD_PLAN.md` section 6's recompute rule asks for; and it
#: has one third the contraction hazard at a seam whose contraction hazard has
#: already cost this lane four card stages once. The house transliteration
#: rule points at upstream's, which is why this is a DEVIATION and not a
#: preference. Upstream's is the required-RED sabotage arm.
comptime M3_2_USES_THE_COLLAPSED_SPELLING = True
comptime M3_2_INNER_FTZ_IS_THE_BARRIER = True
#: **A PRECONDITION, AND IT IS NOT SATISFIED BY THE CARD.** The collapsed
#: spelling needs `rot.q` and `rot.k` PRESENT AS DEVICE BUFFERS, not merely
#: carded. `BACKWARD_SCOPE_M2_M3.md` section 11 item 5 raises the general
#: form of this question (both contracts allow chunk-shaped stages to be
#: ELIDED from the card under a size cap and no document says whether the
#: implementations keep the buffers), and this declaration converts it from an
#: open question into a REQUIREMENT a Mamba-3 forward must meet before this
#: backward can be written.
comptime M3_2_REQUIRES_ROT_Q_AND_ROT_K_AS_BUFFERS = True

# ---------------------------------------------------------------------------
# THE ONE NEW FOLD TOPOLOGY: the whole-sequence reverse angle chain.
# DEVIATION 1367.
# ---------------------------------------------------------------------------
#: Seam S10 is `theta_t = mod2pi(ftz(theta_{t-1} + inc_t))`, SERIAL OVER
#: TOKENS with NO chunk reset, where Mamba-2's cumsum explicitly resets ("the
#: chunk boundary is a hard reset, that is the algorithm"). So the backward
#: carries one whole-sequence reverse chain that Mamba-2 does not have, and
#: **it is the only fold topology Mamba-3 adds over Mamba-2.**
#:
#: DESCENDING in `t` from `L - 1`, seed `+0.0` at `t = L`, one flushed
#: UNFUSED add per step, per `(b, h, r)`. It is a PURE SUFFIX SUM: DEVIATION
#: 1370 pins the mod's local Jacobian at exactly one, so the chain's
#: coefficient is exactly one, the fusion question disappears, and a builder
#: must NOT spell it as `fma(1.0, acc, x)` — that is the same value through a
#: different operation and it is a different kernel to certify.
comptime M3_3_DIRECTION_IS_DESCENDING_IN_T = True
comptime M3_3_SEED_IS_POSITIVE_ZERO = True
comptime M3_3_ADD_IS_UNFUSED = True
comptime M3_3_COEFFICIENT_IS_STRUCTURALLY_ONE = True
comptime M3_3_HAS_NO_CHUNK_RESET = True
#: **THE CONSEQUENCE, AND IT BELONGS IN ANY MAMBA-3 BACKWARD CONTRACT'S
#: SECTION 5.** `IDENTICAL_BACKWARD_PLAN.md` section 4.2 proves
#: sequence-length invariance STRUCTURALLY IMPOSSIBLE for a chain of this
#: shape, and Mamba-3 is the only one of the three models where that bites on
#: a quantity which is not the SSM state. The forward earns
#: decode-equals-prefill from PREFIX STABILITY (contract section 5: a prefill
#: of length `t` gives token `s < t` the scale `gamma_s + beta'_{s+1}` using
#: only tokens `<= t`). **The backward has no prefix property at all**, since
#: every `dtheta_t` depends on every token from `t` to `L-1`, so **THERE IS NO
#: DECODE BACKWARD FOR MAMBA-3, for the same reason there is none for
#: Mamba-1, and the trapezoid does not rescue it.**
comptime M3_3_NO_DECODE_BACKWARD_EXISTS = True
#: Upstream `tl.atomic_add`s into `dAngles` (`mamba3_siso_bwd.py`:1167) and
#: its own comment at `:1064` concedes the interaction with its autotuner:
#: "Do not autotune this kernel. It overwrites dK, dK_bias, dAngles via atomic
#: adds and autotuning will lead to multiple overwrites." **Their own comment
#: says their autotuner and their atomics are incompatible and that the guard
#: is a convention.** Refused here; the chain has a private-slot spelling and
#: needs no atomic.
comptime M3_3_UPSTREAM_ATOMIC_REFUSED = True

# ---------------------------------------------------------------------------
# NEW: the trapezoid's shifted leg. DEVIATION 1368.
# ---------------------------------------------------------------------------
#: Seam S9 builds token `t`'s K-row weight from a SHIFTED load,
#: `scale_t = gamma_t + beta'_{t+1}`, so `dscale_t` feeds `dt_t` and
#: `dsigma_t` through the `gamma` leg AND `dt_{t+1}` and `dsigma_{t+1}`
#: through the `beta'` leg.
#:
#: **PINNED AS A GATHER, NEVER A SCATTER.** Token `t` READS `dscale_{t-1}`
#: for its `beta'` leg. A scatter into `t+1` needs either a float atomic,
#: which IDENTITY_PATHS rows 1 and 2 refuse, or an ordering between writers,
#: which is a launch property; the gather needs neither and is the same
#: refusal T5 makes for the batch fold.
comptime M3_4_IS_A_GATHER_NOT_A_SCATTER = True
#: The two boundaries, both structural and both exactly `+0.0`:
#:   at `t == 0` the `beta'` leg's source `dscale_{-1}` does not exist;
#:   at the LAST REAL TOKEN the leg is absent already in the forward, because
#:   `dt_{t+1}` past the sequence end is `+0.0` (contract section 3: "the
#:   shifted loads cross chunk boundaries but never the sequence end ... a
#:   sequence's last K-row carries only its gamma leg — the beta' leg belongs
#:   to the NEXT call, and that is the trapezoid's seam, not a bug").
comptime M3_4_BOUNDARY_TERMS_ARE_STRUCTURAL_ZERO = True
#: **PREDICTED BITWISE INERT ON ANY FIXTURE WHERE `dt` AND `sigma(trap)` ARE
#: UNIFORM ACROSS TOKENS**, which is `IDENTICAL_BACKWARD_PLAN.md` section 2's
#: off-by-one hazard verbatim ("a plausible gradient that is bit identical on
#: any fixture where all `da` are equal") arriving at a new seam. The gate must
#: plant nonuniform `dt` and nonuniform `trap` and RAISE otherwise.
comptime M3_4_ARM_NEEDS_NONUNIFORM_DT_AND_TRAP = True

# ---------------------------------------------------------------------------
# NEW: the heavy-tail derivative. DEVIATION 1369.
# ---------------------------------------------------------------------------
#: Seam S5 is `A = clamp(-heavy_tail(dd_A), max = -A_floor)` with
#: `heavy_tail(x) = 1 + x` for `x >= 0` and `1 / (1 - x)` for `x < 0`. So the
#: derivative is exactly `1` on the right branch and `1/(1-x)^2` on the left,
#: and the left branch has TWO SPELLINGS THAT ARE DIFFERENT FLOATS.
#:
#: **PINNED: `pinned_mul(hv, hv)` where `hv = ftz(-A.out)`**, the forward's
#: already-rounded value recovered from the carded post-clamp stage by an
#: EXACT negation. The recovery is invalid exactly where the clamp bound
#: binds, and there DEVIATION 1364's mask is `+0.0` anyway, so the invalid
#: region is unreachable and does not need a second spelling.
#: `identical_div(1, ftz(pinned_mul(ftz(1 - x), ftz(1 - x))))` is the refused
#: alternative and the required-RED arm.
comptime M3_5_USES_THE_SQUARED_FORWARD_VALUE = True
#: **THE BRANCH PREDICATE READS `dd_A`, THE INPUT, NOT `A.out`.** That is what
#: makes this a new elementwise seam rather than a transcription: the forward
#: can be spelled branchlessly and the contract records that its branchless
#: `clamp_min + reciprocal(1 - clamp_max)` sum is bit-equal to the branch
#: spelling, "recorded 782-style, spelling differs, bits do not".
#: **THAT ARGUMENT DOES NOT TRANSFER TO THE DERIVATIVE** and must be redone or
#: refused; this declaration refuses it, so the branch IS the profile and a
#: branchless respelling of the derivative is a v2, not an optimization.
comptime M3_5_BRANCHES_ON_THE_INPUT_SIGN = True
comptime M3_5_BRANCHLESS_RESPELLING_REFUSED = True

# ---------------------------------------------------------------------------
# NEW: the mod-2pi Jacobian. DEVIATION 1370.
# ---------------------------------------------------------------------------
#: `mod2pi(x) = ftz(x - pinned_mul(2pi, floor(identical_div(x, 2pi))))` is
#: piecewise affine with slope exactly one away from the wraps, so its local
#: Jacobian is exactly one and **the backward at this seam is a COPY, not a
#: multiply by a computed one.** The falsifiable form is that no float
#: operation occurs at this seam at all; a builder who spells
#: `pinned_mul(d, jac)` with a computed `jac` has added a rounding the profile
#: does not have, and one that is not even guaranteed to be exactly one.
comptime M3_MOD2PI_JACOBIAN_IS_A_STRUCTURAL_COPY = True

# ---------------------------------------------------------------------------
# THE JOIN RULE. DEVIATION 1371. NINE sites.
# ---------------------------------------------------------------------------
#: The rule Mamba-2's DEVIATION 1350 states once, applied here nine times:
#:
#:     Every join is spelled in ASCENDING FORWARD SEAM NUMBER of the
#:     consumers, left-associated, one flushed UNFUSED add per term.
#:
#: The nine sites, with their legs in the pinned order:
#:     `d_da_cs`   S16 , S17 , S20
#:     `dq_rot`    S16 , S17
#:     `dq_pre`    S13 (through the rotation) , S14 (bypassing it)
#:     `dk_pre`    S13 (through the rotation) , S14 (bypassing it)
#:     `dtheta`    the `q` pair , then the `k` pair (upstream's own order)
#:     `dgamma`    S9 , S14
#:     `dsigma`    S9's gamma leg , S9's shifted beta' leg
#:     `ddt`       S7 , S9's gamma leg , S9's shifted beta' leg , S10
#:     `dv`        S16 , S18 , S20
comptime B3_JOIN_RULE_IS_ASCENDING_SEAM_ORDER = True
comptime B3_JOIN_SITES = 9
comptime B3_JOIN_ADDS_ARE_UNFUSED = True
#: **`ddt` IS A FOUR-WAY JOIN AND `dv` IS A THREE-WAY**, which no other model
#: in this family has, and the four-way exists only because of the trapezoid's
#: shifted load. The `dq_pre` and `dk_pre` joins are the interesting two:
#: DEVIATION 830 contracts the PRE-rotation values at seam S14, so each of
#: those joins mixes ONE term that came through three roundings of rotation
#: with ONE that did not. Their legs have different rounding histories by
#: construction and the order is therefore not a formality.
comptime B3_DDT_JOIN_IS_FOUR_WAY = True
comptime B3_DV_JOIN_IS_THREE_WAY = True
#: **ALL NINE ARE PREDICTED BITWISE INERT ON ANY FIXTURE WHERE ONE LEG
#: DOMINATES.** Mamba-1 raised that warning about one site. Nine quiet
#: unfalsifiable arms is the `adv_softplus_guard` failure mode at scale. A
#: gate must plant the legs of each join at COMPARABLE MAGNITUDE and RAISE
#: VACUOUS rather than pass when it cannot.
comptime B3_JOIN_ARMS_PREDICTED_INERT_IF_ONE_LEG_DOMINATES = True

# ---------------------------------------------------------------------------
# INHERITED: the RMSNorm backward, THREE sites. DEVIATION 1372.
# ---------------------------------------------------------------------------
#: T7's closed form, unchanged from Mamba-1 and Mamba-2:
#:
#:     c3   = ftz(pinned_mul(ftz(pinned_mul(rstd, rstd)), rstd))
#:     s    = ftz(identical_div(c3, Float32(<row width>)))
#:     t2   = ftz(pinned_mul(ftz(pinned_mul(s, x[t,j])), drstd[t]))
#:     t1   = ftz(pinned_mul(rstd, dinner[t,j]))
#:     dx_norm[t,j] = ftz(t1 - t2)
comptime B3T7_RSTD3_IS_LEFT_ASSOCIATED = True
comptime B3T7_DIVIDES_BY_ROW_WIDTH_ONCE_PER_ROW = True
comptime B3T7_FINAL_SUBTRACT_IS_UNFUSED = True
#: **THREE SITES, THE MOST OF ANY MODEL HERE**: the block norm at `d_model`
#: (S1-S3), and the B and C norms at `N = 128` per `(b, l, g)` (S21), the
#: latter two fed by DEVIATION 1363's head fold. Mamba-3 has no gated norm at
#: the defaults (`is_outproj_norm = False`), so the gated-norm site Mamba-2
#: has is absent and two new ones replace it.
comptime B3T7_SITES = 3
#: The same trap as Mamba-2's DEVIATION 1351, now at SEVEN sites across four
#: models: T7's collapsed closed form and the transformer lane's node-by-node
#: spelling with `-0.5` and `2.0` kept explicit are the same derivative and
#: DIFFERENT FLOATS, and no existing gate can see the difference because each
#: is self-consistent against its own oracle. **A backward profile that does
#: not pin ONE of the two spellings across all seven sites has already
#: failed.** This file pins T7's; the reconciliation is a deviation owed by
#: the shared kit.
comptime B3T7_RECONCILIATION_WITH_TRANSFORMER_LANE_IS_OWED = True

# ---------------------------------------------------------------------------
# INHERITED: the absorption site. DEVIATION 1373.
# ---------------------------------------------------------------------------
#: T8. `dx[t,j] = ftz(ftz(dres[t,j]) + dx_norm[t,j])`, one add, `dres` LEFT.
#: **THE ABSORPTION SITE.** IDENTITY_PATHS row 55 records that a residual add
#: put a value of order 1e-3 beside one of order 1 and rounded the difference
#: away, so thirteen of sixteen stages moved under a sabotage and
#: `residual.out` was still bit identical. Any backward gate that compares
#: only `dx` is blind in exactly that way; the gate list must be per stage.
comptime B3T8_DRES_IS_THE_LEFT_OPERAND = True

# ---------------------------------------------------------------------------
# THE DELETIONS. DEVIATION 1374.
# ---------------------------------------------------------------------------
#: T2, the `h[t-1]` checkpoint, has no site: `mamba3_backward_state_floats`.
comptime B3_T2_HAS_NO_SITE = True
#: T4, the descending `dA` fold, has no site: Mamba-3's `A` is per token per
#: head and its gradient is elementwise, never a fold over the sequence.
comptime B3_T4_HAS_NO_SITE = True
#: T5, the parameter fold over the batch, has no site: all nine parameter
#: gradients fold the batch inside `M = B * L`, seven through
#: `mamba3_backward_reduce_into` and two through
#: `mamba3_backward_proj_b_into`. A builder who introduces a batch fold has
#: declined a routing and owes a deviation for it.
comptime B3_T5_HAS_NO_SITE = True
#: Mamba-2's DEVIATION 1344, the two-axis `dD` fold, has no site: seam S18's
#: single add makes one coefficient gradient serve both `dD` and `dqk_gamma`.
#: **One forward seam decision removed a topology from the backward**, and it
#: is the only place in this family where Mamba-3 is CHEAPER than Mamba-2.
comptime B3_TWO_AXIS_DD_HAS_NO_SITE = True

# ---------------------------------------------------------------------------
# THE ELEMENTWISE SPELLINGS WITH AN UPSTREAM DISAGREEMENT, inherited.
# ---------------------------------------------------------------------------
#: DEVIATION 1078's answer, INHERITED. `softplus'` is a MULTIPLY by
#: `identical_sigmoid` of the PRE-softplus biased value, not upstream's single
#: division, with the same `<= 20` guard. The distinguishing band is `biased`
#: in roughly `[8, 14]`; a fixture that only straddles 20 passes VACUOUSLY.
#: All three models hit this seam and it is answered ONCE.
comptime B3_SOFTPLUS_D_USES_SIGMOID_MULTIPLY = True
comptime B3_SOFTPLUS_D_GUARD_THRESHOLD_IS_20 = True
#: `exp'` is `exp`, so every decay derivative reuses the forward's rounded
#: value. Four sites: S16's `L`, S17's scaling, S20's `d_rev` and S20's
#: `scale_c`.
comptime B3_EXP_DERIVATIVE_REUSES_THE_FORWARD_VALUE = True
#: `tanh'` is `1 - tanh^2` at the RECOMPUTED `identical_tanh(angle_raw)`, the
#: forward's own function (DEVIATION 821's `portable_tanhf`), never upstream's
#: `sigmoid(2x)*2 - 1` respelling, which DEVIATION 829 already refuses on the
#: forward side. The association of `d_a * pi * (1 - tanh^2)` is pinned LEFT
#: TO RIGHT.
comptime B3_TANH_D_IS_ONE_MINUS_TANH_SQUARED = True
comptime B3_TANH_D_ASSOCIATION_IS_LEFT_TO_RIGHT = True
#: `sigmoid'` is `sigma * (1 - sigma)` at the CARDED `trap.sigma`, not a
#: recomputed sigmoid, for the same reason `exp'` reuses its forward value.
comptime B3_SIGMOID_D_REUSES_THE_CARDED_SIGMA = True
#: Every recomputed forward quantity is spelled with the FORWARD's OWN
#: function. `identical_silu` is the ONE-division spelling; its derivative
#: needs `identical_sigmoid` SEPARATELY, a new call at seam S19's one site.
comptime B3_RECOMPUTE_USES_THE_FORWARD_SPELLING = True

# ---------------------------------------------------------------------------
# WHAT UPSTREAM DOES THAT THIS PROFILE REFUSES.
# ---------------------------------------------------------------------------
#: **UPSTREAM'S MAMBA-3 BACKWARD AUTOTUNES ITS CHUNK SIZE.**
#: `mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py`:22-26 and :1419-1428 wrap
#: its kernels in `@triton.autotune` over `CHUNK_SIZE` in `[32, 64]`, and the
#: forward runs at 64. **So their backward may select a chunk boundary its own
#: forward did not use, inside one training step.** DEVIATION 783's standing,
#: inherited by this profile at `Q = 64`, says `CHUNK_SIZE` "is part of the
#: arithmetic" with exactly the standing of gemm v1's `K_LEAF_MIN`; a backward
#: that autotunes `Q` is not a backward for this forward at all. This is the
#: forward fragility note (the Blackwell `num_stages > 1` silent corruption)
#: reappearing on the backward as a DESIGN CHOICE rather than a bug.
comptime B3_UPSTREAM_AUTOTUNED_CHUNK_SIZE_REFUSED = True
#: Five `tl.atomic_add` sites follow, on `dK` and `dK_bias` (:1138, :1142),
#: `dAngles` (:1167), `dDT` and `dTrap` (:1619, :1620). Refused. Their own
#: file also carries `dq_bias_partial` and `dk_bias_partial` (:1286), so the
#: private-slot idiom and the atomic idiom coexist in one upstream file.
comptime B3_UPSTREAM_ATOMICS_REFUSED = True
#: **NO FLOAT64 GRADIENT REFERENCE IS OBTAINABLE FOR THIS PROFILE and the
#: forward contract already says why**: HF transformers at the pin has NO
#: mamba3 model, so this is a ONE-REPOSITORY profile. A Mamba-3 backward is a
#: one-repository, one-implementation cross-check, weaker than either
#: sibling's, and that has to be said in the claim rather than discovered in
#: the gate.
comptime B3_NO_SECOND_REFERENCE_EXISTS = True


# ---------------------------------------------------------------------------
# THE TOPOLOGY REGISTER, and where each one's kernel WILL live.
# ---------------------------------------------------------------------------
# The sibling register's rule, applied here: a declaration with no named
# implementation is a declaration nobody can find. Every row answers NOT
# WRITTEN, which is the honest state.

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
    """Where the kernel for one declaration WILL live, and that it does not.

    The two files named do not exist. They mirror the forward's ownership
    split (`mamba/impl/mamba_ssm/ops/mamba3_siso.mojo` and
    `mamba/impl/mamba_ssm/modules/mamba3.mojo`):

        mamba/impl/mamba_ssm/ops/mamba3_siso_backward.mojo
            the SISO core's backward, seams S13'-S20'
        mamba/impl/mamba_ssm/modules/mamba3_backward.mojo
            the block's backward, seams S1'-S12' and S21'-S23'
    """
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
    """Every declared choice, as one printable block a gate can diff.

    A card records values. This records DECISIONS, and it exists because a
    build that silently compiled with a different declaration would otherwise
    produce a correctly labelled measurement of the wrong arm. Print it in
    every Mamba-3 backward gate's banner and diff it across vendors along with
    the card.
    """
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
