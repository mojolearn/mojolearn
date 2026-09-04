# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The backward pass of one Mamba-2 (SSD) block, ROUTING AND DECLARATIONS ONLY.

**THIS FILE CONTAINS NO FLOATING-POINT ARITHMETIC.** Not one float multiply,
not one float add, not one `ftz`, not one `identical_mul_add`, not one
`pinned_mul`, and no kernel of its own. It does contain INTEGER shape
arithmetic, because a routing layer is made of index and dimension
bookkeeping and pretending otherwise would be a slogan rather than a claim.
The falsifiable form is that **no expression below has a floating-point
type**. Every buffer is a `DeviceBuffer[DType.float32]` and nothing is ever
loaded out of one, so the token `Float32` occurs in this file only inside
prose. Grepping this file for `Float32` and finding only prose is the whole
test, exactly as it is in `mamba/checks/mamba_backward.mojo` and
`gemm/checks/gemm_backward.mojo`. It does two things and neither touches a
float.

1.  It ROUTES the parts of the Mamba-2 backward that are already somebody
    else's certified arithmetic. **Fourteen calls**, the same total as the
    Mamba-1 layer and a DIFFERENT SPLIT: Mamba-1 is eight projection calls
    and six token reductions, Mamba-2 is FOUR projection calls and TEN token
    reductions. Mamba-2 has two `nn.Linear` projections, not four (no
    `x_proj`, no `dt_proj`: `dt_raw`, `B` and `C` are columns of `in_proj`,
    contract section 2 and the `zxbcdt` split at mamba2.py:211-215), and it
    has three more parameters that reduce over tokens. The coincidence of the
    totals is a coincidence and is recorded so nobody reads it as a pattern.
2.  It DECLARES, and does not implement, every fold and recurrence topology
    the Mamba-2 backward needs that nothing in this repository owns. A
    declaration here is a name, a direction, a seed, a fusion decision, an
    index set and the statement of what would falsify it. **There is
    deliberately no kernel**, because a kernel written before its gate is a
    belief with a compile step.

THE PROFILE IS NOT AMENDED AND MAY NOT BE
------------------------------------------
`mamba/IDENTICAL_MAMBA2_CONTRACT.md` is CONSUMED here and never edited.
Nothing in this file changes a seam decision in its section 4, a constant in
its section 3 (`CHUNK_SIZE` included, DEVIATION 783) or the stage list in its
section 7, so nothing here is a `mojolearn.identical.mamba2.fp32.v1` v2. What
that contract's section 10 says about itself, "no training, no backward",
stays true of v1. The backward is a SEPARATE profile,
`mojolearn.identical.mamba2.bwd.fp32.v1`, which consumes the forward one and
adds clauses of its own, and it does not exist yet.
`gemm/IDENTICAL_FP32_CONTRACT.md` is likewise consumed and never edited.

THE PROPORTION, RE-COUNTED RATHER THAN INHERITED
-------------------------------------------------
`archive/plans/mamba/BACKWARD_SCOPE_M2_M3.md` section 2 counts 70 operations, split
6 copies / 3 (a) / 20 (b) / 41 (c), with 19 of the (c) rows called
c-TOPOLOGY and six of those called NEW topologies. **That document says of
itself that not one number in it was measured**, and reading the contracts
against it moves five rows and deletes one. This file's own count, under the
scope document's counting rule so the two are comparable:

    copies                                    6
    (a)  existing primitive, no new order      3
    (b)  routable to gemm v1                  24
    (c)  genuinely new arithmetic             37
         of which  c-elementwise              20
                   c-inherited order           3
                   c-TOPOLOGY                 14
                                        -------
                                              70

The five moves, each with the reading that forces it, are in the DECLARATIONS
section: `dD`'s inner fold and its token fold are two ROUTED calls once the
fold order is pinned (DEVIATION 1344); `dscale_c` is a routed cell by the
scope document's own sentence (1345); the transposed triangular contraction
is a routed cell at the FULL chunk length because the forward's padding rule
forbids a shortened one (1346); the clamp derivative is an elementwise
predicate, not a fold (1349); and the parameter fold over the batch **has no
site in this profile at all** (1352), because every Mamba-2 parameter
gradient's batch axis lives inside `M = B * L` and is therefore folded by a
routed contraction.

**The two findings that survive the recount are the ones worth carrying.**
The routed fraction is 34 percent, ABOVE Mamba-1's 31 (the scope document put
it below, at 29). And the Mamba-2 backward introduces **TWO new fold
topologies, not six** — DEVIATION 1347's segsum rectangle and DEVIATION
1348's in-chunk suffix cumsum. The other four "new topologies" of the scope
document are an order pin over two routed calls, two routed cells with
declared index sets, and an elementwise predicate.

WHAT THIS FILE DOES NOT BUY
----------------------------
Not identical training, not an identical model, not identical gradients for
one block. Thirty seven operations have no kernel and fourteen fold or
recurrence sites have only a declaration. What is here is the part two other
lanes already paid for, wired to this block's shapes, plus the pinned
decisions for the part that is new, so the remaining work can be counted
honestly rather than estimated.

**Nothing in this file has been compiled or run.**

DEVIATIONS 1340-1353 SPENT, 1354-1359 RESERVED. The band was verified free
by a repository-wide grep before it was opened; the evidence is in the report
that accompanied this file and the pattern that reproduces it is

    grep -rhoE "DEVIATIONS? 13[4-9][0-9]" . | sort -u

which must print nothing but this file's own band and Mamba-3's.

`[[mojo-buffer-freed-at-last-use]]`: every launcher here is an `_into` form
inherited from `gemm_backward.mojo`. It enqueues and returns. The CALLER owns
every buffer, including the ones vector and every pre-product, and must keep
each alive past its own `ctx.synchronize()`. There is deliberately no
synchronizing form, because a backward pass chains fourteen of these and one
wait per call is the wrong shape.
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
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    Mamba2Dims,
)


# ===========================================================================
# THE SABOTAGE SWITCHES
# ===========================================================================
# OFF in every build that does not name them, exactly as the forward file's
# eleven are. None of them touches arithmetic, because this file has none.
# They move WHICH certified product is computed and WHERE it is written,
# which is the only thing here that can be wrong.
#
# Build with, e.g.:
#     tools/with_identical_mode.sh pixi run mojo build \
#         -D MOJOLEARN_MAMBA2_SABOTAGE_BWD_PROJ_SWAP=1 \
#         -I . mamba/checks/mamba2_backward_compile_probe.mojo \
#         -o /tmp/mojolearn_m2b_probe

#: The `dA` and `dB` routings for the two projections are SWAPPED, so the
#: activation gradient is computed at the weight gradient's shape and the
#: other way round. Mamba-2 has no square-ish projection the way Mamba-1's
#: `dt_proj` is at `M == di`, so this arm is HARDER to hide here than there;
#: the gate must still assert the SHAPES and not only the values.
comptime SAB_BWD2_PROJ_SWAP = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_BWD_PROJ_SWAP"
]()
#: The workspace is sized from the FORWARD shape rather than from the
#: backward call's shape. Right at small shapes because allocations have
#: slack, whole `+0.0` regions at large ones. More likely here than in the
#: gemm lane because `dB`'s `k'` is the token count and the forward's `k` is
#: a layer width, so the two numbers differ at every real Mamba-2 shape.
comptime SAB_BWD2_WS_FROM_FORWARD = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_BWD_WS_FROM_FORWARD"
]()
#: DEVIATION 1344's other order: `dD` folds the TOKEN axis first and the
#: `headdim` axis second. Algebraically equal, a different float, and the one
#: that cannot route (see `mamba2_reduction_needs_inner_p_fold`). Bitwise
#: INERT at `P == 1`, which no Mamba-2 shape has, and NEAR-inert on any
#: fixture whose per-token `dD` contributions are of wildly different
#: magnitude; the gate must plant them at comparable magnitude and RAISE
#: rather than pass if it cannot.
comptime SAB_BWD2_DD_TOKENS_FIRST = is_defined[
    "MOJOLEARN_MAMBA2_SABOTAGE_BWD_DD_TOKENS_FIRST"
]()
#: The four `d(conv1d.weight)` taps are reduced into the wrong tap slot, `k`
#: descending. Values right, slot wrong. Bitwise inert when the conv weight
#: is symmetric in `k`, which is why the gate's fixture must plant an
#: asymmetric one and must REFUSE to pass otherwise. Inherited unchanged from
#: the Mamba-1 layer, over the WIDER channel set `CD = d_ssm + 2*G*N`.
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
    """Which sabotage this binary compiled with, for a check's banner.

    A check must print THIS, plus `gemm_backward_sabotage_name()`, plus the
    forward block's and the SSD core's own sabotage names, because a backward
    binary can carry a forward sabotage and a gemm sabotage as well and a
    banner naming only one of them mislabels the run.
    """
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
    """The arms a gate must NOT count as covered by this lane, with the
    reason. Printing this beside a green banner is the difference between a
    gate that is passing and a gate that is blind. Each entry is a PREDICTION
    until a gate turns it into a measurement.
    """
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


# ===========================================================================
# THE TWO PROJECTIONS
# ===========================================================================
# Contract seam S4 and section 2. There is no third projection: `dt_raw`, `B`
# and `C` are COLUMNS of `in_proj.out` (mamba2.py:211-215), not separate
# `nn.Linear` calls, which is the structural difference from Mamba-1 and the
# reason this lane routes four projection calls where Mamba-1 routes eight.

#: `norm.out [M, d_model] . W_in [d_in_proj, d_model]^T` -> `[M, d_in_proj]`,
#: columns `z | xBC | dt_raw`.
comptime PROJ2_IN = 0
#: `gnorm.out [M, d_inner] . W_out [d_model, d_inner]^T` -> `[M, d_model]`.
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
    """`(op, m, n, k)` of the FORWARD projection, from the block's dimensions.

    THE ONE DOOR. `gemm_identical.mojo` makes "launch geometry cannot reach
    the arithmetic" a property of the call graph by giving `(L, P)` exactly
    one producer, `gemm_backward.mojo` gives the backward shape exactly two,
    and `mamba_backward.mojo` gives the Mamba-1 projection shape exactly one.
    **THE MAMBA-2 FORWARD PROJECTION SHAPE HAS EXACTLY ONE PRODUCER, THIS
    FUNCTION**, and every routing, launcher and sizing helper below calls it.
    To change which product a Mamba-2 backward computes you would have to edit
    one pure, host-side, device-free function.

    Both rows are `OP_NT`, which is contract seam S4 and not a convention
    this file may vary. `m` is the token count `M = B * L` and is passed in
    rather than derived, because `Mamba2Dims` carries no batch or sequence
    length.

    `out_proj`'s `k` is `d_inner` and not `d_ssm` only because the profile
    pins `d_mlp = 0`, so `d_ssm == d_inner` (contract section 3). If a later
    profile ever admits `d_mlp > 0` this row changes and that is a v2, which
    is why the equality is spelled here rather than assumed at the call site.

    DEVIATION 1340. Contract section 4 seam S4 and DEVIATION 784; gemm
    contract sections 0.1 and 2.
    """
    var dm = dims.d_model
    var di = dims.d_inner
    if which == PROJ2_IN:
        return (OP_NT, m, dims.d_in_proj(), dm)
    return (OP_NT, m, dm, di)


def mamba2_backward_proj_a_call(
    which: Int, dims: Mamba2Dims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the ACTIVATION gradient of `which`.

    A thin, deliberate pass-through to `gemm_backward_a_call`, so the routing
    table has ONE definition in this repository and this lane cannot hold a
    second opinion about it. `dA` here is the gradient with respect to the
    projection's INPUT stage: `norm.out` for `in_proj`, `gnorm.out` for
    `out_proj`.

    Because both forward rows are `OP_NT` this always returns
    `OP_NN(dC, W) at (m, k, n)` with `dC` LEFT. `k'` is the projection's
    OUTPUT width, a layer width, bounded, so nothing new happens to the leaf
    rule on this side.

    DEVIATION 1340.
    """
    var fwd = mamba2_proj_forward_call(which, dims, m)
    comptime if SAB_BWD2_PROJ_SWAP:
        return gemm_backward_b_call(fwd[0], fwd[1], fwd[2], fwd[3])
    return gemm_backward_a_call(fwd[0], fwd[1], fwd[2], fwd[3])


def mamba2_backward_proj_b_call(
    which: Int, dims: Mamba2Dims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the WEIGHT gradient of `which`.

    Always `OP_TN(dC, A) at (n, k, m)` with `dC` LEFT.

    **`k'` IS `m`, THE TOKEN COUNT, IN BOTH ROWS**, so a microbatched Mamba-2
    training step is not bit equal to an unsplit one for `dW_in` or `dW_out`,
    ever, on any vendor. The forward is batch invariant because gemm contract
    6.1 forbids the leaf size from depending on `m`; the weight gradient
    contracts over the batch, so the batch arrives as `k`, where section 6 of
    the same contract REQUIRES the leaf size to depend on it. Both clauses
    are correct and they are about different calls. `IDENTICAL_BACKWARD_PLAN`
    section 4.1 is the consequence and it carries over verbatim, with the
    Mamba-2 amendment that **TEN more parameter gradients in this block have
    the same property and are not matmuls** (Mamba-1 has six).

    Contract DEVIATION 784's note that every forward projection has a bounded
    `k` does not survive here: at `M > 128` the partition has more than one
    leaf and v1's balanced tree is exercised through a Mamba-2 entry point for
    the first time, so `SAB_FOLD_STRIDE`, `SAB_PAD_PLUS_ZERO`,
    `SAB_FOLD_SERIAL` and `SAB_NODE_ORDER` become reachable here.

    DEVIATION 1340. Gemm contract sections 0.1, 2, 3, 6 and 6.1.
    """
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
    """The ACTIVATION gradient of projection `which`, into `d_input`.

    `d_output` is the gradient arriving at this projection's output stage,
    `[M, n]`. `weight` is the FORWARD weight, unchanged and untransposed.
    `d_input` receives `[M, k]`. `ws` must hold at least
    `mamba2_backward_proj_a_workspace_max_floats(which, dims, m)` floats.

    Everything numerical happens inside `identical_gemm_backward_a_into`,
    which calls `identical_gemm_into`, the certified entry point, unmodified.
    These bits are `mojolearn.identical.gemm.fp32.v1`'s bits at the shape the
    table names and there is no second arithmetic to certify.

    ASYNCHRONOUS, caller-owned buffers, `[[mojo-buffer-freed-at-last-use]]`.
    DEVIATION 1340.
    """
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
    """The WEIGHT gradient of projection `which`, into `d_weight`.

    `input_stage` is the FORWARD input to this projection, one of the recorded
    stages `norm.out` or `gnorm.out`, unchanged. `d_weight` receives the
    weight's own shape.

    **This is one of the two calls whose `k'` is the token count**, so its
    plan, its workspace and its leaf size all move when the batch or the
    sequence length moves. Nothing about that threatens cross-vendor identity
    and everything about it means a training run has to declare `(B, L)` and
    its microbatch schedule as part of its numerical specification.

    `dA` and `dB` write DISJOINT buffers and read the same `d_output`, so the
    two are independent, need no barrier between them, and may share one `ws`
    on one context.

    ASYNCHRONOUS, caller-owned buffers. DEVIATION 1340.
    """
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


# ===========================================================================
# THE TEN TOKEN-AXIS REDUCTIONS
# ===========================================================================
# Each is `sum over the M token rows of a [M, W] matrix` giving `[W]`, which
# is `identical_gemm_backward_bias_into(ctx, out, src, ones, ws, M, W)` and
# therefore an `OP_NN` v1 GEMM at `(1, W, M)`. DEVIATION 851 carries the
# argument that `fma(ftz(1.0), ftz(x), acc)` is ONE rounding of `x + acc`, so
# the ones vector turns the reduction into the gemm contract's own ascending
# flushed leaf chain and its own balanced tree, under the SAME certificate as
# the projections, instead of a second fold shape needing its own clause.
# `core/pinned_reduce.mojo::pinned_block_sum` is NOT reusable for any of them
# and DEVIATION 851 already carries that argument too.
#
# EIGHT OF THE TEN TAKE A PRE-PRODUCT THIS FILE DOES NOT COMPUTE, because the
# reduced quantity is a Hadamard product and not a raw stage. Those are
# ARITHMETIC and therefore NOT here; the launcher takes the already
# materialized product buffer and says so in its signature.
#
# ONE OF THE TEN ALSO TAKES AN INNER FOLD, `dD` over `headdim`, which is
# DEVIATION 1344 and is the only two-axis parameter gradient in any of the
# three Mamba profiles.

#: `dD[h] = sum over tokens of (sum over p of dskip[t,h,p] * x_post[t,h,p])`.
#: `src` is the pre-product ALREADY FOLDED over `p`, shape `[M, H]`.
#: DEVIATION 1344 pins the order and `mamba2_dd_inner_fold_call` gives the
#: inner cell's shape.
comptime RED2_D = 0
#: `d(dt_bias)[h] = sum over tokens of ddt_raw[t,h]`. `src` is the stage
#: directly; no pre-product. Contract seam S9's bias add.
comptime RED2_DT_BIAS = 1
#: `d(A_log)[h] = pinned_mul(sum over tokens of d_dA[t,h] * dt[t,h], A[h])`.
#: `src` is the pre-product `[M, H]`. The final product by `A[h]` is
#: arithmetic and is not here. **This routing is what deletes Mamba-1's T4**:
#: Mamba-1 declined to route `dA` because the routed form needed a third
#: `[M, di, N]` buffer, and Mamba-2's `A` is per HEAD (contract section 3,
#: `A = -exp(A_log)` at `[H]`, seam S5) so the pre-product buffer is `[M, H]`.
#: `IDENTICAL_BACKWARD_PLAN.md` section 3.4 asked for exactly this revisit and
#: said the routed form is strictly preferable when the memory allows.
comptime RED2_A_LOG = 2
#: `d(conv1d.bias)[d] = sum over tokens of dconv[t,d]`, `W = CD`. `src` is the
#: stage directly; no pre-product.
comptime RED2_CONV_BIAS = 3
#: `d(conv1d.weight)[d,k] = sum over tokens of dconv[t,d] * hin[t-3+k, d]`,
#: FOUR separate reductions, one per tap, `W = CD`. `src` is the pre-product
#: for tap `k`, and the shifted read that builds it must take pre-sequence
#: positions from the conv WINDOW exactly as forward seam S6 does.
comptime RED2_CONV_W_TAP0 = 4
comptime RED2_CONV_W_TAP1 = 5
comptime RED2_CONV_W_TAP2 = 6
comptime RED2_CONV_W_TAP3 = 7
#: `d(norm.weight)[j] = sum over tokens of dnrm[t,j] * inner[t,j]`, the BLOCK
#: norm, `W = d_model`. `inner` is seam S2's output BEFORE the weight
#: multiply; recovering it by dividing `norm.out` by the weight is a division
#: and a different answer.
comptime RED2_NORM_W = 8
#: `d(gated norm weight)[j] = sum over tokens of d_gn[t,j] * inner_g[t,j]`,
#: `W = d_inner` (`= d_ssm` at `d_mlp = 0`). Seam S21. Same `inner` warning.
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
    """`W`, the length of the reduced vector.

    Four distinct widths, where Mamba-1 has two. `H = nheads` for the three
    per-head parameters, `CD = conv_dim` for the five conv reductions,
    `d_model` for the block norm and `d_inner` for the gated norm. A gate
    that separates `SAB_BWD_BIAS_AXIS` at one width has not separated it at
    the others, and at `d_model = 32` with `B * L = 64` the token count is a
    small multiple of three of the four.
    """
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
    """The conv tap index `k` a `RED2_CONV_W_*` id writes, or `-1`.

    The slot, not the arithmetic. `SAB_BWD2_CONV_TAP_SLOT` reverses it, which
    puts correct values in the wrong tap and is BITWISE INERT when the conv
    weight is symmetric in `k`. The gate must assert the fixture is
    asymmetric before counting this arm as covered.
    """
    if which < RED2_CONV_W_TAP0 or which > RED2_CONV_W_TAP3:
        return -1
    var k = which - RED2_CONV_W_TAP0
    comptime if SAB_BWD2_CONV_TAP_SLOT:
        return (M2_D_CONV - 1) - k
    return k


def mamba2_reduction_needs_preproduct(which: Int) -> Bool:
    """True where `src` must be a Hadamard product the caller materialized.

    Eight of the ten. `RED2_DT_BIAS` and `RED2_CONV_BIAS` reduce a recorded
    stage directly. A caller that passes a raw stage where a pre-product
    belongs computes a plausible column sum of the wrong quantity and no
    shape check can see it, which is why a finite-difference arm and not an
    assertion here is the gate for this.
    """
    return which != RED2_DT_BIAS and which != RED2_CONV_BIAS


def mamba2_reduction_needs_inner_p_fold(which: Int) -> Bool:
    """True only for `RED2_D`. DEVIATION 1344.

    `D` is per head at `[H]` (contract section 3, `D_has_hdim = False`) and
    seam S20 spells `pinned_mul(x[l,h,p], D[h])`, so `dD[h]` crosses TOKENS
    and `headdim`. Every other parameter gradient in all three Mamba profiles
    crosses exactly one axis. This one crosses two and the order is pinned.
    """
    return which == RED2_D


def mamba2_dd_inner_fold_call(dims: Mamba2Dims) -> Tuple[Int, Int, Int, Int]:
    """`(op, m, n, k)` of `dD`'s INNER fold, the one over `headdim`.

    `OP_NN` at `(1, 1, P)` per `(token, head)`: `P = 64`, one leaf, one serial
    ascending chain seeded `+0.0` with a fused multiply-add per term, which is
    gemm v1's own leaf. The result is the `[M, H]` matrix `RED2_D` reduces.

    **THE ORDER IS PINNED BY THE ROUTING AND IS NOT FREE.** The scope
    document calls the two-axis order "a free choice that moves bits", which
    is true of the arithmetic and false of this profile: the token leg can
    only be the certified ones-vector `OP_NN` at `(1, H, M)` if the matrix
    handed to it is `[M, H]`, and the only way to get one is to fold `P`
    first. Folding tokens first yields `[H, P]` and then needs a SECOND,
    uncertified fold over `P`. So `P` first is the answer that keeps `dD`
    inside `mojolearn.identical.gemm.fp32.v1`, and `SAB_BWD2_DD_TOKENS_FIRST`
    is the other order kept as the required-RED arm.

    DEVIATION 1344.
    """
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
    """`out[w] = sum over the M rows of src[M, W]`, as a v1 `OP_NN` at
    `(1, W, M)`.

    `ones` holds `M` entries of EXACTLY one
    (`identical_gemm_backward_bias_ones_floats(m)`). A wrong value there is a
    wrong gradient with no symptom, because any vector produces a plausible
    weighted column sum; the gate for it is a finite-difference arm, not an
    allocation check.

    For a `RED2_CONV_W_*` id the caller writes into
    `out[d * M2_D_CONV + mamba2_reduction_tap(which)]` for each `d`, since the
    conv weight is `[CD, D_CONV]` row-major and this launcher writes a
    CONTIGUOUS `[W]` vector. The scatter into the tap column is the caller's
    and is a copy, not a seam.

      - `m == 0` gives `k' == 0`, so every entry of `out` is a stored `+0.0`.
        Correct, and gemm contract section 8 requires the value WRITTEN rather
        than skipped.
      - `W == 0` gives an empty output and nothing is written.

    **DEVIATION 1341**, and the token-axis consequence is the projections'.
    `k'` is `M`. These ten are as microbatch-sensitive as the two weight
    matmuls and they are the ten a reader is most likely to assume are exempt,
    because they look like reductions rather than matmuls.

    ASYNCHRONOUS, caller-owned buffers, INCLUDING `ones` and `src`.
    """
    identical_gemm_backward_bias_into(
        ctx, out, src, ones, ws, m, mamba2_reduction_width(which, dims)
    )


def mamba2_backward_ones_floats(m: Int) -> Int:
    """How long the shared ones vector is. One per token count, allocated once
    and reused by all ten reductions."""
    return identical_gemm_backward_bias_ones_floats(m)


# ===========================================================================
# THE INTRA-CHUNK BACKWARD CELL TABLE
# ===========================================================================
# SHAPES ONLY. No launcher, deliberately.
#
# Every contraction the Mamba-2 backward performs inside a chunk is a gemm v1
# cell whose contracted length is a PROFILE CONSTANT, which is the strongest
# structural fact in this lane and the reason `IDENTICAL_BACKWARD_PLAN.md`
# section 4.3's rule ("every crossing fold's partition is a pure function of
# the length of the axis it reduces") is satisfiable here by construction
# rather than by care. This table is what makes that checkable: it names each
# site, its contracted length, and the leaf partition `contract_leaf_size`
# gives at that length, so a gate can print the table and a reader can see
# that no entry moves with `B`, `L`, the launch or the vendor.
#
# There is no launcher because gemm v1's entry point is per cell and the
# BATCHED driver over `(b, chunk, head)` is the builders' execution plan,
# which DEVIATION 784's last sentence forbids from touching the arithmetic.
# Inventing that driver here would be inventing the thing the contract says
# must not be able to reach the numbers.

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
    """The CONTRACTED LENGTH of one intra-chunk backward cell.

    Every value returned here is built from `P = 64`, `N = 128` and
    `Q = 256`, which contract section 3 freezes. **Not one of them is a
    function of `B`, `L`, the launch or the vendor.** That is the sentence
    `IDENTICAL_BACKWARD_PLAN.md` section 4.3 could not write about Mamba-1,
    where T3 contracts over a model width and T4 over the sequence, and the
    chunked forward is what bought it.

    `CELL2_DSCALE` is the exception in magnitude and not in kind: `P * N` is
    8192, still a profile constant, and it is the ONLY site in this profile,
    forward or backward, whose partition is a deep tree (see
    `mamba2_backward_cell_leaves`).
    """
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
    """`(leaf_size, leaf_count)` for one cell, from `contract_leaf_size`.

    Pure, integer, and a compile-time answer at every site because
    `mamba2_backward_cell_k` returns a profile constant. Under
    `CONTRACT_K_LEAF_MIN = 128` the whole table is

        k =   32 .. 128   ->  one leaf, one serial ascending chain
        k =  256          ->  two leaves of 128, ONE fold level
        k = 8192          ->  64 leaves of 128, SIX fold levels

    so `CELL2_DSCALE` is the first and only place either the Mamba-2 forward
    or its backward exercises a balanced tree deeper than one level. A gate
    that fires `SAB_NODE_ORDER` or `SAB_FOLD_STRIDE` only through the `Q`
    cells has tested a two-leaf tree and reported on a sixty-four-leaf one.
    """
    var k = mamba2_backward_cell_k(which)
    var el = contract_leaf_size(k)
    return (el, leaf_count(k, el))


# ===========================================================================
# WORKSPACE SIZING
# ===========================================================================
# Gemm contract 13.5 and `identical_gemm_into`'s documented hazard. Sizing a
# workspace from the FORWARD shape is an out-of-bounds write, and it is more
# likely here than in the gemm lane because `dB`'s `k'` is the token count
# and the forward's `k` is a layer width, so the two numbers differ at every
# real Mamba-2 shape rather than coinciding by luck.


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
    """The `dB` sizing, and the one most likely to surprise, because `k'` is
    the token count and a plan chosen at one batch size is not the plan chosen
    at another."""
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
    """ONE number a backward pass can allocate once and reuse for all fourteen
    routed calls at this shape.

    Sharing one workspace across the fourteen is safe because they are
    enqueued on ONE `DeviceContext` and MAX runs them in order, so a later
    call's leaf kernel cannot start before an earlier call's fold kernel has
    finished reading. **It is not safe across two contexts or two streams**,
    and a second stream would be a change to this function's contract.

    **This number does NOT include the memory the declarations below need.**
    It does not include the intra-chunk cells' own workspaces either, since
    this file provides no launcher for them. `mamba2_backward_state_floats`
    is the one activation term that is large and unavoidable.

    DEVIATION 1340.
    """
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
    """`B * C * H * P * N`, the incoming-per-chunk state the backward reads.

    **THIS IS THE NUMBER THAT DELETES MAMBA-1's T2**, and it is the cleanest
    inheritance in the whole lane. `IDENTICAL_BACKWARD_PLAN.md` section 3.4
    spends DEVIATION 1071 refusing upstream's `a = h[t] - dbu[t]` recovery and
    prices `N` times the activation footprint to avoid it. In Mamba-2 the
    quantity that recovery fights to obtain is **already a carded forward
    stage**, `pass.states [B, C, H, P, N]`, "the state ENTERING each chunk"
    (contract section 7). There is nothing to refuse and nothing to price:
    this is `M * di * N / Q` entries per batch rather than `M * di * N`, a
    factor of `Q = 256` fewer.

    `C = ceil(l / Q)`. The chunk count is the only term here that moves with
    the sequence, and it moves as `1/Q`.
    """
    var c = (l + M2_CHUNK_SIZE - 1) // M2_CHUNK_SIZE
    if l <= 0:
        c = 0
    return b * c * dims.nheads * M2_HEADDIM * M2_D_STATE


# ===========================================================================
# THE DECLARATIONS
# ===========================================================================
# NOT KERNELS. Not stubs either. Each block below is the pinned decision for
# one topology or one order, in the forward contract's own vocabulary, stated
# so that whoever writes the kernel is transcribing rather than deciding, and
# so that a gate can print `mamba2_backward_declarations()` and diff it across
# vendors and across builds.
#
# A declaration is not a measurement. Every one of these is a PREDICTION about
# what will be identical and none of them has run.

# ---------------------------------------------------------------------------
# INHERITED, at a new axis: the reverse recurrence. DEVIATION 1342.
# ---------------------------------------------------------------------------
#: T1's topology, over CHUNKS rather than tokens. Forward seam S17 is
#: `h_c = ftz(fma(scale_c, h_{c-1}, chunk_states_c))`, serial over `c`, so its
#: backward is the same first-order linear recurrence run backwards:
#: `dh_{c-1} = ftz(identical_mul_add(scale_c, dh_c, <the S16 contribution>))`.
#: DESCENDING in `c` from `C - 1`, seeded `+0.0` at `c = C`, ONE rounding,
#: FUSED, because S17 is FUSED and only fusion has a portable spelling.
comptime B2T1_DIRECTION_IS_DESCENDING_IN_C = True
comptime B2T1_SEED_IS_POSITIVE_ZERO = True
comptime B2T1_IS_FUSED = True
#: The `scale` index used at step `c` is `c + 1`, NOT `c`, exactly as Mamba-1's
#: `T1_DA_INDEX_OFFSET`. **The off-by-one is bitwise inert whenever every
#: `scale_c` is equal across chunks, and at `C == 1` there is no step at all.**
#: `C = ceil(L / 256)`, so at every `L <= 256` this recurrence does not run and
#: at every `L <= 512` it runs one step. **NO MAMBA-2 BACKWARD GATE BELOW
#: L = 513 CAN WITNESS THIS DECLARATION.** The forward's shape set
#: (contract section 3, `L` in {1, 4, 256, 257, 513, 770}) contains exactly two
#: shapes that can, and a backward gate that runs the other four and reports
#: green has measured nothing here.
comptime B2T1_SCALE_INDEX_OFFSET = 1
#: The recurrence is one to two orders of magnitude SHORTER than Mamba-1's,
#: which reverses over `L` tokens. It stays inside one thread holding `P * N`
#: values, so it inherits the forward's structural launch invariance for free,
#: and the sequence-length non-invariance of
#: `IDENTICAL_BACKWARD_PLAN.md` section 4.2 applies to `C`, not to `L`.
comptime B2T1_STEPS_ARE_CHUNKS_NOT_TOKENS = True

# ---------------------------------------------------------------------------
# INHERITED, at two lengths: the shared-channel contraction. DEVIATION 1343.
# ---------------------------------------------------------------------------
#: T3. `B` and `C` are shared across heads by COPY (contract section 3,
#: `ngroups = 1`, "broadcast, not arithmetic"), and a copy forward is a
#: CONTRACTION backward. Serial ASCENDING in the crossed index from `+0.0`,
#: one FUSED multiply-add per term, bit for bit gemm v1 at
#: `(1, N, <crossed length>)`.
comptime B2T3_DIRECTION_IS_ASCENDING = True
comptime B2T3_SEED_IS_POSITIVE_ZERO = True
comptime B2T3_IS_FUSED = True
#: **THREE SITES AT TWO LENGTHS**, which Mamba-1 does not have.
#:   `dC` from S18's state read-out crosses heads AND headdim, length
#:       `H * P = d_inner`. Mamba-1's T3 at the same length.
#:   `dB` from S15's chunk state crosses heads only, length `H`.
#:   `dG` from S13's mask product crosses heads only, length `H`.
comptime B2T3_SITES = 3
#: **THE PARTITION QUESTION AT `H` HAS AN ANSWER AND IT IS "ALWAYS ONE LEAF".**
#: `H = d_inner / 64 = d_model / 32`, and `contract_leaf_size(k) = k` for every
#: `k <= 128`, so the `H`-length folds are a single serial ascending chain at
#: every `d_model <= 4096`. The scope document's worry that
#: `contract_partition(H)` "may be one leaf where `contract_partition(di)` is
#: many" resolves to: it is one leaf at every shape this profile can be built
#: at, and the threshold at which that stops being true is `d_model > 4096`.
comptime B2T3_H_FOLD_IS_ONE_LEAF_BELOW_DMODEL = 4096
#: **AND THAT MAKES THE `H` SITES VACUOUS FOR EVERY PARTITION ARM.** At
#: `d_model = 32` the gate shape (contract section 3) gives `H = 1` and the
#: fold is a SINGLE TERM, so every direction, seed and partition sabotage
#: against `dB` and `dG` is INERT BY CONSTRUCTION. At `d_model = 64`, `H = 2`.
#: A gate must ASSERT ITS ARM'S PREDICTED TERM COUNT and raise VACUOUS
#: otherwise, which is `IDENTICAL_BACKWARD_PLAN.md` section 7's lesson pointed
#: at a specific shape.
comptime B2T3_H_SITES_ARE_VACUOUS_AT_DMODEL_32 = True
#: **THE `d_inner` SITE IS ALSO VACUOUS AT BOTH GATE SHAPES**, for the mirror
#: reason: `d_inner` is 64 or 128 there, both one leaf. Witnessing v1's
#: balanced tree through `dC` needs `d_inner > 128`, that is `d_model >= 128`,
#: **a shape the Mamba-2 FORWARD gate set does not contain.** A backward gate
#: must add it. This is a new fixture requirement, not an inherited one.
comptime B2T3_DI_TREE_NEEDS_DMODEL_AT_LEAST = 128

# ---------------------------------------------------------------------------
# NEW: the two-axis dD fold. DEVIATION 1344.
# ---------------------------------------------------------------------------
#: `P` FIRST, then tokens. `mamba2_dd_inner_fold_call` carries the argument:
#: the order is forced by the decision to route the token leg, not chosen.
#: Both legs are then gemm v1 cells and `dD` leaves category (c) entirely.
comptime M2_1_FOLDS_HEADDIM_FIRST = True
comptime M2_1_BOTH_LEGS_ARE_ROUTED = True

# ---------------------------------------------------------------------------
# NEW: the (P, N) crossing fold for dscale_c. DEVIATION 1345.
# ---------------------------------------------------------------------------
#: ROUTED, as a gemm v1 `OP_NN` cell at `(1, 1, P*N)` per `(b, h, chunk)`,
#: `k = 8192`. Routing is the right answer for T5's own reason: Metal has NO
#: threadgroup float atomics and 32 KB of threadgroup memory, so a fold needing
#: scratch proportional to `P * N` inside a threadgroup does not fit, and the
#: private-slot spelling had to exist anyway.
comptime M2_2_IS_ROUTED_TO_GEMM = True
#: **THE LINEARIZATION IS PART OF THE ARITHMETIC AND IS PINNED.** At `k = 8192`
#: the partition is 64 leaves of 128, so WHICH `(p, n)` pairs land in which
#: leaf is decided by the order in which the pairs are enumerated, and two
#: enumerations of the same 8192 terms are two different floats. Pinned
#: `p`-MAJOR, index `p * N + n`, because that is the row-major layout of
#: `pass.states [B, C, H, P, N]` and of `cstate.out`, so the pinned order is
#: the one that reads the buffer contiguously. An `n`-major enumeration is the
#: required-RED arm and it is NOT bitwise inert on any fixture, which makes it
#: one of the few arms in this lane a gate can trust.
comptime M2_2_LINEARIZATION_IS_P_MAJOR = True

# ---------------------------------------------------------------------------
# NEW: the transposed triangular contraction. DEVIATION 1346.
# ---------------------------------------------------------------------------
#: Forward S14 reads `M[i,j]` for `j <= i`; the backward's `dX_d[j]` reads
#: `M[i,j]` for `i >= j`. **THE FOLD RUNS OVER THE FULL CHUNK LENGTH `Q` WITH
#: STRUCTURAL ZEROS, NEVER OVER A SHORTENED RANGE.** The forward contract's
#: section 3 already decided this for the forward: "Every intra-chunk fold is
#: therefore ALWAYS length Q; a padded position contributes an exactly-zero
#: product to a fold and moves no nonzero bit." A shortened chain over only
#: `i >= j` has `k = Q - j`, a contracted length that depends on the OUTPUT
#: INDEX, and then `contract_leaf_size` returns a different partition per
#: column and the answer becomes a function of position. Full length keeps
#: `contract_leaf_size(256) = 128` with two leaves at every `j`.
comptime M2_3_FOLD_IS_FULL_Q_WITH_STRUCTURAL_ZEROS = True
#: **A TRANSPOSED IMPLEMENTATION IS BIT IDENTICAL TO THE CORRECT ONE ON ANY
#: FIXTURE WHOSE `M` IS SYMMETRIC**, and the profile's own zero padding pushes
#: fixtures toward symmetry rather than away from it (the padded rows and
#: columns are exact zeros, which is symmetric). The gate must plant an
#: asymmetric `M` and must ASSERT the asymmetry before counting the arm.
comptime M2_3_TRANSPOSE_ARM_NEEDS_ASYMMETRIC_M = True

# ---------------------------------------------------------------------------
# NEW TOPOLOGY 1 of 2: the segsum backward. DEVIATION 1347.
# ---------------------------------------------------------------------------
#: **IT IS A RECTANGLE, NOT A TRIANGLE, AND THAT IS THE WHOLE DECLARATION.**
#: S11 builds `seg[i][j] = sum over u in (j, i] of dA[u]` for `j < i`. So
#: `d_dA[k]` collects every `(i, j)` with `j < k <= i`, and because
#: `j <= k-1 < k <= i` forces `j < i`, the index set is the FULL RECTANGLE
#: `j in [0, k)` by `i in [k, Q)`. It has `k * (Q - k)` terms, at most 16384 at
#: `k = 128`, and the scope document's reading of it as "a two dimensional
#: reverse prefix fold over a triangle" is a harder problem than the one that
#: is actually there.
#:
#: PINNED AS TWO SERIAL PASSES, NEVER AS ONE `Q^3` fold:
#:
#:     PJ[i][k] = ftz(PJ[i][k-1] + dseg[i][k-1])     j ASCENDING, seed +0.0
#:     d_dA[k]  = ftz(acc + PJ[i][k])   over i = k .. Q-1, ASCENDING, seed +0.0
#:
#: The first pass is a running prefix in `j`, which is EXACTLY the ascending
#: fold of the first `k` terms and is therefore bit-equal to recomputing it
#: fresh at every `k` — this is partial-sum REUSE, not a prefix-difference
#: trick, and no subtraction of accumulated quantities occurs anywhere.
#: `j` ascending is the forward's own direction for the same triangle (S11
#: rebuilds `seg[i][j] = ftz(seg[i-1][j] + dA[i])` ascending).
comptime M2_4_IS_PREFIX_THEN_SUFFIX = True
comptime M2_4_INNER_J_IS_ASCENDING = True
comptime M2_4_OUTER_I_IS_ASCENDING = True
comptime M2_4_SEEDS_ARE_POSITIVE_ZERO = True
comptime M2_4_ADDS_ARE_UNFUSED = True
#: The two refused spellings, both required-RED:
#:   (i) the direct `k * (Q - k)` fold per `k`, `Q` times the work and a
#:       different association;
#:  (ii) a summed-area table with `d_dA[k]` recovered by DIFFERENCING two
#:       accumulated corner sums, which is unbounded relative cancellation
#:       for the same reason DEVIATION 1071 refuses `h[t] - dbu[t]`.
comptime M2_4_DIRECT_RECTANGLE_REFUSED = True
comptime M2_4_SUMMED_AREA_DIFFERENCE_REFUSED = True
#: Upstream ships two implementations of this seam and NAMES ONE OF THEM
#: UNSTABLE, `_chunk_scan_bwd_ddAcs_unstable_kernel` against
#: `_chunk_scan_bwd_ddAcs_stable_kernel` (`ssd_chunk_scan.py`:889, :1098).
#: Their own naming records that they found it numerically delicate, and
#: neither of their spellings is pinned here.
comptime M2_4_UPSTREAM_HAS_TWO_SPELLINGS = True

# ---------------------------------------------------------------------------
# NEW TOPOLOGY 2 of 2: the in-chunk reverse cumsum. DEVIATION 1348.
# ---------------------------------------------------------------------------
#: S11's `dA_cs[i] = ftz(dA_cs[i-1] + dA[i])` is a prefix sum ascending inside
#: the chunk with NO cross-chunk carry ("the chunk boundary is a hard reset,
#: that is the algorithm"). Its backward is the suffix sum, and a suffix sum's
#: running accumulation has to walk DESCENDING. Length `Q`, seed `+0.0` at
#: `i = Q`, one flushed UNFUSED add per step, per `(b, h, chunk)`.
comptime M2_5_DIRECTION_IS_DESCENDING_IN_I = True
comptime M2_5_SEED_IS_POSITIVE_ZERO = True
comptime M2_5_ADD_IS_UNFUSED = True
comptime M2_5_RESETS_AT_EVERY_CHUNK_BOUNDARY = True
#: **THE MAMBA-2 BACKWARD HAS EXACTLY TWO DESCENDING WALKS AND NO DESCENDING
#: CONTRACTION.** They are DEVIATION 1342's reverse chunk recurrence and this
#: one, and both are reverse ACCUMULATIONS whose direction the forward's own
#: direction forces. Mamba-1's T4, the one place that profile pins a
#: descending CONTRACTION against every other fold in both profiles, is
#: DELETED here (`RED2_A_LOG` routes it), and DEVIATION 1347 deliberately does
#: not reintroduce one. A builder who reaches for a descending contraction
#: anywhere in this backward has left the profile.
comptime M2_NO_DESCENDING_CONTRACTION_ANYWHERE = True

# ---------------------------------------------------------------------------
# NEW: the clamp derivative. DEVIATION 1349.
# ---------------------------------------------------------------------------
#: Forward seam S9 ends in `identical_clamp(x, lo, hi)`, DEVIATION 788's
#: primitive, and **no document in this repository defines its DERIVATIVE**.
#: This one does, and it is not a fold, so it is an elementwise predicate and
#: not a topology.
#:
#: `portable_clampf` (`checks/numerics.mojo`:2255-2271) is a TOTAL-ORDER
#: value-pick: it starts at `v = x`, replaces `v` by `lo` when
#: `_total_order_key(lo) > _total_order_key(v)`, then replaces `v` by `hi` when
#: `_total_order_key(hi) < _total_order_key(v)`. The gradient PASSES exactly
#: when the value-pick returned `x` unchanged, so the mask is
#:
#:     pass  <=>  key(lo) <= key(x_pre)  AND  key(hi) >= key(x_pre)
#:
#: with `key` the SAME `_total_order_key` the forward uses, never a float
#: `<=`. Two consequences a subgradient convention would get wrong.
comptime M2_6_MASK_USES_TOTAL_ORDER_KEYS = True
#: **AT `x == lo` AND AT `x == hi` THE GRADIENT PASSES**, because at equal
#: keys neither replacement fires and the clamp is the identity there. This is
#: the boundary case the scope document lists as unanswered, and the rule that
#: answers it is that the derivative agrees with which BRANCH the forward's
#: value-pick took, not with a mathematical convention about a corner.
comptime M2_6_BOUNDARY_IS_INCLUSIVE = True
#: **THE PREDICATE READS THE PRE-CLAMP VALUE, NEVER `dt.out`.** Comparing the
#: POST-clamp value to the limits is a different predicate: a clamped value
#: equals `lo` and so does an unclamped value that was already `lo`, and the
#: two have opposite gradients. `dt.out` is the carded post-clamp stage, so a
#: backward that reads only the card gets this wrong; the softplus output must
#: be carried or recomputed with `identical_softplus`, the forward's own
#: function.
comptime M2_6_PREDICATE_READS_PRE_CLAMP = True
#: A signed-zero note, because total order is not float order. `key(-0.0)` is
#: BELOW `key(+0.0)`, so at the profile default `dt_limit = (0.0, +inf)` an
#: `x_pre` of `-0.0` is CLAMPED and its gradient VANISHES, where a float
#: `x >= lo` compare would say `-0.0 >= +0.0` is true and pass it. The forward
#: cannot produce `-0.0` here (`identical_softplus` of a finite input is
#: `+0.0` or larger), so this is a claim about the SPELLING and not about a
#: reachable value, and it is written down so a builder does not "simplify"
#: the predicate into a float compare.
comptime M2_6_NEGATIVE_ZERO_IS_CLAMPED_BY_TOTAL_ORDER = True

# ---------------------------------------------------------------------------
# THE JOIN RULE. DEVIATION 1350. Five sites.
# ---------------------------------------------------------------------------
#: Mamba-1 has ONE join and spends DEVIATION 1075 on it, pinning the order
#: "following the forward's own data flow since `u` is consumed by S11, then
#: the scan, then S17b". **That is not a fact about `du`; it is a RULE, and
#: this profile states it once and applies it five times.**
#:
#:     Every join is spelled in ASCENDING FORWARD SEAM NUMBER of the
#:     consumers, left-associated, one flushed UNFUSED add per term.
#:
#: The five sites and their legs, in the pinned order:
#:     `d_dA_cs`   S12 (segsum) , S15 (decay) , S18 (the exp scaling)
#:     `ddt`       S10's `X_d` leg , S10's `dA` leg
#:     `dB`        S12 (the `G` contraction) , S15 (`B_decay`)
#:     `dC`        S12 (the `G` contraction) , S18 (the state read-out)
#:     `dx`        S10 (into `X_d`) , S20 (the `D` skip)
comptime B2_JOIN_RULE_IS_ASCENDING_SEAM_ORDER = True
comptime B2_JOIN_SITES = 5
comptime B2_JOIN_ADDS_ARE_UNFUSED = True
#: **EVERY ONE OF THE FIVE IS PREDICTED BITWISE INERT ON ANY FIXTURE WHERE ONE
#: LEG DOMINATES.** Mamba-1 raised that warning about one site; five quiet
#: unfalsifiable arms is the `adv_softplus_guard` failure mode at scale
#: (256 softplus inputs spanning [19.87, 20.10] with ZERO cells in the
#: distinguishing band, and 23 of 23 byte-identical dumps under a sabotage
#: that was reached and inert). A gate must plant the legs of each join at
#: COMPARABLE MAGNITUDE and RAISE VACUOUS rather than pass when it cannot.
comptime B2_JOIN_ARMS_PREDICTED_INERT_IF_ONE_LEG_DOMINATES = True

# ---------------------------------------------------------------------------
# INHERITED: the RMSNorm backward, TWO sites. DEVIATION 1351.
# ---------------------------------------------------------------------------
#: T7's closed form, unchanged:
#:
#:     c3   = ftz(pinned_mul(ftz(pinned_mul(rstd, rstd)), rstd))
#:     s    = ftz(identical_div(c3, Float32(<row width>)))
#:     t2   = ftz(pinned_mul(ftz(pinned_mul(s, x[t,j])), drstd[t]))
#:     t1   = ftz(pinned_mul(rstd, dinner[t,j]))
#:     dx_norm[t,j] = ftz(t1 - t2)
#:
#: `rstd^3` left associated, never `portable_powf`; the division hoisted once
#: per row, which is loop invariant in `j` and moves no bits; the final
#: subtraction UNFUSED.
comptime B2T7_RSTD3_IS_LEFT_ASSOCIATED = True
comptime B2T7_DIVIDES_BY_ROW_WIDTH_ONCE_PER_ROW = True
comptime B2T7_FINAL_SUBTRACT_IS_UNFUSED = True
#: TWO SITES in this model, at DIFFERENT ROW WIDTHS: the block norm at
#: `d_model` (S1-S3) and the gated norm at `d_ssm = d_inner` (S21). The row
#: width enters through `s`, so the two sites are the same spelling at two
#: divisors and a builder must not hoist the divisor out of the site.
comptime B2T7_SITES = 2
#: **THE TRAP, AND IT IS THE LARGEST SINGLE RISK IN THIS FILE.** This
#: repository already contains TWO different, self-consistent RMSNorm
#: backwards. T7 above folds the `-0.5` and the `2.0` into a subtraction;
#: `transformer/checks/transformer_backward.mojo`:824-931 spells the same
#: derivative node by node and defends keeping them, "`-0.5` AND `2.0` ARE
#: SPELLED, NOT ELIDED" (:865-869). They compute the same real number through
#: a different sequence of roundings and **they are different floats**.
#: Mamba-1 needs one such site, Mamba-2 two, Mamba-3 three and the transformer
#: one: SEVEN sites. If each lane picks locally this repository will carry two
#: RMSNorm gradients and no gate anywhere will notice, because each is
#: self-consistent against its own oracle. **A backward profile that does not
#: pin ONE of the two spellings across all seven sites has already failed.**
#: This file pins T7's; the reconciliation with the transformer lane is a
#: DEVIATION owed by whoever writes the shared kit, not a cleanup.
comptime B2T7_RECONCILIATION_WITH_TRANSFORMER_LANE_IS_OWED = True

# ---------------------------------------------------------------------------
# INHERITED: the absorption site. DEVIATION 1352 (a).
# ---------------------------------------------------------------------------
#: T8. `dx[t,j] = ftz(ftz(dres[t,j]) + dx_norm[t,j])`, one add, `dres` on the
#: LEFT. **THE ABSORPTION SITE.** IDENTITY_PATHS row 55 records that at the
#: shape where a forward sabotage was STRONGEST, thirteen of sixteen stages
#: moved, `out_proj.out` differed on 23 of 64 cells, and `residual.out` was
#: STILL BIT IDENTICAL. An output-only gate called that arm inert. Any
#: backward gate that compares only `dx` is blind in exactly the same way, and
#: that is why the gate list must be per stage.
comptime B2T8_DRES_IS_THE_LEFT_OPERAND = True

# ---------------------------------------------------------------------------
# THE DELETIONS. DEVIATION 1353. Three of Mamba-1's eight topologies have no
# site in this profile, and each deletion is a finding rather than an absence.
# ---------------------------------------------------------------------------
#: T2, the `h[t-1]` checkpoint, is not a decision here: the quantity is a
#: carded forward stage. `mamba2_backward_state_floats` carries the argument.
comptime B2_T2_HAS_NO_SITE = True
#: T4, the descending `dA` fold, is deleted because `RED2_A_LOG` ROUTES it.
#: Mamba-2's `A` is per HEAD, so the pre-product buffer is `[M, H]` and not
#: `[M, di, N]`. `IDENTICAL_BACKWARD_PLAN.md` section 3.4 wrote the condition
#: for this revisit before it was earned and said the routed form is "strictly
#: preferable, because it deletes a declared order in favor of a certified
#: one".
comptime B2_T4_HAS_NO_SITE = True
#: **T5, THE PARAMETER FOLD OVER THE BATCH, HAS NO SITE EITHER, AND THIS ONE
#: IS A CORRECTION.** The scope document's operation table gives it a row
#: (C70, "shared by nine parameters") and its shared-kit table marks it "yes"
#: for all three models. In Mamba-2 every parameter gradient's batch axis
#: lives INSIDE `M = B * L`: nine of the eleven parameters reduce through
#: `mamba2_backward_reduce_into` at `k' = M`, and `dW_in` and `dW_out` reduce
#: through `mamba2_backward_proj_b_into` at `k' = M`. A contraction over `M`
#: has already folded the batch. T5's private slots and its refusal of the
#: float atomic remain the right answer for any quantity a thread can only
#: produce per `(b, ...)`, and this profile has none, so the declaration here
#: is that **a builder who introduces a batch fold has declined a routing and
#: owes a deviation for it**, not that T5 was wrong.
comptime B2_T5_HAS_NO_SITE = True
#: The same reading raises a documentation defect in the Mamba-1 lane, recorded
#: here because this file may not edit that one:
#: `IDENTICAL_BACKWARD_PLAN.md` section 3.4 names T5 as "shared by `dA`, `dD`,
#: `db_dt`, `dcw` and `dcb`", and four of those five are routed through
#: `mamba_backward.mojo`'s ones-vector reductions at `k' = M` by DEVIATION
#: 1080, which folds their batch axis already. Only `dA`, which T4 declines to
#: route, plainly needs T5 there.
comptime B2_MAMBA1_T5_SITE_LIST_IS_QUESTIONED = True

# ---------------------------------------------------------------------------
# THE ELEMENTWISE SPELLINGS WITH AN UPSTREAM DISAGREEMENT.
# ---------------------------------------------------------------------------
#: DEVIATION 1078's answer, INHERITED and not re-answered. `softplus'` is a
#: MULTIPLY by `identical_sigmoid` of the PRE-softplus biased value, not
#: upstream's single division. Same `<= 20` guard. **The distinguishing band
#: is `biased` in roughly `[8, 14]` and a fixture that only straddles 20
#: passes this VACUOUSLY**; that has already bitten this lane once.
comptime B2_SOFTPLUS_D_USES_SIGMOID_MULTIPLY = True
comptime B2_SOFTPLUS_D_GUARD_THRESHOLD_IS_20 = True
#: `exp'` is `exp`, so every decay derivative REUSES the forward's already
#: rounded value rather than recomputing it. Four sites: S12's `L`, S15's
#: `decay`, S17's `scale_c` and S18's scaling. Recomputation is bit-equal only
#: if it uses `identical_exp` on the identical argument, and it is a second
#: call for nothing.
comptime B2_EXP_DERIVATIVE_REUSES_THE_FORWARD_VALUE = True
#: Every recomputed forward quantity is spelled with the FORWARD's OWN
#: function. Upstream breaks this rule in the Mamba-1 backward, recomputing
#: `silu(z)` as `z * sigmoid(z)` where its forward wrote `z / (1 + exp(-z))`,
#: which makes their gradient inconsistent with their own output.
#: `identical_silu` is the ONE-division spelling in all three Mamba profiles;
#: its derivative needs `identical_sigmoid` SEPARATELY, which is a new call at
#: two sites here (S7's conv SiLU and S8's gate SiLU).
comptime B2_RECOMPUTE_USES_THE_FORWARD_SPELLING = True

# ---------------------------------------------------------------------------
# WHAT UPSTREAM DOES THAT THIS PROFILE REFUSES, recorded as constants so a
# gate can print the refusals beside the declarations.
# ---------------------------------------------------------------------------
#: Upstream partitions its `dB`, `dC` and `dcb` head folds by
#: `torch.cuda.get_device_properties(...).multi_processor_count`
#: (`ssd_chunk_state.py`:937-940, `ssd_chunk_scan.py`:1485-1488, :1534-1537).
#: **Their summation order is a function of the GPU's SM count**, which varies
#: by SKU inside one vendor: an A100 and an H100 on the same weights and the
#: same tokens compute different `dB` bits. Those are exactly the three sites
#: DEVIATION 1343 covers.
comptime B2_UPSTREAM_SM_COUNT_SPLIT_REFUSED = True
#: Twelve live `tl.atomic_add` calls sit beside them, on `dD`, `ddt`, `dA`,
#: `ddt_bias` and `ddA_cumsum` at five sites. Float atomics are arrival order
#: and therefore not reproducible run to run on ONE device. IDENTITY_PATHS
#: rows 1 and 2 refuse exactly this shape.
comptime B2_UPSTREAM_ATOMICS_REFUSED = True
#: `mamba_ssm/utils/determinism.py`:21-27 gates their deterministic fallbacks
#: behind a `MAMBA_DETERMINISTIC` environment variable. The ambition is real
#: and theirs is not there yet.
comptime B2_UPSTREAM_DETERMINISM_IS_AN_ENV_VAR = True

# ---------------------------------------------------------------------------
# THE INHERITED GAP, stated so it is not rediscovered.
# ---------------------------------------------------------------------------
#: **THE CONV BACKWARD IS INHERITED UNREAD.** `IDENTICAL_BACKWARD_PLAN.md`
#: section 6 records that `causal_conv1d` IS NOT ON DISK at the pin and that
#: every statement about the conv backward is INFERRED rather than
#: transliterated. Mamba-2 runs the same conv over a WIDER channel set,
#: `CD = d_ssm + 2*G*N` against Mamba-1's `d_inner`, so it inherits the gap
#: and widens the surface the gap applies to. Five of the ten token-axis
#: reductions and the four-tap correlation that feeds them sit on the far side
#: of it.
comptime B2_CONV_BACKWARD_IS_INFERRED_NOT_READ = True


# ---------------------------------------------------------------------------
# THE TOPOLOGY REGISTER, and where each one's kernel WILL live.
# ---------------------------------------------------------------------------
# `mamba/checks/mamba_backward.mojo` grew a `mamba_backward_topology_site`
# table on 2026-09-03 with the reason "a declaration with no named
# implementation is a declaration nobody can find". The same table exists here
# and every row answers NOT WRITTEN, which is the honest state and is the
# reason the table is worth printing: a gate that prints it prints a list of
# absences rather than an impression of progress.

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
    """Where the kernel for one declaration WILL live, and that it does not.

    The two files named do not exist. They mirror the forward's ownership
    split and the Mamba-1 backward's, so the register points somewhere
    specific rather than at "the implementation":

        mamba/impl/mamba_ssm/modules/ssd_minimal_backward.mojo
            the SSD core's backward, seams S10'-S19'
        mamba/impl/mamba_ssm/modules/mamba2_backward.mojo
            the block's backward, seams S1'-S9' and S20'-S22'
    """
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
    """Every declared choice, as one printable block a gate can diff.

    A card records values. This records DECISIONS, and it exists because a
    build that silently compiled with a different declaration would otherwise
    produce a correctly labelled measurement of the wrong arm. That is the
    shared-checkout mode-flip lesson applied to a set of comptime constants.
    Print it in every Mamba-2 backward gate's banner and diff it across
    vendors along with the card.
    """
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
