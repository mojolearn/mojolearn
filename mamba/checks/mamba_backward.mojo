# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The backward pass of one Mamba-1 block, ROUTING AND DECLARATIONS ONLY.

**THIS FILE CONTAINS NO FLOATING-POINT ARITHMETIC.** Not one float multiply,
not one float add, not one `ftz`, not one `identical_mul_add`, not one
`pinned_mul`, and no kernel of its own. It does contain INTEGER shape
arithmetic, because a routing layer is made of index and dimension
bookkeeping and pretending otherwise would be a slogan rather than a claim.
The falsifiable form is that **no expression below has a floating-point
type**. Every buffer is a `DeviceBuffer[DType.float32]` and nothing is ever
loaded out of one, so the token `Float32` occurs in this file only inside
prose. It does two things and neither of them touches a float.

1.  It ROUTES the parts of the Mamba-1 backward that are already somebody
    else's certified arithmetic. Fourteen calls in total. Eight are the
    gradients of the block's four `nn.Linear` projections, which are
    `gemm/checks/gemm_backward.mojo`'s six-row table entered at this
    block's shapes, and six are token-axis parameter reductions spelled as
    `mojolearn.identical.gemm.fp32.v1` `OP_NN` products against a vector of
    ones, which is DEVIATION 851's trick reused unchanged.
2.  It DECLARES, and does not implement, the eight fold and recurrence
    topologies that the Mamba-1 backward needs and that nothing in this
    repository owns. A declaration here is a name, a direction, a seed, a
    fusion decision and a pointer at the section of
    `mamba/IDENTICAL_BACKWARD_PLAN.md` that prices the alternative. **There is
    deliberately no kernel**, because a kernel written before its gate is a
    belief with a compile step.

The claim is falsifiable rather than decorative, exactly as
`gemm_backward.mojo`'s is. **If a float-valued expression ever appears below,
the claim is false and `mamba/IDENTICAL_BACKWARD_PLAN.md` has to say so.**
Grepping this file for `Float32` and finding only prose is the whole test.

THE PROPORTION, STATED RATHER THAN PADDED
------------------------------------------
The gemm backward lane could write a 595 line file with no arithmetic in it
because `dA` and `dB` ARE the forward operation at a different transpose. That
is not true here. `IDENTICAL_BACKWARD_PLAN.md` section 3.3 counts 42 backward
operations. Thirteen route, two reuse an existing primitive, four are copies,
and **twenty three are genuinely new arithmetic**, of which eight are new fold
or recurrence topologies. So this file is SHORT and the plan is LONG, and that
is the correct shape for a lane whose finding is that most of the work is new.
Padding the routing layer to look like the gemm one would misreport the
result.

THE PROFILE IS NOT AMENDED AND MAY NOT BE
------------------------------------------
`mamba/IDENTICAL_MAMBA_CONTRACT.md` is frozen and is CONSUMED here. Nothing in
this file changes a seam decision in its section 4, a constant in its section
3, or the stage list in its section 7, so nothing here is a
`mojolearn.identical.mamba1.fp32.v1` v2. What section 9 of that contract says
about itself, "No training, no backward", stays true of v1. The backward is a
SEPARATE profile, `mojolearn.identical.mamba1.bwd.fp32.v1`, which consumes the
forward one and adds clauses of its own, and it does not exist yet.
`gemm/IDENTICAL_FP32_CONTRACT.md` is likewise consumed and never edited.

THE FOUR PROJECTIONS, AND A FINDING ABOUT THE SABOTAGE THAT COVERS THEM
-----------------------------------------------------------------------
Contract seam S17 says all four projections are `gemm.fp32.v1` `OP_NT` cells.
Row-major, no leading dimension, no sub-view.

    which       forward call                  A            B
    in_proj     OP_NT (M, 2*di, dm)           norm.out     W_in  [2di, dm]
    x_proj      OP_NT (M, r+2N, di)           silu.out     W_x   [r+2N, di]
    dt_proj     OP_NT (M, di, r)              dt_low       W_dt  [di, r]
    out_proj    OP_NT (M, dm, di)             gate.out     W_out [dm, di]

Reading `gemm_backward.mojo`'s table at `OP_NT`,

    dA = OP_NN(dC, B) at (m, k, n),   dC LEFT
    dB = OP_TN(dC, A) at (n, k, m),   dC LEFT

**ALL EIGHT OF THIS LANE'S PROJECTION CALLS PUT `dC` ON THE LEFT, AND THAT IS
A REPORTED HAZARD RATHER THAN A CONVENIENCE.** The gemm lane's
`SAB_BWD_OPERAND_ORDER` sabotage ignores the side flag and forces `dC` left.
It is wrong for three of the six rows of that table, but **it is BITWISE INERT
through every entry point in this file**, because Mamba-1 has no `OP_NN` and
no `OP_TN` forward matmul for it to be wrong about. A gate that runs that arm
through this lane and reports green has measured nothing. That is the "reached
but inert" failure this repository has already paid for once, on
`adv_softplus_guard`, where 256 softplus inputs spanned [19.87, 20.10] with
ZERO cells in the distinguishing band and `S14_THRESHOLD_10` produced 23 of 23
byte-identical dumps. `mamba_backward_inert_sabotages()` below returns the
list so a gate can print it beside its own banner instead of discovering it.

**`k'` IS THE TOKEN COUNT IN ALL FOUR `dB` CALLS.** `gemm_backward_b_call`'s
own docstring is the long form. `dW_in`, `dW_x`, `dW_dt` and `dW_out` each
contract over `M = B * L`, where `gemm/IDENTICAL_FP32_CONTRACT.md` section 6
REQUIRES the leaf size to depend on the contracted length. So a microbatched
training step is not bit equal to an unsplit one for any of the four, ever, on
any vendor. `IDENTICAL_BACKWARD_PLAN.md` section 4.1 is the consequence, and
its point is that **six MORE parameter gradients in this block have the same
property and are not matmuls**, so a reader who has absorbed the gemm finding
will assume they are exempt and they are not.

THE SIX TOKEN-AXIS REDUCTIONS
------------------------------
`dD`, `d(dt_proj.bias)`, `d(conv1d.bias)`, `d(conv1d.weight)` at four tap
offsets, and `d(norm.weight)` are each `sum over tokens of a [M, W] matrix`
giving `[W]`, which is `identical_gemm_backward_bias_into` at `(m, n) = (M,
W)` and therefore an `OP_NN` v1 GEMM at `(1, W, M)`. `core/pinned_reduce.mojo`
::`pinned_block_sum` is NOT reusable for any of them and DEVIATION 851 already
carries the argument. It pairs by STRIDE, which `gemm` contract 7.2 clause 1
names as a DIFFERENT ANSWER from v1's adjacent pairing, it folds only within
one block, and it does not flush its own partials.

**Five of the six need a PRE-PRODUCT that this file does not compute**, because
the reduced quantity is a Hadamard product and not a raw stage. `dD` needs
`pinned_mul(dsk, u)`, `d(norm.weight)` needs `pinned_mul(dnrm, inner)`, and
each of the four `d(conv1d.weight)` taps needs
`pinned_mul(dconv, hin_shifted_by_k)`. Those are ARITHMETIC and therefore they
are NOT here. The launchers below take the already-materialized product buffer
and say so in their signatures. `d(dt_proj.bias)` and `d(conv1d.bias)` are the
two that need no pre-product, because they reduce a stage directly.

WHAT THIS FILE DOES NOT BUY
----------------------------
Not identical training. Not an identical model. Not even identical gradients
for one block, because the eight declarations below are declarations and the
twenty three new arithmetic operations of the plan's section 3.2 have no
kernel. What is here is the part that was already paid for by two other lanes,
wired to this block's shapes so the remaining work can be counted honestly.

**Nothing in this file has been compiled or run.**

DEVIATIONS 1070-1078 (the eight declared topologies and the softplus
derivative), 1079 (the fourteen routings and their workspace sizing), 1080
(the six token-axis reductions as ones-vector v1 GEMMs and the pre-product
buffers they require).

`[[mojo-buffer-freed-at-last-use]]`: every launcher here is an `_into` form
inherited from `gemm_backward.mojo`. It enqueues and returns. The CALLER owns
every buffer, including the ones vector and every pre-product, and must keep
every one of them alive past its own `ctx.synchronize()`. A buffer created in
the caller's frame is DEAD at its `.unsafe_ptr()`, so a training step that
allocates a workspace, calls one of these and returns without waiting has
freed the workspace before the kernel ran. There is deliberately no
synchronizing form, because a backward pass chains fourteen of these and one
wait per call is the wrong shape.
"""

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


# ===========================================================================
# THE SABOTAGE SWITCHES
# ===========================================================================
# OFF in every build that does not name them, exactly as the forward file's
# six are (`modeling_mamba.mojo`, `selective_scan_interface.mojo`). None of
# them touches arithmetic, because this file has none. They move WHICH
# certified product is computed and WHERE it is written, which is the only
# thing here that can be wrong.
#
# Build with, e.g.:
#     tools/with_identical_mode.sh pixi run mojo run \
#         -D MOJOLEARN_MAMBA_SABOTAGE_BWD_PROJ_SWAP=1 \
#         -I . mamba/checks/mamba_backward_check.mojo

#: The `dA` and `dB` routings for the four projections are SWAPPED, so the
#: activation gradient is computed at the weight gradient's shape and the other
#: way round. At `dt_proj` with `M == di` and `r == di` the two shapes
#: coincide, so a gate that runs only a square-ish fixture cannot see it. The
#: gate must assert the SHAPES as well as the values.
comptime SAB_BWD_PROJ_SWAP = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_PROJ_SWAP"
]()
#: The workspace is sized from the FORWARD shape rather than from the backward
#: call's shape. This is the near-miss the gemm lane documented and paid for:
#: a 1-float workspace passed to a SPLITK dispatch at `64 x 4` still returned
#: the right answer because the allocation had slack, and only at `64 x 64`
#: did whole regions of the output come back `+0.0`. It is MORE likely here
#: than there, because `dB`'s `k'` is the token count and the forward's `k` is
#: a layer width, so the two numbers genuinely differ at every real shape.
comptime SAB_BWD_WS_FROM_FORWARD = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_WS_FROM_FORWARD"
]()
#: The four `d(conv1d.weight)` taps are reduced in the wrong tap order, `k`
#: descending. The reduction is over TOKENS in each case, so the values are
#: right and land in the wrong slot. Bitwise inert when the conv weight is
#: symmetric in `k`, which is why the gate's fixture must plant an asymmetric
#: one and must REFUSE to pass otherwise.
comptime SAB_BWD_CONV_TAP_SLOT = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_CONV_TAP_SLOT"
]()

comptime ANY_BWD_SABOTAGE = (
    SAB_BWD_PROJ_SWAP or SAB_BWD_WS_FROM_FORWARD or SAB_BWD_CONV_TAP_SLOT
)


def mamba_backward_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for a check's banner.

    A check must print THIS, plus `gemm_backward_sabotage_name()`, plus
    `mamba_block_sabotage_name()` and `mamba_scan_sabotage_name()`, because a
    backward binary can carry a forward sabotage and a gemm sabotage as well
    and a banner naming only one of the four mislabels the run.
    """
    comptime if SAB_BWD_PROJ_SWAP:
        return String("BWD_PROJ_SWAP")
    comptime if SAB_BWD_WS_FROM_FORWARD:
        return String("BWD_WS_FROM_FORWARD")
    comptime if SAB_BWD_CONV_TAP_SLOT:
        return String("BWD_CONV_TAP_SLOT")
    return String("none")


def mamba_backward_inert_sabotages() -> String:
    """The arms a gate must NOT count as covered by this lane, with the reason.

    Printing this beside a green banner is the difference between a gate that
    is passing and a gate that is blind. Each entry is a PREDICTION and gate
    MB1 is what turns it into a measurement.
    """
    return (
        String("SAB_BWD_OPERAND_ORDER: INERT here. All four Mamba-1")
        + " projections are OP_NT, so every one of the eight backward calls"
        + " puts dC on the LEFT and the side flag is never exercised.\n"
        + "SAB_BWD_UNTRANSPOSED: PARTIAL here. It moves the dA and dB shapes"
        + " for OP_NT, so it does bite, but it cannot distinguish OP_NN from"
        + " OP_TN routing because this lane calls neither.\n"
        + "SAB_BWD_BIAS_AXIS: covered, but only at M != W. At a shape where"
        + " the token count equals the reduced width it returns a vector of"
        + " the right length and the wrong contents."
    )


# ===========================================================================
# THE FOUR PROJECTIONS
# ===========================================================================
# Contract seam S17. There is no fifth projection; the contract says so.

#: `norm.out [M, dm] . W_in [2*di, dm]^T` -> `in_proj.out [M, 2*di]`.
comptime PROJ_IN = 0
#: `silu.out [M, di] . W_x [r+2N, di]^T` -> `x_proj.out [M, r+2N]`.
comptime PROJ_X = 1
#: `dt_low [M, r] . W_dt [di, r]^T` -> `dt_proj.out [M, di]`, bias NOT added.
comptime PROJ_DT = 2
#: `gate.out [M, di] . W_out [dm, di]^T` -> `out_proj.out [M, dm]`.
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
    """`(op, m, n, k)` of the FORWARD projection, from the block's dimensions.

    THE ONE DOOR. `gemm_identical.mojo` makes "launch geometry cannot reach
    the arithmetic" a property of the call graph by giving `(L, P)` exactly
    one producer, and `gemm_backward.mojo` gives the backward shape exactly
    two. The same discipline applies here to a third quantity. **THE FORWARD
    PROJECTION SHAPE HAS EXACTLY ONE PRODUCER, THIS FUNCTION**, and every
    routing, launcher and sizing helper below calls it. To change which
    product a Mamba backward computes you would have to edit one pure,
    host-side, device-free function.

    Every row is `OP_NT`, which is contract seam S17 and not a convention this
    file may vary. `m` is the token count `M = B * L` and is passed in rather
    than derived, because `MambaDims` carries no batch or sequence length.

    Pure. It reads `which`, the three dimensions and `m`, and nothing else, so
    a backward call at any shape gets a deterministic plan for the same reason
    a forward one does.

    DEVIATION 1079. Contract section 4 seam S17, gemm contract sections 0.1
    and 2.
    """
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
    """`(op', m', n', k', dc_side)` for the ACTIVATION gradient of `which`.

    A thin, deliberate pass-through to `gemm_backward_a_call`, so that the
    routing table has ONE definition in this repository and this lane cannot
    hold a second opinion about it. `dA` here is the gradient with respect to
    the projection's INPUT stage, which is `norm.out`, `silu.out`, `dt_low`
    and `gate.out` respectively.

    Because every forward row is `OP_NT`, this always returns
    `OP_NN(dC, W) at (m, k, n)` with `dC` LEFT. `k'` is the projection's
    OUTPUT width, which is a layer width and is bounded, so nothing new
    happens to the leaf rule on this side.

    DEVIATION 1079.
    """
    var fwd = mamba_proj_forward_call(which, dims, m)
    comptime if SAB_BWD_PROJ_SWAP:
        # SABOTAGE: the weight gradient's routing returned for the activation
        # gradient. Plausible shape, wrong product.
        return gemm_backward_b_call(fwd[0], fwd[1], fwd[2], fwd[3])
    return gemm_backward_a_call(fwd[0], fwd[1], fwd[2], fwd[3])


def mamba_backward_proj_b_call(
    which: Int, dims: MambaDims, m: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for the WEIGHT gradient of `which`.

    Always `OP_TN(dC, A) at (n, k, m)` with `dC` LEFT, because every forward
    row is `OP_NT`.

    **`k'` IS `m`, THE TOKEN COUNT, IN ALL FOUR ROWS.** That is the gemm
    lane's finding arriving here unchanged, and it is why `IDENTICAL_BACKWARD_
    PLAN.md` section 4.1 says a microbatched Mamba training step is not bit
    equal to an unsplit one. The forward is batch invariant because gemm
    contract 6.1 forbids the leaf size from depending on `m`; the weight
    gradient contracts over the batch, so the batch arrives as `k`, where
    section 6 of the same contract REQUIRES the leaf size to depend on it.
    Both clauses are correct and they are about different calls.

    It also means `k'` is unbounded in a way a forward Mamba `k` never is.
    Contract section 4's S17 note observes that every forward projection has
    `k <= 128` and therefore `P == 1`, a single serial ascending chain. **Not
    one of the four backward `dB` calls has that property.** At `M > 128` the
    partition has more than one leaf and the balanced tree is exercised for
    the first time anywhere in this profile, so `SAB_FOLD_STRIDE`,
    `SAB_PAD_PLUS_ZERO`, `SAB_FOLD_SERIAL` and `SAB_NODE_ORDER` become
    reachable through a Mamba entry point when they never were before. A gate
    that does not run them here is leaving four already-written sabotages
    unfired on a path that has never carried them.

    DEVIATION 1079. Gemm contract sections 0.1, 2, 3, 6 and 6.1.
    """
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
    """The ACTIVATION gradient of projection `which`, into `d_input`.

    `d_output` is the gradient arriving at this projection's output stage,
    `[M, n]`. `weight` is the FORWARD weight, unchanged and untransposed.
    `d_input` receives `[M, k]`. `ws` must hold at least
    `mamba_backward_proj_a_workspace_max_floats(which, dims, m)` floats.

    Everything numerical happens inside `identical_gemm_backward_a_into`,
    which calls `identical_gemm_into`, which is the certified entry point and
    is not modified. This function chooses the projection and gets out of the
    way, so these bits are `mojolearn.identical.gemm.fp32.v1`'s bits at the
    shape the table names. There is no second arithmetic to certify.

    ASYNCHRONOUS, caller-owned buffers, `[[mojo-buffer-freed-at-last-use]]`.
    DEVIATION 1079.
    """
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
    """The WEIGHT gradient of projection `which`, into `d_weight`.

    `input_stage` is the FORWARD input to this projection, which is one of the
    recorded stages `norm.out`, `silu.out`, `dt_low` or `gate.out`, unchanged.
    `d_weight` receives the weight's own shape.

    **This is one of the four calls whose `k'` is the token count**, so its
    plan, its workspace and its leaf size all move when the batch or the
    sequence length moves. Nothing about that threatens cross-vendor identity
    and everything about it means a training run has to declare `(B, L)` and
    its microbatch schedule as part of its numerical specification.

    `dA` and `dB` write DISJOINT buffers and read the same `d_output`, so the
    two are independent and need no barrier between them. They may share one
    `ws` on one context.

    ASYNCHRONOUS, caller-owned buffers. DEVIATION 1079.
    """
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


# ===========================================================================
# THE SIX TOKEN-AXIS REDUCTIONS
# ===========================================================================
# Each is `sum over the M token rows of a [M, W] matrix` giving `[W]`, which
# is `identical_gemm_backward_bias_into(ctx, out, src, ones, ws, M, W)` and
# therefore `OP_NN` at `(1, W, M)`. DEVIATION 851 carries the argument that
# `fma(ftz(1.0), ftz(x), acc)` is ONE rounding of `x + acc`, so the ones
# vector turns the reduction into the gemm contract's own ascending flushed
# leaf chain and its own balanced tree, under the SAME certificate as the
# projections, instead of a second fold shape needing its own clause.
#
# FIVE OF THE SIX TAKE A PRE-PRODUCT THIS FILE DOES NOT COMPUTE. See the
# module docstring. The `src` argument is the already-materialized buffer.

#: `dD[d] = sum over tokens of dsk[t,d] * u[t,d]`. `src` is the pre-product.
comptime RED_D = 0
#: `d(dt_proj.bias)[d] = sum over tokens of ddtp[t,d]`. `src` is `bwd.ddtp`
#: directly; no pre-product.
comptime RED_DT_BIAS = 1
#: `d(conv1d.bias)[d] = sum over tokens of dconv[t,d]`. `src` is `bwd.dconv`
#: directly; no pre-product.
comptime RED_CONV_BIAS = 2
#: `d(conv1d.weight)[d,k] = sum over tokens of dconv[t,d] * hin[t-3+k, d]`,
#: FOUR separate reductions, one per tap. `src` is the pre-product for tap
#: `k`, and the shifted read that builds it must take a pre-sequence position
#: from the conv WINDOW exactly as forward seam S13 does.
comptime RED_CONV_W_TAP0 = 3
comptime RED_CONV_W_TAP1 = 4
comptime RED_CONV_W_TAP2 = 5
comptime RED_CONV_W_TAP3 = 6
#: `d(norm.weight)[j] = sum over tokens of dnrm[t,j] * inner[t,j]`. `src` is
#: the pre-product. Note `inner` is S3's output, BEFORE the weight multiply,
#: and recovering it by dividing `norm.out` by the weight is a division and a
#: different answer.
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
    """`W`, the length of the reduced vector. `d_inner` for six of the eight
    ids and `d_model` for `RED_NORM_W`."""
    if which == RED_NORM_W:
        return dims.d_model
    return dims.d_inner


def mamba_reduction_tap(which: Int) -> Int:
    """The conv tap index `k` a `RED_CONV_W_*` id writes, or `-1`.

    The slot, not the arithmetic. `SAB_BWD_CONV_TAP_SLOT` reverses it, which
    puts correct values in the wrong tap and is BITWISE INERT when the conv
    weight is symmetric in `k`. Gate MB1 must assert the fixture is
    asymmetric before counting this arm as covered.
    """
    if which < RED_CONV_W_TAP0 or which > RED_CONV_W_TAP3:
        return -1
    var k = which - RED_CONV_W_TAP0
    comptime if SAB_BWD_CONV_TAP_SLOT:
        return (D_CONV - 1) - k
    return k


def mamba_reduction_needs_preproduct(which: Int) -> Bool:
    """True where `src` must be a Hadamard product the caller materialized.

    Six of the eight. `RED_DT_BIAS` and `RED_CONV_BIAS` reduce a recorded
    stage directly. A caller that passes a raw stage where a pre-product
    belongs computes a plausible column sum of the wrong quantity and no
    shape check can see it, which is why gate MB2's finite-difference arm is
    the gate for this and not an assertion here.
    """
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
    """`out[w] = sum over the M rows of src[M, W]`, as a v1 `OP_NN` at
    `(1, W, M)`.

    `ones` holds `M` entries of EXACTLY `Float32(1.0)`
    (`identical_gemm_backward_bias_ones_floats(m)`). A wrong value there is a
    wrong gradient with no symptom, because any vector produces a plausible
    weighted column sum, and the gate for it is the finite-difference arm of
    MB2 rather than an allocation check.

    For a `RED_CONV_W_*` id the caller writes into
    `out[d * D_CONV + mamba_reduction_tap(which)]` for each `d`, since the
    conv weight is `[d_inner, D_CONV]` row-major, and this launcher writes a
    CONTIGUOUS `[W]` vector. The scatter into the tap column is the caller's
    and is a copy, not a seam. Four calls, four contiguous vectors, one
    interleave.

      - `m == 0` gives `k' == 0`, so every entry of `out` is a stored `+0.0`.
        Correct, no tokens means no parameter gradient, and gemm contract
        section 8 requires the value to be WRITTEN rather than skipped.
      - `W == 0` gives an empty output and nothing is written.

    **DEVIATION 1080, and the token-axis consequence is the same one the
    projections carry.** `k'` is `M`. These six are as microbatch-sensitive as
    the four weight matmuls and they are the six a reader is most likely to
    assume are exempt, because they look like reductions rather than matmuls.

    ASYNCHRONOUS, caller-owned buffers, INCLUDING `ones` and `src`.
    """
    identical_gemm_backward_bias_into(
        ctx, out, src, ones, ws, m, mamba_reduction_width(which, dims)
    )


def mamba_backward_ones_floats(m: Int) -> Int:
    """How long the shared ones vector is. One per token count, allocated
    once and reused by all six reductions."""
    return identical_gemm_backward_bias_ones_floats(m)


# ===========================================================================
# WORKSPACE SIZING
# ===========================================================================
# Gemm contract 13.5, and `identical_gemm_into`'s documented hazard. Sizing a
# workspace from the FORWARD shape is an out-of-bounds write, and it is more
# likely here than in the gemm lane because `dB`'s `k'` is the token count and
# the forward's `k` is a layer width, so the two numbers differ at every real
# Mamba shape rather than coinciding by luck.


def mamba_backward_proj_a_workspace_max_floats(
    which: Int, dims: MambaDims, m: Int
) -> Int:
    var fwd = mamba_proj_forward_call(which, dims, m)
    comptime if SAB_BWD_WS_FROM_FORWARD:
        # SABOTAGE: the FORWARD call's workspace, `(m, n, k)` rather than the
        # backward call's `(m', n', k')`. Right at small shapes because
        # allocations have slack; whole `+0.0` regions at large ones.
        return identical_gemm_workspace_max_floats(fwd[1], fwd[2], fwd[3])
    return identical_gemm_backward_a_workspace_max_floats(
        fwd[0], fwd[1], fwd[2], fwd[3]
    )


def mamba_backward_proj_b_workspace_max_floats(
    which: Int, dims: MambaDims, m: Int
) -> Int:
    """The `dB` sizing, and the one most likely to surprise, because `k'` is
    the token count and a plan chosen at one batch size is not the plan chosen
    at another."""
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
    """ONE number a backward pass can allocate once and reuse for all fourteen
    routed calls at this shape.

    Sharing one workspace across the fourteen is safe because they are
    enqueued on ONE `DeviceContext` and MAX runs them in order, so a later
    call's leaf kernel cannot start before an earlier call's fold kernel has
    finished reading. The forward gemm gate asserts the stronger form of this,
    running four GEMMs through one DIRTY shared workspace and requiring the
    same bits. **It is not safe across two contexts or two streams**, and a
    second stream would be a change to this function's contract.

    **This number does NOT include the memory the eight declared topologies
    need**, which is `IDENTICAL_BACKWARD_PLAN.md` section 5.3 and which
    dominates. `T2_H_CHECKPOINT_FLOATS` and `T3_DH_FLOATS` below are those.

    DEVIATION 1079.
    """
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


# ===========================================================================
# THE EIGHT DECLARATIONS
# ===========================================================================
# NOT KERNELS. Not stubs either. Each block below is the pinned decision for
# one topology, in the contract's own vocabulary, stated so that whoever
# writes the kernel is transcribing rather than deciding, and so that a gate
# can print `mamba_backward_declarations()` and diff it across vendors and
# across builds. `IDENTICAL_BACKWARD_PLAN.md` section 3.4 carries the
# alternatives and their prices; the constants here are the choices.
#
# A declaration is not a measurement. Every one of these is a PREDICTION about
# what will be identical, and none of them has run.

#: T1, DEVIATION 1070. The reverse state recurrence of seam S9.
#: `dh[t] = fma(da[t+1], dh[t+1], pinned_mul(dy[t], C[t,n]))`.
#: Walked t DESCENDING from `L - 1`, seeded `+0.0` at `t = L`, ONE rounding.
comptime T1_DIRECTION_IS_DESCENDING_IN_T = True
comptime T1_SEED_IS_POSITIVE_ZERO = True
comptime T1_IS_FUSED = True
#: The `da` index used at step `t` is `t + 1`, NOT `t`. Section 2.3 check
#: three. Getting this wrong is bitwise inert when every `delta` is equal
#: across tokens, and at `L == 1`.
comptime T1_DA_INDEX_OFFSET = 1

#: T2, DEVIATION 1071. `h[t-1]` comes from an explicit checkpoint, never from
#: upstream's `h[t] - dbu[t]` subtraction, which has unbounded relative
#: cancellation when `|da*h[t-1]| << |dbu[t]|`.
comptime T2_H_IS_CHECKPOINTED = True
comptime T2_H_SUBTRACTION_REFUSED = True


def t2_h_checkpoint_floats(b: Int, l: Int, dims: MambaDims) -> Int:
    """`B * d_inner * (L + 1) * D_STATE`, the checkpoint's size.

    The `+ 1` is the incoming carried state at index `t = -1`, which the
    backward reads at `t = 0` and which is zeros on prefill (contract section
    5). Section 5.3 of the plan prices this at model scale and the answer is
    uncomfortable.
    """
    return b * dims.d_inner * (l + 1) * D_STATE


#: T3, DEVIATION 1072. `dB` and `dC` contract over `d_inner`, an axis no float
#: crosses in the forward. Serial ASCENDING `d` from `+0.0`, one FUSED
#: multiply-add per term, which is bit for bit gemm v1 at `(1, D_STATE,
#: d_inner)` per token. **If `d_inner > 128` this is NOT one chain**, it is
#: `contract_partition(d_inner)` leaves combined by v1's balanced tree, and an
#: implementation that treats it as one chain is a different answer at every
#: real model width.
comptime T3_DIRECTION_IS_ASCENDING_IN_D = True
comptime T3_SEED_IS_POSITIVE_ZERO = True
comptime T3_IS_FUSED = True
#: The `dB` leaf multiplies a PRE-FORMED `w[t,d] = pinned_mul(delta, u)` by
#: `dh[t,d,n]`, rather than pre-forming `R = pinned_mul(dh, u)` and
#: multiplying by `delta`. The two are algebraically equal, differ in the last
#: bit, and the chosen one costs `D_STATE` times less memory.
comptime T3_DB_PREFORMS_DELTA_TIMES_U = True


def t3_dh_floats(b: Int, l: Int, dims: MambaDims) -> Int:
    """`B * d_inner * L * D_STATE`, the materialized `dh`.

    Paid on top of T2's checkpoint. Together they are the largest open item of
    this lane and section 5.3 of the plan is the table of what they cost at
    model scale. The declared spelling is GATE-SCALE; a blocked variant is
    phase K and its tiling must be `contract_partition(d_inner)`'s leaf
    boundary, never a VRAM budget, or the answer becomes a function of the
    machine.
    """
    return b * dims.d_inner * l * D_STATE


#: T4, DEVIATION 1073. The `dA` fold over tokens runs DESCENDING in `t`, the
#: direction the reverse pass already walks. This is the only descending fold
#: in either profile and the plan states why. `dA` is routable to a
#: ones-vector v1 GEMM and is deliberately not routed, because that would need
#: a THIRD `[M, d_inner, D_STATE]` buffer.
comptime T4_DIRECTION_IS_DESCENDING_IN_T = True
comptime T4_SEED_IS_POSITIVE_ZERO = True
comptime T4_IS_FUSED = True
comptime T4_ROUTED_TO_GEMM = False

#: T5, DEVIATION 1074. Every parameter gradient a thread can only compute per
#: `(b, d)` goes to a PRIVATE SLOT `partial[b, d, ...]` with no atomic
#: anywhere, and a second kernel folds over `b` ASCENDING from `+0.0`.
#: Upstream uses five float `gpuAtomicAdd` calls, which is arrival order and
#: therefore not reproducible run to run on ONE device. IDENTITY_PATHS rows 1
#: and 2 refuse exactly this shape.
comptime T5_USES_PRIVATE_SLOTS = True
comptime T5_DIRECTION_IS_ASCENDING_IN_B = True
comptime T5_SEED_IS_POSITIVE_ZERO = True
comptime T5_ATOMIC_REFUSED = True

#: T6, DEVIATION 1075. The three-way join at `du`, in D-skip then scan then
#: x_proj order, three separate flushed adds, left-associated. The order
#: follows the forward's own data flow. **Predicted BITWISE INERT on any
#: fixture where one contribution dominates the other two.**
comptime T6_JOIN_ORDER_D_THEN_SCAN_THEN_XPROJ = True

#: T7, DEVIATION 1076. The RMSNorm backward closed form. `rstd^3` is
#: `(rstd*rstd)*rstd`, left associated, never `portable_powf`. The division by
#: `d_model` is `identical_div` and is hoisted per row, which is loop
#: invariant in `j` and moves no bits. The final subtraction is UNFUSED.
comptime T7_RSTD3_IS_LEFT_ASSOCIATED = True
comptime T7_DIVIDES_BY_DMODEL_ONCE_PER_ROW = True
comptime T7_FINAL_SUBTRACT_IS_UNFUSED = True

#: T8, DEVIATION 1077. The residual join, `dres` on the left, ONE flushed add.
#: **THE ABSORPTION SITE.** This is the backward's `residual.out`, and
#: IDENTITY_PATHS row 55 records that at the shape where a forward sabotage
#: was STRONGEST, thirteen of sixteen stages moved, `out_proj.out` differed on
#: 23 of 64 cells, and `residual.out` was STILL BIT IDENTICAL. An output-only
#: gate called that arm inert. Any backward gate that compares only `bwd.dx`
#: is blind in exactly the same way.
comptime T8_DRES_IS_THE_LEFT_OPERAND = True

#: DEVIATION 1078, not a topology but the one elementwise spelling with an
#: upstream disagreement. `softplus'` is a MULTIPLY by `identical_sigmoid` of
#: the PRE-softplus biased value, not upstream's single division
#: `ddelta / (1 + expf(-biased))` (`selective_scan_bwd_kernel.cuh:454-457`).
#: The guard is the same, `biased <= 20` takes this arm and above it the
#: derivative is exactly `1.0`. The distinguishing band is `biased` in roughly
#: `[8, 14]`; a fixture that only straddles 20 passes this VACUOUSLY, which
#: has already happened once in this lane on `adv_softplus_guard`.
comptime S14B_USES_SIGMOID_MULTIPLY = True
comptime S14B_GUARD_THRESHOLD_IS_20 = True

#: The forward quantities the backward RECOMPUTES rather than stores, each of
#: which must be spelled with the forward's OWN function and not with an
#: algebraically equal one. Upstream breaks this rule once, recomputing
#: `silu(z)` as `z * sigmoid(z)` in its backward where its forward wrote
#: `z / (1 + expf(-z))`, which makes their gradient inconsistent with their
#: own output. `SAB_S12_MUL_SIGMOID` already exists on the forward side and
#: bit `gate.out` on 15 of 64 cells.
comptime RECOMPUTE_USES_THE_FORWARD_SPELLING = True


def mamba_backward_declarations() -> String:
    """Every declared choice, as one printable block a gate can diff.

    A card records values. This records DECISIONS, and it exists because a
    build that silently compiled with a different declaration would otherwise
    produce a correctly labelled measurement of the wrong arm. That is the
    shared-checkout mode-flip lesson applied to a set of comptime constants.
    Print it in every backward gate's banner and diff it across vendors along
    with the card.
    """
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
    s = s + "sabotage: " + mamba_backward_sabotage_name() + "\n"
    return s
