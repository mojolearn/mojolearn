# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The backward pass of a linear layer under `mojolearn.identical.gemm.fp32.v1`.

**THIS FILE CONTAINS NO ARITHMETIC.** Not one multiply, not one add, not one
`ftz`, not one `identical_mul_add`, and no kernel of its own. It is a routing
layer: it names, for each forward orientation, which of the contract's three
operations computes `dA` and which computes `dB`, at which shape, with which
operand on which side, and then hands the work to
`gemm_identical.mojo::identical_gemm_into` unchanged.

That is the whole claim this file exists to make, and it is falsifiable rather
than decorative. For `C = A . B`,

    dA = dC . B^T
    dB = A^T . dC

and both of those are the operation the forward already implements, at a
different transpose. If a multiply or an add ever appears below, the claim is
false and `gemm/IDENTICAL_BACKWARD_PLAN.md` has to say so.

**THE CONTRACT IS NOT AMENDED AND MAY NOT BE.**
`gemm/IDENTICAL_FP32_CONTRACT.md` is frozen; this file CONSUMES it. Nothing
here changes the leaf rule, the fold topology, the multiply-add policy, the
flush policy or either profile constant, so nothing here is a `...fp32.v2`.
If a backward requirement ever needs one of those to move, that is a new
profile with a new name and a new certificate, announced loudly, never a
quiet edit here.

THE ROUTING TABLE (contract section 0.1, worked in the plan document)
---------------------------------------------------------------------
Row-major throughout, no leading dimension, no sub-view (contract section 2).
`dC` always has `C`'s shape, `m x n`. `dA` always has `A`'s shape and `dB`
always has `B`'s shape, whatever that shape is for the forward op.

    forward      A         B         C       dA                       dB
    OP_NN        m x k     k x n     m x n   OP_NT(dC, B) @ (m, k, n)  OP_TN(A, dC) @ (k, n, m)
    OP_NT        m x k     n x k     m x n   OP_NN(dC, B) @ (m, k, n)  OP_TN(dC, A) @ (n, k, m)
    OP_TN        k x m     k x n     m x n   OP_NT(B, dC) @ (k, m, n)  OP_NN(A, dC) @ (k, n, m)

`@ (m', n', k')` is the shape the backward call is made AT, in the contract's
own `(m, n, k)` vocabulary. Every operand is passed EXACTLY AS IT IS STORED:
not one of the six calls materializes a transpose, which is the point of
having three orientations in the first place (contract section 0.2 counts the
transposes a native `OP_TN` deletes from the forward paths, and this is the
same saving on the backward ones).

Two of the six put `dC` on the RIGHT rather than the left, so operand order is
part of the table and not a convention that can be assumed. That is what
`BWD_DC_LEFT` / `BWD_DC_RIGHT` below carry, and what `SAB_BWD_OPERAND_ORDER`
exists to break.

THE BIAS GRADIENT IS A GEMM, AND THAT IS THE FINDING
-----------------------------------------------------
`db[j] = sum over i of dC[i, j]` is a reduction, not a matrix product, and a
reduction is order dependent, so the obvious reading is that it needs a new
pinned fold of its own. It does not, because

    db[1 x n] = ones[1 x m] . dC[m x n]

is `OP_NN` at `(m', n', k') = (1, n, m)`, which is this contract's arithmetic
already: the leaf loop runs `fma(ftz(1.0), ftz(dC[p, j]), acc)`, and
`fma(1, x, acc)` is ONE rounding of `x + acc` because the product `1 * x` is
exact. So the ones vector turns the reduction into the contract's own
ascending flushed chain inside a leaf and the contract's own balanced tree
across leaves, with no second fold shape anywhere in the tree and no second
thing to certify.

The alternative was `core/pinned_reduce.mojo::pinned_block_sum`, and it is
**NOT directly reusable**, for a reason the contract states about itself:
that fold pairs by STRIDE (`red[t] += red[t + step]`), which contract 7.2
clause 1 names as a DIFFERENT ANSWER from v1's adjacent pairing and which
`gemm_identical.mojo::SAB_FOLD_STRIDE` exists to sabotage. It also folds only
WITHIN one block, so a reduction over more rows than a block has threads needs
a second stage it does not supply, and it does not flush its partials per step
(IDENTITY_PATHS names that as an open residue of that file). Using it here
would put `db` under a different profile from `dA` and `dB`. DEVIATION 851 is
the decision not to.

The cost is real and is stated rather than hidden: `m * n` multiplications by
1.0 that a hand-written reduction would not perform, and `m` floats of ones.
It buys one arithmetic instead of two.

WHAT THIS FILE DOES NOT BUY
----------------------------
Identical gradients for a linear layer's matmul. **Not an identical training
run.** `IDENTICAL_GEMM_PLAN.md` makes the same distinction for the forward
pass ("it gets you deterministic linear layers, NOT deterministic models") and
every word of it applies here with a longer list attached. The list is
`gemm/IDENTICAL_BACKWARD_PLAN.md` section 4, and the item a reader should not
miss is that **`dB`'s `k` is the token count**, so the weight gradient's bits
are a function of how many tokens were in the call. That is the contract
working as specified, and it is also the reason a microbatched step is not
bit equal to an unsplit one.

DEVIATIONS 850 (the routing table as a pure function, and the six calls),
851 (the bias gradient spelled as a v1 `OP_NN` against a ones vector rather
than as a new pinned reduction), 852 (the asynchronous caller-owned-workspace
launchers and their sizing helpers).

`[[mojo-buffer-freed-at-last-use]]`: every launcher here is the `_into` form.
It enqueues and returns. The CALLER owns every buffer, including the ones
vector, and must keep every one of them alive past its own
`ctx.synchronize()`. A buffer created in the caller's frame is dead at its
`.unsafe_ptr()`, so a training step that allocates a workspace, calls one of
these, and returns without waiting has freed the workspace before the kernel
ran. `identical_gemm`'s docstring is the long form of this and the reason the
synchronizing form exists at all; there is deliberately no synchronizing form
here, because a training step chains many of these and one wait per matmul is
the wrong shape.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined

from gemm.original.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.original.gemm_oracle import OP_NN, OP_NT, OP_TN, op_name


# ===========================================================================
# THE SABOTAGE SWITCHES
# ===========================================================================
# OFF in every build that does not name them, exactly as
# `gemm_identical.mojo`'s six are. Each one is a specific way to get the
# ROUTING wrong that a plausible implementation could reach by accident, and
# each exists so the gates named in `IDENTICAL_BACKWARD_PLAN.md` section 5
# can be SHOWN to fail. A gate that has never failed is a gate nobody has
# tested.
#
# None of them touches the arithmetic, because this file has none. They move
# WHICH product is computed, which is the only thing here that can be wrong.
#
# Build with, e.g.:
#     tools/with_identical_mode.sh pixi run mojo run \
#         -D MOJOLEARN_GEMM_SABOTAGE_BWD_UNTRANSPOSED=1 \
#         -I . gemm/original/gemm_backward_check.mojo
#
#: EVERY backward call is routed as `OP_NN`, i.e. the transpose bookkeeping
#: is dropped and `dA = dC . B` / `dB = A . dC` are computed as written. This
#: is the mistake the whole file exists to prevent, and at a square shape with
#: a symmetric fixture it is the mistake most likely to survive a careless
#: gate: it computes a perfectly plausible matrix of the right size. Only a
#: FINITE-DIFFERENCE check or a non-square shape separates it.
comptime SAB_BWD_UNTRANSPOSED = is_defined[
    "MOJOLEARN_GEMM_SABOTAGE_BWD_UNTRANSPOSED"
]()
#: The `dC` side flag is IGNORED and `dC` is always passed as the left
#: operand. Correct for three of the six calls and wrong for the other three
#: (`OP_TN`'s `dA`, and `dB` for forward `OP_NN` and `OP_TN`), so a gate that
#: exercises only forward `OP_NN`'s `dA` cannot see it.
comptime SAB_BWD_OPERAND_ORDER = is_defined[
    "MOJOLEARN_GEMM_SABOTAGE_BWD_OPERAND_ORDER"
]()
#: The bias gradient reduces the WRONG AXIS: `OP_NT(dC, ones)` at `(m, 1, n)`,
#: which is `sum over j` rather than `sum over i`. At `m == n` it produces a
#: vector of the right length and the wrong contents, so the gate for it must
#: run at `m != n` and must compare VALUES rather than shapes.
comptime SAB_BWD_BIAS_AXIS = is_defined[
    "MOJOLEARN_GEMM_SABOTAGE_BWD_BIAS_AXIS"
]()

comptime ANY_BWD_SABOTAGE = (
    SAB_BWD_UNTRANSPOSED or SAB_BWD_OPERAND_ORDER or SAB_BWD_BIAS_AXIS
)


def gemm_backward_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for a check's banner.

    The twin of `gemm_identical.mojo::gemm_sabotage_name`. A check must print
    BOTH, because a backward binary can carry a forward sabotage as well and a
    banner that names only one of them mislabels the run.
    """
    comptime if SAB_BWD_UNTRANSPOSED:
        return String("BWD_UNTRANSPOSED")
    comptime if SAB_BWD_OPERAND_ORDER:
        return String("BWD_OPERAND_ORDER")
    comptime if SAB_BWD_BIAS_AXIS:
        return String("BWD_BIAS_AXIS")
    return String("none")


# ===========================================================================
# THE ROUTING TABLE'S ONE DOOR
# ===========================================================================
# `gemm_identical.mojo` makes "launch geometry cannot reach the arithmetic" a
# property of the call graph by giving `(L, P)` exactly one producer. The same
# discipline applies here to a different quantity: THE BACKWARD SHAPE HAS
# EXACTLY TWO PRODUCERS, `gemm_backward_a_call` and `gemm_backward_b_call`,
# and every launcher and every sizing helper below calls one of them. To
# change which product a backward pass computes you would have to edit one of
# two functions, and both are pure, host-side and testable with no device.

#: `dC` is the LEFT operand of the backward call.
comptime BWD_DC_LEFT = 0
#: `dC` is the RIGHT operand of the backward call.
comptime BWD_DC_RIGHT = 1


def gemm_backward_a_call(
    op: Int, m: Int, n: Int, k: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for `dA`, given the FORWARD `(op, m, n, k)`.

    The first row of the module docstring's table, as a pure function. `op'`
    is one of `OP_NN` / `OP_NT` / `OP_TN`; `(m', n', k')` is the shape the
    call is made at; `dc_side` says whether `dC` goes in the left or the right
    operand slot.

        OP_NN  ->  OP_NT(dC, B) at (m, k, n),  dC LEFT
        OP_NT  ->  OP_NN(dC, B) at (m, k, n),  dC LEFT
        OP_TN  ->  OP_NT(B, dC) at (k, m, n),  dC RIGHT

    **`dA` always has `A`'s shape**, which is `m x k` for `OP_NN` and `OP_NT`
    and `k x m` for `OP_TN`, and each row above returns exactly that as
    `(m', n')`. The derivation and the stride-level check against contract
    section 3 are in `IDENTICAL_BACKWARD_PLAN.md` section 2; the assertion
    that the derivation is right is a GATE, not this docstring.

    **`k'` is `n` in all three rows**, i.e. the contraction of the `dA`
    product is over the OUTPUT width. For a linear layer that is
    `out_features`, so `dA`'s partition is the same order of magnitude as the
    forward's and nothing new happens to the leaf rule. `dB` is where that
    stops being true; see `gemm_backward_b_call`.

    Pure: it reads `op`, `m`, `n`, `k` and nothing else, so the backward call
    at any shape gets a deterministic plan for the same reason the forward
    does (contract 6.1, `choose_gemm_plan`'s own docstring).

    DEVIATION 850, contract sections 0.1, 2 and 3.
    """
    comptime if SAB_BWD_UNTRANSPOSED:
        # SABOTAGE: the transpose bookkeeping dropped. `dA = dC . B` as
        # written, at the shape that happens to typecheck.
        return (OP_NN, m, k, n, BWD_DC_LEFT)
    var bop = OP_NT
    var bm = m
    var bn = k
    var side = BWD_DC_LEFT
    if op == OP_NT:
        bop = OP_NN
    elif op == OP_TN:
        bm = k
        bn = m
        side = BWD_DC_RIGHT
    comptime if SAB_BWD_OPERAND_ORDER:
        # SABOTAGE: the side flag ignored, `dC` always left.
        side = BWD_DC_LEFT
    return (bop, bm, bn, n, side)


def gemm_backward_b_call(
    op: Int, m: Int, n: Int, k: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """`(op', m', n', k', dc_side)` for `dB`, given the FORWARD `(op, m, n, k)`.

    The second row of the module docstring's table, as a pure function.

        OP_NN  ->  OP_TN(A, dC) at (k, n, m),  dC RIGHT
        OP_NT  ->  OP_TN(dC, A) at (n, k, m),  dC LEFT
        OP_TN  ->  OP_NN(A, dC) at (k, n, m),  dC RIGHT

    **`dB` always has `B`'s shape**, which is `k x n` for `OP_NN` and `OP_TN`
    and `n x k` for `OP_NT`, and each row returns exactly that as `(m', n')`.

    **`k'` IS `m` IN ALL THREE ROWS, AND THAT IS THE FINDING OF THIS LANE.**
    `m` is the forward's batch dimension, so the weight gradient contracts
    over the tokens. Contract 6.1 forbids the leaf size from depending on `m`
    or `n` precisely because `m` is a batch dimension and the FORWARD must be
    batch invariant; here the batch dimension arrives as `k`, where section 6
    REQUIRES the leaf size to depend on it. Both statements are the contract
    working correctly and together they say:

      - `dB` is bit identical across vendors, launches and plans at a fixed
        token count. That is what this profile promises and it is unaffected.
      - `dB` at 1024 tokens is NOT the same bits as `dB` at 512 tokens
        accumulated twice, and cannot be under any fixed partition, because
        the two are different sums with different partitions. The microbatch
        schedule is therefore part of a training run's numerical
        specification, not an execution detail.

    It also means `k'` is unbounded in a way a forward `k` never is: a forward
    `k` is a layer width, and this one is however many tokens the caller
    passed. `contract_leaf_size` holds `L` at `K_LEAF_MIN = 128` only while
    `k <= 131072`; above that `L = ceil(k / 1024)` and the serial ascending
    chain inside one leaf grows without bound. The forward gate sweep reaches
    `k = 4,000,000` (`gemm_device_check.mojo::check_device_matches_oracle`,
    case `k4M.P1024.cap`), so a token count up to four million is inside the
    range the partition has actually been exercised at. Past that it is not,
    and a gate has to say so rather than a docstring.

    DEVIATION 850, contract sections 0.1, 2, 3, 6 and 6.1.
    """
    comptime if SAB_BWD_UNTRANSPOSED:
        # SABOTAGE: `dB = A . dC` as written.
        return (OP_NN, k, n, m, BWD_DC_RIGHT)
    var bop = OP_TN
    var bm = k
    var bn = n
    var side = BWD_DC_RIGHT
    if op == OP_NT:
        bm = n
        bn = k
        side = BWD_DC_LEFT
    elif op == OP_TN:
        bop = OP_NN
    comptime if SAB_BWD_OPERAND_ORDER:
        side = BWD_DC_LEFT
    return (bop, bm, bn, m, side)


def gemm_backward_call_name(call: Tuple[Int, Int, Int, Int, Int]) -> String:
    """`"NT(dC,W) 64x32x128"`, for a gate's per-case banner.

    `W` is the OTHER forward operand, which is `B` for a `dA` call and `A` for
    a `dB` call. This function cannot tell which, and deliberately does not
    try: the caller knows and the gate prints it.
    """
    var left = String("dC")
    var right = String("W")
    if call[4] == BWD_DC_RIGHT:
        left = String("W")
        right = String("dC")
    return (
        op_name(call[0])
        + "("
        + left
        + ","
        + right
        + ") "
        + String(call[1])
        + "x"
        + String(call[2])
        + "x"
        + String(call[3])
    )


# ===========================================================================
# WORKSPACE SIZING (contract 13.5, and `identical_gemm_into`'s hazard)
# ===========================================================================


def identical_gemm_backward_a_workspace_max_floats(
    op: Int, m: Int, n: Int, k: Int
) -> Int:
    """Floats of scratch the `dA` call may need, at the FORWARD shape.

    **Sizing a workspace from the forward shape is an out-of-bounds write.**
    The backward call is made at a different `(m, n, k)`, so
    `choose_gemm_plan` may pick a different plan, and a SPLITK plan's
    workspace is `m' * n' * P(k')` where all three of those are the backward
    numbers. `identical_gemm_into`'s docstring records what that mistake cost
    this lane on the forward side: a 1-float workspace passed to a SPLITK
    dispatch at `64 x 4` still returned the right answer because the
    allocation had slack, and only at `64 x 64` did whole regions of the
    output come back `+0.0`. Use this helper. Do not guess, and do not reuse
    the forward number.
    """
    var call = gemm_backward_a_call(op, m, n, k)
    return identical_gemm_workspace_max_floats(call[1], call[2], call[3])


def identical_gemm_backward_b_workspace_max_floats(
    op: Int, m: Int, n: Int, k: Int
) -> Int:
    """Floats of scratch the `dB` call may need, at the FORWARD shape.

    Same hazard, same instruction as the `dA` helper. This one is the more
    likely to surprise, because `dB`'s `k'` is the token count and a plan
    chosen at one batch size is not the plan chosen at another.
    """
    var call = gemm_backward_b_call(op, m, n, k)
    return identical_gemm_workspace_max_floats(call[1], call[2], call[3])


def identical_gemm_backward_bias_workspace_max_floats(m: Int, n: Int) -> Int:
    """Floats of scratch the bias-gradient call may need, given `dC`'s shape.

    The call is `OP_NN` at `(1, n, m)`. `m * n <= 4096` with `P >= 4` is the
    SPLITK condition (`choose_gemm_plan`), and `m' * n'` here is `n`, so the
    bias reduction lands on the SPLITK arm at any `n <= 4096` with more than
    three leaves, which is most training shapes. Its workspace is `n * P(m)`,
    up to `n * 1024` floats. That is not large, and it is not zero either, so
    it must be allocated.
    """
    return identical_gemm_workspace_max_floats(1, n, m)


def identical_gemm_backward_workspace_max_floats(
    op: Int, m: Int, n: Int, k: Int, with_bias: Bool
) -> Int:
    """One number a training step can allocate ONCE and reuse for all of
    `dA`, `dB` and (optionally) `db` at this forward shape.

    Sharing one workspace across the three calls is safe because they are
    enqueued on ONE `DeviceContext` and MAX runs them in order, so the second
    call's leaf kernel cannot start before the first call's fold kernel has
    finished reading. The forward gate already asserts the stronger form of
    this: `check_device_is_batch_invariant` runs four GEMMs through one DIRTY
    shared workspace and requires the same bits. **It is not safe across two
    contexts or two streams**, and if this lane ever grows a second stream
    that is a change to this function's contract and not to its arithmetic.
    """
    var w = identical_gemm_backward_a_workspace_max_floats(op, m, n, k)
    var wb = identical_gemm_backward_b_workspace_max_floats(op, m, n, k)
    if wb > w:
        w = wb
    if with_bias:
        var wc = identical_gemm_backward_bias_workspace_max_floats(m, n)
        if wc > w:
            w = wc
    if w < 1:
        return 1
    return w


def identical_gemm_backward_bias_ones_floats(m: Int) -> Int:
    """How long the ones vector is: `m`, the row count of `dC`.

    The caller allocates it, fills it with EXACTLY `Float32(1.0)` in every
    entry, and keeps it alive past its own synchronize. It is a constant of
    the shape, not per step, so a training loop allocates it once per batch
    size and never touches it again.

    **A wrong value here is a wrong gradient with no symptom**, because any
    vector produces a plausible weighted column sum. The gate for it is not
    "the buffer was allocated", it is the finite-difference check of
    `IDENTICAL_BACKWARD_PLAN.md` section 5 gate B1, which fails if the ones
    are anything else.
    """
    if m < 1:
        return 1
    return m


# ===========================================================================
# THE LAUNCHERS
# ===========================================================================
# All three are the `_into` form: caller-owned buffers, caller-owned
# workspace, ASYNCHRONOUS, nothing waits. A training step enqueues `dA`,
# `dB` and `db` back to back and synchronizes once, which is the shape the
# whole file exists for.


def identical_gemm_backward_a_into(
    ctx: DeviceContext,
    mut da: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """`dA` for the forward `C = op(A) . op(B)` at `(m, n, k)`, into `da`.

    `dc` is `dC`, `m x n` row-major, the gradient arriving from downstream.
    `b` is the FORWARD's `B` operand, unchanged and untransposed. `da`
    receives `A`'s shape: `m x k` for `OP_NN` and `OP_NT`, `k x m` for
    `OP_TN`. `ws` must hold at least
    `identical_gemm_backward_a_workspace_max_floats(op, m, n, k)` floats.

    Everything numerical happens inside `identical_gemm_into`, which is the
    certified entry point and is not modified. This function chooses the op,
    the shape and the operand order and then gets out of the way, so the bits
    of `dA` are the bits of `mojolearn.identical.gemm.fp32.v1` at the shape
    the table names. There is no second arithmetic to certify and no second
    profile.

    **THE DEGENERATE SHAPES FALL OUT AND ARE NOT SPECIAL CASED**, which is
    worth stating because a special case here would be a place to get contract
    section 8 wrong:

      - forward `n == 0` gives `k' == 0`, and section 8 requires every cell of
        the output to be a STORED `+0.0`. `dA` is correctly all zeros: with no
        outputs there is no gradient. `identical_gemm_with_plan` stores it
        rather than skipping the store.
      - forward `k == 0` gives `n' == 0` for `OP_NN` / `OP_NT`, an empty `dA`,
        and nothing is written. Correct: `A` is empty.
      - forward `m == 0` gives `m' == 0` for `OP_NN` / `OP_NT`, an empty `dA`.
        Correct.

    ASYNCHRONOUS. `[[mojo-buffer-freed-at-last-use]]`: `da`, `dc`, `b` and
    `ws` are the CALLER'S and every one of them must outlive the caller's
    `ctx.synchronize()`.

    DEVIATIONS 850 and 852.
    """
    var call = gemm_backward_a_call(op, m, n, k)
    if call[4] == BWD_DC_LEFT:
        identical_gemm_into(
            ctx, da, dc, b, ws, call[1], call[2], call[3], call[0]
        )
        return
    identical_gemm_into(ctx, da, b, dc, ws, call[1], call[2], call[3], call[0])


def identical_gemm_backward_b_into(
    ctx: DeviceContext,
    mut db: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """`dB` for the forward `C = op(A) . op(B)` at `(m, n, k)`, into `db`.

    `dc` is `dC`, `m x n`. `a` is the FORWARD's `A` operand, unchanged and
    untransposed. `db` receives `B`'s shape: `k x n` for `OP_NN` and `OP_TN`,
    `n x k` for `OP_NT`. `ws` must hold at least
    `identical_gemm_backward_b_workspace_max_floats(op, m, n, k)` floats.

    **This is the call whose `k` is the token count** (see
    `gemm_backward_b_call`), so it is the call whose plan, whose workspace and
    whose leaf size all move when the batch size moves. Nothing about that
    threatens cross-vendor identity, and everything about it means a training
    run has to declare its batch and microbatch schedule as part of its
    numerical specification.

    Degenerate shapes, again falling out rather than special cased:

      - forward `m == 0` gives `k' == 0`, so every cell of `dB` is a stored
        `+0.0`. Correct: no tokens, no weight gradient. Section 8 requires the
        value to be WRITTEN, and it is.
      - forward `n == 0` gives an empty `dB` for `OP_NN` / `OP_TN`; forward
        `k == 0` gives an empty `dB` for `OP_NN` / `OP_TN`.

    `dA` and `dB` write DISJOINT buffers and read the same `dC`, so the two
    calls are independent and need no barrier between them. They may share one
    `ws` on one context; see
    `identical_gemm_backward_workspace_max_floats`.

    ASYNCHRONOUS, caller-owned buffers. DEVIATIONS 850 and 852.
    """
    var call = gemm_backward_b_call(op, m, n, k)
    if call[4] == BWD_DC_LEFT:
        identical_gemm_into(
            ctx, db, dc, a, ws, call[1], call[2], call[3], call[0]
        )
        return
    identical_gemm_into(ctx, db, a, dc, ws, call[1], call[2], call[3], call[0])


def identical_gemm_backward_bias_into(
    ctx: DeviceContext,
    mut dbias: DeviceBuffer[DType.float32],
    mut dc: DeviceBuffer[DType.float32],
    mut ones: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
) raises:
    """`dbias[j] = sum over i of dC[i, j]`, as an `OP_NN` GEMM at `(1, n, m)`.

    `dc` is `dC`, `m x n` row-major. `ones` holds `m` entries of exactly
    `Float32(1.0)` (`identical_gemm_backward_bias_ones_floats`). `dbias`
    receives `n` floats. `ws` must hold at least
    `identical_gemm_backward_bias_workspace_max_floats(m, n)`.

    **A REDUCTION ROUTED THROUGH THE PRODUCT, DELIBERATELY** (DEVIATION 851).
    The module docstring carries the argument; the short form is that
    `fma(1, x, acc)` is one rounding of `x + acc`, so the ones vector makes
    this the contract's ascending flushed leaf chain and the contract's
    balanced tree, and `db` therefore lands under the SAME profile and the
    SAME certificate as `dA` and `dB` instead of under a second fold shape
    that would need its own clause, its own fixtures and its own sabotages.
    `core/pinned_reduce.mojo::pinned_block_sum` is not that shape: it pairs by
    stride, which contract 7.2 clause 1 names as a different answer.

    This reduces `dC`'s ROW axis, which is the linear-layer bias case
    (one bias per output column). A caller who wants the column axis wants
    `identical_gemm_into(ctx, out, dc, ones, ws, m, 1, n, OP_NT)`, which is
    the same trick at the other transpose and is deliberately not wrapped
    here, because this lane has no caller for it and an unused wrapper is an
    ungated one.

      - `m == 0` gives `k' == 0`, so every entry of `dbias` is a stored
        `+0.0`. Correct, and section 8 requires the store.
      - `n == 0` gives an empty output and nothing is written.

    ASYNCHRONOUS, caller-owned buffers, INCLUDING `ones`. DEVIATIONS 851 and
    852.
    """
    comptime if SAB_BWD_BIAS_AXIS:
        # SABOTAGE: reduce the COLUMN axis instead. Right length at `m == n`,
        # wrong contents everywhere.
        identical_gemm_into(ctx, dbias, dc, ones, ws, m, 1, n, OP_NT)
        return
    identical_gemm_into(ctx, dbias, ones, dc, ws, 1, n, m, OP_NN)
