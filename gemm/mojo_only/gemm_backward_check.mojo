# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The backward pass's gates: routing, correctness, identity, and the exact
number of cells each sabotage moves.

**WRITTEN AND NEVER RUN.** Not one line below has been compiled, and no
device has executed a backward call. `gemm/IDENTICAL_BACKWARD_PLAN.md`
section 8 says the same about `gemm_backward.mojo`, and this file inherits it
whole: every count in the ledger table below is a PREDICTION derived on
paper, and the point of writing the predictions down before the run is that a
disagreement is then a FINDING rather than something to rationalize after the
fact. When this has run, the ledger becomes MEASURED and this paragraph is
the sentence to delete.

WHAT THIS FILE TURNS FROM AN ARGUMENT INTO A NUMBER
----------------------------------------------------
`gemm/mojo_only/gemm_backward.mojo` is a routing layer that is argued. It
claims, in prose, that `dA` and `dB` are the forward operation at a different
transpose, that the bias gradient is an `OP_NN` GEMM against a ones vector,
and that no arithmetic of its own appears anywhere in it. Three claims, none
measured. This file measures them, and it measures a fourth thing the plan
document only asserts: that `dB`'s bits are a function of the token count.

    G1  the routing table IS the table                    HOST
    G2  the six routes compute the RIGHT derivative       HOST
    G3  the device output IS `gemm_oracle` at the
        backward shape, per cell, bitwise                 DEVICE
    G4  the backward shapes are LAUNCH INVARIANT          DEVICE
    G5  `dB` is not batch invariant, `dA` is              DEVICE
    G6  the bias gradient is the contract's own sum       DEVICE
    G7  the workspace helpers size what actually runs     HOST + DEVICE
    G8  the token sweep, out to four million              DEVICE
    G9  the identity CARD, for the vendor diff            DEVICE, env gated

BIT EQUALITY IS THE ONLY STANDARD HERE. There is no tolerance anywhere in
this file, and there is no epsilon to tune, because both of the questions
being asked have exact answers. "Is this the right derivative" is answered on
a fixture whose every product and every partial sum is exactly representable
in Float32, so every summation order gives the same bits and a plain host sum
is a bit-exact reference. "Is this the same everywhere" is answered against
`gemm_oracle`, which is the profile's definition. A gate that compared to a
tolerance would answer neither.

THE TWO QUESTIONS ARE DIFFERENT AND BOTH ARE NEEDED
-----------------------------------------------------
G3 says the answer is the same on every vendor. It does NOT say the answer is
the right derivative: **a transpose error is bit identical on three vendors**,
and it is the single most likely defect in a routing layer, because every
operand read as its transpose has the same element count and nothing raises.
G2 is the gate for that and it is host only, so it runs with no GPU present.
G3 without G2 certifies a wrong gradient. G2 without G3 certifies a right
gradient that three vendors compute differently. `IDENTICAL_BACKWARD_PLAN.md`
section 5's G2 header makes this argument and this file is where it is spent.

THE SABOTAGE LEDGER, PREDICTED 2026-08-25, NOT MEASURED
--------------------------------------------------------
Six routes, indexed the way `_route_name` indexes them:

    0  dA / forward OP_NN      3  dB / forward OP_NN
    1  dA / forward OP_NT      4  dB / forward OP_NT
    2  dA / forward OP_TN      5  dB / forward OP_TN

    arm                     routes moved       predicted count
    BWD_UNTRANSPOSED        0, 2, 3, 4         4 of 6   mask 29
    BWD_OPERAND_ORDER       2, 3, 5            3 of 6   mask 44
    BWD_BIAS_AXIS           none               0 of 6   mask 0

**TWO OF THE SIX ROUTES ARE INERT UNDER `BWD_UNTRANSPOSED` AND THAT IS
STRUCTURAL, NOT A DEFECT IN THE SABOTAGE.** The arm routes everything as
`OP_NN`, and for `dA` at forward `OP_NT` and for `dB` at forward `OP_TN` the
CORRECT route already is `OP_NN` at the same shape with the same operand
order. The sabotage is then the right answer, bit for bit, and a gate that
reported "BWD_UNTRANSPOSED fails" without saying WHICH routes it failed on
would be reporting a 4-of-6 result as a 6-of-6 one. This is
`[[reached-but-inert]]` in its purest form and it is why G1 checks the moved
ROUTE SET against a predicted bitmask rather than checking that something
somewhere failed.

`BWD_BIAS_AXIS` moves NOTHING in G1, G2's matmul arms, G3, G4 or G5, and
must move G6. `[[mojotrees-verify-reach-not-output]]`: reach is per branch,
and an arm that fires everywhere is an arm that localizes nothing.

WHAT MAKES EACH CLAUSE NON VACUOUS
------------------------------------
Every gate here carries a NEGATIVE CONTROL, because a gate that would pass
with the thing under test absent is worthless and this repository has shipped
three of those.

    G1   the expected table is written out a SECOND time, by hand, in
         `_want_route`. If it were derived from `gemm_backward_a_call` the
         gate would compare a function to itself.
    G2   the exactness of the fixture is CHECKED, not assumed
         (`_exact_guard`), and the gate REFUSES to pass on a square fixture
         or on one where `m`, `n` and `k` are not pairwise distinct.
    G2   `check_the_square_fixture_is_vacuous` DEMONSTRATES the refusal:
         it runs the untransposed route on a symmetric square fixture in a
         CLEAN BUILD and shows it agrees on every cell.
    G3   POISON. Every output buffer is written with `-987654.0` before every
         launch and a surviving poison raises. A never-written cell that
         happens to compare equal is not evidence.
    G3   the all-`-0.0` leaf fixture, which is the only input in this file
         that can tell a CARRY from `+0.0` padding, and which refuses itself
         if the oracle does not come back `-0.0`.
    G5   the split arms are first run on the EXACT fixture, where they MUST
         agree; if they disagree there, the split is buggy and the rounding
         measurement that follows is meaningless. Then they are run on the
         rounding fixture, where a disagreement is REQUIRED or the gate
         raises VACUOUS.
    G6   arm 2 computes the column sum with NO ONES VECTOR ANYWHERE in it.
         Without that arm G6 only checks that a GEMM is a GEMM.
    G7   the shape is CHOSEN by checking that the forward and backward
         workspace numbers differ, and the gate raises if no shape in the
         list separates them.

WHAT THIS FILE DOES NOT DO, STATED SO NOBODY ASSUMES IT DID
-------------------------------------------------------------
1. **It does not run the destructive under-allocation arm** that
   `IDENTICAL_BACKWARD_PLAN.md` section 5's G7 asks for. Handing a
   deliberately too-small `DeviceBuffer` to a SPLITK dispatch is an
   out-of-bounds DEVICE WRITE with undefined behavior, and the repository's
   own record of it (`identical_gemm_into`'s docstring) is a BUG IT HIT, not
   a test it keeps. G7 instead asserts the sizing RELATION on the host at a
   shape where the two numbers provably differ, and runs the positive arm at
   an exactly-sized poisoned workspace. DEVIATION 1058.
2. **It does not emit the fold-ladder levels.** DEVIATION 533's per-level
   hashes live in `bench/gemm_ladder_main.mojo`, which this lane does not
   own. The backward needs no new ladder instrument: a backward call at
   `(m', n', k')` is the forward at `(m', n', k')`, so the existing ladder
   run at the six backward shapes is the whole of it. That is a shape-list
   change in a file this lane may not edit and it is reported rather than
   made.
3. **It does not test a second `DeviceContext` or a second stream.**
   `identical_gemm_backward_workspace_max_floats`'s safety argument is
   explicitly about ONE context, and this file exercises exactly that.
4. **It does not touch `NaN` payload bits** (contract 9.1), and a gradient is
   a place NaNs actually appear.
5. **It does not run past four million tokens.** G8 is the sweep and it stops
   where the forward's sweep stops, and says so.

MODES. `check_backward_routing_is_the_table`, `check_backward_gradients_
are_correct`, G4, G5's `dA` arm and G7 are ASSERTED IN BOTH MODES: they are
claims about the SHAPE of the routing and about exact arithmetic, and FAST
has no excuse for failing them. G3, G6 and G5's rounding arms are ASSERTED
under IDENTICAL and REPORTED under FAST, for the reason
`gemm_device_check.mojo` gives at the same seam.

`[[mojo-string-float-roundtrip]]`: every float printed here carries its hex
bits beside its decimal.

RUN IT
-------
    tools/with_identical_mode.sh pixi run mojo run \
        -I . gemm/mojo_only/gemm_backward_check.mojo

and one sabotage arm at a time, each of which MUST fail, and must fail on
exactly the routes the ledger above names:

    tools/with_identical_mode.sh pixi run mojo run \
        -D MOJOLEARN_GEMM_SABOTAGE_BWD_UNTRANSPOSED=1 \
        -I . gemm/mojo_only/gemm_backward_check.mojo

The forward file's six (`LEAF_READS_LAUNCH`, `FOLD_STRIDE`, `PAD_PLUS_ZERO`,
`FOLD_SERIAL`, `NODE_ORDER`, `LEAF_ROTATE`) must also be shown to fail
THROUGH this file's device gates, which is the proof that the backward path
reaches the contract's arithmetic rather than some other path that happens to
agree.

DEVIATIONS 1050 (the padded-operand fixture), 1051 (the exact-integer
bilinear fixture and its guard), 1052 (the hand-written second routing table
and the predicted-move bitmask), 1053 (the square-fixture vacuity
demonstration in a clean build), 1054 (the no-ones host reduction), 1055 (the
exact ones poison beside the realistic one), 1056 (the split-alignment
taxonomy), 1057 (accumulation spelled as the contract's own fold), 1058 (G7
without the destructive arm), 1059 (the dirty shared workspace across all
three calls), 1060 (the card emitted from this file), 1061 (the
sabotage-inverting verdict), 1062 (the token sweep's shape), 1063 (the
minus-zero leaf fixture routed through `dB`).
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import bitcast

from core.identity_trace import IdentityTrace
from gemm.mojo_only.gemm_backward import (
    ANY_BWD_SABOTAGE,
    BWD_DC_LEFT,
    BWD_DC_RIGHT,
    SAB_BWD_BIAS_AXIS,
    SAB_BWD_OPERAND_ORDER,
    SAB_BWD_UNTRANSPOSED,
    gemm_backward_a_call,
    gemm_backward_b_call,
    gemm_backward_call_name,
    gemm_backward_sabotage_name,
    identical_gemm_backward_a_into,
    identical_gemm_backward_a_workspace_max_floats,
    identical_gemm_backward_b_into,
    identical_gemm_backward_b_workspace_max_floats,
    identical_gemm_backward_bias_into,
    identical_gemm_backward_bias_ones_floats,
    identical_gemm_backward_bias_workspace_max_floats,
    identical_gemm_backward_workspace_max_floats,
)
from gemm.mojo_only.gemm_identical import (
    GEMM_PLAN_COUNT,
    PLAN_FLAT,
    choose_gemm_plan,
    contract_partition,
    gemm_plan_name,
    gemm_sabotage_name,
    identical_gemm_with_plan,
    identical_gemm_workspace_floats,
    identical_gemm_workspace_max_floats,
)
from gemm.mojo_only.gemm_oracle import (
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_count,
    contract_leaf_size,
    fold_balanced_tree,
    gemm_oracle,
    leaf_begin,
    leaf_end,
    op_name,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, numeric_mode_name

comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

#: Written into every output buffer before every launch. A cell still holding
#: it afterwards was NEVER WRITTEN. Contract section 8 makes this load
#: bearing at `k == 0`, where the required `+0.0` must be STORED rather than
#: the store skipped, and it is load bearing again under `BWD_BIAS_AXIS`,
#: which writes `m` entries where `n` are expected.
comptime POISON = Float32(-987654.0)

#: The largest magnitude the exact-integer fixture may reach before Float32
#: stops representing every integer. `2^24`.
comptime EXACT_LIMIT = Float32(16777216.0)


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _show(x: Float32) -> String:
    return String(x) + "/" + hex(_bits(x))


# ===========================================================================
# FIXTURE VALUES
# ===========================================================================
# Three generators, three jobs. None of them is uniform, because
# `[[uniform-test-data-hides-permutation]]`: if every element is the same
# number a transposed read gives the right answer and the whole file is
# vacuous. Every value below is a function of its INDEX, so a cell read from
# the wrong place is a different number.


def _hash64(i: Int, salt: Int) -> UInt64:
    """splitmix64. `[[mojo-amp-plus-is-bitwise-and]]`: the additions here are
    `+` and not `&+`, deliberately, because `&+` is BITWISE AND in Mojo and
    has produced silently wrong hashes in this repository."""
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(
        0xBF58476D1CE4E5B9
    )
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _val(i: Int, salt: Int) -> Float32:
    """THE ROUNDING FIXTURE. A signed value whose significand is 13 BITS and
    whose exponent spreads over eight binades.

    Character for character `gemm_device_check.mojo::_val`, on purpose: the
    forward gate's fixture and the backward gate's fixture must be the same
    generator or a divergence between the two gates is a divergence between
    two fixtures. Two 13-bit significands need 26 bits to multiply exactly
    and Float32 keeps 24, so EVERY product here is inexact, which is the
    requirement: a wrong leaf boundary or a wrong pairing has to move bits
    rather than cancel out.
    """
    var h = _hash64(i, salt)
    var mant = Float32(1.0) + Float32(Int(h & UInt64(0xFFF))) / Float32(4096.0)
    var e = Int((h >> UInt64(13)) & UInt64(7)) - 4
    var scale = Float32(1.0)
    if e >= 0:
        for _ in range(e):
            scale = scale * Float32(2.0)
    else:
        for _ in range(-e):
            scale = scale * Float32(0.5)
    var v = mant * scale
    if (h >> UInt64(20)) & UInt64(1) == UInt64(1):
        return -v
    return v


def _ival(i: Int, salt: Int) -> Float32:
    """THE EXACT FIXTURE. A signed integer in `[-8, 8]`, never zero.

    **This is what makes G2 a bitwise gate with no epsilon in it.** Products
    of two such values are integers in `[-64, 64]` and are exactly
    representable; a sum of `k'` of them is an integer of magnitude at most
    `64 k'`, exactly representable while that stays under `2^24`. So on this
    fixture EVERY summation order gives the SAME BITS: the contract's
    partitioned tree, a plain ascending host loop, and any vendor's library
    all agree, and the only thing that can move a bit is reading the wrong
    element. That is precisely the defect G2 exists to find.

    NEVER ZERO, and that is not cosmetic. A zero operand makes a whole term
    vanish, and a term that vanishes cannot distinguish a right index from a
    wrong one. `_exact_guard` checks the bound; this clause checks the
    fixture has no free passes in it.
    """
    var h = _hash64(i, salt + 7717)
    var mag = Int(h & UInt64(7)) + 1
    var v = Float32(mag)
    if (h >> UInt64(11)) & UInt64(1) == UInt64(1):
        return -v
    return v


def _sym_val(i: Int, j: Int, salt: Int) -> Float32:
    """A SYMMETRIC value: `f(i, j) == f(j, i)`.

    Used by exactly one function, `check_the_square_fixture_is_vacuous`, and
    used there to build the fixture on which a transpose error is BITWISE
    INVISIBLE. It is in this file so that the reason G2's shape constraint
    exists can be demonstrated rather than asserted.
    """
    var lo = i
    var hi = j
    if j < i:
        lo = j
        hi = i
    return _val(lo * 7919 + hi, salt)


def _fill(n_elems: Int, salt: Int) -> List[Float32]:
    var v = List[Float32]()
    for i in range(n_elems):
        v.append(_val(i, salt))
    return v^


def _fill_exact(n_elems: Int, salt: Int) -> List[Float32]:
    var v = List[Float32]()
    for i in range(n_elems):
        v.append(_ival(i, salt))
    return v^


def _fill_const(n_elems: Int, c: Float32) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n_elems):
        v.append(c)
    return v^


def _pad_to(src: List[Float32], want: Int) -> List[Float32]:
    """**DEVIATION 1050.** Grow an operand to `want` floats, filling the tail
    with hashed values.

    THE REASON IS THE SABOTAGE BUILDS AND IT IS NOT OPTIONAL. A sabotaged
    route is made at a DIFFERENT SHAPE from the correct one, so it reads a
    different number of floats out of the same operand. `BWD_OPERAND_ORDER`
    at forward `OP_TN` routes `dA` as `OP_NT(dC, B)` at `(k, m, n)` with `dC`
    forced into the LEFT slot, and the left slot of that call reads `k * n`
    floats while `dC` holds `m * n`. At `k > m` that is a read past the end
    of the buffer.

    An out-of-bounds device read is undefined behavior, so a sabotage arm
    built on one is not a repeatable ledger entry: it might crash, it might
    return the right answer, and it might return a different wrong answer on
    each vendor. Padding every operand to cover the widest route this build
    can make turns it into a DEFINED wrong answer that is the same on every
    vendor and the same on every run. The padding is HASHED and not zero,
    because zeros would make the extra terms vanish and could turn a wrong
    read back into the right number.

    The comparison never looks at the padding. Only the logical extent of the
    OUTPUT is compared, and the oracle is evaluated on the SAME padded lists,
    so the reference and the device see byte-identical inputs.

    **NO SALT PARAMETER, DELIBERATELY.** The pad value is a pure function of
    the ABSOLUTE INDEX, so two independent pads of the same operand to the
    same length are byte identical. If it took a salt, the device runner and
    the host oracle would pad the same buffer with two different tails, and
    on a sabotaged route that reads into the tail they would disagree for a
    reason that has nothing to do with the kernel. That is a gate reporting
    the opposite of the truth and it was two characters away.
    """
    var v = src.copy()
    var i = len(v)
    while i < want:
        v.append(_val(i * 131 + 17, 4242))
        i += 1
    return v^


def _max2(a: Int, b: Int) -> Int:
    if a > b:
        return a
    return b


# ===========================================================================
# SHAPES, WRITTEN OUT INDEPENDENTLY OF THE ROUTING TABLE
# ===========================================================================
# `gemm_backward.mojo` claims `dA` has `A`'s shape and `dB` has `B`'s shape.
# These four functions are where `A`'s shape and `B`'s shape are computed
# from the FORWARD op alone, with the routing table nowhere in scope, so that
# G1 can compare the router's `(m', n')` against them. Deriving them from the
# router would make G1 compare a function to itself.


def _a_rows(op: Int, m: Int, k: Int) -> Int:
    """`A` is `m x k` for `OP_NN` and `OP_NT`, `k x m` for `OP_TN`."""
    if op == OP_TN:
        return k
    return m


def _a_cols(op: Int, m: Int, k: Int) -> Int:
    if op == OP_TN:
        return m
    return k


def _b_rows(op: Int, n: Int, k: Int) -> Int:
    """`B` is `k x n` for `OP_NN` and `OP_TN`, `n x k` for `OP_NT`."""
    if op == OP_NT:
        return n
    return k


def _b_cols(op: Int, n: Int, k: Int) -> Int:
    if op == OP_NT:
        return k
    return n


def _fa(a: List[Float32], op: Int, i: Int, p: Int, m: Int, k: Int) -> Float32:
    """`A_eff[i, p]`, contract section 3.

    Deliberately the same two lines as `gemm_oracle.mojo::_a_at`. **The
    independence this file needs is in the DERIVATIVE and in the SUMMATION
    ORDER, not in the layout**: a second opinion about which byte holds
    `A[i][p]` would be a bug rather than a control, because contract section
    3 is one spelling and there is nothing to cross-check it against.
    """
    if op == OP_TN:
        return a[p * m + i]
    return a[i * k + p]


def _fb(b: List[Float32], op: Int, p: Int, j: Int, n: Int, k: Int) -> Float32:
    """`B_eff[p, j]`, contract section 3."""
    if op == OP_NT:
        return b[j * k + p]
    return b[p * n + j]


def _route_name(r: Int) -> String:
    if r == 0:
        return String("dA/NN")
    if r == 1:
        return String("dA/NT")
    if r == 2:
        return String("dA/TN")
    if r == 3:
        return String("dB/NN")
    if r == 4:
        return String("dB/NT")
    if r == 5:
        return String("dB/TN")
    return String("r?")


def _route_op(r: Int) -> Int:
    """The FORWARD op of route `r`."""
    var t = r % 3
    if t == 1:
        return OP_NT
    if t == 2:
        return OP_TN
    return OP_NN


# ===========================================================================
# GATE 1: THE ROUTING TABLE IS THE TABLE
# ===========================================================================


def _want_route(
    r: Int, m: Int, n: Int, k: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """**THE TABLE, WRITTEN OUT A SECOND TIME, BY HAND. DEVIATION 1052.**

    `IDENTICAL_BACKWARD_PLAN.md` section 2.2, transcribed from the document
    rather than computed from `gemm_backward_a_call`. That is the whole value
    of this function: a gate whose expected value came out of the code under
    test compares a function to itself and passes under every sabotage.

        dA/NN   OP_NT(dC, B) @ (m, k, n)   dC LEFT
        dA/NT   OP_NN(dC, B) @ (m, k, n)   dC LEFT
        dA/TN   OP_NT(B, dC) @ (k, m, n)   dC RIGHT
        dB/NN   OP_TN(A, dC) @ (k, n, m)   dC RIGHT
        dB/NT   OP_TN(dC, A) @ (n, k, m)   dC LEFT
        dB/TN   OP_NN(A, dC) @ (k, n, m)   dC RIGHT

    `(op', m', n', k', dc_side)`, the same five-tuple the router returns.
    """
    if r == 0:
        return (OP_NT, m, k, n, BWD_DC_LEFT)
    if r == 1:
        return (OP_NN, m, k, n, BWD_DC_LEFT)
    if r == 2:
        return (OP_NT, k, m, n, BWD_DC_RIGHT)
    if r == 3:
        return (OP_TN, k, n, m, BWD_DC_RIGHT)
    if r == 4:
        return (OP_TN, n, k, m, BWD_DC_LEFT)
    return (OP_NN, k, n, m, BWD_DC_RIGHT)


def _got_route(r: Int, m: Int, n: Int, k: Int) -> Tuple[Int, Int, Int, Int, Int]:
    """What the router under test says, for route `r`."""
    var op = _route_op(r)
    if r < 3:
        return gemm_backward_a_call(op, m, n, k)
    return gemm_backward_b_call(op, m, n, k)


def _predicted_route_mask() -> Int:
    """**Which routes THIS BUILD's sabotage must move, as a bitmask.**

    The header's ledger table in one integer. The order of the `comptime if`
    blocks mirrors `gemm_backward_a_call`'s own order, where
    `SAB_BWD_UNTRANSPOSED` returns EARLY and therefore wins over
    `SAB_BWD_OPERAND_ORDER` when a build carries both.

        BWD_UNTRANSPOSED   routes 0, 2, 3, 4   ->  1 + 4 + 8 + 16 = 29
        BWD_OPERAND_ORDER  routes 2, 3, 5      ->  4 + 8 + 32     = 44
        BWD_BIAS_AXIS      none                ->  0
        clean              none                ->  0

    THE TWO ZEROS ARE NOT THE SAME ZERO. A clean build must move nothing
    because nothing is broken. A `BWD_BIAS_AXIS` build must move nothing HERE
    because the arm is aimed at a launcher this gate does not exercise, and
    G6 is where it has to bite. `[[reached-but-inert]]` is about the first
    kind of zero appearing where the second was expected, and the only way to
    tell them apart is that the ledger names which gate owns which arm.
    """
    comptime if SAB_BWD_UNTRANSPOSED:
        return 29
    comptime if SAB_BWD_OPERAND_ORDER:
        return 44
    return 0


def check_backward_routing_is_the_table() raises:
    """**G1.** All six routes equal `_want_route`, at three shapes with `m`,
    `n` and `k` pairwise distinct, and the returned `(m', n')` equals `A`'s
    shape or `B`'s shape computed independently from the forward op.

    HOST ONLY, no device, instant, runs in any lane and with no GPU present.

    THE SHAPES MUST BE PAIRWISE DISTINCT AND THE GATE REFUSES OTHERWISE.
    At `m == k` the `dA/TN` row `(k, m, n)` and the `dA/NN` row `(m, k, n)`
    are the same integers, so a router that returned one for the other would
    pass. This is the shape-level twin of G2's symmetric-fixture problem and
    it is the cheaper of the two to get wrong.

    THE `(m', n')` CROSS CHECK IS THE HALF THAT CATCHES A CONSISTENT ERROR.
    Comparing the router to `_want_route` catches a router that disagrees
    with the plan document. It does not catch a plan document that is wrong.
    So the second clause asks a different question with different inputs:
    does the shape this call writes into actually have `A`'s shape, computed
    from `_a_rows` / `_a_cols`, which never saw the table? A transposed row
    in BOTH places fails that.

    ASSERTED IN BOTH MODES. Nothing here is arithmetic.

    Under a sabotage build the gate does not merely expect failure, it
    expects failure ON EXACTLY THE ROUTES `_predicted_route_mask` names, and
    a build that moves a route the mask does not name fails just as hard as
    one that moves nothing. DEVIATION 1052.
    """
    print(
        "check_backward_routing_is_the_table ["
        + _mode_name()
        + "]: six routes against the hand-written table"
    )
    var moved_mask = 0
    var shapes_m = [5, 9, 6]
    var shapes_n = [7, 5, 11]
    var shapes_k = [3, 7, 4]
    var shape_fails = 0
    for s in range(len(shapes_m)):
        var m = shapes_m[s]
        var n = shapes_n[s]
        var k = shapes_k[s]
        if m == n or m == k or n == k:
            raise Error(
                "check_backward_routing_is_the_table: shape "
                + String(m)
                + "x"
                + String(n)
                + "x"
                + String(k)
                + " has two equal dimensions. At equal dimensions a"
                " transposed row is the same integers as the right one and"
                " this gate would pass a router that swapped them. Pick"
                " pairwise-distinct dimensions."
            )
        for r in range(6):
            var want = _want_route(r, m, n, k)
            var got = _got_route(r, m, n, k)
            var same = (
                got[0] == want[0]
                and got[1] == want[1]
                and got[2] == want[2]
                and got[3] == want[3]
                and got[4] == want[4]
            )
            if not same:
                moved_mask = moved_mask | (1 << r)
                if s == 0:
                    print(
                        "    ROUTE MOVED "
                        + _route_name(r)
                        + " at "
                        + String(m)
                        + "x"
                        + String(n)
                        + "x"
                        + String(k)
                        + ": router says "
                        + gemm_backward_call_name(got)
                        + ", the table says "
                        + gemm_backward_call_name(want)
                    )
            # The independent shape cross check, run on the TABLE's answer as
            # well as the router's, so that a wrong table is caught here and
            # not three gates downstream.
            var want_rows = _a_rows(_route_op(r), m, k)
            var want_cols = _a_cols(_route_op(r), m, k)
            if r >= 3:
                want_rows = _b_rows(_route_op(r), n, k)
                want_cols = _b_cols(_route_op(r), n, k)
            if want[1] != want_rows or want[2] != want_cols:
                raise Error(
                    "check_backward_routing_is_the_table: THE TABLE ITSELF"
                    " IS WRONG at "
                    + _route_name(r)
                    + ". _want_route returns output shape "
                    + String(want[1])
                    + "x"
                    + String(want[2])
                    + " and the operand it is the gradient of has shape "
                    + String(want_rows)
                    + "x"
                    + String(want_cols)
                    + ". This is not a router defect; it is a defect in"
                    " IDENTICAL_BACKWARD_PLAN.md section 2.2 as transcribed"
                    " here."
                )
            if same and (got[1] != want_rows or got[2] != want_cols):
                shape_fails += 1
    var predicted = _predicted_route_mask()
    var n_moved = 0
    for r in range(6):
        if moved_mask & (1 << r) != 0:
            n_moved += 1
    print(
        "  routes moved: "
        + String(n_moved)
        + " of 6, mask "
        + String(moved_mask)
        + "   predicted for sabotage '"
        + gemm_backward_sabotage_name()
        + "': mask "
        + String(predicted)
    )
    if shape_fails != 0:
        raise Error(
            "check_backward_routing_is_the_table: "
            + String(shape_fails)
            + " routes agreed with the table and still wrote an output whose"
            " shape is not the shape of the operand they are the gradient"
            " of. The router and the table are consistently wrong."
        )
    if moved_mask != predicted:
        raise Error(
            "check_backward_routing_is_the_table ["
            + _mode_name()
            + "]: the moved-route mask is "
            + String(moved_mask)
            + " and this build predicts "
            + String(predicted)
            + " (sabotage: "
            + gemm_backward_sabotage_name()
            + "). A mask BELOW the prediction is a sabotage that is reached"
            " and inert on a route; a mask ABOVE it is a routing defect the"
            " ledger does not know about. Either way the ledger in this"
            " file's header is now wrong and has to be corrected before"
            " anything here is believed."
        )
    if predicted != 0:
        print(
            "check_backward_routing_is_the_table OK (INVERTED): the"
            " sabotage moved exactly the "
            + String(n_moved)
            + " routes the ledger predicts and left the other "
            + String(6 - n_moved)
            + " bit identical, which is the per-branch reach claim and not"
            " a pass of the clean gate."
        )
        return
    print(
        "check_backward_routing_is_the_table ["
        + _mode_name()
        + "] OK: 6 routes x 3 pairwise-distinct shapes equal the"
        " hand-written table, and every output shape equals the shape of the"
        " operand it is the gradient of"
    )


# ===========================================================================
# GATE 2: THE SIX ROUTES COMPUTE THE RIGHT DERIVATIVE
# ===========================================================================
# `_ref_da`, `_ref_db` and `_ref_bias` are derived from the FORWARD's own
# definition, `C[i][j] = sum over p of A_eff[i][p] * B_eff[p][j]`, and the
# routing table is nowhere in their scope. They are the slow obvious way:
# one plain ascending sum per output cell, no partition, no fold, no ftz.
#
# THEY ARE BIT EXACT ONLY ON `_ival`'S FIXTURE and that is the whole design.
# Every operand is an integer in [-8, 8], every product an integer in
# [-64, 64], and every partial sum an integer of magnitude at most 64 * k'.
# While that stays under 2^24 the sum is exact in Float32 in EVERY order, so
# a plain host loop, the contract's partitioned balanced tree and any
# vendor's library all produce the same bits. `_exact_bound` is what checks
# the "while", and it raises rather than warns.


def _exact_bound(terms: Int, label: String) raises:
    """`64 * terms` must stay below `2^24`, or the fixture stops being exact
    and G2 stops being a bitwise gate.

    Called with the CONTRACTION LENGTH of every sum G2 performs, `k'`, which
    is `n` for a `dA` route, `m` for a `dB` route and `m` for the bias. The
    bound is a static one and it is deliberately not a measurement of the
    actual maximum: a fixture that happens to stay exact at this seed and
    would not at another is a fixture that passes for a reason nobody wrote
    down.
    """
    var bound = Float32(64.0) * Float32(terms)
    if bound >= EXACT_LIMIT:
        raise Error(
            "_exact_bound ["
            + label
            + "]: a contraction over "
            + String(terms)
            + " terms of magnitude <= 64 can reach "
            + _show(bound)
            + ", which is not below 2^24 = "
            + _show(EXACT_LIMIT)
            + ". The integer fixture is no longer exact at this shape, so a"
            " bitwise comparison against a plain host sum would be measuring"
            " summation ORDER and reporting it as a derivative error. Reduce"
            " the shape or move this case to G3."
        )


def _check_operands_are_exact(v: List[Float32], label: String) raises:
    """Every entry is a nonzero integer of magnitude at most 8.

    The NEGATIVE CONTROL for the whole of G2. If somebody hands these
    functions the rounding fixture by mistake the comparison becomes a
    summation-order comparison that fails for a reason that has nothing to do
    with the derivative, and the failure text would send the reader hunting
    the router. This raises with the actual value instead.
    """
    for i in range(len(v)):
        var x = v[i]
        if x != Float32(Int(x)):
            raise Error(
                "_check_operands_are_exact ["
                + label
                + "]: entry "
                + String(i)
                + " is "
                + _show(x)
                + ", which is not an integer. G2's bitwise comparison is"
                " only valid on the exact fixture."
            )
        if x > Float32(8.0) or x < Float32(-8.0):
            raise Error(
                "_check_operands_are_exact ["
                + label
                + "]: entry "
                + String(i)
                + " is "
                + _show(x)
                + ", outside [-8, 8]. Products are no longer bounded by 64"
                " and _exact_bound's guarantee is void."
            )
        if x == Float32(0.0):
            raise Error(
                "_check_operands_are_exact ["
                + label
                + "]: entry "
                + String(i)
                + " is zero. A zero operand makes its whole term vanish, and"
                " a term that vanishes cannot distinguish a right index from"
                " a wrong one, which is exactly what G2 is for."
            )


def _ref_da(
    dc: List[Float32],
    b: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) -> List[Float32]:
    """`dA` the slow obvious way, in `A`'s PHYSICAL layout.

    `dA_eff[i][p] = sum over j of dC[i][j] * B_eff[p][j]`, straight from
    differentiating `C[i][j] = sum over p of A_eff[i][p] * B_eff[p][j]` with
    respect to `A_eff[i][p]`. The routing table is not in scope here and must
    not be: this function is the thing the routing table is checked AGAINST.

    The store at the end is where `OP_TN`'s `A` being `k x m` is handled, and
    it is the one line in this function a reader should look at twice.
    """
    var rows = _a_rows(op, m, k)
    var cols = _a_cols(op, m, k)
    var out = _fill_const(rows * cols, Float32(0.0))
    for i in range(m):
        for p in range(k):
            var acc = Float32(0.0)
            for j in range(n):
                acc = acc + dc[i * n + j] * _fb(b, op, p, j, n, k)
            if op == OP_TN:
                out[p * m + i] = acc
            else:
                out[i * k + p] = acc
    return out^


def _ref_db(
    dc: List[Float32],
    a: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
) -> List[Float32]:
    """`dB` the slow obvious way, in `B`'s PHYSICAL layout.

    `dB_eff[p][j] = sum over i of A_eff[i][p] * dC[i][j]`. The contraction is
    over `i`, THE TOKEN INDEX, which is the whole of section 3.2 visible in
    one loop bound: `dA`'s sum above runs over `j` and does not see the batch
    at all, and this one runs over `i` and sees nothing else.
    """
    var rows = _b_rows(op, n, k)
    var cols = _b_cols(op, n, k)
    var out = _fill_const(rows * cols, Float32(0.0))
    for p in range(k):
        for j in range(n):
            var acc = Float32(0.0)
            for i in range(m):
                acc = acc + _fa(a, op, i, p, m, k) * dc[i * n + j]
            if op == OP_NT:
                out[j * k + p] = acc
            else:
                out[p * n + j] = acc
    return out^


def _ref_bias(dc: List[Float32], m: Int, n: Int) -> List[Float32]:
    """`db[j] = sum over i of dC[i][j]`. **No ones vector anywhere in it.**

    Not a matrix product, not a GEMM, not routed: the reduction a reader
    means when they say "the bias gradient". `identical_gemm_backward_bias_
    into` claims to compute this by multiplying by a vector of ones, and this
    is the function that claim is checked against.
    """
    var out = _fill_const(n, Float32(0.0))
    for j in range(n):
        var acc = Float32(0.0)
        for i in range(m):
            acc = acc + dc[i * n + j]
        out[j] = acc
    return out^


def _route_needs(call: Tuple[Int, Int, Int, Int, Int]) -> Tuple[Int, Int, Int]:
    """`(left_floats, right_floats, out_floats)` a call at this route READS
    and WRITES.

    All three orientations read `m' k'` from the left slot and `k' n'` from
    the right, because a transpose changes which index is major and not how
    many elements there are. That is why a transposed read never raises and
    why `[[read-their-source-against-ours]]` applies to this whole lane.
    """
    return (call[1] * call[3], call[3] * call[2], call[1] * call[2])


def _routed_host(
    call: Tuple[Int, Int, Int, Int, Int],
    dc: List[Float32],
    w: List[Float32],
) raises -> List[Float32]:
    """Evaluate the ROUTED call on the host, through `gemm_oracle`.

    `gemm_oracle` is the profile's DEFINITION, so this is what the device is
    required to produce and it is also, on the exact fixture, the same number
    a plain sum produces. That double duty is what lets G2 be a host-only
    gate: it can answer "is this the right derivative" without a GPU, and G3
    then answers "does the device produce these bits" separately.
    """
    var need = _route_needs(call)
    if call[4] == BWD_DC_RIGHT:
        var lw = _pad_to(w, need[0])
        var rd = _pad_to(dc, need[1])
        return gemm_oracle(lw, rd, call[0], call[1], call[2], call[3])
    var ld = _pad_to(dc, need[0])
    var rw = _pad_to(w, need[1])
    return gemm_oracle(ld, rw, call[0], call[1], call[2], call[3])


def _g2_case(
    op: Int, m: Int, n: Int, k: Int, mut cases: Int, mut moved_mask: Int
) raises:
    """One forward orientation at one shape: both routes and the bias."""
    var a_n = _a_rows(op, m, k) * _a_cols(op, m, k)
    var b_n = _b_rows(op, n, k) * _b_cols(op, n, k)
    var dc = _fill_exact(m * n, 31)
    var a = _fill_exact(a_n, 47)
    var b = _fill_exact(b_n, 59)
    _check_operands_are_exact(dc, String("dC"))
    _check_operands_are_exact(a, String("A"))
    _check_operands_are_exact(b, String("B"))
    _exact_bound(n, String("dA ") + op_name(op))
    _exact_bound(m, String("dB ") + op_name(op))

    var want_a = _ref_da(dc, b, op, m, n, k)
    var got_a = _routed_host(gemm_backward_a_call(op, m, n, k), dc, b)
    var want_b = _ref_db(dc, a, op, m, n, k)
    var got_b = _routed_host(gemm_backward_b_call(op, m, n, k), dc, a)

    var ra = 0
    if op == OP_NT:
        ra = 1
    elif op == OP_TN:
        ra = 2
    _g2_compare(
        want_a,
        got_a,
        _route_name(ra)
        + " "
        + String(m)
        + "x"
        + String(n)
        + "x"
        + String(k),
        ra,
        cases,
        moved_mask,
    )
    _g2_compare(
        want_b,
        got_b,
        _route_name(ra + 3)
        + " "
        + String(m)
        + "x"
        + String(n)
        + "x"
        + String(k),
        ra + 3,
        cases,
        moved_mask,
    )


def _g2_compare(
    want: List[Float32],
    got: List[Float32],
    label: String,
    route: Int,
    mut cases: Int,
    mut moved_mask: Int,
) raises:
    cases += 1
    if len(want) != len(got):
        moved_mask = moved_mask | (1 << route)
        print(
            "    MOVED "
            + label
            + ": the routed call writes "
            + String(len(got))
            + " floats and the gradient has "
            + String(len(want))
            + ". A LENGTH disagreement, which means the route is at the"
            " wrong shape and not merely reading the wrong cells."
        )
        return
    var bad = 0
    var first = -1
    for c in range(len(want)):
        if _bits(want[c]) != _bits(got[c]):
            bad += 1
            if first < 0:
                first = c
    if bad == 0:
        print("    OK    " + label + "  " + String(len(want)) + " cells exact")
        return
    moved_mask = moved_mask | (1 << route)
    print(
        "    MOVED "
        + label
        + ": "
        + String(bad)
        + " of "
        + String(len(want))
        + " cells differ from the derivative. First at flat index "
        + String(first)
        + ": routed = "
        + _show(got[first])
        + "   derivative = "
        + _show(want[first])
    )


def check_backward_gradients_are_correct() raises:
    """**G2, AND THE GATE THAT ANSWERS A DIFFERENT QUESTION FROM ALL THE
    OTHERS.** Every one of the six routes computes the RIGHT derivative,
    bit for bit, on a fixture where "right" has an exact answer.

    HOST ONLY. No device, no GPU, instant.

    WHY THIS IS NOT A TOLERANCE CHECK AND NEEDS NO CENTRAL DIFFERENCE.
    `IDENTICAL_BACKWARD_PLAN.md` section 5's G2 proposes a central difference
    at `h = 1` on an integer fixture, and the argument that it is exact is
    sound. It is also more machinery than the question needs: on the same
    integer fixture the DERIVATIVE ITSELF is exactly representable and can be
    written down directly, one plain ascending sum per cell, and comparing
    two exact numbers is simpler than comparing two exact differences of
    exact numbers. `_ref_da` and `_ref_db` are those sums. DEVIATION 1051.

    The exactness is not assumed. `_check_operands_are_exact` checks every
    operand and `_exact_bound` checks the contraction length, and both raise
    rather than warn, because a fixture that has quietly stopped being exact
    turns this gate from a derivative check into a summation-order check that
    fails for the wrong reason.

    THE SHAPE CONSTRAINT, AND IT IS ENFORCED RATHER THAN DOCUMENTED. `m`, `n`
    and `k` must be pairwise distinct, because at a square shape a transposed
    read has the right length and lands in the right buffer.
    `check_the_square_fixture_is_vacuous` is the demonstration of what that
    costs and it is worth more than this paragraph.

    ASSERTED IN BOTH MODES, which is unusual here and is a consequence of the
    fixture: under FAST `identical_mul_add` becomes `a * b + c` and the
    backend is free to contract, and on exact integers a contracted
    multiply-add and an uncontracted one produce the same bits. A FAST
    failure of this gate is a real routing defect and not a mode artifact.

    Under a sabotage build this gate expects the routes
    `_predicted_route_mask` names to move and the others to be bit
    identical, exactly as G1 does, and reports the mask either way.
    """
    print(
        "check_backward_gradients_are_correct ["
        + _mode_name()
        + "]: six routes against the derivative, on the exact fixture"
    )
    var cases = 0
    var moved_mask = 0
    var shapes_m = [5, 9, 6]
    var shapes_n = [7, 5, 11]
    var shapes_k = [3, 7, 4]
    for s in range(len(shapes_m)):
        var m = shapes_m[s]
        var n = shapes_n[s]
        var k = shapes_k[s]
        if m == n or m == k or n == k:
            raise Error(
                "check_backward_gradients_are_correct: shape "
                + String(m)
                + "x"
                + String(n)
                + "x"
                + String(k)
                + " is not pairwise distinct. At a square shape a transpose"
                " error computes the same numbers and this gate reports the"
                " opposite of the truth. See"
                " check_the_square_fixture_is_vacuous."
            )
        print(
            "  -- "
            + String(m)
            + "x"
            + String(n)
            + "x"
            + String(k)
            + " --"
        )
        _g2_case(OP_NN, m, n, k, cases, moved_mask)
        _g2_case(OP_NT, m, n, k, cases, moved_mask)
        _g2_case(OP_TN, m, n, k, cases, moved_mask)

    # THE BIAS, on the same fixture. Route and shape are fixed and there is
    # only one of them, so it is not part of the six-route mask; it has its
    # own arm and its own sabotage, and G6 is where it is measured on a
    # device.
    var bias_bad = 0
    var bias_cells = 0
    for s in range(len(shapes_m)):
        var m = shapes_m[s]
        var n = shapes_n[s]
        _exact_bound(m, String("bias"))
        var dc = _fill_exact(m * n, 31)
        var ones = _fill_const(identical_gemm_backward_bias_ones_floats(m), Float32(1.0))
        var want = _ref_bias(dc, m, n)
        var bias_call = _bias_call(m, n)
        var got = _routed_host(bias_call, dc, ones)
        bias_cells += n
        if len(got) < n:
            bias_bad += n
            print(
                "    MOVED bias "
                + String(m)
                + "x"
                + String(n)
                + ": the routed call writes "
                + String(len(got))
                + " floats and db has "
                + String(n)
                + "."
            )
            continue
        var bad = 0
        for j in range(n):
            if _bits(want[j]) != _bits(got[j]):
                bad += 1
        bias_bad += bad
        if bad == 0:
            print(
                "    OK    bias "
                + String(m)
                + "x"
                + String(n)
                + "  "
                + String(n)
                + " cells exact, and the ones vector reproduced a sum that"
                " has no ones vector in it"
            )
        else:
            print(
                "    MOVED bias "
                + String(m)
                + "x"
                + String(n)
                + ": "
                + String(bad)
                + " of "
                + String(n)
                + " cells differ from the plain column sum"
            )

    var predicted = _predicted_route_mask()
    var n_moved = 0
    for r in range(6):
        if moved_mask & (1 << r) != 0:
            n_moved += 1
    print(
        "  routes moved: "
        + String(n_moved)
        + " of 6, mask "
        + String(moved_mask)
        + "   predicted mask "
        + String(predicted)
        + "   bias cells moved: "
        + String(bias_bad)
        + " of "
        + String(bias_cells)
    )
    comptime if SAB_BWD_BIAS_AXIS:
        if bias_bad == 0:
            raise Error(
                "check_backward_gradients_are_correct: this build is"
                " BWD_BIAS_AXIS and the bias arm moved NOTHING. The arm is"
                " reached and inert, or it is not reached at all. Either way"
                " the bias clause is UNGATED."
            )
    if not SAB_BWD_BIAS_AXIS and bias_bad != 0:
        raise Error(
            "check_backward_gradients_are_correct ["
            + _mode_name()
            + "]: the ones-vector bias route disagrees with the plain column"
            " sum on "
            + String(bias_bad)
            + " of "
            + String(bias_cells)
            + " cells, on a fixture where every sum is exact. DEVIATION"
            " 851's claim that db[1 x n] = ones[1 x m] . dC[m x n] IS the"
            " reduction is FALSE as spelled, and"
            " IDENTICAL_BACKWARD_PLAN.md section 3.3 has to say so."
        )
    if moved_mask != predicted:
        raise Error(
            "check_backward_gradients_are_correct ["
            + _mode_name()
            + "]: the moved-route mask is "
            + String(moved_mask)
            + " and this build predicts "
            + String(predicted)
            + " (sabotage: "
            + gemm_backward_sabotage_name()
            + "). On the exact fixture a moved route is a WRONG DERIVATIVE,"
            " not a rounding difference, so a mask above the prediction in a"
            " clean build means the routing table in"
            " IDENTICAL_BACKWARD_PLAN.md section 2.2 is wrong."
        )
    if predicted != 0:
        print(
            "check_backward_gradients_are_correct OK (INVERTED): exactly the"
            " predicted routes computed the wrong derivative"
        )
        return
    print(
        "check_backward_gradients_are_correct ["
        + _mode_name()
        + "] OK: "
        + String(cases)
        + " routes and "
        + String(bias_cells)
        + " bias cells equal the derivative bit for bit"
    )


def _bias_call(m: Int, n: Int) -> Tuple[Int, Int, Int, Int, Int]:
    """The route `identical_gemm_backward_bias_into` actually takes, INCLUDING
    this build's sabotage.

    `gemm_backward.mojo` has no `gemm_backward_bias_call` producer -- the
    bias launcher writes its `identical_gemm_into` call inline -- so this is
    a transcription of those two lines and it has to track them. It is the
    one place in this file where the check re-spells the code under test, and
    the cost if it drifts is that the bias allocation is sized for a call
    that is not the call that runs.

    **THAT IS A REQUEST, NOT A DEFECT**, and it is in the report: adding
    `gemm_backward_bias_call(m, n)` to `gemm_backward.mojo` beside the other
    two producers would make the bias route have ONE producer like `dA` and
    `dB` do, and would delete this function.

        clean            OP_NN(ones, dC) @ (1, n, m)   dC RIGHT
        BWD_BIAS_AXIS    OP_NT(dC, ones) @ (m, 1, n)   dC LEFT
    """
    comptime if SAB_BWD_BIAS_AXIS:
        return (OP_NT, m, 1, n, BWD_DC_LEFT)
    return (OP_NN, 1, n, m, BWD_DC_RIGHT)


def check_the_square_fixture_is_vacuous() raises:
    """**THE INTERESTING HALF OF G2, AND IT PASSES BY SHOWING A FAILURE THAT
    IS NOT THERE.** DEVIATION 1053.

    A transpose error is INVISIBLE on a square symmetric fixture. This
    function does not assert that; it computes it, in a CLEAN BUILD, and
    prints the count.

    THE CONSTRUCTION. Take `m = n = k = N` and build `A`, `B` and `dC` all
    symmetric, `X[i][j] == X[j][i]`. Then for forward `OP_NN` the correct
    `dA` is `dC B^T` and the untransposed route computes `dC B`; with `B`
    symmetric those are the SAME MATRIX, bit for bit, at every cell. Same for
    `dB`: `A^T dC` against `A dC`. So the route that
    `MOJOLEARN_GEMM_SABOTAGE_BWD_UNTRANSPOSED` installs returns the correct
    gradient on this fixture, and a G2 built on it would report the sabotage
    green.

    **WHY IT IS DONE IN A CLEAN BUILD RATHER THAN UNDER THE SABOTAGE.** The
    plan's section 5 asks for this demonstration and expects a
    `-D MOJOLEARN_GEMM_SABOTAGE_BWD_UNTRANSPOSED=1` build to be run against a
    square fixture. That works and it costs a whole extra build and an
    operator who remembers to run it. Re-spelling the untransposed route as
    six local integers (`_untransposed_route` below) makes the demonstration
    part of the DEFAULT run, so the reason the shape constraint exists is in
    every transcript rather than in a build somebody has to think of.

    The cost of the re-spelling is that this function tests a route the
    router does not produce in this build. That is acceptable HERE and would
    not be acceptable in G1 or G3, because the claim being made is about the
    FIXTURE and not about the router: it is "no fixture of this shape can
    separate these two routes", and that is true of the routes as
    mathematical objects.

    `[[reached-but-inert]]`, and the mamba lane's finding that an adversarial
    corpus case was BITWISE INERT under a sabotage is the same shape of
    result. It is also the reason this file's header ledger predicts 4 of 6
    rather than 6 of 6.

    THE SECOND HALF IS THE CONTROL. The same comparison is run on an
    ASYMMETRIC square fixture and on the pairwise-distinct fixture G2 uses,
    and both MUST separate. If the symmetric arm shows 0 moved cells and the
    asymmetric arm also shows 0, the demonstration has proved nothing about
    symmetry and this gate raises.
    """
    print(
        "check_the_square_fixture_is_vacuous ["
        + _mode_name()
        + "]: what a square symmetric fixture cannot see"
    )
    var nn = 6
    # SYMMETRIC ARM. Predicted: 0 of 36 cells move on dA and 0 of 36 on dB.
    var sdc = _fill_const(nn * nn, Float32(0.0))
    var sa = _fill_const(nn * nn, Float32(0.0))
    var sb = _fill_const(nn * nn, Float32(0.0))
    for i in range(nn):
        for j in range(nn):
            sdc[i * nn + j] = _sym_val(i, j, 71)
            sa[i * nn + j] = _sym_val(i, j, 83)
            sb[i * nn + j] = _sym_val(i, j, 97)
    var sym_a = _vac_arm(sdc, sa, sb, nn, nn, nn, String("symmetric 6x6x6"))
    # ASYMMETRIC SQUARE ARM, the control for the symmetry claim.
    var adc = _fill(nn * nn, 71)
    var aa = _fill(nn * nn, 83)
    var ab = _fill(nn * nn, 97)
    var asym_a = _vac_arm(adc, aa, ab, nn, nn, nn, String("asymmetric 6x6x6"))
    # PAIRWISE-DISTINCT ARM, the control for the shape claim.
    var pdc = _fill(5 * 7, 71)
    var pa = _fill(5 * 3, 83)
    var pb = _fill(3 * 7, 97)
    var pd_a = _vac_arm(pdc, pa, pb, 5, 7, 3, String("distinct 5x7x3"))

    if sym_a != 0:
        raise Error(
            "check_the_square_fixture_is_vacuous: the SYMMETRIC arm moved "
            + String(sym_a)
            + " cells. It must move ZERO: on a symmetric fixture dC . B and"
            " dC . B^T are the same matrix. Either the fixture is not"
            " symmetric or _untransposed_route no longer spells the"
            " sabotage, and in both cases the demonstration below it is"
            " worthless."
        )
    if asym_a == 0 or pd_a == 0:
        raise Error(
            "check_the_square_fixture_is_vacuous: a CONTROL arm moved"
            " nothing (asymmetric square: "
            + String(asym_a)
            + ", pairwise distinct: "
            + String(pd_a)
            + "). If the untransposed route agrees with the correct one on"
            " EVERY fixture then this function has demonstrated nothing"
            " about symmetry and the whole argument for G2's shape"
            " constraint is unsupported."
        )
    print(
        "check_the_square_fixture_is_vacuous ["
        + _mode_name()
        + "] OK: the untransposed route is BITWISE INERT on the symmetric"
        " square fixture (0 cells moved) and moves "
        + String(asym_a)
        + " cells on the asymmetric square one and "
        + String(pd_a)
        + " on the pairwise-distinct one. That is why G2 refuses a square"
        " fixture, measured rather than asserted."
    )


def _untransposed_route(
    r: Int, m: Int, n: Int, k: Int
) -> Tuple[Int, Int, Int, Int, Int]:
    """What `SAB_BWD_UNTRANSPOSED` installs, re-spelled locally so the
    demonstration above runs in a clean build. Transcribed from
    `gemm_backward.mojo`'s two early returns.

        dA  ->  OP_NN @ (m, k, n)  dC LEFT
        dB  ->  OP_NN @ (k, n, m)  dC RIGHT
    """
    if r < 3:
        return (OP_NN, m, k, n, BWD_DC_LEFT)
    return (OP_NN, k, n, m, BWD_DC_RIGHT)


def _vac_arm(
    dc: List[Float32],
    a: List[Float32],
    b: List[Float32],
    m: Int,
    n: Int,
    k: Int,
    label: String,
) raises -> Int:
    """How many cells the untransposed route moves against the correct route,
    for forward `OP_NN`, summed over `dA` and `dB`. Returns the count."""
    var moved = 0
    var total = 0
    var right_a = _routed_host(_want_route(0, m, n, k), dc, b)
    var wrong_a = _routed_host(_untransposed_route(0, m, n, k), dc, b)
    var right_b = _routed_host(_want_route(3, m, n, k), dc, a)
    var wrong_b = _routed_host(_untransposed_route(3, m, n, k), dc, a)
    var na = len(right_a)
    if len(wrong_a) < na:
        na = len(wrong_a)
    for c in range(na):
        total += 1
        if _bits(right_a[c]) != _bits(wrong_a[c]):
            moved += 1
    var nb = len(right_b)
    if len(wrong_b) < nb:
        nb = len(wrong_b)
    for c in range(nb):
        total += 1
        if _bits(right_b[c]) != _bits(wrong_b[c]):
            moved += 1
    print(
        "    "
        + label
        + ": untransposed vs correct moved "
        + String(moved)
        + " of "
        + String(total)
        + " cells across dA and dB"
    )
    return moved


# ===========================================================================
# THE DEVICE HARNESS
# ===========================================================================
# One runner for all three launchers. It allocates from the ROUTE THIS BUILD
# ACTUALLY TAKES (DEVIATION 1050), poisons the output, launches, drains, and
# reports how many cells came back never written. It does NOT raise on a
# surviving poison: the caller decides, because in a clean build an unwritten
# cell is a defect and under `BWD_BIAS_AXIS` it is the predicted symptom, and
# a runner that raised would turn the second into a stack trace instead of a
# number.

comptime BWD_A = 0
comptime BWD_B = 1
comptime BWD_BIAS = 2


def _bwd_route(which: Int, op: Int, m: Int, n: Int, k: Int) -> Tuple[Int, Int, Int, Int, Int]:
    if which == BWD_A:
        return gemm_backward_a_call(op, m, n, k)
    if which == BWD_B:
        return gemm_backward_b_call(op, m, n, k)
    return _bias_call(m, n)


def _bwd_out_logical(which: Int, op: Int, m: Int, n: Int, k: Int) -> Int:
    """How many floats the GRADIENT has, from the operand's shape and never
    from the route. A route at the wrong shape must not be able to redefine
    how much of its own output gets compared."""
    if which == BWD_A:
        return _a_rows(op, m, k) * _a_cols(op, m, k)
    if which == BWD_B:
        return _b_rows(op, n, k) * _b_cols(op, n, k)
    return n


def _bwd_ws_floats(which: Int, op: Int, m: Int, n: Int, k: Int) -> Int:
    """The workspace, sized for the route that will actually run.

    For `dA` and `dB` the shipped helpers already route through
    `gemm_backward_a_call` / `_b_call` and therefore already follow this
    build's sabotage. **The bias helper does not**: `identical_gemm_backward_
    bias_workspace_max_floats(m, n)` hard-codes `(1, n, m)` and a
    `BWD_BIAS_AXIS` build runs `(m, 1, n)`, so under that arm the shipped
    number can be for the wrong plan. This function takes the max of the two,
    which is safe in every build and is the reason the sabotage arm does not
    turn into an out-of-bounds write.

    That gap is REPORTED and not fixed here: the fix belongs in
    `gemm_backward.mojo`, which this lane may not edit.
    """
    if which == BWD_A:
        return identical_gemm_backward_a_workspace_max_floats(op, m, n, k)
    if which == BWD_B:
        return identical_gemm_backward_b_workspace_max_floats(op, m, n, k)
    var shipped = identical_gemm_backward_bias_workspace_max_floats(m, n)
    var call = _bias_call(m, n)
    var actual = identical_gemm_workspace_max_floats(call[1], call[2], call[3])
    return _max2(shipped, actual)


def _run_bwd(
    ctx: DeviceContext,
    which: Int,
    hdc: List[Float32],
    hw: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    label: String,
    mut unwritten: Int,
) raises -> List[Float32]:
    """One backward launch through the SHIPPED launcher, poisoned first.

    `hdc` and `hw` must already be padded to cover the route (see `_pad_to`).
    Returns the LOGICAL extent of the gradient as host floats.

    `[[mojo-buffer-freed-at-last-use]]`: every buffer is kept alive with a
    trailing `_ =` past the last `synchronize`, because a `DeviceBuffer` is
    dead at its `.unsafe_ptr()` and this is the most common device bug in
    this repository.
    """
    var call = _bwd_route(which, op, m, n, k)
    var out_log = _bwd_out_logical(which, op, m, n, k)
    var out_alloc = _max2(out_log, call[1] * call[2])
    if out_alloc < 1:
        out_alloc = 1
    var ndc = len(hdc)
    if ndc < 1:
        ndc = 1
    var nw = len(hw)
    if nw < 1:
        nw = 1
    var nws = _bwd_ws_floats(which, op, m, n, k)
    if nws < 1:
        nws = 1

    var ddc = ctx.enqueue_create_buffer[DType.float32](ndc)
    var dw = ctx.enqueue_create_buffer[DType.float32](nw)
    var dout = ctx.enqueue_create_buffer[DType.float32](out_alloc)
    var dws = ctx.enqueue_create_buffer[DType.float32](nws)
    var hDC = ctx.enqueue_create_host_buffer[DType.float32](ndc)
    var hW = ctx.enqueue_create_host_buffer[DType.float32](nw)
    var hO = ctx.enqueue_create_host_buffer[DType.float32](out_alloc)
    var hWS = ctx.enqueue_create_host_buffer[DType.float32](nws)
    ctx.synchronize()
    for i in range(len(hdc)):
        hDC.unsafe_ptr().unsafe_store(i, hdc[i])
    for i in range(len(hw)):
        hW.unsafe_ptr().unsafe_store(i, hw[i])
    for i in range(out_alloc):
        hO.unsafe_ptr().unsafe_store(i, POISON)
    # THE WORKSPACE IS POISONED TOO. A plan that reads a slot it did not
    # write reads a partial that is not a partial, and a zeroed workspace
    # would hide that behind a plausible `+0.0`.
    for i in range(nws):
        hWS.unsafe_ptr().unsafe_store(i, POISON)
    ctx.enqueue_copy(dst_buf=ddc, src_ptr=hDC.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dw, src_ptr=hW.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dout, src_ptr=hO.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dws, src_ptr=hWS.unsafe_ptr())
    ctx.synchronize()

    if which == BWD_A:
        identical_gemm_backward_a_into(ctx, dout, ddc, dw, dws, m, n, k, op)
    elif which == BWD_B:
        identical_gemm_backward_b_into(ctx, dout, ddc, dw, dws, m, n, k, op)
    else:
        identical_gemm_backward_bias_into(ctx, dout, ddc, dw, dws, m, n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hO.unsafe_ptr(), src_buf=dout)
    ctx.synchronize()

    var out = List[Float32]()
    for i in range(out_log):
        var v = hO.unsafe_ptr().unsafe_load(i)
        if _bits(v) == _bits(POISON):
            unwritten += 1
        out.append(v)
    _ = ddc
    _ = dw
    _ = dout
    _ = dws
    _ = hDC
    _ = hW
    _ = hO
    _ = hWS
    return out^


def _dc_floats_needed(which: Int, op: Int, m: Int, n: Int, k: Int) -> Int:
    """How many floats THIS BUILD's route reads out of the `dC` slot."""
    var call = _bwd_route(which, op, m, n, k)
    var need = _route_needs(call)
    if call[4] == BWD_DC_LEFT:
        return need[0]
    return need[1]


def _w_floats_needed(which: Int, op: Int, m: Int, n: Int, k: Int) -> Int:
    """How many floats THIS BUILD's route reads out of the OTHER slot."""
    var call = _bwd_route(which, op, m, n, k)
    var need = _route_needs(call)
    if call[4] == BWD_DC_LEFT:
        return need[1]
    return need[0]


def _w_floats_logical(which: Int, op: Int, m: Int, n: Int, k: Int) -> Int:
    """How many floats the OTHER operand actually holds: `B` for a `dA`
    route, `A` for a `dB` route, the ones vector for the bias."""
    if which == BWD_B:
        return _a_rows(op, m, k) * _a_cols(op, m, k)
    if which == BWD_BIAS:
        return identical_gemm_backward_bias_ones_floats(m)
    return _b_rows(op, n, k) * _b_cols(op, n, k)


def _bwd_dc(
    which: Int, op: Int, m: Int, n: Int, k: Int, exact: Bool, salt: Int
) -> List[Float32]:
    """`dC`, `m x n`, padded to whatever this build's route reads.

    **TWO FUNCTIONS RATHER THAN ONE RETURNING A PAIR.** `List` is not
    implicitly copyable on return in Mojo and a `Tuple[List, List]` needs
    explicit transfers that three gate files exactly like this one failed to
    compile on last week. Two single-value producers cost a second call and
    nothing else.
    """
    var dc = _fill(m * n, salt)
    if exact:
        dc = _fill_exact(m * n, salt)
    return _pad_to(dc, _dc_floats_needed(which, op, m, n, k))


def _bwd_w(
    which: Int, op: Int, m: Int, n: Int, k: Int, exact: Bool, salt: Int
) -> List[Float32]:
    """`B`, `A` or the ones vector, padded to whatever the route reads."""
    var wl = _w_floats_logical(which, op, m, n, k)
    var w = _fill(wl, salt + 13)
    if exact:
        w = _fill_exact(wl, salt + 13)
    if which == BWD_BIAS:
        w = _fill_const(wl, Float32(1.0))
    return _pad_to(w, _w_floats_needed(which, op, m, n, k))


# ===========================================================================
# GATE 3: THE DEVICE COMPUTES `gemm_oracle` AT THE BACKWARD SHAPE
# ===========================================================================


def _g3_case(
    ctx: DeviceContext,
    which: Int,
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    label: String,
    exact: Bool,
    mut cases: Int,
    mut fails: Int,
    mut unwritten: Int,
) raises:
    var dc = _bwd_dc(which, op, m, n, k, exact, 101)
    var w = _bwd_w(which, op, m, n, k, exact, 101)
    _g3_case_vals(ctx, which, dc, w, op, m, n, k, label, cases, fails, unwritten)


def _g3_case_vals(
    ctx: DeviceContext,
    which: Int,
    hdc: List[Float32],
    hw: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    label: String,
    mut cases: Int,
    mut fails: Int,
    mut unwritten: Int,
) raises:
    var call = _bwd_route(which, op, m, n, k)
    var want = _routed_host(call, hdc, hw)
    var before = unwritten
    var got = _run_bwd(ctx, which, hdc, hw, op, m, n, k, label, unwritten)
    var out_log = _bwd_out_logical(which, op, m, n, k)
    var bad = 0
    var first = -1
    var nc = out_log
    if len(want) < nc:
        nc = len(want)
    for c in range(nc):
        if _bits(got[c]) != _bits(want[c]):
            bad += 1
            if first < 0:
                first = c
    var part = contract_partition(call[3])
    var never = unwritten - before
    cases += 1
    if bad == 0 and never == 0 and len(want) >= out_log:
        print(
            "    OK   "
            + label
            + "  "
            + gemm_backward_call_name(call)
            + "  L="
            + String(part[0])
            + " P="
            + String(part[1])
            + "  plan="
            + gemm_plan_name(choose_gemm_plan(call[1], call[2], call[3]))
        )
        return
    fails += 1
    var detail = String("")
    if first >= 0:
        detail = (
            ". First at flat index "
            + String(first)
            + ": device = "
            + _show(got[first])
            + "   oracle = "
            + _show(want[first])
        )
    print(
        "    FAIL "
        + label
        + "  "
        + gemm_backward_call_name(call)
        + "  L="
        + String(part[0])
        + " P="
        + String(part[1])
        + ": "
        + String(bad)
        + " of "
        + String(out_log)
        + " cells differ from gemm_oracle at the backward shape, and "
        + String(never)
        + " were NEVER WRITTEN (still POISON)"
        + detail
    )


def _minus_zero_row(k: Int, el: Int) -> List[Float32]:
    """A vector of length `k` such that EVERY leaf's partial against a vector
    of ones is exactly `-0.0`. DEVIATION 1063.

    Lifted from `gemm_device_check.mojo::_minus_zero_leaves` so the forward
    fixture and the backward fixture are the same construction and not two
    authors' opinions about how to reach a negative zero.

    Contract 9.2(a): a `+0.0`-seeded leaf can NEVER reach `-0.0` from
    products, so the only route is `ftz` of a negative subnormal accumulator.
    Each leaf ends with `2^-126` then `-1.5 * 2^-126`, giving `-2^-127`,
    which the flush turns into `-0.0`.

    THE PLACEMENT AT THE END OF THE LEAF IS LOAD BEARING. At the START the
    trailing `0 * 1` products erase the sign, because `fma(0, 1, -0.0)` is
    `+0.0 + (-0.0)` is `+0.0`, and the fixture becomes all `+0.0` partials,
    which separates nothing.
    """
    var a = _fill_const(k, Float32(0.0))
    var p = contract_leaf_count(k)
    for j in range(p):
        var e = (j + 1) * el
        if e > k:
            e = k
        a[e - 2] = bitcast[DType.float32](UInt32(0x00800000))
        a[e - 1] = -bitcast[DType.float32](UInt32(0x00C00000))
    return a^


def _minus_zero_db_case(
    ctx: DeviceContext,
    t: Int,
    n: Int,
    label: String,
    mut cases: Int,
    mut fails: Int,
    mut unwritten: Int,
) raises:
    """**THE ONLY FIXTURE IN THIS FILE THAT CAN TELL A CARRY FROM `+0.0`
    PADDING**, routed through `dB` so that the CARRY is exercised on the
    backward side rather than inferred from the forward side.

    Forward `OP_NN` at `(m, n, k) = (t, n, 1)`. `A` is `t x 1`, so `A`'s
    single column is the planted vector and the `dB` call at `(1, n, t)`
    contracts over the tokens with `dC` all ones. Every leaf partial is then
    `-0.0` and the whole `1 x n` output must be `-0.0`.

    `x + (+0.0) == x` for every finite float, every infinity and every NaN.
    It differs on exactly ONE value, `x = -0.0`, where `(-0) + (+0) = +0`. So
    a `PAD_PLUS_ZERO` build passes every other fixture in this file, exactly
    as it passed all 42 shapes of the forward gate's first sweep with EXIT
    CODE 0.

    THE FIXTURE REFUSES ITSELF if the oracle does not come back `-0.0`. A
    vacuous fixture that reports green is worse than no fixture.
    """
    var el = contract_leaf_size(t)
    var p_count = contract_leaf_count(t)
    var last = t - (p_count - 1) * el
    if p_count < 1 or last < 2:
        raise Error(
            "_minus_zero_db_case ["
            + label
            + "]: the construction needs EVERY leaf to hold at least two"
            " elements and the last leaf of t = "
            + String(t)
            + " holds "
            + String(last)
            + ". Pick a token count whose ragged tail is >= 2."
        )
    var a = _minus_zero_row(t, el)
    var dc = _fill_const(t * n, Float32(1.0))
    var call = _bwd_route(BWD_B, OP_NN, t, n, 1)
    var need = _route_needs(call)
    var dc_need = need[1]
    var a_need = need[0]
    if call[4] == BWD_DC_LEFT:
        dc_need = need[0]
        a_need = need[1]
    var pdc = _pad_to(dc, dc_need)
    var pa = _pad_to(a, a_need)
    var want = _routed_host(call, pdc, pa)
    comptime if IDENTICAL_BUILD:
        for c in range(n):
            if _bits(want[c]) != UInt32(0x80000000):
                raise Error(
                    "_minus_zero_db_case ["
                    + label
                    + "]: the oracle returned "
                    + _show(want[c])
                    + " at cell "
                    + String(c)
                    + " and the construction predicts -0.0 (0x80000000). The"
                    " fixture's leaf partials are not -0.0, so it separates a"
                    " CARRY from +0.0 PADDING on nothing, and reporting it"
                    " green would be reporting the opposite of the truth."
                )
    _g3_case_vals(
        ctx, BWD_B, pdc, pa, OP_NN, t, n, 1, label, cases, fails, unwritten
    )


def check_backward_matches_oracle() raises:
    """**G3.** Every cell of `dA`, `dB` and `db` equals `gemm_oracle`
    evaluated at the BACKWARD `(op', m', n', k')` on the same operand bytes.

    This is the identity gate and it is the backward twin of
    `gemm_device_check.mojo::check_device_matches_oracle`. It says the device
    computes the profile at the backward shapes. **It does not say the
    gradient is right**; a transpose error passes this gate on three vendors
    and G2 is the only thing that catches it.

    THE SHAPE LIST IS DRIVEN BY `k'`, NOT BY `k`. `k'` is `n` for a `dA` call
    and `m` for a `dB` call, so to put a `dB` call at `P = 3` the FORWARD
    shape needs 300 TOKENS, and to put a `dA` call there the forward shape
    needs an output width of 300. That inversion is the whole reason this
    gate cannot reuse the forward gate's shape list, and getting it wrong
    produces a sweep that looks thorough and exercises `P = 1` throughout.

    The list forces, by construction and per route:

        k' = 0        P = 0, and every cell must be a STORED +0.0
        k' = 1        P = 1, the fold that performs no addition
        k' = 128      P = 1, the largest single leaf
        k' = 129      P = 2, ragged
        k' = 300      P = 3, the ragged odd case contract 12.1 names
        k' = 517      P = 5, the smallest P that carries TWICE
        k' = 4097     P = 33, odd with a ONE-ELEMENT last leaf

    plus ragged `m` and `n` that no tile divides, plus one all-`-0.0` leaf
    fixture routed through `dB`, plus one exact-integer fixture per route so
    that this gate ALSO fails if the routing moved (a belt the mask in G2
    already provides, kept because G3 is the gate an operator runs first).

    ASSERTED under IDENTICAL, REPORTED under FAST, for the reason
    `gemm_device_check.mojo` gives at the same seam: under FAST both sides
    are the unpinned spelling and whether the host CPU and the device backend
    agree is a measurement rather than a bug.

    SABOTAGES THAT MUST FAIL: the forward six, invoked THROUGH the backward
    launchers, which is the proof that the backward path reaches the
    contract's arithmetic rather than some other path that agrees with it.
    `FOLD_STRIDE` and `PAD_PLUS_ZERO` need the odd-`P` shapes and the
    `-0.0` fixture respectively, and both are in the list above.
    """
    print(
        "check_backward_matches_oracle ["
        + _mode_name()
        + "]: per-cell bits against gemm_oracle at the backward shape"
    )
    var cases = 0
    var fails = 0
    var unwritten = 0
    with DeviceContext() as ctx:
        print("  -- dA, k' = n: the output width drives the partition --")
        for oi in range(3):
            var op = _tbl_op3(oi)
            _g3_case(ctx, BWD_A, op, 5, 0, 7, String("dA.") + op_name(op) + ".n0.P0", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_A, op, 3, 1, 5, String("dA.") + op_name(op) + ".n1.P1", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_A, op, 5, 128, 3, String("dA.") + op_name(op) + ".n128.P1", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_A, op, 5, 129, 3, String("dA.") + op_name(op) + ".n129.P2", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_A, op, 5, 300, 3, String("dA.") + op_name(op) + ".n300.P3.odd", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_A, op, 17, 517, 33, String("dA.") + op_name(op) + ".n517.P5.2carries", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_A, op, 2, 4097, 3, String("dA.") + op_name(op) + ".n4097.P33.tail1", False, cases, fails, unwritten)
        print("  -- dB, k' = m: the TOKEN COUNT drives the partition --")
        for oi2 in range(3):
            var op2 = _tbl_op3(oi2)
            _g3_case(ctx, BWD_B, op2, 0, 5, 7, String("dB.") + op_name(op2) + ".m0.P0", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_B, op2, 1, 3, 5, String("dB.") + op_name(op2) + ".m1.P1", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_B, op2, 128, 5, 3, String("dB.") + op_name(op2) + ".m128.P1", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_B, op2, 129, 5, 3, String("dB.") + op_name(op2) + ".m129.P2", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_B, op2, 300, 5, 3, String("dB.") + op_name(op2) + ".m300.P3.odd", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_B, op2, 517, 17, 33, String("dB.") + op_name(op2) + ".m517.P5.2carries", False, cases, fails, unwritten)
            _g3_case(ctx, BWD_B, op2, 4097, 2, 3, String("dB.") + op_name(op2) + ".m4097.P33.tail1", False, cases, fails, unwritten)
        print("  -- the bias, k' = m --")
        _g3_case(ctx, BWD_BIAS, OP_NN, 5, 7, 3, String("db.m5.P1"), False, cases, fails, unwritten)
        _g3_case(ctx, BWD_BIAS, OP_NN, 300, 7, 3, String("db.m300.P3.odd"), False, cases, fails, unwritten)
        _g3_case(ctx, BWD_BIAS, OP_NN, 517, 33, 3, String("db.m517.P5.2carries"), False, cases, fails, unwritten)
        print("  -- exact-integer fixtures: a moved ROUTE fails here too --")
        for oi3 in range(3):
            var op3 = _tbl_op3(oi3)
            _g3_case(ctx, BWD_A, op3, 5, 7, 3, String("dA.exact.") + op_name(op3), True, cases, fails, unwritten)
            _g3_case(ctx, BWD_B, op3, 5, 7, 3, String("dB.exact.") + op_name(op3), True, cases, fails, unwritten)
        print(
            "  -- EVERY LEAF PARTIAL IS -0.0, routed through dB (contract"
            " 9.2) -- the ONLY input that separates a CARRY from +0.0"
            " PADDING --"
        )
        _minus_zero_db_case(ctx, 300, 3, String("dB.minuszero.m300.P3.carry"), cases, fails, unwritten)
        _minus_zero_db_case(ctx, 517, 3, String("dB.minuszero.m517.P5.2carries"), cases, fails, unwritten)
        _minus_zero_db_case(ctx, 128, 3, String("dB.minuszero.m128.P1.nofold"), cases, fails, unwritten)
        _minus_zero_db_case(ctx, 4200, 3, String("dB.minuszero.m4200.P33.odd"), cases, fails, unwritten)

    if fails == 0 and unwritten == 0:
        print(
            "check_backward_matches_oracle ["
            + _mode_name()
            + "] OK: "
            + String(cases)
            + " backward calls, every cell bit-identical to gemm_oracle at"
            " the backward shape, no cell left unwritten"
        )
        return
    comptime if IDENTICAL_BUILD:
        raise Error(
            "check_backward_matches_oracle [IDENTICAL]: "
            + String(fails)
            + " of "
            + String(cases)
            + " backward calls DISAGREE with gemm_oracle and "
            + String(unwritten)
            + " output cells were never written (per-case lines above). The"
            " backward path is not computing"
            " mojolearn.identical.gemm.fp32.v1."
        )
    print(
        "check_backward_matches_oracle [FAST] REPORT: "
        + String(fails)
        + " of "
        + String(cases)
        + " calls differ from the host oracle and "
        + String(unwritten)
        + " cells were never written. Under FAST both sides are the unpinned"
        " spelling, so the first number is a measurement; the second is a"
        " defect in either mode."
    )


def _tbl_op3(i: Int) -> Int:
    if i == 1:
        return OP_NT
    if i == 2:
        return OP_TN
    return OP_NN


# ===========================================================================
# GATE 4: THE BACKWARD SHAPES ARE LAUNCH INVARIANT
# ===========================================================================


def _run_plan(
    ctx: DeviceContext,
    call: Tuple[Int, Int, Int, Int, Int],
    hdc: List[Float32],
    hw: List[Float32],
    plan: Int,
    mut unwritten: Int,
) raises -> List[Float32]:
    """One launch of the ROUTED call on a NAMED plan, poisoned first.

    **THIS DELIBERATELY BYPASSES THE BACKWARD LAUNCHERS**, because they take
    no plan argument and must not grow one: naming an execution plan is a
    gate's privilege and a caller's mistake. So G4 asks the kernel's own
    question at the backward SHAPES, and G3 is what asks the launcher's
    question. The operand placement here is a SECOND spelling of the side
    flag, and if it were wrong G4 would test launch invariance at a different
    but still legal shape, which is a weaker result and not a false one. G3
    is the gate where a wrong side is a failure.
    """
    var need = _route_needs(call)
    var out_alloc = need[2]
    if out_alloc < 1:
        out_alloc = 1
    # `.copy()` is explicit because `List[Float32]` is not implicitly
    # copyable (it does not conform to `ImplicitlyCopyable`), and the four
    # bindings below are the exact shape that failed three sibling gate
    # files last week. A gate fixture is small (G4's largest operand is
    # about 10k floats), so the copy is bought deliberately rather than
    # worked around with a swap flag that would fork every read below.
    var left = hdc.copy()
    var right = hw.copy()
    if call[4] == BWD_DC_RIGHT:
        left = hw.copy()
        right = hdc.copy()
    var nl = len(left)
    if nl < 1:
        nl = 1
    var nr = len(right)
    if nr < 1:
        nr = 1
    var nws = identical_gemm_workspace_floats(call[1], call[2], call[3], plan)
    if nws < 1:
        nws = 1
    var dl = ctx.enqueue_create_buffer[DType.float32](nl)
    var dr = ctx.enqueue_create_buffer[DType.float32](nr)
    var dout = ctx.enqueue_create_buffer[DType.float32](out_alloc)
    var dws = ctx.enqueue_create_buffer[DType.float32](nws)
    var hL = ctx.enqueue_create_host_buffer[DType.float32](nl)
    var hR = ctx.enqueue_create_host_buffer[DType.float32](nr)
    var hO = ctx.enqueue_create_host_buffer[DType.float32](out_alloc)
    ctx.synchronize()
    for i in range(len(left)):
        hL.unsafe_ptr().unsafe_store(i, left[i])
    for i in range(len(right)):
        hR.unsafe_ptr().unsafe_store(i, right[i])
    for i in range(out_alloc):
        hO.unsafe_ptr().unsafe_store(i, POISON)
    ctx.enqueue_copy(dst_buf=dl, src_ptr=hL.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dr, src_ptr=hR.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dout, src_ptr=hO.unsafe_ptr())
    ctx.synchronize()
    identical_gemm_with_plan(
        ctx, dout, dl, dr, dws, call[1], call[2], call[3], call[0], plan
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hO.unsafe_ptr(), src_buf=dout)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(need[2]):
        var v = hO.unsafe_ptr().unsafe_load(i)
        if _bits(v) == _bits(POISON):
            unwritten += 1
        out.append(v)
    _ = dl
    _ = dr
    _ = dout
    _ = dws
    _ = hL
    _ = hR
    _ = hO
    return out^


def check_backward_is_launch_invariant() raises:
    """**G4, AND THE PROPERTY THE PROFILE EXISTS FOR, ASSERTED ON BACKWARD
    SHAPES.** Each of the six routes, run under all eight named execution
    plans, must produce ONE byte pattern.

    THE SHAPE IS CHOSEN SO THAT THE SIX ROUTES ARE NOT SIX SPELLINGS OF ONE
    LAUNCH. Forward `(m, n, k) = (520, 517, 17)` gives every `dA` route
    `k' = 517` and every `dB` route `k' = 520`, both `P = 5`, which is the
    smallest `P` that CARRIES TWICE and is therefore the fold shape a
    launch-dependent tree is most likely to get wrong. The `(m', n')` of the
    six routes are `520x17`, `520x17`, `17x520`, `17x517`, `517x17` and
    `17x517`, so the dispatcher's own choice differs across them and the
    eight forced plans cross every tile shape, both grid ranks, all three
    tile orders and both fold locations.

    `PLAN_FLAT` against `PLAN_SPLITK` is the pair that matters most: one
    merges leaf partials into an occupancy-indexed register stack and never
    writes a partial anywhere, the other materializes `m' n' P` named
    partials and folds them level by level through a global workspace. Two
    structurally unrelated realizations of one arithmetic DAG, required to
    agree bit for bit at a shape whose `k` is a token count.

    ASSERTED IN BOTH MODES. Nothing here depends on `fma` or on `ftz`; it is
    a claim about the SHAPE of the kernel and FAST has the same shape.

    SABOTAGES: `LEAF_ROTATE` and `NODE_ORDER`, the two forward arms that move
    a bit only when the launch changes. Neither backward sabotage can reach
    this gate, because all eight plans are given the SAME route: a routing
    defect moves all eight together and is invisible to a comparison of the
    eight against each other. That is not a weakness, it is the division of
    labor, and it is why the ledger says `BWD_*` arms are owned by G1, G2, G3
    and G6.
    """
    print(
        "check_backward_is_launch_invariant ["
        + _mode_name()
        + "]: "
        + String(GEMM_PLAN_COUNT)
        + " execution plans, one numerical plan, at the backward shapes"
    )
    var fails = 0
    var unwritten = 0
    var runs = 0
    var m = 520
    var n = 517
    var k = 17
    with DeviceContext() as ctx:
        for r in range(6):
            var op = _route_op(r)
            var which = BWD_A
            if r >= 3:
                which = BWD_B
            var call = _bwd_route(which, op, m, n, k)
            var idc = _bwd_dc(which, op, m, n, k, False, 211)
            var iw = _bwd_w(which, op, m, n, k, False, 211)
            var reference = _run_plan(ctx, call, idc, iw, PLAN_FLAT, unwritten)
            var part = contract_partition(call[3])
            print(
                "  "
                + _route_name(r)
                + "  "
                + gemm_backward_call_name(call)
                + "  L="
                + String(part[0])
                + " P="
                + String(part[1])
            )
            for plan in range(GEMM_PLAN_COUNT):
                if plan == PLAN_FLAT:
                    continue
                var got = _run_plan(ctx, call, idc, iw, plan, unwritten)
                runs += 1
                var bad = 0
                var first = -1
                for c in range(len(reference)):
                    if _bits(got[c]) != _bits(reference[c]):
                        bad += 1
                        if first < 0:
                            first = c
                if bad == 0:
                    print("      OK   " + gemm_plan_name(plan))
                else:
                    fails += 1
                    print(
                        "      FAIL "
                        + gemm_plan_name(plan)
                        + ": "
                        + String(bad)
                        + " of "
                        + String(len(reference))
                        + " cells differ from "
                        + gemm_plan_name(PLAN_FLAT)
                        + ". First at flat index "
                        + String(first)
                        + ": this plan = "
                        + _show(got[first])
                        + "   FLAT = "
                        + _show(reference[first])
                    )
    if fails != 0 or unwritten != 0:
        raise Error(
            "check_backward_is_launch_invariant ["
            + _mode_name()
            + "]: "
            + String(fails)
            + " plan/route combinations produced DIFFERENT BITS from the"
            " same logical problem and "
            + String(unwritten)
            + " cells were never written. Launch geometry has reached the"
            " arithmetic through the backward shapes, which is the one thing"
            " profile mojolearn.identical.gemm.fp32.v1 exists to prevent."
        )
    print(
        "check_backward_is_launch_invariant ["
        + _mode_name()
        + "] OK: 6 routes x "
        + String(GEMM_PLAN_COUNT)
        + " execution plans ("
        + String(runs)
        + " comparisons), all bit-identical"
    )


# ===========================================================================
# GATE 5: `dB`'s BITS ARE A FUNCTION OF THE TOKEN COUNT, AND `dA`'s ARE NOT
# ===========================================================================
# `IDENTICAL_BACKWARD_PLAN.md` section 3.2 point 2 says:
#
#   > `dB` at `T` tokens is not the same bits as two `dB` calls at `T/2`
#   > summed. It cannot be, under any partition scheme, because those are two
#   > different sums of the same terms in a different order.
#
# **THAT SENTENCE IS TOO STRONG AND THIS GATE IS WRITTEN TO SHOW IT.**
# DEVIATION 1056. The prediction, on paper, before any run:
#
# v1's leaf rule is `L = 128` for every `k` up to 131,072, and v1's fold is a
# balanced binary tree over adjacent leaves. So a split at a token boundary
# that is BOTH a leaf boundary AND a subtree boundary of that tree reproduces
# the unsplit tree exactly, provided the accumulation across microbatches is
# spelled as the fold's own flushed add. Worked for `T = 512` split `256/256`:
#
#     unsplit    L=128, P=4, leaves L0..L3
#                node(1,0) = ftz(ftz(L0) + ftz(L1))
#                node(1,1) = ftz(ftz(L2) + ftz(L3))
#                out       = ftz(ftz(node(1,0)) + ftz(node(1,1)))
#     half 1     k'=256, L=128, P=2  ->  ftz(node(1,0))
#     half 2     k'=256, L=128, P=2  ->  ftz(node(1,1))
#     accum      ftz(ftz(half1) + ftz(half2))  ==  out,  ftz being idempotent
#
# and for `T = 384` split `256/128`, where the second half is a single leaf
# and the unsplit tree's level 1 carries it unchanged. **Predicted: ZERO
# cells move in both.**
#
# A split at 150 of 300, or at 200 of 512, lands inside a leaf, and then the
# two partitions share no boundary at all. **Predicted: nearly every cell
# moves**, and the HOST ORACLE is what says how many, exactly, before the
# device is asked.
#
# The consequence for T11 is sharper than the plan's, not weaker: the
# microbatch schedule is part of a training run's numerical specification
# UNLESS the accumulation factor divides the token count into leaf-aligned
# subtree-aligned pieces and the accumulator is the contract's own add, in
# which case it is free. That is a designable property and somebody building
# a trainer would want to know it.


def _pad_dc_for(
    which: Int, op: Int, m: Int, n: Int, k: Int, dc: List[Float32]
) -> List[Float32]:
    """Pad a caller-built `dC` to whatever THIS BUILD's route reads.

    In a clean build every one of the six routes reads exactly the logical
    extent and this is the identity. It stops being the identity the moment a
    sabotage moves the shape, and an arm that skipped it would read past the
    end of `dC` under `BWD_OPERAND_ORDER` at `k > m`. See `_pad_to`.
    """
    return _pad_to(dc, _dc_floats_needed(which, op, m, n, k))


def _pad_w_for(
    which: Int, op: Int, m: Int, n: Int, k: Int, w: List[Float32]
) -> List[Float32]:
    return _pad_to(w, _w_floats_needed(which, op, m, n, k))


def _slice(v: List[Float32], begin: Int, count: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(count):
        out.append(v[begin + i])
    return out^


def _accum_two(x: List[Float32], y: List[Float32]) -> List[Float32]:
    """The gradient-accumulation add across two microbatches, spelled as
    `fold_balanced_tree` over two elements. DEVIATION 1057.

    NOT a bare `+`. `fold_balanced_tree([x, y])` is `ftz(ftz(x) + ftz(y))`,
    which is exactly the arithmetic node the unsplit call performs at the
    level where the two halves meet. Using the contract's own function rather
    than re-spelling it is what makes the aligned prediction a structural
    claim instead of a coincidence, and it is what T11 will have to adopt if
    a trainer wants the aligned case to stay free.

    A trainer that accumulates with a bare `+` gets the same bits everywhere
    except where a subnormal appears, because `ftz` is then the only
    difference. That is a smaller gap than it sounds and it is still a gap,
    and it is a reason T11 is a PIN and not a "usually fine".
    """
    var out = List[Float32]()
    for i in range(len(x)):
        var pair = List[Float32]()
        pair.append(x[i])
        pair.append(y[i])
        out.append(fold_balanced_tree(pair))
    return out^


def _db_split_arm(
    ctx: DeviceContext,
    t: Int,
    t1: Int,
    n: Int,
    k: Int,
    exact: Bool,
    predict_zero: Bool,
    label: String,
    mut fails: Int,
    mut unwritten: Int,
) raises:
    """One split arm. Forward `OP_NN` at `(t, n, k)`, split into `t1` and
    `t - t1` tokens, `dB` accumulated by `_accum_two`.

    THE VACUITY GUARD IS THE HOST ORACLE AND IT RUNS FIRST. The same
    comparison is evaluated on the host, through `gemm_oracle` at both
    shapes, and its cell count is the PREDICTION. The device then has to
    reproduce that number exactly. A device that agrees where the host says
    it must differ is as much a failure as one that differs where the host
    says it must agree, and `predict_zero` is which of the two this arm
    claims.
    """
    var a = _fill(t * k, 301)
    var dc = _fill(t * n, 302)
    if exact:
        a = _fill_exact(t * k, 301)
        dc = _fill_exact(t * n, 302)
        _exact_bound(t, label)
    var a1 = _slice(a, 0, t1 * k)
    var a2 = _slice(a, t1 * k, (t - t1) * k)
    var dc1 = _slice(dc, 0, t1 * n)
    var dc2 = _slice(dc, t1 * n, (t - t1) * n)

    # THE HOST PREDICTION.
    var h_full = _routed_host(_bwd_route(BWD_B, OP_NN, t, n, k), dc, a)
    var h_1 = _routed_host(_bwd_route(BWD_B, OP_NN, t1, n, k), dc1, a1)
    var h_2 = _routed_host(_bwd_route(BWD_B, OP_NN, t - t1, n, k), dc2, a2)
    var h_acc = _accum_two(h_1, h_2)
    var cells = k * n
    var h_bad = 0
    var h_first = -1
    for c in range(cells):
        if _bits(h_full[c]) != _bits(h_acc[c]):
            h_bad += 1
            if h_first < 0:
                h_first = c

    if predict_zero and h_bad != 0:
        raise Error(
            "_db_split_arm ["
            + label
            + "]: this arm predicts a LEAF-ALIGNED, SUBTREE-ALIGNED split"
            " and the HOST ORACLE moved "
            + String(h_bad)
            + " of "
            + String(cells)
            + " cells (first at "
            + String(h_first)
            + ": unsplit "
            + _show(h_full[h_first])
            + ", accumulated "
            + _show(h_acc[h_first])
            + "). The alignment argument in this section's block comment is"
            " WRONG, which means IDENTICAL_BACKWARD_PLAN.md section 3.2"
            " point 2 was right as written and this file's finding has to be"
            " withdrawn. Do not adjust the fixture. Report it."
        )
    if not predict_zero and not exact and h_bad == 0:
        raise Error(
            "_db_split_arm ["
            + label
            + "]: VACUOUS. This arm predicts a MISALIGNED split and the host"
            " oracle found ZERO of "
            + String(cells)
            + " cells moved, so the fixture cannot separate the two"
            " summation orders and the negative claim is untested. This is"
            " the mamba lane's inert-arm failure with a different subject."
        )
    if exact and h_bad != 0:
        raise Error(
            "_db_split_arm ["
            + label
            + "]: the EXACT-fixture control moved "
            + String(h_bad)
            + " of "
            + String(cells)
            + " cells. On integers every summation order is the same bits,"
            " so this is not a rounding-order difference, it is a BUG IN THE"
            " SPLIT: the two halves are not covering the same terms as the"
            " whole. Every rounding number measured after this point would"
            " be measuring that bug."
        )

    # THE DEVICE, required to reproduce the host's exact count.
    var d_full = _run_bwd(ctx, BWD_B, _pad_dc_for(BWD_B, OP_NN, t, n, k, dc), _pad_w_for(BWD_B, OP_NN, t, n, k, a), OP_NN, t, n, k, label, unwritten)
    var d_1 = _run_bwd(ctx, BWD_B, _pad_dc_for(BWD_B, OP_NN, t1, n, k, dc1), _pad_w_for(BWD_B, OP_NN, t1, n, k, a1), OP_NN, t1, n, k, label, unwritten)
    var d_2 = _run_bwd(ctx, BWD_B, _pad_dc_for(BWD_B, OP_NN, t - t1, n, k, dc2), _pad_w_for(BWD_B, OP_NN, t - t1, n, k, a2), OP_NN, t - t1, n, k, label, unwritten)
    var d_acc = _accum_two(d_1, d_2)
    var d_bad = 0
    var d_first = -1
    for c in range(cells):
        if _bits(d_full[c]) != _bits(d_acc[c]):
            d_bad += 1
            if d_first < 0:
                d_first = c
    var verdict = String("OK  ")
    if d_bad != h_bad or d_first != h_first:
        fails += 1
        verdict = String("FAIL")
    print(
        "    "
        + verdict
        + " "
        + label
        + "  T="
        + String(t)
        + " split "
        + String(t1)
        + "/"
        + String(t - t1)
        + "  host says "
        + String(h_bad)
        + " of "
        + String(cells)
        + " cells move (first "
        + String(h_first)
        + "), device says "
        + String(d_bad)
        + " (first "
        + String(d_first)
        + ")"
    )
    if d_bad != 0 and d_first >= 0:
        print(
            "         first moved cell: unsplit "
            + _show(d_full[d_first])
            + "   accumulated "
            + _show(d_acc[d_first])
        )


def _da_split_arm(
    ctx: DeviceContext,
    t: Int,
    t1: Int,
    n: Int,
    k: Int,
    label: String,
    mut fails: Int,
    mut unwritten: Int,
) raises:
    """`dA` under the SAME split, where the prediction is ZERO moved cells.

    `dA`'s `k'` is `n`, the output width, which a batch split does not touch.
    Splitting the batch splits `dA`'s `m'` instead, and `m'` is the one thing
    contract 6.1 forbids the leaf rule to read. So row `i` of `dA` must come
    back bit identical whether it was computed in a launch of `t` rows or one
    of `t1`. **Predicted: 0 of `t * k` cells move.**

    This is the forward's batch invariance re-asserted through the backward
    routing, and it is the arm that makes G5 a statement about `dB`
    specifically rather than about splitting in general. Without it the gate
    would show that splitting changes bits and would not show WHICH gradient
    it changes, which is the whole content of section 3.2.
    """
    var b = _fill(k * n, 401)
    var dc = _fill(t * n, 402)
    var dc1 = _slice(dc, 0, t1 * n)
    var dc2 = _slice(dc, t1 * n, (t - t1) * n)
    var full = _run_bwd(ctx, BWD_A, _pad_dc_for(BWD_A, OP_NN, t, n, k, dc), _pad_w_for(BWD_A, OP_NN, t, n, k, b), OP_NN, t, n, k, label, unwritten)
    var p1 = _run_bwd(ctx, BWD_A, _pad_dc_for(BWD_A, OP_NN, t1, n, k, dc1), _pad_w_for(BWD_A, OP_NN, t1, n, k, b), OP_NN, t1, n, k, label, unwritten)
    var p2 = _run_bwd(ctx, BWD_A, _pad_dc_for(BWD_A, OP_NN, t - t1, n, k, dc2), _pad_w_for(BWD_A, OP_NN, t - t1, n, k, b), OP_NN, t - t1, n, k, label, unwritten)
    var bad = 0
    var first = -1
    for c in range(t1 * k):
        if _bits(full[c]) != _bits(p1[c]):
            bad += 1
            if first < 0:
                first = c
    for c2 in range((t - t1) * k):
        if _bits(full[t1 * k + c2]) != _bits(p2[c2]):
            bad += 1
            if first < 0:
                first = t1 * k + c2
    if bad == 0:
        print(
            "    OK   "
            + label
            + "  T="
            + String(t)
            + " split "
            + String(t1)
            + "/"
            + String(t - t1)
            + ": 0 of "
            + String(t * k)
            + " dA cells move, as predicted"
        )
        return
    fails += 1
    var split_val = Float32(0.0)
    if first < t1 * k:
        split_val = p1[first]
    else:
        split_val = p2[first - t1 * k]
    print(
        "    FAIL "
        + label
        + ": "
        + String(bad)
        + " of "
        + String(t * k)
        + " dA cells CHANGED BITS with the batch split. First at flat index "
        + String(first)
        + ": unsplit "
        + _show(full[first])
        + "   split "
        + _show(split_val)
        + ". dA's k' is the OUTPUT WIDTH and a batch split does not touch it,"
        " so this is a leaf rule that has started reading m'."
    )


def check_backward_b_depends_on_the_token_count_and_says_so() raises:
    """**G5.** A gate on a NEGATIVE property, so that
    `IDENTICAL_BACKWARD_PLAN.md` section 3.2 is measured rather than
    asserted, and on the POSITIVE property beside it so the negative one
    means something.

    FOUR CLAIMS, and the second is new:

    1. `dB` at `T` tokens differs from two `dB` calls at `T/2` accumulated,
       when the split lands INSIDE a leaf. The host oracle predicts the exact
       cell count and the device must reproduce it.
    2. **`dB` at `T` tokens is BIT IDENTICAL to two accumulated calls when
       the split is leaf aligned and subtree aligned and the accumulator is
       the contract's own flushed add.** Predicted zero moved cells at
       `512 = 256 + 256` and at `384 = 256 + 128`. If this holds, section 3.2
       point 2's "it cannot be, under any partition scheme" is false as
       written and has to be narrowed.
    3. `dA` under the same splits does not move at all, because its `k'` is
       the output width.
    4. The exact-integer control agrees under EVERY split, which is what
       makes claim 1's number a rounding-order measurement rather than a bug
       in the harness.

    ASSERTED IN BOTH MODES for claims 3 and 4, which are shape claims.
    Claims 1 and 2 are rounding claims and are asserted under IDENTICAL and
    reported under FAST.

    Record the measured difference as a number, because that number is the
    blast radius of a microbatch schedule change and somebody will ask for
    it.
    """
    print(
        "check_backward_b_depends_on_the_token_count_and_says_so ["
        + _mode_name()
        + "]: what a microbatch split costs, per cell"
    )
    var fails = 0
    var unwritten = 0
    with DeviceContext() as ctx:
        print("  -- claim 4: the EXACT control, every split must agree --")
        _db_split_arm(ctx, 512, 256, 7, 5, True, True, String("exact.aligned.512"), fails, unwritten)
        _db_split_arm(ctx, 300, 150, 7, 5, True, True, String("exact.misaligned.300"), fails, unwritten)
        print(
            "  -- claim 2: LEAF-ALIGNED and SUBTREE-ALIGNED splits, predicted"
            " ZERO moved cells --"
        )
        _db_split_arm(ctx, 512, 256, 7, 5, False, True, String("aligned.512.256+256"), fails, unwritten)
        _db_split_arm(ctx, 384, 256, 7, 5, False, True, String("aligned.384.256+128"), fails, unwritten)
        print(
            "  -- claim 1: MISALIGNED splits, predicted MANY moved cells, the"
            " host oracle says how many --"
        )
        _db_split_arm(ctx, 300, 150, 7, 5, False, False, String("misaligned.300.150+150"), fails, unwritten)
        _db_split_arm(ctx, 512, 200, 7, 5, False, False, String("misaligned.512.200+312"), fails, unwritten)
        _db_split_arm(ctx, 384, 192, 7, 5, False, False, String("misaligned.384.192+192"), fails, unwritten)
        print("  -- claim 3: dA under the same splits, predicted ZERO --")
        _da_split_arm(ctx, 512, 256, 7, 5, String("dA.aligned.512"), fails, unwritten)
        _da_split_arm(ctx, 300, 150, 7, 5, String("dA.misaligned.300"), fails, unwritten)
        _da_split_arm(ctx, 512, 200, 7, 9, String("dA.misaligned.512.crossplan"), fails, unwritten)
    if unwritten != 0:
        raise Error(
            "check_backward_b_depends_on_the_token_count_and_says_so: "
            + String(unwritten)
            + " output cells were never written."
        )
    if fails != 0:
        comptime if IDENTICAL_BUILD:
            raise Error(
                "check_backward_b_depends_on_the_token_count_and_says_so"
                " [IDENTICAL]: "
                + String(fails)
                + " arms disagreed with their prediction (lines above)."
                " Read the arm names before reading anything else: a FAIL on"
                " an `aligned.*` arm means the alignment finding is wrong and"
                " section 3.2 point 2 stands as written; a FAIL on a"
                " `misaligned.*` arm means the device and the host oracle"
                " disagree about how many cells a rounding-order change"
                " moves, which is a G3 failure wearing a different hat."
            )
        print(
            "check_backward_b_depends_on_the_token_count_and_says_so [FAST]"
            " REPORT: "
            + String(fails)
            + " arms differ from the host prediction. Under FAST the host"
            " and the device may contract differently and this is a"
            " measurement."
        )
        return
    print(
        "check_backward_b_depends_on_the_token_count_and_says_so ["
        + _mode_name()
        + "] OK: every arm matched its prediction, including the aligned"
        " splits at ZERO moved cells and dA at ZERO under every split"
    )


# ===========================================================================
# GATE 6: THE BIAS GRADIENT IS THE CONTRACT'S OWN SUM
# ===========================================================================


def _bias_direct_host(dc: List[Float32], m: Int, n: Int) raises -> List[Float32]:
    """**THE ARM THAT MATTERS. DEVIATION 1054.** `db[j]` computed from `dC`
    with NO ONES VECTOR ANYWHERE IN IT, under the contract's partition and
    the contract's fold.

        L = contract_leaf_size(m),  P = contract_leaf_count(m)
        leaf t:  acc = ftz(acc + ftz(dC[p][j])) for p ascending in the leaf
        db[j] = fold_balanced_tree(the P partials)

    WHY THIS IS BIT EQUAL TO THE ONES TRICK, AND NOT APPROXIMATELY SO. The
    contract's leaf evaluates `acc = ftz(identical_mul_add(ftz(1.0),
    ftz(x), acc))`. `ftz(1.0)` is `1.0`, the product `1.0 * x` is EXACT, and
    a fused multiply-add with an exact product is one correctly-rounded
    `x + acc` -- which is what the line above computes. So the two are the
    same rounding, term by term, and the fold above is literally the same
    function call the GEMM makes.

    **The equality holds under `NUMERIC_FAST` too**, which is worth stating
    because it means the FAST arm of the bias gradient is a correct sum and
    not a different one: under FAST `identical_mul_add` becomes `a * b + c`,
    and whether the backend contracts or not, `1.0 * x` is exact and the
    result is the same correctly-rounded add. G6 arm 2 is therefore
    ASSERTED IN BOTH MODES, unlike arm 1.

    Without this arm G6 only checks that a GEMM is a GEMM: arm 1 compares
    `identical_gemm_backward_bias_into` to `gemm_oracle` at the very shape
    the launcher passes it, which is true by construction of the launcher and
    says nothing about whether the ones trick is the reduction it claims.
    """
    var el = contract_leaf_size(m)
    var pc = contract_leaf_count(m)
    var out = _fill_const(n, Float32(0.0))
    for j in range(n):
        var partials = List[Float32]()
        for t in range(pc):
            var acc = Float32(0.0)
            for p in range(leaf_begin(t, el), leaf_end(t, el, m)):
                acc = ftz(acc + ftz(dc[p * n + j]))
            partials.append(ftz(acc))
        out[j] = fold_balanced_tree(partials)
    return out^


def _bias_arm(
    ctx: DeviceContext,
    m: Int,
    n: Int,
    ones_poison_at: Int,
    ones_poison_val: Float32,
    label: String,
    mut fails: Int,
    mut unwritten: Int,
) raises -> Int:
    """One bias arm. Returns how many of the `n` cells moved against the
    no-ones host reduction. `ones_poison_at < 0` means a clean ones vector."""
    var dc = _fill(m * n, 601)
    var ones = _fill_const(identical_gemm_backward_bias_ones_floats(m), Float32(1.0))
    if ones_poison_at >= 0:
        ones[ones_poison_at] = ones_poison_val
    var pdc = _pad_dc_for(BWD_BIAS, OP_NN, m, n, 1, dc)
    var pones = _pad_w_for(BWD_BIAS, OP_NN, m, n, 1, ones)
    var want_direct = _bias_direct_host(dc, m, n)
    var want_oracle = _routed_host(_bias_call(m, n), pdc, pones)
    var before = unwritten
    var got = _run_bwd(ctx, BWD_BIAS, pdc, pones, OP_NN, m, n, 1, label, unwritten)
    var never = unwritten - before
    var vs_direct = 0
    var vs_oracle = 0
    var first = -1
    for j in range(n):
        if _bits(got[j]) != _bits(want_direct[j]):
            vs_direct += 1
            if first < 0:
                first = j
        if j < len(want_oracle):
            if _bits(got[j]) != _bits(want_oracle[j]):
                vs_oracle += 1
    var verdict = String("OK  ")
    if ones_poison_at < 0 and not SAB_BWD_BIAS_AXIS:
        if vs_direct != 0 or vs_oracle != 0 or never != 0:
            fails += 1
            verdict = String("FAIL")
    print(
        "    "
        + verdict
        + " "
        + label
        + "  m="
        + String(m)
        + " n="
        + String(n)
        + "  vs the no-ones reduction: "
        + String(vs_direct)
        + " of "
        + String(n)
        + " moved   vs gemm_oracle at "
        + gemm_backward_call_name(_bias_call(m, n))
        + ": "
        + String(vs_oracle)
        + " moved   never written: "
        + String(never)
    )
    if first >= 0:
        print(
            "         first moved cell "
            + String(first)
            + ": device "
            + _show(got[first])
            + "   no-ones reduction "
            + _show(want_direct[first])
        )
    return vs_direct + never


def check_backward_bias_is_the_contract_sum() raises:
    """**G6.** The bias gradient is the contract's own column sum, and the
    ones vector is load bearing rather than decorative.

    FOUR ARMS.

    1. **Clean, `m != n`.** `db` equals the no-ones host reduction and
       equals `gemm_oracle` at the routed shape, on every cell. Predicted:
       0 moved, 0 unwritten.
    2. **Clean, `m == n`.** The same, at a shape where a wrong AXIS would
       still produce the right LENGTH. Predicted: 0 moved. This arm exists so
       that arm 4's `m == n` case has a clean twin and the difference between
       them is the sabotage and not the shape.
    3. **A ones vector poisoned to `2.0` at one entry.** Predicted: EXACTLY
       `n` of `n` cells move, because the change to `db[j]` is exactly
       `dC[r][j]`, `_val` never returns zero at these indices, and there is
       no cancellation available in a single extra term. This is the arm with
       an exact prediction. DEVIATION 1055.
    4. **A ones vector poisoned to `1.0000001` at one entry**, the realistic
       version of the same mistake: somebody filled the buffer with a float
       that is not quite one. **This arm is a MEASUREMENT and not an
       assertion**, because whether a relative perturbation of 1.2e-7 on one
       of `m` terms survives the rounding of the sum is a property of the
       fixture and not of the code. The count is printed so the answer is on
       record; `[[reached-but-inert]]` predicts it will be well below `n`.

    Plus, under `SAB_BWD_BIAS_AXIS` only, the two arms the sabotage is aimed
    at, with exact predictions:

        m=5,  n=7   the sabotage writes 5 row sums where 7 column sums are
                    expected. Predicted: 5 of 7 cells hold the wrong value
                    and 2 of 7 are NEVER WRITTEN, so 7 of 7 move.
        m=7,  n=7   the sabotage writes 7 row sums, the RIGHT LENGTH and the
                    wrong contents. Predicted: 7 of 7 move, 0 unwritten.
                    **This is the arm that proves the gate compares VALUES
                    and not shapes**, which `gemm_backward.mojo`'s own
                    docstring demands of it.

    ASSERTED under IDENTICAL. Arm 2's no-ones comparison is asserted under
    FAST as well, for the reason `_bias_direct_host` gives.
    """
    print(
        "check_backward_bias_is_the_contract_sum ["
        + _mode_name()
        + "]: the ones trick against a reduction with no ones in it"
    )
    var fails = 0
    var unwritten = 0
    var poison2_moved = 0
    var poison_eps_moved = 0
    var sab_57 = 0
    var sab_77 = 0
    with DeviceContext() as ctx:
        print("  -- arm 1: clean, m != n --")
        _ = _bias_arm(ctx, 5, 7, -1, Float32(1.0), String("clean.5x7"), fails, unwritten)
        _ = _bias_arm(ctx, 300, 7, -1, Float32(1.0), String("clean.300x7.P3"), fails, unwritten)
        _ = _bias_arm(ctx, 517, 33, -1, Float32(1.0), String("clean.517x33.P5"), fails, unwritten)
        print("  -- arm 2: clean, m == n, where a wrong axis has the right length --")
        sab_77 = _bias_arm(ctx, 7, 7, -1, Float32(1.0), String("clean.7x7"), fails, unwritten)
        print("  -- arm 3: ones poisoned to 2.0, predicted n of n --")
        poison2_moved = _bias_arm(ctx, 5, 7, 2, Float32(2.0), String("poison2.5x7"), fails, unwritten)
        print("  -- arm 4: ones poisoned to 1.0000001, a MEASUREMENT --")
        poison_eps_moved = _bias_arm(
            ctx, 5, 7, 2, bitcast[DType.float32](UInt32(0x3F800001)), String("poisonEps.5x7"), fails, unwritten
        )
        comptime if SAB_BWD_BIAS_AXIS:
            print("  -- the BWD_BIAS_AXIS arms --")
            sab_57 = _bias_arm(ctx, 5, 7, -1, Float32(1.0), String("sab.5x7.short.write"), fails, unwritten)
    comptime if SAB_BWD_BIAS_AXIS:
        if sab_57 != 7 or sab_77 != 7:
            raise Error(
                "check_backward_bias_is_the_contract_sum: BWD_BIAS_AXIS moved "
                + String(sab_57)
                + " of 7 at m=5,n=7 and "
                + String(sab_77)
                + " of 7 at m=7,n=7, and the ledger predicts 7 and 7. A"
                " count BELOW 7 at m == n is the arm being reached and inert"
                " on a value comparison; a count below 7 at m != n means the"
                " short write is not being seen, which is a POISON handling"
                " defect in this file and not in the launcher."
            )
        print(
            "check_backward_bias_is_the_contract_sum OK (INVERTED):"
            " BWD_BIAS_AXIS moved 7 of 7 at m != n (a short write) and 7 of 7"
            " at m == n (the right length, the wrong contents)"
        )
        return
    if poison2_moved != 7:
        raise Error(
            "check_backward_bias_is_the_contract_sum: the ones vector"
            " poisoned to 2.0 at one entry moved "
            + String(poison2_moved)
            + " of 7 cells and the prediction is EXACTLY 7. Below 7 means a"
            " column of dC contributed nothing, so the ones vector is not"
            " load bearing on that column and the clause"
            " `identical_gemm_backward_bias_ones_floats` documents ('a wrong"
            " value here is a wrong gradient with no symptom') is UNGATED"
            " there."
        )
    print(
        "  ones poisoned to 1.0000001 moved "
        + String(poison_eps_moved)
        + " of 7 cells. MEASUREMENT, not an assertion. A number well below 7"
        " is the honest answer that a nearly-one ones vector is a defect this"
        " gate can only partly see, and it is the reason arm 3 exists."
    )
    if fails != 0 or unwritten != 0:
        raise Error(
            "check_backward_bias_is_the_contract_sum ["
            + _mode_name()
            + "]: "
            + String(fails)
            + " clean arms disagreed and "
            + String(unwritten)
            + " cells were never written. If the disagreement is against the"
            " NO-ONES reduction and not against gemm_oracle, then DEVIATION"
            " 851 is wrong: the ones trick is a GEMM but it is not the"
            " contract's column sum, and IDENTICAL_BACKWARD_PLAN.md section"
            " 3.3 has to say so."
        )
    print(
        "check_backward_bias_is_the_contract_sum ["
        + _mode_name()
        + "] OK: db equals a reduction with no ones vector in it, bit for"
        " bit, and a poisoned one is caught"
    )


# ===========================================================================
# GATE 7: THE WORKSPACE HELPERS SIZE WHAT ACTUALLY RUNS
# ===========================================================================


def check_backward_workspace_sizing() raises:
    """**G7.** The backward workspace helpers return what the backward call's
    plan actually needs, at a shape where that is NOT the forward number, and
    one shared workspace carries `dA`, `dB` and `db` back to back with no
    synchronize between them.

    **THE DESTRUCTIVE ARM IS DELIBERATELY NOT WRITTEN. DEVIATION 1058.**
    `IDENTICAL_BACKWARD_PLAN.md` section 5's G7 asks for a deliberate
    under-allocation, showing the output come back with `+0.0` regions.
    Handing a too-small `DeviceBuffer` to a SPLITK dispatch is an
    out-of-bounds DEVICE WRITE, and undefined behavior is not a repeatable
    ledger entry: it can crash, it can return the right answer because the
    allocator had slack (which is exactly what happened on the forward side
    at `64 x 4` before `64 x 64` showed it), and it can do something
    different on each vendor. The repository's record of that under-
    allocation is a BUG IT HIT, not a test it keeps. What this gate asserts
    instead is the two halves that are checkable without corrupting memory:

      (a) THE NUMBER IS RIGHT. On the host, the helper's answer equals
          `identical_gemm_workspace_floats` at the ROUTED shape on the plan
          `choose_gemm_plan` will actually pick for that shape, and the
          combined helper is at least the max of the three.
      (b) THE NUMBER IS NOT THE FORWARD NUMBER. The gate SEARCHES a shape
          list for a case where the forward and backward numbers differ, and
          RAISES if none does, so it cannot pass by testing a coincidence.
          Section 5's own instruction: choose the shape by CHECKING that the
          two differ, not by assuming.
      (c) THE NUMBER IS SUFFICIENT. Every launch in this file already runs
          against a workspace allocated at exactly the helper's answer and
          POISONED before the launch (`_run_bwd`), so a plan that reads a
          slot it did not write reads `-987654.0` and the output is visibly
          wrong rather than plausibly zero.

    The shape the search is expected to find, worked on paper: forward
    `OP_NN` at `(m, n, k) = (520, 5, 17)`. The forward call has `m n = 2600`
    and `P(17) = 1`, so `choose_gemm_plan` refuses SPLITK for want of leaves
    and the forward workspace is the floor, 1. The `dB` route is `OP_TN` at
    `(17, 5, 520)`, where `m' n' = 85` and `P(520) = 5`, so it takes SPLITK
    and needs `85 * 5 = 425`. **425 against 1**, and a training step that
    sized its scratch from the forward shape would write 424 floats past the
    end of it.

    ARM 4, THE SHARED DIRTY WORKSPACE. DEVIATION 1059. `dA`, `dB` and `db`
    enqueued back to back on ONE context through ONE workspace with no
    synchronize between them, each compared to its own solo launch. That is
    the composition `identical_gemm_backward_workspace_max_floats` exists to
    license and its docstring's safety argument is exactly what this arm
    tests: MAX runs them in order, so the second call's leaf kernel cannot
    start before the first call's fold kernel has finished reading. The
    workspace is DIRTY for every launch after the first by construction.

    ASSERTED IN BOTH MODES. Workspace sizing is not arithmetic.
    """
    print(
        "check_backward_workspace_sizing ["
        + _mode_name()
        + "]: what the backward call's plan actually needs"
    )
    var shapes_m = [520, 64, 130, 5, 4096]
    var shapes_n = [5, 4, 70, 7, 3]
    var shapes_k = [17, 4096, 4096, 3, 5]
    var separations = 0
    var relation_fails = 0
    for s in range(len(shapes_m)):
        var m = shapes_m[s]
        var n = shapes_n[s]
        var k = shapes_k[s]
        for oi in range(3):
            var op = _tbl_op3(oi)
            var fwd = identical_gemm_workspace_max_floats(m, n, k)
            var ca = gemm_backward_a_call(op, m, n, k)
            var cb = gemm_backward_b_call(op, m, n, k)
            var wa = identical_gemm_backward_a_workspace_max_floats(op, m, n, k)
            var wb = identical_gemm_backward_b_workspace_max_floats(op, m, n, k)
            var wc = identical_gemm_backward_bias_workspace_max_floats(m, n)
            var want_a = identical_gemm_workspace_floats(
                ca[1], ca[2], ca[3], choose_gemm_plan(ca[1], ca[2], ca[3])
            )
            var want_b = identical_gemm_workspace_floats(
                cb[1], cb[2], cb[3], choose_gemm_plan(cb[1], cb[2], cb[3])
            )
            if wa < want_a or wb < want_b:
                relation_fails += 1
                print(
                    "    FAIL "
                    + op_name(op)
                    + " "
                    + String(m)
                    + "x"
                    + String(n)
                    + "x"
                    + String(k)
                    + ": the dA helper says "
                    + String(wa)
                    + " and the routed plan needs "
                    + String(want_a)
                    + "; the dB helper says "
                    + String(wb)
                    + " and the routed plan needs "
                    + String(want_b)
                )
            var combined = identical_gemm_backward_workspace_max_floats(
                op, m, n, k, True
            )
            if combined < wa or combined < wb or combined < wc:
                relation_fails += 1
                print(
                    "    FAIL "
                    + op_name(op)
                    + " "
                    + String(m)
                    + "x"
                    + String(n)
                    + "x"
                    + String(k)
                    + ": the COMBINED helper says "
                    + String(combined)
                    + " and the three separate ones say "
                    + String(wa)
                    + ", "
                    + String(wb)
                    + ", "
                    + String(wc)
                )
            if wa != fwd or wb != fwd:
                separations += 1
                if separations <= 4:
                    print(
                        "    SEPARATES "
                        + op_name(op)
                        + " "
                        + String(m)
                        + "x"
                        + String(n)
                        + "x"
                        + String(k)
                        + ": forward ws = "
                        + String(fwd)
                        + " floats ("
                        + gemm_plan_name(choose_gemm_plan(m, n, k))
                        + "), dA ws = "
                        + String(wa)
                        + " ("
                        + gemm_plan_name(choose_gemm_plan(ca[1], ca[2], ca[3]))
                        + "), dB ws = "
                        + String(wb)
                        + " ("
                        + gemm_plan_name(choose_gemm_plan(cb[1], cb[2], cb[3]))
                        + ")"
                    )
    if relation_fails != 0:
        raise Error(
            "check_backward_workspace_sizing: "
            + String(relation_fails)
            + " sizing relations are wrong (lines above). A helper that"
            " returns less than the routed plan needs is an out-of-bounds"
            " device write in every caller that trusts it."
        )
    if separations == 0:
        raise Error(
            "check_backward_workspace_sizing: NOT ONE shape in the list"
            " separates the forward workspace number from the backward ones."
            " The gate would then be passing on a coincidence and would say"
            " nothing about the hazard"
            " `identical_gemm_backward_a_workspace_max_floats` exists to"
            " prevent. Add a shape where the backward call takes SPLITK and"
            " the forward one does not, or the reverse."
        )
    print(
        "  "
        + String(separations)
        + " of "
        + String(len(shapes_m) * 3)
        + " shape/orientation pairs give a backward workspace that is NOT"
        " the forward number"
    )
    _shared_workspace_arm()
    print(
        "check_backward_workspace_sizing ["
        + _mode_name()
        + "] OK: every helper covers its routed plan, at least one shape"
        " separates forward from backward, and one dirty shared workspace"
        " carries all three calls"
    )


def _shared_workspace_arm() raises:
    """ARM 4. `dA`, `dB` and `db` through ONE workspace, no synchronize
    between them, against their solo launches."""
    var m = 520
    var n = 5
    var k = 17
    var op = OP_NN
    print(
        "  -- one DIRTY shared workspace across dA, dB and db at "
        + op_name(op)
        + " "
        + String(m)
        + "x"
        + String(n)
        + "x"
        + String(k)
        + " --"
    )
    var unwritten = 0
    var fails = 0
    with DeviceContext() as ctx:
        var sdc = _bwd_dc(BWD_A, op, m, n, k, False, 701)
        var sB = _bwd_w(BWD_A, op, m, n, k, False, 701)
        var sA = _bwd_w(BWD_B, op, m, n, k, False, 701)
        var sOnes = _bwd_w(BWD_BIAS, op, m, n, k, False, 701)
        var solo_a = _run_bwd(ctx, BWD_A, sdc, sB, op, m, n, k, String("solo.dA"), unwritten)
        var solo_b = _run_bwd(ctx, BWD_B, sdc, sA, op, m, n, k, String("solo.dB"), unwritten)
        var solo_c = _run_bwd(ctx, BWD_BIAS, sdc, sOnes, op, m, n, k, String("solo.db"), unwritten)

        var nws = identical_gemm_backward_workspace_max_floats(op, m, n, k, True)
        if nws < 1:
            nws = 1
        var na = _bwd_out_logical(BWD_A, op, m, n, k)
        var nb = _bwd_out_logical(BWD_B, op, m, n, k)
        var nc = _bwd_out_logical(BWD_BIAS, op, m, n, k)
        var dws = ctx.enqueue_create_buffer[DType.float32](nws)
        var ddc = ctx.enqueue_create_buffer[DType.float32](len(sdc))
        var dB_ = ctx.enqueue_create_buffer[DType.float32](len(sB))
        var dA_ = ctx.enqueue_create_buffer[DType.float32](len(sA))
        var dOnes = ctx.enqueue_create_buffer[DType.float32](len(sOnes))
        var oA = ctx.enqueue_create_buffer[DType.float32](na)
        var oB = ctx.enqueue_create_buffer[DType.float32](nb)
        var oC = ctx.enqueue_create_buffer[DType.float32](nc)
        var hdc = ctx.enqueue_create_host_buffer[DType.float32](len(sdc))
        var hb = ctx.enqueue_create_host_buffer[DType.float32](len(sB))
        var ha = ctx.enqueue_create_host_buffer[DType.float32](len(sA))
        var hones = ctx.enqueue_create_host_buffer[DType.float32](len(sOnes))
        var hws = ctx.enqueue_create_host_buffer[DType.float32](nws)
        var hoA = ctx.enqueue_create_host_buffer[DType.float32](na)
        var hoB = ctx.enqueue_create_host_buffer[DType.float32](nb)
        var hoC = ctx.enqueue_create_host_buffer[DType.float32](nc)
        ctx.synchronize()
        for i in range(len(sdc)):
            hdc.unsafe_ptr().unsafe_store(i, sdc[i])
        for i in range(len(sB)):
            hb.unsafe_ptr().unsafe_store(i, sB[i])
        for i in range(len(sA)):
            ha.unsafe_ptr().unsafe_store(i, sA[i])
        for i in range(len(sOnes)):
            hones.unsafe_ptr().unsafe_store(i, sOnes[i])
        for i in range(nws):
            hws.unsafe_ptr().unsafe_store(i, POISON)
        for i in range(na):
            hoA.unsafe_ptr().unsafe_store(i, POISON)
        for i in range(nb):
            hoB.unsafe_ptr().unsafe_store(i, POISON)
        for i in range(nc):
            hoC.unsafe_ptr().unsafe_store(i, POISON)
        ctx.enqueue_copy(dst_buf=ddc, src_ptr=hdc.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dB_, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dA_, src_ptr=ha.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dOnes, src_ptr=hones.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dws, src_ptr=hws.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=oA, src_ptr=hoA.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=oB, src_ptr=hoB.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=oC, src_ptr=hoC.unsafe_ptr())
        ctx.synchronize()
        # THE COMPOSITION: three enqueues, one workspace, no wait between.
        identical_gemm_backward_a_into(ctx, oA, ddc, dB_, dws, m, n, k, op)
        identical_gemm_backward_b_into(ctx, oB, ddc, dA_, dws, m, n, k, op)
        identical_gemm_backward_bias_into(ctx, oC, ddc, dOnes, dws, m, n)
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=hoA.unsafe_ptr(), src_buf=oA)
        ctx.enqueue_copy(dst_ptr=hoB.unsafe_ptr(), src_buf=oB)
        ctx.enqueue_copy(dst_ptr=hoC.unsafe_ptr(), src_buf=oC)
        ctx.synchronize()
        fails += _cmp_host_buf(hoA, solo_a, na, String("dA in the shared loop"))
        fails += _cmp_host_buf(hoB, solo_b, nb, String("dB in the shared loop"))
        fails += _cmp_host_buf(hoC, solo_c, nc, String("db in the shared loop"))
        _ = dws
        _ = ddc
        _ = dB_
        _ = dA_
        _ = dOnes
        _ = oA
        _ = oB
        _ = oC
        _ = hdc
        _ = hb
        _ = ha
        _ = hones
        _ = hws
        _ = hoA
        _ = hoB
        _ = hoC
    if unwritten != 0 or fails != 0:
        raise Error(
            "check_backward_workspace_sizing: the shared-workspace"
            " composition moved bits or left cells unwritten ("
            + String(fails)
            + " calls, "
            + String(unwritten)
            + " unwritten cells in the solo runs). Either the workspace is"
            " undersized for one of the three calls, or a call is reading a"
            " slot the previous call wrote, and"
            " `identical_gemm_backward_workspace_max_floats`'s license to"
            " share one buffer is void."
        )


def _cmp_host_buf(
    hb: HostBuffer[DType.float32], solo: List[Float32], count: Int, label: String
) raises -> Int:
    var bad = 0
    var first = -1
    for i in range(count):
        var v = hb.unsafe_ptr().unsafe_load(i)
        if _bits(v) == _bits(POISON):
            raise Error(
                "POISON SURVIVED in the shared-workspace loop at cell "
                + String(i)
                + " of "
                + label
                + ": the call never wrote it."
            )
        if _bits(v) != _bits(solo[i]):
            bad += 1
            if first < 0:
                first = i
    if bad == 0:
        print(
            "      OK   "
            + label
            + ": "
            + String(count)
            + " cells identical to the solo launch"
        )
        return 0
    print(
        "      FAIL "
        + label
        + ": "
        + String(bad)
        + " of "
        + String(count)
        + " cells CHANGED BITS against the solo launch. First at "
        + String(first)
        + ": in-loop "
        + _show(hb.unsafe_ptr().unsafe_load(first))
        + "   solo "
        + _show(solo[first])
    )
    return 1


# ===========================================================================
# GATE 8: THE TOKEN SWEEP
# ===========================================================================


def check_backward_k_range() raises:
    """**G8.** `dB`'s `k'` swept over the partition's whole tested range,
    because `k'` is the TOKEN COUNT and a token count is whatever the caller
    passed.

    `k' in {0, 1, 128, 129, 300, 517, 4097, 130900, 131073, 1000000,
    4000000}`. The forward gate's own sweep
    (`gemm_device_check.mojo::check_device_matches_oracle`) reaches four
    million, so a token count up to four million is inside the range the
    partition has actually been exercised at. **Beyond four million it is
    not**, and this gate RECORDS that rather than asserting anything about
    it. DEVIATION 1062.

    The interesting members of the list and why:

        131072   the last `k` at which `L` is `K_LEAF_MIN = 128`
        131073   the first at which it is not: `L` becomes 129 and `P` drops
                 from 1024 to 1017
        130900   `P = 1023`, odd, ragged, ten levels of carries
        4000000  `L = 3907`, `P = 1024`, the profile cap, and the shape where
                 one thread walks 3,907 dependent fused multiply-adds

    THE SHAPE IS `(m, n, k) = (T, 1, 2)` AND `m'`, `n'` ARE REDUCED ON
    PURPOSE. Reducing them is sound for the same reason the forward gate
    reduces them: the arithmetic sees `k` and the two profile constants and
    NOTHING ELSE (contract 6.1), so `m'` and `n'` preserve every leaf
    boundary and every tree level while making the host oracle runnable. At
    `T = 4,000,000` the operands are 12 million floats, about 48 MB, which is
    the largest allocation in this file and the reason this gate is last.

    **THIS GATE CANNOT SEE A TRANSPOSE ERROR AND IS NOT FOR THAT.** At
    `n = 1` and `k = 2` the shape is nearly degenerate and a wrong
    orientation could land on the right cells. G2 is the gate for the
    routing and G3 is the gate for the orientation at pairwise-distinct
    shapes; this one is about the PARTITION and only about the partition. A
    reader who takes a green G8 as evidence about transposes has read it
    wrong, and the sentence is here so that nobody does.

    ASSERTED under IDENTICAL, REPORTED under FAST.
    """
    print(
        "check_backward_k_range ["
        + _mode_name()
        + "]: dB's k' is the token count, swept to four million"
    )
    var ts = [0, 1, 128, 129, 300, 517, 4097, 130900, 131073, 1000000, 4000000]
    var cases = 0
    var fails = 0
    var unwritten = 0
    with DeviceContext() as ctx:
        for i in range(len(ts)):
            var t = ts[i]
            var part = contract_partition(t)
            _g3_case(
                ctx,
                BWD_B,
                OP_NN,
                t,
                1,
                2,
                String("dB.tokens")
                + String(t)
                + ".L"
                + String(part[0])
                + ".P"
                + String(part[1]),
                False,
                cases,
                fails,
                unwritten,
            )
    print(
        "  UNTESTED BEYOND FOUR MILLION TOKENS. The partition rule keeps"
        " producing `L = ceil(k / 1024)` without bound and the serial"
        " ascending chain inside one leaf grows with it, but no arm of this"
        " repository has run there and this gate does not pretend otherwise."
    )
    if fails == 0 and unwritten == 0:
        print(
            "check_backward_k_range ["
            + _mode_name()
            + "] OK: "
            + String(cases)
            + " token counts, every cell bit-identical to gemm_oracle"
        )
        return
    comptime if IDENTICAL_BUILD:
        raise Error(
            "check_backward_k_range [IDENTICAL]: "
            + String(fails)
            + " of "
            + String(cases)
            + " token counts disagree with gemm_oracle and "
            + String(unwritten)
            + " cells were never written."
        )
    print(
        "check_backward_k_range [FAST] REPORT: "
        + String(fails)
        + " of "
        + String(cases)
        + " token counts differ from the host oracle."
    )


# ===========================================================================
# GATE 9: THE BACKWARD CARD
# ===========================================================================


def check_backward_card() raises:
    """**G9.** Per-stage hashes for `tools/identity_trace_diff.py`, emitted
    from THIS FILE rather than from a new `bench/` driver. DEVIATION 1060.

    Set `MOJOLEARN_IDENTITY_TRACE` to a path and this gate writes a card; it
    is a no-op otherwise, which is the shipping state.

        MOJOLEARN_IDENTITY_TRACE=/tmp/bwd.apple.card \\
        tools/with_identical_mode.sh pixi run mojo run \\
            -I . gemm/mojo_only/gemm_backward_check.mojo
        python3 tools/identity_trace_diff.py /tmp/bwd.apple.card \\
            /tmp/bwd.h100.card

    **WHY THE CARD IS EMITTED HERE AND NOT FROM A SECOND DRIVER.**
    `IDENTICAL_BACKWARD_PLAN.md` section 5 G9 asks for
    `bench/gemm_bwd_card_main.mojo`, patterned on `bench/gemm_card_main.mojo`.
    Three reasons it is here instead, and the first is the strongest:

    1. A second driver is a SECOND SPELLING OF THE FIXTURES. The forward
       card's own header makes exactly this argument about its oracle arm and
       its device arm ("the SAME DRIVER, so the fixtures, the tags, the
       ordering and the record format are shared by construction rather than
       by two authors agreeing"). A backward card built from its own fixture
       generator would diff two fixtures whenever it diverged, and finding
       that out costs a rented hour.
    2. Every stage the card wants is already computed by G3. Recording it is
       an env check and three lines.
    3. `bench/` is not this lane's to write into.

    WHAT IS RECORDED, AND THE INPUT STAGES COME FIRST. For each of the six
    routes and the bias, at three shapes:

        bwd.<case>.in.dc     the gradient arriving from downstream
        bwd.<case>.in.w      `B` for a dA route, `A` for a dB route, the
                             ones vector for the bias
        bwd.<case>.out       the gradient, recorded OFF THE DEVICE BUFFER

    **Compare the two input stages before comparing any output stage.** Two
    cards whose inputs differ are diffing their fixtures. `record_device` is
    used rather than `record_list_f32` for the output so that
    `MOJOLEARN_IDENTITY_TRACE_DUMP` writes a `.bin` sidecar, and a sidecar is
    the difference between the differ saying "differs" and the differ saying
    WHICH CELL, by how many ulps, and whether the class is denormal-vs-zero.
    On a cross-vendor divergence that is the whole investigation.

    WHAT IS NOT RECORDED, AND IT IS THE PART A READER SHOULD NOT ASSUME.
    **The fold-ladder levels are not here.** DEVIATION 533's per-level hashes
    are `bench/gemm_ladder_main.mojo`'s instrument and it belongs to another
    lane. The backward needs no NEW ladder: a backward call at
    `(m', n', k')` is the forward at `(m', n', k')`, so running the existing
    ladder at the six backward shapes is the whole of it, and that is a
    shape-list change in a file this lane may not edit. Until it is made, a
    backward divergence localizes to a CALL and a CELL and not to a fold
    level, and the forward lane's own experience says that is one instrument
    short: an output-only comparison called a thirteen-stage divergence inert
    on the mamba block, and only the per-stage card could see it.

    A TRACED RUN IS NOT A MEASUREMENT. `record_device` drains the queue.
    """
    var t = IdentityTrace()
    if not t.enabled:
        print(
            "check_backward_card: OFF. Set MOJOLEARN_IDENTITY_TRACE to a"
            " path to emit the card. Nothing was recorded and nothing is"
            " claimed."
        )
        return
    print("check_backward_card [" + _mode_name() + "]: emitting")
    t.header(
        String("gemm backward card, profile mojolearn.identical.gemm.fp32.v1")
    )
    t.header(String("mode ") + _mode_name())
    t.header(String("bwd sabotage ") + gemm_backward_sabotage_name())
    t.header(String("fwd sabotage ") + gemm_sabotage_name())
    t.header(
        String(
            "compare the .in.* stages BEFORE any .out stage; two cards whose"
            " inputs differ are diffing their fixtures"
        )
    )
    var shapes_m = [5, 130, 517]
    var shapes_n = [7, 70, 33]
    var shapes_k = [3, 17, 9]
    var unwritten = 0
    with DeviceContext() as ctx:
        for s in range(len(shapes_m)):
            var m = shapes_m[s]
            var n = shapes_n[s]
            var k = shapes_k[s]
            var tag = String(m) + "x" + String(n) + "x" + String(k)
            for r in range(6):
                var op = _route_op(r)
                var which = BWD_A
                if r >= 3:
                    which = BWD_B
                _card_stage(ctx, t, which, op, m, n, k, String("bwd.") + _route_name(r) + "." + tag, unwritten)
            _card_stage(ctx, t, BWD_BIAS, OP_NN, m, n, k, String("bwd.dbias.") + tag, unwritten)
    if unwritten != 0:
        raise Error(
            "check_backward_card: "
            + String(unwritten)
            + " cells were never written, so the card records a buffer that"
            " is partly poison. A card of an unwritten buffer diffs cleanly"
            " against another card of an unwritten buffer."
        )
    print("check_backward_card OK: card written")


def _card_stage(
    ctx: DeviceContext,
    mut t: IdentityTrace,
    which: Int,
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    tag: String,
    mut unwritten: Int,
) raises:
    var cdc = _bwd_dc(which, op, m, n, k, False, 801)
    var cw = _bwd_w(which, op, m, n, k, False, 801)
    t.record_list_f32(tag + ".in.dc", cdc)
    t.record_list_f32(tag + ".in.w", cw)
    var out = _run_bwd(ctx, which, cdc, cw, op, m, n, k, tag, unwritten)
    t.record_list_f32(tag + ".out", out)


# ===========================================================================


def _gate(name: String, mut ran: Int, mut failed: Int, e: String):
    ran += 1
    if e.byte_length() > 0:
        failed += 1
        print("!! GATE FAILED: " + name)
        print("   " + e)


def main() raises:
    """Runs every gate and reports every verdict BEFORE it raises.

    That is deliberate and it is not a softening: the process still exits
    non-zero if anything failed. Stopping at the first failure under a
    sabotage build would show one gate's opinion when the useful evidence is
    WHICH gates a given defect reaches and which it walks past, and "walks
    past" is the entire content of this file's ledger.

    **UNDER A SABOTAGE BUILD THE VERDICT IS INVERTED. DEVIATION 1061.** A
    clean run is then the FAILURE, because it means the arm was reached and
    made no difference, or was never reached at all. Both are
    `[[reached-but-inert]]`. The individual gates already check their moved
    counts against the ledger, so this is the outer belt: it catches an arm
    that somebody adds and forgets to give a gate.
    """
    print(
        "== gemm/mojo_only/gemm_backward_check.mojo ["
        + _mode_name()
        + "]  bwd sabotage: "
        + gemm_backward_sabotage_name()
        + "  fwd sabotage: "
        + gemm_sabotage_name()
        + " =="
    )
    print(
        "   profile: mojolearn.identical.gemm.fp32.v1   contract:"
        " gemm/IDENTICAL_FP32_CONTRACT.md (CONSUMED, NEVER AMENDED)"
    )
    print(
        "   routing: gemm/mojo_only/gemm_backward.mojo   plan:"
        " gemm/IDENTICAL_BACKWARD_PLAN.md"
    )
    print(
        "   oracle: gemm/mojo_only/gemm_oracle.mojo::gemm_oracle, evaluated"
        " at the BACKWARD shape"
    )
    var ran = 0
    var failed = 0

    try:
        check_backward_routing_is_the_table()
        _gate(String("check_backward_routing_is_the_table"), ran, failed, String(""))
    except e:
        _gate(String("check_backward_routing_is_the_table"), ran, failed, String(e))
    try:
        check_backward_gradients_are_correct()
        _gate(String("check_backward_gradients_are_correct"), ran, failed, String(""))
    except e:
        _gate(String("check_backward_gradients_are_correct"), ran, failed, String(e))
    try:
        check_the_square_fixture_is_vacuous()
        _gate(String("check_the_square_fixture_is_vacuous"), ran, failed, String(""))
    except e:
        _gate(String("check_the_square_fixture_is_vacuous"), ran, failed, String(e))
    try:
        check_backward_matches_oracle()
        _gate(String("check_backward_matches_oracle"), ran, failed, String(""))
    except e:
        _gate(String("check_backward_matches_oracle"), ran, failed, String(e))
    try:
        check_backward_is_launch_invariant()
        _gate(String("check_backward_is_launch_invariant"), ran, failed, String(""))
    except e:
        _gate(String("check_backward_is_launch_invariant"), ran, failed, String(e))
    try:
        check_backward_b_depends_on_the_token_count_and_says_so()
        _gate(String("check_backward_b_depends_on_the_token_count_and_says_so"), ran, failed, String(""))
    except e:
        _gate(String("check_backward_b_depends_on_the_token_count_and_says_so"), ran, failed, String(e))
    try:
        check_backward_bias_is_the_contract_sum()
        _gate(String("check_backward_bias_is_the_contract_sum"), ran, failed, String(""))
    except e:
        _gate(String("check_backward_bias_is_the_contract_sum"), ran, failed, String(e))
    try:
        check_backward_workspace_sizing()
        _gate(String("check_backward_workspace_sizing"), ran, failed, String(""))
    except e:
        _gate(String("check_backward_workspace_sizing"), ran, failed, String(e))
    try:
        check_backward_k_range()
        _gate(String("check_backward_k_range"), ran, failed, String(""))
    except e:
        _gate(String("check_backward_k_range"), ran, failed, String(e))
    try:
        check_backward_card()
        _gate(String("check_backward_card"), ran, failed, String(""))
    except e:
        _gate(String("check_backward_card"), ran, failed, String(e))

    comptime if ANY_BWD_SABOTAGE:
        if failed == 0:
            raise Error(
                "gemm_backward_check [INVERTED]: this binary was built with"
                " backward sabotage '"
                + gemm_backward_sabotage_name()
                + "' and ALL "
                + String(ran)
                + " gates passed. The arm is reached and inert, or it is"
                " never reached at all, and either way the clause it is"
                " aimed at is UNGATED. Do not weaken the arm; find the"
                " fixture that separates it, or record it as vacuous in this"
                " file's ledger."
            )
        print(
            "gemm backward gates: INVERTED PASS ["
            + _mode_name()
            + "]  "
            + String(failed)
            + " of "
            + String(ran)
            + " gates failed under backward sabotage '"
            + gemm_backward_sabotage_name()
            + "', which is the required outcome. Read the per-route masks"
            " above against this file's ledger before believing it."
        )
        return

    if failed != 0:
        raise Error(
            "gemm_backward_check ["
            + _mode_name()
            + "]: "
            + String(failed)
            + " of "
            + String(ran)
            + " gates FAILED (bwd sabotage: "
            + gemm_backward_sabotage_name()
            + ", fwd sabotage: "
            + gemm_sabotage_name()
            + ")"
        )
    print(
        "gemm backward routing + gates: all green ["
        + _mode_name()
        + "]  ("
        + String(ran)
        + " gates, bwd sabotage: "
        + gemm_backward_sabotage_name()
        + ", fwd sabotage: "
        + gemm_sabotage_name()
        + ")"
    )
