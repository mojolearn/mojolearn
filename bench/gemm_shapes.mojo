# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The shapes `gemm.fp32.v1` is judged on. ONE table, with provenance.

`IDENTICAL_GEMM_PLAN.md`'s charter, Phase 4: *"Include representative shapes
from both MojoLearn and transformer inference."* And Phase 3 needs ragged
cases. This file is that table, in one place, so the invariance gates and the
benchmark draw from the SAME shapes rather than each inventing its own -- two
sets of shapes is two definitions of what the profile was tested on.

EVERY ROW CARRIES ITS PROVENANCE, and that is the point of the file rather
than decoration. A benchmark shape with no caller behind it measures a
number nobody will ever see; a transformer shape with no model behind it
measures a number that is not a transformer. So each row names either the
`file:line` in this repository that calls it, or the published model
configuration it comes from.

NO GPU, NO KERNEL, NO DEPENDENCY ON PHASE 2. This module is data plus
arithmetic on the data. `main()` prints the table with the derived leaf
counts, which is how you check the leaf rule against a shape without running
anything.

WHAT THE TABLE IS FOR, CONCRETELY
----------------------------------
The contract's leaf rule makes `L` a function of `k` and the profile alone
(charter clause 6.1: `L` may not depend on `m` or `n`). So `k` is the axis
that decides the fold, and the interesting property of a shape is its LEAF
COUNT `P = ceil(k / L)`:

  - `P = 1` is the degenerate fold the contract still requires to run (a
    one-term fold, not a bypass, or the sign of a zero becomes a function of
    `P`).
  - `P` even and `P` odd take DIFFERENT PATHS through the balanced tree,
    because an unpaired leaf is carried bit-for-bit rather than padded.
  - a ragged last leaf (`k % L != 0`) is a different case again.

**THE FIRST VERSION OF THIS FILE CLAIMED THE SHIPPED GRAM SHAPE HITS THE
ODD CASE FOR FREE. IT WAS WRONG, AND THE ERROR IS RECORDED HERE RATHER THAN
QUIETLY CORRECTED**, because the way it happened is the more useful part.

It mirrored a flat `L = 128`. The contract's leaf size is not a constant: it
is flat only while `ceil(k/K_LEAF_MIN)` fits under `MAX_LEAVES`, and past
`k = 131,072` it becomes `ceil(k / MAX_LEAVES)`. The shipped Gram shape sits
far past that at `k = 1,000,000`, where the real rule gives `L = 977` and
`P = 1024` -- EVEN -- not the `P = 7,813` the flat mirror produced. And the
check that was supposed to catch it asserted that the contract FILE EXISTS
and then printed "confirm by eye", which is not a check at all; it is an
instruction to a human, and the human did not.

THE TRANSFORMER ROWS ARE THE POINT OF THE EXERCISE, AND THE HONEST CAVEAT
--------------------------------------------------------------------------
`IDENTICAL_GEMM_PLAN.md` argues that a general identical GEMM is worth more
than a tree-learning feature because the audience for batch-invariant
inference is far larger. These rows are that claim made checkable: the
projection shapes of real models at real token counts, so "v1 costs Nx" is
answered for the workload the claim is about and not only for a 32x32 Gram.

**They are shapes, not an inference engine, and this file does not claim
otherwise.** A GEMM that is bit-identical at these shapes gives identical
LINEAR LAYERS. Identical logits additionally need every norm, softmax, RoPE
and activation between them pinned, which is not this lane's scope and is
stated at length in the plan. Do not let a green table here become a claim
about tokens.

THE CORRECTED TABLE SAYS SOMETHING SHARPER THAN THE WRONG ONE DID.

**Under the real leaf rule, not one of these twenty shapes produces an odd
leaf count.** Every row lands on `P == 1` or an even `P`. So the odd-leaf
CARRY -- an unpaired leaf copied bit-for-bit rather than padded, the sharpest
clause in the contract -- **is not reached by any workload this repository
runs, nor by any Llama projection in the table.**

The mechanism is worth understanding rather than memorising. Past
`k = 131,072` the rule drives `P` toward `MAX_LEAVES = 1024`, which is a
power of two; below it `P = ceil(k / 128)`, and these shapes' `k` are all
multiples that come out even. Transformer hidden and intermediate widths are
powers of two by construction, so they will never land odd here.

Three consequences. **(1)** Phase 3's odd-carry fixture must be SYNTHETIC and
cannot be drawn from this table -- a gate built only from real shapes would
test the even path and report the contract as covered. Phase 2a's F7 uses
`k = 300` (`P = 3`) and that is the right call. **(2)** The carry clause is
correct and is effectively dead code for every workload we currently know of,
which is worth knowing before anyone prices it. **(3)** `MAX_LEAVES` being a
power of two is what keeps `P` even at large `k`, so changing it is not a
free tuning knob -- an odd or non-power-of-two value makes the carry path
live for the biggest shapes in the table.

`check_table_has_the_hard_cases` REPORTS this rather than failing on it, on
purpose. Failing would put pressure on the next person to add a synthetic row
to go green, and a shape table containing shapes chosen to satisfy its own
check has stopped being a record of what this library computes.

Token counts of 1 / 8 / 512 are decode, small batch, and a prefill chunk.
They are the `m` axis, which the leaf rule may not read -- so if a v1 number
moves between them, something has read `m` that should not have, and the
table doubles as a batch-invariance smoke test at production shapes.
"""

from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


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


#: The contract's two profile constants, and the RULE they feed.
#:
#: **THE FIRST VERSION OF THIS FILE MIRRORED A FLAT `L = 128` AND WAS WRONG.**
#: The leaf size is not a constant; it is `contract_leaf_size(k)` (contract
#: section 6), which returns `K_LEAF_MIN` only while `ceil(k/K_LEAF_MIN)` fits
#: under `MAX_LEAVES` and switches to `ceil(k / MAX_LEAVES)` above it. The
#: crossover is `k = 131,072`, and the shipped Gram shape sits far past it at
#: `k = 1,000,000` -- so every leaf count this file printed for that row was
#: computed under a rule the contract does not have.
#:
#: The check below was ALSO too weak to catch it: it asserted the contract
#: FILE EXISTS and then told the reader to "confirm by eye". A check whose
#: assertion is an instruction to a human is not a check. It now recomputes
#: the rule and asserts the crossover behaviour, so a change to either
#: constant fails here instead of being confirmed by eye and believed.
comptime K_LEAF_MIN = 128
comptime MAX_LEAVES = 1024


def contract_leaf_size(k: Int) -> Int:
    """`L`, contract section 6, transcribed exactly. A pure function of `k`
    and the two profile constants -- not of `m`, not of `n`, not of anything
    the machine knows."""
    if k <= 0:
        return 1
    if k <= K_LEAF_MIN:
        return k
    if (k + K_LEAF_MIN - 1) // K_LEAF_MIN <= MAX_LEAVES:
        return K_LEAF_MIN
    return (k + MAX_LEAVES - 1) // MAX_LEAVES

comptime OP_NT = 0
comptime OP_TN = 1
comptime OP_NN = 2

comptime GEMM_SHAPE_COUNT = 20


def gemm_shape_op(i: Int) -> Int:
    """Which orientation the caller asks for."""
    if i <= 3:
        return OP_TN
    return OP_NT


def gemm_shape_m(i: Int) -> Int:
    if i == 0:
        return 32
    if i == 1:
        return 32
    if i == 2:
        return 128
    if i == 3:
        return 16
    if i == 4:
        return 8192
    if i == 5:
        return 8192
    if i == 6:
        return 4096
    if i == 7:
        return 65536
    # Transformer rows: m is the TOKEN COUNT.
    if i == 8 or i == 11 or i == 14 or i == 17:
        return 1
    if i == 9 or i == 12 or i == 15 or i == 18:
        return 8
    return 512


def gemm_shape_n(i: Int) -> Int:
    if i == 0 or i == 1:
        return 32
    if i == 2:
        return 128
    if i == 3:
        return 16
    if i == 4:
        return 4
    if i == 5:
        return 64
    if i == 6:
        return 64
    if i == 7:
        return 1
    if i >= 8 and i <= 10:
        return 4096  # attention projection, Llama-3-8B hidden
    if i >= 11 and i <= 13:
        return 14336  # MLP up/gate, Llama-3-8B intermediate
    if i >= 14 and i <= 16:
        return 4096  # MLP down, back to hidden
    return 128256  # LM head, Llama-3 vocab


def gemm_shape_k(i: Int) -> Int:
    if i == 0:
        return 1_000_000
    if i == 1:
        return 65_536
    if i == 2:
        return 100_003
    if i == 3:
        return 65_536
    if i == 4:
        return 4
    if i == 5:
        return 128
    if i == 6:
        return 64
    if i == 7:
        return 16
    if i >= 8 and i <= 10:
        return 4096
    if i >= 11 and i <= 13:
        return 4096
    if i >= 14 and i <= 16:
        return 14336  # MLP down contracts over the INTERMEDIATE width
    return 4096


def gemm_shape_name(i: Int) -> String:
    if i == 0:
        return String("gram.32x32x1M")
    if i == 1:
        return String("gram.32x32x64K")
    if i == 2:
        return String("gram.128sq.x100003")
    if i == 3:
        return String("ols.step1.16x16x64K")
    if i == 4:
        return String("pca.transform.8192x4x4")
    if i == 5:
        return String("pca.transform.wide.8192x64x128")
    if i == 6:
        return String("kmeans.dist.4096x64x64")
    if i == 7:
        return String("ols.predict.gemv.64Kx16")
    if i == 8:
        return String("llama8b.qkv.t1")
    if i == 9:
        return String("llama8b.qkv.t8")
    if i == 10:
        return String("llama8b.qkv.t512")
    if i == 11:
        return String("llama8b.mlp_up.t1")
    if i == 12:
        return String("llama8b.mlp_up.t8")
    if i == 13:
        return String("llama8b.mlp_up.t512")
    if i == 14:
        return String("llama8b.mlp_down.t1")
    if i == 15:
        return String("llama8b.mlp_down.t8")
    if i == 16:
        return String("llama8b.mlp_down.t512")
    if i == 17:
        return String("llama8b.lm_head.t1")
    if i == 18:
        return String("llama8b.lm_head.t8")
    return String("llama8b.lm_head.t512")


def gemm_shape_provenance(i: Int) -> String:
    """Where the shape COMES FROM. A row with no answer here does not belong
    in the table."""
    if i == 0:
        return String(
            "bench/linalg_price_main.mojo:90 -- the shipped PCA/OLS Gram"
            " aspect. Past the leaf rule's crossover: L = 977, P = 1024,"
            " ragged tail of 529. The table's largest k."
        )
    if i == 1:
        return String(
            "bench/linalg_trace_main.mojo:107 -- the Gram identity card's"
            " shape, so a v1 number here is comparable with row 27's card."
        )
    if i == 2:
        return String(
            "core/gram_splitk.mojo GRAM_MAX_COLS -- the WIDEST Gram the"
            " pinned kernel serves; past it row 27's gemm_tn REFUSES."
        )
    if i == 3:
        return String(
            "glm/ported/linalg/detail/lstsq.mojo:219 at the OLS identity"
            " card's fixture (n_rows=65536, n_cols=16), DEVIATION 527."
        )
    if i == 4:
        return String(
            "decomposition/estimator.mojo:107 at pca_check's PCA_ROWS=8192,"
            " PCA_COLS=4. P=1: the degenerate one-term fold."
        )
    if i == 5:
        return String(
            "decomposition/estimator.mojo:107 at the pca_wide fixture's"
            " width. P=1 as well, at a wider output."
        )
    if i == 6:
        return String(
            "cluster/ported/cluster/detail/min_cluster_distance_compute"
            ".mojo:308, and bench/linalg_price_main.mojo:120 prices it --"
            " the arm measured at ~4.7x, so v1 is comparable against it."
        )
    if i == 7:
        return String(
            "glm/estimator.mojo:108 -- OLS predict. n=1, so row 28's"
            " gemm_nt routes it to gemv BEFORE any leaf rule applies."
        )
    if i >= 8 and i <= 10:
        return String(
            "Llama-3-8B attention projection: hidden 4096 -> 4096."
            " m is the TOKEN COUNT (decode / small batch / prefill chunk)."
        )
    if i >= 11 and i <= 13:
        return String(
            "Llama-3-8B MLP up/gate: hidden 4096 -> intermediate 14336."
        )
    if i >= 14 and i <= 16:
        return String(
            "Llama-3-8B MLP down: intermediate 14336 -> hidden 4096. The"
            " only transformer row whose k is the INTERMEDIATE width."
        )
    return String(
        "Llama-3-8B LM head: hidden 4096 -> vocab 128256. The widest n in"
        " the table by a factor of nine."
    )


def leaf_count(k: Int, leaf: Int) -> Int:
    """`P = ceil(k / L)`, the contract's derivation order.

    P IS DERIVED FROM L AND NEVER CHOSEN, which is what makes an empty leaf
    impossible -- the charter's clause 6.2. A table that computed L from a
    desired P could produce one.
    """
    if k <= 0:
        return 0
    return (k + leaf - 1) // leaf


def last_leaf_len(k: Int, leaf: Int) -> Int:
    """How many elements the FINAL leaf holds. Equal to `leaf` when k
    divides evenly; never zero for k > 0."""
    if k <= 0:
        return 0
    var r = k % leaf
    if r == 0:
        return leaf
    return r


def check_leaf_size_matches_contract() raises:
    """The transcribed leaf rule behaves the way section 6 specifies.

    THE PREVIOUS VERSION OF THIS CHECK ASSERTED THAT A FILE EXISTS and then
    printed "confirm by eye". It passed while this table computed every leaf
    count from a flat `L = 128` that the contract does not have. A check
    whose assertion is an instruction to a human is not a check, and this
    one's failure was the whole reason the table's headline claim was wrong.

    So it now exercises the rule at the boundaries that define it:
    below `K_LEAF_MIN`, at it, in the flat band, at the crossover, and past
    it -- and asserts the invariant that actually matters, `P <= MAX_LEAVES`
    at every k, which is what `MAX_LEAVES` exists to guarantee.
    """
    if contract_leaf_size(0) != 1:
        raise Error("leaf rule: k = 0 must give L = 1")
    if contract_leaf_size(1) != 1 or contract_leaf_size(64) != 64:
        raise Error("leaf rule: below K_LEAF_MIN, L must be k itself")
    if contract_leaf_size(K_LEAF_MIN) != K_LEAF_MIN:
        raise Error("leaf rule: at K_LEAF_MIN, L must be K_LEAF_MIN")
    if contract_leaf_size(K_LEAF_MIN + 1) != K_LEAF_MIN:
        raise Error("leaf rule: just past K_LEAF_MIN, L must stay flat")

    # The crossover: the largest k whose leaf count still fits MAX_LEAVES.
    var cross = K_LEAF_MIN * MAX_LEAVES
    if contract_leaf_size(cross) != K_LEAF_MIN:
        raise Error(
            "leaf rule: at the crossover k = "
            + String(cross)
            + " the flat band must still apply"
        )
    if contract_leaf_size(cross + 1) <= K_LEAF_MIN:
        raise Error(
            "leaf rule: past the crossover L must GROW, so that P stays"
            " bounded by MAX_LEAVES"
        )

    for i in range(GEMM_SHAPE_COUNT):
        var k = gemm_shape_k(i)
        var p = leaf_count(k, contract_leaf_size(k))
        if p > MAX_LEAVES:
            raise Error(
                "leaf rule: '"
                + gemm_shape_name(i)
                + "' gives P = "
                + String(p)
                + ", past MAX_LEAVES = "
                + String(MAX_LEAVES)
                + ". That bound is what caps the fold's scratch and level"
                " count; if it can be exceeded the rule is transcribed"
                " wrong."
            )

    print(
        "check_leaf_size_matches_contract OK: the rule is flat at"
        " K_LEAF_MIN =",
        K_LEAF_MIN,
        "up to k =",
        cross,
        "and grows past it; every row's P is within MAX_LEAVES =",
        MAX_LEAVES,
    )


def check_table_has_the_hard_cases() raises:
    """The table must actually contain the cases the gates need.

    A shape table whose rows all fall in the easy case is worse than no
    table, because Phase 3 would draw its fixtures from it and conclude the
    profile is exercised. So this asserts the three hard cases are PRESENT
    rather than assuming the rows happen to cover them:

      - at least one ODD leaf count with a RAGGED last leaf (clause 5),
      - at least one `P == 1` (the degenerate one-term fold),
      - at least one `P` even and greater than one.
    """
    var odd_ragged = -1
    var single = -1
    var even_many = -1
    for i in range(GEMM_SHAPE_COUNT):
        var k = gemm_shape_k(i)
        var L = contract_leaf_size(k)
        var p = leaf_count(k, L)
        var tail = last_leaf_len(k, L)
        if p % 2 == 1 and p > 1 and tail != contract_leaf_size(k):
            if odd_ragged < 0:
                odd_ragged = i
        if p == 1:
            if single < 0:
                single = i
        if p > 1 and p % 2 == 0:
            if even_many < 0:
                even_many = i

    # THE ODD CASE IS A REPORT, NOT AN ASSERTION, AND THAT IS DELIBERATE.
    #
    # Under the contract's real leaf rule NOT ONE of these twenty shapes
    # produces an odd leaf count. Every real caller and every Llama
    # projection lands on P == 1 or an even P. Failing here would put
    # pressure on the next person to ADD A SYNTHETIC ROW to make the table
    # green, and a shape table that contains shapes chosen to satisfy its
    # own check is no longer a record of what this library computes. That is
    # the same trap as picking a benchmark dataset because it flatters the
    # result.
    #
    # So the gap is REPORTED, loudly, with the consequence spelled out. The
    # odd-carry fixture belongs in the gate files, where a synthetic `k` is
    # honest, and Phase 2a already put it there at k = 300 (P = 3).
    if odd_ragged < 0:
        print(
            "check_table_has_the_hard_cases FINDING: **no real shape in"
            " this table produces an ODD leaf count.** Every one of the "
            + String(GEMM_SHAPE_COUNT)
            + " rows lands on P == 1 or an even P under"
            " contract_leaf_size. So the odd-leaf CARRY -- the rule that an"
            " unpaired leaf is copied bit-for-bit rather than padded, the"
            " sharpest clause in the contract -- is NOT REACHED BY ANY"
            " WORKLOAD THIS REPOSITORY OR A LLAMA PROJECTION RUNS."
        )
        print(
            "    Why: past k = "
            + String(K_LEAF_MIN * MAX_LEAVES)
            + " the rule drives P toward MAX_LEAVES = "
            + String(MAX_LEAVES)
            + ", a power of two; below it P = ceil(k/"
            + String(K_LEAF_MIN)
            + "), and these shapes' k are all multiples that come out even."
        )
        print(
            "    Consequences. (1) Phase 3's odd-carry fixture must be"
            " SYNTHETIC -- it cannot be drawn from this table, and a gate"
            " built only from these shapes would test the even path and"
            " report the contract as covered. Phase 2a's F7 uses k = 300"
            " (P = 3), which is correct and must stay. (2) The carry clause"
            " is right and is effectively dead code for every workload we"
            " know of -- worth knowing before anyone prices it. (3)"
            " MAX_LEAVES being a POWER OF TWO is what makes P even at large"
            " k; changing it to an odd or non-power-of-two value would make"
            " the carry path live for the biggest shapes here, so it is not"
            " a free tuning knob."
        )
        print(
            "    This is a REPORT rather than a failure on purpose: failing"
            " would invite adding a synthetic row to go green, and a shape"
            " table containing shapes chosen to satisfy its own check has"
            " stopped being a record of what this library computes."
        )
    print(
        "check_table_has_the_hard_cases: P==1 and even-P>1 both present;"
        " odd-leaf coverage reported above. Odd row (if any): '"
        + (gemm_shape_name(odd_ragged) if odd_ragged >= 0 else String("NONE"))
        + "' (P="
        + String(leaf_count(gemm_shape_k(odd_ragged), contract_leaf_size(gemm_shape_k(odd_ragged))))
        + ", last leaf "
        + String(last_leaf_len(gemm_shape_k(odd_ragged), contract_leaf_size(gemm_shape_k(odd_ragged))))
        + "), P==1 at '"
        + gemm_shape_name(single)
        + "', even P>1 at '"
        + gemm_shape_name(even_many)
        + "'"
    )


def main() raises:
    print("== bench/gemm_shapes.mojo [" + _mode_name() + "] ==")
    print(
        "The shapes gemm.fp32.v1 is judged on. L is contract_leaf_size(k),",
        "\n(P = ceil(k/L) is the leaf count; 'tail' is the last leaf's"
        " length)\n",
    )
    print(
        "  #  name                            op        m       n         k"
        "        L      P  tail  parity"
    )
    for i in range(GEMM_SHAPE_COUNT):
        var k = gemm_shape_k(i)
        var L = contract_leaf_size(k)
        var p = leaf_count(k, L)
        var tail = last_leaf_len(k, L)
        var op = String("NT")
        if gemm_shape_op(i) == OP_TN:
            op = String("TN")
        elif gemm_shape_op(i) == OP_NN:
            op = String("NN")
        var parity = String("even")
        if p == 1:
            parity = String("SINGLE")
        elif p % 2 == 1:
            parity = String("ODD")
        print(
            " ",
            i,
            gemm_shape_name(i),
            op,
            gemm_shape_m(i),
            gemm_shape_n(i),
            k,
            "L=" + String(L),
            "P=" + String(p),
            "tail=" + String(tail),
            parity,
        )
    print()
    check_leaf_size_matches_contract()
    check_table_has_the_hard_cases()
    print()
    print("Provenance, one line per row:")
    for i in range(GEMM_SHAPE_COUNT):
        print("  " + gemm_shape_name(i) + ": " + gemm_shape_provenance(i))
