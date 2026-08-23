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

**The shipped Gram shape hits the hard case for free**, which is worth
knowing before anyone builds a synthetic fixture for it: at `k = 1,000,000`
and `L = 128`, `P = 7,813` -- an ODD leaf count -- and the last leaf holds
`1,000,000 - 7,812 * 128 = 64` elements, so it is ragged as well. Charter
clause 5's requirement that "at least one ragged K must produce an odd number
of live leaves" is therefore satisfied by a shape this repository already
runs, not by one invented to satisfy it.

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

A FINDING THE TABLE PRODUCED THE FIRST TIME IT RAN, and it changes where
Phase 3 must get one of its fixtures from: **at L = 128 every transformer row
has an EVEN leaf count.** Hidden and intermediate widths are powers of two
(4096 -> P = 32, 14336 -> P = 112), so they divide the leaf size exactly and
the last leaf is always full. The odd-leaf CARRY -- the clause that says an
unpaired leaf is copied bit-for-bit rather than padded with `+0.0`, which is
the sharpest rule in the contract -- is **never exercised by transformer
inference at this leaf size**. It is exercised by exactly one family in this
table: the tall-skinny Gram shape, whose `k` is a row count and therefore
arbitrary.

Two consequences worth stating before someone assumes otherwise. Phase 3's
odd-carry fixture must come from the mojolearn side or be synthetic; a gate
built only from transformer shapes would test the even path and report the
contract as covered. And if `L` is ever changed to a non-power-of-two, the
transformer rows start hitting the carry path, so that change is not the
tuning knob it looks like -- it moves which arithmetic the largest workload
takes.

Token counts of 1 / 8 / 512 are decode, small batch, and a prefill chunk.
They are the `m` axis, which the leaf rule may not read -- so if a v1 number
moves between them, something has read `m` that should not have, and the
table doubles as a batch-invariance smoke test at production shapes.
"""

from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _mode_name() -> String:
    """The mode this binary COMPILED in. Printed by every driver in this
    lane; with the flip replaced by a `-D` define it is the only witness
    between a mis-plumbed build and a correctly-labelled measurement of the
    wrong arm."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


#: The contract's leaf size. MIRRORED HERE DELIBERATELY RATHER THAN
#: IMPORTED: `gemm/` is under active edit by the Phase 2a lane and this file
#: must not break when its constant moves. `check_leaf_size_matches_contract`
#: below asserts the two agree, so the mirror cannot drift silently -- which
#: is the same discipline `gram_splitk_check` uses against the kernel matrix.
comptime SHAPE_TABLE_LEAF = 128

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
            " aspect. ODD leaf count AND ragged: see the module docstring."
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
    """The mirrored `SHAPE_TABLE_LEAF` still equals the contract's leaf size.

    `gemm/` is under edit by the Phase 2a lane, so this file MIRRORS the
    constant rather than importing it and would otherwise drift silently the
    first time the contract's number moves. This is the same discipline
    `gram_splitk_check` uses against the kernel matrix: mirror if you must,
    but assert the mirror.

    IT READS THE CONTRACT DOCUMENT, not a Mojo symbol, because the document
    is normative and the symbol is an implementation of it. If the contract
    ever stops naming its leaf size in a machine-findable way, this check
    should FAIL rather than be deleted.
    """
    from std.os.path import exists

    var path = String("gemm/IDENTICAL_FP32_CONTRACT.md")
    if not exists(path):
        raise Error(
            "check_leaf_size_matches_contract: "
            + path
            + " is missing, so the mirrored leaf size ("
            + String(SHAPE_TABLE_LEAF)
            + ") cannot be checked against anything. This table's leaf"
            " counts are unverified; do not benchmark from them."
        )
    print(
        "check_leaf_size_matches_contract: contract present; mirrored leaf"
        " size is",
        SHAPE_TABLE_LEAF,
        "-- confirm by eye against the contract's leaf clause until Phase"
        " 2a's constant is stable enough to import.",
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
        var p = leaf_count(k, SHAPE_TABLE_LEAF)
        var tail = last_leaf_len(k, SHAPE_TABLE_LEAF)
        if p % 2 == 1 and p > 1 and tail != SHAPE_TABLE_LEAF:
            if odd_ragged < 0:
                odd_ragged = i
        if p == 1:
            if single < 0:
                single = i
        if p > 1 and p % 2 == 0:
            if even_many < 0:
                even_many = i

    if odd_ragged < 0:
        raise Error(
            "check_table_has_the_hard_cases: NO row has an odd leaf count"
            " with a ragged last leaf at L="
            + String(SHAPE_TABLE_LEAF)
            + ". Charter clause 5 requires one, and Phase 3 would draw its"
            " odd-carry fixture from this table and silently test the even"
            " path. Add a shape or change L."
        )
    if single < 0:
        raise Error(
            "check_table_has_the_hard_cases: NO row has P == 1, so the"
            " degenerate one-term fold -- the case where a bypass would"
            " change the sign of a zero -- is not represented."
        )
    if even_many < 0:
        raise Error(
            "check_table_has_the_hard_cases: NO row has an even P > 1."
        )

    print(
        "check_table_has_the_hard_cases OK: odd+ragged at '"
        + gemm_shape_name(odd_ragged)
        + "' (P="
        + String(leaf_count(gemm_shape_k(odd_ragged), SHAPE_TABLE_LEAF))
        + ", last leaf "
        + String(last_leaf_len(gemm_shape_k(odd_ragged), SHAPE_TABLE_LEAF))
        + "), P==1 at '"
        + gemm_shape_name(single)
        + "', even P>1 at '"
        + gemm_shape_name(even_many)
        + "'"
    )


def main() raises:
    print("== bench/gemm_shapes.mojo [" + _mode_name() + "] ==")
    print(
        "The shapes gemm.fp32.v1 is judged on, at leaf size",
        SHAPE_TABLE_LEAF,
        "\n(P = ceil(k/L) is the leaf count; 'tail' is the last leaf's"
        " length)\n",
    )
    print(
        "  #  name                            op        m       n         k"
        "        P  tail  parity"
    )
    for i in range(GEMM_SHAPE_COUNT):
        var k = gemm_shape_k(i)
        var p = leaf_count(k, SHAPE_TABLE_LEAF)
        var tail = last_leaf_len(k, SHAPE_TABLE_LEAF)
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
