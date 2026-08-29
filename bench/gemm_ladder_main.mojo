"""The `gemm.fp32.v1` FOLD-LADDER card: one stage per TREE LEVEL, not one per
shape.

DEVIATION 533.

WHY THIS FILE EXISTS, AND WHY THE OTHER CARD IS NOT ENOUGH
-----------------------------------------------------------
`bench/gemm_card_main.mojo` hashes a shape's INPUTS and its OUTPUT. That is
the right card for "do Apple, NVIDIA and AMD agree", and it is the wrong
instrument for the question that comes ten seconds later. When a cross-vendor
run finally diverges, the input/output card can say

    llama8b.mlp_down.t1.out   DIFFERS

and nothing more. Three completely different investigations produce exactly
that line:

  1. a LEAF ARITHMETIC defect -- the backend contracted `a*b + c` into an FMA
     where `identical_mul_add` was supposed to pin it, or flushed a denormal
     where `ftz` was supposed to, so the level-0 partials are already wrong;
  2. a FOLD TOPOLOGY defect -- the leaves agree exactly and the tree pairs
     them differently, which is a reduction-ORDER bug and lives in a
     different file than (1);
  3. a CARRY defect -- the leaves and every paired node agree and only the
     odd tail moved, which is contract 7.2 clause 3/4 and is one `if`.

A final-token comparison cannot separate them, and this repository already
knows that from the other direction: the GBDT lane's cross-vendor run found
an NVIDIA divergence at `tree001.winners.scores` while Apple and AMD agreed,
and it found it because the card had a stage there. A card with a stage only
at the end would have reported "the model differs" and cost a week.

So this file walks the reduction tree LEVEL BY LEVEL and hashes each level.
A divergence then localizes to a LEVEL and, through the differ's `.bin`
dumps, to a NODE -- which is the difference between "the output differs" and
"level 0 agrees, level 1 moved at node 3 of cell 17".

HOW THE LADDER IS OBTAINED WITHOUT TOUCHING THE KERNEL
-------------------------------------------------------
**No kernel change was needed and none was made.** `gemm_identical.mojo`
already has an execution plan that materializes every node of the tree in
global memory: `PLAN_SPLITK_STAGED`, contract 13.4's "fully staged"
realization, which the dispatcher never picks and which exists so that
`check_device_is_launch_invariant` has a maximally distant realization to
compare `PLAN_FLAT` against. This driver runs that plan with a workspace it
owns, reads the workspace back, and hashes each level from the host.

THE WORKSPACE SLOT LAYOUT, read out of the kernel rather than assumed
---------------------------------------------------------------------
Four sources, all in `gemm/mojo_only/gemm_identical.mojo`:

  * `identical_gemm_with_plan` sets `stride = fold_node_total(P)` for
    `PLAN_SPLITK_STAGED` (it is `P` for `PLAN_SPLITK`, which materializes
    level 0 ONLY -- that plan cannot serve this card);
  * `identical_gemm_leaf_kernel` stores leaf partial `t` of cell at
    `ws[cell * stride + t]`, and `fold_level_base(P, 0) == 0`, so that IS
    level 0 at its normative address;
  * `identical_gemm_fold_level_kernel` is launched once per level `d` with
    `src_base = fold_level_base(P, d-1)` and `dst_base = fold_level_base(P,
    d)`, writing node `q` at `ws[cell * stride + dst_base + q]`;
  * `identical_gemm_emit_kernel` reads the root at
    `ws[cell * stride + fold_level_base(P, levels - 1)]`.

which is one rule:

    node (d, q) of output cell `c`  lives at
        ws[c * fold_node_total(P) + fold_level_base(P, d) + q]
    level `d` has  fold_level_width(P, d)  nodes,
    and there are  fold_level_count(P)  levels including level 0.

`fold_node_total`, `fold_level_base`, `fold_level_width` and
`fold_level_count` are imported from `gemm_oracle.mojo` -- the SAME functions
the host launcher used to place the nodes. This driver does not re-spell the
topology, because a second spelling of the addressing is a second thing that
can be wrong and would let this card hash a layout the kernel does not use.

WHAT THE CARD RECORDS, IN ORDER, PER SHAPE
-------------------------------------------
    <name>.dims        i32 x 8   m, n, k, L, P, levels, node_total, oracle op
    <name>.in.a        f32       the left operand, from the DEVICE buffer
    <name>.in.b        f32       the right operand, from the DEVICE buffer
    <name>.fold.L00    f32       every cell's LEVEL-0 LEAF PARTIALS
    <name>.fold.L01    f32       every cell's level-1 nodes
    ...                          ascending, one record per level
    <name>.out         f32       the emitted C, from the DEVICE buffer

**INTEGERS BEFORE FLOATS, PER SHAPE AND GLOBALLY.** `ladder.shapes` is the
card's first record and `<name>.dims` is each shape's first, so a run that
took a different partition -- a different `L`, a different `P`, a different
number of levels -- fails the ladder at an INTEGER stage, before a single
float is compared. That ordering is not cosmetic: two runs with different `P`
have different numbers of level records, the differ's tag alignment reports a
structural mismatch, and without the integer stage a reader has to infer from
the tag names why. With it, the first differing record says `P` moved.

A level's record is the level's nodes in `(cell ascending, q ascending)`
order. That order is a pure function of `P` and of the output shape -- it
names a POSITION IN THE ALGORITHM, never a property of the machine, which is
`core/identity_trace.mojo`'s rule 2. `MOJOLEARN_IDENTITY_TRACE_DUMP=fold`
writes each level's raw bytes beside the card so the differ can go per-node.

WHAT THIS CARD CANNOT TELL YOU
-------------------------------
  * **It is one EXECUTION plan.** Every level here is the staged plan's
    realization of the arithmetic DAG. `PLAN_FLAT` computes the same DAG in
    registers and never materializes a node; if the two plans disagreed, this
    card would show the staged one and `check_device_is_launch_invariant` is
    what notices. This card is a localizer, not an invariance gate.
  * **Matching hashes do not prove identical computation** (identity_trace's
    header says this and it is worth repeating here): they prove the two runs
    agree at those checkpoints, on this fixture.
  * **Level 0 agreeing does not prove the leaf loop is right.** It proves the
    two runs' leaf loops agree with each other. Rightness is Proof 2 below,
    against `gemm_oracle`.
  * **A traced run is not a measurement.** Every record drains the queue and
    copies to the host, and this driver additionally reads the whole
    workspace back. Nothing here is a timing.
  * **IT CANNOT SEE `SABOTAGE_PAD_PLUS_ZERO` ON THIS FIXTURE. MEASURED, NOT
    SUSPECTED.** `tools/gemm_ladder.sh sabotage PAD_PLUS_ZERO` builds the
    sabotage (the witness confirms it), runs a shape whose level 4 is SEVEN
    nodes wide so the carry path is genuinely taken, and produces a card
    IDENTICAL to the clean one at every level. The mechanism is arithmetic
    and not a plumbing failure: the sabotage replaces the bit-for-bit carry
    with `ftz(ftz(x) + 0.0)`, and that equals `x` for every `x` that is
    neither `-0.0` nor a denormal. This fixture's operands are
    `<int>/2^20`, so no carried node is ever a signed zero and none is
    subnormal, and the two spellings coincide.

    So a GREEN ladder is not evidence that the carry clause is intact, and
    the shapes must NOT be tuned until this goes red -- a fixture chosen to
    make a check fire has stopped being a record of what the library
    computes. The right fix is a fixture whose carried node is `-0.0` (the
    host-side F7 fixture in `gemm/mojo_only/gemm_oracle_check.mojo` is that
    case) or a subnormal one; the right thing to do until then is to say so
    here. This is exactly row 9's failure class -- 2^20 patterns that scored
    a contracting backend as unfused because not one separated the two
    spellings -- caught by sabotage rather than shipped.

THE THREE THINGS THIS DRIVER ASSERTS
-------------------------------------
1. **THE LADDER IS NOT INERT (`_check_top_level_is_the_output`).** The top
   level this card hashes must be THE BITS THE KERNEL EMITS. Level `D` has
   exactly one node per cell, laid out in cell order, so it must equal `C`
   element for element -- and the driver checks the two FNV hashes as well as
   the cells, and names the first differing cell. If a ladder could hash a
   tower of levels the product does not read, it would be hashing something
   the profile does not use, and a green card would mean nothing. **This
   assertion is live, not decorative: it is the ONLY thing in this file that
   catches `MOJOLEARN_GEMM_SABOTAGE_FOLD_SERIAL`**, which under the staged
   plan leaves every fold level bit-identical and rewrites the answer at the
   emit seam.
2. **THE LADDER AGREES WITH THE NORMATIVE ANSWER
   (`_check_matches_oracle`).** Under `NUMERIC_IDENTICAL` the staged plan's
   `C` must be bit-identical to `gemm/mojo_only/gemm_oracle.mojo::gemm_oracle`
   at every shape. A plan that moves bits is a Phase 3 failure, and this
   driver reports it as one rather than routing around it. Under `FAST` the
   comparison is REPORTED, not asserted -- the same seam
   `gemm_device_check.mojo` draws, for the same reason.
3. **THE WORKSPACE WAS ACTUALLY WRITTEN.** The workspace is poisoned before
   every launch. A node still holding the poison afterwards was never
   written, and hashing it would fold an allocator's leftovers into the card
   -- which differs run to run on ONE machine and would make the instrument
   report divergence everywhere. Surviving poison is counted, printed, and is
   a hard failure in a build with no sabotage active.

THE ORIENTATION TRAP, AND HOW THE MAP IS VERIFIED HERE
--------------------------------------------------------
    bench/gemm_shapes.mojo           NT = 0, TN = 1, NN = 2
    gemm/mojo_only/gemm_oracle.mojo  NN = 0, NT = 1, TN = 2

Two files, two encodings of the same three words. Passing one file's code
into the other has already shipped a whole reference card of plausible,
in-bounds, WRONG products (`bench/gemm_card_main.mojo::_oracle_op` records
it), and nothing trapped, because **`A` holds `m*k` floats whether it is read
as `m x k` or as `k x m`, and `B` holds `n*k` floats either way.** The buffer
sizes are equal in every case, so a size assertion can never see this.

`check_orientation_map` below therefore does NOT check sizes and does not
check the map against a constant written here. It checks the map two ways:

  a. the STRIDE QUADRUPLE. For each table code, this file writes down what
     the TABLE's word means as four strides -- `NT` means `B_eff[p,j] =
     b[j*k + p]`, so `(b_sp, b_sj) = (1, k)` -- and asserts that
     `gemm_operand_strides(map(code), m, n, k)`, the KERNEL's own function,
     returns it, at `m`, `n`, `k` ALL DIFFERENT so that a transposed index
     cannot look correct. A wrong map sends table `NT` to oracle `NN`, whose
     quadruple is `(k, 1, n, 1)` against the expected `(k, 1, 1, k)`, and it
     fails.
  b. the ROUND TRIP THROUGH `op_name`. The oracle's own spelling of the word
     must come back equal to the table's spelling of the same row.

(a) is the load-bearing half: it is checked against the ADDRESSING the kernel
performs, so it would still fail if both files' names agreed and their
addressing did not.

`[[mojo-string-float-roundtrip]]`: every float printed here carries its hex
bits beside its decimal.
`[[mojo-buffer-freed-at-last-use]]`: every device buffer is held past the
`ctx.synchronize()` that the read of it depends on.
"""

from std.memory import bitcast
from std.os import getenv
from std.sys.info import size_of

from max.gpu.host import DeviceBuffer, DeviceContext

from bench.gemm_shapes import (
    GEMM_SHAPE_COUNT,
    gemm_shape_k,
    gemm_shape_m,
    gemm_shape_n,
    gemm_shape_name,
    gemm_shape_op,
)
from bench.gemm_shapes import OP_NN as TBL_OP_NN
from bench.gemm_shapes import OP_NT as TBL_OP_NT
from bench.gemm_shapes import OP_TN as TBL_OP_TN
from core.identity_trace import FNV_OFFSET, IdentityTrace, fnv1a64_bytes
from gemm.mojo_only.gemm_identical import (
    PLAN_SPLITK_STAGED,
    contract_partition,
    gemm_operand_strides,
    gemm_plan_name,
    gemm_sabotage_name,
    identical_gemm_with_plan,
    identical_gemm_workspace_floats,
)
from gemm.mojo_only.gemm_oracle import (
    OP_NN,
    OP_NT,
    OP_TN,
    fold_level_base,
    fold_level_count,
    fold_level_width,
    fold_node_total,
    gemm_oracle,
    op_name,
)
from mojo_only.kernel_matrix import TARGET_COLUMN, column_name
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name

comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


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


#: Written into `C` before every launch. A cell still holding it afterwards
#: was never written, and nothing downstream of it is a comparison of
#: products.
comptime C_POISON = Float32(-987654.0)

#: Written into the WORKSPACE before every launch, and a DIFFERENT value from
#: `C_POISON` so a report can say which buffer a survivor came from. This one
#: is the load-bearing poison for this card: an unwritten fold node would
#: otherwise hash whatever the allocator last left there, which varies run to
#: run on one machine and would make every level record noise.
comptime WS_POISON = Float32(-123456.75)


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _show(x: Float32) -> String:
    return String(x) + "/" + hex(_bits(x))


def _hex16(v: UInt64) -> String:
    """Sixteen lowercase hex digits, the card's own hash spelling. Written
    here rather than imported so this file can print a hash it computed
    itself in the same form the card carries."""
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    for i in range(16):
        var nib = Int((v >> UInt64(60 - 4 * i)) & UInt64(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _hash_f32(values: List[Float32]) raises -> UInt64:
    """FNV-1a64 over `values`' little-endian bytes -- **the same function
    `IdentityTrace._emit` will apply to the same bytes**, imported rather than
    re-implemented. This driver needs the number in its own hands to assert
    Proof 1 (top level == output) without parsing the card back."""
    var tmp = values.copy()
    var h = fnv1a64_bytes(
        FNV_OFFSET,
        tmp.unsafe_ptr().bitcast[UInt8](),
        len(tmp) * size_of[Scalar[DType.float32]](),
    )
    # `[[mojo-buffer-freed-at-last-use]]`'s host cousin: the copy is dead at
    # `.unsafe_ptr()` unless something uses it afterwards.
    _ = tmp^
    return h


# ===========================================================================
# THE FIXTURE: BIT-ASSEMBLED, NO HOST FLOAT CHAIN ANYWHERE
# ===========================================================================


def _mix(i: Int, salt: Int) -> UInt64:
    var z = (
        UInt64(i + 1) * 0x9E3779B97F4A7C15
        + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _exact(i: Int, salt: Int) -> Float32:
    """`<signed int below 2^20> / 2^20`: EXACT in float32 on every backend.

    NO DECIMAL CONSTANT AND NO HOST FLOAT CHAIN. The numerator fits a float32
    mantissa and the divisor is a power of two, so the one division is exact
    and the value is a pure function of its bits -- two machines running this
    driver upload the SAME operands. A generator with a host `acc += v * w`
    in it would put IDENTITY_PATHS row 18's contraction decision upstream of
    the thing being carded, and the card would diff its own fixtures.

    **The PRODUCTS are deliberately inexact.** Two 20-bit numerators need up
    to 40 bits and float32 keeps 24, so every `a*b` rounds -- which is what
    makes the ORDER of the additions observable. A generator whose products
    were exact would make a wrong pairing invisible, and a fixture that
    cannot separate the alternatives is not evidence
    (`gemm_device_check.mojo`'s `_val` documents the same requirement).
    """
    var num = Int(_mix(i, salt) % 2097151) - 1048575
    return Float32(num) / Float32(1048576.0)


def _fill(n_elems: Int, salt: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n_elems):
        out.append(_exact(i, salt))
    return out^


def _a_elems(m: Int, n: Int, k: Int) -> Int:
    """`m * k` under EVERY orientation: `A` is `m x k` for NN and NT and
    `k x m` for TN, and those hold the same number of floats. **This is why
    the orientation trap has no size assertion that can catch it.**"""
    return m * k


def _b_elems(m: Int, n: Int, k: Int) -> Int:
    """`n * k` under every orientation, for the same reason."""
    return n * k


# ===========================================================================
# THE ORIENTATION MAP AND ITS GATE
# ===========================================================================


def _oracle_op(tbl: Int) -> Int:
    """Translate the TABLE's orientation code into the ORACLE's and the
    KERNEL's. See this file's header; `check_orientation_map` is the gate."""
    if tbl == TBL_OP_NT:
        return OP_NT
    if tbl == TBL_OP_TN:
        return OP_TN
    return OP_NN


def _tbl_op_name(tbl: Int) -> String:
    """The TABLE's own spelling of its code, transcribed from
    `bench/gemm_shapes.mojo:541-543` (its `main` prints exactly this)."""
    if tbl == TBL_OP_TN:
        return String("TN")
    if tbl == TBL_OP_NN:
        return String("NN")
    return String("NT")


def check_orientation_map() raises:
    """The map is checked against the KERNEL'S ADDRESSING, not against a
    constant. Header section "THE ORIENTATION TRAP" is the argument.

    `m`, `n` and `k` are all different and all prime, so no two of the three
    quadruples coincide and a transposed index cannot look correct.
    """
    var m = 5
    var n = 7
    var k = 11
    for q in range(3):
        var tbl = TBL_OP_NT
        if q == 1:
            tbl = TBL_OP_TN
        elif q == 2:
            tbl = TBL_OP_NN
        # WHAT THE TABLE'S WORD MEANS, spelled as strides from contract
        # section 3 and written here from the WORD, not from the oracle's
        # constants:
        #   NT : C = A[m x k] . B[n x k]^T -> A[i*k+p], B[j*k+p]
        #   TN : C = A[k x m]^T . B[k x n] -> A[p*m+i], B[p*n+j]
        #   NN : C = A[m x k] . B[k x n]   -> A[i*k+p], B[p*n+j]
        var want_a_si = k
        var want_a_sp = 1
        var want_b_sp = n
        var want_b_sj = 1
        if tbl == TBL_OP_TN:
            want_a_si = 1
            want_a_sp = m
        if tbl == TBL_OP_NT:
            want_b_sp = 1
            want_b_sj = k
        var oop = _oracle_op(tbl)
        var st = gemm_operand_strides(oop, m, n, k)
        if (
            st[0] != want_a_si
            or st[1] != want_a_sp
            or st[2] != want_b_sp
            or st[3] != want_b_sj
        ):
            raise Error(
                "check_orientation_map: table "
                + _tbl_op_name(tbl)
                + " ("
                + String(tbl)
                + ") mapped to oracle "
                + String(oop)
                + " ("
                + op_name(oop)
                + "), whose strides are ("
                + String(st[0])
                + ", "
                + String(st[1])
                + ", "
                + String(st[2])
                + ", "
                + String(st[3])
                + ") -- but the table's word "
                + _tbl_op_name(tbl)
                + " means ("
                + String(want_a_si)
                + ", "
                + String(want_a_sp)
                + ", "
                + String(want_b_sp)
                + ", "
                + String(want_b_sj)
                + "). Every buffer would still be the right size and every"
                " product would still be a plausible float."
            )
        if op_name(oop) != _tbl_op_name(tbl):
            raise Error(
                "check_orientation_map: table "
                + _tbl_op_name(tbl)
                + " maps to an oracle code the oracle itself calls '"
                + op_name(oop)
                + "'"
            )
    # And the map must not be the identity, or every caller that skipped it
    # is accidentally correct. Compared through Lists so the comparison is
    # not comptime-folded away.
    var tbl_codes = List[Int]()
    tbl_codes.append(TBL_OP_NT)
    tbl_codes.append(TBL_OP_TN)
    tbl_codes.append(TBL_OP_NN)
    var orc_codes = List[Int]()
    orc_codes.append(OP_NT)
    orc_codes.append(OP_TN)
    orc_codes.append(OP_NN)
    var same = 0
    for q in range(3):
        if tbl_codes[q] == orc_codes[q]:
            same += 1
    if same == 3:
        raise Error(
            "check_orientation_map: the two encodings now COINCIDE, so this"
            " map is the identity and this gate proves nothing. Either the"
            " two files were unified on purpose -- delete the map and this"
            " gate together -- or one file's constants moved."
        )
    print(
        "check_orientation_map OK: table (NT,TN,NN) = ("
        + String(TBL_OP_NT)
        + ","
        + String(TBL_OP_TN)
        + ","
        + String(TBL_OP_NN)
        + ") -> oracle ("
        + String(OP_NT)
        + ","
        + String(OP_TN)
        + ","
        + String(OP_NN)
        + "), verified against gemm_operand_strides at m=5 n=7 k=11"
    )


# ===========================================================================
# SHAPE SELECTION
# ===========================================================================

#: The host oracle is a scalar loop and Proof 2 runs it for every shape on
#: the ladder, so `m` and `n` are capped to fit this many multiply-
#: accumulates. **`k` IS NEVER CAPPED.** `k` is the axis the leaf rule and
#: the fold topology read (contract section 6); `m` and `n` are the axes
#: section 6.1 forbids them to read. Capping `m` and `n` therefore preserves
#: `L`, `P`, every level width and every pairing -- everything this card is
#: about -- and capping `k` would destroy all of them.
comptime LADDER_MAC_BUDGET = 4_500_000

#: A ladder shape's workspace is `m * n * fold_node_total(P)` floats and it is
#: read back to the host in full. 32 M floats is 128 MB in each place.
comptime LADDER_WS_BUDGET_FLOATS = 32 * 1024 * 1024


def _capped(m: Int, n: Int, k: Int) -> Tuple[Int, Int]:
    var mm = m
    var nn = n
    while mm * nn * k > LADDER_MAC_BUDGET and (mm > 1 or nn > 1):
        if nn >= mm and nn > 1:
            nn = (nn + 1) // 2
        elif mm > 1:
            mm = (mm + 1) // 2
        else:
            break
    return Tuple(mm, nn)


def _default_shapes() -> List[Int]:
    """The rows this ladder runs by default, and WHY each one is here.

    Chosen for VARIETY IN `P`, because `P` is the only thing that changes the
    tree:

      4  pca.transform.8192x4x4   P = 1     THE CONTROL. Contract 7.3: one
                                            level, no fold addition at all.
                                            The ladder must still record it,
                                            and level 0 must equal the output.
      8  llama8b.qkv.t1           P = 32    a five-level power-of-two tree,
                                            every level even, no carry.
     14  llama8b.mlp_down.t1      P = 112   112, 56, 28, 14, 7, 4, 2, 1 --
                                            **level 4 is SEVEN NODES WIDE**,
                                            so level 5 contains a CARRY: an
                                            unpaired node copied bit for bit
                                            rather than padded (contract 7.2
                                            clause 3). The sharpest clause in
                                            the contract, reached from a real
                                            Llama-3-8B MLP-down shape.
      3  ols.step1.16x16x64K      P = 512   ten levels, all powers of two.
                                            Depth, so a defect that only
                                            appears above level 5 has room.
      0  gram.32x32x1M            P = 1024  the profile cap, eleven levels,
                                            and the table's largest k. `L`
                                            here is 977, past the leaf rule's
                                            crossover, with a ragged tail.

    **NO SHAPE IN THE TABLE HAS AN ODD `P`, AND THAT IS NOT A CHOICE MADE
    HERE.** `bench/gemm_shapes.mojo`'s header proves it for all twenty rows
    and explains the mechanism: below the crossover `P = ceil(k/128)` and
    transformer widths are powers of two, above it the rule drives `P` toward
    `MAX_LEAVES = 1024`, itself a power of two. So an odd-`P` row would have
    to be SYNTHETIC, and inventing one to make this list look complete is the
    kind of thing `[[no-dataset-cherry-picking]]` is about in reverse. Row 14
    is the honest answer instead: it reaches the odd-width CARRY -- the same
    clause an odd `P` reaches -- at level 4, from a shape that has a caller.
    """
    var out = List[Int]()
    out.append(4)
    out.append(8)
    out.append(14)
    out.append(3)
    out.append(0)
    return out^


def _selected_shapes() raises -> List[Int]:
    """`MOJOLEARN_GEMM_LADDER_SHAPES=0,3,8` overrides the default list.

    For the SABOTAGE runs, which sometimes want one shape rather than five,
    and for a cross-vendor leg that wants to add a row. The selection is
    recorded as the card's first stage, so two runs that selected differently
    fail at record 0 rather than producing a plausible misalignment.
    """
    var raw = String(getenv("MOJOLEARN_GEMM_LADDER_SHAPES"))
    if raw == "":
        return _default_shapes()
    var out = List[Int]()
    for tok in raw.split(","):
        var s = String(tok).strip()
        if s == "":
            continue
        var idx = Int(atol(s))
        if idx < 0 or idx >= GEMM_SHAPE_COUNT:
            raise Error(
                "MOJOLEARN_GEMM_LADDER_SHAPES: index "
                + String(idx)
                + " is outside the table's 0.."
                + String(GEMM_SHAPE_COUNT - 1)
            )
        out.append(idx)
    if len(out) == 0:
        raise Error(
            "MOJOLEARN_GEMM_LADDER_SHAPES was set but parsed to no shapes,"
            " so this run would emit a card with no ladder in it and exit 0."
        )
    return out^


# ===========================================================================
# ONE SHAPE
# ===========================================================================


def _run_shape(
    ctx: DeviceContext,
    mut t: IdentityTrace,
    idx: Int,
    mut fails: Int,
) raises:
    var m0 = gemm_shape_m(idx)
    var n0 = gemm_shape_n(idx)
    var k = gemm_shape_k(idx)
    var tbl = gemm_shape_op(idx)
    var oop = _oracle_op(tbl)
    var name = gemm_shape_name(idx)

    var mn2 = _capped(m0, n0, k)
    var m = mn2[0]
    var n = mn2[1]
    var mn = m * n

    var part = contract_partition(k)
    var leaf = part[0]
    var p_count = part[1]
    var levels = fold_level_count(p_count)
    var stride = fold_node_total(p_count)
    var nws = identical_gemm_workspace_floats(m, n, k, PLAN_SPLITK_STAGED)

    print(
        "  ",
        name,
        "op=" + _tbl_op_name(tbl) + "(tbl " + String(tbl) + " -> oracle "
        + String(oop) + ")",
        "m=" + String(m) + (
            " (capped from " + String(m0) + ")" if m != m0 else ""
        ),
        "n=" + String(n) + (
            " (capped from " + String(n0) + ")" if n != n0 else ""
        ),
        "k=" + String(k),
        "L=" + String(leaf),
        "P=" + String(p_count),
        "levels=" + String(levels),
        "nodes/cell=" + String(stride),
        "ws=" + String(nws) + " floats",
    )

    if nws > LADDER_WS_BUDGET_FLOATS:
        # REPORTED, NEVER SILENT. A card that quietly omits a shape looks
        # exactly like a card whose shape agreed.
        print(
            "     SKIP -- workspace",
            nws,
            "floats exceeds the ladder budget",
            LADDER_WS_BUDGET_FLOATS,
        )
        fails += 1
        return
    if p_count <= 0:
        print("     SKIP -- P = 0, there is no tree to walk (k =", k, ")")
        fails += 1
        return

    # THE INTEGER STAGE, FIRST. A shape mismatch is visible before any float
    # is compared.
    var dims = List[Int32]()
    dims.append(Int32(m))
    dims.append(Int32(n))
    dims.append(Int32(k))
    dims.append(Int32(leaf))
    dims.append(Int32(p_count))
    dims.append(Int32(levels))
    dims.append(Int32(stride))
    dims.append(Int32(oop))
    t.record_list_i32(name + ".dims", dims)

    var na = _a_elems(m, n, k)
    var nb = _b_elems(m, n, k)
    var ha = _fill(na, 11 + idx)
    var hb = _fill(nb, 22 + idx)

    var da = ctx.enqueue_create_buffer[DType.float32](na)
    var db = ctx.enqueue_create_buffer[DType.float32](nb)
    var dc = ctx.enqueue_create_buffer[DType.float32](mn)
    var dw = ctx.enqueue_create_buffer[DType.float32](nws)
    var hA = ctx.enqueue_create_host_buffer[DType.float32](na)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](nb)
    var hC = ctx.enqueue_create_host_buffer[DType.float32](mn)
    var hW = ctx.enqueue_create_host_buffer[DType.float32](nws)
    ctx.synchronize()

    for i in range(na):
        hA.unsafe_ptr().unsafe_store(i, ha[i])
    for i in range(nb):
        hB.unsafe_ptr().unsafe_store(i, hb[i])
    for i in range(mn):
        hC.unsafe_ptr().unsafe_store(i, C_POISON)
    for i in range(nws):
        hW.unsafe_ptr().unsafe_store(i, WS_POISON)
    ctx.enqueue_copy(dst_buf=da, src_ptr=hA.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=db, src_ptr=hB.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dc, src_ptr=hC.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dw, src_ptr=hW.unsafe_ptr())
    ctx.synchronize()

    # INPUTS FIRST, ALWAYS, AND FROM THE DEVICE BUFFERS. A product diff
    # against different inputs measures nothing, and hashing the DEVICE copy
    # rather than the host list also covers the upload.
    t.record_device(ctx, name + ".in.a", da)
    t.record_device(ctx, name + ".in.b", db)

    # THE NAMED PLAN, NOT THE DISPATCHER. `choose_gemm_plan` never picks the
    # staged plan (`gemm_identical.mojo`'s header says so), so this card has
    # to ask for it by name -- and the workspace above is sized for THAT plan
    # and no other. Sizing for one plan and letting the dispatcher pick
    # another is the out-of-bounds write that cost this lane a run.
    identical_gemm_with_plan(
        ctx, dc, da, db, dw, m, n, k, oop, PLAN_SPLITK_STAGED
    )
    ctx.synchronize()

    ctx.enqueue_copy(dst_ptr=hC.unsafe_ptr(), src_buf=dc)
    ctx.enqueue_copy(dst_ptr=hW.unsafe_ptr(), src_buf=dw)
    ctx.synchronize()

    # ---- POISON: was every node actually written? ----
    var ws_poison = 0
    var first_ws_poison = -1
    for i in range(nws):
        if _bits(hW.unsafe_ptr().unsafe_load(i)) == _bits(WS_POISON):
            ws_poison += 1
            if first_ws_poison < 0:
                first_ws_poison = i
    var c_poison = 0
    for i in range(mn):
        if _bits(hC.unsafe_ptr().unsafe_load(i)) == _bits(C_POISON):
            c_poison += 1
    if ws_poison > 0 or c_poison > 0:
        print(
            "     POISON SURVIVED:",
            ws_poison,
            "of",
            nws,
            "workspace nodes and",
            c_poison,
            "of",
            mn,
            "output cells were never written.",
        )
        if first_ws_poison >= 0:
            var pc = first_ws_poison // stride
            var poff = first_ws_poison - pc * stride
            print(
                "       first unwritten node: cell",
                pc,
                "flat offset",
                poff,
                "-- level bases are",
                _level_bases_text(p_count, levels),
            )
        if gemm_sabotage_name() == "none":
            print(
                "       This build names no sabotage, so this is a real"
                " defect: the ladder is hashing memory the kernel did not"
                " write, and those bytes are an allocator's leftovers."
            )
            fails += 1
        else:
            print(
                "       Sabotage",
                gemm_sabotage_name(),
                "is active; recorded, not treated as this driver's failure.",
            )

    # ---- THE LADDER ----
    var top_level = List[Float32]()
    for d in range(levels):
        var w = fold_level_width(p_count, d)
        var base = fold_level_base(p_count, d)
        var vals = List[Float32]()
        for cell in range(mn):
            var off = cell * stride + base
            for q in range(w):
                vals.append(hW.unsafe_ptr().unsafe_load(off + q))
        var h = _hash_f32(vals)
        var tag = name + ".fold.L" + _two(d)
        t.record_list_f32(tag, vals)
        print(
            "     ",
            tag,
            "w=" + String(w),
            "nodes=" + String(len(vals)),
            "carry=" + ("yes" if (d >= 1 and fold_level_width(
                p_count, d - 1
            ) % 2 == 1) else "no"),
            _hex16(h),
        )
        if d == levels - 1:
            top_level = vals.copy()
        _ = vals^

    var cvals = List[Float32]()
    for i in range(mn):
        cvals.append(hC.unsafe_ptr().unsafe_load(i))
    t.record_device(ctx, name + ".out", dc)
    print("      " + name + ".out", "cells=" + String(mn), _hex16(
        _hash_f32(cvals)
    ))

    # ---- PROOF 1: THE LADDER IS NOT INERT ----
    _check_top_level_is_the_output(name, m, n, levels, top_level, cvals, fails)

    # ---- PROOF 2: THE LADDER AGREES WITH THE NORMATIVE ANSWER ----
    _check_matches_oracle(name, ha, hb, oop, m, n, k, cvals, fails)

    # `[[mojo-buffer-freed-at-last-use]]`: every one of these is dead at its
    # `.unsafe_ptr()` unless it is used after the last read that depends on
    # it. `hW` in particular is read for the whole ladder above.
    _ = da
    _ = db
    _ = dc
    _ = dw
    _ = hA
    _ = hB
    _ = hC
    _ = hW


def _two(d: Int) -> String:
    """Zero-padded level number, so `fold.L02` sorts before `fold.L10`. A tag
    that sorts wrong is a tag a reader mis-reads at 2am."""
    if d < 10:
        return String("0") + String(d)
    return String(d)


def _level_bases_text(p_count: Int, levels: Int) -> String:
    var s = String("")
    for d in range(levels):
        if d > 0:
            s += ","
        s += "L" + _two(d) + "@" + String(fold_level_base(p_count, d))
    return s


def _check_top_level_is_the_output(
    name: String,
    m: Int,
    n: Int,
    levels: Int,
    top_level: List[Float32],
    cvals: List[Float32],
    mut fails: Int,
) raises:
    """PROOF 1. **A path that runs is not a path that is gated.**

    The top level of the tree has exactly one node per output cell, laid out
    in cell order, and `identical_gemm_emit_kernel` copies exactly that node
    into `C` (with an `ftz` that is the identity on a value the fold already
    flushed). So the top level's hash MUST equal the output's hash. If it
    can differ without anyone noticing, this card is a tower of hashes over
    bytes the product does not read, and every green run of it means nothing.

    THIS IS THE ASSERTION THAT CATCHES `SABOTAGE_FOLD_SERIAL`. Under the
    staged plan that sabotage touches neither the leaf kernel nor the
    fold-level kernel -- every level record comes out bit-identical to a clean
    build -- and rewrites the answer inside `identical_gemm_emit_kernel`. A
    ladder without this check would report five green levels under a build
    whose product is wrong.
    """
    if len(top_level) != len(cvals):
        print(
            "      LADDER INERT:",
            name,
            "top level L" + _two(levels - 1),
            "holds",
            len(top_level),
            "nodes but the output holds",
            len(cvals),
            "cells",
        )
        fails += 1
        return
    var first = -1
    for i in range(len(cvals)):
        if _bits(top_level[i]) != _bits(cvals[i]):
            first = i
            break
    var ht = _hash_f32(top_level)
    var hc = _hash_f32(cvals)
    if first < 0 and ht == hc:
        print(
            "      proof-1 OK  top level L" + _two(levels - 1),
            "== .out, bit for bit, hash",
            _hex16(ht),
        )
        return
    print(
        "      LADDER INERT:",
        name,
        "-- the top fold level and the EMITTED OUTPUT disagree. The levels"
        " this card hashes are not the bits the product uses.",
    )
    print("        top level hash", _hex16(ht), " output hash", _hex16(hc))
    if first >= 0:
        print(
            "        first differing cell (",
            first // n,
            ",",
            first - (first // n) * n,
            "): level",
            _show(top_level[first]),
            "vs output",
            _show(cvals[first]),
        )
    if gemm_sabotage_name() != "none":
        print(
            "        Sabotage",
            gemm_sabotage_name(),
            "is active -- this is the check firing, which is what it is for.",
        )
    fails += 1


def _check_matches_oracle(
    name: String,
    ha: List[Float32],
    hb: List[Float32],
    oop: Int,
    m: Int,
    n: Int,
    k: Int,
    cvals: List[Float32],
    mut fails: Int,
) raises:
    """PROOF 2. The staged plan computes `gemm_oracle`, bit for bit.

    ASSERTED UNDER `IDENTICAL`, REPORTED UNDER `FAST`, which is the seam
    `gemm/mojo_only/gemm_device_check.mojo` and `core/gemm_identity_check.mojo`
    both draw: under FAST both sides are the unpinned spelling, the host CPU
    and the device backend are free to contract and to flush differently, and
    whether they agree is a measurement rather than a bug.

    A plan that moves bits is a Phase 3 failure and a finding. It is reported
    here, not worked around.
    """
    var want = gemm_oracle(ha, hb, oop, m, n, k)
    if len(want) != len(cvals):
        print("      ORACLE: length mismatch", len(want), "vs", len(cvals))
        fails += 1
        return
    var bad = 0
    var first = -1
    for i in range(len(want)):
        if _bits(want[i]) != _bits(cvals[i]):
            bad += 1
            if first < 0:
                first = i
    if bad == 0:
        print("      proof-2 OK  device == gemm_oracle, all", len(want), "cells")
        return
    print(
        "      ORACLE MISMATCH:",
        name,
        "--",
        bad,
        "of",
        len(want),
        "cells differ from the normative answer. First at (",
        first // n,
        ",",
        first - (first // n) * n,
        "): device",
        _show(cvals[first]),
        "oracle",
        _show(want[first]),
    )
    comptime if IDENTICAL_BUILD:
        if gemm_sabotage_name() != "none":
            print(
                "        Sabotage",
                gemm_sabotage_name(),
                "is active -- the mismatch is the sabotage, as intended.",
            )
        else:
            print(
                "        This build names no sabotage and compiled"
                " IDENTICAL, so this is a Phase 3 failure: an execution plan"
                " changed the answer's bits."
            )
        fails += 1
    else:
        print(
            "        REPORTED, not asserted: this binary compiled FAST, where"
            " the host oracle and the device backend are both free to"
            " contract and to flush and agreement is a measurement."
        )


# ===========================================================================


def main() raises:
    print("== bench/gemm_ladder_main.mojo [" + _mode_name() + "] ==")
    print("profile mojolearn.identical.gemm.fp32.v1")
    print("plan", PLAN_SPLITK_STAGED, gemm_plan_name(PLAN_SPLITK_STAGED))
    # THE SABOTAGE WITNESS, read from the comptime constant the kernel
    # compiled against. `tools/gemm_ladder.sh` greps it back out: a sabotage
    # define that was passed and silently dropped produces a card identical to
    # the clean one, which reads as "the ladder cannot see this defect" when
    # the truth is "the defect was never built".
    print("sabotage", gemm_sabotage_name())
    print("column", column_name(TARGET_COLUMN))

    var path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if path == "":
        raise Error(
            "bench/gemm_ladder_main: MOJOLEARN_IDENTITY_TRACE is unset, so"
            " this run would emit NO CARD and still exit 0 -- and a driver"
            " that checks nothing records a missing file as agreement."
        )

    check_orientation_map()
    print()

    var shapes = _selected_shapes()
    var t = IdentityTrace()
    t.header(
        String("gemm fold ladder, DEVIATION 533; mode ")
        + _mode_name()
        + "; plan "
        + gemm_plan_name(PLAN_SPLITK_STAGED)
        + "; sabotage "
        + gemm_sabotage_name()
        + "; column "
        + column_name(TARGET_COLUMN)
    )
    var sha = String(getenv("MOJOLEARN_GEMM_LADDER_SHA"))
    if sha != "":
        t.header(String("commit ") + sha)

    # THE CARD'S FIRST RECORD IS AN INTEGER RECORD. Two runs that selected
    # different rows of the table disagree here, at record 0, instead of
    # producing a tag misalignment that a reader has to decode.
    var sel = List[Int32]()
    for i in range(len(shapes)):
        sel.append(Int32(shapes[i]))
    t.record_list_i32("ladder.shapes", sel)

    var fails = 0
    # ONE CONTEXT FOR THE WHOLE CARD. A context per shape would be a
    # different control plane per record, and `HOST_AND_DEVICE.md`'s point is
    # that the control plane is part of what a run is.
    with DeviceContext() as ctx:
        for i in range(len(shapes)):
            _run_shape(ctx, t, shapes[i], fails)
            print()

    print("card:", path)
    print("records:", t.seq)
    print(
        "Localize a divergence:\n"
        "  tools/gemm_ladder.sh emit  -> a card\n"
        "  tools/gemm_ladder.sh diff A.card B.card\n"
        "The first differing record names a LEVEL. Levels below it that agree"
        " are the evidence that the defect is in the fold and not in the"
        " leaves; a level 0 that moves is the evidence that it is not."
    )
    if fails != 0:
        raise Error(
            String("gemm fold ladder: ")
            + String(fails)
            + " failure(s) above. Every verdict was printed before this"
            " raise, because under a sabotage build the useful evidence is"
            " WHICH checks a defect reaches and which it walks past."
        )
    print()
    print("gemm fold ladder: GREEN --", t.seq, "records over", len(shapes),
          "shapes; every top level is the emitted output and every output is"
          " gemm_oracle.")
