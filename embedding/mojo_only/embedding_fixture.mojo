"""The fixture set of profile `mojolearn.identical.embedding.fp32.v1`.

**THIS FILE HAS NEVER BEEN COMPILED AND HAS NEVER BEEN EXECUTED.** Written
2026-08-25 by the embedding GATE lane, DEVIATIONS 1500-1549. No `mojo`
process has read it, no device has run a case from it, and not one bit
produced by it has been observed. Every sentence below that says a fixture
SEPARATES something, or that an arm is INERT on it, is a PREDICTION derived
on paper from `embedding/IDENTICAL_EMBEDDING_CONTRACT.md` and from reading
`embedding_identical.mojo` and `embedding_oracle.mojo`. The whole point of
the file is to turn those predictions into measurements later, and the
`guard_*` functions at the foot are how the turning is done.

WHAT IS OWED AND IS NOT HERE
------------------------------
  * **A COMPILE.** Nothing has been through the front end.
  * **A RUN, ON ANY COLUMN.** Zero bits observed.
  * **`PLAN_SORT`.** Contract 6.2 specifies it and it is not written, so no
    case here can exercise contract clause (d), plan invariance.
  * **THE SHIPPED SHAPE.** Contract 11.2's sweep names `V = 128256`,
    `d = 4096`, `T = 4096`. The largest case here is `V = 300`. A
    128256 x 4096 `dW` is 2.10 GB and `[[no-heavy-local-compute]]` binds
    every author in this tree; the shipped shape belongs on a rented GPU
    and this file must not put it on Andrew's laptop. **That is a COST
    decision and not a confidence one**, and contract 11.2 calls
    `V = 128256` mandatory, so the gap is named here, named in the check's
    SCOPE line, and left open rather than quietly dropped.

WHY THIS FILE LEADS WITH THE FIXTURES THAT SEE NOTHING
--------------------------------------------------------
Contract 11.2 opens with two fixtures that would pass EVERY sabotage in this
lane while gating nothing, and it requires both, with their inert masks
ASSERTED. They are cases 0 and 1 here, and they are first in the table for
the same reason they are first in the contract.

    F-NODUP    every id DISTINCT. Every run has R <= 1, so there is no
               accumulation order to get wrong, and FIVE arms are inert on
               it -- FOLD_DESCENDING, FOLD_BALANCED_TREE, SORT_TIE_REVERSED,
               SORT_KEY_ID_ONLY_UNSTABLE and RANK_BY_ARRIVAL.
    F-DUPSAME  duplicates PRESENT, every duplicate carrying a BITWISE EQUAL
               `dY` row. The order clauses are STILL inert, because a
               permutation of a constant sequence is the same sequence.
               **This is the fixture a lane writes when it remembers to add
               duplicates and forgets that duplicates alone are not
               enough.**

THE FIVE WAYS A FIXTURE WENT BLIND IN THIS REPOSITORY ON 2026-08-25, AND
WHAT THIS FILE DOES ABOUT EACH
--------------------------------------------------------------------------
Four lanes were gated for the first time that night and the gates found four
fixtures that could separate nothing. Each failure has a countermeasure
here, spelled at the code and not only in this header.

1. **EXACTLY-REPRESENTABLE VALUES.** A generator emitting integers scaled by
   `2^-4` made every product exact, so a fold-order arm was bit neutral on
   16 of 20 rows. `emb_hashed_f32` below emits a FULL 23-BIT MANTISSA and
   `guard_hashed_dy_separates` ASSERTS that an ascending fold and a
   descending fold of the generator's own output disagree.
2. **ABSORPTION.** A control that dropped a tail term did not move, because
   the term rounded away against the running sum. `guard_hashed_dy_separates`
   is the same answer: it measures, it does not assume.
3. **ONE BINADE.** Four partials all inside `[1, 2)` made a balanced tree
   and a serial chain agree. `emb_hashed_f32` draws an EXPONENT as well as a
   mantissa, over `2^-14` to `2^14`, so a run's partials differ by up to
   `2^28`. The hand-planted runs (F-ORDER3, F-TREE4) go further and use
   `1.0` beside `2^-24`, which is one ULP of separation by construction and
   is contract 5.4's own table.
4. **A CONTROL WHOSE MECHANISM DOES NOT EXIST IN THE CASE.** A "forgot the
   step index" control was applied to SGD, which never reads a step index.
   Every inert mask in `emb_case_inert_note` names the MECHANISM that makes
   the arm inert, so a reader can check the mechanism exists before
   trusting the mask.
5. **A CLAIM INVERTED.** A case was declared an arm's INERT case where the
   arm actually FIRES. `guard_fold_reads_launch_separates` exists because
   this file found exactly that in the contract it is gating -- see below.

**THE ONE PLACE THIS FILE DISAGREES WITH `embedding_identical.mojo`'s OWN
PROSE (DEVIATION 1502).** That file's `SAB_FOLD_READS_LAUNCH` docstring says
the arm is "**Never inert on a run of length >= 2**". Reading the arm:

    var span = hi - lo
    var start = Int(block_dim.x) % span

`block_dim.x` is `EMB_TPB`, which `_emb_max_tpb` resolves to
`min(EMB_TPB_WANT=256, IDENTITY_FLOOR_BLOCK=512, column cap)` -- a POWER OF
TWO on every column this repository has (256 on the three founding columns,
128 on the portable baseline). **`256 % 2 == 0`, `256 % 4 == 0`, and in
general `EMB_TPB % span == 0` for every `span` that divides it.** At those
run lengths `start` is `0`, the rotation is the identity and the arm is
BITWISE INERT. So the claim is false at `R = 2, 4, 8, 16, ...`, which
includes `R = 2`, the very smallest run the sentence promises.

The consequence is not academic: F-DUPSAME has `R = 2`, F-TREE4 has `R = 4`,
and a gate that picked either as this arm's witness would fire the arm, see
nothing and report a working sabotage as a broken one. **F-ORDER3 has
`R = 3`**, `EMB_TPB % 3 != 0` for every power of two, and it is this file's
witness for the arm. `guard_fold_reads_launch_separates` computes the
rotation from the ACTUAL `EMB_TPB` the binary compiled with and RAISES if it
is zero, so the witness cannot silently stop separating on a column whose
block cap is different.

`[[verify-reach-not-output]]`, and `[[reached-but-inert]]` is what the
alternative would have been.

WHAT EACH CASE IS FOR
-----------------------
`emb_case_note` carries the sentence, per case, beside the case. The table
in `emb_case` is the authority; this list is the map.

     0  f_nodup        NEGATIVE CONTROL, five arms inert, mask ASSERTED
     1  f_dupsame      NEGATIVE CONTROL, order clauses inert, mask ASSERTED
     2  f_order3       R = 3. ascending vs descending, and the ONLY run
                       length in this file at which FOLD_READS_LAUNCH fires
     3  f_tree4        R = 4. the smallest run at which contract 5.4's
                       balanced tree stops being the chain
     4  f_negzero1     R = 1 with a sole `-0.0` contributor. The ONE input
                       at which the `+0.0` seed is visible
     5  f_subacc       contract 7.1's HOLE, planted: row 0 ends `-0.0` and
                       row 1 is the same run plus a `+0.0` and ends `+0.0`
     6  f_subw         a subnormal WEIGHT, `0x00000001`. The only
                       Apple-visible flush arm in this lane
     7  f_empty        13 of 16 rows empty, and the check POISONS `dW`
     8  f_pad          `padding_idx` present AND not the only id
     9  f_hot          every position carries one id. R = T = 300
    10  f_split        one row's contributors on BOTH sides of `t0 = 3`,
                       and one row's on one side only -- its own control
    11  f_oor_high     an id at `V`
    12  f_oor_neg      an id at `-1`
    13  f_multiblock   T = 600, so a run spans THREE `EMB_TPB` bands and
                       RANK_BY_ARRIVAL's phase split is not the identity
    14  f_accum        `accumulate = True`, a carried `dW`
    15  f_wide_v300    V = 300, past the 128 that several sibling profiles
                       treat as a threshold
    16  f_d1           d = 1, the narrowest nonempty width
    17  f_t0           T = 0. Contract section 8's stated value
    18  f_d0           d = 0. Contract section 8's stated value

`[[mojo-buffer-freed-at-last-use]]`: nothing here touches a device.
`[[mojo-string-float-roundtrip]]`: every float this file prints goes out as
`<decimal>/<hex bits>` through `f32_report`, and the hex is the half a
reader may trust.
`[[mojo-amp-plus-is-bitwise-and]]`: `emb_splitmix64` uses PLAIN `+`, which
wraps on `UInt64`, which is what a mixer wants. `&+` computes `x & k` with
no compile error and has produced wrong hashes in this tree twice.
`[[mojo-int-widening-sign-extends]]`: every id this file builds is an
`Int32` and every widening in `emb_hashed_f32` is masked.
"""

from std.memory import bitcast

from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from embedding.mojo_only.embedding_oracle import (
    EMB_NO_PADDING_IDX,
    EmbConfig,
    emb_backward_cell,
    emb_counts,
    emb_fold_balanced_tree_diagnostic,
    emb_max_run_length,
    emb_perm_by_scan,
    emb_run_begin,
)


# ===========================================================================
# BITS
# ===========================================================================


def f32_from_bits(u: UInt32) -> Float32:
    return bitcast[DType.float32](u)


def bits_of(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def bits32_hex(v: Float32) -> String:
    """`0x........` for a Float32.

    `[[mojo-string-float-roundtrip]]`: `String(Float32)` does not round trip
    in this toolchain, so a decimal in a log is a lossy summary and the bit
    pattern is the value. Every float this lane prints carries both."""
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def f32_report(v: Float32) -> String:
    return String(v) + "/" + bits32_hex(v)


def mode_name() -> String:
    """IDENTICAL or FAST, read at COMPILE time.

    `[[the shared checkout's mode flip]]` is the scar: a run whose mode was
    not read back gives correctly labelled measurements of the WRONG arm.
    Every driver in this lane prints this."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def mode_is_identical() -> Bool:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return True
    return False


comptime BITS_POS_ZERO: UInt32 = 0x00000000
comptime BITS_NEG_ZERO: UInt32 = 0x80000000
comptime BITS_ONE: UInt32 = 0x3F800000
comptime BITS_QNAN: UInt32 = 0x7FC00000
comptime BITS_POS_INF: UInt32 = 0x7F800000

#: `2^-24`. Contract 5.4's separator, and the reason it works: `1.0 + 2^-24`
#: is EXACTLY the midpoint of `1.0` and `1 + 2^-23`, so round-half-to-EVEN
#: returns `1.0` -- three times over in a chain. A balanced tree instead
#: pairs the three `2^-24` terms into `2^-23`, and `1.0 + 2^-23` is exact.
comptime BITS_TWO_POW_M24: UInt32 = 0x33800000

#: `1 + 2^-23`, the tree's answer to F-TREE4 and the descending chain's
#: answer to F-ORDER3. It is the value a gate must NOT see from a clean
#: build, and printing it beside the clean answer is how a reader checks the
#: separation is one ULP and not an accident of magnitude.
comptime BITS_ONE_PLUS_ULP: UInt32 = 0x3F800001

#: `-1.5 * 2^-126`, NORMAL, so it survives seam E1. Contract 7.1's `a0`.
comptime BITS_NEG_1P5_MIN_NORMAL: UInt32 = 0x80C00000

#: `+1.0 * 2^-126`, the smallest NORMAL. Contract 7.1's `a1`.
comptime BITS_MIN_NORMAL: UInt32 = 0x00800000

#: `-0.5 * 2^-126`, i.e. `-2^-127`. SUBNORMAL. It is what `a0 + a1` is
#: EXACTLY, and `ftz` of it is `-0.0` -- which is the entire hole in
#: contract 7.1's inertness theorem. Recorded as a constant so the check can
#: print the intermediate rather than assert a result and hope.
comptime BITS_NEG_2_POW_M127: UInt32 = 0x80400000

#: The smallest positive subnormal, `2^-149`. F-SUBW's planted weight.
#: `EMB_GATHER_NO_FLUSH` copies it raw and the pinned gather flushes it to
#: `+0.0`, and **a gather does no arithmetic, so this arm is NOT inert on
#: Apple** -- contract 9.3 and DEVIATION 1310. It is this lane's only
#: single-column flush proof.
comptime BITS_MIN_SUBNORMAL: UInt32 = 0x00000001

#: The poison. A large NEGATIVE NORMAL, `-2^62 * 1.something`, and
#: deliberately NOT a NaN.
#:
#: DEVIATION 1503, and it is a departure from `training/mojo_only/
#: loss_fixture.mojo::BITS_POISON`, which is `0x7FC0DEAD`, a quiet NaN. That
#: choice is right there and wrong here, and the difference is which arm the
#: poison has to survive. `SAB_EMPTY_ROW_SKIPPED` skips the seed store for
#: EVERY cell, not only the empty rows, so under that arm the fold's own
#: accumulator STARTS from the poison and the poison enters the arithmetic.
#: **A NaN entering the arithmetic makes the comparison vendor-shaped** --
#: IDENTITY_PATHS row 39 measured three payloads for one IEEE answer -- and
#: a gate whose sabotage verdict depends on a NaN payload is a gate that
#: says different things on different columns. A large finite normal
#: absorbs every contribution in this fixture set deterministically and is
#: instantly distinguishable from `+0.0`.
comptime BITS_POISON: UInt32 = 0xDEADBE00


def emb_poison(n: Int) -> List[Float32]:
    """`n` cells of `BITS_POISON`, for the `dW` buffer a gate hands the
    backward.

    **THE GATE MUST POISON AND CONTRACT 11.1 SAYS SO IN THE ARM'S OWN INERT
    COLUMN.** `SAB_EMPTY_ROW_SKIPPED` is "ALWAYS INERT if the gate pre-fills
    `dW` with zeros -- which is what a fresh allocation may or may not
    contain". A device buffer that happens to arrive zeroed makes a skipped
    `+0.0` store invisible, and a gate that relies on the allocator is a
    gate that passes for a reason nobody chose."""
    var out = List[Float32](capacity=n if n > 0 else 1)
    for _ in range(n):
        out.append(f32_from_bits(BITS_POISON))
    return out^


def count_poison(values: List[Float32]) -> Int:
    """How many cells still hold `BITS_POISON`, BY BITS.

    Reach is MEASURED. A poison that did not arrive makes the
    `EMPTY_ROW_SKIPPED` verdict meaningless and the check must say so rather
    than infer it."""
    var n = 0
    for i in range(len(values)):
        if bits_of(values[i]) == BITS_POISON:
            n += 1
    return n


def nonfinite_cells(values: List[Float32]) -> Int:
    """NaN or infinity cells, BY BITS AND NEVER BY COMPARES.

    Metal FLUSHES COMPARE OPERANDS (IDENTITY_PATHS row 49), so `v != v` is a
    test with two meanings across columns while a mask-and-compare on the
    exponent field has one. Integer operations do not flush anywhere. This
    must return exactly 1 for a single plant on every column or the clause
    (f) audit is measuring the toolchain instead of the refusal."""
    var n = 0
    for i in range(len(values)):
        var au = bits_of(values[i]) & UInt32(0x7FFFFFFF)
        if au >= UInt32(0x7F800000):
            n += 1
    return n


def subnormal_cells(values: List[Float32]) -> Int:
    """Cells whose exponent field is zero and whose mantissa is not, BY
    BITS. A signed zero is NOT counted."""
    var n = 0
    for i in range(len(values)):
        var u = bits_of(values[i])
        if (u & UInt32(0x7F800000)) == UInt32(0) and (
            u & UInt32(0x007FFFFF)
        ) != UInt32(0):
            n += 1
    return n


# ===========================================================================
# THE GENERATOR
# ===========================================================================


def emb_splitmix64(z_in: UInt64) -> UInt64:
    """splitmix64, Steele/Lea/Flood's finalizer, verbatim.

    **A THIRD COPY, and the cost is stated rather than hidden.** The mamba
    lane has `corpus_splitmix64` and the transformer lane copied it as
    `fixture_splitmix64` under DEVIATION 1000, naming the price -- "two
    copies of a hash have two chances to be edited apart". This is a third
    and the price triples. It is copied and not imported for the reason
    DEVIATION 1000 gives: those modules pull a lane's whole fixture surface
    in behind them, and this file must build with no device and no sibling
    lane present.

    `embedding_check.mojo` OWES AN ASSERTION that this and the transformer
    lane's copy agree over a seed set, and it carries one. That assertion is
    the whole mitigation.

    **`+` HERE IS A REAL ADD AND MUST STAY ONE.**
    `[[mojo-amp-plus-is-bitwise-and]]`: Mojo's `&+` computes `x & k` with NO
    compile error, and it produced wrong hashes in this repository twice,
    the second time on 2026-08-24. Do not "fix" one of these into a `&+`."""
    var z = z_in + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


comptime EMB_SEED_BASE: UInt64 = 0x456D62436F727075
"""ASCII-ish `EmbCorpu`, and distinct from `transformer_fixture.
FIXTURE_SEED_BASE` (`0x546672666D724C6C`) and from
`mamba_fixture.CORPUS_SEED_BASE` (`0x4D616D6261436F72`) ON PURPOSE. Two
lanes sharing a seed base make two different profiles' fixtures correlate,
which is harmless right up to the moment somebody compares a hash across
lanes and reads meaning into it."""


def emb_case_seed(k: Int) -> UInt64:
    """Case `k`'s seed. The `0x1000` stride is the mamba corpus's, kept so
    that the three lanes' seed schedules look the same to a reader."""
    return EMB_SEED_BASE + UInt64(0x1000) * UInt64(k)


comptime TID_DY = 1
comptime TID_W = 2
comptime TID_IDS = 3
comptime TID_DW_PREV = 4


def emb_hashed_f32(seed: UInt64, tensor_id: Int, idx: Int) -> Float32:
    """One FULL-MANTISSA, WIDE-EXPONENT Float32, built from bits.

    **THIS IS THE ANSWER TO TWO OF THE FIVE WAYS A FIXTURE WENT BLIND**, and
    it is built by ASSEMBLING A BIT PATTERN rather than by scaling a
    fraction, because the scaling spellings are exactly the ones that went
    blind.

      * **A full 23-bit mantissa.** `transformer_fixture.mojo::
        fixture_tensor` draws `lo + (hi - lo) * top24 * 2^-24` over a dyadic
        range, which is a fine generator and is one rounding -- but a
        generator that emits integers scaled by `2^-4` is the SAME SHAPE
        with a smaller draw, and that is what made 16 of 20 rows of a
        fold-order arm bit neutral on 2026-08-25. Here every one of the 23
        mantissa bits comes from the hash.
      * **A wide exponent.** Four partials all inside `[1, 2)` made a
        balanced tree and a serial chain agree, which is how a fold-order
        control was reported inert. The exponent here is drawn over
        `2^-14 .. 2^14`, so two contributors to one run can differ by `2^28`
        and the fold order decides which of them is absorbed.

    WHAT IT NEVER EMITS, and each exclusion is load bearing.

      * **No NaN and no infinity.** The exponent is clamped strictly inside
        the normal range, so contract 9.1's refusal is never fired by
        accident. A fixture that trips the refusal it is not testing is a
        fixture that reports a refusal as a failure.
      * **No subnormal and no zero.** The exponent field is never `0`. Both
        are PLANTED where they are wanted (F-SUBW, F-NEGZERO1, F-SUBACC) and
        never drawn, so `EMB_NO_FLUSH_ACC`'s inert mask -- "every fixture
        with no subnormal intermediate" -- is a property this file DECIDES
        rather than one it hopes for.

    `[[mojo-int-widening-sign-extends]]`: every widening below goes through
    a `UInt64` mask and never through a signed intermediate."""
    var key = emb_splitmix64(seed ^ (UInt64(tensor_id) << 32))
    var h = emb_splitmix64(key + UInt64(idx))
    var frac = UInt32(Int((h >> 40) & UInt64(0x007FFFFF)))
    var sign = UInt32(Int((h >> 3) & UInt64(1))) << 31
    # 29 exponent values, 113 through 141, i.e. 2^-14 through 2^14. Never
    # 0 (subnormal or zero) and never 255 (inf or NaN).
    var eb = Int((h >> 8) & UInt64(0x1F))
    var expf = 113 + (eb % 29)
    return f32_from_bits(sign | (UInt32(expf) << 23) | frac)


def emb_hashed_id(seed: UInt64, idx: Int, vocab: Int) -> Int32:
    """One id in `[0, vocab)`.

    `% vocab` on a value already reduced to 32 bits. Integers do not round
    and do not flush, so there is nothing numerical here; what matters is
    that a NEGATIVE never appears, because contract section 8 refuses one by
    name and a fixture that trips a refusal it is not testing reports the
    refusal as a failure."""
    var h = emb_splitmix64(seed ^ (UInt64(TID_IDS) << 32)) + UInt64(idx)
    var r = Int(emb_splitmix64(h) & UInt64(0x7FFFFFFF))
    return Int32(r % vocab)


# ===========================================================================
# THE CASE TABLE
# ===========================================================================

comptime EMB_CASE_COUNT = 19

comptime EMB_PLANT_NONE = 0
comptime EMB_PLANT_NODUP = 1
comptime EMB_PLANT_DUPSAME = 2
comptime EMB_PLANT_ORDER3 = 3
comptime EMB_PLANT_TREE4 = 4
comptime EMB_PLANT_NEGZERO1 = 5
comptime EMB_PLANT_SUBACC = 6
comptime EMB_PLANT_SUBW = 7
comptime EMB_PLANT_EMPTY = 8
comptime EMB_PLANT_PAD = 9
comptime EMB_PLANT_HOT = 10
comptime EMB_PLANT_SPLIT = 11
comptime EMB_PLANT_OOR_HIGH = 12
comptime EMB_PLANT_OOR_NEG = 13


def emb_plant_name(p: Int) -> String:
    if p == EMB_PLANT_NODUP:
        return String("NODUP")
    if p == EMB_PLANT_DUPSAME:
        return String("DUPSAME")
    if p == EMB_PLANT_ORDER3:
        return String("ORDER3")
    if p == EMB_PLANT_TREE4:
        return String("TREE4")
    if p == EMB_PLANT_NEGZERO1:
        return String("NEGZERO1")
    if p == EMB_PLANT_SUBACC:
        return String("SUBACC")
    if p == EMB_PLANT_SUBW:
        return String("SUBW")
    if p == EMB_PLANT_EMPTY:
        return String("EMPTY")
    if p == EMB_PLANT_PAD:
        return String("PAD")
    if p == EMB_PLANT_HOT:
        return String("HOT")
    if p == EMB_PLANT_SPLIT:
        return String("SPLIT")
    if p == EMB_PLANT_OOR_HIGH:
        return String("OOR_HIGH")
    if p == EMB_PLANT_OOR_NEG:
        return String("OOR_NEG")
    return String("none")


@fieldwise_init
struct EmbCase(Copyable, Movable):
    """One runnable case.

    `split` is contract clause (e)'s microbatch boundary, `t0`. It is `-1`
    where the case is not a split case, and it is a FIELD rather than an
    argument because contract 11.2's F-SPLIT requires the boundary to be
    chosen so that one row's contributors land on BOTH sides -- which is a
    property of the ids and the boundary TOGETHER and cannot be picked
    independently of the case.

    `refused` marks a case whose ids or values the profile must REFUSE by
    name. Such a case has no oracle answer, so clause (a) must skip it and
    the refusal audit must run it. **A refused case that leaks into clause
    (a) shows up as an oracle raise and reads like a broken gate**, which is
    the confusion this field exists to prevent."""

    var name: StaticString
    var vocab: Int
    var width: Int
    var n_positions: Int
    var padding_idx: Int
    var accumulate: Bool
    var split: Int
    var plant: Int
    var refused: Bool


def emb_case(k: Int) raises -> EmbCase:
    # name             V     d    T    pad  acc  split plant              refused
    if k == 0:
        # **NEGATIVE CONTROL 1 OF 2, contract 11.2's F-NODUP.** Every id
        # distinct, so every run has R <= 1 and there is no accumulation
        # order to get wrong. FIVE arms are inert on it and the check
        # ASSERTS the mask rather than observing it.
        return EmbCase(
            "f_nodup", 8, 3, 8, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_NODUP, False,
        )
    if k == 1:
        # **NEGATIVE CONTROL 2 OF 2, contract 11.2's F-DUPSAME.** Duplicates
        # present, every duplicate carrying a BITWISE EQUAL `dY` row, so a
        # permutation of a constant sequence is the same sequence and the
        # order clauses are STILL inert. `R_max` is 2, which the check
        # asserts, because a "duplicates" fixture whose duplicates did not
        # survive the generator is the same fixture as F-NODUP under
        # another name.
        return EmbCase(
            "f_dupsame", 4, 3, 8, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_DUPSAME, False,
        )
    if k == 2:
        # Contract 11.2's F-ORDER3. ONE run of THREE, hand planted by bits:
        # `{0x3F800000, 0x33800000, 0x33800000}`. Ascending gives
        # `0x3F800000` and descending `0x3F800001`, and the balanced tree is
        # INERT here (contract 5.4's `R = 3` row) -- which is what makes it
        # a separator for the ORDER clause specifically rather than for the
        # fold shape in general.
        #
        # **AND IT IS THIS FILE'S ONLY WITNESS FOR `EMB_FOLD_READS_LAUNCH`**,
        # DEVIATION 1502: `EMB_TPB` is a power of two on every column, so
        # `EMB_TPB % R == 0` and the arm is inert at every `R` that divides
        # it. `R = 3` does not.
        return EmbCase(
            "f_order3", 2, 1, 3, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_ORDER3, False,
        )
    if k == 3:
        # Contract 11.2's F-TREE4. ONE run of FOUR,
        # `{0x3F800000, 0x33800000, 0x33800000, 0x33800000}`. **The smallest
        # `R` at which the chain and gemm 7.2's balanced tree differ** --
        # chain `0x3F800000`, tree `0x3F800001` -- and contract 5.4 proves
        # they are node for node the same at `R <= 3`, so
        # `EMB_FOLD_BALANCED_TREE` is PROVABLY inert below this case and a
        # gate whose longest run is 3 deletes the arm as broken.
        return EmbCase(
            "f_tree4", 2, 1, 4, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_TREE4, False,
        )
    if k == 4:
        # Contract 11.2's F-NEGZERO1. A single contributor of `-0.0`.
        # **THE ONLY INPUT IN THE WHOLE PROFILE AT WHICH THE `+0.0` SEED IS
        # VISIBLE**: pinned gives `ftz((+0.0) + (-0.0)) = +0.0`, and both
        # `EMB_SINGLE_RUN_BYPASS` and `EMB_SEED_SEEDLESS` give `0x80000000`.
        # Two arms, one input, and contract 11.1 says both exist precisely
        # because a gate carrying only one would not know which clause it
        # had proved.
        return EmbCase(
            "f_negzero1", 3, 1, 1, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_NEGZERO1, False,
        )
    if k == 5:
        # Contract 11.2's F-SUBACC and contract 7.1's HOLE, planted so the
        # exception is MEASURED and not merely admitted.
        #
        # `ids = [0, 0, 1, 1, 1]`. Row 0's run is `{a0, a1}` and row 1's is
        # `{a0, a1, +0.0}`, with `a0 = 0x80C00000` and `a1 = 0x00800000`,
        # both NORMAL so both survive seam E1.
        #
        #     acc after a0 = -1.5 * 2^-126
        #     acc after a1 = ftz(-0.5 * 2^-126) = ftz(0x80400000) = -0.0
        #     row 0 -> 0x80000000
        #     row 1 -> ftz((-0.0) + (+0.0)) = +0.0 -> 0x00000000
        #
        # **An exactly-zero contributor has moved a bit**, which is the
        # counterexample to "a `+0.0` contributor is inert". The seed forbids
        # reaching `-0.0` by ADDITION and does not forbid reaching it
        # through `ftz` of a negative subnormal partial sum.
        #
        # **UNDER `NUMERIC_FAST` THE WHOLE CASE IS DIFFERENT AND THE CHECK
        # MUST NOT ASSERT IT**: `ftz` compiles away, the partial sum stays
        # `0x80400000`, and row 0 ends subnormal rather than `-0.0`. That is
        # not a defect, it is FAST making no identity claim, and
        # `guard_subacc_reaches` reports rather than raises there.
        return EmbCase(
            "f_subacc", 2, 1, 5, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_SUBACC, False,
        )
    if k == 6:
        # Contract 11.2's F-SUBW. A subnormal WEIGHT cell, `0x00000001`.
        # A gather performs no arithmetic, so a raw copy of a subnormal
        # survives on EVERY vendor including an FTZ one -- which makes
        # `EMB_GATHER_NO_FLUSH` the ONE flush arm in this lane that a
        # single-column run can see move (contract 9.3, DEVIATION 1310).
        return EmbCase(
            "f_subw", 4, 2, 4, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_SUBW, False,
        )
    if k == 7:
        # Contract 11.2's F-EMPTY. Three ids over sixteen rows, so THIRTEEN
        # rows are empty and their `+0.0` is a STATED value the
        # implementation must WRITE. The check POISONS `dW` first --
        # contract 11.1 says `EMB_EMPTY_ROW_SKIPPED` is "always inert if the
        # gate pre-fills with zeros", and a fresh device allocation may
        # already be zero.
        return EmbCase(
            "f_empty", 16, 2, 3, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_EMPTY, False,
        )
    if k == 8:
        # Contract 11.2's F-PAD. `padding_idx = 2`, carried by THREE of the
        # six positions, and **`padding_idx` is not the only id** -- which
        # is the half of the requirement a hurried fixture drops. Without
        # other ids present, `EMB_PAD_ROW_CONTRIBUTES` would move
        # `emb.counts` into a vocabulary that has nothing else in it and the
        # `emb.perm` displacement it causes would not be observable.
        return EmbCase(
            "f_pad", 6, 2, 6, 2, False, -1, EMB_PLANT_PAD, False
        )
    if k == 9:
        # Contract 11.2's F-HOT. Every position carries ONE id, so `R == T`
        # and the parallel width collapses from `V * d` to `d`. Contract
        # 12.2 prices it; here it is the degenerate DEPTH and the `T >= 129`
        # shape contract 11.2 calls mandatory.
        return EmbCase(
            "f_hot", 4, 2, 300, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_HOT, False,
        )
    if k == 10:
        # Contract 11.2's F-SPLIT, with BOTH halves in one case.
        # `ids = [0, 1, 0, 2, 1, 0]` and `t0 = 3`.
        #   row 0 has contributors at t = 0, 2 (before) and t = 5 (after)
        #         -> STRADDLES. This is what clause (e) is for.
        #   row 2 has its one contributor at t = 3 -> entirely AFTER.
        #         -> its own INERT control for `EMB_ACCUM_BY_ADD`, whose
        #            inert mask is "any split that leaves every row's
        #            contributors on one side".
        return EmbCase(
            "f_split", 4, 2, 6, EMB_NO_PADDING_IDX, False, 3,
            EMB_PLANT_SPLIT, False,
        )
    if k == 11:
        # Contract 11.2's F-OOR, high half. One id at exactly `V`. REFUSED
        # by name, never clamped: a clamp turns a data bug into a wrong
        # gradient on a REAL vocabulary row and there is no stage at which
        # that becomes visible.
        return EmbCase(
            "f_oor_high", 4, 2, 3, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_OOR_HIGH, True,
        )
    if k == 12:
        # F-OOR, low half. One id at `-1`. It is a SEPARATE case from the
        # high half because they exercise two different branches of
        # `emb_refuse_ids` and because `-1` is also the value of
        # `EMB_NO_PADDING_IDX` -- an id of `-1` reaching the gather would be
        # `w[-1 * d + j]`, a read before the buffer, which the oracle's own
        # docstring names as the reason `padding_idx` gets no exemption from
        # the range check.
        return EmbCase(
            "f_oor_neg", 4, 2, 3, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_OOR_NEG, True,
        )
    if k == 13:
        # `EMB_RANK_BY_ARRIVAL`'s witness, and the shape is not arbitrary.
        #
        # DEVIATION 1504. The arm's device spelling partitions a run by
        # `(t // block_dim.x) % 2` and emits phase 0 then phase 1. With
        # `T <= 2 * EMB_TPB` phase 0 is `[0, EMB_TPB)` and phase 1 is
        # `[EMB_TPB, T)`, and CONCATENATING THEM REPRODUCES ASCENDING
        # ORDER -- the arm is INERT. It only reorders once a run reaches a
        # THIRD band, `[2*EMB_TPB, T)`, which phase 0 also claims and which
        # therefore lands BEFORE the second band's positions.
        #
        # So the requirement is `T > 2 * EMB_TPB`, and `EMB_TPB` is 256 on
        # the three founding columns and 128 on the portable baseline. `600`
        # clears `512`. `guard_rank_by_arrival_separates` recomputes this
        # from the ACTUAL `EMB_TPB` the binary compiled with and raises if
        # the case has stopped separating, because a witness that silently
        # goes inert on a new column is the failure this whole file is
        # about.
        return EmbCase(
            "f_multiblock", 3, 1, 600, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_NONE, False,
        )
    if k == 14:
        # `accumulate = True`, contract 7.4. The carried `dW` comes from
        # `emb_case_dw_prev`. It is a case of its own as well as clause
        # (e)'s machinery, because `EMB_ACCUM_REFILLS` moves `emb.dw_seed`
        # and a lane that only ever ran fresh calls has no stage on which to
        # see it.
        return EmbCase(
            "f_accum", 4, 2, 6, EMB_NO_PADDING_IDX, True, -1,
            EMB_PLANT_SPLIT, False,
        )
    if k == 15:
        # Shape sweep. `V = 300` is past 128, which is the leaf threshold
        # several sibling profiles turn on, and it gives
        # `emb_run_begin_kernel`'s single-threaded serial scan 300 dependent
        # integer additions instead of a handful. Nothing numerical turns on
        # it (integer addition is exactly associative, contract 6.1) and it
        # is here so that the claim is exercised rather than only argued.
        return EmbCase(
            "f_wide_v300", 300, 1, 64, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_NONE, False,
        )
    if k == 16:
        # `d = 1`, the narrowest nonempty width. Contract 11.2's sweep.
        return EmbCase(
            "f_d1", 5, 1, 5, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_NONE, False,
        )
    if k == 17:
        # `T == 0`. Contract section 8: no run has a contributor, `dW` is
        # `+0.0` everywhere STORED, and `Y` is empty. A STATED value, not an
        # early return that leaves the buffer as found.
        return EmbCase(
            "f_t0", 4, 2, 0, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_NONE, False,
        )
    if k == 18:
        # `d == 0`. Contract section 8: `Y` and `dW` are empty and nothing
        # is written. It is a separate case from `T == 0` because the two
        # take different early returns in both halves of the lane and a
        # single degenerate case would exercise one of them.
        return EmbCase(
            "f_d0", 4, 0, 3, EMB_NO_PADDING_IDX, False, -1,
            EMB_PLANT_NONE, False,
        )
    raise Error(
        String("embedding_fixture: no case ")
        + String(k)
        + " (there are "
        + String(EMB_CASE_COUNT)
        + ")"
    )


def emb_case_by_name(name: String) raises -> Int:
    for k in range(EMB_CASE_COUNT):
        var c = emb_case(k)
        if String(c.name) == name:
            return k
    raise Error(
        String("embedding_fixture: no case named '")
        + name
        + "'. The gate names cases by STRING and a name that does not exist"
        + " RAISES rather than falling back to a default, because a green"
        + " for a shape nobody chose is the card-path alias defect in"
        + " another costume."
    )


def emb_case_index(c: EmbCase) raises -> Int:
    """Case `c`'s index in the table, by NAME.

    Every data generator below needs `emb_case_seed(k)` and every one of
    them is handed an `EmbCase` rather than a `k`. Resolving the index here,
    ONCE, is the alternative to threading an index field through nineteen
    constructor calls where a mis-typed one would silently give two cases
    the same seed -- and two cases with one seed correlate, which is
    harmless right up to the moment somebody compares two hashes and reads
    meaning into the agreement.

    A name that is not in the table RAISES. It cannot happen from inside
    this file; it can happen if a caller builds an `EmbCase` by hand, and
    silently seeding such a case from case 0 is worse than refusing it."""
    for k in range(EMB_CASE_COUNT):
        var cc = emb_case(k)
        if String(cc.name) == String(c.name):
            return k
    raise Error(
        String("embedding_fixture: the case named '")
        + String(c.name)
        + "' is not in the table, so it has no seed"
    )


def emb_case_config(c: EmbCase) -> EmbConfig:
    return EmbConfig(c.vocab, c.width, c.padding_idx, c.accumulate)


# ===========================================================================
# THE CASE DATA
# ===========================================================================


def emb_case_ids(c: EmbCase) raises -> List[Int32]:
    """Case `c`'s `[T]` ids.

    Every planted branch below is spelled OUT rather than derived, because
    contract 11.2's whole point is that a fixture's separating power is a
    property of the exact ids and a derived id list is one whose duplicates
    nobody has counted. `emb_case_max_run` is the instrument that checks the
    result and `guard_case_shapes` runs it over the whole table."""
    var t = c.n_positions
    var ids = List[Int32](capacity=t if t > 0 else 1)

    if c.plant == EMB_PLANT_NODUP:
        # ALL DISTINCT. `T == V` here, so this is the identity permutation
        # and every run has length exactly 1.
        for i in range(t):
            ids.append(Int32(i))
        return ids^

    if c.plant == EMB_PLANT_DUPSAME:
        # `0,1,2,3,0,1,2,3`. Every id appears exactly twice, so `R_max == 2`
        # -- and `emb_case_dy` gives position `i` and position `i + V` the
        # SAME BITS, which is what makes the order clauses inert.
        for i in range(t):
            ids.append(Int32(i % c.vocab))
        return ids^

    if (
        c.plant == EMB_PLANT_ORDER3
        or c.plant == EMB_PLANT_TREE4
        or c.plant == EMB_PLANT_HOT
    ):
        # ONE run holding every position. Row 0 for the two planted folds,
        # row 1 for F-HOT so that row 0 is EMPTY and the case exercises both
        # the degenerate depth and the stated empty value at once.
        var v = Int32(0) if c.plant != EMB_PLANT_HOT else Int32(1)
        for _ in range(t):
            ids.append(v)
        return ids^

    if c.plant == EMB_PLANT_NEGZERO1:
        ids.append(Int32(1))
        return ids^

    if c.plant == EMB_PLANT_SUBACC:
        # `[0, 0, 1, 1, 1]`. Row 0 gets the two-contributor run that ends
        # `-0.0`; row 1 gets the same two plus a trailing `+0.0`.
        ids.append(Int32(0))
        ids.append(Int32(0))
        ids.append(Int32(1))
        ids.append(Int32(1))
        ids.append(Int32(1))
        return ids^

    if c.plant == EMB_PLANT_EMPTY:
        ids.append(Int32(2))
        ids.append(Int32(5))
        ids.append(Int32(9))
        return ids^

    if c.plant == EMB_PLANT_PAD:
        # `padding_idx` is 2 and appears at t = 0, 2, 5. Ids 0, 1 and 3 are
        # also present, which is the half of contract 11.2's F-PAD
        # requirement ("`padding_idx` also NOT the only id") that a hurried
        # fixture drops.
        ids.append(Int32(2))
        ids.append(Int32(0))
        ids.append(Int32(2))
        ids.append(Int32(3))
        ids.append(Int32(1))
        ids.append(Int32(2))
        return ids^

    if c.plant == EMB_PLANT_SPLIT:
        # `[0, 1, 0, 2, 1, 0]`. At `t0 = 3`: row 0 straddles (t = 0, 2 | 5),
        # row 1 straddles (t = 1 | 4), row 2 is entirely on the second side
        # (t = 3) and row 3 is empty. Four different relationships to one
        # boundary, in six positions.
        ids.append(Int32(0))
        ids.append(Int32(1))
        ids.append(Int32(0))
        ids.append(Int32(2))
        ids.append(Int32(1))
        ids.append(Int32(0))
        return ids^

    if c.plant == EMB_PLANT_OOR_HIGH:
        ids.append(Int32(1))
        ids.append(Int32(c.vocab))  # REFUSED: at V
        ids.append(Int32(0))
        return ids^

    if c.plant == EMB_PLANT_OOR_NEG:
        ids.append(Int32(1))
        ids.append(Int32(-1))  # REFUSED: below zero
        ids.append(Int32(0))
        return ids^

    if c.plant == EMB_PLANT_SUBW:
        # Ordinary ids; the plant is in the WEIGHT. Two duplicates so the
        # backward half of the case is not itself blind.
        ids.append(Int32(0))
        ids.append(Int32(2))
        ids.append(Int32(0))
        ids.append(Int32(3))
        return ids^

    var seed = emb_case_seed(emb_case_index(c))
    for i in range(t):
        ids.append(emb_hashed_id(seed, i, c.vocab))
    return ids^


def emb_case_dy(c: EmbCase) raises -> List[Float32]:
    """Case `c`'s `[T, d]` incoming gradient.

    The unplanted cells come from `emb_hashed_f32`, full mantissa and wide
    exponent, for the reasons in that function's docstring. The planted
    cases are written OUT BY BITS, because a decimal cannot say
    `0x33800000` and `[[mojo-string-float-roundtrip]]` says a decimal in
    source is not the value either."""
    var t = c.n_positions
    var d = c.width
    var n = t * d
    var dy = List[Float32](capacity=n if n > 0 else 1)

    var seed = emb_case_seed(emb_case_index(c))

    if c.plant == EMB_PLANT_ORDER3:
        # Contract 11.2's F-ORDER3, `d == 1`.
        dy.append(f32_from_bits(BITS_ONE))
        dy.append(f32_from_bits(BITS_TWO_POW_M24))
        dy.append(f32_from_bits(BITS_TWO_POW_M24))
        return dy^

    if c.plant == EMB_PLANT_TREE4:
        dy.append(f32_from_bits(BITS_ONE))
        dy.append(f32_from_bits(BITS_TWO_POW_M24))
        dy.append(f32_from_bits(BITS_TWO_POW_M24))
        dy.append(f32_from_bits(BITS_TWO_POW_M24))
        return dy^

    if c.plant == EMB_PLANT_NEGZERO1:
        dy.append(f32_from_bits(BITS_NEG_ZERO))
        return dy^

    if c.plant == EMB_PLANT_SUBACC:
        # rows 0,1 -> vocabulary row 0; rows 2,3,4 -> vocabulary row 1.
        dy.append(f32_from_bits(BITS_NEG_1P5_MIN_NORMAL))
        dy.append(f32_from_bits(BITS_MIN_NORMAL))
        dy.append(f32_from_bits(BITS_NEG_1P5_MIN_NORMAL))
        dy.append(f32_from_bits(BITS_MIN_NORMAL))
        dy.append(f32_from_bits(BITS_POS_ZERO))
        return dy^

    if c.plant == EMB_PLANT_DUPSAME:
        # **THE WHOLE POINT OF THIS CASE.** Position `i` and position
        # `i + V` carry the SAME id and the SAME BITS, so a permutation of
        # the run is the same sequence of additions and every order clause
        # is inert. Drawing two independent hashes here would make the case
        # a TEST rather than a CONTROL and would silently delete the
        # contract's second negative control.
        for i in range(n):
            var pos = i // d
            var col = i - pos * d
            var canonical = (pos % c.vocab) * d + col
            dy.append(emb_hashed_f32(seed, TID_DY, canonical))
        return dy^

    for i in range(n):
        dy.append(emb_hashed_f32(seed, TID_DY, i))
    return dy^


def emb_case_weight(c: EmbCase) raises -> List[Float32]:
    """Case `c`'s `[V, d]` embedding table.

    F-SUBW plants `0x00000001` at cell `len / 2` -- **not at cell 0**. A
    plant at index 0 is the one a loop that skips its first element would
    still catch, which makes it the weakest possible plant site; the
    transformer lane's clause (e) makes the same choice for the same
    reason."""
    var n = c.vocab * c.width
    var w = List[Float32](capacity=n if n > 0 else 1)
    var seed = emb_case_seed(emb_case_index(c))
    for i in range(n):
        w.append(emb_hashed_f32(seed, TID_W, i))
    if c.plant == EMB_PLANT_SUBW and n > 0:
        w[n // 2] = f32_from_bits(BITS_MIN_SUBNORMAL)
    return w^


def emb_case_dw_prev(c: EmbCase) raises -> List[Float32]:
    """The carried-in `dW` for an `accumulate` case, `[V, d]`.

    Empty when the case does not accumulate: `emb_backward_seed` refuses a
    `dw_prev` of the wrong length under `accumulate`, and an empty list on a
    non-accumulating call is never read.

    **THE CARRIED VALUES ARE HASHED, NOT ZERO.** A zero carry makes the
    carried path bitwise identical to a fresh one, which would make contract
    clause (e) pass on an implementation that ignored the carry entirely --
    `EMB_ACCUM_REFILLS` in particular, whose whole content is erasing it."""
    if not c.accumulate:
        return List[Float32]()
    var n = c.vocab * c.width
    var out = List[Float32](capacity=n if n > 0 else 1)
    var seed = emb_case_seed(emb_case_index(c))
    for i in range(n):
        out.append(emb_hashed_f32(seed, TID_DW_PREV, i))
    return out^


# ===========================================================================
# THE INSTRUMENTS: what a case CAN and CANNOT see
# ===========================================================================


def emb_case_max_run(c: EmbCase) raises -> Int:
    """`R_max` for this case, measured from its own ids.

    **`R_max` IS A PROPERTY OF THE DATA AND NOT OF THE SHAPE**, which is
    exactly why `embedding_oracle.mojo::emb_max_run_length` exists and says
    so. `EMB_FOLD_BALANCED_TREE` is PROVABLY inert at every `R <= 3`, so a
    gate that plants duplicates without measuring how long the longest run
    came out will report a working arm as a broken one."""
    var ids = emb_case_ids(c)
    var cfg = emb_case_config(c)
    return emb_max_run_length(emb_counts(ids, cfg))


def emb_case_distinct_ids(c: EmbCase) raises -> Int:
    var ids = emb_case_ids(c)
    var cfg = emb_case_config(c)
    var counts = emb_counts(ids, cfg)
    var n = 0
    for v in range(len(counts)):
        if Int(counts[v]) > 0:
            n += 1
    return n


def emb_case_empty_rows(c: EmbCase) raises -> Int:
    var ids = emb_case_ids(c)
    var cfg = emb_case_config(c)
    var counts = emb_counts(ids, cfg)
    var n = 0
    for v in range(len(counts)):
        if Int(counts[v]) == 0:
            n += 1
    return n


def emb_case_pad_positions(c: EmbCase) raises -> Int:
    """Positions carrying `padding_idx`. `EMB_PAD_ROW_CONTRIBUTES` and
    `EMB_PAD_ROW_NEG_ZERO` are INERT ENTIRELY when this is zero, and the
    gate must assert the mask rather than observe it."""
    if not emb_case_config(c).has_padding():
        return 0
    var ids = emb_case_ids(c)
    var n = 0
    for t in range(len(ids)):
        if Int(ids[t]) == c.padding_idx:
            n += 1
    return n


def emb_case_straddling_rows(c: EmbCase) raises -> Int:
    """Vocabulary rows with a contributor STRICTLY BEFORE `c.split` and one
    at or after it.

    Contract 11.1 gives `EMB_ACCUM_BY_ADD` the inert mask "any split that
    leaves every row's contributors on one side". This function is that mask
    as a number: zero means the split is the arm's INERT control, and a
    positive count means it is the arm's witness. **A clause (e) whose
    fixture has zero straddling rows passes on the by-add spelling as well
    as on the carry**, which is the shape of a gate that gates nothing."""
    if c.split < 0:
        return 0
    var ids = emb_case_ids(c)
    var cfg = emb_case_config(c)
    var before = List[Bool]()
    var after = List[Bool]()
    for _ in range(cfg.vocab):
        before.append(False)
        after.append(False)
    for t in range(len(ids)):
        var v = Int(ids[t])
        if v < 0 or v >= cfg.vocab or v == cfg.padding_idx:
            continue
        if t < c.split:
            before[v] = True
        else:
            after[v] = True
    var n = 0
    for v in range(cfg.vocab):
        if before[v] and after[v]:
            n += 1
    return n


def emb_case_subnormal_weights(c: EmbCase) raises -> Int:
    """Subnormal cells in `W`. `EMB_GATHER_NO_FLUSH`'s inert mask is "every
    fixture with no subnormal weight", which is every fixture in this file
    except F-SUBW."""
    var w = emb_case_weight(c)
    return subnormal_cells(w)


def emb_case_note(c: EmbCase) raises -> String:
    """One line naming what the case is for and what it CANNOT see.

    The second half is the half that matters and it is the half a fixture
    table usually omits."""
    if c.plant == EMB_PLANT_NODUP:
        return String(
            "NEGATIVE CONTROL: all ids distinct, R_max <= 1. Blind to EVERY"
            " order and sort clause -- FOLD_DESCENDING, FOLD_BALANCED_TREE,"
            " SORT_TIE_REVERSED, SORT_KEY_ID_ONLY_UNSTABLE and"
            " RANK_BY_ARRIVAL are ALL inert, and the gate asserts that mask"
        )
    if c.plant == EMB_PLANT_DUPSAME:
        return String(
            "NEGATIVE CONTROL: duplicates carrying BITWISE EQUAL dY rows."
            " Blind to the ORDER, still, because a permutation of a constant"
            " sequence is the same sequence"
        )
    if c.plant == EMB_PLANT_ORDER3:
        return String(
            "R=3: ascending 0x3f800000 vs descending 0x3f800001; the tree is"
            " INERT here (contract 5.4); the ONLY FOLD_READS_LAUNCH witness"
        )
    if c.plant == EMB_PLANT_TREE4:
        return String(
            "R=4: chain 0x3f800000 vs balanced tree 0x3f800001, the smallest"
            " R at which they differ"
        )
    if c.plant == EMB_PLANT_NEGZERO1:
        return String(
            "R=1 with a sole -0.0 contributor: the ONE input at which the"
            " +0.0 seed is visible. SINGLE_RUN_BYPASS and SEED_SEEDLESS both"
            " need it and NOTHING ELSE separates them"
        )
    if c.plant == EMB_PLANT_SUBACC:
        return String(
            "contract 7.1's HOLE: row 0 ends -0.0 through ftz of a negative"
            " subnormal partial sum, row 1 is the same run plus a +0.0 and"
            " ends +0.0. Asserted as a KNOWN EXCEPTION, IDENTICAL only"
        )
    if c.plant == EMB_PLANT_SUBW:
        return String(
            "a subnormal WEIGHT: GATHER_NO_FLUSH's only witness, and the one"
            " flush arm in this lane that is NOT inert on Apple"
        )
    if c.plant == EMB_PLANT_EMPTY:
        return String(
            "13 of 16 rows empty: the +0.0 STORE, gated only if the buffer"
            " is POISONED first"
        )
    if c.plant == EMB_PLANT_PAD:
        return String(
            "padding_idx present AND not the only id: PAD_ROW_CONTRIBUTES"
            " must move emb.counts and must NOT move emb.dw"
        )
    if c.plant == EMB_PLANT_HOT:
        return String(
            "R == T == 300: the degenerate depth, and T >= 129. Blind to"
            " nothing in particular and expensive"
        )
    if c.plant == EMB_PLANT_SPLIT:
        return String(
            "one row straddles t0=3 and one row does not: clause (e)'s"
            " witness and its own inert control in one case"
        )
    if c.plant == EMB_PLANT_OOR_HIGH or c.plant == EMB_PLANT_OOR_NEG:
        return String(
            "REFUSED by name. No oracle answer exists, so clause (a) must"
            " SKIP it and the refusal audit must run it"
        )
    return String("shape sweep; no plant")


# ===========================================================================
# THE GUARDS: the DEMONSTRATION that each fixture CAN separate
#
# **EVERY FIXTURE THIS FILE SHIPS COMES WITH THE DEMONSTRATION THAT IT CAN
# SEPARATE.** `training/mojo_only/loss_check.mojo`'s guard 4 is the model:
# it proves its exact family separates NO spelling by showing a
# reciprocal-multiply and a true divide agreeing at a power-of-two divisor
# and DISAGREEING at 3.0 -- a one-sided version would be indistinguishable
# from a broken comparison.
#
# Every guard below is TWO-SIDED for the same reason. Each one runs the
# HOST oracle's own arms against the HOST oracle's own pinned fold, on the
# fixture's own bits, and RAISES if the two agree where they must differ OR
# agree everywhere. None of them touches a device, so a guard can run before
# a `DeviceContext` exists and a build with a blind fixture fails in a
# second rather than after a case sweep.
# ===========================================================================


def emb_chain(values: List[Float32], seed: Float32) -> Float32:
    """The contract's fold, on a bare list, ASCENDING. `emb_backward_cell`
    with the permutation flattened away, so a guard can talk about a run
    without building a fixture around it."""
    var acc = seed
    for i in range(len(values)):
        acc = ftz(ftz(acc) + ftz(values[i]))
    return ftz(acc)


def emb_chain_descending(values: List[Float32], seed: Float32) -> Float32:
    """`EMB_FOLD_DESCENDING`'s arm, on the host."""
    var acc = seed
    var i = len(values) - 1
    while i >= 0:
        acc = ftz(ftz(acc) + ftz(values[i]))
        i -= 1
    return ftz(acc)


def emb_chain_rotated(
    values: List[Float32], seed: Float32, block: Int
) -> Float32:
    """`EMB_FOLD_READS_LAUNCH`'s arm, on the host, spelled EXACTLY as
    `emb_backward_kernel` spells it -- `start = block % span`, then a
    wrapping walk.

    It takes `block` as an argument rather than reading `EMB_TPB` so that a
    guard can show the arm going INERT at one block size and firing at
    another, which is DEVIATION 1502's whole finding."""
    var span = len(values)
    if span == 0:
        return ftz(seed)
    var start = block % span
    var acc = seed
    for q in range(span):
        var r = (start + q) % span
        acc = ftz(ftz(acc) + ftz(values[r]))
    return ftz(acc)


def guard_order3_separates() raises -> String:
    """F-ORDER3 separates the ASCENDING order from the descending one, and
    does NOT separate the balanced tree. BOTH halves are asserted.

    The second half is the one that makes the fixture a REACH PROOF rather
    than a smoke test: contract 5.4 proves the tree IS the chain at `R = 3`,
    node for node, so if the tree arm moved here the arm is not the tree
    the contract refused and one of the two documents is wrong."""
    var c = emb_case(emb_case_by_name(String("f_order3")))
    var dy = emb_case_dy(c)
    var zero = f32_from_bits(BITS_POS_ZERO)
    var asc = emb_chain(dy, zero)
    var desc = emb_chain_descending(dy, zero)
    var tree = emb_fold_balanced_tree_diagnostic(dy)
    if bits_of(asc) == bits_of(desc):
        raise Error(
            String("embedding_fixture: F-ORDER3 IS BLIND. Ascending ")
            + bits32_hex(asc)
            + " and descending "
            + bits32_hex(desc)
            + " agree, so EMB_FOLD_DESCENDING cannot be falsified by this"
            + " fixture and every 'the arm did not bite' verdict taken on"
            + " it is meaningless ([[reached-but-inert]])."
        )
    if bits_of(tree) != bits_of(asc):
        raise Error(
            String("embedding_fixture: F-ORDER3's TREE half moved. chain ")
            + bits32_hex(asc)
            + ", tree "
            + bits32_hex(tree)
            + ". Contract 5.4 proves the balanced tree IS the ascending"
            + " chain at R = 3, node for node, so either the diagnostic tree"
            + " is not gemm 7.2's tree or contract 5.4 is wrong. Both are"
            + " findings and neither is this fixture failing."
        )
    return (
        String("F-ORDER3 R=3: ascending ")
        + bits32_hex(asc)
        + " vs descending "
        + bits32_hex(desc)
        + " (SEPARATES), tree "
        + bits32_hex(tree)
        + " (INERT, contract 5.4)"
    )


def guard_tree4_separates() raises -> String:
    """F-TREE4 separates the chain from the balanced tree, and the guard
    prints the ULP so a reader can see the separation is one bit and not an
    accident of magnitude."""
    var c = emb_case(emb_case_by_name(String("f_tree4")))
    var dy = emb_case_dy(c)
    var zero = f32_from_bits(BITS_POS_ZERO)
    var chain = emb_chain(dy, zero)
    var tree = emb_fold_balanced_tree_diagnostic(dy)
    if bits_of(chain) == bits_of(tree):
        raise Error(
            String("embedding_fixture: F-TREE4 IS BLIND. The chain and the")
            + " balanced tree both give "
            + bits32_hex(chain)
            + " at R = 4, and contract 5.4's table says they must differ"
            + " there. Either the four planted bits are not the contract's"
            + " or round-half-to-even is not what this toolchain does"
            + " ([[reached-but-inert]])."
        )
    if bits_of(chain) != BITS_ONE or bits_of(tree) != BITS_ONE_PLUS_ULP:
        raise Error(
            String("embedding_fixture: F-TREE4 SEPARATES BUT NOT AS")
            + " CONTRACT 5.4 PREDICTS. chain "
            + bits32_hex(chain)
            + " (wants 0x3f800000), tree "
            + bits32_hex(tree)
            + " (wants 0x3f800001). A fixture that separates for a reason"
            + " nobody predicted separates something nobody named."
        )
    return (
        String("F-TREE4 R=4: chain ")
        + bits32_hex(chain)
        + " vs balanced tree "
        + bits32_hex(tree)
        + " (SEPARATES, exactly one ULP, contract 5.4's smallest R)"
    )


def guard_negzero1_separates() raises -> String:
    """F-NEGZERO1 separates the SEEDED chain from both wrong spellings, and
    the guard shows that no other fixture value would.

    The second half is spelled: it runs the same two spellings at `+0.0`,
    at `+1.0` and at the hashed generator's first value, and requires them
    to AGREE at all three. That is what makes "inert at every input except a
    sole `-0.0` contributor" a measurement rather than an assertion."""
    var neg = f32_from_bits(BITS_NEG_ZERO)
    var zero = f32_from_bits(BITS_POS_ZERO)
    var one: List[Float32] = [neg]
    var pinned = emb_chain(one, zero)
    var bypass = neg  # the "one contributor needs no adding" spelling
    if bits_of(pinned) == bits_of(bypass):
        raise Error(
            String("embedding_fixture: F-NEGZERO1 IS BLIND. ftz((+0.0) +")
            + " (-0.0)) came out "
            + bits32_hex(pinned)
            + ", the same bits as the bypassed contributor. IEEE-754 says"
            + " (+0) + (-0) is +0 in round-to-nearest on every backend, so"
            + " this is a toolchain finding and not a fixture one."
        )
    var others: List[Float32] = [
        zero,
        f32_from_bits(BITS_ONE),
        emb_hashed_f32(emb_case_seed(0), TID_DY, 0),
    ]
    var agreed = 0
    for i in range(len(others)):
        var single: List[Float32] = [others[i]]
        if bits_of(emb_chain(single, zero)) == bits_of(others[i]):
            agreed += 1
    if agreed != len(others):
        raise Error(
            String("embedding_fixture: F-NEGZERO1's INERT half failed on ")
            + String(len(others) - agreed)
            + " of "
            + String(len(others))
            + " ordinary single contributors. Contract 5.5 says the seeded"
            + " chain and the bypass agree at EVERY input except a sole"
            + " -0.0, and if they do not then SINGLE_RUN_BYPASS moves"
            + " everywhere and is a smoke test rather than a reach proof."
        )
    return (
        String("F-NEGZERO1 R=1: pinned ")
        + bits32_hex(pinned)
        + " vs bypass/seedless "
        + bits32_hex(bypass)
        + " (SEPARATES), and the two agree on "
        + String(agreed)
        + "/"
        + String(len(others))
        + " ordinary single contributors (the arm is a REACH PROOF)"
    )


def guard_subacc_reaches() raises -> String:
    """Contract 7.1's hole, MEASURED on the fixture's own bits.

    Three things are shown and the middle one is the finding:

      1. `a0 + a1` is EXACTLY `0x80400000`, a negative SUBNORMAL. Printed,
         because "near-perfect cancellation at the bottom of the normal
         range" is a sentence and `0x80400000` is a fact.
      2. `ftz` of it is `-0.0`. **The `+0.0` seed forbids reaching `-0.0` by
         ADDITION and does not forbid reaching it through `ftz`.**
      3. A THIRD contributor of exactly `+0.0` then moves the answer from
         `0x80000000` to `0x00000000`, which is the counterexample to
         "an exactly-`+0.0` contributor is inert".

    **UNDER FAST THIS GUARD REPORTS AND DOES NOT RAISE.** `ftz` compiles
    away there, the partial sum stays subnormal and step 2 is simply false;
    that is FAST making no identity claim (the oracle's own header) and not
    a defect. A guard that raised under FAST would be gating the mode rather
    than the fixture."""
    var a0 = f32_from_bits(BITS_NEG_1P5_MIN_NORMAL)
    var a1 = f32_from_bits(BITS_MIN_NORMAL)
    var zero = f32_from_bits(BITS_POS_ZERO)
    var raw_sum = a0 + a1
    var two: List[Float32] = [a0, a1]
    var three: List[Float32] = [a0, a1, zero]
    var r2 = emb_chain(two, zero)
    var r3 = emb_chain(three, zero)
    var line = (
        String("F-SUBACC: a0 ")
        + bits32_hex(a0)
        + " + a1 "
        + bits32_hex(a1)
        + " = "
        + bits32_hex(raw_sum)
        + " (wants 0x80400000, a negative SUBNORMAL); run {a0,a1} -> "
        + bits32_hex(r2)
        + "; run {a0,a1,+0.0} -> "
        + bits32_hex(r3)
    )
    if not mode_is_identical():
        return line + "   [FAST: RECORDED, NOT ASSERTED -- ftz is the identity here]"
    if bits_of(raw_sum) != BITS_NEG_2_POW_M127:
        raise Error(
            String("embedding_fixture: F-SUBACC's PREMISE IS FALSE. a0 + a1")
            + " came out "
            + bits32_hex(raw_sum)
            + " and contract 7.1 computes 0x80400000. The two planted"
            + " constants are not the contract's, or this toolchain's"
            + " addition of two minimum-normal operands is not IEEE."
        )
    if bits_of(r2) != BITS_NEG_ZERO:
        raise Error(
            String("embedding_fixture: F-SUBACC IS BLIND. The two-element")
            + " run gave "
            + bits32_hex(r2)
            + " and contract 7.1 needs 0x80000000. Seam E3's ftz did not"
            + " flush the negative subnormal partial sum, so the ONE"
            + " reachable route to a -0.0 accumulator is not reachable here"
            + " and the KNOWN EXCEPTION is untested"
            + " ([[reached-but-inert]])."
        )
    if bits_of(r3) != BITS_POS_ZERO:
        raise Error(
            String("embedding_fixture: F-SUBACC's THIRD contributor did not")
            + " move the answer. {a0,a1,+0.0} gave "
            + bits32_hex(r3)
            + " and wants 0x00000000. Without that move the case does not"
            + " demonstrate the hole; it only demonstrates a -0.0."
        )
    return line + "   (contract 7.1's hole, MEASURED)"


def guard_nodup_is_blind() raises -> String:
    """**F-NODUP SEES NOTHING AND THAT IS ASSERTED.**

    Contract 11.2 requires this fixture with a five-arm inert mask. The
    guard proves the mask's MECHANISM rather than the mask itself -- every
    run has `R <= 1`, so ascending, descending, one-level pairing and any
    tie order are the same single addition. The five-arm mask is then a
    consequence and the check asserts it against the device."""
    var c = emb_case(emb_case_by_name(String("f_nodup")))
    var rmax = emb_case_max_run(c)
    if rmax > 1:
        raise Error(
            String("embedding_fixture: F-NODUP HAS A RUN OF ")
            + String(rmax)
            + " AND IS NOT A NEGATIVE CONTROL. It is contract 11.2's"
            + " all-ids-distinct fixture and the five-arm inert mask the"
            + " gate asserts on it is false if any id repeats."
        )
    var ids = emb_case_ids(c)
    var cfg = emb_case_config(c)
    var perm = emb_perm_by_scan(ids, cfg)
    var begin = emb_run_begin(emb_counts(ids, cfg))
    var dy = emb_case_dy(c)
    var zero = f32_from_bits(BITS_POS_ZERO)
    var moved = 0
    for v in range(cfg.vocab):
        var lo = Int(begin[v])
        var hi = Int(begin[v + 1])
        for j in range(cfg.width):
            var run = List[Float32]()
            for r in range(lo, hi):
                run.append(dy[Int(perm[r]) * cfg.width + j])
            var asc = emb_backward_cell(
                dy, perm, lo, hi, j, cfg.width, zero
            )
            if bits_of(asc) != bits_of(emb_chain_descending(run, zero)):
                moved += 1
    if moved != 0:
        raise Error(
            String("embedding_fixture: F-NODUP IS NOT BLIND. The ascending")
            + " and descending folds disagreed on "
            + String(moved)
            + " cells, which cannot happen when every run has length <= 1."
            + " The permutation or the run boundaries are wrong."
        )
    return (
        String("F-NODUP: R_max = ")
        + String(rmax)
        + ", ascending == descending on every one of the "
        + String(cfg.vocab * cfg.width)
        + " cells. BLIND BY CONSTRUCTION, and that is what it is for"
    )


def guard_dupsame_is_blind() raises -> String:
    """**F-DUPSAME SEES NOTHING EITHER, AND FOR A DIFFERENT REASON.**

    It HAS duplicates -- the guard asserts `R_max >= 2`, because a
    "duplicates" control whose duplicates did not survive is F-NODUP under
    another name -- and the order clauses are still inert, because a
    permutation of a constant sequence is the same sequence. This is the
    fixture a lane writes when it remembers to add duplicates and forgets
    that duplicates alone are not enough."""
    var c = emb_case(emb_case_by_name(String("f_dupsame")))
    var rmax = emb_case_max_run(c)
    if rmax < 2:
        raise Error(
            String("embedding_fixture: F-DUPSAME HAS R_max = ")
            + String(rmax)
            + " AND IS THEREFORE F-NODUP UNDER ANOTHER NAME. Contract 11.2"
            + " requires two DIFFERENT negative controls and this one is"
            + " supposed to have duplicates."
        )
    var ids = emb_case_ids(c)
    var cfg = emb_case_config(c)
    var perm = emb_perm_by_scan(ids, cfg)
    var begin = emb_run_begin(emb_counts(ids, cfg))
    var dy = emb_case_dy(c)
    var zero = f32_from_bits(BITS_POS_ZERO)
    var moved = 0
    var equal_runs = 0
    for v in range(cfg.vocab):
        var lo = Int(begin[v])
        var hi = Int(begin[v + 1])
        if hi - lo < 2:
            continue
        var run = List[Float32]()
        for r in range(lo, hi):
            run.append(dy[Int(perm[r]) * cfg.width + 0])
        var all_equal = True
        for i in range(1, len(run)):
            if bits_of(run[i]) != bits_of(run[0]):
                all_equal = False
        if all_equal:
            equal_runs += 1
        var asc = emb_backward_cell(dy, perm, lo, hi, 0, cfg.width, zero)
        if bits_of(asc) != bits_of(emb_chain_descending(run, zero)):
            moved += 1
    if equal_runs == 0:
        raise Error(
            "embedding_fixture: F-DUPSAME's DUPLICATES DO NOT CARRY EQUAL"
            " BITS. `emb_case_dy` is supposed to give position i and"
            " position i + V the same hash index, and if it does not then"
            " this fixture is a TEST rather than a CONTROL and contract"
            " 11.2's second negative control has been silently deleted."
        )
    if moved != 0:
        raise Error(
            String("embedding_fixture: F-DUPSAME IS NOT BLIND. Ascending")
            + " and descending disagreed on "
            + String(moved)
            + " runs of bitwise-equal contributors, which is arithmetically"
            + " impossible. The dY generator is not producing equal rows."
        )
    return (
        String("F-DUPSAME: R_max = ")
        + String(rmax)
        + ", "
        + String(equal_runs)
        + " runs of bitwise-EQUAL contributors, ascending == descending on"
        " every one. BLIND, and it is the second control contract 11.2"
        " requires"
    )


def guard_hashed_dy_separates() raises -> String:
    """**THE GENERATOR ITSELF IS SHOWN TO BE ORDER SENSITIVE.**

    This is the guard against the first three of the five ways a fixture
    went blind on 2026-08-25 -- exactly-representable values, absorption,
    and one binade. It takes the hashed `dY` of the multi-block case, folds
    one real run ascending and descending, and RAISES if they agree.

    It also reports the exponent SPREAD inside that run, because "the values
    are full mantissa" is a claim about the generator and "these four
    partials span 2^23" is a claim about the run, and it is the second one
    that decides whether a fold-order arm can be seen."""
    var c = emb_case(emb_case_by_name(String("f_multiblock")))
    var ids = emb_case_ids(c)
    var cfg = emb_case_config(c)
    var perm = emb_perm_by_scan(ids, cfg)
    var begin = emb_run_begin(emb_counts(ids, cfg))
    var dy = emb_case_dy(c)
    var zero = f32_from_bits(BITS_POS_ZERO)
    var separated = 0
    var runs = 0
    var lo_exp = 255
    var hi_exp = 0
    for v in range(cfg.vocab):
        var lo = Int(begin[v])
        var hi = Int(begin[v + 1])
        if hi - lo < 2:
            continue
        runs += 1
        var run = List[Float32]()
        for r in range(lo, hi):
            var val = dy[Int(perm[r]) * cfg.width + 0]
            run.append(val)
            var e = Int((bits_of(val) >> UInt32(23)) & UInt32(0xFF))
            if e < lo_exp:
                lo_exp = e
            if e > hi_exp:
                hi_exp = e
        if bits_of(emb_chain(run, zero)) != bits_of(
            emb_chain_descending(run, zero)
        ):
            separated += 1
    if runs == 0:
        raise Error(
            "embedding_fixture: f_multiblock produced no run of length >= 2,"
            " so the generator guard has nothing to measure"
        )
    if separated == 0:
        raise Error(
            String("embedding_fixture: THE HASHED dY GENERATOR IS BLIND TO")
            + " FOLD ORDER. Ascending and descending agreed on ALL "
            + String(runs)
            + " runs of length >= 2. That is the 2026-08-25 failure exactly"
            + " -- exactly-representable values, or one binade -- and every"
            + " order arm evaluated on a hashed fixture would be reported"
            + " inert ([[reached-but-inert]])."
        )
    return (
        String("hashed dY: ")
        + String(separated)
        + " of "
        + String(runs)
        + " runs separate ascending from descending; exponent field spans "
        + String(lo_exp)
        + ".."
        + String(hi_exp)
        + " (a span of "
        + String(hi_exp - lo_exp)
        + " binades, so the partials are NOT all in one)"
    )


def guard_fold_reads_launch_separates(
    block: Int, must_separate: Bool
) raises -> String:
    """**DEVIATION 1502. `EMB_FOLD_READS_LAUNCH` IS NOT "NEVER INERT ON A
    RUN OF LENGTH >= 2".**

    The arm rotates the walk by `block_dim.x % span`. `block_dim.x` is
    `EMB_TPB`, a POWER OF TWO on every column this repository has, so the
    rotation is ZERO -- and the arm bitwise inert -- at every `span` that
    divides it. That is `R = 2, 4, 8, 16, ...`, which includes the very
    smallest run the device file's docstring promises it fires on.

    This guard takes the ACTUAL block size the binary compiled with and

      * asserts the arm is INERT at F-DUPSAME's `R = 2` and F-TREE4's
        `R = 4`, which is the corrected claim;
      * asserts the arm FIRES at F-ORDER3's `R = 3`, which is this file's
        only witness for it;
      * REPORTS "no witness at this block size" when the rotation on the
        R=3 run happens to be bit neutral -- which is MEASURED at
        `block = 128`, the portable-baseline column's cap, where the
        rotation is 2 and `{2^-24, 1.0, 2^-24}` rounds back to `1.0`. It
        raises there only when `must_separate` says the arm is actually
        compiled in, because killing a whole run over ONE unfalsifiable arm
        would stop the other fifteen from being measured, and the honest
        verdict is "UNGATED here", not "the gate failed".

    `[[verify-reach-not-output]]`: the alternative was to fire the arm,
    watch nothing move on a two-element run, and record "the arm did not
    bite" -- which is the exact inverse of the truth."""
    var zero = f32_from_bits(BITS_POS_ZERO)
    var c3 = emb_case(emb_case_by_name(String("f_order3")))
    var run3 = emb_case_dy(c3)
    var pinned3 = emb_chain(run3, zero)
    var rotated3 = emb_chain_rotated(run3, zero, block)
    if bits_of(pinned3) == bits_of(rotated3):
        # **NO WITNESS AT THIS BLOCK SIZE**, and there is a MEASURED case of
        # it: at `block = 128` the rotation on a three-element run is 2, and
        # `[a1, a0, a2]` over `{1.0, 2^-24, 2^-24}` rounds back to `1.0`
        # exactly as the pinned order does. The portable-baseline column
        # caps a block at 128 (`column_max_block_size`), so THAT COLUMN HAS
        # NO WITNESS FOR THIS ARM IN THIS FIXTURE SET.
        #
        # Reported, not raised, unless the arm is actually compiled in --
        # because a preflight that killed the whole run on a column where
        # ONE arm is unfalsifiable would stop the other fifteen from ever
        # being measured, and the honest verdict is "this arm is UNGATED
        # here", not "the gate failed".
        if not must_separate:
            return (
                String("EMB_FOLD_READS_LAUNCH at block ")
                + String(block)
                + ": **NO WITNESS**. The rotation on the R=3 run is "
                + String(block % 3)
                + " and it moves no bit ("
                + bits32_hex(pinned3)
                + " both ways), and every other run length in this fixture"
                + " set divides the block size. **THE ARM IS UNGATED ON"
                " THIS COLUMN** and must not be reported green. The fix is"
                " a run whose length is coprime to the block size, which is"
                " a FIXTURE edit."
            )
        raise Error(
            String("embedding_fixture: EMB_FOLD_READS_LAUNCH IS ARMED AND")
            + " HAS NO SEPARATING FIXTURE at block size "
            + String(block)
            + ". pinned "
            + bits32_hex(pinned3)
            + ", rotated "
            + bits32_hex(rotated3)
            + ", rotation "
            + String(block % 3)
            + ". Firing this arm here would move no bit and the verdict"
            + " would read as 'the arm did not bite', which is the exact"
            + " inverse of the truth ([[reached-but-inert]])."
        )
    # The CORRECTED inert half, at the two run lengths the device file's own
    # prose says the arm is never inert on.
    var c4 = emb_case(emb_case_by_name(String("f_tree4")))
    var run4 = emb_case_dy(c4)
    var inert4 = bits_of(emb_chain(run4, zero)) == bits_of(
        emb_chain_rotated(run4, zero, block)
    )
    var two: List[Float32] = [
        f32_from_bits(BITS_ONE),
        f32_from_bits(BITS_TWO_POW_M24),
    ]
    var inert2 = bits_of(emb_chain(two, zero)) == bits_of(
        emb_chain_rotated(two, zero, block)
    )
    if block % 2 == 0 and not inert2:
        raise Error(
            String("embedding_fixture: DEVIATION 1502 IS ITSELF WRONG. At")
            + " block size "
            + String(block)
            + ", `block % 2 == 0` so the rotation should be the identity on"
            + " a two-element run, and it moved a bit. Re-read"
            + " emb_backward_kernel's SAB_FOLD_READS_LAUNCH arm before"
            + " trusting either claim."
        )
    return (
        String("EMB_FOLD_READS_LAUNCH at block ")
        + String(block)
        + ": rotation on R=3 is "
        + String(block % 3)
        + " and it MOVES ("
        + bits32_hex(pinned3)
        + " -> "
        + bits32_hex(rotated3)
        + "); on R=2 inert="
        + String(inert2)
        + ", on R=4 inert="
        + String(inert4)
        + ". **The device file's 'never inert on a run of length >= 2' is"
        " FALSE and DEVIATION 1502 records it.**"
    )


def guard_rank_by_arrival_separates(
    block: Int, must_separate: Bool
) raises -> String:
    """`EMB_RANK_BY_ARRIVAL`'s witness needs `T > 2 * block`, and the guard
    computes the arm's own permutation to prove it.

    DEVIATION 1504. The arm emits phase 0 (`(t // block) % 2 == 0`) then
    phase 1. With `T <= 2 * block` phase 0 IS `[0, block)` and phase 1 IS
    `[block, T)`, so concatenating them reproduces ascending order and the
    arm is INERT -- on a fixture that looks, from its `T`, as though it
    should have spanned several blocks. The reordering starts at the THIRD
    band.

    **This is the most dangerous arm in the lane** (contract 6.3 case 3:
    the count an integer atomic hands out is order free and the SLOT is
    not), so a witness that silently goes inert is the worst one to lose."""
    var c = emb_case(emb_case_by_name(String("f_multiblock")))
    if c.n_positions <= 2 * block:
        var msg = (
            String("embedding_fixture: f_multiblock has T = ")
            + String(c.n_positions)
            + " and the block size is "
            + String(block)
            + ", so T <= 2 * block and EMB_RANK_BY_ARRIVAL's phase split"
            + " reproduces ASCENDING ORDER. The arm is INERT on this"
            + " fixture ([[reached-but-inert]])."
        )
        if must_separate:
            raise Error(msg)
        return msg + " **UNGATED on this column**, not green."
    var ids = emb_case_ids(c)
    var cfg = emb_case_config(c)
    var begin = emb_run_begin(emb_counts(ids, cfg))
    var moved = 0
    var runs = 0
    for v in range(cfg.vocab):
        var lo = Int(begin[v])
        var hi = Int(begin[v + 1])
        if hi - lo < 2:
            continue
        runs += 1
        # The arm's permutation, spelled as `emb_perm_kernel` spells it.
        var arm = List[Int32]()
        for phase in range(2):
            for t in range(c.n_positions):
                if Int(ids[t]) != v:
                    continue
                var side = 1 if (t // block) % 2 == 1 else 0
                if side == phase:
                    arm.append(Int32(t))
        var differs = False
        var r = 0
        for t in range(c.n_positions):
            if Int(ids[t]) != v:
                continue
            if Int(arm[r]) != t:
                differs = True
            r += 1
        if differs:
            moved += 1
    if moved == 0:
        var flat = (
            String("embedding_fixture: EMB_RANK_BY_ARRIVAL IS INERT ON")
            + " f_multiblock at block size "
            + String(block)
            + ". None of the "
            + String(runs)
            + " runs came out in a different order from ascending, so the"
            + " arm cannot be falsified here ([[reached-but-inert]])."
        )
        if must_separate:
            raise Error(flat)
        return flat + " **UNGATED on this column**, not green."
    return (
        String("EMB_RANK_BY_ARRIVAL at block ")
        + String(block)
        + ": "
        + String(moved)
        + " of "
        + String(runs)
        + " runs of f_multiblock (T="
        + String(c.n_positions)
        + ") come out in a NON-ascending order under the arm's own phase"
        " split, so the witness separates"
    )


def guard_case_shapes() raises -> String:
    """Every case's measured properties, in one pass, so that a reader can
    see the whole table's separating power at once and a build with a case
    that has drifted fails before any device call.

    It asserts the four properties contract 11.2 makes MANDATORY and that a
    fixture can lose silently:

      * at least one case with `R_max >= 4` (`EMB_FOLD_BALANCED_TREE`);
      * at least one case with `T >= 129` (contract 11.2's sweep);
      * at least one case with a `padding_idx` POSITION (the two pad arms);
      * at least one case with a subnormal WEIGHT (`EMB_GATHER_NO_FLUSH`,
        the only Apple-visible flush arm);
      * at least one split case with a STRADDLING row (`EMB_ACCUM_BY_ADD`).
    """
    var best_run = 0
    var longest_t = 0
    var pad_positions = 0
    var subw = 0
    var straddle = 0
    for k in range(EMB_CASE_COUNT):
        var c = emb_case(k)
        if c.refused:
            continue
        var r = emb_case_max_run(c)
        if r > best_run:
            best_run = r
        if c.n_positions > longest_t:
            longest_t = c.n_positions
        pad_positions += emb_case_pad_positions(c)
        subw += emb_case_subnormal_weights(c)
        straddle += emb_case_straddling_rows(c)
    if best_run < 4:
        raise Error(
            String("embedding_fixture: NO CASE HAS R_max >= 4 (the longest")
            + " is "
            + String(best_run)
            + "). Contract 5.4 proves EMB_FOLD_BALANCED_TREE is inert at"
            + " every R <= 3, so the arm would be fired, seen to move"
            + " nothing, and deleted as broken."
        )
    if longest_t < 129:
        raise Error(
            String("embedding_fixture: NO CASE HAS T >= 129 (the longest is")
            + " "
            + String(longest_t)
            + "). Contract 11.2 calls that shape mandatory."
        )
    if pad_positions == 0:
        raise Error(
            "embedding_fixture: NO CASE CARRIES A padding_idx POSITION, so"
            " EMB_PAD_ROW_CONTRIBUTES and EMB_PAD_ROW_NEG_ZERO are INERT"
            " ENTIRELY and neither can be falsified"
        )
    if subw == 0:
        raise Error(
            "embedding_fixture: NO CASE HAS A SUBNORMAL WEIGHT, so"
            " EMB_GATHER_NO_FLUSH -- the ONLY flush arm in this lane a"
            " single-column run can see move (contract 9.3) -- is"
            " unfalsifiable"
        )
    if straddle == 0:
        raise Error(
            "embedding_fixture: NO SPLIT CASE HAS A ROW WITH CONTRIBUTORS"
            " ON BOTH SIDES OF ITS t0, so clause (e) would pass on the"
            " `dW += dW_micro` spelling as well as on the CARRY and"
            " EMB_ACCUM_BY_ADD is inert"
        )
    return (
        String("case table: ")
        + String(EMB_CASE_COUNT)
        + " cases, R_max up to "
        + String(best_run)
        + ", T up to "
        + String(longest_t)
        + ", "
        + String(pad_positions)
        + " padding_idx positions, "
        + String(subw)
        + " subnormal weight cells, "
        + String(straddle)
        + " straddling rows at the split boundaries"
    )
