# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BLOCK-LEVEL BATCH INVARIANCE: a token's output bits do not depend on how
many other sequences were in the launch.

**THE TIER. This claim is in the IDENTICAL tier and in no other, and the
reason is mechanical rather than a preference.** Every projection in both
blocks goes through `gemm/checks/gemm_identical.mojo::identical_gemm`, and
that function opens with

    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        if _fast_vendor_gemm(ctx, c, a, b, m, n, k, op):
            return

(DEVIATION 1876). So under FAST **and under DETERMINISTIC** the shape is
handed to the vendor's own kernel, which is free to tile differently at
different `m`, and the pinned partition never runs. `PIN_DETERMINISM` is
true under DETERMINISTIC, but the pin this claim rests on is guarded by
`PIN_CROSS_VENDOR`, so:

    IDENTICAL      ASSERTED, bitwise, and a difference is a FAILURE
    DETERMINISTIC  REPORTED  (the vendor GEMM is live; nothing promises it)
    FAST           REPORTED  ([[fast-is-not-identical]]: FAST promises speed
                             only and is never asked a bitwise question)

WHAT MECHANISM THIS LIFTS, AND FROM WHERE
------------------------------------------
`gemm/IDENTICAL_FP32_CONTRACT.md` section 6.1 is titled *"Why `L` may not
depend on `m` or `n` -- this is the batch-invariance clause"*, and its
sentence is the whole of the mechanism:

> because `m` is the batch dimension of a token GEMM. If `L = f(k, m)`, then
> the same row of `A` against the same column of `B` returns different bits
> depending on how many other rows were in the launch.

The code form is `gemm/checks/gemm_oracle.mojo:117`,
`def contract_leaf_size(k: Int) -> Int`, which takes `k` and nothing else,
so the batch dimension is not in scope where the numerical plan is decided.
`core/gemm_identity_check.mojo:460`'s
`check_pinned_gemm_is_batch_invariant` asserts the property ONE LAYER DOWN,
on a bare GEMM.

**THIS FILE IS THE LIFT TO THE BLOCK**, for both blocks the tree has:
`mamba_block_forward` and `llama_decoder_layer_forward`. The other
reductions in either block -- RMSNorm's sum of squares, softmax's max and
denominator, the SSM scan -- reduce over FEATURES PER ROW or over the
sequence, never over the batch, so they *should* be invariant by
construction. "Should be" is not a card, and this file is the card.

WHAT IS ALREADY GATED, SO THAT THIS FILE IS NOT SOLD AS MORE THAN IT IS
-------------------------------------------------------------------------
Both lanes already carry a batch-composition clause and both already carry
the row-slicer negative control:

    mamba/checks/mamba_check.mojo::clause_c          B in {1, 2, 3}
    transformer/checks/transformer_check.mojo::clause_c_batch   B in {1, 2, 3}

This file adds exactly three things to those, and nothing else:

1. **SERVING-SCALE B.** B in {1, 17, 64} rather than {1, 2, 3}, with 17
   there because a non-power-of-two batch is the one a tiled kernel has to
   pad, and 64 because that is a batch a server actually forms.
2. **THE EXECUTION PLAN ACTUALLY MOVES.** `choose_gemm_plan(m, n, k)` reads
   `m`, and is allowed to (contract 6.1's second paragraph). At B = 1 with
   the Mamba fixture the in_proj GEMM is `m = 8`, which takes
   `PLAN_TILE_8_32_32`; at B = 17 it is `m = 136`, which takes
   `PLAN_TILE_16_16_32`. **Different kernels, asserted equal bits.** At
   B in {1, 2, 3} several of these shapes take the SAME plan, so the
   existing clauses are in part asserting that one kernel run twice agrees
   with itself. `check_batch_axis_reaches_the_execution_plan` refuses a run
   in which the plan never moves.
3. **ONE SABOTAGE THE B <= 3 CLAUSES ARE STRUCTURALLY BLIND TO**, below.

THE MAMBA CROSS-TOKEN QUESTION, WHICH IS THE REASON THIS DOCSTRING IS LONG
----------------------------------------------------------------------------
Mamba is a recurrence. `h_t` depends on tokens `0..t`, so "the same token"
is not a well-defined object until the comparison says what a token belongs
to. There are two readings of *"the same tokens at a different batch size"*
and **only one of them is a claim about batch invariance**:

* **RE-BATCHING ONE TOKEN STREAM (ill-posed, and this file refuses to spell
  it).** Take `M` tokens; run them as `(B = 1, L = M)`; run them again as
  `(B = 64, L = M/64)` and compare flat token `i` to flat token `i`. These
  MUST differ and a gate that asserted they agreed would be asserting a
  falsehood. In the first run token 100 attends to (or recurses over) tokens
  `0..100`; in the second, flat token 100 is `(row 12, position 4)` and sees
  only `96..100`. Reshaping across the batch axis CUTS the recurrence at
  different places, so the two runs compute two different functions. Nothing
  about a GEMM's summation order is involved. The same is true of attention,
  for the same reason with the KV cache in place of `h`.

* **COMPOSING A SEQUENCE INTO A LARGER LAUNCH (well-posed, and this is the
  claim).** Fix a sequence `S` of length `L`. Run it alone at `B = 1`. Run
  it again as ROW 0 of a batch of `B` sequences whose other `B - 1` rows are
  arbitrary. Every one of `S`'s `L` tokens must come out bit-identical, and
  so must row 0's end state. `L` is HELD FIXED and only `B` varies.

**So the unit of comparison is the SEQUENCE, and the per-token comparison
happens INSIDE it:** token `t` of row `r` at `B = 64` against token `t` of
row `r` at `B = 1`. That is what "token `i` of the batch-1 run against token
`i` of the batch-64 run" has to mean for a recurrent block, and it is what
this file compares. Written this way the recurrence is not an obstacle at
all -- rows are independent in both blocks (the MAX scan shape is one thread
per `(batch, dim)`, serial over `L`; the KV cache is `[B, n_kv, S, hd]`) --
and the residual question is exactly the batch question: does anything about
the LAUNCH leak into row `r`'s arithmetic.

It is also the reading the serving world means. A server does not re-cut a
prompt across the batch axis; it puts your prompt in a batch beside other
people's and you expect your logits back unchanged.

**Row 0's tokens are bit-identical across the three runs BY CONSTRUCTION,
not by hope**, and the construction is checked rather than trusted: the
fixture generators are element-indexed hashes, so a `B = 64` input's first
`L * d_model` elements ARE the `B = 1` input, and `_refuse_unless_prefix`
compares them bit for bit before any block runs. If that ever stops holding,
this file refuses instead of comparing two different sequences and reporting
the difference as a batch-invariance failure.

WHAT IS COMPARED
-----------------
The BLOCK OUTPUT per token -- `residual.out` for Mamba (contract section 7's
S16) and `residual2` for Llama (section 9's last stage) -- plus, for Mamba,
the recurrent state `scan.h` after the call, which is `[B, d_inner, 16]` and
is the one buffer indexed by the batch axis where a batch-dependent bug
would most naturally live. Stage-by-stage comparison is NOT repeated here:
both lane gates already do that at B <= 3, and the output is the thing the
claim is about.

THE SABOTAGE ARM, AND WHY IT IS THE ONE THIS FILE NEEDED
----------------------------------------------------------
`-D MOJOLEARN_BATCHINV_SABOTAGE_NORM_CHUNK_FROM_M=1`. One `-D` arms BOTH
blocks (the two impl files declare the same string literal; the two
`*_batchinv_sabotage_armed()` accessors are compared here and a
disagreement REFUSES the run, because a misspelled define that silently arms
half a run is this tree's oldest scar).

It replaces RMSNorm's per-row serial ascending sum of squares with a chunked
fold whose CHUNK WIDTH IS DERIVED FROM `m = B * L`:

    chunk = batchinv_norm_chunk(dm, m) = max(1, dm // (1 + m // 64))

That is contract 6.1's forbidden `L = f(k, m)` moved one layer up: a
reduction whose SPLIT is a function of the batch dimension. It is not a
sabotage of the profile's arithmetic in general -- at `m < 64` the divisor
is 1, `chunk == dm`, there is one chunk and the fold IS the clean ascending
chain, bit for bit.

**Which is exactly why it matters: the B <= 3 clauses cannot see it.**
`mamba_check`'s clause (c) tops out at `B = 3, L = 16`, i.e. `m = 48`;
`transformer_check`'s at `B = 3, L = 16`, `m = 48`. Both are below 64. An
arm that is inert for every gate the tree has today, and lethal at the batch
sizes a server uses, is the sharpest available argument that this file is
not redundant.

PREDICTION, so that the measurement can contradict it
-------------------------------------------------------
Stated as falsifiable numbers, none of them measured (this file has not been
run at the time of writing):

* **B = 1 is unmoved.** The sabotage is bitwise inert at `m < 64`, so the
  `B = 1` reference arms (Mamba `m = 8`, Llama `m = 4` and `m = 16`) are
  byte-for-byte what a clean build produces. The arm is BATCH-DERIVED, not
  a fold change, and that is how the file proves it.
* **Mamba, `d_model = 8`, `L = 8`.** At `B = 17` (`m = 136`) chunk is 2, so
  eight squares fold as `((a+b)+(c+d))+((e+f)+(g+h))`-shaped partials rather
  than one chain; at `B = 64` (`m = 512`) chunk is 1, so it is eight
  singletons summed left to right. An 8-term FP32 sum re-associated moves
  its low bits often but not always: expect `norm.sumsq` to move on
  **roughly half to all of the 8 token rows of row 0 at B = 17, and on
  essentially all 8 at B = 64** -- and a floor of at least 1, since 0 would
  make the arm inert and the check says so. `rstd` multiplies the whole row,
  so any row whose `sumsq` moved propagates: expect the block OUTPUT
  `residual.out` to differ on **the great majority of row 0's 8 x 8 = 64
  cells**, and `scan.h` on the great majority of row 0's 16 x 16 = 256.
* **Llama, `d_model = 32`.** At `B = 64` the chunk is 1 with `L = 16`
  (`m = 1024`, divisor 17) and 6 with `L = 4` (`m = 256`, divisor 5). A
  32-term sum re-associated into 32 singletons is different with near
  certainty: expect **all 16 (resp. all 4) of row 0's `sumsq` rows to move**
  and `residual2` to differ on **essentially all 16 x 32 = 512 (resp.
  4 x 32 = 128) cells**.
* **The clean build moves nothing, on any of the three batch sizes, on
  either block.**

A build with the arm on that moves NOTHING is reported as
`[[reached-but-inert]]` and RAISES, exactly like every other sabotage in
this tree; the arm is not credited for a pass it could not have failed.

THE VACUITY CONTROLS
---------------------
An assertion that `B = 1` and `B = 64` agree proves nothing if they agree
trivially. Three controls, and the first two REFUSE:

* **V1, rows are distinguishable.** Row 0 and row 1 of the `B = 17` run must
  differ in the block output. If they do not, the slicer is returning one
  row twice, or the fixture gives every sequence the same answer, and every
  comparison below is a value against itself. Raises VACUOUS.
* **V2, tokens are distinguishable.** Token 0 and token `L - 1` of row 0 at
  `B = 1` must differ. If every token of a row has the same bits, a
  PER-TOKEN comparison is not per-token at all -- and for Mamba it would
  additionally mean the recurrence is not recursing. Raises VACUOUS.
* **V3, the batch axis reaches the execution plan.** The plan
  `choose_gemm_plan(m, n, k)` picks must differ between `B = 1` and
  `B = 64` on at least ONE of the block's GEMM shapes; otherwise the same
  kernel ran three times and the claim degenerates to clause (b)'s
  repeated-launch claim, which is a different and weaker thing. Raises
  VACUOUS under IDENTICAL; REPORTED under the other two tiers, where
  `choose_gemm_plan` is not the dispatcher in the first place.

SCOPE, AND WHAT IS NOT CLAIMED
--------------------------------
One block of each kind, one device, forward only, three batch sizes, two
sequence lengths for Llama and one for Mamba. Nothing here says anything
about a multi-block model, about decode carrying a cache across calls at
mixed batch sizes (a serving system's real question, and OWED), about
backward, or about any vendor the run did not happen on. Report the column
in the claim -- `[[one-box-verdict-is-not-three]]`.
"""

from std.memory import bitcast

from max.gpu.host import DeviceContext

from checks.numerics import PIN_CROSS_VENDOR, numeric_mode_name
from core.identity_trace import IdentityTrace

from gemm.checks.gemm_identical import choose_gemm_plan, gemm_plan_name
from gemm.checks.gemm_oracle import contract_leaf_size

from mamba.checks.mamba_fixture import (
    D_STATE,
    MambaDims,
    MambaWeights,
    corpus_case_seed,
    corpus_weights,
    corpus_x,
)
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    MambaDeviceStages,
    MambaDeviceState,
    MambaDeviceWeights,
    batchinv_norm_chunk,
    mamba_batchinv_sabotage_armed,
    mamba_block_forward,
    mamba_block_sabotage_name,
    mamba_download,
    mamba_upload,
)

from transformer.checks.transformer_fixture import (
    PLANT_NONE,
    RMS_EPS,
    ROPE_THETA,
    FixtureCase,
    TransformerDims,
    TransformerWeights,
    fixture_case,
    fixture_case_by_name,
    fixture_dims,
    fixture_weights,
    fixture_x,
)
from transformer.impl.transformers.models.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    llama_batchinv_sabotage_armed,
    llama_block_sabotage_name,
    llama_decoder_layer_forward,
)


# ===========================================================================
# THE COMPOSITIONS. `1` is the reference a serving request would get alone;
# `17` is deliberately not a power of two (it is the batch a tiled kernel has
# to pad, and padding is where a `+0.0` gets added to something); `64` is a
# batch a server actually forms.
#
# THE MIDDLE ONE IS NOT DECORATION. `choose_gemm_plan`'s thresholds are 16
# and 32 on `m` and `n`, so 17 and 64 can select the SAME plan while 1
# selects a different one -- and a fixture that only ran 1 and 64 could not
# tell "the plan moved once" from "the plan moves with B".
# ===========================================================================

comptime B_REF = 1
comptime B_ODD = 17
comptime B_BIG = 64

#: Mamba's fixture: `d_model = 8` (so `d_inner = 16`, `dt_rank = 1`) and
#: `L = 8`. The smallest shape at which `m = B * L` crosses 64 at `B = 17`
#: (136) and lands well past it at `B = 64` (512). Keeping it small is
#: deliberate: this runs on an M4 first and a card is worth more than a big
#: fixture ([[no-heavy-local-compute]]).
# 32, NOT 8, AND THE SABOTAGE IS WHY.
#
# This was 8 when the file was written, and at 8 the gate CERTIFIED MAMBA
# VACUOUSLY. Measured 2026-08-31: with the norm-chunk arm ARMED and provably
# reached -- the run prints `mamba d_model=8 L=8: B=1 chunk=8 B=17 chunk=2
# B=64 chunk=1`, so three genuinely different fold shapes ran -- all six mamba
# comparisons still came back IDENTICAL. A sum of squares over 8 values
# reassociates into chunks of 2 and of 1 without moving a single bit on this
# fixture, so the mamba half of this check could not have failed no matter
# what the block did. Six of twelve comparisons were decoration.
#
# At 32 the same arm moves all six (residual.out on 2-4 of 8 tokens, scan.h on
# 805-893 of 1024 cells), and the clean build still passes all twelve. That is
# the pair of runs that makes this a gate rather than a report:
#
#   clean,    IDENTICAL, d_model 32:  12 of 12 bit-identical
#   sabotaged, IDENTICAL, d_model 32: 12 of 12 moved
#
# `[[reached-but-inert]]`: a path that runs is not a path that is gated. The
# arm was reached at 8 and inert at 8, and only the sabotage could tell the
# difference between that and a real invariance.
comptime MAMBA_D_MODEL = 32
comptime MAMBA_L = 8

#: Which corpus seed the Mamba weights and input come from. Case 1's seed,
#: the same one `mamba_check::clause_c` uses, so the two clauses are looking
#: at the same numbers at different batch sizes rather than at two fixtures.
comptime MAMBA_SEED_CASE = 1

#: Llama runs TWO shapes, and each is load-bearing.
#:
#: `base_b2_l4_nrep2` is `L = 4`, `d_model = 32`: at `B = 1` that is `m = 4`,
#: which is BELOW `choose_gemm_plan`'s `m >= 16` threshold, so the q/k/v
#: projections take a different plan at `B = 1` than at `B = 17`. This is the
#: case that makes V3 non-vacuous for the transformer.
#:
#: `base_b3_l16_nrep2` is `L = 16`: the causal mask has a real triangle, the
#: softmax folds over 16 keys rather than 4, and `m = 1024` at `B = 64`
#: drives the sabotage's chunk width down to 1.
comptime LLAMA_CASE_SHORT = "base_b2_l4_nrep2"
comptime LLAMA_CASE_LONG = "base_b3_l16_nrep2"

#: How many bitwise comparisons one clean run makes. Written out rather than
#: counted at run time so that a comparison silently dropped from one of the
#: three composition functions shows up as a wrong denominator in the banner
#: instead of disappearing:
#:
#:   mamba   residual.out  row0 B=1 vs 17, row0 B=1 vs 64, row1 B=17 vs 64
#:   mamba   scan.h        row0 B=1 vs 17, row0 B=1 vs 64, row1 B=17 vs 64
#:   llama   residual2     the same three, on base_b2_l4_nrep2
#:   llama   residual2     the same three, on base_b3_l16_nrep2
comptime N_COMPARISONS = 12


# ===========================================================================
# BITS
# ===========================================================================


def bits_of(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def hex_of_bits(u: UInt32) -> String:
    """A bit pattern as `0x` + eight lowercase nibbles.

    `[[mojo-string-float-roundtrip]]`: `String(Float32)` does not round
    trip, so every value this file prints is printed as its BITS. Spelled
    the way `mamba_fixture.bits32_hex` spells it (`String(DIGITS[byte=n])`),
    because `[[mojo-string-not-indexable]]` -- `s[i]` is a compile error.
    """
    comptime DIGITS = "0123456789abcdef"
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


@fieldwise_init
struct Diff(Copyable, Movable):
    """How two equal-length slices differ, BY BITS. `first` is -1 when they
    do not."""

    var n_cells: Int
    var n_diff: Int
    var first: Int
    var a_bits: UInt32
    var b_bits: UInt32


def diff_bits(a: List[Float32], b: List[Float32]) raises -> Diff:
    """Cell by cell, by BIT PATTERN and never by `==`.

    `[[metal-hardware-gaps]]` and IDENTITY_PATHS row 49: Metal flushes
    compare operands, so `x == y` can be true of two different subnormal bit
    patterns on one vendor and false on another, which would make this file
    report a vendor difference as agreement. A LENGTH mismatch raises rather
    than comparing the shorter prefix, because a slice of the wrong size is
    a different defect from a slice of the wrong value.
    """
    if len(a) != len(b):
        raise Error(
            String("batch_invariance: comparing ")
            + String(len(a))
            + " cells against "
            + String(len(b))
            + "; a length mismatch is an indexing bug, not a bit difference"
        )
    var n_diff = 0
    var first = -1
    var ab = UInt32(0)
    var bb = UInt32(0)
    for i in range(len(a)):
        if bits_of(a[i]) != bits_of(b[i]):
            n_diff += 1
            if first < 0:
                first = i
                ab = bits_of(a[i])
                bb = bits_of(b[i])
    return Diff(len(a), n_diff, first, ab, bb)


def slice_of(values: List[Float32], start: Int, count: Int) raises -> List[Float32]:
    if start < 0 or count < 0 or start + count > len(values):
        raise Error(
            String("batch_invariance: slice [")
            + String(start)
            + ", "
            + String(start + count)
            + ") is outside a buffer of "
            + String(len(values))
        )
    var out = List[Float32]()
    for i in range(count):
        out.append(values[start + i])
    return out^


def token_slice(
    values: List[Float32], row: Int, t: Int, l: Int, width: Int
) raises -> List[Float32]:
    """The `width` cells of token `t` of batch row `row`, out of a
    token-major `[B, L, width]` buffer.

    Token-major `M = B * L` is what both contracts specify (mamba section 7,
    transformer section 9), so row `r`'s tokens are contiguous and token `t`
    of row `r` starts at `(r * L + t) * width`. THIS IS THE ONLY INDEX MAP IN
    THE FILE and V1/V2 exist to prove it distinguishes what it claims to."""
    return slice_of(values, (row * l + t) * width, width)


def row_slice(
    values: List[Float32], row: Int, l: Int, width: Int
) raises -> List[Float32]:
    """All `L` tokens of batch row `row`."""
    return slice_of(values, row * l * width, l * width)


def batch_slice(
    values: List[Float32], row: Int, width: Int
) raises -> List[Float32]:
    """The `width` cells of batch row `row` out of a `[B, width]` buffer --
    the shape of `scan.h`, which is `[B, d_inner, D_STATE]` and carries NO
    token axis. It is the recurrent state AFTER the call, so it is comparable
    only as a whole row, never per token."""
    return slice_of(values, row * width, width)


def refuse_unless_prefix(
    label: String, small: List[Float32], big: List[Float32]
) raises:
    """`small` must be `big`'s leading bits, exactly.

    THE COMPARISON'S PRECONDITION, CHECKED RATHER THAN ASSUMED. "The same
    tokens at a different batch size" is only a claim about batch invariance
    if the tokens really are the same; if the fixture generator ever stops
    being element-indexed, the two runs would be fed different inputs and
    this file would report an ordinary difference as a batch-invariance
    failure -- the worst outcome available to it."""
    if len(small) > len(big):
        raise Error(
            String("batch_invariance: ")
            + label
            + ": the small run's input is LONGER than the large run's ("
            + String(len(small))
            + " > "
            + String(len(big))
            + ")"
        )
    var head = slice_of(big, 0, len(small))
    var d = diff_bits(small, head)
    if d.n_diff != 0:
        raise Error(
            String("batch_invariance: ")
            + label
            + ": the B=1 input is NOT the leading rows of the larger"
            + " batch's input -- "
            + String(d.n_diff)
            + " of "
            + String(d.n_cells)
            + " cells differ, first at "
            + String(d.first)
            + " ("
            + hex_of_bits(bits_of(small[d.first]))
            + " vs "
            + hex_of_bits(bits_of(head[d.first]))
            + "). The runs are being fed DIFFERENT SEQUENCES and any"
            + " difference downstream would be misreported as a batch"
            + " dependence."
        )


# ===========================================================================
# THE COMPARISONS THEMSELVES. Printing is the same shape everywhere so that
# the output can be read as a table; a difference prints the TOKEN, the cell
# and both bit patterns, because "it failed" is not a finding.
# ===========================================================================


def compare_row_per_token(
    label: String,
    a: List[Float32],
    b: List[Float32],
    row_a: Int,
    row_b: Int,
    l: Int,
    width: Int,
) raises -> Int:
    """Token by token, bitwise. Returns 1 if ANY token moved, else 0.

    The comparison is per token and reported per token even though the two
    rows are contiguous, because the claim is about a token and a reader who
    sees "3 of 8 tokens moved" learns something a cell count would hide:
    whether the dependence is positional (early tokens clean, late tokens
    dirty, which is what a broken recurrence looks like) or uniform (which is
    what a changed fold order looks like)."""
    var tokens_moved = 0
    var cells_moved = 0
    var cells = 0
    var first_token = -1
    var first_cell = -1
    var first_a = UInt32(0)
    var first_b = UInt32(0)
    for t in range(l):
        var d = diff_bits(
            token_slice(a, row_a, t, l, width),
            token_slice(b, row_b, t, l, width),
        )
        cells += d.n_cells
        if d.n_diff > 0:
            tokens_moved += 1
            cells_moved += d.n_diff
            if first_token < 0:
                first_token = t
                first_cell = d.first
                first_a = d.a_bits
                first_b = d.b_bits
    if tokens_moved == 0:
        print(
            "  ok  "
            + label
            + ": "
            + String(l)
            + " tokens, "
            + String(cells)
            + " cells, IDENTICAL"
        )
        return 0
    print(
        "  MOVED  "
        + label
        + ": "
        + String(tokens_moved)
        + " of "
        + String(l)
        + " tokens and "
        + String(cells_moved)
        + " of "
        + String(cells)
        + " cells, first at token "
        + String(first_token)
        + " cell "
        + String(first_cell)
        + "  "
        + hex_of_bits(first_a)
        + " vs "
        + hex_of_bits(first_b)
    )
    return 1


def compare_batch_row(
    label: String,
    a: List[Float32],
    b: List[Float32],
    row: Int,
    width: Int,
) raises -> Int:
    """A `[B, width]` buffer's row, bitwise. Returns 1 if it moved."""
    var d = diff_bits(
        batch_slice(a, row, width), batch_slice(b, row, width)
    )
    if d.n_diff == 0:
        print(
            "  ok  " + label + ": " + String(d.n_cells) + " cells, IDENTICAL"
        )
        return 0
    print(
        "  MOVED  "
        + label
        + ": "
        + String(d.n_diff)
        + " of "
        + String(d.n_cells)
        + " cells, first at "
        + String(d.first)
        + "  "
        + hex_of_bits(d.a_bits)
        + " vs "
        + hex_of_bits(d.b_bits)
    )
    return 1


def armed_word(armed: Bool) -> String:
    """The banner word for the arm: `ON` or `off`.

    A `def` rather than an inline conditional expression so the banner line
    is one concatenation of Strings and not a mix of String and
    StringLiteral."""
    if armed:
        return String("ON")
    return String("off")


# ===========================================================================
# THE TWO BLOCKS, ONE CALL EACH
# ===========================================================================


def mamba_outputs(
    ctx: DeviceContext,
    w: MambaWeights,
    x: List[Float32],
    b: Int,
    l: Int,
    dims: MambaDims,
) raises -> List[List[Float32]]:
    """One `mamba_block_forward` call from FRESH state, returning
    `[residual.out, scan.h]` on the host.

    A fresh `MambaDeviceState` is a zero conv window and a zero `h`, which is
    contract section 5's prefill. Every device object is a LOCAL and every
    one is alive at the download, because `mamba_block_forward` synchronizes
    before it returns and the explicit moves at the foot keep the owners past
    the last `.unsafe_ptr()` -- `[[mojo-buffer-freed-at-last-use]]`.

    The trace is `disabled()` rather than a path: a card is not what this
    file produces, and a disabled trace also means the run does not change
    behavior for someone who has `MOJOLEARN_IDENTITY_TRACE` exported.
    """
    var dw = MambaDeviceWeights(ctx, w)
    var dstate = MambaDeviceState(ctx, b, dims)
    var dstages = MambaDeviceStages(ctx, b, l, dims)
    var dx = mamba_upload(ctx, x)
    var off = IdentityTrace.disabled()
    mamba_block_forward(ctx, dstages, dstate, dw, dx, b, l, off, "bi")
    var out = List[List[Float32]]()
    out.append(
        mamba_download(ctx, dstages.residual_out, b * l * dims.d_model)
    )
    out.append(
        mamba_download(ctx, dstages.scan_h, b * dims.d_inner * D_STATE)
    )
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    return out^


def llama_output(
    ctx: DeviceContext,
    c: FixtureCase,
    dims: TransformerDims,
    w: TransformerWeights,
    x: List[Float32],
) raises -> List[Float32]:
    """One `llama_decoder_layer_forward` call from a FRESH KV cache at
    `pos0 = 0`, returning `residual2` on the host.

    `llama_decoder_layer_forward` is the unplanted entry point; the planted
    form is for the bit-planting fixture cases and none of them is used here
    (a score plant is a FLAT index into `[B, n_heads, L, S]`, so the same
    plant lands in a different cell at every B and the comparison would be
    measuring the plant -- `transformer_check::clause_c_batch` refuses
    planted cases for the same reason and so does `llama_composition` below).
    """
    var ldims = LlamaDims(
        dims.d_model,
        dims.n_heads,
        dims.n_kv_heads,
        dims.head_dim,
        dims.intermediate,
    )
    var dw = LlamaDeviceWeights(
        ctx,
        ldims,
        RMS_EPS,
        w.norm1_w,
        w.norm2_w,
        w.w_q,
        w.w_k,
        w.w_v,
        w.w_o,
        w.w_gate,
        w.w_up,
        w.w_down,
    )
    var kv = LlamaKVCache(ctx, c.b, ldims, c.cache_cap)
    var rope = LlamaRopeTable(ctx, ldims, ROPE_THETA, dims.rope_positions)
    var stages = LlamaDeviceStages(ctx, c.b, c.l, c.cache_cap, ldims)
    # `mamba_upload` / `mamba_download` rather than `modeling_llama`'s
    # underscore-prefixed `_upload` / `_download`: they are the same four
    # lines over a `DeviceBuffer[float32]` and carry nothing lane-specific,
    # and this file already imports the mamba pair for the other block. One
    # host-transfer spelling in this file is one fewer thing to be wrong
    # about; neither of them is a seam.
    var dx = mamba_upload(ctx, x)
    var off = IdentityTrace.disabled()
    llama_decoder_layer_forward(
        ctx, stages, kv, rope, dw, dx, c.b, c.l, 0, off, "bi"
    )
    var out = mamba_download(ctx, stages.residual2, c.b * c.l * dims.d_model)
    _ = dw^
    _ = kv^
    _ = rope^
    _ = stages^
    _ = dx^
    return out^


def case_at_batch(c: FixtureCase, b: Int) -> FixtureCase:
    """`c` with its batch size replaced and nothing else.

    `FixtureCase` is `@fieldwise_init`, so this is the shape restated with
    one field changed. **The NAME is carried unchanged on purpose**:
    `fixture_weights` and `fixture_x` recover the seed from the name
    (`k_of`), so keeping it is what makes the three runs share one weight set
    and one element-indexed input stream. `cache_cap` is not a function of B
    and stays as it is."""
    return FixtureCase(
        c.name,
        b,
        c.l,
        c.split,
        c.d_model,
        c.n_heads,
        c.n_kv_heads,
        c.head_dim,
        c.intermediate,
        c.cache_cap,
        c.plant,
    )


# ===========================================================================
# V3: DOES THE BATCH AXIS REACH THE EXECUTION PLAN AT ALL
# ===========================================================================


@fieldwise_init
struct PlanProbe(Copyable, Movable):
    var label: String
    var n: Int
    var k: Int


def plan_probes_mamba(dims: MambaDims) -> List[PlanProbe]:
    """The GEMM shapes `mamba_mixer_forward` issues whose `m` is `B * L`.

    `n` and `k` are fixed by the model shape; only `m` moves with the batch,
    which is the whole point. The per-`(batch, head)` score GEMMs of the
    transformer are deliberately absent from the transformer's list below for
    the mirror-image reason: their `m` is `L`, so the batch axis does not
    enter them at all."""
    var di = dims.d_inner
    var out: List[PlanProbe] = [
        PlanProbe(String("mamba in_proj"), 2 * di, dims.d_model),
        PlanProbe(String("mamba x_proj"), dims.x_proj_rows(), di),
        PlanProbe(String("mamba dt_proj"), di, dims.dt_rank),
        PlanProbe(String("mamba out_proj"), dims.d_model, di),
    ]
    return out^


def plan_probes_llama(dims: TransformerDims) -> List[PlanProbe]:
    var out: List[PlanProbe] = [
        PlanProbe(String("llama q_proj"), dims.q_width(), dims.d_model),
        PlanProbe(String("llama k_proj"), dims.kv_width(), dims.d_model),
        PlanProbe(String("llama o_proj"), dims.d_model, dims.q_width()),
        PlanProbe(String("llama gate_proj"), dims.intermediate, dims.d_model),
        PlanProbe(String("llama down_proj"), dims.d_model, dims.intermediate),
    ]
    return out^


def report_plans(
    probes: List[PlanProbe], l: Int, batches: List[Int]
) raises -> Int:
    """Print the plan each batch size selects for each shape, and return how
    many shapes CHANGE PLAN across the batch sizes.

    This is the whole of V3 and it is host arithmetic: `choose_gemm_plan` is
    a pure function of `(m, n, k)` and no device is touched. It also prints
    `contract_leaf_size(k)` beside each row, because the two numbers side by
    side ARE the mechanism: the leaf is constant down the column while the
    plan is not."""
    var moved = 0
    for i in range(len(probes)):
        var label = probes[i].label
        var pn = probes[i].n
        var pk = probes[i].k
        var line = String("  ") + label + ": k=" + String(pk)
        line += " L_leaf=" + String(contract_leaf_size(pk)) + " (never a"
        line += " function of m -- contract 6.1)   plans:"
        var first_plan = -1
        var differs = False
        for j in range(len(batches)):
            var m = batches[j] * l
            var plan = choose_gemm_plan(m, pn, pk)
            if j == 0:
                first_plan = plan
            elif plan != first_plan:
                differs = True
            line += "  B=" + String(batches[j])
            line += " m=" + String(m)
            line += " " + gemm_plan_name(plan)
        if differs:
            moved += 1
            line += "   <- PLAN MOVES WITH B"
        print(line)
    return moved


def check_batch_axis_reaches_the_execution_plan() raises:
    """V3. **The batch axis must actually reach the dispatcher somewhere in
    this run**, or the assertion below is a claim that one kernel run three
    times agrees with itself.

    `choose_gemm_plan` reads `m`, `n` and `k` and is ALLOWED to -- contract
    6.1's second paragraph is explicit: *"the EXECUTION plan may look at `m`,
    `n`, the device, the occupancy and anything else it likes, because under
    section 7 none of that can reach the arithmetic."* This control is what
    turns that sentence from an assurance into a measurement: it finds the
    shapes where a DIFFERENT KERNEL runs at a different batch size, and the
    clause then asserts that the different kernels produce the same bits.

    Note what is NOT claimed by a zero here. Even at a fixed plan the GRID
    changes with `m` in every plan (more blocks, more tiles), so the launch
    is never literally identical across batch sizes. But a grid change is a
    weaker perturbation than a plan change, and a run in which the plan never
    moves is a run whose headline should be softer. So: REFUSED under
    IDENTICAL, REPORTED under the other tiers -- where `identical_gemm`'s
    DEVIATION 1876 branch has handed the shape to the vendor and
    `choose_gemm_plan` is not the dispatcher at all.
    """
    var batches: List[Int] = [B_REF, B_ODD, B_BIG]
    print("")
    print(
        "V3: does the batch axis reach the EXECUTION plan?  (host"
        " arithmetic, no device)"
    )
    var mdims = MambaDims.of(MAMBA_D_MODEL)
    print("  -- mamba, L=" + String(MAMBA_L))
    var m_moved = report_plans(plan_probes_mamba(mdims), MAMBA_L, batches)

    var short = fixture_case(fixture_case_by_name(String(LLAMA_CASE_SHORT)))
    var sdims = fixture_dims(short)
    print("  -- llama " + String(short.name) + ", L=" + String(short.l))
    var s_moved = report_plans(plan_probes_llama(sdims), short.l, batches)

    var long_c = fixture_case(fixture_case_by_name(String(LLAMA_CASE_LONG)))
    var ldims = fixture_dims(long_c)
    print("  -- llama " + String(long_c.name) + ", L=" + String(long_c.l))
    var l_moved = report_plans(plan_probes_llama(ldims), long_c.l, batches)

    var total = m_moved + s_moved + l_moved
    if total == 0:
        if PIN_CROSS_VENDOR:
            raise Error(
                "batch_invariance: V3 IS VACUOUS. Not one GEMM shape in"
                " either block changes EXECUTION PLAN between B=1 and B=64,"
                " so the batch axis never reached the dispatcher and the"
                " clause below reduces to running one kernel three times"
                " ([[reached-but-inert]]). Widen the batch set or pick a"
                " fixture whose m crosses choose_gemm_plan's 16/32"
                " thresholds."
            )
        print(
            "  V3: 0 shapes change plan. Under "
            + numeric_mode_name()
            + " that is EXPECTED and not a finding: identical_gemm's"
            " DEVIATION 1876 branch hands the shape to the vendor kernel"
            " and choose_gemm_plan is not the dispatcher."
        )
        return
    print(
        "  V3: "
        + String(total)
        + " GEMM shapes select a DIFFERENT KERNEL at a different batch size."
        + " The clause below asserts those kernels agree bit for bit."
    )


# ===========================================================================
# THE CLAUSE, MAMBA
# ===========================================================================


def mamba_composition(ctx: DeviceContext) raises -> Int:
    """Row 0 and row 1 of one sequence set, run at B in {1, 17, 64},
    compared PER TOKEN. Returns the number of comparisons that MOVED.

    Read the file docstring's cross-token section before reading this: the
    unit is the SEQUENCE, `L` is fixed at `MAMBA_L`, and only `B` varies.
    Row 1 is compared between B=17 and B=64 (it does not exist at B=1) so
    that the claim is not only about the row a tiled kernel happens to put
    first.
    """
    var dims = MambaDims.of(MAMBA_D_MODEL)
    var dm = dims.d_model
    var di = dims.d_inner
    var l = MAMBA_L
    var seed = corpus_case_seed(MAMBA_SEED_CASE)
    var w = corpus_weights(seed, dims)

    # ONE input stream, cut three ways. `corpus_tensor` is element-indexed
    # (element i is a pure function of the key and i), so the B=1 input IS
    # the leading L*d_model elements of the B=64 input -- and
    # `refuse_unless_prefix` proves it rather than trusting it.
    var x1 = corpus_x(seed, B_REF, l, dm)
    var x17 = corpus_x(seed, B_ODD, l, dm)
    var x64 = corpus_x(seed, B_BIG, l, dm)
    refuse_unless_prefix(String("mamba x, B=1 in B=17"), x1, x17)
    refuse_unless_prefix(String("mamba x, B=1 in B=64"), x1, x64)
    refuse_unless_prefix(String("mamba x, B=17 in B=64"), x17, x64)

    print("")
    print(
        "MAMBA BLOCK  mamba_block_forward, profile"
        " mojolearn.identical.mamba1.fp32.v1"
    )
    print(
        "  d_model="
        + String(dm)
        + " d_inner="
        + String(di)
        + " dt_rank="
        + String(dims.dt_rank)
        + " L="
        + String(l)
        + "   B in {1, 17, 64}, m = B*L in {8, 136, 512}"
    )

    var o1 = mamba_outputs(ctx, w, x1, B_REF, l, dims)
    var o17 = mamba_outputs(ctx, w, x17, B_ODD, l, dims)
    var o64 = mamba_outputs(ctx, w, x64, B_BIG, l, dims)

    # ---- V1: the row map distinguishes rows ----------------------------
    var r0 = row_slice(o17[0], 0, l, dm)
    var r1 = row_slice(o17[0], 1, l, dm)
    var v1 = diff_bits(r0, r1)
    if v1.n_diff == 0:
        raise Error(
            "batch_invariance: MAMBA V1 VACUOUS. Rows 0 and 1 of the B=17"
            " run are bit-identical on all "
            + String(v1.n_cells)
            + " output cells, which cannot be true of two different input"
            + " sequences. Either `row_slice` returns one row twice or the"
            + " block is not reading its input, and either way every"
            + " comparison below is a value against itself"
            + " ([[reached-but-inert]])."
        )
    print(
        "  V1 rows distinguishable: rows 0 and 1 of the B=17 run differ on "
        + String(v1.n_diff)
        + " of "
        + String(v1.n_cells)
        + " output cells"
    )

    # ---- V2: the token map distinguishes tokens ------------------------
    if l < 2:
        raise Error(
            String("batch_invariance: V2 CANNOT BE ASKED at L=")
            + String(l)
            + ". Token 0 and token L-1 are the same token, so the per-token"
            + " comparison has one token in it and proves nothing about"
            + " position. Use a fixture with L >= 2."
        )
    var t_first = token_slice(o1[0], 0, 0, l, dm)
    var t_last = token_slice(o1[0], 0, l - 1, l, dm)
    var v2 = diff_bits(t_first, t_last)
    if v2.n_diff == 0:
        raise Error(
            "batch_invariance: MAMBA V2 VACUOUS. Token 0 and token "
            + String(l - 1)
            + " of the B=1 run are bit-identical, so a PER-TOKEN comparison"
            + " is not per-token here. For Mamba this is the stronger"
            + " statement: it would mean the scan's state is not carrying"
            + " anything from one position to the next."
        )
    print(
        "  V2 tokens distinguishable: token 0 and token "
        + String(l - 1)
        + " of row 0 at B=1 differ on "
        + String(v2.n_diff)
        + " of "
        + String(v2.n_cells)
        + " cells"
    )

    # ---- THE CLAUSE ----------------------------------------------------
    var moved = 0
    moved += compare_row_per_token(
        String("mamba residual.out row0 B=1 vs B=17"),
        o1[0], o17[0], 0, 0, l, dm,
    )
    moved += compare_row_per_token(
        String("mamba residual.out row0 B=1 vs B=64"),
        o1[0], o64[0], 0, 0, l, dm,
    )
    moved += compare_row_per_token(
        String("mamba residual.out row1 B=17 vs B=64"),
        o17[0], o64[0], 1, 1, l, dm,
    )

    # The recurrent state after the call. `[B, d_inner, D_STATE]`, no token
    # axis: it is comparable as a whole row and only as a whole row.
    moved += compare_batch_row(
        String("mamba scan.h row0 B=1 vs B=17"), o1[1], o17[1], 0, di * D_STATE
    )
    moved += compare_batch_row(
        String("mamba scan.h row0 B=1 vs B=64"), o1[1], o64[1], 0, di * D_STATE
    )
    moved += compare_batch_row(
        String("mamba scan.h row1 B=17 vs B=64"),
        o17[1], o64[1], 1, di * D_STATE,
    )
    return moved


# ===========================================================================
# THE CLAUSE, LLAMA
# ===========================================================================


def llama_composition(ctx: DeviceContext, case_name: String) raises -> Int:
    """One Llama fixture case run at B in {1, 17, 64}, compared PER TOKEN.
    Returns the number of comparisons that MOVED."""
    var c = fixture_case(fixture_case_by_name(case_name))
    if c.plant != PLANT_NONE:
        raise Error(
            String("batch_invariance: refusing planted case ")
            + String(c.name)
            + ". A score plant is a FLAT index into [B, n_heads, L, S], so"
            + " the same plant lands in a different cell at every B and the"
            + " comparison would be measuring the plant rather than the"
            + " batch."
        )
    var dims = fixture_dims(c)
    var dm = c.d_model
    var l = c.l
    var w = fixture_weights(c)

    var c1 = case_at_batch(c, B_REF)
    var c17 = case_at_batch(c, B_ODD)
    var c64 = case_at_batch(c, B_BIG)
    var x1 = fixture_x(c1)
    var x17 = fixture_x(c17)
    var x64 = fixture_x(c64)
    refuse_unless_prefix(case_name + " x, B=1 in B=17", x1, x17)
    refuse_unless_prefix(case_name + " x, B=1 in B=64", x1, x64)
    refuse_unless_prefix(case_name + " x, B=17 in B=64", x17, x64)

    print("")
    print(
        "LLAMA DECODER LAYER  llama_decoder_layer_forward, profile"
        " mojolearn.identical.transformer.fp32.v1"
    )
    print(
        "  case "
        + case_name
        + ": d_model="
        + String(dm)
        + " n_heads="
        + String(c.n_heads)
        + " n_kv="
        + String(c.n_kv_heads)
        + " head_dim="
        + String(c.head_dim)
        + " inter="
        + String(c.intermediate)
        + " L="
        + String(l)
        + "   B in {1, 17, 64}, m in {"
        + String(l)
        + ", "
        + String(B_ODD * l)
        + ", "
        + String(B_BIG * l)
        + "}"
    )

    var o1 = llama_output(ctx, c1, dims, w, x1)
    var o17 = llama_output(ctx, c17, dims, w, x17)
    var o64 = llama_output(ctx, c64, dims, w, x64)

    var r0 = row_slice(o17, 0, l, dm)
    var r1 = row_slice(o17, 1, l, dm)
    var v1 = diff_bits(r0, r1)
    if v1.n_diff == 0:
        raise Error(
            String("batch_invariance: LLAMA V1 VACUOUS on case ")
            + case_name
            + ". Rows 0 and 1 of the B=17 run are bit-identical on all "
            + String(v1.n_cells)
            + " output cells, which cannot be true of two different input"
            + " sequences ([[reached-but-inert]])."
        )
    print(
        "  V1 rows distinguishable: rows 0 and 1 of the B=17 run differ on "
        + String(v1.n_diff)
        + " of "
        + String(v1.n_cells)
        + " output cells"
    )

    if l < 2:
        raise Error(
            String("batch_invariance: V2 CANNOT BE ASKED at L=")
            + String(l)
            + ". Token 0 and token L-1 are the same token, so the per-token"
            + " comparison has one token in it and proves nothing about"
            + " position. Use a fixture with L >= 2."
        )
    var v2 = diff_bits(
        token_slice(o1, 0, 0, l, dm), token_slice(o1, 0, l - 1, l, dm)
    )
    if v2.n_diff == 0:
        raise Error(
            String("batch_invariance: LLAMA V2 VACUOUS on case ")
            + case_name
            + ". Token 0 and token "
            + String(l - 1)
            + " of the B=1 run are bit-identical, so a PER-TOKEN comparison"
            + " is not per-token here. Under the causal mask those two"
            + " tokens attend to a different number of keys, so equality"
            + " would mean attention is not reaching the output."
        )
    print(
        "  V2 tokens distinguishable: token 0 and token "
        + String(l - 1)
        + " of row 0 at B=1 differ on "
        + String(v2.n_diff)
        + " of "
        + String(v2.n_cells)
        + " cells"
    )

    var moved = 0
    moved += compare_row_per_token(
        case_name + " residual2 row0 B=1 vs B=17", o1, o17, 0, 0, l, dm
    )
    moved += compare_row_per_token(
        case_name + " residual2 row0 B=1 vs B=64", o1, o64, 0, 0, l, dm
    )
    moved += compare_row_per_token(
        case_name + " residual2 row1 B=17 vs B=64", o17, o64, 1, 1, l, dm
    )
    return moved


# ===========================================================================
# THE SABOTAGE'S OWN WIRING
# ===========================================================================


def sabotage_armed() raises -> Bool:
    """Whether the batch-invariance arm is on, ASKED OF BOTH IMPL FILES.

    `modeling_mamba.mojo` and `modeling_llama.mojo` each declare their own
    `is_defined["MOJOLEARN_BATCHINV_SABOTAGE_NORM_CHUNK_FROM_M"]()`, so the
    `-D` name is spelled twice and a typo in either would arm exactly half
    the run -- the mamba half would go red and the llama half would print a
    clean PASS, which is worse than either failing.
    `[[verify-reach-not-output]]` applied to the sabotage's own switch."""
    var m = mamba_batchinv_sabotage_armed()
    var t = llama_batchinv_sabotage_armed()
    if m != t:
        raise Error(
            String("batch_invariance: the batch-invariance sabotage is")
            + " armed in the mamba block ("
            + String(m)
            + ") and the llama block ("
            + String(t)
            + ") DIFFERENTLY. One of the two `-D` string literals is"
            + " misspelled, so half this run would be a clean build"
            + " reported beside a sabotaged one."
        )
    return m


def print_sabotage_prediction(l_mamba: Int) raises:
    """What the armed arm is predicted to do at the shapes about to run,
    computed from `batchinv_norm_chunk` itself so the prediction cannot drift
    from the code that implements it."""
    print("")
    print(
        "SABOTAGE MOJOLEARN_BATCHINV_SABOTAGE_NORM_CHUNK_FROM_M IS ARMED."
    )
    print(
        "  RMSNorm's per-row sum of squares folds in chunks of width"
        " max(1, d_model // (1 + m//64)) instead of one ascending chain."
        " That is gemm contract 6.1's forbidden L = f(k, m), one layer up."
    )
    var batches: List[Int] = [B_REF, B_ODD, B_BIG]
    var line = String("  mamba d_model=") + String(MAMBA_D_MODEL) + " L="
    line += String(l_mamba) + ":"
    for j in range(len(batches)):
        var m = batches[j] * l_mamba
        line += "  B=" + String(batches[j]) + " m=" + String(m)
        line += " chunk=" + String(batchinv_norm_chunk(MAMBA_D_MODEL, m))
    print(line)
    print(
        "  A chunk equal to d_model IS the clean chain, so B=1 is predicted"
        " to be bit-unmoved and the larger batches to move. That asymmetry"
        " is the point: the arm is BATCH-DERIVED, not a fold change, and"
        " every gate this tree has today runs at m < 64 and cannot see it."
    )


# ===========================================================================
# THE DRIVER
# ===========================================================================


def check_block_batch_invariance() raises:
    """The whole gate. Called by `batch_invariance_main.mojo`."""
    var armed = sabotage_armed()

    print(
        "=== BLOCK-LEVEL BATCH INVARIANCE, lifting"
        " IDENTICAL_FP32_CONTRACT.md section 6.1"
    )
    print(
        "mode "
        + numeric_mode_name()
        + "   mamba arm: "
        + mamba_block_sabotage_name()
        + "   llama arm: "
        + llama_block_sabotage_name()
        + "   batchinv arm: "
        + armed_word(armed)
    )
    if PIN_CROSS_VENDOR:
        print(
            "TIER: IDENTICAL. The comparison below is ASSERTED bitwise; a"
            " difference is a FAILURE."
        )
    else:
        print(
            "TIER: "
            + numeric_mode_name()
            + ". The comparison below is REPORTED, NOT ASSERTED."
            + " identical_gemm's DEVIATION 1876 branch hands every"
            + " projection to the vendor kernel under both FAST and"
            + " DETERMINISTIC, so the pinned partition never runs and"
            + " nothing here promises a bit. [[fast-is-not-identical]]:"
            + " FAST promises SPEED ONLY and is never asked a bitwise"
            + " question."
        )

    if armed:
        print_sabotage_prediction(MAMBA_L)

    check_batch_axis_reaches_the_execution_plan()

    var ctx = DeviceContext()
    var moved = 0
    moved += mamba_composition(ctx)
    moved += llama_composition(ctx, String(LLAMA_CASE_SHORT))
    moved += llama_composition(ctx, String(LLAMA_CASE_LONG))

    print("")
    if armed:
        if moved == 0:
            raise Error(
                "batch_invariance: SABOTAGE"
                " BATCHINV_NORM_CHUNK_FROM_M IS ARMED AND MOVED NO BIT."
                " Either its branch was never reached at these shapes or it"
                " is inert there ([[reached-but-inert]]). It falsifies"
                " NOTHING and must not be reported as a passing arm. Check"
                " that m = B*L really exceeds 64 at B=17 and B=64 and that"
                " the RMSNorm kernels are the ones being launched."
            )
        print(
            "SABOTAGE BATCHINV_NORM_CHUNK_FROM_M BIT: "
            + String(moved)
            + " of "
            + String(N_COMPARISONS)
            + " comparisons moved. The clause is FALSIFIABLE: a"
            " reduction whose split is derived from the batch dimension"
            " breaks it, and the B<=3 clauses in mamba_check and"
            " transformer_check do not see this arm at all."
        )
        print(
            "The invariance claim itself is NOT made by a sabotaged build."
            " Re-run without the -D for the clean arm."
        )
        return

    if moved == 0:
        print(
            String("PASS: ")
            + String(N_COMPARISONS)
            + " of "
            + String(N_COMPARISONS)
            + " comparisons bit-identical. A token's output bits"
            " do not depend on how many other sequences were in the launch,"
            " at B in {1, 17, 64}, for both blocks."
        )
        if not PIN_CROSS_VENDOR:
            print(
                "  ...and that is a REPORT under "
                + numeric_mode_name()
                + ", not a card. It says the vendor kernel happened to"
                + " agree at these shapes on this box, which is a"
                + " measurement worth having and is not the claim."
            )
        print(
            "SCOPE: one block of each kind, one device, forward only,"
            " prefill only. OWED: decode carrying a cache across calls at"
            " mixed batch sizes, a multi-block model, and every column this"
            " run did not happen on ([[one-box-verdict-is-not-three]])."
        )
        return

    if PIN_CROSS_VENDOR:
        raise Error(
            String("batch_invariance: FAILED on ")
            + String(moved)
            + " of "
            + String(N_COMPARISONS)
            + " comparisons: a token's output bits DEPEND on how many"
            + " other sequences shared its launch. Read the MOVED lines"
            + " above; each names the token, the cell and both bit"
            + " patterns."
        )
    print(
        String("REPORTED: ")
        + String(moved)
        + " of "
        + String(N_COMPARISONS)
        + " comparisons moved under "
        + numeric_mode_name()
        + ". This is a MEASUREMENT of what the vendor kernels do at"
        + " different batch sizes, not a failure -- nothing in this tier"
        + " promised otherwise. It is the number that prices the pin."
    )
