# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate file of profile `mojolearn.identical.embedding.fp32.v1`, the
path `embedding/IDENTICAL_EMBEDDING_CONTRACT.md` section 11 and OWED item 1
both name.

NOT A PORT. It runs the device spelling
(`embedding/checks/embedding_identical.mojo`) against the host oracle
(`embedding/checks/embedding_oracle.mojo`) and compares every recorded
stage BY BITS.

**RAN ON TWO COLUMNS.** Apple and AMD both produced `embedding.identical.card` at md5 `c7f824c35336bef2a3d0f672a172ef29`, clause (a) passing on 6,887 cells. What is still true is that NO SABOTAGE ARM WAS EVER BUILT for this lane and there is no NVIDIA leg, so the gate has never been shown capable of failing. Written
2026-08-25, DEVIATIONS 1500 through 1524. No `mojo` process has read it, no
device has run it, no bit produced by it has been observed. Every sentence
below that says a clause "passes", a sabotage "bites", or a stage "moves" is
a PREDICTION about what the source says. **The sabotage ledger below has a
column for every arm and a value in none of them**, because inventing
numbers for a file that has not run would be the single worst thing a gate
could do -- a fabricated ledger reads exactly like evidence.

WHAT IT CHECKS
---------------
1. Contract section 10's NINE stages, device against oracle, BITWISE, over a
   case SET, reporting `<tag> on X of Y cells`. "It failed" is not a finding.
2. THE CARD ITSELF: the nine tags of section 10, in section 10's order,
   emitted once each, at the path the CALLER chose (DEVIATION 1501).
3. Contract section 11's clauses (a) through (g), each with the negative
   control that answers "what would make this pass while gating nothing".
4. Sixteen of the eighteen sabotages of section 11.1, each with the fixture
   that can distinguish it AND its predicted INERT MASK asserted.

THREE FINDINGS THIS FILE MADE BEFORE IT EVER RAN, EACH IN A FILE IT MAY NOT
EDIT
--------------------------------------------------------------------------
Reading the two halves of the lane against the contract produced three
statements that are false as written. They are here at the top because a
finding buried in a function is a finding nobody inherits.

**(1) DEVIATION 1502. `EMB_FOLD_READS_LAUNCH` IS NOT "NEVER INERT ON A RUN
OF LENGTH >= 2".** `embedding_identical.mojo`'s switch docstring says it is.
The arm computes `start = Int(block_dim.x) % span`, `block_dim.x` is
`EMB_TPB`, and `_emb_max_tpb` resolves that to
`min(256, IDENTITY_FLOOR_BLOCK, column_max_block_size)` -- **a power of two
on every column this repository has** (256 on Apple/NVIDIA/AMD, 128 on the
portable baseline). So `start == 0`, and the arm is BITWISE INERT, at every
`span` that divides the block size: `R = 2, 4, 8, 16, ...`. `R = 2` is the
very smallest run the sentence promises it fires on, and F-DUPSAME has
`R = 2` and F-TREE4 has `R = 4`. A gate that took either as this arm's
witness would fire the arm, watch nothing move, and record "the arm did not
bite" -- the exact inverse of the truth. **The witness is F-ORDER3, `R = 3`**,
and `embedding_fixture.mojo::guard_fold_reads_launch_separates` recomputes
the rotation from the block size the binary actually compiled with, because
the fixture stops separating at `block = 128` (rotation 2 on a three-element
run rounds back to the pinned answer) and that is the portable-baseline
column's cap.

**(2) DEVIATION 1504. `EMB_RANK_BY_ARRIVAL` NEEDS `T > 2 * EMB_TPB`, NOT
`T > EMB_TPB`.** The contract says the arm is "INERT on a single-block
launch, where arrival order IS position order -- so the gate must run more
than one block". More than one is not enough. The arm emits phase 0
(`(t // block) % 2 == 0`) and then phase 1; with `T <= 2 * block` phase 0 is
exactly `[0, block)` and phase 1 is exactly `[block, T)`, and concatenating
them REPRODUCES ASCENDING ORDER. The reordering only begins at the third
band, which phase 0 also claims and which therefore lands before the second
band. This is the most dangerous arm in the lane by the contract's own
account (6.3 case 3: an integer atomic's COUNT is order free and its SLOT is
not), so a witness that silently goes inert is the worst one to lose.
`f_multiblock` is `T = 600`.

**(3) DEVIATION 1506, AND IT IS THE SERIOUS ONE. THE DEVICE ENTRY POINTS
PERFORM NO REFUSAL AT ALL.** Contract 9.1 says a NaN or an infinity "is
refused by name, `refuse_nonfinite`... before any recorded stage", and
contract section 8 says an out-of-range id "is REFUSED BY NAME... Not
clamped, not wrapped, not silently dropped". Both are TRUE of
`emb_forward_oracle` and `emb_backward_oracle`, which call
`emb_refuse_shape`, `emb_refuse_ids` and `refuse_nonfinite` on entry. **Both
are FALSE of `identical_embedding_forward_into` and
`identical_embedding_backward_into`, which check nothing.** That file's own
docstring says so and calls it a design -- "THE HOST REFUSALS RUN BEFORE
THIS, NOT INSIDE IT" -- and the design has no enforcement anywhere: there is
no wrapper in the profile that does both, so a caller who forgets gets

  * a NaN folded into `emb.dw` and hashed onto a card, which is exactly the
    vendor-shaped payload IDENTITY_PATHS row 39 forbids; and, worse,
  * an OUT-OF-BOUNDS DEVICE READ for a negative id. `emb_gather_kernel`
    computes `weight.unsafe_load(v * width + j)` with no bounds branch on
    the normative path, so `v == -1` reads BEFORE the buffer. That is a
    memory-safety bug and not a numerics one.

This file MEASURES the first half -- it plants a NaN in `W`, reads it back
OFF THE DEVICE by bits to prove reach, calls the device forward with the
trace DISABLED so no payload can enter a card, and reports whether anything
raised. **It deliberately does NOT run the out-of-range half**, because
running an out-of-bounds read to demonstrate that it is out of bounds is not
a test, it is the bug. Clause (f) raises on the gap by default and names the
one-line fix; `MOJOLEARN_EMB_DEVICE_REFUSAL_GAP_ACK=1` downgrades the raise
to a printed line and the line prints EITHER WAY, so the gap cannot be
suppressed, only acknowledged. **Clause (f) is the LAST clause this file
runs**, precisely so that a raise there still leaves every other clause
measured.

TWO MORE, SMALLER
------------------
**DEVIATION 1505.** Contract 11.1's table has eighteen rows and
`embedding_identical.mojo` has SIXTEEN switches. The contract names
`EMB_SORT_KEY_ID_ONLY_UNSTABLE` as unbuildable (it is a `PLAN_SORT` arm and
`PLAN_SORT` is not written) and it does NOT say the same of
`EMB_FOLD_VIA_GEMM_ONEHOT` -- which nonetheless has no switch anywhere in
the lane. So the table promises an arm that does not exist, and contract
5.2(d)'s claim that the one-hot routing "survives as a SABOTAGE whose job is
to PRINT the difference between routing and pinning" is, today, false. Named
here; not fixed here.

**DEVIATION 1517.** At `d == 0` and at `T == 0` the device backward returns
BEFORE R1, R2 and R3, so `counts`, `run_begin` and `perm` are never written
-- while `emb_backward_stages` computes and records all three. The two cards
therefore disagree at three of nine stages on two shapes contract section 8
declares LEGAL. Clause (a) compares the integer stages only where `T >= 1`
and `d >= 1` and REPORTS the gap by name rather than passing over it.

WHAT WOULD MAKE EACH CLAUSE PASS WHILE GATING NOTHING
-------------------------------------------------------
* **(a)** A device dump and an oracle dump that are the same object, or two
  dumps of different LENGTHS compared to the shorter of the two.
  `compare_stage` RAISES on a length mismatch and the two dumps come from
  two files that share no arithmetic. What clause (a) CANNOT catch is our
  oracle being wrong in the same way as our device; only an independent
  reference can, and this lane has none -- cuML, cuVS and RAFT contain no
  embedding table and there is no PyTorch checkout.
* **(b)** Re-calling one set of buffers. Every launch builds fresh
  everything, including the run scratch.
* **(c)** A padded variant whose padding was never actually appended, so the
  comparison is a value against itself. Each half carries a FIRING control:
  the same extra positions carrying a REAL id and a NONZERO gradient MUST
  move `dW`, and a zero there raises VACUOUS.
* **(d)** **THIS CLAUSE CANNOT RUN AND THE FILE SAYS SO RATHER THAN
  SKIPPING IT.** Plan invariance needs two plans and `PLAN_SORT` is not
  written. What IS run is its host shadow -- `emb_perm_by_total_order_key`
  against `emb_perm_by_scan` at every case -- which proves the KEY and not
  the KERNEL. The launch-geometry half cannot run either: `EMB_TPB` is a
  `comptime`, so "three unrelated launch geometries" is three BUILDS.
* **(e)** A split that leaves every row's contributors on one side. Then the
  carry and the `dW += dW_micro` spelling agree and the clause passes on
  both. `emb_case_straddling_rows` is asserted POSITIVE before the clause
  runs, and the by-add control is asserted to MOVE.
* **(f)** An unconditional refusal. Every plant would be "refused by name"
  and the clause would gate nothing. The control is a CLEAN call that must
  NOT raise.
* **(g)** An arm whose `-D` was misspelled and silently ignored, so a clean
  build reports itself as a bitten sabotage. `MOJOLEARN_EMB_EXPECT_SABOTAGE`
  is DEVIATION 1510 and it is `tools/gemm_ladder.sh:71`'s scar written down.

THE SABOTAGE LEDGER, **NOT MEASURED**
---------------------------------------
Sixteen buildable arms. The `first stage moved`, `cells` and `stages moved`
columns are what a measured ledger reports and what this one CANNOT report,
because nothing here has run.

    arm                    predicted first stage  witness       inert
    FOLD_DESCENDING        emb.dw                 f_order3      f_dupsame
    FOLD_BALANCED_TREE     emb.dw                 f_tree4       f_order3
    SEED_SEEDLESS          emb.dw                 f_negzero1    f_dupsame
    SINGLE_RUN_BYPASS      emb.dw                 f_negzero1    f_dupsame
    EMPTY_ROW_SKIPPED      emb.dw_seed            f_empty       (none)
    EMPTY_ROW_NEG_ZERO     emb.dw_seed            f_empty       f_nodup
    FOLD_READS_LAUNCH      emb.dw                 f_order3      f_tree4
    RANK_BY_ARRIVAL        emb.perm               f_multiblock  f_nodup
    SORT_TIE_REVERSED      emb.perm               f_order3      f_dupsame
    PAD_ROW_CONTRIBUTES    emb.counts             f_pad         f_nodup
    PAD_ROW_NEG_ZERO       emb.dw                 f_pad         f_nodup
    NO_FLUSH_ACC           emb.dw                 (see below)   ALL, on FTZ
    GATHER_NO_FLUSH        emb.fwd                f_subw        f_nodup
    GATHER_CLAMP_OOR       emb.fwd                NOT RUNNABLE  --
    ACCUM_BY_ADD           emb.dw, clause (e)     f_split       f_split row 2
    ACCUM_REFILLS          emb.dw_seed            f_accum       f_nodup

**THREE OF THE SIXTEEN ARE NOT REACH PROOFS AND ARE REPORTED AS SMOKE TESTS
BY NAME**, which is the count a reader should carry rather than "sixteen
arms exist".

  * `NO_FLUSH_ACC` is **INERT ON APPLE ENTIRELY** (contract 9.3): `ftz` is
    bitwise a no-op on an FTZ backend, so pinned and unpinned agree there.
    `check_flush_pin_is_reached` measures whether THIS backend flushes and
    says which; the arm becomes a bit-level reach proof, with no edit, on
    the first non-flushing column.
  * `GATHER_CLAMP_OOR` needs an out-of-range id, and the CLEAN half of that
    comparison is the out-of-bounds read of DEVIATION 1506. Not run.
  * `EMPTY_ROW_SKIPPED` has no inert case: it skips the store for EVERY
    cell, not only the empty rows, so with a poisoned buffer it moves
    everywhere. That is the arm being blunt, not the gate being weak, and
    the poison is what makes it visible at all.

RUNNING IT
-----------
    MOJOLEARN_IDENTITY_TRACE=/tmp/e.card \\
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        embedding/checks/embedding_check.mojo

and one sabotage arm at a time, each of which MUST fail:

    MOJOLEARN_EMB_EXPECT_SABOTAGE=FOLD_BALANCED_TREE \\
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_EMB_SABOTAGE_FOLD_BALANCED_TREE=1 \\
        -I . embedding/checks/embedding_check.mojo

When an arm is armed this file INVERTS its verdict: a clean compare is then
the FAILURE, because it means the sabotage was reached and made no
difference, or was never reached at all. Both are `[[reached-but-inert]]`.

ENVIRONMENT
------------
    MOJOLEARN_IDENTITY_TRACE               where the card goes (DEV 1501)
    MOJOLEARN_EMB_EXPECT_SABOTAGE          guard against a misspelled -D
    MOJOLEARN_EMB_CHECK_CLAUSE_B           run clause (b)
    MOJOLEARN_EMB_CHECK_CLAUSE_C           run clause (c), both halves
    MOJOLEARN_EMB_CHECK_CLAUSE_D           run clause (d)'s host shadow
    MOJOLEARN_EMB_CHECK_CLAUSE_E           run clause (e), the carry
    MOJOLEARN_EMB_CHECK_CLAUSE_F           run clause (f), the row-39 audit
    MOJOLEARN_EMB_DEVICE_REFUSAL_GAP_ACK   acknowledge DEVIATION 1506

Clauses (b) through (f) are OFF by default and each is one line to turn on.
That is a COST decision and not a confidence one, and a leg that reports
only clause (a) has closed only clause (a) -- the printed SCOPE line says so.

OWED, AND THIS FILE COVERS NONE OF IT
---------------------------------------
* **A COMPILE.** Nothing here has been through the front end.
* **A RUN, ON ANY COLUMN.** Zero bits observed.
* **THE 16 SABOTAGE BUILDS.** Sixteen compiles, one arm each. No runner
  script exists and writing one under `tools/` is outside this file's remit.
* **`PLAN_SORT`,** and therefore contract clause (d) in its real form.
* **THE SHIPPED SHAPE.** `V = 128256`, `d = 4096`, `T = 4096`, which
  contract 11.2 calls mandatory. `dW` there is 2.10 GB and
  `[[no-heavy-local-compute]]` binds this author; it belongs on a rented
  GPU.
* **AN INDEPENDENT REFERENCE.** Every clause here compares our device
  against our oracle. There is no embedding table in cuML, cuVS or RAFT and
  no PyTorch checkout, so two halves of one lane agreeing is all the
  evidence this file can produce.
* **EVERY COLUMN THAT IS NOT THE FIRST ONE THIS RUNS ON.** Contract 13:
  nothing cross-vendor until a leg runs, and Apple and AMD agreed bit for
  bit through 302 GEMM stages while NVIDIA diverged at
  `tree001.winners.scores`.

`[[mojo-buffer-freed-at-last-use]]`: every `DeviceBuffer` below is a LOCAL
held past the `ctx.synchronize()` that reads it, with an explicit `_ = x^`
at the foot of every launcher. A buffer is dead at `.unsafe_ptr()` and this
repository has lost a night to that.
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace, read_trace_lines
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from transformer.checks.transformer_fixture import fixture_splitmix64

from embedding.checks.embedding_fixture import (
    BITS_MIN_SUBNORMAL,
    BITS_NEG_ZERO,
    BITS_ONE,
    BITS_POISON,
    BITS_POS_INF,
    BITS_POS_ZERO,
    BITS_QNAN,
    EMB_CASE_COUNT,
    EMB_PLANT_NONE,
    EmbCase,
    bits32_hex,
    bits_of,
    count_poison,
    emb_case,
    emb_case_by_name,
    emb_case_config,
    emb_case_distinct_ids,
    emb_case_dw_prev,
    emb_case_dy,
    emb_case_empty_rows,
    emb_case_ids,
    emb_case_index,
    emb_case_max_run,
    emb_case_note,
    emb_case_pad_positions,
    emb_case_straddling_rows,
    emb_case_subnormal_weights,
    emb_case_weight,
    emb_chain,
    emb_poison,
    emb_splitmix64,
    f32_from_bits,
    guard_case_shapes,
    guard_dupsame_is_blind,
    guard_fold_reads_launch_separates,
    guard_hashed_dy_separates,
    guard_negzero1_separates,
    guard_nodup_is_blind,
    guard_order3_separates,
    guard_rank_by_arrival_separates,
    guard_subacc_reaches,
    guard_tree4_separates,
    mode_is_identical,
    mode_name,
    nonfinite_cells,
)
from embedding.checks.embedding_identical import (
    ANY_EMB_SABOTAGE,
    EMB_TPB,
    emb_run_scratch_ints,
    emb_sabotage_name,
    identical_embedding_backward_into,
    identical_embedding_forward_into,
)
from embedding.checks.embedding_oracle import (
    EMB_MAX_POSITIONS,
    EMB_NO_PADDING_IDX,
    EmbConfig,
    emb_backward_oracle,
    emb_backward_seed,
    emb_counts,
    emb_forward_oracle,
    emb_perm_by_scan,
    emb_perm_by_total_order_key,
    emb_refuse_ids,
    emb_run_begin,
    pack_emb_key,
    refuse_nonfinite,
)


# ===========================================================================
# THE CARD PATH IS THE CALLER'S WHEN THE CALLER NAMES ONE.
#
# DEVIATION 1501, and it is DEVIATION 970 not repeated. There, a
# `comptime TRACE_PATH` was read DIRECTLY by every write site, so the card
# always landed in /tmp no matter what the caller asked for.
# `tools/e1_bootstrap.sh` phase 8 sets `MOJOLEARN_IDENTITY_TRACE` to
# `<out>/lanes/<lane>.identical.card` and its comment claimed the driver
# honored it. It did not. The Apple column of 2026-08-24 ran the mamba lane
# GREEN -- clause (a) PASS, 16/16 stages bit-identical, a 17-tag card
# written -- and phase 8 reported "NO CARD written", because the card was in
# /tmp under the alias while the judge looked in lanes/.
#
# THE SHAPE OF THAT BUG IS WHAT MATTERS: a green check with no card is
# INDISTINGUISHABLE FROM SUCCESS to the gate above it. So the path is read
# from the environment at RUN time and the constant below is only the
# fallback for a standalone run by hand.
# ===========================================================================

comptime TRACE_PATH = "/tmp/mojolearn_embedding.trace"


def card_path() -> String:
    """`MOJOLEARN_IDENTITY_TRACE` when the caller set it, else
    `TRACE_PATH`. Read at RUN time, never at compile time, because the
    harness chooses the directory."""
    var p = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if p.byte_length() > 0:
        return p^
    return String(TRACE_PATH)


comptime TAG_PREFIX = "emb"

#: Contract section 10's nine stages.
comptime EMB_STAGE_COUNT = 9

#: Contract section 11 clause (b)'s repeated-launch count.
comptime CLAUSE_B_LAUNCHES = 8

#: Which of the nine stages are INTEGER. Three of them, and contract section
#: 6 is the whole reason: `emb.perm` is where a divergent sort becomes
#: visible BEFORE it reaches a float. A card that recorded only `emb.dw`
#: would see a wrong tie order as a wrong gradient with no localization.
comptime STAGE_IDS = 0
comptime STAGE_WEIGHT = 1
comptime STAGE_FWD = 2
comptime STAGE_DY = 3
comptime STAGE_COUNTS = 4
comptime STAGE_RUN_BEGIN = 5
comptime STAGE_PERM = 6
comptime STAGE_DW_SEED = 7
comptime STAGE_DW = 8


def emb_stage_tag(i: Int) raises -> String:
    """Contract section 10's nine tags, in section 10's order.

    **THE STRINGS AND THE ORDER ARE BOTH PART OF THE INSTRUMENT.**
    `tools/identity_trace_diff.py` aligns two traces by their TAG SEQUENCES
    before it compares a single hash, so a renamed tag does not produce a
    smaller diff -- it produces a WRONG ALIGNMENT that pairs one run's stage
    against another run's different stage and reports a plausible answer.

    They are spelled HERE and not in the oracle because `EmbStages` carries
    the values and no tag strings at all; contract section 10 is the only
    place they exist, and duplicating a markdown table into exactly one Mojo
    function is the smallest duplication available."""
    if i == STAGE_IDS:
        return String("emb.ids")
    if i == STAGE_WEIGHT:
        return String("emb.weight")
    if i == STAGE_FWD:
        return String("emb.fwd")
    if i == STAGE_DY:
        return String("emb.dy")
    if i == STAGE_COUNTS:
        return String("emb.counts")
    if i == STAGE_RUN_BEGIN:
        return String("emb.run_begin")
    if i == STAGE_PERM:
        return String("emb.perm")
    if i == STAGE_DW_SEED:
        return String("emb.dw_seed")
    if i == STAGE_DW:
        return String("emb.dw")
    raise Error(
        String("embedding_check: no stage ")
        + String(i)
        + " (contract section 10 lists "
        + String(EMB_STAGE_COUNT)
        + ")"
    )


def stage_is_integer(i: Int) -> Bool:
    return i == STAGE_IDS or i == STAGE_COUNTS or i == STAGE_RUN_BEGIN or (
        i == STAGE_PERM
    )


def env_on(name: String) -> Bool:
    return String(getenv(name)) != ""


def env_str(name: String) -> String:
    return String(getenv(name))


def hexbits(v: Float32) -> String:
    return bits32_hex(v)


def hexi32(v: Int32) -> String:
    """An `Int32` in decimal. Integers round-trip through `String` and do not
    flush, so unlike a float they may be printed as themselves --
    `[[mojo-string-float-roundtrip]]` is a float rule and applying it to an
    integer would be cargo cult."""
    return String(Int(v))


# ===========================================================================
# HOST <-> DEVICE
# ===========================================================================


def _upload_f32(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    for i in range(n, n_buf):
        host.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _upload_i32(
    ctx: DeviceContext, values: List[Int32]
) raises -> DeviceBuffer[DType.int32]:
    var n = len(values)
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.int32](n_buf)
    var host = ctx.enqueue_create_host_buffer[DType.int32](n_buf)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    for i in range(n, n_buf):
        host.unsafe_ptr().unsafe_store(i, Int32(0))
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _download_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """The first `n` elements as a host list.

    **`n` IS THE USED LENGTH AND NEVER `len(buf)`.** `core/identity_trace.mojo`
    rule 3 is emphatic about this for a reason -- the run scratch is sized
    with `emb_run_scratch_ints` and used shorter, and hashing the tail folds
    uninitialized memory into a stage, which differs run to run on ONE
    machine and would make the instrument report divergence everywhere."""
    if n <= 0:
        return List[Float32]()
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def _download_i32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int
) raises -> List[Int32]:
    if n <= 0:
        return List[Int32]()
    var host = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.int32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Int32](capacity=n)
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


# ===========================================================================
# ONE CALL, EACH SIDE
# ===========================================================================


struct EmbDump(Movable):
    """The nine stages of one case, one side of the comparison.

    Six are Float32 and three are Int32, and they are held in two lists
    rather than one because a stage's DTYPE is part of the card record
    (`core/identity_trace.mojo` writes `<seq> <tag> <dtype> <count> <hash>`)
    and casting the integers to floats to share a container would put a
    lossy value on the card. `emb.perm` holding `16777217` would hash the
    same as `16777216`."""

    var f: List[List[Float32]]
    var i: List[List[Int32]]

    def __init__(out self):
        self.f = List[List[Float32]]()
        self.i = List[List[Int32]]()
        for _ in range(EMB_STAGE_COUNT):
            self.f.append(List[Float32]())
            self.i.append(List[Int32]())


def host_dump(c: EmbCase) raises -> EmbDump:
    """The oracle's nine stages for one case.

    THE FORWARD AND THE BACKWARD ARE BOTH RUN. Contract section 2 says a
    call is one or the other, and contract section 10's nine tags span both,
    so a card carrying all nine is a card of TWO calls on one fixture. That
    is stated because a reader counting device launches against card records
    will otherwise find one too many.

    **`emb.dw_seed` IS DEFINED HERE AS "the buffer after the seed and the
    `padding_idx` store, before any contribution is folded"**, and it is
    obtained as `emb_backward_oracle` at `T = 0`. It is NOT
    `emb_backward_seed`'s return value, and the difference is real: the
    oracle applies the `padding_idx` zeroing in `emb_backward_oracle` and
    not in `emb_backward_seed`, so under `accumulate` the two differ at row
    `padding_idx`. `preflight` asserts that relationship explicitly rather
    than leaving the stage's meaning to whichever function a reader opens
    first. Taking `T = 0` gives ONE definition that both sides can produce
    -- the device's `identical_embedding_backward_into` at `n_positions < 1`
    enqueues exactly the seed and the pad row and returns -- and it gates
    contract section 8's `T == 0` stated value for free."""
    var cfg = emb_case_config(c)
    var ids = emb_case_ids(c)
    var w = emb_case_weight(c)
    var dy = emb_case_dy(c)
    var prev = emb_case_dw_prev(c)
    var d = EmbDump()

    d.i[STAGE_IDS] = ids.copy()
    d.f[STAGE_WEIGHT] = w.copy()
    d.f[STAGE_FWD] = emb_forward_oracle(w, ids, cfg)
    d.f[STAGE_DY] = dy.copy()
    d.i[STAGE_COUNTS] = emb_counts(ids, cfg)
    d.i[STAGE_RUN_BEGIN] = emb_run_begin(d.i[STAGE_COUNTS])
    d.i[STAGE_PERM] = emb_perm_by_scan(ids, cfg)
    d.f[STAGE_DW_SEED] = emb_backward_oracle(
        List[Float32](), List[Int32](), cfg, prev
    )
    d.f[STAGE_DW] = emb_backward_oracle(dy, ids, cfg, prev)
    return d^


def device_dump(ctx: DeviceContext, c: EmbCase) raises -> EmbDump:
    """The device's nine stages for one case, read back off the device.

    FRESH EVERYTHING on every call -- fresh weight, fresh ids, fresh `dW`,
    fresh run scratch -- which is what makes clause (b) a loop over this
    function rather than a re-call of one set of buffers.

    **`dW` IS POISONED BEFORE THE BACKWARD AND THE POISON'S ARRIVAL IS
    MEASURED.** Contract 11.1 gives `EMB_EMPTY_ROW_SKIPPED` the inert mask
    "any gate that does not POISON the output buffer, since a fresh
    allocation may already be zero". A gate that relied on the allocator
    would be a gate that passed for a reason nobody chose, and `count_poison`
    on the readback is how the reliance is removed rather than assumed away.
    `[[verify-reach-not-output]]`.

    The run scratch is sized with `emb_run_scratch_ints`, never guessed --
    `identical_gemm_into`'s docstring records what a guess cost the GEMM
    lane: a 1-float workspace passed to a SPLITK dispatch still produced the
    right answer at `64 x 4` because the allocation had slack, and only at
    `64 x 64` did whole regions of the output come back `+0.0`."""
    var cfg = emb_case_config(c)
    var ids = emb_case_ids(c)
    var w = emb_case_weight(c)
    var dy = emb_case_dy(c)
    var prev = emb_case_dw_prev(c)
    var t = c.n_positions
    var cells = cfg.vocab * cfg.width
    var d = EmbDump()

    d.i[STAGE_IDS] = ids.copy()
    d.f[STAGE_WEIGHT] = w.copy()
    d.f[STAGE_DY] = dy.copy()

    # ---- the forward -----------------------------------------------------
    var dw_ids = _upload_i32(ctx, ids)
    var dw_w = _upload_f32(ctx, w)
    var y_cells = t * cfg.width
    var dy_out = _upload_f32(ctx, emb_poison(y_cells))
    identical_embedding_forward_into(ctx, dy_out, dw_w, dw_ids, t, cfg)
    ctx.synchronize()
    d.f[STAGE_FWD] = _download_f32(ctx, dy_out, y_cells)
    _ = dy_out^

    # ---- the seed stage, as a T = 0 backward -----------------------------
    var seed_start: List[Float32]
    if cfg.accumulate:
        seed_start = prev.copy()
    else:
        seed_start = emb_poison(cells)
    var dseed = _upload_f32(ctx, seed_start)
    var d_dy0 = _upload_f32(ctx, List[Float32]())
    var d_ids0 = _upload_i32(ctx, List[Int32]())
    var c0 = _upload_i32(ctx, _zeros_i32_list(cfg.vocab))
    var b0 = _upload_i32(ctx, _zeros_i32_list(cfg.vocab + 1))
    var p0 = _upload_i32(ctx, _zeros_i32_list(1))
    identical_embedding_backward_into(
        ctx, dseed, d_dy0, d_ids0, c0, b0, p0, 0, cfg
    )
    ctx.synchronize()
    d.f[STAGE_DW_SEED] = _download_f32(ctx, dseed, cells)
    _ = dseed^
    _ = d_dy0^
    _ = d_ids0^
    _ = c0^
    _ = b0^
    _ = p0^

    # ---- the backward ----------------------------------------------------
    var dw_start: List[Float32]
    if cfg.accumulate:
        dw_start = prev.copy()
    else:
        dw_start = emb_poison(cells)
    # The three run-scratch buffers are sized INDIVIDUALLY here, because the
    # kernels index them independently: `counts` is `V`, `run_begin` is
    # `V + 1` and `perm` is at most `T`. `emb_run_scratch_ints` gives the
    # TOTAL a caller who packs them into one allocation needs, and this
    # caller does not pack, so the total is ASSERTED against the sum rather
    # than used as a length. A check that used the total as one buffer's
    # length would over-allocate silently and hide a real sizing error --
    # which is the shape of what a guess cost the GEMM lane.
    var want = emb_run_scratch_ints(cfg.vocab, t)
    if want != cfg.vocab + cfg.vocab + 1 + t and want != 1:
        raise Error(
            String("embedding_check: emb_run_scratch_ints(")
            + String(cfg.vocab)
            + ", "
            + String(t)
            + ") returned "
            + String(want)
            + " and the three buffers this gate allocates need "
            + String(cfg.vocab + cfg.vocab + 1 + t)
            + ". One of the two is wrong about the run structure's size."
        )
    var ddw = _upload_f32(ctx, dw_start)
    var ddy = _upload_f32(ctx, dy)
    var counts = _upload_i32(ctx, _zeros_i32_list(cfg.vocab))
    var run_begin = _upload_i32(ctx, _zeros_i32_list(cfg.vocab + 1))
    var perm = _upload_i32(ctx, _zeros_i32_list(t if t > 0 else 1))

    # **THE POISON'S ARRIVAL IS MEASURED, NOT ASSUMED.** Reach is per branch
    # and this is the branch `EMB_EMPTY_ROW_SKIPPED` lives on: if the poison
    # never got onto the device, a skipped `+0.0` store is invisible and the
    # arm reports itself inert for a reason that has nothing to do with the
    # kernel ([[verify-reach-not-output]]).
    if not cfg.accumulate and cells > 0:
        var before = _download_f32(ctx, ddw, cells)
        if count_poison(before) != cells:
            raise Error(
                String("embedding_check: the dW POISON did not arrive. ")
                + String(count_poison(before))
                + " of "
                + String(cells)
                + " cells hold BITS_POISON before the backward. Contract"
                + " 11.1 gives EMB_EMPTY_ROW_SKIPPED the inert mask 'any"
                + " gate that does not POISON the output buffer', so a gate"
                + " whose poison is missing cannot falsify that arm"
                + " ([[reached-but-inert]])."
            )
    identical_embedding_backward_into(
        ctx, ddw, ddy, dw_ids, counts, run_begin, perm, t, cfg
    )
    ctx.synchronize()
    d.i[STAGE_COUNTS] = _download_i32(ctx, counts, cfg.vocab)
    d.i[STAGE_RUN_BEGIN] = _download_i32(ctx, run_begin, cfg.vocab + 1)
    var perm_used = len(emb_perm_by_scan(ids, cfg))
    d.i[STAGE_PERM] = _download_i32(ctx, perm, perm_used)
    d.f[STAGE_DW] = _download_f32(ctx, ddw, cells)
    _ = ddw^
    _ = ddy^
    _ = counts^
    _ = run_begin^
    _ = perm^
    _ = dw_w^
    _ = dw_ids^
    return d^


def _zeros_i32_list(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n if n > 0 else 1)
    for _ in range(n):
        out.append(Int32(0))
    return out^


# ===========================================================================
# COMPARING
# ===========================================================================


@fieldwise_init
struct StageDiff(Copyable, Movable):
    var name: String
    var n_cells: Int
    var n_diff: Int
    var first: Int
    var a_bits: UInt32
    var b_bits: UInt32


def compare_f32(
    name: String, host: List[Float32], dev: List[Float32], loud: Bool
) raises -> StageDiff:
    """Bitwise, cell by cell.

    A LENGTH mismatch RAISES rather than being compared to the shorter of
    the two, because a stage of the wrong SIZE is a different defect from a
    stage of the wrong VALUE, and comparing to the shorter is how a
    truncated stage passes.

    **BY BITS AND NEVER BY COMPARES.** `host[i] == dev[i]` would call `+0.0`
    and `-0.0` equal, which would launder the ONE output difference this
    whole profile turns on -- contract 5.5's sole-`-0.0` contributor and
    contract 7.1's `ftz` hole are both signed-zero facts -- and Metal
    flushes compare operands (IDENTITY_PATHS row 49), so a compare-written
    gate has a different meaning on different columns."""
    if len(host) != len(dev):
        raise Error(
            String("embedding_check: stage ")
            + name
            + " has "
            + String(len(host))
            + " cells on one side and "
            + String(len(dev))
            + " on the other. A stage of the wrong SHAPE is a different"
            + " defect from a stage of the wrong VALUE and is not compared"
            + " to the shorter of the two."
        )
    var n_diff = 0
    var first = -1
    for i in range(len(host)):
        if bits_of(host[i]) != bits_of(dev[i]):
            n_diff += 1
            if first < 0:
                first = i
    var ab = UInt32(0)
    var bb = UInt32(0)
    if first >= 0:
        ab = bits_of(host[first])
        bb = bits_of(dev[first])
    if n_diff != 0 and loud:
        print(
            "  MOVED "
            + name
            + "  "
            + String(n_diff)
            + " of "
            + String(len(host))
            + " cells, first cell "
            + String(first)
            + "  a "
            + hexbits(host[first])
            + "  b "
            + hexbits(dev[first])
        )
    elif loud:
        print("  OK    " + name + "  (" + String(len(host)) + " cells)")
    return StageDiff(name, len(host), n_diff, first, ab, bb)


def compare_i32(
    name: String, host: List[Int32], dev: List[Int32], loud: Bool
) raises -> StageDiff:
    """The integer stages. Same discipline; nothing here rounds or flushes,
    so a difference is a WRONG PERMUTATION and not a rounding.

    **THAT IS THE WHOLE OF CONTRACT SECTION 6.** A sort that is wrong is a
    WRONG ANSWER, detectable at `emb.perm` before it ever reaches a float,
    rather than a silent redefinition of what "identical" means."""
    if len(host) != len(dev):
        raise Error(
            String("embedding_check: integer stage ")
            + name
            + " has "
            + String(len(host))
            + " cells on one side and "
            + String(len(dev))
            + " on the other"
        )
    var n_diff = 0
    var first = -1
    for i in range(len(host)):
        if host[i] != dev[i]:
            n_diff += 1
            if first < 0:
                first = i
    var ab = UInt32(0)
    var bb = UInt32(0)
    if first >= 0:
        ab = UInt32(Int(host[first]) & 0xFFFFFFFF)
        bb = UInt32(Int(dev[first]) & 0xFFFFFFFF)
    if n_diff != 0 and loud:
        print(
            "  MOVED "
            + name
            + "  "
            + String(n_diff)
            + " of "
            + String(len(host))
            + " cells, first cell "
            + String(first)
            + "  a "
            + hexi32(host[first])
            + "  b "
            + hexi32(dev[first])
        )
    elif loud:
        print("  OK    " + name + "  (" + String(len(host)) + " cells)")
    return StageDiff(name, len(host), n_diff, first, ab, bb)


def compare_dumps(
    c: EmbCase, a: EmbDump, b: EmbDump, loud: Bool
) raises -> List[StageDiff]:
    """All nine stages, in card order.

    **DEVIATION 1517 IS HANDLED HERE AND NAMED RATHER THAN SMOOTHED OVER.**
    At `d == 0` and at `T == 0` the device backward returns BEFORE R1, R2
    and R3 (`identical_embedding_backward_into`'s two early returns), so
    `counts`, `run_begin` and `perm` are never written -- while
    `emb_backward_stages` computes and records all three. The two cards
    therefore disagree at three of nine stages on two shapes contract
    section 8 declares LEGAL. Those three stages are SKIPPED on those shapes
    and the skip is reported, because comparing an unwritten device buffer
    against a computed oracle value would report a defect that is real but
    is not the one clause (a) is about, and quietly comparing them anyway
    would make every degenerate case red for a reason nobody would trace."""
    var out = List[StageDiff]()
    var degenerate = c.n_positions < 1 or c.width < 1
    for i in range(EMB_STAGE_COUNT):
        var tag = emb_stage_tag(i)
        if degenerate and stage_is_integer(i) and i != STAGE_IDS:
            out.append(
                StageDiff(tag + " [SKIPPED]", 0, 0, -1, UInt32(0), UInt32(0))
            )
            continue
        if stage_is_integer(i):
            out.append(compare_i32(tag, a.i[i], b.i[i], loud))
        else:
            out.append(compare_f32(tag, a.f[i], b.f[i], loud))
    return out^


def count_moved(diffs: List[StageDiff]) -> Int:
    var n = 0
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            n += 1
    return n


def first_moved_index(diffs: List[StageDiff]) -> Int:
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            return i
    return -1


def first_moved(diffs: List[StageDiff]) -> String:
    """The contract's own report shape, verbatim: `<tag> on X of Y cells`.

    Contract 11.1's discipline is that an arm must move the stage its OWN
    clause writes "and no earlier one", so the FIRST moved stage is the
    finding and the count is the evidence that it is not a single-bit
    coincidence."""
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            return (
                diffs[i].name
                + " on "
                + String(diffs[i].n_diff)
                + " of "
                + String(diffs[i].n_cells)
                + " cells"
            )
    return String("")


def total_cells(d: EmbDump) -> Int:
    var n = 0
    for i in range(EMB_STAGE_COUNT):
        n += len(d.f[i]) + len(d.i[i])
    return n


# ===========================================================================
# THE CARD
# ===========================================================================


def write_card(
    mut trace: IdentityTrace, prefix: String, d: EmbDump
) raises:
    """All nine stages onto a trace, in contract section 10's order.

    The three integer stages go through `record_list_i32` and the six float
    stages through `record_list_f32`, so the card's `dtype` column is true.
    Casting the permutation to Float32 to share one code path would put a
    LOSSY value on the card -- `emb.perm` holding `16777217` would hash the
    same as `16777216` -- and a card that cannot represent its own stage is
    worse than one that omits it."""
    for i in range(EMB_STAGE_COUNT):
        var tag = prefix + "." + emb_stage_tag(i)
        if stage_is_integer(i):
            var one = d.i[i].copy()
            trace.record_list_i32(tag, one)
            _ = one^
        else:
            var onef = d.f[i].copy()
            trace.record_list_f32(tag, onef)
            _ = onef^


def check_card_tags(path: String) raises -> Int:
    """The card holds contract section 10's nine tags, in section 10's
    order, each exactly once, each carrying this driver's prefix.

    A CHECK ON THE COMPOSITION AND NOT ON THE ARITHMETIC, **and it is the
    check that catches a card that was never written**, which was DEVIATION
    970's actual damage: `read_trace_lines` on a path nothing wrote raises,
    rather than returning an empty list that a lenient gate would call zero
    differences."""
    var lines = read_trace_lines(path)
    print(
        "card: "
        + String(len(lines))
        + " records at "
        + path
        + ", contract section 10 wants "
        + String(EMB_STAGE_COUNT)
    )
    if len(lines) != EMB_STAGE_COUNT:
        raise Error(
            String("embedding_check: the card has ")
            + String(len(lines))
            + " records and contract section 10 lists "
            + String(EMB_STAGE_COUNT)
        )
    for i in range(len(lines)):
        var fields = lines[i].split("\t")
        if len(fields) < 2:
            raise Error(
                String("embedding_check: malformed trace record: ") + lines[i]
            )
        var got = String(fields[1])
        var expect = String(TAG_PREFIX) + "." + emb_stage_tag(i)
        if got != expect:
            raise Error(
                String("embedding_check: card record ")
                + String(i)
                + " is '"
                + got
                + "', contract section 10 wants '"
                + expect
                + "'. A renamed or reordered tag does not make"
                + " identity_trace_diff.py's diff smaller, it makes its"
                + " ALIGNMENT wrong."
            )
    print(
        "card: "
        + String(EMB_STAGE_COUNT)
        + "/"
        + String(EMB_STAGE_COUNT)
        + " tags in contract section 10's order, all unique, three of them"
        " INTEGER (contract section 6: emb.perm is where a divergent sort"
        " becomes visible before it reaches a float)"
    )
    return len(lines)


# ===========================================================================
# PREFLIGHT: the assertions the fixture and the oracle ASK THIS FILE FOR,
# and the DEMONSTRATIONS that the fixtures can separate anything at all.
#
# They run BEFORE any device call, so a build with a blind fixture fails in
# a second rather than after a case sweep.
# ===========================================================================


def check_flush_pin_is_reached() raises -> Bool:
    """Does THIS backend flush subnormals, and what does that mean for
    `EMB_NO_FLUSH_ACC`?

    Contract 9.3 and gemm 4.1's correction are the standing warning: **a
    gate cannot prove a flush pin is REACHED on an FTZ backend by showing
    that pinned and unpinned bits differ, because on that backend they do
    not.** `cluster/checks/kmeans_identity_check.mojo::
    check_fused_contraction_pin` answers it the only way it can be answered
    -- by REPORTING which arm the backend took -- and this is the same
    question one profile over.

    The measurement: `ftz` of the smallest positive subnormal. If it comes
    back `+0.0` the pin did something; if it comes back `0x00000001` then
    either the pin compiled away (FAST) or the operand never reached it. On
    a hardware-FTZ backend the raw arithmetic already flushes, so
    `EMB_NO_FLUSH_ACC` is **INERT ENTIRELY** and must be reported as a SMOKE
    TEST rather than a passing arm.

    Returns whether the arm is expected to be able to fire on this build."""
    var sub = f32_from_bits(BITS_MIN_SUBNORMAL)
    var flushed = ftz(sub)
    var raw_sum = sub + sub
    print(
        "preflight: ftz(0x00000001) -> "
        + hexbits(flushed)
        + "; raw 0x00000001 + 0x00000001 -> "
        + hexbits(raw_sum)
        + "  (mode "
        + mode_name()
        + ")"
    )
    if not mode_is_identical():
        print(
            "  EMB_NO_FLUSH_ACC: **SMOKE TEST ONLY**. Under FAST the pin"
            " compiles away, so the armed and clean spellings are the same"
            " code and the arm cannot fire. FAST makes no identity claim."
        )
        return False
    if bits_of(flushed) != BITS_POS_ZERO:
        raise Error(
            String("embedding_check: ftz(0x00000001) returned ")
            + bits32_hex(flushed)
            + " under NUMERIC_IDENTICAL. Seam E3's whole content is that"
            + " helper, and if it does not flush the smallest subnormal then"
            + " every flush claim in this profile is unfounded."
        )
    if bits_of(raw_sum) == BITS_POS_ZERO:
        print(
            "  EMB_NO_FLUSH_ACC: **SMOKE TEST ONLY -- THIS BACKEND FLUSHES"
            " IN HARDWARE.** The raw add of two subnormals already returned"
            " +0.0, so the pinned and unpinned spellings are BITWISE EQUAL"
            " here and the arm is INERT ENTIRELY (contract 9.3). Reach"
            " cannot be shown on this column by moving bits; it becomes a"
            " bit-level reach proof, with no edit, on the first"
            " non-flushing backend. Apple's bits not moving is NOT evidence"
            " that the pin is unreached (gemm 4.1's correction)."
        )
        return False
    print(
        "  EMB_NO_FLUSH_ACC: the backend did NOT flush the raw add, so the"
        " arm CAN fire here and its verdict is a real one"
    )
    return True


def preflight() raises -> Bool:
    print("preflight: the assertions the fixture and the oracle asked for")

    # ---- 1. The two refusal bounds are the constants they claim to be ----
    # Contract section 3 freezes `EMB_MAX_POSITIONS` at `2^30 - 1` and the
    # oracle's docstring says it "keeps a position inside the low 32 bits of
    # `pack_emb_key`'s packed key with room to spare". Both halves are
    # asserted, because a bound that is right when it is typed is not the
    # same thing as a bound the toolchain agrees with.
    if EMB_MAX_POSITIONS != 1073741823:
        raise Error(
            String("embedding_check: EMB_MAX_POSITIONS is ")
            + String(EMB_MAX_POSITIONS)
            + " and contract section 3 freezes it at 1073741823 (2^30 - 1)."
            + " This is a v2, not a bug."
        )
    if EMB_NO_PADDING_IDX >= 0:
        raise Error(
            "embedding_check: EMB_NO_PADDING_IDX is not negative, so"
            " `EmbConfig.has_padding` would call the absent case a real row"
        )
    print(
        "  EMB_MAX_POSITIONS = "
        + String(EMB_MAX_POSITIONS)
        + " (2^30 - 1, a REFUSAL bound that reaches no arithmetic),"
        " EMB_NO_PADDING_IDX = "
        + String(EMB_NO_PADDING_IDX)
    )

    # ---- 2. `pack_emb_key` is a TOTAL order, and the sign-extension trap -
    # Contract 6.2(a), DEVIATION 1303. Two Mojo traps live in that
    # expression and both have cost this repository a run.
    #   `[[mojo-int-widening-sign-extends]]`: an Int32 widened to UInt64
    #   SIGN-EXTENDS, so a negative id becomes 0xFFFFFFFF........ and sorts
    #   above every real token. The mask is not decoration and this is the
    #   assertion that proves the mask is there.
    var k_neg = pack_emb_key(Int32(-1), 0)
    var k_max = pack_emb_key(Int32(2147483647), 0)
    if k_neg != UInt64(0xFFFFFFFF) << 32:
        raise Error(
            String("embedding_check: pack_emb_key(-1, 0) is not")
            + " 0xFFFFFFFF00000000. The `& 0xFFFFFFFF` mask that stops an"
            + " Int32 sign-extending through a UInt64 widening is missing or"
            + " wrong ([[mojo-int-widening-sign-extends]])."
        )
    var a = pack_emb_key(Int32(3), 7)
    var b = pack_emb_key(Int32(3), 8)
    var cc = pack_emb_key(Int32(4), 0)
    if not (a < b and b < cc):
        raise Error(
            "embedding_check: pack_emb_key is not a TOTAL order. Contract"
            " 6.2(a)'s whole argument is that no two positions share a key,"
            " so ANY correct sort returns the same permutation -- and that"
            " turns 'the sort must be stable', a property of an"
            " implementation, into 'the sort must be correct', a property"
            " anybody can check. If the key is not ordered by (id, t) the"
            " argument is void."
        )
    _ = k_max
    print(
        "  pack_emb_key: (3,7) < (3,8) < (4,0), and pack_emb_key(-1, 0) ="
        " 0xffffffff00000000 exactly, so the sign-extension mask is present"
    )

    # ---- 3. The two splitmix64 copies agree -----------------------------
    # `embedding_fixture.mojo::emb_splitmix64` is a THIRD copy of one hash
    # (the mamba lane's, the transformer lane's DEVIATION 1000, and this).
    # That file names the price -- "three copies of a hash have three
    # chances to be edited apart" -- and asks for exactly this assertion as
    # the whole mitigation. `[[mojo-amp-plus-is-bitwise-and]]` is what it
    # actually guards: `&+` computes `x & k` with NO compile error, and a
    # `+` "fixed" into a `&+` in one copy and not the other is exactly the
    # edit this catches.
    var seeds: List[UInt64] = [
        UInt64(0),
        UInt64(1),
        UInt64(0x9E3779B97F4A7C15),
        UInt64(0xFFFFFFFFFFFFFFFF),
        UInt64(0x456D62436F727075),
    ]
    for i in range(len(seeds)):
        if emb_splitmix64(seeds[i]) != fixture_splitmix64(seeds[i]):
            raise Error(
                String("embedding_check: the embedding fixture's splitmix64")
                + " and the transformer fixture's disagree at seed "
                + String(i)
                + ". One of the three copies has been edited apart from the"
                + " others ([[mojo-amp-plus-is-bitwise-and]] is the usual"
                + " way)."
            )
    print(
        "  splitmix64: the embedding copy and the transformer copy agree on "
        + String(len(seeds))
        + " seeds (the THIRD copy's stated cost, checked)"
    )

    # ---- 4. `emb.dw_seed`'s two definitions, and where they differ ------
    # `host_dump` defines the stage as `emb_backward_oracle` at `T = 0` and
    # NOT as `emb_backward_seed`'s return value. The two differ at exactly
    # one place and the difference is asserted rather than described: the
    # oracle applies the `padding_idx` zeroing in `emb_backward_oracle` and
    # the seed function does not, so under `accumulate` the carried value
    # survives at row `padding_idx` in one and is `+0.0` in the other.
    var pad_cfg = EmbConfig(4, 2, 1, True)
    var carried = List[Float32]()
    for i in range(8):
        carried.append(f32_from_bits(BITS_ONE + UInt32(i)))
    var seed_only = emb_backward_seed(pad_cfg, carried)
    var seed_stage = emb_backward_oracle(
        List[Float32](), List[Int32](), pad_cfg, carried
    )
    var differ = 0
    var differ_in_pad = 0
    for i in range(len(seed_only)):
        if bits_of(seed_only[i]) != bits_of(seed_stage[i]):
            differ += 1
            if i // pad_cfg.width == pad_cfg.padding_idx:
                differ_in_pad += 1
    if differ == 0 or differ != differ_in_pad:
        raise Error(
            String("embedding_check: emb_backward_seed and the T=0 backward")
            + " differ on "
            + String(differ)
            + " cells of which "
            + String(differ_in_pad)
            + " are in row padding_idx. The stage's definition rests on the"
            + " difference being EXACTLY the padding row under accumulate;"
            + " if it is anywhere else, `emb.dw_seed` means two things and"
            + " the card is comparing them."
        )
    print(
        "  emb.dw_seed: emb_backward_seed and the T=0 backward differ on "
        + String(differ)
        + " cells, ALL of them in row padding_idx under accumulate. The"
        " stage is DEFINED as the T=0 backward, which is what both sides can"
        " produce"
    )

    # ---- 5. THE FIXTURE DEMONSTRATIONS ---------------------------------
    # Every fixture this lane ships comes with the demonstration that it CAN
    # separate. These are `loss_check.mojo`'s guard 4 at ten new sites.
    print("preflight: DEMONSTRATING that each fixture can separate")
    print("  " + guard_case_shapes())
    print("  " + guard_nodup_is_blind())
    print("  " + guard_dupsame_is_blind())
    print("  " + guard_order3_separates())
    print("  " + guard_tree4_separates())
    print("  " + guard_negzero1_separates())
    print("  " + guard_subacc_reaches())
    print("  " + guard_hashed_dy_separates())

    var armed = emb_sabotage_name()
    print(
        "  "
        + guard_fold_reads_launch_separates(
            EMB_TPB, armed == "FOLD_READS_LAUNCH"
        )
    )
    print(
        "  "
        + guard_rank_by_arrival_separates(
            EMB_TPB, armed == "RANK_BY_ARRIVAL"
        )
    )
    var flush_can_fire = check_flush_pin_is_reached()
    print(
        "  resolved block size EMB_TPB = "
        + String(EMB_TPB)
        + "  (DEVIATION 1502 turns on this being a power of two)"
    )
    return flush_can_fire


# ===========================================================================
# THE CASE SET
# ===========================================================================


def clause_a_cases() raises -> List[Int]:
    """Every case except the two the profile must REFUSE.

    DEVIATION 1508. `f_oor_high` and `f_oor_neg` have NO ORACLE ANSWER --
    `emb_refuse_ids` raises on them, by design -- so running them through
    clause (a) would produce an oracle raise that reads exactly like a
    broken gate. They belong to clause (f) and they are used there.

    Nothing else is excluded. In particular the two DEGENERATE cases,
    `f_t0` and `f_d0`, are IN, because contract section 8 gives both a
    STATED value and a stated value nobody checks is a comment. What they
    cannot check is the three integer stages, and DEVIATION 1517 is that."""
    var out = List[Int]()
    for k in range(EMB_CASE_COUNT):
        var c = emb_case(k)
        if c.refused:
            continue
        out.append(k)
    return out^


@fieldwise_init
struct CaseVerdict(Copyable, Movable):
    """One case's clause-(a) result, kept so clause (g)'s expectation table
    can be evaluated ACROSS cases in one binary. That is what makes an
    arm's INERT MASK expressible at all: it needs two cases in one
    process."""

    var name: String
    var n_moved: Int
    var first_index: Int
    var first: String
    var cells: Int
    var moved_dw: Int
    var moved_perm: Int
    var moved_counts: Int
    var moved_seed: Int
    var moved_fwd: Int


def clause_a_case(
    ctx: DeviceContext, k: Int, mut trace: IdentityTrace, prefix: String
) raises -> CaseVerdict:
    """Contract 11(a) at ONE fixture case, all nine stages, BITWISE."""
    var c = emb_case(k)
    var host = host_dump(c)
    var dev = device_dump(ctx, c)
    write_card(trace, prefix, dev)
    var diffs = compare_dumps(c, host, dev, False)
    var moved = count_moved(diffs)
    var fi = first_moved_index(diffs)
    var fname = String("")
    if fi >= 0:
        fname = emb_stage_tag(fi)
    var cells = total_cells(host)
    var line = (
        "  case "
        + String(k)
        + " "
        + String(c.name)
        + "  V="
        + String(c.vocab)
        + " d="
        + String(c.width)
        + " T="
        + String(c.n_positions)
        + " pad="
        + String(c.padding_idx)
        + " acc="
        + String(c.accumulate)
        + " R_max="
        + String(emb_case_max_run(c))
        + " empty_rows="
        + String(emb_case_empty_rows(c))
        + "  "
        + String(cells)
        + " cells: "
    )
    if moved == 0:
        print(line + "9/9 stages bit-identical")
        # The case's own account of what it CANNOT see, printed beside the
        # green. A per-case green with no statement of the case's blind
        # spots is how a fourteen-case sweep comes to read as fourteen
        # independent pieces of evidence when two of them are controls that
        # see nothing by construction.
        print("        " + emb_case_note(c))
    else:
        print(
            line
            + String(moved)
            + " of 9 stages MOVED, first at "
            + first_moved(diffs)
        )
        _ = compare_dumps(c, host, dev, True)
    return CaseVerdict(
        String(c.name),
        moved,
        fi,
        fname,
        cells,
        diffs[STAGE_DW].n_diff,
        diffs[STAGE_PERM].n_diff,
        diffs[STAGE_COUNTS].n_diff,
        diffs[STAGE_DW_SEED].n_diff,
        diffs[STAGE_FWD].n_diff,
    )


# ===========================================================================
# CLAUSE (b): the same bits on every one of eight repeated launches
# ===========================================================================


def clause_b(ctx: DeviceContext, k: Int) raises:
    """Contract 11(b). Eight launches, each with its OWN fresh weight, ids,
    `dW`, run scratch and kernel dispatches, every stage compared to the
    first on every cell.

    Fresh EVERYTHING and not just a fresh call, because the failure this
    clause is for is an execution plan that is not a pure function of the
    input -- a scratch buffer read before it is written, an accumulator that
    survives a call, a launch geometry chosen from a clock. Re-calling one
    set of buffers hides all three, and the run scratch (`counts`,
    `run_begin`, `perm`) is exactly the kind of buffer that would.

    **THE ONE HOLE NO CONTROL CLOSES:** eight identical runs of a
    deterministically WRONG backward pass this. Clause (b) is an invariance
    claim and says nothing about correctness; clause (a) is what covers
    that. It is stated because "8/8 launches identical" reads like a strong
    result and is not one on its own."""
    var c = emb_case(k)
    print(
        "clause (b): "
        + String(CLAUSE_B_LAUNCHES)
        + " repeated launches of "
        + String(c.name)
        + ", every stage, every cell, fresh state each time"
    )
    var base = device_dump(ctx, c)
    var cells = total_cells(base)
    for run in range(2, CLAUSE_B_LAUNCHES + 1):
        var got = device_dump(ctx, c)
        var diffs = compare_dumps(c, base, got, False)
        if count_moved(diffs) != 0:
            raise Error(
                String("embedding_check: CLAUSE (b) FAILED, launch ")
                + String(run)
                + " differs from launch 1 at "
                + first_moved(diffs)
            )
    print(
        "clause (b): PASS, launches 2.."
        + String(CLAUSE_B_LAUNCHES)
        + " bit-identical to launch 1 on all "
        + String(cells)
        + " cells of all 9 stages"
    )


# ===========================================================================
# CLAUSE (c): PADDING and SEQUENCE-LENGTH invariance
#
# **CONTRACT 7.2 IS WHY THIS CLAUSE IS NOT THE TRANSFORMER'S CLAUSE (c), AND
# THE DIFFERENCE IS NOT A WEAKNESS OF THE GATE.** Batch composition
# invariance is FALSE for an embedding gradient and no construction can make
# it true: if sequence S shares a launch with S' and S' also contains token
# v, then dW[v, :] sums both, and it SHOULD -- that is what the gradient of
# a shared weight means. A contract that claimed otherwise would be claiming
# that adding data does not change a gradient. So this file does not run a
# batch-composition half at all, and says so rather than running a weaker
# version of it and calling it clause (c).
#
# What IS claimed, and what these two halves gate: `dW` is a pure function
# of the bits of `dY`, the bits of `ids`, `V`, `d`, `padding_idx` and the
# seed -- and never of how many INERT positions surround the real ones.
# ===========================================================================


def _extend_ids(c: EmbCase, extra: Int, extra_id: Int32) raises -> List[
    Int32
]:
    """Case `c`'s ids with `extra` positions APPENDED.

    Appended and not prepended, deliberately: contract 5.1 clause 1 pins
    ASCENDING ABSOLUTE POSITION, so prepending would renumber every real
    position and the comparison would be measuring the renumbering. This is
    the padded-batch shape a caller actually has."""
    var ids = emb_case_ids(c)
    for _ in range(extra):
        ids.append(extra_id)
    return ids^


def _extend_dy(c: EmbCase, extra: Int, zero: Bool) raises -> List[Float32]:
    """Case `c`'s `dY` with `extra` rows APPENDED, all `+0.0` or all
    distinctly NONZERO.

    The nonzero branch is the FIRING control's, and the value is chosen so
    it CANNOT BE ABSORBED. **Absorption is one of the ways a control went
    dead on 2026-08-25**: a control that dropped a tail term did not move,
    because the term rounded away against the running sum. `1.0` and its
    neighbours are comparable to the hashed generator's mid range and
    enormous beside its small end, so an appended contributor changes the
    sum on every case here."""
    var dy = emb_case_dy(c)
    for _ in range(extra):
        for j in range(c.width):
            if zero:
                dy.append(f32_from_bits(BITS_POS_ZERO))
            else:
                dy.append(f32_from_bits(BITS_ONE + UInt32(j)))
    return dy^


def _device_dw(
    ctx: DeviceContext,
    cfg: EmbConfig,
    ids: List[Int32],
    dy: List[Float32],
    prev: List[Float32],
) raises -> List[Float32]:
    """One backward on the device, `dW` back on the host. `dW` is POISONED
    first, always."""
    var t = len(ids)
    var cells = cfg.vocab * cfg.width
    var start: List[Float32]
    if cfg.accumulate:
        start = prev.copy()
    else:
        start = emb_poison(cells)
    var ddw = _upload_f32(ctx, start)
    var ddy = _upload_f32(ctx, dy)
    var dids = _upload_i32(ctx, ids)
    var counts = _upload_i32(ctx, _zeros_i32_list(cfg.vocab))
    var run_begin = _upload_i32(ctx, _zeros_i32_list(cfg.vocab + 1))
    var perm = _upload_i32(ctx, _zeros_i32_list(t if t > 0 else 1))
    identical_embedding_backward_into(
        ctx, ddw, ddy, dids, counts, run_begin, perm, t, cfg
    )
    ctx.synchronize()
    var out = _download_f32(ctx, ddw, cells)
    _ = ddw^
    _ = ddy^
    _ = dids^
    _ = counts^
    _ = run_begin^
    _ = perm^
    return out^


def clause_c(ctx: DeviceContext) raises:
    """Contract 11(c), BOTH halves, EACH with its own FIRING control, plus
    contract 7.1's counterexample asserted as a KNOWN EXCEPTION.

    **HALF ONE, PADDING.** A row's bits are identical whether the
    contributing positions are surrounded by 0, 1 or 200 positions carrying
    `padding_idx`. Those positions are dropped AT THE SOURCE (contract
    section 8, DEVIATION 1311) and enter no run, so this holds
    unconditionally -- there is no `-0.0` hole here, because there is no
    addition.

    **HALF TWO, SEQUENCE LENGTH.** A row's bits are identical whether the
    contributing positions are surrounded by 0, 1 or 200 positions carrying
    a REAL id whose `dY` row is exactly `+0.0`. That is what a right-padded
    batch produces and what the loss lane's `ignore_index` PROVES it
    produces ("the ignored row is `+0.0`, STORED, and provably inert"). It
    holds by contract 7.1 -- **and it has the one hole, which the third
    part asserts.**

    **EACH HALF CARRIES A FIRING CONTROL AND WITHOUT ONE IT IS WORTHLESS.**
    If the extension never happened -- if `_extend_ids` and `_extend_dy` returned the
    original lists, or if the device ignored the extra positions entirely --
    every comparison below would be a value against ITSELF and would pass
    for ever, on every vendor, hiding any length dependence there is. So
    each half first runs the SAME extension with a REAL id and a NONZERO
    `dY`, and requires `dW` to MOVE. A zero there raises VACUOUS, not
    FAILED. `[[verify-reach-not-output]]`.

    **AND THE THIRD PART IS THE HONEST ONE.** Contract 7.1's theorem is
    "inert everywhere except at a `-0.0` accumulator, which is reachable
    only through `ftz` of a negative subnormal partial sum", and F-SUBACC
    plants exactly that. The clause asserts the EXCEPTION -- row 0 ends
    `0x80000000` and row 1, the same run plus a trailing `+0.0`, ends
    `0x00000000` -- rather than letting the case pass quietly as though the
    theorem had no hole. Two other contracts in this tree assert the theorem
    WITHOUT the hole (transformer 7.1 and loss 7.3 both write "and the seed
    forbids that"), and contract OWED item 5 carries the correction to them.
    """
    var c = emb_case(emb_case_by_name(String("f_pad")))
    var cfg = emb_case_config(c)
    var prev = emb_case_dw_prev(c)
    var base_ids = emb_case_ids(c)
    var base_dy = emb_case_dy(c)
    var base = _device_dw(ctx, cfg, base_ids, base_dy, prev)
    var cells = cfg.vocab * cfg.width

    print(
        "clause (c): padding and sequence-length invariance on "
        + String(c.name)
        + ", "
        + String(cells)
        + " cells, each half with its own FIRING control"
    )
    if not cfg.has_padding():
        raise Error(
            "embedding_check: clause (c)'s padding half needs a case with a"
            " real padding_idx and this one has none, so the whole half"
            " would compare a value against itself"
        )
    if emb_case_pad_positions(c) == 0:
        raise Error(
            "embedding_check: clause (c)'s case has a padding_idx and NO"
            " POSITION CARRYING IT, so the drop-at-source path is never"
            " reached ([[reached-but-inert]])"
        )

    # ---- THE FIRING CONTROL, shared by both halves ----------------------
    var ctrl_ids = _extend_ids(c, 3, Int32(0))
    var ctrl_dy = _extend_dy(c, 3, False)
    var ctrl_dw = _device_dw(ctx, cfg, ctrl_ids, ctrl_dy, prev)
    var ctrl_moved = 0
    for i in range(cells):
        if bits_of(base[i]) != bits_of(ctrl_dw[i]):
            ctrl_moved += 1
    if ctrl_moved == 0:
        raise Error(
            "embedding_check: CLAUSE (c) IS VACUOUS. Appending three"
            " positions carrying a REAL id with a NONZERO dY row moved NO"
            " CELL of dW. Either the extension never reached the device or"
            " the backward is ignoring positions past the original T, and"
            " every 'invariant' comparison below is a value against itself"
            " ([[reached-but-inert]])."
        )
    print(
        "clause (c) control: 3 extra REAL positions with nonzero dY move "
        + String(ctrl_moved)
        + " of "
        + String(cells)
        + " cells, so the extension reaches the arithmetic"
    )

    # ---- HALF ONE: padding positions ------------------------------------
    var pad_counts: List[Int] = [1, 5, 200]
    for pi in range(len(pad_counts)):
        var pids = _extend_ids(c, pad_counts[pi], Int32(cfg.padding_idx))
        var pdy = _extend_dy(c, pad_counts[pi], True)
        var dw = _device_dw(ctx, cfg, pids, pdy, prev)
        var moved = 0
        for i in range(cells):
            if bits_of(base[i]) != bits_of(dw[i]):
                moved += 1
        if moved != 0:
            raise Error(
                String("embedding_check: CLAUSE (c) PADDING FAILED. ")
                + String(pad_counts[pi])
                + " appended positions carrying padding_idx moved "
                + String(moved)
                + " of "
                + String(cells)
                + " cells of dW. Contract section 8 says such a position is"
                + " dropped AT THE SOURCE and enters no run, so this is a"
                + " finding about the profile and not about the gate."
            )
    print(
        "clause (c) padding: PASS, dW bit-identical with 0, 1, 5 and 200"
        " appended padding_idx positions"
    )

    # ---- HALF TWO: real ids with exactly-+0.0 gradient rows -------------
    for pi in range(len(pad_counts)):
        var lids = _extend_ids(c, pad_counts[pi], Int32(0))
        var ldy = _extend_dy(c, pad_counts[pi], True)
        var dw = _device_dw(ctx, cfg, lids, ldy, prev)
        var moved = 0
        var first = -1
        for i in range(cells):
            if bits_of(base[i]) != bits_of(dw[i]):
                moved += 1
                if first < 0:
                    first = i
        if moved != 0:
            raise Error(
                String("embedding_check: CLAUSE (c) LENGTH FAILED. ")
                + String(pad_counts[pi])
                + " appended positions carrying a REAL id with an exactly"
                + " +0.0 dY row moved "
                + String(moved)
                + " of "
                + String(cells)
                + " cells, first at cell "
                + String(first)
                + " ("
                + hexbits(base[first])
                + " -> "
                + hexbits(dw[first])
                + "). Contract 7.1 says such a contributor is inert UNLESS"
                + " the accumulator is -0.0 at that step, and this case has"
                + " no subnormal intermediate, so it is a finding about the"
                + " profile."
            )
    print(
        "clause (c) length: PASS, dW bit-identical with 0, 1, 5 and 200"
        " appended REAL positions whose dY rows are exactly +0.0 (contract"
        " 7.1's theorem)"
    )

    # ---- THE KNOWN EXCEPTION --------------------------------------------
    clause_c_known_exception(ctx)


def clause_c_known_exception(ctx: DeviceContext) raises:
    """Contract 7.1's HOLE, asserted on the device rather than admitted in
    prose.

    F-SUBACC's row 0 is the run `{0x80C00000, 0x00800000}` and its row 1 is
    the same run plus a trailing `0x00000000`. Under IDENTICAL the two must
    come out `0x80000000` and `0x00000000` -- **an exactly-`+0.0`
    contributor has moved a bit**, which is the counterexample to the
    inertness theorem the half above just gated.

    Asserting the exception is the point. A gate that ran F-SUBACC through
    clause (a) and saw it agree with the oracle would have proved that our
    device and our oracle share the hole, and would have said nothing about
    whether the hole is there at all."""
    var c = emb_case(emb_case_by_name(String("f_subacc")))
    var cfg = emb_case_config(c)
    var dw = _device_dw(
        ctx, cfg, emb_case_ids(c), emb_case_dy(c), List[Float32]()
    )
    if not mode_is_identical():
        print(
            "clause (c) KNOWN EXCEPTION [FAST]: RECORDED, NOT ASSERTED. row"
            " 0 -> "
            + hexbits(dw[0])
            + ", row 1 -> "
            + hexbits(dw[1])
            + ". Under FAST `ftz` is the identity, the partial sum stays"
            " subnormal and contract 7.1's route to -0.0 does not exist."
        )
        return
    if bits_of(dw[0]) != BITS_NEG_ZERO or bits_of(dw[1]) != BITS_POS_ZERO:
        raise Error(
            String("embedding_check: CONTRACT 7.1's KNOWN EXCEPTION DID NOT")
            + " REPRODUCE ON THE DEVICE. row 0 -> "
            + hexbits(dw[0])
            + " (wants 0x80000000), row 1 -> "
            + hexbits(dw[1])
            + " (wants 0x00000000). Either seam E3 is not flushing the"
            + " negative subnormal partial sum on this column, or the hole"
            + " is not reachable and contract 7.1's counterexample is"
            + " wrong. Both are findings; neither is this gate failing."
        )
    print(
        "clause (c) KNOWN EXCEPTION: contract 7.1's hole REPRODUCES on the"
        " device. run {a0,a1} -> "
        + hexbits(dw[0])
        + ", run {a0,a1,+0.0} -> "
        + hexbits(dw[1])
        + ". An exactly-+0.0 contributor MOVED A BIT, so the theorem is"
        " 'inert except at a -0.0 accumulator, reachable only through ftz of"
        " a negative subnormal partial sum' and not the slogan."
    )


# ===========================================================================
# CLAUSE (d): PLAN INVARIANCE -- **THIS CLAUSE CANNOT RUN**
# ===========================================================================


def clause_d(ctx: DeviceContext) raises:
    """Contract 11(d), and the honest report of what is missing.

    **THE CLAUSE AS WRITTEN CANNOT RUN AND THIS FUNCTION SAYS SO RATHER
    THAN SKIPPING IT.** DEVIATION 1507. It asks for two things and neither
    is available:

      1. `PLAN_SCAN` and `PLAN_SORT` producing identical `emb.perm` and
         identical `emb.dw` at every fixture. **`PLAN_SORT` IS NOT WRITTEN**
         (contract 6.2, OWED item 2), so there is one plan. The contract
         calls this clause "the strongest evidence available that the
         arithmetic does not read the plan", and it has never been run and
         cannot be.
      2. `emb.dw` identical under at least three unrelated launch
         geometries. **`EMB_TPB` IS A `comptime`** resolved by
         `_emb_max_tpb` at build time, so three geometries is three BUILDS
         and not three calls. One binary cannot do it and pretending
         otherwise would be a green line for a comparison that never
         happened.

    WHAT DOES RUN IS THE HOST SHADOW, and it is worth running because it
    proves the KEY even though it cannot prove the KERNEL.
    `emb_perm_by_total_order_key` sorts contract 6.2(a)'s packed `(id, t)`
    key with a stable merge sort and must return `emb_perm_by_scan`'s
    permutation at every case. Three separate claims collapse into that one
    assertion:

      * the packed key is a TOTAL order, so the sorted list is a pure
        function of `ids` (6.2(a));
      * sorting on it gives ascending `t` inside every run, which is
        contract 5.1 clause 1;
      * therefore 6.2(b) holds -- a STABLE sort by `id` ALONE over a
        POSITION-ORDERED input is the same permutation, because stability
        supplies the low half of the key. That is
        `gbdt/gpu_util/kernel/radix_sort.mojo`'s own argument about
        CatBoost's `(bin || permutationPosition)`, and this is how it stops
        being a quotation.

    **AND ITS NEGATIVE CONTROL.** If the two spellings were the same
    function -- if `emb_perm_by_total_order_key` called `emb_perm_by_scan`
    -- the assertion would pass for ever while comparing nothing. So the
    clause first shows the merge sort producing a DIFFERENT permutation from
    `emb_perm_by_scan` when it is fed a key that is NOT a total order: the
    id alone, with the positions handed to it in REVERSE. That is precisely
    `EMB_SORT_TIE_REVERSED`'s shape and precisely the tie order an unstable
    id-keyed sort is permitted to return, and if it does not differ then the
    sort cannot tell two orders apart and this clause gates nothing."""
    _ = ctx
    print(
        "clause (d): **CANNOT RUN AS SPECIFIED.** PLAN_SORT is not written"
        " (contract 6.2, OWED item 2), so there is ONE plan and plan"
        " invariance -- which the contract calls the strongest evidence that"
        " the arithmetic does not read the plan -- has never been run. The"
        " launch-geometry half needs three BUILDS, because EMB_TPB is a"
        " comptime. What follows is the HOST SHADOW: it proves the KEY, not"
        " the KERNEL."
    )
    var agreed = 0
    var checked = 0
    for k in range(EMB_CASE_COUNT):
        var c = emb_case(k)
        if c.refused:
            continue
        var cfg = emb_case_config(c)
        var ids = emb_case_ids(c)
        var by_scan = emb_perm_by_scan(ids, cfg)
        var by_key = emb_perm_by_total_order_key(ids, cfg)
        checked += 1
        if len(by_scan) != len(by_key):
            raise Error(
                String("embedding_check: CLAUSE (d) SHADOW FAILED on ")
                + String(c.name)
                + ": emb_perm_by_scan has "
                + String(len(by_scan))
                + " entries and emb_perm_by_total_order_key has "
                + String(len(by_key))
                + ". The two must drop the same padding_idx positions."
            )
        var bad = -1
        for i in range(len(by_scan)):
            if by_scan[i] != by_key[i] and bad < 0:
                bad = i
        if bad >= 0:
            raise Error(
                String("embedding_check: CLAUSE (d) SHADOW FAILED on ")
                + String(c.name)
                + " at entry "
                + String(bad)
                + ": scan says "
                + hexi32(by_scan[bad])
                + " and the total-order key says "
                + hexi32(by_key[bad])
                + ". Contract 6.2(a)'s argument is that ANY correct sort"
                + " returns the same permutation because no two positions"
                + " share a key; if the two disagree, one of them is not"
                + " ascending in t inside a run and contract 5.1 clause 1 is"
                + " violated by whichever it is."
            )
        agreed += 1

    # ---- THE NEGATIVE CONTROL -------------------------------------------
    # A key that is NOT a total order, fed in reverse. If this does not
    # differ from the scan, the comparison above cannot see a tie order at
    # all.
    var c2 = emb_case(emb_case_by_name(String("f_dupsame")))
    var cfg2 = emb_case_config(c2)
    var ids2 = emb_case_ids(c2)
    var scan2 = emb_perm_by_scan(ids2, cfg2)
    var reversed_ties = List[Int32]()
    var counts2 = emb_counts(ids2, cfg2)
    var begin2 = emb_run_begin(counts2)
    for _ in range(len(scan2)):
        reversed_ties.append(Int32(0))
    for v in range(cfg2.vocab):
        var lo = Int(begin2[v])
        var hi = Int(begin2[v + 1])
        var back = hi - 1
        for t in range(len(ids2)):
            if Int(ids2[t]) == v:
                reversed_ties[back] = Int32(t)
                back -= 1
    var control_moved = 0
    for i in range(len(scan2)):
        if scan2[i] != reversed_ties[i]:
            control_moved += 1
    if control_moved == 0:
        raise Error(
            "embedding_check: CLAUSE (d)'s SHADOW IS VACUOUS. The"
            " reverse-tie permutation is bit-identical to PLAN_SCAN's on a"
            " fixture with duplicates, so the comparison cannot distinguish"
            " two tie orders and 'the two spellings agree' proves nothing"
            " ([[reached-but-inert]])."
        )
    print(
        "clause (d) shadow: PASS, emb_perm_by_scan == emb_perm_by_total_"
        "order_key on "
        + String(agreed)
        + " of "
        + String(checked)
        + " cases; the reverse-tie control differs on "
        + String(control_moved)
        + " entries, so the comparison can see a tie order. **THE PLAN"
        " CLAUSE ITSELF REMAINS UNGATED.**"
    )


# ===========================================================================
# CLAUSE (e): THE MICROBATCH CARRY
# ===========================================================================


def _slice_ids(ids: List[Int32], lo: Int, hi: Int) -> List[Int32]:
    var out = List[Int32]()
    for t in range(lo, hi):
        out.append(ids[t])
    return out^


def _slice_dy(dy: List[Float32], lo: Int, hi: Int, width: Int) -> List[
    Float32
]:
    var out = List[Float32]()
    for t in range(lo, hi):
        for j in range(width):
            out.append(dy[t * width + j])
    return out^


def clause_e(ctx: DeviceContext) raises:
    """Contract 11(e) and contract 7.4, DEVIATION 1309. **The contract's own
    strongest claim, and the one this file was most keen to gate.**

    > With the accumulator CARRIED rather than two `dW`s added, microbatch
    > splitting is bit-exact at EVERY split point with NO ALIGNMENT
    > CONDITION.

    The unsplit chain for cell `(v, j)` is `((((+0 + a0) + a1) + a2) + ...)`
    over that cell's contributors in ascending `t`. The first microbatch
    computes a PREFIX of that chain and stores it through E4; the second
    loads it through E0 and CONTINUES it. Both seams are `ftz` of an
    already-flushed value, so neither moves a bit, and the resulting
    sequence of additions is the unsplit one TERM FOR TERM. A row with no
    contributor in the second microbatch is carried through unchanged,
    because its "fold" is zero additions on the loaded seed.

    Set beside `archive/plans/gemm/IDENTICAL_BACKWARD_PLAN.md` 3.2, where `dB` splits
    reproduce only at an ALIGNED split -- one that is both a leaf boundary
    and a subtree boundary of v1's balanced tree. **A CHAIN HAS NO
    BOUNDARIES TO ALIGN TO.** And do NOT cite that section's measurement as
    evidence that an accumulator must be a tree: its splits used TWO pieces,
    and over two pieces a serial running sum and a balanced tree are the
    SAME operation.

    **THE NEGATIVE CONTROL IS `EMB_ACCUM_BY_ADD` AND THIS FUNCTION SPELLS
    IT ITSELF.** Contract 11.1 says of that arm: "NO KERNEL IN THIS FILE
    READS THIS SWITCH. It is the one arm whose wrong spelling lives in the
    CALLER, so the check file is what must branch on it." So the by-add
    spelling is computed here on every run, armed or not, and required to
    MOVE. Without it clause (e) would pass on the by-add implementation as
    well as on the carry and would gate nothing.

    **AND THE BY-ADD CONTROL HAS ITS OWN INERT HALF.** The arm's predicted
    mask is "any split that leaves every row's contributors on one side", so
    the clause asserts that the by-add result agrees with the unsplit one on
    the NON-STRADDLING rows and differs on at least one straddling row. An
    arm that moved everywhere would be a smoke test; the row-level mask is
    what makes it a reach proof.

    Every split point of `f_split` is run, `t0 = 1` through `T - 1`, plus a
    THREE-piece split -- because "no alignment condition" is a claim about
    every boundary and a two-piece test cannot distinguish it from the
    gemm lane's aligned-two-piece result.

    Asserted in BOTH modes. The carry theorem is a claim about the SEQUENCE
    of operations and not about their rounding, so it holds under FAST as
    well, and a FAST failure here is a real routing defect rather than a
    numerics one -- `archive/plans/gemm/IDENTICAL_BACKWARD_PLAN.md`'s G2 argument."""
    var c = emb_case(emb_case_by_name(String("f_split")))
    var cfg = emb_case_config(c)
    var ids = emb_case_ids(c)
    var dy = emb_case_dy(c)
    var t = c.n_positions
    var cells = cfg.vocab * cfg.width
    var straddling = emb_case_straddling_rows(c)

    print(
        "clause (e): the microbatch CARRY on "
        + String(c.name)
        + ", T="
        + String(t)
        + ", every split point 1.."
        + String(t - 1)
        + " plus a three-piece split, "
        + String(cells)
        + " cells"
    )
    if straddling == 0:
        raise Error(
            "embedding_check: CLAUSE (e) IS VACUOUS. No vocabulary row of"
            " f_split has contributors on both sides of its split boundary,"
            " so the CARRY and the `dW += dW_micro` spelling agree and the"
            " clause passes on both ([[reached-but-inert]])."
        )
    print(
        "clause (e): "
        + String(straddling)
        + " vocabulary rows straddle the boundary, so the by-add control has"
        " something to move"
    )

    var unsplit = _device_dw(ctx, cfg, ids, dy, List[Float32]())

    # ---- TWO PIECES, EVERY BOUNDARY -------------------------------------
    for t0 in range(1, t):
        var carried = _carry_two(ctx, cfg, ids, dy, t0)
        var moved = 0
        var first = -1
        for i in range(cells):
            if bits_of(unsplit[i]) != bits_of(carried[i]):
                moved += 1
                if first < 0:
                    first = i
        if moved != 0:
            raise Error(
                String("embedding_check: CLAUSE (e) FAILED at t0 = ")
                + String(t0)
                + ". The carried two-piece split moved "
                + String(moved)
                + " of "
                + String(cells)
                + " cells, first at cell "
                + String(first)
                + " ("
                + hexbits(unsplit[first])
                + " -> "
                + hexbits(carried[first])
                + "). Contract 7.4 says a carried chain reproduces the"
                + " unsplit call BIT FOR BIT at EVERY split point with no"
                + " alignment condition, so this is a finding about the"
                + " profile's strongest claim."
            )

    # ---- THREE PIECES ---------------------------------------------------
    var three = _carry_three(ctx, cfg, ids, dy, 2, 4)
    var moved3 = 0
    for i in range(cells):
        if bits_of(unsplit[i]) != bits_of(three[i]):
            moved3 += 1
    if moved3 != 0:
        raise Error(
            String("embedding_check: CLAUSE (e) FAILED on a THREE-piece")
            + " split at (2, 4), "
            + String(moved3)
            + " of "
            + String(cells)
            + " cells. Two pieces are the case a serial chain and a balanced"
            + " tree cannot be told apart on; THREE is where the chain's"
            + " advantage is real, and the gemm lane's own aligned-split"
            + " measurement used two."
        )
    print(
        "clause (e): PASS, the carried split reproduces the unsplit dW BIT"
        " FOR BIT at all "
        + String(t - 1)
        + " two-piece boundaries and at the three-piece split (2, 4)"
    )

    # ---- THE BY-ADD CONTROL, WHICH MUST MOVE ----------------------------
    var t0c = c.split
    var by_add = _by_add_two(ctx, cfg, ids, dy, t0c)
    var straddle_rows = _straddle_row_mask(c, t0c)
    var moved_straddle = 0
    var moved_clean = 0
    for v in range(cfg.vocab):
        for j in range(cfg.width):
            var i = v * cfg.width + j
            if bits_of(unsplit[i]) == bits_of(by_add[i]):
                continue
            if straddle_rows[v]:
                moved_straddle += 1
            else:
                moved_clean += 1
    if moved_straddle == 0:
        raise Error(
            String("embedding_check: CLAUSE (e)'s BY-ADD CONTROL IS DEAD.")
            + " `dW = ftz(dW_first + dW_second)` at t0 = "
            + String(t0c)
            + " gave the SAME BITS as the carried call on every straddling"
            + " row. Contract 7.4 says ADD does not reproduce the unsplit"
            + " chain in general -- `((a0+a1)+a2)+a3` is not"
            + " `(a0+a1) + (a2+a3)` -- so either this fixture's straddling"
            + " runs are too short or their values are exactly"
            + " representable, and clause (e) would pass on the by-add"
            + " spelling as well as on the carry"
            + " ([[reached-but-inert]])."
        )
    if moved_clean != 0:
        raise Error(
            String("embedding_check: CLAUSE (e)'s BY-ADD CONTROL MOVED ")
            + String(moved_clean)
            + " cells on rows whose contributors are ENTIRELY ON ONE SIDE of"
            + " the boundary. Contract 11.1's inert mask for EMB_ACCUM_BY_ADD"
            + " is exactly those rows, and an arm that moves everywhere is a"
            + " smoke test rather than a reach proof"
            + " ([[verify-reach-not-output]])."
        )
    print(
        "clause (e) by-add control: `dW_first + dW_second` at t0 = "
        + String(t0c)
        + " moves "
        + String(moved_straddle)
        + " cells on STRADDLING rows and "
        + String(moved_clean)
        + " on rows entirely on one side. The carry is what the clause"
        " gates and the arm is a REACH PROOF, not a smoke test."
    )


def _carry_two(
    ctx: DeviceContext,
    cfg_in: EmbConfig,
    ids: List[Int32],
    dy: List[Float32],
    t0: Int,
) raises -> List[Float32]:
    """Two microbatches, the accumulator CARRIED.

    Contract 7.4's price, spelled: the caller must (i) `+0.0`-fill `dW`
    exactly once before the FIRST microbatch -- which is what the first call
    with `accumulate` FALSE does -- (ii) NOT fill it again, which is what
    `accumulate` TRUE means, and (iii) present the microbatches in ASCENDING
    `t`, which the loop below does by construction. Out of order and
    contract 5.1 clause 1 is violated and the bits move."""
    var cells = cfg_in.vocab * cfg_in.width
    var first_cfg = EmbConfig(
        cfg_in.vocab, cfg_in.width, cfg_in.padding_idx, False
    )
    var second_cfg = EmbConfig(
        cfg_in.vocab, cfg_in.width, cfg_in.padding_idx, True
    )
    var ddw = _upload_f32(ctx, emb_poison(cells))

    var i1 = _slice_ids(ids, 0, t0)
    var y1 = _slice_dy(dy, 0, t0, cfg_in.width)
    _run_backward_into(ctx, ddw, first_cfg, i1, y1)

    var i2 = _slice_ids(ids, t0, len(ids))
    var y2 = _slice_dy(dy, t0, len(ids), cfg_in.width)
    _run_backward_into(ctx, ddw, second_cfg, i2, y2)

    var out = _download_f32(ctx, ddw, cells)
    _ = ddw^
    return out^


def _carry_three(
    ctx: DeviceContext,
    cfg_in: EmbConfig,
    ids: List[Int32],
    dy: List[Float32],
    t1: Int,
    t2: Int,
) raises -> List[Float32]:
    var cells = cfg_in.vocab * cfg_in.width
    var fresh = EmbConfig(
        cfg_in.vocab, cfg_in.width, cfg_in.padding_idx, False
    )
    var carry = EmbConfig(
        cfg_in.vocab, cfg_in.width, cfg_in.padding_idx, True
    )
    var ddw = _upload_f32(ctx, emb_poison(cells))
    _run_backward_into(
        ctx, ddw, fresh, _slice_ids(ids, 0, t1),
        _slice_dy(dy, 0, t1, cfg_in.width),
    )
    _run_backward_into(
        ctx, ddw, carry, _slice_ids(ids, t1, t2),
        _slice_dy(dy, t1, t2, cfg_in.width),
    )
    _run_backward_into(
        ctx, ddw, carry, _slice_ids(ids, t2, len(ids)),
        _slice_dy(dy, t2, len(ids), cfg_in.width),
    )
    var out = _download_f32(ctx, ddw, cells)
    _ = ddw^
    return out^


def _by_add_two(
    ctx: DeviceContext,
    cfg_in: EmbConfig,
    ids: List[Int32],
    dy: List[Float32],
    t0: Int,
) raises -> List[Float32]:
    """`EMB_ACCUM_BY_ADD`'s spelling: two microbatches EACH seeded `+0.0`,
    then `dW = ftz(dW_first + dW_second)`.

    The add is `ftz` of a plain add, which is what an implementation that
    got this wrong would write -- contract 4.1 says a fold node is a plain
    add with nothing to fuse, so using `identical_mul_add(x, 1.0, y)` here
    would be a DIFFERENT wrong spelling and would confuse two arms."""
    var cells = cfg_in.vocab * cfg_in.width
    var fresh = EmbConfig(
        cfg_in.vocab, cfg_in.width, cfg_in.padding_idx, False
    )
    var d1 = _upload_f32(ctx, emb_poison(cells))
    _run_backward_into(
        ctx, d1, fresh, _slice_ids(ids, 0, t0),
        _slice_dy(dy, 0, t0, cfg_in.width),
    )
    var a = _download_f32(ctx, d1, cells)
    _ = d1^
    var d2 = _upload_f32(ctx, emb_poison(cells))
    _run_backward_into(
        ctx, d2, fresh, _slice_ids(ids, t0, len(ids)),
        _slice_dy(dy, t0, len(ids), cfg_in.width),
    )
    var b = _download_f32(ctx, d2, cells)
    _ = d2^
    var out = List[Float32](capacity=cells if cells > 0 else 1)
    for i in range(cells):
        out.append(ftz(ftz(a[i]) + ftz(b[i])))
    return out^


def _run_backward_into(
    ctx: DeviceContext,
    mut ddw: DeviceBuffer[DType.float32],
    cfg: EmbConfig,
    ids: List[Int32],
    dy: List[Float32],
) raises:
    """One backward into an EXISTING `dW` buffer, with fresh run scratch.

    `dW` is the caller's on purpose: it is the carried accumulator and
    re-uploading it between microbatches would be a host round trip that
    launders whatever the device left there, which is the one thing contract
    7.4's carry must not do."""
    var t = len(ids)
    var ddy = _upload_f32(ctx, dy)
    var dids = _upload_i32(ctx, ids)
    var counts = _upload_i32(ctx, _zeros_i32_list(cfg.vocab))
    var run_begin = _upload_i32(ctx, _zeros_i32_list(cfg.vocab + 1))
    var perm = _upload_i32(ctx, _zeros_i32_list(t if t > 0 else 1))
    identical_embedding_backward_into(
        ctx, ddw, ddy, dids, counts, run_begin, perm, t, cfg
    )
    ctx.synchronize()
    _ = ddy^
    _ = dids^
    _ = counts^
    _ = run_begin^
    _ = perm^


def _straddle_row_mask(c: EmbCase, t0: Int) raises -> List[Bool]:
    """Which vocabulary rows have contributors on BOTH sides of `t0`."""
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
        if t < t0:
            before[v] = True
        else:
            after[v] = True
    var out = List[Bool]()
    for v in range(cfg.vocab):
        out.append(before[v] and after[v])
    return out^


# ===========================================================================
# CLAUSE (f): THE ROW-39 AUDIT, AND DEVIATION 1506
# ===========================================================================


def clause_f(ctx: DeviceContext) raises:
    """Contract 11(f) and contract section 9.1.

    A NaN or an infinity in `W` or in `dY` must be REFUSED BY NAME before
    any recorded stage, because NaN payloads are vendor-shaped
    (IDENTITY_PATHS row 39 measured three payloads for one IEEE answer,
    `0x7fc00000` on Apple, `0x7fffffff` on NVIDIA, `0xffc00000` on AMD) and
    a certified stage may never hold one. Out-of-range ids must be refused
    by name too (contract section 8), never clamped.

    Plants go in at cell `len / 2` and never at cell 0 -- a plant at index 0
    is the one a loop that skips its first element would still catch, which
    makes it the weakest possible plant site.

    **THE CONTROL, WHICH IS THE ONE THAT MATTERS.** If the refusal fired
    UNCONDITIONALLY -- a stray raise, a walk over the wrong buffer, a mask
    that matched every finite value -- then every plant would be "refused by
    name" and the clause would pass for ever WHILE GATING NOTHING. So the
    clause first runs a CLEAN call and requires that nothing raises.

    **AND THEN DEVIATION 1506, WHICH IS WHY THIS CLAUSE RUNS LAST.** Every
    assertion above is about the HOST ORACLE. The DEVICE entry points check
    NOTHING -- read `identical_embedding_forward_into`, which says so in its
    own docstring -- so the profile has no entry point that satisfies
    contract 9.1. This function MEASURES the gap on the safe half (a NaN in
    `W`, planted, read BACK OFF THE DEVICE by bits so reach is measured, run
    with the trace DISABLED so no vendor-shaped payload can enter a card)
    and then raises unless the operator has acknowledged it.

    **IT DELIBERATELY DOES NOT RUN THE OUT-OF-RANGE HALF.** With a negative
    id, `emb_gather_kernel` computes `weight.unsafe_load(v * width + j)`
    with no bounds branch on the normative path, so `v == -1` reads BEFORE
    the buffer. Running an out-of-bounds read to demonstrate that it is out
    of bounds is not a test, it is the bug, and it would be a bug this gate
    committed on purpose on somebody's machine. The host half proves the id
    is refused where a refusal exists; the device half is reported from the
    SOURCE and left unrun."""
    print("clause (f): the row-39 audit, contract section 9.1")

    # ---- THE CONTROL ----------------------------------------------------
    var c = emb_case(emb_case_by_name(String("f_split")))
    var cfg = emb_case_config(c)
    var clean_w = emb_case_weight(c)
    var clean_ids = emb_case_ids(c)
    var clean_dy = emb_case_dy(c)
    var control_raised = False
    try:
        _ = emb_forward_oracle(clean_w, clean_ids, cfg)
        _ = emb_backward_oracle(clean_dy, clean_ids, cfg, List[Float32]())
    except e:
        control_raised = True
    if control_raised:
        raise Error(
            "embedding_check: CLAUSE (f) IS VACUOUS. The refusal fires on"
            " CLEAN inputs, so every plant below would be 'refused' whatever"
            " it held and this clause gates nothing."
        )
    print(
        "clause (f) control: a clean forward and a clean backward do NOT"
        " raise, so the refusal is not unconditional"
    )

    # ---- THE PLANTS, HOST SIDE ------------------------------------------
    var patterns: List[UInt32] = [BITS_QNAN, BITS_POS_INF]
    var pat_names: List[String] = [String("NaN"), String("infinity")]
    var checked = 0
    for pk in range(len(patterns)):
        var v = f32_from_bits(patterns[pk])

        # W, through the forward.
        var w = emb_case_weight(c)
        w[len(w) // 2] = v
        if nonfinite_cells(w) != 1:
            raise Error(
                String("embedding_check: CLAUSE (f) IS VACUOUS for W / ")
                + pat_names[pk]
                + ": the plant did not survive into the host list ("
                + String(nonfinite_cells(w))
                + " non-finite cells, expected exactly 1). Any refusal after"
                + " this fired for another reason"
                + " ([[reached-but-inert]])."
            )
        var raised = False
        var msg = String("")
        try:
            _ = emb_forward_oracle(w, clean_ids, cfg)
        except e:
            raised = True
            msg = String(e)
        if not raised:
            raise Error(
                String("embedding_check: CLAUSE (f) FAILED. A ")
                + pat_names[pk]
                + " in W was NOT refused by emb_forward_oracle."
            )
        if msg.find(String("in W at flat index")) < 0:
            raise Error(
                String("embedding_check: CLAUSE (f) FAILED. A ")
                + pat_names[pk]
                + " in W was refused, but the message does not NAME it: "
                + msg
            )
        checked += 1

        # dY, through the backward.
        var dy = emb_case_dy(c)
        dy[len(dy) // 2] = v
        if nonfinite_cells(dy) != 1:
            raise Error(
                String("embedding_check: CLAUSE (f) IS VACUOUS for dY / ")
                + pat_names[pk]
            )
        var raised2 = False
        var msg2 = String("")
        try:
            _ = emb_backward_oracle(dy, clean_ids, cfg, List[Float32]())
        except e:
            raised2 = True
            msg2 = String(e)
        if not raised2:
            raise Error(
                String("embedding_check: CLAUSE (f) FAILED. A ")
                + pat_names[pk]
                + " in dY was NOT refused by emb_backward_oracle."
            )
        if msg2.find(String("in dY at flat index")) < 0:
            raise Error(
                String("embedding_check: CLAUSE (f) FAILED. A ")
                + pat_names[pk]
                + " in dY was refused, but the message does not NAME it: "
                + msg2
            )
        checked += 1

    # ---- THE OUT-OF-RANGE IDS, HOST SIDE --------------------------------
    var oor: List[String] = [String("f_oor_high"), String("f_oor_neg")]
    for oi in range(len(oor)):
        var oc = emb_case(emb_case_by_name(oor[oi]))
        var ocfg = emb_case_config(oc)
        var oids = emb_case_ids(oc)
        var oraised = False
        var omsg = String("")
        try:
            emb_refuse_ids(oids, ocfg)
        except e:
            oraised = True
            omsg = String(e)
        if not oraised:
            raise Error(
                String("embedding_check: CLAUSE (f) FAILED. ")
                + oor[oi]
                + " carries an out-of-range id and emb_refuse_ids did not"
                + " raise. A clamp turns a data bug into a wrong gradient on"
                + " a REAL vocabulary row and there is no stage at which"
                + " that becomes visible."
            )
        if omsg.find(String("REFUSED")) < 0:
            raise Error(
                String("embedding_check: CLAUSE (f) FAILED. ")
                + oor[oi]
                + "'s refusal does not say REFUSED: "
                + omsg
            )
        checked += 1
    print(
        "clause (f) host: PASS, "
        + String(checked)
        + " refusals, each fired, each NAMING its input"
    )

    # ---- DEVIATION 1506: THE DEVICE HALF --------------------------------
    var w2 = emb_case_weight(c)
    w2[len(w2) // 2] = f32_from_bits(BITS_QNAN)
    var dw_w = _upload_f32(ctx, w2)
    var dw_ids = _upload_i32(ctx, clean_ids)
    var back = _download_f32(ctx, dw_w, len(w2))
    var reached = nonfinite_cells(back)
    if reached != 1:
        raise Error(
            String("embedding_check: the DEVIATION 1506 audit is VACUOUS:")
            + " the planted NaN did not arrive on the device ("
            + String(reached)
            + " non-finite cells read back, expected exactly 1). Whatever"
            + " happens below happens for another reason"
            + " ([[reached-but-inert]])."
        )
    var y_cells = c.n_positions * cfg.width
    var y = _upload_f32(ctx, emb_poison(y_cells))
    var device_raised = False
    try:
        identical_embedding_forward_into(
            ctx, y, dw_w, dw_ids, c.n_positions, cfg
        )
        ctx.synchronize()
    except e:
        device_raised = True
    var out = _download_f32(ctx, y, y_cells)
    var leaked = nonfinite_cells(out)
    _ = y^
    _ = dw_w^
    _ = dw_ids^
    print(
        "DEVIATION 1506 AUDIT: a NaN planted in W at cell "
        + String(len(w2) // 2)
        + " was READ BACK OFF THE DEVICE ("
        + String(reached)
        + " non-finite cell, so reach is MEASURED). The device forward"
        " raised: "
        + String(device_raised)
        + ". Non-finite cells in the device output: "
        + String(leaked)
        + "."
    )
    if device_raised and leaked == 0:
        print(
            "DEVIATION 1506: CLOSED. The device entry point refused. If this"
            " line ever prints, `identical_embedding_forward_into` has grown"
            " a refusal since 2026-08-25 and this whole block should be"
            " deleted along with the deviation."
        )
        return
    var complaint = (
        String("embedding_check: **DEVIATION 1506 IS OPEN.** Contract 9.1")
        + " says a NaN 'is refused by name, refuse_nonfinite ... before any"
        + " recorded stage' and contract section 8 says an out-of-range id"
        + " 'is REFUSED BY NAME ... Not clamped, not wrapped, not silently"
        + " dropped'. Both are TRUE of emb_forward_oracle and"
        + " emb_backward_oracle and FALSE of"
        + " identical_embedding_forward_into and"
        + " identical_embedding_backward_into, which check NOTHING. The"
        + " planted NaN reached the device (measured, 1 cell) and "
        + String(leaked)
        + " non-finite cells came back out of the gather. Nothing in the"
        + " profile forces a caller to run the host refusals first, and for"
        + " a NEGATIVE id the consequence is not a bad number, it is"
        + " emb_gather_kernel computing weight.unsafe_load(v * width + j)"
        + " with v = -1, a read BEFORE the buffer. THE FIX, one function:"
        + " a refusing wrapper in embedding_identical.mojo that calls"
        + " emb_refuse_shape, emb_refuse_ids and refuse_nonfinite and then"
        + " the _into form -- which is the only way the contract's sentence"
        + " becomes true of anything a caller can reach. This gate may not"
        + " edit that file. The out-of-range half was NOT RUN: running an"
        + " out-of-bounds read to demonstrate that it is out of bounds is"
        + " not a test."
    )
    if env_on("MOJOLEARN_EMB_DEVICE_REFUSAL_GAP_ACK"):
        print(complaint)
        print(
            "DEVIATION 1506: ACKNOWLEDGED by"
            " MOJOLEARN_EMB_DEVICE_REFUSAL_GAP_ACK, **NOT FIXED**. The gap"
            " is open and this line is the record that somebody chose to"
            " proceed past it."
        )
        return
    raise Error(complaint)


# ===========================================================================
# CLAUSE (g): THE SABOTAGE ARMS
#
# DEVIATION 1509. The expectation table below is contract 11.1's, DUPLICATED
# into code, and a duplicated table is a table that can drift. The
# alternative was to have the check read the contract, which is a markdown
# file, which would make the gate depend on parsing prose. The duplication
# is accepted and the mitigations are that every row cites section 11.1 and
# that the ONE place this file's table knowingly departs from the contract's
# -- FOLD_READS_LAUNCH's inert set, DEVIATION 1502 -- is argued at length in
# the header.
# ===========================================================================


@fieldwise_init
struct ArmExpectation(Copyable, Movable):
    """What one sabotage arm must do.

    `first_stage` is the tag the arm's OWN CLAUSE writes, and contract
    11.1's discipline is that the arm must move THAT stage "and no earlier
    one". An arm that moves an earlier stage is not aimed where it says it
    is; an arm that moves a later one has been absorbed on the way and its
    clause is being gated by the wrong stage.

    `inert_case` is the half that turns a smoke test into a REACH PROOF, and
    contract 11(g) requires the predicted inert set be "asserted as a mask"
    rather than merely observed to have moved something.

    `must_not_move_dw` is `PAD_ROW_CONTRIBUTES`'s clause and nothing else's:
    that arm must move `emb.counts`, `emb.run_begin` and `emb.perm` and must
    NOT move `emb.dw`, because contract section 8 says the drop-at-source
    and overwrite-afterwards spellings are PROVABLY bit-equal in `dW`. The
    arm is the PROOF of that equivalence, and **the card is the only
    instrument that can see it at all** -- a gate comparing only `emb.dw`
    would call it inert and delete it."""

    var arm: String
    var first_stage: String
    var witness_case: String
    var inert_case: String
    var must_not_move_dw: Bool
    var smoke_only: Bool
    var note: String


def arm_expectation(arm: String) raises -> ArmExpectation:
    if arm == "FOLD_DESCENDING":
        return ArmExpectation(
            arm, String("emb.dw"), String("f_order3"), String("f_dupsame"),
            False, False,
            String(
                "contract 5.1 clause 1, the ascending order. Inert on"
                " f_dupsame because a permutation of a constant sequence is"
                " the same sequence"
            ),
        )
    if arm == "FOLD_BALANCED_TREE":
        return ArmExpectation(
            arm, String("emb.dw"), String("f_tree4"), String("f_order3"),
            False, False,
            String(
                "contract 5.1 clause 2. **PROVABLY inert at every R <= 3**"
                " (contract 5.4: at R = 2 and 3 the tree IS the chain, node"
                " for node), so f_order3 at R = 3 is the sharp inert case"
                " and not merely a quiet one"
            ),
        )
    if arm == "SEED_SEEDLESS":
        return ArmExpectation(
            arm, String("emb.dw"), String("f_negzero1"), String("f_dupsame"),
            False, False,
            String(
                "contract 9.2(c), the departure from gemm v1's SEEDLESS"
                " tree. Inert at every input except a sole -0.0 contributor"
            ),
        )
    if arm == "SINGLE_RUN_BYPASS":
        return ArmExpectation(
            arm, String("emb.dw"), String("f_negzero1"), String("f_dupsame"),
            False, False,
            String(
                "contract 5.5, the seeded R == 1. **The SAME inert mask as"
                " SEED_SEEDLESS and that is exactly why both exist** -- two"
                " wrong spellings agreeing on one input, and a gate carrying"
                " only one would not know which clause it had proved"
            ),
        )
    if arm == "EMPTY_ROW_SKIPPED":
        return ArmExpectation(
            arm, String("emb.dw_seed"), String("f_empty"), String(""),
            False, False,
            String(
                "contract 5.5, the store is required. **NO INERT CASE IS"
                " ASSERTED**: the arm skips the store for EVERY cell, not"
                " only the empty rows, so against a POISONED buffer it moves"
                " everywhere. That is the arm being blunt, not the gate"
                " being weak, and the poison is what makes it visible at all"
            ),
        )
    if arm == "EMPTY_ROW_NEG_ZERO":
        return ArmExpectation(
            arm, String("emb.dw_seed"), String("f_empty"), String("f_nodup"),
            False, False,
            String(
                "contract 5.5, +0.0 and not -0.0. Inert on f_nodup because"
                " that case has NO empty row -- V == T and every id is"
                " distinct -- and a -0.0 seed under a nonempty run is"
                " laundered by the first add"
            ),
        )
    if arm == "FOLD_READS_LAUNCH":
        return ArmExpectation(
            arm, String("emb.dw"), String("f_order3"), String("f_tree4"),
            False, False,
            String(
                "contract 5.1's last paragraph. **DEVIATION 1502: the device"
                " file's 'never inert on a run of length >= 2' is FALSE.**"
                " The arm rotates by `EMB_TPB % span` and EMB_TPB is a power"
                " of two on every column, so it is bitwise INERT at R = 2,"
                " 4, 8, ... -- which is why f_tree4 (R = 4) is the inert"
                " case and f_order3 (R = 3) is the only witness"
            ),
        )
    if arm == "RANK_BY_ARRIVAL":
        return ArmExpectation(
            arm, String("emb.perm"), String("f_multiblock"),
            String("f_nodup"), False, False,
            String(
                "contract 6.3 case 3, **the most dangerous spelling in this"
                " lane** -- an integer atomic's COUNT is order free and its"
                " SLOT is not. DEVIATION 1504: it needs T > 2 * EMB_TPB, not"
                " merely more than one block"
            ),
        )
    if arm == "SORT_TIE_REVERSED":
        return ArmExpectation(
            arm, String("emb.perm"), String("f_order3"), String("f_dupsame"),
            False, False,
            String(
                "contract 6.2(a), the total order. **HALF GATED BY"
                " CONSTRUCTION**: this is the PLAN_SCAN spelling of the arm"
                " and its PLAN_SORT half does not exist, so the sort clause"
                " is half gated and half not gated at all -- contract 11.1"
                " says so and this table repeats it"
            ),
        )
    if arm == "PAD_ROW_CONTRIBUTES":
        return ArmExpectation(
            arm, String("emb.counts"), String("f_pad"), String("f_nodup"),
            True, False,
            String(
                "contract section 8, DEVIATION 1311. It MUST move"
                " emb.counts, emb.run_begin and emb.perm and MUST NOT move"
                " emb.dw -- that is not a weakness, it IS the clause: the"
                " two padding_idx spellings are provably bit-equal in dW,"
                " and the card is the only instrument that can see the"
                " difference at all"
            ),
        )
    if arm == "PAD_ROW_NEG_ZERO":
        return ArmExpectation(
            arm, String("emb.dw"), String("f_pad"), String("f_nodup"),
            False, False,
            String("contract section 8, +0.0 STORED at row padding_idx"),
        )
    if arm == "NO_FLUSH_ACC":
        return ArmExpectation(
            arm, String("emb.dw"), String("f_subacc"), String(""),
            False, True,
            String(
                "seam E3. **INERT ON APPLE ENTIRELY** (contract 9.3): ftz is"
                " bitwise a no-op on an FTZ backend, so the pinned and"
                " unpinned spellings agree there. gemm 4.1's correction is"
                " the standing warning -- Apple's bits not moving is NOT"
                " evidence that a pin is unreached. It becomes a bit-level"
                " reach proof, with no edit, on the first non-flushing"
                " backend"
            ),
        )
    if arm == "GATHER_NO_FLUSH":
        return ArmExpectation(
            arm, String("emb.fwd"), String("f_subw"), String("f_nodup"),
            False, False,
            String(
                "seams G1 and G2, DEVIATION 1310. **NOT inert on Apple** --"
                " a gather does no arithmetic, so a raw copy of a subnormal"
                " survives on every vendor -- which makes this the ONLY"
                " flush arm in the lane a single-column run can see move"
            ),
        )
    if arm == "GATHER_CLAMP_OOR":
        return ArmExpectation(
            arm, String("emb.fwd"), String(""), String(""), False, True,
            String(
                "contract section 8, the refusal. **NOT RUNNABLE.** Its"
                " witness needs an out-of-range id, and the CLEAN half of"
                " that comparison is emb_gather_kernel reading"
                " weight.unsafe_load(v * width + j) out of bounds"
                " (DEVIATION 1506). Running an out-of-bounds read to"
                " demonstrate that it is out of bounds is not a test"
            ),
        )
    if arm == "ACCUM_BY_ADD":
        return ArmExpectation(
            arm, String("emb.dw"), String("f_split"), String(""),
            False, False,
            String(
                "contract 7.4, the CARRY. **NO KERNEL READS THIS SWITCH** --"
                " the wrong spelling lives in the CALLER -- so clause (e)"
                " computes the by-add result ITSELF on every run, armed or"
                " not, and requires it to move on the straddling rows and"
                " NOT to move on the others. The arm is gated whether or not"
                " its -D is set"
            ),
        )
    if arm == "ACCUM_REFILLS":
        return ArmExpectation(
            arm, String("emb.dw_seed"), String("f_accum"), String("f_nodup"),
            False, False,
            String(
                "contract 7.4's price. Inert on a single-microbatch gate,"
                " which is what a lane that never wrote clause (e) has"
            ),
        )
    raise Error(
        String("embedding_check: '")
        + arm
        + "' is not one of the SIXTEEN sabotage names"
        + " embedding_identical.mojo carries. Contract 11.1's table has"
        + " EIGHTEEN rows: EMB_SORT_KEY_ID_ONLY_UNSTABLE is a PLAN_SORT arm"
        + " and PLAN_SORT is not written, and EMB_FOLD_VIA_GEMM_ONEHOT has"
        + " no switch anywhere in the lane (DEVIATION 1505). If a"
        + " seventeenth switch was added, this table and the contract both"
        + " owe it a row."
    )


def find_verdict(
    verdicts: List[CaseVerdict], name: String
) raises -> CaseVerdict:
    for i in range(len(verdicts)):
        if verdicts[i].name == name:
            return verdicts[i].copy()
    raise Error(
        String("embedding_check: the sabotage expectation names case '")
        + name
        + "' and it was not in the clause-(a) set this build ran. The arm"
        + " CANNOT BE EVALUATED, which is not the same as the arm passing"
        + " ([[reached-but-inert]])."
    )


def stage_index_of(tag: String) raises -> Int:
    for i in range(EMB_STAGE_COUNT):
        if emb_stage_tag(i) == tag:
            return i
    raise Error(
        String("embedding_check: '")
        + tag
        + "' is not one of contract section 10's nine tags"
    )


def clause_g(
    arm: String, verdicts: List[CaseVerdict], flush_can_fire: Bool
) raises:
    """The INVERTED verdict of a sabotage build.

    When an arm is armed this file INVERTS: a clean compare is the FAILURE,
    because it means the sabotage was reached and made no difference, or was
    never reached at all. Both are `[[reached-but-inert]]` and both are
    reported as such rather than as a pass.

    **THREE ARMS ARE REPORTED AS SMOKE TESTS BY NAME AND MUST NOT READ AS
    PASSES**, which is the count a reader should carry rather than "sixteen
    arms exist": `NO_FLUSH_ACC` (inert on any FTZ backend),
    `GATHER_CLAMP_OOR` (its clean half is an out-of-bounds read) and
    `EMPTY_ROW_SKIPPED` (no inert case exists)."""
    var exp = arm_expectation(arm)
    print("clause (g): arm " + arm + " -- " + exp.note)

    if arm == "NO_FLUSH_ACC" and not flush_can_fire:
        print(
            "clause (g): "
            + arm
            + " is a **SMOKE TEST ON THIS COLUMN**, not a passing arm. The"
            " backend flushes subnormals in hardware (preflight measured"
            " it), so the pinned and unpinned spellings are BITWISE EQUAL"
            " here and no verdict this build produces is evidence about the"
            " pin. Contract 9.3, and gemm 4.1's correction is the standing"
            " warning. It becomes a real arm on the first non-flushing"
            " column, with no edit to anything."
        )
        return

    if exp.smoke_only and exp.witness_case == "":
        raise Error(
            String("embedding_check: SABOTAGE ")
            + arm
            + " HAS NO RUNNABLE WITNESS and this binary was built with it."
            + " "
            + exp.note
            + " A build carrying an unrunnable arm produces a verdict about"
            + " nothing, and the honest report is that the arm is UNGATED."
        )

    if arm == "ACCUM_BY_ADD":
        print(
            "clause (g): "
            + arm
            + " is evaluated INSIDE CLAUSE (e), because no kernel reads its"
            " switch -- the wrong spelling lives in the caller. Run clause"
            " (e); its by-add control is this arm and it is asserted on"
            " every build."
        )
        return

    var wv = find_verdict(verdicts, exp.witness_case)
    if wv.n_moved == 0:
        raise Error(
            String("embedding_check: SABOTAGE ")
            + arm
            + " IS ARMED AND MOVED NO BIT on its witness case "
            + exp.witness_case
            + ". Either its branch was never reached at this shape or it is"
            + " inert there ([[reached-but-inert]]). It falsifies NOTHING"
            + " and must not be reported as a passing arm."
        )
    var want = stage_index_of(exp.first_stage)
    if wv.first_index != want:
        raise Error(
            String("embedding_check: SABOTAGE ")
            + arm
            + " moved '"
            + wv.first
            + "' FIRST on "
            + exp.witness_case
            + ", and contract 11.1 says its own clause writes '"
            + exp.first_stage
            + "'. Each arm must move the stage its OWN clause writes and no"
            + " earlier one; an earlier stage means the arm is not aimed"
            + " where it says it is."
        )
    if exp.must_not_move_dw and wv.moved_dw != 0:
        raise Error(
            String("embedding_check: SABOTAGE ")
            + arm
            + " moved emb.dw on "
            + String(wv.moved_dw)
            + " cells of "
            + exp.witness_case
            + ", and contract section 8 requires it to move NOTHING there."
            + " The drop-at-source and overwrite-afterwards spellings of"
            + " padding_idx are PROVABLY bit-equal in dW, because a position"
            + " carrying padding_idx can only ever contribute to row"
            + " padding_idx, which emb_pad_row_kernel overwrites. **This arm"
            + " IS that proof**, and a dW that moved means the equivalence"
            + " is false."
        )
    print(
        "clause (g): "
        + arm
        + " BIT on "
        + exp.witness_case
        + ": "
        + String(wv.n_moved)
        + " of 9 stages moved, FIRST at "
        + exp.first_stage
        + ", which is the stage its own clause writes."
    )
    if exp.must_not_move_dw:
        print(
            "clause (g): "
            + arm
            + " left emb.dw UNMOVED, which is the half that proves contract"
            " section 8's two padding_idx spellings are bit-equal. The card"
            " is the only instrument that can see this clause at all."
        )
    if exp.inert_case != "":
        var iv = find_verdict(verdicts, exp.inert_case)
        if iv.n_moved != 0:
            raise Error(
                String("embedding_check: SABOTAGE ")
                + arm
                + " moved "
                + String(iv.n_moved)
                + " stages on "
                + exp.inert_case
                + ", first at "
                + iv.first
                + ", and contract 11.1's PREDICTED INERT MASK requires it to"
                + " move NOTHING there. An arm that moves everywhere is a"
                + " smoke test; the inert case is what makes it a REACH"
                + " PROOF ([[verify-reach-not-output]])."
            )
        print(
            "clause (g): "
            + arm
            + " is INERT on "
            + exp.inert_case
            + ", 9/9 stages unmoved, which is the half that makes it a reach"
            " proof rather than a smoke test."
        )
    else:
        print(
            "clause (g): "
            + arm
            + " has NO ASSERTED INERT CASE. "
            + exp.note
        )


# ===========================================================================


def main() raises:
    var armed = emb_sabotage_name()

    print(
        "=== embedding identity gate, profile"
        " mojolearn.identical.embedding.fp32.v1"
    )
    print(
        "=== embedding_check: clause (a) ran on Apple and AMD, cards md5"
        " c7f824c3, 6887 cells. NO SABOTAGE ARM EXISTS, so this gate has"
        " never been shown capable of failing. Read the header."
    )
    print(
        "mode "
        + mode_name()
        + "   sabotage: "
        + armed
        + "   EMB_TPB: "
        + String(EMB_TPB)
    )

    # ---- THE LEDGER OF WHICH ARM THIS BINARY WAS BUILT WITH -------------
    # DEVIATION 1510, and it is `tools/gemm_ladder.sh:71`'s scar written
    # down. A `-D MOJOLEARN_EMB_SABOTAGE_...` with a typo in it is SILENTLY
    # IGNORED by the compiler: `is_defined` returns False and the build is
    # clean. The operator then sees a green gate and records it as "arm X
    # did not bite", which is the exact inverse of the truth.
    var expect = env_str("MOJOLEARN_EMB_EXPECT_SABOTAGE")
    if expect != "":
        if expect != armed:
            raise Error(
                String("embedding_check: the caller expected sabotage '")
                + expect
                + "' and this BINARY was built with '"
                + armed
                + "'. A misspelled -D is silently ignored by the compiler"
                + " and produces a clean build that a caller reads as 'the"
                + " arm did not bite'. Fix the -D or the expectation."
            )
        print(
            "ledger: the caller expected '"
            + expect
            + "' and the binary agrees, so the -D was not silently dropped"
        )
    elif armed == "none":
        print(
            "ledger: this binary is CLEAN -- no sabotage arm is compiled in."
            " Set MOJOLEARN_EMB_EXPECT_SABOTAGE to have the binary check its"
            " own -D."
        )
    else:
        print(
            "ledger: this binary carries sabotage '"
            + armed
            + "' and the caller did not say so. Set"
            " MOJOLEARN_EMB_EXPECT_SABOTAGE to close the misspelled -D hole."
        )

    var flush_can_fire = preflight()

    var ctx = DeviceContext()

    # ---- CLAUSE (a) ------------------------------------------------------
    # THE CARD CASE IS CHOSEN BY NAME AND IT IS NOT THE FIRST CASE IN THE
    # SET. DEVIATION 1512. `f_nodup` is case 0 and it is a fine clause-(a)
    # case, but the card must carry NINE POPULATED stages and the degenerate
    # cases (`f_t0`, `f_d0`) have empty ones -- so the card case is picked
    # for having every stage nonempty, not for being first. Every other case
    # runs with the trace DISABLED, because `IdentityTrace` enforces tag
    # UNIQUENESS within one trace and nineteen cases would emit `emb.ids`
    # nineteen times and raise. A per-case prefix would be the alternative
    # and it is deliberately not taken: the card a round judge reads must
    # have the nine tags of contract section 10 and nothing else.
    var cases = clause_a_cases()
    print(
        "clause (a): "
        + String(len(cases))
        + " fixture cases, all 9 stages, device vs host oracle, BITWISE"
    )
    var card_case = emb_case_by_name(String("f_split"))
    var cpath = card_path()
    var verdicts = List[CaseVerdict]()
    for ci in range(len(cases)):
        if cases[ci] == card_case:
            var trace = IdentityTrace.to_path(cpath)
            verdicts.append(clause_a_case(ctx, cases[ci], trace, TAG_PREFIX))
        else:
            var off = IdentityTrace.disabled()
            verdicts.append(
                clause_a_case(ctx, cases[ci], off, "case" + String(cases[ci]))
            )

    _ = check_card_tags(cpath)

    var moved_cases = 0
    var first_case = String("")
    var all_cells = 0
    for i in range(len(verdicts)):
        all_cells += verdicts[i].cells
        if verdicts[i].n_moved > 0:
            moved_cases += 1
            if first_case == "":
                first_case = verdicts[i].name + " at " + verdicts[i].first

    comptime if ANY_EMB_SABOTAGE:
        clause_g(armed, verdicts, flush_can_fire)
        print(
            "clauses (b), (c) and (d) are NOT run under a sabotage build:"
            " they are INVARIANCE claims and a deterministic sabotage"
            " satisfies them. Clause (f) is not run either: the refusals are"
            " upstream of every sabotaged seam."
        )
    else:
        if moved_cases != 0:
            var n = verdicts[0].n_moved
            var f = verdicts[0].first
            for i in range(len(verdicts)):
                if verdicts[i].n_moved > 0:
                    n = verdicts[i].n_moved
                    f = verdicts[i].first
                    break
            if mode_is_identical():
                raise Error(
                    String("embedding_check: CLAUSE (a) FAILED, ")
                    + String(n)
                    + " stages differ from the oracle, first at "
                    + f
                    + "  (case "
                    + first_case
                    + ")"
                )
            else:
                # FAST arms of (a) are RECORDED, not asserted, where they
                # are vendor-shaped -- contract section 11, and the metrics
                # lane's leg-11 lesson. Under FAST every `ftz` compiles away
                # and this profile makes NO IDENTITY CLAIM AT ALL, which is
                # the oracle's own disclaimer about itself.
                print(
                    "clause (a) [FAST]: RECORDED, NOT ASSERTED. "
                    + String(moved_cases)
                    + " of "
                    + String(len(verdicts))
                    + " cases differ from the oracle, first at "
                    + first_case
                    + ". FAST is unversioned and makes no identity claim."
                )
        else:
            print(
                "clause (a): PASS, "
                + String(len(verdicts))
                + " cases, 9/9 stages bit-identical to the oracle on all "
                + String(all_cells)
                + " cells, "
                + String(EMB_STAGE_COUNT)
                + "/"
                + String(EMB_STAGE_COUNT)
                + " card tags"
            )
        print(
            "clause (a) NOTE, DEVIATION 1517: at d == 0 and at T == 0 the"
            " device backward returns BEFORE R1, R2 and R3, so emb.counts,"
            " emb.run_begin and emb.perm are never written -- while"
            " emb_backward_stages computes and records all three. Those"
            " three stages are SKIPPED on f_t0 and f_d0 and the skip is"
            " reported here rather than passed over. Contract section 8"
            " declares both shapes LEGAL, so the two cards disagree at three"
            " of nine stages on two legal shapes."
        )

        if env_on("MOJOLEARN_EMB_CHECK_CLAUSE_B"):
            clause_b(ctx, emb_case_by_name(String("f_split")))
        else:
            print("clause (b): SKIPPED (set MOJOLEARN_EMB_CHECK_CLAUSE_B=1)")

        if env_on("MOJOLEARN_EMB_CHECK_CLAUSE_C"):
            clause_c(ctx)
        else:
            print(
                "clause (c): SKIPPED (set MOJOLEARN_EMB_CHECK_CLAUSE_C=1)."
                " BOTH HALVES ARE SKIPPED TOGETHER and that is deliberate:"
                " padding and sequence length are two different fixtures and"
                " a lane that ran one would have half a clause with a whole"
                " clause's name."
            )

        if env_on("MOJOLEARN_EMB_CHECK_CLAUSE_D"):
            clause_d(ctx)
        else:
            print(
                "clause (d): SKIPPED (set MOJOLEARN_EMB_CHECK_CLAUSE_D=1)."
                " NOTE: even when it runs it is a HOST SHADOW -- PLAN_SORT"
                " is not written, so plan invariance, which the contract"
                " calls the strongest evidence that the arithmetic does not"
                " read the plan, has never been run and cannot be."
            )

        if env_on("MOJOLEARN_EMB_CHECK_CLAUSE_E"):
            clause_e(ctx)
        else:
            print(
                "clause (e): SKIPPED (set MOJOLEARN_EMB_CHECK_CLAUSE_E=1)."
                " NOTE: EMB_ACCUM_BY_ADD is falsifiable ONLY here, because"
                " no kernel reads its switch, so a lane that never runs"
                " clause (e) has fifteen arms and not sixteen."
            )

        print(
            "SCOPE: this build, this column, "
            + mode_name()
            + " only. What is NOT closed by anything printed above: **plan"
            " invariance** (PLAN_SORT is not written, contract clause (d)"
            " has never run); **the shipped shape** V=128256 d=4096 T=4096,"
            " which contract 11.2 calls mandatory and which is 2.10 GB of dW"
            " and belongs on a rented GPU rather than this laptop; **the"
            " device-side refusal**, DEVIATION 1506, which is a real defect"
            " in a file this gate may not edit; **EMB_FOLD_VIA_GEMM_ONEHOT**"
            " and **EMB_SORT_KEY_ID_ONLY_UNSTABLE**, two of contract 11.1's"
            " eighteen arms that have no switch anywhere (DEVIATION 1505);"
            " **EMB_NO_FLUSH_ACC on any FTZ column**, where it is inert by"
            " construction; **EMB_GATHER_CLAMP_OOR**, whose clean half is an"
            " out-of-bounds read; **an INDEPENDENT reference** -- there is"
            " no embedding table in cuML, cuVS or RAFT and no PyTorch"
            " checkout, so every clause here is our device against our"
            " oracle and both are ours; **FAST mode**; **the fifteen"
            " sabotage builds this binary is not**; and **every column that"
            " is not this one** -- Apple and AMD agreed bit for bit through"
            " 302 GEMM stages while NVIDIA diverged at"
            " tree001.winners.scores, so two backends agreeing closes"
            " nothing."
        )

        # CLAUSE (f) RUNS LAST, ON PURPOSE. DEVIATION 1506 makes it raise on
        # a defect this gate cannot fix, and a clause that raises should not
        # be the reason the six clauses above went unmeasured.
        if env_on("MOJOLEARN_EMB_CHECK_CLAUSE_F"):
            clause_f(ctx)
        else:
            print(
                "clause (f): SKIPPED (set MOJOLEARN_EMB_CHECK_CLAUSE_F=1)."
                " NOTE: it runs LAST when it runs, because DEVIATION 1506"
                " makes it RAISE on a defect in a file this gate may not"
                " edit -- the device entry points perform NO refusal at all"
                " -- and a raise there must not cost the other clauses their"
                " measurements."
            )


# ===========================================================================
# OWED, AND WHY I DID NOT DO IT HERE
#
# This file and `embedding_fixture.mojo` are the only two this lane's gate
# agent was permitted to write. Everything below is a belief about ANOTHER
# file, recorded rather than acted on, so that the next agent inherits the
# finding instead of rediscovering it. **None of it has been verified by
# running anything.**
#
# 1. **A REFUSING ENTRY POINT, DEVIATION 1506.** The one-line shape:
#
#        def identical_embedding_forward(ctx, out_y, weight, ids, w_host,
#                                        ids_host, n_positions, cfg) raises:
#            emb_refuse_shape(cfg, n_positions)
#            emb_refuse_ids(ids_host, cfg)
#            refuse_nonfinite(String("W"), w_host)
#            identical_embedding_forward_into(...)
#
#    and its backward twin over `dY`. It belongs in
#    `embedding_identical.mojo` beside the `_into` forms. Until it exists,
#    contract 9.1's "refused by name before any recorded stage" is true of
#    the oracle and false of everything a caller can reach on a device, and
#    a negative id is an out-of-bounds read in `emb_gather_kernel`.
#
# 2. **`EMB_FOLD_READS_LAUNCH`'s DOCSTRING, DEVIATION 1502.** The sentence
#    "**Never inert on a run of length >= 2**" is false and should read
#    "inert at every run length that DIVIDES the block size, which is every
#    power of two up to EMB_TPB; the witness must be a run whose length is
#    coprime to it". `[[fix-docs-on-discovery]]` binds whoever touches that
#    file: delete the false sentence, do not soften it.
#
# 3. **`EMB_RANK_BY_ARRIVAL`'s INERT SET, DEVIATION 1504.** Contract 11.1
#    says "a single-block launch, where arrival order IS position order".
#    More than one block is not enough -- the phase split reproduces
#    ascending order for every `T <= 2 * EMB_TPB`. The row should say
#    `T <= 2 * EMB_TPB`.
#
# 4. **CONTRACT 11.1's TABLE HAS EIGHTEEN ROWS AND THE LANE HAS SIXTEEN
#    SWITCHES, DEVIATION 1505.** `EMB_FOLD_VIA_GEMM_ONEHOT` has no switch
#    anywhere, and contract 5.2(d)'s claim that the one-hot routing
#    "survives as a SABOTAGE whose job is to PRINT the difference between
#    routing and pinning" is therefore false today. Either write the arm or
#    mark the row unbuilt, as the contract already does for
#    `EMB_SORT_KEY_ID_ONLY_UNSTABLE`.
#
# 5. **THE DEGENERATE-SHAPE STAGE GAP, DEVIATION 1517.** At `d == 0` and
#    `T == 0` the device never writes `counts`, `run_begin` or `perm` and
#    `emb_backward_stages` records all three. One of the two should change:
#    either the device should run R1-R3 on those shapes (cheap -- they are
#    `V` threads over zero positions) or `emb_backward_stages` should record
#    them empty. This gate compares neither and says so.
#
# 6. **A RUNNER FOR THE SIXTEEN SABOTAGE BUILDS.** `is_defined` is a
#    compile-time query, so exercising the set is sixteen compiles.
#    `tools/gemm_ladder.sh` is the pattern and its line 71 is the scar
#    DEVIATION 1510 answers. What is owed is a script that, per arm, builds
#    with the arm's `-D`, exports `MOJOLEARN_EMB_EXPECT_SABOTAGE=<arm>` and
#    `MOJOLEARN_IDENTITY_TRACE=<out>/<arm>.card`, runs this file and
#    requires a NONZERO exit. `ACCUM_BY_ADD` additionally needs
#    `MOJOLEARN_EMB_CHECK_CLAUSE_E=1`. A shell file under `tools/` is
#    outside this agent's remit.
#
# 7. **`IDENTITY_PATHS.md`, `SUPPORT_MATRIX.md`, `archive/plans/CARD_GAPS.md` and
#    `archive/plans/UNWIRED.md`** all enumerate lanes and none mentions `embedding/`. Four
#    entries owed, all in files this lane may not edit. `archive/plans/UNWIRED.md` in
#    particular is where "specified, never compiled" belongs, and that is
#    now the state of five files rather than three.
#
# 8. **`tools/e1_bootstrap.sh` PHASE 8.** This lane's card is not wired into
#    the leg's judge. What phase 8 needs is one entry that sets
#    `MOJOLEARN_IDENTITY_TRACE` to `<out>/lanes/embedding.identical.card`
#    and runs this file. DEVIATION 1501 is why that will now work and
#    DEVIATION 970 is why it would not have.
#
# 9. **`embedding/corpus/`.** The only thing that could catch our oracle
#    being wrong in the same way as our device. It does not exist and there
#    is nothing to build it from: cuML, cuVS and RAFT have no embedding
#    table and there is no PyTorch checkout. Until something does, every
#    green line this file prints is two halves of one lane agreeing with
#    each other.
# ===========================================================================
