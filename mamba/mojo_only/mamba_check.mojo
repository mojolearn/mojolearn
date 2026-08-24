"""The gate file of profile `mojolearn.identical.mamba1.fp32.v1`, the path
`mamba/IDENTICAL_MAMBA_CONTRACT.md` section 8 names.

NOT A PORT. It runs the device block
(`mamba/ported/transformers/models/mamba/modeling_mamba.mojo`) against the
host oracle (`mamba/mojo_only/mamba_oracle.mojo`) and compares every recorded
stage BY BITS.

**THIS GATE IS DELIBERATELY TINY AND IT IS NOT THE CONTRACT'S GATE YET.**
The DEFAULT is one shape, B = 1, L = 4, d_model = 8, and one launch. The
shape is read from the environment (`MOJOLEARN_MAMBA_CHECK_B`, `_L`, `_DM`)
so that closing a clause that demands a bigger fold costs ONE build and not
one build per shape; nothing here sweeps. Andrew's machine was
crashed on 2026-08-23 by seven agents compiling Mojo at once, and this lane
was cut to a single compile at a time at the smallest shape that exists. What
this file gates is clause (a) of section 8 at one point, plus clause (f), the
falsifiability of the seams this lane owns. Everything else is OWED and is
listed at the foot of this docstring so that no reader mistakes a green line
for the contract's claim.

WHAT IT CHECKS
---------------
1. Every computed stage of contract section 7, device vs oracle, BITWISE, and
   on a difference it names the stage, the number of differing cells and the
   first one in both hex bit patterns. "It failed" is not a finding; "conv.out
   moved on 48 of 64 cells starting at cell 3" is.
2. The CARD ITSELF: the seventeen tags of section 7, in the section's order,
   emitted once each. That is a check on the composition (this file's block
   emits fourteen and `selective_scan_fn` emits `scan.y`, `scan.h` and
   `skip.out`), and the trace's tag-uniqueness invariant is what would catch
   the two sides both claiming one.
3. REACH, MEASURED, NOT ASSUMED. Section 4's softplus note says the `x <= 20`
   guard cannot move a bit at the boundary itself and that the distinguishing
   range is delta in about [8, 14]. A fixture that does not enter that band
   passes the `S14_THRESHOLD_10` sabotage VACUOUSLY, which is worse than not
   running it (`[[reached-but-inert]]`). So the fixture PLANTS the band and
   this file COUNTS the cells that land in it, and refuses to report the arm
   as falsifying anything if the count is zero.

THE FIXTURE, AND WHY IT IS NOT THE CORPUS'S DEFAULT
-----------------------------------------------------
`corpus_weights` at the corpus's own ranges, with ONE plant: `dt_proj.bias`
for channels `d % 4 == 3` is drawn from [8, 14] instead of [-7, -2], through
the SAME hashed generator (`corpus_tensor`), so no value here is hand-picked.
That is contract section 4's fixture F7, and it is the only way the softplus
guard's distinguishing range is reachable at all. The other twelve channels
keep the corpus range, so one launch exercises both the ordinary path and the
guard's band. The oracle is handed the identical weights, so the bitwise
comparison is unaffected by the plant: the oracle is the authority for
whatever weights it is given.

RUNNING IT
-----------
    tools/with_identical_mode.sh pixi run mojo run -I . \
        mamba/mojo_only/mamba_check.mojo

and one sabotage arm at a time, each of which MUST fail:

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
        -D MOJOLEARN_MAMBA_SABOTAGE_S14_THRESHOLD_10=1 \
        -I . mamba/mojo_only/mamba_check.mojo

When an arm is armed this file INVERTS its verdict: a clean compare is then
the FAILURE, because it means the sabotage was reached and made no
difference, or was never reached at all. Both are `[[reached-but-inert]]` and
both are reported as such.

THE SABOTAGE LEDGER, MEASURED 2026-08-23 (Apple, IDENTICAL, this shape)
------------------------------------------------------------------------
All six arms of `modeling_mamba.mojo` were run one at a time, each its own
build. Every one BIT. What matters is not that they failed but WHERE, so the
first stage to move and its cell count are recorded here:

    arm                    first stage moved   cells    stages moved
    S14_THRESHOLD_10       softplus.out        13 / 64      4 / 16
    S13_BIAS_LAST          conv.out            33 / 64     10 / 16
    S13_TAPS_REVERSED      conv.out            24 / 64      9 / 16
    S1_FOLD_DESCENDING     norm.sumsq           1 / 4       1 / 16
    S12_MUL_SIGMOID        gate.out            15 / 64      3 / 16
    S17_OP_NUMBERING       in_proj.out        128 / 128    13 / 16

Each arm moves the stage its own seam writes and no earlier one, which is
the evidence that it is reached where it is aimed and nowhere else.

THREE THINGS THE LEDGER SAYS THAT A PASS OR FAIL WOULD NOT
------------------------------------------------------------
1. **The card is what makes S1 falsifiable at all.** `S1_FOLD_DESCENDING`
   moves `norm.sumsq` on one row and moves NOTHING ELSE: `norm.out` is
   already identical again, because a 1 ulp change in the sum of squares is
   absorbed by the divide and the reciprocal square root. A gate that
   compared only the block OUTPUT would call that arm inert and would then
   have licensed any fold order at S1. The per-row `norm.sumsq` stage is the
   only reason contract section 4's S1 clause is falsifiable here.
2. **Absorption is everywhere and it is not a bug.** `S14_THRESHOLD_10`
   moves 13 cells of `softplus.out` and 13 of `scan.y` but only ONE cell of
   `skip.out` and no cell of `gate.out`, `out_proj.out` or `residual.out`:
   `y + u * D` adds a tiny y to a much larger `u * D` and rounds the
   difference away. `S13_TAPS_REVERSED` reaches `residual.out` on zero
   cells. So four of the six arms are invisible in the block's output at
   this shape and are caught only by the intermediate stages.
3. **The orientation trap is exactly as advertised.**
   `S17_OP_NUMBERING` moves 13 of 16 stages, `in_proj.out` on ALL 128 cells,
   and NOTHING RAISES: no bound is exceeded and no size check fires, because
   every operand read as its transpose has the same element count. That is
   the whole reference card of plausible wrong products, reproduced on
   demand.

THE THIN CLAUSE, MEASURED AT d_model = 16 AND NO LONGER THIN
--------------------------------------------------------------
`S1_FOLD_DESCENDING` bit on ONE of four rows at `d_model = 8` and moved
`norm.sumsq` and NOTHING ELSE. That was reported as a weak falsification with
`d_model = 16` owed. It has now been run there, B = 1, L = 4, d_model = 16:

    d_model    rows moved    stages moved    reaches
    8          1 of 4        1 of 16         norm.sumsq only
    16         3 of 4        13 of 16        through to out_proj.out

Eight terms was simply too short a fold for most rows to round differently in
the two directions. At sixteen the arm propagates the whole way down the
block. The S1 clause is WELL GATED at d_model = 16 and only marginally gated
at 8, and the honest reading is that the fold-order clause needs the wider
row to be tested at all.

**AND THE BLOCK'S OUTPUT STILL DOES NOT MOVE.** At d_model = 16 thirteen of
sixteen stages differ, `out_proj.out` among them on 23 of 64 cells, and
`residual.out` is STILL bit-identical: `residual + out_proj` adds an
out_proj of order 1e-3 to an input of order 1, and the difference rounds
away. So even at the shape where this arm is strongest, a gate that compared
only the block's output would report it inert. That is the sharpest form of
the absorption finding in this file and it is an argument about the card's
design rather than about this block.

`S14_THRESHOLD_10` is reported only because the fixture plants the band and
the reach count above is nonzero; against the corpus default ranges it would
be VACUOUS.

OWED, and this file does not cover any of it
----------------------------------------------
L = 16, 64 and 257; FAST mode; clause (e), the row-39 planted NaN and
infinity audit; the corpus cross-check against `mamba/corpus/`; and every
column that is not this Apple box.

CLAUSES (b) AND (c) ARE CHECKED HERE, and each says what would have hidden a
failure:

* **(b), eight repeated launches.** The block is run eight times in one
  process, each run its own fresh state, stage buffers and kernel dispatches,
  and runs 2 through 8 are compared to run 1 on every cell of every stage.
  What would hide a failure: comparing only the final `residual.out`, since
  four of the six sabotage arms already showed that stage absorbing an
  upstream difference entirely. The comparison is therefore per stage.
* **(c), batch composition.** The same sequence is run alone (B = 1), beside
  one other (B = 2) and beside two others (B = 3), from ONE `x` that is
  sliced, so row 0's input bits are identical by construction. Row 0's cells
  are then compared across all three compositions, and row 1's between B = 2
  and B = 3. PASS on 4,152 compared cells of all 16 stages at B up to 3,
  L = 4, d_model = 8, and on 7,896 cells at d_model = 16. What would hide a failure: comparing whole buffers
  rather than ROW SLICES, because the buffers are three different lengths
  and a whole-buffer compare cannot even be spelled; and forgetting the
  per-batch stages (`conv.window`, `scan.h`), which are indexed `[B, ...]`
  rather than by token and are exactly where a batch-dependent bug would
  live.

  **AND IT CARRIES A NEGATIVE CONTROL, because without one it is worthless.**
  If `row_slice` were wrong -- if it returned row 0 whatever row it was asked
  for -- every comparison would compare a row to ITSELF and pass for ever, on
  every vendor, hiding any batch dependence there is. So the clause first
  proves the slicer can tell two rows apart: rows 0 and 1 of the B = 2 run
  have different input tokens and must differ somewhere. They differ on 15 of
  15 batched stages. A zero there RAISES and calls the clause vacuous rather
  than passing it (`[[verify-reach-not-output]]`).

* **(d), decode == prefill AT THE COMPOSITION POINT.** Gate D is green for
  `mamba_simple.mojo` in isolation; this is the same claim for
  `mamba_block_forward`, which is a different thing and is what the card is
  made of. A length-L sequence is run once as a prefill and then one token at
  a time through the SAME entry point with the state carried, and every token
  of every stage is compared. PASS on 2,152 cells at d_model = 8 and 4,168 at
  d_model = 16. This is the clause DEVIATION 721's bias seed exists to make
  true BY CONSTRUCTION, and it is now measured rather than argued.

  **AND ITS CONTROL, because this is the clause most exposed to a blind
  gate.** If the decode path and the prefill path shared a buffer or a cached
  card, the comparison would be a value against ITSELF and would pass for
  ever on every vendor. So the clause first compares decode step `t` against
  prefill token `t + 1`, a deliberate MISALIGNMENT that must differ: it
  differs on 39 stage comparisons. A zero there raises VACUOUS, not FAILED.
  The per-batch stages (`conv.window`, `scan.h`) are the state AFTER the
  call, so they are comparable only at the last token, and that is the only
  token at which they are compared.

Clause (c) is a device-versus-device invariance claim. It does not re-check
correctness: a block that was wrong in a batch-INDEPENDENT way would pass it.
Clause (a), the oracle compare above, is what covers that. The same is true
of clause (d).
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace, read_trace_lines
from mamba.mojo_only.mamba_fixture import (
    D_CONV,
    D_STATE,
    TID_B_DT,
    MambaDims,
    MambaWeights,
    corpus_case_seed,
    corpus_tensor,
    corpus_weights,
    corpus_x,
    mode_name,
)
from mamba.mojo_only.mamba_oracle import MambaState, MambaStages, mamba_block_oracle
from mamba.ported.transformers.models.mamba.modeling_mamba import (
    BLOCK_ANY_SABOTAGE,
    MambaDeviceStages,
    MambaDeviceState,
    MambaDeviceWeights,
    mamba_block_forward,
    mamba_block_sabotage_name,
    mamba_download,
    mamba_upload,
)


comptime TRACE_PATH = "/tmp/mojolearn_mamba_block_tiny.trace"
comptime TAG_PREFIX = "tiny"

#: The number of repeated launches contract section 8 clause (b) names.
comptime CLAUSE_B_LAUNCHES = 8

#: Stage layout kinds. TOKEN: `[M, W]`, `M = B * L` rows. BATCH: `[B, W]`,
#: one block per sequence. GLOBAL: one buffer for the whole call.
comptime KIND_TOKEN = 0
comptime KIND_BATCH = 1
comptime KIND_GLOBAL = 2


def env_int(name: String, dflt: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return dflt
    var v = 0
    for i in range(s.byte_length()):
        var c = ord(String(s[byte=i]))
        if c < 48 or c > 57:
            raise Error(
                String("mamba_check: ") + name + " is not a number: '" + s + "'"
            )
        v = v * 10 + (c - 48)
    return v


def env_on(name: String) raises -> Bool:
    return String(getenv(name)) != ""


def hexbits(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


# ===========================================================================
# THE STAGE TABLE. Card order, contract section 7, minus `input.x` (which is
# the input and not computed). Widths and kinds are what makes a ROW SLICE
# expressible, which is the whole of clause (c).
# ===========================================================================


def stage_names() -> List[String]:
    var n: List[String] = [
        String("norm.sumsq"),
        String("norm.out"),
        String("in_proj.out"),
        String("A.out"),
        String("conv.out"),
        String("silu.out"),
        String("conv.window"),
        String("x_proj.out"),
        String("dt_proj.out"),
        String("softplus.out"),
        String("scan.y"),
        String("scan.h"),
        String("skip.out"),
        String("gate.out"),
        String("out_proj.out"),
        String("residual.out"),
    ]
    return n^


def stage_kind(i: Int) -> Int:
    if i == 3:
        return KIND_GLOBAL  # A.out, [d_inner, D_STATE], no batch axis
    if i == 6 or i == 11:
        return KIND_BATCH  # conv.window, scan.h: [B, ...]
    return KIND_TOKEN


def stage_width(i: Int, dims: MambaDims) -> Int:
    var dm = dims.d_model
    var di = dims.d_inner
    if i == 0:
        return 1
    if i == 1 or i == 14 or i == 15:
        return dm
    if i == 2:
        return 2 * di
    if i == 3:
        return di * D_STATE
    if i == 6:
        return di * D_CONV
    if i == 7:
        return dims.x_proj_rows()
    if i == 11:
        return di * D_STATE
    return di


def row_slice(
    values: List[Float32], i: Int, bb: Int, l: Int, dims: MambaDims
) -> List[Float32]:
    """The cells of stage `i` belonging to batch row `bb`. THE POINT OF
    CLAUSE (c): a row's bits must not depend on who else shares the launch,
    and the only way to say that is to cut the row out of buffers of three
    different lengths."""
    var w = stage_width(i, dims)
    var kind = stage_kind(i)
    var start: Int
    var count: Int
    if kind == KIND_TOKEN:
        start = bb * l * w
        count = l * w
    elif kind == KIND_BATCH:
        start = bb * w
        count = w
    else:
        start = 0
        count = w
    var out = List[Float32]()
    for j in range(count):
        out.append(values[start + j])
    return out^


def token_slice(
    values: List[Float32], i: Int, t: Int, dims: MambaDims
) -> List[Float32]:
    """The cells of TOKEN-kind stage `i` belonging to token `t` of a B = 1
    call. Clause (d) needs this because a decode step's whole buffer is ONE
    token and a prefill's is L of them: the comparison is only expressible
    per token."""
    var w = stage_width(i, dims)
    var out = List[Float32]()
    for j in range(w):
        out.append(values[t * w + j])
    return out^


def oracle_dump(st: MambaStages) -> List[List[Float32]]:
    """The host oracle's stages, card order, matching `stage_names()`."""
    var out = List[List[Float32]]()
    out.append(st.norm_sumsq.copy())
    out.append(st.norm_out.copy())
    out.append(st.in_proj.copy())
    out.append(st.a_out.copy())
    out.append(st.conv_out.copy())
    out.append(st.silu_out.copy())
    out.append(st.conv_win.copy())
    out.append(st.x_proj.copy())
    out.append(st.dt_proj.copy())
    out.append(st.softplus_out.copy())
    out.append(st.scan_y.copy())
    out.append(st.scan_h.copy())
    out.append(st.skip_out.copy())
    out.append(st.gate_out.copy())
    out.append(st.out_proj.copy())
    out.append(st.residual_out.copy())
    return out^


def device_dump(
    ctx: DeviceContext, mut d: MambaDeviceStages, b: Int, l: Int, dims: MambaDims
) raises -> List[List[Float32]]:
    """Every stage buffer back on the host, card order. Downloading rather
    than comparing on device keeps every comparison in one place and makes a
    row slice a list slice."""
    var dm = dims.d_model
    var di = dims.d_inner
    var xr = dims.x_proj_rows()
    var m = b * l
    var out = List[List[Float32]]()
    out.append(mamba_download(ctx, d.norm_sumsq, m))
    out.append(mamba_download(ctx, d.norm_out, m * dm))
    out.append(mamba_download(ctx, d.in_proj, m * 2 * di))
    out.append(mamba_download(ctx, d.a_out, di * D_STATE))
    out.append(mamba_download(ctx, d.conv_out, m * di))
    out.append(mamba_download(ctx, d.silu_out, m * di))
    out.append(mamba_download(ctx, d.conv_win, b * di * D_CONV))
    out.append(mamba_download(ctx, d.x_proj, m * xr))
    out.append(mamba_download(ctx, d.dt_proj, m * di))
    out.append(mamba_download(ctx, d.softplus_out, m * di))
    out.append(mamba_download(ctx, d.scan_y, m * di))
    out.append(mamba_download(ctx, d.scan_h, b * di * D_STATE))
    out.append(mamba_download(ctx, d.skip_out, m * di))
    out.append(mamba_download(ctx, d.gate_out, m * di))
    out.append(mamba_download(ctx, d.out_proj, m * dm))
    out.append(mamba_download(ctx, d.residual_out, m * dm))
    return out^


def run_block(
    ctx: DeviceContext,
    w: MambaWeights,
    x: List[Float32],
    b: Int,
    l: Int,
    dims: MambaDims,
    mut trace: IdentityTrace,
    prefix: String,
) raises -> List[List[Float32]]:
    """One whole block call on the device, stages returned on the host.

    `[[mojo-buffer-freed-at-last-use]]`: the weights, the state and the input
    are locals here and every one of them is still alive when
    `mamba_block_forward` returns, because that function synchronizes before
    it does."""
    var dw = MambaDeviceWeights(ctx, w)
    var dstate = MambaDeviceState(ctx, b, dims)
    var dstages = MambaDeviceStages(ctx, b, l, dims)
    var dx = mamba_upload(ctx, x)
    mamba_block_forward(ctx, dstages, dstate, dw, dx, b, l, trace, prefix)
    var out = device_dump(ctx, dstages, b, l, dims)
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    return out^


def run_step(
    ctx: DeviceContext,
    mut dw: MambaDeviceWeights,
    mut dstate: MambaDeviceState,
    x: List[Float32],
    b: Int,
    l: Int,
    dims: MambaDims,
    mut trace: IdentityTrace,
    prefix: String,
) raises -> List[List[Float32]]:
    """One block call against a CALLER-OWNED state, so the caller can carry
    it from one call to the next. That is the whole of the decode path:
    contract section 5 says the decode step is this same function at `l == 1`
    with the conv window and the SSM state carried."""
    var dstages = MambaDeviceStages(ctx, b, l, dims)
    var dx = mamba_upload(ctx, x)
    mamba_block_forward(ctx, dstages, dstate, dw, dx, b, l, trace, prefix)
    var out = device_dump(ctx, dstages, b, l, dims)
    _ = dstages^
    _ = dx^
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


def compare_stage(
    name: String, host: List[Float32], dev: List[Float32], loud: Bool
) raises -> StageDiff:
    """Bitwise, cell by cell. A LENGTH mismatch is reported as such rather
    than compared to the shorter of the two, because a stage that is the
    wrong size is a different defect from a stage that is the wrong value.

    `loud` prints per stage. It is False wherever a difference is EXPECTED
    (clause (c)'s negative control) or wherever the caller reports the
    failure itself with more context (clauses (b) and (c)), so that the only
    lines on stdout are lines a reader should act on."""
    if len(host) != len(dev):
        raise Error(
            String("mamba_check: stage ")
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
        if bitcast[DType.uint32](host[i]) != bitcast[DType.uint32](dev[i]):
            n_diff += 1
            if first < 0:
                first = i
    if n_diff == 0:
        if loud:
            print("  OK    " + name + "  (" + String(len(host)) + " cells)")
    elif loud:
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
    return StageDiff(name, len(host), n_diff, first)


def compare_dumps(
    a: List[List[Float32]], b: List[List[Float32]], loud: Bool
) raises -> List[StageDiff]:
    var names = stage_names()
    var out = List[StageDiff]()
    for i in range(len(names)):
        out.append(compare_stage(names[i], a[i], b[i], loud))
    return out^


def count_moved(diffs: List[StageDiff]) -> Int:
    var n = 0
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            n += 1
    return n


def first_moved(diffs: List[StageDiff]) -> String:
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


# ===========================================================================
# THE CARD
# ===========================================================================


def contract_section_7_tags() -> List[String]:
    """The card, verbatim from contract section 7, in the section's order.
    `selective_scan_fn` emits `scan.y`, `scan.h` and `skip.out`; the block
    emits the other fourteen."""
    var want: List[String] = [
        String("input.x"),
        String("norm.sumsq"),
        String("norm.out"),
        String("in_proj.out"),
        String("A.out"),
        String("conv.out"),
        String("silu.out"),
        String("conv.window"),
        String("x_proj.out"),
        String("dt_proj.out"),
        String("softplus.out"),
        String("scan.y"),
        String("scan.h"),
        String("skip.out"),
        String("gate.out"),
        String("out_proj.out"),
        String("residual.out"),
    ]
    return want^


def check_card_tags(path: String) raises -> Int:
    var want = contract_section_7_tags()
    var lines = read_trace_lines(path)
    print(
        "card: "
        + String(len(lines))
        + " records, contract section 7 wants "
        + String(len(want))
    )
    if len(lines) != len(want):
        raise Error(
            String("mamba_check: the card has ")
            + String(len(lines))
            + " records and contract section 7 lists "
            + String(len(want))
        )
    for i in range(len(lines)):
        var fields = lines[i].split("\t")
        if len(fields) < 2:
            raise Error(
                String("mamba_check: malformed trace record: ") + lines[i]
            )
        var got = String(fields[1])
        var expect = String(TAG_PREFIX) + "." + want[i]
        if got != expect:
            raise Error(
                String("mamba_check: card record ")
                + String(i)
                + " is '"
                + got
                + "', contract section 7 wants '"
                + expect
                + "'"
            )
    print("card: 17/17 tags in contract section 7's order, all unique")
    return len(lines)


# ===========================================================================
# THE FIXTURE
# ===========================================================================


def planted_weights(dims: MambaDims) raises -> MambaWeights:
    """The corpus's default ranges with contract section 4's fixture F7
    planted: `dt_proj.bias` for channels `d % 4 == 3` drawn from [8, 14],
    through the SAME hashed generator, so the softplus guard's
    distinguishing range is REACHED. Every other tensor is the corpus's."""
    var seed = corpus_case_seed(1)
    var w = corpus_weights(seed, dims)
    var high = corpus_tensor(seed, TID_B_DT, dims.d_inner, 8.0, 14.0)
    for d in range(dims.d_inner):
        if d % 4 == 3:
            w.b_dt[d] = high[d]
    return w^


def softplus_band_reach(
    w: MambaWeights, st: MambaStages, m: Int, di: Int
) -> Int:
    """How many cells of S14's input land in (10, 20], the ONLY band in
    which the `x <= 20` guard and a guard moved to 10 can differ. Section 4
    is explicit that the boundary at 20 itself cannot move a bit in FP32.
    Zero here means `S14_THRESHOLD_10` falsifies NOTHING, whatever the gate
    prints."""
    var n = 0
    for t in range(m):
        for d in range(di):
            var biased = st.dt_proj[t * di + d] + w.b_dt[d]
            if biased > Float32(10.0) and biased <= Float32(20.0):
                n += 1
    return n


# ===========================================================================
# CLAUSE (b): the same bits on every one of eight repeated launches
# ===========================================================================


def clause_b(
    ctx: DeviceContext,
    w: MambaWeights,
    x: List[Float32],
    b: Int,
    l: Int,
    dims: MambaDims,
    base: List[List[Float32]],
) raises:
    """Contract section 8 clause (b). Eight launches, each with its OWN
    fresh state, stage buffers and kernel dispatches, every stage compared to
    the first on every cell.

    Comparing only `residual.out` would hide a failure: the sabotage ledger
    in this file's header measured four of six arms being absorbed before
    that stage. So every stage is compared."""
    print(
        "clause (b): "
        + String(CLAUSE_B_LAUNCHES)
        + " repeated launches, every stage, every cell"
    )
    var total_cells = 0
    for i in range(len(base)):
        total_cells += len(base[i])
    for run in range(2, CLAUSE_B_LAUNCHES + 1):
        var off = IdentityTrace.disabled()
        var got = run_block(ctx, w, x, b, l, dims, off, "repeat")
        var diffs = compare_dumps(base, got, False)
        var moved = count_moved(diffs)
        if moved != 0:
            raise Error(
                String("mamba_check: CLAUSE (b) FAILED, launch ")
                + String(run)
                + " differs from launch 1 at "
                + first_moved(diffs)
            )
    print(
        "clause (b): PASS, launches 2.."
        + String(CLAUSE_B_LAUNCHES)
        + " bit-identical to launch 1 on all "
        + String(total_cells)
        + " cells of all 16 stages"
    )


# ===========================================================================
# CLAUSE (c): batch composition invariance
# ===========================================================================


def clause_c(
    ctx: DeviceContext, w: MambaWeights, l: Int, dims: MambaDims
) raises:
    """Contract section 8 clause (c). A row's bits are identical whether its
    sequence shares the launch with 0, 1 or 2 others.

    The contract notes this is the clause vLLM's batch-invariant mode cannot
    give Mamba (`supports_batch_invariance()` is False for its Mamba
    backends), so it is the most interesting one in the section.

    ONE `x` is generated at B = 3 and SLICED, so row 0's input bits are
    identical across the three compositions by construction rather than by
    coincidence. Then row 0 is compared across all three and row 1 between
    B = 2 and B = 3. Whole-buffer comparison cannot even be spelled here (the
    buffers are three different lengths), which is exactly why the per-batch
    stages `conv.window` and `scan.h` are the ones to watch: they are indexed
    `[B, ...]` and are where a batch-dependent bug would live."""
    var dm = dims.d_model
    var x3 = corpus_x(corpus_case_seed(1), 3, l, dm)
    var row_len = l * dm
    var x1 = List[Float32]()
    for i in range(row_len):
        x1.append(x3[i])
    var x2 = List[Float32]()
    for i in range(2 * row_len):
        x2.append(x3[i])

    print("clause (c): batch composition, the same row at B=1, B=2 and B=3")
    var off1 = IdentityTrace.disabled()
    var d1 = run_block(ctx, w, x1, 1, l, dims, off1, "b1")
    var off2 = IdentityTrace.disabled()
    var d2 = run_block(ctx, w, x2, 2, l, dims, off2, "b2")
    var off3 = IdentityTrace.disabled()
    var d3 = run_block(ctx, w, x3, 3, l, dims, off3, "b3")

    var names = stage_names()

    # THE NEGATIVE CONTROL, and clause (c) is worthless without it. If
    # `row_slice` were wrong -- if it returned row 0 whatever `bb` it was
    # given -- every comparison below would compare a row to ITSELF and pass
    # on every cell, for ever, on every vendor. So first prove the slicer
    # can tell two rows apart: rows 0 and 1 of the B = 2 run have DIFFERENT
    # input tokens, so they must differ somewhere. `[[verify-reach-not-output]]`.
    var control = 0
    for i in range(len(names)):
        var a0 = row_slice(d2[i], i, 0, l, dims)
        var a1 = row_slice(d2[i], i, 1, l, dims)
        if stage_kind(i) == KIND_GLOBAL:
            continue  # A.out has no batch axis; equal by construction
        var d = compare_stage(names[i] + " CONTROL row0 vs row1", a0, a1, False)
        if d.n_diff > 0:
            control += 1
    if control == 0:
        raise Error(
            "mamba_check: CLAUSE (c) IS VACUOUS. Rows 0 and 1 of the B=2 run"
            " are bit-identical on every stage, which cannot be true of two"
            " different input sequences. `row_slice` is not cutting distinct"
            " rows, so every comparison below is a row against itself"
            " ([[reached-but-inert]])."
        )
    print(
        "clause (c) control: rows 0 and 1 of the B=2 run differ on "
        + String(control)
        + " of 15 batched stages, so the row slicer distinguishes rows"
    )

    var cells = 0
    var bad = 0
    var first_bad = String("")
    for i in range(len(names)):
        var r0_1 = row_slice(d1[i], i, 0, l, dims)
        var r0_2 = row_slice(d2[i], i, 0, l, dims)
        var r0_3 = row_slice(d3[i], i, 0, l, dims)
        cells += len(r0_1) * 2
        var a = compare_stage(
            names[i] + " row0 B=1 vs B=2", r0_1, r0_2, True
        )
        var b = compare_stage(
            names[i] + " row0 B=1 vs B=3", r0_1, r0_3, True
        )
        if a.n_diff > 0 or b.n_diff > 0:
            bad += 1
            if first_bad == "":
                first_bad = names[i] + " row 0"
        # row 1 exists in B=2 and B=3 and must also be composition-free
        var r1_2 = row_slice(d2[i], i, 1, l, dims)
        var r1_3 = row_slice(d3[i], i, 1, l, dims)
        cells += len(r1_2)
        var c = compare_stage(
            names[i] + " row1 B=2 vs B=3", r1_2, r1_3, True
        )
        if c.n_diff > 0:
            bad += 1
            if first_bad == "":
                first_bad = names[i] + " row 1"
    if bad != 0:
        raise Error(
            String("mamba_check: CLAUSE (c) FAILED on ")
            + String(bad)
            + " stages, first at "
            + first_bad
            + ": a row's bits depend on who shares its launch"
        )
    print(
        "clause (c): PASS, row 0 identical at B=1, B=2 and B=3 and row 1"
        " identical at B=2 and B=3, on all "
        + String(cells)
        + " compared cells of all 16 stages"
    )


# ===========================================================================
# CLAUSE (d): decode == prefill, bitwise, per token
# ===========================================================================


def clause_d(
    ctx: DeviceContext,
    w: MambaWeights,
    x: List[Float32],
    l: Int,
    dims: MambaDims,
) raises:
    """Contract section 8 clause (d), AT THE COMPOSITION POINT. Gate D is
    green for `mamba_simple.mojo` in isolation; this is the same claim for
    `mamba_block_forward`, which is a different thing and is what the block's
    card is made of.

    This clause is what DEVIATION 721 exists to make true BY CONSTRUCTION.
    `Mamba.step` sums the conv taps and adds the bias afterwards, while the
    prefill kernels seed the accumulator WITH the bias; the profile uses the
    bias seed on BOTH paths, precisely so that two spellings cannot make this
    clause false. So it should pass, and a failure here would be the most
    important finding of the night rather than a bug in this file.

    A length-L sequence is run once as a prefill and then one token at a time
    through the SAME entry point with the state carried, and every token of
    every stage is compared. The per-batch stages (`conv.window`, `scan.h`)
    are the state AFTER the call, so they are comparable only at the last
    token, and that is where they are compared.

    THE NEGATIVE CONTROL, and clause (d) is the clause most exposed without
    one. If the decode path and the prefill path shared a buffer or a cached
    card, the comparison would be a value against ITSELF and would pass for
    ever on every vendor. So the clause first compares decode step `t`
    against prefill token `t + 1` -- a DELIBERATE MISALIGNMENT that MUST
    differ. If it does not, the comparison cannot tell two tokens apart and
    the clause raises VACUOUS rather than FAILED."""
    var dm = dims.d_model
    var names = stage_names()

    print("clause (d): decode == prefill at the block, per token, per stage")

    var dw1 = MambaDeviceWeights(ctx, w)
    var st1 = MambaDeviceState(ctx, 1, dims)
    var off1 = IdentityTrace.disabled()
    var pre = run_step(ctx, dw1, st1, x, 1, l, dims, off1, "prefill")

    var dw2 = MambaDeviceWeights(ctx, w)
    var st2 = MambaDeviceState(ctx, 1, dims)
    var steps = List[List[List[Float32]]]()
    for t in range(l):
        var xt = List[Float32]()
        for j in range(dm):
            xt.append(x[t * dm + j])
        var off2 = IdentityTrace.disabled()
        steps.append(run_step(ctx, dw2, st2, xt, 1, 1, dims, off2, "decode"))

    # ---- the control: misaligned tokens MUST differ ----------------------
    var control = 0
    for t in range(l - 1):
        for i in range(len(names)):
            if stage_kind(i) != KIND_TOKEN:
                continue
            var a = token_slice(pre[i], i, t + 1, dims)
            var b = token_slice(steps[t][i], i, 0, dims)
            var d = compare_stage("control", a, b, False)
            if d.n_diff > 0:
                control += 1
    if l > 1 and control == 0:
        raise Error(
            "mamba_check: CLAUSE (d) IS VACUOUS. Decode step t is"
            " bit-identical to prefill token t+1 on every stage, which"
            " cannot be true of different tokens. The decode and prefill"
            " paths are not two computations here -- they share a buffer or"
            " a cached card -- so the aligned comparison below is a value"
            " against itself ([[reached-but-inert]])."
        )
    print(
        "clause (d) control: decode step t vs prefill token t+1 differs on "
        + String(control)
        + " misaligned stage comparisons, so the comparison distinguishes"
        " tokens"
    )

    # ---- the clause: aligned tokens must MATCH ---------------------------
    var cells = 0
    var bad = 0
    var first_bad = String("")
    for t in range(l):
        for i in range(len(names)):
            var kind = stage_kind(i)
            if kind == KIND_BATCH and t != l - 1:
                continue  # state AFTER the call: comparable at the last token
            var a: List[Float32]
            var b: List[Float32]
            if kind == KIND_TOKEN:
                a = token_slice(pre[i], i, t, dims)
                b = token_slice(steps[t][i], i, 0, dims)
            else:
                a = pre[i].copy()
                b = steps[t][i].copy()
            cells += len(a)
            var d = compare_stage(
                names[i] + " token " + String(t), a, b, True
            )
            if d.n_diff > 0:
                bad += 1
                if first_bad == "":
                    first_bad = names[i] + " at token " + String(t)
    if bad != 0:
        raise Error(
            String("mamba_check: CLAUSE (d) FAILED on ")
            + String(bad)
            + " stage-tokens, first at "
            + first_bad
            + ". DEVIATION 721's bias seed was supposed to make this true by"
            + " construction, so this is a finding about the profile and not"
            + " about the gate."
        )
    print(
        "clause (d): PASS, "
        + String(l)
        + " decode steps bit-identical to the prefill on all "
        + String(cells)
        + " compared cells"
    )


# ===========================================================================


def main() raises:
    var b = env_int("MOJOLEARN_MAMBA_CHECK_B", 1)
    var l = env_int("MOJOLEARN_MAMBA_CHECK_L", 4)
    var dm = env_int("MOJOLEARN_MAMBA_CHECK_DM", 8)
    var dims = MambaDims.of(dm)
    var di = dims.d_inner
    var r = dims.dt_rank
    var m = b * l
    var armed = mamba_block_sabotage_name()

    print(
        "=== mamba block identity gate, profile"
        " mojolearn.identical.mamba1.fp32.v1"
    )
    print("mode " + mode_name() + "   block sabotage: " + armed)
    print(
        "shape B="
        + String(b)
        + " L="
        + String(l)
        + " d_model="
        + String(dm)
        + " d_inner="
        + String(di)
        + " dt_rank="
        + String(r)
    )

    var w = planted_weights(dims)
    var x = corpus_x(corpus_case_seed(1), b, l, dm)

    var hstate = MambaState(b, dims)
    var st = mamba_block_oracle(w, x, b, l, hstate)
    var host = oracle_dump(st)

    var reach = softplus_band_reach(w, st, m, di)
    print(
        "reach: S14's input lands in (10, 20] on "
        + String(reach)
        + " of "
        + String(m * di)
        + " cells  (fixture F7's band; 0 would make S14_THRESHOLD_10 vacuous)"
    )

    var ctx = DeviceContext()
    var trace = IdentityTrace.to_path(TRACE_PATH)
    var dev = run_block(ctx, w, x, b, l, dims, trace, TAG_PREFIX)

    print("clause (a): stages, device vs host oracle, BITWISE:")
    var diffs = compare_dumps(host, dev, True)
    var n_moved = count_moved(diffs)

    _ = check_card_tags(TRACE_PATH)

    comptime if BLOCK_ANY_SABOTAGE:
        if n_moved == 0:
            raise Error(
                String("mamba_check: SABOTAGE ")
                + armed
                + " IS ARMED AND MOVED NO BIT. Either its branch was never"
                + " reached at this shape or it is inert there"
                + " ([[reached-but-inert]]). It falsifies NOTHING and must"
                + " not be reported as a passing arm."
            )
        print(
            "SABOTAGE "
            + armed
            + " BIT: "
            + String(n_moved)
            + " of "
            + String(len(diffs))
            + " stages moved, first at "
            + first_moved(diffs)
            + ". The clause it targets is falsifiable."
        )
        print(
            "clauses (b) and (c) are NOT run under a sabotage build: they are"
            " invariance claims and a deterministic sabotage satisfies them."
        )
    else:
        if n_moved != 0:
            raise Error(
                String("mamba_check: CLAUSE (a) FAILED, ")
                + String(n_moved)
                + " stages differ from the oracle, first at "
                + first_moved(diffs)
            )
        print(
            "clause (a): PASS, "
            + String(len(diffs))
            + "/"
            + String(len(diffs))
            + " stages bit-identical to the oracle, 17/17 card tags."
        )
        clause_b(ctx, w, x, b, l, dims, dev)
        if env_on("MOJOLEARN_MAMBA_CHECK_CLAUSE_C"):
            clause_c(ctx, w, l, dims)
        else:
            print(
                "clause (c): SKIPPED (set MOJOLEARN_MAMBA_CHECK_CLAUSE_C=1)"
            )
        if env_on("MOJOLEARN_MAMBA_CHECK_CLAUSE_D"):
            if b != 1:
                raise Error(
                    "mamba_check: clause (d) is written for B=1; set"
                    " MOJOLEARN_MAMBA_CHECK_B=1"
                )
            clause_d(ctx, w, x, l, dims)
        else:
            print(
                "clause (d): SKIPPED (set MOJOLEARN_MAMBA_CHECK_CLAUSE_D=1)"
            )
        print(
            "SCOPE: this shape only, IDENTICAL only. L=16/64/257, FAST,"
            " clauses (d) and (e) and the corpus cross-check are OWED."
        )
