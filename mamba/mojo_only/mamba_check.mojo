"""The gate file of profile `mojolearn.identical.mamba1.fp32.v1`, the path
`mamba/IDENTICAL_MAMBA_CONTRACT.md` section 8 names.

NOT A PORT. It runs the device block
(`mamba/ported/transformers/models/mamba/modeling_mamba.mojo`) against the
host oracle (`mamba/mojo_only/mamba_oracle.mojo`) and compares every recorded
stage BY BITS.

**THIS GATE IS DELIBERATELY TINY AND IT IS NOT THE CONTRACT'S GATE YET.**
ONE shape, B = 1, L = 4, d_model = 8, and ONE launch. Andrew's machine was
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

WHERE THE LEDGER IS THIN, and it is thin
------------------------------------------
`S1_FOLD_DESCENDING` bites on ONE of four rows at `d_model = 8`. Eight terms
is a short fold and most rows round the same way in both directions. The arm
is genuinely falsifying, but weakly, and `d_model = 16` is owed before the
S1 clause should be called well gated. `S14_THRESHOLD_10` is reported only
because the fixture plants the band and the reach count above is nonzero;
against the corpus default ranges it would be VACUOUS.

OWED, and this file does not cover any of it
----------------------------------------------
The rest of contract section 3's sweep (B in {1,2,3}, L in {1,16,64,257},
d_model 16); FAST mode; clause (b), eight repeated launches; clause (c),
batch composition; clause (d), decode == prefill at this composition point;
clause (e), the row-39 planted NaN and infinity audit; the corpus cross-check
against `mamba/corpus/`; and every column that is not this Apple box.
"""

from std.memory import bitcast

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


comptime CHECK_B = 1
comptime CHECK_L = 4
comptime CHECK_DM = 8
comptime TRACE_PATH = "/tmp/mojolearn_mamba_block_tiny.trace"
comptime TAG_PREFIX = "tiny"


def hexbits(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


@fieldwise_init
struct StageDiff(Copyable, Movable):
    var name: String
    var n_cells: Int
    var n_diff: Int
    var first: Int


def compare_stage(
    name: String, host: List[Float32], dev: List[Float32]
) raises -> StageDiff:
    """Bitwise, cell by cell. A LENGTH mismatch is reported as such rather
    than compared to the shorter of the two, because a stage that is the
    wrong size is a different defect from a stage that is the wrong value."""
    if len(host) != len(dev):
        raise Error(
            String("mamba_check: stage ")
            + name
            + " has "
            + String(len(host))
            + " cells on the host and "
            + String(len(dev))
            + " on the device"
        )
    var n_diff = 0
    var first = -1
    for i in range(len(host)):
        if bitcast[DType.uint32](host[i]) != bitcast[DType.uint32](dev[i]):
            n_diff += 1
            if first < 0:
                first = i
    if n_diff == 0:
        print("  OK    " + name + "  (" + String(len(host)) + " cells)")
    else:
        print(
            "  MOVED "
            + name
            + "  "
            + String(n_diff)
            + " of "
            + String(len(host))
            + " cells, first cell "
            + String(first)
            + "  host "
            + hexbits(host[first])
            + "  device "
            + hexbits(dev[first])
        )
    return StageDiff(name, len(host), n_diff, first)


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
    """The emitted tag SEQUENCE against contract section 7. Returns the
    number of tags read. The differ aligns two traces by their tag
    sequences, so an order that drifts between vendors is a divergence
    report on every stage; an order that drifts from the CONTRACT is a card
    that is no longer the card the contract froze."""
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


def main() raises:
    var b = CHECK_B
    var l = CHECK_L
    var dm = CHECK_DM
    var dims = MambaDims.of(dm)
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    var m = b * l
    var armed = mamba_block_sabotage_name()

    print("=== mamba block identity gate, profile mojolearn.identical.mamba1.fp32.v1")
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
        + "   (ONE shape, ONE launch: the machine-load freeze)"
    )

    var w = planted_weights(dims)
    var x = corpus_x(corpus_case_seed(1), b, l, dm)

    # ---- the host oracle, the ANSWER -------------------------------------
    var hstate = MambaState(b, dims)
    var st = mamba_block_oracle(w, x, b, l, hstate)

    var reach = softplus_band_reach(w, st, m, di)
    print(
        "reach: S14's input lands in (10, 20] on "
        + String(reach)
        + " of "
        + String(m * di)
        + " cells  (fixture F7's band; 0 would make S14_THRESHOLD_10 vacuous)"
    )

    # ---- the device block -------------------------------------------------
    var ctx = DeviceContext()
    var dw = MambaDeviceWeights(ctx, w)
    var dstate = MambaDeviceState(ctx, b, dims)
    var dstages = MambaDeviceStages(ctx, b, l, dims)
    var dx = mamba_upload(ctx, x)
    var trace = IdentityTrace.to_path(TRACE_PATH)
    mamba_block_forward(ctx, dstages, dstate, dw, dx, b, l, trace, TAG_PREFIX)

    # ---- every computed stage, bitwise ------------------------------------
    print("stages, device vs host oracle, BITWISE:")
    var diffs = List[StageDiff]()
    diffs.append(
        compare_stage(
            "norm.sumsq", st.norm_sumsq, mamba_download(ctx, dstages.norm_sumsq, m)
        )
    )
    diffs.append(
        compare_stage(
            "norm.out", st.norm_out, mamba_download(ctx, dstages.norm_out, m * dm)
        )
    )
    diffs.append(
        compare_stage(
            "in_proj.out",
            st.in_proj,
            mamba_download(ctx, dstages.in_proj, m * 2 * di),
        )
    )
    diffs.append(
        compare_stage(
            "A.out", st.a_out, mamba_download(ctx, dstages.a_out, di * D_STATE)
        )
    )
    diffs.append(
        compare_stage(
            "conv.out", st.conv_out, mamba_download(ctx, dstages.conv_out, m * di)
        )
    )
    diffs.append(
        compare_stage(
            "silu.out", st.silu_out, mamba_download(ctx, dstages.silu_out, m * di)
        )
    )
    diffs.append(
        compare_stage(
            "conv.window",
            st.conv_win,
            mamba_download(ctx, dstages.conv_win, b * di * D_CONV),
        )
    )
    diffs.append(
        compare_stage(
            "x_proj.out", st.x_proj, mamba_download(ctx, dstages.x_proj, m * xr)
        )
    )
    diffs.append(
        compare_stage(
            "dt_proj.out", st.dt_proj, mamba_download(ctx, dstages.dt_proj, m * di)
        )
    )
    diffs.append(
        compare_stage(
            "softplus.out",
            st.softplus_out,
            mamba_download(ctx, dstages.softplus_out, m * di),
        )
    )
    diffs.append(
        compare_stage(
            "scan.y", st.scan_y, mamba_download(ctx, dstages.scan_y, m * di)
        )
    )
    diffs.append(
        compare_stage(
            "scan.h",
            st.scan_h,
            mamba_download(ctx, dstages.scan_h, b * di * D_STATE),
        )
    )
    diffs.append(
        compare_stage(
            "skip.out", st.skip_out, mamba_download(ctx, dstages.skip_out, m * di)
        )
    )
    diffs.append(
        compare_stage(
            "gate.out", st.gate_out, mamba_download(ctx, dstages.gate_out, m * di)
        )
    )
    diffs.append(
        compare_stage(
            "out_proj.out",
            st.out_proj,
            mamba_download(ctx, dstages.out_proj, m * dm),
        )
    )
    diffs.append(
        compare_stage(
            "residual.out",
            st.residual_out,
            mamba_download(ctx, dstages.residual_out, m * dm),
        )
    )

    var n_moved = 0
    var first_moved = String("")
    var first_moved_cells = 0
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            n_moved += 1
            if first_moved == "":
                first_moved = diffs[i].name
                first_moved_cells = diffs[i].n_diff

    # ---- the card ---------------------------------------------------------
    _ = check_card_tags(TRACE_PATH)

    # ---- the verdict, INVERTED when an arm is armed -----------------------
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
            + first_moved
            + " on "
            + String(first_moved_cells)
            + " cells. The clause it targets is falsifiable."
        )
    else:
        if n_moved != 0:
            raise Error(
                String("mamba_check: CLEAN BUILD FAILED, ")
                + String(n_moved)
                + " stages differ from the oracle, first at "
                + first_moved
            )
        print(
            "CLEAN: "
            + String(len(diffs))
            + "/"
            + String(len(diffs))
            + " stages bit-identical to the oracle, 17/17 card tags."
        )
        print(
            "SCOPE: one shape, one launch, IDENTICAL only. The sweep, FAST,"
            " clauses (b)-(e) and the corpus cross-check are OWED."
        )
