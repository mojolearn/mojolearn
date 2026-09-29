# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate of the AMD plans of the int8 unit product, one product and
four: every plan of `gemm/checks/gemm_int8_mma_amd.mojo` against the
reference unit plan, the flat plan and the host.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_int8_mma_amd_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_AMD_SABOTAGE=1 -I . gemm/checks/gemm_int8_mma_amd_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_TUNED_SABOTAGE=1 -I . gemm/checks/gemm_int8_mma_amd_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_PIECES_SABOTAGE=1 -I . gemm/checks/gemm_int8_mma_amd_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_SABOTAGE=1 -I . gemm/checks/gemm_int8_mma_amd_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_TUNED_UNSTATED=1 -I . gemm/checks/gemm_int8_mma_amd_check.mojo

The first must pass every gate. The second breaks the padding rule of the
DIRECT loads (a step beyond `k` reads the row's last code) and must FAIL
every gate that compares a product or a sum, through the direct plans, at
the shapes whose `k` is not a whole number of the plan's windows; its staged
plans must still agree. The third breaks the padding rule of the STAGING and
must fail the same gates through the staged plans; its direct plans must
still agree. The fourth pairs the wrong fragments in the four-product
kernels (HL twice, LH never) and must FAIL the two four-product gates; the
one-product gates must pass under it. The fifth flips every value stored
and must FAIL every gate. The sixth answers "not aligned" at every launch,
so every load is the byte path, and must pass every gate with the same
bits. `tools/lowbit_amd_tuned/gate_job.sh` runs the six and reads the logs.

Lane lane/lowbit-amd-tuned, 2026-09-29. The shapes, the plants and the
host's sums are lane/lowbit-mma-speed's and lane/lowbit-units', imported
and not respelled: `gemm/checks/gemm_int8_mma_tuned_check.mojo`,
`gemm/checks/gemm_int8_pieces_tuned_check.mojo`,
`gemm/checks/gemm_int8_apple_chunk_check.mojo`,
`gemm/checks/gemm_lowbit_check.mojo`. The answer of one product is
`gemm/host/gemm_lowbit_oracle.mojo::gemm_int8_oracle`; of four,
`piece_sums_host`.

WHAT IS PLANTED, beyond those files' plants. The fifteen-bit profile's high
plane holds -128, which no code of `int8i32.v1` is, and its piece products
run through the ONE-product kernel when they are four launches. So
`check_amd_minus_128_piece` plants -128 on both operands of one product, up
to the four-product kernel's largest `k`, 65536, where the sum is 2^30.

A PLAN'S SABOTAGE MUST REACH WHAT IT CAN AND NOTHING ELSE. Under either
scheduling arm every case prints whether the defect could reach it (the
plan reads the arm, and `k` is not a whole number of its windows) and the
gate `check_amd_sabotage_reach` FAILS when a case the defect cannot reach
differs or a case it must reach at a planted shape agrees. It passes in a
clean build, where no case differs.

ON A COLUMN THAT IS NOT AMD no gate here runs, which main says and which is
not a pass.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.sys import has_accelerator, is_defined

from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.numerics import numeric_mode_name
from checks.numerics_int15 import dequant_int15_pinned, int15_recombine
from gemm.checks.gemm_int8_apple_chunk_check import (
    PLANT_SHAPE_COUNT,
    _plant_exponents,
)
from gemm.checks.gemm_int8_mma import identical_gemm_int8_mma_into
from gemm.checks.gemm_int8_mma_amd import (
    INT8_AMD_AVAILABLE,
    INT8_AMD_PIECES_PLAN_COUNT,
    INT8_AMD_PLAN_COUNT,
    INT8_AMD_SABOTAGE,
    identical_gemm_int8_mma_amd_into,
    identical_gemm_int8_mma_amd_with_plan,
    identical_gemm_int8_pieces_amd_fused_into,
    identical_gemm_int8_pieces_amd_fused_with_plan,
    identical_gemm_int8_pieces_amd_into,
    identical_gemm_int8_pieces_amd_with_plan,
    int8_amd_dispatch,
    int8_amd_pieces_dispatch,
    int8_amd_pieces_plan_is_direct,
    int8_amd_pieces_plan_name,
    int8_amd_plan_is_direct,
    int8_amd_plan_name,
    int8_amd_plan_window,
    int8_amd_sabotage_name,
)
from gemm.checks.gemm_int8_mma_tuned import (
    INT8_PIECES_MAX_K,
    INT8_PIECES_MAX_K_ANY_INT8,
    INT8_TUNED_SABOTAGE,
    identical_gemm_int8_pieces_flat_into,
)
from gemm.checks.gemm_int8_mma_tuned_check import (
    TUNED_PLANT_COUNT,
    TUNED_PLANT_SHAPE_COUNT,
    TUNED_SHAPE_COUNT,
    _tuned_plant_a,
    _tuned_plant_b,
    _tuned_plant_name,
    _tuned_plant_shape,
    _tuned_shape,
)
from gemm.checks.gemm_int8_pieces_tuned_check import (
    PIECES_PLANT_ALL_MIN,
    PIECES_PLANT_COUNT,
    PIECES_PLANT_SHAPE_COUNT,
    _constant_plane,
    _pieces_plant_name,
    _pieces_plant_shape,
    _pieces_plant_value,
    _plane,
    _poisoned_sums,
    _read_sums,
    _sums_diff,
    piece_sums_host,
)
from gemm.checks.gemm_lowbit import identical_gemm_int8_flat_into
from gemm.checks.gemm_lowbit_check import (
    MMA_SHAPE_COUNT,
    SHAPE_COUNT,
    _download_f32,
    _fill,
    _first_diff,
    _gate,
    _mma_shape,
    _poisoned,
    _shape,
    _show,
    _upload_i32,
    _upload_i8,
)
from gemm.checks.quantize_int8_par_check import Tally, _diff, _verdict
from gemm.host.gemm_lowbit_oracle import gemm_int8_oracle, quantize_rows_int8
from gemm.host.gemm_oracle import OP_NT

#: Whether this build has the plans: AMD only.
comptime HAS_AMD = INT8_AMD_AVAILABLE

#: Whether the build carries a scheduling arm, and whose plans read it.
comptime REACH_DIRECT = INT8_AMD_SABOTAGE
comptime REACH_STAGED = INT8_TUNED_SABOTAGE

#: `-D MOJOLEARN_INT8_AMD_TRACE=1`: every launch is named on stdout, flushed,
#: before it is enqueued, so a launch that faults the device (the process
#: dies with no line of its own, job 1790657510941) is the last one named.
comptime INT8_AMD_TRACE = is_defined["MOJOLEARN_INT8_AMD_TRACE"]()


def _trace(what: String):
    comptime if INT8_AMD_TRACE:
        print("   launch " + what, flush=True)


struct Reach(Movable):
    """What a scheduling arm did against what it can do: the cases it could
    not reach that differed anyway, and the cases it could reach, with how
    many of them differed."""

    var unreachable_failed: Int
    var reachable: Int
    var reachable_failed: Int
    var first: String

    def __init__(out self):
        self.unreachable_failed = 0
        self.reachable = 0
        self.reachable_failed = 0
        self.first = String("")

    def note(mut self, can_reach: Bool, verdict: String, tag: String):
        if can_reach:
            self.reachable += 1
            if verdict.byte_length() > 0:
                self.reachable_failed += 1
        elif verdict.byte_length() > 0:
            self.unreachable_failed += 1
            if self.first.byte_length() == 0:
                self.first = tag.copy()


def _can_reach(direct: Bool, window: Int, k: Int) -> Bool:
    """Whether the build's scheduling arm can change this plan's output at
    this `k`: the plan reads the arm, and the last window holds a step
    beyond `k`."""
    var reads = False
    comptime if REACH_DIRECT:
        reads = direct
    elif REACH_STAGED:
        reads = not direct
    return reads and k % window != 0


# ===========================================================================
# ONE PRODUCT
# ===========================================================================


def _run_every_amd_plan(
    ctx: DeviceContext,
    qa: List[Int8],
    ea: List[Int32],
    qb: List[Int8],
    eb: List[Int32],
    m: Int,
    n: Int,
    k: Int,
    tag: String,
    mut tally: Tally,
    mut reach: Reach,
) raises:
    """One shape and one pair of operands through the flat plan, the
    reference unit plan and every plan of the AMD file, each against the
    oracle, the reference and the flat plan. The operands are uploaded
    once; every plan writes its own poisoned output."""
    var want = gemm_int8_oracle(qa, ea, qb, eb, m, n, k)
    var dqa = _upload_i8(ctx, qa)
    var dea = _upload_i32(ctx, ea)
    var dqb = _upload_i8(ctx, qb)
    var deb = _upload_i32(ctx, eb)
    var dflat = _poisoned(ctx, m * n)
    var dref = _poisoned(ctx, m * n)
    _trace(tag + " flat and reference")
    identical_gemm_int8_flat_into(ctx, dflat, dqa, dea, dqb, deb, m, n, k)
    identical_gemm_int8_mma_into(ctx, dref, dqa, dea, dqb, deb, m, n, k)
    ctx.synchronize()
    var flat = _download_f32(ctx, dflat, m * n, tag + " flat")
    var ref_ = _download_f32(ctx, dref, m * n, tag + " reference")
    tally.note(_diff(flat, want, tag + " (flat vs oracle)"))
    tally.note(_diff(ref_, want, tag + " (reference unit plan vs oracle)"))
    for plan in range(INT8_AMD_PLAN_COUNT):
        var ptag = tag + " " + int8_amd_plan_name(plan)
        var dc = _poisoned(ctx, m * n)
        var got: String
        # A launch the device refuses is a failed case of THAT plan, named,
        # and the plans after it still run.
        try:
            _trace(ptag)
            identical_gemm_int8_mma_amd_with_plan(
                ctx, dc, dqa, dea, dqb, deb, m, n, k, plan
            )
            ctx.synchronize()
            var out = _download_f32(ctx, dc, m * n, ptag)
            got = _diff(out, want, ptag + " (amd plan vs oracle)")
            if got.byte_length() == 0:
                got = _diff(out, ref_, ptag + " (amd plan vs reference unit plan)")
            if got.byte_length() == 0:
                got = _diff(out, flat, ptag + " (amd plan vs flat)")
        except e:
            got = ptag + ": " + String(e)
        reach.note(
            _can_reach(int8_amd_plan_is_direct(plan), int8_amd_plan_window(plan), k),
            got,
            ptag,
        )
        tally.note(got)
        _ = dc
    _ = dqa
    _ = dea
    _ = dqb
    _ = deb
    _ = dflat
    _ = dref


def check_amd_plans_match_reference_flat_oracle(
    ctx: DeviceContext, mut reach: Reach
) raises:
    """GATE: on codes quantized from the float fixtures every AMD plan
    returns the oracle's bits, the reference unit plan's and the flat
    plan's, at every OP_NT shape of `gemm_lowbit_check.mojo`, its ragged
    shapes and the tuned gate's."""
    var tally = Tally()
    var shapes = 0
    for s in range(SHAPE_COUNT + MMA_SHAPE_COUNT + TUNED_SHAPE_COUNT):
        var m = 0
        var n = 0
        var k = 0
        if s < SHAPE_COUNT:
            var sh = _shape(s)
            if sh[3] != OP_NT:
                continue
            m = sh[0]
            n = sh[1]
            k = sh[2]
        elif s < SHAPE_COUNT + MMA_SHAPE_COUNT:
            var sh = _mma_shape(s - SHAPE_COUNT)
            m = sh[0]
            n = sh[1]
            k = sh[2]
        else:
            var sh = _tuned_shape(s - SHAPE_COUNT - MMA_SHAPE_COUNT)
            m = sh[0]
            n = sh[1]
            k = sh[2]
        var qa = quantize_rows_int8(_fill(m * k, 311 + s), m, k)
        var qb = quantize_rows_int8(_fill(n * k, 331 + s), n, k)
        var tag = "int8 amd " + String(m) + "x" + String(n) + "x" + String(k)
        var before = tally.failed
        _run_every_amd_plan(ctx, qa.q, qa.e, qb.q, qb.e, m, n, k, tag, tally, reach)
        shapes += 1
        if tally.failed == before:
            print("   ok " + tag)
    print("   shapes: " + String(shapes))
    _verdict(tally, String("quantized-fixture"))


def check_amd_planted_worst_cases(ctx: DeviceContext, mut reach: Reach) raises:
    """GATE: the planted codes of the tuned gate (lane/lowbit-units' five
    plants up to the profile's largest `k`, a row of zero codes, one code
    alone at the last step) through every AMD plan, against the oracle, the
    reference unit plan and the flat plan."""
    var tally = Tally()
    for s in range(PLANT_SHAPE_COUNT + TUNED_PLANT_SHAPE_COUNT):
        var sh = _tuned_plant_shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var ea = _plant_exponents(m, -6, 5)
        var eb = _plant_exponents(n, -7, 3)
        for plant in range(TUNED_PLANT_COUNT):
            var qa = _tuned_plant_a(plant, m, k)
            var qb = _tuned_plant_b(plant, n, k)
            var tag = (
                "int8 amd planted " + _tuned_plant_name(plant) + " " + String(m) + "x"
                + String(n) + "x" + String(k)
            )
            var before = tally.failed
            _run_every_amd_plan(ctx, qa, ea, qb, eb, m, n, k, tag, tally, reach)
            if tally.failed == before:
                print("   ok " + tag)
    _verdict(tally, String("planted"))


comptime MINUS_128_SHAPE_COUNT = 5


def _minus_128_shape(i: Int) -> Tuple[Int, Int, Int]:
    """Off the plans' tiles and windows, and the four-product kernel's two
    largest `k`."""
    if i == 0:
        return (17, 33, 4100)
    if i == 1:
        return (65, 67, 4096)
    if i == 2:
        return (1, 130, 4097)
    if i == 3:
        return (2, 3, INT8_PIECES_MAX_K_ANY_INT8)
    return (3, 2, INT8_PIECES_MAX_K)


def check_amd_minus_128_piece(ctx: DeviceContext, mut reach: Reach) raises:
    """GATE: a PIECE OF -128 on both operands of one product. -128 is no
    code of `int8i32.v1` (L-4 clamps to [-127, 127]) and is a code of the
    fifteen-bit profile's high plane, whose products this kernel computes
    when they are four launches. Every product is +16384, the largest any
    two int8 give; at `k` 65536 the sum is 2^30. One code of each operand
    differs, so a cell is not every other cell."""
    var tally = Tally()
    for s in range(MINUS_128_SHAPE_COUNT):
        var sh = _minus_128_shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var ea = _plant_exponents(m, -6, 5)
        var eb = _plant_exponents(n, -7, 3)
        var qa = _constant_plane(m * k, -128)
        var qb = _constant_plane(n * k, -128)
        qa[(m - 1) * k + k // 3] = Int8(5)
        qb[(n // 2) * k + k - 1] = Int8(-3)
        var tag = "int8 amd planted minus-128 " + String(m) + "x" + String(n) + "x" + String(k)
        var before = tally.failed
        _run_every_amd_plan(ctx, qa, ea, qb, eb, m, n, k, tag, tally, reach)
        if tally.failed == before:
            print("   ok " + tag)
    _verdict(tally, String("minus-128"))


def _run_amd_dispatched(
    ctx: DeviceContext,
    qa: List[Int8],
    ea: List[Int32],
    mut dqb: DeviceBuffer[DType.int8],
    mut deb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
    tag: String,
) raises -> List[Float32]:
    """`identical_gemm_int8_mma_amd_into` on uploaded left codes."""
    var dqa = _upload_i8(ctx, qa)
    var dea = _upload_i32(ctx, ea)
    var dc = _poisoned(ctx, m * n)
    identical_gemm_int8_mma_amd_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
    ctx.synchronize()
    var out = _download_f32(ctx, dc, m * n, tag)
    _ = dqa
    _ = dea
    _ = dc
    return out^


def check_amd_dispatch_is_batch_invariant(ctx: DeviceContext) raises:
    """GATE: through the plan the AMD launcher picks, row `i` of an `m`-row
    call equals the one-row call on row `i` alone, and the batch equals the
    oracle. The two calls take DIFFERENT plans; the gate refuses to pass
    when they take one."""
    var m = 40
    var n = 150
    var k = 1000
    var qa = quantize_rows_int8(_fill(m * k, 401), m, k)
    var qb = quantize_rows_int8(_fill(n * k, 409), n, k)
    var want = gemm_int8_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
    var dqb = _upload_i8(ctx, qb.q)
    var deb = _upload_i32(ctx, qb.e)
    var batch = _run_amd_dispatched(ctx, qa.q, qa.e, dqb, deb, m, n, k, String("amd batch"))
    _first_diff(batch, want, String("amd batch (dispatched vs oracle)"))
    if int8_amd_dispatch(m, n, k) == int8_amd_dispatch(1, n, k):
        raise Error("the batch and the single row took the same plan; the gate compares nothing")
    for i in range(m):
        var row = List[Int8]()
        for p in range(k):
            row.append(qa.q[i * k + p])
        var er = List[Int32]()
        er.append(qa.e[i])
        var one = _run_amd_dispatched(ctx, row, er, dqb, deb, 1, n, k, "amd row " + String(i))
        for j in range(n):
            if bitcast[DType.uint32](one[j]) != bitcast[DType.uint32](batch[i * n + j]):
                raise Error(
                    "amd row " + String(i) + " col " + String(j)
                    + " differs between the batch and the single-row call: "
                    + _show(batch[i * n + j]) + " vs " + _show(one[j])
                )
    _ = dqb
    _ = deb
    print(
        "   ok " + String(m) + " rows agree with their single-row calls; plans "
        + int8_amd_plan_name(int8_amd_dispatch(m, n, k)) + " and "
        + int8_amd_plan_name(int8_amd_dispatch(1, n, k))
    )


# ===========================================================================
# FOUR PRODUCTS
# ===========================================================================


def _fused_host(
    want: List[Int32], ea: List[Int32], eb: List[Int32], m: Int, n: Int
) -> List[Float32]:
    """What the FUSED form stores, from the host's three sums: the
    fifteen-bit seam (`int15_store_cell`'s arithmetic, W-5 to W-7) on the
    host, with no define of the device's defect arms."""
    var out = List[Float32]()
    for i in range(m):
        for j in range(n):
            var at_ = 3 * (i * n + j)
            out.append(
                dequant_int15_pinned(
                    int15_recombine(want[at_], want[at_ + 1], want[at_ + 2]),
                    Int(ea[i]) + Int(eb[j]),
                )
            )
    return out^


def _run_every_amd_pieces_plan(
    ctx: DeviceContext,
    ah: List[Int8],
    al: List[Int8],
    bh: List[Int8],
    bl: List[Int8],
    m: Int,
    n: Int,
    k: Int,
    tag: String,
    mut tally: Tally,
    mut reach: Reach,
) raises:
    """One shape and four planes through the reference device plan (one
    thread per cell) and every four-product plan of the AMD file, each
    against the host and against the reference device plan."""
    var want = piece_sums_host(ah, al, bh, bl, m, n, k)
    # The FUSED form's operands and answer: row exponents that differ row
    # to row, and the seam on the host's sums.
    var ea = _plant_exponents(m, -6, 5)
    var eb = _plant_exponents(n, -7, 3)
    var want_c = _fused_host(want, ea, eb, m, n)
    var dea = _upload_i32(ctx, ea)
    var deb = _upload_i32(ctx, eb)
    var dah = _upload_i8(ctx, ah)
    var dal = _upload_i8(ctx, al)
    var dbh = _upload_i8(ctx, bh)
    var dbl = _upload_i8(ctx, bl)
    var flat = List[Int32]()
    var flat_ok = False
    var dflat = _poisoned_sums(ctx, 3 * m * n)
    var verdict: String
    try:
        _trace(tag + " pieces flat")
        identical_gemm_int8_pieces_flat_into(ctx, dflat, dah, dal, dbh, dbl, m, n, k)
        ctx.synchronize()
        flat = _read_sums(ctx, dflat, 3 * m * n, tag + " flat")
        flat_ok = True
        verdict = _sums_diff(flat, want, n, tag + " (reference device plan vs host)")
    except e:
        verdict = tag + " flat: " + String(e)
    tally.note(verdict)
    for plan in range(INT8_AMD_PIECES_PLAN_COUNT):
        var ptag = tag + " " + int8_amd_pieces_plan_name(plan)
        var ds = _poisoned_sums(ctx, 3 * m * n)
        var got: String
        try:
            _trace(ptag)
            identical_gemm_int8_pieces_amd_with_plan(
                ctx, ds, dah, dal, dbh, dbl, m, n, k, plan
            )
            ctx.synchronize()
            var out = _read_sums(ctx, ds, 3 * m * n, ptag)
            got = _sums_diff(out, want, n, ptag + " (amd plan vs host)")
            if got.byte_length() == 0 and flat_ok:
                got = _sums_diff(out, flat, n, ptag + " (amd plan vs reference device plan)")
            # THE FUSED FORM of the same plan: the seam in the last step.
            if got.byte_length() == 0:
                var dc = _poisoned(ctx, m * n)
                _trace(ptag + " fused")
                identical_gemm_int8_pieces_amd_fused_with_plan(
                    ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k, plan
                )
                ctx.synchronize()
                var outc = _download_f32(ctx, dc, m * n, ptag + " fused")
                got = _diff(outc, want_c, ptag + " (amd plan, fused form, vs host seam)")
                _ = dc
        except e:
            got = ptag + ": " + String(e)
        # Every four-product plan's window is 64 steps but the first's,
        # whose loads are eight bytes.
        var window = 64
        if plan == 0:
            window = 32
        reach.note(_can_reach(int8_amd_pieces_plan_is_direct(plan), window, k), got, ptag)
        tally.note(got)
        _ = ds
    # The fused form on the plan the AMD launcher names.
    var dispatched: String
    try:
        var dc = _poisoned(ctx, m * n)
        _trace(tag + " fused dispatched")
        identical_gemm_int8_pieces_amd_fused_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k)
        ctx.synchronize()
        var outc = _download_f32(ctx, dc, m * n, tag + " fused dispatched")
        dispatched = _diff(outc, want_c, tag + " fused dispatched (vs host seam)")
        _ = dc
    except e:
        dispatched = tag + " fused dispatched: " + String(e)
    reach.note(
        _can_reach(
            int8_amd_pieces_plan_is_direct(int8_amd_pieces_dispatch(m, n, k)),
            32 if int8_amd_pieces_dispatch(m, n, k) == 0 else 64,
            k,
        ),
        dispatched,
        tag + " fused dispatched",
    )
    tally.note(dispatched)
    _ = dea
    _ = deb
    _ = dah
    _ = dal
    _ = dbh
    _ = dbl
    _ = dflat


def check_amd_pieces_match_flat_and_host(ctx: DeviceContext, mut reach: Reach) raises:
    """GATE: on hashed planes of both fixtures (the fifteen-bit profile's
    ranges, and any int8 on all four planes) every four-product plan
    returns the host's three sums and the reference device plan's, at the
    ragged shapes of `gemm_lowbit_check.mojo` and of the tuned gate."""
    var tally = Tally()
    var shapes = 0
    for s in range(MMA_SHAPE_COUNT + TUNED_SHAPE_COUNT):
        var m = 0
        var n = 0
        var k = 0
        if s < MMA_SHAPE_COUNT:
            var sh = _mma_shape(s)
            m = sh[0]
            n = sh[1]
            k = sh[2]
        else:
            var sh = _tuned_shape(s - MMA_SHAPE_COUNT)
            m = sh[0]
            n = sh[1]
            k = sh[2]
        for fixture in range(2):
            var low = fixture == 0
            var ah = _plane(m * k, 701 + s, False)
            var al = _plane(m * k, 709 + s, low)
            var bh = _plane(n * k, 719 + s, False)
            var bl = _plane(n * k, 727 + s, low)
            var tag = (
                "int8 amd pieces " + String(m) + "x" + String(n) + "x" + String(k)
                + (" profile-ranges" if low else " any-int8")
            )
            var before = tally.failed
            _run_every_amd_pieces_plan(ctx, ah, al, bh, bl, m, n, k, tag, tally, reach)
            if tally.failed == before:
                print("   ok " + tag)
        shapes += 1
    print("   shapes: " + String(shapes) + ", two fixtures each")
    _verdict(tally, String("hashed-plane"))


def check_amd_pieces_planted_worst_cases(ctx: DeviceContext, mut reach: Reach) raises:
    """GATE: the four-product gate's plants through every AMD plan: every
    code -128 on all four planes at `k` 65535, where the middle sum is
    2147450880, the largest any input gives; high planes -128 and low
    planes +127 at the largest `k` the kernel admits, 65536, where HL + LH
    is -2130706432, nearest its Int32 bound for the operands the bound is
    stated for; and each plane alone."""
    var tally = Tally()
    for s in range(PIECES_PLANT_SHAPE_COUNT):
        var sh = _pieces_plant_shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        for plant in range(PIECES_PLANT_COUNT):
            if plant == PIECES_PLANT_ALL_MIN and k > INT8_PIECES_MAX_K_ANY_INT8:
                print(
                    "   -- " + _pieces_plant_name(plant) + " is not run at k = " + String(k)
                    + ": its low planes are outside [0, 127]"
                )
                continue
            var ah = _constant_plane(m * k, _pieces_plant_value(plant, 0))
            var al = _constant_plane(m * k, _pieces_plant_value(plant, 1))
            var bh = _constant_plane(n * k, _pieces_plant_value(plant, 2))
            var bl = _constant_plane(n * k, _pieces_plant_value(plant, 3))
            # The four-product gate's four odd codes, at its positions.
            ah[(m - 1) * k + k // 3] = Int8(5)
            al[(m // 2) * k + k - 1] = Int8(7)
            bh[(n - 1) * k] = Int8(-3)
            bl[(n // 2) * k + k // 2] = Int8(9)
            var tag = (
                "int8 amd pieces planted " + _pieces_plant_name(plant) + " " + String(m)
                + "x" + String(n) + "x" + String(k)
            )
            var before = tally.failed
            _run_every_amd_pieces_plan(ctx, ah, al, bh, bl, m, n, k, tag, tally, reach)
            if tally.failed == before:
                print("   ok " + tag)
    _verdict(tally, String("planted"))


def check_amd_pieces_refuses_above_its_bound(ctx: DeviceContext) raises:
    """GATE: one step above `INT8_PIECES_MAX_K` is refused BY NAME by the
    AMD launcher before anything is launched, and the launcher's two plans
    are two."""
    var k = INT8_PIECES_MAX_K + 1
    var dah = _upload_i8(ctx, _constant_plane(k, 1))
    var dal = _upload_i8(ctx, _constant_plane(k, 1))
    var dbh = _upload_i8(ctx, _constant_plane(k, 1))
    var dbl = _upload_i8(ctx, _constant_plane(k, 1))
    var ds = _poisoned_sums(ctx, 3)
    var refused = False
    try:
        identical_gemm_int8_pieces_amd_into(ctx, ds, dah, dal, dbh, dbl, 1, 1, k)
    except e:
        refused = String(e).find(String(INT8_PIECES_MAX_K)) >= 0
    if not refused:
        raise Error("the AMD launcher did not refuse k = " + String(k) + " by name")
    if int8_amd_pieces_dispatch(512, 4096, 4096) == int8_amd_pieces_dispatch(1, 4096, 4096):
        raise Error("the AMD launcher takes one plan at 1 row and at 512")
    ctx.synchronize()
    _ = dah
    _ = dal
    _ = dbh
    _ = dbl
    _ = ds
    print("   ok k = " + String(k) + " refused by name; bound " + String(INT8_PIECES_MAX_K))


def check_amd_sabotage_reach(reach: Reach) raises:
    """GATE: what the build's scheduling arm did against what it can do. In
    a build with no scheduling arm nothing is reachable and nothing may
    differ. In a build with one, a case it cannot reach (the other kind of
    plan, or a `k` of whole windows) must agree, and the cases it can reach
    must not ALL agree: an arm that changes nothing is no arm."""
    print(
        "   reach: " + String(reach.reachable) + " cases the arm can reach, "
        + String(reach.reachable_failed) + " of them differed; "
        + String(reach.unreachable_failed) + " cases it cannot reach differed"
    )
    var armed = False
    comptime if REACH_DIRECT:
        armed = True
    elif REACH_STAGED:
        armed = True
    if not armed:
        # The value arm and the fragment arm reach every case and are read
        # by the gates above; this gate has nothing to say of them.
        return
    if reach.unreachable_failed > 0:
        raise Error(
            String(reach.unreachable_failed) + " cases the scheduling arm cannot"
            " reach differed; first: " + reach.first
        )
    if reach.reachable == 0 or reach.reachable_failed == 0:
        raise Error(
            "the scheduling arm changed no case of the "
            + String(reach.reachable) + " it can reach"
        )


def main() raises:
    print(
        "== gemm/checks/gemm_int8_mma_amd_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + int8_amd_sabotage_name() + " =="
    )
    print("   profile: mojolearn.identical.gemm.int8i32.v1, the AMD plans, one product and four")
    print("   column: " + column_name(TARGET_COLUMN))
    var ran = 0
    var failed = 0
    comptime if not has_accelerator():
        print("   no accelerator: the device gates DID NOT RUN, which is not a pass")
        raise Error("the AMD int8 gate needs a device")
    else:
        comptime if HAS_AMD:
            var ctx = DeviceContext()
            var reach = Reach()
            for plan in range(INT8_AMD_PLAN_COUNT):
                print("   plan " + String(plan) + ": " + int8_amd_plan_name(plan))
            for plan in range(INT8_AMD_PIECES_PLAN_COUNT):
                print(
                    "   four-product plan " + String(plan) + ": "
                    + int8_amd_pieces_plan_name(plan)
                )
            try:
                check_amd_plans_match_reference_flat_oracle(ctx, reach)
                _gate(String("check_amd_plans_match_reference_flat_oracle"), ran, failed, String(""))
            except e:
                _gate(String("check_amd_plans_match_reference_flat_oracle"), ran, failed, String(e))
            try:
                check_amd_planted_worst_cases(ctx, reach)
                _gate(String("check_amd_planted_worst_cases"), ran, failed, String(""))
            except e:
                _gate(String("check_amd_planted_worst_cases"), ran, failed, String(e))
            try:
                check_amd_minus_128_piece(ctx, reach)
                _gate(String("check_amd_minus_128_piece"), ran, failed, String(""))
            except e:
                _gate(String("check_amd_minus_128_piece"), ran, failed, String(e))
            try:
                check_amd_dispatch_is_batch_invariant(ctx)
                _gate(String("check_amd_dispatch_is_batch_invariant"), ran, failed, String(""))
            except e:
                _gate(String("check_amd_dispatch_is_batch_invariant"), ran, failed, String(e))
            try:
                check_amd_pieces_match_flat_and_host(ctx, reach)
                _gate(String("check_amd_pieces_match_flat_and_host"), ran, failed, String(""))
            except e:
                _gate(String("check_amd_pieces_match_flat_and_host"), ran, failed, String(e))
            try:
                check_amd_pieces_planted_worst_cases(ctx, reach)
                _gate(String("check_amd_pieces_planted_worst_cases"), ran, failed, String(""))
            except e:
                _gate(String("check_amd_pieces_planted_worst_cases"), ran, failed, String(e))
            try:
                check_amd_pieces_refuses_above_its_bound(ctx)
                _gate(String("check_amd_pieces_refuses_above_its_bound"), ran, failed, String(""))
            except e:
                _gate(String("check_amd_pieces_refuses_above_its_bound"), ran, failed, String(e))
            try:
                check_amd_sabotage_reach(reach)
                _gate(String("check_amd_sabotage_reach"), ran, failed, String(""))
            except e:
                _gate(String("check_amd_sabotage_reach"), ran, failed, String(e))
            print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
            if failed > 0:
                raise Error(String(failed) + " of " + String(ran) + " gates failed")
        else:
            # Not a pass: the gates did not run.
            print(
                "   THE AMD GATES DID NOT RUN, which is not a pass: column "
                + column_name(TARGET_COLUMN) + " is not AMD"
            )
            raise Error("the AMD int8 gate needs the AMD column")
