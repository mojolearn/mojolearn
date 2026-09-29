# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate of FOUR PRODUCTS, ONE STAGING: the three Int32 sums of every
plan of `identical_gemm_int8_pieces_tuned_kernel` against the reference
device plan (one thread per cell) and the host's integers.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_int8_pieces_tuned_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_PIECES_SABOTAGE=1 -I . gemm/checks/gemm_int8_pieces_tuned_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_TUNED_SABOTAGE=1 -I . gemm/checks/gemm_int8_pieces_tuned_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_SABOTAGE=1 -I . gemm/checks/gemm_int8_pieces_tuned_check.mojo

The first must pass every gate. The second pairs the wrong fragments (the
middle sum takes HL twice and never LH) and must FAIL the two gates that
compare sums. The third breaks the staging's padding rule and must FAIL the
same two, at the ragged `k`. The fourth flips the lowest bit of every sum
stored, by the reference device plan too, and must FAIL them against the
host. `check_pieces_refuses_above_its_bound` reads no sum and passes under
all four. `tools/lowbit_mma_speed/gate_job.sh pieces` runs the four.

Lane lane/lowbit-mma-speed, 2026-09-29. Kernel
`gemm/checks/gemm_int8_mma_tuned.mojo`. The answer is this file's
`piece_sums_host`: three sums of products of int8 in Int64, each checked to
be an Int32. There is no float in the kernel and none here.

WHAT IS PLANTED. The kernel's claim is that its three accumulators hold HH,
HL + LH and LL of the right cell, read from the right plane. So:

    two fixtures     planes of the fifteen-bit profile's ranges (high in
                     [-128, 127], low in [0, 127]) and planes of any int8
                     on both sides; the four planes differ at every
                     position, so a fragment read from the wrong plane, or
                     paired with the wrong one, changes a sum;
    ragged extents   lane/lowbit-units' and the one-product gate's;
    the bound        high planes -128 and low planes +127 at the largest
                     `k` the kernel admits, 65536, the fifteen-bit
                     profile's own: the middle sum is -2130706432 and HH is
                     2^30; and every code -128 on ALL FOUR planes (outside
                     the operands the bound is stated for, inside any
                     int8's) at 65535, where the middle sum is 2147450880,
                     the largest any input gives;
    one plane alone  three planes of zero codes and one of +127, in turn,
                     so each plane's path to each sum is seen alone.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import has_accelerator

from checks.kernel_matrix import TARGET_COLUMN, column_name, lib_int8_matrix_unit_for
from checks.numerics import numeric_mode_name
from gemm.checks.gemm_int8_mma_tuned import (
    INT8_PIECES_MAX_K,
    INT8_PIECES_MAX_K_ANY_INT8,
    INT8_PIECES_PLAN_COUNT,
    identical_gemm_int8_pieces_flat_into,
    identical_gemm_int8_pieces_tuned_fused_with_plan,
    identical_gemm_int8_pieces_tuned_into,
    identical_gemm_int8_pieces_tuned_with_plan,
    int8_pieces_dispatch,
    int8_pieces_plan_name,
    int8_pieces_sabotage_name,
)
from gemm.checks.gemm_int8_mma_tuned_check import TUNED_SHAPE_COUNT, _tuned_shape
from checks.numerics_int15 import dequant_int15_pinned, int15_recombine
from gemm.checks.gemm_lowbit_check import (
    MMA_SHAPE_COUNT,
    _download_f32,
    _download_i32,
    _poisoned,
    _gate,
    _hash64,
    _mma_shape,
    _upload_i32,
    _upload_i8,
)
from gemm.checks.quantize_int8_par_check import Tally, _diff, _verdict

comptime HAS_UNIT = lib_int8_matrix_unit_for[TARGET_COLUMN]()

#: What no sum is: the middle sum's largest magnitude is 2147450880.
comptime SUM_POISON = Int32(2147483647)

comptime FIXTURE_PROFILE = 0  #: high in [-128, 127], low in [0, 127]
comptime FIXTURE_ANY = 1  #: any int8 on both planes


def _plane(count: Int, salt: Int, low: Bool) -> List[Int8]:
    """A plane of hashed codes: any int8, or `[0, 127]` when `low`."""
    var q = List[Int8]()
    for i in range(count):
        var h = Int(_hash64(i, salt) & UInt64(0xFF))
        if low:
            q.append(Int8(h & 127))
        else:
            q.append(Int8(h - 128))
    return q^


def _constant_plane(count: Int, value: Int) -> List[Int8]:
    var q = List[Int8]()
    for _ in range(count):
        q.append(Int8(value))
    return q^


def piece_sums_host(
    ah: List[Int8],
    al: List[Int8],
    bh: List[Int8],
    bl: List[Int8],
    m: Int,
    n: Int,
    k: Int,
) raises -> List[Int32]:
    """THE ANSWER: per cell HH, HL + LH, LL, summed in Int64 with `p`
    ascending, each refused if it is not an Int32."""
    var out = List[Int32]()
    for i in range(m):
        for j in range(n):
            var hh = 0
            var mid = 0
            var ll = 0
            for p in range(k):
                var a_hi = Int(ah[i * k + p])
                var a_lo = Int(al[i * k + p])
                var b_hi = Int(bh[j * k + p])
                var b_lo = Int(bl[j * k + p])
                hh += a_hi * b_hi
                mid += a_hi * b_lo + a_lo * b_hi
                ll += a_lo * b_lo
            for v in [hh, mid, ll]:
                if v > 2147483647 or v < -2147483648:
                    raise Error("piece_sums_host: a sum left the Int32: " + String(v))
                out.append(Int32(v))
    return out^


def _exponents(count: Int, salt: Int) -> List[Int32]:
    """Row exponents in [-40, 9], hashed: the fused form's scale."""
    var e = List[Int32]()
    for i in range(count):
        e.append(Int32(Int(_hash64(i, salt) % UInt64(50)) - 40))
    return e^


def fused_host(
    sums: List[Int32], ea: List[Int32], eb: List[Int32], m: Int, n: Int
) -> List[Float32]:
    """What the FUSED form stores per cell: lane/lowbit-int15's
    `int15_store_cell` rule on the host's three sums, spelled with the same
    seams (`int15_recombine`, `dequant_int15_pinned`), scaled by
    `ea[i] + eb[j]`. No sabotage arm reaches it."""
    var out = List[Float32]()
    for i in range(m):
        for j in range(n):
            var at_ = 3 * (i * n + j)
            var v = int15_recombine(sums[at_], sums[at_ + 1], sums[at_ + 2])
            out.append(dequant_int15_pinned(v, Int(ea[i]) + Int(eb[j])))
    return out^


def _sums_diff(got: List[Int32], want: List[Int32], n: Int, tag: String) -> String:
    """Empty when every sum agrees; else the count and the first."""
    var bad = 0
    var first = -1
    for i in range(len(want)):
        if got[i] != want[i]:
            bad += 1
            if first < 0:
                first = i
    if bad == 0:
        return String("")
    var cell = first // 3
    var piece = first - 3 * cell
    var name = String("HH")
    if piece == 1:
        name = String("HL+LH")
    elif piece == 2:
        name = String("LL")
    return (
        tag + ": " + String(bad) + " of " + String(len(want)) + " sums differ; first "
        + name + " of cell (" + String(cell // n) + ", " + String(cell % n) + ") got "
        + String(got[first]) + " want " + String(want[first])
    )


def _poisoned_sums(ctx: DeviceContext, count: Int) raises -> DeviceBuffer[DType.int32]:
    var h = List[Int32]()
    for _ in range(count):
        h.append(SUM_POISON)
    return _upload_i32(ctx, h)


def _read_sums(
    ctx: DeviceContext, mut d: DeviceBuffer[DType.int32], count: Int, tag: String
) raises -> List[Int32]:
    var out = _download_i32(ctx, d, count)
    for i in range(count):
        if out[i] == SUM_POISON:
            raise Error("POISON SURVIVED at sum " + String(i) + " of " + tag)
    return out^


def _run_every_pieces_plan(
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
) raises:
    """One shape and four planes through the reference device plan and, on
    a column that has the unit, every staged plan, each against the host
    and the staged ones against the reference device plan too."""
    var want = piece_sums_host(ah, al, bh, bl, m, n, k)
    var ea = _exponents(m, 733 + k)
    var eb = _exponents(n, 739 + k)
    var fwant = fused_host(want, ea, eb, m, n)
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
        identical_gemm_int8_pieces_flat_into(ctx, dflat, dah, dal, dbh, dbl, m, n, k)
        ctx.synchronize()
        flat = _read_sums(ctx, dflat, 3 * m * n, tag + " flat")
        flat_ok = True
        verdict = _sums_diff(flat, want, n, tag + " (reference device plan vs host)")
    except e:
        verdict = tag + " flat: " + String(e)
    tally.note(verdict)
    comptime if HAS_UNIT:
        for plan in range(INT8_PIECES_PLAN_COUNT):
            var ptag = tag + " " + int8_pieces_plan_name(plan)
            var ds = _poisoned_sums(ctx, 3 * m * n)
            var got: String
            try:
                identical_gemm_int8_pieces_tuned_with_plan(
                    ctx, ds, dah, dal, dbh, dbl, m, n, k, plan
                )
                ctx.synchronize()
                var out = _read_sums(ctx, ds, 3 * m * n, ptag)
                got = _sums_diff(out, want, n, ptag + " (staged vs host)")
                if got.byte_length() == 0 and flat_ok:
                    got = _sums_diff(out, flat, n, ptag + " (staged vs reference device plan)")
            except e:
                got = ptag + ": " + String(e)
            tally.note(got)
            _ = ds
            # THE FUSED FORM of the same plan: one launch, the stand-in
            # epilogue's float32 per cell, against the host's.
            var ftag = tag + " fused." + int8_pieces_plan_name(plan)
            var dc = _poisoned(ctx, m * n)
            var fgot: String
            try:
                identical_gemm_int8_pieces_tuned_fused_with_plan(
                    ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k, plan
                )
                ctx.synchronize()
                var cout = _download_f32(ctx, dc, m * n, ftag)
                fgot = _diff(cout, fwant, ftag + " (fused vs host)")
            except e:
                fgot = ftag + ": " + String(e)
            tally.note(fgot)
            _ = dc
    _ = dah
    _ = dal
    _ = dbh
    _ = dbl
    _ = dflat
    _ = dea
    _ = deb


def check_pieces_plans_match_flat_and_host(ctx: DeviceContext) raises:
    """GATE: on hashed planes of both fixtures every staged plan returns the
    host's three sums and the reference device plan's, at the ragged shapes
    of `gemm_lowbit_check.mojo` and of the one-product gate."""
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
            var low = fixture == FIXTURE_PROFILE
            var ah = _plane(m * k, 701 + s, False)
            var al = _plane(m * k, 709 + s, low)
            var bh = _plane(n * k, 719 + s, False)
            var bl = _plane(n * k, 727 + s, low)
            var tag = (
                "int8 pieces " + String(m) + "x" + String(n) + "x" + String(k)
                + (" profile-ranges" if low else " any-int8")
            )
            var before = tally.failed
            _run_every_pieces_plan(ctx, ah, al, bh, bl, m, n, k, tag, tally)
            if tally.failed == before:
                print("   ok " + tag)
        shapes += 1
    print("   shapes: " + String(shapes) + ", two fixtures each")
    _verdict(tally, String("hashed-plane"))


comptime PIECES_PLANT_ALL_MIN = 0  #: every code -128 on all four planes
comptime PIECES_PLANT_LOW_MAX = 1  #: high planes -128, low planes +127
comptime PIECES_PLANT_ONLY_AH = 2  #: +127 on one plane, zero codes on three
comptime PIECES_PLANT_ONLY_AL = 3
comptime PIECES_PLANT_ONLY_BH = 4
comptime PIECES_PLANT_ONLY_BL = 5
comptime PIECES_PLANT_COUNT = 6


def _pieces_plant_name(plant: Int) -> String:
    if plant == PIECES_PLANT_ALL_MIN:
        return String("all-minus-128")
    if plant == PIECES_PLANT_LOW_MAX:
        return String("high-minus-128-low-plus-127")
    if plant == PIECES_PLANT_ONLY_AH:
        return String("only-a-high")
    if plant == PIECES_PLANT_ONLY_AL:
        return String("only-a-low")
    if plant == PIECES_PLANT_ONLY_BH:
        return String("only-b-high")
    return String("only-b-low")


def _pieces_plant_value(plant: Int, plane: Int) -> Int:
    """The constant code of plane 0 (A high), 1 (A low), 2 (B high) or 3
    (B low) under the plant."""
    if plant == PIECES_PLANT_ALL_MIN:
        return -128
    if plant == PIECES_PLANT_LOW_MAX:
        if plane == 1 or plane == 3:
            return 127
        return -128
    if plane == plant - PIECES_PLANT_ONLY_AH:
        return 127
    return 0


comptime PIECES_PLANT_SHAPE_COUNT = 8


def _pieces_plant_shape(i: Int) -> Tuple[Int, Int, Int]:
    """Off every tile and window of the plans, and the two largest `k`."""
    if i == 0:
        return (9, 7, 1025)
    if i == 1:
        return (65, 67, 4096)
    if i == 2:
        return (1, 130, 4097)
    if i == 3:
        return (8, 129, 14336)
    if i == 4:
        return (2, 3, INT8_PIECES_MAX_K_ANY_INT8)
    if i == 5:
        return (129, 257, 96)
    if i == 6:
        return (17, 33, 4100)
    return (3, 2, INT8_PIECES_MAX_K)


def check_pieces_planted_worst_cases(ctx: DeviceContext) raises:
    """GATE: the planted planes through every plan, against the host and
    the reference device plan. Every case is run and printed; the gate
    raises after the last one."""
    var tally = Tally()
    for s in range(PIECES_PLANT_SHAPE_COUNT):
        var sh = _pieces_plant_shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        for plant in range(PIECES_PLANT_COUNT):
            if plant == PIECES_PLANT_ALL_MIN and k > INT8_PIECES_MAX_K_ANY_INT8:
                # Low planes of -128 are outside the operands the larger
                # bound is stated for: at this `k` their middle sum is 2^31.
                print(
                    "   -- " + _pieces_plant_name(plant) + " is not run at k = " + String(k)
                    + ": its low planes are outside [0, 127]"
                )
                continue
            var ah = _constant_plane(m * k, _pieces_plant_value(plant, 0))
            var al = _constant_plane(m * k, _pieces_plant_value(plant, 1))
            var bh = _constant_plane(n * k, _pieces_plant_value(plant, 2))
            var bl = _constant_plane(n * k, _pieces_plant_value(plant, 3))
            # One code of each plane differs from its constant, at a row
            # and a step that differ per plane, so a cell is not every
            # other cell. Each is smaller in magnitude than the constant it
            # replaces or sits in a plane of zero codes, so no sum grows.
            ah[(m - 1) * k + k // 3] = Int8(5)
            al[(m // 2) * k + k - 1] = Int8(7)
            bh[(n - 1) * k] = Int8(-3)
            bl[(n // 2) * k + k // 2] = Int8(9)
            var tag = (
                "int8 pieces planted " + _pieces_plant_name(plant) + " " + String(m) + "x"
                + String(n) + "x" + String(k)
            )
            var before = tally.failed
            _run_every_pieces_plan(ctx, ah, al, bh, bl, m, n, k, tag, tally)
            if tally.failed == before:
                print("   ok " + tag)
    _verdict(tally, String("planted"))


def check_pieces_refuses_above_its_bound(ctx: DeviceContext) raises:
    """GATE: one step above `INT8_PIECES_MAX_K` is refused BY NAME, by the
    reference device plan and by the staged launcher, before anything is
    launched; and the launcher's two plans are two."""
    var k = INT8_PIECES_MAX_K + 1
    var dah = _upload_i8(ctx, _constant_plane(k, 1))
    var dal = _upload_i8(ctx, _constant_plane(k, 1))
    var dbh = _upload_i8(ctx, _constant_plane(k, 1))
    var dbl = _upload_i8(ctx, _constant_plane(k, 1))
    var ds = _poisoned_sums(ctx, 3)
    var refused = False
    try:
        identical_gemm_int8_pieces_flat_into(ctx, ds, dah, dal, dbh, dbl, 1, 1, k)
    except e:
        refused = String(e).find(String(INT8_PIECES_MAX_K)) >= 0
    if not refused:
        raise Error("the reference device plan did not refuse k = " + String(k) + " by name")
    comptime if HAS_UNIT:
        refused = False
        try:
            identical_gemm_int8_pieces_tuned_into(ctx, ds, dah, dal, dbh, dbl, 1, 1, k)
        except e:
            refused = String(e).find(String(INT8_PIECES_MAX_K)) >= 0
        if not refused:
            raise Error("the staged launcher did not refuse k = " + String(k) + " by name")
        if int8_pieces_dispatch(512, 4096, 4096) == int8_pieces_dispatch(1, 4096, 4096):
            raise Error("the launcher takes one plan at 1 row and at 512")
    ctx.synchronize()
    var left = _download_i32(ctx, ds, 3)
    for i in range(3):
        if left[i] != SUM_POISON:
            raise Error("a refused launch wrote a sum")
    _ = dah
    _ = dal
    _ = dbh
    _ = dbl
    _ = ds
    print("   ok k = " + String(k) + " refused by name; bound " + String(INT8_PIECES_MAX_K))


def main() raises:
    print(
        "== gemm/checks/gemm_int8_pieces_tuned_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + int8_pieces_sabotage_name() + " =="
    )
    print("   four int8 products of two-plane operands, one staging; three Int32 sums per cell")
    print("   column: " + column_name(TARGET_COLUMN))
    var ran = 0
    var failed = 0
    comptime if not has_accelerator():
        print("   no accelerator: the device gates DID NOT RUN, which is not a pass")
        raise Error("the four-product gate needs a device")
    else:
        var ctx = DeviceContext()
        comptime if HAS_UNIT:
            for plan in range(INT8_PIECES_PLAN_COUNT):
                print("   plan " + String(plan) + ": " + int8_pieces_plan_name(plan))
        else:
            print(
                "   THE STAGED PLANS DID NOT RUN, which is not a pass: column "
                + column_name(TARGET_COLUMN) + " has no int8 matrix unit. The reference"
                + " device plan runs against the host."
            )
        try:
            check_pieces_plans_match_flat_and_host(ctx)
            _gate(String("check_pieces_plans_match_flat_and_host"), ran, failed, String(""))
        except e:
            _gate(String("check_pieces_plans_match_flat_and_host"), ran, failed, String(e))
        try:
            check_pieces_planted_worst_cases(ctx)
            _gate(String("check_pieces_planted_worst_cases"), ran, failed, String(""))
        except e:
            _gate(String("check_pieces_planted_worst_cases"), ran, failed, String(e))
        try:
            check_pieces_refuses_above_its_bound(ctx)
            _gate(String("check_pieces_refuses_above_its_bound"), ran, failed, String(""))
        except e:
            _gate(String("check_pieces_refuses_above_its_bound"), ran, failed, String(e))
        print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
        if failed > 0:
            raise Error(String(failed) + " of " + String(ran) + " gates failed")
