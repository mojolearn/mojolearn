# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate of the Apple exact-chunk probe: int8 codes held as float32 on
Metal's float matrix unit, against the flat int8 kernel and the host oracle.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_int8_apple_chunk_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_APPLE_CHUNK_SABOTAGE=1 -I . gemm/checks/gemm_int8_apple_chunk_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_SABOTAGE=1 -I . gemm/checks/gemm_int8_apple_chunk_check.mojo

The first must pass every gate. The second removes the chunk boundary and
must FAIL `check_apple_chunk_planted_worst_cases`; the third flips every
stored value and must FAIL both device gates that compare with the oracle.
`tools/lowbit_units/chunk_job.sh` runs the three and reads the logs.

Lane lane/lowbit-units, 2026-09-29. Kernel
`gemm/checks/gemm_int8_apple_chunk.mojo`; answer
`gemm/host/gemm_lowbit_oracle.mojo::gemm_int8_oracle`; the flat kernel
`gemm/checks/gemm_lowbit.mojo::identical_gemm_int8_flat_into`.

WHY THE WORST CASES ARE PLANTED. The probe's claim is that no partial sum
inside a chunk leaves the integers a float32 holds. Codes quantized from
random floats cannot test it: their products cancel, a sum over k = 4096 of
them stays near 10^5, and a kernel with NO chunk boundary passes every
random fixture (the second build above shows it, gate by gate). So the gate
plants the codes that reach the bound:

    all-positive     every product +16129
    all-negative     every product -16129
    cancel-halves    +16129 for the first half of k, -16129 for the rest:
                     the partial sum peaks in the middle and the answer is
                     0 (k even) or -16129 (k odd)
    odd-windows      +16129 except one +16002 in every eight steps, so the
                     sum of any whole 8-step fragment is ODD: a unit that
                     summed a fragment before it added it would still have
                     to round past 2^24
    checkerboard     the sign of a row alternates with i and of a column
                     with j, so neighbouring cells of one fragment hold
                     +max and -max together

at k on both sides of the chunk boundary (1024, 1025, 1040, 1041, 1042, 2081), at
the transformer widths (4096, 4097, 14336) and at the profile's largest k
(131072, where the Int32 sum is 2114060288), with m and n off the 8-wide
fragment, the 64-wide WIDE tile and the 128-wide ROW tile.

WHERE A MISSING BOUNDARY FIRST SHOWS (m2pro, 2026-09-29, the boundary
removed). Not at k = 1041: there the float chain rounds ONCE, at its last
step, and one rounding of the exact sum is what the epilogue's own
`i32_to_f32_pinned` does to the Int32, so the stored cell is the same. It
shows from k = 1042, where the chain has rounded twice (16129 * 1042 =
16806418 is a float32 and the chain holds 16806416). The k = 1042 shape is
planted for that reason; cancel-halves at k = 2081 does not show it either,
its partial sum peaking at 16129 * 1040.

MAIN RUNS EVERY GATE AND REPORTS EVERY VERDICT before it raises, and the
planted gate reports EVERY case before it raises, as
`gemm_lowbit_check.mojo` does and for the same reason: under a sabotage
build the evidence is which cases the defect reaches.
"""

from max.gpu.host import DeviceContext
from std.sys import has_accelerator

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN, column_name
from checks.numerics import numeric_mode_name
from gemm.checks.gemm_int8_apple_chunk import (
    F32_EXACT_INTEGER_BOUND,
    INT8_APPLE_CHUNK_GEOMETRY_ROW,
    INT8_APPLE_CHUNK_GEOMETRY_WIDE,
    INT8_APPLE_CHUNK_KB,
    INT8_APPLE_CHUNK_MAX_STEPS,
    INT8_APPLE_CHUNK_WINDOWS,
    INT8_PRODUCT_MAX,
    identical_gemm_int8_apple_chunk_with_geometry,
    int8_apple_chunk_geometry,
    int8_apple_chunk_geometry_name,
    int8_apple_chunk_sabotage_name,
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
from gemm.contract import INT8_MAX_K
from gemm.host.gemm_lowbit_oracle import gemm_int8_oracle, quantize_rows_int8
from gemm.contract import OP_NT

comptime PLANT_ALL_POSITIVE = 0
comptime PLANT_ALL_NEGATIVE = 1
comptime PLANT_CANCEL_HALVES = 2
comptime PLANT_ODD_WINDOWS = 3
comptime PLANT_CHECKERBOARD = 4
comptime PLANT_COUNT = 5


def _plant_name(plant: Int) -> String:
    if plant == PLANT_ALL_POSITIVE:
        return String("all-positive")
    if plant == PLANT_ALL_NEGATIVE:
        return String("all-negative")
    if plant == PLANT_CANCEL_HALVES:
        return String("cancel-halves")
    if plant == PLANT_ODD_WINDOWS:
        return String("odd-windows")
    return String("checkerboard")


def _plant_a(plant: Int, m: Int, k: Int) -> List[Int8]:
    """The left operand's codes: +127 everywhere, and under the
    checkerboard -127 on every odd row."""
    var q = List[Int8]()
    for i in range(m):
        var v = Int8(127)
        if plant == PLANT_CHECKERBOARD and i % 2 == 1:
            v = Int8(-127)
        for _ in range(k):
            q.append(v)
    return q^


def _plant_b(plant: Int, n: Int, k: Int) -> List[Int8]:
    """The right operand's codes, which carry the pattern."""
    var q = List[Int8]()
    for j in range(n):
        for p in range(k):
            var v = Int8(127)
            if plant == PLANT_ALL_NEGATIVE:
                v = Int8(-127)
            elif plant == PLANT_CANCEL_HALVES:
                if p >= k // 2:
                    v = Int8(-127)
            elif plant == PLANT_ODD_WINDOWS:
                if p % 8 == 0:
                    v = Int8(126)
            elif plant == PLANT_CHECKERBOARD:
                if j % 3 == 0:
                    v = Int8(-127)
            q.append(v)
    return q^


def _plant_exponents(rows: Int, base: Int, period: Int) -> List[Int32]:
    """Row exponents that differ from row to row, so the epilogue reads the
    right row's and the right column's."""
    var e = List[Int32]()
    for r in range(rows):
        e.append(Int32(base + r % period))
    return e^


#: The planted shapes: k on both sides of the chunk boundary, the
#: transformer widths, the profile's largest k; m and n off every tile edge.
comptime PLANT_SHAPE_COUNT = 11


def _plant_shape(i: Int) -> Tuple[Int, Int, Int]:
    if i == 0:
        return (9, 7, 1024)
    if i == 1:
        return (9, 7, 1025)
    if i == 2:
        return (3, 5, 1040)
    if i == 3:
        return (3, 5, 1041)
    if i == 4:
        return (70, 9, 2081)
    if i == 5:
        return (65, 67, 4096)
    if i == 6:
        return (1, 130, 4097)
    if i == 7:
        return (8, 129, 14336)
    if i == 8:
        return (2, 3, INT8_MAX_K)
    if i == 9:
        return (3, 5, 1042)
    return (1, 1, 1041)


#: The random-fixture shapes this file adds to `gemm_lowbit_check.mojo`'s:
#: k around the chunk boundary and the transformer width, on both tiles.
comptime CHUNK_SHAPE_COUNT = 6


def _chunk_shape(i: Int) -> Tuple[Int, Int, Int]:
    if i == 0:
        return (5, 70, 1023)
    if i == 1:
        return (5, 70, 1024)
    if i == 2:
        return (5, 70, 1025)
    if i == 3:
        return (66, 130, 2049)
    if i == 4:
        return (8, 128, 4096)
    return (9, 129, 4096)


def _run_both(
    ctx: DeviceContext,
    qa: List[Int8],
    ea: List[Int32],
    qb: List[Int8],
    eb: List[Int32],
    m: Int,
    n: Int,
    k: Int,
    geometry: Int,
    tag: String,
) raises -> Tuple[List[Float32], List[Float32]]:
    """The flat kernel and the probe on the same codes: (flat, chunk)."""
    var dqa = _upload_i8(ctx, qa)
    var dea = _upload_i32(ctx, ea)
    var dqb = _upload_i8(ctx, qb)
    var deb = _upload_i32(ctx, eb)
    var dflat = _poisoned(ctx, m * n)
    var dchunk = _poisoned(ctx, m * n)
    identical_gemm_int8_flat_into(ctx, dflat, dqa, dea, dqb, deb, m, n, k)
    identical_gemm_int8_apple_chunk_with_geometry(ctx, dchunk, dqa, dea, dqb, deb, m, n, k, geometry)
    ctx.synchronize()
    var flat = _download_f32(ctx, dflat, m * n, tag + " flat")
    var chunk = _download_f32(ctx, dchunk, m * n, tag + " chunk")
    _ = dqa
    _ = dea
    _ = dqb
    _ = deb
    _ = dflat
    _ = dchunk
    return (flat^, chunk^)


# ===========================================================================
# THE BOUND, ON THE HOST
# ===========================================================================


def check_chunk_bound_is_exact_and_tight() raises:
    """GATE: the chunk the kernel runs keeps every partial sum a float32,
    and one step past `INT8_APPLE_CHUNK_MAX_STEPS` does not. Integer
    arithmetic on the constants, then the same fact as a float32 chain on
    the host: `+16129` added 1040 times is the integer, and the 1041st sum,
    16790289, is odd and above 2^24, so no float32 holds it."""
    var steps = INT8_APPLE_CHUNK_WINDOWS * INT8_APPLE_CHUNK_KB
    if steps > INT8_APPLE_CHUNK_MAX_STEPS:
        raise Error("the kernel's chunk, " + String(steps) + " steps, is above the bound " + String(INT8_APPLE_CHUNK_MAX_STEPS))
    if INT8_PRODUCT_MAX * steps >= F32_EXACT_INTEGER_BOUND:
        raise Error("16129 * " + String(steps) + " is not below 2^24")
    if INT8_PRODUCT_MAX * INT8_APPLE_CHUNK_MAX_STEPS >= F32_EXACT_INTEGER_BOUND:
        raise Error("16129 * INT8_APPLE_CHUNK_MAX_STEPS is not below 2^24")
    if INT8_PRODUCT_MAX * (INT8_APPLE_CHUNK_MAX_STEPS + 1) < F32_EXACT_INTEGER_BOUND:
        raise Error("INT8_APPLE_CHUNK_MAX_STEPS is not the largest chunk: one more step still fits")
    var acc = Float32(0.0)
    var exact = 0
    for _ in range(INT8_APPLE_CHUNK_MAX_STEPS):
        acc = acc + Float32(INT8_PRODUCT_MAX)
        exact += INT8_PRODUCT_MAX
        if Int(acc) != exact:
            raise Error("a float32 chain of +16129 left the integers inside the bound, at " + String(exact))
    acc = acc + Float32(INT8_PRODUCT_MAX)
    exact += INT8_PRODUCT_MAX
    if Int(acc) == exact:
        raise Error("step 1041 of the chain is still exact: the bound this file states is not the float32 bound")
    print(
        "   ok chunk " + String(steps) + " steps (" + String(INT8_APPLE_CHUNK_WINDOWS) + " windows of "
        + String(INT8_APPLE_CHUNK_KB) + "), bound " + String(INT8_APPLE_CHUNK_MAX_STEPS)
        + "; the host chain at 1041 steps is " + String(Int(acc)) + ", the integer is " + String(exact)
    )


# ===========================================================================
# THE DEVICE GATES
# ===========================================================================


def check_apple_chunk_matches_flat_and_oracle(ctx: DeviceContext) raises:
    """GATE: on codes quantized from the file's float fixtures the probe
    returns the flat kernel's bits and the oracle's, on the geometry its
    launcher picks, at every OP_NT shape of `gemm_lowbit_check.mojo`, its
    ragged shapes and this file's."""
    var shapes = 0
    for s in range(SHAPE_COUNT + MMA_SHAPE_COUNT + CHUNK_SHAPE_COUNT):
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
            var sh = _chunk_shape(s - SHAPE_COUNT - MMA_SHAPE_COUNT)
            m = sh[0]
            n = sh[1]
            k = sh[2]
        var qa = quantize_rows_int8(_fill(m * k, 211 + s), m, k)
        var qb = quantize_rows_int8(_fill(n * k, 223 + s), n, k)
        var want = gemm_int8_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
        var tag = "int8 chunk " + String(m) + "x" + String(n) + "x" + String(k)
        var both = _run_both(ctx, qa.q, qa.e, qb.q, qb.e, m, n, k, int8_apple_chunk_geometry(m), tag)
        _first_diff(both[1], both[0], tag + " (chunk vs flat)")
        _first_diff(both[1], want, tag + " (chunk vs oracle)")
        _first_diff(both[0], want, tag + " (flat vs oracle)")
        shapes += 1
        print("   ok " + tag + "  c[0]=" + _show(both[1][0]))
    print("   ok chunk == flat == oracle on " + String(shapes) + " shapes of quantized fixtures")


def check_apple_chunk_geometries_agree(ctx: DeviceContext) raises:
    """GATE: the WIDE and the ROW tile return the same bits at every shape,
    so the launcher's choice between them is scheduling."""
    var shapes = 0
    for s in range(MMA_SHAPE_COUNT + CHUNK_SHAPE_COUNT):
        var m = 0
        var n = 0
        var k = 0
        if s < MMA_SHAPE_COUNT:
            var sh = _mma_shape(s)
            m = sh[0]
            n = sh[1]
            k = sh[2]
        else:
            var sh = _chunk_shape(s - MMA_SHAPE_COUNT)
            m = sh[0]
            n = sh[1]
            k = sh[2]
        var qa = quantize_rows_int8(_fill(m * k, 239 + s), m, k)
        var qb = quantize_rows_int8(_fill(n * k, 251 + s), n, k)
        var tag = "int8 chunk geometry " + String(m) + "x" + String(n) + "x" + String(k)
        var wide = _run_both(ctx, qa.q, qa.e, qb.q, qb.e, m, n, k, INT8_APPLE_CHUNK_GEOMETRY_WIDE, tag + " WIDE")
        var row = _run_both(ctx, qa.q, qa.e, qb.q, qb.e, m, n, k, INT8_APPLE_CHUNK_GEOMETRY_ROW, tag + " ROW")
        _first_diff(row[1], wide[1], tag + " (ROW vs WIDE)")
        _first_diff(wide[1], wide[0], tag + " (WIDE vs flat)")
        shapes += 1
    print("   ok ROW == WIDE == flat on " + String(shapes) + " shapes")


def check_apple_chunk_planted_worst_cases(ctx: DeviceContext) raises:
    """GATE: the planted codes, on BOTH geometries, against the flat kernel
    and the oracle. Every case is run and printed; the gate raises after
    the last one with the count and the first failure. This is the gate the
    chunk-boundary sabotage must fail, and it says at which k."""
    var cases = 0
    var failed = 0
    var first = String("")
    for s in range(PLANT_SHAPE_COUNT):
        var sh = _plant_shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var ea = _plant_exponents(m, -6, 5)
        var eb = _plant_exponents(n, -7, 3)
        for plant in range(PLANT_COUNT):
            var qa = _plant_a(plant, m, k)
            var qb = _plant_b(plant, n, k)
            var want = gemm_int8_oracle(qa, ea, qb, eb, m, n, k)
            for geometry in range(2):
                var tag = (
                    "int8 chunk planted " + _plant_name(plant) + " " + String(m) + "x" + String(n)
                    + "x" + String(k) + " " + ("ROW" if geometry == INT8_APPLE_CHUNK_GEOMETRY_ROW else "WIDE")
                )
                cases += 1
                try:
                    var both = _run_both(ctx, qa, ea, qb, eb, m, n, k, geometry, tag)
                    _first_diff(both[0], want, tag + " (flat vs oracle)")
                    _first_diff(both[1], want, tag + " (chunk vs oracle)")
                    _first_diff(both[1], both[0], tag + " (chunk vs flat)")
                    print("   ok " + tag + "  c[0]=" + _show(both[1][0]))
                except e:
                    failed += 1
                    print("   !! " + String(e))
                    if first.byte_length() == 0:
                        first = String(e)
    print("   planted: " + String(cases) + " cases, " + String(failed) + " failed")
    if failed > 0:
        raise Error(String(failed) + " of " + String(cases) + " planted cases differ; first: " + first)


def main() raises:
    print(
        "== gemm/checks/gemm_int8_apple_chunk_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + int8_apple_chunk_sabotage_name() + " =="
    )
    print("   profile: mojolearn.identical.gemm.int8i32.v1, the Apple exact-chunk PROBE")
    print("   column: " + column_name(TARGET_COLUMN))
    print(
        "   geometries: " + int8_apple_chunk_geometry_name(INT8_APPLE_CHUNK_GEOMETRY_WIDE)
        + "; " + int8_apple_chunk_geometry_name(INT8_APPLE_CHUNK_GEOMETRY_ROW)
    )
    var ran = 0
    var failed = 0
    try:
        check_chunk_bound_is_exact_and_tight()
        _gate(String("check_chunk_bound_is_exact_and_tight"), ran, failed, String(""))
    except e:
        _gate(String("check_chunk_bound_is_exact_and_tight"), ran, failed, String(e))
    comptime if not has_accelerator():
        print("   no accelerator: the device gates DID NOT RUN, which is not a pass")
        raise Error("the Apple exact-chunk gate needs the Metal column")
    elif TARGET_COLUMN != COLUMN_APPLE:
        print(
            "   the device gates DID NOT RUN, which is not a pass: column "
            + column_name(TARGET_COLUMN) + " is not Apple"
        )
        raise Error("the Apple exact-chunk gate needs the Metal column")
    else:
        var ctx = DeviceContext()
        try:
            check_apple_chunk_matches_flat_and_oracle(ctx)
            _gate(String("check_apple_chunk_matches_flat_and_oracle"), ran, failed, String(""))
        except e:
            _gate(String("check_apple_chunk_matches_flat_and_oracle"), ran, failed, String(e))
        try:
            check_apple_chunk_geometries_agree(ctx)
            _gate(String("check_apple_chunk_geometries_agree"), ran, failed, String(""))
        except e:
            _gate(String("check_apple_chunk_geometries_agree"), ran, failed, String(e))
        try:
            check_apple_chunk_planted_worst_cases(ctx)
            _gate(String("check_apple_chunk_planted_worst_cases"), ran, failed, String(""))
        except e:
            _gate(String("check_apple_chunk_planted_worst_cases"), ran, failed, String(e))
        print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
        if failed > 0:
            raise Error(String(failed) + " of " + String(ran) + " gates failed")
