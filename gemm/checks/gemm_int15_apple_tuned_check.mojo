# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate of the TUNED Apple float-unit plans of
`mojolearn.identical.gemm.int15i64.v1`: every variant against the host
oracle AND the flat kernel on every case, planted worst cases at every
chunk boundary, the quality lane's simulation vectors, a launch in slices,
and the two arms that show the gate can fail.

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_int15_apple_tuned_check.mojo
        every gate passes
    ... -D MOJOLEARN_INT15_APPLE_TUNED_CHUNK_SABOTAGE=1 ...
        MUST FAIL check_tuned_planted_worst_cases and
        check_tuned_chunk_boundaries, and must leave
        check_tuned_shapes_inside_one_chunk passing
    ... -D MOJOLEARN_LOWBIT_SABOTAGE=1 ...
        MUST FAIL every device gate

Lane lane/lowbit-apple-tuned, 2026-09-29. Kernels
`gemm/checks/gemm_int15_apple_tuned.mojo`. Apple only. It extends
`gemm/checks/gemm_int15_check.mojo` (lane/lowbit-int15): the shapes, the
fills, the planted rows, the digest and the cell comparison are that
file's, imported, and the cases it shares with that gate carry that gate's
NAMES and OPERANDS, so a `DIGEST <case> tuned.<variant> <hex>` line here is
compared with the H100's and the MI325X's lines of the same case by
`tools/lowbit_int15/digests.py`.

WHAT IS PLANTED AT A CHUNK BOUNDARY, AND WHY THESE. Form FOUR cuts `k`
every 512 steps and form THREE every 256 (the proofs are in the kernel
file). The cases are constant codes whose accumulators grow by the most a
step can add, at the `k` that ends a chunk, one step short of it, one step
past it, eight past it (the first step of the unit past it), and at odd
`k` beyond the point where an unbroken float sum would pass 2^24:

  16383 x 16383     pieces (127, 127): `PP = 254 * 254 = 64516`, THREE's
                    largest term; `HH` and `LL` are 16129, odd
  -16383 x -16383   pieces (-128, 1): `HH = 16384`, the largest piece
                    product
  -16257 x -16257   pieces (-128, 127): `MID = -32512` a step, FOUR's
                    largest term
  16382 x 16382     pieces (127, 126): `PP = 253 * 253 = 64009`, ODD, so an
                    unbroken `PP` passes 2^24 on an odd integer from
                    `k = 263`, where `HH`, `MID` and `LL` are still far
                    below it: the case that fails when THREE's boundary
                    alone is missing
  16383 x 16382     `MID = 127 * 126 + 127 * 127 = 32131` a step, ODD: an
                    unbroken `MID` passes 2^24 on an odd integer at odd
                    `k` from 523: the case that fails when FOUR's boundary
                    is missing, below the `k` where `HH` would
  +16383 then -16383 against +16383   the halves cancel: every accumulator
                    climbs to its largest and returns to zero
  kind 9            lane/lowbit-int15's odd sums across two steps of the
                    unit
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.os import getenv
from std.sys import has_accelerator

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN, column_name
from checks.numerics import numeric_mode_name
from gemm.checks.gemm_int15 import (
    Int15QuantWorkspace,
    identical_gemm_int15_flat_into,
    int15_quant_chunks,
    quantize_planes_int15_parallel_device,
    split_int15_device,
)
from gemm.checks.gemm_int15_apple_tuned import (
    INT15_TUNED2_CHUNK_STEPS,
    INT15_TUNED3_CHUNK_STEPS,
    INT15_TUNED4_CHUNK_STEPS,
    TUNED_VARIANT_COUNT,
    identical_gemm_int15_apple_tuned_into,
    int15_apple_tuned_sabotage_name,
    int15_apple_tuned_variant_name,
)
from gemm.checks.gemm_int15_apple_tuned_dev import (
    TUNED_DEV_COUNT,
    identical_gemm_int15_apple_dev_from_pieces_into,
    int15_apple_tuned_dev_variant_name,
)
from gemm.checks.gemm_int15_check import (
    SHAPE_COUNT,
    _digest,
    _download_cells,
    _fill,
    _first_diff,
    _planted_rows,
    _poisoned,
    _shape,
    _tag,
    _upload,
)
from gemm.checks.gemm_int15_sim_check import _load as _sim_load
from gemm.host.gemm_int15_oracle import (
    INT15_MAX_K,
    Int15Rows,
    gemm_int15_oracle,
    quantize_rows_int15,
)
from gemm.host.gemm_oracle import GEMM_ORACLE_HOST_SABOTAGE

comptime IS_APPLE = TARGET_COLUMN == COLUMN_APPLE

#: The default slice of a launch in this gate (the kernel file's default).
comptime GATE_SLICE_MACS = 4_294_967_296


def _constant_rows(rows: Int, k: Int, code: Int) -> Int15Rows:
    """Every code the same, exponent 0, written directly."""
    var q = List[Int16]()
    for _ in range(rows * k):
        q.append(Int16(code))
    var ex = List[Int32]()
    for _ in range(rows):
        ex.append(Int32(0))
    return Int15Rows(q^, ex^, rows, k)


def _variants_on_planes(
    ctx: DeviceContext,
    mut dah: DeviceBuffer[DType.int8],
    mut dal: DeviceBuffer[DType.int8],
    mut dea: DeviceBuffer[DType.int32],
    mut dbh: DeviceBuffer[DType.int8],
    mut dbl: DeviceBuffer[DType.int8],
    mut deb: DeviceBuffer[DType.int32],
    want: List[Float32],
    flat: List[Float32],
    m: Int,
    n: Int,
    k: Int,
    name: String,
    slice_macs: Int,
) raises -> String:
    """Every variant on one case's device planes, each output poisoned
    first and read back whole, each against the oracle and, where the case
    ran it, the flat kernel. Every variant runs before anything is raised;
    the failures come back as one string."""
    var failures = String("")
    for v in range(TUNED_VARIANT_COUNT + TUNED_DEV_COUNT):
        var vn = String("tuned.") + int15_apple_tuned_variant_name(v)
        if v >= TUNED_VARIANT_COUNT:
            vn = String("tuned.") + int15_apple_tuned_dev_variant_name(v - TUNED_VARIANT_COUNT)
        var dc = _poisoned(ctx, m * n)
        try:
            if v < TUNED_VARIANT_COUNT:
                identical_gemm_int15_apple_tuned_into(
                    ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k, v, slice_macs
                )
            else:
                identical_gemm_int15_apple_dev_from_pieces_into(
                    ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k, v - TUNED_VARIANT_COUNT, slice_macs
                )
            ctx.synchronize()
            var got = _download_cells(ctx, dc, m * n, name + " " + vn)
            print("   DIGEST " + name + " " + vn + " " + _digest(got))
            _first_diff(got, want, name + " (" + vn + " vs oracle)")
            if len(flat) > 0:
                _first_diff(got, flat, name + " (" + vn + " vs flat)")
        except e:
            if failures.byte_length() > 0:
                failures += "; "
            failures += String(e)
        _ = dc
    return failures


def _case(
    ctx: DeviceContext,
    qa: Int15Rows,
    qb: Int15Rows,
    m: Int,
    n: Int,
    k: Int,
    name: String,
    slice_macs: Int = GATE_SLICE_MACS,
) raises -> String:
    """One case on host codes: the oracle, the flat kernel on the device,
    then every tuned variant on planes split ON THE DEVICE. Returns the
    failures, empty when there are none."""
    var want = gemm_int15_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
    print("   DIGEST " + name + " oracle " + _digest(want))
    var dea = _upload[DType.int32](ctx, qa.e)
    var deb = _upload[DType.int32](ctx, qb.e)
    var dqa = _upload[DType.int16](ctx, qa.q)
    var dqb = _upload[DType.int16](ctx, qb.q)
    var dah = ctx.enqueue_create_buffer[DType.int8](m * k)
    var dal = ctx.enqueue_create_buffer[DType.int8](m * k)
    var dbh = ctx.enqueue_create_buffer[DType.int8](n * k)
    var dbl = ctx.enqueue_create_buffer[DType.int8](n * k)
    split_int15_device(ctx, dah, dal, dqa, m * k)
    split_int15_device(ctx, dbh, dbl, dqb, n * k)
    ctx.synchronize()
    var failures = String("")
    var flat = List[Float32]()
    var dcf = _poisoned(ctx, m * n)
    try:
        identical_gemm_int15_flat_into(ctx, dcf, dqa, dea, dqb, deb, m, n, k)
        ctx.synchronize()
        flat = _download_cells(ctx, dcf, m * n, name + " flat")
        print("   DIGEST " + name + " flat " + _digest(flat))
        _first_diff(flat, want, name + " (flat vs oracle)")
    except e:
        failures += String(e)
    var more = _variants_on_planes(
        ctx, dah, dal, dea, dbh, dbl, deb, want, flat, m, n, k, name, slice_macs
    )
    if more.byte_length() > 0:
        if failures.byte_length() > 0:
            failures += "; "
        failures += more
    _ = dcf
    _ = dqa
    _ = dqb
    _ = dea
    _ = deb
    _ = dah
    _ = dal
    _ = dbh
    _ = dbl
    return failures


def _collect(mut failures: String, more: String):
    if more.byte_length() > 0:
        if failures.byte_length() > 0:
            failures += "; "
        failures += more


def _shapes(ctx: DeviceContext, inside_one_chunk: Bool) raises -> Int:
    """lane/lowbit-int15's fifteen shapes with its `plans-` operands, the
    ragged ones included. `inside_one_chunk` takes the shapes whose `k` is
    at most THREE's chunk (no boundary but the end of `k`), the other call
    the rest."""
    var failures = String("")
    var ran = 0
    for s in range(SHAPE_COUNT):
        var sh = _shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        if (k <= INT15_TUNED3_CHUNK_STEPS) != inside_one_chunk:
            continue
        var qa = quantize_rows_int15(_fill(m * k, 131 + s), m, k)
        var qb = quantize_rows_int15(_fill(n * k, 149 + s), n, k)
        _collect(failures, _case(ctx, qa, qb, m, n, k, "plans-" + _tag(m, n, k)))
        ran += 1
    if failures.byte_length() > 0:
        raise Error(failures)
    return ran


def check_tuned_shapes_inside_one_chunk(ctx: DeviceContext) raises:
    """GATE: every variant equals the oracle and the flat kernel on the
    shapes no chunk boundary cuts. The chunk arm does not reach them."""
    var ran = _shapes(ctx, True)
    print("   ok " + String(ran) + " shapes with k at most " + String(INT15_TUNED3_CHUNK_STEPS))


def check_tuned_shapes_across_chunks(ctx: DeviceContext) raises:
    """GATE: the same on the shapes a chunk boundary cuts."""
    var ran = _shapes(ctx, False)
    print("   ok " + String(ran) + " shapes with k above " + String(INT15_TUNED3_CHUNK_STEPS))


def check_tuned_planted_worst_cases(ctx: DeviceContext) raises:
    """GATE: lane/lowbit-int15's planted cases, its names and its operands
    (`check_int15_planted_worst_cases`), on every variant: every code at
    its largest magnitude, the high piece at -128 on both sides, the cross
    term at its most negative, cancelling halves, a row of zeros, the walk
    over the whole range, the largest admitted `k`, the scale exponent at
    both ends and the largest sum."""
    var cases = 0
    var failures = String("")
    comptime geom_count = 4
    for geom in range(geom_count):
        var m = 3
        var n = 5
        var k = 17
        if geom == 1:
            m = 17
            n = 33
            k = 1000
        elif geom == 2:
            m = 1
            n = 40
            k = 4097
        elif geom == 3:
            m = 2
            n = 3
            k = INT15_MAX_K
        comptime left_count = 7
        for ka in range(left_count):
            for kb in range(left_count):
                if geom == 3 and not ((ka < 3 and kb < 3) or (ka == 3 and kb == 0)):
                    continue
                if (geom == 1 or geom == 2) and (ka == 5) != (kb == 5):
                    continue
                var qa = _planted_rows(m, k, ka, 0)
                var qb = _planted_rows(n, k, kb, 0)
                var name = "planted-" + String(ka) + "." + String(kb) + "-" + _tag(m, n, k)
                _collect(failures, _case(ctx, qa, qb, m, n, k, name))
                cases += 1
        var qa9 = _planted_rows(m, k, 9, 0)
        var qb9 = _planted_rows(n, k, 0, 0)
        _collect(failures, _case(ctx, qa9, qb9, m, n, k, "planted-9.0-" + _tag(m, n, k)))
        cases += 1
    comptime scale_count = 6
    for s in range(scale_count):
        var ea = 115
        var eb = 12
        if s == 1:
            ea = 115
            eb = 13
        elif s == 2:
            ea = -139
            eb = 13
        elif s == 3:
            ea = -139
            eb = 12
        elif s == 4:
            ea = 115
            eb = 115
        elif s == 5:
            ea = -139
            eb = -139
        var qa = _planted_rows(3, 33, 7, ea)
        var qb = _planted_rows(5, 33, 8, eb)
        _collect(failures, _case(ctx, qa, qb, 3, 5, 33, "scale-" + String(ea + eb)))
        cases += 1
    for top in range(2):
        var qa = _planted_rows(2, INT15_MAX_K, 0, 84 + top)
        var qb = _planted_rows(3, INT15_MAX_K, 0, 0)
        _collect(
            failures,
            _case(ctx, qa, qb, 2, 3, INT15_MAX_K, "largest-sum-scale-" + String(84 + top)),
        )
        cases += 1
    if failures.byte_length() > 0:
        raise Error(failures)
    print("   ok " + String(cases) + " planted cases, every variant equal to the oracle and the flat kernel")


def _boundary_k(i: Int) -> Int:
    """The `k` of the chunk-boundary cases: around TWO's boundary (8, one
    step of the unit; 512 is also where the deferred carry of f2d flushes),
    around THREE's boundary (256),
    around FOUR's (512), around the second boundary of each, and odd `k`
    past the points where an unbroken sum passes 2^24 (263 for `PP`, 523
    for `MID`, 1041 for `HH` and `LL`)."""
    var ks: List[Int] = [
        INT15_TUNED2_CHUNK_STEPS - 1,
        INT15_TUNED2_CHUNK_STEPS,
        INT15_TUNED2_CHUNK_STEPS + 1,
        2 * INT15_TUNED2_CHUNK_STEPS + 1,
        INT15_TUNED3_CHUNK_STEPS - 1,
        INT15_TUNED3_CHUNK_STEPS,
        INT15_TUNED3_CHUNK_STEPS + 1,
        INT15_TUNED3_CHUNK_STEPS + 8,
        265,
        INT15_TUNED4_CHUNK_STEPS - 1,
        INT15_TUNED4_CHUNK_STEPS,
        INT15_TUNED4_CHUNK_STEPS + 1,
        INT15_TUNED4_CHUNK_STEPS + 8,
        525,
        2 * INT15_TUNED4_CHUNK_STEPS,
        2 * INT15_TUNED4_CHUNK_STEPS + 1,
        1043,
        4097,
    ]
    return ks[i]


comptime BOUNDARY_K_COUNT = 18
comptime BOUNDARY_PAIR_COUNT = 7


def _boundary_pair_name(pair: Int) -> String:
    if pair == 0:
        return String("16383x16383")
    if pair == 1:
        return String("m16383xm16383")
    if pair == 2:
        return String("m16257xm16257")
    if pair == 3:
        return String("16382x16382")
    if pair == 4:
        return String("16383x16382")
    if pair == 5:
        return String("halves")
    return String("kind9")


def _boundary_rows(rows: Int, k: Int, pair: Int, left: Bool) -> Int15Rows:
    """One operand of a chunk-boundary pair (the header's list)."""
    if pair == 0:
        return _constant_rows(rows, k, 16383)
    if pair == 1:
        return _constant_rows(rows, k, -16383)
    if pair == 2:
        return _constant_rows(rows, k, -16257)
    if pair == 3:
        return _constant_rows(rows, k, 16382)
    if pair == 4:
        return _constant_rows(rows, k, 16383 if left else 16382)
    if pair == 5:
        return _planted_rows(rows, k, 3 if left else 0, 0)
    return _planted_rows(rows, k, 9 if left else 0, 0)


def check_tuned_chunk_boundaries(ctx: DeviceContext) raises:
    """GATE: the planted cases AT the chunk boundaries of both forms (the
    header says which and why), a ragged 3 x 5 output, every variant equal
    to the oracle and the flat kernel. The chunk arm MUST fail here."""
    var failures = String("")
    var cases = 0
    var m = 3
    var n = 5
    for ki in range(BOUNDARY_K_COUNT):
        var k = _boundary_k(ki)
        for pair in range(BOUNDARY_PAIR_COUNT):
            var qa = _boundary_rows(m, k, pair, True)
            var qb = _boundary_rows(n, k, pair, False)
            var name = "chunk-" + _boundary_pair_name(pair) + "-" + _tag(m, n, k)
            _collect(failures, _case(ctx, qa, qb, m, n, k, name))
            cases += 1
    if failures.byte_length() > 0:
        raise Error(failures)
    print("   ok " + String(cases) + " cases at the chunk boundaries, every variant equal to the oracle and the flat kernel")


def check_tuned_launch_in_slices(ctx: DeviceContext) raises:
    """GATE: a launch cut into slices of ONE row of tiles each (a slice of
    one multiply-accumulate asks for the smallest) writes every cell and
    the oracle's bits: 129 x 129 is three rows of 64-row tiles, five of
    32-row tiles and seventeen of 8-row tiles, the last one ragged."""
    var m = 129
    var n = 129
    var k = 264
    var qa = quantize_rows_int15(_fill(m * k, 307), m, k)
    var qb = quantize_rows_int15(_fill(n * k, 311), n, k)
    var failures = _case(ctx, qa, qb, m, n, k, "slices-" + _tag(m, n, k), 1)
    if failures.byte_length() > 0:
        raise Error(failures)
    print("   ok a launch in slices of one row of tiles, every variant")


def check_tuned_simulation_vectors(ctx: DeviceContext, path: String) raises:
    """GATE (contract 6.6): from the simulation's exported float32
    operands, the device quantizer straight to planes and then every
    variant give the simulation's product, bit for bit, NaN cells as NaN."""
    var cases = _sim_load(path)
    var failures = String("")
    var cells = 0
    for ci in range(len(cases)):
        ref sc = cases[ci]
        var name = "sim-" + _tag(sc.m, sc.n, sc.k)
        print("   DIGEST " + name + " simulation " + _digest(sc.c))
        var da = _upload[DType.float32](ctx, sc.a)
        var db = _upload[DType.float32](ctx, sc.b)
        var dea = ctx.enqueue_create_buffer[DType.int32](sc.m)
        var deb = ctx.enqueue_create_buffer[DType.int32](sc.n)
        var dah = ctx.enqueue_create_buffer[DType.int8](sc.m * sc.k)
        var dal = ctx.enqueue_create_buffer[DType.int8](sc.m * sc.k)
        var dbh = ctx.enqueue_create_buffer[DType.int8](sc.n * sc.k)
        var dbl = ctx.enqueue_create_buffer[DType.int8](sc.n * sc.k)
        var quant = Int15QuantWorkspace(ctx)
        var rows_max = sc.m if sc.m > sc.n else sc.n
        quant.ensure(ctx, rows_max * int15_quant_chunks(sc.k))
        quantize_planes_int15_parallel_device(ctx, dah, dal, dea, da, quant, sc.m, sc.k, False)
        quantize_planes_int15_parallel_device(ctx, dbh, dbl, deb, db, quant, sc.n, sc.k, False)
        ctx.synchronize()
        var no_flat = List[Float32]()
        _collect(
            failures,
            _variants_on_planes(
                ctx, dah, dal, dea, dbh, dbl, deb, sc.c, no_flat, sc.m, sc.n, sc.k, name,
                GATE_SLICE_MACS,
            ),
        )
        cells += sc.m * sc.n
        _ = quant^
        _ = da
        _ = db
        _ = dea
        _ = deb
        _ = dah
        _ = dal
        _ = dbh
        _ = dbl
    if failures.byte_length() > 0:
        raise Error(failures)
    print("   ok " + String(cells) + " cells of " + String(len(cases)) + " exported products are the simulation's on every variant")


def _gate(name: String, mut ran: Int, mut failed: Int, e: String):
    ran += 1
    if e.byte_length() > 0:
        failed += 1
        print("!! GATE FAILED: " + name)
        print("   " + e)
    else:
        print("ok " + name)


def main() raises:
    print(
        "== gemm/checks/gemm_int15_apple_tuned_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + int15_apple_tuned_sabotage_name()
        + "  host sabotage: " + String(GEMM_ORACLE_HOST_SABOTAGE) + " =="
    )
    print("   profile: mojolearn.identical.gemm.int15i64.v1, the tuned Apple float-unit plans")
    print("   column: " + column_name(TARGET_COLUMN) + "  variants: " + String(TUNED_VARIANT_COUNT) + " staged + " + String(TUNED_DEV_COUNT) + " device-fragment")
    comptime if not IS_APPLE:
        raise Error("gemm_int15_apple_tuned_check: this column is not Apple; NOTHING RAN, which is not a pass")
    else:
        comptime if not has_accelerator():
            raise Error("gemm_int15_apple_tuned_check: no accelerator; NOTHING RAN, which is not a pass")
        else:
            var path = String(getenv("MOJOLEARN_INT15_SIM_VECTORS"))
            if path.byte_length() == 0:
                path = String("gemm/checks/vectors/int15_sim_vectors.q15")
            var ran = 0
            var failed = 0
            var ctx = DeviceContext()
            try:
                check_tuned_shapes_inside_one_chunk(ctx)
                _gate(String("check_tuned_shapes_inside_one_chunk"), ran, failed, String(""))
            except e:
                _gate(String("check_tuned_shapes_inside_one_chunk"), ran, failed, String(e))
            try:
                check_tuned_shapes_across_chunks(ctx)
                _gate(String("check_tuned_shapes_across_chunks"), ran, failed, String(""))
            except e:
                _gate(String("check_tuned_shapes_across_chunks"), ran, failed, String(e))
            try:
                check_tuned_planted_worst_cases(ctx)
                _gate(String("check_tuned_planted_worst_cases"), ran, failed, String(""))
            except e:
                _gate(String("check_tuned_planted_worst_cases"), ran, failed, String(e))
            try:
                check_tuned_chunk_boundaries(ctx)
                _gate(String("check_tuned_chunk_boundaries"), ran, failed, String(""))
            except e:
                _gate(String("check_tuned_chunk_boundaries"), ran, failed, String(e))
            try:
                check_tuned_launch_in_slices(ctx)
                _gate(String("check_tuned_launch_in_slices"), ran, failed, String(""))
            except e:
                _gate(String("check_tuned_launch_in_slices"), ran, failed, String(e))
            try:
                check_tuned_simulation_vectors(ctx, path)
                _gate(String("check_tuned_simulation_vectors"), ran, failed, String(""))
            except e:
                _gate(String("check_tuned_simulation_vectors"), ran, failed, String(e))
            _ = ctx^
            print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
            if failed > 0:
                raise Error(String(failed) + " gate(s) failed")
