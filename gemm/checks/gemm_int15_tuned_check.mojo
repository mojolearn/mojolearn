# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate of the fifteen-bit profile's TUNED plan: lane/lowbit-mma-speed's
four products with one staging, then this profile's seams, against the
host oracle, the flat kernel and the reference unit plan.

    pixi run check-gemm-int15-tuned                     every gate passes
    pixi run check-gemm-int15-tuned-sabotage            MUST FAIL the two product gates (every stored cell flipped)
    pixi run check-gemm-int15-tuned-pieces-sabotage     MUST FAIL them (the sums kernel's middle sum takes HL twice)
    pixi run check-gemm-int15-tuned-epilogue-sabotage   MUST FAIL them (the epilogue reads the wrong sum)
    pixi run check-gemm-int15-tuned-exponent-sabotage   MUST FAIL check_int15_tuned_row_scales (the column exponent read at the row index)
    pixi run check-gemm-int15-tuned-host-sabotage       MUST FAIL them (every oracle cell flipped)

Profile `mojolearn.identical.gemm.int15i64.v1`, contract clause W-13. Plan
`gemm/checks/gemm_int15_tuned.mojo`. Lane lane/lowbit-int15, 2026-09-29.

It runs on a column that has the integer matrix unit (NVIDIA, AMD). On any
other column main says it did not run, which is not a pass.

FUSED AND TWO-LAUNCH (the epilogue fold, 2026-09-29): every plan runs on
both paths, `tuned.<plan>` (fused, the entry points' path) and
`two.<plan>` (the sums stored, then the epilogue launch), and the
dispatched plan on both. Each must print the oracle's digest, so fused and
two-launch print one digest. While `INT15_FUSED_IS_STUB` the fused path is
the two-launch path and the header says so.

EVERY PLAN OF THE SUMS KERNEL is run on every shape and every planted case
of `gemm_int15_check.mojo` (its fixtures, its ragged extents, its planted
worst cases), and each must print the digest the host oracle prints, which
is the digest the flat kernel and the reference unit plan print in that
gate. The digest lines are `DIGEST <case> tuned.<plan> <hex>`.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import has_accelerator

from checks.kernel_matrix import TARGET_COLUMN, column_name, lib_int8_matrix_unit_for
from checks.numerics import numeric_mode_name
from gemm.checks.gemm_int15 import (
    identical_gemm_int15_flat_into,
    identical_gemm_int15_mma_into,
    int15_sabotage_name,
    split_int15_device,
)
from gemm.checks.gemm_int15_check import (
    SHAPE_COUNT,
    _digest,
    _download_cells,
    _fill,
    _first_diff,
    _planted_rows,
    _distinct_exponents,
    _poisoned,
    _row_scaled,
    _shape,
    _tag,
    _upload,
)
from gemm.checks.gemm_int15_tuned import (
    Int15SumsWorkspace,
    identical_gemm_int15_tuned_into,
    identical_gemm_int15_tuned_two_launch_into,
    identical_gemm_int15_tuned_two_launch_with_plan,
    identical_gemm_int15_tuned_with_plan,
    int15_fused_name,
    int15_tuned_admits,
    int15_tuned_sabotage_name,
)
from gemm.checks.gemm_int8_mma_tuned import (
    INT8_PIECES_MAX_K,
    INT8_PIECES_PLAN_COUNT,
    int8_pieces_plan_name,
    int8_pieces_sabotage_name,
)
from gemm.host.gemm_int15_oracle import (
    INT15_MAX_K,
    Int15Rows,
    gemm_int15_oracle,
    quantize_rows_int15,
)
from gemm.host.gemm_oracle import GEMM_ORACLE_HOST_SABOTAGE

comptime HAS_UNIT = lib_int8_matrix_unit_for[TARGET_COLUMN]()

#: `plan` values of `_run`: the sums kernel's plans, then these.
comptime RUN_DISPATCH = 100
comptime RUN_FLAT = 101
comptime RUN_REFERENCE_UNIT = 102
comptime RUN_DISPATCH_TWO_LAUNCH = 103
#: `RUN_TWO_LAUNCH + plan`: the sums kernel's plan on the two-launch path.
comptime RUN_TWO_LAUNCH = 200


def _run(
    ctx: DeviceContext, qa: Int15Rows, qb: Int15Rows, m: Int, n: Int, k: Int, plan: Int, tag: String
) raises -> List[Float32]:
    """One plan on host codes, the planes split on the device as a caller's
    are."""
    var dea = _upload[DType.int32](ctx, qa.e)
    var deb = _upload[DType.int32](ctx, qb.e)
    var dqa = _upload[DType.int16](ctx, qa.q)
    var dqb = _upload[DType.int16](ctx, qb.q)
    var dc = _poisoned(ctx, m * n)
    var dah = ctx.enqueue_create_buffer[DType.int8](m * k)
    var dal = ctx.enqueue_create_buffer[DType.int8](m * k)
    var dbh = ctx.enqueue_create_buffer[DType.int8](n * k)
    var dbl = ctx.enqueue_create_buffer[DType.int8](n * k)
    var work = Int15SumsWorkspace(ctx)
    split_int15_device(ctx, dah, dal, dqa, m * k)
    split_int15_device(ctx, dbh, dbl, dqb, n * k)
    if plan == RUN_FLAT:
        identical_gemm_int15_flat_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
    elif plan == RUN_REFERENCE_UNIT:
        identical_gemm_int15_mma_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, m, n, k)
    elif plan == RUN_DISPATCH:
        identical_gemm_int15_tuned_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, work, m, n, k)
    elif plan == RUN_DISPATCH_TWO_LAUNCH:
        identical_gemm_int15_tuned_two_launch_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, work, m, n, k)
    elif plan >= RUN_TWO_LAUNCH:
        identical_gemm_int15_tuned_two_launch_with_plan(
            ctx, dc, dah, dal, dea, dbh, dbl, deb, work, m, n, k, plan - RUN_TWO_LAUNCH
        )
    else:
        identical_gemm_int15_tuned_with_plan(ctx, dc, dah, dal, dea, dbh, dbl, deb, work, m, n, k, plan)
    ctx.synchronize()
    var out = _download_cells(ctx, dc, m * n, tag)
    _ = dea
    _ = deb
    _ = dqa
    _ = dqb
    _ = dc
    _ = dah
    _ = dal
    _ = dbh
    _ = dbl
    _ = work^
    return out^


def _every_tuned_plan_equals(
    ctx: DeviceContext, qa: Int15Rows, qb: Int15Rows, m: Int, n: Int, k: Int, name: String
) raises:
    """The oracle, the flat kernel, the reference unit plan, then every
    plan of the sums kernel and the dispatched one, each against the
    oracle. Every plan runs before the first difference is raised."""
    var want = gemm_int15_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
    print("   DIGEST " + name + " oracle " + _digest(want))
    var failures = String("")
    comptime P = INT8_PIECES_PLAN_COUNT
    for at_ in range(2 * P + 4):
        var plan = at_
        var label = String("")
        if at_ == 2 * P:
            plan = RUN_DISPATCH
            label = String("tuned.dispatch")
        elif at_ == 2 * P + 1:
            plan = RUN_DISPATCH_TWO_LAUNCH
            label = String("two.dispatch")
        elif at_ == 2 * P + 2:
            plan = RUN_FLAT
            label = String("flat")
        elif at_ == 2 * P + 3:
            plan = RUN_REFERENCE_UNIT
            label = String("mma")
        else:
            if at_ < P:
                label = String("tuned.") + int8_pieces_plan_name(at_)
            else:
                plan = RUN_TWO_LAUNCH + (at_ - P)
                label = String("two.") + int8_pieces_plan_name(at_ - P)
            if not int15_tuned_admits(m, n, k):
                # k = 65536: the sums kernel refuses, and the gate below
                # holds it to that refusal.
                continue
        var got = _run(ctx, qa, qb, m, n, k, plan, name + " " + label)
        print("   DIGEST " + name + " " + label + " " + _digest(got))
        try:
            _first_diff(got, want, name + " (" + label + " vs oracle)")
        except e:
            if failures.byte_length() > 0:
                failures += "; "
            failures += String(e)
    if failures.byte_length() > 0:
        raise Error(failures)


def check_int15_tuned_matches_oracle(ctx: DeviceContext) raises:
    """GATE: every plan of the sums kernel, through the epilogue, returns
    the oracle's bits on every shape of the fifteen-bit gate, the ragged
    ones included."""
    for s in range(SHAPE_COUNT):
        var sh = _shape(s)
        var m = sh[0]
        var n = sh[1]
        var k = sh[2]
        var qa = quantize_rows_int15(_fill(m * k, 131 + s), m, k)
        var qb = quantize_rows_int15(_fill(n * k, 149 + s), n, k)
        _every_tuned_plan_equals(ctx, qa, qb, m, n, k, "plans-" + _tag(m, n, k))
    print("   ok " + String(INT8_PIECES_PLAN_COUNT) + " tuned plans fused and two-launch, the dispatched one on both, flat and the reference unit plan equal the oracle on " + String(SHAPE_COUNT) + " shapes")


def check_int15_tuned_planted_worst_cases(ctx: DeviceContext) raises:
    """GATE: the planted cases of the fifteen-bit gate (every code at
    +16383 or -16383, the high piece at -128 on both sides, the cross term
    at its most negative, cancelling halves, a row of zeros, the walk), at
    ragged extents, at the largest `k` the sums kernel admits and at the
    largest the profile admits, where the dispatched plan is the reference
    unit plan."""
    var cases = 0
    var failures = String("")
    comptime geom_count = 5
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
            k = INT8_PIECES_MAX_K
        elif geom == 4:
            m = 2
            n = 3
            k = INT15_MAX_K
        comptime kind_count = 7
        for ka in range(kind_count):
            for kb in range(kind_count):
                if geom >= 3 and not ((ka < 3 and kb < 3) or (ka == 3 and kb == 0)):
                    continue
                if (geom == 1 or geom == 2) and (ka == 5) != (kb == 5):
                    continue
                var qa = _planted_rows(m, k, ka, 0)
                var qb = _planted_rows(n, k, kb, 0)
                var name = "planted-" + String(ka) + "." + String(kb) + "-" + _tag(m, n, k)
                try:
                    _every_tuned_plan_equals(ctx, qa, qb, m, n, k, name)
                except e:
                    if failures.byte_length() > 0:
                        failures += "; "
                    failures += String(e)
                cases += 1
    if failures.byte_length() > 0:
        raise Error(failures)
    print("   ok " + String(cases) + " planted cases, every tuned plan equal to the oracle")


def check_int15_tuned_row_scales(ctx: DeviceContext) raises:
    """GATE (clause W-7, the scale `2^(ea[i] + eb[j])`): operands whose rows
    carry DIFFERENT exponents, so a cell scaled by any exponent but its own
    row's and its own column's is seen. The other gates' fills give every
    row of an operand one exponent, and an epilogue that read the column
    exponent at the row index passed them (run 6, nvc3-0032, and the MI325X,
    1790658677079: the exponent arm held=no). Row `r` of A is scaled by
    `2^((r mod 7) - 3)`, row `c` of B by `2^(2 - (c mod 5))`; the gate first
    holds the fixture to that (at least 5 distinct exponents per operand),
    so it cannot pass by being blind."""
    var m = 37
    var n = 41
    var k = 300
    var qa = quantize_rows_int15(_row_scaled(m, k, 311, 7, 3, 1), m, k)
    var qb = quantize_rows_int15(_row_scaled(n, k, 313, 5, 2, -1), n, k)
    var da = _distinct_exponents(qa.e)
    var db = _distinct_exponents(qb.e)
    if da < 5 or db < 5:
        raise Error(
            "the fixture is blind: " + String(da) + " distinct row exponents in A and "
            + String(db) + " in B, 5 each required"
        )
    _every_tuned_plan_equals(ctx, qa, qb, m, n, k, "row-scales-" + _tag(m, n, k))
    print("   ok every tuned plan, fused and two-launch, equal to the oracle with " + String(da) + " and " + String(db) + " distinct row exponents")


def check_int15_tuned_refuses(ctx: DeviceContext) raises:
    """GATE: every path of the tuned plan refuses, by name and before it
    launches, the first extent above the bound it states: a named plan,
    fused and two-launch, at `INT8_PIECES_MAX_K + 1` (the sums kernel's
    bound, read from its file, not assumed) and at `INT15_MAX_K + 1` (the
    profile's); the dispatched plan, fused and two-launch, at
    `INT15_MAX_K + 1`.

    The operands are as long as the largest shape probed (`1 x k`), so a
    call that failed to refuse launches inside its buffers and the gate
    fails on the count, never by reading outside an allocation. (Until
    2026-09-29 the gate assumed the sums kernel refused `k = 65536` and
    passed one-byte buffers; lane/lowbit-mma-speed raised that bound to
    65536 in 520406a38 and the MI325X read outside the allocation, job
    1790657862351.)"""
    var k_sums = INT8_PIECES_MAX_K + 1
    var k_prof = INT15_MAX_K + 1
    var k_big = k_sums if k_sums > k_prof else k_prof
    var ones = List[Int8](length=k_big, fill=Int8(1))
    var e0: List[Int32] = [0]
    var dah = _upload[DType.int8](ctx, ones)
    var dal = _upload[DType.int8](ctx, ones)
    var dbh = _upload[DType.int8](ctx, ones)
    var dbl = _upload[DType.int8](ctx, ones)
    var dea = _upload[DType.int32](ctx, e0)
    var deb = _upload[DType.int32](ctx, e0)
    var dc = _poisoned(ctx, 1)
    var work = Int15SumsWorkspace(ctx)
    var tried = 0
    var refused = 0
    var extents: List[Int] = [k_sums, k_prof]
    for at_ in range(len(extents)):
        var k = extents[at_]
        tried += 2
        try:
            identical_gemm_int15_tuned_with_plan(ctx, dc, dah, dal, dea, dbh, dbl, deb, work, 1, 1, k, 0)
        except e:
            refused += 1
        try:
            identical_gemm_int15_tuned_two_launch_with_plan(ctx, dc, dah, dal, dea, dbh, dbl, deb, work, 1, 1, k, 0)
        except e:
            refused += 1
    tried += 2
    try:
        identical_gemm_int15_tuned_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, work, 1, 1, k_prof)
    except e:
        refused += 1
    try:
        identical_gemm_int15_tuned_two_launch_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, work, 1, 1, k_prof)
    except e:
        refused += 1
    ctx.synchronize()
    _ = dah
    _ = dal
    _ = dbh
    _ = dbl
    _ = dea
    _ = deb
    _ = dc
    _ = work^
    if refused != tried:
        raise Error("only " + String(refused) + " of " + String(tried) + " launches refused")
    print(
        "   ok " + String(tried) + " launches refuse the first extent above their bound (sums kernel "
        + String(INT8_PIECES_MAX_K) + ", profile " + String(INT15_MAX_K) + ")"
    )


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
        "== gemm/checks/gemm_int15_tuned_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + int15_sabotage_name()
        + "  epilogue sabotage: " + int15_tuned_sabotage_name()
        + "  sums kernel sabotage: " + int8_pieces_sabotage_name()
        + "  host sabotage: " + String(GEMM_ORACLE_HOST_SABOTAGE) + " =="
    )
    print("   profile: mojolearn.identical.gemm.int15i64.v1, the TUNED plan (contract W-13)")
    print("   fused path: " + int15_fused_name())
    print("   column: " + column_name(TARGET_COLUMN))
    var ran = 0
    var failed = 0
    comptime if not has_accelerator():
        print("   no accelerator: the gates did not run")
    elif not HAS_UNIT:
        print(
            "   THE TUNED PLAN'S GATES DID NOT RUN: column " + column_name(TARGET_COLUMN)
            + " has no int8 matrix unit. That is not a pass."
        )
    else:
        var ctx = DeviceContext()
        try:
            check_int15_tuned_matches_oracle(ctx)
            _gate(String("check_int15_tuned_matches_oracle"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_tuned_matches_oracle"), ran, failed, String(e))
        try:
            check_int15_tuned_planted_worst_cases(ctx)
            _gate(String("check_int15_tuned_planted_worst_cases"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_tuned_planted_worst_cases"), ran, failed, String(e))
        try:
            check_int15_tuned_row_scales(ctx)
            _gate(String("check_int15_tuned_row_scales"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_tuned_row_scales"), ran, failed, String(e))
        try:
            check_int15_tuned_refuses(ctx)
            _gate(String("check_int15_tuned_refuses"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_tuned_refuses"), ran, failed, String(e))
        _ = ctx^
    print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
    if failed > 0:
        raise Error(String(failed) + " gate(s) failed")
