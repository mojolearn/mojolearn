# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate of the fifteen-bit profile's TUNED plan: lane/lowbit-mma-speed's
four products with one staging, then this profile's seams, against the
host oracle, the flat kernel and the reference unit plan.

    pixi run check-gemm-int15-tuned                     every gate passes
    pixi run check-gemm-int15-tuned-sabotage            MUST FAIL the two product gates (every stored cell flipped)
    pixi run check-gemm-int15-tuned-pieces-sabotage     MUST FAIL them (the sums kernel's middle sum takes HL twice)
    pixi run check-gemm-int15-tuned-epilogue-sabotage   MUST FAIL them (the epilogue reads the wrong sum)
    pixi run check-gemm-int15-tuned-host-sabotage       MUST FAIL them (every oracle cell flipped)

Profile `mojolearn.identical.gemm.int15i64.v1`, contract clause W-13. Plan
`gemm/checks/gemm_int15_tuned.mojo`. Lane lane/lowbit-int15, 2026-09-29.

It runs on a column that has the integer matrix unit (NVIDIA, AMD). On any
other column main says it did not run, which is not a pass.

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
    _poisoned,
    _shape,
    _tag,
    _upload,
)
from gemm.checks.gemm_int15_tuned import (
    Int15SumsWorkspace,
    identical_gemm_int15_tuned_into,
    identical_gemm_int15_tuned_with_plan,
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
    for at_ in range(INT8_PIECES_PLAN_COUNT + 3):
        var plan = at_
        var label = String("")
        if at_ == INT8_PIECES_PLAN_COUNT:
            plan = RUN_DISPATCH
            label = String("tuned.dispatch")
        elif at_ == INT8_PIECES_PLAN_COUNT + 1:
            plan = RUN_FLAT
            label = String("flat")
        elif at_ == INT8_PIECES_PLAN_COUNT + 2:
            plan = RUN_REFERENCE_UNIT
            label = String("mma")
        else:
            label = String("tuned.") + int8_pieces_plan_name(plan)
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
    print("   ok " + String(INT8_PIECES_PLAN_COUNT) + " tuned plans, the dispatched one, flat and the reference unit plan equal the oracle on " + String(SHAPE_COUNT) + " shapes")


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


def _refused_by_name(e: String, bound: Int) -> Bool:
    return e.find(String(bound)) >= 0


def check_int15_tuned_refuses(ctx: DeviceContext) raises:
    """GATE: every call above a bound is refused BY NAME before anything is
    launched. A named plan of the sums kernel refuses one step above ITS
    bound (`INT8_PIECES_MAX_K`) and one step above the profile's
    (`INT15_MAX_K`); the dispatched plan refuses one step above the
    profile's. The bounds are read, not assumed: lane/lowbit-mma-speed
    raised the sums kernel's to 65536 (520406a38), the profile's own, and
    the gate that expected k = 65536 to be refused then LAUNCHED a 1 x 1 x
    65536 product on buffers of one code (an out-of-bounds read; MI325X job
    1790658495381). The operands here hold every code the largest `k` asked
    reads, so a call that is not refused reads nothing outside them."""
    var kmax = INT15_MAX_K + 1
    if INT8_PIECES_MAX_K + 1 > kmax:
        kmax = INT8_PIECES_MAX_K + 1
    var codes = List[Int8]()
    for _ in range(kmax):
        codes.append(Int8(1))
    var e0: List[Int32] = [0]
    var dah = _upload[DType.int8](ctx, codes)
    var dal = _upload[DType.int8](ctx, codes)
    var dbh = _upload[DType.int8](ctx, codes)
    var dbl = _upload[DType.int8](ctx, codes)
    var dea = _upload[DType.int32](ctx, e0)
    var deb = _upload[DType.int32](ctx, e0)
    var dc = _poisoned(ctx, 1)
    var work = Int15SumsWorkspace(ctx)
    var bad = String("")
    var sums_k = INT8_PIECES_MAX_K + 1
    try:
        identical_gemm_int15_tuned_with_plan(ctx, dc, dah, dal, dea, dbh, dbl, deb, work, 1, 1, sums_k, 0)
        bad += "the named plan did not refuse k = " + String(sums_k) + "; "
    except e:
        if not _refused_by_name(String(e), INT8_PIECES_MAX_K) and not _refused_by_name(String(e), INT15_MAX_K):
            bad += "the named plan refused k = " + String(sums_k) + " but not by its bound: " + String(e) + "; "
    try:
        identical_gemm_int15_tuned_with_plan(ctx, dc, dah, dal, dea, dbh, dbl, deb, work, 1, 1, INT15_MAX_K + 1, 0)
        bad += "the named plan did not refuse k = " + String(INT15_MAX_K + 1) + "; "
    except e:
        if not _refused_by_name(String(e), INT15_MAX_K):
            bad += "the named plan refused k = " + String(INT15_MAX_K + 1) + " but not by the profile's bound: " + String(e) + "; "
    try:
        identical_gemm_int15_tuned_into(ctx, dc, dah, dal, dea, dbh, dbl, deb, work, 1, 1, INT15_MAX_K + 1)
        bad += "the dispatched plan did not refuse k = " + String(INT15_MAX_K + 1) + "; "
    except e:
        if not _refused_by_name(String(e), INT15_MAX_K):
            bad += "the dispatched plan refused k = " + String(INT15_MAX_K + 1) + " but not by the profile's bound: " + String(e) + "; "
    ctx.synchronize()
    _ = dah
    _ = dal
    _ = dbh
    _ = dbl
    _ = dea
    _ = deb
    _ = dc
    _ = work^
    if bad.byte_length() > 0:
        raise Error(bad)
    print(
        "   ok 3 calls refuse an extent above their bound by name (sums kernel "
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
            check_int15_tuned_refuses(ctx)
            _gate(String("check_int15_tuned_refuses"), ran, failed, String(""))
        except e:
            _gate(String("check_int15_tuned_refuses"), ran, failed, String(e))
        _ = ctx^
    print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
    if failed > 0:
        raise Error(String(failed) + " gate(s) failed")
