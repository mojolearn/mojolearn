# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate of the TUNED int8 unit plans: every plan against the reference
unit plan, the flat plan and the host oracle.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_int8_mma_tuned_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_TUNED_SABOTAGE=1 -I . gemm/checks/gemm_int8_mma_tuned_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_SABOTAGE=1 -I . gemm/checks/gemm_int8_mma_tuned_check.mojo

The first must pass every gate. The second breaks the padding rule of the
staging and must FAIL the two gates that hold ragged `k`
(`check_tuned_plans_match_reference_flat_oracle`,
`check_tuned_planted_worst_cases`). The third flips every stored value and
must FAIL every gate. `tools/lowbit_mma_speed/unit_gate_job.sh` runs the
three and reads the logs. The parallel quantizer has its own gate,
`gemm/checks/quantize_int8_par_check.mojo`.

Lane lane/lowbit-mma-speed, 2026-09-29. Kernel
`gemm/checks/gemm_int8_mma_tuned.mojo`; the reference unit plan
`gemm/checks/gemm_int8_mma.mojo`; the flat plan
`gemm/checks/gemm_lowbit.mojo`; the answer
`gemm/host/gemm_lowbit_oracle.mojo::gemm_int8_oracle`.

WHAT A SCHEDULE CAN GET WRONG, and so what is planted. No plan here can
round: every sum is an integer. What a plan can do is read the wrong code
(a fragment index off by a row, a window's tail left stale, a pad that is
not the zero code), own the wrong cell, or store under the wrong row's
exponent. So the fixtures are:

    quantized floats   codes that differ at every position, so a fragment
                       read from the wrong row, column or k step changes
                       the sum;
    ragged extents     m and n off the 16-wide unit tile, the 64-wide warp
                       tile and the 128- and 256-wide blocks; k off the
                       unit's 32, the staging loads' 4 and 16 and the
                       windows' 32, 64 and 128;
    planted codes      lane/lowbit-units' five plants (every product
                       +16129, every product -16129, cancelling halves, odd
                       windows, the checkerboard) up to the profile's
                       largest k, where the Int32 holds 2114060288, and two
                       of this file's: a row of zero codes in each operand,
                       and one nonzero code alone in a row of zeros, at the
                       LAST step, so a tail that is dropped or stale shows;
    row exponents      that differ from row to row.

ON A COLUMN WITHOUT THE UNIT (Apple) no gate here runs, which main says
and which is not a pass.

MAIN RUNS EVERY GATE AND REPORTS EVERY VERDICT before it raises, and a gate
reports every case before it raises, as `gemm_lowbit_check.mojo` does and
for the same reason: under a sabotage build the evidence is which cases
the defect reaches.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.sys import has_accelerator

from checks.kernel_matrix import (
    COLUMN_NVIDIA,
    TARGET_COLUMN,
    column_name,
    lib_int8_matrix_unit_for,
)
from checks.numerics import numeric_mode_name
from gemm.checks.gemm_int8_apple_chunk_check import (
    PLANT_COUNT,
    PLANT_SHAPE_COUNT,
    _plant_a,
    _plant_b,
    _plant_exponents,
    _plant_name,
    _plant_shape,
)
from gemm.checks.gemm_int8_mma import identical_gemm_int8_mma_into
from gemm.checks.gemm_int8_mma_tuned import (
    INT8_DIRECT_PROBE_HOISTED,
    INT8_TUNED_PLAN_COUNT,
    identical_gemm_int8_mma_direct_into,
    identical_gemm_int8_mma_tuned_into,
    identical_gemm_int8_mma_tuned_with_plan,
    int8_direct_name,
    int8_tuned_dispatch,
    int8_tuned_plan_available,
    int8_tuned_plan_name,
    int8_tuned_sabotage_name,
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
from gemm.host.gemm_lowbit_oracle import (
    INT8_MAX_K,
    gemm_int8_oracle,
    quantize_rows_int8,
)
from gemm.host.gemm_oracle import OP_NT

#: Whether this build has the unit, and the direct kernel (NVIDIA only).
comptime HAS_UNIT = lib_int8_matrix_unit_for[TARGET_COLUMN]()
comptime HAS_DIRECT = TARGET_COLUMN == COLUMN_NVIDIA

#: The direct kernel's instantiations that are PLANS: the ones below its
#: first probe.
comptime DIRECT_PLAN_COUNT = INT8_DIRECT_PROBE_HOISTED


#: The ragged shapes this file adds: m and n off the 64-wide warp tile and
#: the 128- and 256-wide blocks; k off the staging loads (4100 is a multiple
#: of 4 and not of 16, 4104 of 8 and not of 16, 4112 of 16 and not of 32)
#: and off the windows (63, 65, 127, 129, 191).
comptime TUNED_SHAPE_COUNT = 12


def _tuned_shape(i: Int) -> Tuple[Int, Int, Int]:
    if i == 0:
        return (65, 129, 63)
    if i == 1:
        return (63, 127, 65)
    if i == 2:
        return (129, 65, 127)
    if i == 3:
        return (127, 257, 129)
    if i == 4:
        return (130, 255, 191)
    if i == 5:
        return (17, 300, 4100)
    if i == 6:
        return (1, 257, 4104)
    if i == 7:
        return (16, 513, 4112)
    if i == 8:
        return (8, 128, 4096)
    if i == 9:
        return (128, 128, 256)
    if i == 10:
        return (64, 256, 1000)
    return (200, 40, 96)


def _run_every_plan(
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
) raises:
    """One shape and one pair of operands through the flat plan, the
    reference unit plan, every staged plan and every direct plan, each
    against the oracle and against the reference. The operands are uploaded
    once; every plan writes its own poisoned output."""
    var want = gemm_int8_oracle(qa, ea, qb, eb, m, n, k)
    var dqa = _upload_i8(ctx, qa)
    var dea = _upload_i32(ctx, ea)
    var dqb = _upload_i8(ctx, qb)
    var deb = _upload_i32(ctx, eb)
    var dflat = _poisoned(ctx, m * n)
    var dref = _poisoned(ctx, m * n)
    identical_gemm_int8_flat_into(ctx, dflat, dqa, dea, dqb, deb, m, n, k)
    identical_gemm_int8_mma_into(ctx, dref, dqa, dea, dqb, deb, m, n, k)
    ctx.synchronize()
    var flat = _download_f32(ctx, dflat, m * n, tag + " flat")
    var ref_ = _download_f32(ctx, dref, m * n, tag + " reference")
    tally.note(_diff(flat, want, tag + " (flat vs oracle)"))
    tally.note(_diff(ref_, want, tag + " (reference unit plan vs oracle)"))
    for plan in range(INT8_TUNED_PLAN_COUNT):
        if not int8_tuned_plan_available(plan):
            continue
        var ptag = tag + " " + int8_tuned_plan_name(plan)
        var dc = _poisoned(ctx, m * n)
        var got: String
        # A launch the device refuses is a failed case of THAT plan, named,
        # and the plans after it still run (nvc3-0018: a refused launch
        # outside this `try` ended the gate without saying whose it was).
        try:
            identical_gemm_int8_mma_tuned_with_plan(ctx, dc, dqa, dea, dqb, deb, m, n, k, plan)
            ctx.synchronize()
            var out = _download_f32(ctx, dc, m * n, ptag)
            got = _diff(out, want, ptag + " (tuned vs oracle)")
            if got.byte_length() == 0:
                got = _diff(out, ref_, ptag + " (tuned vs reference unit plan)")
            if got.byte_length() == 0:
                got = _diff(out, flat, ptag + " (tuned vs flat)")
        except e:
            got = ptag + ": " + String(e)
        tally.note(got)
        _ = dc
    comptime if HAS_DIRECT:
        for which in range(DIRECT_PLAN_COUNT):
            var ptag = tag + " " + int8_direct_name(which)
            var dc = _poisoned(ctx, m * n)
            var got: String
            try:
                identical_gemm_int8_mma_direct_into(ctx, dc, dqa, dea, dqb, deb, m, n, k, which)
                ctx.synchronize()
                var out = _download_f32(ctx, dc, m * n, ptag)
                got = _diff(out, want, ptag + " (direct vs oracle)")
                if got.byte_length() == 0:
                    got = _diff(out, ref_, ptag + " (direct vs reference unit plan)")
            except e:
                got = ptag + ": " + String(e)
            tally.note(got)
            _ = dc
    _ = dqa
    _ = dea
    _ = dqb
    _ = deb
    _ = dflat
    _ = dref


# ===========================================================================
# THE UNIT GATES
# ===========================================================================


def check_tuned_plans_match_reference_flat_oracle(ctx: DeviceContext) raises:
    """GATE: on codes quantized from the file's float fixtures every staged
    plan and every direct plan returns the oracle's bits, the reference unit
    plan's and the flat plan's, at every OP_NT shape of
    `gemm_lowbit_check.mojo`, its ragged shapes and this file's."""
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
        var tag = "int8 tuned " + String(m) + "x" + String(n) + "x" + String(k)
        var before = tally.failed
        _run_every_plan(ctx, qa.q, qa.e, qb.q, qb.e, m, n, k, tag, tally)
        shapes += 1
        if tally.failed == before:
            print("   ok " + tag)
    print("   shapes: " + String(shapes))
    _verdict(tally, String("quantized-fixture"))


comptime TUNED_PLANT_ZERO_ROWS = PLANT_COUNT
comptime TUNED_PLANT_LONE_LAST = PLANT_COUNT + 1
comptime TUNED_PLANT_COUNT = PLANT_COUNT + 2


def _tuned_plant_name(plant: Int) -> String:
    if plant == TUNED_PLANT_ZERO_ROWS:
        return String("zero-rows")
    if plant == TUNED_PLANT_LONE_LAST:
        return String("lone-last-step")
    return _plant_name(plant)


def _tuned_plant_a(plant: Int, m: Int, k: Int) -> List[Int8]:
    """The left operand. ZERO_ROWS: +127, and every third row zero codes.
    LONE_LAST: zero codes, and -127 at the last step of every row."""
    if plant < PLANT_COUNT:
        return _plant_a(plant, m, k)
    var q = List[Int8]()
    for i in range(m):
        for p in range(k):
            var v = Int8(127)
            if plant == TUNED_PLANT_ZERO_ROWS:
                if i % 3 == 1:
                    v = Int8(0)
            else:
                v = Int8(0)
                if p == k - 1:
                    v = Int8(-127)
            q.append(v)
    return q^


def _tuned_plant_b(plant: Int, n: Int, k: Int) -> List[Int8]:
    """The right operand. ZERO_ROWS: -127, and every fifth row zero codes.
    LONE_LAST: +126 everywhere, so only the last step's product is not 0."""
    if plant < PLANT_COUNT:
        return _plant_b(plant, n, k)
    var q = List[Int8]()
    for j in range(n):
        for _ in range(k):
            var v = Int8(-127)
            if plant == TUNED_PLANT_ZERO_ROWS:
                if j % 5 == 2:
                    v = Int8(0)
            else:
                v = Int8(126)
            q.append(v)
    return q^


#: The planted shapes this file adds to lane/lowbit-units': off this file's
#: tiles and windows, at small k so the host oracle stays cheap.
comptime TUNED_PLANT_SHAPE_COUNT = 5


def _tuned_plant_shape(i: Int) -> Tuple[Int, Int, Int]:
    if i < PLANT_SHAPE_COUNT:
        return _plant_shape(i)
    var j = i - PLANT_SHAPE_COUNT
    if j == 0:
        return (129, 257, 96)
    if j == 1:
        return (65, 130, 161)
    if j == 2:
        return (17, 33, 4100)
    if j == 3:
        return (3, 300, 4112)
    return (131, 5, 33)


def check_tuned_planted_worst_cases(ctx: DeviceContext) raises:
    """GATE: the planted codes through every plan, against the oracle, the
    reference unit plan and the flat plan. Every case is run and printed;
    the gate raises after the last one with the count and the first
    failure. With `check_tuned_plans_match_reference_flat_oracle` it is the
    gate the staging sabotage must fail, and it says at which k."""
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
                "int8 tuned planted " + _tuned_plant_name(plant) + " " + String(m) + "x"
                + String(n) + "x" + String(k)
            )
            var before = tally.failed
            _run_every_plan(ctx, qa, ea, qb, eb, m, n, k, tag, tally)
            if tally.failed == before:
                print("   ok " + tag)
    _verdict(tally, String("planted"))


def _run_dispatched(
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
    """`identical_gemm_int8_mma_tuned_into` on uploaded left codes."""
    var dqa = _upload_i8(ctx, qa)
    var dea = _upload_i32(ctx, ea)
    var dc = _poisoned(ctx, m * n)
    identical_gemm_int8_mma_tuned_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
    ctx.synchronize()
    var out = _download_f32(ctx, dc, m * n, tag)
    _ = dqa
    _ = dea
    _ = dc
    return out^


def check_tuned_dispatch_is_batch_invariant(ctx: DeviceContext) raises:
    """GATE: through the plan the launcher picks, row `i` of an `m`-row
    call equals the one-row call on row `i` alone. The two calls take
    DIFFERENT plans (the block plan and the ROW plan), so this is also the
    launcher's choice shown to be scheduling; and the batch equals the
    oracle."""
    var m = 40
    var n = 150
    var k = 1000
    var qa = quantize_rows_int8(_fill(m * k, 401), m, k)
    var qb = quantize_rows_int8(_fill(n * k, 409), n, k)
    var want = gemm_int8_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
    var dqb = _upload_i8(ctx, qb.q)
    var deb = _upload_i32(ctx, qb.e)
    var batch = _run_dispatched(ctx, qa.q, qa.e, dqb, deb, m, n, k, String("tuned batch"))
    _first_diff(batch, want, String("tuned batch (dispatched vs oracle)"))
    if int8_tuned_dispatch(m, n, k) == int8_tuned_dispatch(1, n, k):
        raise Error("the batch and the single row took the same plan; the gate compares nothing")
    for i in range(m):
        var row = List[Int8]()
        for p in range(k):
            row.append(qa.q[i * k + p])
        var er = List[Int32]()
        er.append(qa.e[i])
        var one = _run_dispatched(ctx, row, er, dqb, deb, 1, n, k, "tuned row " + String(i))
        for j in range(n):
            if bitcast[DType.uint32](one[j]) != bitcast[DType.uint32](batch[i * n + j]):
                raise Error(
                    "tuned row " + String(i) + " col " + String(j)
                    + " differs between the batch and the single-row call: "
                    + _show(batch[i * n + j]) + " vs " + _show(one[j])
                )
    _ = dqb
    _ = deb
    print(
        "   ok " + String(m) + " rows agree with their single-row calls; plans "
        + int8_tuned_plan_name(int8_tuned_dispatch(m, n, k)) + " and "
        + int8_tuned_plan_name(int8_tuned_dispatch(1, n, k))
    )


def main() raises:
    print(
        "== gemm/checks/gemm_int8_mma_tuned_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + int8_tuned_sabotage_name() + " =="
    )
    print("   profile: mojolearn.identical.gemm.int8i32.v1, the tuned unit plans")
    print("   column: " + column_name(TARGET_COLUMN))
    var ran = 0
    var failed = 0
    comptime if not has_accelerator():
        print("   no accelerator: the device gates DID NOT RUN, which is not a pass")
        raise Error("the tuned int8 gate needs a device")
    else:
        var ctx = DeviceContext()
        comptime if HAS_UNIT:
            for plan in range(INT8_TUNED_PLAN_COUNT):
                if int8_tuned_plan_available(plan):
                    print("   plan " + String(plan) + ": " + int8_tuned_plan_name(plan))
                else:
                    print(
                        "   plan " + String(plan) + ": " + int8_tuned_plan_name(plan)
                        + "  NOT RUN on this column (a block of it is above 1024 threads),"
                        + " which is not a pass"
                    )
            try:
                check_tuned_plans_match_reference_flat_oracle(ctx)
                _gate(String("check_tuned_plans_match_reference_flat_oracle"), ran, failed, String(""))
            except e:
                _gate(String("check_tuned_plans_match_reference_flat_oracle"), ran, failed, String(e))
            try:
                check_tuned_planted_worst_cases(ctx)
                _gate(String("check_tuned_planted_worst_cases"), ran, failed, String(""))
            except e:
                _gate(String("check_tuned_planted_worst_cases"), ran, failed, String(e))
            try:
                check_tuned_dispatch_is_batch_invariant(ctx)
                _gate(String("check_tuned_dispatch_is_batch_invariant"), ran, failed, String(""))
            except e:
                _gate(String("check_tuned_dispatch_is_batch_invariant"), ran, failed, String(e))
            print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
            if failed > 0:
                raise Error(String(failed) + " of " + String(ran) + " gates failed")
        else:
            # Not a pass: the gates did not run. The column has no integer
            # matrix unit, so no unit plan can be launched here.
            print(
                "   THE UNIT GATES DID NOT RUN, which is not a pass: column "
                + column_name(TARGET_COLUMN) + " has no int8 matrix unit"
            )
            raise Error("the tuned int8 gate needs a column with the integer matrix unit")
