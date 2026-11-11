# SPDX-License-Identifier: Apache-2.0
"""Native preference promotion, rounding, shape/refusal and storage contracts."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.sequential_input_plan import SequenceInputPlan
from core.borrowed_buffer import BorrowedBufferLayout
from core.affinity_preference_plan import AffinityPreferencePlan

comptime AFFINITY_PREFERENCE_PLAN_FIXTURES = 10


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def fixture(unit: Int) raises:
    var result = AffinityPreferencePlan()
    if unit == 9:
        require(result.span().address == 0 and result.span().count == 0, "absent preference has no scratch")
        return
    var plan = SequenceInputPlan()
    plan.container(0, 1)
    var singles: List[Float32] = [-2, 0.5, 7]
    var doubles: List[Float64] = [-3, 999, 1.25, 999, 8, 999]
    if unit == 7 or unit == 8:
        var shape: List[Int] = [3]
        var strides: List[Int] = [4 if unit == 7 else 16]
        var indirect: List[Int] = [-1]
        var layout = BorrowedBufferLayout(Int(singles.unsafe_ptr()) if unit == 7 else Int(doubles.unsafe_ptr()),
            4 if unit == 7 else 8, True, shape^, strides^, indirect^)
        plan.array(1, layout^, "f", True)
    else:
        var count = 3 if unit == 0 else (2 if unit == 1 else 1)
        plan.container(1, count)
        if unit == 0:
            plan.scalar(2, bitcast[DType.uint64](Int64(-1)), "i", 8)
            plan.scalar(2, 0, "i", 8)
            plan.scalar(2, 16777217, "i", 8)
        elif unit == 1:
            plan.scalar(2, bitcast[DType.uint64](Int64(-1)), "i", 8)
            # A value one integer above the Float32 midpoint: Float64 first
            # loses the +1, then ties to the even lower Float32 significand.
            plan.scalar(2, (UInt64(1) << 63) + (UInt64(1) << 39) + 1, "u", 8)
        elif unit == 2 or unit == 3:
            plan.scalar(2, UInt64.MAX, "u", 8)
        elif unit == 6:
            plan.container(2, 1)
            plan.scalar(3, 1, "i", 8)
        else:
            plan.scalar(2, UInt64(0x7ff0000000000000) if unit == 4 else UInt64(0x7ff8000000000001), "f", 8)
    var refused = False
    try:
        result.prepare(plan, unit != 3)
    except problem:
        if unit == 3:
            require(String(problem).find("holds integers beyond the int64 range") >= 0, "no-NumPy unsigned refusal")
        elif unit == 4 or unit == 5:
            require(String(problem) == "mojolearn: preference contains NaN or infinity", "nonfinite preference refusal")
        elif unit == 6:
            require(String(problem) == "mojolearn: preference must be 2-D, got 3-D shape (1, 1, 1)", "rank refusal includes native shape")
        else:
            raise Error("unexpected preference refusal: " + String(problem))
        refused = True
    require(refused == (unit >= 3 and unit <= 6), "expected preference refusal was lost")
    if not refused:
        var values = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=result.span().address)
        if unit == 0:
            require(values[0] == -1 and values[1] == 0 and values[2] == 16777216, "signed integer Float32 conversion")
        elif unit == 1:
            require(values[0] == -1 and bitcast[DType.uint32](values[1]) == UInt32(0x5f000000), "mixed integer promotion retains double rounding")
        elif unit == 2:
            require(bitcast[DType.uint32](values[0]) == UInt32(0x5f800000), "unsigned native preference conversion")
        elif unit == 7:
            require(result.direct and len(result.values) == 0 and len(result.inferred) == 12,
                "already Float32 uses its inferred storage without another tensor copy")
            require(values[0] == -2 and values[1] == 0.5 and values[2] == 7, "direct Float32 values")
        else:
            require(not result.direct and len(result.inferred) == 0 and len(result.values) == 3,
                "inferred intermediate released after the required cast")
            require(values[0] == -3 and values[1] == 1.25 and values[2] == 8, "strided Float64 conversion")
    _ = result^
    _ = plan^
    _ = singles^
    _ = doubles^


def run_checks() raises:
    var failures = List[Int](length=AFFINITY_PREFERENCE_PLAN_FIXTURES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(unit)
        except problem:
            flags[unit] = 1
            print("FAIL affinity_preference_plan fixture", unit, String(problem))
    host_parallelize(run, AFFINITY_PREFERENCE_PLAN_FIXTURES)
    var failed = 0
    for value in failures:  # metadata: independent fixture statuses
        failed += value
    require(failed == 0, "affinity preference plan failed " + String(failed) + " fixtures")
    print("PASS affinity_preference_plan:", AFFINITY_PREFERENCE_PLAN_FIXTURES, "native contracts")
