# SPDX-License-Identifier: Apache-2.0
"""Native scalar validation, unit decisions and Float32 class-weight words."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.class_weight_plan import (
    ClassWeightInputs, class_weight_kind, class_weight_value,
    class_weight_all_one, class_weight_plan,
)

comptime CLASS_WEIGHT_PLAN_FIXTURES = 24


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def publication(unit: Int) raises:
    var sizes: List[Int] = [0, 1, 3, 257, 65535, 65537, 131073, 262271]
    var count = sizes[unit]
    var values = List[Float64](unsafe_uninit_length=count)
    var source = values.unsafe_ptr()
    var tasks = host_predict_task_count_work(max(1, count), max(1, count))
    var chunk = host_predict_chunk(max(1, count), tasks)
    def input(worker: Int) {imm source, imm count, imm chunk}:
        for i in range(worker * chunk, min((worker + 1) * chunk, count)):
            source[i] = 0.5 if i % 3 == 0 else (2.0 if i % 3 == 1 else -0.0)
    host_parallelize(input, tasks)
    var all_one = class_weight_all_one(values)
    require(all_one == (count == 0), "balanced scalar unit decision")
    var plan = class_weight_plan(values, all_one)
    require(plan.all_one == (count == 0) and len(plan.words) == count, "weight plan extent")
    var actual = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(plan.words.unsafe_ptr()))
    var failed = List[Int](length=tasks, fill=0)
    var fp = failed.unsafe_ptr()
    def check(worker: Int) {imm actual, imm count, imm chunk, imm fp}:
        var errors = 0
        for i in range(worker * chunk, min((worker + 1) * chunk, count)):
            var expected = UInt32(0x3F000000) if i % 3 == 0 else (UInt32(0x40000000) if i % 3 == 1 else UInt32(0x80000000))
            errors += Int(bitcast[DType.uint32](actual[i]) != expected)
        fp[worker] = errors
    host_parallelize(check, tasks)
    for errors in failed:
        require(errors == 0, "Float32 publication changed expected words")
    _ = plan^
    _ = values^


def refusals(unit: Int) raises:
    var message = String()
    try:
        if unit < 4:
            var kinds: List[Int] = [0, 1, 2, 5]
            class_weight_kind(kinds[unit])
        elif unit < 8:
            var words: List[UInt64] = [0xBFF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000042]
            class_weight_value(bitcast[DType.float64](words[unit - 4]))
        else:
            var values: List[Float64] = [0.0]
            if unit == 8:
                values[0] = 1.0e-300
            elif unit == 9:
                values[0] = 1.0e300
            elif unit == 11:
                values[0] = -0.0
            _ = class_weight_plan(values, class_weight_all_one(values))
    except error:
        message = String(error)
    var expected = String("class weights must be finite nonnegative real numbers")
    if unit == 8 or unit == 9:
        expected = "class weights must remain finite and nonzero when positive in Float32"
    elif unit >= 10:
        expected = "class weights must have positive total"
    require(message == expected, "class weight refusal/order changed")


def scalar_contract(unit: Int) raises:
    if unit == 0 or unit == 1:
        var inputs = ClassWeightInputs()
        inputs.append(1.0, unit == 0)
        var plan = class_weight_plan(inputs.values, inputs.all_one)
        require(plan.all_one == (unit == 0), "exact source unit flag was rounded away")
        require(len(plan.words) == (0 if unit == 0 else 1), "source equality controls row materialization")
        if unit == 1:
            require(bitcast[DType.uint32](plan.words[0]) == UInt32(0x3F800000), "rounded unit remains Float32 one")
        return
    if unit == 2:
        var smallest = Float64(bitcast[DType.float32](UInt32(1)))
        var largest = Float64(bitcast[DType.float32](UInt32(0x7F7FFFFF)))
        var values: List[Float64] = [smallest, largest]
        var plan = class_weight_plan(values, class_weight_all_one(values))
        require(bitcast[DType.uint32](plan.words[0]) == UInt32(1)
            and bitcast[DType.uint32](plan.words[1]) == UInt32(0x7F7FFFFF), "Float32 representable endpoints")
        return
    class_weight_kind(3)
    class_weight_kind(4)
    var inputs = ClassWeightInputs()
    inputs.append(0.5, False)
    var refused = False
    try:
        inputs.append(-1.0, False)
    except error:
        refused = String(error) == "class weights must be finite nonnegative real numbers"
    require(refused and len(inputs.values) == 1 and inputs.values[0] == 0.5,
            "invalid scalar was published into native input state")


def run_checks() raises:
    var failed = List[Int](length=CLASS_WEIGHT_PLAN_FIXTURES, fill=0)
    var fp = failed.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit < 8:
                publication(unit)
            elif unit < 20:
                refusals(unit - 8)
            else:
                scalar_contract(unit - 20)
        except error:
            fp[unit] = 1
            print("FAIL class_weight_plan fixture", unit, String(error))
    host_parallelize(task, CLASS_WEIGHT_PLAN_FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    require(failures == 0, "class_weight_plan failed " + String(failures) + " fixtures")
    print("PASS class_weight_plan:", CLASS_WEIGHT_PLAN_FIXTURES, "native validation/publication fixtures")
