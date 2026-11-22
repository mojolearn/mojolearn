# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Native scalar validation, unit decisions and Float32 class-weight words.

`test_class_weight_rows` (lane py2mojo-C, 2026-11-22) runs the row pipeline every class-weighted fit runs
(bindings/class_weight_rows_boundary_pack.mojo `class_weight_plan_admitted` and `class_weight_rows_admitted`, the
same steps in the same order: COUNTS, the Int64 cast, the balanced ratio n / (k n_c), the Float32 plan, ROWS, then
the door's MULTIPLY against `sample_weight`) and checks the row words the fit receives, ported from
`test_class_weight_shared_families.py::test_shared_class_weights_match_explicit_sample_weights` (balanced, partial
and unit over its 8-row fixture; the three families share this pipeline, their scalar and dictionary admission and
the fitted `classes_` stay in Python) and `test_rf_class_weight.py::test_balanced_and_partial_mapping` (40 rows,
codes stored by `tools/py2mojo_native_fixtures_b4.py`). The public tests compare two fits; the fits read nothing of
the class weights but these row words, so the words are what is asserted here, bit for bit.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.label_weights_native_api import (
    LabelWeightsDriver, LabelWeightsRequest, LABEL_WEIGHTS_COUNTS, LABEL_WEIGHTS_ROWS, LABEL_WEIGHTS_MULTIPLY,
)
from xtrees.label_weights_native import label_weights_native_v1
from core.labels_native_api import LabelsRatioRequest
from core.labels_native_host import LabelsHostKernels
from core.admitted_cast_elements import admitted_cast_elements
from bindings.class_weight_rows_run import class_weight_rows_plan
from checks.native_tests.tsa_metrics_fixture_words import fixture_i32
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


# ---- the class-weight row pipeline (module docstring) ----
def weight_rows(codes: List[Int32], k: Int, var inputs: ClassWeightInputs, balanced: Bool,
                sample: List[Float32]) raises -> List[Float32]:
    """The row weights a class-weighted fit receives; empty when the plan is all one (the unweighted or
    sample-weight-only fit)."""
    var n = len(codes)
    var driver = LabelWeightsDriver(label_weights_native_v1)
    if balanced:
        var counts = List[Int32](length=k, fill=0)
        _ = driver.run(LABEL_WEIGHTS_COUNTS, LabelWeightsRequest(Int(codes.unsafe_ptr()), Int(counts.unsafe_ptr()),
            0, n, k))
        var wide = List[Int64](length=k, fill=0)
        require(admitted_cast_elements(Int(counts.unsafe_ptr()), 2, Int(wide.unsafe_ptr()), 3, k) == 0, "count cast")
        inputs.values = List[Float64](length=k, fill=0.0)
        require(LabelsHostKernels.ratio(LabelsRatioRequest(Int(wide.unsafe_ptr()),
            Int(inputs.values.unsafe_ptr()), k, n, 0)) == 0, "balanced ratio")
        _ = counts^
        _ = wide^
    var plan = class_weight_rows_plan(inputs, balanced)
    if plan.all_one:
        if len(sample) == 0:
            return List[Float32]()
        return sample.copy()
    var rows = List[Float32](length=n, fill=Float32(-1))
    _ = driver.run(LABEL_WEIGHTS_ROWS, LabelWeightsRequest(Int(codes.unsafe_ptr()), Int(plan.words.unsafe_ptr()),
        Int(rows.unsafe_ptr()), n, k))
    if len(sample) == 0:
        return rows^
    var product = List[Float32](length=n, fill=Float32(-1))
    _ = driver.run(LABEL_WEIGHTS_MULTIPLY, LabelWeightsRequest(Int(rows.unsafe_ptr()), Int(sample.unsafe_ptr()),
        Int(product.unsafe_ptr()), n, 0))
    _ = rows^
    _ = plan^
    return product^


def same_words(got: List[Float32], want: List[Float32], label: String) raises:
    require(len(got) == len(want), label + ": " + String(len(got)) + " row weights, expected " + String(len(want)))
    for i in range(len(want)):  # small-loop(oracle: one fixture's rows)
        require(bitcast[DType.uint32](got[i]) == bitcast[DType.uint32](want[i]), label + ": row " + String(i) + " is "
            + String(got[i]) + ", expected " + String(want[i]))


def class_weight_rows(unit: Int) raises:
    if unit <= 2:
        # test_shared_class_weights_match_explicit_sample_weights: classes [1, 7], codes of y = [1, 1, 7 x 6]
        var codes: List[Int32] = [0, 0, 1, 1, 1, 1, 1, 1]
        var sample: List[Float32] = [0.5, 1.0, 2.0, 1.0, 0.5, 2.0, 1.0, 1.0]
        var inputs = ClassWeightInputs()
        var per_class = List[Float32](length=8, fill=Float32(1))
        var label = String("shared ")
        if unit == 0:
            label += "balanced"
            for i in range(8):  # small-loop(np.where(y == 1, 2.0, 2.0 / 3).astype("f4"))
                per_class[i] = Float32(2.0) if codes[i] == 0 else Float32(2.0 / 3.0)
        elif unit == 1:
            # {7: 0.25}: the missing class 1 is 1 (the dictionary admission's default)
            label += "partial"
            inputs.append(1.0, True)
            inputs.append(0.25, False)
            for i in range(8):  # small-loop(np.where(y == 7, 0.25, 1).astype("f4"))
                per_class[i] = Float32(0.25) if codes[i] == 1 else Float32(1)
        else:
            # {1.0: float32(1)}: every class exactly one
            label += "unit"
            inputs.append(1.0, True)
            inputs.append(1.0, True)
        var want = List[Float32](length=8, fill=Float32(0))
        for i in range(8):  # small-loop(sample * per_class in float32)
            want[i] = sample[i] * per_class[i]
        var before = sample.copy()
        same_words(weight_rows(codes, 2, inputs^, unit == 0, sample), want, label)
        same_words(sample, before, label + " (sample_weight untouched)")
        return
    # test_balanced_and_partial_mapping: 40 codes, 'a' 0 and 'b' 1, no sample weights
    var codes = fixture_i32("checks/native_tests/fixtures/trees_b4/cw-codes.i32", 40)
    var n_a = 0
    for c in codes:  # small-loop(the 40 codes)
        if c == 0:
            n_a += 1
    var want = List[Float32](length=40, fill=Float32(0))
    var inputs = ClassWeightInputs()
    var label = String("rf ")
    if unit == 3:
        label += "balanced"
        for i in range(40):  # small-loop(np.where(y == 'a', n / (2 n_a), n / (2 n_b)).astype(float32))
            want[i] = Float32(40.0 / (2.0 * Float64(n_a))) if codes[i] == 0 else Float32(40.0 / (2.0 * Float64(40 - n_a)))
    else:
        # {'a': 0}: rows of 'a' weigh 0, 'b' (missing) 1
        label += "partial"
        inputs.append(0.0, False)
        inputs.append(1.0, True)
        for i in range(40):  # small-loop(np.where(y == 'a', 0.0, 1.0).astype(float32))
            want[i] = Float32(0) if codes[i] == 0 else Float32(1)
    same_words(weight_rows(codes, 2, inputs^, unit == 3, List[Float32]()), want, label)


def test_class_weight_rows() raises:
    check_cases("class_weight_rows", 5, class_weight_rows)


def test_publication() raises:
    check_cases("publication", 8, publication)


def test_refusals() raises:
    check_cases("refusals", 12, refusals)


def test_scalar_contract() raises:
    check_cases("scalar_contract", 4, scalar_contract)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
