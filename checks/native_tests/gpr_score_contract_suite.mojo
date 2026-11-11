# SPDX-License-Identifier: Apache-2.0
"""Pure score storage, snapshot admission and input-range ownership checks."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.gp_native_score import GPRScoreValues, GPRScoreProgram, gpr_score_prepare
from checks.native_tests.gp_state_suite import require

comptime GPR_SCORE_CONTRACT_CASES = 7


def fixture(unit: Int) raises:
    if unit == 0:
        var empty = GPRScoreValues(0)
        require(len(empty.prediction) == 0 and len(empty.scalar) == 1 and empty.scalar[0] == 0,
            "empty score storage keeps one zero scalar")
        var values = GPRScoreValues(3)
        require(len(values.prediction) == 3 and values.prediction[0] == 0
            and values.prediction[2] == 0 and values.scalar[0] == 0, "zero initialized score storage")
        var prediction = values.prediction_address()
        var scalar = values.scalar_address()
        var moved = values^
        require(moved.prediction_address() == prediction and moved.scalar_address() == scalar,
            "whole score-storage move retains both allocations")
        _ = empty
        _ = moved
        return
    if unit == 6:
        var source: List[Float32] = [1.0, 2.0, 3.0, 4.0]
        var program = GPRScoreProgram(4)
        program.input(0, 0, 0, -1)
        require(len(program.inputs) == 0, "zero input span neither reads nor records a range")
        program.input(0, 2, Int(source.unsafe_ptr()), -1)
        program.input(2, 2, Int(source.unsafe_ptr() + 2), -1)
        require(len(program.inputs) == 3 and program.inputs[0] == 0
            and program.inputs[1] == 4 and program.inputs[2] == -1,
            "adjacent admitted host chunks form one upload interval")
        source[0] = 99.0
        require(program.arena[0] == Float32(1.0) and program.arena[3] == Float32(4.0),
            "input-range storage retains the original admitted values")
        _ = source
        _ = program
        return
    var values = GPRScoreValues(2)
    values.prediction = [1.0, 2.0]
    var truth: List[Float32] = [1.0, 2.0]
    var nan_value = bitcast[DType.float32](UInt32(0x7FC00000))
    var infinity = bitcast[DType.float32](UInt32(0x7F800000))
    if unit == 1 or unit == 5:
        truth[0] = nan_value
    elif unit == 2:
        truth[0] = infinity
    if unit == 3:
        values.prediction[0] = nan_value
    elif unit == 4 or unit == 5:
        values.prediction[0] = infinity
    var expected = String("y_true") if unit <= 2 or unit == 5 else String("y_pred")
    var refused = False
    try:
        var program = gpr_score_prepare(values, Int(truth.unsafe_ptr()))
        _ = program
    except problem:
        require(String(problem) == "mojolearn score: " + expected + " contains NaN or infinity",
            "finite refusal preserves target-before-prediction precedence")
        refused = True
    require(refused, "nonfinite metric input was accepted")
    _ = values
    _ = truth


def run_checks() raises:
    var failures = List[Int](length=GPR_SCORE_CONTRACT_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(unit)
        except problem:
            flags[unit] = 1
            print("FAIL gpr_score_contract fixture", unit, String(problem))
    host_parallelize(run, GPR_SCORE_CONTRACT_CASES)
    var failed = 0
    for flag in failures:  # metadata: independent fixture statuses
        failed += flag
    require(failed == 0, "GPR score contract failures: " + String(failed))
    print("PASS gpr_score_contract:", GPR_SCORE_CONTRACT_CASES, "fixtures")
    _ = failures
