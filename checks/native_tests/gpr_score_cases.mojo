# SPDX-License-Identifier: Apache-2.0
"""Native GPR score composition with analytic and independent saved oracles.

The original 60x4 case retains the public default kernel, ridge, data stream
and 1e-3 score tolerance. Other cases are supplementary score-policy checks.
"""
from std.math import abs, isfinite
from core.native_descriptor_abi import NativeDescriptorCall
from core.gp_native_api import GPDriver, GPRFitRequest, GPRQueryRequest, GP_CAPABILITIES, GP_CAP_POINTER, GP_CAP_RESIDENT, GP_ROUTE_LIST, GP_ROUTE_POINTER, GP_ROUTE_RESIDENT
from core.gp_native_storage import GPKernelStorage
from core.gp_native_score import GPRScoreValues, GPRScoreProgram, gpr_score_prepare, gp_r2_value
from checks.native_tests.gp_state_suite import require
from checks.native_tests.gp_native_data import input_values, reference_values, GP_GUARD, GP_GUARD64

comptime GPR_SCORE_NATIVE_CASES = 10


def default_kernel() -> GPKernelStorage:
    # Exactly the public default Constant(1, fixed) * RBF(1, fixed).
    var value = GPKernelStorage()
    value.kinds = [0, 2, 5]
    value.parameters = [1.0, 0.0, 0.0]
    value.lengths = [0, 1, 0]
    value.scales = [1.0]
    value.finish()
    return value^


def owned_program(unit: Int) raises -> GPRScoreProgram:
    var rows = 2 if unit == 3 else 4
    var values = GPRScoreValues(rows)
    var truth: List[Float32] = [0.0, 1.0, 2.0, 4.0]
    if unit == 0:
        values.prediction = [0.5, 1.0, 1.0, 5.0]
    elif unit == 1:
        truth = [3.0, 3.0, 3.0, 3.0]
        values.prediction = [3.0, 3.0, 3.0, 3.0]
    elif unit == 2:
        truth = [3.0, 3.0, 3.0, 3.0]
        values.prediction = [2.0, 3.0, 4.0, 3.0]
    else:
        # The exact mean is 2**24+1, which rounds to 2**24 in Float32.
        # The established GP metric therefore has SS_tot=4, not 2.
        truth = [16777216.0, 16777218.0]
        values.prediction = [16777216.0, 16777216.0]
    var program = gpr_score_prepare(values, Int(truth.unsafe_ptr()))
    require(len(program.inputs) == 3 and program.inputs[0] == 0
        and Int(program.inputs[1]) == 2 * rows + 1 and program.inputs[2] == -1,
        "truth/prediction/scalar spans form one contiguous host input record")
    # The returned program must own the admitted snapshots. Neither source
    # owner survives this function; mutation also distinguishes snapshots.
    truth[0] = -97.0
    values.prediction[0] = 89.0
    values.scalar[0] = 71.0
    _ = truth
    _ = values
    return program^


def analytic_case(metrics: NativeDescriptorCall, unit: Int, ranges: Bool) raises:
    var program = owned_program(unit)
    var address = Int(program.arena.unsafe_ptr())
    var moved = program^
    try:
        require(Int(moved.arena.unsafe_ptr()) == address, "moving score owner retains its allocation")
        var sums = moved.run(metrics, ranges)
        var expected_num = Float64(2.25 if unit == 0 else (0.0 if unit == 1 else (2.0 if unit == 2 else 4.0)))
        var expected_den = Float64(8.75 if unit == 0 else (4.0 if unit == 3 else 0.0))
        require(sums.numerator == expected_num and sums.denominator == expected_den,
            "analytic residual/target sums retain complete binary64 outputs")
        var expected = Float64(26.0 / 35.0 if unit == 0 else (1.0 if unit == 1 else 0.0))
        var score = gp_r2_value(sums)
        require(Bool(isfinite(score)) and abs(score - expected) <= 0.00000000000001,
            "independent rational/constant-target/rounded-mean score oracle")
    finally:
        _ = moved


def original_case(driver: GPDriver, metrics: NativeDescriptorCall, ranges: Bool) raises:
    var rows = 60
    var features = 4
    var source = input_values("gpr-score60", "source", rows * features)
    var target = input_values("gpr-score60", "target", rows)
    var reference = reference_values("gpr-score60", "reference", 3)
    var kernel = default_kernel()
    var factor = List[Float32](length=rows * rows + 2, fill=GP_GUARD)
    var dual = List[Float32](length=rows + 2, fill=GP_GUARD)
    var scalars = List[Float64](length=7, fill=GP_GUARD64)
    var values = GPRScoreValues(rows)
    var lease = 0
    try:
        var capabilities = driver.metadata(GP_CAPABILITIES)
        var status = driver.gpr_fit(GPRFitRequest(Int(source.unsafe_ptr()), Int(target.unsafe_ptr()),
            Int(factor.unsafe_ptr() + 1), Int(dual.unsafe_ptr() + 1), Int(scalars.unsafe_ptr() + 1),
            rows, features, kernel.view(), 0.00000095367431640625, (capabilities & GP_CAP_POINTER) != 0))
        require(status == 0, "original default GPR fit succeeds")
        var route = GP_ROUTE_POINTER if (capabilities & GP_CAP_POINTER) != 0 else GP_ROUTE_LIST
        if (capabilities & GP_CAP_RESIDENT) != 0:
            lease = driver.lease_open()
            require(lease > 0, "resident score query owns a live lease")
            route = GP_ROUTE_RESIDENT
        # Mean-only score keeps the real pointer/resident query route and
        # leaves the unrequested variance/std/clamp addresses absent.
        var request = GPRQueryRequest(Int(source.unsafe_ptr()), Int(factor.unsafe_ptr() + 1),
            Int(dual.unsafe_ptr() + 1), Int(source.unsafe_ptr()), values.prediction_address(),
            0, 0, 0, rows, features, rows, kernel.view(), status, False, False,
            1.0, 0.0, route, lease, -1, 0, -1)
        var result = driver.gpr_query(request)
        require(result.finite == 0 and result.range_status == 0, "original score query status")
        var program = gpr_score_prepare(values, Int(target.unsafe_ptr()))
        try:
            var sums = program.run(metrics, ranges)
            var score = gp_r2_value(sums)
            require(Bool(isfinite(score)) and abs(score - reference[0]) <= 0.001,
                "original 60x4 GP score NumPy oracle at original 1e-3 tolerance")
        finally:
            _ = program
        require(factor[0] == GP_GUARD and factor[rows * rows + 1] == GP_GUARD
            and dual[0] == GP_GUARD and dual[rows + 1] == GP_GUARD
            and scalars[0] == GP_GUARD64 and scalars[6] == GP_GUARD64,
            "fit output guards around original score workload")
    finally:
        if lease > 0:
            driver.lease_close(lease)
        _ = source
        _ = target
        _ = reference
        _ = kernel
        _ = factor
        _ = dual
        _ = scalars
        _ = values


def fixture(driver: GPDriver, metrics: NativeDescriptorCall, unit: Int) raises:
    var ranges = unit % 2 != 0
    if unit < 8:
        analytic_case(metrics, unit // 2, ranges)
    else:
        original_case(driver, metrics, ranges)
