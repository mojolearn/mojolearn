# SPDX-License-Identifier: Apache-2.0
"""Independent factored Adafactor equations for the former NumPy fixture.

The 5x4 matrix and length-three vector retain six steps, lr 0.02, decay 0.05,
PyTorch-rule relative step/clipping, and original rtol1e-4/atol1e-5. Native
fixed dyadic data replaces RNG setup; expected arithmetic is Float64 and
never calls a product numerical helper. Matrix row/column and update-cell
units use the host pool; whole-tensor sums and optimizer steps are ordered.
"""
from std.math import pow, sqrt
from core.host_parallel import host_parallelize
from sequence.exec_trait import Exec
from sequence.ops_impl import adafactor_step_impl

comptime ADAFACTOR_FIXTURES = 2
comptime AF_STEPS = 6
comptime AF_EPS1 = Float64(0.00000011920928955078125)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def af_shape(fixture_id: Int) -> Tuple[Int, Int]:
    return (5, 4) if fixture_id == 0 else (3, 0)


def af_config() -> List[Float64]:
    return [0.02, -0.8, AF_EPS1, 0.001, 1, 0.05]


def af_data(fixture_id: Int) -> Tuple[List[Float32], List[Float32]]:
    var shape = af_shape(fixture_id)
    var n = shape[0] * shape[1] if shape[1] else shape[0]
    var parameters = List[Float32](length=n, fill=0)
    var gradients = List[Float32](length=AF_STEPS * n, fill=0)
    var pp = parameters.unsafe_ptr()
    var gp = gradients.unsafe_ptr()
    def cell(index: Int) {imm pp, imm gp, imm n}:
        pp.unsafe_store(index, Float32((index * 11) % 23 - 11) * 0.0625)
        for epoch in range(AF_STEPS):
            gp.unsafe_store(epoch * n + index, Float32(((epoch + 1) * 19 + index * 7) % 29 - 14) * 0.03125)
    host_parallelize(cell, n)
    return (parameters^, gradients^)


def af_expected(fixture_id: Int, data: Tuple[List[Float32], List[Float32]]) -> Tuple[List[Float64], List[Float64], List[Float64]]:
    var shape = af_shape(fixture_id)
    var rows = shape[0]
    var cols = shape[1]
    var n = rows * cols if cols else rows
    var parameters = List[Float64](length=n, fill=0)
    var first = List[Float64](length=rows, fill=0)
    var second = List[Float64](length=max(cols, 1), fill=0)
    var update = List[Float64](length=n, fill=0)
    var pp = parameters.unsafe_ptr()
    var firstp = first.unsafe_ptr()
    var secondp = second.unsafe_ptr()
    var up = update.unsafe_ptr()
    var original = data[0].unsafe_ptr()
    var gp = data[1].unsafe_ptr()
    def initialize(index: Int) {imm pp, imm original}:
        pp.unsafe_store(index, Float64(original.unsafe_load(index)))
    host_parallelize(initialize, n)
    for step in range(1, AF_STEPS + 1):  # dependent second-moment/parameter recurrence
        var weight = pow(Float64(step), -0.8)
        var rho = min(Float64(0.02), pow(Float64(step), -0.5))
        var offset = (step - 1) * n
        var parameter_norm_sq = Float64(0)
        for index in range(n):  # one small whole-tensor RMS reduction
            parameter_norm_sq += parameters[index] * parameters[index]
        var alpha = max(Float64(0.001), sqrt(parameter_norm_sq / Float64(n))) * rho
        if cols:
            def row_task(row: Int) {imm firstp, imm gp, imm offset, imm cols, imm weight}:
                var squares = Float64(0)
                for col in range(cols):
                    var g = Float64(gp.unsafe_load(offset + row * cols + col))
                    squares += g * g
                var prior = firstp.unsafe_load(row)
                firstp.unsafe_store(row, prior + weight * (squares / Float64(cols) - prior))
            host_parallelize(row_task, rows)
            def col_task(col: Int) {imm secondp, imm gp, imm offset, imm rows, imm cols, imm weight}:
                var squares = Float64(0)
                for row in range(rows):
                    var g = Float64(gp.unsafe_load(offset + row * cols + col))
                    squares += g * g
                var prior = secondp.unsafe_load(col)
                secondp.unsafe_store(col, prior + weight * (squares / Float64(rows) - prior))
            host_parallelize(col_task, cols)
        else:
            def variance_task(index: Int) {imm firstp, imm gp, imm offset, imm weight}:
                var g = Float64(gp.unsafe_load(offset + index))
                var prior = firstp.unsafe_load(index)
                firstp.unsafe_store(index, prior + weight * (g * g - prior))
            host_parallelize(variance_task, n)
        var row_mean = Float64(0)
        if cols:
            for row in range(rows):
                row_mean += first[row]
            row_mean /= Float64(rows)
        var denominator = max(row_mean, AF_EPS1)
        def update_task(index: Int) {imm firstp, imm secondp, imm gp, imm up, imm offset, imm cols, imm denominator}:
            var estimate = firstp.unsafe_load(index) if cols == 0 else firstp.unsafe_load(index // cols) * secondp.unsafe_load(index % cols) / denominator
            var g = Float64(gp.unsafe_load(offset + index))
            up.unsafe_store(index, g / sqrt(max(estimate, AF_EPS1 * AF_EPS1)))
        host_parallelize(update_task, n)
        var update_norm_sq = Float64(0)
        for index in range(n):  # one small whole-tensor clipping RMS reduction
            update_norm_sq += update[index] * update[index]
        var coefficient = alpha / max(Float64(1), sqrt(update_norm_sq / Float64(n)))
        def apply(index: Int) {imm pp, imm up, imm coefficient}:
            pp.unsafe_store(index, pp.unsafe_load(index) * (1.0 - 0.02 * 0.05) - coefficient * up.unsafe_load(index))
        host_parallelize(apply, n)
    _ = data
    _ = update
    return (parameters^, first^, second^)


def af_check(actual: List[Float32], wanted: List[Float64], label: String) raises:
    require(len(actual) == len(wanted), label + " extent")
    for index in range(len(actual)):
        require(abs(Float64(actual[index]) - wanted[index]) <= 1e-5 + 1e-4 * abs(wanted[index]),
            "Adafactor independent " + label + " cell " + String(index))


def run_adafactor_fixture[E: Exec](mut ex: E, fixture_id: Int) raises:
    var shape = af_shape(fixture_id)
    var rows = shape[0]
    var cols = shape[1]
    var n = rows * cols if cols else rows
    var data = af_data(fixture_id)
    var expected = af_expected(fixture_id, data)
    var parameters = data[0].copy()
    var first = List[Float32](length=rows, fill=0)
    var second = List[Float32](length=max(cols, 1), fill=0)
    for step in range(1, AF_STEPS + 1):  # dependent optimizer recurrence
        var addresses: List[Int] = [Int(parameters.unsafe_ptr()), Int(data[1].unsafe_ptr()) + 4 * (step - 1) * n,
            Int(first.unsafe_ptr()), Int(second.unsafe_ptr())]
        var ints: List[Int] = [rows, cols, step]
        require(adafactor_step_impl(ex, addresses, ints, af_config()) == n, "Adafactor native step extent")
    ex.sync()
    af_check(parameters, expected[0], "parameters")
    af_check(first, expected[1], "row/variance state")
    af_check(second, expected[2], "column state")
    _ = data
    _ = parameters
    _ = first
    _ = second
