# SPDX-License-Identifier: Apache-2.0
"""Native GP score storage and the existing R² program's distinct GP policy."""
from std.memory import memcpy
from core.hostptr import f32_ptr
from core.admitted_convert import admitted_convert
from core.cast_in import CAST_KIND_FLOAT
from core.host_finite_scan import host_all_finite_f32
from core.linear_score_program import LinearR2Layout
from core.linear_score_run import linear_r2_execute, MetricR2Sums
from core.native_descriptor_abi import NativeDescriptorCall


struct GPRScoreValues(Movable):
    var rows: Int
    var prediction: List[Float32]
    var scalar: List[Float32]

    def __init__(out self, rows: Int):
        self.rows = rows
        self.prediction = List[Float32](length=rows, fill=0.0)
        self.scalar = List[Float32](length=1, fill=0.0)

    def prediction_address(self) -> Int:
        return Int(self.prediction.unsafe_ptr())

    def scalar_address(self) -> Int:
        return Int(self.scalar.unsafe_ptr())


struct GPRScoreProgram(Movable):
    var layout: LinearR2Layout
    var arena: List[Float32]
    var inputs: List[Int32]

    def __init__(out self, rows: Int) raises:
        self.layout = LinearR2Layout(rows)
        self.arena = List[Float32](length=self.layout.program.size, fill=0.0)
        self.inputs = List[Int32]()

    def input(mut self, offset: Int, count: Int, source: Int, slot: Int) raises:
        if count == 0:
            return
        if slot < 0:
            memcpy(dest=self.arena.unsafe_ptr() + offset, src=f32_ptr(source), count=count)
        # Same adjacent host-range coalescing as the arena boundary runner.
        var length = len(self.inputs)
        if slot < 0 and length >= 3 and self.inputs[length - 1] < 0 and Int(self.inputs[length - 2]) == offset:
            self.inputs[length - 2] = Int32(offset + count)
        else:
            self.inputs.append(Int32(offset))
            self.inputs.append(Int32(offset + count))
            self.inputs.append(Int32(slot))

    def run(self, entry: NativeDescriptorCall, ranges: Bool) raises -> MetricR2Sums:
        return linear_r2_execute(entry, self.layout, Int(self.arena.unsafe_ptr()), self.inputs, ranges)


def gp_r2_value(sums: MetricR2Sums) -> Float64:
    # GP's existing constant-target policy differs from the linear score's
    # historical denominator-zero policy; keep each explicit at its caller.
    if sums.denominator == 0.0:
        return Float64(1.0) if sums.numerator == 0.0 else Float64(0.0)
    return Float64(1.0) - sums.numerator / sums.denominator


def _gpr_score_snapshot(source: Int, rows: Int, name: String) raises -> List[Float32]:
    # Same forced float32 metric-target snapshot and parallel same-type copy
    # used by metric_reg_target/admitted_typed_c, followed by its finite scan.
    var values = List[Float32](length=rows, fill=0.0)
    var shape: List[Int] = [rows]
    var strides: List[Int] = [4]
    admitted_convert(source, shape, strides, rows, 4, CAST_KIND_FLOAT,
        4, CAST_KIND_FLOAT, "<f4", Int(values.unsafe_ptr()))
    if not host_all_finite_f32(f32_ptr(Int(values.unsafe_ptr())), rows):
        raise Error("mojolearn score: " + name + " contains NaN or infinity")
    return values^


def gpr_score_prepare(values: GPRScoreValues, target: Int) raises -> GPRScoreProgram:
    # Original _Reg target order, including each forced input snapshot, then
    # program input copies. Private native ownership replaces Array wrappers.
    var truth = _gpr_score_snapshot(target, values.rows, "y_true")
    var prediction = _gpr_score_snapshot(values.prediction_address(), values.rows, "y_pred")
    var program = GPRScoreProgram(values.rows)
    var y_offset = program.layout.y
    var p_offset = program.layout.p
    program.input(y_offset, values.rows, Int(truth.unsafe_ptr()), -1)
    program.input(p_offset, values.rows, Int(prediction.unsafe_ptr()), -1)
    program.input(2 * values.rows, 1, values.scalar_address(), -1)
    _ = truth
    _ = prediction
    _ = values
    return program^
