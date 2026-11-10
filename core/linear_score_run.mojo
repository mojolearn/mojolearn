# SPDX-License-Identifier: Apache-2.0
"""Native R² program ownership and prediction directly into its arena."""
from std.memory import memcpy, bitcast
from std.os import getenv
from core.hostptr import f32_ptr
from core.linear_boundary_run import linear_predict_run
from core.linear_model_state import LinearModelState, linear_state_predict_request
from core.linear_score_program import LinearR2Layout
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from x_metrics.descriptor_schema import MP_MODE, MP_VENDOR, MP_CAPABILITIES, MP_HAS_RANGES, MP_RUN_OUT, MP_RUN_RANGES


def linear_r2_check_target(rows: Int, target_rows: Int) raises:
    if target_rows != rows:
        raise Error("mojolearn score: y_true and y_pred have different shapes (" + String(target_rows)
            + ", 1) and (" + String(rows) + ", 1)")


def linear_r2_finish(entry: NativeDescriptorCall, layout: LinearR2Layout, address: Int,
    rows: Int, target: Int, vendor: Int, mode: Int) raises -> Float64:
    if native_descriptor_call(entry, MP_VENDOR, [], [], []) != vendor:
        raise Error("RuntimeError: score vendor differs from selected execution")
    if native_descriptor_call(entry, MP_MODE, [], [], []) != mode:
        raise Error("RuntimeError: score mode differs from process default")
    memcpy(dest=f32_ptr(address) + layout.y, src=f32_ptr(target), count=rows)
    var inputs: List[Int32] = [0, Int32(2 * rows + 1), -1]
    var numerator = layout.numerator
    var denominator = layout.denominator
    var outputs: List[Int32] = [Int32(numerator), Int32(numerator + 2), -1, 1,
        Int32(denominator), Int32(denominator + 2), -1, 1]
    var stages = len(layout.program.words) // 16
    var words_address = Int(layout.program.words.unsafe_ptr())
    var ranges = String(getenv("MOJOLEARN_ARENA_RANGES", "1").strip()) != "0"
    try:
        var capabilities = native_descriptor_call(entry, MP_CAPABILITIES, [], [], [])
        if ranges and capabilities & MP_HAS_RANGES:
            _ = native_descriptor_call(entry, MP_RUN_RANGES,
                [address, words_address, Int(inputs.unsafe_ptr()), Int(outputs.unsafe_ptr())],
                [layout.program.size, stages, 1, 2], [])
        else:
            _ = native_descriptor_call(entry, MP_RUN_OUT,
                [address, words_address, Int(outputs.unsafe_ptr())], [layout.program.size, stages, 2], [])
        var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)
        var numerator_bits = UInt64(words[numerator]) | (UInt64(words[numerator + 1]) << UInt64(32))
        var denominator_bits = UInt64(words[denominator]) | (UInt64(words[denominator + 1]) << UInt64(32))
        var num = bitcast[DType.float64](numerator_bits)
        var den = bitcast[DType.float64](denominator_bits)
        return Float64(0.0) if den == 0 else Float64(1.0) - num / den
    finally:
        _ = inputs
        _ = outputs


def linear_score_run(predict: NativeDescriptorCall, metrics: NativeDescriptorCall,
    state: LinearModelState, x: Int, rows: Int, cols: Int, target: Int,
    target_rows: Int, vendor: Int, mode: Int, estimator: String) raises -> Float64:
    linear_r2_check_target(rows, target_rows)
    var layout = LinearR2Layout(rows)
    var arena = List[Float32](length=layout.program.size, fill=0.0)
    var address = Int(arena.unsafe_ptr())
    try:
        var request = linear_state_predict_request(state, x, address + 4 * layout.p, rows, cols, False, estimator)
        linear_predict_run(predict, request)
        return linear_r2_finish(metrics, layout, address, rows, target, vendor, mode)
    finally:
        _ = layout
        _ = arena


def linear_r2_run(metrics: NativeDescriptorCall, prediction: Int, rows: Int,
    target: Int, target_rows: Int, vendor: Int, mode: Int) raises -> Float64:
    linear_r2_check_target(rows, target_rows)
    var layout = LinearR2Layout(rows)
    var arena = List[Float32](length=layout.program.size, fill=0.0)
    var address = Int(arena.unsafe_ptr())
    try:
        memcpy(dest=f32_ptr(address) + layout.p, src=f32_ptr(prediction), count=rows)
        return linear_r2_finish(metrics, layout, address, rows, target, vendor, mode)
    finally:
        _ = layout
        _ = arena
