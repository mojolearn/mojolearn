# SPDX-License-Identifier: Apache-2.0
"""Native R² program ownership and prediction directly into its arena."""
from std.memory import unsafe_memcpy, bitcast
from std.os import getenv
from core.hostptr import f32_ptr
from core.entry_table import EntryRef, entry_ref_vendor, entry_ref_mode
from core.host_finite_scan import host_all_finite_f32
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


@fieldwise_init
struct MetricR2Sums(ImplicitlyCopyable, Movable):
    var numerator: Float64
    var denominator: Float64


def linear_r2_execute(entry: NativeDescriptorCall, layout: LinearR2Layout, address: Int,
    inputs: List[Int32], ranges: Bool) raises -> MetricR2Sums:
    """Execute the existing program with admitted native input-range records.

    Input preparation is separate so callers that already own resident inputs
    retain their exact store handles and copy route. Output ranges, stages and
    numerical kernels are shared with the existing linear score path.
    """
    var numerator = layout.numerator
    var denominator = layout.denominator
    var outputs: List[Int32] = [Int32(numerator), Int32(numerator + 2), -1, 1,
        Int32(denominator), Int32(denominator + 2), -1, 1]
    var stages = len(layout.program.words) // 16
    var words_address = Int(layout.program.words.unsafe_ptr())
    try:
        var capabilities = native_descriptor_call(entry, MP_CAPABILITIES, [], [], [])
        if ranges and capabilities & MP_HAS_RANGES:
            _ = native_descriptor_call(entry, MP_RUN_RANGES,
                [address, words_address, Int(inputs.unsafe_ptr()), Int(outputs.unsafe_ptr())],
                [layout.program.size, stages, len(inputs) // 3, 2], [])
        else:
            _ = native_descriptor_call(entry, MP_RUN_OUT,
                [address, words_address, Int(outputs.unsafe_ptr())], [layout.program.size, stages, 2], [])
        var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)
        var numerator_bits = UInt64(words[numerator]) | (UInt64(words[numerator + 1]) << UInt64(32))
        var denominator_bits = UInt64(words[denominator]) | (UInt64(words[denominator + 1]) << UInt64(32))
        var num = bitcast[DType.float64](numerator_bits)
        var den = bitcast[DType.float64](denominator_bits)
        return MetricR2Sums(num, den)
    finally:
        _ = inputs
        _ = outputs



def linear_r2_sums(entry: NativeDescriptorCall, layout: LinearR2Layout, address: Int,
    rows: Int, target: Int, vendor: Int, mode: Int) raises -> MetricR2Sums:
    if native_descriptor_call(entry, MP_VENDOR, [], [], []) != vendor:
        raise Error("RuntimeError: score vendor differs from selected execution")
    if native_descriptor_call(entry, MP_MODE, [], [], []) != mode:
        raise Error("RuntimeError: score mode differs from process default")
    unsafe_memcpy(dest=f32_ptr(address) + layout.y, src=f32_ptr(target), count=rows)
    var inputs: List[Int32] = [0, Int32(2 * rows + 1), -1]
    var ranges = String(getenv("MOJOLEARN_ARENA_RANGES", "1").strip()) != "0"
    return linear_r2_execute(entry, layout, address, inputs, ranges)


def linear_r2_finish(entry: NativeDescriptorCall, layout: LinearR2Layout, address: Int,
    rows: Int, target: Int, vendor: Int, mode: Int) raises -> Float64:
    var sums = linear_r2_sums(entry, layout, address, rows, target, vendor, mode)
    return Float64(0.0) if sums.denominator == 0 else Float64(1.0) - sums.numerator / sums.denominator


def linear_score_run(predict: NativeDescriptorCall, metrics: NativeDescriptorCall,
    state: LinearModelState, x: Int, rows: Int, cols: Int, target: Int,
    target_rows: Int, vendor: Int, mode: Int, estimator: String) raises -> Float64:
    linear_r2_check_target(rows, target_rows)
    var layout = LinearR2Layout(rows)
    var arena = List[Float32](length=layout.program.size, fill=0.0)
    var address = Int(arena.unsafe_ptr())
    try:
        var request = linear_state_predict_request(state, x, address + 4 * layout.p, rows, cols, estimator)
        _ = linear_predict_run(predict, request)
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
        unsafe_memcpy(dest=f32_ptr(address) + layout.p, src=f32_ptr(prediction), count=rows)
        return linear_r2_finish(metrics, layout, address, rows, target, vendor, mode)
    finally:
        _ = layout
        _ = arena


def member_r2_score(metrics: EntryRef, y_true: Int, y_pred: Int, n: Int) raises -> Float64:
    """A regressor member's `score`: the R^2 its estimator's own `score`
    returns (`1 - SS_res / SS_tot` of the `_Reg` program, binary64 sums,
    0.0 for a constant y; `classical_r2_score` and `svc_r2_sums` run the same
    program), over n host float32 targets and predictions. `metrics` is the
    x_metrics program entry on the member's route (`fifth`;
    providers/python/metrics_route_boundary_door.mojo
    `member_metrics_program_fifth`). The members of LinearSVR, QNRegressor,
    Lasso and ElasticNet called the metrics image's `r2_score` entry (MR_R2),
    a float32 result, so a cross_val_score of host folds (member route)
    differed from the same folds as DeviceRows (the estimator's own score)
    in the 8th digit (verify smoke of e282d53b19,
    test_device_rows_door_cv.py; lane sr3-smoke, 2026-11-22)."""
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=y_true), n):
        raise Error("mojolearn score: y_true contains NaN or infinity")
    if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=y_pred), n):
        raise Error("mojolearn score: y_pred contains NaN or infinity")
    return linear_r2_run(metrics.call, y_pred, n, y_true, n, entry_ref_vendor(metrics), entry_ref_mode(metrics))
