# SPDX-License-Identifier: Apache-2.0
"""The RMS-norm and linear training primitives against exact and Float64
references (ported from the `arm_rms_norm` and `arm_linear` arms of
`python/mojolearn/tests/test_training_primitives_surface.py`, lane
tmig-training, 2026-11-14).

The suites call the primitives the training bindings export: on the host
column `core/mojolearn_training_host_impl.mojo` (`rms_norm_*_host_into`,
`linear_*_host_into`), on a device `training/native_primitive_dispatch.mojo`
`native_primitive_operation` (operations 2..5). Both take the same address
layout: rms forward [y, x, w], rms backward [dx, dw, dy, x, w], linear
forward [c, a, w] and linear backward [da, dw, dc, a, w]. This module holds
the inputs, the references and the assertions; no kernel call.

Inputs are small integers as float32 (the Python test's `_ints`): every
linear product and sum is exact in float32, so a wrong slot or count shows
as a wrong number. Tolerances are the Python test's.
"""
from std.memory import bitcast
from std.math import sqrt
from checks.native_tests.training_check_common import same_words, all_finite, uniform_int
from core.host_parallel import host_parallelize
from core.mojolearn_training_host_impl import (
    rms_norm_forward_host_into,
    rms_norm_backward_host_into,
    linear_forward_host_into,
    linear_backward_host_into,
    accumulate_host_into,
)
from training.native_primitive_dispatch import native_primitive_operation
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime RMS_M = 5
comptime RMS_D = 8
comptime LIN_M = 7
comptime LIN_K = 5
comptime LIN_N = 3


def small_ints(seed: Int, count: Int, lo: Int, hi: Int) -> List[Float32]:
    """Integers in [lo, hi) as float32."""
    var values = List[Float32](length=count, fill=0)
    for i in range(count):  # small-loop(fixture fill): at most 40 values
        values[i] = Float32(lo + uniform_int(seed, i, hi - lo))
    return values^


def rms_x() -> List[Float32]:
    return small_ints(3, RMS_M * RMS_D, 1, 4)


def rms_w() -> List[Float32]:
    return List[Float32](length=RMS_D, fill=2.0)


def rms_dy() -> List[Float32]:
    return small_ints(4, RMS_M * RMS_D, -4, 5)


def _inverse_rms(x: List[Float32], row: Int) -> Float64:
    var acc = Float64(0)
    for d in range(RMS_D):  # small-loop(oracle row): eight values
        acc += Float64(x[row * RMS_D + d]) * Float64(x[row * RMS_D + d])
    return 1.0 / sqrt(acc / Float64(RMS_D))


def check_rms(x: List[Float32], w: List[Float32], dy: List[Float32], y: List[Float32],
        dx: List[Float32], dw: List[Float32]) raises:
    """Forward within 2e-6 of max |ref|; backward shapes and finiteness;
    dweight = sum_rows dy * x_hat within 1e-4 * max(1, max |ref|)."""
    require(len(y) == RMS_M * RMS_D, "rms_norm_forward is not (M, D)")
    var worst = Float64(0)
    var scale = Float64(0)
    for r in range(RMS_M):  # small-loop(oracle rows): five rows
        var inv = _inverse_rms(x, r)
        for d in range(RMS_D):
            var want = Float64(Float32(Float64(w[d]) * Float64(x[r * RMS_D + d]) * inv))
            worst = max(worst, abs(Float64(y[r * RMS_D + d]) - want))
            scale = max(scale, abs(want))
    require(worst <= 2e-6 * scale, "rms_norm_forward is not the host formula: max diff " + String(worst))
    require(len(dx) == RMS_M * RMS_D and len(dw) == RMS_D, "rms_norm_backward shapes are not (M, D) and (D,)")
    require(all_finite(dx) and all_finite(dw), "rms_norm_backward is not finite")
    var dw_worst = Float64(0)
    var dw_scale = Float64(1)
    for d in range(RMS_D):  # small-loop(oracle columns): eight sums
        var want = Float64(0)
        for r in range(RMS_M):
            want += Float64(dy[r * RMS_D + d]) * Float64(x[r * RMS_D + d]) * _inverse_rms(x, r)
        dw_worst = max(dw_worst, abs(Float64(dw[d]) - want))
        dw_scale = max(dw_scale, abs(want))
    require(dw_worst <= 1e-4 * dw_scale, "dweight is not sum_rows dy * x_hat: max diff " + String(dw_worst))


def lin_a() -> List[Float32]:
    return small_ints(5, LIN_M * LIN_K, -4, 5)


def lin_w() -> List[Float32]:
    return small_ints(6, LIN_N * LIN_K, -4, 5)


def lin_dc() -> List[Float32]:
    return small_ints(7, LIN_M * LIN_N, -4, 5)


def _exact(rows: Int, cols: Int, inner: Int, left: List[Float32], left_row: Int, left_inner: Int,
        right: List[Float32], right_col: Int, right_inner: Int) -> List[Float32]:
    """out[i, j] = sum_t left[i*left_row + t*left_inner] * right[j*right_col + t*right_inner],
    in Float64 (exact for these integers), rounded once."""
    var result = List[Float32](length=rows * cols, fill=0)
    for i in range(rows):  # small-loop(oracle cells): at most 35 cells
        for j in range(cols):
            var acc = Float64(0)
            for t in range(inner):
                acc += Float64(left[i * left_row + t * left_inner]) * Float64(right[j * right_col + t * right_inner])
            result[i * cols + j] = Float32(acc)
    return result^


def check_linear(a: List[Float32], w: List[Float32], dc: List[Float32], c: List[Float32],
        da: List[Float32], dw: List[Float32], c_first_row: List[Float32]) raises:
    """c = a . w^T, da = dc . w, dweight = dc^T . a, all exact; one row alone
    equals that row of the batch, word for word."""
    require(len(c) == LIN_M * LIN_N, "linear_forward is not (M, N)")
    require(same_words(c, _exact(LIN_M, LIN_N, LIN_K, a, LIN_K, 1, w, LIN_K, 1)), "linear_forward is not a . w^T, exact")
    require(same_words(da, _exact(LIN_M, LIN_K, LIN_N, dc, LIN_N, 1, w, 1, LIN_K)), "da is not dc . w, exact")
    require(same_words(dw, _exact(LIN_N, LIN_K, LIN_M, dc, 1, LIN_N, a, 1, LIN_K)),
            "dweight is not dc^T . a (the contraction over M tokens), exact")
    var first = List[Float32](length=LIN_N, fill=0)
    for j in range(LIN_N):  # small-loop(row copy): three values
        first[j] = c[j]
    require(same_words(c_first_row, first), "one row alone is not that row of the batch, word for word")


# ---------------------------------------------------------------- ordered gradient fold
# Ported from `test_ordered_gradient_fold_has_fixed_left_association_and_owns_inputs`
# (test_parallel_mlp_native.py): `ordered_sum_gradients` stacks the shards
# [S, count] (shard k in slot k) and folds them in ascending order IN PLACE
# with the accumulate primitive (addresses [slot k, slot k-1], ints
# [count, 2, -1]), bindings/parallel_mlp_native_door.mojo `neural_ordered_sum`.

comptime FOLD_SHARDS = 3
comptime FOLD_COUNT = 8  # a (2,) tensor then a (2, 3) tensor


def fold_stack() -> List[Float32]:
    """Shard 0: [1, -1] and arange(6); shards 1 and 2: [2^-24, -2^-24] with
    ones and twos."""
    var tiny = Float32(5.9604644775390625e-08)  # 2^-24
    var stack = List[Float32](length=FOLD_SHARDS * FOLD_COUNT, fill=0)
    stack[0] = 1.0
    stack[1] = -1.0
    for i in range(6):  # small-loop(fixture fill)
        stack[2 + i] = Float32(i)
    for shard in range(1, FOLD_SHARDS):  # small-loop(fixture fill)
        stack[shard * FOLD_COUNT] = tiny
        stack[shard * FOLD_COUNT + 1] = -tiny
        for i in range(6):
            stack[shard * FOLD_COUNT + 2 + i] = Float32(shard)
    return stack^


def check_ordered_fold(stack: List[Float32]) raises:
    """The last slot holds the left-associated sum: both separate additions of
    the tiny terms tie to even (a tree combining them first moves one ULP)."""
    var base = (FOLD_SHARDS - 1) * FOLD_COUNT
    require(bitcast[DType.uint32](stack[base]) == UInt32(0x3F800000) and bitcast[DType.uint32](stack[base + 1]) == UInt32(0xBF800000),
            "the ordered fold is not left-associated: " + String(stack[base]) + ", " + String(stack[base + 1]))
    for i in range(6):  # small-loop(check comparison)
        require(stack[base + 2 + i] == Float32(i) + 3, "the ordered fold of the second tensor is not first + 3")


# ---- The host column (was training_primitives_cpu_suite.mojo) ----
# RMS-norm and linear training primitives on the host column
# (training_primitives_cases.mojo); the two independent fixtures run across
# host cores.
comptime PRIMITIVE_FIXTURES = 3


def _rms() raises:
    var x = rms_x()
    var w = rms_w()
    var dy = rms_dy()
    var y = List[Float32](length=RMS_M * RMS_D, fill=0)
    var dx = List[Float32](length=RMS_M * RMS_D, fill=0)
    var dw = List[Float32](length=RMS_D, fill=0)
    _ = rms_norm_forward_host_into([Int(y.unsafe_ptr()), Int(x.unsafe_ptr()), Int(w.unsafe_ptr())], RMS_M, RMS_D, Float32(0))
    _ = rms_norm_backward_host_into([Int(dx.unsafe_ptr()), Int(dw.unsafe_ptr()), Int(dy.unsafe_ptr()),
                                     Int(x.unsafe_ptr()), Int(w.unsafe_ptr())], RMS_M, RMS_D, Float32(0))
    check_rms(x, w, dy, y, dx, dw)


def _linear() raises:
    var a = lin_a()
    var w = lin_w()
    var dc = lin_dc()
    var c = List[Float32](length=LIN_M * LIN_N, fill=0)
    var c1 = List[Float32](length=LIN_N, fill=0)
    var da = List[Float32](length=LIN_M * LIN_K, fill=0)
    var dw = List[Float32](length=LIN_N * LIN_K, fill=0)
    _ = linear_forward_host_into([Int(c.unsafe_ptr()), Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], LIN_M, LIN_N, LIN_K)
    _ = linear_forward_host_into([Int(c1.unsafe_ptr()), Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], 1, LIN_N, LIN_K)
    _ = linear_backward_host_into([Int(da.unsafe_ptr()), Int(dw.unsafe_ptr()), Int(dc.unsafe_ptr()),
                                   Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], LIN_M, LIN_N, LIN_K)
    check_linear(a, w, dc, c, da, dw, c1)


def _fold() raises:
    var stack = fold_stack()
    var base = Int(stack.unsafe_ptr())
    for shard in range(1, FOLD_SHARDS):  # dependent ordered fold: ascending shard order
        var dst = base + 4 * shard * FOLD_COUNT
        var src = base + 4 * (shard - 1) * FOLD_COUNT
        _ = accumulate_host_into([dst, src], FOLD_COUNT, 2, -1)
    check_ordered_fold(stack)


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(PRIMITIVE_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            if unit == 0:
                _rms()
            elif unit == 1:
                _linear()
            else:
                _fold()
        except error:
            cases.fail(unit, error)
    report.parallel(task, PRIMITIVE_FIXTURES)


# ---- The device column (was training_primitives_gpu_suite.mojo) ----
# RMS-norm and linear training primitives on the device
# (training_primitives_cases.mojo), through `native_primitive_operation`
# (operations 2..5). The two fixtures run in order: every call shares the one
# process training context (`neural_ctx`); the kernels keep their own
# parallelism.
comptime PRIMITIVE_FIXTURES_gpu = 3


def _rms_gpu() raises:
    var x = rms_x()
    var w = rms_w()
    var dy = rms_dy()
    var y = List[Float32](length=RMS_M * RMS_D, fill=0)
    var dx = List[Float32](length=RMS_M * RMS_D, fill=0)
    var dw = List[Float32](length=RMS_D, fill=0)
    var eps: List[Float64] = [0.0]
    _ = native_primitive_operation(2, [Int(y.unsafe_ptr()), Int(x.unsafe_ptr()), Int(w.unsafe_ptr())], [RMS_M, RMS_D], eps)
    _ = native_primitive_operation(3, [Int(dx.unsafe_ptr()), Int(dw.unsafe_ptr()), Int(dy.unsafe_ptr()),
                                       Int(x.unsafe_ptr()), Int(w.unsafe_ptr())], [RMS_M, RMS_D], eps)
    check_rms(x, w, dy, y, dx, dw)


def _linear_gpu() raises:
    var a = lin_a()
    var w = lin_w()
    var dc = lin_dc()
    var c = List[Float32](length=LIN_M * LIN_N, fill=0)
    var c1 = List[Float32](length=LIN_N, fill=0)
    var da = List[Float32](length=LIN_M * LIN_K, fill=0)
    var dw = List[Float32](length=LIN_N * LIN_K, fill=0)
    var none = List[Float64]()
    _ = native_primitive_operation(4, [Int(c.unsafe_ptr()), Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], [LIN_M, LIN_N, LIN_K], none)
    _ = native_primitive_operation(4, [Int(c1.unsafe_ptr()), Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], [1, LIN_N, LIN_K], none)
    _ = native_primitive_operation(5, [Int(da.unsafe_ptr()), Int(dw.unsafe_ptr()), Int(dc.unsafe_ptr()),
                                       Int(a.unsafe_ptr()), Int(w.unsafe_ptr())], [LIN_M, LIN_N, LIN_K], none)
    check_linear(a, w, dc, c, da, dw, c1)


def _fold_gpu() raises:
    var stack = fold_stack()
    var base = Int(stack.unsafe_ptr())
    for shard in range(1, FOLD_SHARDS):  # dependent ordered fold: ascending shard order
        var dst = base + 4 * shard * FOLD_COUNT
        var src = base + 4 * (shard - 1) * FOLD_COUNT
        _ = native_primitive_operation(9, [dst, src], [FOLD_COUNT, 2, -1], List[Float64]())
    check_ordered_fold(stack)


def _run_gpu(mut report: Report) raises:
    var cases = report.cases(PRIMITIVE_FIXTURES_gpu)
    for unit in range(PRIMITIVE_FIXTURES_gpu):  # ordered: one shared device context
        try:
            if unit == 0:
                _rms_gpu()
            elif unit == 1:
                _linear_gpu()
            else:
                _fold_gpu()
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("training_primitives", cpu=True, gpu=True, serial="", run=run_cases)
