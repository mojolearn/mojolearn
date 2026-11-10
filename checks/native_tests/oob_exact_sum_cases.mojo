# SPDX-License-Identifier: Apache-2.0
"""Five OOB targets x covered/uncovered, with independent closed-form words.

This replaces Python math.fsum tests with exact expected binary64 encodings.
In the cancellation fixture [2^60, 1, -2^60], the exact sum is one and the
rounded mean is 1/3; separately rounded squared deviations sum to 2^121.
No product exact-sum or soft-float helper supplies the expectations.
"""
from std.memory import bitcast
from core.hostptr import f32_ptr, f64_ptr, i32_ptr
from xtrees.oob import oob_r2_host, oob_r2_device

comptime OOB_EXACT_FIXTURES = 10


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def run_oob_fixture[device: Bool](fixture_id: Int) raises:
    require(fixture_id >= 0 and fixture_id < OOB_EXACT_FIXTURES, "unknown OOB fixture")
    var target_id = fixture_id // 2
    var covered = fixture_id % 2 != 0
    var y: List[Float32] = [1.0]
    if target_id == 1:
        y = [-1.0]
    elif target_id == 2:
        var big = bitcast[DType.float32](UInt32(0x5D800000))  # 2^60
        y = [big, 1.0, -big]
    elif target_id == 3:
        y = [2.0, -2.0, 0.0]
    elif target_id == 4:
        y = List[Float32](length=65, fill=1.0)
    var n = len(y)
    var counts = List[Int32](length=n, fill=Int32(2 if covered else 0))
    var acc = List[Float64](length=n, fill=0)
    for i in range(n):
        if covered:
            acc[i] = Float64(y[i]) * 2.0
    var original_y = y.copy()
    var original_acc = acc.copy()
    var original_counts = counts.copy()
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000000))
    var pred = List[Float64](length=n, fill=nan)
    var words = List[Float64](length=4, fill=nan)
    var flags = List[Int32](length=4, fill=-1)
    comptime if device:
        oob_r2_device(f64_ptr(Int(acc.unsafe_ptr())), i32_ptr(Int(counts.unsafe_ptr())), f32_ptr(Int(y.unsafe_ptr())), n,
                      f64_ptr(Int(pred.unsafe_ptr())), f64_ptr(Int(words.unsafe_ptr())), i32_ptr(Int(flags.unsafe_ptr())))
    else:
        oob_r2_host(f64_ptr(Int(acc.unsafe_ptr())), i32_ptr(Int(counts.unsafe_ptr())), f32_ptr(Int(y.unsafe_ptr())), n,
                    f64_ptr(Int(pred.unsafe_ptr())), f64_ptr(Int(words.unsafe_ptr())), i32_ptr(Int(flags.unsafe_ptr())))
    var total = Float64(1)
    var centered = Float64(0)
    var residual = Float64(1)
    var mean = Float64(1)
    if target_id == 1:
        total = -1.0
        mean = -1.0
    elif target_id == 2:
        centered = bitcast[DType.float64](UInt64(0x4780000000000000))  # 2^121
        residual = centered
        mean = bitcast[DType.float64](UInt64(0x3FD5555555555555))  # correctly rounded 1/3
    elif target_id == 3:
        total = 0.0
        centered = 8.0
        residual = 8.0
        mean = 0.0
    elif target_id == 4:
        total = 65.0
        residual = 65.0
    if covered:
        residual = 0.0
    var expected: List[Float64] = [total, centered, residual, mean]
    for i in range(4):
        require(flags[i] == 0, "OOB finite fixture status flag " + String(i))
        require(bitcast[DType.uint64](words[i]) == bitcast[DType.uint64](expected[i]), "OOB sum/centered/residual/mean word " + String(i))
    for i in range(n):
        var expected_pred = Float64(original_y[i]) if covered else Float64(0)
        require(bitcast[DType.uint64](pred[i]) == bitcast[DType.uint64](expected_pred), "OOB covered/zero-count prediction")
        require(bitcast[DType.uint64](acc[i]) == bitcast[DType.uint64](original_acc[i]), "OOB accumulator input mutated")
        require(bitcast[DType.uint32](y[i]) == bitcast[DType.uint32](original_y[i]), "OOB target input mutated")
        require(counts[i] == original_counts[i], "OOB count input mutated")
