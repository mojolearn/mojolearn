# SPDX-License-Identifier: Apache-2.0
"""Main-lane-only small-k selector bit gate and synchronized component timing.

Author has not compiled/run this candidate. Build/run with IDENTICAL and -I .
Small cases compare legacy radix, candidate, and a host composite-key top-k.
Every selected value/index is compared raw, including signed zeros, infinities,
and NaN payloads (the radix selector accepts arbitrary float bit patterns).
Output CELL records permit exact cross-device comparison of selected pairs.

MOJOLEARN_SMALLK_LARGE=1 adds separately labelled random and duplicate-heavy
100k-index cases for k8/10/16; ROWS defaults32, may be128. COLS defaults100000.
SAMPLES defaults9, minimum7; every sample rotates legacy/candidate order.
Times include COMPLETE selection launch+sync, exclude input/allocations and
exclude distance calculation. There is no end-to-end kNN claim. GPU memory
is bounded by rows*cols <= 16000000; one context/fixture runs at a time.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns
from bench.gemv_serial_layout_main import _env_int, _upload, _read, _same
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.select_radix_identical import composite_key, radix_topk_identical_kernel
from neighbors.checks.select_smallk_identical_candidate import smallk_identical_into
from neighbors.impl.matrix.detail.select_radix import SELECT_BLOCK


def _values(rows: Int, n: Int, profile: Int) -> List[Float32]:
    var result = List[Float32]()
    for row in range(rows):
        for col in range(n):
            var z = UInt64(col + 1) * UInt64(0x9E3779B97F4A7C15) + UInt64(row + 3) * UInt64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
            z = z ^ (z >> 31)
            var value = Float32(Int((z >> 40) & UInt64(0xFFFFFF))) / Float32(262144)
            if profile == 1:
                value = Float32((col + row) % 19) / Float32(32)
            elif profile == 2:
                value = Float32(col) - Float32(n // 2)
            elif profile == 3:
                value = Float32(n // 2) - Float32(col)
            elif profile == 4:
                var slot = (col + row) % 12
                var bits = UInt32(0)
                if slot == 0:
                    bits = UInt32(0)
                elif slot == 1:
                    bits = UInt32(0x80000000)
                elif slot == 2:
                    bits = UInt32(1)
                elif slot == 3:
                    bits = UInt32(0x80000001)
                elif slot == 4:
                    bits = UInt32(0x7F800000)
                elif slot == 5:
                    bits = UInt32(0xFF800000)
                elif slot == 6:
                    bits = UInt32(0x7FC00001)
                elif slot == 7:
                    bits = UInt32(0xFFC00001)
                elif slot == 8:
                    bits = UInt32(0x7F7FFFFF)
                elif slot == 9:
                    bits = UInt32(0xFF7FFFFF)
                elif slot == 10:
                    bits = UInt32(0x00800000)
                else:
                    bits = UInt32(0x7FC00002)
                value = bitcast[DType.float32](bits)
            result.append(value)
    return result^


def _launch(
    ctx: DeviceContext, mut values: DeviceBuffer[DType.float32],
    mut ov: DeviceBuffer[DType.float32], mut oi: DeviceBuffer[DType.uint32],
    mut bv: DeviceBuffer[DType.float32], mut bi: DeviceBuffer[DType.uint32],
    rows: Int, n: Int, k: Int, select_min: Bool, arm: Int,
) raises:
    if arm == 0:
        ctx.enqueue_function[radix_topk_identical_kernel](
            values.unsafe_ptr(), ov.unsafe_ptr(), oi.unsafe_ptr(), bv.unsafe_ptr(), bi.unsafe_ptr(),
            Int32(n), Int32(k), Int32(n), Int32(select_min),
            grid_dim=(rows, 1, 1), block_dim=(SELECT_BLOCK, 1, 1),
        )
    else:
        smallk_identical_into(ctx, values, ov, oi, rows, n, k, select_min)


def _oracle(values: List[Float32], rows: Int, n: Int, k: Int, select_min: Bool) -> List[UInt32]:
    var output = List[UInt32]()
    for row in range(rows):
        var keys = List[UInt64]()
        for slot in range(k):
            keys.append(UInt64(18446744073709551615))
        for col in range(n):
            var pending = composite_key(values[row * n + col], UInt32(col), select_min)
            for slot in range(k):
                if pending < keys[slot]:
                    var previous = keys[slot]
                    keys[slot] = pending
                    pending = previous
        for slot in range(k):
            output.append(UInt32(keys[slot] & UInt64(4294967295)))
    return output^


def _case(rows: Int, n: Int, k: Int, profile: Int, select_min: Bool, timing: Bool, samples: Int) raises:
    var host = _values(rows, n, profile)
    var oracle = List[UInt32]()
    if not timing:
        oracle = _oracle(host, rows, n, k, select_min)
    with DeviceContext() as ctx:
        var values = _upload(ctx, host)
        var ov = ctx.enqueue_create_buffer[DType.float32](rows * k)
        var oi = ctx.enqueue_create_buffer[DType.uint32](rows * k)
        var bv = ctx.enqueue_create_buffer[DType.float32](2 * rows * n)
        var bi = ctx.enqueue_create_buffer[DType.uint32](2 * rows * n)
        var hi = ctx.enqueue_create_host_buffer[DType.uint32](rows * k)
        var baseline_values = List[Float32]()
        var baseline_indices = List[UInt32]()
        for repeat in range(2):
            for arm in range(2):
                ov.enqueue_fill(Float32(-987654))
                oi.enqueue_fill(UInt32(4294967295))
                ctx.synchronize()
                _launch(ctx, values, ov, oi, bv, bi, rows, n, k, select_min, arm)
                ctx.synchronize()
                var got = _read(ctx, ov)
                ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=oi)
                ctx.synchronize()
                for slot in range(rows * k):
                    var index = hi.unsafe_ptr().unsafe_load(slot)
                    if index >= UInt32(n):
                        raise Error("selected index unwritten/out of range")
                    if bitcast[DType.uint32](got[slot]) != bitcast[DType.uint32](host[(slot // k) * n + Int(index)]):
                        raise Error("selected value changed original input bits")
                    if not timing and index != oracle[slot]:
                        raise Error("selected index differs from host key oracle")
                    if repeat == 0 and arm == 0:
                        baseline_indices.append(index)
                    elif index != baseline_indices[slot]:
                        raise Error("candidate/repeat selected index differs from legacy")
                if repeat == 0 and arm == 0:
                    baseline_values = got.copy()
                else:
                    _same(baseline_values, got, "selected distances")
        _same(host, _read(ctx, values), "selection mutated input")
        for slot in range(rows * k):
            print("CELL", rows, n, k, profile, Int(select_min), slot,
                  baseline_indices[slot], bitcast[DType.uint32](baseline_values[slot]))
        print("BITGATE PASS", rows, n, k, "profile", profile, "select_min", Int(select_min))
        if timing:
            print("TIMING complete_selection launch_sync allocations_input_distance_excluded", rows, n, k,
                  "profile", profile, "legacy_scratch_bytes", 16 * rows * n, "candidate_global_scratch_bytes", 0)
            for warm in range(2):
                for arm in range(2):
                    _launch(ctx, values, ov, oi, bv, bi, rows, n, k, select_min, arm)
                    ctx.synchronize()
            for sample in range(samples):
                for position in range(2):
                    var arm = (sample + position) % 2
                    ctx.synchronize()
                    var begin = perf_counter_ns()
                    _launch(ctx, values, ov, oi, bv, bi, rows, n, k, select_min, arm)
                    ctx.synchronize()
                    var elapsed = Float64(perf_counter_ns() - begin) / Float64(1000000)
                    var label = String("legacy_radix")
                    if arm == 1:
                        label = "candidate_local_topk_merge"
                    print("SAMPLE", rows, n, k, profile, sample, label, elapsed)
            # Check both arms again after timing, not merely the last writer.
            for arm in range(2):
                _launch(ctx, values, ov, oi, bv, bi, rows, n, k, select_min, arm)
                ctx.synchronize()
                _same(baseline_values, _read(ctx, ov), "post-timing selected distances")
                ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=oi)
                ctx.synchronize()
                for slot in range(rows * k):
                    if hi.unsafe_ptr().unsafe_load(slot) != baseline_indices[slot]:
                        raise Error("post-timing selected index differs")
            _same(host, _read(ctx, values), "post-timing input mutation")
        _ = values^
        _ = ov^
        _ = oi^
        _ = bv^
        _ = bi^
        _ = hi^


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("small-k qualification requires IDENTICAL")
    var samples = _env_int("MOJOLEARN_SMALLK_SAMPLES", 9)
    if samples < 7:
        raise Error("at least seven timing samples required")
    for profile in range(5):
        for direction in range(2):
            _case(1, 1, 1, profile, direction == 1, False, samples)
            _case(3, 7, 1, profile, direction == 1, False, samples)
            _case(3, 16, 16, profile, direction == 1, False, samples)
            _case(3, 33, 8, profile, direction == 1, False, samples)
            _case(3, 257, 10, profile, direction == 1, False, samples)
            _case(3, 1025, 16, profile, direction == 1, False, samples)
    if String(getenv("MOJOLEARN_SMALLK_LARGE")) == "1":
        var rows = _env_int("MOJOLEARN_SMALLK_ROWS", 32)
        var n = _env_int("MOJOLEARN_SMALLK_COLS", 100000)
        if rows <= 0 or rows > 128 or n < 16 or n > 1000000 or rows * n > 16000000:
            raise Error("small-k fixture exceeds bounded dimensions/storage")
        for profile in range(2):
            _case(rows, n, 8, profile, True, True, samples)
            _case(rows, n, 10, profile, True, True, samples)
            _case(rows, n, 16, profile, True, True, samples)
    print("KNN SMALL-K SELECTION QUALIFICATION PASS")
