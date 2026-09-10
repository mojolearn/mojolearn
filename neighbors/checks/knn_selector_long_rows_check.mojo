# SPDX-License-Identifier: Apache-2.0
"""Check long/ragged selector rows against exhaustive host order statistics.

Exercises all capacity buckets and common-k specializations, both ordering
directions, duplicate values, signed zero, subnormals, infinities and NaN
payloads. The host oracle repeatedly scans the entire row for its next
composite key; it uses no per-thread lists, warp reductions or device output.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.select_radix_identical import composite_key
from neighbors.checks.select_smallk_identical_candidate import smallk_select_launch


def check_case(length: Int, k: Int, select_min: Bool) raises:
    comptime ROWS = 3
    with DeviceContext() as ctx:
        var host = ctx.enqueue_create_host_buffer[DType.float32](ROWS * length)
        var got_d = ctx.enqueue_create_host_buffer[DType.float32](ROWS * k)
        var got_i = ctx.enqueue_create_host_buffer[DType.uint32](ROWS * k)
        ctx.synchronize()
        var special: List[UInt32] = [0, 2147483648, 1, 2147483649, 2139095040, 4286578688, 2143289345, 2143289346, 4290772993]
        for row in range(ROWS):
            for col in range(length):
                var bits = UInt32(1065353216 + ((col * 37 + row * 13) % 257))
                if row == 1:
                    bits = special[col % len(special)]
                elif row == 2:
                    bits = UInt32(1065353216)
                host.unsafe_ptr().unsafe_store(row * length + col, bitcast[DType.float32](bits))
        var values = ctx.enqueue_create_buffer[DType.float32](ROWS * length)
        var distances = ctx.enqueue_create_buffer[DType.float32](ROWS * k)
        var indices = ctx.enqueue_create_buffer[DType.uint32](ROWS * k)
        ctx.enqueue_copy(dst_buf=values, src_ptr=host.unsafe_ptr())
        smallk_select_launch(
            ctx, values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            distances.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), indices.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            ROWS, length, k, select_min,
        )
        ctx.enqueue_copy(dst_ptr=got_d.unsafe_ptr(), src_buf=distances)
        ctx.enqueue_copy(dst_ptr=got_i.unsafe_ptr(), src_buf=indices)
        ctx.synchronize()
        for row in range(ROWS):
            var previous = UInt64(0)
            for rank in range(k):
                var best = UInt64(18446744073709551615)
                var selected = -1
                for col in range(length):
                    var value = host.unsafe_ptr().unsafe_load(row * length + col)
                    var key = composite_key(value, UInt32(col), select_min)
                    if (rank == 0 or key > previous) and key < best:
                        best = key
                        selected = col
                var cell = row * k + rank
                if selected < 0 or got_i.unsafe_ptr().unsafe_load(cell) != UInt32(selected):
                    raise Error("long-row selector index differs from exhaustive host rank")
                var expected = host.unsafe_ptr().unsafe_load(row * length + selected)
                if bitcast[DType.uint32](got_d.unsafe_ptr().unsafe_load(cell)) != bitcast[DType.uint32](expected):
                    raise Error("long-row selector distance bits differ from host input")
                previous = best
        print("SELECTOR_LONG_CASE_PASS", length, k, select_min, ROWS * k)
        _ = host^
        _ = values^


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("long-row selector gate requires IDENTICAL")
    var lengths: List[Int] = [8192, 8193, 65537]
    var counts: List[Int] = [1, 10, 15, 17, 33, 64]
    for length in lengths:
        for k in counts:
            check_case(length, k, True)
            check_case(length, k, False)
    print("KNN LONG-ROW SELECTOR PASS", "cases", 36)
