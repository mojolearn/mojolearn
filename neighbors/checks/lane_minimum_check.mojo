# SPDX-License-Identifier: Apache-2.0
"""Check all lanes, two successive reductions, native and shared schedules.

Widths above this device's native group test shared-memory portability only;
they do not constitute execution evidence for another physical GPU column.
"""
from std.gpu import thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from checks.kernel_matrix import TARGET_COLUMN, column_lane_width, column_lane_width_is_fixed, column_max_block_size, column_is_simulated, column_is_buildable, column_name, COLUMN_AMD_RDNA
from neighbors.checks.lane_minimum import logical_min_u64

comptime BLOCK = 128 if column_max_block_size(TARGET_COLUMN) < 256 else 256

@always_inline
def fixture(lane: Int, iteration: Int) -> UInt64:
    # High-word ties, alternating high-bit values and descending low words.
    var high = UInt64((lane * 17 + iteration * 11) % 7)
    if iteration == 1:
        high |= UInt64(0x80000000)
    return (high << UInt64(32)) | UInt64(255 - lane)


def kernel[WIDTH: Int, FIXED: Bool](output: MutPointer[UInt64, MutAnyOrigin]):
    var tid = Int(thread_idx.x)
    for iteration in range(2):
        var result = logical_min_u64[WIDTH, BLOCK, column_lane_width(TARGET_COLUMN), FIXED](fixture(tid, iteration))
        output.unsafe_store(iteration * BLOCK + tid, result)


def check[WIDTH: Int, FIXED: Bool](ctx: DeviceContext) raises:
    var output = ctx.enqueue_create_buffer[DType.uint64](BLOCK * 2)
    var host = ctx.enqueue_create_host_buffer[DType.uint64](BLOCK * 2)
    ctx.enqueue_function[kernel[WIDTH, FIXED]](output.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(BLOCK, 1, 1))
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=output)
    ctx.synchronize()
    for iteration in range(2):
        for tid in range(BLOCK):
            var expected = UInt64(0xffffffffffffffff)
            var base = tid // WIDTH * WIDTH
            for lane in range(base, base + WIDTH):
                var v = fixture(lane, iteration)
                if v < expected:
                    expected = v
            if host.unsafe_ptr().unsafe_load(iteration * BLOCK + tid) != expected:
                print("LANE_MIN_FAIL", WIDTH, FIXED, iteration, tid)
                raise Error("logical lane minimum mismatch")
    print("LANE_MIN_PASS", "width", WIDTH, "fixed", FIXED, "cells", BLOCK * 2)


def main() raises:
    comptime if is_defined["MOJOLEARN_REQUIRE_RDNA_TARGET"]():
        comptime assert TARGET_COLUMN == COLUMN_AMD_RDNA and not column_is_simulated(), "RDNA qualification requires a real RDNA compilation target"
    print("DECLARED_COLUMN", column_name(TARGET_COLUMN), "simulated", column_is_simulated(), "backend_declared_buildable", column_is_buildable(TARGET_COLUMN), "block", BLOCK)
    with DeviceContext() as ctx:
        comptime for exponent in range(8):
            comptime width = 1 << exponent
            check[width, column_lane_width_is_fixed(TARGET_COLUMN) and not column_is_simulated()](ctx)
            check[width, False](ctx)
    print("LANE MINIMUM WIDTH PASS", "cases", 16, "cells", BLOCK * 32)
