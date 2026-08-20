"""Check their reorder path against a host stable partition.

`ReorderOneBitImpl` produces a STABLE partition: all zero-flag rows first, in
their original relative order, then all one-flag rows, in theirs. Both halves
matter. An unstable partition still conserves rows and still puts the right
count on each side, so a count check cannot see the failure -- only comparing
the ORDER can.

The check plants a scattered flag pattern and a distinct value per row, runs
the device path, and compares against a host partition element for element.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_util.kernel.reorder_one_bit import (
    REORDER_BLOCK,
    launch_reorder_one_bit,
)


def check_reorder_one_bit(n: Int = 20000, offset: Int = 37) raises:
    var ctx = DeviceContext()
    var total = offset + n + 11  # slack either side, to catch overruns

    var h_flags = ctx.enqueue_create_host_buffer[DType.uint8](total)
    var h_vals = ctx.enqueue_create_host_buffer[DType.uint32](total)
    # sentinel everywhere, so a write outside [offset, offset+n) shows up
    for i in range(total):
        h_flags.unsafe_ptr().unsafe_store(i, UInt8(9))
        h_vals.unsafe_ptr().unsafe_store(i, UInt32(0xDEADBEEF))

    var host_flag = List[Int]()
    for i in range(n):
        var x = UInt32(i * 2654435761 + 0x9E3779B9)
        x ^= x << 13
        x ^= x >> 17
        x ^= x << 5
        var f = Int(x & 1)
        host_flag.append(f)
        h_flags.unsafe_ptr().unsafe_store(offset + i, UInt8(f))
        h_vals.unsafe_ptr().unsafe_store(offset + i, UInt32(i))

    var temp_flags = ctx.enqueue_create_buffer[DType.uint8](total)
    var temp_vals = ctx.enqueue_create_buffer[DType.uint32](total)
    var out_flags = ctx.enqueue_create_buffer[DType.uint8](total)
    var out_vals = ctx.enqueue_create_buffer[DType.uint32](total)
    ctx.enqueue_copy(dst_buf=temp_flags, src_ptr=h_flags.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=temp_vals, src_ptr=h_vals.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=out_flags, src_ptr=h_flags.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=out_vals, src_ptr=h_vals.unsafe_ptr())

    var n_blocks = (n + REORDER_BLOCK - 1) // REORDER_BLOCK
    var offsets = ctx.enqueue_create_buffer[DType.int32](n)
    var block_sums = ctx.enqueue_create_buffer[DType.int32](n_blocks)
    ctx.synchronize()

    launch_reorder_one_bit(
        ctx, offset, n, temp_flags, temp_vals, offsets, block_sums,
        out_flags, out_vals,
    )
    ctx.synchronize()

    var got = ctx.enqueue_create_host_buffer[DType.uint32](total)
    var gotf = ctx.enqueue_create_host_buffer[DType.uint8](total)
    ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=out_vals)
    ctx.enqueue_copy(dst_ptr=gotf.unsafe_ptr(), src_buf=out_flags)
    ctx.synchronize()

    # host reference: zeros in order, then ones in order
    var want = List[Int]()
    for i in range(n):
        if host_flag[i] == 0:
            want.append(i)
    for i in range(n):
        if host_flag[i] == 1:
            want.append(i)

    var wrong = 0
    for i in range(n):
        if Int(got.unsafe_ptr().unsafe_load(offset + i)) != want[i]:
            wrong += 1
    print("    stable partition, order wrong:", wrong, "of", n)

    var spill = 0
    for i in range(offset):
        if got.unsafe_ptr().unsafe_load(i) != UInt32(0xDEADBEEF):
            spill += 1
    for i in range(offset + n, total):
        if got.unsafe_ptr().unsafe_load(i) != UInt32(0xDEADBEEF):
            spill += 1
    print("    wrote outside the leaf:", spill, "cells")

    var zeros = 0
    for i in range(n):
        if host_flag[i] == 0:
            zeros += 1
    var flagwrong = 0
    for i in range(n):
        var wf = 0 if i < zeros else 1
        if Int(gotf.unsafe_ptr().unsafe_load(offset + i)) != wf:
            flagwrong += 1
    print("    flags misplaced:", flagwrong, "of", n)

    if wrong != 0:
        raise Error(
            String("reorder is not a stable partition: ") + String(wrong)
            + " of " + String(n) + " out of order"
        )
    if spill != 0:
        raise Error("reorder wrote outside the leaf it was given")
    if flagwrong != 0:
        raise Error("reorder did not move the flags with the values")
    print("  ReorderOneBitImpl matches a host stable partition")
