# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""cpu2-l6-bindings (2026-10-04): the sum of |v| over a Float32 vector in
ONE fixed binary64 order, on the device and on the host column.

WHY. The forest regressors pick their fixed-point label scale from
`sum over all rows of |y|` (`checks/fixed_point.choose_scale`). That sum was
a serial host loop over every row inside the GPU fit. A device sum must give
the host column the same word, so the ORDER is part of the contract and is
the same on every vendor and on the host
(`core/abs_sum_blocked_host.mojo` holds the host column, GPU-import free):

1. rows are cut into chunks of `ABS_SUM_CHUNK` consecutive rows; chunk `c`
   sums `|v[r]|` for its rows in ascending `r`, starting from +0;
2. the chunk sums are then folded pairwise, level by level: at each level
   `out[i] = in[2i] + in[2i + 1]` (or `in[2i]` alone for an odd tail),
   until one value is left.

Every addition is `checks/soft_f64.sf64_add` (binary64, round-to-nearest-
even; the Apple GPU has no float64), on the device and on the host, so the
words agree everywhere. `|v|` is the float's bits with the sign cleared and
`sf64_from_f32` widens exactly. The order is fixed by `n` alone (no grid
size, no vendor width), so no vendor can disagree.

This order differs from the old serial loop, so the scale CAN move against
the previous version (old bits do not matter); device and host column move
together.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.soft_f64 import SF64_ZERO, sf64_add
from core.abs_sum_blocked_host import ABS_SUM_CHUNK, abs_word_f64
from core.device_zero import enqueue_fill

comptime ABS_SUM_TPB = 256


def abs_sum_chunks_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[UInt64, MutAnyOrigin],
    n_in: Int32,
    n_chunks_in: Int32,
):
    """Step 1: one thread per chunk."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(n_chunks_in):
        return
    var n = Int(n_in)
    var lo = c * ABS_SUM_CHUNK
    var hi = min(lo + ABS_SUM_CHUNK, n)
    var acc = SF64_ZERO
    for r in range(lo, hi):
        acc = sf64_add(acc, abs_word_f64(src.unsafe_load(r)))
    part.unsafe_store(c, acc)


def abs_sum_pair_kernel(
    src: MutPointer[UInt64, MutAnyOrigin],
    dst: MutPointer[UInt64, MutAnyOrigin],
    m_in: Int32,
):
    """Step 2, one level: `dst[i] = src[2i] + src[2i + 1]`, odd tail copied."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var m = Int(m_in)
    var half = (m + 1) // 2
    if i >= half:
        return
    if 2 * i + 1 < m:
        dst.unsafe_store(i, sf64_add(src.unsafe_load(2 * i), src.unsafe_load(2 * i + 1)))
    else:
        dst.unsafe_store(i, src.unsafe_load(2 * i))


def device_abs_sum_blocked(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> Float64:
    """The blocked sum of |buf[0:n]| (module docstring), computed on the
    device; ONE 8-byte word is read back."""
    if n <= 0:
        return Float64(0)
    if n > 2147483647:
        raise Error("device_abs_sum_blocked: n exceeds Int32")
    var m = (n + ABS_SUM_CHUNK - 1) // ABS_SUM_CHUNK
    var a = ctx.enqueue_create_buffer[DType.uint64](m)
    var b = ctx.enqueue_create_buffer[DType.uint64]((m + 1) // 2 if m > 1 else 1)
    ctx.enqueue_function[abs_sum_chunks_kernel](
        buf.unsafe_ptr(), a.unsafe_ptr(), Int32(n), Int32(m),
        grid_dim=((m + ABS_SUM_TPB - 1) // ABS_SUM_TPB, 1, 1),
        block_dim=(ABS_SUM_TPB, 1, 1),
    )
    var in_a = True
    while m > 1:
        var half = (m + 1) // 2
        if in_a:
            ctx.enqueue_function[abs_sum_pair_kernel](
                a.unsafe_ptr(), b.unsafe_ptr(), Int32(m),
                grid_dim=((half + ABS_SUM_TPB - 1) // ABS_SUM_TPB, 1, 1),
                block_dim=(ABS_SUM_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[abs_sum_pair_kernel](
                b.unsafe_ptr(), a.unsafe_ptr(), Int32(m),
                grid_dim=((half + ABS_SUM_TPB - 1) // ABS_SUM_TPB, 1, 1),
                block_dim=(ABS_SUM_TPB, 1, 1),
            )
        in_a = not in_a
        m = half
    var host = ctx.enqueue_create_host_buffer[DType.uint64](1)
    if in_a:
        var head = a.create_sub_buffer[DType.uint64](0, 1)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=head)
        ctx.synchronize()
        _ = head^
    else:
        var head = b.create_sub_buffer[DType.uint64](0, 1)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=head)
        ctx.synchronize()
        _ = head^
    var word = host.unsafe_ptr().unsafe_load(0)
    _ = host^
    _ = a^
    _ = b^
    return bitcast[DType.float64](word)


def index_out_of_range_kernel(
    idx: MutPointer[Int32, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    bound_in: Int32,
):
    """Sets `flag[0] = 1` when any `idx[i]` lies outside `[0, bound)`.
    Every writer stores the same word, so the race is benign and the
    answer is order-free."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var v = idx.unsafe_load(i)
    if v < Int32(0) or v >= bound_in:
        flag.unsafe_store(0, Int32(1))


def device_any_index_out_of_range(
    ctx: DeviceContext, mut idx: DeviceBuffer[DType.int32], n: Int, bound: Int
) raises -> Bool:
    """cpu2-l6-bindings: the row-id range refusal as one device pass over
    the resident ids; ONE Int32 is read back, never the ids."""
    if n <= 0:
        return False
    if n > 2147483647 or bound > 2147483647:
        raise Error("device_any_index_out_of_range: size exceeds Int32")
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_fill(ctx, flag, Int32(0))
    ctx.enqueue_function[index_out_of_range_kernel](
        idx.unsafe_ptr(), flag.unsafe_ptr(), Int32(n), Int32(bound),
        grid_dim=((n + ABS_SUM_TPB - 1) // ABS_SUM_TPB, 1, 1),
        block_dim=(ABS_SUM_TPB, 1, 1),
    )
    var host = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=flag)
    ctx.synchronize()
    var bad = host.unsafe_ptr().unsafe_load(0) != Int32(0)
    _ = host^
    _ = flag^
    return bad
