# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The query grouping (`group_sizes`) as device offsets, validated ON THE
DEVICE (lane/cpu3-gbdt-b, 2026-10-04).

`gbdt/train.mojo::train`, `make_querywise_target_buffers` (QueryRMSE) and
`make_pairwise_group_buffers` (PairLogit, the fused group layout) each walked
the sizes on the host: a running sum into a staged offsets buffer and a
refusal of an empty query or a sum that misses `n_rows`. Now the sizes go up
once, `launch_scan_vector_u32` (the device prefix sum the CTR block already
uses) forms the inclusive sums, and one kernel writes the `n + 1` offsets
(`offsets[0] = 0`, `offsets[q + 1]` the end of query `q`) and flags:

  word 0  the lowest query with size 0 (`GROUP_NO_Q` = none)
  word 1  1 when the running sum wrapped 2^32 (some `incl[q] <= incl[q-1]`
          with every size positive), so the total below is not the true sum
  word 2  the total (`offsets[n]`)

Integer work only: every vendor and the host get the same offsets.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from gbdt.gpu_util.kernel.scan import SCAN_BLOCK, launch_scan_vector_u32

comptime GROUP_TPB = 256
comptime GROUP_NO_Q = Int32(2147483647)


struct GroupLayout(Movable):
    """`sizes` (n) and `offsets` (n + 1) of the grouping, on the device."""

    var n_groups: Int
    var sizes: DeviceBuffer[DType.uint32]
    var offsets: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        n_groups: Int,
        var sizes: DeviceBuffer[DType.uint32],
        var offsets: DeviceBuffer[DType.uint32],
    ):
        self.n_groups = n_groups
        self.sizes = sizes^
        self.offsets = offsets^


def group_offsets_kernel(
    sizes: MutPointer[UInt32, MutAnyOrigin],
    incl: MutPointer[UInt32, MutAnyOrigin],
    offsets: MutPointer[UInt32, MutAnyOrigin],
    words: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if q >= n:
        return
    var end = incl.unsafe_load(q)
    offsets.unsafe_store(q + 1, end)
    if q == 0:
        offsets.unsafe_store(0, UInt32(0))
    if sizes.unsafe_load(q) == UInt32(0):
        _ = Atomic.min(words.unsafe_offset(0), Int32(q))
    elif q > 0 and end <= incl.unsafe_load(q - 1):
        _ = Atomic.max(words.unsafe_offset(1), Int32(1))


def device_group_layout(
    ctx: DeviceContext,
    group_sizes: List[UInt32],
    n_rows: Int,
    empty_prefix: String,
    cover_prefix: String,
) raises -> GroupLayout:
    """Upload, scan and check the grouping. Refuses, in the callers' words,
    an empty query (`empty_prefix + q + " has no rows"`) and a total other
    than `n_rows` (`cover_prefix + total + " rows of " + n_rows`). The
    caller refuses an empty `group_sizes` first. One 12 B readback."""
    var n = len(group_sizes)
    var d_sizes = ctx.enqueue_create_buffer[DType.uint32](max(1, n))
    var d_off = ctx.enqueue_create_buffer[DType.uint32](n + 1)
    if n == 0:
        if n_rows != 0:
            raise Error(cover_prefix + "0 rows of " + String(n_rows))
        return GroupLayout(0, d_sizes^, d_off^)
    ctx.enqueue_copy(dst_buf=d_sizes, src_ptr=group_sizes.unsafe_ptr())
    var d_incl = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_bs = ctx.enqueue_create_buffer[DType.uint32](
        (n + SCAN_BLOCK - 1) // SCAN_BLOCK
    )
    launch_scan_vector_u32(ctx, n, True, d_sizes, d_incl, d_bs)
    var h = ctx.enqueue_create_host_buffer[DType.int32](3)
    h.unsafe_ptr().unsafe_store(0, GROUP_NO_Q)
    h.unsafe_ptr().unsafe_store(1, Int32(0))
    h.unsafe_ptr().unsafe_store(2, Int32(0))
    var d_w = ctx.enqueue_create_buffer[DType.int32](3)
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h.unsafe_ptr())
    ctx.enqueue_function[group_offsets_kernel](
        d_sizes.unsafe_ptr(),
        d_incl.unsafe_ptr(),
        d_off.unsafe_ptr(),
        d_w.unsafe_ptr(),
        Int32(n),
        grid_dim=((n + GROUP_TPB - 1) // GROUP_TPB, 1, 1),
        block_dim=(GROUP_TPB, 1, 1),
    )
    var h_total = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var tail = d_off.create_sub_buffer[DType.uint32](n, 1)
    ctx.enqueue_copy(dst_ptr=h_total.unsafe_ptr(), src_buf=tail)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_w)
    ctx.synchronize()
    _ = len(group_sizes)
    var empty_q = h.unsafe_ptr().unsafe_load(0)
    var wrapped = h.unsafe_ptr().unsafe_load(1) != Int32(0)
    var total = Int(h_total.unsafe_ptr().unsafe_load(0))
    _ = tail^
    _ = d_incl^
    _ = d_bs^
    _ = d_w^
    _ = h^
    _ = h_total^
    if empty_q != GROUP_NO_Q:
        raise Error(empty_prefix + String(Int(empty_q)) + " has no rows")
    if wrapped:
        raise Error(cover_prefix + "more than 2^32 rows of " + String(n_rows))
    if total != n_rows:
        raise Error(cover_prefix + String(total) + " rows of " + String(n_rows))
    return GroupLayout(n, d_sizes^, d_off^)
