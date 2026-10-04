# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""cpu3-bindings (2026-10-04): the row-weight refusals and total of a GPU
fit, as one device pass over the resident weights.

The weighted forest fits used to walk every row on the host (finite and
nonnegative check, all-ones check, a serial binary64 total) before the
upload. Here the weights are uploaded once and checked where they live:

- `bad`: some weight is not in [0, FLT_MAX] (NaN, inf or negative);
- `all_unit`: every weight is exactly 1.0 (the caller then fits unweighted);
- `total`: the blocked binary64 sum (`core/abs_sum_blocked`, a fixed order
  that is the same on every vendor and on the host column's
  `host_abs_sum_blocked`); weights are nonnegative, so |w| = w.

The two flags are order-free (every writer stores the same word). Two Int32
and one 8-byte word are read back, never the weights.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.abs_sum_blocked import device_abs_sum_blocked
from core.device_zero import enqueue_fill

comptime WEIGHT_SCAN_TPB = 256


def weight_flags_kernel(
    w: MutPointer[Float32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """flags[0] = 1 when any weight is outside [0, FLT_MAX];
    flags[1] = 1 when any weight differs from 1.0."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var v = w.unsafe_load(i)
    if not (v >= Float32(0) and v <= Float32(3.4028234663852886e38)):
        flags.unsafe_store(0, Int32(1))
    if v != Float32(1):
        flags.unsafe_store(1, Int32(1))


@fieldwise_init
struct WeightScan(Copyable, Movable):
    var bad: Bool
    var all_unit: Bool
    var total: Float64


def device_scan_weights(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> WeightScan:
    """Scan `buf[0:n]` on the device (module docstring). `total` is computed
    only when no weight is bad (it is 0 otherwise)."""
    if n <= 0:
        return WeightScan(False, True, Float64(0))
    if n > 2147483647:
        raise Error("device_scan_weights: n exceeds Int32")
    var flags = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, flags, Int32(0))
    ctx.enqueue_function[weight_flags_kernel](
        buf.unsafe_ptr(), flags.unsafe_ptr(), Int32(n),
        grid_dim=((n + WEIGHT_SCAN_TPB - 1) // WEIGHT_SCAN_TPB, 1, 1),
        block_dim=(WEIGHT_SCAN_TPB, 1, 1),
    )
    var host = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=flags)
    ctx.synchronize()
    var bad = host.unsafe_ptr().unsafe_load(0) != Int32(0)
    var all_unit = host.unsafe_ptr().unsafe_load(1) == Int32(0)
    _ = host^
    _ = flags^
    var total = Float64(0)
    if not bad:
        total = device_abs_sum_blocked(ctx, buf, n)
    return WeightScan(bad, all_unit, total)
