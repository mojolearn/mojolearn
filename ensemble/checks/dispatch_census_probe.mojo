# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What counts as a Compute dispatch on this Metal stack? Measured, not
assumed.

The launch-log join in `profile_metal.py` aligns host enqueue order with
the trace's Compute intervals, so it must know which `DeviceContext`
operations EMIT a Compute interval. CUDA intuition says memsets and
copies are DMA/blit work; on Apple unified memory any of them may be a
hidden compute kernel, a blit, or a plain host memcpy. This probe issues
PRIME counts of each -- 7 device memsets, 11 host-to-device copies, 13
device-to-device copies, 5 device-to-host copies, 17 trivial kernel
launches -- so `xctrace`'s Compute-row count decomposes UNIQUELY over
{7, 11, 13, 5, 17}: any observed total has exactly one explanation.

Run under xctrace, then count Compute rows for this process:

    xctrace record --template 'Metal System Trace' \
        --output census.trace --launch -- ./build/dispatch_census_probe
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_idx, thread_idx


def touch_kernel(p: MutPointer[Int32, MutAnyOrigin], n: Int32):
    var i = Int(block_idx.x) * 64 + Int(thread_idx.x)
    if i < Int(n):
        p[unsafe_offset=i] = p[unsafe_offset=i] + 1


def main() raises:
    var ctx = DeviceContext()
    comptime N = 4096
    var a = ctx.enqueue_create_buffer[DType.int32](N)
    var b = ctx.enqueue_create_buffer[DType.int32](N)
    var h = ctx.enqueue_create_host_buffer[DType.int32](N)
    for i in range(N):
        h.unsafe_ptr().unsafe_store(i, Int32(i))
    ctx.synchronize()

    # 7 device memsets
    for _ in range(7):
        ctx.enqueue_memset(a, Int32(0))
    # 11 host-to-device copies
    for _ in range(11):
        ctx.enqueue_copy(dst_buf=a, src_ptr=h.unsafe_ptr())
    # 13 device-to-device copies
    for _ in range(13):
        ctx.enqueue_copy(dst_buf=b, src_buf=a)
    # 5 device-to-host copies
    for _ in range(5):
        ctx.enqueue_copy(dst_buf=h, src_buf=b)
    # 17 trivial kernel launches
    for _ in range(17):
        ctx.enqueue_function[touch_kernel](
            a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            Int32(N),
            grid_dim=(N + 63) // 64,
            block_dim=64,
        )
    ctx.synchronize()
    print("census done", h.unsafe_ptr().unsafe_load(0))
    _ = a^
    _ = b^
    _ = h^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^
