# SPDX-License-Identifier: Apache-2.0
"""Mamba3 IDENTICAL byte transfers; no floating-point operations.

One bulk copy each way through a pinned stage. The legacy per-call host copy
(`-D MOJOLEARN_MAMBA3_LEGACY_HOST_COPY`, the old A/B arm) is gone
(cpu-gpu-cleanup n-train-mamba, 2026-10-02). Mamba1/2 imports are unchanged.
"""
from std.memory import memcpy
from max.gpu.host import DeviceBuffer, DeviceContext
from mamba.impl.modeling.modeling_mamba import (
    mamba_copy_in,
    mamba_device_alloc,
)


def m3_upload(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var n_buf = max(n, 1)
    var dev = mamba_device_alloc(ctx, n_buf)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
    ctx.synchronize()
    if n > 0:
        memcpy(dest=host.unsafe_ptr(), src=values.unsafe_ptr(), count=n)
    else:
        host.unsafe_ptr().unsafe_store(0, Float32(0.0))
    mamba_copy_in(ctx, dev, host.unsafe_ptr(), n_buf)
    ctx.synchronize()
    _ = host^
    return dev^


def m3_download(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32](length=n, fill=Float32(0.0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=host.unsafe_ptr(), count=n)
    _ = host^
    return out^
