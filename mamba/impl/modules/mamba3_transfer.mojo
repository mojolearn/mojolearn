# SPDX-License-Identifier: Apache-2.0
"""Mamba3 IDENTICAL byte transfers; no floating-point operations.

The legacy switch exists for same-source public-API A/B measurements. Other
numeric modes retain the original helpers. Mamba1/2 imports are unchanged.
"""
from std.memory import memcpy
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from mamba.impl.modeling.modeling_mamba import (
    mamba_upload, mamba_download,
)

comptime M3_BULK_TRANSFER = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_MAMBA3_LEGACY_HOST_COPY"]()


def m3_upload(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    comptime if not M3_BULK_TRANSFER:
        return mamba_upload(ctx, values)
    else:
        var n = len(values)
        var n_buf = max(n, 1)
        var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
        var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
        ctx.synchronize()
        if n > 0:
            memcpy(dest=host.unsafe_ptr(), src=values.unsafe_ptr(), count=n)
        else:
            host.unsafe_ptr().unsafe_store(0, Float32(0.0))
        ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
        ctx.synchronize()
        _ = host^
        return dev^


def m3_download(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    comptime if not M3_BULK_TRANSFER:
        return mamba_download(ctx, buf, n)
    else:
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
