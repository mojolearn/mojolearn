# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The transformer backward's check readback: one device span copied to a host
list, for the independent gradient checks and for naming the one offending
word of a refused non-finite gradient (bwd_refuse_nonfinite_d_out). Moved
unchanged out of autograd/kernels/transformer.mojo, which held it while that
file lived under transformer/checks/."""

from max.gpu.host import DeviceBuffer, DeviceContext
from core.step_phase import step_count_d2h, step_count_host_alloc, step_count_sync


def _download(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """Read a device span for checks or nonfinite-error classification."""
    step_count_host_alloc()
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    step_count_sync()
    ctx.synchronize()
    if n == len(buf):
        step_count_d2h()
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        step_count_d2h()
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    step_count_sync()
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^
