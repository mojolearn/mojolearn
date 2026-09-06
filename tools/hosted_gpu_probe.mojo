# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Smallest possible answer to: does Metal compute work on this machine?

Written for GitHub's hosted macOS runners. They are real Apple silicon, but
they are virtualized, and whether a VM exposes the GPU to Metal compute is a
FACT ABOUT THE RUNNER rather than something to reason about from the chip. If
this prints OK in CI, the release workflow can build and test a wheel on a
hosted runner and no self-hosted machine is needed.

Deliberately not part of probe_main.mojo: that file is large, is under active
edit by another session, and a CI probe that fails for an unrelated reason
answers the wrong question.
"""

from max.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx


def double_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    inp: MutPointer[Float32, MutAnyOrigin],
):
    var i = Int(block_idx.x) * 64 + Int(thread_idx.x)
    dst.unsafe_store(i, inp.unsafe_load(i) * Float32(2.0))


def main() raises:
    comptime N = 256
    var ctx = DeviceContext()
    var din = ctx.enqueue_create_buffer[DType.float32](N)
    var dout = ctx.enqueue_create_buffer[DType.float32](N)
    var h = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.synchronize()

    for i in range(N):
        h.unsafe_ptr().unsafe_store(i, Float32(i))
    ctx.enqueue_copy(dst_buf=din, src_ptr=h.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[double_kernel](
        dout.unsafe_ptr(), din.unsafe_ptr(),
        grid_dim=(N // 64, 1, 1), block_dim=(64, 1, 1),
    )
    ctx.synchronize()

    var back = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=back.unsafe_ptr(), src_buf=dout)
    ctx.synchronize()

    var wrong = 0
    for i in range(N):
        if back.unsafe_ptr().unsafe_load(i) != Float32(i) * Float32(2.0):
            wrong += 1
    if wrong != 0:
        raise Error(
            "GPU ran but produced wrong values: "
            + String(wrong)
            + " of "
            + String(N)
        )
    print("HOSTED_GPU_OK: device context created, kernel ran,", N, "values correct")
