# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Isolate WHICH Metal feature the hosted runner lacks.

`hosted_gpu_probe.mojo` runs there: a plain elementwise multiply. The
library's `core_row_norms_row_norm_kernel` does not, failing at "Failed to
create Metal function". The difference between them is `block_sum`, the
block-wide reduction from `max.gpu.primitives.block`, and the threadgroup size
it is instantiated at.

So this probe is the trivial kernel plus exactly one block reduction. If it
fails where the trivial one passed, the block primitive is the boundary, and
the claim "runs on any Apple silicon" needs qualifying rather than widening.
"""

from max.gpu.host import DeviceContext
from max.gpu.primitives.block import sum as block_sum
from std.gpu import block_idx, thread_idx

comptime TPB = 256


def block_reduce_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_cols: Int32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var c = tid
    while c < Int(n_cols):
        acc += src.unsafe_load(row * Int(n_cols) + c)
        c += TPB
    var total = block_sum[block_size=TPB](acc)
    if tid == 0:
        dst.unsafe_store(row, total)


def main() raises:
    comptime ROWS = 8
    comptime COLS = 64
    var ctx = DeviceContext()
    var src = ctx.enqueue_create_buffer[DType.float32](ROWS * COLS)
    var dst = ctx.enqueue_create_buffer[DType.float32](ROWS)
    var h = ctx.enqueue_create_host_buffer[DType.float32](ROWS * COLS)
    ctx.synchronize()
    for i in range(ROWS * COLS):
        h.unsafe_ptr().unsafe_store(i, Float32(1.0))
    ctx.enqueue_copy(dst_buf=src, src_ptr=h.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[block_reduce_kernel](
        dst.unsafe_ptr(), src.unsafe_ptr(), Int32(COLS),
        grid_dim=(ROWS, 1, 1), block_dim=(TPB, 1, 1),
    )
    ctx.synchronize()

    var back = ctx.enqueue_create_host_buffer[DType.float32](ROWS)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=back.unsafe_ptr(), src_buf=dst)
    ctx.synchronize()
    for r in range(ROWS):
        var got = back.unsafe_ptr().unsafe_load(r)
        if got != Float32(COLS):
            raise Error(
                "row " + String(r) + " summed to " + String(got)
                + ", expected " + String(COLS)
            )
    print("BLOCK_PRIMITIVE_OK: block_sum at TPB", TPB, "over", ROWS, "rows")
