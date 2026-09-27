# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""THE PREP LANE'S PROOF DUMMY (lane/algos-prep, 2026-09-27; removed before
merge). Column L1 mean on the device: one thread per column, rows summed in
ascending order, every operand through `ftz`, the division `identical_div`.
The host oracle x_prep/host/dummy_oracle.mojo is the same loop."""
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from checks.numerics import ftz, identical_div
from metrics.checks.device_io import upload_f32, download_f32


def l1_mean_kernel(
    x: MutPointer[Float32, MutAnyOrigin], n: Int32, d: Int32, output: MutPointer[Float32, MutAnyOrigin],
):
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c < Int(d):
        var acc = Float32(0)
        for i in range(Int(n)):
            acc = ftz(acc + ftz(abs(ftz(x.unsafe_load(i * Int(d) + c)))))
        var mean = ftz(identical_div(acc, Float32(Int(n))))
        # a zero column scales by one: 0/0 is NaN, and NaN payloads are per vendor
        output.unsafe_store(c, mean if mean > Float32(0) else Float32(1))


def scale_kernel(
    x: MutPointer[Float32, MutAnyOrigin], s: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], count: Int32, d: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        output.unsafe_store(i, ftz(identical_div(ftz(x.unsafe_load(i)), ftz(s.unsafe_load(i % Int(d))))))


def l1_mean_fit_device(x: List[Float32], n: Int, d: Int) raises -> List[Float32]:
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var out = ctx.enqueue_create_buffer[DType.float32](d)
    ctx.enqueue_function[l1_mean_kernel](
        dx.unsafe_ptr(), Int32(n), Int32(d), out.unsafe_ptr(), grid_dim=(d + 63) // 64, block_dim=64,
    )
    var result = download_f32(ctx, out, d)
    _ = out^
    _ = dx^
    _ = ctx^
    return result^


def scale_device(x: List[Float32], s: List[Float32], n: Int, d: Int) raises -> List[Float32]:
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var ds = upload_f32(ctx, s)
    var out = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.enqueue_function[scale_kernel](
        dx.unsafe_ptr(), ds.unsafe_ptr(), out.unsafe_ptr(), Int32(n * d), Int32(d),
        grid_dim=(n * d + 255) // 256, block_dim=256,
    )
    var result = download_f32(ctx, out, n * d)
    _ = out^
    _ = ds^
    _ = dx^
    _ = ctx^
    return result^
