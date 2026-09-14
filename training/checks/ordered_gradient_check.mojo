# SPDX-License-Identifier: Apache-2.0
"""Cloud gate: cancellation must distinguish a left fold from a balanced tree."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from training.byte_lm_parallel import _ordered_add_kernel
from training.checks.train_loop import _upload, download_f32


def main() raises:
    var ctx = DeviceContext()
    var first: List[Float32] = [Float32(1e20), bitcast[DType.float32](UInt32(0x80000000)), Float32(1)]
    var second: List[Float32] = [Float32(1), bitcast[DType.float32](UInt32(0x80000000)), bitcast[DType.float32](UInt32(1))]
    var third: List[Float32] = [Float32(-1e20), Float32(0), Float32(-1)]
    var fourth: List[Float32] = [Float32(1), Float32(0), Float32(1)]
    var total = _upload(ctx, first)
    var parts: List[List[Float32]] = [second^, third^, fourth^]
    for k in range(len(parts)):
        var incoming = _upload(ctx, parts[k])
        ctx.enqueue_function[_ordered_add_kernel](total.unsafe_ptr(), incoming.unsafe_ptr(), Int32(3),
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1))
        ctx.synchronize()
        _ = incoming^
    var result = download_f32(ctx, total, 3)
    if bitcast[DType.uint32](result[0]) != UInt32(0x3f800000):
        raise Error("gradient reduction is not the prescribed left fold")
    if bitcast[DType.uint32](result[1]) != UInt32(0) or bitcast[DType.uint32](result[2]) != UInt32(0x3f800000):
        raise Error("gradient reduction FTZ/zero contract differs")
    print("PASS ordered gradient cancellation, signed zero and subnormal seams")
    _ = total^
    ctx.synchronize()
