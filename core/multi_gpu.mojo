# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving transport for independently owned matrix slices.

These operations perform no floating-point arithmetic. Device contexts own
their allocations; callers retain both contexts through every copy and join.
"""
from max.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_dim, block_idx, thread_idx


def peer_clone[dt: DType](source_ctx: DeviceContext, target_ctx: DeviceContext,
                        mut source: DeviceBuffer[dt]) raises -> DeviceBuffer[dt]:
    var target = target_ctx.enqueue_create_buffer[dt](len(source))
    target_ctx.synchronize()
    source.enqueue_copy_to(target)
    source_ctx.synchronize()
    return target^


def copy_columns_kernel[scatter: Bool](
    full: MutPointer[Float32, MutAnyOrigin],
    packed: MutPointer[Float32, MutAnyOrigin],
    stride: Int32, first: Int32, width: Int32, cells: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(cells):
        var j = (i // Int(width)) * Int(stride) + Int(first) + i % Int(width)
        comptime if scatter:
            full[j] = packed[i]
        else:
            packed[i] = full[j]


def copy_scalar_kernel(source: MutPointer[Float32, MutAnyOrigin],
                      target: MutPointer[Float32, MutAnyOrigin]):
    target[0] = source[0]
