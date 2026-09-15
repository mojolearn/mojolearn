# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving transport for independently owned matrix slices.

These operations perform no floating-point arithmetic. Device contexts own
their allocations; callers retain both contexts through every copy and join.
"""
from max.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.info import has_amd_gpu_accelerator


def transfer_bytes[dt: DType](source_ctx: DeviceContext, target_ctx: DeviceContext,
                             mut source: DeviceBuffer[dt], mut target: DeviceBuffer[dt],
                             cells: Int, cross_device: Bool) raises:
    """Copy `cells` elements of `source` into `target`.

    AMD (HIP) CROSS-DEVICE COPIES ARE STAGED THROUGH HOST MEMORY. On two
    RunPod MI300X (SR-IOV virtual functions) a kernel on the target device,
    launched after `enqueue_copy_to` and `synchronize()` on BOTH contexts,
    read cells the copy had not yet written: their bytes were the previous
    contents of that memory (training/checks/peer_copy_check.mojo PEERSOLVE;
    bench/results/multi_gpu/2026-09-15/peer-copy-mi300x/). The same source
    on two H100s never did. Staging reads the source into pinned host memory
    through the source context and writes it through the target context, so
    the target context's own drain covers the write. A same-device copy, and
    every copy on other vendors, stays a device copy.
    """
    if len(source) < cells or len(target) < cells:
        raise Error("transfer_bytes: a buffer is shorter than the copy")
    comptime if has_amd_gpu_accelerator():
        if cross_device:
            var host = source_ctx.enqueue_create_host_buffer[dt](cells)
            # A whole buffer is copied as itself, so a caller's sub-buffer is
            # never viewed again.
            if cells == len(source):
                source_ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=source)
            else:
                var sv = source.create_sub_buffer[dt](0, cells)
                source_ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=sv)
                source_ctx.synchronize()
                _ = sv^
            source_ctx.synchronize()
            if cells == len(target):
                target_ctx.enqueue_copy(dst_buf=target, src_ptr=host.unsafe_ptr())
            else:
                var tv = target.create_sub_buffer[dt](0, cells)
                target_ctx.enqueue_copy(dst_buf=tv, src_ptr=host.unsafe_ptr())
                target_ctx.synchronize()
                _ = tv^
            target_ctx.synchronize()
            _ = host^
            return
    # Unchanged from the call sites this replaces: the device copy and a
    # drain of the source context.
    source.enqueue_copy_to(target)
    source_ctx.synchronize()


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
