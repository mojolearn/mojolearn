# SPDX-License-Identifier: Apache-2.0
"""Mamba3 IDENTICAL nonfinite refusal without downloading every operand.

The reduction compares integers only. Its code is 2*index plus 0 for NaN
or 1 for infinity, so a minimum selects the first offending cell, whatever
its sign or NaN payload. Callers retain the existing named-buffer order.
Other numeric modes keep the original host refusal and transfer helpers.
"""
from std.gpu import block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from mamba.impl.modules.mamba3_transfer import m3_download
from mamba.impl.modeling.modeling_mamba import _refuse_nonfinite_named_host

comptime M3_REFUSAL_THREADS = 256
comptime M3_REFUSAL_BLOCKS = 128
comptime M3_REFUSAL_NONE: Int64 = 9223372036854775807
comptime M3_DEVICE_REFUSAL = not is_defined["MOJOLEARN_MAMBA3_LEGACY_REFUSAL"]()


def m3_nonfinite_partial_kernel(
    part: MutPointer[Int64, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin],
    n_in: Int64,
):
    var red = stack_allocation[M3_REFUSAL_THREADS, Scalar[DType.int64], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * M3_REFUSAL_THREADS + tid
    var stride = Int(grid_dim.x) * M3_REFUSAL_THREADS
    var best = M3_REFUSAL_NONE
    while i < Int(n_in):
        var bits = bitcast[DType.uint32](values.unsafe_load(i)) & UInt32(0x7FFFFFFF)
        if bits >= UInt32(0x7F800000):
            best = Int64(i) * 2
            if bits == UInt32(0x7F800000):
                best += 1
            break
        i += stride
    red.unsafe_store(tid, best)
    barrier()
    var active = M3_REFUSAL_THREADS // 2
    while active > 0:
        if tid < active:
            var other = red.unsafe_load(tid + active)
            if other < red.unsafe_load(tid):
                red.unsafe_store(tid, other)
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def m3_first_nonfinite_code(
    ctx: DeviceContext, mut values: DeviceBuffer[DType.float32], n: Int,
) raises -> Int64:
    """First bad index/type, or M3_REFUSAL_NONE; only n leading cells.
    At most 1 KiB is copied to the host, independent of operand size."""
    if n < 0 or n > len(values):
        raise Error("m3_first_nonfinite_code: invalid buffer extent")
    if n == 0:
        return M3_REFUSAL_NONE
    var blocks = min((n + M3_REFUSAL_THREADS - 1) // M3_REFUSAL_THREADS, M3_REFUSAL_BLOCKS)
    var part = ctx.enqueue_create_buffer[DType.int64](blocks)
    var host = ctx.enqueue_create_host_buffer[DType.int64](blocks)
    ctx.synchronize()
    ctx.enqueue_function[m3_nonfinite_partial_kernel](
        part.unsafe_ptr(), values.unsafe_ptr(), Int64(n),
        grid_dim=(blocks, 1, 1), block_dim=(M3_REFUSAL_THREADS, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=part)
    ctx.synchronize()
    var best = M3_REFUSAL_NONE
    for i in range(blocks):  # small-loop(blocks: scan partials, at most M3_REFUSAL_BLOCKS): integer min fold of the device scan partials
        var candidate = host.unsafe_ptr().unsafe_load(i)
        if candidate < best:
            best = candidate
    _ = host^
    _ = part^
    return best


def m3_refuse_nonfinite_named(
    ctx: DeviceContext, name: String,
    mut values: DeviceBuffer[DType.float32], n: Int,
) raises:
    """The existing refusal's exact text, buffer name and first index."""
    comptime if not M3_DEVICE_REFUSAL:
        _refuse_nonfinite_named_host(name, m3_download(ctx, values, n))
    else:
        var code = m3_first_nonfinite_code(ctx, values, n)
        if code == M3_REFUSAL_NONE:
            return
        var index = Int(code // 2)
        if code % 2 == 0:
            raise Error(
                String("mamba: NaN in ") + name + " at flat index "
                + String(index)
                + " REFUSED (row 39: NaN payloads are vendor-shaped; no"
                + " stage may record one)"
            )
        raise Error(
            String("mamba: infinity in ") + name + " at flat index "
            + String(index) + " REFUSED (row 39)"
        )


def _refuse_grad_output_host(tag: String, values: List[Float32]) raises:
    """The host column's grad_output refusal (a column without device
    scans, where `M3_DEVICE_REFUSAL` is False): exponent all ones, by bits,
    first flat index."""
    for i in range(len(values)):
        var bits = bitcast[DType.uint32](values[i])
        if (bits & UInt32(0x7f800000)) == UInt32(0x7f800000):
            raise Error(tag + " backward: non-finite grad_output at flat index " + String(i))


def mamba_refuse_grad_output(
    ctx: DeviceContext, tag: String,
    mut uploaded: DeviceBuffer[DType.float32], values: List[Float32],
) raises:
    """cpu3-seq (2026-10-04): the Mamba-1/2/3 backward's grad_output finite
    refusal on the UPLOADED buffer (one scan, at most 1 KiB read back), with
    the message and first flat index of the host walk it replaces. `values`
    is read only by the host column."""
    comptime if M3_DEVICE_REFUSAL:
        var code = m3_first_nonfinite_code(ctx, uploaded, len(values))
        if code != M3_REFUSAL_NONE:
            raise Error(tag + " backward: non-finite grad_output at flat index " + String(Int(code // 2)))
    else:
        _refuse_grad_output_host(tag, values)
