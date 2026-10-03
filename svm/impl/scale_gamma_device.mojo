# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""gamma='scale' exact sums ON THE DEVICE (lane/apple-fast-py2mojo-linear).

The limbs of `svm/impl/scale_gamma_limbs.mojo` from a grid: every thread
adds a grid-stride share of the cells into its own limbs, carries them, and
each block folds its threads' limbs slot by slot in shared memory into one
partial row; a second launch, one block per slot, folds the partial rows.
Integer sums are exact, so the words' value equals the host column's.
"""

from std.gpu import block_idx, thread_idx
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.memory import stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.py2mojo_linear import py2mojo_linear_flags

from svm.impl.scale_gamma_limbs import (
    SG_CHUNK,
    SG_SLOTS,
    sg_add_cell,
    sg_normalize,
    sg_zero,
)

comptime SG_TPB = 256
comptime SG_MAX_BLOCKS = 1024
#: cells per thread the grid is sized for (more threads past this pay only
#: the fold)
comptime SG_PER_THREAD = 16

comptime _U32P = MutPointer[UInt32, MutAnyOrigin]
comptime _I64P = MutPointer[Int64, MutAnyOrigin]


def sg_partial_kernel(x: _U32P, count_in: Int64, nb_in: Int32, part: _I64P):
    var count = Int(count_in)
    var nb = Int(nb_in)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var acc = sg_zero()
    var stride = SG_TPB * nb
    var i = blk * SG_TPB + tid
    var since = 0
    while i < count:
        sg_add_cell(acc, x[i])
        since += 1
        if since == SG_CHUNK:
            sg_normalize(acc)
            since = 0
        i += stride
    sg_normalize(acc)
    var s = stack_allocation[SG_TPB, Scalar[DType.int64], address_space = AddressSpace.SHARED]()
    comptime for slot in range(SG_SLOTS):
        s[tid] = acc[slot]
        barrier()
        var step = SG_TPB // 2
        while step > 0:
            if tid < step:
                s[tid] = s[tid] + s[tid + step]
            barrier()
            step //= 2
        if tid == 0:
            part[slot * nb + blk] = s[0]
        barrier()


def sg_finish_kernel(part: _I64P, nb_in: Int32, dst: _I64P):
    """Block `slot` folds that slot's `nb` partials."""
    var nb = Int(nb_in)
    var tid = Int(thread_idx.x)
    var slot = Int(block_idx.x)
    var v = Int64(0)
    var j = tid
    while j < nb:
        v += part[slot * nb + j]
        j += SG_TPB
    var s = stack_allocation[SG_TPB, Scalar[DType.int64], address_space = AddressSpace.SHARED]()
    s[tid] = v
    barrier()
    var step = SG_TPB // 2
    while step > 0:
        if tid < step:
            s[tid] = s[tid] + s[tid + step]
        barrier()
        step //= 2
    if tid == 0:
        dst[slot] = s[0]


@always_inline
def _grid_blocks(count: Int) -> Int:
    var want = (count + SG_TPB * SG_PER_THREAD - 1) // (SG_TPB * SG_PER_THREAD)
    return max(1, min(SG_MAX_BLOCKS, want))


def scale_gamma_limbs_device(ctx: DeviceContext, x_addr: Int, count: Int, dst_addr: Int) raises:
    """Uploads the `count` float32 cells at `x_addr` (as bits) and writes the
    SG_SLOTS int64 words at `dst_addr`."""
    var dst = _I64P(unsafe_from_address=dst_addr)
    if count == 0:
        for k in range(SG_SLOTS):
            dst[k] = 0
        return
    var nb = _grid_blocks(count)
    var d_x = ctx.enqueue_create_buffer[DType.uint32](count)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=_U32P(unsafe_from_address=x_addr))
    var d_part = ctx.enqueue_create_buffer[DType.int64](nb * SG_SLOTS)
    var d_out = ctx.enqueue_create_buffer[DType.int64](SG_SLOTS)
    ctx.enqueue_function[sg_partial_kernel](
        d_x.unsafe_ptr(), Int64(count), Int32(nb), d_part.unsafe_ptr(), grid_dim=nb, block_dim=SG_TPB,
    )
    ctx.enqueue_function[sg_finish_kernel](
        d_part.unsafe_ptr(), Int32(nb), d_out.unsafe_ptr(), grid_dim=SG_SLOTS, block_dim=SG_TPB,
    )
    ctx.enqueue_copy(dst_ptr=dst, src_buf=d_out)
    ctx.synchronize()
    _ = d_x^
    _ = d_part^
    _ = d_out^


def scale_gamma_limbs_device_binding(
    ctx: DeviceContext, x_addr: PythonObject, count: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """The body of each GPU binding's `scale_gamma_limbs(x, count, out)`;
    the binding hands in its family context. Returns count."""
    var n = Int(py=count)
    var xa = Int(py=x_addr)
    var oa = Int(py=out_addr)
    if n < 0 or oa == 0 or (n > 0 and xa == 0):
        raise Error("scale_gamma_limbs: null buffer or negative count")
    with GILReleased(Python()):
        scale_gamma_limbs_device(ctx, xa, n, oa)
    return PythonObject(n)


def py2mojo_linear_flags_binding() raises -> PythonObject:
    return PythonObject(py2mojo_linear_flags())
