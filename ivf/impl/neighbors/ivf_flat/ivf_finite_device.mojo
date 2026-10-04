# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF finiteness on the device (lane fix-k1-neighbors, audit K1 / F7
`ivf_finiteness`, 2026-10-04).

`ivf_validate_data` (ivf_flat_index.mojo) walks every value of the dataset,
the extension rows and the queries on one host thread before the upload.
Under `IVF_IDN_DEVICE_FINITE` the device routes upload first and scan the
uploaded words here instead:

    ivf_bound_partial_kernel   grid-stride, one Int32 per block: the
                               smallest flat index whose word has
                               `(bits & 0x7FFFFFFF) >= bits(2^63)`, or
                               `IVF_FINITE_NONE`
    ivf_bound_fold_kernel      one block: the minimum over the partials

then ONE 4 B readback of the verdict, which the host needs to raise. The
predicate is `ivf_validate_data`'s, by bits: NaN, both infinities and every
|v| >= 2^63 are exactly the words whose low 31 bits are at or above 2^63's
(the order of nonnegative floats is the order of their words). An integer
minimum is exact and order-free, so the verdict and the first index are the
same on every vendor and at every grid size. Nothing here reaches a card or
an output: on a hit the caller re-runs the host walk ONLY to raise the same
message (the failure path), so the refusal text is unchanged.

IDENTICAL only; FAST keeps the host walk. `-D MOJOLEARN_IDN_IVF_DEVICE_FINITE_OFF`
(or `MOJOLEARN_IDN_ALL_OFF`) restores the host walk before the upload. No bit
of any output moves either way (a scan only refuses).
"""

from std.gpu import block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import (
    IVF_MAGNITUDE_BOUND,
    ivf_validate_data,
)

comptime IVF_IDN_DEVICE_FINITE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_IVF_DEVICE_FINITE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
"""Default ON in IDENTICAL; see the module docstring."""

comptime IVF_FINITE_TPB = 256
"""Threads per block of both kernels (a halving tree over it)."""
comptime IVF_FINITE_BLOCKS = 512
"""Grid cap of the partial scan; above it every thread grid-strides."""
comptime IVF_FINITE_NONE: Int32 = 2147483647
"""A block's partial when its slice holds no hit."""


def ivf_bound_partial_kernel(
    part: MutPointer[Int32, MutAnyOrigin],
    buf: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    bound_bits: UInt32,
):
    """One partial per block: the smallest flat index in this block's
    grid-stride slice with `(bits & 0x7FFFFFFF) >= bound_bits`."""
    var n = Int(n_in)
    var red = stack_allocation[
        IVF_FINITE_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * IVF_FINITE_TPB
    var i = Int(block_idx.x) * IVF_FINITE_TPB + tid
    var best = IVF_FINITE_NONE
    while i < n:
        var au = bitcast[DType.uint32](buf.unsafe_load(i)) & UInt32(0x7FFFFFFF)
        if au >= bound_bits:
            best = Int32(i)
            break
        i += stride
    red.unsafe_store(tid, best)
    barrier()
    var active = IVF_FINITE_TPB // 2
    while active > 0:
        if tid < active:
            var o = red.unsafe_load(tid + active)
            if o < red.unsafe_load(tid):
                red.unsafe_store(tid, o)
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def ivf_bound_fold_kernel(
    out: MutPointer[Int32, MutAnyOrigin],
    part: MutPointer[Int32, MutAnyOrigin],
    blocks_in: Int32,
):
    """One block: the minimum over `blocks_in` partials into `out[0]`."""
    var blocks = Int(blocks_in)
    var red = stack_allocation[
        IVF_FINITE_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var best = IVF_FINITE_NONE
    var i = tid
    while i < blocks:
        var v = part.unsafe_load(i)
        if v < best:
            best = v
        i += IVF_FINITE_TPB
    red.unsafe_store(tid, best)
    barrier()
    var active = IVF_FINITE_TPB // 2
    while active > 0:
        if tid < active:
            var o = red.unsafe_load(tid + active)
            if o < red.unsafe_load(tid):
                red.unsafe_store(tid, o)
        barrier()
        active = active // 2
    if tid == 0:
        out.unsafe_store(0, red.unsafe_load(0))


def ivf_device_first_over_bound(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> Int:
    """The first flat index in `buf[0:n]` that `ivf_validate_data` would
    refuse, or -1. Two launches and one 4 B readback."""
    if n <= 0:
        return -1
    if n > Int(IVF_FINITE_NONE):
        raise Error("ivf_flat: finiteness scan length outside the Int32 index range")
    var blocks = (n + IVF_FINITE_TPB - 1) // IVF_FINITE_TPB
    if blocks > IVF_FINITE_BLOCKS:
        blocks = IVF_FINITE_BLOCKS
    var part = ctx.enqueue_create_buffer[DType.int32](blocks)
    var verdict = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_function[ivf_bound_partial_kernel](
        part.unsafe_ptr(),
        buf.unsafe_ptr(),
        Int32(n),
        bitcast[DType.uint32](IVF_MAGNITUDE_BOUND),
        grid_dim=(blocks, 1, 1),
        block_dim=(IVF_FINITE_TPB, 1, 1),
    )
    ctx.enqueue_function[ivf_bound_fold_kernel](
        verdict.unsafe_ptr(),
        part.unsafe_ptr(),
        Int32(blocks),
        grid_dim=(1, 1, 1),
        block_dim=(IVF_FINITE_TPB, 1, 1),
    )
    var host = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=verdict)
    # LOAD-BEARING: the host reads the verdict on the next line.
    ctx.synchronize()
    var best = host.unsafe_ptr().unsafe_load(0)
    _ = host^
    _ = verdict^
    _ = part^
    if best == IVF_FINITE_NONE:
        return -1
    return Int(best)


def ivf_validate_device(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    values: List[Float32],
    n_rows: Int,
    dim: Int,
    where: String,
) raises:
    """`ivf_validate_data` over words already uploaded to `buf`.

    The length check is the host one (no data walk). On a device hit the
    host walk runs ONLY to raise `ivf_validate_data`'s own message; the
    trailing raise is unreachable while the two predicates agree."""
    if len(values) != n_rows * dim:
        ivf_validate_data(values, n_rows, dim, where)
    var first = ivf_device_first_over_bound(ctx, buf, n_rows * dim)
    if first < 0:
        return
    ivf_validate_data(values, n_rows, dim, where)
    raise Error(
        "ivf_flat: "
        + where
        + " value at flat position "
        + String(first)
        + " is not finite or has magnitude at or above 2^63 (device scan)."
    )
