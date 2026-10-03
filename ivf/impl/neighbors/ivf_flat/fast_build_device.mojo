# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-FLAT build, FAST on Apple: the host passes of `ivf_flat_build` done on
the device (lane/apple-fast-ann, 2026-10-02; x_ann/fast_env.mojo switches,
each read on the host at dispatch, default off).

`-D MOJOLEARN_IVF_FAST_DEVICE_TRAINSET=1` (`fast_trainset_device`,
`fast_trainset_scale`): the coarse quantizer's FAST training sample
(`IVF_FAST_TRAINSET`, 262,144 rows at 1,024 lists) gathered on the device
from the rows already uploaded, and the fixed-point scale of its centroid
accumulation from device column sums of |x| (float32 partials of COL_ROWS
rows, joined in Float64 on the host in order, then `choose_scale` on the
largest column with a 2^-10 margin: the snap to a power of two makes the
scale the same number unless the magnitude sits within that margin of a
boundary). Cause (`ivf_flat_build.mojo`): the sample was built on the host
one float at a time (57.7 M appends on Istella), summed by `plan_quantizer_scale`
on the host and uploaded a second time beside the whole dataset.

`-D MOJOLEARN_IVF_FAST_DEVICE_CSR=1` (`fast_list_layout_device`): the CSR
lists by a device histogram, scan and ranked scatter, the closure condition
`ivf/checks/list_layout.mojo` states for DEVIATION 1800: `csr_count_kernel`
(one threadgroup per CSR_ROWS rows: each row's rank among the earlier rows
of its block with the same label, and the block's count per list, both from
the staged labels, no atomics), `csr_prefix_kernel` (per list, the running
count over the blocks in block order, and the list size),
`csr_offsets_block_kernel` + `csr_offsets_add_kernel` (the exclusive scan
of the sizes in two levels: a block-local scan per FB_T lists writing block
totals, then each block adding the totals before it; at most CSR_OFF_BLOCKS
blocks, a comptime bound, so no one-block launch over n_lists),
`csr_scatter_kernel` (slot = offset of the list + the earlier blocks' count
+ the rank), `csr_gather_data_kernel` (the permuted vectors). The slot of
row i is its position among the rows of its label in ascending i, which is
the slot `build_list_layout`'s ascending host pass gives it: the same
offsets, the same carried ids, the same vectors, so the same bits. Cause:
`build_list_layout` is three host passes over the rows, the third moving
n x dim floats (352 MB on Istella) one row at a time."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import memcpy, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.fixed_point import choose_scale
from x_ann.io import download_f32, download_i32, upload_i32

comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime U32P = MutPointer[UInt32, MutAnyOrigin]

#: threads per threadgroup
comptime FB_T = 256
#: rows per column-sum partial
comptime COL_ROWS = 1024
#: rows per CSR block, and the most lists the one-threadgroup scan takes
comptime CSR_ROWS = 1024
comptime CSR_LISTS_MAX = 4096


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + FB_T - 1) // FB_T


def gather_rows_kernel(count: Int32, dim: Int32, x: F32P, rows: I32P, dst: F32P):
    """dst[s, c] = x[rows[s], c], one thread per cell."""
    var e = _tid()
    if e < Int(count):
        var d = Int(dim)
        var s = e // d
        var c = e % d
        dst.unsafe_store(e, x.unsafe_load(Int(rows.unsafe_load(s)) * d + c))


def colsum_part_kernel(count: Int32, n: Int32, dim: Int32, x: F32P, part: F32P):
    """part[g, c] = sum of |x[i, c]| over rows g COL_ROWS .. g COL_ROWS +
    COL_ROWS - 1, in row order (float32)."""
    var e = _tid()
    if e < Int(count):
        var d = Int(dim)
        var g = e // d
        var c = e % d
        var i0 = g * COL_ROWS
        var i1 = i0 + COL_ROWS
        if i1 > Int(n):
            i1 = Int(n)
        var acc = Float32(0.0)
        for i in range(i0, i1):
            acc = acc + abs(x.unsafe_load(i * d + c))
        part.unsafe_store(e, acc)


def csr_count_kernel(n: Int32, n_lists: Int32, labels: U32P, ranks: I32P, bcount: I32P):
    """Threadgroup b, rows i0 = b CSR_ROWS .. i0 + CSR_ROWS - 1 (labels
    staged): ranks[i] = the number of earlier rows of the block with row i's
    label; bcount[b, l] = the block's rows with label l."""
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var i0 = b * CSR_ROWS
    var cnt = CSR_ROWS if Int(n) - i0 > CSR_ROWS else Int(n) - i0
    var nl = Int(n_lists)
    var lab = stack_allocation[CSR_ROWS, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    for r in range(t, CSR_ROWS, FB_T):
        var v = Int32(-1)
        if r < cnt:
            v = Int32(Int(labels.unsafe_load(i0 + r)))
        lab[r] = v
    barrier()
    for r in range(t, cnt, FB_T):
        var l = lab[r]
        var k = 0
        for p in range(r):
            if lab[p] == l:
                k += 1
        ranks.unsafe_store(i0 + r, Int32(k))
    for l in range(t, nl, FB_T):
        var k = 0
        for r in range(cnt):
            if Int(lab[r]) == l:
                k += 1
        bcount.unsafe_store(b * nl + l, Int32(k))


def csr_prefix_kernel(nb: Int32, n_lists: Int32, bcount: I32P, bprefix: I32P, sizes: I32P):
    """List l (one thread): bprefix[b, l] = its rows in blocks before b;
    sizes[l] = its rows in all."""
    var l = _tid()
    if l < Int(n_lists):
        var nl = Int(n_lists)
        var run = Int32(0)
        for b in range(Int(nb)):
            bprefix.unsafe_store(b * nl + l, run)
            run = run + bcount.unsafe_load(b * nl + l)
        sizes.unsafe_store(l, run)


#: blocks of the two-level offsets scan: n_lists <= CSR_LISTS_MAX, FB_T lists
#: per block, so the block totals are at most this many (a comptime bound)
comptime CSR_OFF_BLOCKS = CSR_LISTS_MAX // FB_T


def csr_offsets_block_kernel(n_lists: Int32, sizes: I32P, offsets: I32P, btot: I32P):
    """Level 1 of the offsets scan. Block b, list l = b FB_T + t:
    offsets[l] = the exclusive scan of the block's sizes (Hillis-Steele in
    threadgroup memory), btot[b] = the block's total. ceil(n_lists / FB_T)
    blocks (<= CSR_OFF_BLOCKS)."""
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var nl = Int(n_lists)
    var l = b * FB_T + t
    var sc = stack_allocation[FB_T, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var v0 = Int32(0)
    if l < nl:
        v0 = sizes.unsafe_load(l)
    sc[t] = v0
    barrier()
    var step = 1
    while step < FB_T:
        var v = Int32(0)
        if t >= step:
            v = sc[t - step]
        barrier()
        sc[t] = sc[t] + v
        barrier()
        step *= 2
    if l < nl:
        offsets.unsafe_store(l, sc[t] - v0)
    if t == 0:
        btot.unsafe_store(b, sc[FB_T - 1])


def csr_offsets_add_kernel(n_lists: Int32, sizes: I32P, btot: I32P, offsets: I32P):
    """Level 2, the same grid: block b adds the totals of the blocks before
    it (CSR_OFF_BLOCKS candidates, a comptime bound) to its lists' offsets;
    the thread of the last list writes offsets[n_lists] = its offset + its
    size (the total). The same integer sums as one serial scan: the same
    offsets."""
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var nl = Int(n_lists)
    var l = b * FB_T + t
    var carry = Int32(0)
    comptime for q in range(CSR_OFF_BLOCKS):
        if q < b:
            carry = carry + btot.unsafe_load(q)
    if l < nl:
        var o = offsets.unsafe_load(l) + carry
        offsets.unsafe_store(l, o)
        if l == nl - 1:
            offsets.unsafe_store(nl, o + sizes.unsafe_load(l))
    if nl == 0 and b == 0 and t == 0:
        offsets.unsafe_store(0, Int32(0))


def csr_scatter_kernel(
    n: Int32, n_lists: Int32, labels: U32P, ranks: I32P, bprefix: I32P, offsets: I32P, list_indices: U32P
):
    """list_indices[offsets[l] + bprefix[b, l] + ranks[i]] = i for row i of
    block b with label l: the carry, ascending in i within every list."""
    var i = _tid()
    if i < Int(n):
        var nl = Int(n_lists)
        var l = Int(labels.unsafe_load(i))
        if l < 0 or l >= nl:
            return
        var b = i // CSR_ROWS
        var slot = Int(offsets.unsafe_load(l)) + Int(bprefix.unsafe_load(b * nl + l)) + Int(ranks.unsafe_load(i))
        list_indices.unsafe_store(slot, UInt32(i))


def csr_gather_data_kernel(count: Int32, dim: Int32, x: F32P, list_indices: U32P, dst: F32P):
    """dst[slot, c] = x[list_indices[slot], c]: the permuted vectors."""
    var e = _tid()
    if e < Int(count):
        var d = Int(dim)
        var slot = e // d
        var c = e % d
        dst.unsafe_store(e, x.unsafe_load(Int(list_indices.unsafe_load(slot)) * d + c))


def _download_u32(ctx: DeviceContext, buf: DeviceBuffer[DType.uint32], n: Int) raises -> List[UInt32]:
    var h = ctx.enqueue_create_host_buffer[DType.uint32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[UInt32](length=n, fill=UInt32(0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=h.unsafe_ptr(), count=n)
    _ = h^
    return out^


def fast_trainset_device(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], rows: List[Int], n_train: Int, dim: Int,
) raises -> DeviceBuffer[DType.float32]:
    """The n_train x dim sample of the uploaded rows, gathered on the device
    (`rows` ascending, `ivf_trainset_rows`). Drained before the return."""
    var ids = List[Int32](capacity=n_train)
    for s in range(n_train):
        ids.append(Int32(rows[s]))
    var dids = upload_i32(ctx, ids)
    var dxt = ctx.enqueue_create_buffer[DType.float32](n_train * dim)
    ctx.enqueue_function[gather_rows_kernel](
        Int32(n_train * dim), Int32(dim), dx.unsafe_ptr(), dids.unsafe_ptr(), dxt.unsafe_ptr(),
        grid_dim=_grid(n_train * dim), block_dim=FB_T,
    )
    ctx.synchronize()
    _ = dids^
    return dxt^


def fast_trainset_scale(
    ctx: DeviceContext, mut dxt: DeviceBuffer[DType.float32], n_train: Int, dim: Int,
) raises -> Float64:
    """`plan_quantizer_scale` of the device sample: float32 column partials
    over COL_ROWS rows, joined in Float64 on the host in row order, the
    largest column with a 2^-10 margin into `choose_scale`."""
    var parts = (n_train + COL_ROWS - 1) // COL_ROWS
    var dpart = ctx.enqueue_create_buffer[DType.float32](parts * dim)
    ctx.enqueue_function[colsum_part_kernel](
        Int32(parts * dim), Int32(n_train), Int32(dim), dxt.unsafe_ptr(), dpart.unsafe_ptr(),
        grid_dim=_grid(parts * dim), block_dim=FB_T,
    )
    var part = download_f32(ctx, dpart, parts * dim)
    var largest = Float64(0.0)
    for c in range(dim):
        var column = Float64(0.0)
        for g in range(parts):
            column += Float64(part[g * dim + c])
        if column > largest:
            largest = column
    _ = dpart^
    return choose_scale(largest * (1.0 + 1.0 / 1024.0), n_train)


def fast_list_layout_device(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], mut labels: DeviceBuffer[DType.uint32],
    n_rows: Int, dim: Int, n_lists: Int, with_data: Bool,
    mut offsets: List[Int32], mut list_indices: List[UInt32], mut list_data: List[Float32],
) raises:
    """`build_list_layout`'s three lists from the device labels: the CSR
    offsets (n_lists + 1), the carried ids (n_rows) and, under `with_data`,
    the permuted vectors (n_rows x dim; empty otherwise). The caller checks
    n_lists <= CSR_LISTS_MAX."""
    var nb = (n_rows + CSR_ROWS - 1) // CSR_ROWS
    var dranks = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var dbcount = ctx.enqueue_create_buffer[DType.int32](nb * n_lists)
    var dbprefix = ctx.enqueue_create_buffer[DType.int32](nb * n_lists)
    var dsizes = ctx.enqueue_create_buffer[DType.int32](n_lists)
    var doff = ctx.enqueue_create_buffer[DType.int32](n_lists + 1)
    var dbtot = ctx.enqueue_create_buffer[DType.int32](CSR_OFF_BLOCKS)
    var dind = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var ddata = ctx.enqueue_create_buffer[DType.float32]((n_rows * dim) if with_data else 1)
    ctx.enqueue_function[csr_count_kernel](
        Int32(n_rows), Int32(n_lists), labels.unsafe_ptr(), dranks.unsafe_ptr(), dbcount.unsafe_ptr(),
        grid_dim=nb, block_dim=FB_T,
    )
    ctx.enqueue_function[csr_prefix_kernel](
        Int32(nb), Int32(n_lists), dbcount.unsafe_ptr(), dbprefix.unsafe_ptr(), dsizes.unsafe_ptr(),
        grid_dim=_grid(n_lists), block_dim=FB_T,
    )
    # the offsets in two launches over ceil(n_lists / FB_T) blocks (never one
    # block over n_lists: the block totals are CSR_OFF_BLOCKS at most)
    var nob = (n_lists + FB_T - 1) // FB_T
    ctx.enqueue_function[csr_offsets_block_kernel](
        Int32(n_lists), dsizes.unsafe_ptr(), doff.unsafe_ptr(), dbtot.unsafe_ptr(), grid_dim=nob, block_dim=FB_T,
    )
    ctx.enqueue_function[csr_offsets_add_kernel](
        Int32(n_lists), dsizes.unsafe_ptr(), dbtot.unsafe_ptr(), doff.unsafe_ptr(), grid_dim=nob, block_dim=FB_T,
    )
    ctx.enqueue_function[csr_scatter_kernel](
        Int32(n_rows), Int32(n_lists), labels.unsafe_ptr(), dranks.unsafe_ptr(), dbprefix.unsafe_ptr(),
        doff.unsafe_ptr(), dind.unsafe_ptr(), grid_dim=_grid(n_rows), block_dim=FB_T,
    )
    if with_data:
        ctx.enqueue_function[csr_gather_data_kernel](
            Int32(n_rows * dim), Int32(dim), dx.unsafe_ptr(), dind.unsafe_ptr(), ddata.unsafe_ptr(),
            grid_dim=_grid(n_rows * dim), block_dim=FB_T,
        )
    ctx.synchronize()
    offsets = download_i32(ctx, doff, n_lists + 1)
    list_indices = _download_u32(ctx, dind, n_rows)
    if with_data:
        list_data = download_f32(ctx, ddata, n_rows * dim)
    else:
        list_data = List[Float32]()
    _ = ddata^
    _ = dind^
    _ = dbtot^
    _ = doff^
    _ = dsizes^
    _ = dbprefix^
    _ = dbcount^
    _ = dranks^
