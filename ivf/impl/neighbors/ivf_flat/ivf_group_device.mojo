# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IVF search plan on the device (lane cgr4-download-loop, 2026-10-03).

The batched search used to download every query's probes, sort them on the
host, sum the probed lists' sizes per query on the host, and group the
(query, probe) pairs by list with a host counting sort before uploading the
groups for `identical_ivf_scan_grouped_kernel`. Every one of those steps is
here, on the device, in parallel:

  * `ivf_probe_counts_device`: the candidate count of each query (the kept
    rows of its probed lists under a filter, else their sizes), one thread
    per query over its n_probes lists; under a filter the kept rows per list
    are counted first, one thread per list slot, by integer atomics.
  * `ivf_group_pairs_device`: the pairs sorted by list id with a stable
    device radix sort (ascending pair index within a list, the host counting
    sort's order), the per-list offsets and the GQPB-pair blocks by device
    scans, each block's list found by a binary search over the block
    offsets.

Everything is integer and exact. The grouped kernel's result is a set
function of each (query, probe)'s candidates, and the merge folds a query's
partials by the total order (distance bits, original index), so the probe
numbering (the device's selection order, where the host numbered the probes
after sorting them) moves no bit.
"""

from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.device_fold import device_exclusive_scan_total
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len

comptime _TPB = 256
comptime _I32P = MutPointer[Int32, MutAnyOrigin]
comptime _U32P = MutPointer[UInt32, MutAnyOrigin]


def _grid(n: Int) -> Int:
    return max((n + _TPB - 1) // _TPB, 1)


def _upper_bound(a: _I32P, len_in: Int, v: Int) -> Int:
    """The first index i in [0, len) with a[i] > v (a ascending)."""
    var lo = 0
    var hi = len_in
    while lo < hi:
        var mid = (lo + hi) // 2
        if Int(a[mid]) <= v:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _kept_per_list_kernel(
    off: _I32P, ind: _U32P, keep: _I32P, n_lists_in: Int32, n_slots_in: Int32, kept: _I32P
):
    """Slot s: its list l (off[l] <= s < off[l + 1]) gains one kept row when
    the slot's original row passes the filter. Integer atomics: exact."""
    var s = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if s >= Int(n_slots_in):
        return
    if keep[Int(ind[s])] == 0:
        return
    var l = _upper_bound(off, Int(n_lists_in) + 1, s) - 1
    _ = Atomic.fetch_add(kept.unsafe_offset(l), Int32(1))


def _probe_counts_kernel(
    probe: _U32P, off: _I32P, kept: _I32P, filtered: Int32,
    n_queries_in: Int32, n_probes_in: Int32, counts: _I32P,
):
    """Query q: the sum over its probes' lists of the kept rows (filtered)
    or the list sizes."""
    var q = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if q >= Int(n_queries_in):
        return
    var n_probes = Int(n_probes_in)
    var c = 0
    for p in range(n_probes):
        var l = Int(probe[q * n_probes + p])
        if filtered != 0:
            c += Int(kept[l])
        else:
            c += Int(off[l + 1]) - Int(off[l])
    counts[q] = Int32(c)


def _pair_keys_kernel(probe: _U32P, n_in: Int32, keys: _U32P, vals: _U32P):
    var i = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if i < Int(n_in):
        keys[i] = probe[i]
        vals[i] = UInt32(i)


def _zero_kernel(buf: _I32P, n_in: Int32):
    var i = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if i < Int(n_in):
        buf[i] = Int32(0)


def _list_hist_kernel(probe: _U32P, n_in: Int32, gcount: _I32P):
    var i = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if i < Int(n_in):
        _ = Atomic.fetch_add(gcount.unsafe_offset(Int(probe[i])), Int32(1))


def _split_pairs_kernel(vals: _U32P, n_in: Int32, n_probes_in: Int32, gq: _I32P, gp: _I32P):
    var j = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if j < Int(n_in):
        var v = Int(vals[j])
        var n_probes = Int(n_probes_in)
        gq[j] = Int32(v // n_probes)
        gp[j] = Int32(v % n_probes)


def _list_blocks_kernel(goff: _I32P, n_lists_in: Int32, gqpb: Int32, nblk: _I32P):
    var l = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if l < Int(n_lists_in):
        var cnt = Int(goff[l + 1]) - Int(goff[l])
        nblk[l] = Int32((cnt + Int(gqpb) - 1) // Int(gqpb))


def _emit_blocks_kernel(
    boff: _I32P, goff: _I32P, n_lists_in: Int32, n_blocks_in: Int32, gqpb: Int32, bl: _I32P, bs: _I32P
):
    """Block b of the scan: its list l (boff[l] <= b < boff[l + 1]) and its
    first pair goff[l] + (b - boff[l]) * GQPB, the host loop's (l, j)."""
    var b = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if b >= Int(n_blocks_in):
        return
    var l = _upper_bound(boff, Int(n_lists_in) + 1, b) - 1
    bl[b] = Int32(l)
    bs[b] = Int32(Int(goff[l]) + (b - Int(boff[l])) * Int(gqpb))


def ivf_probe_counts_device(
    ctx: DeviceContext,
    mut probe: DeviceBuffer[DType.uint32],
    mut off: DeviceBuffer[DType.int32],
    mut ind: DeviceBuffer[DType.uint32],
    mut keep: DeviceBuffer[DType.int32],
    filtered: Bool,
    n_queries: Int,
    n_probes: Int,
    n_lists: Int,
    n_slots: Int,
) raises -> DeviceBuffer[DType.int32]:
    """The candidate count of every query, on the device."""
    var kept = ctx.enqueue_create_buffer[DType.int32](max(n_lists, 1))
    if filtered:
        ctx.enqueue_function[_zero_kernel](
            kept.unsafe_ptr(), Int32(n_lists), grid_dim=_grid(n_lists), block_dim=_TPB,
        )
        if n_slots > 0:
            ctx.enqueue_function[_kept_per_list_kernel](
                off.unsafe_ptr(), ind.unsafe_ptr(), keep.unsafe_ptr(), Int32(n_lists),
                Int32(n_slots), kept.unsafe_ptr(), grid_dim=_grid(n_slots), block_dim=_TPB,
            )
    var counts = ctx.enqueue_create_buffer[DType.int32](max(n_queries, 1))
    ctx.enqueue_function[_probe_counts_kernel](
        probe.unsafe_ptr(), off.unsafe_ptr(), kept.unsafe_ptr(), Int32(1) if filtered else Int32(0),
        Int32(n_queries), Int32(n_probes), counts.unsafe_ptr(),
        grid_dim=_grid(n_queries), block_dim=_TPB,
    )
    ctx.synchronize()
    _ = kept^
    return counts^


struct IvfPairGroups(Movable):
    """`identical_ivf_scan_grouped_kernel`'s grouping arrays, on the device."""

    var goff: DeviceBuffer[DType.int32]
    var gq: DeviceBuffer[DType.int32]
    var gp: DeviceBuffer[DType.int32]
    var bl: DeviceBuffer[DType.int32]
    var bs: DeviceBuffer[DType.int32]
    var n_blocks: Int

    def __init__(
        out self,
        var goff: DeviceBuffer[DType.int32],
        var gq: DeviceBuffer[DType.int32],
        var gp: DeviceBuffer[DType.int32],
        var bl: DeviceBuffer[DType.int32],
        var bs: DeviceBuffer[DType.int32],
        n_blocks: Int,
    ):
        self.goff = goff^
        self.gq = gq^
        self.gp = gp^
        self.bl = bl^
        self.bs = bs^
        self.n_blocks = n_blocks


def ivf_group_pairs_device(
    ctx: DeviceContext,
    mut probe: DeviceBuffer[DType.uint32],
    n_queries: Int,
    n_probes: Int,
    n_lists: Int,
    gqpb: Int,
) raises -> IvfPairGroups:
    """The (query, probe) pairs grouped by list, on the device: `gq`/`gp`
    the pairs sorted by list (stable, ascending pair index within a list),
    `goff` each list's first pair (n_lists + 1), `bl`/`bs` each scan block's
    list and first pair. Only the block count comes back to the host (the
    grid size)."""
    var n = n_queries * n_probes
    var keys = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
    var vals = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
    var tk = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
    var tv = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
    var cnt = ctx.enqueue_create_buffer[DType.int32](max(frs_counts_len(n), 1))
    var gq = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var gp = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var gcount = ctx.enqueue_create_buffer[DType.int32](n_lists + 1)
    var goff = ctx.enqueue_create_buffer[DType.int32](n_lists + 1)
    var nblk = ctx.enqueue_create_buffer[DType.int32](n_lists + 1)
    var boff = ctx.enqueue_create_buffer[DType.int32](n_lists + 1)
    ctx.enqueue_function[_pair_keys_kernel](
        probe.unsafe_ptr(), Int32(n), keys.unsafe_ptr(), vals.unsafe_ptr(),
        grid_dim=_grid(n), block_dim=_TPB,
    )
    fast_radix_sort_pairs_u32(ctx, n, keys, vals, tk, tv, cnt)
    ctx.enqueue_function[_split_pairs_kernel](
        vals.unsafe_ptr(), Int32(n), Int32(n_probes), gq.unsafe_ptr(), gp.unsafe_ptr(),
        grid_dim=_grid(n), block_dim=_TPB,
    )
    ctx.enqueue_function[_zero_kernel](
        gcount.unsafe_ptr(), Int32(n_lists + 1), grid_dim=_grid(n_lists + 1), block_dim=_TPB,
    )
    ctx.enqueue_function[_list_hist_kernel](
        probe.unsafe_ptr(), Int32(n), gcount.unsafe_ptr(), grid_dim=_grid(n), block_dim=_TPB,
    )
    device_exclusive_scan_total(ctx, gcount.unsafe_ptr(), goff, n_lists)
    ctx.enqueue_function[_list_blocks_kernel](
        goff.unsafe_ptr(), Int32(n_lists), Int32(gqpb), nblk.unsafe_ptr(),
        grid_dim=_grid(n_lists), block_dim=_TPB,
    )
    device_exclusive_scan_total(ctx, nblk.unsafe_ptr(), boff, n_lists)
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=boff.create_sub_buffer[DType.int32](n_lists, 1))
    ctx.synchronize()
    var n_blocks = Int(h.unsafe_ptr()[0])
    _ = h^
    var bl = ctx.enqueue_create_buffer[DType.int32](max(n_blocks, 1))
    var bs = ctx.enqueue_create_buffer[DType.int32](max(n_blocks, 1))
    if n_blocks > 0:
        ctx.enqueue_function[_emit_blocks_kernel](
            boff.unsafe_ptr(), goff.unsafe_ptr(), Int32(n_lists), Int32(n_blocks), Int32(gqpb),
            bl.unsafe_ptr(), bs.unsafe_ptr(), grid_dim=_grid(n_blocks), block_dim=_TPB,
        )
    ctx.synchronize()
    _ = keys^
    _ = vals^
    _ = tk^
    _ = tv^
    _ = cnt^
    _ = gcount^
    _ = nblk^
    _ = boff^
    return IvfPairGroups(goff^, gq^, gp^, bl^, bs^, n_blocks)
