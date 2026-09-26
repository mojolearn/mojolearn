# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple: the kNN connectivity graph and its Laplacian built on the
device, from the kNN indices to the negated Laplacian the Lanczos reads.

The reference sequence (`spectral_embedding.mojo`'s docstring) reads the kNN
back, builds a host COO, uploads it for `coo_symmetrize`, downloads `2 *
nnz` slots, sorts them on the host, compacts, inserts the missing diagonal,
sorts AGAIN and uploads the result: at 30,000 rows and the default `n / 10`
neighbors that is 180 M entries through host lists four times (60 of 77 s).

Here the same matrix is assembled where it lives, from the kNN indices
left on the device (`knn_self_search_device_indices`: the same search,
no readback, no host sort). Symmetrized, row `r` holds column `c` when `c`
is among r's neighbors (A) or r is among c's (T): value `0.5 * (1 + 1) = 1`
in both, `0.5 * (1 + 0) = 0.5` in one; and `(r, r, 0.0)` where the
Laplacian inserts a missing diagonal.
  n <= FG_BM_MAX_N (`_bitmap_graph`, no sorting at all): T by an in-degree
     count, a scan and an atomic scatter; then per row two threadgroup
     bitmaps (A, T) give the row length (count pass, then one scan) and the
     entries in ascending column order (write pass, popcount prefix).
  larger n: each row's columns sorted (bitonic, threadgroup memory), `r`
     binary-searched in row `c` for mutuality, the owed transposes
     scattered and sorted per row, each row written by merge ranks.
The result is the row-sorted COO + `indptr` `compute_graph_laplacian`
uploads, entry for entry, so the Laplacian and everything after it read
the same bytes as the host path.

Taken only when `k` fits one threadgroup sort (`FG_MAX_K`), the counts fit
Int32, and no identity trace is being written (the trace records the host
COO stages). `-D MOJOLEARN_SPECTRAL_GRAPH_FAST_OFF` keeps the host path.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import prefix_sum as _warp_prefix_sum
from std.bit import pop_count
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from neighbors.estimator import knn_self_search_device_indices
from spectral.impl.sparse.linalg.detail.laplacian import (
    DeviceCoo,
    laplacian_from_sorted_device,
)

comptime SPECTRAL_GRAPH_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SPECTRAL_GRAPH_FAST_OFF"]()
)
comptime FG_MAX_K = 4096
comptime FG_SORT_TPB = 1024
comptime FG_TPB = 256
comptime FG_SCAN_TPB = 1024
#: rows up to this many columns are assembled from two threadgroup bitmaps
#: (the row's own neighbors, and the rows that list it) instead of sorts
comptime FG_BM_WORDS = 4000
comptime FG_BM_MAX_N = FG_BM_WORDS * 32


def fast_graph_eligible(n: Int, k: Int) -> Bool:
    comptime if not SPECTRAL_GRAPH_FAST:
        return False
    if n <= 1 or k < 1 or k > FG_MAX_K or k > n:
        return False
    # every count below is Int32: 2 n k + n entries at most
    return 2 * n * k + n < (1 << 31) - 1


def fg_sort_rows_kernel(
    acols: MutPointer[UInt32, MutAnyOrigin],
    has_self: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    k_in: Int32,
    kp_in: Int32,
):
    """Row `r`'s k neighbor columns sorted ascending in place; `has_self[r]`
    set when `r` is among them; `flags[0]` set on a repeated column."""
    var row = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var k = Int(k_in)
    var kp = Int(kp_in)
    var base = row * k
    var sh = stack_allocation[
        FG_MAX_K, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var i = t
    while i < kp:
        sh[i] = acols[base + i] if i < k else UInt32(0xFFFFFFFF)
        i += FG_SORT_TPB
    barrier()
    var size = 2
    while size <= kp:
        var stride = size // 2
        while stride > 0:
            var e = t
            while e < kp // 2:
                var lo = 2 * e - (e & (stride - 1))
                var hi = lo + stride
                var up = (lo & size) == 0
                var a = sh[lo]
                var b = sh[hi]
                if (a > b) == up:
                    sh[lo] = b
                    sh[hi] = a
                e += FG_SORT_TPB
            barrier()
            stride //= 2
        size *= 2
    i = t
    while i < k:
        var v = sh[i]
        acols[base + i] = v
        if v == UInt32(row):
            has_self[row] = 1
        if i + 1 < k and sh[i + 1] == v:
            flags[0] = 1
        i += FG_SORT_TPB


def _lower_bound(
    p: MutPointer[UInt32, MutAnyOrigin], start: Int, length: Int, target: UInt32
) -> Int:
    var lo = 0
    var hi = length
    while lo < hi:
        var mid = (lo + hi) // 2
        if p[start + mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def fg_mutual_kernel(
    acols: MutPointer[UInt32, MutAnyOrigin],
    mutual: MutPointer[UInt8, MutAnyOrigin],
    ecnt: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """Entry `(r, c)`: is `r` in row `c`? If not, row `c` is owed `(c, r)`."""
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var k = Int(k_in)
    if e >= Int(n_in) * k:
        return
    var r = e // k
    var c = Int(acols[e])
    var lb = _lower_bound(acols, c * k, k, UInt32(r))
    if lb < k and acols[c * k + lb] == UInt32(r):
        mutual[e] = 1
    else:
        mutual[e] = 0
        _ = Atomic.fetch_add(ecnt.unsafe_offset(c), Int32(1))


def fg_scan_kernel(
    ecnt: MutPointer[Int32, MutAnyOrigin],
    has_self: MutPointer[Int32, MutAnyOrigin],
    indptr: MutPointer[Int32, MutAnyOrigin],
    eoff: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """One threadgroup: `indptr` = exclusive scan of `k + ecnt + !has_self`,
    `eoff` = exclusive scan of `ecnt`, both with the total at `[n]`."""
    var t = Int(thread_idx.x)
    var n = Int(n_in)
    var k = Int(k_in)
    var chunk = (n + FG_SCAN_TPB - 1) // FG_SCAN_TPB
    var lo = t * chunk
    var hi = lo + chunk
    if hi > n:
        hi = n
    var s_len = stack_allocation[
        FG_SCAN_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var s_e = stack_allocation[
        FG_SCAN_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var a_len = Int32(0)
    var a_e = Int32(0)
    var r = lo
    while r < hi:
        var ec = ecnt[r]
        a_e += ec
        a_len += Int32(k) + ec + (Int32(1) - has_self[r])
        r += 1
    s_len[t] = a_len
    s_e[t] = a_e
    barrier()
    var off = 1
    while off < FG_SCAN_TPB:
        var v_len = s_len[t]
        var v_e = s_e[t]
        if t >= off:
            v_len += s_len[t - off]
            v_e += s_e[t - off]
        barrier()
        s_len[t] = v_len
        s_e[t] = v_e
        barrier()
        off *= 2
    var p_len = s_len[t] - a_len
    var p_e = s_e[t] - a_e
    r = lo
    while r < hi:
        indptr[r] = p_len
        eoff[r] = p_e
        var ec = ecnt[r]
        p_e += ec
        p_len += Int32(k) + ec + (Int32(1) - has_self[r])
        r += 1
    if t == FG_SCAN_TPB - 1:
        indptr[n] = s_len[t]
        eoff[n] = s_e[t]


def fg_scatter_kernel(
    acols: MutPointer[UInt32, MutAnyOrigin],
    mutual: MutPointer[UInt8, MutAnyOrigin],
    eoff: MutPointer[Int32, MutAnyOrigin],
    ecur: MutPointer[Int32, MutAnyOrigin],
    etmp: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """The owed `(c, r)` of every non-mutual entry into row `c`'s slots."""
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var k = Int(k_in)
    if e >= Int(n_in) * k:
        return
    if mutual[e] != 0:
        return
    var c = Int(acols[e])
    var p = Int(eoff[c]) + Int(Atomic.fetch_add(ecur.unsafe_offset(c), Int32(1)))
    etmp[p] = UInt32(e // k)


def fg_sort_extras_kernel(
    eoff: MutPointer[Int32, MutAnyOrigin],
    etmp: MutPointer[UInt32, MutAnyOrigin],
    esorted: MutPointer[UInt32, MutAnyOrigin],
):
    """Row `r`'s owed columns (distinct) sorted into `esorted`: bitonic in
    threadgroup memory, or by rank counting for a row longer than that."""
    var row = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var base = Int(eoff[row])
    var ne = Int(eoff[row + 1]) - base
    if ne == 0:
        return
    if ne > FG_MAX_K:
        var i = t
        while i < ne:
            var v = etmp[base + i]
            var rank = 0
            for j in range(ne):
                if etmp[base + j] < v:
                    rank += 1
            esorted[base + rank] = v
            i += FG_SORT_TPB
        return
    var kp = 1
    while kp < ne:
        kp *= 2
    var sh = stack_allocation[
        FG_MAX_K, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var i = t
    while i < kp:
        sh[i] = etmp[base + i] if i < ne else UInt32(0xFFFFFFFF)
        i += FG_SORT_TPB
    barrier()
    var size = 2
    while size <= kp:
        var stride = size // 2
        while stride > 0:
            var e = t
            while e < kp // 2:
                var lo = 2 * e - (e & (stride - 1))
                var hi = lo + stride
                var up = (lo & size) == 0
                var a = sh[lo]
                var b = sh[hi]
                if (a > b) == up:
                    sh[lo] = b
                    sh[hi] = a
                e += FG_SORT_TPB
            barrier()
            stride //= 2
        size *= 2
    i = t
    while i < ne:
        esorted[base + i] = sh[i]
        i += FG_SORT_TPB


def fg_place_kernel(
    acols: MutPointer[UInt32, MutAnyOrigin],
    mutual: MutPointer[UInt8, MutAnyOrigin],
    has_self: MutPointer[Int32, MutAnyOrigin],
    eoff: MutPointer[Int32, MutAnyOrigin],
    esorted: MutPointer[UInt32, MutAnyOrigin],
    indptr: MutPointer[Int32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    k_in: Int32,
):
    """Row `r` in ascending column order: its own k entries (1 or 0.5), the
    owed transposes (0.5) and, when it has no self entry, `(r, r, 0.0)`."""
    var row = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var k = Int(k_in)
    var abase = row * k
    var ebase = Int(eoff[row])
    var ne = Int(eoff[row + 1]) - ebase
    var out = Int(indptr[row])
    var need_diag = has_self[row] == 0
    var ur = UInt32(row)
    var i = t
    while i < k:
        var a = acols[abase + i]
        var pos = out + i + _lower_bound(esorted, ebase, ne, a)
        if need_diag and ur < a:
            pos += 1
        rows[pos] = Int32(row)
        cols[pos] = Int32(Int(a))
        vals[pos] = Float32(1.0) if mutual[abase + i] != 0 else Float32(0.5)
        i += FG_TPB
    var j = t
    while j < ne:
        var b = esorted[ebase + j]
        var pos = out + j + _lower_bound(acols, abase, k, b)
        if need_diag and ur < b:
            pos += 1
        rows[pos] = Int32(row)
        cols[pos] = Int32(Int(b))
        vals[pos] = Float32(0.5)
        j += FG_TPB
    if t == 0 and need_diag:
        var pos = out + _lower_bound(acols, abase, k, ur) + _lower_bound(esorted, ebase, ne, ur)
        rows[pos] = Int32(row)
        cols[pos] = Int32(row)
        vals[pos] = Float32(0.0)


def fg_tcount_kernel(
    acols: MutPointer[UInt32, MutAnyOrigin],
    tcnt: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """In-degree: row `c` of the transpose gets one entry per `(r, c)`."""
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if e >= Int(n_in) * Int(k_in):
        return
    _ = Atomic.fetch_add(tcnt.unsafe_offset(Int(acols[e])), Int32(1))


def fg_exscan_kernel(
    inp: MutPointer[Int32, MutAnyOrigin],
    outp: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """One threadgroup: `outp` = exclusive scan of `inp`, total at `[n]`."""
    var t = Int(thread_idx.x)
    var n = Int(n_in)
    var chunk = (n + FG_SCAN_TPB - 1) // FG_SCAN_TPB
    var lo = t * chunk
    var hi = lo + chunk
    if hi > n:
        hi = n
    var sh = stack_allocation[
        FG_SCAN_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var a = Int32(0)
    var r = lo
    while r < hi:
        a += inp[r]
        r += 1
    sh[t] = a
    barrier()
    var off = 1
    while off < FG_SCAN_TPB:
        var v = sh[t]
        if t >= off:
            v += sh[t - off]
        barrier()
        sh[t] = v
        barrier()
        off *= 2
    var p = sh[t] - a
    r = lo
    while r < hi:
        outp[r] = p
        p += inp[r]
        r += 1
    if t == FG_SCAN_TPB - 1:
        outp[n] = sh[t]


def fg_tscatter_kernel(
    acols: MutPointer[UInt32, MutAnyOrigin],
    toff: MutPointer[Int32, MutAnyOrigin],
    tcur: MutPointer[Int32, MutAnyOrigin],
    tlist: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """`r` into row `c` of the transpose for every `(r, c)` (unordered)."""
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var k = Int(k_in)
    if e >= Int(n_in) * k:
        return
    var c = Int(acols[e])
    var p = Int(toff[c]) + Int(Atomic.fetch_add(tcur.unsafe_offset(c), Int32(1)))
    tlist[p] = UInt32(e // k)


def fg_bm_row_kernel[write: Bool](
    acols: MutPointer[UInt32, MutAnyOrigin],
    toff: MutPointer[Int32, MutAnyOrigin],
    tlist: MutPointer[UInt32, MutAnyOrigin],
    lens: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """Row `r` of the symmetrized graph from two bitmaps: `A` (its own
    neighbors) and `T` (the rows listing it). Column `c` is present when
    in `A` or `T` (value 1 in both, 0.5 in one) and at `c = r` always (the
    Laplacian's inserted `0.0` when in neither). Without `write`, the row
    length goes to `lens[r]`; with it, the entries go to `lens[r]`
    (= `indptr[r]`) onward in ascending column order."""
    var row = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var n = Int(n_in)
    var k = Int(k_in)
    var nw = (n + 31) // 32
    var bma = stack_allocation[
        FG_BM_WORDS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var bmt = stack_allocation[
        FG_BM_WORDS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        FG_SORT_TPB // 32, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var w = t
    while w < nw:
        bma[w] = 0
        bmt[w] = 0
        w += FG_SORT_TPB
    barrier()
    var i = t
    while i < k:
        var c = Int(acols[row * k + i])
        var bit = bitcast[DType.int32](UInt32(1) << UInt32(c & 31))
        var old = Atomic.fetch_add(bma.unsafe_offset(c >> 5), bit)
        comptime if not write:
            if (old & bit) != 0:
                flags[0] = 1
        i += FG_SORT_TPB
    var tb = Int(toff[row])
    var te = Int(toff[row + 1])
    var j = tb + t
    while j < te:
        var c = Int(tlist[j])
        var bit = bitcast[DType.int32](UInt32(1) << UInt32(c & 31))
        _ = Atomic.fetch_add(bmt.unsafe_offset(c >> 5), bit)
        j += FG_SORT_TPB
    barrier()
    var chunk = (nw + FG_SORT_TPB - 1) // FG_SORT_TPB
    var w0 = t * chunk
    var w1 = w0 + chunk
    if w1 > nw:
        w1 = nw
    var dw = row >> 5
    var dbit = UInt32(1) << UInt32(row & 31)
    var cnt = Int32(0)
    w = w0
    while w < w1:
        var u = bitcast[DType.uint32](bma[w]) | bitcast[DType.uint32](bmt[w])
        if w == dw:
            u |= dbit
        cnt += Int32(pop_count(u))
        w += 1
    # two-level scan: within each simdgroup, then over the 32 group totals
    var incl = _warp_prefix_sum(cnt)
    var lane = t % 32
    var wid = t // 32
    if lane == 31:
        sc[wid] = incl
    barrier()
    if wid == 0:
        var tot = sc[lane]
        var winc = _warp_prefix_sum(tot)
        sc[lane] = winc - tot
    barrier()
    var before = sc[wid] + incl - cnt
    comptime if not write:
        if t == FG_SORT_TPB - 1:
            lens[row] = before + cnt
    else:
        var pos = Int(lens[row]) + Int(before)
        w = w0
        while w < w1:
            var ua = bitcast[DType.uint32](bma[w])
            var ut = bitcast[DType.uint32](bmt[w])
            var u = ua | ut
            if w == dw:
                u |= dbit
            while u != 0:
                var low = u & (~u + 1)
                var b = Int(pop_count(low - 1))
                var v = Float32(0.0)
                if (ua & low) != 0 and (ut & low) != 0:
                    v = Float32(1.0)
                elif ((ua | ut) & low) != 0:
                    v = Float32(0.5)
                rows[pos] = Int32(row)
                cols[pos] = Int32(w * 32 + b)
                vals[pos] = v
                pos += 1
                u ^= low
            w += 1


def _bitmap_graph(
    ctx: DeviceContext,
    var acols: DeviceBuffer[DType.uint32],
    n: Int,
    k: Int,
    tpb: Int,
) raises -> DeviceCoo:
    """`fast_knn_graph` for `n <= FG_BM_MAX_N`: no sorts at all."""
    var nk = n * k
    var small = ctx.enqueue_create_buffer[DType.int32](3 * n + 8)
    var tcnt = small.create_sub_buffer[DType.int32](0, n)
    var tcur = small.create_sub_buffer[DType.int32](n, n)
    var lens = small.create_sub_buffer[DType.int32](2 * n, n)
    var flags = small.create_sub_buffer[DType.int32](3 * n, 4)
    var offs = ctx.enqueue_create_buffer[DType.int32](2 * n + 2)
    var toff = offs.create_sub_buffer[DType.int32](0, n + 1)
    var indptr_v = offs.create_sub_buffer[DType.int32](n + 1, n + 1)
    var tlist = ctx.enqueue_create_buffer[DType.uint32](nk)
    var dummy = ctx.enqueue_create_buffer[DType.int32](2)
    var dummy_f = ctx.enqueue_create_buffer[DType.float32](1)
    var dummy_a = dummy.create_sub_buffer[DType.int32](0, 1)
    var dummy_b = dummy.create_sub_buffer[DType.int32](1, 1)
    ctx.enqueue_memset(small, Int32(0))
    var grid_e = (nk + FG_TPB - 1) // FG_TPB
    ctx.enqueue_function[fg_tcount_kernel](
        acols.unsafe_ptr(), tcnt.unsafe_ptr(), Int32(n), Int32(k),
        grid_dim=(grid_e, 1, 1), block_dim=(FG_TPB, 1, 1),
    )
    ctx.enqueue_function[fg_exscan_kernel](
        tcnt.unsafe_ptr(), toff.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(FG_SCAN_TPB, 1, 1),
    )
    ctx.enqueue_function[fg_tscatter_kernel](
        acols.unsafe_ptr(), toff.unsafe_ptr(), tcur.unsafe_ptr(),
        tlist.unsafe_ptr(), Int32(n), Int32(k),
        grid_dim=(grid_e, 1, 1), block_dim=(FG_TPB, 1, 1),
    )
    ctx.enqueue_function[fg_bm_row_kernel[False]](
        acols.unsafe_ptr(), toff.unsafe_ptr(), tlist.unsafe_ptr(),
        lens.unsafe_ptr(), flags.unsafe_ptr(), dummy_a.unsafe_ptr(),
        dummy_b.unsafe_ptr(), dummy_f.unsafe_ptr(), Int32(n), Int32(k),
        grid_dim=(n, 1, 1), block_dim=(FG_SORT_TPB, 1, 1),
    )
    ctx.enqueue_function[fg_exscan_kernel](
        lens.unsafe_ptr(), indptr_v.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(FG_SCAN_TPB, 1, 1),
    )
    var tot = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=tot.unsafe_ptr(), src_buf=flags.create_sub_buffer[DType.int32](0, 1))
    ctx.enqueue_copy(dst_ptr=tot.unsafe_ptr() + 1, src_buf=indptr_v.create_sub_buffer[DType.int32](n, 1))
    ctx.synchronize()
    if tot.unsafe_ptr()[0] != 0:
        raise Error(
            "connectivity_graph: a kNN row holds a repeated neighbor --"
            " refused by name (DEVIATION 775)"
        )
    var nnz = Int(tot.unsafe_ptr()[1])
    _ = tot^
    var rows = ctx.enqueue_create_buffer[DType.int32](nnz)
    var cols = ctx.enqueue_create_buffer[DType.int32](nnz)
    var vals = ctx.enqueue_create_buffer[DType.float32](nnz)
    var indptr = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.enqueue_copy(dst_buf=indptr, src_buf=indptr_v)
    ctx.enqueue_function[fg_bm_row_kernel[True]](
        acols.unsafe_ptr(), toff.unsafe_ptr(), tlist.unsafe_ptr(),
        indptr.unsafe_ptr(), flags.unsafe_ptr(), rows.unsafe_ptr(),
        cols.unsafe_ptr(), vals.unsafe_ptr(), Int32(n), Int32(k),
        grid_dim=(n, 1, 1), block_dim=(FG_SORT_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = acols^
    _ = tlist^
    _ = small^
    _ = offs^
    _ = dummy^
    _ = dummy_f^
    return laplacian_from_sorted_device(ctx, n, nnz, rows^, cols^, vals^, indptr^, tpb)


def fast_knn_graph(
    ctx: DeviceContext,
    dataset: List[Float32],
    n: Int,
    n_features: Int,
    k: Int,
    tpb: Int,
) raises -> DeviceCoo:
    """The kNN search, symmetrize (`0.5 * (a + b)`), sort, zero removal and
    diagonal insertion of the host path, on the device, then
    `compute_graph_laplacian`'s device half: returns `D - A`. The caller
    checked `fast_graph_eligible(n, k)` and the dataset."""
    var h_data = ctx.enqueue_create_host_buffer[DType.float32](n * n_features)
    ctx.synchronize()
    var src = dataset.unsafe_ptr()
    for i in range(n * n_features):
        h_data.unsafe_ptr().unsafe_store(i, src.unsafe_load(i))
    var acols = knn_self_search_device_indices(ctx, h_data.unsafe_ptr(), n, n_features, k)
    _ = h_data^
    if n <= FG_BM_MAX_N:
        return _bitmap_graph(ctx, acols^, n, k, tpb)
    var kp = 2
    while kp < k:
        kp *= 2
    var nk = n * k
    var small = ctx.enqueue_create_buffer[DType.int32](4 * n + 4)
    var has_self = small.create_sub_buffer[DType.int32](0, n)
    var ecnt = small.create_sub_buffer[DType.int32](n, n)
    var ecur = small.create_sub_buffer[DType.int32](2 * n, n)
    var flags = small.create_sub_buffer[DType.int32](3 * n, 4)
    var offs = ctx.enqueue_create_buffer[DType.int32](2 * n + 2)
    var indptr_v = offs.create_sub_buffer[DType.int32](0, n + 1)
    var eoff = offs.create_sub_buffer[DType.int32](n + 1, n + 1)
    var mutual = ctx.enqueue_create_buffer[DType.uint8](nk)
    ctx.enqueue_memset(small, Int32(0))
    ctx.enqueue_function[fg_sort_rows_kernel](
        acols.unsafe_ptr(), has_self.unsafe_ptr(), flags.unsafe_ptr(),
        Int32(k), Int32(kp), grid_dim=(n, 1, 1), block_dim=(FG_SORT_TPB, 1, 1),
    )
    ctx.enqueue_function[fg_mutual_kernel](
        acols.unsafe_ptr(), mutual.unsafe_ptr(), ecnt.unsafe_ptr(),
        Int32(n), Int32(k),
        grid_dim=((nk + FG_TPB - 1) // FG_TPB, 1, 1), block_dim=(FG_TPB, 1, 1),
    )
    ctx.enqueue_function[fg_scan_kernel](
        ecnt.unsafe_ptr(), has_self.unsafe_ptr(), indptr_v.unsafe_ptr(),
        eoff.unsafe_ptr(), Int32(n), Int32(k),
        grid_dim=(1, 1, 1), block_dim=(FG_SCAN_TPB, 1, 1),
    )
    var tot = ctx.enqueue_create_host_buffer[DType.int32](4)
    ctx.enqueue_copy(dst_ptr=tot.unsafe_ptr(), src_buf=flags)
    ctx.enqueue_copy(dst_ptr=tot.unsafe_ptr() + 1, src_buf=indptr_v.create_sub_buffer[DType.int32](n, 1))
    ctx.enqueue_copy(dst_ptr=tot.unsafe_ptr() + 2, src_buf=eoff.create_sub_buffer[DType.int32](n, 1))
    ctx.synchronize()
    if tot.unsafe_ptr()[0] != 0:
        raise Error(
            "connectivity_graph: a kNN row holds a repeated neighbor --"
            " refused by name (DEVIATION 775)"
        )
    var nnz = Int(tot.unsafe_ptr()[1])
    var n_extra = Int(tot.unsafe_ptr()[2])
    _ = tot^
    var ebuf = ctx.enqueue_create_buffer[DType.uint32](2 * n_extra + 2)
    var etmp = ebuf.create_sub_buffer[DType.uint32](0, n_extra + 1)
    var esorted = ebuf.create_sub_buffer[DType.uint32](n_extra + 1, n_extra + 1)
    ctx.enqueue_function[fg_scatter_kernel](
        acols.unsafe_ptr(), mutual.unsafe_ptr(), eoff.unsafe_ptr(),
        ecur.unsafe_ptr(), etmp.unsafe_ptr(), Int32(n), Int32(k),
        grid_dim=((nk + FG_TPB - 1) // FG_TPB, 1, 1), block_dim=(FG_TPB, 1, 1),
    )
    ctx.enqueue_function[fg_sort_extras_kernel](
        eoff.unsafe_ptr(), etmp.unsafe_ptr(), esorted.unsafe_ptr(),
        grid_dim=(n, 1, 1), block_dim=(FG_SORT_TPB, 1, 1),
    )
    var rows = ctx.enqueue_create_buffer[DType.int32](nnz)
    var cols = ctx.enqueue_create_buffer[DType.int32](nnz)
    var vals = ctx.enqueue_create_buffer[DType.float32](nnz)
    var indptr = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.enqueue_function[fg_place_kernel](
        acols.unsafe_ptr(), mutual.unsafe_ptr(), has_self.unsafe_ptr(),
        eoff.unsafe_ptr(), esorted.unsafe_ptr(), indptr_v.unsafe_ptr(),
        rows.unsafe_ptr(), cols.unsafe_ptr(), vals.unsafe_ptr(), Int32(k),
        grid_dim=(n, 1, 1), block_dim=(FG_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=indptr, src_buf=indptr_v)
    ctx.synchronize()
    _ = acols^
    _ = mutual^
    _ = ebuf^
    _ = small^
    _ = offs^
    return laplacian_from_sorted_device(ctx, n, nnz, rows^, cols^, vals^, indptr^, tpb)
