# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DBSCAN without a CSR: `-D MOJOLEARN_IDN_DBSCAN_EDGE_FREE_OFF` disables it.

WHY (lane dbscan-taxi-speed, 2026-10-10). The reference route (`runner.mojo`)
materializes every eps edge as an int32 CSR and min-propagates labels over
it. Its cost is O(E) per label pass, and E is not bounded by n: on a dataset
whose eps balls hold a large share of the rows (standardized taxi at eps 3
holds ~1e12 edges in 1e6 rows) the route

  - splits every memory-sized range in halves until each holds at most
    `edge_cap` (2^31) edges, recounting each half (log2(E_range / 2^31)
    extra distance passes over every row: ~6 on the L40S plan),
  - then runs, per CSR batch (~E / 2^31 of them, ~500), a fill, a
    `weak_cc_batched` that initializes and walks all N labels plus the
    batch's 2^31 columns per pass, and a `merge_labels` over all N per pass,
  - then the border pass refills every batch with a labelled border row.

That is ~4 TB of column traffic per label pass plus 500 x passes x N label
work: the race times out on every arm, on NVIDIA and AMD.

WHAT THIS DOES INSTEAD. The labels depend on two things only: the core mask
and the connected components of the core-core edge graph. Neither needs the
edges stored.

  A. Degrees: D1's tiled count (`rbc_eps_pass_degrees_tile`, the same kernel
     and predicate loop 1 uses) over every row, range by range; the core mask
     is `deg >= min_pts` (`core_points_compute`, unchanged).
  B. Components: lock-free union-find over the original row ids, linking the
     larger root under the smaller (so a root is always its set's smallest
     row id). Range by range, D2's bit matrix (`rbc_eps_pass_bits_tile`, the
     same bits loop 1 writes) is scanned by one thread per query row; every
     set bit (q, j) with both rows core is a union. Before each range the
     parents are compressed and every 64-slot block of the index gets its
     single root (`broot`: the one root all its core rows share, -1 when they
     differ, -2 when it holds no core row). A query row skips a block whose
     single root is already its own root and, for a single-root block on
     another root, makes ONE union with its first core hit. So once a range
     has joined a dense region, every later range reads that region as
     n / 64 words per row, not n edges.
  C. Border rows: a non-core row has fewer than `min_pts` neighbours, so its
     whole neighbour list (from the bits) fits a small CSR sized by the
     scan of the non-core degrees. After the last range each non-core row
     takes the smallest `root + 1` over its core neighbours, `MAX_LABEL`
     when it has none (DEVIATION 5130's border rule, `border_pull_kernel`).
  D. A core row's label is `root + 1`: the component's smallest row id + 1,
     which is `weak_cc`'s fixed point (`i + 1` min-propagated) and the merged
     labelling's. `_dbscan_finish` then runs the same `make_monotonic` +
     `relabelForSkl` tail.

ACCELERATION, NOT SEMANTICS: THE DENSE PREFIX. A row of landmark L's slice
with `d1 <= h = eps / 2 * (1 - 1e-3)` is within `2h < eps` of every other
such row of L (triangle inequality through L; `d1` and the squared distance
each carry a relative float error of at most ~n_features * 2^-24, which the
1e-3 margin covers while `n_features <= EF_DENSE_MAX_DIMS`). Every such pair
also passes D1's three tests (test 1: `d(q, L) <= h < eps + radius`; test 2:
`d(q, L) - d1[s] <= h < eps`; test 3: the squared distance is at most
`(2h)^2 (1 + err) < eps^2`), so it IS an edge of the reference graph. Its
core rows are unioned before range 0, which makes the dense blocks
single-rooted before the first bit scan. Skipping it changes the work, never
the partition.

THE SAME LABELS. The edge set is D1's predicate (the bits), the core mask is
D1's degrees, the partition of the core rows is the connected components of
those edges (union-find's final partition does not depend on the order the
unions land in), the root of each set is its minimum row id by construction,
and the border rule is the batched reference's. So the output equals the
CSR route's for any batching; `dbscan_edge_split_check` compares it to the
one-batch fit bit for bit. No float is folded anywhere in B-D, so NVIDIA and
AMD give the same words whenever their bits agree, which they must already
for the CSR route.

COST. Two D1 distance passes (A and B), one bit-matrix write and read per
row (n / 8 bytes), O(n) compress and block-root passes per range, union work
bounded by the multi-root blocks a row meets (the first range in a dense
region, the slice tails), and border work below n * min_pts. No int32 edge
bound and no per-batch O(N) label pass.

WHEN IT RUNS (runner.mojo): IDENTICAL on NVIDIA and AMD with the bit matrix
on (IDN_DBSCAN_ADJ_BITMAP), unweighted, Euclidean ball cover, and only once a
memory-sized range's exact count passes `edge_cap`: the point where the CSR
route would start splitting ranges below the memory plan and its cost turns
O(E). A fit whose ranges fit one CSR each keeps the CSR route unchanged.
"""

from std.atomic import Atomic, Ordering
from std.bit import count_trailing_zeros
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from core.device_zero import enqueue_fill
from dbscan.impl.corepoints.compute import core_points_compute
from dbscan.impl.sparse.detail.csr import MAX_LABEL
from neighbors.impl.ball_cover.registers import (
    IDN_DBSCAN_ADJ_BITMAP,
    rbc_eps_bitmap_words,
    rbc_eps_pass_bits_tile,
    rbc_eps_pass_degrees_tile,
)
from neighbors.impl.ball_cover.scan import rbc_exclusive_scan_launch


#: The edge-free route. Default ON wherever the bit matrix is (IDENTICAL,
#: NVIDIA and AMD): a clear asymptotic fix (O(E) label passes and int32
#: splits -> O(n * n / 64) bit words). `-D MOJOLEARN_IDN_DBSCAN_EDGE_FREE_OFF`
#: (or MOJOLEARN_IDN_ALL_OFF) restores the CSR route everywhere.
#: KEPT 2026-10-10 (lane grid-act-15; A/B on main ca25d9321, one run per arm,
#: nv2 v1234 default / v1235 _OFF, amd a1558 / a1559): taxi default 13336 ms
#: NV / 7057 ms AMD, hash d6fa652f on both, 36 clusters, noise 0.000174; the
#: _OFF arm is REFUSED(timeout) on both vendors. istella default 84826 / 54245
#: ms, _OFF 84697 / 54157 ms (1.00x / 1.00x), hash 7494ce8e on all four
#: cells, 40131 clusters, noise 0.219391 (no change on istella).
comptime IDN_DBSCAN_EDGE_FREE = IDN_DBSCAN_ADJ_BITMAP and not (
    is_defined["MOJOLEARN_IDN_DBSCAN_EDGE_FREE_OFF"]()
    or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

comptime EF_TPB = 128
comptime EF_BROOT_TPB = 64
#: `broot` cell of a 64-slot block with no core row.
comptime EF_NO_CORE = Int32(-2)
#: `broot` cell of a block whose core rows sit on more than one root.
comptime EF_MULTI = Int32(-1)
#: The dense-prefix margin (1e-3, relative) covers the float error of `d1`
#: and of the squared distance, each at most ~n_features * 2^-24 relative
#: (a sum of n_features non-negative rounded terms). At 2,048 features that
#: is 1.2e-4, eight times inside the margin. Above it the prefix step is
#: skipped (work only; the partition is the same without it).
comptime EF_DENSE_MAX_DIMS = 2048


@always_inline
def _ef_find(parent: MutPointer[Int32, MutAnyOrigin], x0: Int32) -> Int32:
    var r = x0
    while True:
        var p = Atomic.load[ordering = Ordering.RELAXED](parent + Int(r))
        if p == r:
            return r
        r = p


@always_inline
def _ef_union(parent: MutPointer[Int32, MutAnyOrigin], a: Int32, b: Int32):
    """Link the larger root under the smaller (CAS), so every root is its
    set's smallest id. Retries when another thread moved the root first."""
    var x = a
    var y = b
    while True:
        x = _ef_find(parent, x)
        y = _ef_find(parent, y)
        if x == y:
            return
        if x < y:
            var t = x
            x = y
            y = t
        var expected = x
        if Atomic.compare_exchange[
            success_ordering = Ordering.RELAXED,
            failure_ordering = Ordering.RELAXED,
            weak=True,
        ](parent + Int(x), expected, y):
            return


def ef_init_parent_kernel(parent: MutPointer[Int32, MutAnyOrigin], n_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        parent.unsafe_store(i, Int32(i))


def ef_compress_kernel(parent: MutPointer[Int32, MutAnyOrigin], n_in: Int32):
    """`parent[i] = root(i)`. Run with no union in flight, so every thread
    sees the same roots and the result is the partition's root map."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        parent.unsafe_store(i, _ef_find(parent, Int32(i)))


def ef_border_deg_kernel(
    deg: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    bdeg: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """A non-core row keeps its degree (below min_pts), a core row 0."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        if core.unsafe_load(i) != 0:
            bdeg.unsafe_store(i, Int32(0))
        else:
            bdeg.unsafe_store(i, deg.unsafe_load(i))


def ef_dense_anchor_kernel(
    d1: MutPointer[Float32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    nearest: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    anchor: MutPointer[Int32, MutAnyOrigin],
    h: Float32,
    n_in: Int32,
):
    """One thread per index slot: the smallest core row id of each slice's
    dense prefix (`d1 <= h`). A minimum, so the order of the atomics does
    not matter."""
    var s = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if s >= Int(n_in):
        return
    if d1.unsafe_load(s) <= h:
        var c = cols.unsafe_load(s)
        if core.unsafe_load(Int(c)) != 0:
            var k = Int(nearest.unsafe_load(Int(c)))
            _ = Atomic.min(anchor.unsafe_offset(k), c)


def ef_dense_union_kernel(
    d1: MutPointer[Float32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    nearest: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    anchor: MutPointer[Int32, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    h: Float32,
    n_in: Int32,
):
    """Every core row of a dense prefix joins its prefix's anchor: the pair
    is a true edge of the reference graph (module docstring)."""
    var s = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if s >= Int(n_in):
        return
    if d1.unsafe_load(s) <= h:
        var c = cols.unsafe_load(s)
        if core.unsafe_load(Int(c)) != 0:
            var k = Int(nearest.unsafe_load(Int(c)))
            var a = anchor.unsafe_load(k)
            if a != c and a != MAX_LABEL:
                _ef_union(parent, c, a)


def ef_block_root_kernel(
    r_indptr: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    broot: MutPointer[Int32, MutAnyOrigin],
):
    """One block per landmark, one thread per 64-slot block of its slice:
    the block's single root, `EF_MULTI` or `EF_NO_CORE`. `parent` is
    compressed and no union is in flight, so `parent[j]` is j's root. The
    word index is D2's (`lm + indptr[lm] // 64 + t`)."""
    var lm = Int(block_idx.x)
    var rs = Int(r_indptr.unsafe_load(lm))
    var rsz = Int(r_indptr.unsafe_load(lm + 1)) - rs
    var wbase = lm + rs // 64
    var nw = (rsz + 63) // 64
    var t = Int(thread_idx.x)
    while t < nw:
        var lo = MAX_LABEL
        var hi = Int32(-1)
        var p_end = 64 * t + 64
        if p_end > rsz:
            p_end = rsz
        for p in range(64 * t, p_end):
            var j = Int(cols.unsafe_load(rs + p))
            if core.unsafe_load(j) != 0:
                var rj = parent.unsafe_load(j)
                if rj < lo:
                    lo = rj
                if rj > hi:
                    hi = rj
        var cell = EF_MULTI
        if hi < Int32(0):
            cell = EF_NO_CORE
        elif lo == hi:
            cell = lo
        broot.unsafe_store(wbase + t, cell)
        t += EF_BROOT_TPB


def ef_union_kernel(
    bm: MutPointer[UInt64, MutAnyOrigin],
    n_queries_in: Int32,
    start_in: Int32,
    n_landmarks_in: Int32,
    r_indptr: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    core: MutPointer[UInt8, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    broot: MutPointer[Int32, MutAnyOrigin],
    boff: MutPointer[Int32, MutAnyOrigin],
    blist: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per query row of the range. A core row unions with every
    core row its bits name (skipping blocks already on its root, one union
    per single-root block); a non-core row copies its neighbour ids to its
    border list. The lanes of a warp walk the same (landmark, word) in step,
    so the `broot` and root loads are broadcasts and the word loads are
    adjacent (D2's word-major layout)."""
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nq = Int(n_queries_in)
    if q >= nq:
        return
    var g = Int(start_in) + q
    var gi = Int32(g)
    var is_core = core.unsafe_load(g) != 0
    var pos = 0
    if not is_core:
        pos = Int(boff.unsafe_load(g))
    for lm in range(Int(n_landmarks_in)):
        var rs = Int(r_indptr.unsafe_load(lm))
        var rsz = Int(r_indptr.unsafe_load(lm + 1)) - rs
        var wbase = lm + rs // 64
        var nw = (rsz + 63) // 64
        for t in range(nw):
            var w = bm.unsafe_load((wbase + t) * nq + q)
            if w == UInt64(0):
                continue
            var base = rs + 64 * t
            if not is_core:
                while w != UInt64(0):
                    var c = Int(count_trailing_zeros(w))
                    w &= w - UInt64(1)
                    blist.unsafe_store(pos, cols.unsafe_load(base + c))
                    pos += 1
                continue
            var br = broot.unsafe_load(wbase + t)
            if br == EF_NO_CORE:
                continue
            if br >= Int32(0):
                if _ef_find(parent, br) == _ef_find(parent, gi):
                    continue
                # every core row of the block is on br's root: one union
                # with the first core hit joins them all
                while w != UInt64(0):
                    var c = Int(count_trailing_zeros(w))
                    w &= w - UInt64(1)
                    var j = cols.unsafe_load(base + c)
                    if core.unsafe_load(Int(j)) != 0:
                        _ef_union(parent, gi, j)
                        break
                continue
            while w != UInt64(0):
                var c = Int(count_trailing_zeros(w))
                w &= w - UInt64(1)
                var j = cols.unsafe_load(base + c)
                if core.unsafe_load(Int(j)) != 0:
                    _ef_union(parent, gi, j)


def ef_label_kernel(
    core: MutPointer[UInt8, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    boff: MutPointer[Int32, MutAnyOrigin],
    blist: MutPointer[Int32, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """Core: root + 1. Non-core: the smallest root + 1 over its core
    neighbours, `MAX_LABEL` (noise) when it has none. `parent` is
    compressed. Reads `parent` only and writes `labels` only."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    if core.unsafe_load(i) != 0:
        labels.unsafe_store(i, parent.unsafe_load(i) + Int32(1))
        return
    var best = MAX_LABEL
    for e in range(Int(boff.unsafe_load(i)), Int(boff.unsafe_load(i + 1))):
        var j = Int(blist.unsafe_load(e))
        if core.unsafe_load(j) != 0:
            var lj = parent.unsafe_load(j) + Int32(1)
            if lj < best:
                best = lj
    labels.unsafe_store(i, best)


def edge_free_admits(n_rows: Int, min_pts: Int) -> Bool:
    """The border lists total at most n * (min_pts - 1) ids, int32-indexed."""
    return min_pts >= 1 and n_rows * (min_pts - 1) < Int(MAX_LABEL)


def dbscan_edge_free_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_reordered: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut nearest: DeviceBuffer[DType.int32],
    mut core: DeviceBuffer[DType.uint8],
    mut labels: DeviceBuffer[DType.int32],
    mut bm: DeviceBuffer[DType.uint64],
    n_rows: Int,
    n_features: Int,
    n_landmarks: Int,
    eps: Float32,
    min_pts: Int,
    batch: Int,
    phase_timing: Bool,
) raises -> Int:
    """Steps A-D of the module docstring. `bm` holds `batch` query rows of
    D2's bit matrix (the runner's memory plan). Writes `core` and the
    merged-stage `labels`; returns the number of query ranges."""
    var n_ranges = (n_rows + batch - 1) // batch
    var blocks_n = (n_rows + EF_TPB - 1) // EF_TPB
    var t0 = perf_counter_ns()

    # --- A: degrees over every row, then the core mask ----------------------
    var deg = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var deg_scratch = ctx.enqueue_create_buffer[DType.int32](batch)
    for k in range(n_ranges):  # small-loop(n_ranges: one D1 launch per memory-sized query range): the plan, no row data
        var st = k * batch
        var np = min(n_rows - st, batch)
        var qa = x.create_sub_buffer[DType.float32](st * n_features, np * n_features)
        var da = deg.create_sub_buffer[DType.int32](st, np)
        rbc_eps_pass_degrees_tile(
            ctx, x_reordered, qa, r, r_indptr, r_1nn_cols, r_1nn_dists,
            r_radius, da, np, n_features, n_landmarks, eps,
        )
    core_points_compute(ctx, deg, core, min_pts, 0, n_rows)
    ctx.synchronize()
    var t_a = perf_counter_ns()

    # --- C (sizing): the border lists of the non-core rows ------------------
    var bdeg = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var boff = ctx.enqueue_create_buffer[DType.int32](n_rows + 1)
    ctx.enqueue_function[ef_border_deg_kernel](
        deg.unsafe_ptr(), core.unsafe_ptr(), bdeg.unsafe_ptr(), Int32(n_rows),
        grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
    )
    rbc_exclusive_scan_launch(ctx, boff, bdeg, n_rows)
    var h_tot = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(
        dst_ptr=h_tot.unsafe_ptr(),
        src_buf=boff.create_sub_buffer[DType.int32](n_rows, 1),
    )
    ctx.synchronize()
    var n_border = Int(h_tot.unsafe_ptr().unsafe_load(0))
    var blist = ctx.enqueue_create_buffer[DType.int32](n_border if n_border > 0 else 1)

    # --- B: union-find over the core-core bits ------------------------------
    var parent = ctx.enqueue_create_buffer[DType.int32](n_rows)
    ctx.enqueue_function[ef_init_parent_kernel](
        parent.unsafe_ptr(), Int32(n_rows),
        grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
    )
    if n_features <= EF_DENSE_MAX_DIMS:
        var anchor = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
        enqueue_fill(ctx, anchor, MAX_LABEL)
        var h = eps * Float32(0.5) * Float32(1.0 - 1.0e-3)
        ctx.enqueue_function[ef_dense_anchor_kernel](
            r_1nn_dists.unsafe_ptr(), r_1nn_cols.unsafe_ptr(),
            nearest.unsafe_ptr(), core.unsafe_ptr(), anchor.unsafe_ptr(), h,
            Int32(n_rows),
            grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
        )
        ctx.enqueue_function[ef_dense_union_kernel](
            r_1nn_dists.unsafe_ptr(), r_1nn_cols.unsafe_ptr(),
            nearest.unsafe_ptr(), core.unsafe_ptr(), anchor.unsafe_ptr(),
            parent.unsafe_ptr(), h, Int32(n_rows),
            grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
        )
        ctx.synchronize()
        _ = anchor^
    var broot = ctx.enqueue_create_buffer[DType.int32](
        rbc_eps_bitmap_words(n_rows, n_landmarks)
    )
    for k in range(n_ranges):  # small-loop(n_ranges: one bit-matrix range per memory-sized query range): the plan, no row data
        var st = k * batch
        var np = min(n_rows - st, batch)
        ctx.enqueue_function[ef_compress_kernel](
            parent.unsafe_ptr(), Int32(n_rows),
            grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
        )
        ctx.enqueue_function[ef_block_root_kernel](
            r_indptr.unsafe_ptr(), r_1nn_cols.unsafe_ptr(), core.unsafe_ptr(),
            parent.unsafe_ptr(), broot.unsafe_ptr(),
            grid_dim=(n_landmarks, 1, 1), block_dim=(EF_BROOT_TPB, 1, 1),
        )
        var qb = x.create_sub_buffer[DType.float32](st * n_features, np * n_features)
        rbc_eps_pass_bits_tile(
            ctx, x_reordered, qb, r, r_indptr, r_1nn_cols, r_1nn_dists,
            r_radius, deg_scratch, bm, np, n_features, n_landmarks, eps,
        )
        ctx.enqueue_function[ef_union_kernel](
            bm.unsafe_ptr(), Int32(np), Int32(st), Int32(n_landmarks),
            r_indptr.unsafe_ptr(), r_1nn_cols.unsafe_ptr(), core.unsafe_ptr(),
            parent.unsafe_ptr(), broot.unsafe_ptr(), boff.unsafe_ptr(),
            blist.unsafe_ptr(),
            grid_dim=((np + EF_TPB - 1) // EF_TPB, 1, 1),
            block_dim=(EF_TPB, 1, 1),
        )
        ctx.synchronize()
    var t_b = perf_counter_ns()

    # --- C, D: labels -------------------------------------------------------
    ctx.enqueue_function[ef_compress_kernel](
        parent.unsafe_ptr(), Int32(n_rows),
        grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
    )
    ctx.enqueue_function[ef_label_kernel](
        core.unsafe_ptr(), parent.unsafe_ptr(), boff.unsafe_ptr(),
        blist.unsafe_ptr(), labels.unsafe_ptr(), Int32(n_rows),
        grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
    )
    ctx.synchronize()
    if phase_timing:
        print(
            "PHASE edge_free ranges " + String(n_ranges) + " border_ids "
            + String(n_border) + " degrees "
            + String(Float64(t_a - t0) / 1.0e6) + " unions "
            + String(Float64(t_b - t_a) / 1.0e6) + " labels "
            + String(Float64(perf_counter_ns() - t_b) / 1.0e6)
        )
    _ = deg^
    _ = deg_scratch^
    _ = bdeg^
    _ = boff^
    _ = blist^
    _ = parent^
    _ = broot^
    return n_ranges
