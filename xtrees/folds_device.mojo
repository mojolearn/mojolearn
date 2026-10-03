# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The cv fold bookkeeping of Stacking / CalibratedClassifierCV on the device
(lane/apple-fast-trees-ensembles, 2026-10-02; every GPU build and the host
column since lane apple-fast-py2mojo-trees, 2026-10-03).

The law is sklearn's StratifiedKFold(shuffle=False) / KFold(shuffle=False),
exactly as python/mojolearn/_expansion_trees.py `_trees_stratified_folds` /
`_trees_kfolds` state it: no draw anywhere (shuffle=False has no RNG), so the
folds are the Python routine's and no model bit moves. Stratified: classes in
order of first appearance; the sorted labels dealt round robin give class e
(block [s_e, s_e + c_e) of the sorted labels) its per-fold allocation
alloc[f][e] = #{p in the block : p mod n_splits == f}; the rows of a class, in
row order, take fold 0 alloc[0][e] times, then fold 1, ... So the row with
within-class rank j is in the first fold f with sum_{i <= f} alloc[i][e] > j.
KFold: contiguous folds, the first n mod n_splits one row longer.

Every step is a row-per-thread launch or a block-parallel scan:
  1. `fold_hist_kernel`: the class histogram and each class's first row
     (atomics), one thread per row;
  2. `fold_classes_kernel` / `fold_starts_kernel`: the first-appearance rank
     of each class, its count and block start (one thread per class; the
     loops inside are over the k classes, not the rows);
  3. per class: `fold_flag_kernel` + the exclusive scan + `fold_rank_kernel`
     give every row its within-class rank (a per-class scan);
  4. `fold_assign_strat_kernel` (or `fold_assign_kfold_kernel`): the fold id
     per row and the fold sizes (atomics);
  5. per fold: `fold_flag_kernel` + the scan + `fold_scatter_kernel` compact
     the rows outside the fold (ascending) and inside it (ascending) into
     the fold's slot of one output buffer.
The host receives the fold sizes and that buffer in one download at the end;
nothing is read back in between. The scan is the three-phase block scan of
gbdt/gpu_util/kernel/scan.mojo with its phase 2 made parallel: the block
totals are scanned by the same block kernel, recursively, until one tile of
at most FOLD_SCAN_BLOCK totals remains (n <= FOLD_SCAN_BLOCK^3 rows)."""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import prefix_sum
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN
from xtrees.ops import folds_serial
from core.neural_context import process_ctx

comptime FOLD_TPB = 256
comptime FOLD_SCAN_BLOCK = 512
comptime FOLD_GRID_CAP = 65535 * 16
comptime TE_DEVICE_FOLDS = TARGET_COLUMN != COLUMN_CPU
"""Every GPU build (both tiers, every vendor) takes the device fold path
since lane apple-fast-py2mojo-trees (2026-10-03; it was FAST + Apple only);
the CPU column (`-D MOJOLEARN_COLUMN_CPU`, the host binding) runs
`xtrees/ops.mojo` `folds_serial`, the same integers."""


def _grid(n: Int) -> Int:
    return max(1, min((n + FOLD_TPB - 1) // FOLD_TPB, FOLD_GRID_CAP))


# ---------------------------------------------------------------- kernels --


def fold_hist_kernel(
    codes: MutPointer[Int32, MutAnyOrigin],
    n_in: Int64,
    k_in: Int32,
    n_splits_in: Int32,
    hist: MutPointer[Int32, MutAnyOrigin],
    first: MutPointer[Int32, MutAnyOrigin],
    fcount: MutPointer[Int32, MutAnyOrigin],
):
    """hist[c] += 1 and first[c] = min(first[c], r) for every row r of code
    c; a code outside [0, k) sets the status word fcount[n_splits] to 2."""
    var n = Int(n_in)
    var k = Int(k_in)
    var ns = Int(n_splits_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < n:
        var c = Int(codes[r])
        if c < 0 or c >= k:
            _ = Atomic.max(fcount.unsafe_offset(ns), Int32(2))
        else:
            _ = Atomic.fetch_add(hist.unsafe_offset(c), Int32(1))
            _ = Atomic.min(first.unsafe_offset(c), Int32(r))
        r += stride


def fold_classes_kernel(
    k_in: Int32,
    n_splits_in: Int32,
    hist: MutPointer[Int32, MutAnyOrigin],
    first: MutPointer[Int32, MutAnyOrigin],
    enc: MutPointer[Int32, MutAnyOrigin],
    count_enc: MutPointer[Int32, MutAnyOrigin],
    fcount: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per code c: enc[c] = the number of present classes whose
    first row comes earlier (its index in order of first appearance; -1 for
    a code with no rows), count_enc[enc[c]] = hist[c]. Thread 0 sets the
    status word fcount[n_splits] to 1 when n_splits exceeds every class
    count (the Python refusal)."""
    var k = Int(k_in)
    var ns = Int(n_splits_in)
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c < k:
        if Int(hist[c]) == 0:
            enc[c] = Int32(-1)
        else:
            var fc = Int(first[c])
            var e = 0
            for c2 in range(k):
                if Int(hist[c2]) > 0 and Int(first[c2]) < fc:
                    e += 1
            enc[c] = Int32(e)
            count_enc[e] = hist[c]
    if c == 0:
        var cmax = 0
        for c2 in range(k):
            if Int(hist[c2]) > cmax:
                cmax = Int(hist[c2])
        if ns > cmax:
            _ = Atomic.max(fcount.unsafe_offset(ns), Int32(1))


def fold_starts_kernel(
    k_in: Int32,
    count_enc: MutPointer[Int32, MutAnyOrigin],
    start: MutPointer[Int32, MutAnyOrigin],
):
    """start[e] = the number of rows of the classes before e in first-
    appearance order (class e's block of the sorted labels starts there)."""
    var k = Int(k_in)
    var e = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if e < k:
        var s = 0
        for e2 in range(e):
            s += Int(count_enc[e2])
        start[e] = Int32(s)


def fold_flag_kernel(
    keys: MutPointer[Int32, MutAnyOrigin],
    n_in: Int64,
    key: Int32,
    flags: MutPointer[Int32, MutAnyOrigin],
):
    """flags[r] = 1 where keys[r] == key, else 0."""
    var n = Int(n_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < n:
        flags[r] = Int32(1) if keys[r] == key else Int32(0)
        r += stride


def fold_scan_block_kernel(
    values: MutPointer[Int32, MutAnyOrigin],
    size_in: Int32,
    scanned: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
):
    """The block-local EXCLUSIVE sum of `values` plus each block's total
    (gbdt/gpu_util/kernel/scan.mojo `scan_block_u32_kernel`, Int32)."""
    var size = Int(size_in)
    var tid = Int(thread_idx.x)
    var start = Int(block_idx.x) * FOLD_SCAN_BLOCK
    var v = Int32(0)
    if start + tid < size:
        v = values.unsafe_load(start + tid)
    var exclusive = prefix_sum[block_size=FOLD_SCAN_BLOCK, exclusive=True](v)
    if start + tid < size:
        scanned.unsafe_store(start + tid, exclusive)
    if tid == FOLD_SCAN_BLOCK - 1:
        block_sums.unsafe_store(Int(block_idx.x), exclusive + v)


def fold_scan_carry_kernel(
    scanned: MutPointer[Int32, MutAnyOrigin],
    carries: MutPointer[Int32, MutAnyOrigin],
    size_in: Int32,
):
    """scanned[i] += carries[block of i] (launched at FOLD_SCAN_BLOCK threads
    per block, so block_idx.x is the scan block)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        scanned.unsafe_store(
            i, scanned.unsafe_load(i) + carries.unsafe_load(Int(block_idx.x))
        )


def fold_rank_kernel(
    keys: MutPointer[Int32, MutAnyOrigin],
    n_in: Int64,
    key: Int32,
    scanned: MutPointer[Int32, MutAnyOrigin],
    rank: MutPointer[Int32, MutAnyOrigin],
):
    """rank[r] = scanned[r] (the number of earlier rows with this key) where
    keys[r] == key."""
    var n = Int(n_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < n:
        if keys[r] == key:
            rank[r] = scanned[r]
        r += stride


def fold_assign_strat_kernel(
    codes: MutPointer[Int32, MutAnyOrigin],
    n_in: Int64,
    n_splits_in: Int32,
    enc: MutPointer[Int32, MutAnyOrigin],
    start: MutPointer[Int32, MutAnyOrigin],
    count_enc: MutPointer[Int32, MutAnyOrigin],
    rank: MutPointer[Int32, MutAnyOrigin],
    folds: MutPointer[Int32, MutAnyOrigin],
    fcount: MutPointer[Int32, MutAnyOrigin],
):
    """Row r of class e with within-class rank j: the first fold f with
    sum_{i <= f} alloc[i][e] > j, alloc[i][e] = #{p in [s_e, s_e + c_e) :
    p mod n_splits == i}, counted in closed form; fcount[f] += 1."""
    var n = Int(n_in)
    var ns = Int(n_splits_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < n:
        var e = Int(enc[Int(codes[r])])
        var j = Int(rank[r])
        var s = Int(start[e])
        var c = Int(count_enc[e])
        var sm = s % ns
        var cum = 0
        var f = 0
        while f < ns - 1:
            # the first p >= s with p mod ns == f, then every ns-th up to s + c
            var p0 = s + ((f - sm + ns) % ns)
            if p0 < s + c:
                cum += (s + c - 1 - p0) // ns + 1
            if j < cum:
                break
            f += 1
        folds[r] = Int32(f)
        _ = Atomic.fetch_add(fcount.unsafe_offset(f), Int32(1))
        r += stride


def fold_assign_kfold_kernel(
    n_in: Int64,
    n_splits_in: Int32,
    folds: MutPointer[Int32, MutAnyOrigin],
    fcount: MutPointer[Int32, MutAnyOrigin],
):
    """KFold(shuffle=False): the first n mod n_splits folds hold
    n // n_splits + 1 contiguous rows, the rest n // n_splits."""
    var n = Int(n_in)
    var ns = Int(n_splits_in)
    var q = n // ns
    var rem = n % ns
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < n:
        var f = 0
        if r < rem * (q + 1):
            f = r // (q + 1)
        else:
            f = rem + (r - rem * (q + 1)) // q
        folds[r] = Int32(f)
        _ = Atomic.fetch_add(fcount.unsafe_offset(f), Int32(1))
        r += stride


def fold_scatter_kernel(
    folds: MutPointer[Int32, MutAnyOrigin],
    n_in: Int64,
    i_in: Int32,
    scanned: MutPointer[Int32, MutAnyOrigin],
    fcount: MutPointer[Int32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
):
    """Fold i's slot rows[i * n : (i + 1) * n]: the rows outside fold i,
    ascending, then the rows inside it, ascending. `scanned[r]` is the
    number of earlier rows inside fold i."""
    var n = Int(n_in)
    var i = Int(i_in)
    var cnt = Int(fcount[i])
    var base = i * n
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < n:
        var inside = Int(scanned[r])
        if Int(folds[r]) == i:
            rows[base + (n - cnt) + inside] = Int32(r)
        else:
            rows[base + (r - inside)] = Int32(r)
        r += stride


# --------------------------------------------------------------- launches --


struct FoldScanWorkspace(Movable):
    """The block-total buffers of the recursive exclusive scan over n."""
    var nb1: Int
    var nb2: Int
    var s1: DeviceBuffer[DType.int32]
    var s1s: DeviceBuffer[DType.int32]
    var s2: DeviceBuffer[DType.int32]
    var s2s: DeviceBuffer[DType.int32]
    var s3: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, n: Int) raises:
        self.nb1 = (n + FOLD_SCAN_BLOCK - 1) // FOLD_SCAN_BLOCK
        self.nb2 = (self.nb1 + FOLD_SCAN_BLOCK - 1) // FOLD_SCAN_BLOCK
        if self.nb2 > FOLD_SCAN_BLOCK:
            raise Error("x_trees device_folds: more rows than the fold scan covers")
        self.s1 = ctx.enqueue_create_buffer[DType.int32](max(1, self.nb1))
        self.s1s = ctx.enqueue_create_buffer[DType.int32](max(1, self.nb1))
        self.s2 = ctx.enqueue_create_buffer[DType.int32](max(1, self.nb2))
        self.s2s = ctx.enqueue_create_buffer[DType.int32](max(1, self.nb2))
        self.s3 = ctx.enqueue_create_buffer[DType.int32](1)


def _exclusive_scan(
    ctx: DeviceContext,
    mut values: DeviceBuffer[DType.int32],
    mut scanned: DeviceBuffer[DType.int32],
    n: Int,
    mut ws: FoldScanWorkspace,
) raises:
    """scanned = the exclusive prefix sum of values[0:n], every phase
    parallel: the block totals are scanned by the block kernel again, down
    to one tile of at most FOLD_SCAN_BLOCK totals."""
    ctx.enqueue_function[fold_scan_block_kernel](
        values.unsafe_ptr(), Int32(n), scanned.unsafe_ptr(), ws.s1.unsafe_ptr(),
        grid_dim=ws.nb1, block_dim=FOLD_SCAN_BLOCK,
    )
    if ws.nb1 > 1:
        ctx.enqueue_function[fold_scan_block_kernel](
            ws.s1.unsafe_ptr(), Int32(ws.nb1), ws.s1s.unsafe_ptr(), ws.s2.unsafe_ptr(),
            grid_dim=ws.nb2, block_dim=FOLD_SCAN_BLOCK,
        )
        if ws.nb2 > 1:
            # at most FOLD_SCAN_BLOCK totals: one tile, its size bounded at compile time
            ctx.enqueue_function[fold_scan_block_kernel](
                ws.s2.unsafe_ptr(), Int32(ws.nb2), ws.s2s.unsafe_ptr(), ws.s3.unsafe_ptr(),
                grid_dim=1, block_dim=FOLD_SCAN_BLOCK,
            )
            ctx.enqueue_function[fold_scan_carry_kernel](
                ws.s1s.unsafe_ptr(), ws.s2s.unsafe_ptr(), Int32(ws.nb1),
                grid_dim=ws.nb2, block_dim=FOLD_SCAN_BLOCK,
            )
        ctx.enqueue_function[fold_scan_carry_kernel](
            scanned.unsafe_ptr(), ws.s1s.unsafe_ptr(), Int32(n),
            grid_dim=ws.nb1, block_dim=FOLD_SCAN_BLOCK,
        )


def device_folds(
    codes: MutPointer[Int32, MutUntrackedOrigin],
    n: Int,
    k: Int,
    n_splits: Int,
    rows: MutPointer[Int32, MutUntrackedOrigin],
    counts: MutPointer[Int32, MutUntrackedOrigin],
) raises -> Int:
    """`k` > 0: StratifiedKFold over `codes` (int32 in [0, k)); `k` == 0:
    KFold (codes unread). Writes counts[0:n_splits] (the fold sizes) and
    rows[i * n : (i + 1) * n] for every fold i (the rows outside fold i
    ascending, then inside it ascending). Returns 0, 1 (n_splits exceeds
    every class count: the Python refusal) or 2 (a code outside [0, k))."""
    comptime if TE_DEVICE_FOLDS:
        var ctx = process_ctx["MojoXTreesPermContext"]()
        var kk = max(1, k)
        var d_flags = ctx.enqueue_create_buffer[DType.int32](n)
        var d_scanned = ctx.enqueue_create_buffer[DType.int32](n)
        var d_folds = ctx.enqueue_create_buffer[DType.int32](n)
        var d_fcount = ctx.enqueue_create_buffer[DType.int32](n_splits + 1)
        var d_rows = ctx.enqueue_create_buffer[DType.int32](n_splits * n)
        d_fcount.enqueue_fill(Int32(0))
        var ws = FoldScanWorkspace(ctx, n)
        var blocks = _grid(n)
        # the status word is fcount[n_splits] (counts[n_splits] on the host)
        if k > 0:
            var d_codes = ctx.enqueue_create_buffer[DType.int32](n)
            var d_rank = ctx.enqueue_create_buffer[DType.int32](n)
            var d_hist = ctx.enqueue_create_buffer[DType.int32](kk)
            var d_first = ctx.enqueue_create_buffer[DType.int32](kk)
            var d_enc = ctx.enqueue_create_buffer[DType.int32](kk)
            var d_count = ctx.enqueue_create_buffer[DType.int32](kk)
            var d_start = ctx.enqueue_create_buffer[DType.int32](kk)
            d_hist.enqueue_fill(Int32(0))
            d_first.enqueue_fill(Int32(n))
            d_count.enqueue_fill(Int32(0))
            ctx.enqueue_copy(dst_buf=d_codes, src_ptr=codes)
            ctx.enqueue_function[fold_hist_kernel](
                d_codes.unsafe_ptr(), Int64(n), Int32(k), Int32(n_splits), d_hist.unsafe_ptr(),
                d_first.unsafe_ptr(), d_fcount.unsafe_ptr(),
                grid_dim=blocks, block_dim=FOLD_TPB,
            )
            ctx.enqueue_function[fold_classes_kernel](
                Int32(k), Int32(n_splits), d_hist.unsafe_ptr(), d_first.unsafe_ptr(),
                d_enc.unsafe_ptr(), d_count.unsafe_ptr(), d_fcount.unsafe_ptr(),
                grid_dim=_grid(k), block_dim=FOLD_TPB,
            )
            ctx.enqueue_function[fold_starts_kernel](
                Int32(k), d_count.unsafe_ptr(), d_start.unsafe_ptr(),
                grid_dim=_grid(k), block_dim=FOLD_TPB,
            )
            # the within-class rank: one flag-and-scan per class
            for c in range(k):
                ctx.enqueue_function[fold_flag_kernel](
                    d_codes.unsafe_ptr(), Int64(n), Int32(c), d_flags.unsafe_ptr(),
                    grid_dim=blocks, block_dim=FOLD_TPB,
                )
                _exclusive_scan(ctx, d_flags, d_scanned, n, ws)
                ctx.enqueue_function[fold_rank_kernel](
                    d_codes.unsafe_ptr(), Int64(n), Int32(c), d_scanned.unsafe_ptr(),
                    d_rank.unsafe_ptr(),
                    grid_dim=blocks, block_dim=FOLD_TPB,
                )
            ctx.enqueue_function[fold_assign_strat_kernel](
                d_codes.unsafe_ptr(), Int64(n), Int32(n_splits), d_enc.unsafe_ptr(),
                d_start.unsafe_ptr(), d_count.unsafe_ptr(), d_rank.unsafe_ptr(),
                d_folds.unsafe_ptr(), d_fcount.unsafe_ptr(),
                grid_dim=blocks, block_dim=FOLD_TPB,
            )
            # the fold lists: one flag-and-scan per fold
            for i in range(n_splits):
                ctx.enqueue_function[fold_flag_kernel](
                    d_folds.unsafe_ptr(), Int64(n), Int32(i), d_flags.unsafe_ptr(),
                    grid_dim=blocks, block_dim=FOLD_TPB,
                )
                _exclusive_scan(ctx, d_flags, d_scanned, n, ws)
                ctx.enqueue_function[fold_scatter_kernel](
                    d_folds.unsafe_ptr(), Int64(n), Int32(i), d_scanned.unsafe_ptr(),
                    d_fcount.unsafe_ptr(), d_rows.unsafe_ptr(),
                    grid_dim=blocks, block_dim=FOLD_TPB,
                )
            ctx.enqueue_copy(dst_ptr=counts, src_buf=d_fcount)
            ctx.enqueue_copy(dst_ptr=rows, src_buf=d_rows)
            ctx.synchronize()
            _ = d_codes^
            _ = d_rank^
            _ = d_hist^
            _ = d_first^
            _ = d_enc^
            _ = d_count^
            _ = d_start^
        else:
            ctx.enqueue_function[fold_assign_kfold_kernel](
                Int64(n), Int32(n_splits), d_folds.unsafe_ptr(), d_fcount.unsafe_ptr(),
                grid_dim=blocks, block_dim=FOLD_TPB,
            )
            for i in range(n_splits):
                ctx.enqueue_function[fold_flag_kernel](
                    d_folds.unsafe_ptr(), Int64(n), Int32(i), d_flags.unsafe_ptr(),
                    grid_dim=blocks, block_dim=FOLD_TPB,
                )
                _exclusive_scan(ctx, d_flags, d_scanned, n, ws)
                ctx.enqueue_function[fold_scatter_kernel](
                    d_folds.unsafe_ptr(), Int64(n), Int32(i), d_scanned.unsafe_ptr(),
                    d_fcount.unsafe_ptr(), d_rows.unsafe_ptr(),
                    grid_dim=blocks, block_dim=FOLD_TPB,
                )
            ctx.enqueue_copy(dst_ptr=counts, src_buf=d_fcount)
            ctx.enqueue_copy(dst_ptr=rows, src_buf=d_rows)
            ctx.synchronize()
        var st = Int(counts[unsafe_offset=n_splits])
        _ = d_flags^
        _ = d_scanned^
        _ = d_folds^
        _ = d_fcount^
        _ = d_rows^
        _ = ws^
        return st
    else:
        return folds_serial(codes, n, k, n_splits, rows, counts)


def leaf_numbering(
    left: MutPointer[Int32, MutUntrackedOrigin], nn: Int, node_col: MutPointer[Int32, MutUntrackedOrigin],
) raises -> Int:
    """RandomTreesEmbedding's output columns (lane apple-fast-py2mojo-trees):
    node_col[g] = the number of leaves (left == -1) before node g in node
    order for a leaf, -1 for a split node; returns the leaf count. The device
    (flag, the parallel exclusive scan, rank) on every GPU build, one loop on
    the host column; integers, the same on both."""
    if nn <= 0:
        return 0
    comptime if TE_DEVICE_FOLDS:
        var ctx = process_ctx["MojoXTreesPermContext"]()
        var d_left = ctx.enqueue_create_buffer[DType.int32](nn)
        ctx.enqueue_copy(dst_buf=d_left, src_ptr=left)
        var d_flags = ctx.enqueue_create_buffer[DType.int32](nn)
        var d_scanned = ctx.enqueue_create_buffer[DType.int32](nn)
        var d_col = ctx.enqueue_create_buffer[DType.int32](nn)
        d_col.enqueue_fill(Int32(-1))
        var ws = FoldScanWorkspace(ctx, nn)
        var blocks = _grid(nn)
        ctx.enqueue_function[fold_flag_kernel](
            d_left.unsafe_ptr(), Int64(nn), Int32(-1), d_flags.unsafe_ptr(),
            grid_dim=blocks, block_dim=FOLD_TPB,
        )
        _exclusive_scan(ctx, d_flags, d_scanned, nn, ws)
        ctx.enqueue_function[fold_rank_kernel](
            d_left.unsafe_ptr(), Int64(nn), Int32(-1), d_scanned.unsafe_ptr(), d_col.unsafe_ptr(),
            grid_dim=blocks, block_dim=FOLD_TPB,
        )
        var d_tail = ctx.enqueue_create_buffer[DType.int32](2)
        var v_scan = d_scanned.create_sub_buffer[DType.int32](nn - 1, 1)
        var v_flag = d_flags.create_sub_buffer[DType.int32](nn - 1, 1)
        var h_a = d_tail.create_sub_buffer[DType.int32](0, 1)
        var h_b = d_tail.create_sub_buffer[DType.int32](1, 1)
        ctx.enqueue_copy(dst_buf=h_a, src_buf=v_scan)
        ctx.enqueue_copy(dst_buf=h_b, src_buf=v_flag)
        var h_tail = ctx.enqueue_create_host_buffer[DType.int32](2)
        ctx.enqueue_copy(dst_buf=h_tail, src_buf=d_tail)
        ctx.enqueue_copy(dst_ptr=node_col, src_buf=d_col)
        ctx.synchronize()
        var count = Int(h_tail.unsafe_ptr().unsafe_load(0)) + Int(h_tail.unsafe_ptr().unsafe_load(1))
        _ = v_scan^
        _ = v_flag^
        _ = h_a^
        _ = h_b^
        _ = h_tail^
        _ = d_tail^
        _ = d_left^
        _ = d_flags^
        _ = d_scanned^
        _ = d_col^
        _ = ws^
        return count
    else:
        var col = 0
        for g in range(nn):
            if Int(left[unsafe_offset=g]) == -1:
                node_col[unsafe_offset=g] = Int32(col)
                col += 1
            else:
                node_col[unsafe_offset=g] = Int32(-1)
        return col
