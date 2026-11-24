# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Brute-force DBSCAN without a CSR (the default brute route of identical mode since 2026-11-20; was the
experiment MOJOLEARN_DBSCAN_FAST_BRUTE_EDGE_FREE; its _OFF old route was deleted 2026-11-23).

WHERE THE BRUTE TIME GOES (the reference route, `runner.mojo`, with
`sparse_rbc_mode` off). `compute_batch_size` clamps the brute batch to
`MAX_LABEL // n_rows` rows (`dbscan.mojo`), because the dense `batch x N`
byte adjacency is int32-indexed: 2,147 rows at 1e6 rows, so ~466 batches.
Per batch the route runs

  - loop 1: one distance pass of the batch against all N rows, writing the
    `batch x N` byte adjacency (2 GB) and the degrees, plus a host readback;
  - loop 2: the distance pass AGAIN (every batch but 0), `adj_graph_run`
    (scan + compaction of the 2 GB adjacency into a CSR), `weak_cc_batched`
    (initialises and min-propagates all N labels, several passes, each pass a
    host round trip) and `merge_labels_run` (more passes over all N);
  - the border pass: for every batch holding a labelled border row, the
    distance pass a THIRD time and the CSR again.

So the cost is three N x N distance passes plus ~466 x (passes x N) label
work and ~466 x 2 GB of adjacency writes and compactions. The label work and
the per-batch round trips are what grow with N / batch; on a dense eps (most
of the rows within eps of a large share of the others) each batch's CSR also
holds ~batch x N edges, the most it can.

WHAT THIS DOES INSTEAD: the edge-free route (`edge_free.mojo`, lane
dbscan-taxi-speed) on the brute predicate.

  A. Degrees: `bef_eps_kernel[WRITE_BITS=False]`, the reference kernel's tile
     and accumulation, counting only. The core mask is `deg >= min_pts`.
  B. Components: per memory-sized query range, `bef_eps_kernel[WRITE_BITS=
     True]` writes the range's bit matrix (one bit per (query, column), 64
     consecutive column ids per word, word-major `word * n_queries + q`).
     The union-find of `edge_free.mojo` (the same `_ef_find` / `_ef_union`:
     larger root linked under the smaller, so a root is its set's smallest
     row id) consumes it in query CHUNKS: before each chunk the parents are
     compressed and every 64-row block gets its single root (`broot`); a
     core query skips a block already on its root and makes ONE union with a
     single-root block on another root. Chunks start at `BEF_CHUNK0` rows
     and double, so the first rows of a dense region (every block still
     multi-root) are few, and once the region has collapsed onto one root a
     row reads it as N / 64 words, not N edges.
  C. Border lists of the non-core rows (fewer than `min_pts` neighbours),
     sized by a scan of their degrees, filled during B.
  D. `ef_label_kernel`: core -> root + 1, non-core -> smallest root + 1 over
     its core neighbours, `MAX_LABEL` (noise) when none.

THE SAME LABELS. The bit for (q, j) is the reference kernel's `adj` byte for
(q, j): the same Policy4x4 tile, the same `kk` walk over the same K-blocks,
the same `_eps_acc` and the same `<= thresh` compare, so the same float
accumulator decides it (the accumulation below is that kernel's statement
for statement and must change with it). The degrees are those bits' counts,
so the core mask is the reference's. Union-find's final partition does not
depend on the order of the unions, a root is its component's smallest id, so
a core label is `min id + 1`, which is `weak_cc`'s fixed point; the border
rule is DEVIATION 5130's. `_dbscan_finish` runs the same relabel. No float is
folded in B-D. The labels equal the reference route's for every batching;
the only reported difference is `n_iter_` (0 here: no propagation passes).

COST. Two distance passes instead of up to three; no byte adjacency, no CSR,
no per-batch label or merge pass, no host round trip inside the loops (one
scalar readback sizes the border lists). Bit traffic: N / 8 bytes written
and read per query row. Union work: bounded by the multi-root blocks a row
meets.

MEMORY. The bit matrix of all N rows is N * N / 8 bytes (125 GB at 1e6 rows)
and is never held whole: a range holds `rows = DB_BM_SHARE_PCT % of the free
device memory / (8 * ceil(N / 64))` query rows, the same share the ball-cover
bit matrix takes (`runner.mojo`), floored at 1 and capped by the launch grid
(`BEF_MAX_RANGE_ROWS`). Size and device reasoning only, for any N.

WHEN IT RUNS (`brute_edge_free_applies`): the define, IDENTICAL on NVIDIA and
AMD (where the ball-cover edge-free route runs), the brute arm, a metric the
fused kernel serves (L2, L1, cosine; precomputed keeps its own kernel),
unweighted, more than one reference batch (a one-batch fit makes one
distance pass and one label pass already), and the border lists int32-sized.
"""

from std.atomic import Atomic
from std.bit import count_trailing_zeros
from std.time import perf_counter_ns
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.primitives.warp import shuffle_xor
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from checks.numerics import ftz_simd
from dbscan.impl.corepoints.compute import core_points_compute
from dbscan.impl.edge_free import (
    EF_MULTI,
    EF_NO_CORE,
    EF_TPB,
    IDN_DBSCAN_EDGE_FREE,
    _ef_find,
    _ef_union,
    edge_free_admits,
    ef_border_deg_kernel,
    ef_compress_kernel,
    ef_init_parent_kernel,
    ef_label_kernel,
)
from dbscan.impl.neighbors.epsilon_neighborhood import (
    DBSCAN_METRIC_L1,
    DBSCAN_METRIC_PRECOMPUTED,
    EPS_MBLK,
    EPS_NBLK,
    EPS_THREADS,
    GEMM_ACC_COLS_PER_TH,
    GEMM_ACC_ROWS_PER_TH,
    GEMM_ACC_TH_COLS,
    GEMM_ACC_TH_ROWS,
    GEMM_KBLK,
    GEMM_MBLK,
    GEMM_NBLK,
    GEMM_SMEM_PAGE_X,
    GEMM_SMEM_PAGE_Y,
    GEMM_SMEM_STRIDE,
    _eps_acc,
    dbscan_metric_threshold,
)
from dbscan.impl.brute_eps_filter import (
    DBF_G_SHARE_PCT,
    dbf_band_scale,
    dbf_filter_pays,
    dbf_filter_serves,
    dbf_gemm_rows,
    dbf_prepare,
    dbf_range_launch,
)
from gemm.checks.gemm_identical import GemmWorkspace
from neighbors.impl.ball_cover.scan import rbc_exclusive_scan_launch


#: The route of identical mode (IDN_DBSCAN_EDGE_FREE) since 2026-11-20. A fit `brute_edge_free_applies` does not
#: admit (weights, precomputed, one batch, the ball cover, fast mode) keeps the reference brute route (byte
#: adjacency, CSR, weak_cc, merge).
# A/B round 3, 2026-11-20 (lane fx-verdicts5; NVIDIA L40S host 2ef132a6de04, wheel 28a780604c and its B set, base and
# arm back to back in one pod job, tag nvc-ab-20261120T063324Z-c630ad; one scored run per arm; labels, quality and
# digests equal on every cell). Base -> arm ms, fit: classical/dbscan taxi 1,000,000 x 11 1,040,400 -> 351,860
# (0.338; 36 clusters, noise 0.000174); taxi 250,000 rows (the neighbouring size) 37,950 -> 19,701 (0.519);
# istella 1,000,000 x 220 597,190 -> 591,530 (0.991: no change on that row). PROMOTED.
# The mechanism is general (per-batch label work and 2 GB adjacency compactions removed; range sized from free
# memory), no shape or dataset key. Not timed: the non-board make_blobs 1,000,000 x 16 case of the EXPERIMENTS row.
# The _OFF switch for the admitted fits is gone (decided from this A/B, lane ab-prep-classical).
# TOMBSTONE: MOJOLEARN_DBSCAN_FAST_BRUTE_EDGE_FREE_OFF (decided: winner promoted, A/B round 3) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_DBSCAN_FAST_BRUTE_EDGE_FREE_OFF.patch; record in docs/TOMBSTONES.md.
comptime DBSCAN_BRUTE_EDGE_FREE = IDN_DBSCAN_EDGE_FREE

#: One column tile of the reference kernel is one 64-bit word: `EPS_NBLK`
#: columns, thread column `acccolid + j * GEMM_ACC_TH_COLS` -> bit `acccolid
#: + 16 j`. Asserted where it is used.
comptime BEF_WORD_COLS = 64
#: The row tiles of a range sit on grid.y (65,535 at most), so a range holds
#: at most this many query rows.
comptime BEF_MAX_RANGE_ROWS = 65535 * EPS_MBLK
#: The first union chunk of a range (rows), doubled per chunk. Two blocks of
#: `EF_TPB`: small enough that the rows meeting a dense region while every
#: block is still multi-root are few, and the O(N) compress + block-root
#: launches between chunks are ~log2(range / BEF_CHUNK0) per range.
comptime BEF_CHUNK0 = 2 * EF_TPB
#: The bit matrix's share of the free device memory, in percent (the runner's
#: `DB_BM_SHARE_PCT` for the ball-cover bit matrix).
comptime BEF_BM_SHARE_PCT = 25


def bef_words(n_rows: Int) -> Int:
    """64-bit words per query row: one per 64 consecutive column ids."""
    return (n_rows + BEF_WORD_COLS - 1) // BEF_WORD_COLS


def brute_edge_free_applies(
    brute: Bool,
    metric: Int,
    has_weights: Bool,
    n_rows: Int,
    batch: Int,
    min_pts: Int,
) -> Bool:
    """The one admission test, read by `dbscan.mojo` (which then allocates
    no byte adjacency) and by the runner (which then takes this route)."""
    comptime if DBSCAN_BRUTE_EDGE_FREE:
        return (
            brute
            and metric != DBSCAN_METRIC_PRECOMPUTED
            and not has_weights
            and batch < n_rows
            and edge_free_admits(n_rows, min_pts)
        )
    else:
        return False


def bef_eps_kernel[
    metric: Int, WRITE_BITS: Bool
](
    bm: MutPointer[UInt64, MutAnyOrigin],
    deg: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    thresh_in: Float32,
):
    """`eps_unexp_neigh_kernel[metric]`'s tile and accumulation, emitting
    bits (`WRITE_BITS`) or degrees (not `WRITE_BITS`, `deg` zeroed first)
    instead of the byte adjacency. `x` is the range (`m` rows), `y` the
    dataset (`n` rows). grid = (ceil(n / 64) column words, ceil(m / 64) row
    tiles), block = `EPS_THREADS`. Every (word, query) cell has one writer.
    """
    comptime assert EPS_NBLK == BEF_WORD_COLS, "one column tile is one word"
    comptime assert GEMM_ACC_COLS_PER_TH == 4 and GEMM_ACC_TH_COLS == 16, (
        "bit acccolid + 16 j for j < 4"
    )
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)

    var tid = Int(thread_idx.x)
    var accrowid = tid // GEMM_ACC_TH_COLS
    var acccolid = tid % GEMM_ACC_TH_COLS

    var word = Int(block_idx.x)
    var m0 = Int(block_idx.y) * GEMM_MBLK
    var n0 = word * GEMM_NBLK

    var sx = stack_allocation[
        GEMM_SMEM_PAGE_X,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var sy = stack_allocation[
        GEMM_SMEM_PAGE_Y,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var acc0 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
    var acc1 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
    var acc2 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
    var acc3 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)

    # The reference kernel's `loop()`, statement for statement
    # (`epsilon_neighborhood.mojo`, eps_unexp_neigh_kernel): the same float
    # accumulator decides every bit. Change both together.
    var kt = 0
    while kt < k:
        var e = tid
        while e < GEMM_MBLK * GEMM_KBLK:
            var r = e // GEMM_KBLK
            var c = e % GEMM_KBLK
            var v = Float32(0.0)
            if m0 + r < m and kt + c < k:
                v = x.unsafe_load((m0 + r) * k + kt + c)
            sx[r * GEMM_SMEM_STRIDE + c] = v
            e += EPS_THREADS
        e = tid
        while e < GEMM_NBLK * GEMM_KBLK:
            var r2 = e // GEMM_KBLK
            var c2 = e % GEMM_KBLK
            var v2 = Float32(0.0)
            if n0 + r2 < n and kt + c2 < k:
                v2 = y.unsafe_load((n0 + r2) * k + kt + c2)
            sy[r2 * GEMM_SMEM_STRIDE + c2] = v2
            e += EPS_THREADS
        barrier()

        for kk in range(GEMM_KBLK):
            var regy = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
            for j in range(GEMM_ACC_COLS_PER_TH):
                regy[j] = sy[
                    (acccolid + j * GEMM_ACC_TH_COLS) * GEMM_SMEM_STRIDE + kk
                ]
            regy = ftz_simd[GEMM_ACC_COLS_PER_TH](regy)
            acc0 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](
                acc0,
                sx[(accrowid + 0 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk],
                regy,
            )
            acc1 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](
                acc1,
                sx[(accrowid + 1 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk],
                regy,
            )
            acc2 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](
                acc2,
                sx[(accrowid + 2 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk],
                regy,
            )
            acc3 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](
                acc3,
                sx[(accrowid + 3 * GEMM_ACC_TH_ROWS) * GEMM_SMEM_STRIDE + kk],
                regy,
            )
        barrier()
        kt += GEMM_KBLK

    # --- epilogue: the reference's radius test, then bits or counts --------
    var startx = m0 + accrowid
    var starty = n0 + acccolid
    for i in range(GEMM_ACC_ROWS_PER_TH):
        var xid = startx + i * GEMM_ACC_TH_ROWS
        var lo = UInt32(0)
        var hi = UInt32(0)
        var s = Int32(0)
        for j in range(GEMM_ACC_COLS_PER_TH):
            var yid = starty + j * GEMM_ACC_TH_COLS
            var a = acc0[j]
            if i == 1:
                a = acc1[j]
            elif i == 2:
                a = acc2[j]
            elif i == 3:
                a = acc3[j]
            if a <= thresh_in and xid < m and yid < n:
                s += Int32(1)
                if j < 2:
                    lo |= UInt32(1) << UInt32(acccolid + j * GEMM_ACC_TH_COLS)
                else:
                    hi |= UInt32(1) << UInt32(
                        acccolid + (j - 2) * GEMM_ACC_TH_COLS
                    )
        # The 16 lanes of one accrowid are one aligned 16-lane group; every
        # lane reaches every shuffle (comptime bound, nothing conditional).
        comptime if WRITE_BITS:
            var off = 1
            while off < GEMM_ACC_TH_COLS:
                lo |= shuffle_xor(lo, UInt32(off))
                hi |= shuffle_xor(hi, UInt32(off))
                off *= 2
            if acccolid == 0 and xid < m:
                bm.unsafe_store(
                    word * m + xid,
                    (hi.cast[DType.uint64]() << UInt64(32))
                    | lo.cast[DType.uint64](),
                )
        else:
            var off2 = 1
            while off2 < GEMM_ACC_TH_COLS:
                s += shuffle_xor(s, UInt32(off2))
                off2 *= 2
            if acccolid == 0 and xid < m:
                _ = Atomic.fetch_add(deg.unsafe_offset(xid), s)


def bef_block_root_kernel(
    core: MutPointer[UInt8, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    broot: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """One thread per 64-row block of row ids: its single root, `EF_MULTI`
    or `EF_NO_CORE`. `parent` is compressed, no union in flight."""
    var w = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if w * BEF_WORD_COLS >= n:
        return
    var lo = Int32(2147483647)
    var hi = Int32(-1)
    var j_end = min(w * BEF_WORD_COLS + BEF_WORD_COLS, n)
    for j in range(w * BEF_WORD_COLS, j_end):
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
    broot.unsafe_store(w, cell)


def bef_union_kernel(
    bm: MutPointer[UInt64, MutAnyOrigin],
    n_queries_in: Int32,
    q0_in: Int32,
    n_chunk_in: Int32,
    start_in: Int32,
    n_words_in: Int32,
    core: MutPointer[UInt8, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin],
    broot: MutPointer[Int32, MutAnyOrigin],
    boff: MutPointer[Int32, MutAnyOrigin],
    blist: MutPointer[Int32, MutAnyOrigin],
):
    """`ef_union_kernel` over the brute bit matrix: one thread per query row
    `q0 + t` of the range's chunk; column id = 64 word + bit. A core row
    unions with the core rows its bits name (skipping blocks on its root,
    one union per single-root block); a non-core row copies its neighbour
    ids to its border list."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(n_chunk_in):
        return
    var nq = Int(n_queries_in)
    var q = Int(q0_in) + t
    var g = Int(start_in) + q
    var gi = Int32(g)
    var is_core = core.unsafe_load(g) != 0
    var pos = 0
    if not is_core:
        pos = Int(boff.unsafe_load(g))
    for wd in range(Int(n_words_in)):
        var w = bm.unsafe_load(wd * nq + q)
        if w == UInt64(0):
            continue
        var base = wd * BEF_WORD_COLS
        if not is_core:
            while w != UInt64(0):
                var c = Int(count_trailing_zeros(w))
                w &= w - UInt64(1)
                blist.unsafe_store(pos, Int32(base + c))
                pos += 1
            continue
        var br = broot.unsafe_load(wd)
        if br == EF_NO_CORE:
            continue
        if br >= Int32(0):
            if _ef_find(parent, br) == _ef_find(parent, gi):
                continue
            while w != UInt64(0):
                var c = Int(count_trailing_zeros(w))
                w &= w - UInt64(1)
                var j = base + c
                if core.unsafe_load(j) != 0:
                    _ef_union(parent, gi, Int32(j))
                    break
            continue
        while w != UInt64(0):
            var c = Int(count_trailing_zeros(w))
            w &= w - UInt64(1)
            var j = base + c
            if core.unsafe_load(j) != 0:
                _ef_union(parent, gi, Int32(j))


def bef_filtered_fit[
    metric: Int
](
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut core: DeviceBuffer[DType.uint8],
    mut labels: DeviceBuffer[DType.int32],
    n_rows: Int,
    n_features: Int,
    eps: Float64,
    min_pts: Int,
    phase_timing: Bool,
) raises -> Int:
    """`dbscan_brute_edge_free_fit` with both distance passes on the GEMM filter of `brute_eps_filter.mojo`
    (the default where `dbf_filter_pays`; lane ab-outcome-4). Steps A-D, the union-find, the border
    lists and the labels are that function's, statement for statement; only the launch that fills the degrees
    (A) and the range's bit matrix (B) differs, and it writes the same degrees and the same words (the filter
    decides a pair only where its bound proves the reference accumulator's compare; the rest run `_eps_acc`).

    Returns the number of bit-matrix ranges, or 0 with NOTHING written when the centred copy or one GEMM range
    does not fit the device (the caller then runs the reference passes).

    ONE ROUTE (Andrew, 2026-11-24: "the route should be one thing only", optimised for large inputs; lane
    ab-outcome-3 / o3-route). Every fit whose buffers fit runs the filter for both passes; nothing times or
    declines it (the timed probe of lane ab-outcome-1, which ran the filter and `bef_eps_kernel` on the first
    rows and kept the faster, is removed). The fixed costs the filter carries per pair are in the module
    docstring of `brute_eps_filter.mojo`.

    The union side (lane ab-outcome-1, 2026-11-24): the filter's buffers took memory before the bit-matrix plan,
    so the bit ranges were smaller and more of them ran, and every range reruns the union rounds over all n
    rows (`ef_compress_kernel` and `bef_block_root_kernel` per doubling chunk). The bit-matrix plan is now
    made FIRST, from the same BEF_BM_SHARE_PCT of the free memory the reference fit uses, and the filter sizes
    its centred copy and its GEMM range from what is left: the same number of bit ranges as the reference.

    Launch sites (the loops they sit in): `dbf_prepare` once a fit; `dbf_range_launch` once per GEMM-sized
    query range of A and of B (driver loops over the memory plan, about n / gemm_rows iterations each). Nothing
    is added per row or per pair on the host."""
    var thresh = dbscan_metric_threshold(metric, eps)
    var n_words = bef_words(n_rows)
    var blocks_n = (n_rows + EF_TPB - 1) // EF_TPB
    var t0 = perf_counter_ns()

    # --- the bit-matrix plan first: the reference fit's share of the free memory (the union side, above) -----
    var free0 = Int(ctx.get_memory_info()[0])
    var range_rows = (free0 * BEF_BM_SHARE_PCT // 100) // (n_words * 8)
    if range_rows < 1:
        range_rows = 1
    if range_rows > BEF_MAX_RANGE_ROWS:
        range_rows = BEF_MAX_RANGE_ROWS
    if range_rows > n_rows:
        range_rows = n_rows
    var bm_bytes = range_rows * n_words * 8
    var free_f = free0 - bm_bytes
    if free_f < 0:
        free_f = 0

    # --- the filter's plan: the centred copy, then one GEMM range, from what the bit matrix leaves ----------
    if 4 * n_rows * (n_features + 1) > free_f * DBF_G_SHARE_PCT // 100:
        if phase_timing:
            print(
                "PHASE plan.brute_eps_filter not taken: the centred copy does not fit "
                + String(DBF_G_SHARE_PCT) + " percent of the " + String(free_f)
                + " free bytes the bit-matrix plan leaves"
            )
        return 0
    var xc = ctx.enqueue_create_buffer[DType.float32](n_rows * n_features)
    var nc = ctx.enqueue_create_buffer[DType.float32](n_rows)
    ctx.synchronize()
    var free_g = Int(ctx.get_memory_info()[0]) - bm_bytes
    if free_g < 0:
        free_g = 0
    var g_rows = dbf_gemm_rows(free_g, n_rows, n_features)
    if g_rows < 1:
        if phase_timing:
            print(
                "PHASE plan.brute_eps_filter not taken: no GEMM range (one query row against "
                + String(n_rows) + " rows, with the GEMM's scratch) fits the device"
            )
        _ = xc^
        _ = nc^
        return 0
    dbf_prepare(ctx, x, xc, nc, n_rows, n_features)
    var g = ctx.enqueue_create_buffer[DType.float32](g_rows * n_rows)
    var gws = GemmWorkspace(ctx)
    # the fit never counts rechecked pairs (COUNT_RECHECKS off): one word stands in for the count
    var unc_none = ctx.enqueue_create_buffer[DType.int32](1)
    var band = dbf_band_scale(n_features)

    # --- A: degrees over every row -----------------------------------------
    var deg = ctx.enqueue_create_buffer[DType.int32](n_rows)
    # the degree instantiation writes no bits; one word stands in for `bm`
    var bm_none = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_memset(deg, Int32(0))
    var st_a = 0
    while st_a < n_rows:  # driver-loop: over GEMM-sized query ranges (the plan, no row data)
        var np_f = min(n_rows - st_a, g_rows)
        dbf_range_launch[metric, False](
            ctx, gws, g, xc, nc, x, bm_none, deg, unc_none, st_a, np_f,
            n_rows, n_features, np_f, 0, thresh, band,
        )
        st_a += np_f
    core_points_compute(ctx, deg, core, min_pts, 0, n_rows)
    if phase_timing:
        print(
            "PHASE plan.brute_eps_filter gemm_rows " + String(g_rows)
            + " band " + String(Float64(band))
        )

    # --- C (sizing): the border lists of the non-core rows -----------------
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
    var t_a = perf_counter_ns()
    var n_border = Int(h_tot.unsafe_ptr().unsafe_load(0))
    var blist = ctx.enqueue_create_buffer[DType.int32](
        n_border if n_border > 0 else 1
    )
    var parent = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var broot = ctx.enqueue_create_buffer[DType.int32](n_words)
    ctx.enqueue_function[ef_init_parent_kernel](
        parent.unsafe_ptr(), Int32(n_rows),
        grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
    )
    ctx.synchronize()

    # --- B: the bit matrix of the plan made first, then bits + unions range by range ----------------------
    # The plan reserved `bm_bytes` before the filter's buffers; if the free memory measured now does not hold it
    # (another tenant took memory meanwhile), the reference's share of what is free now.
    var bm_free = Int(ctx.get_memory_info()[0])
    if bm_bytes > bm_free:
        range_rows = (bm_free * BEF_BM_SHARE_PCT // 100) // (n_words * 8)
        if range_rows < 1:
            range_rows = 1
        if range_rows > BEF_MAX_RANGE_ROWS:
            range_rows = BEF_MAX_RANGE_ROWS
        if range_rows > n_rows:
            range_rows = n_rows
    var bm = ctx.enqueue_create_buffer[DType.uint64](range_rows * n_words)
    var n_ranges = (n_rows + range_rows - 1) // range_rows
    if phase_timing:
        print(
            "PHASE plan.brute_edge_free words " + String(n_words)
            + " range_rows " + String(range_rows) + " ranges "
            + String(n_ranges) + " border_ids " + String(n_border)
        )
    var root_blocks = (n_words + EF_TPB - 1) // EF_TPB
    for kr in range(n_ranges):  # driver-loop: over memory-sized query ranges (the plan, no row data)
        var st = kr * range_rows
        var np = min(n_rows - st, range_rows)
        var s0 = 0
        while s0 < np:  # driver-loop: over GEMM-sized query sub-ranges of the bit range (the plan, no row data)
            var ns = min(np - s0, g_rows)
            dbf_range_launch[metric, True](
                ctx, gws, g, xc, nc, x, bm, deg, unc_none, st + s0, ns,
                n_rows, n_features, np, s0, thresh, band,
            )
            s0 += ns
        var q0 = 0
        var chunk = BEF_CHUNK0
        while q0 < np:  # driver-loop: over doubling union chunks of the range (log2(range / BEF_CHUNK0) + 1 of them)
            var nc_rows = min(np - q0, chunk)
            ctx.enqueue_function[ef_compress_kernel](
                parent.unsafe_ptr(), Int32(n_rows),
                grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
            )
            ctx.enqueue_function[bef_block_root_kernel](
                core.unsafe_ptr(), parent.unsafe_ptr(), broot.unsafe_ptr(),
                Int32(n_rows),
                grid_dim=(root_blocks, 1, 1), block_dim=(EF_TPB, 1, 1),
            )
            ctx.enqueue_function[bef_union_kernel](
                bm.unsafe_ptr(), Int32(np), Int32(q0), Int32(nc_rows), Int32(st),
                Int32(n_words), core.unsafe_ptr(), parent.unsafe_ptr(),
                broot.unsafe_ptr(), boff.unsafe_ptr(), blist.unsafe_ptr(),
                grid_dim=((nc_rows + EF_TPB - 1) // EF_TPB, 1, 1),
                block_dim=(EF_TPB, 1, 1),
            )
            q0 += nc_rows
            chunk *= 2
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
            "PHASE brute_edge_free ranges " + String(n_ranges) + " degrees "
            + String(Float64(t_a - t0) / 1.0e6) + " unions "
            + String(Float64(t_b - t_a) / 1.0e6) + " labels "
            + String(Float64(perf_counter_ns() - t_b) / 1.0e6)
        )
    _ = deg^
    _ = bm_none^
    _ = bdeg^
    _ = boff^
    _ = blist^
    _ = parent^
    _ = broot^
    _ = bm^
    _ = h_tot^
    _ = xc^
    _ = nc^
    _ = g^
    _ = gws^
    _ = unc_none^
    return n_ranges


def dbscan_brute_edge_free_fit[
    metric: Int
](
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut core: DeviceBuffer[DType.uint8],
    mut labels: DeviceBuffer[DType.int32],
    n_rows: Int,
    n_features: Int,
    eps: Float64,
    min_pts: Int,
    phase_timing: Bool,
) raises -> Int:
    """Steps A-D of the module docstring. Writes `core` and the merged-stage
    `labels`; returns the number of bit-matrix ranges."""
    # The GEMM filter (`brute_eps_filter.mojo`; the default since 2026-11-24, lane ab-outcome-4, where the
    # per-pair cost rule `dbf_filter_pays` holds: from DBF_MIN_FEATURES features up): the two distance passes as
    # a GEMM filter with this kernel's own recheck; the same degrees, bits and labels. It returns 0, with nothing
    # written, when its buffers do not fit the device, and the passes below run. One integer compare per fit.
    comptime if IDN_DBSCAN_EDGE_FREE and metric != DBSCAN_METRIC_L1:
        if dbf_filter_serves(metric, n_features) and dbf_filter_pays(n_features):
            var filtered_ranges = bef_filtered_fit[metric](
                ctx, x, core, labels, n_rows, n_features, eps, min_pts,
                phase_timing,
            )
            if filtered_ranges > 0:
                return filtered_ranges
    var thresh = dbscan_metric_threshold(metric, eps)
    var n_words = bef_words(n_rows)
    var blocks_n = (n_rows + EF_TPB - 1) // EF_TPB
    var t0 = perf_counter_ns()

    # --- A: degrees over every row (ranges sized by the launch grid only) --
    var deg = ctx.enqueue_create_buffer[DType.int32](n_rows)
    # the degree instantiation writes no bits; one word stands in for `bm`
    var bm_none = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_memset(deg, Int32(0))
    var deg_range = min(n_rows, BEF_MAX_RANGE_ROWS)
    var st_a = 0
    while st_a < n_rows:  # driver-loop: over memory/grid-sized query ranges (the plan, no row data)
        var np_a = min(n_rows - st_a, deg_range)
        var xa = x.create_sub_buffer[DType.float32](
            st_a * n_features, np_a * n_features
        )
        var da = deg.create_sub_buffer[DType.int32](st_a, np_a)
        ctx.enqueue_function[bef_eps_kernel[metric, False]](
            bm_none.unsafe_ptr(), da.unsafe_ptr(),
            xa.unsafe_ptr(), x.unsafe_ptr(), Int32(np_a), Int32(n_rows),
            Int32(n_features), thresh,
            grid_dim=(n_words, (np_a + EPS_MBLK - 1) // EPS_MBLK, 1),
            block_dim=(EPS_THREADS, 1, 1),
        )
        st_a += np_a
    core_points_compute(ctx, deg, core, min_pts, 0, n_rows)

    # --- C (sizing): the border lists of the non-core rows -----------------
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
    var t_a = perf_counter_ns()
    var n_border = Int(h_tot.unsafe_ptr().unsafe_load(0))
    var blist = ctx.enqueue_create_buffer[DType.int32](
        n_border if n_border > 0 else 1
    )
    var parent = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var broot = ctx.enqueue_create_buffer[DType.int32](n_words)
    ctx.enqueue_function[ef_init_parent_kernel](
        parent.unsafe_ptr(), Int32(n_rows),
        grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
    )
    ctx.synchronize()

    # --- B: the memory plan, then bits + unions range by range -------------
    # Bytes per query row: 8 * n_words (= N / 8). The range takes
    # BEF_BM_SHARE_PCT of the free memory measured now, after every other
    # buffer of the fit exists; any N, any device.
    var bm_free = Int(ctx.get_memory_info()[0])
    var range_rows = (bm_free * BEF_BM_SHARE_PCT // 100) // (n_words * 8)
    if range_rows < 1:
        range_rows = 1
    if range_rows > BEF_MAX_RANGE_ROWS:
        range_rows = BEF_MAX_RANGE_ROWS
    if range_rows > n_rows:
        range_rows = n_rows
    var bm = ctx.enqueue_create_buffer[DType.uint64](range_rows * n_words)
    var n_ranges = (n_rows + range_rows - 1) // range_rows
    if phase_timing:
        print(
            "PHASE plan.brute_edge_free words " + String(n_words)
            + " range_rows " + String(range_rows) + " ranges "
            + String(n_ranges) + " border_ids " + String(n_border)
        )
    var root_blocks = (n_words + EF_TPB - 1) // EF_TPB
    for kr in range(n_ranges):  # driver-loop: over memory-sized query ranges (the plan, no row data)
        var st = kr * range_rows
        var np = min(n_rows - st, range_rows)
        var xb = x.create_sub_buffer[DType.float32](
            st * n_features, np * n_features
        )
        ctx.enqueue_function[bef_eps_kernel[metric, True]](
            bm.unsafe_ptr(), deg.unsafe_ptr(), xb.unsafe_ptr(), x.unsafe_ptr(),
            Int32(np), Int32(n_rows), Int32(n_features), thresh,
            grid_dim=(n_words, (np + EPS_MBLK - 1) // EPS_MBLK, 1),
            block_dim=(EPS_THREADS, 1, 1),
        )
        var q0 = 0
        var chunk = BEF_CHUNK0
        while q0 < np:  # driver-loop: over doubling union chunks of the range (log2(range / BEF_CHUNK0) + 1 of them)
            var nc = min(np - q0, chunk)
            ctx.enqueue_function[ef_compress_kernel](
                parent.unsafe_ptr(), Int32(n_rows),
                grid_dim=(blocks_n, 1, 1), block_dim=(EF_TPB, 1, 1),
            )
            ctx.enqueue_function[bef_block_root_kernel](
                core.unsafe_ptr(), parent.unsafe_ptr(), broot.unsafe_ptr(),
                Int32(n_rows),
                grid_dim=(root_blocks, 1, 1), block_dim=(EF_TPB, 1, 1),
            )
            ctx.enqueue_function[bef_union_kernel](
                bm.unsafe_ptr(), Int32(np), Int32(q0), Int32(nc), Int32(st),
                Int32(n_words), core.unsafe_ptr(), parent.unsafe_ptr(),
                broot.unsafe_ptr(), boff.unsafe_ptr(), blist.unsafe_ptr(),
                grid_dim=((nc + EF_TPB - 1) // EF_TPB, 1, 1),
                block_dim=(EF_TPB, 1, 1),
            )
            q0 += nc
            chunk *= 2
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
            "PHASE brute_edge_free ranges " + String(n_ranges) + " degrees "
            + String(Float64(t_a - t0) / 1.0e6) + " unions "
            + String(Float64(t_b - t_a) / 1.0e6) + " labels "
            + String(Float64(perf_counter_ns() - t_b) / 1.0e6)
        )
    _ = deg^
    _ = bm_none^
    _ = bdeg^
    _ = boff^
    _ = blist^
    _ = parent^
    _ = broot^
    _ = bm^
    _ = h_tot^
    return n_ranges
