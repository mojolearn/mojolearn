# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The brute epsilon pass of DBSCAN as a GEMM FILTER plus the exact kernel's recheck
(experiment `-D MOJOLEARN_DBSCAN_FAST_BRUTE_EPS_FILTER`, default OFF; lane sp-dbscan-filter, 2026-11-20).

WHY. `bef_eps_kernel` (`brute_edge_free.mojo`) accumulates `(x - y)^2` per pair over the features in a 4x4
register tile: about 0.74 TFLOP/s on the L40S on the 1,000,000 x 220 board row, where the identical GEMM
(`gemm/checks/gemm_identical.mojo`, tuned 8x8 register tiles) runs 19.5-25 TFLOP/s. The squared distance is
`|a|^2 + |b|^2 - 2 a.b`, and `a.b` for a whole range of query rows against every row is one GEMM.

THE EXPANDED VALUE NEVER DECIDES A PAIR THE EXACT KERNEL COULD DECIDE DIFFERENTLY. It rounds differently from
the unexpanded chain, so it is used only as a filter with a PROVEN error bound: a pair whose expanded value is
below `thresh` by more than the bound is inside, one above by more than the bound is outside, and every other
pair (and every pair of a row the bound does not cover) is recomputed by `_eps_acc`, the reference kernel's own
statements in the reference kernel's order. So every adjacency bit and every degree equals `bef_eps_kernel`'s,
and labels, core mask and the bits of `identical` are unchanged by construction; only the time moves.
`dbscan/checks/dbscan_eps_filter_check.mojo` compares the two passes word for word on adversarial inputs.

THE BOUND. `u = 2^-24` (FP32, round to nearest), `g_n = n u / (1 - n u)`, `k` = features. The filter works on
CENTRED rows `a = fl(x_q - mu)`, `b = fl(x_j - mu)` (`mu` = the column means, any float vector: a translation
does not change a distance, and it shrinks the norms to the data's spread, which is what the bound is relative
to). `Na = |a|^2`, `Nb = |b|^2` exactly; `D' = |a - b|^2`; `D = |x_q - x_j|^2` (the true value); `A` = the
reference kernel's accumulator for the pair; `approx = fl(fl(na + nb) - 2 g)` with `na`, `nb` the computed
norms and `g` the GEMM's cell.

  1. `|g - a.b| <= g_(k+1) sum|a_c b_c| <= g_(k+1) (Na + Nb) / 2` for ANY summation order and with or without
     fused multiply-adds (a product meets at most one multiply rounding and k additions), so `2 g` is within
     `(k + 1) u (Na + Nb)`.
  2. `|na - Na| <= g_(k+1) Na`, the same for `nb`: `(k + 1) u (Na + Nb)`.
  3. `fl(na + nb)`: `u (Na + Nb)`. The final subtraction: `u |.| <= 2 u (Na + Nb)` since `D' <= 2 (Na + Nb)`.
     So `|approx - D'| <= (2k + 5) u (Na + Nb)`.
  4. Centring rounds each coordinate once: `|(a_c - b_c) - (x_qc - x_jc)| <= u (|a_c| + |b_c|)`, so
     `|sqrt(D') - sqrt(D)| <= u (|a| + |b|)` and `|D' - D| <= 4 u (Na + Nb)`.
  5. The reference chain `diff = fl(x - y); acc = fma(diff, diff, acc)`: all terms are non-negative, a term
     meets at most `k + 2` roundings, so `|A - D| <= g_(k+2) D <= (2k + 4) u (Na + Nb)`.

  Sum, first order: `|approx - A| <= (4k + 13) u (Na + Nb)`. With `(k + 2) u <= 2^-8` (`k <= DBF_MAX_FEATURES`)
  every second-order term, the rounding of the band itself and the use of the COMPUTED norms in it stay below a
  factor `1 + 1/32`, so the band `M_k (na + nb)` with `M_k = (4k + 16) u (1 + 1/16)` (`dbf_band_scale`) covers
  the error. 220 features: `M = 5.7e-5`; 16 features: `5.1e-6`.

  Underflow and the flushes (`ftz` of the reference chain and of the GEMM's operands and accumulator, products
  below the normal range) add ABSOLUTE terms, each at most `2^-125 (1 + |a|_max + |b|_max)` per feature; the
  part proportional to the norms is far inside `M_k`, the rest is below `k 2^-122 <= 2^-106`.
  `DBF_ABS_SLACK = 2^-100` covers it.

  Overflow: a row whose centred squared norm is not finite or above `DBF_NORM_CAP = 2^100` is UNUSABLE: its
  centred row is stored as zeros (the GEMM never sees a non-finite operand, every cell of `g` is finite) and its
  norm as `DBF_NORM_UNUSABLE` (negative), and every pair with such a row takes the exact recheck. Below the cap
  every coordinate is under `2^50` and every sum under `2^120`: nothing overflows.

  The compares are safe under rounding: `fl` is monotone, so `fl(approx + band) < thresh` implies
  `approx + band < thresh` in the reals, hence `A < thresh` and the reference's `A <= thresh` holds; and
  `fl(approx - band) > thresh` implies `A > thresh`. Anything else, a NaN included, is rechecked.

WHAT RUNS. Once per fit: the column means, the centred rows `xc` (n x k floats) and their norms. Per query
range (sized so the range's `m x n` product and the GEMM's own scratch take `DBF_G_SHARE_PCT` of the free
device memory, and `m n` fits an int32): one `identical_gemm` (`xc[range] . xc^T`), then `dbf_filter_kernel`,
which has the reference kernel's tile and epilogue (64 x 64 pairs a block, 4 x 4 a thread, the same bit and
degree reductions) with the accumulate loop replaced by one load of `g` per pair.

WHEN IT IS NOT WORTH IT, DECIDED AT RUN TIME FROM THE DATA IN FRONT OF IT (lane ab-outcome-1, 2026-11-24;
`brute_edge_free.bef_filtered_fit`, THE COST RULE). The first A/B (nvc-ab-20261124T070727Z-69dac2, L40S) halved
a 220-feature fit (575 -> 296 s) and did not speed an 11-feature one (1.02 at 250,000 rows, not faster at full
size). Per pair the reference spends work proportional to k; the filter spends the GEMM's (also proportional
to k, at a far higher rate), a fixed round trip of the product through device memory, and the recheck of the
open pairs (a warp reruns its row's k features from global memory while any of its 32 lanes x 4 pairs is
open, so the open fraction costs about 128x its size in warps). Which side wins depends on k AND on how the
distances crowd the threshold, so the fit times the filter and `bef_eps_kernel` on the same probe rows of the
degrees pass (at most an eighth of the rows) and keeps the faster. The first range also counts its open pairs
(one scalar readback), printed with the probe's times under phase timing: the evidence a decline leaves. The
old fixed rule (decline above 1 / 256 open pairs) is replaced by the timing; it could not see the k-independent
costs. Cost reasoning only; the next A/B decides it.

SCOPE. The brute edge-free route of `identical` (NVIDIA and AMD), squared-L2 arithmetic (euclidean, and cosine
on its unit rows); L1 has no expanded form and keeps the reference kernel. Not carried to radius neighbours,
LOF, OPTICS or k-means here (after the A/B).
"""

from std.atomic import Atomic
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.primitives.warp import shuffle_xor
from max.gpu.primitives.block import sum as block_sum
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz_simd
from dbscan.impl.edge_free import IDN_DBSCAN_EDGE_FREE
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
    _eps_acc,
)
from gemm.checks.gemm_identical import (
    GemmWorkspace,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_NT


#: The switch: default OFF, A/B owed with the wave-3 sha (bench/results/paper_compare/owed-pre0867.json arm
#: dbscan-brute-eps-filter): the first A/B was split (module docstring), and the timed probe is the new form.
#: Read only on the brute edge-free route, which exists under IDENTICAL on NVIDIA and AMD.
comptime DBSCAN_BRUTE_EPS_FILTER = (
    is_defined["MOJOLEARN_DBSCAN_FAST_BRUTE_EPS_FILTER"]() and IDN_DBSCAN_EDGE_FREE
)

#: `(k + 2) u <= 2^-8`: the width up to which the second-order terms of the bound stay inside the band's 1/16.
comptime DBF_MAX_FEATURES = 65534
#: 2^100: the largest centred squared norm the bound serves (coordinates under 2^50, sums under 2^120).
comptime DBF_NORM_CAP = Float32(1.2676506002282294e30)
#: The stored norm of a row the bound does not serve; negative, so one ordinary compare finds it.
comptime DBF_NORM_UNUSABLE = Float32(-1.0)
#: 2^-100: the absolute part of the band (underflow and flush terms, see the module docstring).
comptime DBF_ABS_SLACK = Float32(7.888609052210118e-31)
#: The largest `m x n` product of one GEMM: the cell count travels as an Int32 in the GEMM's launches, and the
#: probe's uncertain count of one range is an Int32.
comptime DBF_MAX_CELLS = 2147483647
#: The product buffer's (and the centred copy's) share of the free device memory, in percent: the share the
#: bit matrix of the route takes (`BEF_BM_SHARE_PCT`).
comptime DBF_G_SHARE_PCT = 25
#: Row-strided partial sums of the column means (blocks along grid.x): enough blocks to fill any device.
comptime DBF_MEAN_PARTS = 1024
comptime DBF_TPB = 256


def dbf_filter_serves(metric: Int, n_features: Int) -> Bool:
    """Whether the filter's arithmetic and bound apply: the define, a squared-L2 accumulator, a width the bound
    covers. Memory and the data itself are decided by the fit."""
    comptime if not DBSCAN_BRUTE_EPS_FILTER:
        return False
    return (
        metric != DBSCAN_METRIC_L1
        and metric != DBSCAN_METRIC_PRECOMPUTED
        and n_features >= 1
        and n_features <= DBF_MAX_FEATURES
    )


def dbf_band_scale(n_features: Int) -> Float32:
    """`M_k = (4k + 16) u (1 + 1/16)`, `u = 2^-24`: the band relative to `na + nb` (module docstring)."""
    return Float32(
        Float64(4 * n_features + 16) * 1.0625 * 5.9604644775390625e-08
    )


def dbf_gemm_rows(free_bytes: Int, n_rows: Int, n_features: Int) -> Int:
    """Query rows of one GEMM range, or 0 when no range fits: the range's `rows x n_rows` product plus the
    GEMM's own scratch at that shape within `DBF_G_SHARE_PCT` of `free_bytes`, `rows x n_rows` an int32, the
    row tiles within the launch grid. Size and device reasoning only, for any N."""
    if n_rows < 1 or n_rows > DBF_MAX_CELLS:
        return 0
    var share = free_bytes * DBF_G_SHARE_PCT // 100
    var rows = share // (4 * n_rows)
    if rows > n_rows:
        rows = n_rows
    if rows > DBF_MAX_CELLS // n_rows:
        rows = DBF_MAX_CELLS // n_rows
    if rows > 65535 * EPS_MBLK:
        rows = 65535 * EPS_MBLK
    while rows > 1 and 4 * (
        rows * n_rows
        + identical_gemm_workspace_max_floats(rows, n_rows, n_features)
    ) > share:  # driver-loop: over halvings of the plan (no row data)
        rows = rows // 2
    if rows < 1:
        return 0
    if 4 * (
        rows * n_rows
        + identical_gemm_workspace_max_floats(rows, n_rows, n_features)
    ) > share:
        return 0
    # whole 128-row GEMM tiles where the range holds one (execution plan only)
    if rows >= 128:
        rows = (rows // 128) * 128
    return rows


def dbf_colsum_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
    parts_in: Int32,
):
    """`part[p * k + f]` = the sum of column `f` over rows `p, p + parts, ...`. grid = (parts, ceil(k /
    block)), one thread a column. A fixed order (a function of n, k and parts); the centre it feeds never
    reaches an answer, only how many pairs the filter decides."""
    var n = Int(n_in)
    var k = Int(k_in)
    var parts = Int(parts_in)
    var p = Int(block_idx.x)
    var f = Int(block_idx.y) * Int(block_dim.x) + Int(thread_idx.x)
    if f >= k:
        return
    var s = Float32(0.0)
    var r = p
    while r < n:
        s += x.unsafe_load(r * k + f)
        r += parts
    part.unsafe_store(p * k + f, s)


def dbf_colmean_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    mean: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
    parts_in: Int32,
):
    """`mean[f]` = the partial sums of column `f` added in ascending part order, over n. One thread a column."""
    var k = Int(k_in)
    var parts = Int(parts_in)
    var f = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if f >= k:
        return
    var t = Float32(0.0)
    for p in range(parts):
        t += part.unsafe_load(p * k + f)
    mean.unsafe_store(f, t / Float32(Int(n_in)))


def dbf_center_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    mean: MutPointer[Float32, MutAnyOrigin],
    xc: MutPointer[Float32, MutAnyOrigin],
    nc: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """`xc = x - mean` and `nc = |xc|^2`, one thread a row. A row whose centred norm is not finite or above
    `DBF_NORM_CAP` is stored as zeros with the norm `DBF_NORM_UNUSABLE` (the exponent bits, not only a float
    compare, find the non-finite ones)."""
    var n = Int(n_in)
    var k = Int(k_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= n:
        return
    var s = Float32(0.0)
    for c in range(k):
        var v = x.unsafe_load(r * k + c) - mean.unsafe_load(c)
        xc.unsafe_store(r * k + c, v)
        s += v * v
    var eb = bitcast[DType.uint32](s) & 0x7F800000
    if eb != 0x7F800000 and s <= DBF_NORM_CAP:
        nc.unsafe_store(r, s)
    else:
        for c in range(k):
            xc.unsafe_store(r * k + c, Float32(0.0))
        nc.unsafe_store(r, DBF_NORM_UNUSABLE)


def dbf_filter_kernel[
    metric: Int, WRITE_BITS: Bool
](
    bm: MutPointer[UInt64, MutAnyOrigin],
    deg: MutPointer[Int32, MutAnyOrigin],
    unc: MutPointer[Int32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    nc: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    start_in: Int32,
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    bm_rows_in: Int32,
    q_off_in: Int32,
    thresh_in: Float32,
    band_in: Float32,
    probe_in: Int32,
):
    """`bef_eps_kernel[metric, WRITE_BITS]`'s answer for the query rows `[start, start + m)` of the dataset
    against all `n` rows, from `g` (`m x n`, `g[q, j]` = centred row `start + q` . centred row `j`) and the
    centred norms `nc`, with the reference accumulation only for the pairs the band leaves open.

    grid = (ceil(n / 64) column words, ceil(m / 64) row tiles), block = `EPS_THREADS`: the reference tile, the
    same thread to (row, column) map. `WRITE_BITS`: the word of (query `q`, column word `w`) goes to
    `bm[w * bm_rows + q_off + q]` (the range's bit matrix holds `bm_rows` queries; this launch fills the
    queries from `q_off`). Not `WRITE_BITS`: the count is added to `deg[start + q]` (zeroed first).
    `x` is the ORIGINAL dataset, read only by the recheck. With `probe_in` set, the block's rechecked pairs are
    added to `unc[0]`."""
    comptime assert EPS_NBLK == 64 and GEMM_NBLK == 64, "one column tile is one word"
    comptime assert GEMM_ACC_COLS_PER_TH == 4 and GEMM_ACC_TH_COLS == 16, (
        "bit acccolid + 16 j for j < 4"
    )
    var start = Int(start_in)
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var bm_rows = Int(bm_rows_in)
    var q_off = Int(q_off_in)

    var tid = Int(thread_idx.x)
    var accrowid = tid // GEMM_ACC_TH_COLS
    var acccolid = tid % GEMM_ACC_TH_COLS

    var word = Int(block_idx.x)
    var m0 = Int(block_idx.y) * GEMM_MBLK
    var n0 = word * GEMM_NBLK
    var startx = m0 + accrowid
    var starty = n0 + acccolid
    # the reference walks whole K-blocks, zero past k: the same count of steps here
    var k_walk = ((k + GEMM_KBLK - 1) // GEMM_KBLK) * GEMM_KBLK

    # this thread's four columns: their centred norms (unusable past n)
    var ncol = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](DBF_NORM_UNUSABLE)
    for j in range(GEMM_ACC_COLS_PER_TH):
        var yid0 = starty + j * GEMM_ACC_TH_COLS
        if yid0 < n:
            ncol[j] = nc.unsafe_load(yid0)

    var n_unc = Int32(0)
    for i in range(GEMM_ACC_ROWS_PER_TH):
        var xid = startx + i * GEMM_ACC_TH_ROWS
        var inside = UInt32(0)
        var pending = UInt32(0)
        if xid < m:
            var nq = nc.unsafe_load(start + xid)
            for j in range(GEMM_ACC_COLS_PER_TH):
                var yid = starty + j * GEMM_ACC_TH_COLS
                if yid < n:
                    var nj = ncol[j]
                    var usable = nq >= Float32(0.0) and nj >= Float32(0.0)
                    var sn = nq + nj
                    var approx = sn - Float32(2.0) * g.unsafe_load(xid * n + yid)
                    var band = band_in * sn + DBF_ABS_SLACK
                    if usable and approx + band < thresh_in:
                        inside |= UInt32(1) << UInt32(j)
                    elif usable and approx - band > thresh_in:
                        pass
                    else:
                        pending |= UInt32(1) << UInt32(j)
        if pending != UInt32(0):
            # THE EXACT RECHECK: `bef_eps_kernel`'s accumulation for this thread's row `i`, statement for
            # statement (the operands it stages through shared memory are read from global memory here: a
            # staging copy is bit exact). `xv` unflushed, `regy` flushed, `_eps_acc` at the same width, the
            # features in ascending order, zeros past k and past n. Change both together.
            var acc = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
            for c in range(k_walk):
                var xv = Float32(0.0)
                var regy = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
                if c < k:
                    xv = x.unsafe_load((start + xid) * k + c)
                    for j in range(GEMM_ACC_COLS_PER_TH):
                        var yid2 = starty + j * GEMM_ACC_TH_COLS
                        if yid2 < n:
                            regy[j] = x.unsafe_load(yid2 * k + c)
                regy = ftz_simd[GEMM_ACC_COLS_PER_TH](regy)
                acc = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](acc, xv, regy)
            for j in range(GEMM_ACC_COLS_PER_TH):
                if ((pending >> UInt32(j)) & UInt32(1)) != UInt32(0):
                    n_unc += Int32(1)
                    # the reference's radius test
                    if acc[j] <= thresh_in:
                        inside |= UInt32(1) << UInt32(j)

        # --- the reference epilogue: bits or counts --------------------------
        var lo = UInt32(0)
        var hi = UInt32(0)
        var s = Int32(0)
        for j in range(GEMM_ACC_COLS_PER_TH):
            if ((inside >> UInt32(j)) & UInt32(1)) != UInt32(0):
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
                    word * bm_rows + q_off + xid,
                    (hi.cast[DType.uint64]() << UInt64(32))
                    | lo.cast[DType.uint64](),
                )
        else:
            var off2 = 1
            while off2 < GEMM_ACC_TH_COLS:
                s += shuffle_xor(s, UInt32(off2))
                off2 *= 2
            if acccolid == 0 and xid < m:
                _ = Atomic.fetch_add(deg.unsafe_offset(start + xid), s)

    # the probe: every thread reaches the block reduction; thread 0 adds when asked
    var block_unc = block_sum[block_size=EPS_THREADS](n_unc)
    if tid == 0 and probe_in != Int32(0):
        _ = Atomic.fetch_add(unc.unsafe_offset(0), block_unc)


def dbf_prepare(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut xc: DeviceBuffer[DType.float32],
    mut nc: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_features: Int,
) raises:
    """Once a fit: the column means of `x`, then `xc` (n x k centred rows) and `nc` (n centred squared norms,
    `DBF_NORM_UNUSABLE` for a row the bound does not serve). Three launches over every row and column, drained
    before it returns (the partial sums and the means live only here)."""
    var parts = DBF_MEAN_PARTS
    if parts > n_rows:
        parts = n_rows
    var kblocks = (n_features + DBF_TPB - 1) // DBF_TPB
    var part = ctx.enqueue_create_buffer[DType.float32](parts * n_features)
    var mean = ctx.enqueue_create_buffer[DType.float32](n_features)
    ctx.enqueue_function[dbf_colsum_kernel](
        x.unsafe_ptr(), part.unsafe_ptr(), Int32(n_rows), Int32(n_features),
        Int32(parts),
        grid_dim=(parts, kblocks, 1), block_dim=(DBF_TPB, 1, 1),
    )
    ctx.enqueue_function[dbf_colmean_kernel](
        part.unsafe_ptr(), mean.unsafe_ptr(), Int32(n_rows),
        Int32(n_features), Int32(parts),
        grid_dim=(kblocks, 1, 1), block_dim=(DBF_TPB, 1, 1),
    )
    ctx.enqueue_function[dbf_center_kernel](
        x.unsafe_ptr(), mean.unsafe_ptr(), xc.unsafe_ptr(), nc.unsafe_ptr(),
        Int32(n_rows), Int32(n_features),
        grid_dim=((n_rows + DBF_TPB - 1) // DBF_TPB, 1, 1),
        block_dim=(DBF_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = part^
    _ = mean^


def dbf_range_launch[
    metric: Int, WRITE_BITS: Bool
](
    ctx: DeviceContext,
    mut gws: GemmWorkspace,
    mut g: DeviceBuffer[DType.float32],
    mut xc: DeviceBuffer[DType.float32],
    mut nc: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut bm: DeviceBuffer[DType.uint64],
    mut deg: DeviceBuffer[DType.int32],
    mut unc: DeviceBuffer[DType.int32],
    start: Int,
    m: Int,
    n_rows: Int,
    n_features: Int,
    bm_rows: Int,
    q_off: Int,
    thresh: Float32,
    band: Float32,
    probe: Bool,
) raises:
    """One query range `[start, start + m)`: the GEMM `g[m x n] = xc[range] . xc^T`, then the filter launch
    (bits into `bm` at query offset `q_off` of a `bm_rows`-query bit matrix, or degrees into `deg`). `g` holds
    at least `m * n_rows` floats; `gws` is the fit's GEMM scratch. Asynchronous apart from what the GEMM's own
    plan waits for: the caller keeps every buffer named past its synchronize."""
    var xq = xc.create_sub_buffer[DType.float32](
        start * n_features, m * n_features
    )
    var gv = g.create_sub_buffer[DType.float32](0, m * n_rows)
    gws.run(ctx, gv, xq, xc, m, n_rows, n_features, OP_NT)
    var probe_flag = Int32(0)
    if probe:
        probe_flag = Int32(1)
    ctx.enqueue_function[dbf_filter_kernel[metric, WRITE_BITS]](
        bm.unsafe_ptr(), deg.unsafe_ptr(), unc.unsafe_ptr(), gv.unsafe_ptr(),
        nc.unsafe_ptr(), x.unsafe_ptr(), Int32(start), Int32(m),
        Int32(n_rows), Int32(n_features), Int32(bm_rows), Int32(q_off),
        thresh, band, probe_flag,
        grid_dim=(
            (n_rows + EPS_NBLK - 1) // EPS_NBLK,
            (m + EPS_MBLK - 1) // EPS_MBLK,
            1,
        ),
        block_dim=(EPS_THREADS, 1, 1),
    )
