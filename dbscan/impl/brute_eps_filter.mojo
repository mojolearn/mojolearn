# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The brute epsilon pass of DBSCAN as a GEMM FILTER plus the exact kernel's recheck
(lane sp-dbscan-filter, 2026-11-20; the default since 2026-11-24 wherever the per-pair cost rule `dbf_filter_pays`
says it pays, lane ab-outcome-4: the opt-in define MOJOLEARN_DBSCAN_FAST_BRUTE_EPS_FILTER is gone).

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

NO PROBE, NO RUN-TIME TIMING (Andrew, 2026-11-24: "the route should be one thing only", optimised for large
inputs; no probe choosing between two). Nothing times or declines the filter at run time (lane ab-outcome-3 /
o3-route removed the timed probe of lane ab-outcome-1, which had replaced the 1 / 256 open-pair decline). The
first A/B (nvc-ab-20261124T070727Z-69dac2, L40S) halved a 220-feature fit (575 -> 296 s), 1.02 on an 11-feature
one at 250,000 rows, not faster at full size; the wave-3 A/B (below, `dbf_filter_pays`) was split the same way,
so which pass a fit takes is decided before it runs by the per-pair cost rule of `dbf_filter_pays`, from the
feature count alone. Per pair the reference spends work proportional to k; the
filter spends the GEMM's (also proportional to k, at a far higher rate) plus two costs that do not shrink with
k: the round trip of the product through device memory (the GEMM writes `g`, the filter reads it: the identical
GEMM has no epilogue hook to fuse the band test into), and the recheck of the open pairs. The recheck is one
pass over the features per thread however many of its 4 rows have an open pair, loading only the open
rows' and the open columns' features (lane ab-outcome-3 / o3-route; it was one pass per row with an open pair,
loading all 4 columns): a warp runs it at most once per tile, and the words are the same (each open pair's
accumulator runs the reference's statements in the reference's order).

SCOPE. The brute edge-free route of `identical` (NVIDIA and AMD), squared-L2 arithmetic (euclidean, and cosine
on its unit rows); L1 has no expanded form and keeps the reference kernel. Not carried to radius neighbours,
LOF, OPTICS or k-means here (after the A/B).
"""

from std.atomic import Atomic
from std.memory import bitcast
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.primitives.warp import shuffle_xor
from max.gpu.primitives.block import sum as block_sum
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz_simd
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


# The opt-in switch is gone (lane ab-outcome-4, 2026-11-24): the filter is the default wherever `dbf_filter_pays`
# holds, on the brute edge-free route (IDN_DBSCAN_EDGE_FREE: IDENTICAL on NVIDIA and AMD).
# TOMBSTONE: MOJOLEARN_DBSCAN_FAST_BRUTE_EPS_FILTER (SPLIT: NV L40S istella full 0.58, taxi full 0.99, taxi 250000 1.12; the filter is now the default by the per-pair cost rule dbf_filter_pays, the define is gone) deleted 2026-11-24 by lane/ab-outcome-4; code recoverable at 2631af4005.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_DBSCAN_FAST_BRUTE_EPS_FILTER.ab-outcome-4.patch

#: `(k + 2) u <= 2^-8`: the width up to which the second-order terms of the bound stay inside the band's 1/16.
comptime DBF_MAX_FEATURES = 65534
#: 2^100: the largest centred squared norm the bound serves (coordinates under 2^50, sums under 2^120).
comptime DBF_NORM_CAP = Float32(1.2676506002282294e30)
#: The stored norm of a row the bound does not serve; negative, so one ordinary compare finds it.
comptime DBF_NORM_UNUSABLE = Float32(-1.0)
#: 2^-100: the absolute part of the band (underflow and flush terms, see the module docstring).
comptime DBF_ABS_SLACK = Float32(7.888609052210118e-31)
#: The largest `m x n` product of one GEMM: the cell count travels as an Int32 in the GEMM's launches, and the
#: checks' rechecked-pair count of one range is an Int32.
comptime DBF_MAX_CELLS = 2147483647
#: The product buffer's (and the centred copy's) share of the free device memory, in percent: the share the
#: bit matrix of the route takes (`BEF_BM_SHARE_PCT`).
comptime DBF_G_SHARE_PCT = 25
#: Row-strided partial sums of the column means (blocks along grid.x): enough blocks to fill any device.
comptime DBF_MEAN_PARTS = 1024
comptime DBF_TPB = 256


#: THE PER-PAIR COST RULE (lane ab-outcome-4, 2026-11-24): the feature count from which the filter costs less per
#: pair than the reference accumulation. Per pair (both distance passes, the union side equal on both since the
#: bit-matrix plan is made first):
#:   reference: k feature steps of `bef_eps_kernel`'s 4x4 tile, c_ref each;
#:   filter:    k feature steps of the identical GEMM, c_gemm each (19.5-25 against 0.74 TFLOP/s: c_gemm ~ c_ref / 27),
#:              plus a FIXED part F that does not shrink with k: the product's 8-byte round trip through device
#:              memory (the GEMM writes `g`, `dbf_filter_kernel` reads it; no epilogue hook fuses them) and the
#:              band test, plus the open pairs' recheck (a fraction of k c_ref, data dependent, small on wide data).
#:   The filter pays when k (c_ref - c_gemm) > F, i.e. k > F / (c_ref (1 - 1/27)).
#: Both constants from the wave-3 A/B (nvc-ab-20261124T182912Z-977cc8, NVIDIA L40S, wheel 71db45de5d, job nvc1-0004,
#: one scored run per arm; 2 passes x n^2 pairs each):
#:   c_ref = 0.51 ps: the two reference fits at 1,000,000 rows differ by 579116 - 364446 ms over 2e12 pairs and
#:           220 - 11 features;
#:   F     = 24 ps:   the filter fit at 250,000 x 11 cost 22229 - 19857 ms more over 1.25e11 pairs (19 ps a pair),
#:           plus the 11 (c_ref - c_gemm) = 5 ps it saved on the accumulation.
#:   k* = 24 / (0.51 x 26 / 27) = 49, rounded up to DBF_MIN_FEATURES = 50.
#: The verdicts it reproduces: istella full (220 features) 0.58 FILTER; taxi 250000 (11 features) 1.12 REFERENCE;
#: taxi full (11 features) 0.99, within noise, REFERENCE. Per pair, so it holds for any row count; it depends on the
#: device's ratio of memory bandwidth to the reference kernel's rate (measured on the L40S only; the MI300X has more
#: bandwidth per flop, so the threshold is conservative there) and on the data only through the open fraction.
#: Neighbouring shapes it must hold for (owed driver row dbscan-brute-eps-filter-shapes): 50,000 x 8 and 50,000 x 64
#: bracket it (reference / filter), and the non-board 200,000 x 100 takes the filter. Never a dataset or board shape:
#: the board's two dbscan widths (11, 220) are far on either side of it.
comptime DBF_MIN_FEATURES = 50


def dbf_filter_pays(n_features: Int) -> Bool:
    """The per-pair cost rule above: the GEMM filter costs less per pair than the reference accumulation from
    DBF_MIN_FEATURES features up. Either pass writes the same words; only the time differs."""
    return n_features >= DBF_MIN_FEATURES


def dbf_filter_serves(metric: Int, n_features: Int) -> Bool:
    """Whether the filter's arithmetic and bound apply: a squared-L2 accumulator, a width the bound covers. Memory
    and the data itself are decided by the fit."""
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


@always_inline
def _dbf_recheck_bits(
    acc: SIMD[DType.float32, GEMM_ACC_COLS_PER_TH], pend: UInt32, thresh: Float32
) -> UInt32:
    """The reference's radius test on the open pairs of one row: bit j where pair j was open and its exact
    accumulator is within `thresh`."""
    var bits = UInt32(0)
    for j in range(GEMM_ACC_COLS_PER_TH):
        if ((pend >> UInt32(j)) & UInt32(1)) != UInt32(0):
            if acc[j] <= thresh:
                bits |= UInt32(1) << UInt32(j)
    return bits


def dbf_filter_kernel[
    metric: Int, WRITE_BITS: Bool, COUNT_RECHECKS: Bool
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
):
    """`bef_eps_kernel[metric, WRITE_BITS]`'s answer for the query rows `[start, start + m)` of the dataset
    against all `n` rows, from `g` (`m x n`, `g[q, j]` = centred row `start + q` . centred row `j`) and the
    centred norms `nc`, with the reference accumulation only for the pairs the band leaves open.

    grid = (ceil(n / 64) column words, ceil(m / 64) row tiles), block = `EPS_THREADS`: the reference tile, the
    same thread to (row, column) map. `WRITE_BITS`: the word of (query `q`, column word `w`) goes to
    `bm[w * bm_rows + q_off + q]` (the range's bit matrix holds `bm_rows` queries; this launch fills the
    queries from `q_off`). Not `WRITE_BITS`: the count is added to `deg[start + q]` (zeroed first).
    `x` is the ORIGINAL dataset, read only by the recheck. `COUNT_RECHECKS` (the word-for-word check only; the fit
    never counts): the block's rechecked pairs are added to `unc[0]`."""
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

    comptime assert GEMM_ACC_ROWS_PER_TH == 4, "four row accumulators below"
    # --- the band test of this thread's 4 x 4 pairs ---------------------------
    var inside = SIMD[DType.uint32, GEMM_ACC_ROWS_PER_TH](0)
    var pending = SIMD[DType.uint32, GEMM_ACC_ROWS_PER_TH](0)
    for i in range(GEMM_ACC_ROWS_PER_TH):
        var xid = startx + i * GEMM_ACC_TH_ROWS
        var ins = UInt32(0)
        var pen = UInt32(0)
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
                        ins |= UInt32(1) << UInt32(j)
                    elif usable and approx - band > thresh_in:
                        pass
                    else:
                        pen |= UInt32(1) << UInt32(j)
        inside[i] = ins
        pending[i] = pen

    # --- THE EXACT RECHECK of the open pairs ----------------------------------
    # `bef_eps_kernel`'s accumulation, statement for statement, for every open pair of this thread in ONE pass
    # over the features (the operands it stages through shared memory are read from global memory here: a
    # staging copy is bit exact). `xv` unflushed, `regy` flushed, `_eps_acc` at the same width, the features in
    # ascending order, zeros past k and past n. Only the open rows' `xv` and the open columns' `regy` are loaded,
    # and a row with no open pair skips its `_eps_acc`: `_eps_acc` is lane-wise, so every open pair's accumulator
    # is the one the reference computes, and the lanes of a closed pair are never read. Change both together.
    var pend_cols = pending[0] | pending[1] | pending[2] | pending[3]
    if pend_cols != UInt32(0):
        var xid0 = startx
        var xid1 = startx + GEMM_ACC_TH_ROWS
        var xid2 = startx + 2 * GEMM_ACC_TH_ROWS
        var xid3 = startx + 3 * GEMM_ACC_TH_ROWS
        var acc0 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
        var acc1 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
        var acc2 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
        var acc3 = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
        for c in range(k_walk):
            var xv0 = Float32(0.0)
            var xv1 = Float32(0.0)
            var xv2 = Float32(0.0)
            var xv3 = Float32(0.0)
            var regy = SIMD[DType.float32, GEMM_ACC_COLS_PER_TH](0.0)
            if c < k:
                for j in range(GEMM_ACC_COLS_PER_TH):
                    if ((pend_cols >> UInt32(j)) & UInt32(1)) != UInt32(0):
                        var yid2 = starty + j * GEMM_ACC_TH_COLS
                        if yid2 < n:
                            regy[j] = x.unsafe_load(yid2 * k + c)
                if pending[0] != UInt32(0):
                    xv0 = x.unsafe_load((start + xid0) * k + c)
                if pending[1] != UInt32(0):
                    xv1 = x.unsafe_load((start + xid1) * k + c)
                if pending[2] != UInt32(0):
                    xv2 = x.unsafe_load((start + xid2) * k + c)
                if pending[3] != UInt32(0):
                    xv3 = x.unsafe_load((start + xid3) * k + c)
            regy = ftz_simd[GEMM_ACC_COLS_PER_TH](regy)
            if pending[0] != UInt32(0):
                acc0 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](acc0, xv0, regy)
            if pending[1] != UInt32(0):
                acc1 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](acc1, xv1, regy)
            if pending[2] != UInt32(0):
                acc2 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](acc2, xv2, regy)
            if pending[3] != UInt32(0):
                acc3 = _eps_acc[metric, GEMM_ACC_COLS_PER_TH](acc3, xv3, regy)
        # the reference's radius test on the open pairs
        inside[0] = inside[0] | _dbf_recheck_bits(acc0, pending[0], thresh_in)
        inside[1] = inside[1] | _dbf_recheck_bits(acc1, pending[1], thresh_in)
        inside[2] = inside[2] | _dbf_recheck_bits(acc2, pending[2], thresh_in)
        inside[3] = inside[3] | _dbf_recheck_bits(acc3, pending[3], thresh_in)

    for i in range(GEMM_ACC_ROWS_PER_TH):
        var xid = startx + i * GEMM_ACC_TH_ROWS
        # --- the reference epilogue: bits or counts --------------------------
        var lo = UInt32(0)
        var hi = UInt32(0)
        var s = Int32(0)
        for j in range(GEMM_ACC_COLS_PER_TH):
            if ((inside[i] >> UInt32(j)) & UInt32(1)) != UInt32(0):
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

    # the check's count of rechecked pairs: every thread reaches the block reduction (compile time: the fit
    # instantiates COUNT_RECHECKS False and runs no reduction)
    comptime if COUNT_RECHECKS:
        var n_unc = Int32(0)
        for i in range(GEMM_ACC_ROWS_PER_TH):
            for j in range(GEMM_ACC_COLS_PER_TH):
                if ((pending[i] >> UInt32(j)) & UInt32(1)) != UInt32(0):
                    n_unc += Int32(1)
        var block_unc = block_sum[block_size=EPS_THREADS](n_unc)
        if tid == 0:
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
    metric: Int, WRITE_BITS: Bool, COUNT_RECHECKS: Bool = False
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
) raises:
    """One query range `[start, start + m)`: the GEMM `g[m x n] = xc[range] . xc^T`, then the filter launch
    (bits into `bm` at query offset `q_off` of a `bm_rows`-query bit matrix, or degrees into `deg`). `g` holds
    at least `m * n_rows` floats; `gws` is the fit's GEMM scratch. Asynchronous apart from what the GEMM's own
    plan waits for: the caller keeps every buffer named past its synchronize. `COUNT_RECHECKS` (the word-for-word
    check only) adds the launch's rechecked pairs to `unc[0]`; otherwise `unc` is not touched."""
    var xq = xc.create_sub_buffer[DType.float32](
        start * n_features, m * n_features
    )
    var gv = g.create_sub_buffer[DType.float32](0, m * n_rows)
    gws.run(ctx, gv, xq, xc, m, n_rows, n_features, OP_NT)
    ctx.enqueue_function[dbf_filter_kernel[metric, WRITE_BITS, COUNT_RECHECKS]](
        bm.unsafe_ptr(), deg.unsafe_ptr(), unc.unsafe_ptr(), gv.unsafe_ptr(),
        nc.unsafe_ptr(), x.unsafe_ptr(), Int32(start), Int32(m),
        Int32(n_rows), Int32(n_features), Int32(bm_rows), Int32(q_off),
        thresh, band,
        grid_dim=(
            (n_rows + EPS_NBLK - 1) // EPS_NBLK,
            (m + EPS_MBLK - 1) // EPS_MBLK,
            1,
        ),
        block_dim=(EPS_THREADS, 1, 1),
    )
