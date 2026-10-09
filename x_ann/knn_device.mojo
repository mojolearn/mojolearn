# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The exact k-NN graph of t-SNE and CAGRA on the device, tiled (lane
ann-apple, 2026-09-28). SAME BITS as one thread per row running
`ts_knn_cell`:

  * one thread per row i, as before; the candidate rows j arrive in tiles of
    KTJ rows staged in threadgroup memory, tiles and rows ascending, so row
    i offers its candidates in the cell's order j = 0, 1, ..., n - 1;
  * the staged values are `ftz(x)`, and row i's own values are `ftz(x)` in
    registers: the cell's `ftz(ftz(x_i) - ftz(x_j))` on the same words
    (ftz is idempotent);
  * the fold runs over a comptime width MAXD >= d with zero padding past d:
    a padded step is `fma(+0, +0, acc)`, which returns acc unchanged (acc is
    a sum of squares from +0, never -0), so the sum is the cell's ascending
    fused fold over c < d;
  * each candidate goes through the cell's own `ts_knn_offer` (DEVIATION
    5810); a candidate that cannot enter a full list is skipped first by the
    same `ts_knn_beats` test against a register copy of the list's last
    entry.

Rows wider than 64 features take the untiled cell."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_NVIDIA, COLUMN_AMD

from checks.numerics import ftz, identical_mul_add, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_ann.tsne_core import F32P, I32P, ts_ftz_nonneg, ts_knn_beats, ts_knn_cell, ts_knn_offer
from x_ann.fast_env import FAST_KNN_BIGD

#: K13 (IDENTICAL, lane ml-cluster-nbrs 2026-10-04): partial-distance pruning
#: in `knn_tiled_kernel`. Once the list is full, a candidate's fold stops as
#: soon as its partial sum reaches the list's last distance ld. Tie-aware: the
#: running sum only grows (non-negative fused terms from +0, flushed), so the
#: full distance is >= the partial, or NaN; candidates are offered in
#: ascending j, so every listed index is below j, and (dist >= ld, j > li)
#: never beats (ld, li) under `ts_knn_beats`; a NaN never beats either. A
#: pruned candidate is exactly one the full fold would reject: the same list,
#: the same bits. Tested every KNP_EVERY features. `-D
#: MOJOLEARN_IDN_ANN_KNN_PRUNE_OFF` (or MOJOLEARN_IDN_ALL_OFF) folds every
#: feature.
comptime IDN_ANN_KNN_PRUNE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_ANN_KNN_PRUNE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: features between two prune tests (a multiple of 4)
comptime KNP_EVERY = 16

#: rows per threadgroup (one thread each) and candidate rows per tile
comptime KTB = 128
comptime KTJ = 64
comptime TPB = 64


def knn_cell_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        ts_knn_cell(i, x, Int(n), Int(d), Int(nn), nn_d, nn_i)


def knn_tiled_kernel[MAXD: Int](n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    var t = Int(thread_idx.x)
    var nr = Int(n)
    var dd = Int(d)
    var k = Int(nn)
    var i = Int(block_idx.x) * KTB + t
    var tile = stack_allocation[KTJ * MAXD, Scalar[DType.float32], alignment=16, address_space=AddressSpace.SHARED]()
    var live = i < nr
    var xi = InlineArray[Float32, MAXD](fill=Float32(0.0))
    if live:
        comptime for c in range(MAXD):
            if c < dd:
                xi[c] = ftz(x.unsafe_load(i * dd + c))
    var base = i * k
    var filled = 0
    var ld = Float32(0.0)
    var li = 0
    var j0 = 0
    while j0 < nr:
        for e in range(t, KTJ * MAXD, KTB):
            var r = e // MAXD
            var c = e % MAXD
            var v = Float32(0.0)
            if j0 + r < nr and c < dd:
                v = ftz(x.unsafe_load((j0 + r) * dd + c))
            tile[e] = v
        barrier()
        if live:
            var jn = KTJ if nr - j0 > KTJ else nr - j0
            for r in range(jn):
                var j = j0 + r
                if j == i:
                    continue
                var acc = Float32(0.0)
                # the staged row four floats per load; the fold is the same
                # statements in the same order, lane by lane
                # lane ann-apple2: acc, a sum of squares from +0, is flushed
                # by `ts_ftz_nonneg` (the same word). A skip of the padded
                # groups was measured slower (m4pro-b 1790604321939) and is
                # not taken.
                var dead = False
                comptime for c4 in range(MAXD // 4):
                    if not dead:
                        var tv = tile.unsafe_load[width=4, alignment=16](r * MAXD + 4 * c4)
                        comptime for u in range(4):
                            var diff = ftz(xi[4 * c4 + u] - tv[u])
                            acc = ts_ftz_nonneg(identical_mul_add(diff, diff, acc))
                        # K13: the tie-aware prune (see IDN_ANN_KNN_PRUNE)
                        comptime if IDN_ANN_KNN_PRUNE and (4 * c4 + 4) % KNP_EVERY == 0 and 4 * c4 + 4 < MAXD:
                            if filled == k and acc >= ld:
                                dead = True
                if dead:
                    continue
                if filled == k and not ts_knn_beats(acc, j, ld, li):
                    continue
                filled = ts_knn_offer(acc, j, base, k, filled, nn_d, nn_i)
                if filled == k:
                    ld = nn_d.unsafe_load(base + k - 1)
                    li = Int(nn_i.unsafe_load(base + k - 1))
        barrier()
        j0 += KTJ


#: the chunked arm (rows wider than 64 features): candidate rows per tile
#: and features per staged chunk
comptime KBJ = 32
comptime KBC = 32


def knn_tiled_bigd_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    """FAST on Apple default (off: `-D MOJOLEARN_ANN_FAST_KNN_BIGD_OFF`; lane/apple-fast-ann,
    2026-10-02): the tiled k-NN for rows wider than 64 features. One thread
    per row i (KTB per threadgroup); the candidate rows arrive in tiles of
    KBJ rows, each tile staged KBC features at a time (a KBJ x KBC slab of
    ftz(x) in threadgroup memory), and thread i keeps KBJ running sums, one
    per candidate, folding feature c ascending across the chunks: the
    cell's ascending fused fold on the same words, then the cell's own
    `ts_knn_offer` per candidate in j order. Cause: rows wider than 64
    features took `knn_cell_kernel` (`knn_enqueue`), one thread per row
    reading every candidate row from device memory with nothing staged,
    400,000 x 400,000 x 220 loads at CAGRA's Istella build. Same bits
    expected (the same statements in the same order)."""
    var t = Int(thread_idx.x)
    var nr = Int(n)
    var dd = Int(d)
    var k = Int(nn)
    var i = Int(block_idx.x) * KTB + t
    var slab = stack_allocation[KBJ * KBC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var live = i < nr
    var base = i * k
    var filled = 0
    var ld = Float32(0.0)
    var li = 0
    var acc = InlineArray[Float32, KBJ](fill=Float32(0.0))
    var j0 = 0
    while j0 < nr:
        var jn = KBJ if nr - j0 > KBJ else nr - j0
        comptime for r in range(KBJ):
            acc[r] = Float32(0.0)
        var c0 = 0
        while c0 < dd:
            var cn = KBC if dd - c0 > KBC else dd - c0
            for e in range(t, KBJ * KBC, KTB):
                var r = e // KBC
                var c = e % KBC
                var v = Float32(0.0)
                if r < jn and c < cn:
                    v = ftz(x.unsafe_load((j0 + r) * dd + c0 + c))
                slab[e] = v
            barrier()
            if live:
                for c in range(cn):
                    var xi = ftz(x.unsafe_load(i * dd + c0 + c))
                    comptime for r in range(KBJ):
                        var diff = ftz(xi - slab[r * KBC + c])
                        acc[r] = ts_ftz_nonneg(identical_mul_add(diff, diff, acc[r]))
            barrier()
            c0 += KBC
        if live:
            for r in range(jn):
                var j = j0 + r
                if j == i:
                    continue
                var a = acc[r]
                if filled == k and not ts_knn_beats(a, j, ld, li):
                    continue
                filled = ts_knn_offer(a, j, base, k, filled, nn_d, nn_i)
                if filled == k:
                    ld = nn_d.unsafe_load(base + k - 1)
                    li = Int(nn_i.unsafe_load(base + k - 1))
        j0 += KBJ


# lane/gap-nv-classical2: rows wider than 64 features on NVIDIA and AMD.
# `knn_cell_kernel` (one thread per row re-reading both rows from global
# memory) ran istella's 20k x 220 graph. `knn_wide_kernel` computes a
# KW_TI x KW_TJ block of squared distances with both sides staged KW_KC
# features at a time and 4 x 4 cells per thread, each cell the cell's chain
# `ts_ftz_nonneg(fma(diff, diff, acc))`, diff = ftz(ftz(x_i) - ftz(x_j)),
# c ascending over exactly d; then row i's owner offers the tile's
# candidates in ascending j through the same `ts_knn_beats` /
# `ts_knn_offer`. The same bits. -D MOJOLEARN_KNN_WIDE_OFF=1 restores the
# cell kernel.
comptime KNN_WIDE = (
    (TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD)
    and not is_defined["MOJOLEARN_KNN_WIDE_OFF"]()
)
comptime KW_TI = 64
comptime KW_TJ = 64
comptime KW_KC = 16
comptime KW_TX = 16
comptime KW_TY = 16
comptime KW_RI = KW_TI // KW_TY
comptime KW_RJ = KW_TJ // KW_TX
comptime KW_TPB = KW_TX * KW_TY
comptime KW_DS = KW_TJ + 1


def knn_wide_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * KW_TX + tx
    var nr = Int(n)
    var dd = Int(d)
    var k = Int(nn)
    var i0 = Int(block_idx.x) * KW_TI
    var a_s = stack_allocation[KW_KC * KW_TI, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var b_s = stack_allocation[KW_KC * KW_TJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var dist_s = stack_allocation[KW_TI * KW_DS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var i = i0 + tid
    var owner = tid < KW_TI and i < nr
    var base = i * k
    var filled = 0
    var ld = Float32(0.0)
    var li = 0
    var j0 = 0
    while j0 < nr:
        var acc = InlineArray[Float32, KW_RI * KW_RJ](fill=Float32(0.0))
        var k0 = 0
        while k0 < dd:
            comptime for q in range(KW_KC * KW_TI // KW_TPB):
                var e = tid + q * KW_TPB
                var ii = e // KW_KC
                var kk = e - ii * KW_KC
                var c = k0 + kk
                var v = Float32(0.0)
                if c < dd and i0 + ii < nr:
                    v = ftz(x.unsafe_load((i0 + ii) * dd + c))
                a_s[kk * KW_TI + ii] = v
            comptime for q in range(KW_KC * KW_TJ // KW_TPB):
                var e = tid + q * KW_TPB
                var jj = e // KW_KC
                var kk = e - jj * KW_KC
                var c = k0 + kk
                var v = Float32(0.0)
                if c < dd and j0 + jj < nr:
                    v = ftz(x.unsafe_load((j0 + jj) * dd + c))
                b_s[kk * KW_TJ + jj] = v
            barrier()
            var kmax = dd - k0
            if kmax > KW_KC:
                kmax = KW_KC
            for kk in range(kmax):
                var av = InlineArray[Float32, KW_RI](fill=Float32(0.0))
                var bv = InlineArray[Float32, KW_RJ](fill=Float32(0.0))
                comptime for r in range(KW_RI):
                    av[r] = a_s[kk * KW_TI + ty + r * KW_TY]
                comptime for c in range(KW_RJ):
                    bv[c] = b_s[kk * KW_TJ + tx + c * KW_TX]
                comptime for r in range(KW_RI):
                    comptime for c in range(KW_RJ):
                        var diff = ftz(av[r] - bv[c])
                        acc[r * KW_RJ + c] = ts_ftz_nonneg(identical_mul_add(diff, diff, acc[r * KW_RJ + c]))
            barrier()
            k0 += KW_KC
        comptime for r in range(KW_RI):
            comptime for c in range(KW_RJ):
                dist_s[(ty + r * KW_TY) * KW_DS + tx + c * KW_TX] = acc[r * KW_RJ + c]
        barrier()
        if owner:
            var jn = KW_TJ if nr - j0 > KW_TJ else nr - j0
            for r in range(jn):
                var j = j0 + r
                if j == i:
                    continue
                var dv = dist_s[tid * KW_DS + r]
                if filled == k and not ts_knn_beats(dv, j, ld, li):
                    continue
                filled = ts_knn_offer(dv, j, base, k, filled, nn_d, nn_i)
                if filled == k:
                    ld = nn_d.unsafe_load(base + k - 1)
                    li = Int(nn_i.unsafe_load(base + k - 1))
        barrier()
        j0 += KW_TJ


#: fg-tsne-dbscan T1 (IDENTICAL, NVIDIA and AMD, DEFAULT OFF, lane
#: fg-tsne-dbscan 2026-10-09): THE TILE'S TOP-K SELECT IN THREADGROUP
#: MEMORY. `knn_wide_kernel` (and the one-thread-per-row tiled kernel) end
#: every 64 x 64 distance tile with a serial owner phase: 64 of the block's
#: 256 threads offer the tile's candidates one by one to the row's sorted
#: list in DEVICE memory (`ts_knn_offer`, an insertion shift of up to k
#: (dist, idx) pairs per accepted candidate, ~k ln(n / k) accepted offers
#: per row over the fit, each a dependent chain of device loads and stores,
#: divergent across the owner warp) while 192 threads wait.
#: `knn_wide_select_kernel` keeps the distance tile (the same chain
#: `ts_ftz_nonneg(fma(diff, diff, acc))`, c ascending over exactly d) and
#: replaces the owner phase by three block-wide steps per tile:
#:   A. every thread (4 per row, 16 candidates each) marks the candidates
#:      that can enter the row's list: j < n, j != i, and the list not full
#:      or (dist, j) before the list's last entry;
#:   B. every marked candidate takes its RANK among the row's marked
#:      candidates under (dist, j) (64 compares in threadgroup memory) and
#:      writes its column to slot `rank`: the marked set sorted, in parallel;
#:   C. the row's owner merges the sorted marked set into the list from the
#:      back, in place, keeping the min(k, filled + m) first entries: one
#:      pass over the list's tail per tile instead of one shift chain per
#:      accepted candidate.
#: The list after a tile is the first min(k, seen) entries of every
#: candidate seen so far under (squared distance, index): a total order on
#: finite and infinite distances (indices are distinct), so the SET and its
#: ORDER are unique and equal the serial offers' result: the same nn_d /
#: nn_i words, the same P, the same embedding. Bits: none. (A NaN distance
#: has no place in the serial order, whose result then depends on offer
#: order; here NaN sorts after every number, ties by index, the same on
#: every vendor. A NaN can only come from a non-finite input row.)
#: Under the switch every width d takes this kernel on NVIDIA and AMD (the
#: d <= 64 rows took the one-thread-per-row tile: n / 128 blocks of 128
#: threads, a quarter of the threads of the 64 x 64 tile's n / 64 blocks of
#: 256, and the same serial offers per thread); padded features are not
#: folded (the loop runs over exactly d), which the tiled kernel's zero
#: padding equals (a padded step is the identity). Threadgroup memory: the
#: wide kernel's 25 KB plus 8 KB of marks and order and 0.75 KB of row
#: state, independent of k (the list stays in device memory). Expected:
#: the kNN stage's insertion cost (~1.5 s at 20k x 220) to the tile's
#: distance cost. Also the CAGRA build's exact graph (`cagra_knn_enqueue`),
#: same bits. IDENTICAL default on NVIDIA and AMD since 2026-10-09
#: (lane/postmerge-act-4; post-merge A/B, one run per arm, ratios arm /
#: default, nv L40S n0608 -> v0979..v0981, amd MI325X a1067 -> a1104..a1106):
#: alone istella NV 1389.9 -> 1342 ms (0.97x) / AMD 1957.9 -> 2000 (1.02x),
#: taxi NV 1563.6 -> 1368 (0.87x) / AMD 1902.7 -> 1831 (0.96x); with
#: IDN_TSNE_REP_TILE (both promoted together) istella NV 1191 (0.86x) / AMD
#: 1219 (0.62x), avg 0.74x; taxi NV 1831 (1.17x) / AMD 1048 (0.55x), avg
#: 0.86x. Vendor split: NV taxi 1.17x slower with both on (the average of
#: the two vendors decides). Trustworthiness@15 equal and digests identical
#: in every arm on both vendors (istella ce69cdffd6f889b6, taxi
#: cee32a8a52defb1c): no bit change. -D MOJOLEARN_IDN_TSNE_KNN_TILE_SELECT_OFF
#: (or MOJOLEARN_IDN_ALL_OFF) restores the serial owner insertion (and the
#: one-thread-per-row tile for d <= 64).
comptime IDN_TSNE_KNN_TILE_SELECT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and (TARGET_COLUMN == COLUMN_NVIDIA or TARGET_COLUMN == COLUMN_AMD)
    and not (
        is_defined["MOJOLEARN_IDN_TSNE_KNN_TILE_SELECT_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)
#: candidates of a row per thread in steps A and B (4 threads per row)
comptime KS_PER = KW_TJ * KW_TI // KW_TPB


@always_inline
def ks_before(d1: Float32, j1: Int, d2: Float32, j2: Int) -> Bool:
    """(d1, j1) before (d2, j2) under (squared distance, index), NaN after
    every number (ties by index): `ts_knn_beats` on every non-NaN pair."""
    if d1 != d1:
        return d2 != d2 and j1 < j2
    if d2 != d2:
        return True
    return d1 < d2 or (d1 == d2 and j1 < j2)


def knn_wide_select_kernel(n: Int32, x: F32P, d: Int32, nn: Int32, nn_d: F32P, nn_i: I32P):
    """`knn_wide_kernel`'s distance tile with the block-wide select and merge
    of IDN_TSNE_KNN_TILE_SELECT: the same lists."""
    comptime assert KW_TI * 4 == KW_TPB, "four threads per row in the select steps"
    comptime assert KS_PER * 4 == KW_TJ, "the four threads of a row cover the tile's candidates"
    comptime assert KW_TJ <= 256, "a column fits a UInt8"
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * KW_TX + tx
    var nr = Int(n)
    var dd = Int(d)
    var k = Int(nn)
    var i0 = Int(block_idx.x) * KW_TI
    var a_s = stack_allocation[KW_KC * KW_TI, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var b_s = stack_allocation[KW_KC * KW_TJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var dist_s = stack_allocation[KW_TI * KW_DS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ok_s = stack_allocation[KW_TI * KW_TJ, Scalar[DType.uint8], address_space=AddressSpace.SHARED]()
    var ord_s = stack_allocation[KW_TI * KW_TJ, Scalar[DType.uint8], address_space=AddressSpace.SHARED]()
    var rld_s = stack_allocation[KW_TI, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var rli_s = stack_allocation[KW_TI, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var rfull_s = stack_allocation[KW_TI, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    # the owner phase's row (C) and the select steps' row and quarter (A, B)
    var i = i0 + tid
    var owner = tid < KW_TI and i < nr
    var base = i * k
    var filled = 0
    var sr = tid >> 2
    var sq = tid & 3
    var si = i0 + sr
    if tid < KW_TI:
        rld_s[tid] = Float32(0.0)
        rli_s[tid] = Int32(0)
        rfull_s[tid] = Int32(0)
    var j0 = 0
    while j0 < nr:
        var acc = InlineArray[Float32, KW_RI * KW_RJ](fill=Float32(0.0))
        var k0 = 0
        while k0 < dd:
            comptime for q in range(KW_KC * KW_TI // KW_TPB):
                var e = tid + q * KW_TPB
                var ii = e // KW_KC
                var kk = e - ii * KW_KC
                var c = k0 + kk
                var v = Float32(0.0)
                if c < dd and i0 + ii < nr:
                    v = ftz(x.unsafe_load((i0 + ii) * dd + c))
                a_s[kk * KW_TI + ii] = v
            comptime for q in range(KW_KC * KW_TJ // KW_TPB):
                var e = tid + q * KW_TPB
                var jj = e // KW_KC
                var kk = e - jj * KW_KC
                var c = k0 + kk
                var v = Float32(0.0)
                if c < dd and j0 + jj < nr:
                    v = ftz(x.unsafe_load((j0 + jj) * dd + c))
                b_s[kk * KW_TJ + jj] = v
            barrier()
            var kmax = dd - k0
            if kmax > KW_KC:
                kmax = KW_KC
            for kk in range(kmax):
                var av = InlineArray[Float32, KW_RI](fill=Float32(0.0))
                var bv = InlineArray[Float32, KW_RJ](fill=Float32(0.0))
                comptime for r in range(KW_RI):
                    av[r] = a_s[kk * KW_TI + ty + r * KW_TY]
                comptime for c in range(KW_RJ):
                    bv[c] = b_s[kk * KW_TJ + tx + c * KW_TX]
                comptime for r in range(KW_RI):
                    comptime for c in range(KW_RJ):
                        var diff = ftz(av[r] - bv[c])
                        acc[r * KW_RJ + c] = ts_ftz_nonneg(identical_mul_add(diff, diff, acc[r * KW_RJ + c]))
            barrier()
            k0 += KW_KC
        comptime for r in range(KW_RI):
            comptime for c in range(KW_RJ):
                dist_s[(ty + r * KW_TY) * KW_DS + tx + c * KW_TX] = acc[r * KW_RJ + c]
        barrier()
        # A: the candidates that can enter row sr's list
        var full = rfull_s[sr] != Int32(0)
        var ld = rld_s[sr]
        var li = Int(rli_s[sr])
        comptime for u in range(KS_PER):
            var c = sq * KS_PER + u
            var j = j0 + c
            var ok = si < nr and j < nr and j != si
            if ok and full:
                ok = ks_before(dist_s[sr * KW_DS + c], j, ld, li)
            ok_s[sr * KW_TJ + c] = UInt8(1) if ok else UInt8(0)
        barrier()
        # B: each marked candidate's rank among the row's marked candidates
        comptime for u in range(KS_PER):
            var c = sq * KS_PER + u
            if ok_s[sr * KW_TJ + c] != UInt8(0):
                var dc = dist_s[sr * KW_DS + c]
                var rank = 0
                for c2 in range(KW_TJ):
                    if ok_s[sr * KW_TJ + c2] != UInt8(0):
                        if ks_before(dist_s[sr * KW_DS + c2], c2, dc, c):
                            rank += 1
                ord_s[sr * KW_TJ + rank] = UInt8(c)
        barrier()
        # C: row i's owner merges the m sorted marked candidates into its
        # list from the back, in place (write slot p = a + b + 1 >= a, so no
        # unread entry is overwritten), keeping the first min(k, filled + m)
        if owner:
            var m = 0
            for c2 in range(KW_TJ):
                if ok_s[tid * KW_TJ + c2] != UInt8(0):
                    m += 1
            if m > 0:
                var nf = filled + m
                if nf > k:
                    nf = k
                var a = filled - 1
                var b = m - 1
                # drop the (filled + m - nf) last entries of the union
                var drop = filled + m - nf
                while drop > 0:
                    var cb = Int(ord_s[tid * KW_TJ + b])
                    if a >= 0 and ks_before(dist_s[tid * KW_DS + cb], j0 + cb, nn_d.unsafe_load(base + a),
                                            Int(nn_i.unsafe_load(base + a))):
                        a -= 1
                    else:
                        b -= 1
                    drop -= 1
                var p = nf - 1
                while b >= 0:
                    var cb = Int(ord_s[tid * KW_TJ + b])
                    var db = dist_s[tid * KW_DS + cb]
                    var jb = j0 + cb
                    if a >= 0 and ks_before(db, jb, nn_d.unsafe_load(base + a), Int(nn_i.unsafe_load(base + a))):
                        nn_d.unsafe_store(base + p, nn_d.unsafe_load(base + a))
                        nn_i.unsafe_store(base + p, nn_i.unsafe_load(base + a))
                        a -= 1
                    else:
                        nn_d.unsafe_store(base + p, db)
                        nn_i.unsafe_store(base + p, Int32(jb))
                        b -= 1
                    p -= 1
                filled = nf
                if filled == k:
                    rld_s[tid] = nn_d.unsafe_load(base + k - 1)
                    rli_s[tid] = nn_i.unsafe_load(base + k - 1)
                    rfull_s[tid] = Int32(1)
        barrier()
        j0 += KW_TJ


def knn_enqueue(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], n: Int, d: Int, nn: Int,
    mut dnd: DeviceBuffer[DType.float32], mut dni: DeviceBuffer[DType.int32],
) raises:
    """Enqueue the k-NN graph of the n x d rows in dx (no sync)."""
    var blocks = (n + KTB - 1) // KTB
    # fg-tsne-dbscan T1: every width d takes the block-wide select on NVIDIA
    # and AMD (see IDN_TSNE_KNN_TILE_SELECT), the same lists
    comptime if IDN_TSNE_KNN_TILE_SELECT:
        ctx.enqueue_function[knn_wide_select_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn),
                                                     dnd.unsafe_ptr(), dni.unsafe_ptr(),
                                                     grid_dim=(n + KW_TI - 1) // KW_TI,
                                                     block_dim=(KW_TX, KW_TY, 1))
        return
    # lane/apple-fast-ann: the chunked arm for wide rows, FAST+Apple default since the M3 A/B (CAGRA istella -88%)
    comptime if FAST_KNN_BIGD:
        if d > 64:
            ctx.enqueue_function[knn_tiled_bigd_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn),
                                                        dnd.unsafe_ptr(), dni.unsafe_ptr(), grid_dim=blocks,
                                                        block_dim=KTB)
            return
    # lane ann-apple2: the fold width is d rounded up to a multiple of 4 up
    # to 32 (fewer padded steps; a padded step is the identity, see above),
    # then 48 and 64
    comptime for w4 in range(1, 9):
        if d <= 4 * w4 and d > 4 * (w4 - 1):
            ctx.enqueue_function[knn_tiled_kernel[4 * w4]](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn),
                                                            dnd.unsafe_ptr(), dni.unsafe_ptr(), grid_dim=blocks,
                                                            block_dim=KTB)
            return
    if d <= 48:
        ctx.enqueue_function[knn_tiled_kernel[48]](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                                    dni.unsafe_ptr(), grid_dim=blocks, block_dim=KTB)
    elif d <= 64:
        ctx.enqueue_function[knn_tiled_kernel[64]](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                                    dni.unsafe_ptr(), grid_dim=blocks, block_dim=KTB)
    else:
        comptime if KNN_WIDE:
            ctx.enqueue_function[knn_wide_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                                  dni.unsafe_ptr(), grid_dim=(n + KW_TI - 1) // KW_TI,
                                                  block_dim=(KW_TX, KW_TY, 1))
            return
        ctx.enqueue_function[knn_cell_kernel](Int32(n), dx.unsafe_ptr(), Int32(d), Int32(nn), dnd.unsafe_ptr(),
                                              dni.unsafe_ptr(), grid_dim=(n + TPB - 1) // TPB, block_dim=TPB)
