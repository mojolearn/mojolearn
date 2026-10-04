# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-w4-linalg (2026-10-04): LU_FAST_TSLU (candidate, opt-in
`-D MOJOLEARN_LU_FAST_TSLU`; FAST on Apple only, needs LU_FAST_MMA), the
32-column inner panel of `lu_fast_mma_factor` by tournament pivoting
(tall-skinny LU, "CALU" / TSLU) instead of one launch per column.

Why: under LU_FAST_MMA the trailing updates are matrix-unit GEMMs (about
3.7e11 flops at n = 8192, tens of ms on the M3), but every 32-column panel
still takes `lu_fast_panel`'s load + 32 dependent step launches + store +
left swaps, then `lfm_swaps_trsm_kernel`: about 37 launches a panel, 9,500
dependent launches at n = 8192, each a full grid drain on Metal. Partial
pivoting needs a grid-wide pivot decision per column; tournament pivoting
picks the panel's 32 pivot rows with a reduction tree instead:

  level 0   one block per 128 rows of the panel (rows k0 .. n-1): the block's
            rows x 32 columns into threadgroup memory, an LU with partial
            pivoting over them (simdgroup shuffle pivot search, 2 barriers a
            column), and the 32 chosen rows' indices out;
  level l   one block per 4 candidate sets (128 rows), the same LU on the
            ORIGINAL rows of the candidates, 32 indices out, until one set
            is left (n = 8192: 64 -> 16 -> 4 -> 1 sets, 4 launches);
  last      the final block's pivot order is the panel's pivot order and its
            factored candidates are L11 \ U11 (the LU of the winners in that
            order depends only on the winners). It also writes piv (getrf's
            sequential swaps, k0 .. k1-1), act / info, the moved rows' map
            (at most 64 positions) and the moved rows' original panel cells;
  post      ONE launch, three disjoint column ranges by block: every panel
            row i >= k0 (rows < k1 take L11 \ U11; the others solve their
            source row against U11, a 32-step row solve, one thread a row);
            the swaps on every column left of k0; and on every column right
            of the panel the swaps then, for columns inside the outer block,
            the 32-row unit-lower solve against L11 (from the last level's
            scratch, so it needs no ordering against the row part).

A panel is then (levels + 1) launches plus `lu_fast_mma_factor`'s GEMM:
about 6 at n = 8192 instead of 37.

Bits and quality: tournament pivots are not partial pivoting's pivots (a
row can win its block yet lose globally to a row eliminated differently),
so FAST words change, and the growth factor is CALU's (Grigori, Demmel,
Xiang 2011: on random matrices close to partial pivoting's, a few tens of
percent above). The quality gate is tools/lu_fast_tslu_quality.py (factor
and solve residual vs main within max(1.5x, +2e-7), info equal). Zero
pivots: an exactly-zero U11[c, c] records act = 0 and info = k + 1 as main
does, and its multipliers are 0 (div0), the same as main's skipped step
when the column is exactly zero. IDENTICAL and every other vendor compile
main's route unchanged.
"""

from std.gpu import WARP_SIZE, block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import COLUMN_APPLE, lib_smem_page_fits_for
from checks.numerics import ftz, identical_mul_add
from x_decomp.cells import F32Ptr, I32Ptr, div0, lu_swap_elem
from x_decomp.lu_fast import LFS_TPB, LU_FAST_STEP1, lfs_blocks

#: CANDIDATE, off unless `-D MOJOLEARN_LU_FAST_TSLU` (lane/apple-fast-w4-linalg,
#: A/B and quality owed: tools/lu_fast_tslu_pair.py). Only read inside
#: `lu_fast_mma_factor`, so it also needs LU_FAST_MMA (FAST + Apple).
comptime LU_FAST_TSLU = LU_FAST_STEP1 and is_defined["MOJOLEARN_LU_FAST_TSLU"]()
#: Rows a tournament block factors (= its threads; 4 candidate sets of LTS_W).
comptime LTS_RB = 128
#: The panel width (LU_FAST_MMA's LFM_PANEL).
comptime LTS_W = 32
#: Threadgroup row stride (33: a row per thread without bank conflicts).
comptime LTS_VS = LTS_W + 1
comptime LTS_SG = LTS_RB // 32


def lts_sets(rows: Int) -> Int:
    """Level-0 candidate sets (blocks) over `rows` panel rows."""
    return max(1, (rows + LTS_RB - 1) // LTS_RB)


def lts_cand_floats(n: Int) -> Int:
    """Floats of one candidate buffer (two are needed, ping-pong)."""
    return lts_sets(n) * LTS_W


comptime LTS_U11_FLOATS = LTS_W * LTS_W
comptime LTS_ORIG_FLOATS = 2 * LTS_W * LTS_W
comptime LTS_MAP_FLOATS = 1 + 4 * LTS_W


@always_inline
def _lts_better(ov: Float32, orow: Int32, oslot: Int32, cv: Float32, crow: Int32, cslot: Int32) -> Bool:
    """(ov, orow) beats (cv, crow): a real candidate over none, then the
    greater |value|, then the lower row (`_lfs_better`'s rule)."""
    if oslot < 0:
        return False
    if cslot < 0:
        return True
    return ov > cv or (ov == cv and orow < crow)


def lts_tour_kernel(
    a: F32Ptr, cin: F32Ptr, cout: F32Ptr, u11: F32Ptr, orig: F32Ptr, mapb: F32Ptr,
    piv: I32Ptr, act: F32Ptr, info: F32Ptr,
    k0_in: Int32, k1_in: Int32, n_in: Int32, level0: Int32, nin: Int32, last: Int32,
):
    """One tournament level (see the module docstring), LTS_RB threads a
    block, block b factoring candidate rows b * LTS_RB .. + LTS_RB - 1 (level
    0: panel rows k0 + b * LTS_RB + t; else the indices cin[b * LTS_RB + t],
    -1 = none) and writing its LTS_W winners, in pivot order, to
    cout[b * LTS_W ..]. With `last` (a one-block level) it also writes U11,
    piv, act, info, the row map and the moved rows' cells."""
    comptime W = LTS_W
    comptime VS = LTS_VS
    comptime RB = LTS_RB
    comptime PAGE_BYTES = (RB * VS + RB + 3 * LTS_SG + W + 4 * W + 1) * 4
    comptime assert lib_smem_page_fits_for[COLUMN_APPLE, PAGE_BYTES](), (
        "lts_tour_kernel: the block's rows must fit Apple's threadgroup memory"
    )
    comptime assert WARP_SIZE == 32, "lts_tour_kernel: 32-lane simdgroups (Apple)"
    var v = stack_allocation[RB * VS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rid = stack_allocation[RB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var pv = stack_allocation[LTS_SG, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var pr = stack_allocation[LTS_SG, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var ps = stack_allocation[LTS_SG, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var wslot = stack_allocation[W, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var mpos = stack_allocation[2 * W, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var mrow = stack_allocation[2 * W, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var ecount = stack_allocation[1, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var n = Int(n_in)
    var k0 = Int(k0_in)
    var w = Int(k1_in) - k0
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var sg = tid // 32
    var g = -1
    if Int(level0) != 0:
        var r = k0 + b * RB + tid
        if r < n:
            g = r
    else:
        var q = b * RB + tid
        if q < Int(nin):
            g = Int(cin.unsafe_load(q))
    rid[tid] = Int32(g)
    for c in range(W):
        var x = Float32(0)
        if g >= 0 and c < w:
            x = a.unsafe_load(g * n + k0 + c)
        v[tid * VS + c] = x
    var alive = g >= 0
    if tid < W:
        wslot[tid] = Int32(-1)
    barrier()
    for c in range(w):
        # pivot search over the alive rows: simdgroup butterfly, then the
        # LTS_SG simdgroup winners folded by every thread
        var cv = Float32(-1)
        var cr = Int32(-1)
        var cs = Int32(-1)
        if alive:
            cv = abs(ftz(v[tid * VS + c]))
            cr = Int32(g)
            cs = Int32(tid)
        comptime for sh in range(5):
            var ov = shuffle_xor(cv, UInt32(1 << sh))
            var orr = shuffle_xor(cr, UInt32(1 << sh))
            var os = shuffle_xor(cs, UInt32(1 << sh))
            if _lts_better(ov, orr, os, cv, cr, cs):
                cv = ov
                cr = orr
                cs = os
        if tid % 32 == 0:
            pv[sg] = cv
            pr[sg] = cr
            ps[sg] = cs
        barrier()
        var bv = pv[0]
        var br = pr[0]
        var bs = ps[0]
        comptime for q in range(1, LTS_SG):
            if _lts_better(pv[q], pr[q], ps[q], bv, br, bs):
                bv = pv[q]
                br = pr[q]
                bs = ps[q]
        var s = Int(bs)
        if s >= 0:
            if tid == 0:
                wslot[c] = Int32(s)
            var d = ftz(v[s * VS + c])
            if tid == s:
                alive = False
            elif alive and d != Float32(0):
                var l = div0(v[tid * VS + c], d)
                v[tid * VS + c] = l
                for c2 in range(c + 1, w):
                    v[tid * VS + c2] = ftz(identical_mul_add(-l, ftz(v[s * VS + c2]), ftz(v[tid * VS + c2])))
        barrier()
    if tid < w:
        var ws = Int(wslot[tid])
        cout.unsafe_store(b * W + tid, Float32(Int(rid[ws])) if ws >= 0 else Float32(-1))
    elif tid < W:
        cout.unsafe_store(b * W + tid, Float32(-1))
    if Int(last) != 0:
        # L11 \ U11 of the winners (identity padding past w)
        if tid < W:
            var ws = Int(wslot[tid]) if tid < w else -1
            for c2 in range(W):
                var x = Float32(1) if tid == c2 else Float32(0)
                if ws >= 0 and c2 < w:
                    x = v[ws * VS + c2]
                u11.unsafe_store(tid * W + c2, x)
        if tid == 0:
            # getrf's sequential swaps that bring winner t to row k0 + t, and
            # the map position -> original row of every moved row
            var e_n = 0
            for t in range(w):
                var tgt = Int(rid[Int(wslot[t])])
                var q = tgt
                for e in range(e_n):
                    if Int(mrow[e]) == tgt:
                        q = Int(mpos[e])
                var pos = k0 + t
                piv.unsafe_store(pos, Int32(q))
                if q != pos:
                    var v1 = pos
                    for e in range(e_n):
                        if Int(mpos[e]) == pos:
                            v1 = Int(mrow[e])
                    var hit_p = False
                    var hit_q = False
                    for e in range(e_n):
                        if Int(mpos[e]) == pos:
                            mrow[e] = Int32(tgt)
                            hit_p = True
                        elif Int(mpos[e]) == q:
                            mrow[e] = Int32(v1)
                            hit_q = True
                    if not hit_p:
                        mpos[e_n] = Int32(pos)
                        mrow[e_n] = Int32(tgt)
                        e_n += 1
                    if not hit_q:
                        mpos[e_n] = Int32(q)
                        mrow[e_n] = Int32(v1)
                        e_n += 1
                var dd = ftz(v[Int(wslot[t]) * VS + t])
                var on = dd != Float32(0)
                act.unsafe_store(pos, Float32(1) if on else Float32(0))
                if not on and info.unsafe_load(0) == Float32(0):
                    info.unsafe_store(0, Float32(pos + 1))
            ecount[0] = Int32(e_n)
        barrier()
        var e_all = Int(ecount[0])
        var idx = tid
        while idx < e_all * W:
            var e = idx // W
            var c = idx % W
            var x = Float32(0)
            if c < w:
                x = a.unsafe_load(Int(mrow[e]) * n + k0 + c)
            orig.unsafe_store(idx, x)
            idx += RB
        if tid < e_all:
            mapb.unsafe_store(1 + tid, Float32(Int(mpos[tid])))
            mapb.unsafe_store(1 + 2 * W + tid, Float32(Int(mrow[tid])))
        if tid == 0:
            mapb.unsafe_store(0, Float32(e_all))


def lts_post_kernel(
    a: F32Ptr, u11: F32Ptr, orig: F32Ptr, mapb: F32Ptr, piv: I32Ptr, act: F32Ptr,
    k0_in: Int32, k1_in: Int32, kb_in: Int32, n_in: Int32, g_rows: Int32, g_left: Int32,
):
    """The panel's write-back, LFS_TPB threads a block, by block range:
    [0, g_rows) one thread per panel row i >= k0; [g_rows, g_rows + g_left)
    one thread per column j < k0 (the swaps); the rest one thread per column
    j >= k1 (the swaps, then for j < kb the U rows k0+1 .. k1-1 solved
    against L11, `lfm_swaps_trsm_kernel`'s statements with L11 from u11).
    The three ranges touch disjoint columns of `a`."""
    comptime W = LTS_W
    comptime PAGE_BYTES = (W * W + 4 * W) * 4
    comptime assert lib_smem_page_fits_for[COLUMN_APPLE, PAGE_BYTES](), "lts_post_kernel: U11 page"
    var us = stack_allocation[W * W, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var mp = stack_allocation[2 * W, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var mr = stack_allocation[2 * W, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var n = Int(n_in)
    var k0 = Int(k0_in)
    var k1 = Int(k1_in)
    var w = k1 - k0
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var gr = Int(g_rows)
    var gl = Int(g_left)
    if b < gr:
        var idx = tid
        while idx < W * W:
            us[idx] = u11.unsafe_load(idx)
            idx += LFS_TPB
        var e_all = Int(mapb.unsafe_load(0))
        if tid < e_all:
            mp[tid] = Int32(Int(mapb.unsafe_load(1 + tid)))
            mr[tid] = Int32(Int(mapb.unsafe_load(1 + 2 * W + tid)))
        barrier()
        var i = k0 + b * LFS_TPB + tid
        if i < n:
            if i < k1:
                for c in range(w):
                    a.unsafe_store(i * n + k0 + c, us[(i - k0) * W + c])
            else:
                var hit = -1
                for e in range(e_all):
                    if Int(mp[e]) == i:
                        hit = e
                var x = InlineArray[Float32, W](fill=Float32(0))
                comptime for c in range(W):
                    if c < w:
                        if hit >= 0:
                            x[c] = orig.unsafe_load(hit * W + c)
                        else:
                            x[c] = a.unsafe_load(i * n + k0 + c)
                # row i's steps k0 .. k1-1 in order (padding: U = I, x = 0)
                comptime for c in range(W):
                    var l = div0(x[c], us[c * W + c])
                    x[c] = l
                    comptime for c2 in range(c + 1, W):
                        x[c2] = ftz(identical_mul_add(-l, ftz(us[c * W + c2]), ftz(x[c2])))
                comptime for c in range(W):
                    if c < w:
                        a.unsafe_store(i * n + k0 + c, x[c])
    elif b < gr + gl:
        var j = (b - gr) * LFS_TPB + tid
        if j < k0:
            for k in range(k0, k1):
                lu_swap_elem(a, piv, k, j, n)
    else:
        var j = k1 + (b - gr - gl) * LFS_TPB + tid
        if j < n:
            for k in range(k0, k1):
                lu_swap_elem(a, piv, k, j, n)
            if j < Int(kb_in):
                for k in range(k0 + 1, k1):
                    var acc = ftz(a.unsafe_load(k * n + j))
                    for kp in range(k0, k):
                        if act.unsafe_load(kp) != Float32(0):
                            var l = u11.unsafe_load((k - k0) * W + kp - k0)
                            acc = ftz(identical_mul_add(-l, ftz(a.unsafe_load(kp * n + j)), ftz(acc)))
                    a.unsafe_store(k * n + j, acc)


def lu_tslu_panel(
    ctx: DeviceContext, a: F32Ptr, piv: I32Ptr, info: F32Ptr, act: F32Ptr,
    ca: F32Ptr, cb: F32Ptr, u11: F32Ptr, orig: F32Ptr, mapb: F32Ptr, k0: Int, k1: Int, kb: Int, n: Int,
) raises:
    """The panel [k0, k1) (k1 - k0 <= LTS_W) factored in place by tournament
    pivoting, piv / act / info set, its swaps applied to every column of `a`
    and, for columns k1 .. kb-1, the U rows solved (what `lu_fast_panel` plus
    `_lfm_cols(.., k0, k1, k1, kb, n, n, True)` do). ca / cb hold
    lts_cand_floats(n) floats each, u11 / orig / mapb LTS_U11_FLOATS /
    LTS_ORIG_FLOATS / LTS_MAP_FLOATS. Enqueued, no sync."""
    var sets = lts_sets(n - k0)
    ctx.enqueue_function[lts_tour_kernel](
        a, cb, ca, u11, orig, mapb, piv, act, info,
        Int32(k0), Int32(k1), Int32(n), Int32(1), Int32(0), Int32(1 if sets == 1 else 0),
        grid_dim=sets, block_dim=LTS_RB,
    )
    var cur_a = True
    while sets > 1:
        var nxt = (sets + 3) // 4
        ctx.enqueue_function[lts_tour_kernel](
            a, ca if cur_a else cb, cb if cur_a else ca, u11, orig, mapb, piv, act, info,
            Int32(k0), Int32(k1), Int32(n), Int32(0), Int32(sets * LTS_W), Int32(1 if nxt == 1 else 0),
            grid_dim=nxt, block_dim=LTS_RB,
        )
        cur_a = not cur_a
        sets = nxt
    var g_rows = lfs_blocks(n - k0)
    var g_left = (k0 + LFS_TPB - 1) // LFS_TPB
    var g_right = (n - k1 + LFS_TPB - 1) // LFS_TPB
    ctx.enqueue_function[lts_post_kernel](
        a, u11, orig, mapb, piv, act, Int32(k0), Int32(k1), Int32(kb), Int32(n), Int32(g_rows), Int32(g_left),
        grid_dim=g_rows + g_left + g_right, block_dim=LFS_TPB,
    )
