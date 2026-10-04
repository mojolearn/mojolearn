# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_MCD_BATCH_COMPAT (2026-10-04) is a separate opt-in repair:
compact support before legacy scalar 4096-term folds, FMA covariance/distance/LU cells,
and a parallel round-robin eigensolve using main's rotation and convergence
rules. It does NOT inherit the failed route's quality approval. Validate
with tools/mcd_compat_quality.py against current main before any default.
The initial experiment targets taxi (2 <= d <= 64); wide istella still
uses main and needs a separate extension after narrow quality passes.

MinCovDet's fast_mcd with every C-step of every candidate on the device
(lane/apple-fast-robust, 2026-10-02; FAST + Apple only, OPT-IN since
2026-10-03: `-D MOJOLEARN_MCD_DEVICE_CSTEPS` turns it on; the default is the
per-candidate kit route). M3 A/B: min-cov-det taxi 79,925 -> 215 ms; M2 A/B
robust-ee-taxi-x: elliptic-envelope taxi 64,578 -> 267.5 ms.

DROPPED-quality (tools/mcd_quality_ab.sh, M2, taxi 100k, 2026-10-03): it
lands on a different robust fit than the kit route. Jaccard of the flagged
Xq mask vs the kit route (OFF) .8805 MinCovDet (chi2 .975 cut) / .9645
EllipticEnvelope, below the .99 bar, while OFF matches IDENTICAL at .994 /
.999; location_ shifts 14% and covariance_ 18% (relative Frobenius), the
MinCovDet flag rate .231 -> .203. The raw h-subset covariance is singular
in every build (exact fit), but this route keeps rank 8 where OFF and
IDENTICAL keep rank 10. Kept as opt-in code for a future correct parallel
C-step.

x_decomp/kit_device.mojo's `fast_mcd_dev` runs x_decomp/mcd.mojo's search
one candidate and one C-step at a time: each C-step is ten to fifteen kit
launches with a host sync for the distances (the h-smallest selection ran
on the host), the log determinant and the Jacobi eigenpairs, and the taxi
board (100,000 x 16: 333 subsets x 10 trials x 2 steps, then 3,330
candidates x up to 30 steps on 1,500 rows, then 10 x 30 on all rows) is
tens of thousands of such syncs. Here the search is three PHASES, each a
set of candidates that step together:

  * the two permutations (`_perm`) are a bitonic sort of the draws' (float
    image, index) keys on the device (`mf_keys_kernel`, `mf_bitonic_kernel`);
  * a candidate's rows are a group of the permutation (phase A: subset i's
    n_ss rows; phase B: the first n_m of the second permutation; phase C: all
    rows); its support is a 0/1 mask over its rows;
  * a step: masked column sums per (candidate, 256-row tile, feature), the
    mean; masked centered products per (candidate, tile, feature pair), the
    covariance (`emp_cov`'s `1/h` scale); one thread per candidate takes the
    log determinant (`fast_logdet`: partial-pivot LU, the sign rule, the logs
    floored at FLT_MIN) and the C-step control (`det < prev_det and iters > 0
    and det != -inf`; at the exit sklearn's `use_prev` rule) and records
    which parity holds the chosen (mean, covariance, mask); one thread per
    still-active candidate takes `pinvh` (a cyclic Jacobi eigensolve, the
    `|w| > max|w| * d * eps` cut, V diag(1/w) V^T); one thread per
    (candidate, row) the Mahalanobis distance (`mahal`'s statements); one
    block per candidate the h smallest by (distance image, index) (an 8-pass
    radix select over the 32-bit image, then a block scan that takes the
    first `need` ties in index order), written as the next parity's mask;
  * the stop word (any candidate active) is read once per step; a candidate
    ranks in its group by (det image, index) with one thread per candidate
    counting the keys below its own, the kept ones hand their (mean,
    covariance) to the next phase in rank order.

The statements are mcd.mojo's (the kit's elementwise and fold cells are
ftz float32; folds are blocked here, the eigensolve is a one-thread Jacobi,
the `cut` is float32 since Metal has no float64), so the covariance matches
within float32, not bit for bit. X is uploaded once. d <= MF_DMAX; larger
d or a p == 1 input keeps main's path.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add
from x_decomp.cells import F32Ptr, I32Ptr, add, sub, mul, div0, sqrt0, log_floor, rand_cell
from x_decomp.device import xd_ctx, _down, _down_i, _launch_gemm_mma, DECOMP_FAST_GEMM_MMA
from x_decomp.mcd_mma import mc_center_kernel, mc_publish_matrix_kernel, mc_mahal_reduce_kernel
from x_decomp.kit import Mat
from x_decomp.cells import FOLD_BLOCK
from x_decomp.mcd_compat import mc_compact_kernel, mc_moment_kernel, mc_pinvh_kernel
from x_decomp.mcd_bmma import launch_gemm_mma_batched
from x_decomp.jacobi2 import dev_barrier

# FAILED gap26-mcdcompat-taxi at 948c4e7b1: B rejected by the batched
# eigensolve gate (A=70877.713 ms). Repair: make collective entry uniform
# before clearing needp for inactive singular candidates. Validation of
# this race repair is recorded below; keep opt-in and retain both gates.
# FAIL-quality gap26-mcdrepair-small-ready, ab4265c9a, M3 taxi cap3000:
# quality-fit A5595.80ms / B828.96ms (not full-board timing); location_rel
# .032817, covariance_rel .085678, precision_rel .999817, distances_rel
# .997423; support Jaccard .941431, raw support .798209; both raw ranks10.
# Both flag fractions1 are uninformative. This is a changed fitted model.
# Source audit: main DMcd.emp_cov/pinvh/mahal use DKit.mm -> Apple MMA,
# including split-K covariance at support sizes>=1024. COMPAT's scalar
# 4096-term folds do not replay that arithmetic. A causal first-divergence
# trace is still owed; do NOT interpret the race repair as quality approval.
# See docs/apple-fast/ab/mcd-compat-review.md for the semantics-preserving plan.
# New candidate after gap26-mcdrepair-small-ready's scalar/MMA mismatch.
# Keeps batching control/eigen work; main's actual MMA launches compute all
# three products with unchanged per-candidate shapes and split-K policy.
# FAST+Apple DEFAULT since source 9ac9871b3 (kernel 77520069e). M3, one run
# per arm, taxi board size: MinCovDet fit 70877.7 -> 3587.2 ms, EllipticEnvelope
# 70092.1 -> 3627.9 ms; capped EE 5613.6 -> 956.8 ms. Quality: gap26-mcd-mma-
# {small,ee-small,full,ee-full} all MCDQ-PAIR-PASS against the original 1%
# fitted-state / .99 support gates (small: covariance rel 4.4e-8, masks exact).
# MOJOLEARN_MCD_BATCH_MMA_OFF restores main's per-candidate path.
comptime MCD_BATCH_MMA = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and DECOMP_FAST_GEMM_MMA and not is_defined["MOJOLEARN_MCD_BATCH_MMA_OFF"]()
)

comptime MCD_BATCH_COMPAT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and (is_defined["MOJOLEARN_MCD_BATCH_COMPAT"]() or MCD_BATCH_MMA)
)

# DROP-quality: mcdq4 (M2 taxi 100k), flagged-mask Jaccard vs OFF
# .8805 MCD / .9645 EE fails .99; location/covariance shift 14%/18%.
# Opt-in pending a corrected C-step; evidence above and in
# docs/apple-fast/EXPERIMENTS.md (MCD_DEVICE_CSTEPS).
comptime MCD_DEVICE_CSTEPS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and (is_defined["MOJOLEARN_MCD_DEVICE_CSTEPS"]() or MCD_BATCH_COMPAT)
)

# DEFAULT (FAST + Apple, on top of MCD_BATCH_MMA; lane/apple-fast-w2-mcd2):
# each of the three per-candidate GEMM loops is ONE launch over the phase's
# candidates (x_decomp/mcd_bmma.mojo: main's tile kernel, same tiles / K
# windows / split-K per candidate); inactive candidates launch no GEMM work.
# M3, one run per arm: MinCovDet taxi 3002.9 -> 294.3 ms (w2-mcdb-t-mcd-taxi),
# EllipticEnvelope taxi 3010.3 -> 299.3 ms (w2-mcdb-t-ee-taxi); quality
# w2-mcdb-q-mcd-taxi, w2-mcdb-q-ee-taxi MCDQ-PAIR-PASS (1% fitted state,
# .99 support). MOJOLEARN_MCD_BMMA_OFF restores the per-candidate launches.
comptime MCD_BMMA = MCD_BATCH_MMA and not is_defined["MOJOLEARN_MCD_BMMA_OFF"]()
# MCD_WIDE, DEFAULT (FAST + Apple, only with MCD_BMMA on; lane/apple-fast-w2-mcd2):
# this batched search for 64 < d <= MF_WIDE_DMAX too
# (istella d = 220 fell back to fast_mcd_dev: per candidate, per C-step kit
# launches + host syncs + a 220-wide round-robin eigh with a sync per sweep,
# killed > 20 min on the board). Wide d adds a block-per-candidate LU
# (mf_det_wide_kernel, the same pivots / cells / log sum as mf_det_kernel)
# and mc_pinvh_kernel with 256-entry eigen tables; everything else is the
# narrow MMA path (the distance never holds a row in registers there).
# Quality w2-mcdw-q-mcd-istella-r1, w2-mcdw-q-ee-istella-r1 MCDQ-PAIR-PASS
# (istella cap 3000, B = BMMA + WIDE). M3 full istella, B only (w2-mcdw-t-*):
# MinCovDet 86333.5 ms, EllipticEnvelope 86640.8 ms; main's fallback measured
# 1253937 ms for EE istella (py2mojo-decomp-elliptic-envelope-istella arm A).
# Cost: ~20 GB peak memory at the istella board size. MOJOLEARN_MCD_WIDE_OFF (or
# MOJOLEARN_MCD_BMMA_OFF) restores the fast_mcd_dev fallback for d > 64.
comptime MCD_WIDE = MCD_BMMA and not is_defined["MOJOLEARN_MCD_WIDE_OFF"]()
comptime MF_WIDE_DMAX = 256

comptime U64Ptr = MutPointer[UInt64, MutAnyOrigin]
comptime MF_TPB = 256
#: Rows per tile of the masked moments.
comptime MF_TILE = 256
#: Largest feature count served here (the per-row distance keeps the
#: centered row in registers).
comptime MF_DMAX = 64
#: Jacobi sweeps of the per-candidate eigensolve.
comptime MF_SWEEPS = 60
#: `_F32_EPS` of mcd.mojo as float32, `_FLT_MIN` likewise.
comptime MF_F32_EPS = Float32(1.1920928955078125e-07)
comptime MF_FLT_MIN = Float32(1.1754943508222875e-38)
#: Error words: NaN draw/distance, no pinvh, eigensolve budget, Frobenius drift.
comptime MF_ERR = 4
comptime MF_SH_INT = MF_TPB * 16 + 32 + MF_TPB


def _blocks(count: Int) -> Int:
    return max((count + MF_TPB - 1) // MF_TPB, 1)


@always_inline
def _neg_inf32() -> Float32:
    return bitcast[DType.float32](UInt32(0xFF800000))


@always_inline
def _pos_inf32() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


@always_inline
def _image(v: Float32) -> UInt32:
    """`f32_key` (x_decomp/moves.mojo): the monotone uint32 image, -0.0 as
    +0.0; a NaN is flagged by the kernels that produce one."""
    var w = v
    if w == Float32(0):
        w = Float32(0)
    var ub = bitcast[DType.uint32](w)
    return ub ^ UInt32(0xFFFFFFFF) if (ub >> 31) == 1 else ub | UInt32(0x80000000)


@always_inline
def _row_of(rows: I32Ptr, ident: Int32, g: Int, r: Int, i: Int) -> Int:
    """Row i of row group g: the identity or the permutation's group."""
    if ident != 0:
        return i
    return Int(rows.unsafe_load(g * r + i))


# ------------------------------------------------------------ permutations
def mf_keys_kernel(dst: U64Ptr, npad: Int32, count: Int32, seed: UInt32, stream: UInt32):
    """Key i = (image of draw i) << 32 | i; the padding above `count` sorts
    last."""
    var i = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    if i < Int(npad):
        if i < Int(count):
            var v = rand_cell(i, seed, stream, 0)
            dst.unsafe_store(i, (UInt64(_image(v)) << 32) | UInt64(i))
        else:
            dst.unsafe_store(i, UInt64(0xFFFFFFFFFFFFFFFF))


def mf_bitonic_kernel(keys: U64Ptr, npad: Int32, j: Int32, k: Int32):
    """One compare-exchange step of the bitonic network (ascending)."""
    var i = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    if i < Int(npad):
        var jj = Int(j)
        var ixj = i ^ jj
        if ixj > i:
            var a = keys.unsafe_load(i)
            var b = keys.unsafe_load(ixj)
            var up = (i & Int(k)) == 0
            if (up and a > b) or ((not up) and a < b):
                keys.unsafe_store(i, b)
                keys.unsafe_store(ixj, a)


def mf_extract_kernel(keys: U64Ptr, rows: I32Ptr, count: Int32):
    var i = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    if i < Int(count):
        rows.unsafe_store(i, Int32(keys.unsafe_load(i) & UInt64(0xFFFFFFFF)))


def mf_draw_kernel(dist: F32Ptr, nc: Int32, r: Int32, seed: UInt32, stream0: UInt32, err: I32Ptr):
    """Candidate c's draws over its r rows: stream0 + c (mcd.mojo's
    `perm_head_sorted`, one draw per candidate in order)."""
    var t = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    var rr = Int(r)
    if t < Int(nc) * rr:
        var c = t // rr
        var i = t - c * rr
        var v = rand_cell(i, seed, stream0 + UInt32(c), 0)
        if v != v:
            err.unsafe_store(0, Int32(1))
        dist.unsafe_store(t, v)


def mf_fill_i_kernel(dst: I32Ptr, count: Int32, v: Int32):
    var i = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    if i < Int(count):
        dst.unsafe_store(i, v)


def mf_fill_f_kernel(dst: F32Ptr, count: Int32, v: Float32):
    var i = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    if i < Int(count):
        dst.unsafe_store(i, v)


# ---------------------------------------------------------------- the step
def mf_colsum_kernel(
    x: F32Ptr, rows: I32Ptr, ident: Int32, per: Int32, mask: I32Ptr, part: F32Ptr, nc: Int32, r: Int32,
    tiles: Int32, d: Int32, active: I32Ptr,
):
    """part[(c, tile, j)] = the masked column sum over the tile's rows."""
    var t = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var nt = Int(tiles)
    if t < Int(nc) * nt * dd:
        var j = t % dd
        var ct = t // dd
        var tile = ct % nt
        var c = ct // nt
        if active.unsafe_load(c) != 0:
            var rr = Int(r)
            var g = c // Int(per)
            var lo = tile * MF_TILE
            var hi = min(lo + MF_TILE, rr)
            var acc = Float32(0)
            for i in range(lo, hi):
                if mask.unsafe_load(c * rr + i) != 0:
                    acc = add(acc, x.unsafe_load(_row_of(rows, ident, g, rr, i) * dd + j))
            part.unsafe_store(t, acc)


def mf_mean_kernel(part: F32Ptr, loc: F32Ptr, nc: Int32, tiles: Int32, d: Int32, hinv: Float32, active: I32Ptr):
    """loc[c, j] = (sum of the tiles) * (1/h): `colmean`."""
    var t = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    var dd = Int(d)
    if t < Int(nc) * dd:
        var c = t // dd
        var j = t - c * dd
        if active.unsafe_load(c) != 0:
            var nt = Int(tiles)
            var acc = Float32(0)
            for tile in range(nt):
                acc = add(acc, part.unsafe_load((c * nt + tile) * dd + j))
            loc.unsafe_store(t, mul(acc, hinv))


def mf_cov_part_kernel(
    x: F32Ptr, rows: I32Ptr, ident: Int32, per: Int32, mask: I32Ptr, loc: F32Ptr, pj: I32Ptr, pk: I32Ptr,
    part: F32Ptr, nc: Int32, r: Int32, tiles: Int32, d: Int32, npair: Int32, active: I32Ptr,
):
    """part[(c, tile, q)] = the masked sum over the tile of (x_j - loc_j)
    (x_k - loc_k) for pair q = (j, k), j <= k: `emp_cov`'s centered Gram."""
    var t = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    var np = Int(npair)
    var nt = Int(tiles)
    if t < Int(nc) * nt * np:
        var q = t % np
        var ct = t // np
        var tile = ct % nt
        var c = ct // nt
        if active.unsafe_load(c) != 0:
            var dd = Int(d)
            var rr = Int(r)
            var g = c // Int(per)
            var j = Int(pj.unsafe_load(q))
            var k = Int(pk.unsafe_load(q))
            var mj = loc.unsafe_load(c * dd + j)
            var mk = loc.unsafe_load(c * dd + k)
            var lo = tile * MF_TILE
            var hi = min(lo + MF_TILE, rr)
            var acc = Float32(0)
            for i in range(lo, hi):
                if mask.unsafe_load(c * rr + i) != 0:
                    var row = _row_of(rows, ident, g, rr, i) * dd
                    var a = sub(x.unsafe_load(row + j), mj)
                    var b = sub(x.unsafe_load(row + k), mk)
                    acc = add(acc, mul(a, b))
            part.unsafe_store(t, acc)


def mf_cov_kernel(
    part: F32Ptr, cov: F32Ptr, pj: I32Ptr, pk: I32Ptr, nc: Int32, tiles: Int32, d: Int32, npair: Int32,
    hinv: Float32, active: I32Ptr,
):
    var t = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    var np = Int(npair)
    if t < Int(nc) * np:
        var c = t // np
        var q = t - c * np
        if active.unsafe_load(c) != 0:
            var nt = Int(tiles)
            var dd = Int(d)
            var acc = Float32(0)
            for tile in range(nt):
                acc = add(acc, part.unsafe_load((c * nt + tile) * np + q))
            var v = mul(acc, hinv)
            var j = Int(pj.unsafe_load(q))
            var k = Int(pk.unsafe_load(q))
            cov.unsafe_store(c * dd * dd + j * dd + k, v)
            cov.unsafe_store(c * dd * dd + k * dd + j, v)


@always_inline
def _logdet_kern(a: F32Ptr, o: Int, dd: Int) -> Float32:
    """`fast_logdet` on the d x d block at a[o:]: partial-pivot LU in place
    (x_decomp/host.mojo `lu`: the first largest |a_ik| by `>`), -inf for a
    zero pivot or an odd sign, else the ascending sum of log(max(|u_ii|,
    FLT_MIN))."""
    var neg = 0
    for k in range(dd):
        var p = k
        var best = abs(a.unsafe_load(o + k * dd + k))
        for i in range(k + 1, dd):
            var v = abs(a.unsafe_load(o + i * dd + k))
            if v > best:
                best = v
                p = i
        if p != k:
            neg += 1
            for j in range(dd):
                var t = a.unsafe_load(o + k * dd + j)
                a.unsafe_store(o + k * dd + j, a.unsafe_load(o + p * dd + j))
                a.unsafe_store(o + p * dd + j, t)
        var piv = a.unsafe_load(o + k * dd + k)
        if piv == Float32(0):
            return _neg_inf32()
        if piv < Float32(0):
            neg += 1
        for i in range(k + 1, dd):
            var f = a.unsafe_load(o + i * dd + k) / piv
            comptime if MCD_BATCH_COMPAT:
                f = div0(a.unsafe_load(o + i * dd + k), ftz(piv))
            a.unsafe_store(o + i * dd + k, f)
            for j in range(k + 1, dd):
                comptime if MCD_BATCH_COMPAT:
                    a.unsafe_store(o + i * dd + j, ftz(identical_mul_add(-f, ftz(a.unsafe_load(o + k * dd + j)), ftz(a.unsafe_load(o + i * dd + j)))))
                else:
                    a.unsafe_store(o + i * dd + j, a.unsafe_load(o + i * dd + j) - f * a.unsafe_load(o + k * dd + j))
    if neg % 2 != 0:
        return _neg_inf32()
    var total = Float32(0)
    for i in range(dd):
        total = add(total, log_floor(abs(a.unsafe_load(o + i * dd + i)), MF_FLT_MIN))
    return total


def mf_det_kernel(
    cov: F32Ptr, work: F32Ptr, det: F32Ptr, det_prev: F32Ptr, nc: Int32, d: Int32, step: Int32, n_iter: Int32,
    active: I32Ptr, needp: I32Ptr, fin: I32Ptr, err: I32Ptr,
):
    """One thread per candidate: the log determinant of this parity's
    covariance, then mcd.mojo's C-step control. `fin[c]` records the parity
    of the chosen (mean, covariance, mask) when the candidate stops;
    `needp[c]` asks for one pinvh of a candidate whose first determinant is
    -inf (mcd.mojo's `if det == -inf: P = pinvh(cov)`)."""
    var c = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    if c < Int(nc) and active.unsafe_load(c) != 0:
        var dd = Int(d)
        var o = c * dd * dd
        for q in range(dd * dd):
            work.unsafe_store(o + q, cov.unsafe_load(o + q))
        var cur = _logdet_kern(work, o, dd)
        det.unsafe_store(c, cur)
        var s = Int(step)
        var iters = Int(n_iter) - s
        var prev = _pos_inf32()
        if s > 0:
            prev = det_prev.unsafe_load(c)
        if s == 0 and cur == _neg_inf32():
            needp.unsafe_store(c, Int32(1))
        var cont = cur < prev and iters > 0 and cur != _neg_inf32()
        if not cont:
            active.unsafe_store(c, Int32(0))
            # has_p: a pinvh ran in an iteration (s > 0) or at the -inf start
            if s == 0 and cur != _neg_inf32():
                err.unsafe_store(1, Int32(1))
            var use_prev = s > 0 and cur > prev
            if iters == 0:
                use_prev = False
            fin.unsafe_store(c, Int32(((s + 1) % 2) if use_prev else (s % 2)))


def mf_det_wide_kernel(
    cov: F32Ptr, work: F32Ptr, det: F32Ptr, det_prev: F32Ptr, nc: Int32, d: Int32, step: Int32, n_iter: Int32,
    active: I32Ptr, needp: I32Ptr, fin: I32Ptr, err: I32Ptr,
):
    """MCD_WIDE: mf_det_kernel with one BLOCK per candidate. The
    LU is _logdet_kern's (MCD_BATCH_COMPAT cells): the pivot is the first largest
    |a_ik| by `>` (a block reduction, ties to the lower row), the row swap,
    the factors and the trailing update are parallel over rows / cells (row
    k is not written during step k, so each cell sees _logdet_kern's operands),
    the log sum is one thread's ascending sum. Then the same C-step control."""
    var c = Int(block_idx.x)
    if active.unsafe_load(c) == 0:
        return
    var tid = Int(thread_idx.x)
    var dd = Int(d)
    var o = c * dd * dd
    var a = work + o
    var sv = stack_allocation[MF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var si = stack_allocation[MF_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var z = tid
    while z < dd * dd:
        a.unsafe_store(z, cov.unsafe_load(o + z))
        z += MF_TPB
    dev_barrier()
    var neg = 0  # thread 0's count
    var zero = False
    for k in range(dd):
        var best = Float32(-1)
        var bi = dd
        var i = k + tid
        while i < dd:
            var v = abs(a.unsafe_load(i * dd + k))
            if v > best:
                best = v
                bi = i
            i += MF_TPB
        sv[tid] = best
        si[tid] = Int32(bi)
        dev_barrier()
        var w = MF_TPB // 2
        while w > 0:
            if tid < w:
                var v2 = sv[tid + w]
                var i2 = si[tid + w]
                if v2 > sv[tid] or (v2 == sv[tid] and i2 < si[tid]):
                    sv[tid] = v2
                    si[tid] = i2
            dev_barrier()
            w //= 2
        var p = Int(si[0])
        if p >= dd:
            p = k  # every |a_ik| unordered (NaN): _logdet_kern keeps p = k
        if p != k:
            if tid == 0:
                neg += 1
            var j = tid
            while j < dd:
                var t = a.unsafe_load(k * dd + j)
                a.unsafe_store(k * dd + j, a.unsafe_load(p * dd + j))
                a.unsafe_store(p * dd + j, t)
                j += MF_TPB
        dev_barrier()
        var piv = a.unsafe_load(k * dd + k)
        if piv == Float32(0):
            zero = True
            break
        if piv < Float32(0) and tid == 0:
            neg += 1
        var r = k + 1 + tid
        while r < dd:
            a.unsafe_store(r * dd + k, div0(a.unsafe_load(r * dd + k), ftz(piv)))
            r += MF_TPB
        dev_barrier()
        var span = dd - k - 1
        var q = tid
        while q < span * span:
            var ii = k + 1 + q // span
            var jj = k + 1 + q % span
            var f = a.unsafe_load(ii * dd + k)
            a.unsafe_store(ii * dd + jj, ftz(identical_mul_add(-f, ftz(a.unsafe_load(k * dd + jj)), ftz(a.unsafe_load(ii * dd + jj)))))
            q += MF_TPB
        dev_barrier()
    if tid != 0:
        return
    var cur = _neg_inf32()
    if not zero and neg % 2 == 0:
        var total = Float32(0)
        for i in range(dd):
            total = add(total, log_floor(abs(a.unsafe_load(i * dd + i)), MF_FLT_MIN))
        cur = total
    det.unsafe_store(c, cur)
    var s = Int(step)
    var iters = Int(n_iter) - s
    var prev = _pos_inf32()
    if s > 0:
        prev = det_prev.unsafe_load(c)
    if s == 0 and cur == _neg_inf32():
        needp.unsafe_store(c, Int32(1))
    var cont = cur < prev and iters > 0 and cur != _neg_inf32()
    if not cont:
        active.unsafe_store(c, Int32(0))
        if s == 0 and cur != _neg_inf32():
            err.unsafe_store(1, Int32(1))
        var use_prev = s > 0 and cur > prev
        if iters == 0:
            use_prev = False
        fin.unsafe_store(c, Int32(((s + 1) % 2) if use_prev else (s % 2)))


def mf_pinvh_kernel(cov: F32Ptr, a: F32Ptr, v: F32Ptr, p: F32Ptr, nc: Int32, d: Int32, active: I32Ptr, needp: I32Ptr):
    """One thread per candidate: `pinvh` of its covariance (a cyclic
    Jacobi eigensolve in a[c], V in v[c]; the cut |w| > max|w| * d * eps;
    P = V diag(1/w) V^T with k ascending) into p[c]."""
    var c = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    if c < Int(nc) and (active.unsafe_load(c) != 0 or needp.unsafe_load(c) != 0):
        needp.unsafe_store(c, Int32(0))
        var dd = Int(d)
        var o = c * dd * dd
        var fro = Float32(0)
        for q in range(dd * dd):
            var x = cov.unsafe_load(o + q)
            a.unsafe_store(o + q, x)
            fro = fro + x * x
        for q in range(dd * dd):
            v.unsafe_store(o + q, Float32(0))
        for i in range(dd):
            v.unsafe_store(o + i * dd + i, Float32(1))
        var limit = Float32(1e-14) * fro
        for _ in range(MF_SWEEPS):
            var off = Float32(0)
            for pp in range(dd - 1):
                for qq in range(pp + 1, dd):
                    var apq = a.unsafe_load(o + pp * dd + qq)
                    off = off + apq * apq
            if not (off > limit):
                break
            for pp in range(dd - 1):
                for qq in range(pp + 1, dd):
                    var apq = a.unsafe_load(o + pp * dd + qq)
                    if apq == Float32(0):
                        continue
                    var theta = (a.unsafe_load(o + qq * dd + qq) - a.unsafe_load(o + pp * dd + pp)) / (Float32(2) * apq)
                    var at = abs(theta)
                    var tt = Float32(1) / (at + sqrt0(theta * theta + Float32(1)))
                    if theta < Float32(0):
                        tt = -tt
                    var cc = Float32(1) / sqrt0(tt * tt + Float32(1))
                    var ss = tt * cc
                    for k in range(dd):
                        var akp = a.unsafe_load(o + k * dd + pp)
                        var akq = a.unsafe_load(o + k * dd + qq)
                        a.unsafe_store(o + k * dd + pp, cc * akp - ss * akq)
                        a.unsafe_store(o + k * dd + qq, ss * akp + cc * akq)
                    for k in range(dd):
                        var apk = a.unsafe_load(o + pp * dd + k)
                        var aqk = a.unsafe_load(o + qq * dd + k)
                        a.unsafe_store(o + pp * dd + k, cc * apk - ss * aqk)
                        a.unsafe_store(o + qq * dd + k, ss * apk + cc * aqk)
                    for k in range(dd):
                        var vkp = v.unsafe_load(o + k * dd + pp)
                        var vkq = v.unsafe_load(o + k * dd + qq)
                        v.unsafe_store(o + k * dd + pp, cc * vkp - ss * vkq)
                        v.unsafe_store(o + k * dd + qq, ss * vkp + cc * vkq)
        # the cut and the kept reciprocals (mcd.mojo `pinvh`; the cut in float32)
        var wmax = abs(a.unsafe_load(o))
        for j in range(1, dd):
            var w = abs(a.unsafe_load(o + j * dd + j))
            if w > wmax:
                wmax = w
        var cut = mul(mul(wmax, Float32(dd)), MF_F32_EPS)
        for j in range(dd):
            var w = a.unsafe_load(o + j * dd + j)
            var inv = div0(Float32(1), w) if abs(w) > cut else Float32(0)
            a.unsafe_store(o + j, inv)   # row 0 of the finished a holds inv (the diagonal was read already)
        for i in range(dd):
            for j in range(dd):
                var acc = Float32(0)
                for k in range(dd):
                    acc = add(acc, mul(mul(v.unsafe_load(o + i * dd + k), a.unsafe_load(o + k)), v.unsafe_load(o + j * dd + k)))
                p.unsafe_store(o + i * dd + j, acc)


def mf_dist_kernel(
    x: F32Ptr, rows: I32Ptr, ident: Int32, per: Int32, loc0: F32Ptr, loc1: F32Ptr, par: Int32, fin: I32Ptr,
    use_fin: Int32, p: F32Ptr, dist: F32Ptr, nc: Int32, r: Int32, d: Int32, active: I32Ptr, err: I32Ptr,
):
    """dist[c, i] = `mahal`: xc = x_row - loc; (xc P) . xc, k then j
    ascending. The mean is this parity's, or the candidate's chosen parity
    (`use_fin`, the final distances); an inactive candidate is skipped
    unless `use_fin`."""
    var t = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    var rr = Int(r)
    if t < Int(nc) * rr:
        var c = t // rr
        var i = t - c * rr
        if use_fin != 0 or active.unsafe_load(c) != 0:
            var dd = Int(d)
            var g = c // Int(per)
            var row = _row_of(rows, ident, g, rr, i) * dd
            var pr = Int(fin.unsafe_load(c)) if use_fin != 0 else Int(par)
            var xc = InlineArray[Float32, MF_DMAX](fill=Float32(0))
            for k in range(dd):
                var m = loc1.unsafe_load(c * dd + k) if pr == 1 else loc0.unsafe_load(c * dd + k)
                xc[k] = sub(x.unsafe_load(row + k), m)
            var o = c * dd * dd
            var acc = Float32(0)
            for j in range(dd):
                var tj = Float32(0)
                for k in range(dd):
                    comptime if MCD_BATCH_COMPAT:
                        tj = ftz(identical_mul_add(ftz(xc[k]), ftz(p.unsafe_load(o + k * dd + j)), tj))
                    else:
                        tj = add(tj, mul(xc[k], p.unsafe_load(o + k * dd + j)))
                acc = add(acc, mul(tj, xc[j]))
            if acc != acc:
                err.unsafe_store(0, Int32(1))
            dist.unsafe_store(t, acc)


def mf_select_kernel(dist: F32Ptr, mask: I32Ptr, r: Int32, h: Int32, active: I32Ptr):
    """One block per candidate: the h rows with the smallest (image, index)
    keys of its r distances (`smallest_sorted`'s set) as a 0/1 mask. An
    8-pass radix select over the 32-bit image (a 16-bin count per pass,
    every thread's private bins folded across the block), then a pass in
    index order that takes every row below the selected image and the
    first `need` rows at it (a block scan of the ties)."""
    var c = Int(block_idx.x)
    if active.unsafe_load(c) == 0:
        return
    var tid = Int(thread_idx.x)
    var rr = Int(r)
    var base = c * rr
    var sh = stack_allocation[MF_SH_INT, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tot = 16 * MF_TPB
    var scan = tot + 32
    var prefix = UInt32(0)
    var need = Int(h)
    var shift = 28
    var first = True
    while shift >= 0:
        var cnt = InlineArray[Int32, 16](fill=Int32(0))
        var i = tid
        while i < rr:
            var im = _image(dist.unsafe_load(base + i))
            if first or (im >> UInt32(shift + 4)) == prefix:
                cnt[Int((im >> UInt32(shift)) & UInt32(15))] += 1
            i += MF_TPB
        for b in range(16):
            sh[tid * 16 + b] = cnt[b]
        barrier()
        if tid < 16:
            var s = Int32(0)
            for t in range(MF_TPB):
                s += sh[t * 16 + tid]
            sh[tot + tid] = s
        barrier()
        var acc = 0
        var dsel = 15
        var found = False
        for b in range(16):
            if not found:
                var nb = Int(sh[tot + b])
                if acc + nb >= need:
                    dsel = b
                    found = True
                else:
                    acc += nb
        need -= acc
        prefix = (prefix << 4) | UInt32(dsel)
        first = False
        shift -= 4
        barrier()
    var vstar = prefix
    var done = 0
    var i0 = 0
    while i0 < rr:
        var i = i0 + tid
        var valid = i < rr
        var eq = 0
        var lt = False
        if valid:
            var im = _image(dist.unsafe_load(base + i))
            lt = im < vstar
            eq = 1 if im == vstar else 0
        sh[scan + tid] = Int32(eq)
        barrier()
        var off = 1
        while off < MF_TPB:
            var v = sh[scan + tid]
            var a = sh[scan + tid - off] if tid >= off else Int32(0)
            barrier()
            sh[scan + tid] = v + a
            barrier()
            off *= 2
        var rank = Int(sh[scan + tid]) - eq
        var total = Int(sh[scan + MF_TPB - 1])
        if valid:
            var take = lt or (eq == 1 and done + rank < need)
            mask.unsafe_store(base + i, Int32(1) if take else Int32(0))
        done += total
        barrier()
        i0 += MF_TPB


def mf_any_kernel(active: I32Ptr, nc: Int32, dst: I32Ptr):
    var c = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    if c < Int(nc) and active.unsafe_load(c) != 0:
        dst.unsafe_store(0, Int32(1))


def mf_rank_kernel(det0: F32Ptr, det1: F32Ptr, fin: I32Ptr, nc: Int32, per: Int32, keep: Int32, order: I32Ptr):
    """One thread per candidate: its rank in its group of `per` by (det
    image, index) (`_order_by_det`); ranks below `keep` write
    order[g * keep + rank] = c."""
    var c = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    if c < Int(nc):
        var pp = Int(per)
        var g = c // pp
        var lo = g * pp
        var hi = min(lo + pp, Int(nc))
        var mine = _image(det1.unsafe_load(c) if fin.unsafe_load(c) == 1 else det0.unsafe_load(c))
        var rank = 0
        for o in range(lo, hi):
            var im = _image(det1.unsafe_load(o) if fin.unsafe_load(o) == 1 else det0.unsafe_load(o))
            if im < mine or (im == mine and o < c):
                rank += 1
        if rank < Int(keep):
            order.unsafe_store(g * Int(keep) + rank, Int32(c))


def mf_handoff_kernel(
    order: I32Ptr, fin: I32Ptr, loc0: F32Ptr, loc1: F32Ptr, cov0: F32Ptr, cov1: F32Ptr, loc_dst: F32Ptr,
    cov_dst: F32Ptr, nc: Int32, d: Int32,
):
    """Candidate c of the next phase takes the chosen (mean, covariance) of
    order[c] (one thread per element)."""
    var t = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var cells = dd + dd * dd
    if t < Int(nc) * cells:
        var c = t // cells
        var e = t - c * cells
        var src = Int(order.unsafe_load(c))
        var pr = Int(fin.unsafe_load(src))
        if e < dd:
            var v = loc1.unsafe_load(src * dd + e) if pr == 1 else loc0.unsafe_load(src * dd + e)
            loc_dst.unsafe_store(c * dd + e, v)
        else:
            var q = e - dd
            var v = cov1.unsafe_load(src * dd * dd + q) if pr == 1 else cov0.unsafe_load(src * dd * dd + q)
            cov_dst.unsafe_store(c * dd * dd + q, v)


def mf_scatter_kernel(
    rows: I32Ptr, ident: Int32, mask0: I32Ptr, mask1: I32Ptr, dist: F32Ptr, fin: I32Ptr, best: I32Ptr, r: Int32,
    sup_out: I32Ptr, dist_out: F32Ptr,
):
    """The best candidate's support and distances scattered to its rows'
    ids (mcd.mojo's `support[selection[...]]`, `dist[selection[a]]`)."""
    var i = Int(block_idx.x) * MF_TPB + Int(thread_idx.x)
    var rr = Int(r)
    if i < rr:
        var c = Int(best.unsafe_load(0))
        var pr = Int(fin.unsafe_load(c))
        var m = mask1.unsafe_load(c * rr + i) if pr == 1 else mask0.unsafe_load(c * rr + i)
        var row = _row_of(rows, ident, 0, rr, i)
        sup_out.unsafe_store(row, m)
        dist_out.unsafe_store(row, dist.unsafe_load(c * rr + i))


# ------------------------------------------------------------------ driver
def _f(buf: DeviceBuffer[DType.float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


def _i(buf: DeviceBuffer[DType.int32]) -> I32Ptr:
    return I32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


def _u(buf: DeviceBuffer[DType.uint64]) -> U64Ptr:
    return U64Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


struct MfPhase(Movable):
    """One phase: nc candidates in groups of `per`, each over r rows (row
    group g of `rows`, or the identity), support size h, n_iter steps."""
    var nc: Int
    var per: Int
    var r: Int
    var ident: Int
    var h: Int
    var n_iter: Int
    var tiles: Int
    var mm_x: DeviceBuffer[DType.float32]
    var mm_y: DeviceBuffer[DType.float32]
    var mm_out: DeviceBuffer[DType.float32]
    var mm_v: DeviceBuffer[DType.float32]
    var mm_vw: DeviceBuffer[DType.float32]
    var mm_ran: DeviceBuffer[DType.int32]
    var selected: DeviceBuffer[DType.int32]
    var mask0: DeviceBuffer[DType.int32]
    var mask1: DeviceBuffer[DType.int32]
    var dist: DeviceBuffer[DType.float32]
    var loc0: DeviceBuffer[DType.float32]
    var loc1: DeviceBuffer[DType.float32]
    var cov0: DeviceBuffer[DType.float32]
    var cov1: DeviceBuffer[DType.float32]
    var pm: DeviceBuffer[DType.float32]
    var wa: DeviceBuffer[DType.float32]
    var wv: DeviceBuffer[DType.float32]
    var det0: DeviceBuffer[DType.float32]
    var det1: DeviceBuffer[DType.float32]
    var part: DeviceBuffer[DType.float32]
    var active: DeviceBuffer[DType.int32]
    var needp: DeviceBuffer[DType.int32]
    var fin: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, nc: Int, per: Int, r: Int, ident: Int, h: Int, n_iter: Int, d: Int,
                 npair: Int) raises:
        self.nc = nc
        self.per = per
        self.r = r
        self.ident = ident
        self.h = h
        self.n_iter = n_iter
        self.tiles = max((r + MF_TILE - 1) // MF_TILE, 1)
        comptime if MCD_BATCH_COMPAT:
            self.tiles = max((h + FOLD_BLOCK - 1) // FOLD_BLOCK, 1)
        var cr = max(nc * r, 1)
        var cd = max(nc * d, 1)
        var cdd = max(nc * d * d, 1)
        self.mm_x = ctx.enqueue_create_buffer[DType.float32](cr*d if MCD_BATCH_MMA else 1)
        self.mm_y = ctx.enqueue_create_buffer[DType.float32](cr*d if MCD_BATCH_MMA else 1)
        self.mm_out = ctx.enqueue_create_buffer[DType.float32](cdd if MCD_BATCH_MMA else 1)
        self.mm_v = ctx.enqueue_create_buffer[DType.float32](cdd if MCD_BATCH_MMA else 1)
        self.mm_vw = ctx.enqueue_create_buffer[DType.float32](cdd if MCD_BATCH_MMA else 1)
        self.mm_ran = ctx.enqueue_create_buffer[DType.int32](max(nc, 1) if MCD_BATCH_MMA else 1)
        self.selected = ctx.enqueue_create_buffer[DType.int32](cr if MCD_BATCH_COMPAT else 1)
        self.mask0 = ctx.enqueue_create_buffer[DType.int32](cr)
        self.mask1 = ctx.enqueue_create_buffer[DType.int32](cr)
        self.dist = ctx.enqueue_create_buffer[DType.float32](cr)
        self.loc0 = ctx.enqueue_create_buffer[DType.float32](cd)
        self.loc1 = ctx.enqueue_create_buffer[DType.float32](cd)
        self.cov0 = ctx.enqueue_create_buffer[DType.float32](cdd)
        self.cov1 = ctx.enqueue_create_buffer[DType.float32](cdd)
        self.pm = ctx.enqueue_create_buffer[DType.float32](cdd)
        self.wa = ctx.enqueue_create_buffer[DType.float32](cdd)
        self.wv = ctx.enqueue_create_buffer[DType.float32](cdd)
        self.det0 = ctx.enqueue_create_buffer[DType.float32](max(nc, 1))
        self.det1 = ctx.enqueue_create_buffer[DType.float32](max(nc, 1))
        var pw = max(d, npair)
        comptime if MCD_WIDE:
            if d > MF_DMAX:
                pw = d  # the MMA route folds column sums only (npair = 24,310 at d = 220)
        self.part = ctx.enqueue_create_buffer[DType.float32](max(nc * self.tiles * pw, 1))
        self.active = ctx.enqueue_create_buffer[DType.int32](max(nc, 1))
        self.needp = ctx.enqueue_create_buffer[DType.int32](max(nc, 1))
        self.fin = ctx.enqueue_create_buffer[DType.int32](max(nc, 1))


def _perm(ctx: DeviceContext, keys: DeviceBuffer[DType.uint64], npad: Int, count: Int, seed: Int, stream: Int,
          rows: DeviceBuffer[DType.int32], take: Int) raises:
    """`_perm`: the first `take` of argsort((draw, index)) into rows."""
    ctx.enqueue_function[mf_keys_kernel](
        _u(keys), Int32(npad), Int32(count), UInt32(seed & 0xFFFFFFFF), UInt32(stream & 0xFFFFFFFF),
        grid_dim=_blocks(npad), block_dim=MF_TPB,
    )
    var k = 2
    while k <= npad:
        var j = k // 2
        while j >= 1:
            ctx.enqueue_function[mf_bitonic_kernel](
                _u(keys), Int32(npad), Int32(j), Int32(k), grid_dim=_blocks(npad), block_dim=MF_TPB,
            )
            j //= 2
        k *= 2
    ctx.enqueue_function[mf_extract_kernel](_u(keys), _i(rows), Int32(take), grid_dim=_blocks(take), block_dim=MF_TPB)


def _mma_precision(ctx: DeviceContext, ph: MfPhase, d: Int) raises:
    comptime if MCD_BMMA:
        # gate = mm_ran (pinvh ran for the candidate), the publication's mask
        launch_gemm_mma_batched(ctx, _f(ph.mm_vw), _f(ph.mm_v), _f(ph.mm_out), d, d, d, False, True,
                                ph.nc, d*d, d*d, d*d, _i(ph.mm_ran), False)
    else:
        for c in range(ph.nc):
            var off = c*d*d
            _launch_gemm_mma(ctx, _f(ph.mm_vw)+off, _f(ph.mm_v)+off,
                             _f(ph.mm_out)+off, d, d, d, False, True)
    ctx.enqueue_function[mc_publish_matrix_kernel](
        _f(ph.mm_out), _f(ph.pm), _i(ph.mm_ran), Int32(ph.nc), Int32(d), Float32(1),
        grid_dim=_blocks(ph.nc*d*d), block_dim=MF_TPB,
    )


def _mma_covariance(ctx: DeviceContext, ph: MfPhase, dx: DeviceBuffer[DType.float32],
                    rows: DeviceBuffer[DType.int32], d: Int, par: Int, hinv: Float32) raises:
    ctx.enqueue_function[mc_center_kernel[MCD_BMMA]](
        _f(dx), _i(rows), _i(ph.selected), _f(ph.loc0), _f(ph.loc1),
        _f(ph.mm_x), _i(ph.active), _i(ph.fin), Int32(ph.nc), Int32(ph.r), Int32(d),
        Int32(ph.per), Int32(ph.ident), Int32(ph.h), Int32(1), Int32(par), Int32(0),
        grid_dim=_blocks(ph.nc*ph.h*d), block_dim=MF_TPB,
    )
    comptime if MCD_BMMA:
        launch_gemm_mma_batched(ctx, _f(ph.mm_x), _f(ph.mm_x), _f(ph.mm_out), d, ph.h, d, True, False,
                                ph.nc, ph.r*d, ph.r*d, d*d, _i(ph.active), False)
    else:
        for c in range(ph.nc):
            var x = _f(ph.mm_x)+c*ph.r*d
            _launch_gemm_mma(ctx, x, x, _f(ph.mm_out)+c*d*d, d, ph.h, d, True, False)
    ctx.enqueue_function[mc_publish_matrix_kernel](
        _f(ph.mm_out), _f(ph.cov1) if par == 1 else _f(ph.cov0), _i(ph.active),
        Int32(ph.nc), Int32(d), hinv, grid_dim=_blocks(ph.nc*d*d), block_dim=MF_TPB,
    )


def _mma_distance(ctx: DeviceContext, ph: MfPhase, dx: DeviceBuffer[DType.float32],
                  rows: DeviceBuffer[DType.int32], d: Int, par: Int, use_fin: Bool,
                  err: DeviceBuffer[DType.int32]) raises:
    ctx.enqueue_function[mc_center_kernel[MCD_BMMA]](
        _f(dx), _i(rows), _i(ph.selected), _f(ph.loc0), _f(ph.loc1),
        _f(ph.mm_x), _i(ph.active), _i(ph.fin), Int32(ph.nc), Int32(ph.r), Int32(d),
        Int32(ph.per), Int32(ph.ident), Int32(ph.r), Int32(0), Int32(par), Int32(1 if use_fin else 0),
        grid_dim=_blocks(ph.nc*ph.r*d), block_dim=MF_TPB,
    )
    comptime if MCD_BMMA:
        # gate = active, every candidate for the final distances (use_fin)
        launch_gemm_mma_batched(ctx, _f(ph.mm_x), _f(ph.pm), _f(ph.mm_y), ph.r, d, d, False, False,
                                ph.nc, ph.r*d, d*d, ph.r*d, _i(ph.active), use_fin)
    else:
        for c in range(ph.nc):
            var off = c*ph.r*d
            _launch_gemm_mma(ctx, _f(ph.mm_x)+off, _f(ph.pm)+c*d*d,
                             _f(ph.mm_y)+off, ph.r, d, d, False, False)
    ctx.enqueue_function[mc_mahal_reduce_kernel](
        _f(ph.mm_x), _f(ph.mm_y), _f(ph.dist), _i(ph.active), _i(err),
        Int32(ph.nc), Int32(ph.r), Int32(d), Int32(1 if use_fin else 0),
        grid_dim=_blocks(ph.nc*ph.r), block_dim=MF_TPB,
    )


def _compat_pinvh(ctx: DeviceContext, ph: MfPhase, cov: F32Ptr, d: Int,
                  err: DeviceBuffer[DType.int32]) raises:
    """mc_pinvh_kernel over the phase (64-entry tables; MCD_WIDE
    takes the 256-entry instance for d > MF_DMAX)."""
    comptime if MCD_WIDE:
        if d > MF_DMAX:
            ctx.enqueue_function[mc_pinvh_kernel[MCD_BATCH_MMA, MF_WIDE_DMAX]](
                cov, _f(ph.wa), _f(ph.wv), _f(ph.pm), Int32(ph.nc), Int32(d), _i(ph.active), _i(ph.needp),
                _i(err), _f(ph.mm_v), _f(ph.mm_vw), _i(ph.mm_ran),
                grid_dim=ph.nc, block_dim=MF_TPB,
            )
            return
    ctx.enqueue_function[mc_pinvh_kernel[MCD_BATCH_MMA]](
        cov, _f(ph.wa), _f(ph.wv), _f(ph.pm), Int32(ph.nc), Int32(d), _i(ph.active), _i(ph.needp), _i(err),
        _f(ph.mm_v), _f(ph.mm_vw), _i(ph.mm_ran),
        grid_dim=ph.nc, block_dim=MF_TPB,
    )


def _det(ctx: DeviceContext, ph: MfPhase, d: Int, s: Int, err: DeviceBuffer[DType.int32]) raises:
    """Step s's log determinant + C-step control (MCD_WIDE: one
    block per candidate for d > MF_DMAX)."""
    var par = s % 2
    comptime if MCD_WIDE:
        if d > MF_DMAX:
            ctx.enqueue_function[mf_det_wide_kernel](
                _f(ph.cov1) if par == 1 else _f(ph.cov0), _f(ph.wa), _f(ph.det1) if par == 1 else _f(ph.det0),
                _f(ph.det0) if par == 1 else _f(ph.det1), Int32(ph.nc), Int32(d), Int32(s), Int32(ph.n_iter),
                _i(ph.active), _i(ph.needp), _i(ph.fin), _i(err), grid_dim=ph.nc, block_dim=MF_TPB,
            )
            return
    ctx.enqueue_function[mf_det_kernel](
        _f(ph.cov1) if par == 1 else _f(ph.cov0), _f(ph.wa), _f(ph.det1) if par == 1 else _f(ph.det0),
        _f(ph.det0) if par == 1 else _f(ph.det1), Int32(ph.nc), Int32(d), Int32(s), Int32(ph.n_iter), _i(ph.active),
        _i(ph.needp), _i(ph.fin), _i(err), grid_dim=_blocks(ph.nc), block_dim=MF_TPB,
    )


def _run_phase(
    ctx: DeviceContext, ph: MfPhase, dx: DeviceBuffer[DType.float32], rows: DeviceBuffer[DType.int32],
    pj: DeviceBuffer[DType.int32], pk: DeviceBuffer[DType.int32], d: Int, npair: Int, has_init: Bool,
    want_dist: Bool, err: DeviceBuffer[DType.int32], anyb: DeviceBuffer[DType.int32], mut hany: List[Int32],
) raises:
    """The C-steps of every candidate of the phase, together. Without an
    init the mask0 holds the draws' selection; with one, loc1/cov1 hold the
    candidates' (loc0, cov0) and the first selection is `mahal` under
    pinvh(cov0). Ends with every candidate stopped (fin set) and, when
    `want_dist`, its final distances under its chosen mean and its last P."""
    var nc = ph.nc
    var r = ph.r
    var hinv = Float32(1.0 / Float64(ph.h))
    ctx.enqueue_function[mf_fill_i_kernel](_i(ph.active), Int32(nc), Int32(1), grid_dim=_blocks(nc), block_dim=MF_TPB)
    ctx.enqueue_function[mf_fill_i_kernel](_i(ph.needp), Int32(nc), Int32(0), grid_dim=_blocks(nc), block_dim=MF_TPB)
    ctx.enqueue_function[mf_fill_i_kernel](_i(ph.fin), Int32(nc), Int32(0), grid_dim=_blocks(nc), block_dim=MF_TPB)
    if has_init:
        # P0 = pinvh(cov0), dist = mahal(X, loc0, P0), sel = the h smallest
        comptime if MCD_BATCH_COMPAT:
            _compat_pinvh(ctx, ph, _f(ph.cov1), d, err)
            comptime if MCD_BATCH_MMA:
                _mma_precision(ctx, ph, d)
        else:
            ctx.enqueue_function[mf_pinvh_kernel](
                _f(ph.cov1), _f(ph.wa), _f(ph.wv), _f(ph.pm), Int32(nc), Int32(d), _i(ph.active), _i(ph.needp),
                grid_dim=_blocks(nc), block_dim=MF_TPB,
            )
        comptime if MCD_BATCH_MMA:
            _mma_distance(ctx, ph, dx, rows, d, 1, False, err)
        else:
            ctx.enqueue_function[mf_dist_kernel](
                _f(dx), _i(rows), Int32(ph.ident), Int32(ph.per), _f(ph.loc0), _f(ph.loc1), Int32(1), _i(ph.fin), Int32(0),
                _f(ph.pm), _f(ph.dist), Int32(nc), Int32(r), Int32(d), _i(ph.active), _i(err),
                grid_dim=_blocks(nc * r), block_dim=MF_TPB,
            )
    ctx.enqueue_function[mf_select_kernel](
        _f(ph.dist), _i(ph.mask0), Int32(r), Int32(ph.h), _i(ph.active), grid_dim=nc, block_dim=MF_TPB,
    )
    for s in range(ph.n_iter + 1):
        var par = s % 2
        comptime if MCD_BATCH_COMPAT:
            ctx.enqueue_function[mc_compact_kernel](
                _i(ph.mask1) if par == 1 else _i(ph.mask0), _i(ph.selected), Int32(r), _i(ph.active),
                grid_dim=nc, block_dim=MF_TPB,
            )
            ctx.enqueue_function[mc_moment_kernel](
                _f(dx), _i(rows), Int32(ph.ident), Int32(ph.per), _i(ph.selected),
                _f(ph.loc1) if par == 1 else _f(ph.loc0), _i(pj), _i(pk), _f(ph.part),
                Int32(nc), Int32(r), Int32(ph.h), Int32(ph.tiles), Int32(d), Int32(npair), Int32(0), _i(ph.active),
                grid_dim=_blocks(nc * ph.tiles * d), block_dim=MF_TPB,
            )
        else:
            ctx.enqueue_function[mf_colsum_kernel](
                _f(dx), _i(rows), Int32(ph.ident), Int32(ph.per), _i(ph.mask1) if par == 1 else _i(ph.mask0), _f(ph.part),
                Int32(nc), Int32(r), Int32(ph.tiles), Int32(d), _i(ph.active),
                grid_dim=_blocks(nc * ph.tiles * d), block_dim=MF_TPB,
            )
        ctx.enqueue_function[mf_mean_kernel](
            _f(ph.part), _f(ph.loc1) if par == 1 else _f(ph.loc0), Int32(nc), Int32(ph.tiles), Int32(d), hinv,
            _i(ph.active), grid_dim=_blocks(nc * d), block_dim=MF_TPB,
        )
        comptime if MCD_BATCH_MMA:
            _mma_covariance(ctx, ph, dx, rows, d, par, hinv)
        else:
            comptime if MCD_BATCH_COMPAT:
                ctx.enqueue_function[mc_moment_kernel](
                    _f(dx), _i(rows), Int32(ph.ident), Int32(ph.per), _i(ph.selected),
                    _f(ph.loc1) if par == 1 else _f(ph.loc0), _i(pj), _i(pk), _f(ph.part),
                    Int32(nc), Int32(r), Int32(ph.h), Int32(ph.tiles), Int32(d), Int32(npair), Int32(1), _i(ph.active),
                    grid_dim=_blocks(nc * ph.tiles * npair), block_dim=MF_TPB,
                )
            else:
                ctx.enqueue_function[mf_cov_part_kernel](
                    _f(dx), _i(rows), Int32(ph.ident), Int32(ph.per), _i(ph.mask1) if par == 1 else _i(ph.mask0),
                    _f(ph.loc1) if par == 1 else _f(ph.loc0), _i(pj), _i(pk), _f(ph.part), Int32(nc), Int32(r), Int32(ph.tiles),
                    Int32(d), Int32(npair), _i(ph.active), grid_dim=_blocks(nc * ph.tiles * npair), block_dim=MF_TPB,
                )
            ctx.enqueue_function[mf_cov_kernel](
                _f(ph.part), _f(ph.cov1) if par == 1 else _f(ph.cov0), _i(pj), _i(pk), Int32(nc), Int32(ph.tiles), Int32(d),
                Int32(npair), hinv, _i(ph.active), grid_dim=_blocks(nc * npair), block_dim=MF_TPB,
            )
        _det(ctx, ph, d, s, err)
        ctx.enqueue_function[mf_fill_i_kernel](_i(anyb), Int32(1), Int32(0), grid_dim=1, block_dim=MF_TPB)
        ctx.enqueue_function[mf_any_kernel](_i(ph.active), Int32(nc), _i(anyb), grid_dim=_blocks(nc), block_dim=MF_TPB)
        ctx.enqueue_copy(dst_ptr=hany.unsafe_ptr(), src_buf=anyb)
        ctx.synchronize()
        # the -inf start's pinvh (needp) runs even when no candidate is active
        comptime if MCD_BATCH_COMPAT:
            _compat_pinvh(ctx, ph, _f(ph.cov1) if par == 1 else _f(ph.cov0), d, err)
            comptime if MCD_BATCH_MMA:
                _mma_precision(ctx, ph, d)
        else:
            ctx.enqueue_function[mf_pinvh_kernel](
                _f(ph.cov1) if par == 1 else _f(ph.cov0), _f(ph.wa), _f(ph.wv), _f(ph.pm), Int32(nc), Int32(d),
                _i(ph.active), _i(ph.needp), grid_dim=_blocks(nc), block_dim=MF_TPB,
            )
        if Int(hany[0]) == 0:
            break
        comptime if MCD_BATCH_MMA:
            _mma_distance(ctx, ph, dx, rows, d, par, False, err)
        else:
            ctx.enqueue_function[mf_dist_kernel](
                _f(dx), _i(rows), Int32(ph.ident), Int32(ph.per), _f(ph.loc0), _f(ph.loc1), Int32(par), _i(ph.fin), Int32(0),
                _f(ph.pm), _f(ph.dist), Int32(nc), Int32(r), Int32(d), _i(ph.active), _i(err),
                grid_dim=_blocks(nc * r), block_dim=MF_TPB,
            )
        ctx.enqueue_function[mf_select_kernel](
            _f(ph.dist), _i(ph.mask0) if par == 1 else _i(ph.mask1), Int32(r), Int32(ph.h), _i(ph.active),
            grid_dim=nc, block_dim=MF_TPB,
        )
    if want_dist:
        comptime if MCD_BATCH_MMA:
            _mma_distance(ctx, ph, dx, rows, d, 0, True, err)
        else:
            ctx.enqueue_function[mf_dist_kernel](
                _f(dx), _i(rows), Int32(ph.ident), Int32(ph.per), _f(ph.loc0), _f(ph.loc1), Int32(0), _i(ph.fin), Int32(1),
                _f(ph.pm), _f(ph.dist), Int32(nc), Int32(r), Int32(d), _i(ph.active), _i(err),
                grid_dim=_blocks(nc * r), block_dim=MF_TPB,
            )


def _rank(ctx: DeviceContext, ph: MfPhase, keep: Int, order: DeviceBuffer[DType.int32]) raises:
    ctx.enqueue_function[mf_rank_kernel](
        _f(ph.det0), _f(ph.det1), _i(ph.fin), Int32(ph.nc), Int32(ph.per), Int32(keep), _i(order),
        grid_dim=_blocks(ph.nc), block_dim=MF_TPB,
    )


def _handoff(ctx: DeviceContext, src: MfPhase, order: DeviceBuffer[DType.int32], dst: MfPhase, d: Int) raises:
    ctx.enqueue_function[mf_handoff_kernel](
        _i(order), _i(src.fin), _f(src.loc0), _f(src.loc1), _f(src.cov0), _f(src.cov1), _f(dst.loc1), _f(dst.cov1),
        Int32(dst.nc), Int32(d), grid_dim=_blocks(dst.nc * (d + d * d)), block_dim=MF_TPB,
    )


def _finish(
    ctx: DeviceContext, ph: MfPhase, order: DeviceBuffer[DType.int32], rows: DeviceBuffer[DType.int32], n: Int,
    d: Int, loc_out: F32Ptr, cov_out: F32Ptr, sup_out: I32Ptr, dist_out: F32Ptr, err: DeviceBuffer[DType.int32],
) raises:
    """The best candidate (order[0]) home: its mean, covariance, support
    mask and distances scattered to its rows' ids; zeros elsewhere."""
    var dsup = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var ddist = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dloc = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dcov = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    ctx.enqueue_function[mf_fill_i_kernel](_i(dsup), Int32(n), Int32(0), grid_dim=_blocks(n), block_dim=MF_TPB)
    ctx.enqueue_function[mf_fill_f_kernel](_f(ddist), Int32(n), Float32(0), grid_dim=_blocks(n), block_dim=MF_TPB)
    ctx.enqueue_function[mf_scatter_kernel](
        _i(rows), Int32(ph.ident), _i(ph.mask0), _i(ph.mask1), _f(ph.dist), _i(ph.fin), _i(order), Int32(ph.r),
        _i(dsup), _f(ddist), grid_dim=_blocks(ph.r), block_dim=MF_TPB,
    )
    # the chosen mean and covariance through the handoff into a one-candidate phase
    ctx.enqueue_function[mf_handoff_kernel](
        _i(order), _i(ph.fin), _f(ph.loc0), _f(ph.loc1), _f(ph.cov0), _f(ph.cov1), _f(dloc), _f(dcov), Int32(1), Int32(d),
        grid_dim=_blocks(d + d * d), block_dim=MF_TPB,
    )
    var herr = List[Int32](length=MF_ERR, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=herr.unsafe_ptr(), src_buf=err)
    _down(ctx, dloc, loc_out, d)
    _down(ctx, dcov, cov_out, d * d)
    _down_i(ctx, dsup, sup_out, n)
    _down(ctx, ddist, dist_out, n)
    ctx.synchronize()
    _ = dsup^
    _ = ddist^
    _ = dloc^
    _ = dcov^
    if Int(herr[0]) != 0:
        raise Error("x_decomp MinCovDet: a NaN distance or draw has no order (refused)")
    if Int(herr[1]) != 0:
        raise Error("x_decomp MinCovDet: the first C-step's log determinant is not finite")
    if Int(herr[2]) != 0:
        raise Error("x_decomp MinCovDet: batched round-robin eigensolve failed convergence budget")
    if Int(herr[3]) != 0:
        raise Error("x_decomp MinCovDet: batched round-robin eigensolve failed Frobenius gate")
    _ = herr^


def mf_pair_table_kernel(pj: I32Ptr, pk: I32Ptr, d_in: Int32):
    """One thread a (j, k) cell of d x d: for k >= j, pair index
    j * d - j * (j - 1) / 2 + (k - j) gets (j, k), the row-major upper
    triangle order the host loop built. Integer only."""
    var d = Int(d_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < d * d:
        var j = t // d
        var k = t % d
        if k >= j:
            var p = j * d - j * (j - 1) // 2 + (k - j)
            pj.unsafe_store(p, Int32(j))
            pk.unsafe_store(p, Int32(k))


def fast_mcd_fast(
    X: Mat, p: List[Int], loc_out: F32Ptr, cov_out: F32Ptr, sup_out: I32Ptr, dist_out: F32Ptr,
) raises -> Bool:
    """`fast_mcd` (x_decomp/mcd.mojo) with every candidate's C-steps on the
    device; False when the input is outside this module's range (then the
    caller runs main's search). p as `fast_mcd_dev`'s plan."""
    var n = p[0]
    var d = p[1]
    var h = p[2]
    var seed = p[3]
    var dmax = MF_DMAX
    comptime if MCD_WIDE:
        dmax = MF_WIDE_DMAX
    if d > dmax or d < 2 or n < 1:
        return False
    var ctx = xd_ctx()
    var npair = d * (d + 1) // 2
    # the (j, k <= j..d) pair table is written on the device (lane cpu3-core)
    var pj = ctx.enqueue_create_buffer[DType.int32](npair)
    var pk = ctx.enqueue_create_buffer[DType.int32](npair)
    ctx.enqueue_function[mf_pair_table_kernel](
        _i(pj), _i(pk), Int32(d), grid_dim=_blocks(d * d), block_dim=MF_TPB,
    )
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n * d, 1))
    ctx.enqueue_copy(dst_buf=dx, src_ptr=X.p())
    var err = ctx.enqueue_create_buffer[DType.int32](MF_ERR)
    ctx.enqueue_function[mf_fill_i_kernel](_i(err), Int32(MF_ERR), Int32(0), grid_dim=1, block_dim=MF_TPB)
    var anyb = ctx.enqueue_create_buffer[DType.int32](1)
    var hany = List[Int32](length=1, fill=Int32(0))
    var ident = ctx.enqueue_create_buffer[DType.int32](1)
    if n > 500:
        var n_sub = p[4]
        var n_ss = p[5]
        var h_sub = p[6]
        var n_trials = p[7]
        var n_m = p[8]
        var h_m = p[9]
        var n_best_m = p[10]
        var npad = 1
        while npad < n:
            npad *= 2
        var keys = ctx.enqueue_create_buffer[DType.uint64](npad)
        # draw 1: the subsets' shuffle
        var shuf = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
        _perm(ctx, keys, npad, n, seed, 1000 + 1, shuf, n)
        # phase A: n_sub subsets x n_trials candidates, draws 2 .. 1 + nc_a
        var nc_a = n_sub * n_trials
        var keep_a = min(10, n_trials)
        var pa = MfPhase(ctx, nc_a, n_trials, n_ss, 0, h_sub, 2, d, npair)
        ctx.enqueue_function[mf_draw_kernel](
            _f(pa.dist), Int32(nc_a), Int32(n_ss), UInt32(seed & 0xFFFFFFFF), UInt32((1000 + 2) & 0xFFFFFFFF), _i(err),
            grid_dim=_blocks(nc_a * n_ss), block_dim=MF_TPB,
        )
        _run_phase(ctx, pa, dx, shuf, pj, pk, d, npair, False, False, err, anyb, hany)
        var order_a = ctx.enqueue_create_buffer[DType.int32](max(n_sub * keep_a, 1))
        _rank(ctx, pa, keep_a, order_a)
        # draw 2 + nc_a: the merged selection
        var sel = ctx.enqueue_create_buffer[DType.int32](max(n_m, 1))
        _perm(ctx, keys, npad, n, seed, 1000 + 2 + nc_a, sel, n_m)
        # phase B: the pool on the n_m selected rows, up to 30 steps
        var nc_b = n_sub * keep_a
        var pb = MfPhase(ctx, nc_b, nc_b, n_m, 0, h_m, 30, d, npair)
        _handoff(ctx, pa, order_a, pb, d)
        _run_phase(ctx, pb, dx, sel, pj, pk, d, npair, True, n < 1500, err, anyb, hany)
        var order_b = ctx.enqueue_create_buffer[DType.int32](max(n_best_m, 1))
        _rank(ctx, pb, n_best_m, order_b)
        if n < 1500:
            _finish(ctx, pb, order_b, sel, n, d, loc_out, cov_out, sup_out, dist_out, err)
            _ = pa^
            _ = pb^
            _ = keys^
            _ = shuf^
            _ = sel^
            _ = order_a^
            _ = order_b^
        else:
            # phase C: the n_best_m on every row, up to 30 steps, distances kept
            var pc = MfPhase(ctx, n_best_m, n_best_m, n, 1, h, 30, d, npair)
            _handoff(ctx, pb, order_b, pc, d)
            _run_phase(ctx, pc, dx, ident, pj, pk, d, npair, True, True, err, anyb, hany)
            var order_c = ctx.enqueue_create_buffer[DType.int32](1)
            _rank(ctx, pc, 1, order_c)
            _finish(ctx, pc, order_c, ident, n, d, loc_out, cov_out, sup_out, dist_out, err)
            _ = pa^
            _ = pb^
            _ = pc^
            _ = keys^
            _ = shuf^
            _ = sel^
            _ = order_a^
            _ = order_b^
            _ = order_c^
    else:
        # 30 random starts on every row (draws 1 .. 30), two steps, keep 10;
        # then the 10 on every row, up to 30 steps
        var pa = MfPhase(ctx, 30, 30, n, 1, h, 2, d, npair)
        ctx.enqueue_function[mf_draw_kernel](
            _f(pa.dist), Int32(30), Int32(n), UInt32(seed & 0xFFFFFFFF), UInt32((1000 + 1) & 0xFFFFFFFF), _i(err),
            grid_dim=_blocks(30 * n), block_dim=MF_TPB,
        )
        _run_phase(ctx, pa, dx, ident, pj, pk, d, npair, False, False, err, anyb, hany)
        var order_a = ctx.enqueue_create_buffer[DType.int32](10)
        _rank(ctx, pa, 10, order_a)
        var pc = MfPhase(ctx, 10, 10, n, 1, h, 30, d, npair)
        _handoff(ctx, pa, order_a, pc, d)
        _run_phase(ctx, pc, dx, ident, pj, pk, d, npair, True, True, err, anyb, hany)
        var order_c = ctx.enqueue_create_buffer[DType.int32](1)
        _rank(ctx, pc, 1, order_c)
        _finish(ctx, pc, order_c, ident, n, d, loc_out, cov_out, sup_out, dist_out, err)
        _ = pa^
        _ = pc^
        _ = order_a^
        _ = order_c^
    _ = hany^
    _ = pj^
    _ = pk^
    _ = dx^
    _ = err^
    _ = anyb^
    _ = ident^
    return True
