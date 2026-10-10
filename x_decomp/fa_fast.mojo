# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple (lane/apple-fast-fa@3efbce2af, 2026-10-03; recovered onto
main 2026-10-04 by lane/apple-fast-rec-fa-robust, every define default OFF and
READY-AB): FactorAnalysis's fit and transform on device-resident operands.
The recovery keeps main's two-pass mean and its cancellation-free psi update
(lane/apple-fast-quality-glmfa) on these routes; LIVEBUF turns
ITER_DEVICE on. NOT an IDENTICAL path: the binding
registers these entries only under `FA_FAST_APPLE` (a FAST build for the
Apple GPU), each route only when its `-D MOJOLEARN_FA_<NAME>` define is set
(read with `is_defined`, never an env read), and python/mojolearn/
_expansion_decomp.py `FactorAnalysis` takes a route only when the binding
reports its define (`x_decomp_fa_defines`). IDENTICAL and every other vendor
compile main's code unchanged: no IDENTICAL launch reaches a kernel here.

Profile of main's fit and the mechanisms: docs/apple-fast/notes/fa.md. The
defines (docs/apple-fast/ab/fa.md):

- MOJOLEARN_FA_GRAM_ONCE: `fa_gram_tile_kernel` + `fa_gram_fold_kernel`, the
  centered Gram G = (X - mean)^T (X - mean) (d x d) and var = diag(G) / n in
  ONE tiled pass over the resident X (64 x 64 output tiles of 4 x 4 per
  thread over 16-row slabs in threadgroup memory, a partial per 8192 rows,
  the fold over the partials). No Xc buffer, no sq buffer, no download, no
  host copy, no QR of the n x d data. EM's per-iteration SVD of R D / sqrt(n)
  becomes the eigh of D G D / n (same spectrum: R^T R = G), the route main
  already takes when n < d. The Python loop stays Python.
- MOJOLEARN_FA_ITER_DEVICE (the FAST + Apple default since 2026-10-04,
  rab7-faiterfix; rollback MOJOLEARN_FA_ITER_DEVICE_OFF): `fa_em_py`, the
  whole EM loop as one call on the
  resident G: per iteration `fa_scale_kernel` (D G D / n and sqrt(psi) +
  1e-12), the round-robin eigh (main's kernels and sweeps, no sign-flip and
  ordering launches: `fa_finish_kernel` orders and signs the nc columns it
  uses), `fa_finish_kernel` (W, the psi update, the 2 d log terms) and ONE
  readback of 2 d + 4 floats; the log-likelihood is summed in float64 on the
  host in main's order and the tol test is main's. Implies GRAM_ONCE's pass.
- MOJOLEARN_FA_LIVEBUF: ITER_DEVICE's scratch as one arena buffer (one live
  Metal buffer instead of ~12) and W + psi read back in one copy.
- MOJOLEARN_FA_TRANSFORM_FUSED: `fa_transform_kernel`, transform as one
  launch over rows with P = (W / psi)^T cov_z (d x nc) and the mean in
  threadgroup memory, one row per thread, the n x nc result read back once.
- (QUALITY-FIX, lane/apple-fast-fa-quality 2026-10-04) every ITER_DEVICE
  arm forms G in double-float (`fa_gram_tile_df_kernel`), factors it once
  (`fa_chol_df_kernel`, R^T R = G) and takes main's one-sided SVD of
  R D / sqrt(n) per iteration (`fa_rs_svd_block_kernel` / `rs_round_kernel`)
  instead of the float32 eigh of D G D / n, which squared the condition
  number (Istella held-out mean_log_likelihood 92.898 against main's
  99.487). MOJOLEARN_FA_GRAM_QOLD keeps the float32 route; GRAM_ONCE alone
  (the Python loop) keeps it too. See FA_GRAM_DF below.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from std.memory import stack_allocation
from std.python import PythonObject
from std.sys import llvm_intrinsic
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz
from core.device_zero import enqueue_fill
from decomposition.checks.jacobi_eigh_device import JACOBI_TOL
from checks.numerics import identical_mul_add, identical_sqrt
from x_decomp.cells import X_DECOMP_SVD_SWEEPS, X_DECOMP_SVD_TOL, F32Ptr, add, div0, log_floor, mul, sqrt0
from x_decomp.device import PJ_SYNC_ROUNDS, TPB, _blocks, _pj_blocks, _pj_off_blocks, xd_ctx
from x_decomp.jacobi2 import dev_barrier
from x_decomp.eigh_scale import enqueue_es_scale_strided, enqueue_es_unscale_diag_ptr, enqueue_es_unscale_ptr, es_strided_words
from x_decomp.jacobi_par import (
    PJ_TPB,
    eigh_par_cs_kernel,
    eigh_par_off_fold_kernel,
    eigh_par_off_part_kernel,
    eigh_par_update_kernel,
    pj_identity_kernel,
)
from x_decomp.resident import _id, _n, _ptr, pool_alloc, pool_free
from x_decomp.rr import (
    RR_EIGH_SWEEPS,
    RR_OFF_TPB,
    rr_add,
    rr_block,
    rr_converged,
    rr_cs,
    rr_fro_kept,
    rr_row_off,
    rr_sub,
    rr_vrow,
)
from x_decomp.rr_svd import RS_TPB, rs_decide, rs_pair
from x_decomp.rr_svd_device import rs_norm_kernel, rs_round_kernel

#: the guard of every route in this file: a FAST build for the Apple GPU
comptime FA_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
# TOMBSTONE: MOJOLEARN_FA_ALL (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: every FA define at once; factor-analysis istella 10300.9 -> 20530.8 ms (+99.3%, rab6-faqfix), quality noise.
# Restore: git apply experiments/removed/MOJOLEARN_FA_ALL.patch; record in docs/TOMBSTONES.md.
#: (FAST + Apple, default OFF) FactorAnalysis.fit forms the centred Gram G
#: (d x d) in ONE tiled pass over the resident X (`fa_gram_tile_kernel` +
#: `fa_gram_fold_kernel`), and the Python EM loop takes the eigh of D G D / n
#: in place of main's QR of the n x d data and SVD of R D / sqrt(n) per
#: iteration (the same spectrum: R^T R = G). Source lane/apple-fast-fa@3efbce2af.
#: Prior M3 B arms (source tree): taxi 277 ms, istella 1262 ms against board
#: 308 / 10351. Never recorded in EXPERIMENTS. Recovery fixes: main's two-pass
#: mean and its cancellation-free psi update (lane/apple-fast-quality-glmfa,
#: 2026-10-03) are kept on this route too; the old route had psi =
#: var - colsum(W^2), whose float32 floor near var * 1e-7 cost Istella's
#: held-out log-likelihood. Quality risk: G squares the condition number the
#: QR route sees; the A/B's quality check decides.
comptime FA_GRAM_ONCE = FA_FAST_APPLE and is_defined["MOJOLEARN_FA_GRAM_ONCE"]()
#: (FAST + Apple, default ON since 2026-10-04; implies GRAM_ONCE's pass) the whole EM loop as
#: ONE binding call on the resident G (`fa_em_py`): per iteration
#: `fa_scale_kernel`, main's round-robin eigh, `fa_finish_kernel` (order, sign,
#: W, psi update, the 2 d log terms) and one readback of 2 d + 4 floats; the
#: log-likelihood summed in float64 on the host in main's order. Source
#: lane/apple-fast-fa@3efbce2af. Prior M3 B arms: taxi 168.7 ms (with
#: LIVEBUF), istella about 1253 ms against board 308 / 10351: the largest
#: classical lead found, never recorded. Recovery fix: psi update in
#: `fa_finish_kernel` is main's cancellation-free form (see GRAM_ONCE).
#: LIVEBUF acts only inside this loop, so it
#: turns it on (a lone define never builds a no-op arm).
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, tag
#: rab7-faiterfix, with the double-float Gram FA_GRAM_DF of lane/apple-fast-
#: fa-quality): factor-analysis istella 10299.65 -> 5636.55 ms (-45.3%),
#: mean_log_likelihood 99.487208 -> 99.487138 (noise; the float32 Gram's
#: 92.898 loss is gone); taxi 317.85 -> 157.98 ms (-50.3%), -14.823653 ->
#: -14.823710 (noise). KEEP: the FAST + Apple default since then (FA_GRAM_DF
#: with it, since FA_GRAM_DF follows FA_ITER_DEVICE); rollback
#: -D MOJOLEARN_FA_ITER_DEVICE_OFF (the old -D name is harmless; the
#: LIVEBUF arm still turns it on).
comptime FA_ITER_DEVICE = FA_FAST_APPLE and (
    not is_defined["MOJOLEARN_FA_ITER_DEVICE_OFF"]()
    or is_defined["MOJOLEARN_FA_LIVEBUF"]()
)
# TOMBSTONE: MOJOLEARN_FA_EIG_SMALL (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: the FA EM loop's d x d eigh / SVD as ONE launch of one threadgroup (fa_rr_eigh_block_kernel, fa_rs_svd_block_kernel); in FA_ALL istella 10300.9 -> 20530.8 ms (+99.3%).
# Restore: git apply experiments/removed/MOJOLEARN_FA_EIG_SMALL.patch; record in docs/TOMBSTONES.md.
#: (FAST + Apple, default OFF; implies ITER_DEVICE) the loop's scratch as one
#: arena buffer (one live Metal buffer instead of ~14) and W + psi read back
#: in one copy. Source lane/apple-fast-fa@3efbce2af. Prior M3 B arms
#: (ITER_DEVICE + LIVEBUF): istella 1258 ms, taxi 168.7 ms. No failure.
comptime FA_LIVEBUF = FA_FAST_APPLE and is_defined["MOJOLEARN_FA_LIVEBUF"]()
# TOMBSTONE: MOJOLEARN_FA_LL_DEVICE (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: the FA EM convergence test on the device (double-float ll sum in fa_finish_kernel, a flag read every 4 iterations); only timed with EIG_SMALL (FA_ALL istella +99.3%), alone never timed.
# Restore: git apply experiments/removed/MOJOLEARN_FA_LL_DEVICE.patch; record in docs/TOMBSTONES.md.
#: (FAST + Apple, default OFF; RECORD rab7-fatransform 2026-10-04: istella
#: 10313.84 -> 10524.26 ms (+2.0%), taxi +0.3%, quality identical: stays off)
#: FactorAnalysis.transform as one launch over
#: rows (`fa_transform_kernel`): P = (W / psi)^T cov_z (d x nc) and the mean
#: in threadgroup memory, one row per thread, in place of a sub, two GEMMs
#: and their n x d / n x nc intermediates. Limits from the kernel: nc <=
#: FA_TR_MAXK register accumulators, d (nc + 1) <= FA_TR_FLOATS (a 16 KB
#: threadgroup page, checked against the column's page). Source
#: lane/apple-fast-fa@3efbce2af. Prior M3 B arms: istella 6201 ms, taxi
#: 243 ms against board 10351 / 308 (whole fit + transform lane). No failure.
comptime FA_TRANSFORM_FUSED = FA_FAST_APPLE and is_defined["MOJOLEARN_FA_TRANSFORM_FUSED"]()
#: (FAST + Apple, QUALITY-FIX, lane/apple-fast-fa-quality 2026-10-04) the
#: ITER_DEVICE loop's spectral step. Every ITER_DEVICE arm takes it unless -D MOJOLEARN_FA_GRAM_QOLD keeps the float32 route.
#: Cause: the float32 Gram + float32 eigh of D G D / n resolve B = D G D / n
#: only to eps ||B||, but the psi of a column the factors explain sits at
#: relative psi_j / var_j (1e-8 on Istella) of B_jj; FA_ALL's Istella held-out
#: mean_log_likelihood fell 99.487 (FAST main) -> 92.898 on all three ITER
#: variants. The fix keeps the one pass over X and the device loop:
#:   1. `fa_gram_tile_df_kernel` + `fa_gram_fold_df_kernel`: G accumulated in
#:      double-float float32 (fma two-product, two-sum; about 2^-48 relative
#:      to sum |x_i x_j|), hi words then lo words (2 d x d floats);
#:   2. `fa_chol_df_kernel`, ONCE per fit: the double-float semidefinite
#:      Cholesky G = R^T R (one threadgroup; a pivot under FA_CHOL_DROP G_jj
#:      drops its row: that column is in the span of the earlier ones to
#:      float32 resolution), R^T rounded to float32;
#:   3. per iteration main's spectral route on that R: the one-sided
#:      round-robin Jacobi SVD of R D / sqrt(n) (x_decomp/rr_svd.mojo cells,
#:      relative rotation test X_DECOMP_SVD_TOL), one launch a round
#:      (`rs_round_kernel`, main's grid kernel). The SVD sees sqrt(B):
#:      the residual it must resolve is sqrt(psi / var) ~ 1e-4 >> eps.
#: float32 numpy model (~/mojolearn-evidence/fa-quality/fa_model.py, results
#: beside it): see docs/apple-fast/EXPERIMENTS.md `FA_GRAM_DF`.
comptime FA_GRAM_QOLD = is_defined["MOJOLEARN_FA_GRAM_QOLD"]()
comptime FA_GRAM_DF = FA_ITER_DEVICE and not FA_GRAM_QOLD
#: a Cholesky pivot under this fraction of G_jj (2^-44, 16x the double-float
#: word's 2^-48) drops its row: the residual left is 2^-22 of the column's
#: norm, the float32 Householder QR's own backward error
comptime FA_CHOL_DROP = Float32(5.684341886080802e-14)

#: features the one-threadgroup kernels accept: one row of the d x d per
#: lane of the FA_TPB-lane threadgroup (a kernel limit, not a dataset window)
comptime FA_MAX_D = 256
#: threads of the one-threadgroup kernels (= RR_OFF_TPB: the convergence
#: test's fold keeps main's lane partition and tree)
comptime FA_TPB = 256
#: sklearn's SMALL and the kit's log floor (FLT_MIN), as FactorAnalysis.fit
comptime FA_SMALL = Float32(1.0e-12)
comptime FA_TINY = Float32(1.1754943508222875e-38)
#: log(2 pi), `_expansion_decomp._LOG_2PI`
comptime FA_LOG_2PI: Float64 = 1.8378770664093453

# ------------------------------------------------------------ the Gram once
#: output tile side, slab rows, threads (16 x 16, each a 4 x 4 micro-tile),
#: rows per partial (grid y)
comptime FG_TILE = 64
comptime FG_SLAB = 16
comptime FG_TPB = 256
comptime FG_ROWS = 8192
comptime FG_SMEM_BYTES = 2 * FG_SLAB * FG_TILE * 4
comptime FG_FITS = lib_smem_page_fits_for[TARGET_COLUMN, FG_SMEM_BYTES]()


def fa_gram_tiles(d: Int) -> Int:
    """Tiles along one side of the d x d output."""
    return (d + FG_TILE - 1) // FG_TILE


def fa_gram_parts(nrows: Int) -> Int:
    """Partials (grid y) over nrows rows."""
    return (nrows + FG_ROWS - 1) // FG_ROWS if nrows > 0 else 1


def fa_gram_tile_kernel(x: F32Ptr, mean: F32Ptr, part: F32Ptr, n_in: Int32, d_in: Int32, nt_in: Int32):
    """Block (pair, z): pair = block_idx.x names an upper-triangle tile
    (ti <= tj) of the nt x nt tile grid, z = block_idx.y the partial over rows
    [z FG_ROWS, min(n, (z + 1) FG_ROWS)). Slabs of FG_SLAB rows of the tile's
    two column ranges, centered at the mean as they are loaded (columns past
    d and rows past the range load 0), go to threadgroup memory; thread
    (tr, tc) accumulates its 4 x 4 cells over the slab. part[z d d + i d + j]
    for the tile's cells with i < d, j < d (the fold reads i <= j)."""
    var n = Int(n_in)
    var d = Int(d_in)
    var nt = Int(nt_in)
    var pair = Int(block_idx.x)
    var z = Int(block_idx.y)
    var ti = 0
    var rem = pair
    while rem >= nt - ti:
        rem -= nt - ti
        ti += 1
    var tj = ti + rem
    var i0 = ti * FG_TILE
    var j0 = tj * FG_TILE
    var r0 = z * FG_ROWS
    var r1 = min(n, r0 + FG_ROWS)
    var tid = Int(thread_idx.x)
    var tr = tid // 16
    var tc = tid - tr * 16
    var sa = stack_allocation[FG_SLAB * FG_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sb = stack_allocation[FG_SLAB * FG_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[Float32, 16](fill=Float32(0.0))
    var r = r0
    while r < r1:
        barrier()
        for idx in range(tid, FG_SLAB * FG_TILE, FG_TPB):
            var sr = idx // FG_TILE
            var sc = idx - sr * FG_TILE
            var row = r + sr
            var va = Float32(0.0)
            var vb = Float32(0.0)
            if row < r1:
                var ca = i0 + sc
                if ca < d:
                    va = x.unsafe_load(row * d + ca) - mean.unsafe_load(ca)
                var cb = j0 + sc
                if cb < d:
                    vb = x.unsafe_load(row * d + cb) - mean.unsafe_load(cb)
            sa[idx] = va
            sb[idx] = vb
        barrier()
        for sr in range(FG_SLAB):
            var a0 = sa[sr * FG_TILE + tr * 4]
            var a1 = sa[sr * FG_TILE + tr * 4 + 1]
            var a2 = sa[sr * FG_TILE + tr * 4 + 2]
            var a3 = sa[sr * FG_TILE + tr * 4 + 3]
            var b0 = sb[sr * FG_TILE + tc * 4]
            var b1 = sb[sr * FG_TILE + tc * 4 + 1]
            var b2 = sb[sr * FG_TILE + tc * 4 + 2]
            var b3 = sb[sr * FG_TILE + tc * 4 + 3]
            acc[0] += a0 * b0
            acc[1] += a0 * b1
            acc[2] += a0 * b2
            acc[3] += a0 * b3
            acc[4] += a1 * b0
            acc[5] += a1 * b1
            acc[6] += a1 * b2
            acc[7] += a1 * b3
            acc[8] += a2 * b0
            acc[9] += a2 * b1
            acc[10] += a2 * b2
            acc[11] += a2 * b3
            acc[12] += a3 * b0
            acc[13] += a3 * b1
            acc[14] += a3 * b2
            acc[15] += a3 * b3
        r += FG_SLAB
    var dd = d * d
    for a in range(4):
        var i = i0 + tr * 4 + a
        if i < d:
            for b in range(4):
                var j = j0 + tc * 4 + b
                if j < d:
                    part.unsafe_store(z * dd + i * d + j, acc[a * 4 + b])


def fa_gram_fold_kernel(part: F32Ptr, g: F32Ptr, var_out: F32Ptr, d_in: Int32, nz_in: Int32, inv_n: Float32):
    """Cell (i, j), i <= j: the partials z ascending into g[i, j] and
    g[j, i]; the diagonal scaled by 1 / n into var_out."""
    var d = Int(d_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < d * d:
        var i = t // d
        var j = t - i * d
        if i <= j:
            var dd = d * d
            var acc = Float32(0.0)
            for z in range(Int(nz_in)):
                acc += part.unsafe_load(z * dd + t)
            g.unsafe_store(t, acc)
            g.unsafe_store(j * d + i, acc)
            if i == j:
                var_out.unsafe_store(i, acc * inv_n)


# ------------------------------------------------ double-float (FA_GRAM_DF)
# A value is hi + lo. Every add is `_rn` (fma(a, 1, b): one rounding, and no
# fadd for the compiler to reassociate or contract into), every exact
# product error an explicit fma: the error-free transformations survive any
# FAST flag. Metal has no float64.
comptime DF = SIMD[DType.float32, 2]


@always_inline
def _rn(a: Float32, b: Float32) -> Float32:
    return llvm_intrinsic["llvm.fma.f32", Float32, has_side_effect=False](a, Float32(1.0), b)


@always_inline
def _fma(a: Float32, b: Float32, c: Float32) -> Float32:
    return llvm_intrinsic["llvm.fma.f32", Float32, has_side_effect=False](a, b, c)


@always_inline
def _df_ts(a: Float32, b: Float32) -> DF:
    """Knuth's two-sum: (a + b rounded, its error)."""
    var s = _rn(a, b)
    var v = _rn(s, -a)
    return DF(s, _rn(_rn(a, -_rn(s, -v)), _rn(b, -v)))


@always_inline
def _df_fts(a: Float32, b: Float32) -> DF:
    """Dekker's fast two-sum, |a| >= |b| or a == 0."""
    var s = _rn(a, b)
    return DF(s, _rn(b, -_rn(s, -a)))


@always_inline
def _df_prod(a: Float32, b: Float32) -> DF:
    """a b exactly: (a b rounded, fma(a, b, -p))."""
    var p = a * b
    return DF(p, _fma(a, b, -p))


@always_inline
def _df_add(x: DF, y: DF) -> DF:
    """x + y, error about 2^-48 (|x| + |y|) (the sloppy double-float add:
    relative to the operands, what a sum of products needs)."""
    var st = _df_ts(x[0], y[0])
    return _df_fts(st[0], _rn(st[1], _rn(x[1], y[1])))


@always_inline
def _df_mul(x: DF, y: DF) -> DF:
    var p = _df_prod(x[0], y[0])
    var e = _fma(x[0], y[1], p[1])
    e = _fma(x[1], y[0], e)
    return _df_fts(p[0], e)


@always_inline
def _df_mul_f(x: DF, b: Float32) -> DF:
    var p = _df_prod(x[0], b)
    return _df_fts(p[0], _fma(x[1], b, p[1]))


@always_inline
def _df_div(x: DF, y: DF) -> DF:
    """x / y: the float32 quotient, then one correction from the exact
    residual x - y q1."""
    var q1 = x[0] / y[0]
    var r = _df_add(x, -_df_mul_f(y, q1))
    return _df_fts(q1, r[0] / y[0])


@always_inline
def _df_sqrt(x: DF) -> DF:
    """x > 0: the float32 root, then one Newton correction from the exact
    residual x - s^2."""
    var s = sqrt0(x[0])
    var r = _df_add(x, -_df_prod(s, s))
    return _df_fts(s, r[0] / _rn(s, s))


def fa_gram_tile_df_kernel(x: F32Ptr, mean: F32Ptr, part: F32Ptr, n_in: Int32, d_in: Int32, nt_in: Int32):
    """`fa_gram_tile_kernel` (same tiles, slabs and partials) with every cell
    accumulated in double-float: each product exact (`_df_prod`), added by
    `_df_add`. part[z 2 d d + i d + j] the hi words, part[z 2 d d + d d +
    i d + j] the lo words."""
    var n = Int(n_in)
    var d = Int(d_in)
    var nt = Int(nt_in)
    var pair = Int(block_idx.x)
    var z = Int(block_idx.y)
    var ti = 0
    var rem = pair
    while rem >= nt - ti:
        rem -= nt - ti
        ti += 1
    var tj = ti + rem
    var i0 = ti * FG_TILE
    var j0 = tj * FG_TILE
    var r0 = z * FG_ROWS
    var r1 = min(n, r0 + FG_ROWS)
    var tid = Int(thread_idx.x)
    var tr = tid // 16
    var tc = tid - tr * 16
    var sa = stack_allocation[FG_SLAB * FG_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sb = stack_allocation[FG_SLAB * FG_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var hi = InlineArray[Float32, 16](fill=Float32(0.0))
    var lo = InlineArray[Float32, 16](fill=Float32(0.0))
    var r = r0
    while r < r1:
        barrier()
        for idx in range(tid, FG_SLAB * FG_TILE, FG_TPB):
            var sr = idx // FG_TILE
            var sc = idx - sr * FG_TILE
            var row = r + sr
            var va = Float32(0.0)
            var vb = Float32(0.0)
            if row < r1:
                var ca = i0 + sc
                if ca < d:
                    va = x.unsafe_load(row * d + ca) - mean.unsafe_load(ca)
                var cb = j0 + sc
                if cb < d:
                    vb = x.unsafe_load(row * d + cb) - mean.unsafe_load(cb)
            sa[idx] = va
            sb[idx] = vb
        barrier()
        for sr in range(FG_SLAB):
            var av = SIMD[DType.float32, 4](
                sa[sr * FG_TILE + tr * 4], sa[sr * FG_TILE + tr * 4 + 1],
                sa[sr * FG_TILE + tr * 4 + 2], sa[sr * FG_TILE + tr * 4 + 3],
            )
            var bv = SIMD[DType.float32, 4](
                sb[sr * FG_TILE + tc * 4], sb[sr * FG_TILE + tc * 4 + 1],
                sb[sr * FG_TILE + tc * 4 + 2], sb[sr * FG_TILE + tc * 4 + 3],
            )
            comptime for a in range(4):
                comptime for b in range(4):
                    var s = _df_add(DF(hi[a * 4 + b], lo[a * 4 + b]), _df_prod(av[a], bv[b]))
                    hi[a * 4 + b] = s[0]
                    lo[a * 4 + b] = s[1]
        r += FG_SLAB
    var dd = d * d
    for a in range(4):
        var i = i0 + tr * 4 + a
        if i < d:
            for b in range(4):
                var j = j0 + tc * 4 + b
                if j < d:
                    part.unsafe_store(z * 2 * dd + i * d + j, hi[a * 4 + b])
                    part.unsafe_store(z * 2 * dd + dd + i * d + j, lo[a * 4 + b])


def fa_gram_fold_df_kernel(part: F32Ptr, g: F32Ptr, var_out: F32Ptr, d_in: Int32, nz_in: Int32, inv_n: Float32):
    """Cell (i, j), i <= j: the double-float partials z ascending into
    g[i, j] and g[j, i] (hi words g[0, d d), lo words g[d d, 2 d d)); the
    diagonal (hi + lo) / n into var_out."""
    var d = Int(d_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < d * d:
        var i = t // d
        var j = t - i * d
        if i <= j:
            var dd = d * d
            var acc = DF(0.0, 0.0)
            for z in range(Int(nz_in)):
                acc = _df_add(acc, DF(part.unsafe_load(z * 2 * dd + t), part.unsafe_load(z * 2 * dd + dd + t)))
            g.unsafe_store(t, acc[0])
            g.unsafe_store(dd + t, acc[1])
            g.unsafe_store(j * d + i, acc[0])
            g.unsafe_store(dd + j * d + i, acc[1])
            if i == j:
                var_out.unsafe_store(i, _rn(acc[0], acc[1]) * inv_n)


def fa_chol_df_kernel(g: F32Ptr, rt: F32Ptr, d_in: Int32):
    """ONE threadgroup (FA_TPB lanes, d <= FA_MAX_D), once per fit: the
    double-float Cholesky G = R^T R (R upper) of the double-float G (hi words
    g[0, d d), lo words g[d d, 2 d d)), in place on G's upper triangle, column
    j ascending: lane i >= j forms t_i = G[j, i] - sum_{k < j} R[k, j] R[k, i]
    (k ascending), lane j the pivot; a pivot not above FA_CHOL_DROP G[j, j]
    (or G[j, j] = 0, a constant column) sets row j of R to zero (the column is
    in the span of the earlier ones to float32 resolution), else R[j, j] =
    sqrt(t_j), R[j, i] = t_i / R[j, j]. Then rt = R^T rounded to float32 (row
    p = column p of R, zero below its diagonal entry)."""
    var d = Int(d_in)
    var dd = d * d
    var tid = Int(thread_idx.x)
    var piv = stack_allocation[4, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for j in range(d):
        var t = DF(0.0, 0.0)
        if tid >= j and tid < d:
            t = DF(g.unsafe_load(j * d + tid), g.unsafe_load(dd + j * d + tid))
            for k in range(j):
                var rkj = DF(g.unsafe_load(k * d + j), g.unsafe_load(dd + k * d + j))
                var rki = DF(g.unsafe_load(k * d + tid), g.unsafe_load(dd + k * d + tid))
                t = _df_add(t, -_df_mul(rkj, rki))
            if tid == j:
                var gjj = g.unsafe_load(j * d + j)
                if gjj > Float32(0.0) and t[0] > FA_CHOL_DROP * gjj:
                    var rj = _df_sqrt(t)
                    piv[0] = rj[0]
                    piv[1] = rj[1]
                    piv[2] = Float32(1.0)
                else:
                    piv[0] = Float32(0.0)
                    piv[1] = Float32(0.0)
                    piv[2] = Float32(0.0)
        dev_barrier()
        if tid >= j and tid < d:
            var o = DF(0.0, 0.0)
            if piv[2] != Float32(0.0):
                if tid == j:
                    o = DF(piv[0], piv[1])
                else:
                    o = _df_div(t, DF(piv[0], piv[1]))
            g.unsafe_store(j * d + tid, o[0])
            g.unsafe_store(dd + j * d + tid, o[1])
        dev_barrier()
    for u in range(tid, dd, FA_TPB):
        var p = u // d
        var i = u - p * d
        var val = Float32(0.0)
        if i <= p:
            val = _rn(g.unsafe_load(i * d + p), g.unsafe_load(dd + i * d + p))
        rt.unsafe_store(u, val)


def fa_gram_py(x: PythonObject, mean: PythonObject, g: PythonObject, var_: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, d]: G (d x d, device id g) and var (d, device id var_) of the
    resident X (n x d) centered at mean (d); enqueued, no sync. Partials from
    the pool, freed at once (the context runs in order). FA_GRAM_DF: G in
    double-float, g holds 2 d x d floats (hi words, then lo words)."""
    comptime if not FG_FITS:
        raise Error("x_decomp fa_gram: the tile slabs do not fit this column's threadgroup memory")
    var n = _n(p, 0)
    var d = _n(p, 1)
    if n < 1 or d < 1:
        raise Error("x_decomp fa_gram: empty input")
    if d > FA_MAX_D:
        raise Error("x_decomp fa_gram: more than " + String(FA_MAX_D) + " features")
    var nt = fa_gram_tiles(d)
    var nz = fa_gram_parts(n)
    var px = _ptr(_id(x), n * d)
    var pm = _ptr(_id(mean), d)
    var words = 2 if FA_GRAM_DF else 1
    var pg = _ptr(_id(g), words * d * d)
    var pv = _ptr(_id(var_), d)
    var sid = pool_alloc(nz * words * d * d)
    var pp = _ptr(sid, nz * words * d * d)
    var ctx = xd_ctx()
    comptime if FA_GRAM_DF:
        ctx.enqueue_function[fa_gram_tile_df_kernel](
            px, pm, pp, Int32(n), Int32(d), Int32(nt), grid_dim=(nt * (nt + 1) // 2, nz, 1), block_dim=FG_TPB
        )
        ctx.enqueue_function[fa_gram_fold_df_kernel](
            pp, pg, pv, Int32(d), Int32(nz), Float32(1.0 / Float64(n)), grid_dim=_blocks(d * d), block_dim=TPB
        )
    else:
        ctx.enqueue_function[fa_gram_tile_kernel](
            px, pm, pp, Int32(n), Int32(d), Int32(nt), grid_dim=(nt * (nt + 1) // 2, nz, 1), block_dim=FG_TPB
        )
        ctx.enqueue_function[fa_gram_fold_kernel](
            pp, pg, pv, Int32(d), Int32(nz), Float32(1.0 / Float64(n)), grid_dim=_blocks(d * d), block_dim=TPB
        )
    pool_free(sid)
    return PythonObject(d)


# ------------------------------------------------------------ the EM loop
def fa_scale_kernel(g: F32Ptr, psi: F32Ptr, b: F32Ptr, sp: F32Ptr, flag: F32Ptr, d_in: Int32, inv_n: Float32):
    """b = D G D / n with D = diag(1 / (sqrt(psi) + 1e-12)), the kit's
    `div` / `mul` cells; sp = sqrt(psi) + 1e-12. A set convergence flag
    (LL_DEVICE) makes the launch a no-op."""
    if flag.unsafe_load(0) != Float32(0.0):
        return
    var d = Int(d_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < d * d:
        var i = t // d
        var j = t - i * d
        var spi = add(sqrt0(psi.unsafe_load(i)), FA_SMALL)
        var spj = add(sqrt0(psi.unsafe_load(j)), FA_SMALL)
        b.unsafe_store(t, mul(div0(div0(g.unsafe_load(t), spj), spi), inv_n))
        if j == 0:
            sp.unsafe_store(i, spi)


# TOMBSTONE: MOJOLEARN_FA_EIG_SMALL (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: fa_rr_eigh_block_kernel (the eigh as one threadgroup launch).
# Restore: git apply experiments/removed/MOJOLEARN_FA_EIG_SMALL.patch; record in docs/TOMBSTONES.md.


def fa_svd_scale_kernel(
    rt0: F32Ptr, psi: F32Ptr, rt: F32Ptr, sp: F32Ptr, flag: F32Ptr, d_in: Int32, inv_sqrt_n: Float32
):
    """FA_GRAM_DF: rt = (R D / sqrt(n))^T, row p = column p of R divided by
    sp_p = sqrt(psi_p) + 1e-12 then scaled by 1 / sqrt(n) (main's `div` and
    `scale` of Rx, the kit's cells); rt0 = R^T from `fa_chol_df_kernel`. A
    set convergence flag (LL_DEVICE) makes the launch a no-op."""
    if flag.unsafe_load(0) != Float32(0.0):
        return
    var d = Int(d_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < d * d:
        var p = t // d
        var i = t - p * d
        var spp = add(sqrt0(psi.unsafe_load(p)), FA_SMALL)
        rt.unsafe_store(t, mul(div0(rt0.unsafe_load(t), spp), inv_sqrt_n))
        if i == 0:
            sp.unsafe_store(p, spp)


# TOMBSTONE: MOJOLEARN_FA_EIG_SMALL (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: fa_rs_svd_block_kernel (the FA_GRAM_DF SVD as one threadgroup launch).
# Restore: git apply experiments/removed/MOJOLEARN_FA_EIG_SMALL.patch; record in docs/TOMBSTONES.md.


@always_inline
def _fa_ev[SVD: Bool](a: F32Ptr, d: Int, k: Int) -> Float32:
    """Eigenvalue k of D G D / n: a's diagonal (the eigh), or the square of
    singular value k of R D / sqrt(n) (FA_GRAM_DF, a = the d norms)."""
    comptime if SVD:
        var sv = a.unsafe_load(k)
        return mul(sv, sv)
    else:
        return a.unsafe_load(k * d + k)


@always_inline
def _fa_v[SVD: Bool](v: F32Ptr, d: Int, i: Int, col: Int) -> Float32:
    """Entry i of vector col: column col of v (the eigh), row col of vt (the SVD)."""
    comptime if SVD:
        return v.unsafe_load(col * d + i)
    else:
        return v.unsafe_load(i * d + col)


def fa_finish_kernel[SVD: Bool = False](
    a: F32Ptr, v: F32Ptr, sp: F32Ptr, psi: F32Ptr, w: F32Ptr, psi_new: F32Ptr, small: F32Ptr,
    stat: F32Ptr, llst: F32Ptr, llrec: F32Ptr, d_in: Int32, nc_in: Int32, it_in: Int32, half_n: Float32, tol: Float32,
):
    """One threadgroup, after the eigh: the diagonal of a ranked descending
    (ties to the lower index), s2 = max(eigenvalue, 0), the nc leading
    columns of v signed as `sign_flip_kernel` signs them (largest |.| entry
    positive, first on a tie), W[j, i] = v[i, col_j] sqrt(max(s2_j - 1, 0))
    sp_i (the kit's `mul`), psi_new_i = max(sp_i^2 sum_r wt_r v[i, col_r]^2,
    1e-12) over every rank r ascending, wt_r = min(s2_r, 1) for r < nc and s2_r
    past them (main's cancellation-free psi update, lane/apple-fast-quality-
    glmfa: the same value as var - colsum(W^2) in exact arithmetic, a sum of
    nonnegative terms here), and small = [log(max(s2_j, FLT_MIN)) for
    j < nc | s2_j for nc <= j < d | log(max(psi_i, FLT_MIN)) | stat]: the
    terms of main's log-likelihood, which the host sums in float64.
    llst[0] (always zero since the device test went) and llrec are kept in
    the signature only.
    SVD (FA_GRAM_DF): a = the d singular values of R D / sqrt(n) (s2 their
    squares) and v = V^T (vector p in row p), `_fa_ev` / `_fa_v`."""
    if llst.unsafe_load(0) != Float32(0.0):
        return
    var d = Int(d_in)
    var nc = Int(nc_in)
    var tid = Int(thread_idx.x)
    var posd = stack_allocation[FA_MAX_D, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var fct = stack_allocation[FA_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sgn = stack_allocation[FA_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var wt = stack_allocation[FA_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if tid < d:
        var wv = _fa_ev[SVD](a, d, tid)
        var rank = 0
        for k in range(d):
            var wk = _fa_ev[SVD](a, d, k)
            if wk > wv or (wk == wv and k < tid):
                rank += 1
        posd[rank] = Int32(tid)
    barrier()
    if tid < d:
        var col = Int(posd[tid])
        var s2 = _fa_ev[SVD](a, d, col)
        if not (s2 > Float32(0.0)):
            s2 = Float32(0.0)
        wt[tid] = s2
        if tid < nc:
            wt[tid] = s2 if s2 < Float32(1.0) else Float32(1.0)
            var sm1 = sub_s(s2, Float32(1.0))
            fct[tid] = sqrt0(sm1 if sm1 > Float32(0.0) else Float32(0.0))
            small.unsafe_store(tid, log_floor(s2, FA_TINY))
            # the sign of column `col`: its largest-|.| entry, the first on a tie
            var biggest = Float32(0.0)
            for i in range(d):
                var m = abs(_fa_v[SVD](v, d, i, col))
                if m > biggest:
                    biggest = m
            var first = d
            for i in range(d):
                if abs(_fa_v[SVD](v, d, i, col)) == biggest and i < first:
                    first = i
            var neg = first < d and _fa_v[SVD](v, d, first, col) < Float32(0.0)
            sgn[tid] = Float32(-1.0) if neg else Float32(1.0)
        else:
            small.unsafe_store(tid, s2)
        small.unsafe_store(d + tid, log_floor(psi.unsafe_load(tid), FA_TINY))
    barrier()
    for t in range(tid, nc * d, FA_TPB):
        var j = t // d
        var i = t - j * d
        var col = Int(posd[j])
        var val = _fa_v[SVD](v, d, i, col)
        if sgn[j] < Float32(0.0):
            val = -val
        w.unsafe_store(t, mul(mul(val, fct[j]), sp.unsafe_load(i)))
    dev_barrier()
    if tid < d:
        var acc = Float32(0.0)
        for r in range(d):
            var x = _fa_v[SVD](v, d, tid, Int(posd[r]))
            acc = add(acc, mul(wt[r], mul(x, x)))
        var q = sp.unsafe_load(tid)
        var pn = mul(mul(q, q), acc)
        psi_new.unsafe_store(tid, pn if pn > FA_SMALL else FA_SMALL)
    if tid == 0:
        small.unsafe_store(2 * d, stat.unsafe_load(0))
        small.unsafe_store(2 * d + 1, stat.unsafe_load(1))
        small.unsafe_store(2 * d + 2, stat.unsafe_load(2))
        small.unsafe_store(2 * d + 3, stat.unsafe_load(3))
    # TOMBSTONE: MOJOLEARN_FA_LL_DEVICE (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
    # Tried: the double-float ll sum and the device tol test here.
    # Restore: git apply experiments/removed/MOJOLEARN_FA_LL_DEVICE.patch; record in docs/TOMBSTONES.md.


@always_inline
def sub_s(a: Float32, b: Float32) -> Float32:
    """The kit's `sub` cell (x_decomp/cells.mojo): ftz(ftz(a) - ftz(b))."""
    return ftz(ftz(a) - ftz(b))


struct _FaMem(Movable):
    """The loop's device scratch: one arena buffer (FA_LIVEBUF) or one buffer
    per slot. Slots are named by index; offsets are 64-float aligned."""
    var bufs: List[DeviceBuffer[DType.float32]]
    var which: List[Int]
    var offs: List[Int]
    var lens: List[Int]
    var arena: Bool
    var total: Int

    def __init__(out self, arena: Bool):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.which = List[Int]()
        self.offs = List[Int]()
        self.lens = List[Int]()
        self.arena = arena
        self.total = 0

    def add(mut self, ctx: DeviceContext, count: Int) raises -> Int:
        var cnt = max(count, 1)
        var padded = (cnt + 63) // 64 * 64
        if self.arena:
            self.which.append(0)
            self.offs.append(self.total)
            self.total += padded
        else:
            self.bufs.append(ctx.enqueue_create_buffer[DType.float32](padded))
            self.which.append(len(self.bufs) - 1)
            self.offs.append(0)
        self.lens.append(cnt)
        return len(self.lens) - 1

    def seal(mut self, ctx: DeviceContext) raises:
        if self.arena:
            self.bufs.append(ctx.enqueue_create_buffer[DType.float32](max(self.total, 1)))

    def ptr(self, s: Int) -> F32Ptr:
        return F32Ptr(unsafe_from_address=Int(self.bufs[self.which[s]].unsafe_ptr())) + self.offs[s]

    def sub(self, s: Int) raises -> DeviceBuffer[DType.float32]:
        return self.bufs[self.which[s]].create_sub_buffer[DType.float32](self.offs[s], self.lens[s])


def _fa_eigh_grid(
    ctx: DeviceContext, mem: _FaMem, s_b: Int, s_v: Int, s_cs: Int, s_off: Int, s_part: Int, s_fold: Int,
    mut hfold: HostBuffer[DType.float32], d: Int, s_es: Int,
) raises -> SIMD[DType.float32, 4]:
    """Main's `DevExec._eigh_par_on` rounds and per-sweep test on the loop's
    pointers (no sign flip, no ordering, no download: `fa_finish_kernel`
    orders and signs the columns it uses). Returns (converged, sweeps run,
    off, fro)."""
    var dm = d + (d % 2)
    var h = dm // 2
    var nb = _pj_off_blocks(d)
    var pa = mem.ptr(s_b)
    var pv = mem.ptr(s_v)
    var pcs = mem.ptr(s_cs)
    var poff = mem.ptr(s_off)
    var ppart = mem.ptr(s_part)
    var pfold = mem.ptr(s_fold)
    ctx.enqueue_function[pj_identity_kernel](pv, Int32(d), grid_dim=_pj_blocks(d * d), block_dim=PJ_TPB)
    # lane idn-cov-overflow: the eigh's power-of-two range scale
    # (x_decomp/eigh_scale.mojo; no write inside the band), the diagonal
    # unscaled below for `fa_finish_kernel`
    var pes = mem.ptr(s_es)
    enqueue_es_scale_strided(ctx, pa, d * d, 1, d, pes)
    var converged = False
    var executed = 0
    var fro_in = Float32(-1.0)
    var fro_now = Float32(0.0)
    var off_last = Float32(0.0)
    for sweep in range(RR_EIGH_SWEEPS + 1):
        var spart = mem.sub(s_part)
        var sfold = mem.sub(s_fold)
        enqueue_fill(ctx, spart, Float32(-1.0))
        enqueue_fill(ctx, sfold, Float32(-1.0))
        ctx.enqueue_function[eigh_par_off_part_kernel](pa, poff, ppart, Int32(d), grid_dim=nb, block_dim=RR_OFF_TPB)
        ctx.enqueue_function[eigh_par_off_fold_kernel](ppart, pfold, Int32(nb), grid_dim=1, block_dim=RR_OFF_TPB)
        ctx.enqueue_copy(dst_ptr=hfold.unsafe_ptr(), src_buf=sfold)
        ctx.synchronize()
        var hp = F32Ptr(unsafe_from_address=Int(hfold.unsafe_ptr()))
        var off = hp.unsafe_load(0)
        var dg = hp.unsafe_load(1)
        if not (hp.unsafe_load(2) >= Float32(0.0)):
            raise Error("x_decomp fa_em: a block of the eigh's convergence test did not run (a launch failure)")
        off_last = off
        fro_now = ftz(off + dg)
        if fro_in < Float32(0.0):
            fro_in = fro_now
        if rr_converged(off, dg, Float32(JACOBI_TOL)):
            converged = True
            break
        if sweep == RR_EIGH_SWEEPS:
            break
        executed += 1
        for rd in range(dm - 1):
            ctx.enqueue_function[eigh_par_cs_kernel](
                pa, pcs, Int32(d), Int32(dm), Int32(rd), grid_dim=_pj_blocks(h), block_dim=PJ_TPB
            )
            ctx.enqueue_function[eigh_par_update_kernel](
                pa, pv, pcs, Int32(d), Int32(dm), Int32(rd), grid_dim=_pj_blocks(h * h + d * h), block_dim=PJ_TPB
            )
            if rd % PJ_SYNC_ROUNDS == PJ_SYNC_ROUNDS - 1:
                ctx.synchronize()
    if converged and not rr_fro_kept(fro_in, fro_now):
        converged = False
    enqueue_es_unscale_diag_ptr(ctx, pa, pes, d)
    enqueue_es_unscale_ptr(ctx, poff + 2 * d, pes, 1, d)
    return SIMD[DType.float32, 4](Float32(1.0) if converged else Float32(0.0), Float32(executed), off_last, fro_now)


def _fa_svd_grid(
    ctx: DeviceContext, mem: _FaMem, s_rt: Int, s_vt: Int, s_s: Int, s_flags: Int,
    mut hflags: HostBuffer[DType.float32], d: Int, s_es: Int,
) raises -> SIMD[DType.float32, 4]:
    """FA_GRAM_DF: main's `DevExec.svd_cells` rounds on the
    loop's pointers (one `rs_round_kernel` launch a round, a pair a block,
    the pair flags read once per sweep; a sweep with no rotation is
    converged), then `rs_norm_kernel`. Returns (converged, sweeps run, 0, 0)."""
    var dm = d + (d % 2)
    var h = dm // 2
    var prt = mem.ptr(s_rt)
    var pvt = mem.ptr(s_vt)
    var ps = mem.ptr(s_s)
    var pfl = mem.ptr(s_flags)
    ctx.enqueue_function[pj_identity_kernel](pvt, Int32(d), grid_dim=_pj_blocks(d * d), block_dim=PJ_TPB)
    # the power-of-two range scale (x_decomp/eigh_scale.mojo), s unscaled
    var pes = mem.ptr(s_es)
    enqueue_es_scale_strided(ctx, prt, d * d, 1, d, pes)
    var converged = d < 2
    var executed = 0
    while not converged and executed < X_DECOMP_SVD_SWEEPS:
        executed += 1
        var sfl = mem.sub(s_flags)
        enqueue_fill(ctx, sfl, Float32(0.0))
        for rd in range(dm - 1):
            ctx.enqueue_function[rs_round_kernel](
                prt, pvt, pfl, Int32(d), Int32(dm), Int32(rd), X_DECOMP_SVD_TOL, grid_dim=h, block_dim=RS_TPB
            )
            if rd % PJ_SYNC_ROUNDS == PJ_SYNC_ROUNDS - 1:
                ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=hflags.unsafe_ptr(), src_buf=sfl)
        ctx.synchronize()
        var hp = F32Ptr(unsafe_from_address=Int(hflags.unsafe_ptr()))
        var rotated = False
        for b in range(h):
            if hp.unsafe_load(b) != Float32(0.0):
                rotated = True
        if not rotated:
            converged = True
    ctx.enqueue_function[rs_norm_kernel](prt, ps, Int32(d), grid_dim=d, block_dim=RS_TPB)
    enqueue_es_unscale_ptr(ctx, ps, pes, 1, d)
    return SIMD[DType.float32, 4](Float32(1.0) if converged else Float32(0.0), Float32(executed), 0.0, 0.0)


def _fa_ll(hp: F32Ptr, d: Int, nc: Int, llconst: Float64, neg_half_n: Float64) -> Float64:
    """Main's log-likelihood from the readback: slog, unexp and plog each a
    sequential float64 sum of float32 terms (`_dsum`), then
    (llconst + slog + unexp + plog) * (-n / 2)."""
    var slog = Float64(0.0)
    for j in range(nc):
        slog += Float64(hp.unsafe_load(j))
    var unexp = Float64(0.0)
    for j in range(nc, d):
        unexp += Float64(hp.unsafe_load(j))
    var plog = Float64(0.0)
    var dd2 = 2 * d
    for j in range(d, dd2):
        plog += Float64(hp.unsafe_load(j))
    return (llconst + slog + unexp + plog) * neg_half_n


def _fa_check_eigh(conv: Float32, sweeps: Float32, off: Float32, fro: Float32, d: Int) raises:
    if conv == Float32(0.0):
        raise Error(
            "x_decomp fa_em: the round-robin Jacobi did not converge in " + String(Int(sweeps)) + " sweeps at d = "
            + String(d) + " (off-diagonal mass " + String(off) + " of " + String(fro)
            + "). An unconverged decomposition is not returned as if it were one (DEVIATION 590)."
        )


def fa_em_py(
    g: PythonObject, psi0: PythonObject, w_out: PythonObject, psi_out: PythonObject,
    ll_out: PythonObject, p: PythonObject, tol: PythonObject,
) raises -> PythonObject:
    """FactorAnalysis.fit's EM loop on the resident Gram: p = [d, nc, n,
    max_iter]; g the device id of G (d x d), psi0 host floats (d),
    w_out host floats (nc x d), psi_out host floats (d), ll_out host float64
    (max_iter). Returns the iterations run (len(loglike_))."""
    var d = _n(p, 0)
    var nc = _n(p, 1)
    var n = _n(p, 2)
    var max_iter = _n(p, 3)
    if d < 1 or d > FA_MAX_D:
        raise Error("x_decomp fa_em: d must be in [1, " + String(FA_MAX_D) + "]")
    if nc < 1 or nc > d or n < 1 or max_iter < 1:
        raise Error("x_decomp fa_em: bad shape")
    var pg = _ptr(_id(g), (2 if FA_GRAM_DF else 1) * d * d)
    var psi_src = F32Ptr(unsafe_from_address=Int(py=psi0))
    var w_dst = F32Ptr(unsafe_from_address=Int(py=w_out))
    var psi_dst = F32Ptr(unsafe_from_address=Int(py=psi_out))
    var ll_dst = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=ll_out))
    var tol64 = Float64(py=tol)
    var dm = d + (d % 2)
    var h = dm // 2
    var nb = _pj_off_blocks(d)
    var ctx = xd_ctx()
    var mem = _FaMem(FA_LIVEBUF)
    var s_b = mem.add(ctx, d * d)
    var s_v = mem.add(ctx, d * d)
    var s_sp = mem.add(ctx, d)
    var s_w = mem.add(ctx, nc * d)
    var s_pa = mem.add(ctx, d)
    var s_pb = mem.add(ctx, d)
    var s_small = mem.add(ctx, 2 * d + 4)
    var s_cs = mem.add(ctx, 2 * h)
    var s_stat = mem.add(ctx, 4)
    var s_off = mem.add(ctx, 3 * d)
    var s_part = mem.add(ctx, 3 * nb)
    var s_fold = mem.add(ctx, 3)
    var s_es = mem.add(ctx, es_strided_words(1, d))
    var s_llst = mem.add(ctx, 4)
    var s_llrec = mem.add(ctx, 2 * max_iter)
    # FA_GRAM_DF: R^T (d x d), the singular values, the grid SVD's pair flags
    # (after W, psi A and psi B: LIVEBUF reads those three as one span)
    var s_r0 = mem.add(ctx, d * d if FA_GRAM_DF else 1)
    var s_s = mem.add(ctx, d if FA_GRAM_DF else 1)
    var s_flags = mem.add(ctx, h if FA_GRAM_DF else 1)
    mem.seal(ctx)
    var hsmall = ctx.enqueue_create_host_buffer[DType.float32](2 * d + 4)
    var hfold = ctx.enqueue_create_host_buffer[DType.float32](3)
    var hflags = ctx.enqueue_create_host_buffer[DType.float32](max(h, 1))
    # psi0 up, the flag and the status words zero
    var spa = mem.sub(s_pa)
    ctx.enqueue_copy(dst_buf=spa, src_ptr=psi_src)
    var sllst = mem.sub(s_llst)
    enqueue_fill(ctx, sllst, Float32(0.0))
    var sstat = mem.sub(s_stat)
    enqueue_fill(ctx, sstat, Float32(1.0))
    var pstat = mem.ptr(s_stat)
    var pllst = mem.ptr(s_llst)
    var pllrec = mem.ptr(s_llrec)
    var inv_n = Float32(1.0 / Float64(n))
    var inv_sqrt_n = Float32(1.0 / sqrt(Float64(n)))
    comptime if FA_GRAM_DF:
        # R^T R = G once, double-float, one threadgroup (G's words overwritten)
        ctx.enqueue_function[fa_chol_df_kernel](pg, mem.ptr(s_r0), Int32(d), grid_dim=1, block_dim=FA_TPB)
    var llconst = Float64(d) * FA_LOG_2PI + Float64(nc)
    var neg_half_n = -Float64(n) / 2.0
    var half_n = Float32(Float64(n) / 2.0)
    var tol32 = Float32(tol64)
    var old_ll = Float64.MIN_FINITE
    var it = 0
    var cur = s_pa
    var nxt = s_pb
    var i = 0
    while i < max_iter:
        it = i + 1
        var est = SIMD[DType.float32, 4](1.0, 0.0, 0.0, 0.0)
        comptime if FA_GRAM_DF:
            # the SVD of R D / sqrt(n) (rt in s_b, vt in s_v, s in s_s)
            ctx.enqueue_function[fa_svd_scale_kernel](
                mem.ptr(s_r0), mem.ptr(cur), mem.ptr(s_b), mem.ptr(s_sp), pllst, Int32(d), inv_sqrt_n,
                grid_dim=_blocks(d * d), block_dim=TPB,
            )
            est = _fa_svd_grid(ctx, mem, s_b, s_v, s_s, s_flags, hflags, d, s_es)
            _fa_check_eigh(est[0], est[1], est[2], est[3], d)
            ctx.enqueue_function[fa_finish_kernel[True]](
                mem.ptr(s_s), mem.ptr(s_v), mem.ptr(s_sp), mem.ptr(cur), mem.ptr(s_w), mem.ptr(nxt),
                mem.ptr(s_small), pstat, pllst, pllrec, Int32(d), Int32(nc), Int32(i), half_n, tol32,
                grid_dim=1, block_dim=FA_TPB,
            )
        else:
            ctx.enqueue_function[fa_scale_kernel](
                pg, mem.ptr(cur), mem.ptr(s_b), mem.ptr(s_sp), pllst, Int32(d), inv_n, grid_dim=_blocks(d * d), block_dim=TPB
            )
            est = _fa_eigh_grid(ctx, mem, s_b, s_v, s_cs, s_off, s_part, s_fold, hfold, d, s_es)
            _fa_check_eigh(est[0], est[1], est[2], est[3], d)
            ctx.enqueue_function[fa_finish_kernel[False]](
                mem.ptr(s_b), mem.ptr(s_v), mem.ptr(s_sp), mem.ptr(cur), mem.ptr(s_w), mem.ptr(nxt),
                mem.ptr(s_small), pstat, pllst, pllrec, Int32(d), Int32(nc), Int32(i), half_n, tol32,
                grid_dim=1, block_dim=FA_TPB,
            )
        var ssm = mem.sub(s_small)
        ctx.enqueue_copy(dst_ptr=hsmall.unsafe_ptr(), src_buf=ssm)
        ctx.synchronize()
        var hp = F32Ptr(unsafe_from_address=Int(hsmall.unsafe_ptr()))
        var ll = _fa_ll(hp, d, nc, llconst, neg_half_n)
        ll_dst.unsafe_store(i, ll)
        if (ll - old_ll) < tol64:
            break
        old_ll = ll
        var tmp = cur
        cur = nxt
        nxt = tmp
        i += 1
    # TOMBSTONE: MOJOLEARN_FA_LL_DEVICE (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
    # Tried: the stop iteration and ll pairs read back from the device here.
    # Restore: git apply experiments/removed/MOJOLEARN_FA_LL_DEVICE.patch; record in docs/TOMBSTONES.md.
    comptime if FA_LIVEBUF:
        # W, psi A and psi B are adjacent in the arena: one copy
        var span = (mem.offs[s_pb] + mem.lens[s_pb]) - mem.offs[s_w]
        var hout = ctx.enqueue_create_host_buffer[DType.float32](span)
        var sall = mem.bufs[0].create_sub_buffer[DType.float32](mem.offs[s_w], span)
        ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=sall)
        ctx.synchronize()
        var op = F32Ptr(unsafe_from_address=Int(hout.unsafe_ptr()))
        for k in range(nc * d):
            w_dst.unsafe_store(k, op.unsafe_load(k))
        var poff = mem.offs[cur] - mem.offs[s_w]
        for k in range(d):
            psi_dst.unsafe_store(k, op.unsafe_load(poff + k))
        _ = hout^
    else:
        ctx.enqueue_copy(dst_ptr=w_dst, src_buf=mem.sub(s_w))
        ctx.enqueue_copy(dst_ptr=psi_dst, src_buf=mem.sub(cur))
        ctx.synchronize()
    _ = hsmall^
    _ = hfold^
    _ = hflags^
    _ = mem^
    ctx.synchronize()
    return PythonObject(it)


# ------------------------------------------------------------ transform
comptime FA_TR_TPB = 256
comptime FA_TR_MAXK = 16
comptime FA_TR_FLOATS = 4096
comptime FA_TR_SMEM_BYTES = FA_TR_FLOATS * 4
comptime FA_TR_FITS = lib_smem_page_fits_for[TARGET_COLUMN, FA_TR_SMEM_BYTES]()


def fa_transform_kernel(x: F32Ptr, mean: F32Ptr, pm: F32Ptr, dst: F32Ptr, n_in: Int32, d_in: Int32, nc_in: Int32):
    """out[row] = (x[row] - mean) P, P d x nc in threadgroup memory with the
    mean behind it, one row per thread, nc <= FA_TR_MAXK accumulators."""
    var n = Int(n_in)
    var d = Int(d_in)
    var nc = Int(nc_in)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[FA_TR_FLOATS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var np = d * nc
    for idx in range(tid, np + d, FA_TR_TPB):
        sh[idx] = pm.unsafe_load(idx) if idx < np else mean.unsafe_load(idx - np)
    barrier()
    var row = Int(block_idx.x) * FA_TR_TPB + tid
    if row >= n:
        return
    var acc = InlineArray[Float32, FA_TR_MAXK](fill=Float32(0.0))
    for j in range(d):
        var xj = x.unsafe_load(row * d + j) - sh[np + j]
        for kk in range(nc):
            acc[kk] += xj * sh[j * nc + kk]
    for kk in range(nc):
        dst.unsafe_store(row * nc + kk, acc[kk])


def fa_transform_py(x: PythonObject, mean: PythonObject, pm: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, d, nc]: dst (n x nc, device id) = (X - mean) P for the
    resident X (n x d), mean (d) and P (d x nc); enqueued, no sync."""
    comptime if not FA_TR_FITS:
        raise Error("x_decomp fa_transform: the P page does not fit this column's threadgroup memory")
    var n = _n(p, 0)
    var d = _n(p, 1)
    var nc = _n(p, 2)
    if n < 1 or d < 1 or nc < 1:
        raise Error("x_decomp fa_transform: empty input")
    if nc > FA_TR_MAXK or d * nc + d > FA_TR_FLOATS:
        raise Error("x_decomp fa_transform: P and the mean do not fit the page (nc <= 16, d (nc + 1) <= 4096)")
    var px = _ptr(_id(x), n * d)
    var pmean = _ptr(_id(mean), d)
    var pp = _ptr(_id(pm), d * nc)
    var po = _ptr(_id(dst), n * nc)
    var ctx = xd_ctx()
    ctx.enqueue_function[fa_transform_kernel](
        px, pmean, pp, po, Int32(n), Int32(d), Int32(nc), grid_dim=(n + FA_TR_TPB - 1) // FA_TR_TPB, block_dim=FA_TR_TPB
    )
    return PythonObject(n)


# ------------------------------------------------------------ the defines
def fa_defines_py() raises -> PythonObject:
    """The FactorAnalysis switches this FAST Apple binding was built with
    (`-D MOJOLEARN_FA_...`), comma-joined, so python/mojolearn/
    _expansion_decomp.py `_fa_fast_define` picks a route without an env read.
    Registered only under FA_FAST_APPLE (bindings/_mojolearn_x_decomp.mojo)."""
    var s = String("")
    comptime if FA_GRAM_ONCE:
        s += "MOJOLEARN_FA_GRAM_ONCE,"
    comptime if FA_ITER_DEVICE:
        s += "MOJOLEARN_FA_ITER_DEVICE,"
    comptime if FA_LIVEBUF:
        s += "MOJOLEARN_FA_LIVEBUF,"
    comptime if FA_TRANSFORM_FUSED:
        s += "MOJOLEARN_FA_TRANSFORM_FUSED,"
    # not a -D name: the route ITER_DEVICE takes unless MOJOLEARN_FA_GRAM_QOLD
    # (python/mojolearn/_expansion_decomp.py sizes G's buffer from it)
    comptime if FA_GRAM_DF:
        s += "MOJOLEARN_FA_GRAM_DF,"
    comptime if FA_FAST_APPLE and FA_GRAM_QOLD:
        s += "MOJOLEARN_FA_GRAM_QOLD,"
    return PythonObject(s)
