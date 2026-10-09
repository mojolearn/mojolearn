# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE (lane/classical-structural, 2026-10-07):
one resident centered-Gram fit for LinearRegression (alpha = 0) and Ridge
(alpha > 0), IDENTICAL only, default off.

WHAT IT REPLACES. OLS: `DevExec.ols_tsqr_factor` writes a centered m x (d+1)
copy and runs a 14-panel blocked Householder TSQR over it (each panel
re-reads and re-writes the trailing matrix), then the Python glue runs the
small SVD of R through more binding cells. Ridge: four PCIe crossings of X
(column sums, center, the centered download, the fit's upload), a
single-block Jacobi eigensolve and the untiled `U = A V` that writes a full
m x d U. Both end in the same normal equations the fp32 opponents solve.

WHAT IT DOES. X and y go up once. The exact column sums and the binary64
means are the existing items (`col_sums_pair_buf`, `col_means_buf`). The
centered Gram G = (X - mu)^T (X - mu) and the centered cross c = (X - mu)^T
(y - ybar) are read straight from the resident X (no centered copy): the
repo's blocked leaf kernels (`bm_centered_gram_panels`, `bm_centered_cross_panels`,
core/blocked_moments.mojo) with the binary-counter fold of the leaf partials,
leaves of `contract_leaf_size(n)` rows. Then the power-of-two equilibration
of `ols_equilibration_scale` (DEVIATION 2620, exact), A = S (G + alpha I) S,
b = S c; the IDENTICAL Cholesky (the `chol_diag` / `chol_col_elem` cells of
x_decomp/cells.mojo, one launch per column, one thread per row, as
`DevExec.chol`); a trust gate (the Cholesky reported a non-positive pivot,
or a pivot square below 2^-12 of its equilibrated diagonal, x_linear/ridge.mojo's
RIDGE_FF_GATE); the two triangular solves, one launch per column with the
rows in parallel; and w = S w'. Only w (d floats) and the means come home.
No U is formed.

COST REASONING. Two reads of X (Gram, cross) at full width replace the
TSQR's 14 panel sweeps or Ridge's seven crossings; the solve is about 3 d
small launches. That holds for any tall n x d with d a few hundred.

FALLBACK (the quality guard for ill-conditioned designs). When the Gram does
not factor or the trust gate fails, the entry returns status 1 with nothing
written, and the Python glue runs the incumbent route (TSQR + SVD cutoff for
OLS, the eigen route for Ridge). Both columns compute the same booleans, so
both take the same route.
Lane fg-linear L2 (IDN_RIDGE_RESIDENT, default on): a rejected RIDGE fit no
longer goes back to Python; `_gs_ridge_resident_fallback` runs the same eigen
route on the resident buffers (status 0). The host column keeps returning 1
and running its host eigen route: the same words.

BITS. They change (normal equations instead of TSQR/eig). The host column,
glm/host/gram_solve_host.mojo, runs the same cells in the same order:
`centered_gram_v1_cell` / `centered_cross_v1_cell` (the reference cells the
leaf kernels reproduce), `chol_serial`, and the same solve statements.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, identical_mul, identical_mul_add
from x_decomp.cells import F32Ptr, chol_diag, div0, sqrt0
from core.blocked_moments import bm_centered_cross_panels, bm_centered_gram_panels
from core.device_zero import enqueue_fill
from gemm.contract import contract_leaf_size
from glm.impl.center_device import col_means_buf, col_sums_pair_buf
from glm.impl.linalg.detail.lstsq import ols_equilibration_scale
from glm.impl.gram_solve_cells import gs_equilibrated_cell, gs_pivot_trusted, gs_regularized_diag, gs_scaled_rhs
from experiments.classical_identical_ideas.fg_linear_controls import IDN_GRAM_FF_FALLBACK, IDN_RIDGE_RESIDENT
from glm.impl.gram_ff_cells import (
    GFF_LEAVES,
    gff_backward_cell,
    gff_cells,
    gff_chol_cell,
    gff_coef,
    gff_equilibrated_cell,
    gff_fold_cell,
    gff_forward_cell,
    gff_leaf_cell,
    gff_mean_split,
    gff_pivot_trusted,
    gff_regularized_diag_f32,
)
from x_linear.ff import ff_ld, ff_mul_f, ff_st
from glm.impl.center_device import center_buf
from glm.impl.ridge import RIDGE_ALGO_EIG, ridge_eig_scratch_traced
from core.identity_trace import IdentityTrace
from glm.impl.pinned_upload import linear_upload_f32

comptime GS_TPB = 128


def _gs_blocks(count: Int) -> Int:
    return (count + GS_TPB - 1) // GS_TPB if count > 0 else 1


@always_inline
def gs_scale_of(g: F32Ptr, i: Int, d: Int, alpha: Float32) -> Float32:
    """The equilibration scale of column i (`ols_equilibration_scale` of the
    regularized diagonal)."""
    return ols_equilibration_scale(gs_regularized_diag(g.unsafe_load(i * d + i), alpha))


def gs_equilibrate_kernel(g: F32Ptr, c: F32Ptr, a: F32Ptr, adiag: F32Ptr, b: F32Ptr, sv: F32Ptr, d_in: Int32, alpha: Float32):
    """One thread per cell of A; the diagonal threads also write the
    equilibrated diagonal copy (the trust gate reads it after the Cholesky
    overwrote A), the scaled right-hand side and the scale vector."""
    var d = Int(d_in)
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if q >= d * d:
        return
    var i = q // d
    var j = q - i * d
    var si = gs_scale_of(g, i, d, alpha)
    var sj = gs_scale_of(g, j, d, alpha)
    var v = gs_equilibrated_cell(g, i, j, d, alpha, si, sj)
    a.unsafe_store(q, v)
    if i == j:
        adiag.unsafe_store(i, v)
        b.unsafe_store(i, gs_scaled_rhs(c, i, si))
        sv.unsafe_store(i, si)


def gs_info_init_kernel(info: F32Ptr):
    if block_idx.x == 0 and thread_idx.x == 0:
        info.unsafe_store(0, Float32(0))


def gs_chol_step_kernel(a: F32Ptr, adiag: F32Ptr, info: F32Ptr, j: Int32, n: Int32):
    """Column step j of `chol_serial`, one thread per row i >= j (the
    statements of x_decomp/device.mojo `chol_step_kernel`: row j's thread is
    `chol_diag`; every row i > j re-forms the same pivot chain and runs
    `chol_col_elem`'s statements with it). The chain starts from `adiag[j]`,
    the equilibrated diagonal's untouched copy, which is the word a[j, j]
    holds when the step starts (only this step's row-j thread ever writes
    a[j, j]): the same value, read from a word no thread of the step writes."""
    var jj = Int(j)
    var nn = Int(n)
    var i = jj + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= nn:
        return
    if i == jj:
        chol_diag(a, info, jj, nn)
        return
    var acc = ftz(adiag.unsafe_load(jj))
    for p in range(jj):
        var l = ftz(a.unsafe_load(jj * nn + p))
        acc = ftz(identical_mul_add(-l, l, acc))
    if not (acc > Float32(0)):
        acc = Float32(1)
    var dpiv = sqrt0(acc)
    var s = ftz(a.unsafe_load(i * nn + jj))
    for p in range(jj):
        s = ftz(identical_mul_add(-ftz(a.unsafe_load(i * nn + p)), ftz(a.unsafe_load(jj * nn + p)), s))
    a.unsafe_store(i * nn + jj, div0(s, dpiv))
    a.unsafe_store(jj * nn + i, Float32(0))


def gs_trust_kernel(l: F32Ptr, adiag: F32Ptr, info: F32Ptr, flag: F32Ptr, d_in: Int32):
    """flag[0] = 1 when the factor is trusted (info 0 and every pivot passes
    the gate), else 0. One thread over the d pivots (d = the feature count;
    the whole launch is d loads)."""
    if block_idx.x != 0 or thread_idx.x != 0:
        return
    var d = Int(d_in)
    var ok = info.unsafe_load(0) == Float32(0)
    for j in range(d):
        if not gs_pivot_trusted(l.unsafe_load(j * d + j), adiag.unsafe_load(j)):
            ok = False
    flag.unsafe_store(0, Float32(1) if ok else Float32(0))


def gs_forward_step_kernel(l: F32Ptr, b: F32Ptr, z: F32Ptr, j: Int32, n: Int32):
    """Forward solve L z = b, column step j: z_j = b_j / L[j, j] (every
    thread forms the same quotient), rows i > j take b_i -= L[i, j] z_j;
    thread j stores z_j into `z` (a separate vector, so no thread of the
    step writes a word another thread of the step reads)."""
    var jj = Int(j)
    var nn = Int(n)
    var i = jj + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= nn:
        return
    var zj = div0(ftz(b.unsafe_load(jj)), l.unsafe_load(jj * nn + jj))
    if i == jj:
        z.unsafe_store(jj, zj)
    else:
        b.unsafe_store(i, ftz(identical_mul_add(-ftz(l.unsafe_load(i * nn + jj)), zj, ftz(b.unsafe_load(i)))))


def gs_backward_step_kernel(l: F32Ptr, z: F32Ptr, w: F32Ptr, j: Int32, n: Int32):
    """Backward solve L^T w = z, column step j (descending): w_j = z_j /
    L[j, j], rows i < j take z_i -= L[j, i] w_j; thread j stores w_j into
    `w` (reads and writes of the step are disjoint, as above)."""
    var jj = Int(j)
    var nn = Int(n)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i > jj:
        return
    var wj = div0(ftz(z.unsafe_load(jj)), l.unsafe_load(jj * nn + jj))
    if i == jj:
        w.unsafe_store(jj, wj)
    else:
        z.unsafe_store(i, ftz(identical_mul_add(-ftz(l.unsafe_load(jj * nn + i)), wj, ftz(z.unsafe_load(i)))))


def gs_unscale_kernel(w: F32Ptr, sv: F32Ptr, d_in: Int32):
    """w_i = s_i w'_i (exact power-of-two scaling, flushed)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(d_in):
        w.unsafe_store(i, ftz(identical_mul(ftz(sv.unsafe_load(i)), ftz(w.unsafe_load(i)))))


def _p(buf: DeviceBuffer[DType.float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


# ---------------------------------------------------------------------------
# Lane fg-linear L3 (IDN_GRAM_FF_FALLBACK, default off): the float-float
# second chance of a rejected Ridge Gram (glm/impl/gram_ff_cells.mojo).
# ---------------------------------------------------------------------------
comptime _U64Ptr = MutPointer[UInt64, MutAnyOrigin]


def gff_split_kernel(m64: _U64Ptr, mh: F32Ptr, ml: F32Ptr, d_in: Int32, center: Int32):
    """The d + 1 binary64 means (X then y) as float-float words; zeros when
    the fit does not center."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j > Int(d_in):
        return
    if center != 0:
        var f = gff_mean_split(m64.unsafe_load(j))
        mh.unsafe_store(j, f.hi)
        ml.unsafe_store(j, f.lo)
    else:
        mh.unsafe_store(j, Float32(0))
        ml.unsafe_store(j, Float32(0))


def gff_leaf_kernel(x: F32Ptr, y: F32Ptr, mh: F32Ptr, ml: F32Ptr, ph: F32Ptr, pl: F32Ptr, n_in: Int64, d_in: Int32):
    """grid (cells / GS_TPB, GFF_LEAVES): one thread per (leaf, cell);
    adjacent threads take adjacent Gram columns j of one row i, so a row's
    loads of x[r, j] coalesce and x[r, i] broadcasts."""
    var d = Int(d_in)
    var t = gff_cells(d)
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var leaf = Int(block_idx.y)
    if q >= t:
        return
    var v = gff_leaf_cell(x, y, mh, ml, Int(n_in), d, leaf, q)
    ph.unsafe_store(leaf * t + q, v.hi)
    pl.unsafe_store(leaf * t + q, v.lo)


def gff_fold_kernel(ph: F32Ptr, pl: F32Ptr, gh: F32Ptr, gl: F32Ptr, ch: F32Ptr, cl: F32Ptr, d_in: Int32):
    var d = Int(d_in)
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if q < gff_cells(d):
        gff_fold_cell(ph, pl, gh, gl, ch, cl, d, q)


def gff_equilibrate_kernel(
    gh: F32Ptr, gl: F32Ptr, ch: F32Ptr, cl: F32Ptr, ah: F32Ptr, al: F32Ptr, adh: F32Ptr, adl: F32Ptr,
    bh: F32Ptr, bl: F32Ptr, sv: F32Ptr, d_in: Int32, alpha: Float32,
):
    """One thread per cell of A (float-float); the diagonal threads also
    write the diagonal copy, the scaled right-hand side and the scale."""
    var d = Int(d_in)
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if q >= d * d:
        return
    var i = q // d
    var j = q - i * d
    var si = ols_equilibration_scale(gff_regularized_diag_f32(gh, gl, i, d, alpha))
    var sj = ols_equilibration_scale(gff_regularized_diag_f32(gh, gl, j, d, alpha))
    var v = gff_equilibrated_cell(gh, gl, i, j, d, alpha, si, sj)
    ff_st(ah, al, q, v)
    if i == j:
        ff_st(adh, adl, i, v)
        ff_st(bh, bl, i, ff_mul_f(ff_ld(ch, cl, i), si))
        sv.unsafe_store(i, si)


def gff_chol_step_kernel(ah: F32Ptr, al: F32Ptr, adh: F32Ptr, adl: F32Ptr, info: F32Ptr, j: Int32, n: Int32):
    var jj = Int(j)
    var i = jj + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        gff_chol_cell(ah, al, adh, adl, info, i, jj, Int(n))


def gff_trust_kernel(lh: F32Ptr, adh: F32Ptr, info: F32Ptr, flag: F32Ptr, d_in: Int32):
    """flag[0] = 1 when info is 0 and every float-float pivot passes
    GFF_TRUST_GATE. One thread over the d pivots."""
    if block_idx.x != 0 or thread_idx.x != 0:
        return
    var d = Int(d_in)
    var ok = info.unsafe_load(0) == Float32(0)
    for j in range(d):
        if not gff_pivot_trusted(lh.unsafe_load(j * d + j), adh.unsafe_load(j)):
            ok = False
    flag.unsafe_store(0, Float32(1) if ok else Float32(0))


def gff_forward_step_kernel(lh: F32Ptr, ll: F32Ptr, bh: F32Ptr, bl: F32Ptr, zh: F32Ptr, zl: F32Ptr, j: Int32, n: Int32):
    var jj = Int(j)
    var i = jj + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        gff_forward_cell(lh, ll, bh, bl, zh, zl, i, jj, Int(n))


def gff_backward_step_kernel(lh: F32Ptr, ll: F32Ptr, zh: F32Ptr, zl: F32Ptr, wh: F32Ptr, wl: F32Ptr, j: Int32, n: Int32):
    var jj = Int(j)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i <= jj:
        gff_backward_cell(lh, ll, zh, zl, wh, wl, i, jj, Int(n))


def gff_coef_kernel(wh: F32Ptr, wl: F32Ptr, sv: F32Ptr, dst: F32Ptr, d_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(d_in):
        dst.unsafe_store(i, gff_coef(wh, wl, sv, i))


def _gs_ff_fit(
    ctx: DeviceContext,
    mut d_x: DeviceBuffer[DType.float32],
    mut d_y: DeviceBuffer[DType.float32],
    mut d_m64: DeviceBuffer[DType.uint64],
    center: Bool,
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    d: Int,
    alpha: Float32,
) raises -> Int:
    """IDN_GRAM_FF_FALLBACK: 0 with coef written when the float-float factor
    is trusted, else 1 with nothing written. All on the device; one wait for
    the trust flag, one for coef."""
    var t = gff_cells(d)
    var d_mh = ctx.enqueue_create_buffer[DType.float32](d + 1)
    var d_ml = ctx.enqueue_create_buffer[DType.float32](d + 1)
    ctx.enqueue_function[gff_split_kernel](
        _U64Ptr(unsafe_from_address=Int(d_m64.unsafe_ptr())), _p(d_mh), _p(d_ml), Int32(d), Int32(1 if center else 0),
        grid_dim=_gs_blocks(d + 1), block_dim=GS_TPB,
    )
    var d_ph = ctx.enqueue_create_buffer[DType.float32](GFF_LEAVES * t)
    var d_pl = ctx.enqueue_create_buffer[DType.float32](GFF_LEAVES * t)
    ctx.enqueue_function[gff_leaf_kernel](
        _p(d_x), _p(d_y), _p(d_mh), _p(d_ml), _p(d_ph), _p(d_pl), Int64(n_rows), Int32(d),
        grid_dim=(_gs_blocks(t), GFF_LEAVES, 1), block_dim=(GS_TPB, 1, 1),
    )
    var d_gh = ctx.enqueue_create_buffer[DType.float32](d * d)
    var d_gl = ctx.enqueue_create_buffer[DType.float32](d * d)
    var d_ch = ctx.enqueue_create_buffer[DType.float32](d)
    var d_cl = ctx.enqueue_create_buffer[DType.float32](d)
    ctx.enqueue_function[gff_fold_kernel](
        _p(d_ph), _p(d_pl), _p(d_gh), _p(d_gl), _p(d_ch), _p(d_cl), Int32(d), grid_dim=_gs_blocks(t), block_dim=GS_TPB
    )
    var d_ah = ctx.enqueue_create_buffer[DType.float32](d * d)
    var d_al = ctx.enqueue_create_buffer[DType.float32](d * d)
    var d_adh = ctx.enqueue_create_buffer[DType.float32](d)
    var d_adl = ctx.enqueue_create_buffer[DType.float32](d)
    var d_bh = ctx.enqueue_create_buffer[DType.float32](d)
    var d_bl = ctx.enqueue_create_buffer[DType.float32](d)
    var d_s = ctx.enqueue_create_buffer[DType.float32](d)
    ctx.enqueue_function[gff_equilibrate_kernel](
        _p(d_gh), _p(d_gl), _p(d_ch), _p(d_cl), _p(d_ah), _p(d_al), _p(d_adh), _p(d_adl),
        _p(d_bh), _p(d_bl), _p(d_s), Int32(d), alpha,
        grid_dim=_gs_blocks(d * d), block_dim=GS_TPB,
    )
    var d_info = ctx.enqueue_create_buffer[DType.float32](1)
    var d_flag = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[gs_info_init_kernel](_p(d_info), grid_dim=1, block_dim=1)
    for j in range(d):
        ctx.enqueue_function[gff_chol_step_kernel](
            _p(d_ah), _p(d_al), _p(d_adh), _p(d_adl), _p(d_info), Int32(j), Int32(d),
            grid_dim=_gs_blocks(d - j), block_dim=GS_TPB,
        )
    ctx.enqueue_function[gff_trust_kernel](
        _p(d_ah), _p(d_adh), _p(d_info), _p(d_flag), Int32(d), grid_dim=1, block_dim=1
    )
    var h_flag = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=h_flag, src_buf=d_flag)
    ctx.synchronize()
    var status = 1
    if h_flag.unsafe_ptr().unsafe_load(0) != Float32(0):
        var d_zh = ctx.enqueue_create_buffer[DType.float32](d)
        var d_zl = ctx.enqueue_create_buffer[DType.float32](d)
        var d_wh = ctx.enqueue_create_buffer[DType.float32](d)
        var d_wl = ctx.enqueue_create_buffer[DType.float32](d)
        var d_w = ctx.enqueue_create_buffer[DType.float32](d)
        for j in range(d):
            ctx.enqueue_function[gff_forward_step_kernel](
                _p(d_ah), _p(d_al), _p(d_bh), _p(d_bl), _p(d_zh), _p(d_zl), Int32(j), Int32(d),
                grid_dim=_gs_blocks(d - j), block_dim=GS_TPB,
            )
        for jj in range(d):
            var j = d - 1 - jj
            ctx.enqueue_function[gff_backward_step_kernel](
                _p(d_ah), _p(d_al), _p(d_zh), _p(d_zl), _p(d_wh), _p(d_wl), Int32(j), Int32(d),
                grid_dim=_gs_blocks(j + 1), block_dim=GS_TPB,
            )
        ctx.enqueue_function[gff_coef_kernel](
            _p(d_wh), _p(d_wl), _p(d_s), _p(d_w), Int32(d), grid_dim=_gs_blocks(d), block_dim=GS_TPB
        )
        ctx.enqueue_copy(dst_ptr=coef_ptr, src_buf=d_w)
        ctx.synchronize()
        status = 0
        _ = d_zh^
        _ = d_zl^
        _ = d_wh^
        _ = d_wl^
        _ = d_w^
    _ = h_flag^
    _ = d_mh^
    _ = d_ml^
    _ = d_ph^
    _ = d_pl^
    _ = d_gh^
    _ = d_gl^
    _ = d_ch^
    _ = d_cl^
    _ = d_ah^
    _ = d_al^
    _ = d_adh^
    _ = d_adl^
    _ = d_bh^
    _ = d_bl^
    _ = d_s^
    _ = d_info^
    _ = d_flag^
    return status


def _gs_ridge_resident_fallback(
    ctx: DeviceContext,
    mut d_x: DeviceBuffer[DType.float32],
    mut d_y: DeviceBuffer[DType.float32],
    mut d_mx: DeviceBuffer[DType.float32],
    mut d_my: DeviceBuffer[DType.float32],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    alpha: Float32,
    center: Bool,
) raises:
    """IDN_RIDGE_RESIDENT (lane fg-linear L2, experiments/classical_identical_ideas/
    fg_linear_controls.mojo): Ridge's eig route on the X and y this entry
    already uploaded and the means it already formed, instead of status 1 and
    the Python glue's four crossings. The statements are
    `ridge_fit_resident_host`'s (glm/estimator.mojo) after its means: center X
    and y by the float32 means (`center_buf`), then `ridge_eig_scratch_traced`
    with the raw X (dead after the center) as the Gram's scratch. With
    `center` False the raw X and y are the design, a fresh buffer the scratch.
    Writes coef (n_features floats)."""
    var cells = n_rows * n_features
    var w = ctx.enqueue_create_buffer[DType.float32](n_features)
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            String("ridge n=") + String(n_rows) + " d=" + String(n_features)
            + " algo=" + String(RIDGE_ALGO_EIG)
        )
    if center:
        var d_cx = ctx.enqueue_create_buffer[DType.float32](cells)
        var d_cy = ctx.enqueue_create_buffer[DType.float32](n_rows)
        center_buf(ctx, d_x, d_mx, d_cx, n_rows, n_features)
        center_buf(ctx, d_y, d_my, d_cy, n_rows, 1)
        ctx.synchronize()
        trace.record_device[DType.float32](ctx, "ridge.input.A", d_cx, cells)
        trace.record_device[DType.float32](ctx, "ridge.input.b", d_cy, n_rows)
        trace.record_scalar_f32("ridge.input.alpha", alpha)
        # d_x is dead from here: the Gram's gemm_tn scratch.
        ridge_eig_scratch_traced(ctx, d_cx, n_rows, n_features, d_cy, alpha, w, d_x, trace)
        _ = d_cx^
        _ = d_cy^
    else:
        var xa = ctx.enqueue_create_buffer[DType.float32](cells)
        ctx.synchronize()
        trace.record_device[DType.float32](ctx, "ridge.input.A", d_x, cells)
        trace.record_device[DType.float32](ctx, "ridge.input.b", d_y, n_rows)
        trace.record_scalar_f32("ridge.input.alpha", alpha)
        ridge_eig_scratch_traced(ctx, d_x, n_rows, n_features, d_y, alpha, w, xa, trace)
        _ = xa^
    trace.record_device[DType.float32](ctx, "ridge.coef", w, n_features)
    ctx.enqueue_copy(dst_ptr=coef_ptr, src_buf=w)
    ctx.synchronize()
    _ = w^


def linear_gram_fit_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mu_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ymean_ptr: MutPointer[Float64, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    alpha: Float32,
    center: Bool,
) raises -> Int:
    """The entry (module docstring). Returns 0 and writes coef (d floats)
    and, with `center`, mu (float32 d) and the y mean (binary64); returns 1
    with nothing written when the Gram is not trusted (the caller's route)."""
    if n_features <= 0 or n_rows <= 0:
        raise Error("linear_gram_fit: n_rows and n_features must be positive")
    if alpha < Float32(0.0):
        raise Error("linear_gram_fit: alpha must be non-negative")
    var d = n_features
    var cells = n_rows * d
    var d_x = ctx.enqueue_create_buffer[DType.float32](cells)
    var d_y = ctx.enqueue_create_buffer[DType.float32](n_rows)
    # lane fg-linear L4 (IDN_LINEAR_PINNED_UPLOAD, default on): through the
    # pinned stage when large; the direct copy otherwise (no bit moves)
    linear_upload_f32(ctx, d_x, x_ptr, cells)
    linear_upload_f32(ctx, d_y, y_ptr, n_rows)
    var d_mx = ctx.enqueue_create_buffer[DType.float32](d)
    var d_my = ctx.enqueue_create_buffer[DType.float32](1)
    # the binary64 means stay alive for the fit (lane fg-linear L3 reads them)
    var d_m64 = ctx.enqueue_create_buffer[DType.uint64](d + 1)
    if center:
        var d_sx = ctx.enqueue_create_buffer[DType.uint64](d)
        var d_sy = ctx.enqueue_create_buffer[DType.uint64](1)
        col_sums_pair_buf(ctx, d_x, d_y, d_sx, d_sy, n_rows, d)
        col_means_buf(ctx, d_sx, d_sy, d_mx, d_my, d_m64, n_rows, d)
        # the fit's outputs: mu (float32) and the y mean (binary64 bits)
        ctx.enqueue_copy(dst_ptr=mu_ptr, src_buf=d_mx)
        ctx.enqueue_copy(
            dst_ptr=MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(ymean_ptr)),
            src_buf=d_m64.create_sub_buffer[DType.uint64](d, 1),
        )
        ctx.synchronize()
        _ = d_sx^
        _ = d_sy^
    else:
        enqueue_fill(ctx, d_mx, Float32(0))
        enqueue_fill(ctx, d_my, Float32(0))
    var leaf = contract_leaf_size(n_rows)
    var d_g = ctx.enqueue_create_buffer[DType.float32](d * d)
    var d_c = ctx.enqueue_create_buffer[DType.float32](d)
    bm_centered_gram_panels(ctx, _p(d_g), _p(d_x), _p(d_mx), n_rows, d, leaf)
    bm_centered_cross_panels(ctx, _p(d_c), _p(d_x), _p(d_mx), _p(d_y), _p(d_my), n_rows, d, leaf)
    var d_a = ctx.enqueue_create_buffer[DType.float32](d * d)
    var d_adiag = ctx.enqueue_create_buffer[DType.float32](d)
    var d_b = ctx.enqueue_create_buffer[DType.float32](d)
    var d_z = ctx.enqueue_create_buffer[DType.float32](d)
    var d_w = ctx.enqueue_create_buffer[DType.float32](d)
    var d_s = ctx.enqueue_create_buffer[DType.float32](d)
    var d_info = ctx.enqueue_create_buffer[DType.float32](1)
    var d_flag = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[gs_equilibrate_kernel](
        _p(d_g), _p(d_c), _p(d_a), _p(d_adiag), _p(d_b), _p(d_s), Int32(d), alpha,
        grid_dim=_gs_blocks(d * d), block_dim=GS_TPB,
    )
    ctx.enqueue_function[gs_info_init_kernel](_p(d_info), grid_dim=1, block_dim=1)
    for j in range(d):
        ctx.enqueue_function[gs_chol_step_kernel](
            _p(d_a), _p(d_adiag), _p(d_info), Int32(j), Int32(d), grid_dim=_gs_blocks(d - j), block_dim=GS_TPB
        )
    ctx.enqueue_function[gs_trust_kernel](
        _p(d_a), _p(d_adiag), _p(d_info), _p(d_flag), Int32(d), grid_dim=1, block_dim=1
    )
    var h_flag = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=h_flag, src_buf=d_flag)
    ctx.synchronize()
    var trusted = h_flag.unsafe_ptr().unsafe_load(0) != Float32(0)
    var status = 1
    if trusted:
        for j in range(d):
            ctx.enqueue_function[gs_forward_step_kernel](
                _p(d_a), _p(d_b), _p(d_z), Int32(j), Int32(d), grid_dim=_gs_blocks(d - j), block_dim=GS_TPB
            )
        for jj in range(d):
            var j = d - 1 - jj
            ctx.enqueue_function[gs_backward_step_kernel](
                _p(d_a), _p(d_z), _p(d_w), Int32(j), Int32(d), grid_dim=_gs_blocks(j + 1), block_dim=GS_TPB
            )
        ctx.enqueue_function[gs_unscale_kernel](_p(d_w), _p(d_s), Int32(d), grid_dim=_gs_blocks(d), block_dim=GS_TPB)
        ctx.enqueue_copy(dst_ptr=coef_ptr, src_buf=d_w)
        ctx.synchronize()
        status = 0
    comptime if IDN_GRAM_FF_FALLBACK:
        # lane fg-linear L3 (default off): a rejected Ridge Gram first tries
        # the float-float Gram + Cholesky on the resident X (no Jacobi).
        if status != 0 and alpha > Float32(0.0):
            status = _gs_ff_fit(ctx, d_x, d_y, d_m64, center, coef_ptr, n_rows, d, alpha)
    if status != 0 and alpha > Float32(0.0) and d > 1 and n_rows > 1:
        # IDN_RIDGE_RESIDENT (lane fg-linear L2, default on): a rejected
        # Ridge Gram runs the eig route on the resident X / y and the means
        # formed above (the same words the Python glue's route reaches),
        # instead of status 1 and four PCIe crossings of X. OLS (alpha 0)
        # still returns 1 (its incumbent is the TSQR). The d == 1 and n <= 1
        # cases keep status 1 (the incumbent's own refusals by name).
        comptime if IDN_RIDGE_RESIDENT:
            _gs_ridge_resident_fallback(ctx, d_x, d_y, d_mx, d_my, coef_ptr, n_rows, d, alpha, center)
            status = 0
    _ = h_flag^
    _ = d_m64^
    _ = d_x^
    _ = d_y^
    _ = d_mx^
    _ = d_my^
    _ = d_g^
    _ = d_c^
    _ = d_a^
    _ = d_adiag^
    _ = d_b^
    _ = d_z^
    _ = d_w^
    _ = d_s^
    _ = d_info^
    _ = d_flag^
    return status
