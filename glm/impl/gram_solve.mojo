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


def gs_chol_step_kernel(a: F32Ptr, info: F32Ptr, j: Int32, n: Int32):
    """Column step j of `chol_serial`, one thread per row i >= j (the
    statements of x_decomp/device.mojo `chol_step_kernel`: row j's thread is
    `chol_diag`; every row i > j re-forms the same pivot chain and runs
    `chol_col_elem`'s statements with it)."""
    var jj = Int(j)
    var nn = Int(n)
    var i = jj + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= nn:
        return
    if i == jj:
        chol_diag(a, info, jj, nn)
        return
    var acc = ftz(a.unsafe_load(jj * nn + jj))
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
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=d_y, src_ptr=y_ptr)
    var d_mx = ctx.enqueue_create_buffer[DType.float32](d)
    var d_my = ctx.enqueue_create_buffer[DType.float32](1)
    if center:
        var d_sx = ctx.enqueue_create_buffer[DType.uint64](d)
        var d_sy = ctx.enqueue_create_buffer[DType.uint64](1)
        col_sums_pair_buf(ctx, d_x, d_y, d_sx, d_sy, n_rows, d)
        var d_m64 = ctx.enqueue_create_buffer[DType.uint64](d + 1)
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
        _ = d_m64^
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
            _p(d_a), _p(d_info), Int32(j), Int32(d), grid_dim=_gs_blocks(d - j), block_dim=GS_TPB
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
    _ = h_flag^
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
