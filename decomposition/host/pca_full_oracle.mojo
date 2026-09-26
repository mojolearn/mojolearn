# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""`PCA(svd_solver='full')` TRAINING on the host, for a box with no GPU (the
pca-full-whiten lane, 2026-09-14): the R-SVD of the centered data, the
Householder QR and the one-sided Jacobi as the device path runs them.

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu` or a `DeviceContext`,
and no GPU binding imports this file. Every arithmetic statement is a
`checks/numerics.mojo` leaf (`ftz`, `identical_mul_add`, `identical_sqrt`,
`identical_div`) or a plain IEEE float32 operation the device kernel also
performs unpinned, spelled a SECOND time from the device kernels named
below, in their order. The pieces the covariance arm already restated
(the column mean, the halving tree, the rotation `(c, s)`, the rotation
update pair, the sign flip and the Float64 tail) are imported from
`decomposition/host/pca_oracle.mojo`, because the device arm imports the
same kernels from the covariance arm.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS:

  `host_qr_panel_slice`    `qr_panel_kernel`, `core/householder_qr.mojo:288`,
                           one block (one row slice `b`). The slice bounds
                           `(b * m) // n_slices .. ((b + 1) * m) // n_slices`.
                           Per column `j`: the norm partials, lane `t` adding
                           rows `j + t, j + t + QR_TPB, ...` through
                           `ftz(identical_mul_add(v, v, acc))`, folded by the
                           unflushed halving tree (`fold_and_broadcast`,
                           `halving_block_sum[QR_TPB]`); `normx =
                           ftz(identical_sqrt(sigma))`; a zero `normx` stores
                           `R_jj = 0` and applies nothing (DEVIATION 588);
                           otherwise the reflector's three scalars
                           (`qr_reflector_r`, `qr_reflector_u1`,
                           `qr_reflector_tau`), the packed `w` rows `j + 1 ..`
                           divided by `u1` through `identical_div`, and per
                           trailing column `c` the dot partials starting at
                           row `j + 1 + t`, the fold, `total = ftz(ajc +
                           tail)`, `td = ftz(tau * total)`, the head
                           `ftz(ajc - td)` and the rows `ftz(identical_mul_add(
                           -td, w, cur))`. Then the upper triangle copied from
                           the slice's leading rows (zero past `ms`) and the
                           strict lower triangle zeroed.
  `host_qr_factor`         `qr_factor`, `:431`, one device:
                           `qr_slice_count(n_rows, n_cols)` slices (a pure
                           function of the shape, DEVIATION 589), each slice's
                           `R` into its `n x n` tile, then the SAME panel on
                           the stacked `(ns * n) x n` tiles as one slice. One
                           slice factors `x` straight into `R`.
                           `MOJOLEARN_QR_DEVICE_COUNT` above 1 moves each
                           slice to a device and keeps its rows and lanes
                           (`qr_parallel_panels`), so the bits do not depend
                           on it and it is not restated.
  `host_one_sided_jacobi_svd`
                           `one_sided_jacobi_svd_kernel`, `decomposition/impl/
                           linalg/detail/svd_full.mojo:179`, the tall arm
                           (`wide_rotation = False`): `W = I`; per sweep, per
                           pair `p < q`, the three Gram partials over rows
                           `t, t + SVD_TPB, ...` folded by the halving tree
                           (`SVD_TPB = JACOBI_TPB`), the relative test
                           `|apq| > ftz(tol * ftz(sqrt(app) * sqrt(aqq)))`,
                           the rotation of the columns of `R` and of `W`; a
                           sweep with no rotation converges; the singular
                           values are the flushed roots of the folded column
                           norms of the rotated `R`.
  `host_pca_fit_full`      `pca_fit_full`, `svd_full.mojo:417`, tall only:
                           `pca_full_validate`, `column_mean_kernel`, the
                           unfused `shift_columns_kernel` at `sign = -1`, the
                           QR, the SVD of `R` with its convergence refusal in
                           its words, `sign_flip_kernel` on the right basis,
                           `diag[i] = S_i^2 / (n_rows - 1)` in Float64 and the
                           shared tail at `singular_scale = n_rows - 1`.

WHAT IS NOT RESTATED, AND IS REFUSED BY NAME: the WIDE route (DEVIATION 593,
`n_rows < n_cols`: the transposed single-panel QR, the wide rotation at 64
sweeps and 1e-6, and the reflectors applied to the basis). No identity_break
fixture is wide, so it would be unmeasured host code; the host fit raises
instead of hashing something the gate never saw.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` (read here through
`PCA_ORACLE_HOST_SABOTAGE`, the estimators family's one define) folds every
QR trailing-column dot product's lane partials SERIALLY DESCENDING instead of
through the halving tree, a different association for every column with more
than one live lane, so `R`, and every output after it, moves in its last
bits.

The restatement is a prediction until measured. The CPU identity gate
(`tools/identity_break.py --diff <3 GPU columns> <cpu json> --lanes
pca-full-whiten --require-columns 4`) is the measurement.
"""
from checks.numerics import ftz, identical_div, identical_mul_add, identical_sqrt
from decomposition.host.pca_oracle import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_TPB,
    PCA_ORACLE_HOST_SABOTAGE,
    PCAHostFit,
    _rot_add,
    _rot_sub,
    host_column_mean,
    host_halving_sum,
    host_jacobi_rotation_cs,
    host_order_truncate_spectrum,
    host_pca_validate,
    host_shift_columns,
    host_sign_flip,
)


#: `QR_TPB`, `QR_MAX_SLICES` and `QR_SLICE_ROWS_PER_COL`
#: (`core/householder_qr.mojo:175-183`), restated because that file imports
#: the GPU. The first is the fold width (a flat numeric constant there, not a
#: matrix row); the other two decide the slice count from the shape alone.
comptime QR_TPB = 32
comptime QR_MAX_SLICES = 64
comptime QR_SLICE_ROWS_PER_COL = 4

#: `SVD_TPB = JACOBI_TPB` (`svd_full.mojo:176`), the same fold width as the
#: covariance arm's eigensolver, read from the kernel matrix there and here.
comptime SVD_TPB = JACOBI_TPB


def host_qr_slice_count(n_rows: Int, n_cols: Int) -> Int:
    """`qr_slice_count`: halve from QR_MAX_SLICES while a slice would hold
    fewer than QR_SLICE_ROWS_PER_COL rows per column."""
    var s = QR_MAX_SLICES
    while s > 1 and n_rows < s * (QR_SLICE_ROWS_PER_COL * n_cols):
        s //= 2
    return s


def _qr_fold(partials: List[Float32], sabotage: Bool) -> Float32:
    """`fold_and_broadcast[QR_TPB]`: the unflushed halving tree. The
    sabotage arm (dot products only) adds the lanes serially descending."""
    if sabotage:
        var acc = Float32(0.0)
        for tt in range(len(partials)):
            acc = acc + partials[len(partials) - 1 - tt]
        return acc
    return host_halving_sum(partials)


def host_qr_panel_slice(
    mut a: List[Float32],
    mut r_out: List[Float32],
    m: Int,
    n: Int,
    n_slices: Int,
    b: Int,
):
    """`qr_panel_kernel` for block `b`; `a` is `m x n` row major and is
    factored in place, `r_out` receives the slice's `R` at `b * n * n`."""
    var rb = (b * m) // n_slices
    var re = ((b + 1) * m) // n_slices
    var ms = re - rb
    var rbase = b * n * n

    for j in range(n):
        var norm_partials = List[Float32](length=QR_TPB, fill=Float32(0.0))
        for t in range(QR_TPB):
            var acc = Float32(0.0)
            var i = j + t
            while i < ms:
                var v = ftz(a[(rb + i) * n + j])
                acc = ftz(identical_mul_add(v, v, acc))
                i += QR_TPB
            norm_partials[t] = acc
        var sigma = host_halving_sum(norm_partials)
        var normx = ftz(identical_sqrt(sigma))
        var ajj = ftz(a[(rb + j) * n + j])

        if normx == Float32(0.0):
            r_out[rbase + j * n + j] = Float32(0.0)
        else:
            # `qr_reflector_r`, `qr_reflector_u1`, `qr_reflector_tau`.
            var s = Float32(-1.0) if ajj >= Float32(0.0) else Float32(1.0)
            var r_jj = ftz(s * normx)
            var u1 = ftz(ajj - r_jj)
            var tau = ftz(identical_div(ftz(ftz(-s) * u1), normx))
            r_out[rbase + j * n + j] = r_jj

            for i2 in range(j + 1, ms):
                var cur = ftz(a[(rb + i2) * n + j])
                a[(rb + i2) * n + j] = ftz(identical_div(cur, u1))

            for c in range(j + 1, n):
                var dot_partials = List[Float32](length=QR_TPB, fill=Float32(0.0))
                for t in range(QR_TPB):
                    var dacc = Float32(0.0)
                    var i3 = j + 1 + t
                    while i3 < ms:
                        var w = ftz(a[(rb + i3) * n + j])
                        var x = ftz(a[(rb + i3) * n + c])
                        dacc = ftz(identical_mul_add(w, x, dacc))
                        i3 += QR_TPB
                    dot_partials[t] = dacc
                var tail = _qr_fold(dot_partials, PCA_ORACLE_HOST_SABOTAGE)
                var ajc = ftz(a[(rb + j) * n + c])
                var total = ftz(ajc + tail)
                var td = ftz(tau * total)
                a[(rb + j) * n + c] = ftz(ajc - td)
                for i4 in range(j + 1, ms):
                    var w2 = ftz(a[(rb + i4) * n + j])
                    var cur2 = ftz(a[(rb + i4) * n + c])
                    a[(rb + i4) * n + c] = ftz(identical_mul_add(-td, w2, cur2))

    for t in range(n * n):
        var rr = t // n
        var cc = t - rr * n
        if cc > rr:
            if rr < ms:
                r_out[rbase + t] = ftz(a[(rb + rr) * n + cc])
            else:
                r_out[rbase + t] = Float32(0.0)
        elif cc < rr:
            r_out[rbase + t] = Float32(0.0)


def host_qr_factor(mut a: List[Float32], n_rows: Int, n_cols: Int) raises -> List[Float32]:
    """`qr_factor` on one device: `R` (`n_cols x n_cols`, row major) of the
    tall `a`, which is destroyed."""
    if n_rows < n_cols:
        raise Error(
            "qr_factor needs at least as many rows as columns, got "
            + String(n_rows)
            + " x "
            + String(n_cols)
            + ". The route for a wide matrix is an LQ factorization of the"
            " transpose, which this file does not carry; see DEVIATION 593"
            " in decomposition/impl/linalg/detail/svd_full.mojo"
        )
    var ns = host_qr_slice_count(n_rows, n_cols)
    var nn = n_cols * n_cols
    var r = List[Float32](length=nn, fill=Float32(0.0))
    if ns == 1:
        host_qr_panel_slice(a, r, n_rows, n_cols, 1, 0)
        return r^
    var scratch = List[Float32](length=ns * nn, fill=Float32(0.0))
    for b in range(ns):
        host_qr_panel_slice(a, scratch, n_rows, n_cols, ns, b)
    host_qr_panel_slice(scratch, r, ns * n_cols, n_cols, 1, 0)
    return r^


@fieldwise_init
struct OneSidedJacobiHostResult(Movable):
    """What `one_sided_jacobi_svd_kernel` leaves: the right basis (vector
    `i` in COLUMN `i`), the unordered singular values and the info slots."""

    var v: List[Float32]
    var s: List[Float32]
    var converged: Bool
    var executed: Int
    var last_rots: Int


def _svd_fold_partials(
    r: List[Float32], n: Int, p: Int, q: Int,
) -> SIMD[DType.float32, 4]:
    """The three Gram folds of columns `p` and `q` (`app`, `aqq`, `apq`)."""
    var lp = List[Float32](length=SVD_TPB, fill=Float32(0.0))
    var lq = List[Float32](length=SVD_TPB, fill=Float32(0.0))
    var lpq = List[Float32](length=SVD_TPB, fill=Float32(0.0))
    for t in range(SVD_TPB):
        var ap = Float32(0.0)
        var aq = Float32(0.0)
        var apq = Float32(0.0)
        var i = t
        while i < n:
            var xp = ftz(r[i * n + p])
            var xq = ftz(r[i * n + q])
            ap = ftz(identical_mul_add(xp, xp, ap))
            aq = ftz(identical_mul_add(xq, xq, aq))
            apq = ftz(identical_mul_add(xp, xq, apq))
            i += SVD_TPB
        lp[t] = ap
        lq[t] = aq
        lpq[t] = apq
    return SIMD[DType.float32, 4](
        host_halving_sum(lp), host_halving_sum(lq), host_halving_sum(lpq), Float32(0.0)
    )


def host_one_sided_jacobi_svd(
    mut r: List[Float32], n: Int, max_sweeps: Int, tol: Float32,
) -> OneSidedJacobiHostResult:
    """`one_sided_jacobi_svd_kernel[wide_rotation=False]` replayed serially;
    `r` is consumed (its columns end as `U * S`)."""
    var v = List[Float32](length=n * n, fill=Float32(0.0))
    for i in range(n):
        v[i * n + i] = Float32(1.0)

    var executed = 0
    var converged = False
    var last_rots = 0
    for _sweep in range(max_sweeps):
        var rots = 0
        for p in range(n):
            for q in range(p + 1, n):
                var g = _svd_fold_partials(r, n, p, q)
                var app = g[0]
                var aqq = g[1]
                var apq = g[2]
                var np_ = ftz(identical_sqrt(app))
                var nq_ = ftz(identical_sqrt(aqq))
                var thresh = ftz(tol * ftz(np_ * nq_))
                if abs(apq) > thresh:
                    rots += 1
                    var cs = host_jacobi_rotation_cs(app, aqq, apq)
                    var c = cs[0]
                    var s = cs[1]
                    for k in range(n):
                        var rkp = ftz(r[k * n + p])
                        var rkq = ftz(r[k * n + q])
                        r[k * n + p] = _rot_sub(c, rkp, s, rkq)
                        r[k * n + q] = _rot_add(s, rkp, c, rkq)
                    for k in range(n):
                        var vkp = ftz(v[k * n + p])
                        var vkq = ftz(v[k * n + q])
                        v[k * n + p] = _rot_sub(c, vkp, s, vkq)
                        v[k * n + q] = _rot_add(s, vkp, c, vkq)
        executed += 1
        last_rots = rots
        if rots == 0:
            converged = True
            break

    var sv = List[Float32](length=n, fill=Float32(0.0))
    for j in range(n):
        var partials = List[Float32](length=SVD_TPB, fill=Float32(0.0))
        for t in range(SVD_TPB):
            var acc = Float32(0.0)
            var i2 = t
            while i2 < n:
                var vv = ftz(r[i2 * n + j])
                acc = ftz(identical_mul_add(vv, vv, acc))
                i2 += SVD_TPB
            partials[t] = acc
        sv[j] = ftz(identical_sqrt(host_halving_sum(partials)))
    return OneSidedJacobiHostResult(v^, sv^, converged, executed, last_rots)


def host_pca_full_validate(n_rows: Int, n_cols: Int, n_components: Int) raises:
    """`pca_full_validate` under IDENTICAL, with the wide route refused by
    name (see the module docstring) after the checks the device makes."""
    host_pca_validate(n_rows, n_cols, n_components)
    if n_rows < n_cols:
        if n_components > n_rows:
            raise Error("full SVD n_components cannot exceed min(n_samples, n_features)")
        raise Error(
            "pca_fit_full on the CPU: the wide route (n_rows < n_cols, the"
            " transposed QR of DEVIATION 593) has no host restatement, so a "
            + String(n_rows)
            + " x "
            + String(n_cols)
            + " fit is refused by name on a CPU-only install rather than"
            " computed by an arithmetic the CPU identity gate never measured."
            " decomposition/host/pca_full_oracle.mojo carries the tall route"
            " only; svd_solver='covariance_eigh' handles this shape on the CPU"
        )


def host_pca_fit_full(
    x: List[Float32], n_rows: Int, n_cols: Int, n_components: Int,
) raises -> PCAHostFit:
    """`pca_fit_full` without the DeviceContext, tall matrices only."""
    host_pca_full_validate(n_rows, n_cols, n_components)
    var mu = host_column_mean(x, n_rows, n_cols)
    var centered = host_shift_columns(x, mu, n_rows, n_cols, Float32(-1.0))
    var r = host_qr_factor(centered, n_rows, n_cols)
    var svd = host_one_sided_jacobi_svd(r, n_cols, JACOBI_SWEEPS, Float32(JACOBI_TOL))
    if not svd.converged:
        raise Error(
            "the one-sided Jacobi SVD did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n_cols = "
            + String(n_cols)
            + ": the last sweep still performed "
            + String(Float32(svd.last_rots))
            + " rotations against a tolerance of "
            + String(Float32(JACOBI_TOL))
            + ". The remedy is more sweeps, the same one cuSOLVER's syevj"
            " has. An unconverged decomposition is not returned as if it"
            " were one; see DEVIATION 590."
        )
    var vecs32 = svd.v.copy()
    host_sign_flip(vecs32, n_cols)
    var denom = Float64(n_rows - 1)
    var diag = List[Float64]()
    for i in range(n_cols):
        var sv = Float64(svd.s[i])
        diag.append(sv * sv / denom)
    var vecs = List[Float64]()
    for i in range(n_cols * n_cols):
        vecs.append(Float64(vecs32[i]))
    var result = host_order_truncate_spectrum(
        diag, vecs, n_cols, n_components, n_rows - 1
    )
    return PCAHostFit(result^, mu^)
