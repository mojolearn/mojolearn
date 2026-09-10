# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`svd_solver='full'`: R-SVD of the CENTERED data, not of its covariance.

WHAT THIS ARM IS AND WHY IT IS NOT THE ONE ALREADY SHIPPED
==========================================================
`decomposition/NOT_IMPLEMENTED.tsv` carried this row: cuML maps BOTH `'auto'`
AND `'full'` onto `Solver.COV_EIG_DQ` (`pca.pyx:394-395`), so for THEM the two
names are one solver on the covariance and `'full'` means nothing extra. The
name only means a dense SVD of `X` in scikit-learn (`_pca.py:539-540` sends
`'full'` to `_fit_full`, which calls `scipy.linalg.svd` on the centered `X`),
and scikit-learn's is the signature `mojolearn.PCA` copies, so `'full'` must
mean the dense route or mean nothing.

THE VALUE OVER THE SHIPPED ARM IS REAL AND IS THE REASON TO CARRY IT.
Forming `X^T X` SQUARES the condition number. Float32 has 7.2 decimal digits
and `1/eps = 8.4e6`, so a design with `kappa(X) = 3e3` -- unremarkable for
correlated features -- gives `kappa(X^T X) = 9e6`, past the Float32 limit, and
the smallest eigenvalues of the covariance come back as noise. The R-SVD's
sensitivity is `kappa(X)`, not `kappa(X)^2`. This is the same argument
DEVIATION 678 made for the arima lane's least squares, and like that one it is
MEASURED here rather than argued:
`check_full_beats_covariance_on_ill_conditioning`.

That is also exactly why scikit-learn keeps `'full'` beside
`'covariance_eigh'` instead of collapsing them the way cuML does.

THE ROUTE, WHICH IS WHAT LAPACK DOES FOR A TALL MATRIX ANYWAY
-------------------------------------------------------------
    1. column means (`column_mean_kernel`), then center in place
       (`shift_columns_kernel`, `sign = -1`). THE SAME TWO KERNELS the
       covariance arm calls, so `mean_` is bit-identical between the arms
       and `check_full_mean_matches_covariance_arm` says so.
    2. `X_c = Q R`, `core/householder_qr.mojo::qr_factor`. `Q` is NEVER
       FORMED (see below).
    3. one-sided Jacobi SVD of the `n_cols x n_cols` `R`:
       `R W = U_r S`, so `X_c = (Q U_r) S W^T`.
    4. the right singular vectors are `W`, the singular values are `S`, and
       neither one mentions `Q`. `sign_flip_kernel` then
       `order_truncate_spectrum`, both shared with the covariance arm.

WHY `Q` IS NEVER FORMED. PCA's outputs are `components_` (`W^T`),
`singular_values_` (`S`), `explained_variance_` (`S^2/(n-1)`),
`explained_variance_ratio_` and `noise_variance_`, and `transform` is
`X_c W`. The left singular vectors appear in none of them. Not forming `Q`
removes an `O(n_samples x n_features)` output buffer and a whole `orgqr`
kernel from this arm, and it is the reason
`TruncatedSVD(algorithm='randomized')` is NOT closed in this pass: RAFT's
randomized route calls `qrGetQ` three times (`rsvd.cuh:198`, `:218`, `:241`)
and genuinely needs the explicit `Q` this arm does without.

DEVIATION 590: THE ONE-SIDED JACOBI, AND WHY IT IS NOT THE SHIPPED ONE
======================================================================
`decomposition/checks/jacobi_eigh_device.mojo` is a TWO-SIDED cyclic Jacobi
on a SYMMETRIC matrix. Feeding it `R^T R` would put the condition squaring
back in at the last step and throw away everything the QR just bought, so
the SVD of `R` is taken ONE-SIDED instead: the rotations are applied to the
COLUMNS of `R` only, and the 2x2 problem each one solves is built from three
column inner products rather than read off a matrix. At convergence the
columns of `R W` are mutually orthogonal, their norms ARE the singular
values, and `W` is the accumulated rotation.

That is Demmel-Veselic's construction and its whole point is that the small
singular values are computed to high RELATIVE accuracy, which is the
property this arm exists to deliver.

THE ROTATION ITSELF IS NOT A SECOND COPY. `jacobi_rotation_cs` and the
`_rot_sub` / `_rot_add` update pair are imported from the shipped
eigensolver, which was refactored in the same commit to call the same
helper. There is ONE spelling in the tree of `theta`, `t`, `c`, `s` and of
the contraction rule on `c*x - s*y`.

WHAT IS THIS FILE'S OWN, AND IS THEREFORE ITS OWN TO GATE:

  * THE GRAM OF EACH COLUMN PAIR is three block folds (`app`, `aqq`, `apq`),
    every partial through `identical_mul_add` and `ftz`, folded by
    `pinned_block_sum`. DEVIATION 587's argument applies unchanged: the
    association is a pure function of `SVD_TPB` and no lane primitive
    appears in it.
  * THE CONVERGENCE TEST IS RELATIVE AND PER PAIR:
    `|apq| > tol * sqrt(app) * sqrt(aqq)`. It is written as a product of
    two square roots and NOT as `sqrt(app * aqq)` because `app * aqq` is a
    product of two sums of squares and overflows Float32 on data a user can
    plausibly hand us (column norms around 1e20 are not exotic after a bad
    scaling), where the two-root form cannot. The test is scale invariant
    by construction, which is the defect DEVIATION BLOCK 1 of the shipped
    eigensolver had to be repaired for; `check_full_scale_invariance` is
    the gate and the sabotage arm is the absolute test.
  * THE EXIT IS A ROTATION COUNT, NOT A NORM. A sweep that performs zero
    rotations has converged, because every pair passed the test above.
    This is a quantity the machine gets a vote in exactly as the shipped
    solver's `off` is -- a last-bit disagreement about one `apq` changes
    THE NUMBER OF SWEEPS, and one more sweep is `n(n-1)/2` more rotations
    applied to `R` and `W`. That is why every input to the comparison is
    pinned, and why `SVD_TPB` is `JACOBI_TPB` rather than a second number.
  * NON-CONVERGENCE IS REFUSED, not returned. Same argument as
    `eig_and_truncate`'s: the arm cuML's default reaches (`eigDC`) aborts
    with "eigensolver couldn't converge to a solution" and their Jacobi arm
    silently does not, and a silently unconverged decomposition returned as
    if it were one is a wrong answer with no error.

DEVIATION 591: THE HOST TAIL IS SHARED, NOT COPIED.
`order_truncate_spectrum` in `pca.mojo` does the descending order, the
truncation, the ratios and `truncCompExpVars`' noise variance for BOTH arms.
This arm hands it `diag[i] = S_i^2 / (n_rows - 1)` and `singular_scale =
n_rows - 1`, so its `sqrt(lam * singular_scale)` returns `S_i` and the two
arms report the same five quantities by the same rules. The round trip
through the square costs less than one Float64 ulp on a Float32-derived
value; a private copy of the ordering rule would cost a second definition of
`noise_variance_`, which is the expensive one.

DEVIATION 593: WIDE IDENTICAL MATRICES USE TRANSPOSED QR (2026-09-10).
Center X using the shared kernels, factor X^T = Q R with a single QR panel,
SVD R^T = U S V^T, and form the PCA right basis Q V by applying stored
reflectors in reverse order. No covariance and no division by singular
values are used, including on rank-deficient inputs. The existing square
caller buffers retain their shape, with compact results zero-padded for
tracing. The shared host tail uses min(rows, cols) singular values and the
same count for noise variance. n_components cannot exceed that count.

The new wide Jacobi route uses 64 sweeps and relative tolerance 1e-6;
the inherited 1e-7 can cycle on roundoff-sized rank-deficient columns.
A local large-theta rotation avoids overflowing theta squared. Columns
whose pinned squared norms flush to zero are treated as zero in the relative
rotation test as well as in singular-value extraction; rotating against
zero norms made rank-deficient columns cycle on H100. Tall and
legacy numeric-mode behavior retain the original rotation, budget and
threshold. Nonconvergence is still refused. Single-panel Q reconstruction
and square caller buffers are capacity/performance limitations, not a
TSQR speed claim. Gates: decomposition/checks/svd_wide_check.mojo.

WHAT THIS ARM DOES NOT INHERIT FROM THE COVARIANCE ARM: the
`n_features > 128 under NUMERIC_IDENTICAL` refusal. That limit is the pinned
split-K Gram kernel's capacity (IDENTITY_PATHS row 27) and this arm never
builds a Gram. Tall feature counts above 128 remain RUN OWED; the standard
tall suite and wide 129-feature gate ran on Apple and NVIDIA (September 10);
`one-box-verdict-is-not-three` applies to a capability claim as much as to a
speed one.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import thread_idx, block_idx
from std.memory import stack_allocation

from checks.numerics import ftz, identical_mul_add, identical_sqrt, identical_div, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.column_stats import (
    STATS_TPB,
    column_mean_kernel,
    shift_columns_kernel,
)
from core.householder_qr import fold_and_broadcast, qr_factor, qr_slice_count, QR_TPB, qr_reflector_u1, qr_reflector_tau
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_TPB,
    _rot_add,
    _rot_sub,
    jacobi_rotation_cs,
)
from decomposition.impl.linalg.detail.pca import (
    PCAResult,
    SIGNFLIP_TPB,
    order_truncate_spectrum,
    pca_validate,
    sign_flip_kernel,
)


#: THE SAME NUMBER AS `JACOBI_TPB`, ON PURPOSE AND NOT BY COINCIDENCE. It is
#: the width of a fold that decides a sweep count, so a second value would be
#: a second summation order for the same 2x2 problems the shipped
#: eigensolver solves. Written as an assignment rather than restated so it
#: cannot be moved on one side only.
comptime SVD_TPB = JACOBI_TPB


def one_sided_jacobi_svd_kernel[wide_rotation: Bool = False](
    r: MutPointer[Float32, MutAnyOrigin],
    v_out: MutPointer[Float32, MutAnyOrigin],
    s_out: MutPointer[Float32, MutAnyOrigin],
    info_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    max_sweeps_in: Int32,
    tol_in: Float32,
):
    """One-sided cyclic Jacobi SVD of an `n x n` row-major `r`, ONE BLOCK.

    `r` in: the matrix. Out: its columns are `U * S`, which no shipped
    caller reads and which the gates use to check `U`'s orthonormality
    without a second kernel.

    `v_out` ends with right singular vector `i` in COLUMN `i`. That is
    LAPACK's convention, cuSOLVER's, `jacobi_eigh_kernel`'s and the one
    `sign_flip_kernel` and `order_truncate_spectrum` both expect.

    `s_out` is the `n` singular values, NOT ordered: a cyclic Jacobi orders
    nothing, exactly as the two-sided one does not, and the host tail sorts.
    Every entry is a column norm and is therefore non-negative by
    construction.

    `info_out` is three slots: `[0]` 1 if a sweep performed no rotations,
    `[1]` the sweeps executed, `[2]` the rotations in the last sweep.

    LAUNCH WITH EXACTLY `SVD_TPB` THREADS -- `pinned_block_sum` writes one
    threadgroup slot per thread into a `SVD_TPB`-wide slab.
    """
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var rot = stack_allocation[
        2,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    # `W = I`. In GLOBAL memory, like the shipped solver's basis and for the
    # same reason: an `n x n` threadgroup array is what imposed the
    # `JACOBI_MAX_N = 32` cap this section spent a round removing.
    var idx = tid
    while idx < n * n:
        var rr = idx // n
        var cc = idx - rr * n
        v_out.unsafe_store(idx, Float32(1.0) if rr == cc else Float32(0.0))
        idx += SVD_TPB
    barrier()

    var executed = 0
    var converged = False
    var last_rots = 0

    for _sweep in range(Int(max_sweeps_in)):
        var rots = 0
        for p in range(n):
            for q in range(p + 1, n):
                # The 2x2 Gram of columns p and q, THREE folds over one pass
                # of the two columns. Reading each column once and building
                # all three partials together is a bandwidth choice, not an
                # arithmetic one: each partial is its own serial ascending
                # accumulation and is folded by its own halving tree.
                var lp = Float32(0.0)
                var lq = Float32(0.0)
                var lpq = Float32(0.0)
                var i = tid
                while i < n:
                    var xp = ftz(r.unsafe_load(i * n + p))
                    var xq = ftz(r.unsafe_load(i * n + q))
                    lp = ftz(identical_mul_add(xp, xp, lp))
                    lq = ftz(identical_mul_add(xq, xq, lq))
                    lpq = ftz(identical_mul_add(xp, xq, lpq))
                    i += SVD_TPB
                var app = fold_and_broadcast[SVD_TPB](lp)
                var aqq = fold_and_broadcast[SVD_TPB](lq)
                var apq = fold_and_broadcast[SVD_TPB](lpq)

                # THE TEST. Two square roots and not one, so a column norm
                # near the Float32 ceiling cannot overflow the product; and
                # relative, so the same data in different units converges in
                # the same number of sweeps. Every operand is a broadcast
                # value, so the branch is UNIFORM across the block, which is
                # what makes the barriers inside it legal.
                var np_ = ftz(identical_sqrt(app))
                var nq_ = ftz(identical_sqrt(aqq))
                var thresh = ftz(tol_in * ftz(np_ * nq_))
                var rotate = abs(apq) > thresh
                comptime if wide_rotation:
                    # The declared FTZ squared-norm arithmetic represents
                    # these columns as zero. A nonzero cross product cannot
                    # define a relative angle against a zero norm.
                    rotate = rotate and np_ != Float32(0.0) and nq_ != Float32(0.0)
                if rotate:
                    rots += 1
                    if tid == 0:
                        var cs = jacobi_rotation_cs(app, aqq, apq)
                        comptime if wide_rotation:
                            # Large theta overflows theta^2 in the inherited
                            # rotation. Here t=apq/(aqq-app) is its limiting
                            # value, with error below Float32 rounding once
                            # |theta| > 1e10. This is local to the new route.
                            var delta = ftz(aqq - app)
                            if abs(delta) > abs(apq) * Float32(2e10):
                                var t = ftz(identical_div(apq, delta))
                                cs = SIMD[DType.float32, 2](Float32(1.0), t)
                        rot[0] = cs[0]
                        rot[1] = cs[1]
                    barrier()
                    var c = rot[0]
                    var s = rot[1]

                    # Columns p and q of R. ONE side, where the eigensolver
                    # does two: there is no row update here, and that
                    # absence IS the "one-sided" in the name.
                    var k = tid
                    while k < n:
                        var rkp = ftz(r.unsafe_load(k * n + p))
                        var rkq = ftz(r.unsafe_load(k * n + q))
                        r.unsafe_store(k * n + p, _rot_sub(c, rkp, s, rkq))
                        r.unsafe_store(k * n + q, _rot_add(s, rkp, c, rkq))
                        k += SVD_TPB
                    barrier()

                    # The accumulated right basis.
                    k = tid
                    while k < n:
                        var vkp = ftz(v_out.unsafe_load(k * n + p))
                        var vkq = ftz(v_out.unsafe_load(k * n + q))
                        v_out.unsafe_store(
                            k * n + p, _rot_sub(c, vkp, s, vkq)
                        )
                        v_out.unsafe_store(
                            k * n + q, _rot_add(s, vkp, c, vkq)
                        )
                        k += SVD_TPB
                    barrier()
        executed += 1
        last_rots = rots
        if rots == 0:
            converged = True
            break

    # The singular values are the column norms of the rotated R.
    for j in range(n):
        var acc = Float32(0.0)
        var i2 = tid
        while i2 < n:
            var vv = ftz(r.unsafe_load(i2 * n + j))
            acc = ftz(identical_mul_add(vv, vv, acc))
            i2 += SVD_TPB
        var nrm2 = fold_and_broadcast[SVD_TPB](acc)
        if tid == 0:
            s_out.unsafe_store(j, ftz(identical_sqrt(nrm2)))
    barrier()

    if tid == 0:
        info_out.unsafe_store(0, Float32(1.0) if converged else Float32(0.0))
        info_out.unsafe_store(1, Float32(executed))
        info_out.unsafe_store(2, Float32(last_rots))


def svd_of_r[wide_rotation: Bool = False](
    ctx: DeviceContext,
    mut r: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    mut s: DeviceBuffer[DType.float32],
    n_cols: Int,
    max_sweeps: Int = JACOBI_SWEEPS,
    tolerance: Float32 = Float32(JACOBI_TOL),
) raises:
    """Launch the one-sided Jacobi and REFUSE a non-converged answer.

    Same refusal, same argument and the same wording as
    `eig_and_truncate`'s. `r` is destroyed (its columns become `U * S`).

    `max_sweeps` DEFAULTS to `JACOBI_SWEEPS` and the shipped caller passes
    nothing, so the budget is RAFT's own on every fit. It is a parameter
    only so `check_full_refuses_unconverged` can drive the refusal branch:
    a refusal that has never been shown to fire is a comment, and this one
    is unreachable on ordinary data at fifteen sweeps, which is exactly why
    it cannot be gated without a knob.
    """
    var info = ctx.enqueue_create_buffer[DType.float32](3)
    ctx.synchronize()
    ctx.enqueue_function[one_sided_jacobi_svd_kernel[wide_rotation]](
        r.unsafe_ptr(),
        v.unsafe_ptr(),
        s.unsafe_ptr(),
        info.unsafe_ptr(),
        Int32(n_cols),
        Int32(max_sweeps),
        tolerance,
        grid_dim=(1, 1, 1),
        block_dim=(SVD_TPB, 1, 1),
    )
    var h_info = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_ptr=h_info.unsafe_ptr(), src_buf=info)
    ctx.synchronize()
    if h_info.unsafe_ptr().unsafe_load(0) == Float32(0.0):
        raise Error(
            "the one-sided Jacobi SVD did not converge in "
            + String(max_sweeps)
            + " sweeps at n_cols = "
            + String(n_cols)
            + ": the last sweep still performed "
            + String(h_info.unsafe_ptr().unsafe_load(2))
            + " rotations against a tolerance of "
            + String(tolerance)
            + ". The remedy is more sweeps, the same one cuSOLVER's syevj"
            " has. An unconverged decomposition is not returned as if it"
            " were one; see DEVIATION 590."
        )


def pca_full_validate(n_rows: Int, n_cols: Int, n_components: Int) raises:
    """Shared shape checks; wide IDENTICAL additionally bounds k by rows."""
    pca_validate(n_rows, n_cols, n_components)
    if n_rows < n_cols:
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            if n_components > n_rows:
                raise Error("full SVD n_components cannot exceed min(n_samples, n_features)")
            return
        raise Error(
            "mojolearn PCA(svd_solver='full') needs at least as many samples"
            " as features and got "
            + String(n_rows)
            + " x "
            + String(n_cols)
            + ". The dense route is an R-SVD, which needs a TALL matrix; the"
            " portable route for a wide one is an LQ factorization of the"
            " transpose and it is not written (DEVIATION 593,"
            " decomposition/NOT_IMPLEMENTED.tsv). svd_solver='covariance_eigh'"
            " handles this shape. Silently substituting it here is the"
            " substitution this class refuses to make for the solver name"
            " itself, so this raises instead"
        )


def pca_fit_full(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut r_scratch: DeviceBuffer[DType.float32],
    mut r_out: DeviceBuffer[DType.float32],
    mut v_buf: DeviceBuffer[DType.float32],
    mut s_buf: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_components: Int,
) raises -> PCAResult:
    """`scikit-learn`'s `_fit_full` shape: an SVD of the centered data.

    `x` IS DESTROYED, twice over -- centered in place and then overwritten
    by the QR's packed reflectors. It is NOT restored, which is the one
    place this arm's contract differs from `pca_fit`'s `restore_input`. The
    covariance arm restores because `raft::stats::meanAdd` does
    (`pca.cuh:138`) and because its `x` is the caller's; here the caller is
    `decomposition/estimator.mojo`, which uploads its own copy and drops it,
    and a restore would be an `O(rows)` pass to undo a buffer nobody reads
    again. A caller that needs `x` afterwards must pass a copy, which is
    `geqrf`'s contract and arima's.

    `r_scratch` must hold `qr_slice_count(n_rows, n_cols) * n_cols^2`
    floats; `r_out` and `v_buf` `n_cols^2`; `s_buf` and `mu` `n_cols`.
    """
    pca_full_validate(n_rows, n_cols, n_components)

    # THE SAME TWO KERNELS THE COVARIANCE ARM CALLS, with the same launch
    # geometry, so `mean_` agrees bitwise between the arms and the centering
    # cannot drift between them. `compute_covariance` is not called because
    # its fused split-K arm does not center in place at all (DEVIATION 42):
    # it folds `x - mu` into the Gram's tile read, and there is no Gram here
    # to fold anything into.
    ctx.enqueue_function[column_mean_kernel](
        mu.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    var cells = n_rows * n_cols
    ctx.enqueue_function[shift_columns_kernel](
        x.unsafe_ptr(),
        mu.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        Float32(-1.0),
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    if n_rows < n_cols:
        _pca_wide_basis(ctx, x, r_out, v_buf, s_buf, n_rows, n_cols)
    else:
        _ = qr_factor(ctx, x, r_scratch, r_out, n_rows, n_cols)
        svd_of_r(ctx, r_out, v_buf, s_buf, n_cols)

    # `signFlipKernel` on the RIGHT basis, which is where DEVIATION 525 pins
    # it for both shipped arms: largest-absolute-value entry, ties to the
    # LOWEST index, negate iff that entry is negative. Called here rather
    # than reimplemented so the third arm cannot acquire a fourth
    # convention.
    ctx.enqueue_function[sign_flip_kernel](
        v_buf.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(SIGNFLIP_TPB, 1, 1),
    )
    ctx.synchronize()

    var h_s = ctx.enqueue_create_host_buffer[DType.float32](n_cols)
    var h_v = ctx.enqueue_create_host_buffer[DType.float32](n_cols * n_cols)
    ctx.enqueue_copy(dst_ptr=h_s.unsafe_ptr(), src_buf=s_buf)
    ctx.enqueue_copy(dst_ptr=h_v.unsafe_ptr(), src_buf=v_buf)
    ctx.synchronize()

    # DEVIATION 591: the eigenvalue the SHARED tail expects is the variance
    # along the component, `S^2 / (n_rows - 1)`, which is scikit-learn's
    # `explained_variance_` (`_pca.py::_fit_full`) and cuML's covariance
    # eigenvalue. Squaring here and taking the root again inside the tail
    # costs under one Float64 ulp on a Float32-derived value and buys ONE
    # definition of the ordering, the ratios and `noise_variance_`.
    var denom = Float64(n_rows - 1)
    var diag = List[Float64]()
    for i in range(n_cols):
        var sv = Float64(h_s.unsafe_ptr().unsafe_load(i))
        diag.append(sv * sv / denom)
    var vecs = List[Float64]()
    for i in range(n_cols * n_cols):
        vecs.append(Float64(h_v.unsafe_ptr().unsafe_load(i)))
    return order_truncate_spectrum(
        diag, vecs, n_cols, n_components, n_rows - 1, min(n_rows, n_cols)
    )


def pca_full_scratch_cells(n_rows: Int, n_cols: Int) -> Int:
    """How large `r_scratch` must be. One place, so a caller cannot size it
    from a slice count it computed itself."""
    if n_rows < n_cols:
        return 1  # The wide route owns its compact transposed QR buffers.
    return qr_slice_count(n_rows, n_cols) * n_cols * n_cols



def _wide_transpose_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    rows: Int32, cols: Int32,
):
    var t = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if t < Int(rows) * Int(cols):
        dst.unsafe_store((t % Int(cols)) * Int(rows) + t // Int(cols), src.unsafe_load(t))


def _wide_seed_kernel(
    small_v: MutPointer[Float32, MutAnyOrigin],
    small_r: MutPointer[Float32, MutAnyOrigin],
    basis: MutPointer[Float32, MutAnyOrigin],
    report_r: MutPointer[Float32, MutAnyOrigin],
    singular: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32, n_in: Int32,
):
    var m = Int(m_in)
    var n = Int(n_in)
    var t = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if t < n * n:
        var row = t // n
        var col = t % n
        var v = Float32(0.0)
        var r = Float32(0.0)
        if row < m and col < m:
            v = small_v.unsafe_load(row * m + col)
            r = small_r.unsafe_load(row * m + col)
        basis.unsafe_store(t, v)
        report_r.unsafe_store(t, r)
    if t >= m and t < n:
        singular.unsafe_store(t, Float32(0.0))


def _wide_apply_q_kernel(
    packed: MutPointer[Float32, MutAnyOrigin],
    r: MutPointer[Float32, MutAnyOrigin],
    basis: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32, n_in: Int32,
):
    # X^T=Q R and R^T=U S V^T imply X=U S (Q V)^T.
    # Apply stored reflectors in reverse order to [V;0]. No division by S,
    # so rank deficiency preserves an orthonormal null-space basis as well.
    var m = Int(m_in)
    var n = Int(n_in)
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    for jj in range(m):
        var j = m - 1 - jj
        var rjj = r.unsafe_load(j * m + j)
        if rjj != Float32(0.0):
            var ajj = ftz(packed.unsafe_load(j * m + j))
            var tau = qr_reflector_tau(ajj, abs(rjj), qr_reflector_u1(ajj, rjj))
            var acc = Float32(0.0)
            var row = j + 1 + tid
            while row < n:
                acc = ftz(identical_mul_add(packed.unsafe_load(row * m + j), basis.unsafe_load(row * n + c), acc))
                row += QR_TPB
            # Read before the fold barrier: tid0 must not overwrite the
            # head while another warp is still loading it.
            var head = basis.unsafe_load(j * n + c)
            var tail = fold_and_broadcast[QR_TPB](acc)
            var td = ftz(tau * ftz(head + tail))
            if tid == 0:
                basis.unsafe_store(j * n + c, ftz(head - td))
            row = j + 1 + tid
            while row < n:
                basis.unsafe_store(row * n + c, ftz(identical_mul_add(-td, packed.unsafe_load(row * m + j), basis.unsafe_load(row * n + c))))
                row += QR_TPB
            barrier()


def _pca_wide_basis(
    ctx: DeviceContext,
    mut centered: DeviceBuffer[DType.float32],
    mut report_r: DeviceBuffer[DType.float32],
    mut basis: DeviceBuffer[DType.float32],
    mut singular: DeviceBuffer[DType.float32],
    m: Int, n: Int,
) raises:
    var xt = ctx.enqueue_create_buffer[DType.float32](m * n)
    var r = ctx.enqueue_create_buffer[DType.float32](m * m)
    var rt = ctx.enqueue_create_buffer[DType.float32](m * m)
    var small_v = ctx.enqueue_create_buffer[DType.float32](m * m)
    var scratch = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[_wide_transpose_kernel](centered.unsafe_ptr(), xt.unsafe_ptr(), Int32(m), Int32(n), grid_dim=((m*n+255)//256,1,1), block_dim=(256,1,1))
    # A single panel preserves all reflectors. TSQR's R-only interface does
    # not retain the second-level Q; using it here would silently lose Q.
    _ = qr_factor(ctx, xt, scratch, r, n, m, slices_override=1)
    ctx.enqueue_function[_wide_transpose_kernel](r.unsafe_ptr(), rt.unsafe_ptr(), Int32(m), Int32(m), grid_dim=((m*m+255)//256,1,1), block_dim=(256,1,1))
    svd_of_r[True](ctx, rt, small_v, singular, m, max_sweeps=64, tolerance=Float32(1e-6))
    ctx.enqueue_function[_wide_seed_kernel](small_v.unsafe_ptr(), r.unsafe_ptr(), basis.unsafe_ptr(), report_r.unsafe_ptr(), singular.unsafe_ptr(), Int32(m), Int32(n), grid_dim=((n*n+255)//256,1,1), block_dim=(256,1,1))
    ctx.enqueue_function[_wide_apply_q_kernel](xt.unsafe_ptr(), r.unsafe_ptr(), basis.unsafe_ptr(), Int32(m), Int32(n), grid_dim=(m,1,1), block_dim=(QR_TPB,1,1))
    ctx.synchronize()
    _ = xt^
    _ = r^
    _ = rt^
    _ = small_v^
    _ = scratch^
