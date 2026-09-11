# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Least squares through the normal equations and an eigendecomposition.

FOLLOWS `raft/linalg/detail/lstsq.cuh::lstsqEig` at RAFT `661a3b8`.
Followed statement for statement.

This is cuML's OLS solver `algo = 1` (`cuml/cpp/src/glm/ols.cuh:120`). Their
six steps, copied:

    covA = A^T A                 O(rows * cols^2)
    Ab   = A^T b                 O(rows * cols)
    Q S Q* = eig(covA)           O(cols^3)
    QS   = Q invS                O(cols^2)   with DivideByNonZero
    covA = QS Q^T                O(cols^3)   == inv(A^T A)
    w    = covA Ab               O(cols^2)

**Only the first two touch rows.** That is the same shape as PCA and it is
why this section cost almost nothing: `core/gemm.mojo::gemm_tn` is step 1,
`jacobi_eigh_kernel` is step 3, and `gemm_nt` and `gemv_n`, both of them one
call to a MAX kernel, are steps 5 and 6. The genuinely new code is
`xty_kernel` and the column division.

WHY THEY DEFAULT TO SVD AND NOT TO THIS
---------------------------------------
`olsFit`'s default is `algo = 0`, `lstsqSvdJacobi`. Forming `A^T A` SQUARES
the condition number, so this route loses roughly twice the digits an SVD
route would on an ill-conditioned design. `DivideByNonZero` is the guard: a
direction the data barely constrains appears as a near-zero eigenvalue and
gets DROPPED rather than divided by, which turns the inverse into a
pseudo-inverse.

TWO STEPS THAT ARE NOT IN `lstsqEig` (DEVIATIONS 2620, 2621, 2026-09-11)
------------------------------------------------------------------------
Their guard is an ABSOLUTE `1e-10` against an eigenvalue of `A^T A`, and on
Istella-S (raw columns spanning seven orders of magnitude) this route
returned R^2 -115.6 where scikit-learn gets 0.164. Two settled pieces now
sit around the eigensolver:

    2b  G <- S G S, Ab <- S Ab     S = diag(2^-ceil(e_i / 2)), G_ii = m 2^e_i:
                                   LAPACK xPOEQUB's power-of-radix
                                   equilibration, EXACT in float32
    3b  keep |lam| > n eps32 max|lam|   scipy.linalg.pinvh's default rtol
    6b  w <- S w~                  exact

Both decisions (the scales and the threshold) are made on the HOST from
bits, so every vendor keeps the same directions with the same weights.
They move bits on every design, well conditioned or not, because `S` is not
the identity.

Implementing their non-default solver is a deliberate choice and it is recorded
in `glm/NOT_IMPLEMENTED.tsv`: it is the one that reuses machinery this repository
already has, and their SVD route needs a one-sided Jacobi SVD that does not
exist here yet. The accuracy difference is real and belongs in any
comparison against scikit-learn, whose `LinearRegression` uses LAPACK
`gelsd`, an SVD route.

STEP 6 IS ON THE VENDOR GEMV, AND THE SYMBOL IS `gemv_gpu` NOT `gemv`
---------------------------------------------------------------------
Step 6 is `raft::linalg::gemv` upstream and now calls MAX's gemv here too,
through `core/gemm.mojo::gemv_n`. Which symbol is not a detail. The obvious
one is wrong: `linalg.gemv.gemv` takes no `DeviceContext` and no `target` and
its own docstring opens "Computes a CPU matrix-vector product", so handing it
device pointers would be the `linalg.transpose` failure again. The GPU
counterpart in the same module, `linalg.gemv.gemv_gpu`, is the real mirror of
`raft::linalg::gemv` and is what runs.

`archive/reference/VENDOR_LIBRARIES.md` used to list `linalg.gemv.gemv` as AVAILABLE, where
AVAILABLE meant only that the import compiled, which is exactly how the wrong
symbol got recorded as the finished form. That row and the `linalg.matmul` at
`n = 1` row are corrected. `bench/results/VENDOR_PATH_2026-08-19.md` carries
the same stale sentence and is not this lane's to edit.

**STEP 1 GOES THROUGH `gemm_tn`'s DISPATCH, AND STEP 6 HAS NO SECOND ARM ANY
MORE.** At OLS's shipped shapes (n_cols <= 128) `gemm_tn` takes the split-K
Gram kernel (`core/gram_splitk.mojo`) — the vendor matmul measured ~25
GFLOP/s there, one output tile being its only parallelism — and the
transpose + `linalg.matmul` arm serves larger outputs; `transpose_a` is
still refused and is not used on either arm. Step 6 called
`gemv_gpu` or a implemented RAFT contraction depending on a `use_vendor_gemv` flag,
and the flag and the contraction are both deleted. What checks the vendor call
is `check_ols_beats_truth_on_noise`, which recomputes both residuals on the
host: least squares cannot lose to the planted coefficients on its own sample,
so a wrong step 6 fails it. A host property is a better witness than a second
device kernel and costs no code.

THE STREAM OVERLAP IS NOT IMPLEMENTED
--------------------------------
Theirs computes `A^T A` and `A^T b` on TWO CUDA streams concurrently, with
events to join them (`lstsq.cuh`, `multAbStream`). Mojo's `DeviceContext`
gives one queue here, so ours runs them in sequence. That is a real
throughput deviation and not a correctness one, and it is exactly the kind of
control-plane concurrency `archive/reference/HOST_AND_DEVICE.md` says the incumbents get for
free from CUDA and we do not.

WHAT DEVIATION 527 ADDED HERE, AND WHY IT MOVES NO BITS
-------------------------------------------------------
This file is `glm/impl/`, and under `NUMERIC_FAST`
the shipped bits must not move at all. Two things were added and neither is
arithmetic:

1. **A STAGE CARD.** `lstsq_eig_traced` is the same six steps carrying an
   `IdentityTrace` (`core/identity_trace.mojo`), which hashes RAW BYTES of a
   named buffer after each one. A single final hash says a model moved; a
   card says WHICH STEP moved it first, which is the difference between a
   claim and a diagnosis. `lstsq_eig` is the untraced entry every existing
   caller already had, and it constructs a DISABLED trace, whose every
   `record_*` returns on one boolean test. No launch, no copy, no float.
2. **`elem_tpb`, THE ELEMENTWISE LAUNCH WIDTH.** Steps 4 and the
   eigenvalue extraction launch one thread per OUTPUT CELL
   (`divide_columns_by_nonzero_kernel`, `diagonal_to_vector_kernel`: `idx =
   block_idx.x * block_dim.x + thread_idx.x`, one store per `idx`, no fold,
   no atomic, no cross-thread combination). The block width therefore moves
   WHICH thread computes a cell and never what any cell is computed from --
   `numerics.mojo`'s own definition of a SCHEDULING row. It exists so that
   `check_ols_is_launch_invariant` can vary a launch geometry that must not
   matter and require the coefficient BYTES not to move. The default is the
   256 that was hardcoded here before, so the shipped launch is unchanged.

   It is deliberately NOT threaded into the Gram product, the `xty` fold or
   the Jacobi block: those three block widths ARE fold widths (rows 20/21),
   they are pinned at the matrix and in `pinned_block_sum`, and handing a
   caller a knob onto them would be handing a caller a knob onto a summation
   order.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast

from cluster.checks.reduce_by_key import copy_f32_kernel
from core.gemm import gemm_nt, gemm_tn, gemv_n
from core.column_stats import (
    STATS_TPB,
    diagonal_to_vector_kernel,
    divide_columns_by_nonzero_kernel,
    xty_kernel,
)
from core.identity_trace import IdentityTrace
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_TPB,
    jacobi_eigh_kernel,
)
from glm.impl.matrix.math import (
    matrix_vector_binary_mult_kernel,
    row_vector_binary_mult_kernel,
    vector_binary_mult_kernel,
)


#: The elementwise launch width steps 3b and 4 used to hardcode. SCHEDULING,
#: and provably so: each thread owns one output cell, so this number cannot
#: reach the value of any cell. See the module docstring.
comptime OLS_ELEM_TPB = 256

#: `DivideByNonZero`'s ABSOLUTE threshold, 1e-10 (`lstsq.cuh:105`).
#:
#: **NO LONGER ON THE TALL ROUTE (2026-09-11).** An absolute 1e-10 against an
#: eigenvalue of `A^T A`, which scales with the square of the data, gave the
#: same design rank 8 at scale 1 and rank 0 at scale 2^-24, and on Istella-S
#: (raw columns spanning seven orders of magnitude) an R^2 of -115.6 where
#: scikit-learn gets 0.164. `lstsq_eig` now equilibrates and cuts relative
#: (`ols_equilibration_scale`, `ols_pinv_threshold`, DEVIATIONS 2620, 2621).
#: `lstsq_min_norm` left it too (DEVIATION 2622, the same equilibration and
#: relative cutoff on `A A^T`); nothing on either route cuts at it now.
comptime OLS_NONZERO_THRESH = Float32(1.0e-10)

#: `numpy.finfo(numpy.float32).eps`, 2^-23 exactly.
comptime OLS_PINV_EPS32 = Float64(1.1920928955078125e-07)


def ols_pinv_threshold(max_abs_eig: Float32, n: Int) -> Float32:
    """DEVIATION 2621. The eigenvalue cutoff of a Hermitian pseudo-inverse:
    an eigenvalue `lam` of the `n x n` matrix is kept when
    `|lam| > n * eps32 * max|lam|`.

    SETTLED, NOT INVENTED. `scipy.linalg.pinvh` defaults `rtol` to `N * eps`
    of the dtype and keeps `abs(s) > atol + rtol * max(abs(s))`;
    `torch.linalg.pinv(hermitian=True)` and `numpy.linalg.matrix_rank` cut
    at the same place. An eigenvalue of a float32 Gram matrix below it is
    rounding, not data, and dividing by it turns rounding into a coefficient.

    Host arithmetic: one exact integer-times-power-of-two product, one
    Float64 rounding, one narrowing. The threshold is a function of the
    eigenvalue bits alone on every machine.
    """
    return Float32(Float64(n) * OLS_PINV_EPS32 * Float64(max_abs_eig))


def ols_equilibration_scale(diag: Float32) -> Float32:
    """DEVIATION 2620. The power-of-two scale `s_i` for one diagonal entry
    `G_ii` of the Gram matrix, chosen so that `s_i^2 G_ii` lies in [0.5, 2).

    SETTLED, NOT INVENTED. LAPACK `xPOEQUB` equilibrates a symmetric positive
    definite matrix by `S = diag(1 / sqrt(A_ii))` restricted to powers of the
    radix, so that the scaling introduces no rounding, and `xPOSVX` solves
    the equilibrated system. By van der Sluis (1969) that diagonal scaling is
    within a factor `n` of the best condition number any diagonal scaling
    reaches, and the power-of-two restriction costs at most a further 4. On
    the columns of the design it is what cuML's `normalize=True` does
    (`preprocess.cuh`), up to the centering the Python layer already did.

    ONE ROUNDING CHOICE DIFFERS FROM `xPOEQUB`, ON PURPOSE. Theirs is
    `INT(-0.5 * log2(A_ii))`, truncation toward zero. Ours writes
    `G_ii = m * 2^e` with `m` in [1, 2) and uses `2^-ceil(e / 2)`, which is
    exactly shift-invariant: the same design in units scaled by `2^j` gets
    scales multiplied by exactly `2^-j` and a BIT-IDENTICAL equilibrated
    matrix (`check_ols_rank_guard_is_scale_invariant`). Truncation toward
    zero loses that across `e = 0`.

    Integer arithmetic on the bits and no floating-point operation, so every
    machine computes the same scale. A zero, negative or non-finite diagonal
    gets scale 1: there is nothing to equilibrate, and a non-finite input
    keeps propagating rather than being masked.
    """
    var bits = bitcast[DType.uint32](diag)
    var field = Int((bits >> 23) & UInt32(0xFF))
    var frac = bits & UInt32(0x7FFFFF)
    if (bits >> 31) != UInt32(0) or field == 255:
        return Float32(1.0)
    if field == 0 and frac == UInt32(0):
        return Float32(1.0)
    var e = field - 127
    if field == 0:
        # Subnormal: value = frac * 2^-149, so e = floor(log2(frac)) - 149.
        var width = 0
        var f = frac
        while f != UInt32(0):
            f = f >> 1
            width += 1
        e = (width - 1) - 149
    var k = (e + 1) // 2 if e >= 0 else -((-e) // 2)
    return bitcast[DType.float32](UInt32(127 - k) << 23)


def lstsq_eig(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut cov_a: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut qs: DeviceBuffer[DType.float32],
    mut s_vec: DeviceBuffer[DType.float32],
    mut ab: DeviceBuffer[DType.float32],
    mut inv: DeviceBuffer[DType.float32],
    mut a_alias: DeviceBuffer[DType.float32],
    mut a_alias2: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    elem_tpb: Int = OLS_ELEM_TPB,
) raises:
    """`w = inv(A^T A) A^T b`, their step order. The untraced entry.

    Every caller this file ever had lands here. It constructs a DISABLED
    `IdentityTrace` and calls `lstsq_eig_traced`, so there is exactly one
    implementation of the six steps and the traced runs certify the shipped
    path rather than a copy of it. A disabled trace's `record_*` returns on
    one boolean test: no launch, no copy, no arithmetic, no bits moved.
    """
    var off = IdentityTrace.disabled()
    lstsq_eig_traced(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        n_rows, n_cols, elem_tpb, off,
    )


def lstsq_eig_traced(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut cov_a: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut qs: DeviceBuffer[DType.float32],
    mut s_vec: DeviceBuffer[DType.float32],
    mut ab: DeviceBuffer[DType.float32],
    mut inv: DeviceBuffer[DType.float32],
    mut a_alias: DeviceBuffer[DType.float32],
    mut a_alias2: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    elem_tpb: Int,
    mut trace: IdentityTrace,
) raises:
    """`w = inv(A^T A) A^T b`, their step order, with a stage card.

    Steps 5 and 6 are on MAX's tuned kernels, which is what their cuBLAS
    calls are; step 1 goes through `gemm_tn`, which dispatches the shipped
    small-output Gram shapes to the split-K kernel (see the module
    docstring). There is no second implementation selected by any flag; the
    old `use_vendor_gemv` flag and its contraction are gone.

    THE CARD, DEVIATION 527. One record after each of the six steps, plus
    the eigensolver's `info` and the RANK. Tags name a position in the
    algorithm and carry no machine number, per `core/identity_trace.mojo`
    rule 2, so two vendors' cards align. The rank record is the one that is
    not a rounding: `divide_columns_by_nonzero_kernel` DROPS a direction
    whose eigenvalue is at or below `ols_pinv_threshold` (DEVIATION 2621), so a last-bit
    move in step 3 can change how many directions the pseudo-inverse keeps
    -- a DISCRETE output of a float comparison, and an integer stage the
    differ reads before it reads any float one.
    """
    # covA <- A^T A. `raft::linalg::gemm(CUBLAS_OP_T, CUBLAS_OP_N, alpha=1)`
    # — a Gram matrix, not a covariance, so no scale. This is the ONLY
    # step here that touches rows, and it was still on the hand-written
    # contraction while every other section had moved: OLS sat at 28 ms
    # across five benchmark rounds because nothing I changed was on its path.
    gemm_tn(ctx, cov_a, a, a_alias, a_alias2, n_cols, n_cols, n_rows)

    # Ab <- A^T b. Theirs overlaps this with the line above on a second
    # stream; see the module docstring.
    ctx.enqueue_function[xty_kernel](
        ab.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    ctx.synchronize()

    # THE CARD'S FIRST TWO DEVICE STAGES. `cov_a` is recorded HERE and not
    # later because the Jacobi below CONSUMES it in place -- after the sweep
    # the buffer holds the eigenvalues on its diagonal and rotated rubble
    # off it, so a record taken after the launch would hash a different
    # object under a step-1 name.
    trace.record_device[DType.float32](
        ctx, "ols.step1.covA", cov_a, n_cols * n_cols
    )
    trace.record_device[DType.float32](ctx, "ols.step2.Ab", ab, n_cols)

    # STEP 2b, DEVIATION 2620: EQUILIBRATE BEFORE THE EIGENSOLVER.
    #
    # `G_ii = m 2^e` gives `s_i = 2^-ceil(e / 2)`, computed on the host from
    # the bits (`ols_equilibration_scale`); then `G <- S G S` and
    # `Ab <- S Ab` on the device. Every multiply is by a power of two, so it
    # is EXACT: no rounding is added and no vendor can round it differently.
    # Step 6b multiplies the solution back, `w = S w~`, also exactly.
    #
    # WHY IT IS HERE, MEASURED (DigitalOcean MI325X, 2026-09-11, Istella-S
    # 2,043,304 x 220, tools/ols_illconditioned_probe.py). The raw Gram's
    # largest eigenvalue is 4.75e19, so the Jacobi's float32 `||G||_F^2`
    # overflows to inf, its stopping test `2 off <= tol^2 ||G||_F^2` holds
    # before the first rotation, and the eigendecomposition it returns is the
    # diagonal: every coefficient became `Ab_i / G_ii` and R^2 was -115.6 (a
    # float32 emulation of that Jacobi does 0 sweeps and gives -115.8).
    # Without the overflow (50,000 rows) the same global test stops after 2
    # sweeps with column pairs at relative correlation 0.82 unrotated.
    # Equilibrated, the diagonal lies in [0.5, 2), nothing overflows, and the
    # test does its job (12 sweeps at full shape).
    #
    # The card recorded `step1.covA` and `step2.Ab` just above, BEFORE this
    # step. The scales are a pure function of `step1.covA`'s bits, so they
    # need no stage of their own and the card keeps its 11 stages.
    var scale = ctx.enqueue_create_buffer[DType.float32](n_cols)
    ctx.enqueue_function[diagonal_to_vector_kernel](
        s_vec.unsafe_ptr(),
        cov_a.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=((n_cols + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    var h_diag = ctx.enqueue_create_host_buffer[DType.float32](n_cols)
    ctx.enqueue_copy(dst_ptr=h_diag.unsafe_ptr(), src_buf=s_vec)
    ctx.synchronize()
    var h_scale = ctx.enqueue_create_host_buffer[DType.float32](n_cols)
    ctx.synchronize()
    for i in range(n_cols):
        h_scale.unsafe_ptr().unsafe_store(
            i, ols_equilibration_scale(h_diag.unsafe_ptr().unsafe_load(i))
        )
    ctx.enqueue_copy(dst_buf=scale, src_ptr=h_scale.unsafe_ptr())
    ctx.synchronize()
    # `[[mojo-buffer-freed-at-last-use]]`: a host buffer is dead at its last
    # `.unsafe_ptr()` unless something uses it after the synchronize.
    _ = h_diag^
    _ = h_scale^
    var cells = n_cols * n_cols
    ctx.enqueue_function[row_vector_binary_mult_kernel](
        cov_a.unsafe_ptr(),
        scale.unsafe_ptr(),
        Int32(n_cols),
        Int32(n_cols),
        grid_dim=((cells + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.enqueue_function[matrix_vector_binary_mult_kernel](
        cov_a.unsafe_ptr(),
        scale.unsafe_ptr(),
        Int32(n_cols),
        Int32(n_cols),
        grid_dim=((cells + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.enqueue_function[vector_binary_mult_kernel](
        ab.unsafe_ptr(),
        scale.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=((n_cols + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.synchronize()

    # Q S Q* <- covA. Jacobi consumes covA and leaves S on its diagonal.
    #
    # `info` is the eigensolver's convergence report and it is not optional:
    # `eigDC`, the arm `lstsqEig` reaches (`lstsq.cuh:315`), aborts on a
    # non-zero `dev_info` (`raft/linalg/detail/eig.cuh:149-151`; the
    # identically worded ASSERT at `:79-81` belongs to `eigDC_legacy`, which
    # nothing calls). An OLS built on an
    # unconverged eigendecomposition of `A^T A` is a wrong answer with no
    # error. `tol` and `sweeps` are `raft::linalg::eigJacobi`'s own defaults
    # (`raft/linalg/eig.cuh:108-109`); they used to be a hardcoded 80 and
    # 1e-10, and the 1e-10 was an ABSOLUTE test on a quantity that scales
    # with the square of the data.
    var info_buf = ctx.enqueue_create_buffer[DType.float32](3)
    ctx.synchronize()
    ctx.enqueue_function[jacobi_eigh_kernel](
        cov_a.unsafe_ptr(),
        q.unsafe_ptr(),
        info_buf.unsafe_ptr(),
        Int32(n_cols),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
    )
    ctx.enqueue_function[diagonal_to_vector_kernel](
        s_vec.unsafe_ptr(),
        cov_a.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=((n_cols + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    var h_info = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_ptr=h_info.unsafe_ptr(), src_buf=info_buf)
    ctx.synchronize()
    # `h_info` is alive across the copy TODAY only because the reads below sit
    # after the synchronize. It is the one host buffer on this path without an
    # explicit guard; step 3b's `h_eig` below carries one, and the trap is
    # `[[mojo-buffer-freed-at-last-use]]`. If the convergence branch is ever made
    # conditional or deleted, this buffer's last use becomes the enqueue and
    # the copy writes freed host memory. Hygiene, not the OLS defect.
    _ = h_info
    if h_info.unsafe_ptr().unsafe_load(0) == Float32(0.0):
        raise Error(
            "lstsq_eig: the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps on the "
            + String(n_cols)
            + " x "
            + String(n_cols)
            + " Gram matrix; ||offdiag(A^T A)||_F / ||A^T A||_F is still "
            + String(h_info.unsafe_ptr().unsafe_load(1))
            + ". A rank-deficient or badly scaled design produces this."
        )

    # THE EIGENDECOMPOSITION'S THREE STAGES. `info` is recorded as a stage
    # of its own because slot 2 is the SWEEP COUNT, and a sweep count is a
    # discrete quantity that a last-bit disagreement in the convergence fold
    # can move (`jacobi_eigh_device.mojo` DEVIATION BLOCK 3). Two vendors
    # whose cards first differ HERE differed about how much work to do, not
    # about how to round it, and the differ should say so.
    trace.record_device[DType.float32](ctx, "ols.step3.eigvals", s_vec, n_cols)
    trace.record_device[DType.float32](
        ctx, "ols.step3.eigvecs", q, n_cols * n_cols
    )
    trace.record_device[DType.float32](ctx, "ols.step3.info", info_buf, 3)
    # STEP 3b, DEVIATION 2621: THE CUTOFF IS RELATIVE TO THE LARGEST
    # EIGENVALUE (`ols_pinv_threshold`, scipy.linalg.pinvh's rtol). The host
    # reads the eigenvalues back, takes max|lam| exactly and hands the kernel
    # one threshold. The predicate below is the kernel's, character for
    # character, so the card's rank is the rank the kernel uses; a NaN
    # eigenvalue fails both comparisons and counts as DROPPED, as it does
    # in the kernel.
    var h_eig = ctx.enqueue_create_host_buffer[DType.float32](n_cols)
    ctx.enqueue_copy(dst_ptr=h_eig.unsafe_ptr(), src_buf=s_vec)
    ctx.synchronize()
    var max_abs = Float32(0.0)
    for i in range(n_cols):
        var mag = abs(h_eig.unsafe_ptr().unsafe_load(i))
        if mag > max_abs:
            max_abs = mag
    var thresh = ols_pinv_threshold(max_abs, n_cols)
    var kept = 0
    for i in range(n_cols):
        var lam = h_eig.unsafe_ptr().unsafe_load(i)
        if lam > thresh or lam < -thresh:
            kept += 1
    _ = h_eig^
    if trace.enabled:
        var one = List[Int32]()
        one.append(Int32(kept))
        trace.record_list_i32("ols.step4.rank", one)

    # QS <- Q invS, with DivideByNonZero.
    ctx.enqueue_function[divide_columns_by_nonzero_kernel](
        qs.unsafe_ptr(),
        q.unsafe_ptr(),
        s_vec.unsafe_ptr(),
        Int32(n_cols),
        thresh,
        grid_dim=((n_cols * n_cols + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    trace.record_device[DType.float32](
        ctx, "ols.step4.QS", qs, n_cols * n_cols
    )
    # inv <- QS Q^T == Q invS Q^T == inv(A^T A)
    gemm_nt(ctx, inv, qs, q, n_cols, n_cols, n_cols)
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, "ols.step5.inv", inv, n_cols * n_cols
    )

    # w <- inv Ab.
    #
    # RAFT does not call gemm here: it calls `raft::linalg::gemv`
    # (`lstsq.cuh`, "w <- covA Ab"), because a matrix against ONE vector is a
    # different BLAS routine with a different tuning. Expressing it as a
    # matmul with `n = 1` produced zeros for some coefficients, which is what
    # sent me to read their line again.
    #
    # THE OBVIOUS SWAP IS THE WRONG SYMBOL. `linalg.gemv.gemv` is HOST-ONLY.
    # Checked, not assumed: its signature is
    # `gemv[parallelize: Bool, elementwise_lambda_fn](c_buf, a_buf, b_buf)`
    # with **no `ctx: DeviceContext` and no `target`**, and its own docstring
    # opens "Computes a CPU matrix-vector product". It is the same tell that
    # caught `nn.cumsum` in `archive/reference/VENDOR_LIBRARIES.md`: the GPU-capable calls in
    # this toolchain (`matmul`, `top_k`, `argsort`, `gather`) all carry a
    # context and this one does not. Recorded so nobody re-derives it.
    #
    # THE SYMBOL THAT IS RIGHT is `linalg.gemv.gemv_gpu`, same module, wrapped
    # as `core/gemm.mojo::gemv_n`. The orientation below is READ OFF
    # `max/kernels/src/linalg/gemv.mojo` at tag `max/v26.5.0` (the toolchain
    # pinned here is max 26.5.0), not inferred from the fact that it compiles:
    #
    #   * `gemv_gpu` derives its dimensions from C and A ONLY. It calls
    #     `GemmShape.get`, which returns `(c.dim[0], c.dim[1], a.dim[1])` and
    #     documents that B is skipped because B may be pre-packed. So with
    #     c = `w` shaped `(n_cols, 1)` and a = `inv` shaped
    #     `(n_cols, n_cols)`, it sees m = n_cols, n = 1, k = n_cols.
    #   * `n == 1` with a float32 A selects `GEMVAlgorithm.GEMV_KERNEL`, whose
    #     body is `accum += a[global_warp_id * k + idx] * b[idx]` over
    #     `idx < k`, then `c[global_warp_id] = accum`, one warp per output
    #     row. That IS `w[i] = sum_j inv[i][j] * ab[j]` for row-major `inv`,
    #     which is the product this step wants.
    #   * `transpose_b` stays FALSE. This is the part worth not guessing: the
    #     `transpose_b == True` arm of `gemv_gpu_dispatch` SWAPS a and b and
    #     passes `(n, m, k)`, so at n = 1 it would launch one warp and write
    #     one coefficient instead of `n_cols` of them.
    #   * `ab` is shaped `(n_cols, 1)` and not `(1, n_cols)`. `GEMV_KERNEL`
    #     reads B linearly so both give the same answer THERE, but the
    #     `MATMUL_NAIVE` fallback in the same dispatcher indexes B as
    #     `(k, n)`. `(n_cols, 1)` is correct under both arms.
    #   * `pdl_level` keeps its `PDLLevel.ON` default. PDL is Hopper-only and
    #     gated on `_SUPPORT_PDL_LAUNCH`, which is
    #     `has_nvidia_gpu_accelerator() and compute >= H100`, so on Metal the
    #     grid-dependency barriers compile out and `pdl_launch_attributes`
    #     returns an empty list. Nothing Apple-specific is being relied on.
    #
    # THE IMPLEMENTED CONTRACTION IS GONE. It stood here as a second arm behind
    # `use_vendor_gemv=False`, on the argument that a vendor call needs
    # something to be checked against. The check that matters is
    # `glm/checks/ols_check.mojo::check_ols_beats_truth_on_noise`, which
    # recomputes both residuals on the HOST and fails if the fitted
    # coefficients lose to the planted ones. A wrong step 6 cannot pass that,
    # and it needs no second device kernel. Keeping one meant carrying a
    # hand-written GEMM to check a tuned GEMV a host property already covers.
    gemv_n(ctx, w, inv, ab, n_cols, n_cols)
    # STEP 6b, DEVIATION 2620: w = S w~, one exact multiply per coefficient.
    ctx.enqueue_function[vector_binary_mult_kernel](
        w.unsafe_ptr(),
        scale.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=((n_cols + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, "ols.step6.coef", w, n_cols)
    _ = scale^

