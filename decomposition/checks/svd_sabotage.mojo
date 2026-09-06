# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The R-SVD's kernels with their decisions BROKEN ON PURPOSE, for the check.

NOT A PORT, NOT REACHED by any driver, by `decomposition/estimator.mojo` or
by the identity card. One copy of `core/householder_qr.mojo::qr_panel_kernel`
and one of
`decomposition/impl/linalg/detail/svd_full.mojo::one_sided_jacobi_svd_kernel`,
carrying arms that `decomposition/checks/svd_full_check.mojo` selects through
the `arm` argument and that nothing else can reach. At `SVD_SAB_NONE` the
host dispatchers below call the SHIPPED code by name, so the gates exercise
the shipped kernels and not a second copy of them, and no shipped bit depends
on this file.

Same construction, and for the same reason, as
`decomposition/checks/pca_sabotage.mojo` (DEVIATION 585),
`cholesky/checks/chol_sabotage.mojo` and `hierarchy/checks/sabotage_tile.mojo`:
a sabotage arm does not belong in a production kernel, and a sabotage that
requires editing source cannot be run by an orchestrator that is forbidden to
edit source.

DEVIATION 592.

WHY EVERY ARM EXISTS. A gate that has never been shown capable of failing
does not count in this tree, so each decision the two kernels make gets an
arm that is a PLAUSIBLE way to make that decision wrongly, and the check
records what the arm must move and what it actually moved.

`SVD_SAB_QR_SIGN`       takes `s = +sign(a_jj)` instead of `-sign(a_jj)`, so
                        `u1 = a_jj - s*normx` becomes a DIFFERENCE OF
                        NEAR-EQUAL NUMBERS whenever the column is already
                        nearly axis-aligned. This is DEVIATION 678's chosen
                        sign, `archive/research/arima/SABOTAGES.md` arm (l), and the arm that
                        proves the sign is load bearing rather than
                        cosmetic. It is NOT expected to fail on a
                        well-conditioned fixture -- the reflector is still
                        orthogonal in exact arithmetic -- so the gate that
                        uses it is comparative and runs on a
                        deliberately ill-conditioned one:
                        `check_full_reflector_sign_earns_its_place` requires
                        the shipped sign to be STRICTLY better on every
                        fixture, which is the same shape as arima's
                        `check_qr_beats_normal_equations_on_ill_conditioning`.
`SVD_SAB_QR_NO_SCALE`   never divides the reflector tail by `u1`, so `w` is
                        not the unit-first-entry vector `tau` was computed
                        for and `H` is not orthogonal at all. This is the
                        shape of "I read LAPACK's packed layout and forgot
                        that `w_0 = 1` is IMPLICIT". `R^T R` stops equalling
                        `A^T A` immediately, so
                        `check_qr_gram_matches_float64` MUST fail, and it
                        must fail by orders of magnitude rather than in the
                        last bit.
`SVD_SAB_QR_RANK_TEST`  reinstates DEVIATION 678's refusal: a zero column
                        norm returns `info = j + 1` and the factorization
                        stops. This is the arm for DEVIATION 588 -- it is
                        the RIGHT behaviour for a routine that back
                        substitutes and the WRONG behaviour for one that
                        feeds an SVD -- and
                        `check_full_survives_a_constant_column` requires the
                        shipped path to accept a centered constant column
                        with a zero singular value while this arm refuses
                        it. Without this arm the deviation is a paragraph;
                        with it, it is a measurement.
`SVD_SAB_QR_NO_FTZ`     drops the flushes on the column-norm partial.
                        EXPECTED INERT ON APPLE and on any column that
                        flushes denormals in hardware, so the check asserts
                        nothing about the cell count for this arm and
                        RECORDS it instead. It moves on a denormal-honoring
                        column, which is the column the pin exists for. Same
                        expectation and same treatment as
                        `PCA_SAB_WHITEN_NO_FTZ`.
`SVD_SAB_JAC_ABS_TOL`   compares `|apq|` against `tol` itself rather than
                        against `tol * sqrt(app) * sqrt(aqq)`. This is
                        EXACTLY the defect DEVIATION BLOCK 1 of the shipped
                        eigensolver had to be repaired for -- an absolute
                        test on a quantity that scales with the square of
                        the data -- so the arm is not invented, it is the
                        bug this tree already paid for once.
                        `check_full_scale_invariance` MUST see the sweep
                        count move between `X` and `1000 * X` on this arm
                        and MUST see it stay put on the shipped one.
`SVD_SAB_JAC_NO_V`      rotates `R` and forgets to rotate `W`, so the
                        singular values are right and the components are the
                        identity. This is the arm that proves the
                        reconstruction gate is testing the BASIS and not
                        just the spectrum: `V^T V = I` still passes (the
                        identity is orthonormal), the singular values still
                        match Float64, and
                        `check_full_truncated_reconstruction_is_optimal`
                        fails. A gate suite without it would pass a PCA that
                        returned the coordinate axes.

`SVD_SAB_QR_DROP_SLICE` has no kernel arm. It is a HOST arm inside
`qr_factor_sab`: TSQR's second pass is launched over `(ns - 1) * n` rows
instead of `ns * n`, so the last slice's `R` is silently left out of the
reduction. It exists because the reduction is the one part of DEVIATION 589
that a correct per-slice factorization cannot vouch for, and because
"forgot the last tile" is the classic way to break a two-pass reduction.
`check_tsqr_agrees_with_one_block` runs it and the Gram error MUST move.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation

from checks.numerics import (
    ftz,
    identical_div,
    identical_mul_add,
    identical_sqrt,
)
from core.householder_qr import (
    QR_TPB,
    fold_and_broadcast,
    qr_factor,
    qr_panel_kernel,
    qr_reflector_r,
    qr_reflector_tau,
    qr_reflector_u1,
    qr_slice_count,
)
from decomposition.checks.jacobi_eigh_device import (
    _rot_add,
    _rot_sub,
    jacobi_rotation_cs,
)
from decomposition.impl.linalg.detail.svd_full import (
    SVD_TPB,
    one_sided_jacobi_svd_kernel,
)


comptime SVD_SAB_NONE = 0
comptime SVD_SAB_QR_SIGN = 1
comptime SVD_SAB_QR_NO_SCALE = 2
comptime SVD_SAB_QR_RANK_TEST = 3
comptime SVD_SAB_QR_NO_FTZ = 4
comptime SVD_SAB_JAC_ABS_TOL = 5
comptime SVD_SAB_JAC_NO_V = 6
comptime SVD_SAB_QR_DROP_SLICE = 7

#: DEVIATION 678's `LS_RANK_TOL`, restated here rather than imported so this
#: file does not create a `decomposition/` -> `arima/` dependency for the
#: sake of one constant in a sabotage arm. The arm only needs the SHAPE of
#: their refusal, which is "a zero or tiny diagonal stops the factorization".
comptime SAB_RANK_TOL = Float32(1.0e-5)


@always_inline
def _sab_sign(ajj: Float32, arm: Int32) -> Float32:
    """`SVD_SAB_QR_SIGN` flips the reflector's sign; every other arm keeps
    DEVIATION 586's."""
    if Int(arm) == SVD_SAB_QR_SIGN:
        return Float32(1.0) if ajj >= Float32(0.0) else Float32(-1.0)
    return Float32(-1.0) if ajj >= Float32(0.0) else Float32(1.0)


def qr_panel_kernel_sab(
    a: MutPointer[Float32, MutAnyOrigin],
    r_out: MutPointer[Float32, MutAnyOrigin],
    info_out: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    lda_in: Int32,
    n_slices_in: Int32,
    arm: Int32,
):
    """`qr_panel_kernel` with the arms above. `info_out[block]` is 0 unless
    `SVD_SAB_QR_RANK_TEST` refused, in which case it is `j + 1`, which is
    arima's own return convention."""
    var m = Int(m_in)
    var n = Int(n_in)
    var lda = Int(lda_in)
    var n_slices = Int(n_slices_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var rb = (b * m) // n_slices
    var re = ((b + 1) * m) // n_slices
    var ms = re - rb
    var rbase = b * n * n
    if tid == 0:
        info_out.unsafe_store(b, Int32(0))

    for j in range(n):
        var acc = Float32(0.0)
        var i = j + tid
        while i < ms:
            if Int(arm) == SVD_SAB_QR_NO_FTZ:
                var vraw = a.unsafe_load((rb + i) * lda + j)
                acc = identical_mul_add(vraw, vraw, acc)
            else:
                var v = ftz(a.unsafe_load((rb + i) * lda + j))
                acc = ftz(identical_mul_add(v, v, acc))
            i += QR_TPB
        var sigma = fold_and_broadcast[QR_TPB](acc)
        var normx = ftz(identical_sqrt(sigma))
        var ajj = ftz(a.unsafe_load((rb + j) * lda + j))

        if normx == Float32(0.0):
            if Int(arm) == SVD_SAB_QR_RANK_TEST:
                # arima's branch: refuse, and stop. Uniform across the block
                # because `normx` came out of a broadcast fold.
                if tid == 0:
                    info_out.unsafe_store(b, Int32(j + 1))
                return
            if tid == 0:
                r_out.unsafe_store(rbase + j * n + j, Float32(0.0))
            barrier()
        else:
            var s = _sab_sign(ajj, arm)
            var r_jj = ftz(s * normx)
            var u1 = ftz(ajj - r_jj)
            var tau = ftz(identical_div(ftz(ftz(-s) * u1), normx))
            if tid == 0:
                r_out.unsafe_store(rbase + j * n + j, r_jj)
            if u1 == Float32(0.0):
                # Only reachable on the flipped-sign arm, where `u1` is a
                # cancellation and CAN be exactly zero. Leaving the column
                # alone is the least-wrong thing this arm can do; the gate
                # it feeds is comparative, not a crash test.
                barrier()
            else:
                if Int(arm) != SVD_SAB_QR_NO_SCALE:
                    var i2 = j + 1 + tid
                    while i2 < ms:
                        var cur = ftz(a.unsafe_load((rb + i2) * lda + j))
                        a.unsafe_store(
                            (rb + i2) * lda + j, ftz(identical_div(cur, u1))
                        )
                        i2 += QR_TPB
                barrier()
                for c in range(j + 1, n):
                    var dacc = Float32(0.0)
                    var i3 = j + 1 + tid
                    while i3 < ms:
                        var w = ftz(a.unsafe_load((rb + i3) * lda + j))
                        var x = ftz(a.unsafe_load((rb + i3) * lda + c))
                        dacc = ftz(identical_mul_add(w, x, dacc))
                        i3 += QR_TPB
                    var tail = fold_and_broadcast[QR_TPB](dacc)
                    var ajc = ftz(a.unsafe_load((rb + j) * lda + c))
                    var total = ftz(ajc + tail)
                    var td = ftz(tau * total)
                    if tid == 0:
                        a.unsafe_store((rb + j) * lda + c, ftz(ajc - td))
                    var i4 = j + 1 + tid
                    while i4 < ms:
                        var w2 = ftz(a.unsafe_load((rb + i4) * lda + j))
                        var cur2 = ftz(a.unsafe_load((rb + i4) * lda + c))
                        a.unsafe_store(
                            (rb + i4) * lda + c,
                            ftz(identical_mul_add(-td, w2, cur2)),
                        )
                        i4 += QR_TPB
                    barrier()

    barrier()
    var t = tid
    while t < n * n:
        var rr = t // n
        var cc = t - rr * n
        if cc > rr:
            if rr < ms:
                r_out.unsafe_store(
                    rbase + t, ftz(a.unsafe_load((rb + rr) * lda + cc))
                )
            else:
                r_out.unsafe_store(rbase + t, Float32(0.0))
        elif cc < rr:
            r_out.unsafe_store(rbase + t, Float32(0.0))
        t += QR_TPB


def one_sided_jacobi_svd_kernel_sab(
    r: MutPointer[Float32, MutAnyOrigin],
    v_out: MutPointer[Float32, MutAnyOrigin],
    s_out: MutPointer[Float32, MutAnyOrigin],
    info_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    max_sweeps_in: Int32,
    tol_in: Float32,
    arm: Int32,
):
    """`one_sided_jacobi_svd_kernel` with `SVD_SAB_JAC_ABS_TOL` and
    `SVD_SAB_JAC_NO_V`."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var rot = stack_allocation[
        2,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

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
                var thresh = Float32(0.0)
                if Int(arm) == SVD_SAB_JAC_ABS_TOL:
                    thresh = tol_in
                else:
                    var np_ = ftz(identical_sqrt(app))
                    var nq_ = ftz(identical_sqrt(aqq))
                    thresh = ftz(tol_in * ftz(np_ * nq_))
                if abs(apq) > thresh:
                    rots += 1
                    if tid == 0:
                        var cs = jacobi_rotation_cs(app, aqq, apq)
                        rot[0] = cs[0]
                        rot[1] = cs[1]
                    barrier()
                    var c = rot[0]
                    var s = rot[1]
                    var k = tid
                    while k < n:
                        var rkp = ftz(r.unsafe_load(k * n + p))
                        var rkq = ftz(r.unsafe_load(k * n + q))
                        r.unsafe_store(k * n + p, _rot_sub(c, rkp, s, rkq))
                        r.unsafe_store(k * n + q, _rot_add(s, rkp, c, rkq))
                        k += SVD_TPB
                    barrier()
                    if Int(arm) != SVD_SAB_JAC_NO_V:
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


def qr_factor_sab(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut r_scratch: DeviceBuffer[DType.float32],
    mut r_out: DeviceBuffer[DType.float32],
    mut info: DeviceBuffer[DType.int32],
    n_rows: Int,
    n_cols: Int,
    arm: Int,
    slices_override: Int = 0,
) raises -> Int:
    """`qr_factor` with an arm. AT `SVD_SAB_NONE` THIS CALLS THE SHIPPED
    `qr_factor`, so every gate below runs against the shipped kernel."""
    var ns = qr_slice_count(n_rows, n_cols) if slices_override <= 0 else slices_override
    if arm == SVD_SAB_NONE:
        return qr_factor(ctx, a, r_scratch, r_out, n_rows, n_cols, ns)
    if arm == SVD_SAB_QR_DROP_SLICE:
        # The kernel is the SHIPPED one on both passes; what is sabotaged is
        # the REDUCTION, by telling the second pass the stack is one slice
        # shorter than it is. Meaningless at `ns == 1`, so the caller must
        # force a multi-slice run, and the gate does.
        if ns < 2:
            raise Error(
                "SVD_SAB_QR_DROP_SLICE needs at least two slices; pass"
                " slices_override"
            )
        ctx.enqueue_function[qr_panel_kernel](
            a.unsafe_ptr(), r_scratch.unsafe_ptr(),
            Int32(n_rows), Int32(n_cols), Int32(n_cols), Int32(ns),
            grid_dim=(ns, 1, 1), block_dim=(QR_TPB, 1, 1),
        )
        ctx.enqueue_function[qr_panel_kernel](
            r_scratch.unsafe_ptr(), r_out.unsafe_ptr(),
            Int32((ns - 1) * n_cols), Int32(n_cols), Int32(n_cols), Int32(1),
            grid_dim=(1, 1, 1), block_dim=(QR_TPB, 1, 1),
        )
        ctx.synchronize()
        return ns
    if ns == 1:
        ctx.enqueue_function[qr_panel_kernel_sab](
            a.unsafe_ptr(), r_out.unsafe_ptr(), info.unsafe_ptr(),
            Int32(n_rows), Int32(n_cols), Int32(n_cols), Int32(1), Int32(arm),
            grid_dim=(1, 1, 1), block_dim=(QR_TPB, 1, 1),
        )
        ctx.synchronize()
        return 1
    ctx.enqueue_function[qr_panel_kernel_sab](
        a.unsafe_ptr(), r_scratch.unsafe_ptr(), info.unsafe_ptr(),
        Int32(n_rows), Int32(n_cols), Int32(n_cols), Int32(ns), Int32(arm),
        grid_dim=(ns, 1, 1), block_dim=(QR_TPB, 1, 1),
    )
    ctx.enqueue_function[qr_panel_kernel_sab](
        r_scratch.unsafe_ptr(), r_out.unsafe_ptr(), info.unsafe_ptr(),
        Int32(ns * n_cols), Int32(n_cols), Int32(n_cols), Int32(1), Int32(arm),
        grid_dim=(1, 1, 1), block_dim=(QR_TPB, 1, 1),
    )
    ctx.synchronize()
    return ns


def svd_of_r_sab(
    ctx: DeviceContext,
    mut r: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    mut s: DeviceBuffer[DType.float32],
    mut info: DeviceBuffer[DType.float32],
    n_cols: Int,
    max_sweeps: Int,
    tol: Float32,
    arm: Int,
) raises:
    """The one-sided Jacobi with an arm, and WITHOUT the refusal, so a gate
    can inspect a non-converged run instead of catching an exception.

    AT `SVD_SAB_NONE` THIS LAUNCHES THE SHIPPED KERNEL. `max_sweeps` and
    `tol` are parameters here and constants at the shipped call site, which
    is what lets `check_full_refuses_unconverged` and
    `check_full_scale_invariance` reach the sweep count at all."""
    if arm == SVD_SAB_NONE:
        ctx.enqueue_function[one_sided_jacobi_svd_kernel](
            r.unsafe_ptr(), v.unsafe_ptr(), s.unsafe_ptr(), info.unsafe_ptr(),
            Int32(n_cols), Int32(max_sweeps), tol,
            grid_dim=(1, 1, 1), block_dim=(SVD_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[one_sided_jacobi_svd_kernel_sab](
            r.unsafe_ptr(), v.unsafe_ptr(), s.unsafe_ptr(), info.unsafe_ptr(),
            Int32(n_cols), Int32(max_sweeps), tol, Int32(arm),
            grid_dim=(1, 1, 1), block_dim=(SVD_TPB, 1, 1),
        )
    ctx.synchronize()
