# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Gaussian-process kernels with their pins BROKEN ON PURPOSE.

NOT A PORT, NOT REACHED by any driver, by `gaussian_process/estimator.mojo`
or by the card. Copies of `gaussian_process/mojo_only/kernels.mojo`'s leaf,
combine, mean and variance kernels, each carrying arms that
`gp_check.mojo` selects through the `sabotage` argument threaded down from
`gp_kernel_matrix`, `gpr_fit_host` and `gpr_predict_host`, and that nothing
else can reach. The un-sabotaged arm of every kernel HERE is never launched:
the drivers call the real kernel when `sabotage == GP_SAB_NONE` and one of
these otherwise, so the shipped bits never depend on this file.

Same construction as `cholesky/mojo_only/chol_sabotage.mojo`, and for the
same reason: a sabotage arm does not belong in a production kernel, and a
sabotage that requires editing source cannot be run by an orchestrator that
is forbidden to edit source. **No source edit and no rebuild is required for
any arm below.**

DEVIATION 1769.

THE ARMS, and what each is a plausible way to get wrong
--------------------------------------------------------
`GP_SAB_DIST_DESCENDING`    the feature axis of the squared distance walks
                            `f` DESCENDING. Same multiset of products, a
                            different summation order. This is the classic
                            transcription slip: the caller's loop happens to
                            run backwards and no tolerance test can see it.
`GP_SAB_STD_EXP`            `std.math.exp` on the kernel value instead of
                            `identical_exp`. A device `exp` is a VENDOR
                            CHOICE in its last bit (IDENTITY_PATHS row 12)
                            and EVERY cell of a GP Gram matrix goes through
                            one, so this is the single widest seam in the
                            lane.
`GP_SAB_EXPANDED_RBF`       the squared distance becomes `||x||^2 + ||y||^2
                            - 2 x.y` instead of `sum_f (x_f - y_f)^2`. This
                            is the arithmetic `svm/ported/distance/
                            kernel_matrices.mojo::rbf_kernel_expanded_kernel`
                            ships, correctly, for an SVM. It is the WRONG
                            arithmetic for a Gaussian process and DEVIATION
                            1755 is the argument. Catastrophic cancellation
                            for near points is what a Gram matrix is made of.
`GP_SAB_NO_FTZ_KERNEL`      `ftz` dropped at the kernel-matrix seam. Metal
                            flushes in hardware, so this is expected INERT on
                            Apple and live on a denormal-honoring column.
                            RECORDED, never claimed.
`GP_SAB_ALGEBRA_REASSOCIATE` a Sum or Product node combines `b op a` instead
                            of `a op b`. Float addition and multiplication
                            are exactly commutative away from NaN payloads
                            and zero signs, so this arm is expected to be
                            LARGELY INERT and to move only where a `-0.0` or
                            a NaN reaches a node. It is here precisely
                            because a reader will assume the opposite, and
                            what it establishes is which of the two it is.
`GP_SAB_VDOTV_PAIRWISE`     the predictive variance's `v^T v` folds PAIRWISE
                            instead of ascending serial. Same multiset,
                            different order, and a fold shape is exactly what
                            IDENTITY_PATHS row 21 is about.
`GP_SAB_NO_CLAMP`           the negative-variance clamp is dropped, so a
                            variance comes back negative and `sqrt` of it
                            comes back NaN. This is the defect the clamp
                            exists to stop and it must be visible.
`GP_SAB_CLAMP_UNCOUNTED`    the clamp still fires but the per-test-point flag
                            is never written. The output is unchanged and the
                            RECORD of it is gone, which is the failure mode
                            DEVIATION 1760 exists to forbid: a GP that clamps
                            silently is a GP that lies quietly.
`GP_SAB_MEAN_DESCENDING`    DRIVER arm. The posterior mean is computed by a
                            hand-written kernel folding the training axis
                            DESCENDING instead of by `identical_gemm_into`,
                            so the mean leaves profile
                            `mojolearn.identical.gemm.fp32.v1` entirely.
`GP_SAB_LOGDET_RECOMPUTED`  HOST DRIVER arm. `log |K|` is recomputed as
                            `log(prod_j L_jj^2)` instead of being taken from
                            the factor `cholesky_logdet_host` already
                            produced. It is the spelling a reader reaches for
                            and it underflows to zero on any factor with
                            diagonal entries below one, which every
                            correlation-shaped factor has.
`GP_SAB_YALPHA_DESCENDING`  HOST DRIVER arm. `y^T alpha` folds DESCENDING.

The three DRIVER arms have no kernel here except `sabotage_mean_kernel`;
`GP_SAB_LOGDET_RECOMPUTED` and `GP_SAB_YALPHA_DESCENDING` are host branches
in `gaussian_process/estimator.mojo` taken only when `sabotage` names them.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp
from std.memory import bitcast

from kde.ported.distance.distance_ops import l2_unexp_core
from mojo_only.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)


#: SABOTAGE-ONLY. The register-stack depth `GP_SAB_VDOTV_PAIRWISE` folds in.
#: At module scope rather than inside the kernel because a `comptime for`
#: needs a static bound and a static bound belongs where a reader can see it.
comptime GP_SAB_FOLD_LEVELS = 24


#: The production path. Every driver default.
comptime GP_SAB_NONE = 0
comptime GP_SAB_DIST_DESCENDING = 1
comptime GP_SAB_STD_EXP = 2
comptime GP_SAB_EXPANDED_RBF = 3
comptime GP_SAB_NO_FTZ_KERNEL = 4
comptime GP_SAB_ALGEBRA_REASSOCIATE = 5
comptime GP_SAB_VDOTV_PAIRWISE = 6
comptime GP_SAB_NO_CLAMP = 7
comptime GP_SAB_CLAMP_UNCOUNTED = 8
comptime GP_SAB_MEAN_DESCENDING = 9
comptime GP_SAB_LOGDET_RECOMPUTED = 10
comptime GP_SAB_YALPHA_DESCENDING = 11
comptime GP_SAB_COUNT = 12


def gp_sabotage_name(sab: Int) -> String:
    """The arm's name, for the check's banner and for an error message."""
    if sab == GP_SAB_NONE:
        return String("NONE")
    if sab == GP_SAB_DIST_DESCENDING:
        return String("DIST_DESCENDING")
    if sab == GP_SAB_STD_EXP:
        return String("STD_EXP")
    if sab == GP_SAB_EXPANDED_RBF:
        return String("EXPANDED_RBF")
    if sab == GP_SAB_NO_FTZ_KERNEL:
        return String("NO_FTZ_KERNEL")
    if sab == GP_SAB_ALGEBRA_REASSOCIATE:
        return String("ALGEBRA_REASSOCIATE")
    if sab == GP_SAB_VDOTV_PAIRWISE:
        return String("VDOTV_PAIRWISE")
    if sab == GP_SAB_NO_CLAMP:
        return String("NO_CLAMP")
    if sab == GP_SAB_CLAMP_UNCOUNTED:
        return String("CLAMP_UNCOUNTED")
    if sab == GP_SAB_MEAN_DESCENDING:
        return String("MEAN_DESCENDING")
    if sab == GP_SAB_LOGDET_RECOMPUTED:
        return String("LOGDET_RECOMPUTED")
    if sab == GP_SAB_YALPHA_DESCENDING:
        return String("YALPHA_DESCENDING")
    return String("UNKNOWN")


def gp_sabotage_touches_kernel_matrix(sab: Int) -> Bool:
    """True when the arm changes a LEAF or COMBINE kernel, so the driver
    launches the copy in this file instead of the production one."""
    if sab == GP_SAB_DIST_DESCENDING:
        return True
    if sab == GP_SAB_STD_EXP:
        return True
    if sab == GP_SAB_EXPANDED_RBF:
        return True
    if sab == GP_SAB_NO_FTZ_KERNEL:
        return True
    if sab == GP_SAB_ALGEBRA_REASSOCIATE:
        return True
    return False


def gp_sabotage_touches_variance(sab: Int) -> Bool:
    """True when the arm changes the predictive-variance kernel."""
    if sab == GP_SAB_VDOTV_PAIRWISE:
        return True
    if sab == GP_SAB_NO_CLAMP:
        return True
    if sab == GP_SAB_CLAMP_UNCOUNTED:
        return True
    return False


# ===========================================================================
# THE SABOTAGED DISTANCE
# ===========================================================================


def sabotage_scaled_sqdist(
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    ls: MutPointer[Float32, MutAnyOrigin],
    i: Int,
    j: Int,
    d: Int,
    ls_len: Int,
    sab: Int,
) -> Float32:
    """`kernels.mojo::gp_scaled_sqdist` with the two distance arms in it.

    The un-sabotaged path here is never reached: `gp_kernel_matrix` calls the
    real helper when `sabotage == GP_SAB_NONE`.
    """
    if sab == GP_SAB_EXPANDED_RBF:
        # `||x||^2 + ||y||^2 - 2 x.y`, the expanded identity. Every one of
        # the three sums is an ascending serial chain here, so on the
        # DIAGONAL the three coincide exactly and `s` is exactly `+0.0`;
        # this arm therefore moves NOTHING on the diagonal and everything
        # off it. DEVIATION 1755 states what that does and does not prove.
        var nx = Float32(0.0)
        var ny = Float32(0.0)
        var dot = Float32(0.0)
        for f in range(d):
            var li = f
            if ls_len == 1:
                li = 0
            var lv = ftz(ls.unsafe_load(li))
            var xv = ftz(identical_div(ftz(x.unsafe_load(i * d + f)), lv))
            var yv = ftz(identical_div(ftz(y.unsafe_load(j * d + f)), lv))
            nx = ftz(identical_mul_add(xv, xv, nx))
            ny = ftz(identical_mul_add(yv, yv, ny))
            dot = ftz(identical_mul_add(xv, yv, dot))
        return ftz(ftz(nx + ny) - ftz(Float32(2.0) * dot))

    var acc = Float32(0.0)
    if sab == GP_SAB_DIST_DESCENDING:
        for ff in range(d):
            var f = d - 1 - ff
            var li = f
            if ls_len == 1:
                li = 0
            var lv = ftz(ls.unsafe_load(li))
            var xv = ftz(identical_div(ftz(x.unsafe_load(i * d + f)), lv))
            var yv = ftz(identical_div(ftz(y.unsafe_load(j * d + f)), lv))
            acc = l2_unexp_core(acc, xv, yv)
        return ftz(acc)

    if sab == GP_SAB_NO_FTZ_KERNEL:
        # No `ftz` at any seam of the distance. On an FTZ backend this is
        # bitwise a no-op; on a denormal-honoring one it keeps subnormal
        # coordinates and subnormal partial sums that IDENTICAL flushes.
        for f in range(d):
            var li = f
            if ls_len == 1:
                li = 0
            var lv = ls.unsafe_load(li)
            var xv = identical_div(x.unsafe_load(i * d + f), lv)
            var yv = identical_div(y.unsafe_load(j * d + f), lv)
            var diff = xv - yv
            acc = identical_mul_add(diff, diff, acc)
        return acc

    for f in range(d):
        var li = f
        if ls_len == 1:
            li = 0
        var lv = ftz(ls.unsafe_load(li))
        var xv = ftz(identical_div(ftz(x.unsafe_load(i * d + f)), lv))
        var yv = ftz(identical_div(ftz(y.unsafe_load(j * d + f)), lv))
        acc = l2_unexp_core(acc, xv, yv)
    return ftz(acc)


def sabotage_rbf_kernel(
    out_k: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    ls: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    d_in: Int32,
    ls_len_in: Int32,
    sab_in: Int32,
):
    """`kernels.mojo::gp_rbf_kernel` with the distance and `exp` arms."""
    var m = Int(m_in)
    var n = Int(n_in)
    var d = Int(d_in)
    var ls_len = Int(ls_len_in)
    var sab = Int(sab_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m * n:
        return
    var i = t // n
    var j = t - i * n
    var d2 = sabotage_scaled_sqdist(x, y, ls, i, j, d, ls_len, sab)
    if sab == GP_SAB_NO_FTZ_KERNEL:
        out_k.unsafe_store(t, identical_exp(Float32(-0.5) * d2))
        return
    var e = ftz(identical_mul(Float32(-0.5), d2))
    if sab == GP_SAB_STD_EXP:
        out_k.unsafe_store(t, ftz(exp(e)))
        return
    out_k.unsafe_store(t, ftz(identical_exp(e)))


def sabotage_matern_kernel(
    out_k: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    ls: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    d_in: Int32,
    ls_len_in: Int32,
    nu_sel_in: Int32,
    sqrt3: Float32,
    sqrt5: Float32,
    sab_in: Int32,
):
    """`kernels.mojo::gp_matern_kernel` with the distance and `exp` arms."""
    var m = Int(m_in)
    var n = Int(n_in)
    var d = Int(d_in)
    var ls_len = Int(ls_len_in)
    var nu_sel = Int(nu_sel_in)
    var sab = Int(sab_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m * n:
        return
    var i = t // n
    var j = t - i * n
    var d2 = sabotage_scaled_sqdist(x, y, ls, i, j, d, ls_len, sab)
    var dist = ftz(identical_sqrt(d2))

    if nu_sel == 1:
        var s3v = ftz(identical_mul(dist, sqrt3))
        var pre3 = ftz(Float32(1.0) + s3v)
        if sab == GP_SAB_STD_EXP:
            out_k.unsafe_store(t, ftz(identical_mul(pre3, ftz(exp(-s3v)))))
            return
        out_k.unsafe_store(
            t, ftz(identical_mul(pre3, ftz(identical_exp(-s3v))))
        )
        return
    if nu_sel == 2:
        var s5v = ftz(identical_mul(dist, sqrt5))
        var ss = ftz(identical_mul(s5v, s5v))
        var third = ftz(identical_div(ss, Float32(3.0)))
        var pre5 = ftz(ftz(Float32(1.0) + s5v) + third)
        if sab == GP_SAB_STD_EXP:
            out_k.unsafe_store(t, ftz(identical_mul(pre5, ftz(exp(-s5v)))))
            return
        out_k.unsafe_store(
            t, ftz(identical_mul(pre5, ftz(identical_exp(-s5v))))
        )
        return
    if sab == GP_SAB_STD_EXP:
        out_k.unsafe_store(t, ftz(exp(-dist)))
        return
    out_k.unsafe_store(t, ftz(identical_exp(-dist)))


def sabotage_combine_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    mn_in: Int32,
    is_prod_in: Int32,
    sab_in: Int32,
):
    """`kernels.mojo::gp_combine_kernel` with the operand order reversed.

    `GP_SAB_ALGEBRA_REASSOCIATE`. Expected LARGELY INERT: float `+` and `*`
    are exactly commutative except on NaN payloads and on the sign of a zero
    sum. The arm exists so the check can say WHICH, rather than a reader
    assuming.
    """
    var mn = Int(mn_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= mn:
        return
    var av = ftz(a.unsafe_load(t))
    var bv = ftz(b.unsafe_load(t))
    if Int(sab_in) == GP_SAB_ALGEBRA_REASSOCIATE:
        if Int(is_prod_in) != 0:
            a.unsafe_store(t, ftz(identical_mul(bv, av)))
        else:
            a.unsafe_store(t, ftz(bv + av))
        return
    if Int(is_prod_in) != 0:
        a.unsafe_store(t, ftz(identical_mul(av, bv)))
    else:
        a.unsafe_store(t, ftz(av + bv))


def sabotage_variance_kernel(
    var_out: MutPointer[Float32, MutAnyOrigin],
    std_out: MutPointer[Float32, MutAnyOrigin],
    clamped: MutPointer[Int32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_train_in: Int32,
    n_star_in: Int32,
    kss: Float32,
    sab_in: Int32,
):
    """`kernels.mojo::gp_variance_kernel` with its three arms.

    `GP_SAB_VDOTV_PAIRWISE` folds the column of `v` pairwise in a small
    register array instead of ascending serial. `GP_SAB_NO_CLAMP` stores the
    raw variance. `GP_SAB_CLAMP_UNCOUNTED` clamps and stores a zero flag.
    """
    var n_train = Int(n_train_in)
    var n_star = Int(n_star_in)
    var sab = Int(sab_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n_star:
        return

    var acc = Float32(0.0)
    if sab == GP_SAB_VDOTV_PAIRWISE:
        # A register stack merged whenever a level is occupied: the shape
        # `gemm_identical.mojo::_fold_push` uses, transcribed with a
        # `comptime for` over static levels so the slot index is never a
        # runtime value. A different bracketing of the same multiset.
        var stack = SIMD[DType.float32, 32](0.0)
        var occ = 0
        for i in range(n_train):
            var vv = ftz(v.unsafe_load(i * n_star + t))
            var carry = ftz(identical_mul(vv, vv))
            var placed = False
            comptime for lvl in range(GP_SAB_FOLD_LEVELS):
                if not placed:
                    if ((occ >> lvl) & 1) == 1:
                        carry = ftz(ftz(stack[lvl]) + ftz(carry))
                        occ = occ - (1 << lvl)
                    else:
                        stack[lvl] = carry
                        occ = occ + (1 << lvl)
                        placed = True
        comptime for lvl in range(GP_SAB_FOLD_LEVELS):
            if ((occ >> lvl) & 1) == 1:
                acc = ftz(acc + ftz(stack[lvl]))
    else:
        for i in range(n_train):
            var vv = ftz(v.unsafe_load(i * n_star + t))
            acc = ftz(identical_mul_add(vv, vv, acc))

    var raw = ftz(ftz(kss) - acc)
    if sab == GP_SAB_NO_CLAMP:
        var_out.unsafe_store(t, raw)
        clamped.unsafe_store(t, Int32(0))
        std_out.unsafe_store(t, ftz(identical_sqrt(raw)))
        return
    var outv = raw
    if not (raw > Float32(0.0)):
        outv = Float32(0.0)
    var moved = bitcast[DType.uint32](outv) != bitcast[DType.uint32](raw)
    var_out.unsafe_store(t, outv)
    if sab == GP_SAB_CLAMP_UNCOUNTED:
        clamped.unsafe_store(t, Int32(0))
    else:
        clamped.unsafe_store(t, Int32(1) if moved else Int32(0))
    std_out.unsafe_store(t, ftz(identical_sqrt(outv)))


def sabotage_mean_kernel(
    mean: MutPointer[Float32, MutAnyOrigin],
    kcross: MutPointer[Float32, MutAnyOrigin],
    dual: MutPointer[Float32, MutAnyOrigin],
    n_train_in: Int32,
    n_star_in: Int32,
):
    """`GP_SAB_MEAN_DESCENDING`. The posterior mean by hand, folding the
    training axis DESCENDING, so the mean leaves profile
    `mojolearn.identical.gemm.fp32.v1` altogether.

    `kcross` is `n_train x n_star` row-major, which is the orientation
    `trsm_lower` wants; the production path reads the same buffer through
    `identical_gemm_into` at `OP_TN`.
    """
    var n_train = Int(n_train_in)
    var n_star = Int(n_star_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n_star:
        return
    var acc = Float32(0.0)
    for ii in range(n_train):
        var i = n_train - 1 - ii
        acc = ftz(
            identical_mul_add(
                ftz(kcross.unsafe_load(i * n_star + t)),
                ftz(dual.unsafe_load(i)),
                acc,
            )
        )
    mean.unsafe_store(t, ftz(acc))
