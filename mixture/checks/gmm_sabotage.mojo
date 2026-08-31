# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The mixture kernels with their pins BROKEN ON PURPOSE, for the check.

NOT A PORT, NOT REACHED by any driver, by `mixture/estimator.mojo` or by the
card. Copies of `mixture/checks/estep.mojo`'s and `mstep.mojo`'s kernels,
each carrying arms that `gmm_check.mojo` selects through the `sabotage`
argument threaded down from `gmm_e_step` / `gmm_m_step` and that nothing else
can reach. The un-sabotaged arm of every kernel HERE is never launched: the
drivers call the real kernel when `sabotage == GMM_SAB_NONE` and one of these
otherwise, so the shipped bits never depend on this file.

Same construction as `cholesky/checks/chol_sabotage.mojo` and
`hierarchy/checks/sabotage_tile.mojo`, and for the same reason: a sabotage
arm does not belong in a production kernel, and **a sabotage that requires
editing source cannot be run by an orchestrator that is forbidden to edit
source.** That is the house standard now.

DEVIATION 1740.

THE ARMS, and what each is a plausible way to get wrong
--------------------------------------------------------
`GMM_SAB_TOL_ULP`          DRIVER arm, and **the most important one in this
                           lane**. The value compared against `tol` is moved
                           by ONE ULP toward zero before the comparison. It
                           changes no parameter and no probability; it
                           changes only whether the loop stops, and therefore
                           the ITERATION COUNT and every number after it.
                           This is the arm that shows the convergence test is
                           the whole ballgame, and it is what
                           `check_iteration_count_is_identical` exists to
                           see.
`GMM_SAB_LSE_DESCENDING`   the logsumexp's sum walks the components
                           DESCENDING. Same multiset, different order.
`GMM_SAB_LSE_ROTATE`       the logsumexp's sum starts at
                           `k0 = block_idx.x % K` and wraps, so the order is
                           a function of LAUNCH GEOMETRY. At one block this
                           is INERT, which is the point: only the launch gate
                           can see it.
`GMM_SAB_ROWMAX_GE`        the row max becomes `>=`, so the HIGHER index wins
                           a tie. Row 39: on a row carrying `-0.0` and
                           `+0.0` this returns the other zero.
`GMM_SAB_ROWMAX_HARDWARE`  the row max becomes the hardware `max(acc, v)`
                           instead of the positional strict `>`. `max(-0,+0)`
                           is the SECOND operand on Apple and the LARGER on
                           NVIDIA and AMD (IDENTITY_PATHS row 39, measured on
                           all three), so this arm is the one that can
                           genuinely disagree across vendors.
`GMM_SAB_MAHAL_DESCENDING` the Mahalanobis fold walks the feature axis
                           DESCENDING.
`GMM_SAB_NK_DESCENDING`    the `nk` fold walks the sample axis DESCENDING.
                           `nk` divides the mean and the covariance, so this
                           moves the whole M-step.
`GMM_SAB_MEANLL_PAIRWISE`  the mean log likelihood folds PAIRWISE instead of
                           ascending serial. The mean log likelihood is the
                           CONVERGENCE QUANTITY, so this is a second, subtler
                           way to move the iteration count.
`GMM_SAB_VENDOR_MATMUL`    DRIVER arm. The E-step's `X . P` calls
                           `core/gemm.mojo::gemm_nt` (MAX `linalg.matmul`, a
                           CLOSED library whose k-split is a per-vendor
                           summation order) instead of
                           `identical_gemm_into`.
`GMM_SAB_COLLAPSE_RESET`   DRIVER arm. A component whose Cholesky returns
                           `info != 0` is silently RESET to the identity
                           covariance instead of refusing. This is what a
                           library does when it wants its fit never to fail,
                           and it is exactly the outcome DEVIATION 1723
                           forbids: the fit then succeeds on a vendor whose
                           pivot was positive and succeeds DIFFERENTLY on one
                           whose pivot was not.
`GMM_SAB_LOGDET_FROM_DIAG` DRIVER plus kernel arm. `log_det_chol` is computed
                           as scikit-learn spells it, `sum_j log(P[j][j])`
                           over the precision Cholesky's own diagonal,
                           instead of `-0.5 * chol_logdet(L)`. Both are the
                           same quantity; they are not the same bits, and
                           DEVIATION 1726 is the choice between them.
`GMM_SAB_NO_FTZ_RESP`      `ftz` dropped on `resp = exp(log_resp)`. A
                           responsibility is an exponential of a large
                           negative number, so SUBNORMALS ARE WHERE THEY
                           LIVE: this is the one place in the lane a
                           denormal-honoring column and a flushing one
                           routinely see different values. Expected INERT on
                           Apple (Metal flushes in hardware).
`GMM_SAB_COV_PRESCALE`     the division by `nk[k]` is applied to the
                           responsibility BEFORE the outer-product
                           accumulation instead of to the accumulated matrix
                           after it. Mathematically identical, arithmetically
                           not, and it is the spelling an implementation
                           reaches for when it wants one fewer kernel.

The four DRIVER arms have no kernel in this file; they are branches in
`estep.mojo`'s and `mstep.mojo`'s drivers taken only when `sabotage` names
them, exactly as `cholesky/checks/potrf.mojo` leaves its three driver arms
on the production path.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import max as hardware_max
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
    identical_mul_add,
)


#: The production path. Every driver default.
comptime GMM_SAB_NONE = 0
comptime GMM_SAB_TOL_ULP = 1
comptime GMM_SAB_LSE_DESCENDING = 2
comptime GMM_SAB_LSE_ROTATE = 3
comptime GMM_SAB_ROWMAX_GE = 4
comptime GMM_SAB_ROWMAX_HARDWARE = 5
comptime GMM_SAB_MAHAL_DESCENDING = 6
comptime GMM_SAB_NK_DESCENDING = 7
comptime GMM_SAB_MEANLL_PAIRWISE = 8
comptime GMM_SAB_VENDOR_MATMUL = 9
comptime GMM_SAB_COLLAPSE_RESET = 10
comptime GMM_SAB_LOGDET_FROM_DIAG = 11
comptime GMM_SAB_NO_FTZ_RESP = 12
comptime GMM_SAB_COV_PRESCALE = 13
comptime GMM_SAB_COUNT = 14

#: Depth of `sabotage_meanll_kernel`'s equal-rank merge stack. Only the
#: PAIRWISE arm uses it. `log2(n) + 1` entries suffice and 40 is far above
#: `log2` of any `n` a float32 sum means anything over.
comptime GMM_STACK_MAX = 40


def gmm_sabotage_name(sab: Int) -> String:
    """The arm's name, for the check's banner and for an error message."""
    if sab == GMM_SAB_NONE:
        return String("NONE")
    if sab == GMM_SAB_TOL_ULP:
        return String("TOL_ULP")
    if sab == GMM_SAB_LSE_DESCENDING:
        return String("LSE_DESCENDING")
    if sab == GMM_SAB_LSE_ROTATE:
        return String("LSE_ROTATE")
    if sab == GMM_SAB_ROWMAX_GE:
        return String("ROWMAX_GE")
    if sab == GMM_SAB_ROWMAX_HARDWARE:
        return String("ROWMAX_HARDWARE")
    if sab == GMM_SAB_MAHAL_DESCENDING:
        return String("MAHAL_DESCENDING")
    if sab == GMM_SAB_NK_DESCENDING:
        return String("NK_DESCENDING")
    if sab == GMM_SAB_MEANLL_PAIRWISE:
        return String("MEANLL_PAIRWISE")
    if sab == GMM_SAB_VENDOR_MATMUL:
        return String("VENDOR_MATMUL")
    if sab == GMM_SAB_COLLAPSE_RESET:
        return String("COLLAPSE_RESET")
    if sab == GMM_SAB_LOGDET_FROM_DIAG:
        return String("LOGDET_FROM_DIAG")
    if sab == GMM_SAB_NO_FTZ_RESP:
        return String("NO_FTZ_RESP")
    if sab == GMM_SAB_COV_PRESCALE:
        return String("COV_PRESCALE")
    return String("UNKNOWN")


def gmm_sabotage_is_driver_arm(sab: Int) -> Bool:
    """True when the arm is a BRANCH IN A DRIVER rather than a kernel here.

    The drivers use this to decide which kernel to launch, so the production
    kernel is reached at every driver arm and a driver arm can never be
    mistaken for a kernel one.
    """
    if sab == GMM_SAB_TOL_ULP:
        return True
    if sab == GMM_SAB_VENDOR_MATMUL:
        return True
    if sab == GMM_SAB_COLLAPSE_RESET:
        return True
    return False


def gmm_ulp_down(v: Float32) -> Float32:
    """`v` moved ONE ULP toward zero, by its bits. `GMM_SAB_TOL_ULP`'s whole
    mechanism, on the HOST, where the convergence comparison lives.

    A magnitude of exactly zero is returned unchanged: there is no float
    below it in magnitude, and a run whose change is exactly zero has
    converged on any spelling of the test. Anything else has its magnitude
    field decremented, which for a positive normal is exactly the next
    representable value down.

    This is the smallest possible perturbation of the convergence test. If
    the iteration count does not move under it on ANY fixture, then either
    no fixture in this lane ever stops within one ulp of the threshold -- in
    which case the gate is not exercising the boundary it claims to -- or
    the count is not being read back from the run.
    """
    var u = bitcast[DType.uint32](v)
    if (u & UInt32(0x7FFFFFFF)) == UInt32(0):
        return v
    return bitcast[DType.float32](u - UInt32(1))


# ===========================================================================
# THE E-STEP KERNELS, SABOTAGED
# ===========================================================================


def sabotage_mahal_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    murow: MutPointer[Float32, MutAnyOrigin],
    mahal: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    kcomp_in: Int32,
    ncomp_in: Int32,
    sabotage_in: Int32,
):
    """`estep.mojo::mahal_kernel` with the DESCENDING arm.

    Everything not named by an arm is character for character the production
    kernel, so a failure names the arm and nothing else.
    """
    var n = Int(n_in)
    var d = Int(d_in)
    var kc = Int(kcomp_in)
    var ncomp = Int(ncomp_in)
    var sab = Int(sabotage_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var acc = Float32(0.0)
    if sab == GMM_SAB_MAHAL_DESCENDING:
        for jj in range(d):
            var j = d - 1 - jj
            var t = ftz(ftz(y.unsafe_load(i * d + j)) - ftz(murow.unsafe_load(j)))
            acc = ftz(identical_mul_add(t, t, acc))
    else:
        for j in range(d):
            var t = ftz(ftz(y.unsafe_load(i * d + j)) - ftz(murow.unsafe_load(j)))
            acc = ftz(identical_mul_add(t, t, acc))
    mahal.unsafe_store(i * ncomp + kc, acc)


def sabotage_logsumexp_kernel(
    wlp: MutPointer[Float32, MutAnyOrigin],
    lse: MutPointer[Float32, MutAnyOrigin],
    rowmax: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    ncomp_in: Int32,
    sabotage_in: Int32,
):
    """`estep.mojo::logsumexp_kernel` with FOUR arms: the two row-max ones
    (row 39) and the two summation-order ones.

    `LSE_ROTATE` reads `block_idx.x`, which is what makes it a LAUNCH
    GEOMETRY arm: at one block it is inert and only a launch sweep can see
    it.
    """
    var n = Int(n_in)
    var ncomp = Int(ncomp_in)
    var sab = Int(sabotage_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var base = i * ncomp

    var max_exp = wlp.unsafe_load(base)
    for k in range(1, ncomp):
        var v = wlp.unsafe_load(base + k)
        if sab == GMM_SAB_ROWMAX_HARDWARE:
            # THE HARDWARE MAX, accumulator first. Measured on all three
            # vendors (IDENTITY_PATHS row 39): `max(+0.0, -0.0)` is `-0.0`
            # on Apple (the second operand) and `+0.0` on NVIDIA and AMD
            # (IEEE-2019 maximum), so this arm is the one that can genuinely
            # disagree ACROSS VENDORS rather than merely against the oracle.
            max_exp = hardware_max(max_exp, v)
        elif sab == GMM_SAB_ROWMAX_GE:
            if v >= max_exp:
                max_exp = v
        else:
            if v > max_exp:
                max_exp = v
    rowmax.unsafe_store(i, max_exp)

    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    if max_exp == neg_inf:
        lse.unsafe_store(i, max_exp)
        return

    var s = Float32(0.0)
    if sab == GMM_SAB_LSE_DESCENDING:
        for kk in range(ncomp):
            var k = ncomp - 1 - kk
            s = ftz(
                s
                + ftz(identical_exp(ftz(wlp.unsafe_load(base + k) - max_exp)))
            )
    elif sab == GMM_SAB_LSE_ROTATE:
        var k0 = Int(block_idx.x) % ncomp
        for t in range(ncomp):
            var k = (k0 + t) % ncomp
            s = ftz(
                s
                + ftz(identical_exp(ftz(wlp.unsafe_load(base + k) - max_exp)))
            )
    else:
        for k in range(ncomp):
            s = ftz(
                s
                + ftz(identical_exp(ftz(wlp.unsafe_load(base + k) - max_exp)))
            )
    lse.unsafe_store(i, ftz(identical_log(s) + max_exp))


def sabotage_meanll_kernel(
    lse: MutPointer[Float32, MutAnyOrigin],
    out_scalar: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    sabotage_in: Int32,
):
    """`estep.mojo::meanll_kernel` with the PAIRWISE arm.

    ONE THREAD in both arms; only the BRACKETING differs, and the divide by
    `n` is identical in both, so a bit that moves here moved because of the
    fold and not because of the normalization.

    The pairwise arm is the standard equal-rank merge: push each term, and
    while the top two entries of the stack cover blocks of EQUAL LENGTH, pop
    them and add. That produces exactly the balanced adjacent-pair tree, needs
    a stack no deeper than `log2(n) + 1`, and is the fold shape a
    numerically-minded implementation reaches for -- which is the point of
    the arm. `GMM_STACK_MAX = 40` is far above `log2` of any `n` a float32
    sum is meaningful over.
    """
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var n = Int(n_in)
    var sab = Int(sabotage_in)
    if sab != GMM_SAB_MEANLL_PAIRWISE:
        var acc = Float32(0.0)
        for i in range(n):
            acc = ftz(acc + ftz(lse.unsafe_load(i)))
        out_scalar.unsafe_store(0, ftz(identical_div(acc, Float32(n))))
        return

    var vals = stack_allocation[
        GMM_STACK_MAX, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var ranks = stack_allocation[
        GMM_STACK_MAX, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var top = 0
    for i in range(n):
        vals[unsafe_offset=top] = ftz(lse.unsafe_load(i))
        ranks[unsafe_offset=top] = Int32(0)
        top += 1
        while top >= 2 and (
            ranks[unsafe_offset = top - 1] == ranks[unsafe_offset = top - 2]
        ):
            var b = vals[unsafe_offset = top - 1]
            var a = vals[unsafe_offset = top - 2]
            vals[unsafe_offset = top - 2] = ftz(a + b)
            ranks[unsafe_offset = top - 2] = (
                ranks[unsafe_offset = top - 2] + Int32(1)
            )
            top -= 1
    # Ragged tail: combine the leftover blocks from the SMALLEST (top) to the
    # largest, which is the same rule the balanced tree applies to a ragged
    # leaf count.
    var acc2 = Float32(0.0)
    if top > 0:
        acc2 = vals[unsafe_offset = top - 1]
        var t = top - 2
        while t >= 0:
            acc2 = ftz(vals[unsafe_offset=t] + acc2)
            t -= 1
    out_scalar.unsafe_store(0, ftz(identical_div(acc2, Float32(n))))


def sabotage_logdet_from_diag_kernel(
    prec: MutPointer[Float32, MutAnyOrigin],
    logdet: MutPointer[Float32, MutAnyOrigin],
    d_in: Int32,
    kcomp_in: Int32,
):
    """`GMM_SAB_LOGDET_FROM_DIAG`: scikit-learn's own spelling of
    `log_det_chol`, `sum_j log(P[j][j])` over the precision Cholesky's
    diagonal (`_compute_log_det_cholesky`, `_gaussian_mixture.py:448-487`).

    There is NO production counterpart to this kernel, because DEVIATION 1726
    routes the quantity through `cholesky/checks/potrf.mojo::chol_logdet`
    instead. It lives here so the check can show that the two spellings give
    DIFFERENT BITS, which is what makes 1726 a decision rather than a
    preference: `log(1/x)` and `-log(x)` are the same number and not the same
    float32, and a lane that computed it one way and an oracle that computed
    it the other would disagree for a reason nobody would find.
    """
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var d = Int(d_in)
    var kc = Int(kcomp_in)
    var acc = Float32(0.0)
    for j in range(d):
        acc = ftz(
            acc + ftz(identical_log(ftz(prec.unsafe_load(j * d + j))))
        )
    logdet.unsafe_store(kc, acc)


# ===========================================================================
# THE M-STEP KERNELS, SABOTAGED
# ===========================================================================


def sabotage_resp_exp_kernel(
    logresp: MutPointer[Float32, MutAnyOrigin],
    resp: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    ncomp_in: Int32,
    sabotage_in: Int32,
):
    """`mstep.mojo::resp_exp_kernel` with the NO_FTZ arm.

    `exp` of a large negative log responsibility is exactly where a
    Gaussian mixture manufactures subnormals, so this is the lane's denormal
    site. Metal flushes in hardware, so the arm is expected INERT on Apple
    and to move bits on NVIDIA and AMD, both of which keep subnormals
    (IDENTITY_PATHS row 39, measured on all three).
    """
    var n = Int(n_in)
    var ncomp = Int(ncomp_in)
    var sab = Int(sabotage_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * ncomp:
        return
    var v = identical_exp(ftz(logresp.unsafe_load(idx)))
    if sab == GMM_SAB_NO_FTZ_RESP:
        resp.unsafe_store(idx, v)
        return
    resp.unsafe_store(idx, ftz(v))


def sabotage_nk_kernel(
    resp: MutPointer[Float32, MutAnyOrigin],
    nk: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    ncomp_in: Int32,
    ten_eps: Float32,
    sabotage_in: Int32,
):
    """`mstep.mojo::nk_kernel` with the DESCENDING arm. One thread per
    component; only the direction of the sample walk differs."""
    var n = Int(n_in)
    var ncomp = Int(ncomp_in)
    var sab = Int(sabotage_in)
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k >= ncomp:
        return
    var acc = Float32(0.0)
    if sab == GMM_SAB_NK_DESCENDING:
        for ii in range(n):
            var i = n - 1 - ii
            acc = ftz(acc + ftz(resp.unsafe_load(i * ncomp + k)))
    else:
        for i in range(n):
            acc = ftz(acc + ftz(resp.unsafe_load(i * ncomp + k)))
    nk.unsafe_store(k, ftz(acc + ten_eps))


def sabotage_center_scale_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin],
    resp: MutPointer[Float32, MutAnyOrigin],
    nk: MutPointer[Float32, MutAnyOrigin],
    diff: MutPointer[Float32, MutAnyOrigin],
    scaled: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    kcomp_in: Int32,
    ncomp_in: Int32,
    sabotage_in: Int32,
):
    """`mstep.mojo::center_scale_kernel` with the PRESCALE arm.

    Production: `scaled = resp * diff`, and the accumulated outer product is
    divided by `nk` afterwards. PRESCALE: `scaled = (resp / nk) * diff` and
    the division afterwards is skipped, which the driver arranges. One
    rounding moves from after the sum to before it, on every term.
    """
    var n = Int(n_in)
    var d = Int(d_in)
    var kc = Int(kcomp_in)
    var ncomp = Int(ncomp_in)
    var sab = Int(sabotage_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * d:
        return
    var i = idx // d
    var j = idx % d
    var dv = ftz(ftz(x.unsafe_load(idx)) - ftz(means.unsafe_load(kc * d + j)))
    diff.unsafe_store(idx, dv)
    var r = ftz(resp.unsafe_load(i * ncomp + kc))
    if sab == GMM_SAB_COV_PRESCALE:
        r = ftz(identical_div(r, ftz(nk.unsafe_load(kc))))
    scaled.unsafe_store(idx, ftz(identical_mul(r, dv)))
