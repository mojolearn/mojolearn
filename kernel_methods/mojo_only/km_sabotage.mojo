"""This lane's kernels with their pins BROKEN ON PURPOSE, for the check.

NOT A PORT, NOT REACHED by any driver at `KMSAB_NONE`, not by
`kernel_methods/estimator.mojo` and not by the card. Copies of the epilogues
in `kernel_methods/ported/distance/kernel_matrices.mojo`, of
`svm/ported/distance/kernel_matrices.mojo::rbf_kernel_expanded_kernel`, of
`kernel_methods/mojo_only/kernel_matrix.mojo::laplacian_epilogue_kernel`, of
`kernel_methods/ported/kernel_ridge/kernel_ridge.mojo::add_ridge_diag_kernel`
and of `random_features.mojo`'s two, each carrying arms that `km_check.mojo`
selects through the `sabotage` argument threaded down from the drivers and
that nothing else can reach.

**THE UN-SABOTAGED ARM OF EVERY KERNEL HERE IS NEVER LAUNCHED**: the drivers
call the real kernel when `sabotage == KMSAB_NONE` and one of these
otherwise, so the shipped bits never depend on this file.

Same construction as `cholesky/mojo_only/chol_sabotage.mojo` and
`hierarchy/mojo_only/sabotage_tile.mojo`, and for the same two reasons: a
sabotage arm does not belong in a production kernel, and a sabotage that
requires editing source cannot be run by an orchestrator that is forbidden to
edit source. DEVIATION 1687.

ONE DUPLICATION IS TAKEN HERE AND IT IS NAMED. `sabotage_rbf_epilogue_kernel`
is a copy of a kernel in `svm/`, not in this lane. It has to be:
`STD_TRANSCENDENTAL` must reach the RBF exponential, `svm/` may not be
edited by this lane, and `svm/`'s own compile-time `SAB_STD_EXP` define
cannot be selected at run time. So the RBF production path is svm's
`kernel_op` and the RBF sabotage path is `identical_gemm` (through the same
`kernel_op` at a LINEAR parameter block) plus the copy below. **If svm's
epilogue ever changes, this copy is stale and the gate becomes a comparison
of two of our own old ideas.** `check_km_sabotage_copies_agree` exists for exactly
that: driven at `KMSAB_COPY_ONLY`, this file's copies must reproduce the
production kernels bit for bit on every fixture, which is what makes a
failing arm attributable to the arm.

THE ARMS, and what each is a plausible way to get wrong
--------------------------------------------------------
`KMSAB_STD_TRANSCENDENTAL`  KERNEL. The RBF and laplacian `exp`, the sigmoid
                            `tanh` and the feature map's `cos` go through
                            `std.math` instead of the `identical_*` family.
                            The single largest transcendental surface in the
                            lane: one call per cell of an `n x n` matrix.
                            IDENTITY_PATHS row 12.
`KMSAB_POLY_VIA_POW`        KERNEL. The polynomial power becomes
                            `identical_pow(base, degree)`, i.e.
                            `exp(p log x)`. DEVIATION 1663's rejected
                            spelling, and on any fixture with a NEGATIVE base
                            it returns NaN rather than a wrong number, which
                            is the point.
`KMSAB_RIDGE_RELATIVE`      KERNEL. `K_ii += alpha * K_ii` instead of
                            `K_ii += alpha`. **MUST BE SWEPT.** On an RBF or
                            laplacian kernel matrix the two nearly coincide
                            because the diagonal is nearly 1
                            (`cholesky/README.md`'s finding, and DEVIATION
                            1666 says why "nearly" and not "exactly").
`KMSAB_RIDGE_PLUS_JITTER`   DRIVER. The Cholesky profile's `2^-20` ridge is
                            applied IN ADDITION to `alpha`. DEVIATION 1660's
                            forbidden state, and the one a careful reader
                            reaches for by accident because `cholesky/`'s
                            entry point asks for a jitter and refuses to
                            default it.
`KMSAB_NO_SIGN_FLIP`        DRIVER. `sign_flip_kernel` is not launched.
                            **EXPECTED to move the eigenvector stage and to
                            be INERT on the normalization and the
                            embedding**, and that asymmetry is DEVIATION
                            1668's claim checked rather than asserted.
`KMSAB_EIGEN_ORDER_ASCENDING` DRIVER. The eigenvalue order is ascending.
                            Same multiset, different `k` axis for the
                            normalization product.
`KMSAB_EIGEN_TIE_UNSTABLE`  DRIVER. The sort's tie break keeps the HIGHER
                            index instead of the lower, so the order stops
                            being the total order the convention names.
                            Inert on any fixture without a repeated
                            eigenvalue, which is why the sweep is required
                            and why `FIX_KM_ORTHO` plants four-way ties.
`KMSAB_NO_EIGEN_CLIP`       DRIVER. sklearn's `clip(S, 1e-12, None)` is
                            dropped, so a zero or negative eigenvalue -- which
                            a float32 Jacobi produces on a rank-deficient
                            Gram -- becomes an infinity or a NaN in
                            `s^{-1/2}`.
`KMSAB_BASIS_FROM_LAUNCH`   DRIVER. The basis rows are a LAUNCH-STRIDED slice
                            `[0, stride, 2*stride, ...]` instead of the
                            position-mapped rank prefix, so the fit depends
                            on block geometry. DEVIATION 1671's gate.
`KMSAB_RF_STREAM_DRAW`      KERNEL. `W[f][j]` is drawn at the SEQUENTIAL
                            position `f * n_components + j` instead of at
                            `(f, j // 2)`, which is what numpy does and what
                            DEVIATION 1671's argument is against. **INERT at
                            a single `n_components`**, which is exactly how a
                            stream-shaped defect hides from a gate that never
                            varies the width.
`KMSAB_NO_BOXMULLER_GUARD`  KERNEL. DEVIATION 1676's `log(0)` guard dropped.
                            REPORT in the sweep -- a `2^-24` event will not
                            occur in a fixture -- and driven DIRECTLY at
                            `u1 = +0.0` by `check_boxmuller_guard`.
`KMSAB_EMBED_OP_NN`         DRIVER. The embedding is `embedded @
                            normalization` instead of sklearn's
                            `@ normalization.T`. DEVIATION 1674, and it would
                            be inert if `normalization` were bitwise
                            symmetric, which it is not.
`KMSAB_RF_SCALE_IN_KERNEL`  KERNEL. `sqrt(2 / n_components)` is recomputed
                            per thread inside the epilogue instead of once on
                            the host. DEVIATION 1678, classified REPORT and
                            EXPECTED INERT: what it proves is that the
                            constant is not secretly a fold or a
                            launch-shaped quantity.

The DRIVER arms have no kernel in this file; they are branches in
`kernel_methods/estimator.mojo`'s and `kernel_matrix.mojo`'s drivers taken
only when `sabotage` names them, exactly as `cholesky/mojo_only/potrf.mojo`
leaves its three driver arms on the production path.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import cos, exp, tanh

from core.philox import PhiloxState
from kernel_methods.ported.random.rng_device import (
    km_boxmuller_pair,
    km_guard_unit,
    km_unit_float_from,
)
from mojo_only.numerics import (
    ftz,
    identical_cos,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_pow,
    identical_sqrt,
    identical_tanh,
)
from resample.mojo_only.index_map import key_join, position_subsequence


#: The production path. Every driver default.
comptime KMSAB_NONE = 0
comptime KMSAB_STD_TRANSCENDENTAL = 1
comptime KMSAB_POLY_VIA_POW = 2
comptime KMSAB_RIDGE_RELATIVE = 3
comptime KMSAB_RIDGE_PLUS_JITTER = 4
comptime KMSAB_NO_SIGN_FLIP = 5
comptime KMSAB_EIGEN_ORDER_ASCENDING = 6
comptime KMSAB_EIGEN_TIE_UNSTABLE = 7
comptime KMSAB_NO_EIGEN_CLIP = 8
comptime KMSAB_BASIS_FROM_LAUNCH = 9
comptime KMSAB_RF_STREAM_DRAW = 10
comptime KMSAB_NO_BOXMULLER_GUARD = 11
comptime KMSAB_EMBED_OP_NN = 12
comptime KMSAB_RF_SCALE_IN_KERNEL = 13

#: NOT AN ARM. Routes the drivers through the COPIES in this file with no
#: arm engaged, so `check_km_sabotage_copies_agree` can prove the copies
#: reproduce the production kernels BIT FOR BIT before any arm is trusted.
#:
#: **THIS IS WHAT MAKES A FAILING ARM ATTRIBUTABLE.** Without it, an arm that
#: moves bits could be moving them because the arm did something or because
#: the copy drifted from the kernel it copies -- and the RBF copy is a copy of
#: a kernel in ANOTHER LANE (`svm/`), which this lane cannot edit and which
#: can change without anything here noticing. A sabotage suite whose control
#: is untested is a suite that reports its own transcription errors as
#: evidence.
comptime KMSAB_COPY_ONLY = 14

comptime KMSAB_COUNT = 15


def km_sabotage_name(sab: Int) -> String:
    """The arm's name, for the check's banner and for an error message."""
    if sab == KMSAB_NONE:
        return String("NONE")
    if sab == KMSAB_STD_TRANSCENDENTAL:
        return String("STD_TRANSCENDENTAL")
    if sab == KMSAB_POLY_VIA_POW:
        return String("POLY_VIA_POW")
    if sab == KMSAB_RIDGE_RELATIVE:
        return String("RIDGE_RELATIVE")
    if sab == KMSAB_RIDGE_PLUS_JITTER:
        return String("RIDGE_PLUS_JITTER")
    if sab == KMSAB_NO_SIGN_FLIP:
        return String("NO_SIGN_FLIP")
    if sab == KMSAB_EIGEN_ORDER_ASCENDING:
        return String("EIGEN_ORDER_ASCENDING")
    if sab == KMSAB_EIGEN_TIE_UNSTABLE:
        return String("EIGEN_TIE_UNSTABLE")
    if sab == KMSAB_NO_EIGEN_CLIP:
        return String("NO_EIGEN_CLIP")
    if sab == KMSAB_BASIS_FROM_LAUNCH:
        return String("BASIS_FROM_LAUNCH")
    if sab == KMSAB_RF_STREAM_DRAW:
        return String("RF_STREAM_DRAW")
    if sab == KMSAB_NO_BOXMULLER_GUARD:
        return String("NO_BOXMULLER_GUARD")
    if sab == KMSAB_EMBED_OP_NN:
        return String("EMBED_OP_NN")
    if sab == KMSAB_RF_SCALE_IN_KERNEL:
        return String("RF_SCALE_IN_KERNEL")
    if sab == KMSAB_COPY_ONLY:
        return String("COPY_ONLY")
    return String("UNKNOWN")


def km_sabotage_is_kernel_arm(sab: Int) -> Bool:
    """True when the arm lives in a kernel HERE, false when it is a branch in
    a driver. The drivers use this to decide which kernel to launch, so the
    production kernel is reached at every driver arm."""
    if sab == KMSAB_STD_TRANSCENDENTAL:
        return True
    if sab == KMSAB_POLY_VIA_POW:
        return True
    if sab == KMSAB_RIDGE_RELATIVE:
        return True
    if sab == KMSAB_RF_STREAM_DRAW:
        return True
    if sab == KMSAB_NO_BOXMULLER_GUARD:
        return True
    if sab == KMSAB_RF_SCALE_IN_KERNEL:
        return True
    if sab == KMSAB_COPY_ONLY:
        return True
    return False


def km_sabotage_touches_kernel_matrix(sab: Int) -> Bool:
    """True for the three arms the kernel-matrix driver has to route around.
    Separated from `km_sabotage_is_kernel_arm` because the random-feature
    arms live in a different driver and routing a kernel matrix through the
    sabotage epilogues for them would change bits for no reason."""
    if sab == KMSAB_STD_TRANSCENDENTAL:
        return True
    if sab == KMSAB_POLY_VIA_POW:
        return True
    if sab == KMSAB_COPY_ONLY:
        return True
    return False


# ===========================================================================
# The kernel-matrix epilogues
# ===========================================================================


def sabotage_rbf_epilogue_kernel(
    inout_k: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
    norm_x: MutPointer[Float32, MutAnyOrigin],
    norm_y: MutPointer[Float32, MutAnyOrigin],
    gain: Float32,
    sabotage_in: Int32,
):
    """`svm/ported/distance/kernel_matrices.mojo::rbf_kernel_expanded_kernel`,
    its IDENTICAL arm, character for character, plus one arm.

    THE COPY IS DELIBERATE AND IT IS THIS FILE'S ONE DUPLICATION; the module
    header says why and names the check that keeps it honest. Everything not
    named by the arm is their line exactly, so a failure names the arm and
    nothing else.
    """
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var sab = Int(sabotage_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= rows * cols:
        return
    var i = t // cols
    var j = t - i * cols
    var dot = inout_k.unsafe_load(t)
    var s = ftz(
        ftz(ftz(norm_x.unsafe_load(i)) + ftz(norm_y.unsafe_load(j)))
        - ftz(Float32(2.0) * ftz(dot))
    )
    var e = ftz((-gain) * s)
    if sab == KMSAB_STD_TRANSCENDENTAL:
        inout_k.unsafe_store(t, ftz(exp(e)))
        return
    inout_k.unsafe_store(t, ftz(identical_exp(e)))


def sabotage_polynomial_epilogue_kernel(
    inout_k: MutPointer[Float32, MutAnyOrigin],
    len_in: Int32,
    degree_in: Int32,
    gain: Float32,
    offset: Float32,
    sabotage_in: Int32,
):
    """`kernel_methods/ported/distance/kernel_matrices.mojo::
    polynomial_epilogue_kernel` with the `POLY_VIA_POW` arm.

    The arm is `identical_pow(base, Float32(degree))`, which is
    `portable_powf` = `exp(p * log(x))`. On a NEGATIVE base that function
    returns a quiet NaN by its own documented contract, so on `FIX_KM_MIXED`
    this arm does not produce a slightly different number, it produces NaN
    across roughly half the matrix -- which is DEVIATION 1663's argument made
    visible rather than argued.
    """
    var n = Int(len_in)
    var sab = Int(sabotage_in)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= n:
        return
    var dot = ftz(inout_k.unsafe_load(tid))
    var t = ftz(identical_mul_add(gain, dot, offset))
    if sab == KMSAB_POLY_VIA_POW:
        inout_k.unsafe_store(tid, ftz(identical_pow(t, Float32(Int(degree_in)))))
        return
    var acc = Float32(1.0)
    for _ in range(Int(degree_in)):
        acc = ftz(identical_mul(acc, t))
    inout_k.unsafe_store(tid, acc)


def sabotage_tanh_epilogue_kernel(
    inout_k: MutPointer[Float32, MutAnyOrigin],
    len_in: Int32,
    gain: Float32,
    offset: Float32,
    sabotage_in: Int32,
):
    """`tanh_epilogue_kernel` with the `STD_TRANSCENDENTAL` arm."""
    var n = Int(len_in)
    var sab = Int(sabotage_in)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= n:
        return
    var dot = ftz(inout_k.unsafe_load(tid))
    var t = ftz(identical_mul_add(gain, dot, offset))
    if sab == KMSAB_STD_TRANSCENDENTAL:
        inout_k.unsafe_store(tid, ftz(tanh(t)))
        return
    inout_k.unsafe_store(tid, ftz(identical_tanh(t)))


def sabotage_laplacian_epilogue_kernel(
    inout_k: MutPointer[Float32, MutAnyOrigin],
    len_in: Int32,
    gain: Float32,
    sabotage_in: Int32,
):
    """`kernel_matrix.mojo::laplacian_epilogue_kernel` with the
    `STD_TRANSCENDENTAL` arm."""
    var n = Int(len_in)
    var sab = Int(sabotage_in)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= n:
        return
    var d = ftz(inout_k.unsafe_load(tid))
    var e = ftz(identical_mul(gain, d))
    if sab == KMSAB_STD_TRANSCENDENTAL:
        inout_k.unsafe_store(tid, ftz(exp(e)))
        return
    inout_k.unsafe_store(tid, ftz(identical_exp(e)))


# ===========================================================================
# The ridge
# ===========================================================================


def sabotage_ridge_diag_kernel(
    k_io: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    alpha: Float32,
    sabotage_in: Int32,
):
    """`kernel_ridge.mojo::add_ridge_diag_kernel` with the RELATIVE arm.

    `K_ii += alpha * K_ii` is not a wrong idea in general -- it is what a
    Gaussian-process library usually does, because an absolute ridge is
    meaningless without knowing the kernel's scale. It is wrong HERE because
    `alpha` is scikit-learn's and cuML's parameter and theirs is absolute
    (`K.flat[::n+1] += alpha[0]`), so a relative ridge answers a different
    question under the same parameter name.

    `cholesky/mojo_only/chol_sabotage.mojo::sabotage_jitter_diag_kernel`
    carries the same arm for the same reason, and this lane's version exists
    because `alpha` is not the Cholesky profile's jitter and the two are
    added by two different kernels (DEVIATION 1685).
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var d = ftz(k_io.unsafe_load(i * n + i))
    if Int(sabotage_in) == KMSAB_RIDGE_RELATIVE:
        k_io.unsafe_store(i * n + i, ftz(identical_mul_add(alpha, d, d)))
        return
    k_io.unsafe_store(i * n + i, ftz(d + alpha))


# ===========================================================================
# The random features
# ===========================================================================


def sabotage_random_weights_kernel(
    w_out: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    n_features_in: Int32,
    n_components_in: Int32,
    sigma: Float32,
    sabotage_in: Int32,
):
    """`random_features.mojo::random_weights_kernel` with two arms.

    `RF_STREAM_DRAW` is the important one and it is written to look
    REASONABLE, because the defect it models is the one a competent person
    ships: "one draw per weight, indexed by the weight's flat position" is a
    perfectly sensible sentence and it makes `W[f][j]` a function of
    `n_components`. numpy does exactly that -- `random_state.normal(size=(d,
    D))` fills row-major from a sequential stream -- and a run at `D = 256`
    then shares NOTHING with a run at `D = 64`.

    `NO_BOXMULLER_GUARD` drops DEVIATION 1676's `u1 == +0.0` substitution.
    """
    var d = Int(n_components_in)
    var nf = Int(n_features_in)
    var sab = Int(sabotage_in)
    var total = nf * d
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= total:
        return
    var f = t // d
    var j = t - f * d
    var key = key_join(lo_bits, hi_bits)

    var sub = position_subsequence(UInt64(f), UInt64(j // 2))
    var take_second = (j % 2) == 1
    if sab == KMSAB_RF_STREAM_DRAW:
        # ARM: the FLAT position in the weight matrix, which is a function of
        # `n_components`. One normal per position, `val2` discarded.
        sub = position_subsequence(UInt64(0), UInt64(f * d + j))
        take_second = False

    var gen = PhiloxState.init(key, sub, UInt64(0))
    var u1 = km_unit_float_from(gen)
    if sab != KMSAB_NO_BOXMULLER_GUARD:
        u1 = km_guard_unit(u1)
    var u2 = km_unit_float_from(gen)
    var pair = km_boxmuller_pair(u1, u2, sigma, Float32(0.0))
    if take_second:
        w_out.unsafe_store(t, pair[1])
    else:
        w_out.unsafe_store(t, pair[0])


def sabotage_feature_map_epilogue_kernel(
    proj_io: MutPointer[Float32, MutAnyOrigin],
    b_in: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_components_in: Int32,
    scale: Float32,
    sabotage_in: Int32,
):
    """`random_features.mojo::feature_map_epilogue_kernel` with two arms.

    `RF_SCALE_IN_KERNEL` recomputes `sqrt(2 / D)` per thread. It is EXPECTED
    INERT wherever the device's `div` and `sqrt` are the host's, which is
    every column under IDENTICAL by construction, and it is driven anyway:
    an expected-inert arm that MOVES is a finding about that column's
    division, and an expected-inert arm nobody drives is an assumption.
    """
    var d = Int(n_components_in)
    var sab = Int(sabotage_in)
    var total = Int(n_rows_in) * d
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= total:
        return
    var j = t % d
    var p = ftz(proj_io.unsafe_load(t))
    var shifted = ftz(p + ftz(b_in.unsafe_load(j)))

    var s = scale
    if sab == KMSAB_RF_SCALE_IN_KERNEL:
        s = ftz(
            identical_sqrt(ftz(identical_div(Float32(2.0), Float32(d))))
        )

    if sab == KMSAB_STD_TRANSCENDENTAL:
        proj_io.unsafe_store(t, ftz(identical_mul(cos(shifted), s)))
        return
    proj_io.unsafe_store(t, ftz(identical_mul(identical_cos(shifted), s)))
