# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Five kernels, and FOUR of them are somebody else's code called, not copied.

**READ THIS BEFORE ADDING A KERNEL HERE.** This file is a DISPATCHER and an
argument for why it is only a dispatcher. The matrix product, the RBF
expansion, the L1 distance, the row norms and the two polynomial epilogues all
live in files this lane does not own or has ported beside the caller, and
`km_kernel_matrix` below is the ten lines that route between them.

    kernel      the dot / distance                       the epilogue
    ---------   --------------------------------------   -----------------------
    LINEAR      svm kernel_op (identical_gemm OP_NT)      none
    RBF         svm kernel_op (gemm + THEIR expansion)    theirs, in svm/
    POLYNOMIAL  svm kernel_op at a LINEAR KernelParams    ported here, cuVS
    SIGMOID     svm kernel_op at a LINEAR KernelParams    ported here, cuVS
    LAPLACIAN   kde pairwise_distance at DIST_L1          here (DEVIATION 1665)

**NOTE THE THIRD AND FOURTH ROWS.** The polynomial and sigmoid kernels need
`X Y^T` and nothing else before their epilogue, which is exactly what
`svm/impl/distance/kernel_matrices.mojo::kernel_op` computes when its
`KernelParams.kernel` is `KERNEL_LINEAR`. So this lane obtains the dot product
by CALLING THAT with a linear parameter block and then launching its own
epilogue on the result. No second spelling of the matrix product exists in
`kernel_methods/`, and `mojolearn.identical.gemm.fp32.v1` is the only
contraction any kernel here rides.

THE NAME COLLIDES WITH `checks/kernel_matrix.mojo` AT THE REPOSITORY ROOT
AND THE TWO ARE UNRELATED. That file is the per-vendor TUNABLES matrix
(`lib_block_size_for`, `TARGET_COLUMN`); this one is about kernel matrices in
the machine-learning sense. The brief that opened this lane named the path, so
it is kept, and this paragraph is the disambiguation a grep will land on.

WHAT IS NOT HERE, AND WHERE IT IS
---------------------------------
- The contraction: `gemm/checks/gemm_identical.mojo`, profile
  `mojolearn.identical.gemm.fp32.v1`.
- The RBF expansion and the squared row norms: `svm/impl/distance/
  kernel_matrices.mojo::rbf_kernel_expanded_kernel`, `row_norms_l2sq`, a port
  of cuVS `kernel_matrices.cu` under that lane's DEVIATION 630.
- The Manhattan distance the laplacian kernel needs: `kde/impl/distance/
  distance.mojo::pairwise_distance` at `DIST_L1`, a port of RAFT's `l1.cuh`,
  one thread per cell with an ascending feature walk and every seam already
  flushed.
- `KernelParams` itself: `svm/impl/svm/svm_parameter.mojo`, which is
  `ML::matrix::KernelParams {kernel, degree, gamma, coef0}`. This lane adds
  ONE value to its kernel enumeration -- `KM_KERNEL_LAPLACIAN` -- and adds it
  HERE rather than in `svm/`, which this lane may not edit and which would
  gain a kernel its solver refuses.

# =========================================================================
# DEVIATION 1666: WHICH RBF THE LANE COMPUTES, BECAUSE THERE ARE TWO
# UPSTREAMS AND THEY DISAGREE.
#
# cuML's `rbf_kernel` (`metrics/pairwise_kernels.py:41-48`) is
# `exp(-gamma * pairwise_distances(X, Y, metric="sqeuclidean"))`. cuVS's
# `RBFKernel::evaluate` (`kernel_matrices.cu`) is the EXPANDED form,
# `exp(-gamma * (|x|^2 + |y|^2 - 2 x.y))`, computed as a GEMM plus an
# epilogue over precomputed row norms. scikit-learn is expanded too
# (`euclidean_distances`), with a `maximum(D, 0)` clamp and an exact-zero
# diagonal fix that cuVS does not have.
#
# THIS LANE COMPUTES THE EXPANDED ONE, cuVS's, with NO clamp at zero --
# because that is the arm already ported, already gated and already carrying
# a DEVIATION (630) in this repository, and a second RBF would be a second
# thing to get wrong. The difference is not cosmetic: the expansion
# catastrophically cancels for nearby rows, so `|x|^2 + |y|^2 - 2 x.y` can
# come out slightly NEGATIVE where the true squared distance is a small
# positive, and `exp` of a small positive exponent then returns a kernel
# value just ABOVE 1. On the diagonal, where `x` and `y` are the same row,
# the expansion returns exactly `exp(-gamma * (2|x|^2 - 2|x|^2))` only if
# the GEMM's `x.x` equals the norm kernel's `x.x` bit for bit -- and it does
# NOT in general, because one is `identical_gemm`'s pinned fold and the other
# is `row_norm_l2sq_kernel`'s serial chain.
#
# **CONSEQUENCE THE KERNEL-RIDGE LANE HAS TO LIVE WITH, STATED RATHER THAN
# DISCOVERED LATER: the RBF Gram matrix's diagonal is NOT exactly 1.0.** So
# DEVIATION 1660's second argument -- that the absolute and relative jitter
# policies coincide on a unit diagonal -- is an argument about the
# MATHEMATICAL diagonal, and `check_km_sabotages` sweeps its fixtures instead
# of relying on it. `check_kernel_matrix_vs_oracle` prints the worst
# |diag - 1| it saw so the size of the effect is on the record.
# =========================================================================

# =========================================================================
# DEVIATION 1665: THE LAPLACIAN KERNEL HAS NO UPSTREAM EPILOGUE, SO IT IS
# WRITTEN HERE OVER A PORTED DISTANCE.
#
# cuVS's `kernel_matrices.cu` has four kernel types -- linear, polynomial,
# tanh and RBF -- and no laplacian. cuML has one
# (`pairwise_kernels.py:51-56`) and it is `exp(-gamma * manhattan)` in Python
# over `pairwise_distances`. So the ALGORITHM is upstream and the KERNEL is
# not, and the honest form of the port is: call the ported Manhattan distance
# (`kde/impl/distance/distance.mojo`, RAFT's `l1.cuh`) and write the
# four-token epilogue here.
#
# `PORTING_RULES 0b-i` is satisfied by that shape rather than violated by it:
# cuML's dispatch for `metric="laplacian"` reaches a device-wide distance
# computation followed by a device-wide elementwise `exp`, unfused, in two
# passes, and so does this. Their fused arm does not exist.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from kde.impl.distance.distance import pairwise_distance
from kde.impl.distance.distance_ops import DIST_L1
from kernel_methods.checks.km_sabotage import (
    KMSAB_NONE,
    km_sabotage_touches_kernel_matrix,
    sabotage_laplacian_epilogue_kernel,
    sabotage_polynomial_epilogue_kernel,
    sabotage_rbf_epilogue_kernel,
    sabotage_tanh_epilogue_kernel,
)
from kernel_methods.impl.distance.kernel_matrices import (
    KM_EPILOGUE_TPB,
    KM_MAX_DEGREE,
    polynomial_epilogue_kernel,
    tanh_epilogue_kernel,
)
from checks.numerics import (
    ftz,
    identical_exp,
    identical_mul,
)
from svm.impl.distance.kernel_matrices import (
    kernel_op,
    kernel_workspace_floats,
    row_norms_l2sq,
)
from svm.impl.svm.svm_parameter import (
    KERNEL_LINEAR,
    KERNEL_POLYNOMIAL,
    KERNEL_PRECOMPUTED,
    KERNEL_RBF,
    KERNEL_TANH,
    KernelParams,
)


# ===========================================================================
# The kernel enumeration. The first five values ARE
# `svm/impl/svm/svm_parameter.mojo`'s, imported rather than restated, so a
# `KernelParams` built here is the same struct their solver reads and a value
# can never mean two things in one repository.
# ===========================================================================

comptime KM_KERNEL_LINEAR = KERNEL_LINEAR
comptime KM_KERNEL_POLYNOMIAL = KERNEL_POLYNOMIAL
comptime KM_KERNEL_RBF = KERNEL_RBF
comptime KM_KERNEL_SIGMOID = KERNEL_TANH
comptime KM_KERNEL_PRECOMPUTED = KERNEL_PRECOMPUTED

#: THE ONE VALUE THIS LANE ADDS. `svm/`'s enumeration stops at
#: `KERNEL_PRECOMPUTED = 4` and this lane may not edit that file, so the
#: laplacian kernel takes the next value here. A `KernelParams` carrying it
#: is legal input to `km_kernel_matrix` and is NOT legal input to anything in
#: `svm/`, which refuses it by name at `kernel_op`'s final `elif`. That
#: asymmetry is real and is why `km_kernel_matrix` never routes a laplacian
#: through `kernel_op`.
comptime KM_KERNEL_LAPLACIAN = 5

comptime KM_KERNEL_COUNT = 6

#: SCHEDULING: the block width for this lane's own elementwise kernels.
comptime KM_TPB = 256


def km_kernel_name(kernel: Int) -> String:
    if kernel == KM_KERNEL_LINEAR:
        return String("linear")
    if kernel == KM_KERNEL_POLYNOMIAL:
        return String("polynomial")
    if kernel == KM_KERNEL_RBF:
        return String("rbf")
    if kernel == KM_KERNEL_SIGMOID:
        return String("sigmoid")
    if kernel == KM_KERNEL_PRECOMPUTED:
        return String("precomputed")
    if kernel == KM_KERNEL_LAPLACIAN:
        return String("laplacian")
    return String("unknown")


def km_kernel_from_name(name: String) raises -> Int:
    """scikit-learn's `PAIRWISE_KERNEL_FUNCTIONS` keys, for the five this
    lane supports, plus `poly` which is their alias for `polynomial`.

    REFUSES BY NAME (DEVIATION 1686). The refusal text lists what IS
    supported and names where each unsupported one would go, because a
    caller who typed `chi2` needs to know it is a deferral and not a typo.
    """
    if name == "linear":
        return KM_KERNEL_LINEAR
    if name == "polynomial" or name == "poly":
        return KM_KERNEL_POLYNOMIAL
    if name == "rbf":
        return KM_KERNEL_RBF
    if name == "sigmoid":
        return KM_KERNEL_SIGMOID
    if name == "laplacian":
        return KM_KERNEL_LAPLACIAN
    raise Error(
        "kernel_methods: unsupported kernel '"
        + name
        + "'. This lane ports linear, polynomial (alias poly), rbf, sigmoid"
        " and laplacian. scikit-learn's cosine, chi2 and additive_chi2, and"
        " every callable kernel, are UNPORTED and carry rows in"
        " kernel_methods/NOT_IMPLEMENTED.tsv; 'precomputed' is refused separately"
        " because it is a shape contract rather than a kernel and nothing"
        " here validates it (DEVIATION 1683)"
    )


def km_gamma_default(n_features: Int) -> Float64:
    """`gamma = 1.0 / X.shape[1]` when the caller passes none.

    THEIR default, and it is theirs three times over: cuML's
    `polynomial_kernel`, `sigmoid_kernel`, `rbf_kernel` and
    `laplacian_kernel` each open with `if gamma is None: gamma = 1.0 /
    X.shape[1]` (`pairwise_kernels.py:21, 31, 42, 52`), and scikit-learn's
    do the same. Computed in FLOAT64 on the host and narrowed once at the
    kernel-argument boundary, because `1 / d` for a non-power-of-two `d` is
    inexact and doing it twice in two precisions is two numbers.

    NOT applied silently: `km_validate_kernel_params` refuses a non-positive
    gamma, and the estimator surfaces record the gamma they used in the
    card's header so a run is reproducible from its own transcript.
    """
    return 1.0 / Float64(n_features)


def km_validate_kernel_params(kp: KernelParams, what: String) raises:
    """Every refusal a kernel parameter block can earn, BY NAME, on the host,
    before a buffer is allocated. DEVIATION 1686.

    `degree` is the interesting one and DEVIATION 1663 is its argument: it
    must be a non-negative integer at or below `KM_MAX_DEGREE`, because the
    power is a repeated product and because `identical_pow` -- the only other
    spelling available -- returns NaN on the negative bases a polynomial
    kernel routinely produces. `KernelParams.degree` is already an `Int` in
    this tree, so the integrality is a property of the type; what is checked
    here is the RANGE and the fact that a caller who wanted `degree = 2.5`
    was refused upstream at the estimator's own argument rather than having
    it silently floored.
    """
    if kp.kernel == KM_KERNEL_PRECOMPUTED:
        raise Error(
            what
            + ": kernel='precomputed' is refused by name. cuML and"
            " scikit-learn both accept it and both treat X as an already-"
            " formed kernel matrix, which is a SHAPE CONTRACT this lane does"
            " not validate and cannot check an oracle against. DEVIATION"
            " 1683; kernel_methods/NOT_IMPLEMENTED.tsv carries the row"
        )
    if (
        kp.kernel != KM_KERNEL_LINEAR
        and kp.kernel != KM_KERNEL_POLYNOMIAL
        and kp.kernel != KM_KERNEL_RBF
        and kp.kernel != KM_KERNEL_SIGMOID
        and kp.kernel != KM_KERNEL_LAPLACIAN
    ):
        raise Error(
            what
            + ": kernel value "
            + String(kp.kernel)
            + " is not one of the five this lane ports (linear="
            + String(KM_KERNEL_LINEAR)
            + ", polynomial="
            + String(KM_KERNEL_POLYNOMIAL)
            + ", rbf="
            + String(KM_KERNEL_RBF)
            + ", sigmoid="
            + String(KM_KERNEL_SIGMOID)
            + ", laplacian="
            + String(KM_KERNEL_LAPLACIAN)
            + ")"
        )
    if kp.gamma != kp.gamma:
        raise Error(what + ": gamma is NaN")
    if kp.coef0 != kp.coef0:
        raise Error(what + ": coef0 is NaN")
    var needs_gamma = (
        kp.kernel == KM_KERNEL_POLYNOMIAL
        or kp.kernel == KM_KERNEL_RBF
        or kp.kernel == KM_KERNEL_SIGMOID
        or kp.kernel == KM_KERNEL_LAPLACIAN
    )
    if needs_gamma and not (kp.gamma > 0.0):
        raise Error(
            what
            + ": the "
            + km_kernel_name(kp.kernel)
            + " kernel needs a POSITIVE gamma; got a value that is not"
            " greater than zero. A zero gamma collapses the RBF and"
            " laplacian kernels to the all-ones matrix, which is singular"
            " at every size above one, and a negative one turns them into"
            " a divergent exponential. Spelled `not (gamma > 0)` so a NaN"
            " that got past the test above would still be refused"
        )
    if kp.kernel == KM_KERNEL_POLYNOMIAL:
        if kp.degree < 0:
            raise Error(
                what
                + ": degree must be a NON-NEGATIVE integer, got "
                + String(kp.degree)
                + ". DEVIATION 1663: the polynomial power is an ascending"
                " repeated product, so a negative degree has no spelling"
                " here, and identical_pow (exp(p log x)) cannot stand in"
                " because a polynomial kernel's base is routinely negative"
                " and portable_powf returns NaN there"
            )
        if kp.degree > KM_MAX_DEGREE:
            raise Error(
                what
                + ": degree "
                + String(kp.degree)
                + " exceeds KM_MAX_DEGREE = "
                + String(KM_MAX_DEGREE)
                + ". To close this refusal, decide what a float32 kernel"
                " matrix raised to that power is supposed to mean -- at"
                " degree 33 a base of 2 already overflows float32 -- and"
                " then re-gate check_kernel_matrix_vs_oracle at the larger"
                " degree. DEVIATION 1663"
            )


def km_validate_matrix(
    values: List[Float32], n_rows: Int, n_cols: Int, what: String
) raises:
    """Shape and finiteness, refused BY NAME with the offending flat index.

    DEVIATION 1686. NaN and infinity are refused rather than propagated for
    the reason `cholesky/checks/potrf.mojo::chol_validate_matrix` gives:
    a NaN that reaches a kernel matrix reaches the pivot decision, and the
    pivot decision is DATA-DEPENDENT CONTROL FLOW, so one non-finite input
    can make two vendors disagree about whether the problem is solvable at
    all -- and no downstream bitwise gate ever runs on a run that took two
    different branches.
    """
    if n_rows <= 0 or n_cols <= 0:
        raise Error(
            what
            + ": need positive dimensions, got "
            + String(n_rows)
            + " x "
            + String(n_cols)
        )
    if len(values) != n_rows * n_cols:
        raise Error(
            what
            + " holds "
            + String(len(values))
            + " floats, "
            + String(n_rows)
            + " x "
            + String(n_cols)
            + " needs "
            + String(n_rows * n_cols)
        )
    for i in range(len(values)):
        var v = values[i]
        if v != v:
            raise Error(
                what
                + ": NaN at flat index "
                + String(i)
                + "; refused by name (DEVIATION 1686)"
            )
        if v > Float32(3.4028234663852886e38) or v < Float32(
            -3.4028234663852886e38
        ):
            raise Error(
                what
                + ": infinity at flat index "
                + String(i)
                + "; refused by name (DEVIATION 1686)"
            )


# ===========================================================================
# The one epilogue this lane owns outright
# ===========================================================================


def laplacian_epilogue_kernel(
    inout_k: MutPointer[Float32, MutAnyOrigin],
    len_in: Int32,
    gain: Float32,
):
    """`K = exp(-gamma * manhattan(X, Y))`, one thread per cell.

    cuML's line is `K = -gamma * pairwise_distances(..., "manhattan"); exp(K,
    K)` (`pairwise_kernels.py:54-55`), so the NEGATION IS FOLDED INTO THE
    GAIN by the caller and this kernel multiplies by an already-negative
    number. Written that way rather than as `identical_exp(-(gamma * d))`
    because a negate-then-multiply and a multiply-by-a-negative are the same
    bits, and folding it host-side means one fewer float operation inside a
    kernel that runs once per cell of an `n x n` matrix.

    `identical_exp` because a device `exp` is a vendor choice in its last bit
    (IDENTITY_PATHS row 12); the whole matrix goes through it.
    """
    var n = Int(len_in)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= n:
        return
    var d = ftz(inout_k.unsafe_load(tid))
    inout_k.unsafe_store(tid, ftz(identical_exp(ftz(identical_mul(gain, d)))))


# ===========================================================================
# The dispatcher
# ===========================================================================


def km_kernel_workspace_floats(m: Int, n: Int, k: Int) -> Int:
    """What `km_kernel_matrix` needs in `ws` for an `m x n` kernel matrix
    over `k` features. `svm/impl/distance/kernel_matrices.mojo`'s helper,
    re-exported so this lane never guesses a GEMM workspace -- the gemm
    lane's own docstring records that sizing a workspace for one plan and
    letting the dispatcher pick another is an out-of-bounds write a small
    shape will not show you."""
    return kernel_workspace_floats(m, n, k)


def km_kernel_matrix(
    ctx: DeviceContext,
    kp: KernelParams,
    mut out: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    mut norm_a: DeviceBuffer[DType.float32],
    mut norm_b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    elem_tpb: Int = KM_EPILOGUE_TPB,
    sabotage: Int = KMSAB_NONE,
) raises:
    """`out[m x n] = K(a_i, b_j)`, row-major, for the five ported kernels.

    ASYNCHRONOUS. `ws` must hold at least `km_kernel_workspace_floats(m, n,
    k)` floats and every buffer must outlive the caller's own
    `ctx.synchronize()`.

    `elem_tpb` is SCHEDULING and the checks vary it. Nothing in this file
    reads a block index, a block count or a lane id into a value.

    `sabotage` is `KMSAB_NONE` on every production path. When it names an arm
    this driver launches `km_sabotage.mojo`'s COPY of the epilogue instead of
    the real one, so no production kernel in this lane carries a sabotage
    branch and the shipped bits cannot depend on the sabotage file
    (`cholesky`'s DEVIATION 1642 construction; DEVIATION 1687 here).
    `KMSAB_COPY_ONLY` routes through the copies with no arm engaged, which is
    how `check_km_sabotage_copies_agree` proves the copies are faithful before
    any arm is believed.
    """
    if m <= 0 or n <= 0:
        return

    var via_copy = km_sabotage_touches_kernel_matrix(sabotage)
    var grid_all = (m * n + elem_tpb - 1) // elem_tpb

    if kp.kernel == KM_KERNEL_LAPLACIAN:
        # No dot product and therefore no norms. Zero them so nothing
        # downstream can hash uninitialized memory and report a divergence
        # that is really an allocator.
        ctx.enqueue_memset(norm_a, Float32(0.0))
        ctx.enqueue_memset(norm_b, Float32(0.0))
        pairwise_distance(ctx, out, a, b, m, n, k, DIST_L1, elem_tpb)
        if via_copy:
            ctx.enqueue_function[sabotage_laplacian_epilogue_kernel](
                out.unsafe_ptr(),
                Int32(m * n),
                Float32(-kp.gamma),
                Int32(sabotage),
                grid_dim=(grid_all, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
            return
        ctx.enqueue_function[laplacian_epilogue_kernel](
            out.unsafe_ptr(),
            Int32(m * n),
            Float32(-kp.gamma),
            grid_dim=(grid_all, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        return

    # Every remaining kernel starts from `a . b^T`. `row_norms_l2sq` is
    # only READ by the RBF expansion, and it is computed unconditionally
    # anyway: it is one pass over the data against an O(m n k) contraction,
    # and a buffer that is sometimes written is a buffer whose card stage
    # sometimes means something.
    row_norms_l2sq(ctx, norm_a, a, m, k)
    row_norms_l2sq(ctx, norm_b, b, n, k)

    if kp.kernel == KM_KERNEL_LINEAR:
        # THEIR CODE, CALLED, and there is no epilogue to sabotage: a linear
        # kernel IS the pinned GEMM, whose own six sabotages live in the gemm
        # lane and are not this lane's to re-drive.
        kernel_op(ctx, kp, out, a, b, m, n, k, norm_a, norm_b, ws)
        return

    if kp.kernel == KM_KERNEL_RBF and not via_copy:
        # THEIR CODE, CALLED. `kernel_op` issues the pinned GEMM and svm's
        # ported expansion epilogue in one call.
        kernel_op(ctx, kp, out, a, b, m, n, k, norm_a, norm_b, ws)
        return

    # RBF-under-a-copy, POLYNOMIAL and SIGMOID all start from the dot alone,
    # and they get it from the SAME `kernel_op` at a LINEAR parameter block.
    var dot_only = KernelParams(KM_KERNEL_LINEAR, kp.degree, kp.gamma, kp.coef0)
    kernel_op(ctx, dot_only, out, a, b, m, n, k, norm_a, norm_b, ws)

    if kp.kernel == KM_KERNEL_RBF:
        ctx.enqueue_function[sabotage_rbf_epilogue_kernel](
            out.unsafe_ptr(),
            Int32(m),
            Int32(n),
            norm_a.unsafe_ptr(),
            norm_b.unsafe_ptr(),
            Float32(kp.gamma),
            Int32(sabotage),
            grid_dim=(grid_all, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        return

    if kp.kernel == KM_KERNEL_POLYNOMIAL:
        if via_copy:
            ctx.enqueue_function[sabotage_polynomial_epilogue_kernel](
                out.unsafe_ptr(),
                Int32(m * n),
                Int32(kp.degree),
                Float32(kp.gamma),
                Float32(kp.coef0),
                Int32(sabotage),
                grid_dim=(grid_all, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
            return
        ctx.enqueue_function[polynomial_epilogue_kernel](
            out.unsafe_ptr(),
            Int32(m * n),
            Int32(kp.degree),
            Float32(kp.gamma),
            Float32(kp.coef0),
            grid_dim=(grid_all, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        return

    if via_copy:
        ctx.enqueue_function[sabotage_tanh_epilogue_kernel](
            out.unsafe_ptr(),
            Int32(m * n),
            Float32(kp.gamma),
            Float32(kp.coef0),
            Int32(sabotage),
            grid_dim=(grid_all, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        return

    ctx.enqueue_function[tanh_epilogue_kernel](
        out.unsafe_ptr(),
        Int32(m * n),
        Float32(kp.gamma),
        Float32(kp.coef0),
        grid_dim=(grid_all, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
