# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Five kernels, and FOUR of them are somebody else's code called, not copied.

**READ THIS BEFORE ADDING A KERNEL HERE.** This file is a DISPATCHER and an
argument for why it is only a dispatcher. The matrix product, the RBF
expansion, the L1 distance, the row norms and the two polynomial epilogues all
live in files this lane does not own or has implemented beside the caller, and
`km_kernel_matrix` below is the ten lines that route between them.

    kernel      the dot / distance                       the epilogue
    ---------   --------------------------------------   -----------------------
    LINEAR      svm kernel_op (identical_gemm OP_NT)      none
    RBF         svm kernel_op (gemm + THEIR expansion)    theirs, in svm/
    POLYNOMIAL  svm kernel_op at a LINEAR KernelParams    implemented here, cuVS
    SIGMOID     svm kernel_op at a LINEAR KernelParams    implemented here, cuVS
    LAPLACIAN   kde pairwise_distance at DIST_L1          here (DEVIATION 1665)

**NOTE THE THIRD AND FOURTH ROWS.** The polynomial and sigmoid kernels need
`X Y^T` and nothing else before their epilogue, which is exactly what
`svm/impl/distance/kernel_matrices.mojo::kernel_op` computes when its
`KernelParams.kernel` is `KERNEL_LINEAR`. So this lane obtains the dot product
by CALLING THAT with a linear parameter block and then launching its own
epilogue on the result. No second spelling of the matrix product exists in
`kernel_methods/`, and `mojolearn.identical.gemm.fp32.v1` is the only
contraction any kernel here rides.

UNDER FAST THE CALL IS `kernel_op`'s LINES RATHER THAN `kernel_op` ITSELF
(`_svm_kernel_op`, 2026-09-14), because svm's FAST-only fused RBF tile made every FAST
build that names `kernel_op` compile sixteen extra device kernels, and
`km_check.mojo` stopped compiling inside 25 minutes. The table above is what
runs in every mode; the fused tile is not in it.

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
  kernel_matrices.mojo::rbf_kernel_expanded_kernel`, `row_norms_l2sq`, an implementation
  of cuVS `kernel_matrices.cu` under that lane's DEVIATION 630.
- The Manhattan distance the laplacian kernel needs: `kde/impl/distance/
  distance.mojo::pairwise_distance` at `DIST_L1`, an implementation of RAFT's `l1.cuh`,
  one thread per cell with an ascending feature walk and every seam already
  flushed.
- `KernelParams` itself: `svm/impl/svm_parameter.mojo`, which is
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
# because that is the arm already implemented, already gated and already carrying
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
# WRITTEN HERE OVER A IMPLEMENTED DISTANCE.
#
# cuVS's `kernel_matrices.cu` has four kernel types -- linear, polynomial,
# tanh and RBF -- and no laplacian. cuML has one
# (`pairwise_kernels.py:51-56`) and it is `exp(-gamma * manhattan)` in Python
# over `pairwise_distances`. So the ALGORITHM is the reference's and the KERNEL is
# not, and the honest form of the implementation is: call the implemented Manhattan distance
# (`kde/impl/distance/distance.mojo`, RAFT's `l1.cuh`) and write the
# four-token epilogue here.
#
# `CONTRIBUTING.md (Algorithms and references)` is satisfied by that shape rather than violated by it:
# cuML's dispatch for `metric="laplacian"` reaches a device-wide distance
# computation followed by a device-wide elementwise `exp`, unfused, in two
# passes, and so does this. Their fused arm does not exist.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from kde.impl.distance.distance import pairwise_distance
from kde.impl.distance.distance_ops import DIST_L1
from neighbors.impl.distance.detail.distance_ops import l1_core
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
from std.os import getenv

from std.sys.compile import is_defined
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
    identical_tanh,
)
from core.gemm import gemm_nt
from svm.impl.distance.kernel_matrices import (
    kernel_op,
    kernel_workspace_floats,
    rbf_kernel_expanded_kernel,
    row_norms_l2sq,
)
from svm.impl.svm_parameter import (
    KERNEL_LINEAR,
    KERNEL_POLYNOMIAL,
    KERNEL_PRECOMPUTED,
    KERNEL_RBF,
    KERNEL_TANH,
    KernelParams,
)


# ===========================================================================
# The kernel enumeration. The first five values ARE
# `svm/impl/svm_parameter.mojo`'s, imported rather than restated, so a
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

#: scikit-learn's `cosine`, `chi2` and `additive_chi2` pairwise kernels
#: (2026-09-27, lane x-neighbors-km-kernels). Legal input to
#: `km_kernel_matrix` only, as the laplacian is.
comptime KM_KERNEL_COSINE = 6
comptime KM_KERNEL_CHI2 = 7
comptime KM_KERNEL_ADDITIVE_CHI2 = 8

comptime KM_KERNEL_COUNT = 9

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
    if kernel == KM_KERNEL_COSINE:
        return String("cosine")
    if kernel == KM_KERNEL_CHI2:
        return String("chi2")
    if kernel == KM_KERNEL_ADDITIVE_CHI2:
        return String("additive_chi2")
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
    if name == "cosine":
        return KM_KERNEL_COSINE
    if name == "chi2":
        return KM_KERNEL_CHI2
    if name == "additive_chi2":
        return KM_KERNEL_ADDITIVE_CHI2
    raise Error(
        "kernel_methods: unsupported kernel '"
        + name
        + "'. This lane implements linear, polynomial (alias poly), rbf, sigmoid"
        " and laplacian. scikit-learn's cosine, chi2 and additive_chi2, and"
        " every callable kernel, are UNIMPLEMENTED and carry rows in"
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
        and kp.kernel != KM_KERNEL_COSINE
        and kp.kernel != KM_KERNEL_CHI2
        and kp.kernel != KM_KERNEL_ADDITIVE_CHI2
    ):
        raise Error(
            what
            + ": kernel value "
            + String(kp.kernel)
            + " is not one of the five this lane implements (linear="
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
        or kp.kernel == KM_KERNEL_CHI2
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


#: fam2-kernel-gp (2026-10-04), IDENTICAL, CANDIDATE ARM, OFF by default
#: (`-D MOJOLEARN_IDN_KM_RBF_CELL` turns it on in the device binding AND in
#: the host column, `km_host_oracle.mojo::KMH_RBF_CELL` reads the same
#: defines; `MOJOLEARN_IDN_ALL_OFF` turns it off): the RBF kernel matrix at
#: k <= KM_RBF_CELL_MAX_D features is ONE launch, one thread per cell, the
#: squared distance as one chain over the features ascending of
#: `(x - y)^2` (`identical_mul_add`) and the exponential in the same thread,
#: in place of two row-norm launches, the GEMM over k and the expansion
#: epilogue `|x|^2 + |y|^2 - 2 x.y`. BITS CHANGE (a direct distance, not the
#: expansion): NVIDIA, AMD and Apple run this kernel and the host column
#: runs the same line. The direct form has no cancellation (K(x, x) is
#: exactly 1 and no distance is negative) and is bitwise symmetric, so a
#: self-kernel (`self_kernel=True`: KernelRidge fit, Nystroem's basis kernel)
#: computes the upper triangle and mirrors it. `-D
#: MOJOLEARN_IDN_KM_RBF_CELL_D16` lowers the feature bound to 16 on both
#: columns, for timing the crossover against the GEMM.
comptime KM_IDN_RBF_CELL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_KM_RBF_CELL"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime KM_RBF_CELL_MAX_D = 16 if is_defined["MOJOLEARN_IDN_KM_RBF_CELL_D16"]() else 64


def km_rbf_cell_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    neg_gamma: Float32,
    sym_in: Int32,
):
    """KM_IDN_RBF_CELL: cell (i, j) = exp(-gamma sum_c (a_ic - b_jc)^2), the
    sum one chain over c ascending. `sym_in` != 0 (a IS b, m == n): threads
    below the diagonal return and each thread at or above it writes its
    cell and the mirrored one (the chain reads (x - y)^2, which is the same
    word for (y - x)^2, so the mirror is the cell's own bits)."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m * n:
        return
    var i = t // n
    var j = t - i * n
    if sym_in != Int32(0) and j < i:
        return
    var acc = Float32(0.0)
    for c in range(k):
        var d = ftz(ftz(a.unsafe_load(i * k + c)) - ftz(b.unsafe_load(j * k + c)))
        acc = ftz(identical_mul_add(d, d, acc))
    var v = ftz(identical_exp(ftz(identical_mul(neg_gamma, acc))))
    output.unsafe_store(t, v)
    if sym_in != Int32(0) and j != i:
        output.unsafe_store(j * n + i, v)


#: fix-kg1-kernel (2026-10-04, audit B11), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_KM_DOT_CELL_OFF` or `MOJOLEARN_IDN_ALL_OFF` restores the
#: GEMM + epilogue route; `km_host_oracle.mojo::KMH_DOT_CELL` and
#: `km_oracle.mojo` read the same defines): the POLYNOMIAL and SIGMOID kernel
#: matrices at k <= KM_DOT_CELL_MAX_D features are ONE launch, one thread per
#: cell, the dot as one chain over the features ascending
#: (`identical_mul_add(a, b, acc)`) and the epilogue line of
#: `polynomial_epilogue_kernel` / `tanh_epilogue_kernel` in the same thread,
#: in place of two row-norm launches, the GEMM over k and the epilogue launch.
#: BITS CHANGE (the chain, not the GEMM profile's fold): NVIDIA, AMD and Apple
#: run this kernel and the host column runs the same line. The chain reads
#: a_ic * b_jc, the same word as b_jc * a_ic, so a self-kernel computes the
#: upper triangle and mirrors it. `-D MOJOLEARN_IDN_KM_DOT_CELL_D16` lowers
#: the bound to 16 on every column, for timing the crossover.
comptime KM_IDN_DOT_CELL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_KM_DOT_CELL_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime KM_DOT_CELL_MAX_D = 16 if is_defined["MOJOLEARN_IDN_KM_DOT_CELL_D16"]() else 64


def km_dot_cell_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    gain: Float32,
    offset: Float32,
    degree_in: Int32,
    poly_in: Int32,
    sym_in: Int32,
):
    """KM_IDN_DOT_CELL: cell (i, j) = epilogue(sum_c a_ic b_jc), the sum one
    chain over c ascending; `poly_in` != 0 is `(gain dot + offset)^degree` by
    repeated multiplication, else `tanh(gain dot + offset)`, each the line of
    `impl/distance/kernel_matrices.mojo`'s epilogue. `sym_in` != 0 (a IS b,
    m == n): threads below the diagonal return and each thread at or above
    it writes its cell and the mirrored one."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m * n:
        return
    var i = t // n
    var j = t - i * n
    if sym_in != Int32(0) and j < i:
        return
    var dot = Float32(0.0)
    for c in range(k):
        dot = ftz(
            identical_mul_add(
                ftz(a.unsafe_load(i * k + c)), ftz(b.unsafe_load(j * k + c)), dot
            )
        )
    var base = ftz(identical_mul_add(gain, dot, offset))
    var v: Float32
    if poly_in != Int32(0):
        var acc = Float32(1.0)
        for _ in range(Int(degree_in)):
            acc = ftz(identical_mul(acc, base))
        v = acc
    else:
        v = ftz(identical_tanh(base))
    output.unsafe_store(t, v)
    if sym_in != Int32(0) and j != i:
        output.unsafe_store(j * n + i, v)


#: fix-kg1-kernel (2026-10-04, audit B12), IDENTICAL, ON by default
#: (`-D MOJOLEARN_IDN_KM_LAP_CELL_OFF` or `MOJOLEARN_IDN_ALL_OFF` restores the
#: two launches): the Laplacian kernel matrix (KernelRidge fit/predict,
#: Nystroem basis and transform cross kernel) as ONE launch, the L1 chain of
#: `kde pairwise_unexpanded_kernel` (`l1_core`, ascending) and the line of
#: `laplacian_epilogue_kernel` in the same thread. NO BIT MOVES: the same
#: operations in the same order, only the m x n intermediate store and
#: reload and one launch are gone, so the host column is unchanged.
#: abs(x - y) is the same word as abs(y - x), so a self-kernel computes the
#: upper triangle and mirrors it.
comptime KM_IDN_LAP_CELL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_KM_LAP_CELL_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def km_laplacian_cell_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    gain: Float32,
    sym_in: Int32,
):
    """KM_IDN_LAP_CELL: cell (i, j) = exp(gain * sum_c |a_ic - b_jc|), the
    caller's `gain = -gamma`; the chain is `l1_core` over c ascending."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m * n:
        return
    var i = t // n
    var j = t - i * n
    if sym_in != Int32(0) and j < i:
        return
    var acc = Float32(0.0)
    for c in range(k):
        acc = l1_core(
            acc, ftz(a.unsafe_load(i * k + c)), ftz(b.unsafe_load(j * k + c))
        )
    var d = ftz(acc)
    var v = ftz(identical_exp(ftz(identical_mul(gain, d))))
    output.unsafe_store(t, v)
    if sym_in != Int32(0) and j != i:
        output.unsafe_store(j * n + i, v)


def chi2_cell_kernel(
    out_k: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    gain: Float32,
    exp_it: Int32,
):
    """scikit-learn's `_chi2_kernel_fast`: `res = sum_k (x - y)^2 / (x + y)`
    over the features where `x + y != 0`, ascending, one thread per cell;
    `additive_chi2` writes `-res`, `chi2` writes `exp(gain * res)` with the
    caller's `gain = -gamma` (their `K = -res; K *= gamma; exp(K)`). Every
    operation rounds once (ftz, identical_mul / identical_div /
    identical_exp); `km_host_oracle.mojo::kmh_chi2_cell` is the same line."""
    var n = Int(n_in)
    var k = Int(k_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(m_in) * n:
        return
    var i = t // n
    var j = t - i * n
    var acc = Float32(0.0)
    for c in range(k):
        var x = ftz(a.unsafe_load(i * k + c))
        var y = ftz(b.unsafe_load(j * k + c))
        var s = ftz(x + y)
        if s != Float32(0.0):
            var d = ftz(x - y)
            acc = ftz(acc + ftz(identical_div(ftz(identical_mul(d, d)), s)))
    if exp_it != 0:
        out_k.unsafe_store(t, ftz(identical_exp(ftz(identical_mul(gain, acc)))))
    else:
        out_k.unsafe_store(t, -acc)


def cosine_rows_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    norms_sq: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    k_in: Int32,
):
    """scikit-learn's `normalize(X)` inside `cosine_similarity`: each row
    divided by its l2 norm, `ftz(x / ftz(sqrt(ftz(norm^2))))` with the
    squared norm from `row_norms_l2sq`'s ascending chain; a zero-norm row is
    left as it is (their `norms[norms == 0] = 1`). One thread per cell."""
    var k = Int(k_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(rows_in) * k:
        return
    var r = t // k
    var v = ftz(src.unsafe_load(t))
    var q = ftz(norms_sq.unsafe_load(r))
    if q == Float32(0.0):
        dst.unsafe_store(t, v)
    else:
        dst.unsafe_store(t, ftz(identical_div(v, ftz(identical_sqrt(q)))))


# ===========================================================================
# The svm call, and why FAST does not take the fused RBF tile
# ===========================================================================

#: `svm/impl/distance/kernel_matrices.mojo`'s `KM_TPB`, the block width
#: `kernel_op` launches `rbf_kernel_expanded_kernel` at. SCHEDULING, since that
#: kernel is one thread per output cell with no fold across threads.
comptime _SVM_EPILOGUE_TPB = 256


def _svm_kernel_op(
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
) raises:
    """`kernel_op`, CALLED, under IDENTICAL and DETERMINISTIC. Under FAST,
    `kernel_op`'s own non-fused lines, called one by one.

    **WHY FAST DOES NOT CALL `kernel_op`, WHICH IS COMPILE TIME.** Since
    DEVIATION 2492 (svm, 2026-09-10) `kernel_op` carries, under FAST only, a
    run-time branch into `rbf_fused_tile`, which instantiates
    `rbf_fused_tile_kernel[KPAD]` at all 16 register widths 4, 8, ..., 64,
    sixteen device kernels, each with two `comptime for` loops unrolled
    KPAD statements long (1,088 unrolled statements in all) around two
    `barrier()` phases. The branch is chosen at run time, so every build
    that NAMES `kernel_op` under FAST compiles all sixteen whatever `k` is.
    The binding builds IDENTICAL only and never compiles them, and neither
    does `cholesky/checks/cholesky_check.mojo`, which does not reach
    `kernel_op`; `km_check.mojo` under FAST did, and on the MI300X and the
    H100 its compile passed 25 minutes (legs 2026-09-14 d-followup) where
    it had compiled in all three modes on 2026-09-10 before 2492 landed.

    **WHY NO IDENTICAL OR DETERMINISTIC BIT CAN MOVE.** In those modes this
    function is the one `kernel_op` call it replaced, same arguments, same
    order; the FAST branch below is not compiled.

    **WHAT FAST RUNS INSTEAD, AND IT IS WHAT FAST RAN BEFORE 2492.** Exactly
    the lines `kernel_op` executes under FAST when its fused branch declines:
    the device count refusal, `gemm_nt` over `k`, then
    `rbf_kernel_expanded_kernel` for RBF, the refusal of any other
    non-linear kernel. The LINEAR kernel, the POLYNOMIAL and SIGMOID dots and
    the RBF-under-a-copy dot never took the fused branch (it requires
    `kp.kernel == KERNEL_RBF`), so their FAST bits are unchanged too. Only
    the production RBF kernel matrix under FAST at `k <= 64` changes route,
    from the fused tile back to `gemm_nt` plus svm's expansion epilogue,
    which is the route this file's header table and
    `check_km_sabotage_copies_agree`'s copy both describe. FAST makes no bit
    claim, and `bindings/build_kernel_methods.sh` builds IDENTICAL only, so
    no shipped bit moves.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
        if m <= 0 or n <= 0:
            return
        var setting = String(getenv("MOJOLEARN_SVM_DEVICE_COUNT"))
        if setting != "" and setting != "1":
            # `kernel_op`'s refusal, which under FAST fires on every value
            # that parses (and `Int` raises on one that does not).
            var count = Int(setting)
            _ = count
            raise Error("parallel SVM kernels require IDENTICAL and 1..64 devices")
        gemm_nt(ctx, out, a, b, m, n, k)
        if kp.kernel == KERNEL_RBF:
            ctx.enqueue_function[rbf_kernel_expanded_kernel](
                out.unsafe_ptr(), Int32(m), Int32(n),
                norm_a.unsafe_ptr(), norm_b.unsafe_ptr(), Float32(kp.gamma),
                grid_dim=(m * n + _SVM_EPILOGUE_TPB - 1) // _SVM_EPILOGUE_TPB,
                block_dim=_SVM_EPILOGUE_TPB,
            )
        elif kp.kernel != KERNEL_LINEAR:
            raise Error("svm kernel_op: unimplemented kernel " + String(kp.kernel))
    else:
        kernel_op(ctx, kp, out, a, b, m, n, k, norm_a, norm_b, ws)


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
    a_input: DeviceBuffer[DType.float32],
    b_input: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    mut norm_a: DeviceBuffer[DType.float32],
    mut norm_b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    elem_tpb: Int = KM_EPILOGUE_TPB,
    sabotage: Int = KMSAB_NONE,
    self_kernel: Bool = False,
) raises:
    """`out[m x n] = K(a_i, b_j)`, row-major, for the five implemented kernels.

    `self_kernel` (fam2-kernel-gp): the caller states `a_input` IS `b_input`
    and m == n. Read only by the KM_IDN_RBF_CELL, KM_IDN_DOT_CELL and
    KM_IDN_LAP_CELL arms, which then compute one triangle and mirror it;
    every other route ignores it.

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

    # DEVIATION 2487: read-only handles can name one input twice.
    # Views bridge legacy mutable-handle callees; the input kernels only read.
    var a = a_input.create_sub_buffer[DType.float32](0, len(a_input))
    var b = b_input.create_sub_buffer[DType.float32](0, len(b_input))
    var via_copy = km_sabotage_touches_kernel_matrix(sabotage)
    var grid_all = (m * n + elem_tpb - 1) // elem_tpb

    if kp.kernel == KM_KERNEL_LAPLACIAN:
        # No dot product and therefore no norms. Zero them so nothing
        # downstream can hash uninitialized memory and report a divergence
        # that is really an allocator.
        ctx.enqueue_memset(norm_a, Float32(0.0))
        ctx.enqueue_memset(norm_b, Float32(0.0))
        comptime if KM_IDN_LAP_CELL:
            if not via_copy:
                ctx.enqueue_function[km_laplacian_cell_kernel](
                    out.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(),
                    Int32(m), Int32(n), Int32(k), Float32(-kp.gamma),
                    Int32(1) if (self_kernel and m == n) else Int32(0),
                    grid_dim=(grid_all, 1, 1),
                    block_dim=(elem_tpb, 1, 1),
                )
                return
        # elem_tpb BY KEYWORD. `pairwise_distance` gained a `metric_arg`
        # parameter BEFORE `elem_tpb` when Minkowski landed, so this
        # positional call started handing the thread count to metric_arg.
        pairwise_distance(
            ctx, out, a, b, m, n, k, DIST_L1, elem_tpb=elem_tpb
        )
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

    if kp.kernel == KM_KERNEL_CHI2 or kp.kernel == KM_KERNEL_ADDITIVE_CHI2:
        ctx.enqueue_memset(norm_a, Float32(0.0))
        ctx.enqueue_memset(norm_b, Float32(0.0))
        ctx.enqueue_function[chi2_cell_kernel](
            out.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(),
            Int32(m), Int32(n), Int32(k), Float32(-kp.gamma),
            Int32(1) if kp.kernel == KM_KERNEL_CHI2 else Int32(0),
            grid_dim=(grid_all, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        return

    if kp.kernel == KM_KERNEL_COSINE:
        # normalize both operands, then the pinned GEMM of the normalized
        # rows (`cosine_similarity` = normalize(X) . normalize(Y)^T).
        row_norms_l2sq(ctx, norm_a, a, m, k)
        row_norms_l2sq(ctx, norm_b, b, n, k)
        var an = ctx.enqueue_create_buffer[DType.float32](m * k)
        var bn = ctx.enqueue_create_buffer[DType.float32](n * k)
        ctx.enqueue_function[cosine_rows_kernel](
            an.unsafe_ptr(), a.unsafe_ptr(), norm_a.unsafe_ptr(), Int32(m), Int32(k),
            grid_dim=((m * k + elem_tpb - 1) // elem_tpb, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        ctx.enqueue_function[cosine_rows_kernel](
            bn.unsafe_ptr(), b.unsafe_ptr(), norm_b.unsafe_ptr(), Int32(n), Int32(k),
            grid_dim=((n * k + elem_tpb - 1) // elem_tpb, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        var dot_only = KernelParams(KM_KERNEL_LINEAR, kp.degree, kp.gamma, kp.coef0)
        _svm_kernel_op(ctx, dot_only, out, an, bn, m, n, k, norm_a, norm_b, ws)
        # the normalized copies are this call's own: drain before they go
        ctx.synchronize()
        _ = an^
        _ = bn^
        return

    # Every remaining kernel starts from `a . b^T`. `row_norms_l2sq` is
    # only READ by the RBF expansion, and it is computed unconditionally
    # anyway: it is one pass over the data against an O(m n k) contraction,
    # and a buffer that is sometimes written is a buffer whose card stage
    # sometimes means something.
    comptime if KM_IDN_RBF_CELL:
        if kp.kernel == KM_KERNEL_RBF and not via_copy and k <= KM_RBF_CELL_MAX_D:
            ctx.enqueue_memset(norm_a, Float32(0.0))
            ctx.enqueue_memset(norm_b, Float32(0.0))
            ctx.enqueue_function[km_rbf_cell_kernel](
                out.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(),
                Int32(m), Int32(n), Int32(k), -Float32(kp.gamma),
                Int32(1) if (self_kernel and m == n) else Int32(0),
                grid_dim=(grid_all, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
            return

    comptime if KM_IDN_DOT_CELL:
        if (
            (kp.kernel == KM_KERNEL_POLYNOMIAL or kp.kernel == KM_KERNEL_SIGMOID)
            and not via_copy
            and k <= KM_DOT_CELL_MAX_D
        ):
            ctx.enqueue_memset(norm_a, Float32(0.0))
            ctx.enqueue_memset(norm_b, Float32(0.0))
            ctx.enqueue_function[km_dot_cell_kernel](
                out.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(),
                Int32(m), Int32(n), Int32(k),
                Float32(kp.gamma), Float32(kp.coef0), Int32(kp.degree),
                Int32(1) if kp.kernel == KM_KERNEL_POLYNOMIAL else Int32(0),
                Int32(1) if (self_kernel and m == n) else Int32(0),
                grid_dim=(grid_all, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
            return

    row_norms_l2sq(ctx, norm_a, a, m, k)
    row_norms_l2sq(ctx, norm_b, b, n, k)

    if kp.kernel == KM_KERNEL_LINEAR:
        # THEIR CODE, CALLED, and there is no epilogue to sabotage: a linear
        # kernel IS the pinned GEMM, whose own six sabotages live in the gemm
        # lane and are not this lane's to re-drive. `_svm_kernel_op` is
        # `kernel_op` outside FAST; see it for what FAST runs and why.
        _svm_kernel_op(ctx, kp, out, a, b, m, n, k, norm_a, norm_b, ws)
        return

    if kp.kernel == KM_KERNEL_RBF and not via_copy:
        # THEIR CODE, CALLED. `kernel_op` issues the pinned GEMM and svm's
        # implemented expansion epilogue in one call.
        _svm_kernel_op(ctx, kp, out, a, b, m, n, k, norm_a, norm_b, ws)
        return

    # RBF-under-a-copy, POLYNOMIAL and SIGMOID all start from the dot alone,
    # and they get it from the SAME `kernel_op` at a LINEAR parameter block.
    var dot_only = KernelParams(KM_KERNEL_LINEAR, kp.degree, kp.gamma, kp.coef0)
    _svm_kernel_op(ctx, dot_only, out, a, b, m, n, k, norm_a, norm_b, ws)

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
