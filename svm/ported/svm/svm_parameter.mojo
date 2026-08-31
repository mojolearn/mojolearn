# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`SvmParameter`, `SvmModel`, `KernelParams`: the parameter and model records.

PORT OF `cuml/cpp/include/cuml/svm/svm_parameter.h`, `svm_model.h` and
`cuml/cpp/include/cuml/matrix/kernel_params.hpp` at cuML v26.08.00.
Every field of theirs is here; every field we do not honor RAISES BY NAME
in `check_rung1_scope` (called by `svcFit`), never silently ignored.

What the record carries that the brief's rung 1 does not use, and what
happens to it:

    svmType != C_SVC       raised by name (SVR is rung 2, NU_* unported)
    epsilon                honored only as SVR's parameter: raised if non-zero
                           with C_SVC, exactly because C_SVC ignores it and a
                           silently ignored parameter is the failure mode
    cache_size != 0        raised by name (the LRU cache; README "cache decision")
    verbosity              accepted and ignored: it selects LOG LINES in theirs
                           (`CUML_LOG_DEBUG`), we print none
    kernel POLYNOMIAL,     raised by name (TANH has no identical_tanh;
      TANH, PRECOMPUTED    POLYNOMIAL is one identical_pow away and is left
                           unported rather than written blind)
    degree, coef0          only read by the two refused kernels
    sample_weight          the Solve/InitPenalty argument; raised by name

# =========================================================================
# DEVIATION 636 (IDENTITY_PATHS row 39, FACT 2): NON-FINITE inputs are
# REFUSED BY NAME. Theirs validates none of `C`, `tol`, `gamma`, `X`,
# `labels` (the C++ entry points take what they are given; a NaN surfaces,
# if at all, as the "NaN found during fitting" throw of
# `CheckStoppingCondition`). Ours raises, naming the parameter, on: `C`
# not finite, `tol` not finite (`tol = NaN` passes their `tol > 0` and
# never stops), RBF `gamma` not finite or negative (`gamma = inf` makes
# `-inf * 0 = NaN` on the diagonal of the kernel matrix; a negative gamma
# is an exp that overflows to inf and then to NaN in f; sklearn's own
# constraint is `gamma >= 0`), and any non-finite cell of `X` or `labels`
# (`svc_impl.mojo`, fit and predict). Why: a computed NaN carries a
# VENDOR-SPECIFIC payload (Apple 0x7fc00000, NVIDIA 0x7fffffff, AMD
# 0xffc00000), so a NaN can never sit in a card-hashed stage, and every
# one of these inputs puts one there (`svm.iter000.f` at the latest).
# Refusing is the guard; the stages stay raw bits. Gated by
# `svc_check.mojo::check_refusals` (each named) and `check_nan_never_recorded`.
# =========================================================================
"""

from std.math import isfinite


comptime C_SVC = 0
comptime NU_SVC = 1
comptime EPSILON_SVR = 2
comptime NU_SVR = 3

comptime KERNEL_LINEAR = 0
comptime KERNEL_POLYNOMIAL = 1
comptime KERNEL_RBF = 2
comptime KERNEL_TANH = 3
comptime KERNEL_PRECOMPUTED = 4


@fieldwise_init
struct KernelParams(Copyable, Movable):
    """`ML::matrix::KernelParams {kernel, degree, gamma, coef0}`."""

    var kernel: Int
    var degree: Int
    var gamma: Float64
    var coef0: Float64

    @staticmethod
    def linear() -> Self:
        """`KernelParams{LINEAR, 3, 1, 0}`, the `SVC` constructor default
        (`svc.hpp`)."""
        return Self(KERNEL_LINEAR, 3, 1.0, 0.0)

    @staticmethod
    def rbf(gamma: Float64) -> Self:
        return Self(KERNEL_RBF, 3, gamma, 0.0)


@fieldwise_init
struct SvmParameter(Copyable, Movable):
    """`ML::SVM::SvmParameter`. Field order and defaults are the python
    binding's (`svm_base.pyx:364-372`) and the `SVC` constructor's
    (`svc.cu:121-132`): C = 1, cache_size (ours defaults to 0, see the
    module docstring), max_outer_iter = -1, max_iter = -1, nochange_steps =
    1000, tol = 1e-3, verbosity, epsilon = 0, svmType = C_SVC."""

    var C: Float64
    var cache_size: Float64
    var max_outer_iter: Int
    var max_iter: Int
    var nochange_steps: Int
    var tol: Float64
    var verbosity: Int
    var epsilon: Float64
    var svmType: Int

    @staticmethod
    def default() -> Self:
        return Self(1.0, 0.0, -1, -1, 1000, 1.0e-3, 0, 0.0, C_SVC)


struct SvmModel(Movable):
    """`ML::SVM::SvmModel<float>` with DENSE support storage only
    (`SupportStorage.data`, `nnz = -1`). Host copies of the device arrays:
    `svmFreeBuffers` is the scope exit."""

    var n_support: Int
    var n_cols: Int
    var b: Float32
    var dual_coefs: List[Float32]
    var support_matrix: List[Float32]
    var support_idx: List[Int32]
    var n_classes: Int
    var unique_labels: List[Float32]
    var n_iter: Int

    def __init__(out self):
        self.n_support = 0
        self.n_cols = 0
        self.b = Float32(0.0)
        self.dual_coefs = List[Float32]()
        self.support_matrix = List[Float32]()
        self.support_idx = List[Int32]()
        self.n_classes = 0
        self.unique_labels = List[Float32]()
        self.n_iter = 0


def check_rung1_scope(
    param: SvmParameter, kp: KernelParams, has_sample_weight: Bool
) raises:
    """Every parameter of theirs is honored or refused by NAME. Rung 1."""
    # RUNG 2 ADMITS EPSILON_SVR (2026-08-31). This block used to refuse every
    # svmType but C_SVC and refuse a non-zero `epsilon` outright. Upstream
    # solves BOTH problems with the same SMO -- `smoblocksolve.cuh:99-101`
    # says so, and takes `svmType` as a parameter its body never reads -- so
    # the two differ in the gradient initialization, the domain size and how
    # the coefficients are combined, not in the solver.
    #
    # NU_SVC and NU_SVR stay refused, and that is upstream's own boundary
    # rather than ours: `smosolver.cuh`'s Initialize has no arm for them
    # either.
    if param.svmType != C_SVC and param.svmType != EPSILON_SVR:
        raise Error(
            "svm: svmType=" + String(param.svmType)
            + " is not ported (C_SVC and EPSILON_SVR are; NU_SVC/NU_SVR are"
            + " unported upstream too)"
        )
    # `epsilon` IS THE SVR PARAMETER AND ONLY THAT. C_SVC ignores it upstream,
    # so a non-zero value on a classifier is still refused rather than
    # silently dropped; on a regressor it is required to be finite and
    # non-negative, which is DEVIATION 636's family (NaN fails `< 0.0` and
    # would otherwise pass).
    if param.svmType == C_SVC:
        if param.epsilon != 0.0:
            raise Error(
                "svm: epsilon=" + String(param.epsilon)
                + " is the SVR parameter; C_SVC ignores it upstream, so it is"
                + " refused rather than silently dropped"
            )
    else:
        if not isfinite(param.epsilon):
            raise Error(
                "svm: epsilon must be finite for EPSILON_SVR, got "
                + String(param.epsilon) + " (DEVIATION 636)"
            )
        if param.epsilon < 0.0:
            raise Error(
                "svm: epsilon must be non-negative for EPSILON_SVR, got "
                + String(param.epsilon)
            )
    if param.cache_size != 0.0:
        raise Error(
            "svm: cache_size=" + String(param.cache_size)
            + " MiB: the raft::cache LRU is not ported in rung 1; pass 0 (their"
            + " n_cache_sets == 0 path, taken exactly). See svm/UNPORTED.tsv"
        )
    if kp.kernel == KERNEL_POLYNOMIAL:
        raise Error("svm: kernel=POLYNOMIAL is not ported in rung 1 (degree, coef0 unused)")
    if kp.kernel == KERNEL_TANH:
        raise Error("svm: kernel=TANH is not ported in rung 1 (coef0 unused)")
    if kp.kernel == KERNEL_PRECOMPUTED:
        raise Error("svm: kernel=PRECOMPUTED is not ported in rung 1")
    if kp.kernel != KERNEL_LINEAR and kp.kernel != KERNEL_RBF:
        raise Error("svm: unknown kernel " + String(kp.kernel))
    if has_sample_weight:
        raise Error("svm: sample_weight is not ported in rung 1 (InitPenalty's weighted arm)")
    # DEVIATION 636: NaN fails `<= 0.0` and would pass; ask for finite first.
    if not isfinite(param.C):
        raise Error("svm: C must be finite, got " + String(param.C) + " (DEVIATION 636)")
    if param.C <= 0.0:
        raise Error("svm: C must be positive, got " + String(param.C))
    if not isfinite(param.tol):
        raise Error("svm: tol must be finite, got " + String(param.tol) + " (DEVIATION 636)")
    if param.tol <= 0.0:
        raise Error("svm: tol must be positive, got " + String(param.tol))
    if kp.kernel == KERNEL_RBF:
        if not isfinite(kp.gamma) or kp.gamma < 0.0:
            raise Error(
                "svm: gamma must be finite and >= 0 for the RBF kernel, got "
                + String(kp.gamma) + " (DEVIATION 636)"
            )


def check_finite_list(values: List[Float32], what: String) raises:
    """DEVIATION 636: every cell of `what` (`X`, `labels`, the predict `X`)
    must be finite; raises naming the first offending index."""
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error(
                "svm: " + what + " contains a non-finite value at flat index "
                + String(i) + " (DEVIATION 636: a NaN or inf input cannot be"
                " fitted; a computed NaN has a vendor-specific payload)"
            )
