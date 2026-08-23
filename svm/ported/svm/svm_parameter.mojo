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
"""


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
    if param.svmType != C_SVC:
        raise Error(
            "svm: svmType=" + String(param.svmType)
            + " is not ported (rung 1 is C_SVC only; EPSILON_SVR is rung 2,"
            + " NU_SVC/NU_SVR are unported upstream too)"
        )
    if param.epsilon != 0.0:
        raise Error(
            "svm: epsilon=" + String(param.epsilon)
            + " is the SVR parameter; C_SVC ignores it upstream, so it is"
            + " refused rather than silently dropped"
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
    if param.C <= 0.0:
        raise Error("svm: C must be positive, got " + String(param.C))
    if param.tol <= 0.0:
        raise Error("svm: tol must be positive, got " + String(param.tol))
