# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host entry points for the C-SVC and epsilon-SVR lanes, what `bindings/` calls.

Shaped like `kde/estimator.mojo` and `glm/estimator.mojo`, host lists in,
host values out, one `DeviceContext` per call, nothing retained. The
binding (`bindings/_mojolearn_svm.mojo`) marshals NumPy addresses into
these lists and back out; the wrapper is `python/mojolearn/_svm_impl.py`.

WHAT THIS LANE HAS, IN ONE SENTENCE
-----------------------------------
BINARY C-SVC AND EPSILON-SVR, dense FP32, LINEAR and RBF. The regression
half reached this file on 2026-09-01 and nothing about the solver changed
to let it in; the four entry points below are two thin pairs over one
`SmoSolver`. This paragraph said "There is no SVR ENTRY POINT here" for a
day, which was true while `svr_fit_host` did not exist, and before that it
said `svmType != C_SVC` raises, which stopped being the shape of the
boundary at `fea6becc`.

WHAT SEPARATES THE TWO PAIRS, and it is a short list. `svc_*` pins
`svmType` at `C_SVC` and `epsilon` at 0; `svr_*` pins `svmType` at
`EPSILON_SVR` and takes `epsilon` from the caller. `svc_fit_host` validates
`y` as exactly two distinct labels (the solver does, and raises upstream's
own sentence); `svr_fit_host` validates nothing about `y` but its
finiteness, because the targets are continuous and upstream hands them
straight to `Solve`. `svc_predict_host` can ask for the class epilogue;
`svr_predict_host` cannot, and has no argument for it.

THE 2 * n_rows DOMAIN DOES NOT REACH THIS BOUNDARY. `SmoSolver` solves SVR
over `n_train = 2 * n_rows` (`smosolver.mojo:443`), but `Results` folds the
two alpha halves and then selects over `n_rows` (`results.mojo:124-130`),
so `svr_fit_host` returns the same shapes `svc_fit_host` does and its
worst-case output buffers are the same size. That is stated again on
`svr_fit_host` itself, because it is the one thing a caller allocating
buffers can get wrong in a way that costs memory rather than an error.

THE MODEL CROSSES BACK AS ARRAYS, AND THAT IS DELIBERATE (DEVIATION 873)
------------------------------------------------------------------------
`bindings/_mojolearn_estimators.mojo` states the rule this lane inherits,
"all device buffers and contexts live for one call and no pointer is
retained." So `svc_fit_host` returns the whole `SvmModel` as host values
and `svc_predict_host` rebuilds one from them and uploads again; the SVR
pair does the same thing with a shorter record. The consequences, both real
and both on the class docstring:

  * `n_support` is not known until the solve finishes, so the caller has
    to hand `fit` worst-case output buffers -- `n_rows` dual coefficients,
    `n_rows` support indices and `n_rows * n_cols` support-vector floats.
    On a matrix that is mostly support vectors this is a second copy of X.
  * The support matrix is uploaded again at every `predict`. Theirs keeps
    it on the device between calls; this surface does not.

Nothing numeric changes. `svc_predict` takes exactly the model `svc_fit`
produced, `svr_predict` exactly the model `svr_fit` produced, and the round
trip through host float32 is exact.

THE IDENTITY CARD
-----------------
All four entries take `IdentityTrace()`, which reads
`MOJOLEARN_IDENTITY_TRACE` from the environment once, exactly as
`kde/estimator.mojo` does. Unset (the shipping state) it is off and costs
nothing. Set, a fit and its predicts append their stages to the one file;
the classifier fit's stages are the 32 the 2026-08-23 three-vendor round
compared (E3_RESULTS.md round 11). THE SVR STAGES HAVE NOT BEEN IN A
THREE-VENDOR ROUND. The SVR gates are 44 of 44 at `fea6becc` on the box
that ran them, and no cross-vendor card has been diffed for this header.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from svm.impl.svm.smosolver import SmoTrace
from svm.impl.svm.svc_impl import svc_fit, svc_predict
from svm.impl.svm.svr_impl import svr_fit, svr_predict
from svm.impl.svm.svm_parameter import (
    C_SVC,
    EPSILON_SVR,
    KERNEL_LINEAR,
    KERNEL_RBF,
    KernelParams,
    SvmModel,
    SvmParameter,
    check_rung1_scope,
)


struct SvcFitOutputs(Copyable, Movable):
    """One fit's `SvmModel`, flattened to host values so the binding can
    copy it into the caller's NumPy arrays. `dual_coefs` and `support_idx`
    have `n_support` entries; `support_matrix` has `n_support * n_cols`,
    row-major, in support order."""

    var n_support: Int
    var n_iter: Int
    var b: Float32
    var label0: Float32
    var label1: Float32
    """`unique_labels[0]` and `[1]`: the SORTED distinct labels. `label1`
    is the one `ovr_labels_kernel` maps to +1 (`getOvrlabels(..., idx=1)`),
    so the sign of the decision function is fixed by this pair and not by
    the order the labels appeared in `y`."""
    var dual_coefs: List[Float32]
    var support_idx: List[Int32]
    var support_matrix: List[Float32]

    def __init__(out self):
        self.n_support = 0
        self.n_iter = 0
        self.b = Float32(0.0)
        self.label0 = Float32(0.0)
        self.label1 = Float32(0.0)
        self.dual_coefs = List[Float32]()
        self.support_idx = List[Int32]()
        self.support_matrix = List[Float32]()


def _kernel_params(kernel: Int, gamma: Float64) raises -> KernelParams:
    """`ML::matrix::KernelParams` for the two ported kernels. `degree` and
    `coef0` are their constructor defaults (3 and 0); both are read only by
    POLYNOMIAL and TANH, which `check_rung1_scope` refuses by name, so
    there is no value a caller could pass that would reach a kernel."""
    if kernel != KERNEL_LINEAR and kernel != KERNEL_RBF:
        raise Error(
            "svm: kernel=" + String(kernel) + " is not ported in rung 1;"
            + " only LINEAR (" + String(KERNEL_LINEAR) + ") and RBF ("
            + String(KERNEL_RBF) + ") are (svm/NOT_IMPLEMENTED.tsv)"
        )
    return KernelParams(kernel, 3, gamma, 0.0)


def svc_fit_host(
    x: List[Float32],
    labels: List[Float32],
    n_rows: Int,
    n_cols: Int,
    kernel: Int,
    gamma: Float64,
    C: Float64,
    tol: Float64,
    max_iter: Int,
    nochange_steps: Int,
) raises -> SvcFitOutputs:
    """`SVC(C, kernel, gamma, tol, max_iter, nochange_steps).fit(X, y)`,
    one shot. `x` is ROW-MAJOR `n_rows x n_cols` (theirs is column-major;
    a layout, not an arithmetic, recorded in `svm/DERIVATION_MAP.tsv`).

    `max_outer_iter` is fixed at -1, which is what cuML's own Python layer
    does at `svm_base.pyx:371` -- it is not on their Python surface either,
    and -1 becomes `max(100000, 100 * n_train)` in the solver.

    `cache_size` is fixed at 0, the `n_cache_sets == 0` arm, because the
    `raft::cache` LRU is not ported (DEVIATION 871; `svm/NOT_IMPLEMENTED.tsv`).
    `epsilon` is fixed at 0 and `svmType` at `C_SVC`; both would raise by
    name otherwise, and neither has a value a C-SVC caller could want.

    Every other refusal is `check_rung1_scope`'s and it is called HERE,
    before the `DeviceContext` exists, so a refused parameter costs no
    device work. `svc_fit` calls it again; that is cheap and keeps the
    guarantee independent of this file.
    """
    if n_rows <= 0:
        raise Error("Parameter n_rows: number of rows cannot be less than one")
    if n_cols <= 0:
        raise Error("Parameter n_cols: number of columns cannot be less than one")
    if len(x) != n_rows * n_cols:
        raise Error(
            "svc_fit_host: X has " + String(len(x)) + " values, n_rows x n_cols is "
            + String(n_rows * n_cols)
        )
    if len(labels) != n_rows:
        raise Error(
            "svc_fit_host: y has " + String(len(labels)) + " values, n_rows is "
            + String(n_rows)
        )
    var kp = _kernel_params(kernel, gamma)
    var param = SvmParameter.default()
    param.C = C
    param.tol = tol
    param.max_iter = max_iter
    param.max_outer_iter = -1
    param.nochange_steps = nochange_steps
    param.cache_size = 0.0
    param.epsilon = 0.0
    param.svmType = C_SVC
    param.verbosity = 0
    check_rung1_scope(param, kp, False)

    var ctx = DeviceContext()
    var card = IdentityTrace()
    card.header(
        "svc_fit_host: n_rows=" + String(n_rows) + " n_cols=" + String(n_cols)
        + " kernel=" + String(kernel) + " gamma=" + String(gamma)
        + " C=" + String(C) + " tol=" + String(tol)
        + " max_iter=" + String(max_iter)
        + " nochange_steps=" + String(nochange_steps)
    )
    var trace = SmoTrace()
    var model = svc_fit(ctx, x, labels, n_rows, n_cols, param, kp, card, trace)

    var out = SvcFitOutputs()
    out.n_support = model.n_support
    out.n_iter = model.n_iter
    out.b = model.b
    out.label0 = model.unique_labels[0]
    out.label1 = model.unique_labels[1]
    out.dual_coefs = model.dual_coefs.copy()
    out.support_idx = model.support_idx.copy()
    out.support_matrix = model.support_matrix.copy()
    _ = model^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out^


def svc_predict_host(
    x: List[Float32],
    n_rows: Int,
    n_cols: Int,
    support_matrix: List[Float32],
    dual_coefs: List[Float32],
    n_support: Int,
    b: Float32,
    label0: Float32,
    label1: Float32,
    kernel: Int,
    gamma: Float64,
    predict_class: Bool,
    buffer_size_mib: Float64,
) raises -> List[Float32]:
    """`svcPredict` on a model rebuilt from `svc_fit_host`'s output.
    `predict_class` picks their `applyPrediction` epilogue (the class
    label) over the raw decision value; `buffer_size_mib` is their
    `param.cache_size` at the `SVC::predict` call site, the n_rows BATCH
    knob, and `check_device_is_launch_invariant` holds the answer fixed
    over it (arm (c) runs 0.001 MiB against 200 MiB).

    `support_idx` is NOT taken: `svc_predict` never reads it. It is
    returned by `fit` for `support_` alone.
    """
    if n_rows <= 0:
        raise Error("svc_predict_host: n_rows must be at least one")
    if n_cols <= 0:
        raise Error("svc_predict_host: n_cols must be at least one")
    if len(x) != n_rows * n_cols:
        raise Error(
            "svc_predict_host: X has " + String(len(x)) + " values, n_rows x n_cols is "
            + String(n_rows * n_cols)
        )
    if n_support < 0:
        raise Error("svc_predict_host: n_support cannot be negative")
    if len(dual_coefs) != n_support:
        raise Error(
            "svc_predict_host: dual_coefs has " + String(len(dual_coefs))
            + " values, n_support is " + String(n_support)
        )
    if len(support_matrix) != n_support * n_cols:
        raise Error(
            "svc_predict_host: support_matrix has " + String(len(support_matrix))
            + " values, n_support x n_cols is " + String(n_support * n_cols)
        )
    if not (buffer_size_mib > 0.0):
        raise Error(
            "svc_predict_host: the predict buffer (cache_size) must be a"
            " positive number of MiB, got " + String(buffer_size_mib)
        )
    var kp = _kernel_params(kernel, gamma)

    var model = SvmModel()
    model.n_support = n_support
    model.n_cols = n_cols
    model.b = b
    model.dual_coefs = dual_coefs.copy()
    model.support_matrix = support_matrix.copy()
    model.support_idx = List[Int32]()
    model.n_classes = 2
    model.unique_labels = List[Float32]()
    model.unique_labels.append(label0)
    model.unique_labels.append(label1)
    model.n_iter = 0

    var ctx = DeviceContext()
    var card = IdentityTrace()
    card.header(
        "svc_predict_host: n_rows=" + String(n_rows) + " n_cols=" + String(n_cols)
        + " n_support=" + String(n_support) + " kernel=" + String(kernel)
        + " gamma=" + String(gamma)
        + " predict_class=" + String(predict_class)
        + " buffer_mib=" + String(buffer_size_mib)
    )
    var out = svc_predict(
        ctx, model, x, n_rows, n_cols, kp, buffer_size_mib, predict_class, card
    )
    _ = model^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out^


struct SvrFitOutputs(Copyable, Movable):
    """One epsilon-SVR fit's `SvmModel`, flattened to host values so the
    binding can copy it into the caller's NumPy arrays. `dual_coefs` and
    `support_idx` have `n_support` entries; `support_matrix` has
    `n_support * n_cols`, row-major, in support order.

    NO LABEL PAIR, and that is the whole difference from `SvcFitOutputs`. A
    regressor has no classes, so there is nothing for `label0` / `label1` to
    carry and nothing at predict that would read them.
    """

    var n_support: Int
    var n_iter: Int
    var b: Float32
    var dual_coefs: List[Float32]
    var support_idx: List[Int32]
    var support_matrix: List[Float32]

    def __init__(out self):
        self.n_support = 0
        self.n_iter = 0
        self.b = Float32(0.0)
        self.dual_coefs = List[Float32]()
        self.support_idx = List[Int32]()
        self.support_matrix = List[Float32]()


def svr_fit_host(
    x: List[Float32],
    targets: List[Float32],
    n_rows: Int,
    n_cols: Int,
    kernel: Int,
    gamma: Float64,
    C: Float64,
    epsilon: Float64,
    tol: Float64,
    max_iter: Int,
    nochange_steps: Int,
) raises -> SvrFitOutputs:
    """`SVR(C, epsilon, kernel, gamma, tol, max_iter, nochange_steps).fit(X,
    y)`, one shot. `x` is ROW-MAJOR `n_rows x n_cols` (theirs is
    column-major; a layout, not an arithmetic, recorded in
    `svm/DERIVATION_MAP.tsv`).

    `targets` are CONTINUOUS VALUES, one per row. Nothing here validates
    them as `+-1` and nothing sorts them into a label pair, because
    `svr_impl.cuh:68` hands `y` straight to `SmoSolver::Solve`. The only
    check they get is DEVIATION 636's, every cell finite, and it is the
    solver's own.

    WHAT THE CALLER MUST ALLOCATE, AND WHY IT IS NOT DOUBLED
    ---------------------------------------------------------
    Exactly what a C-SVC caller allocates:

        dual_coefs      n_rows          float32
        support_idx     n_rows          int32
        support_matrix  n_rows * n_cols float32

    Worst case, because `n_support` is not known until the solve finishes
    (DEVIATION 873), and only the first `n_support` (and
    `n_support * n_cols`) entries are written. Every one of those sizes is a
    function of `(n_rows, n_cols)` alone, so a Python caller can compute all
    three before the call.

    THE 2 * n_rows DOMAIN IS INTERNAL. The solver carries alpha+ and alpha-
    as one `n_train = 2 * n_rows` vector, and `WorkingSet`, `SmoSolver` and
    `Results` are all sized by it. But `Results::combine_coefs` folds the
    two halves (`coef_i = alpha_i - alpha*_i`, spelled upstream as an add
    because `y` is `[+1]*n ++ [-1]*n`) and `get_dual_coefs`,
    `get_support_vector_indices` and `collect_support_vector_matrix` then
    select over `n_rows`. So `n_support <= n_rows`, never `2 * n_rows`, and
    a caller sizing at `2 * n_rows` allocates twice what can be written.

    `max_outer_iter` is fixed at -1, which is what cuML's own Python layer
    does at `svm_base.pyx:371`, and -1 becomes `max(100000, 100 * n_train)`
    in the solver, over the DOUBLED `n_train` here.

    `cache_size` is fixed at 0, the `n_cache_sets == 0` arm, because the
    `raft::cache` LRU is not ported (DEVIATION 871;
    `svm/NOT_IMPLEMENTED.tsv`). `svmType` is fixed at `EPSILON_SVR`.
    `epsilon` is the caller's, unlike the classifier's pair, and it is
    passed through UNCLAMPED so that `check_rung1_scope`'s two refusals, not
    finite and negative, stay reachable from every surface above this one.

    Every other refusal is `check_rung1_scope`'s and it is called HERE,
    before the `DeviceContext` exists, so a refused parameter costs no
    device work. `svr_fit` calls it again; that is cheap and keeps the
    guarantee independent of this file.
    """
    if n_rows <= 0:
        raise Error("Parameter n_rows: number of rows cannot be less than one")
    if n_cols <= 0:
        raise Error("Parameter n_cols: number of columns cannot be less than one")
    if len(x) != n_rows * n_cols:
        raise Error(
            "svr_fit_host: X has " + String(len(x)) + " values, n_rows x n_cols is "
            + String(n_rows * n_cols)
        )
    if len(targets) != n_rows:
        raise Error(
            "svr_fit_host: y has " + String(len(targets)) + " values, n_rows is "
            + String(n_rows)
        )
    var kp = _kernel_params(kernel, gamma)
    var param = SvmParameter.default()
    param.C = C
    param.tol = tol
    param.max_iter = max_iter
    param.max_outer_iter = -1
    param.nochange_steps = nochange_steps
    param.cache_size = 0.0
    param.epsilon = epsilon
    param.svmType = EPSILON_SVR
    param.verbosity = 0
    check_rung1_scope(param, kp, False)

    var ctx = DeviceContext()
    var card = IdentityTrace()
    card.header(
        "svr_fit_host: n_rows=" + String(n_rows) + " n_cols=" + String(n_cols)
        + " kernel=" + String(kernel) + " gamma=" + String(gamma)
        + " C=" + String(C) + " epsilon=" + String(epsilon)
        + " tol=" + String(tol)
        + " max_iter=" + String(max_iter)
        + " nochange_steps=" + String(nochange_steps)
    )
    var trace = SmoTrace()
    var model = svr_fit(ctx, x, targets, n_rows, n_cols, param, kp, card, trace)

    var out = SvrFitOutputs()
    out.n_support = model.n_support
    out.n_iter = model.n_iter
    out.b = model.b
    out.dual_coefs = model.dual_coefs.copy()
    out.support_idx = model.support_idx.copy()
    out.support_matrix = model.support_matrix.copy()
    _ = model^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out^


def svr_predict_host(
    x: List[Float32],
    n_rows: Int,
    n_cols: Int,
    support_matrix: List[Float32],
    dual_coefs: List[Float32],
    n_support: Int,
    b: Float32,
    kernel: Int,
    gamma: Float64,
    buffer_size_mib: Float64,
) raises -> List[Float32]:
    """`SVR::predict` on a model rebuilt from `svr_fit_host`'s output. One
    float32 per row, `sum_j K(x, sv_j) dual_j + b`.

    NO `predict_class`, because a regressor has no epilogue to choose, and
    NO label pair, because there are no classes to map onto. The two dummy
    labels the launch needs are written here and are never read;
    `svm/impl/svm/svr_impl.mojo`'s header says why they exist at all.

    `buffer_size_mib` is their `param.cache_size` at the `SVM::predict` call
    site, the n_rows BATCH knob, and `check_svr_device_is_launch_invariant`
    holds the answer fixed over it on the regression path specifically (its
    arm (c) runs 0.001 MiB against 200 MiB, and the regression path has
    twice the launches to be invariant over because `UpdateF` runs on both
    halves).

    `support_idx` is NOT taken: `svc_predict` never reads it. It is returned
    by `fit` for `support_` alone.
    """
    if n_rows <= 0:
        raise Error("svr_predict_host: n_rows must be at least one")
    if n_cols <= 0:
        raise Error("svr_predict_host: n_cols must be at least one")
    if len(x) != n_rows * n_cols:
        raise Error(
            "svr_predict_host: X has " + String(len(x)) + " values, n_rows x n_cols is "
            + String(n_rows * n_cols)
        )
    if n_support < 0:
        raise Error("svr_predict_host: n_support cannot be negative")
    if len(dual_coefs) != n_support:
        raise Error(
            "svr_predict_host: dual_coefs has " + String(len(dual_coefs))
            + " values, n_support is " + String(n_support)
        )
    if len(support_matrix) != n_support * n_cols:
        raise Error(
            "svr_predict_host: support_matrix has " + String(len(support_matrix))
            + " values, n_support x n_cols is " + String(n_support * n_cols)
        )
    if not (buffer_size_mib > 0.0):
        raise Error(
            "svr_predict_host: the predict buffer (cache_size) must be a"
            " positive number of MiB, got " + String(buffer_size_mib)
        )
    var kp = _kernel_params(kernel, gamma)

    var model = SvmModel()
    model.n_support = n_support
    model.n_cols = n_cols
    model.b = b
    model.dual_coefs = dual_coefs.copy()
    model.support_matrix = support_matrix.copy()
    model.support_idx = List[Int32]()
    model.n_classes = 0
    var dummy_labels: List[Float32] = [Float32(-1.0), Float32(1.0)]
    model.unique_labels = dummy_labels^
    model.n_iter = 0

    var ctx = DeviceContext()
    var card = IdentityTrace()
    card.header(
        "svr_predict_host: n_rows=" + String(n_rows) + " n_cols=" + String(n_cols)
        + " n_support=" + String(n_support) + " kernel=" + String(kernel)
        + " gamma=" + String(gamma)
        + " buffer_mib=" + String(buffer_size_mib)
    )
    var out = svr_predict(
        ctx, model, x, n_rows, n_cols, kp, buffer_size_mib, card
    )
    _ = model^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return out^
