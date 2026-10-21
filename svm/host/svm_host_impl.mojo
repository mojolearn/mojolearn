# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host entries of `bindings/_mojolearn_svm_host.mojo`: the SVC and SVR
fits and predictions over the SMO oracle and the isolation forest run, with
no Python import. The binding unpacks its PythonObject arguments, calls one
function here and converts the result back (lane thin-core, 2026-10-21: the
bodies moved word for word, no bit change)."""
from std.math import isnan

from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from core.host_predict_threads import host_predict_task_count
from checks.numerics import ftz
from isolation_forest.checks.if_oracle import (
    OracleForest,
    oracle_fit,
    oracle_path_lengths,
    oracle_scores,
)
from isolation_forest.estimator import (
    IF_WANT_DECISION_FUNCTION,
    IF_WANT_PREDICT,
    IF_WANT_SCORE_SAMPLES,
    percentile_linear_host,
)
from isolation_forest.impl.isolation_forest import (
    IF_params,
    check_finite_by_name_host,
)
from isolation_forest.impl.rng.xorwow import build_xorwow_tables
from svm.host.smo_oracle import (
    OracleResult,
    smo_oracle_decision_into,
    smo_oracle_fit,
)
from svm.impl.svc_impl import unique_labels_sorted
from svm.impl.svm_parameter import (
    C_SVC,
    EPSILON_SVR,
    KERNEL_LINEAR,
    KERNEL_POLYNOMIAL,
    KERNEL_PRECOMPUTED,
    KERNEL_RBF,
    KERNEL_TANH,
    KernelParams,
    SvmParameter,
    check_finite_list,
    check_rung1_scope,
)
from svm.host.finite_scan import check_finite_ptr

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def _kernel_params(
    kernel: Int, gamma: Float64, degree: Int = 3, coef0: Float64 = 0.0
) raises -> KernelParams:
    """`svm/estimator.mojo::_kernel_params`, the same refusal and the same
    constructor defaults (degree 3, coef0 0, read by no implemented
    kernel)."""
    if kernel != KERNEL_LINEAR and kernel != KERNEL_RBF and kernel != KERNEL_POLYNOMIAL and kernel != KERNEL_TANH and kernel != KERNEL_PRECOMPUTED:
        raise Error(
            "svm: kernel=" + String(kernel) + " is not implemented in rung 1;"
            + " only LINEAR (" + String(KERNEL_LINEAR) + ") and RBF ("
            + String(KERNEL_RBF) + ") are (svm/NOT_IMPLEMENTED.tsv)"
        )
    return KernelParams(kernel, degree, gamma, coef0)


def _iforest_host_scores(
    forest: OracleForest, x: List[Float32], n_rows: Int, n_cols: Int
) raises -> List[Float32]:
    """`_score_samples_device`'s guards in its words, then the oracle's
    path lengths and PAPER scores (1 = anomaly, 0.5 = normal; the Python
    layer's negation is the caller's)."""
    if n_rows <= 0:
        raise Error("Invalid n_rows " + String(n_rows))
    if n_cols != forest.n_features:
        raise Error(
            "X_query has "
            + String(n_cols)
            + " features, the model was fitted with "
            + String(forest.n_features)
        )
    check_finite_by_name_host("X_query", x, n_rows, n_cols)
    var pl = oracle_path_lengths(forest, x, n_rows, n_cols)
    return oracle_scores(forest, pl)


def svc_fit_host(
    xp: _F32P, labels: List[Float32], c_rows: List[Float32],
    dp: _F32P, sip: _I32P, smp: _F32P, ip: _F64P,
    n_rows: Int, n_cols: Int, kernel: Int, gamma: Float64, c: Float64,
    tol: Float64, max_iter: Int, nochange_steps: Int, degree: Int, coef0: Float64,
) raises -> Int:
    """`SVC.fit` on the host by `smo_oracle_fit` (the binding `svc_fit`'s
    contract): writes the support, dual coefficients and five info words.
    Returns `n_support`."""
    var n_support = 0
    # `svc_fit_host_borrowed`'s guards and parameter pins, in its order.
    if n_rows <= 0:
        raise Error("Parameter n_rows: number of rows cannot be less than one")
    if n_cols <= 0:
        raise Error("Parameter n_cols: number of columns cannot be less than one")
    if len(labels) != n_rows:
        raise Error(
            "svc_fit_host: y has " + String(len(labels)) + " values, n_rows is "
            + String(n_rows)
        )
    var kp = _kernel_params(kernel, gamma, degree, coef0)
    var param = SvmParameter.default()
    param.C = c
    param.tol = tol
    param.max_iter = max_iter
    param.max_outer_iter = -1
    param.nochange_steps = nochange_steps
    param.cache_size = 0.0
    param.epsilon = 0.0
    param.svmType = C_SVC
    param.verbosity = 0
    check_rung1_scope(param, kp, False)
    # `svc_fit_borrowed`'s, then `_svc_label_model`'s.
    check_rung1_scope(param, kp, False)
    check_finite_ptr(xp, n_rows * n_cols, "X")
    check_finite_list(labels, "labels")
    var unique = unique_labels_sorted(labels)
    if len(unique) != 2:
        raise Error(
            "Only binary classification is implemented at the moment (got "
            + String(len(unique)) + " classes)"
        )
    var label0 = unique[0]
    var label1 = unique[1]
    # `ovr_labels_kernel`: +1 where the label equals the LARGER one.
    var y = List[Float32]()
    y.reserve(n_rows)
    for i in range(n_rows):
        y.append(Float32(1.0) if labels[i] == label1 else Float32(-1.0))
    var x = read_f32(Int(xp), n_rows * n_cols)
    # THE ONE CALL THAT COMPUTES ANYTHING.
    var res = smo_oracle_fit[DType.float32](x, y, n_rows, n_cols, param, kp, c_rows=c_rows)
    if isnan(res.b):
        # DEVIATION 637, the device's refusal in its words.
        raise Error(
            "SMO error: NaN found during fitting (DEVIATION 637: the"
            " intercept b is NaN, floating point overflow in f)"
        )
    n_support = len(res.dual_coefs)
    if len(res.support_idx) != n_support or n_support > n_rows:
        raise Error("svc_fit: the host oracle returned support of an unexpected length; nothing written")
    for j in range(n_support):
        dp[j] = res.dual_coefs[j]
        sip[j] = res.support_idx[j]
        # `CollectSupportVectorMatrix`: row `support_idx[j]` of X.
        var r = Int(res.support_idx[j])
        if r < 0 or r >= n_rows:
            raise Error("svc_fit: the host oracle returned a support index out of range; nothing written")
        for col in range(n_cols):
            smp[j * n_cols + col] = x[r * n_cols + col]
    ip[0] = Float64(res.b)
    ip[1] = Float64(n_support)
    ip[2] = Float64(res.n_iter)
    ip[3] = Float64(label0)
    ip[4] = Float64(label1)
    return n_support


def svc_predict_host(
    x_address: Int, op: _F32P, n_rows: Int, n_cols: Int, n_support: Int,
    b: Float32, label0: Float32, label1: Float32, kernel: Int, gamma: Float64,
    predict_class: Bool, buffer_mib: Float64, degree: Int, coef0: Float64,
    dual_address: Int, support_address: Int,
) raises:
    """`SVC.decision_function` / `SVC.predict` on the host (the binding
    `svc_predict`'s contract): writes `n_rows` float32 to `op`."""
    # `svc_predict_host`'s guards, in its order and words.
    if n_rows <= 0:
        raise Error("svc_predict_host: n_rows must be at least one")
    if n_cols <= 0:
        raise Error("svc_predict_host: n_cols must be at least one")
    if kernel == KERNEL_PRECOMPUTED:
        # X IS the n_rows x n_support cross-kernel (the caller gathered
        # the support columns); no support rows are read.
        if n_support > 0 and n_cols != n_support:
            raise Error(
                "svc_predict_host: kernel='precomputed' needs X with one"
                " column per support vector"
            )
    if not (buffer_mib > 0.0):
        raise Error(
            "svc_predict_host: the predict buffer (cache_size) must be a"
            " positive number of MiB, got " + String(buffer_mib)
        )
    var kp = _kernel_params(kernel, gamma, degree, coef0)
    var xp = f32_ptr(x_address)
    if n_support == 0:
        for i in range(n_rows):
            op[i] = label0 if predict_class and b < Float32(0.0) else (
                label1 if predict_class else b
            )
    else:
        # Borrow all three arrays and write decisions directly into the
        # caller's output. Class prediction safely maps those completed
        # rows in place; no model/query/output List is materialized.
        smo_oracle_decision_into(
            f32_ptr(dual_address), f32_ptr(support_address), n_support,
            xp, n_rows, n_cols, kp, b, op,
            host_predict_task_count(n_rows),
        )
        if predict_class:
            for i in range(n_rows):
                op[i] = label0 if op[i] < Float32(0.0) else label1


def svr_fit_host(
    xp: _F32P, targets: List[Float32], c_rows: List[Float32],
    dp: _F32P, sip: _I32P, smp: _F32P, ip: _F64P,
    n_rows: Int, n_cols: Int, kernel: Int, gamma: Float64, c: Float64,
    epsilon: Float64, tol: Float64, max_iter: Int, nochange_steps: Int,
    degree: Int, coef0: Float64,
) raises -> Int:
    """`SVR.fit` on the host by `smo_oracle_fit` under EPSILON_SVR (the
    binding `svr_fit`'s contract). Returns `n_support`."""
    var n_support = 0
    # `svr_fit_host`'s guards and parameter pins, in its order.
    if n_rows <= 0:
        raise Error("Parameter n_rows: number of rows cannot be less than one")
    if n_cols <= 0:
        raise Error("Parameter n_cols: number of columns cannot be less than one")
    if len(targets) != n_rows:
        raise Error(
            "svr_fit_host: y has " + String(len(targets)) + " values, n_rows is "
            + String(n_rows)
        )
    var kp = _kernel_params(kernel, gamma, degree, coef0)
    var param = SvmParameter.default()
    param.C = c
    param.tol = tol
    param.max_iter = max_iter
    param.max_outer_iter = -1
    param.nochange_steps = nochange_steps
    param.cache_size = 0.0
    param.epsilon = epsilon
    param.svmType = EPSILON_SVR
    param.verbosity = 0
    check_rung1_scope(param, kp, False)
    # `svr_fit`'s: the scope again, then DEVIATION 636's finite scans.
    check_rung1_scope(param, kp, False)
    check_finite_ptr(xp, n_rows * n_cols, "X")
    check_finite_list(targets, "labels")
    var x = read_f32(Int(xp), n_rows * n_cols)
    # THE ONE CALL THAT COMPUTES ANYTHING. `y` is the regression
    # targets; the oracle builds the +-1 label vector and the gradient
    # as `SvrInit` does.
    var res = smo_oracle_fit[DType.float32](x, targets, n_rows, n_cols, param, kp, c_rows=c_rows)
    if isnan(res.b):
        # DEVIATION 637, the device's refusal in its words.
        raise Error(
            "SMO error: NaN found during fitting (DEVIATION 637: the"
            " intercept b is NaN, floating point overflow in f)"
        )
    n_support = len(res.dual_coefs)
    if len(res.support_idx) != n_support or n_support > n_rows:
        raise Error("svr_fit: the host oracle returned support of an unexpected length; nothing written")
    for j in range(n_support):
        var r = Int(res.support_idx[j])
        if r < 0 or r >= n_rows:
            raise Error("svr_fit: the host oracle returned a support index out of range; nothing written")
    for j in range(n_support):
        dp[j] = res.dual_coefs[j]
        sip[j] = res.support_idx[j]
        # `CollectSupportVectorMatrix`: row `support_idx[j]` of X.
        var r = Int(res.support_idx[j])
        for col in range(n_cols):
            smp[j * n_cols + col] = x[r * n_cols + col]
    ip[0] = Float64(res.b)
    ip[1] = Float64(n_support)
    ip[2] = Float64(res.n_iter)
    return n_support


def svr_predict_host(
    x_address: Int, op: _F32P, n_rows: Int, n_cols: Int, n_support: Int,
    b: Float32, kernel: Int, gamma: Float64, buffer_mib: Float64,
    degree: Int, coef0: Float64, dual_address: Int, support_address: Int,
) raises:
    """`SVR.predict` on the host (the binding `svr_predict`'s contract):
    writes `n_rows` float32 to `op`."""
    # `svr_predict_host`'s guards, in its order and words.
    if n_rows <= 0:
        raise Error("svr_predict_host: n_rows must be at least one")
    if n_cols <= 0:
        raise Error("svr_predict_host: n_cols must be at least one")
    if kernel == KERNEL_PRECOMPUTED:
        # X IS the n_rows x n_support cross-kernel (the caller gathered
        # the support columns); no support rows are read.
        if n_support > 0 and n_cols != n_support:
            raise Error(
                "svr_predict_host: kernel='precomputed' needs X with one"
                " column per support vector"
            )
    if not (buffer_mib > 0.0):
        raise Error(
            "svr_predict_host: the predict buffer (cache_size) must be a"
            " positive number of MiB, got " + String(buffer_mib)
        )
    var kp = _kernel_params(kernel, gamma, degree, coef0)
    if n_support == 0:
        for i in range(n_rows):
            op[i] = b
    else:
        smo_oracle_decision_into(
            f32_ptr(dual_address), f32_ptr(support_address), n_support,
            f32_ptr(x_address), n_rows, n_cols, kp, b, op,
            host_predict_task_count(n_rows),
        )


@fieldwise_init
struct IForestHostRun(Movable):
    """What `iforest_run_host` hands back: the float32 outputs (score_samples
    or decision_function), the int32 labels (predict), `offset_` and the
    resolved `max_samples_`."""
    var values: List[Float32]
    var labels: List[Int32]
    var offset_: Float64
    var max_samples_: Int


def iforest_run_host(
    train_address: Int, query_address: Int, n_train: Int, n_features: Int,
    n_query: Int, n_estimators: Int, max_samples_mode: Int, max_samples_int: Int,
    max_samples_frac: Float64, max_depth: Int, max_features_mode: Int,
    max_features_int: Int, max_features_frac: Float64, bootstrap: Bool,
    random_state: Int, contamination_auto: Bool, contamination: Float64, want: Int,
) raises -> IForestHostRun:
    """`IsolationForest.fit` plus ONE of `score_samples`, `decision_function`
    or `predict` on the host (the binding `iforest_run`'s contract)."""
    var values = List[Float32]()
    var labels = List[Int32]()
    var offset_ = Float64(-0.5)
    var max_samples_ = 0
    # `iforest_run_host`'s remaining guard.
    if want < IF_WANT_SCORE_SAMPLES or want > IF_WANT_PREDICT:
        raise Error(
            "iforest_run_host: want=" + String(want) + " is not one of 0"
            " (score_samples), 1 (decision_function), 2 (predict)"
        )
    var train = read_f32(train_address, n_train * n_features)
    var query = read_f32(query_address, n_query * n_features)
    # THE DEVICE STAGES EVERY INPUT CELL THROUGH `ftz` AT UPLOAD
    # (`_upload_f32`, `_upload_rowmajor_as_colmajor` in
    # `isolation_forest/impl/isolation_forest.mojo`; DEVIATION 1942 row),
    # for the training matrix and for every query, so a denormal input
    # is a signed zero to every kernel. The oracle reads its lists raw,
    # so the flush is applied here, once, at the same boundary. MEASURED
    # 2026-09-14: without it the CPU column's iforest/denormal train and
    # infer cells diverged from all three GPU columns on `scores` (the
    # GPU columns' denormal and denormal_ftz hashes are equal), the other
    # eight fixtures identical.
    for i in range(n_train * n_features):
        train[i] = ftz(train[i])
    for i in range(n_query * n_features):
        query[i] = ftz(query[i])
    # `IsolationForestEstimator.fit` (`:616-712`), in its order.
    var actual_max_features: Int
    if max_features_mode == 1:
        if max_features_int < 1 or max_features_int > n_features:
            raise Error(
                "max_features must be an int in [1, n_features] or a float in (0.0, 1.0]."
            )
        actual_max_features = max_features_int
    else:
        if max_features_frac <= 0.0 or max_features_frac > 1.0:
            raise Error(
                "max_features must be an int in [1, n_features] or a float in (0.0, 1.0]."
            )
        actual_max_features = Int(max_features_frac * Float64(n_features))
        if actual_max_features < 1:
            actual_max_features = 1
    var use_quantile = False
    if not contamination_auto:
        if contamination <= 0.0 or contamination > 0.5:
            raise Error(
                "contamination must be 'auto' or a float in the range (0.0, 0.5]."
            )
        use_quantile = True
    var actual_max_samples: Int
    if max_samples_mode == 0:
        actual_max_samples = 256 if n_train > 256 else n_train
    elif max_samples_mode == 1:
        if max_samples_int <= 0:
            raise Error("max_samples must be a positive integer.")
        actual_max_samples = max_samples_int if max_samples_int < n_train else n_train
    else:
        if max_samples_frac <= 0.0 or max_samples_frac > 1.0:
            raise Error("float max_samples must be in (0.0, 1.0].")
        actual_max_samples = Int(max_samples_frac * Float64(n_train))
        if actual_max_samples < 1:
            raise Error(
                "max_samples resolves to 0 samples; increase max_samples or the number of rows."
            )
    max_samples_ = actual_max_samples
    if random_state < 0 or random_state >= 4294967296:
        raise Error(
            "Expected `0 <= random_state <= 2**32 - 1`, got " + String(random_state)
        )
    var if_params = IF_params.default()
    if_params.n_estimators = n_estimators
    if_params.max_samples = actual_max_samples
    if_params.max_depth = max_depth if max_depth > 0 else -1
    if_params.max_features = actual_max_features
    if_params.bootstrap = bootstrap
    if_params.seed = UInt64(random_state)
    # `IsolationForest.error_checking` (`:538-549`), then DEVIATION 680's
    # finite scan (the borrowed path's threaded scan raises the same
    # words), then the fit.
    if if_params.n_estimators <= 0:
        raise Error(
            "n_estimators must be > 0, got " + String(if_params.n_estimators)
        )
    check_finite_by_name_host("X", train, n_train, n_features)
    var tables = build_xorwow_tables()
    var forest = oracle_fit(train, n_train, n_features, if_params, tables)
    if use_quantile:
        var paper_train = _iforest_host_scores(forest, train, n_train, n_features)
        var training_scores = List[Float32](capacity=n_train)
        for i in range(n_train):
            training_scores.append(-paper_train[i])
        offset_ = percentile_linear_host(training_scores, 100.0 * contamination)
    else:
        offset_ = -0.5
    var paper = _iforest_host_scores(forest, query, n_query, n_features)
    if want == IF_WANT_PREDICT:
        var threshold = Float32(-offset_)
        for i in range(n_query):
            var raw = Int32(1) if paper[i] > threshold else Int32(-1)
            labels.append(-raw)
    elif want == IF_WANT_DECISION_FUNCTION:
        var off = Float32(offset_)
        for i in range(n_query):
            values.append((-paper[i]) - off)
    else:
        for i in range(n_query):
            values.append(-paper[i])
    _ = tables^
    return IForestHostRun(values^, labels^, offset_, max_samples_)
