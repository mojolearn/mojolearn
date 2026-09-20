# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_estimators` family: KernelDensity (the
CPU training lane, phase 1, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md sections 1.1 kde and 3.2) and,
since the classical host inference lane the same day, the INFERENCE entries
of LinearRegression, Ridge, TruncatedSVD, LogisticRegression and PCA
(docs/lanes/BRIEF_forest_host_inference_2026-09-13.md, "Classical lanes").

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The KDE arithmetic is
`kde/host/kde_oracle.mojo::oracle_score_samples`, the float32 serial
replay the device arm is gated against bit for bit under IDENTICAL ("every
formula spelled here a SECOND time rather than imported from `kde/impl/`").
The validation is the GPU entry's, in the GPU entry's order
(`kde/estimator.mojo::kde_score_samples_host_ptr`: kernel and metric names,
`kde_fit_validate`, `n_query`, train data, query data), through the same
host-only functions of `kde/impl/neighbors/kernel_density.mojo`, so a bad
call raises the same error and nothing is written on a refusal. The
classical inference arithmetic is `core/classical_host_predict.mojo`, the
statement-for-statement restatement of the pinned gemm/gemv kernel, the
intercept and bias epilogues, the centering kernel and the host sigmoid;
that file's header names every original by file and line.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for what this covers, so
`python/mojolearn/density.py::KernelDensity`, `linear_model.py` and
`decomposition.py` run unchanged on a CPU-only install through
`_backend._HOST_MODULES` (`"_mojolearn_estimators":
"_mojolearn_estimators_host"`): `kde_score_samples` with the SAME address
contract (train, query, weights, out, the five-value params list, kernel,
metric; mirrored word for word in `density.py`), `ols_predict`,
`tsvd_transform`, `qn_decision_function`, `qn_sigmoid`, `pca_transform`,
since lane/logistic-multiclass (2026-09-14) `qn_softmax` and the 4-field
`qn_decision_function` (the softmax shape of LogisticRegression with more
than two classes), and, since the kde svc host lane (2026-09-14), the whitened pair
`pca_whiten_transform` and `pca_whiten_inverse_transform`, with the SAME
address contracts as `bindings/_mojolearn_estimators.mojo` (each docstring
below repeats its params list), `estimators_numeric_mode` and
`estimators_vendor` (answering "cpu"). Since workstream E
(lane/cpu-training-e, 2026-09-14) the TRAINING entries `pca_fit`,
`tsvd_fit`, `ols_fit` and `ridge_fit` as well, over
`decomposition/host/pca_oracle.mojo` (the column mean, the split-K Gram,
the Float32 Jacobi at the device's settings, the sign flip and the Float64
tail, each restated from its kernel) and `glm/host/glm_oracle.mojo` (the
equilibrated pseudo-inverse of `lstsq_eig` and the `svd_eig` plus
`ridge_solve` pair), and `dbscan_fit` over `dbscan/host/dbscan_oracle.mojo`
(the ball cover index and query replayed, the brute arm, the label
propagation and the relabel; `sample_weight` and an explicit
`max_mbytes_per_batch` refused by name), and (batch 2, lane/cpu-training-e2)
`qn_fit` over `glm/host/qn_oracle.mojo` (the L-BFGS arm of cuML's
quasi-Newton solver with the binary logistic loss; the softmax loss, an l1
or elasticnet penalty and `sample_weight` refused by name). Every other
function of the GPU binding (inverse_transform, ...) is
deliberately absent, so those surfaces refuse BY NAME through
`_HostBinding` and never hash something else.
"""
from std.math import isfinite
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, ftz
from core.classical_host_predict import (
    CLASSICAL_HOST_SABOTAGE,
    host_ols_predict_into,
    host_pca_transform_into,
    host_pca_whiten_inverse_transform_into,
    host_pca_whiten_transform_into,
    host_qn_decision_into,
    host_qn_decision_multi_into,
    host_qn_sigmoid_into,
    host_qn_softmax_into,
    host_tsvd_transform_into,
)
from core.labeled_reference_host_predict import (
    LABELED_PREDICT_HOST_SABOTAGE,
    host_labeled_reference_predict,
    labeled_reference_threshold,
    labeled_reference_validate,
)
from dbscan.host.dbscan_oracle import (
    DBSCAN_ORACLE_HOST_SABOTAGE,
    host_dbscan_fit,
)
from decomposition.host.pca_oracle import (
    PCA_ORACLE_HOST_SABOTAGE,
    host_pca_fit,
    host_pca_validate,
    host_tsvd_fit,
)
from decomposition.host.pca_full_oracle import (
    host_pca_fit_full,
    host_pca_full_validate,
)
from gemm.host.identical_gemm import GEMM_ORACLE_HOST_SABOTAGE, OP_TN, gemm_oracle
from glm.host.glm_oracle import host_ols_fit, host_ridge_fit
from glm.host.qn_oracle import QN_ORACLE_HOST_SABOTAGE, host_qn_fit
from core.host_predict_threads import HostF32Ptr, host_list_ptr, host_predict_task_count
from kde.host.kde_oracle import KDE_ORACLE_HOST_SABOTAGE, oracle_score_samples_into
from kde.impl.neighbors.kernel_density import (
    kde_fit_validate,
    kde_validate_data_ptr,
    kernel_from_name,
    metric_from_name,
)
from kernel_methods.host.km_host_oracle import (
    kmh_kernel_ridge_predict,
    kmh_nystroem_transform,
    kmh_rbf_sampler_transform,
)
from preprocessing.host.scaler_oracle import (
    SCALER_ORACLE_HOST_SABOTAGE,
    host_minmax_transform,
    host_standard_transform,
)


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("estimators host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def estimators_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def estimators_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def estimators_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_estimators_host.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "estimators host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_estimators_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `estimators_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build, so its name would land
# in the vendor-neutral binary. The comptime assert above is the check.


def estimators_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary sums every logsumexp row, walks every dot
    product descending, folds the split-K Gram's chunks descending and
    increments the first final DBSCAN label on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control; one define,
    every arithmetic this binding carries)."""
    return PythonObject(
        KDE_ORACLE_HOST_SABOTAGE
        or CLASSICAL_HOST_SABOTAGE
        or PCA_ORACLE_HOST_SABOTAGE
        or DBSCAN_ORACLE_HOST_SABOTAGE
        or QN_ORACLE_HOST_SABOTAGE
        or LABELED_PREDICT_HOST_SABOTAGE
        or SCALER_ORACLE_HOST_SABOTAGE
        or GEMM_ORACLE_HOST_SABOTAGE
    )


# The GPU binding's names, same contract.


def estimators_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here, a host
    binding is IDENTICAL only (the build script refuses any other)."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def estimators_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def kde_score_samples_binding(
    train_addr: PythonObject,
    query_addr: PythonObject,
    weights_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    kernel: PythonObject,
    metric: PythonObject,
) raises -> PythonObject:
    """KernelDensity.score_samples on the host: log density of each query
    row under the fitted training set, by `oracle_score_samples`. Writes
    `n_query` float32 to `out_addr`. `params`, in this order (mirrored in
    `python/mojolearn/density.py` and in the GPU binding):

        0  n_train
        1  n_query
        2  n_features
        3  bandwidth   (float)
        4  has_weights (0/1; weights_addr is read only when 1)

    `kernel` and `metric` are the sklearn/cuML names; every unimplemented
    one is refused BY NAME by `kernel_from_name` and `metric_from_name`, as
    on the device. Returns n_query."""
    if len(params) != 5:
        raise Error(
            "kde_score_samples: params must contain 5 values, got "
            + String(len(params))
        )
    var tp = f32_ptr(_index(train_addr))
    var qp = f32_ptr(_index(query_addr))
    var op = f32_ptr(_index(out_addr))
    var n_train = _index(params[0])
    var n_query = _index(params[1])
    var n_features = _index(params[2])
    var bandwidth = Float32(Float64(py=params[3]))
    var has_weights = _index(params[4]) != 0
    var kname = String(py=kernel)
    var mname = String(py=metric)
    var weights = List[Float32]()
    if has_weights:
        weights = read_f32(_index(weights_addr), max(0, n_train))
    var train_address = _index(train_addr)
    var query_address = _index(query_addr)
    with GILReleased(Python()):
        # The GPU entry's checks, in its order, so a bad call raises the
        # same error before anything is read or written.
        var k = kernel_from_name(kname)
        var m = metric_from_name(mname)
        kde_fit_validate(n_train, n_features, bandwidth, k, m, weights, has_weights)
        if n_query <= 0:
            raise Error("kde: X must have at least one row (n_query)")
        kde_validate_data_ptr(tp, n_train, n_features, m, "train")
        kde_validate_data_ptr(qp, n_query, n_features, m, "query")
        # THE ONE CALL THAT COMPUTES ANYTHING. metric_arg is Minkowski's p,
        # 2.0 here as in the GPU binding, which passes no other value.
        # DEVIATION 2920: read in place and scored straight into `out`, the
        # query rows split across the host thread policy; the per-row
        # arithmetic is `oracle_score_samples`'s, which the checks keep.
        oracle_score_samples_into(
            tp, qp, weights, has_weights, n_train, n_query, n_features,
            bandwidth, k, m, op, host_predict_task_count(n_query), Float32(2.0),
        )
    return PythonObject(n_query)


# ===========================================================================
# THE TRAINING ENTRIES (workstream E, lane/cpu-training-e, 2026-09-14): the
# GPU binding's `pca_fit`, `tsvd_fit`, `ols_fit` and `ridge_fit`, same
# names, same arity, same params lists, over decomposition/host/pca_oracle.mojo
# and glm/host/glm_oracle.mojo; batch 2 adds `qn_fit` over
# glm/host/qn_oracle.mojo; the pca-full-whiten lane adds `pca_fit_full`
# (the R-SVD arm, tall matrices) over decomposition/host/pca_full_oracle.mojo.
# `inverse_transform` stays absent and refuses BY NAME.
# ===========================================================================


def pca_fit_binding(
    x_addr: PythonObject,
    components_addr: PythonObject,
    mean_addr: PythonObject,
    explained_addr: PythonObject,
    ratio_addr: PythonObject,
    singular_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`PCA.fit` (svd_solver 'auto', 'covariance_eigh', 'jacobi') on the
    host by `host_pca_fit`: the GPU binding's contract, params `n_rows,
    n_features, n_components`, the five public arrays written, the noise
    variance returned. The shape refusals are `pca_validate`'s, raised
    BEFORE anything is read."""
    if len(params) != 3:
        raise Error("pca_fit: params must contain n_rows, n_features, n_components")
    var x_address = _index(x_addr)
    var cp = f32_ptr(_index(components_addr))
    var mp = f32_ptr(_index(mean_addr))
    var ep = f32_ptr(_index(explained_addr))
    var rp = f32_ptr(_index(ratio_addr))
    var sp = f32_ptr(_index(singular_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    var noise = Float64(0.0)
    with GILReleased(Python()):
        host_pca_validate_first(nr, nf, nc)
        var x = read_f32(x_address, nr * nf)
        var fit = host_pca_fit(x, nr, nf, nc)
        for i in range(nc * nf):
            cp[i] = Float32(fit.result.components[i])
        for i in range(nc):
            ep[i] = Float32(fit.result.explained_var[i])
            rp[i] = Float32(fit.result.explained_var_ratio[i])
            sp[i] = Float32(fit.result.singular_vals[i])
        for i in range(nf):
            mp[i] = fit.mean[i]
        noise = fit.result.noise_var
    return PythonObject(noise)


def pca_fit_full_binding(
    x_addr: PythonObject,
    components_addr: PythonObject,
    mean_addr: PythonObject,
    explained_addr: PythonObject,
    ratio_addr: PythonObject,
    singular_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`PCA.fit` with svd_solver='full' on the host by `host_pca_fit_full`
    (decomposition/host/pca_full_oracle.mojo, the pca-full-whiten lane): the
    GPU binding's contract, params `n_rows, n_features, n_components`, the
    five public arrays written, the noise variance returned. The shape
    refusals are `pca_full_validate`'s, raised BEFORE anything is read; a
    wide matrix is refused by name (the host carries the tall route only)."""
    if len(params) != 3:
        raise Error("pca_fit_full: params must contain n_rows, n_features, n_components")
    var x_address = _index(x_addr)
    var cp = f32_ptr(_index(components_addr))
    var mp = f32_ptr(_index(mean_addr))
    var ep = f32_ptr(_index(explained_addr))
    var rp = f32_ptr(_index(ratio_addr))
    var sp = f32_ptr(_index(singular_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    host_pca_full_validate(nr, nf, nc)
    var noise = Float64(0.0)
    with GILReleased(Python()):
        var x = read_f32(x_address, nr * nf)
        var fit = host_pca_fit_full(x, nr, nf, nc)
        for i in range(nc * nf):
            cp[i] = Float32(fit.result.components[i])
        for i in range(nc):
            ep[i] = Float32(fit.result.explained_var[i])
            rp[i] = Float32(fit.result.explained_var_ratio[i])
            sp[i] = Float32(fit.result.singular_vals[i])
        for i in range(nf):
            mp[i] = fit.mean[i]
        noise = fit.result.noise_var
    return PythonObject(noise)


def host_pca_validate_first(nr: Int, nf: Int, nc: Int) raises:
    """`pca_validate` before the read, so a refused shape reads no address;
    the fit validates again on entry, the device order."""
    host_pca_validate(nr, nf, nc)


def tsvd_fit_binding(
    x_addr: PythonObject,
    components_addr: PythonObject,
    singular_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`TruncatedSVD.fit` on the host by `host_tsvd_fit`: params `n_rows,
    n_features, n_components`; components and singular values written.
    Returns 0."""
    if len(params) != 3:
        raise Error("tsvd_fit: params must contain 3 values")
    var x_address = _index(x_addr)
    var cp = f32_ptr(_index(components_addr))
    var sp = f32_ptr(_index(singular_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    with GILReleased(Python()):
        host_pca_validate_first(nr, nf, nc)
        var x = read_f32(x_address, nr * nf)
        var result = host_tsvd_fit(x, nr, nf, nc)
        for i in range(nc * nf):
            cp[i] = Float32(result.components[i])
        for i in range(nc):
            sp[i] = Float32(result.singular_vals[i])
    return PythonObject(0)


def ols_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    coef_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`LinearRegression.fit` on the host by `host_ols_fit` (the centered
    and, when weighted, root-scaled design the Python layer hands over, as
    on the GPU): params `n_rows, n_features`; `n_features` coefficients
    written. Returns 0."""
    if len(params) != 2:
        raise Error("ols_fit: params must contain n_rows, n_features")
    var x_address = _index(x_addr)
    var y_address = _index(y_addr)
    var wp = f32_ptr(_index(coef_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        var x = read_f32(x_address, nr * nf)
        var y = read_f32(y_address, nr)
        var w = host_ols_fit(x, y, nr, nf)
        for i in range(nf):
            wp[i] = w[i]
    return PythonObject(0)


def ridge_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    coef_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`Ridge.fit` (the eig arm, DEVIATION 545) on the host by
    `host_ridge_fit`: params `n_rows, n_features, alpha` (`alpha` a float,
    `Float32(Float64(...))` as in the GPU binding). Returns 0."""
    if len(params) != 3:
        raise Error("ridge_fit: params must contain n_rows, n_features, alpha")
    var x_address = _index(x_addr)
    var y_address = _index(y_addr)
    var wp = f32_ptr(_index(coef_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var alpha = Float32(Float64(py=params[2]))
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        var x = read_f32(x_address, nr * nf)
        var y = read_f32(y_address, nr)
        var w = host_ridge_fit(x, y, nr, nf, alpha)
        for i in range(nf):
            wp[i] = w[i]
    return PythonObject(0)


def qn_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    coef_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`qnFit` on the host by `host_qn_fit` (L-BFGS, or OWL-QN when the
    l1 penalty is nonzero; the binary logistic loss, or the softmax loss at
    `n_classes > 2`, lane/cpu-training-batch3). params: n_rows, n_features,
    n_classes, penalty_l1, penalty_l2, grad_tol, change_tol, max_iter,
    linesearch_max_iter, lbfgs_memory, fit_intercept, penalty_normalized,
    has_sample_weight, and an OPTIONAL 14th, the loss id (QN_LOSS_LOGISTIC,
    the value a 13-field call gets). `coef_addr` holds `n_targets *
    (n_features + fit_intercept)` floats, written; `info_addr[0]` receives
    the objective, `[1]` the OPT_RETCODE; returns num_iters.
    `sample_weight` is refused BY NAME."""
    if len(params) != 13 and len(params) != 14:
        raise Error("qn_fit: params must carry the 13 qn_params fields, plus an optional 14th, the loss id")
    var x_address = _index(x_addr)
    var y_address = _index(y_addr)
    var wp = f32_ptr(_index(coef_addr))
    var ip = f32_ptr(_index(info_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    var l1 = Float64(py=params[3])
    var l2 = Float64(py=params[4])
    var grad_tol = Float64(py=params[5])
    var change_tol = Float64(py=params[6])
    var max_iter = _index(params[7])
    var ls_max = _index(params[8])
    var mem = _index(params[9])
    var fit_intercept = _index(params[10]) != 0
    var normalized = _index(params[11]) != 0
    var has_sw = _index(params[12]) != 0
    var loss = _index(params[13]) if len(params) == 14 else 0
    var iters = 0
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        var x = read_f32(x_address, nr * nf)
        var y = read_f32(y_address, nr)
        var coef = List[Float32]()
        var r = host_qn_fit(
            x, y, nr, nf, nc, l1, l2, grad_tol, change_tol, max_iter, ls_max,
            mem, fit_intercept, normalized, has_sw, loss, coef,
        )
        for i in range(len(coef)):
            wp[i] = coef[i]
        ip[0] = r.fx
        ip[1] = Float32(r.retcode)
        iters = r.n_iter
    return PythonObject(iters)


def dbscan_fit_binding(
    x_addr: PythonObject,
    labels_addr: PythonObject,
    weight_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`DBSCAN.fit` on the host by `host_dbscan_fit`: the GPU binding's
    contract (params `n_rows, n_features, eps, min_samples, budget_mb,
    max_iter, eps_nn_method, metric`; `weight_addr` 0 for no weights),
    `n_rows` int32 labels written, the propagation pass count returned
    (the one-thread schedule's; no column hashes it). `weight_addr != 0`
    reads `n_rows` float32 weights and takes the weighted core test
    (`host_weighted_degree`, lane/cpu-training-batch3). `budget_mb != 0` is
    refused BY NAME: the device-sized batch has no host restatement."""
    return _dbscan_fit_run(x_addr, labels_addr, weight_addr, 0, params)


def dbscan_fit_core_binding(
    x_addr: PythonObject,
    labels_addr: PythonObject,
    weight_addr: PythonObject,
    core_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`dbscan_fit` with the fit's core mask written to `core_addr`
    (`n_rows` uint8, `DBSCANHostFit.core`), the GPU binding's contract
    (lane/inference-transductive-predict, 2026-09-15)."""
    var ca = _index(core_addr)
    if ca == 0:
        raise Error("dbscan_fit_core: core_addr must be an array address, got 0")
    return _dbscan_fit_run(x_addr, labels_addr, weight_addr, ca, params)


def labeled_reference_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Out-of-sample labels for DBSCAN and AgglomerativeClustering on the
    host by `host_labeled_reference_predict`, the GPU binding's contract
    (addrs refs, keys, ref_labels, queries, out_labels, out_refs; params
    n_refs, n_queries, n_features, metric, eps, has_thresh). Returns 0."""
    if len(addrs) != 6:
        raise Error(
            "labeled_reference_predict: addrs must contain 6 addresses, got "
            + String(len(addrs))
        )
    if len(params) != 6:
        raise Error(
            "labeled_reference_predict: params must contain 6 values, got "
            + String(len(params))
        )
    var refs_address = _index(addrs[0])
    var keys_address = _index(addrs[1])
    var labels_address = _index(addrs[2])
    var queries_address = _index(addrs[3])
    var olp = i32_ptr(_index(addrs[4]))
    var orp = i32_ptr(_index(addrs[5]))
    var n_refs = _index(params[0])
    var n_queries = _index(params[1])
    var n_features = _index(params[2])
    var metric = _index(params[3])
    var eps = Float64(py=params[4])
    var has_thresh = _index(params[5]) != 0
    with GILReleased(Python()):
        labeled_reference_validate(n_refs, n_queries, n_features, metric)
        var refs = read_f32(refs_address, n_refs * n_features)
        var queries = read_f32(queries_address, n_queries * n_features)
        var kp = i32_ptr(keys_address)
        var lp = i32_ptr(labels_address)
        var keys = List[Int32](capacity=n_refs)
        var ref_labels = List[Int32](capacity=n_refs)
        for i in range(n_refs):
            keys.append(kp[i])
            ref_labels.append(lp[i])
        var thresh = Float32(0.0)
        if has_thresh:
            thresh = labeled_reference_threshold(metric, eps)
        var out = host_labeled_reference_predict(
            refs, n_refs, keys, ref_labels, queries, n_queries, n_features,
            metric, thresh, has_thresh,
        )
        for i in range(n_queries):
            olp[i] = out.labels[i]
            orp[i] = out.refs[i]
    return PythonObject(0)


def _dbscan_fit_run(
    x_addr: PythonObject,
    labels_addr: PythonObject,
    weight_addr: PythonObject,
    core_address: Int,
    params: PythonObject,
) raises -> PythonObject:
    if len(params) != 8:
        raise Error(
            "dbscan_fit: params must contain 8 values, got "
            + String(len(params))
        )
    var x_address = _index(x_addr)
    var lp = i32_ptr(_index(labels_addr))
    var wa = _index(weight_addr)
    var nr = _index(params[0])
    var nf = _index(params[1])
    var eps = Float64(py=params[2])
    var min_samples = _index(params[3])
    var budget = _index(params[4])
    var max_iter = _index(params[5])
    var eps_nn_method = _index(params[6])
    var metric = _index(params[7])
    if budget != 0:
        raise Error(
            "mojolearn: no CPU implementation of DBSCAN.fit with"
            " max_mbytes_per_batch yet; the host runs one batch of every row"
            " and does not size a device it does not have"
        )
    var passes = 0
    with GILReleased(Python()):
        if nr < 1 or nf < 1:
            raise Error(
                "dbscan_fit needs n_samples and n_features >= 1: got "
                + String(nr)
                + ", "
                + String(nf)
            )
        var x = read_f32(x_address, nr * nf)
        var weights = List[Float32]()
        if wa != 0:
            weights = read_f32(wa, nr)
        var fit = host_dbscan_fit(
            x, nr, nf, eps, min_samples, max_iter, eps_nn_method, metric,
            weights, wa != 0,
        )
        for i in range(nr):
            lp[i] = fit.labels[i]
        if core_address != 0:
            var cp = MutPointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=core_address
            )
            for i in range(nr):
                cp.unsafe_store(i, fit.core[i])
        passes = fit.passes
    return PythonObject(passes)


# The classical inference entries (the classical host inference lane,
# 2026-09-13). Each keeps the GPU binding's name, arity and params list.


def _positive(value: Int, what: String) raises:
    if value < 1:
        raise Error("estimators host: " + what + " must be at least 1, got " + String(value))


def ols_predict_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`LinearRegression.predict` and `Ridge.predict` on the host:
    `out[n_rows] = X[n_rows x n_features] . coef + intercept` by
    `host_ols_predict`. params, as in the GPU binding: n_rows, n_features,
    intercept (a float; `Float32(Float64(...))` as there). Returns 0."""
    if len(params) != 3:
        raise Error("ols_predict: params must contain n_rows, n_features, intercept")
    var x_address = _index(x_addr)
    var coef_address = _index(coef_addr)
    var op = f32_ptr(_index(out_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var intercept = Float32(Float64(py=params[2]))
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        # DEVIATION 2920: X read in place, the rows split across the host
        # thread policy, the predictions written straight to `out`.
        var coef = read_f32(coef_address, nf)
        host_ols_predict_into(
            f32_ptr(x_address), host_list_ptr(coef), op, nr, nf, intercept,
            host_predict_task_count(nr),
        )
        _ = coef^
    return PythonObject(0)


def tsvd_transform_binding(
    x_addr: PythonObject,
    components_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`TruncatedSVD.transform` on the host: `out[n_rows x n_components] =
    X . components^T` by `host_tsvd_transform`. params: n_rows, n_features,
    n_components. Returns 0."""
    if len(params) != 3:
        raise Error("tsvd_transform: params must contain 3 values")
    var x_address = _index(x_addr)
    var c_address = _index(components_addr)
    var op = f32_ptr(_index(out_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        _positive(nc, "n_components")
        var components = read_f32(c_address, nc * nf)
        host_tsvd_transform_into(
            f32_ptr(x_address), host_list_ptr(components), op, nr, nf, nc,
            host_predict_task_count(nr),
        )
        _ = components^
    return PythonObject(0)


def pca_transform_binding(
    x_addr: PythonObject,
    mean_addr: PythonObject,
    components_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`PCA.transform` (whiten=False) on the host: center by `mean_`, then
    `out = (X - mean) . components^T` by `host_pca_transform`. params:
    n_rows, n_features, n_components. Returns 0. The whitened pair is
    `pca_whiten_transform` and `pca_whiten_inverse_transform` below."""
    if len(params) != 3:
        raise Error("pca_transform: params must contain 3 values")
    var x_address = _index(x_addr)
    var m_address = _index(mean_addr)
    var c_address = _index(components_addr)
    var op = f32_ptr(_index(out_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        _positive(nc, "n_components")
        var mu = read_f32(m_address, nf)
        var components = read_f32(c_address, nc * nf)
        host_pca_transform_into(
            f32_ptr(x_address), host_list_ptr(mu), host_list_ptr(components),
            op, nr, nf, nc, host_predict_task_count(nr),
        )
        _ = mu^
        _ = components^
    return PythonObject(0)


def _whiten_finite(values: List[Float32], what: String) raises:
    """`_pca_whiten_finite` of the GPU binding, over the copy this side
    reads, with the GPU binding's sentence."""
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error("PCA whitening requires finite inputs and outputs")


def _whiten_finite_ptr(values: HostF32Ptr, count: Int, what: String) raises:
    """The same finite refusal over caller-owned memory, without a copy."""
    for i in range(count):
        if not isfinite(values.unsafe_load(i)):
            raise Error("PCA whitening requires finite inputs and outputs")


def _pca_whiten_apply(
    input_addr: PythonObject,
    mean_addr: PythonObject,
    components_addr: PythonObject,
    singular_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    inverse: Bool,
) raises -> PythonObject:
    """`bindings/_mojolearn_estimators.mojo::_pca_whiten_apply` without the
    DeviceContext: the same params check, the same dimension refusals in the
    same words, the same overlap refusal, the same finiteness and sign
    refusals BEFORE anything is computed, the same finiteness refusal of the
    output after, and the arithmetic by `host_pca_whiten_transform` or
    `host_pca_whiten_inverse_transform`. params: n_rows, n_features,
    n_components, n_fit_rows (the FIT's row count, DEVIATION 580)."""
    if len(params) != 4:
        raise Error("PCA whitening params require rows, features, components, fit_rows")
    var nr = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    var nfit = _index(params[3])
    if nr < 1 or nf < 2 or nc < 1 or nc > nf or nfit < 2:
        raise Error("PCA whitening requires positive rows/components, features>=2 and fit_rows>=2")
    if nr > 2147483647 or nf > 2147483647 or nc > 2147483647 or nfit > 2147483647:
        raise Error("PCA whitening dimensions exceed Int32")
    if nr > 2147483647 // nf or nc > 2147483647 // nf:
        raise Error("PCA whitening matrix cell count exceeds Int32")
    var input_count = nr * (nc if inverse else nf)
    var output_count = nr * (nf if inverse else nc)
    var xa = _index(input_addr)
    var ma = _index(mean_addr)
    var ca = _index(components_addr)
    var sa = _index(singular_addr)
    var oa = _index(out_addr)
    var starts = List[Int]()
    starts.append(xa)
    starts.append(ma)
    starts.append(ca)
    starts.append(sa)
    var counts = List[Int]()
    counts.append(input_count)
    counts.append(nf)
    counts.append(nc * nf)
    counts.append(nc)
    for i in range(4):
        if starts[i] <= 0 or starts[i] % 4 != 0:
            raise Error("PCA whitening requires positive aligned FP32 pointers")
    if oa <= 0 or oa % 4 != 0:
        raise Error("PCA whitening requires positive aligned FP32 pointers")
    for i in range(4):
        if oa < starts[i] + counts[i] * 4 and starts[i] < oa + output_count * 4:
            raise Error("PCA whitening output must not overlap any input")
    var op = f32_ptr(oa)
    with GILReleased(Python()):
        var mu = read_f32(ma, nf)
        var components = read_f32(ca, nc * nf)
        var singular = read_f32(sa, nc)
        var xp = f32_ptr(xa)
        _whiten_finite_ptr(xp, input_count, "input")
        _whiten_finite(mu, "mean")
        _whiten_finite(components, "components")
        _whiten_finite(singular, "singular")
        for i in range(nc):
            if singular[i] < Float32(0):
                raise Error("PCA whitening singular values must be nonnegative")
        if inverse:
            host_pca_whiten_inverse_transform_into(
                xp, components, singular, mu, op, nr, nf, nc, nfit,
                host_predict_task_count(nr),
            )
        else:
            host_pca_whiten_transform_into(
                xp, mu, components, singular, op, nr, nf, nc, nfit,
                host_predict_task_count(nr),
            )
        _whiten_finite_ptr(op, output_count, "output")
    return PythonObject(0)


def pca_whiten_transform_binding(
    x_addr: PythonObject, mean_addr: PythonObject,
    components_addr: PythonObject, singular_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`PCA.transform` (whiten=True) on the host, the GPU binding's arity
    and argument order (x, mean, components, singular_values, out, params).
    Returns 0."""
    return _pca_whiten_apply(
        x_addr, mean_addr, components_addr, singular_addr, out_addr, params, False)


def pca_whiten_inverse_transform_binding(
    scores_addr: PythonObject, components_addr: PythonObject,
    singular_addr: PythonObject, mean_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`PCA.inverse_transform` (whiten=True) on the host, the GPU binding's
    arity and argument order (scores, components, singular_values, mean, out,
    params). Returns 0."""
    return _pca_whiten_apply(
        scores_addr, mean_addr, components_addr, singular_addr, out_addr, params, True)


def qn_decision_function_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`LogisticRegression.decision_function` on the host: `scores = X w + b`
    by `host_qn_decision`, `w` the fitted `_w` of `n_features +
    fit_intercept` entries, the bias its LAST entry. params: n_rows,
    n_features, fit_intercept (0/1) and, since lane/logistic-multiclass
    (2026-09-14), an OPTIONAL 4th, n_classes, the GPU binding's contract:
    absent or below 3 is the binary shape; `n_classes > 2` the softmax
    shape by `host_qn_decision_multi`, `w` the column-major `C x dims`
    block and out `n_rows * n_classes` floats row-major. Returns 0."""
    if len(params) != 3 and len(params) != 4:
        raise Error("qn_decision_function: params must contain n_rows, n_features, fit_intercept and an optional n_classes")
    var x_address = _index(x_addr)
    var w_address = _index(coef_addr)
    var op = f32_ptr(_index(out_addr))
    var nr = _index(params[0])
    var nf = _index(params[1])
    var fi = _index(params[2]) != 0
    var nc = _index(params[3]) if len(params) == 4 else 1
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        _positive(nf, "n_features")
        # DEVIATION 2920: X read in place, the rows split across the host
        # thread policy, the scores written straight to `out`.
        if nc > 2:
            var wm = read_f32(w_address, (nf + (1 if fi else 0)) * nc)
            host_qn_decision_multi_into(
                f32_ptr(x_address), host_list_ptr(wm), op, nr, nf, nc, fi,
                host_predict_task_count(nr),
            )
            _ = wm^
        else:
            var w = read_f32(w_address, nf + (1 if fi else 0))
            host_qn_decision_into(
                f32_ptr(x_address), host_list_ptr(w), op, nr, nf, fi,
                host_predict_task_count(nr),
            )
            _ = w^
    return PythonObject(0)


def qn_softmax_binding(
    scores_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The multinomial `predict_proba` link (lane/logistic-multiclass,
    2026-09-14) by `host_qn_softmax`, the GPU binding's contract: scores
    float32 `(n_rows, n_classes)` row-major, out float64 `(n_rows,
    n_classes)`. params: n_rows, n_classes. Returns 0."""
    if len(params) != 2:
        raise Error("qn_softmax: params must contain n_rows, n_classes")
    var s_address = _index(scores_addr)
    var op = f64_ptr(_index(out_addr))
    var nr = _index(params[0])
    var nc = _index(params[1])
    if nc < 3:
        raise Error("qn_softmax: n_classes must be at least 3; the binary link is qn_sigmoid")
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        host_qn_softmax_into(
            f32_ptr(s_address), op, nr, nc, host_predict_task_count(nr)
        )
    return PythonObject(0)


def qn_sigmoid_binding(
    scores_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The binary `predict_proba` link (DEVIATION 549) by `host_qn_sigmoid`:
    out is float64 `(n_rows, 2)`, `[1 - p, p]`. params: n_rows. Returns 0."""
    if len(params) != 1:
        raise Error("qn_sigmoid: params must contain n_rows")
    var s_address = _index(scores_addr)
    var op = f64_ptr(_index(out_addr))
    var nr = _index(params[0])
    with GILReleased(Python()):
        _positive(nr, "n_rows")
        host_qn_sigmoid_into(
            f32_ptr(s_address), op, nr, host_predict_task_count(nr)
        )
    return PythonObject(0)


# ===========================================================================
# SAVED-MODEL INFERENCE FOR THE SCALERS, COORDINATE DESCENT AND THE KERNEL
# METHODS (lane/inference-linear-svm, 2026-09-15). Their training bindings
# (preprocessing, solver, kernel_methods) are reference-only and do not ship
# in a wheel, so the transform and predict entries a saved model needs are
# served from this shipped binding under the GPU bindings' names and
# contracts. Each entry below repeats the reference host binding's guards
# and calls the same host restatement:
#   standard_transform, minmax_transform  bindings/_mojolearn_preprocessing_host.mojo
#                                         over preprocessing/host/scaler_oracle.mojo
#   cd_predict                            bindings/_mojolearn_solver_host.mojo
#                                         (gemm_oracle at OP_TN, then ftz(v + intercept))
#   kernel_ridge_predict, nystroem_transform, rbf_sampler_transform
#                                         bindings/_mojolearn_kernel_methods_host.mojo
#                                         over kernel_methods/host/km_host_oracle.mojo
# No fit entry of those families is compiled here; the fits stay in the
# reference bindings.
# ===========================================================================


def _pp_load(addr: Int, n: Int) raises -> List[Float32]:
    if addr == 0:
        raise Error("preprocessing: null Float32 pointer")
    return read_f32(addr, max(0, n))


def _pp_out(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("preprocessing: null Float32 pointer")
    return f32_ptr(addr)


def _pp_minmax_finite(values: List[Float32]) raises:
    for value in values:
        if not isfinite(value):
            raise Error("MinMaxScaler: nonfinite input or Float32 arithmetic overflow")


def _pp_standard_finite(values: List[Float32]) raises:
    for value in values:
        if not isfinite(value):
            raise Error("StandardScaler: nonfinite input or Float32 arithmetic overflow")


def standard_transform_binding(
    x_addr: PythonObject, mean_addr: PythonObject, scale_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`standard_transform`, the preprocessing host binding's entry and
    contract: params=[n,d,inverse,with_mean,with_std]; output n*d row-major
    by `host_standard_transform`. Returns n*d."""
    if len(params) != 5:
        raise Error("standard_transform: requires 5 parameters")
    var n = _index(params[0])
    var d = _index(params[1])
    var inverse = _index(params[2])
    var with_mean = _index(params[3])
    var with_std = _index(params[4])
    if n <= 0 or d <= 0 or d > 2147483647 or n > 2147483647 // d:
        raise Error("StandardScaler: positive dimensions with n*d<=Int32.max required")
    if with_mean < 0 or with_mean > 1 or with_std < 0 or with_std > 1:
        raise Error("StandardScaler: flags must be 0 or 1")
    var x = _pp_load(_index(x_addr), n * d)
    var mean = _pp_load(_index(mean_addr), d)
    var scale = _pp_load(_index(scale_addr), d)
    var output = _pp_out(_index(out_addr))
    with GILReleased(Python()):
        if len(x) < n * d or len(mean) < d or len(scale) < d or inverse < 0 or inverse > 1:
            raise Error("StandardScaler: invalid transform parameters")
        _pp_standard_finite(x)
        if with_mean != 0:
            _pp_standard_finite(mean)
        if with_std != 0:
            _pp_standard_finite(scale)
            for c in range(d):
                if scale[c] <= 0:
                    raise Error("StandardScaler: scale must be positive")
        var result = host_standard_transform(x, mean, scale, n, d, inverse, with_mean, with_std)
        _pp_standard_finite(result)
        for i in range(n * d):
            output[i] = result[i]
    return PythonObject(n * d)


def minmax_transform_binding(
    x_addr: PythonObject, scale_addr: PythonObject, min_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`minmax_transform`, the preprocessing host binding's entry and
    contract: params=[n,d,inverse,clip,feature_min,feature_max]; output n*d
    row-major by `host_minmax_transform`. Returns n*d."""
    if len(params) != 6:
        raise Error("minmax_transform: requires 6 parameters")
    var n = _index(params[0])
    var d = _index(params[1])
    var inverse = _index(params[2])
    var clip = _index(params[3])
    var lower = Float32(Float64(py=params[4]))
    var upper = Float32(Float64(py=params[5]))
    if n <= 0 or d <= 0 or d > 2147483647 or n > 2147483647 // d:
        raise Error("MinMaxScaler: positive dimensions with n*d<=Int32.max required")
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise Error("MinMaxScaler: finite increasing Float32 feature range required")
    var x = _pp_load(_index(x_addr), n * d)
    var scale = _pp_load(_index(scale_addr), d)
    var offset = _pp_load(_index(min_addr), d)
    var output = _pp_out(_index(out_addr))
    with GILReleased(Python()):
        if len(x) < n * d or len(scale) < d or len(offset) < d or inverse < 0 or inverse > 1 or clip < 0 or clip > 1:
            raise Error("MinMaxScaler: invalid transform parameters")
        _pp_minmax_finite(x)
        _pp_minmax_finite(scale)
        _pp_minmax_finite(offset)
        for c in range(d):
            if scale[c] <= 0:
                raise Error("MinMaxScaler: scale must be positive")
        var result = host_minmax_transform(x, scale, offset, n, d, inverse, clip, lower, upper)
        _pp_minmax_finite(result)
        for i in range(n * d):
            output[i] = result[i]
    return PythonObject(n * d)


def cd_predict_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`cd_predict`, the solver host binding's entry and contract: `x_addr`
    COLUMN-MAJOR `n_rows x n_cols`, params [n_rows, n_cols, intercept],
    `pred = gemm_oracle(x, coef, OP_TN, n_rows, 1, n_cols)` then
    `ftz(v + intercept)` when the intercept is not zero. Returns 0."""
    if len(params) != 3:
        raise Error(
            "cd_predict: params must contain n_rows, n_cols, intercept"
        )
    var x_address = _index(x_addr)
    var coef_address = _index(coef_addr)
    var op = f32_ptr(_index(out_addr))
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var intercept = Float32(Float64(py=params[2]))
    with GILReleased(Python()):
        if n_rows < 1 or n_cols < 1:
            raise Error(
                "cd_predict_host needs n_rows and n_cols >= 1: got "
                + String(n_rows) + ", " + String(n_cols)
            )
        if n_cols <= 0:
            raise Error(
                "Parameter n_cols: number of columns cannot be less than one"
            )
        if n_rows <= 1:
            raise Error("Parameter n_rows: number of rows cannot be less than two")
        var x = read_f32(x_address, n_rows * n_cols)
        var coef = read_f32(coef_address, n_cols)
        var pred = gemm_oracle(x, coef, OP_TN, n_rows, 1, n_cols)
        for i in range(n_rows):
            var v = pred[i]
            if intercept != Float32(0.0):
                v = ftz(v + intercept)
            op[i] = v
    return PythonObject(0)


def kernel_ridge_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`kernel_ridge_predict`, the kernel_methods host binding's entry and
    contract. `addrs`: 0 x_fit, 1 dual, 2 x_new, 3 out. `params`: 0 n, 1 d,
    2 t, 3 kernel, 4 degree, 5 gamma, 6 coef0, 7 alpha, 8 info, 9 q.
    Returns 0."""
    if len(addrs) != 4:
        raise Error(
            "kernel_ridge_predict: addrs must contain 4 addresses (x_fit,"
            " dual, x_new, out), got "
            + String(len(addrs))
        )
    if len(params) != 10:
        raise Error(
            "kernel_ridge_predict: params must contain 10 values (n, d, t,"
            " kernel, degree, gamma, coef0, alpha, info, q), got "
            + String(len(params))
        )
    var op = f32_ptr(Int(py=addrs[3]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var t = Int(py=params[2])
    var kernel = Int(py=params[3])
    var degree = Int(py=params[4])
    var gamma = Float64(py=params[5])
    var coef0 = Float64(py=params[6])
    var q = Int(py=params[9])
    var x_fit = read_f32(Int(py=addrs[0]), max(0, n * d))
    var dual = read_f32(Int(py=addrs[1]), max(0, n * t))
    var x_new = read_f32(Int(py=addrs[2]), max(0, q * d))
    with GILReleased(Python()):
        var out = kmh_kernel_ridge_predict(
            x_fit, dual, n, d, t, kernel, degree, gamma, coef0, x_new, q
        )
        for i in range(q * t):
            op.unsafe_store(i, out[i])
        _ = out^
    _ = x_fit^
    _ = dual^
    _ = x_new^
    return PythonObject(0)


def nystroem_transform_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`nystroem_transform`, the kernel_methods host binding's entry and
    contract. `addrs`: 0 components, 1 indices, 2 normalization,
    3 eigenvalues, 4 eigenvectors, 5 x, 6 out. `params`: 0 q, 1 d, 2 kernel,
    3 degree, 4 gamma, 5 coef0, 6 seed, 7 sweeps, 8 m. Returns 0."""
    if len(addrs) != 7:
        raise Error(
            "nystroem_transform: addrs must contain 7 addresses (components,"
            " indices, normalization, eigenvalues, eigenvectors, x, out),"
            " got "
            + String(len(addrs))
        )
    if len(params) != 9:
        raise Error(
            "nystroem_transform: params must contain 9 values (q, d, kernel,"
            " degree, gamma, coef0, seed, sweeps, m), got "
            + String(len(params))
        )
    var q = Int(py=params[0])
    var d = Int(py=params[1])
    var kernel = Int(py=params[2])
    var degree = Int(py=params[3])
    var gamma = Float64(py=params[4])
    var coef0 = Float64(py=params[5])
    var m = Int(py=params[8])
    var components = read_f32(Int(py=addrs[0]), max(0, q * d))
    var normalization = read_f32(Int(py=addrs[2]), max(0, q * q))
    var x = read_f32(Int(py=addrs[5]), max(0, m * d))
    var op = f32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        var out = kmh_nystroem_transform(
            components, normalization, q, d, kernel, degree, gamma, coef0, x, m
        )
        for i in range(m * q):
            op.unsafe_store(i, out[i])
        _ = out^
    _ = components^
    _ = normalization^
    _ = x^
    return PythonObject(0)


def rbf_sampler_transform_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`rbf_sampler_transform`, the kernel_methods host binding's entry and
    contract. `addrs`: 0 weights, 1 offset, 2 x, 3 out. `params`: 0 d, 1 q,
    2 gamma, 3 seed, 4 sigma, 5 scale, 6 m. Returns 0."""
    if len(addrs) != 4:
        raise Error(
            "rbf_sampler_transform: addrs must contain 4 addresses (weights,"
            " offset, x, out), got "
            + String(len(addrs))
        )
    if len(params) != 7:
        raise Error(
            "rbf_sampler_transform: params must contain 7 values (d, q,"
            " gamma, seed, sigma, scale, m), got "
            + String(len(params))
        )
    var d = Int(py=params[0])
    var q = Int(py=params[1])
    var scale = Float32(Float64(py=params[5]))
    var m = Int(py=params[6])
    var weights = read_f32(Int(py=addrs[0]), max(0, d * q))
    var offset = read_f32(Int(py=addrs[1]), max(0, q))
    var x = read_f32(Int(py=addrs[2]), max(0, m * d))
    var op = f32_ptr(Int(py=addrs[3]))
    with GILReleased(Python()):
        var out = kmh_rbf_sampler_transform(weights, offset, d, q, scale, x, m)
        for i in range(m * q):
            op.unsafe_store(i, out[i])
        _ = out^
    _ = weights^
    _ = offset^
    _ = x^
    return PythonObject(0)


@export
def PyInit__mojolearn_estimators_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_estimators_host")
        module.def_function[estimators_host_numeric_mode_binding]("estimators_host_numeric_mode")
        module.def_function[estimators_host_vendor_binding]("estimators_host_vendor")
        module.def_function[estimators_host_column_binding]("estimators_host_column")
        module.def_function[estimators_host_sabotage_binding]("estimators_host_sabotage")
        module.def_function[estimators_vendor_binding]("estimators_vendor")
        module.def_function[estimators_numeric_mode_binding]("estimators_numeric_mode")
        module.def_function[kde_score_samples_binding]("kde_score_samples")
        module.def_function[pca_fit_binding]("pca_fit")
        module.def_function[pca_fit_full_binding]("pca_fit_full")
        module.def_function[tsvd_fit_binding]("tsvd_fit")
        module.def_function[ols_fit_binding]("ols_fit")
        module.def_function[ridge_fit_binding]("ridge_fit")
        module.def_function[dbscan_fit_binding]("dbscan_fit")
        module.def_function[dbscan_fit_core_binding]("dbscan_fit_core")
        module.def_function[labeled_reference_predict_binding]("labeled_reference_predict")
        module.def_function[qn_fit_binding]("qn_fit")
        module.def_function[ols_predict_binding]("ols_predict")
        module.def_function[tsvd_transform_binding]("tsvd_transform")
        module.def_function[pca_transform_binding]("pca_transform")
        module.def_function[pca_whiten_transform_binding]("pca_whiten_transform")
        module.def_function[pca_whiten_inverse_transform_binding]("pca_whiten_inverse_transform")
        module.def_function[qn_decision_function_binding]("qn_decision_function")
        module.def_function[qn_sigmoid_binding]("qn_sigmoid")
        module.def_function[qn_softmax_binding]("qn_softmax")
        module.def_function[standard_transform_binding]("standard_transform")
        module.def_function[minmax_transform_binding]("minmax_transform")
        module.def_function[cd_predict_binding]("cd_predict")
        module.def_function[kernel_ridge_predict_binding]("kernel_ridge_predict")
        module.def_function[nystroem_transform_binding]("nystroem_transform")
        module.def_function[rbf_sampler_transform_binding]("rbf_sampler_transform")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_estimators_host: ", error))
