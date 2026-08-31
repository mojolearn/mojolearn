# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the verified DBSCAN, PCA, tSVD, OLS, Ridge and logistic kernels.

Kept in a separate extension so the independently changing primary binding
does not become a merge point. Arrays cross as borrowed NumPy addresses; all
device buffers and contexts live for one call and no pointer is retained.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext

from dbscan.estimator import dbscan_fit
from kde.estimator import kde_score_samples_host
from decomposition.estimator import (
    inverse_transform_host,
    pca_fit_host,
    pca_transform_host,
    tsvd_fit_host,
    tsvd_transform_host,
)
from glm.estimator import (
    ols_fit_host,
    ols_predict_host,
    qn_decision_function_host,
    qn_fit_host,
    qn_sigmoid_host,
    ridge_fit_host,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float64 buffer address")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def dbscan_fit_binding(
    x_addr: PythonObject,
    labels_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Fit DBSCAN. Returns the propagation pass count.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/density.py`):

        0  n_rows
        1  n_features
        2  eps            (float)
        3  min_samples
        4  budget_mb      (max_mbytes_per_batch; 0 = cuML's own estimate)
        5  max_iter       (max_iterations of the label propagation; 0 =
                           run to the fixed point, DEVIATION 519)
        6  eps_nn_method  (0 = EPS_NN_BRUTE_FORCE, 1 = EPS_NN_RBC)

    Slot 6 was added 2026-08-23 (DEVIATION 516): the wrapper used to have
    no way to choose the eps-neighbourhood arm, so the ball cover -- the
    shipped DEFAULT, DEVIATION 35 -- was the only arm a Python caller
    could reach and the brute-force arm E1U certified was unreachable
    from Python. `dbscan/estimator.mojo` refuses any other value by name.
    """
    if len(params) != 7:
        raise Error(
            "dbscan_fit: params must contain 7 values, got "
            + String(len(params))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var lp = _i32_ptr(Int(py=labels_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var eps = Float64(py=params[2])
    var min_samples = Int(py=params[3])
    var budget = Int(py=params[4])
    var max_iter = Int(py=params[5])
    var eps_nn_method = Int(py=params[6])
    var passes = 0
    with GILReleased(Python()):
        var ctx = DeviceContext()
        passes = dbscan_fit(
            ctx, xp, nr, nf, eps, min_samples, lp, budget, max_iter,
            eps_nn_method,
        )
    return PythonObject(passes)


def pca_fit_binding(
    x_addr: PythonObject,
    components_addr: PythonObject,
    mean_addr: PythonObject,
    explained_addr: PythonObject,
    ratio_addr: PythonObject,
    singular_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Fit PCA and write its five public arrays; return noise variance."""
    if len(params) != 3:
        raise Error("pca_fit: params must contain n_rows, n_features, n_components")
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=components_addr))
    var mp = _f32_ptr(Int(py=mean_addr))
    var ep = _f32_ptr(Int(py=explained_addr))
    var rp = _f32_ptr(Int(py=ratio_addr))
    var sp = _f32_ptr(Int(py=singular_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    var noise = Float64(0.0)
    with GILReleased(Python()):
        var ctx = DeviceContext()
        noise = pca_fit_host(
            ctx, xp, cp, mp, ep, rp, sp, nr, nf, nc
        )
    return PythonObject(noise)


def pca_transform_binding(
    x_addr: PythonObject,
    mean_addr: PythonObject,
    components_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    if len(params) != 3:
        raise Error("pca_transform: params must contain 3 values")
    var xp = _f32_ptr(Int(py=x_addr))
    var mp = _f32_ptr(Int(py=mean_addr))
    var cp = _f32_ptr(Int(py=components_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    with GILReleased(Python()):
        var ctx = DeviceContext()
        pca_transform_host(ctx, xp, mp, cp, op, nr, nf, nc)
    return PythonObject(0)


def tsvd_fit_binding(
    x_addr: PythonObject,
    components_addr: PythonObject,
    singular_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Fit tSVD. Only components and singular values are claimed."""
    if len(params) != 3:
        raise Error("tsvd_fit: params must contain 3 values")
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=components_addr))
    var sp = _f32_ptr(Int(py=singular_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    with GILReleased(Python()):
        var ctx = DeviceContext()
        tsvd_fit_host(ctx, xp, cp, sp, nr, nf, nc)
    return PythonObject(0)


def tsvd_transform_binding(
    x_addr: PythonObject,
    components_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    if len(params) != 3:
        raise Error("tsvd_transform: params must contain 3 values")
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=components_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    with GILReleased(Python()):
        var ctx = DeviceContext()
        tsvd_transform_host(ctx, xp, cp, op, nr, nf, nc)
    return PythonObject(0)


def inverse_transform_binding(
    scores_addr: PythonObject,
    components_addr: PythonObject,
    mean_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Reconstruct scores @ components, optionally adding a PCA mean.

    params: n_rows, n_features, n_components, add_mean.
    """
    if len(params) != 4:
        raise Error("inverse_transform: params must contain 4 values")
    var zp = _f32_ptr(Int(py=scores_addr))
    var cp = _f32_ptr(Int(py=components_addr))
    var mp = _f32_ptr(Int(py=mean_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    var add_mean = Int(py=params[3]) != 0
    with GILReleased(Python()):
        var ctx = DeviceContext()
        inverse_transform_host(ctx, zp, cp, mp, op, nr, nf, nc, add_mean)
    return PythonObject(0)


def ols_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    coef_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    if len(params) != 2:
        raise Error("ols_fit: params must contain n_rows, n_features")
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var wp = _f32_ptr(Int(py=coef_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    with GILReleased(Python()):
        var ctx = DeviceContext()
        ols_fit_host(ctx, xp, yp, wp, nr, nf)
    return PythonObject(0)


def ols_predict_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """GPU matrix-vector prediction; Python adds the scalar intercept."""
    if len(params) != 3:
        raise Error("ols_predict: params must contain n_rows, n_features, intercept")
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=coef_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var intercept = Float32(Float64(py=params[2]))
    with GILReleased(Python()):
        var ctx = DeviceContext()
        ols_predict_host(ctx, xp, cp, op, nr, nf, intercept)
    return PythonObject(0)


def ridge_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    coef_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`ridgeFit`, the eig arm (DEVIATION 545). params: n_rows, n_features,
    alpha. The intercept is the Python layer's host centering, as for OLS."""
    if len(params) != 3:
        raise Error("ridge_fit: params must contain n_rows, n_features, alpha")
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var wp = _f32_ptr(Int(py=coef_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var alpha = Float32(Float64(py=params[2]))
    with GILReleased(Python()):
        var ctx = DeviceContext()
        ridge_fit_host(ctx, xp, yp, wp, nr, nf, alpha)
    return PythonObject(0)


def qn_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    coef_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`qnFit`, QN_LOSS_LOGISTIC (DEVIATIONS 546-549). params: n_rows,
    n_features, n_classes, penalty_l1, penalty_l2, grad_tol, change_tol,
    max_iter, linesearch_max_iter, lbfgs_memory, fit_intercept,
    penalty_normalized, has_sample_weight -- cuML's `qn_params` in its
    field order. Returns num_iters; info[0] = objective, info[1] = retcode."""
    if len(params) != 13:
        raise Error("qn_fit: params must carry the 13 qn_params fields")
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var wp = _f32_ptr(Int(py=coef_addr))
    var ip = _f32_ptr(Int(py=info_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    var l1 = Float64(py=params[3])
    var l2 = Float64(py=params[4])
    var grad_tol = Float64(py=params[5])
    var change_tol = Float64(py=params[6])
    var max_iter = Int(py=params[7])
    var ls_max = Int(py=params[8])
    var mem = Int(py=params[9])
    var fit_intercept = Int(py=params[10]) != 0
    var normalized = Int(py=params[11]) != 0
    var has_sw = Int(py=params[12]) != 0
    var iters = 0
    with GILReleased(Python()):
        var ctx = DeviceContext()
        iters = qn_fit_host(
            ctx, xp, yp, wp, ip, nr, nf, nc, l1, l2, grad_tol, change_tol,
            max_iter, ls_max, mem, fit_intercept, normalized, has_sw,
        )
    return PythonObject(iters)


def qn_decision_function_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`qnDecisionFunction`: scores = X w + b. params: n_rows, n_features,
    fit_intercept."""
    if len(params) != 3:
        raise Error("qn_decision_function: params must contain n_rows, n_features, fit_intercept")
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=coef_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var fi = Int(py=params[2]) != 0
    with GILReleased(Python()):
        var ctx = DeviceContext()
        qn_decision_function_host(ctx, xp, cp, op, nr, nf, fi)
    return PythonObject(0)


def qn_sigmoid_binding(
    scores_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The binary predict_proba link on the host through identical_exp64
    (DEVIATION 549): out is float64 (n_rows, 2)."""
    if len(params) != 1:
        raise Error("qn_sigmoid: params must contain n_rows")
    var sp = _f32_ptr(Int(py=scores_addr))
    var op = _f64_ptr(Int(py=out_addr))
    var nr = Int(py=params[0])
    with GILReleased(Python()):
        qn_sigmoid_host(sp, op, nr)
    return PythonObject(0)


def kde_score_samples_binding(
    train_addr: PythonObject,
    query_addr: PythonObject,
    weights_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    kernel: PythonObject,
    metric: PythonObject,
) raises -> PythonObject:
    """KernelDensity.score_samples (kde/, DEVIATIONS 600-604): log density
    of each query row under the fitted training set. Writes `n_query`
    float32 to `out_addr`. `params`, in this order (mirrored in
    `python/mojolearn/density.py`):

        0  n_train
        1  n_query
        2  n_features
        3  bandwidth   (float)
        4  has_weights (0/1; weights_addr is read only when 1)

    `kernel` and `metric` are the sklearn/cuML names; the host entry
    refuses every unported one BY NAME (kde/estimator.mojo). Returns
    n_query. Added 2026-08-23 by the identity lane on the kde lane's
    hand-off (kde/README.md).
    """
    if len(params) != 5:
        raise Error(
            "kde_score_samples: params must contain 5 values, got "
            + String(len(params))
        )
    var tp = _f32_ptr(Int(py=train_addr))
    var qp = _f32_ptr(Int(py=query_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var n_train = Int(py=params[0])
    var n_query = Int(py=params[1])
    var n_features = Int(py=params[2])
    var bandwidth = Float32(Float64(py=params[3]))
    var has_weights = Int(py=params[4]) != 0
    var kname = String(py=kernel)
    var mname = String(py=metric)
    var train = List[Float32]()
    var query = List[Float32]()
    var weights = List[Float32]()
    for i in range(n_train * n_features):
        train.append(tp.unsafe_load(i))
    for i in range(n_query * n_features):
        query.append(qp.unsafe_load(i))
    if has_weights:
        var wp = _f32_ptr(Int(py=weights_addr))
        for i in range(n_train):
            weights.append(wp.unsafe_load(i))
    var out = List[Float32]()
    with GILReleased(Python()):
        out = kde_score_samples_host(
            train, n_train, query, n_query, n_features, bandwidth, kname,
            mname, weights, has_weights,
        )
    for i in range(n_query):
        op.unsafe_store(i, out[i])
    return PythonObject(n_query)


def estimators_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_estimators() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_estimators")
        m.def_function[estimators_vendor_binding]("estimators_vendor")
        m.def_function[dbscan_fit_binding]("dbscan_fit")
        m.def_function[kde_score_samples_binding]("kde_score_samples")
        m.def_function[pca_fit_binding]("pca_fit")
        m.def_function[pca_transform_binding]("pca_transform")
        m.def_function[tsvd_fit_binding]("tsvd_fit")
        m.def_function[tsvd_transform_binding]("tsvd_transform")
        m.def_function[inverse_transform_binding]("inverse_transform")
        m.def_function[ols_fit_binding]("ols_fit")
        m.def_function[ols_predict_binding]("ols_predict")
        m.def_function[ridge_fit_binding]("ridge_fit")
        m.def_function[qn_fit_binding]("qn_fit")
        m.def_function[qn_decision_function_binding]("qn_decision_function")
        m.def_function[qn_sigmoid_binding]("qn_sigmoid")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_estimators: ", e))
