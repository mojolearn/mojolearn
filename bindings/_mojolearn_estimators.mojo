# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the verified DBSCAN, PCA, tSVD, OLS, Ridge and logistic kernels.

Kept in a separate extension so the independently changing primary binding
does not become a merge point. Arrays cross as borrowed NumPy addresses; all
device buffers and contexts live for one call and no pointer is retained.
"""

# DEVIATION 2486: shared byte-preserving host copies.
from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32, copy_f32
from std.os import abort
from std.math import isfinite
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext

from core.labeled_reference_predict import labeled_reference_predict
from dbscan.estimator import dbscan_fit
from kde.estimator import kde_score_samples_host_ptr
from kde.resident_fit import (
    kde_fit_prepare,
    kde_fit_release,
    kde_score_samples_resident,
)
from decomposition.estimator import (
    inverse_transform_host,
    pca_fit_host,
    pca_fit_full_host,
    pca_transform_host,
    pca_whiten_transform_host,
    pca_whiten_inverse_transform_host,
    tsvd_fit_host,
    tsvd_transform_host,
)
from glm.estimator import (
    ols_fit_host,
    ols_predict_host,
    qn_decision_function_host,
    qn_fit_host,
    qn_predict_binary_host,
    qn_sigmoid_host,
    qn_softmax_host,
    ridge_fit_host,
)
from decomposition.impl.linalg.detail.svd_full import pca_full_validate


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    return f64_ptr(addr)


def dbscan_fit_binding(
    x_addr: PythonObject,
    labels_addr: PythonObject,
    weight_addr: PythonObject,
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
        7  metric         (0 = DBSCAN_METRIC_L2, 1 = DBSCAN_METRIC_L1)

    Slot 6 was added 2026-08-23 (DEVIATION 516): the wrapper used to have
    no way to choose the eps-neighbourhood arm, so the ball cover -- the
    shipped DEFAULT, DEVIATION 35 -- was the only arm a Python caller
    could reach and the brute-force arm E1U certified was unreachable
    from Python. `dbscan/estimator.mojo` refuses any other value by name.

    Slot 7 and `weight_addr` were added 2026-09-01 with the L1 arm
    (DEVIATION 27) and `sample_weight`. `weight_addr` is a SEPARATE
    ARGUMENT and not a params slot because it is an ARRAY ADDRESS and the
    params list is scalars; `0` is their `sample_weight == nullptr` and is
    what an unweighted fit passes. The array must be `n_rows` contiguous
    float32, which `density.py` guarantees with `as_f32_c`.
    """
    return _dbscan_fit_run(x_addr, labels_addr, weight_addr, 0, params)


def dbscan_fit_core_binding(
    x_addr: PythonObject,
    labels_addr: PythonObject,
    weight_addr: PythonObject,
    core_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`dbscan_fit` with the fit's core mask read back, for
    `DBSCAN(prediction_data=True)` (lane/inference-transductive-predict,
    2026-09-15). The params list is `dbscan_fit`'s; `core_addr` receives
    `n_rows` uint8, 1 where the fit's core test held. The mask is a COPY of
    the device buffer the fit computed (`dbscan_fit_impl_weighted`'s
    `out_core_addr`); the labels and the pass count are the same call's."""
    var ca = Int(py=core_addr)
    if ca == 0:
        raise Error("dbscan_fit_core: core_addr must be an array address, got 0")
    return _dbscan_fit_run(x_addr, labels_addr, weight_addr, ca, params)


def labeled_reference_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Out-of-sample labels for DBSCAN and AgglomerativeClustering
    (`core/labeled_reference_predict.mojo`; the rule is stated in
    `core/labeled_reference_host_predict.mojo`). DEVIATION 2740: new
    capability, neither cuML nor scikit-learn has it.

    `addrs`, in this exact order (mirrored in `python/mojolearn/density.py`
    and `python/mojolearn/_hierarchy_impl.py`): refs (float32 n_refs x
    n_features), keys (int32 n_refs), ref_labels (int32 n_refs), queries
    (float32 n_queries x n_features), out_labels (int32 n_queries), out_refs
    (int32 n_queries). `params`: n_refs, n_queries, n_features, metric (0 L2,
    1 L1), eps (float, read only when has_thresh), has_thresh (0/1).
    Returns 0."""
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
    var rp = _f32_ptr(Int(py=addrs[0]))
    var kp = _i32_ptr(Int(py=addrs[1]))
    var lp = _i32_ptr(Int(py=addrs[2]))
    var qp = _f32_ptr(Int(py=addrs[3]))
    var olp = _i32_ptr(Int(py=addrs[4]))
    var orp = _i32_ptr(Int(py=addrs[5]))
    var n_refs = Int(py=params[0])
    var n_queries = Int(py=params[1])
    var n_features = Int(py=params[2])
    var metric = Int(py=params[3])
    var eps = Float64(py=params[4])
    var has_thresh = Int(py=params[5]) != 0
    with GILReleased(Python()):
        var ctx = DeviceContext()
        labeled_reference_predict(
            ctx, rp, n_refs, kp, lp, qp, n_queries, n_features, metric, eps,
            has_thresh, olp, orp,
        )
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
    var xp = _f32_ptr(Int(py=x_addr))
    var lp = _i32_ptr(Int(py=labels_addr))
    var wa = Int(py=weight_addr)
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var eps = Float64(py=params[2])
    var min_samples = Int(py=params[3])
    var budget = Int(py=params[4])
    var max_iter = Int(py=params[5])
    var eps_nn_method = Int(py=params[6])
    var metric = Int(py=params[7])
    var passes = 0
    with GILReleased(Python()):
        var ctx = DeviceContext()
        passes = dbscan_fit(
            ctx, xp, nr, nf, eps, min_samples, lp, budget, max_iter,
            eps_nn_method, metric, wa, core_address,
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


def pca_fit_full_binding(
    x_addr: PythonObject,
    components_addr: PythonObject,
    mean_addr: PythonObject,
    explained_addr: PythonObject,
    ratio_addr: PythonObject,
    singular_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Additive R-SVD fit export; same five outputs as pca_fit.

    This exposes existing arithmetic, not a new numerical certificate.
    Validate its tall-matrix contract before creating a device context.
    """
    if len(params) != 3:
        raise Error("pca_fit_full: params must contain n_rows, n_features, n_components")
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    pca_full_validate(nr, nf, nc)
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=components_addr))
    var mp = _f32_ptr(Int(py=mean_addr))
    var ep = _f32_ptr(Int(py=explained_addr))
    var rp = _f32_ptr(Int(py=ratio_addr))
    var sp = _f32_ptr(Int(py=singular_addr))
    var noise = Float64(0.0)
    with GILReleased(Python()):
        var ctx = DeviceContext()
        noise = pca_fit_full_host(ctx, xp, cp, mp, ep, rp, sp, nr, nf, nc)
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


def _pca_whiten_pointer(
    address: Int, count: Int,
) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if address <= 0 or address % 4 != 0:
        raise Error("PCA whitening requires positive aligned FP32 pointers")
    if address > 9223372036854775807 - count * 4:
        raise Error("PCA whitening pointer span overflows signed Int")
    return _f32_ptr(address)


def _pca_whiten_finite(
    pointer: MutPointer[Float32, MutUntrackedOrigin], count: Int,
) raises:
    for i in range(count):
        if not isfinite(pointer.unsafe_load(i)):
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
    # Additive ABI only: the unwhitened transform exports keep their arity
    # and arithmetic. All host input checks precede the DeviceContext.
    if len(params) != 4:
        raise Error("PCA whitening params require rows, features, components, fit_rows")
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    var nfit = Int(py=params[3])
    if nr < 1 or nf < 2 or nc < 1 or nc > nf or nfit < 2:
        raise Error("PCA whitening requires positive rows/components, features>=2 and fit_rows>=2")
    if nr > 2147483647 or nf > 2147483647 or nc > 2147483647 or nfit > 2147483647:
        raise Error("PCA whitening dimensions exceed Int32")
    if nr > 2147483647 // nf or nc > 2147483647 // nf:
        raise Error("PCA whitening matrix cell count exceeds Int32")
    var input_count = nr * (nc if inverse else nf)
    var output_count = nr * (nf if inverse else nc)
    var xa = Int(py=input_addr)
    var ma = Int(py=mean_addr)
    var ca = Int(py=components_addr)
    var sa = Int(py=singular_addr)
    var oa = Int(py=out_addr)
    var xp = _pca_whiten_pointer(xa, input_count)
    var mp = _pca_whiten_pointer(ma, nf)
    var cp = _pca_whiten_pointer(ca, nc * nf)
    var sp = _pca_whiten_pointer(sa, nc)
    var op = _pca_whiten_pointer(oa, output_count)
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
        if oa < starts[i] + counts[i] * 4 and starts[i] < oa + output_count * 4:
            raise Error("PCA whitening output must not overlap any input")
    _pca_whiten_finite(xp, input_count)
    _pca_whiten_finite(mp, nf)
    _pca_whiten_finite(cp, nc * nf)
    _pca_whiten_finite(sp, nc)
    for i in range(nc):
        if sp.unsafe_load(i) < Float32(0):
            raise Error("PCA whitening singular values must be nonnegative")
    with GILReleased(Python()):
        var ctx = DeviceContext()
        if inverse:
            pca_whiten_inverse_transform_host(ctx, xp, cp, sp, mp, op, nr, nf, nc, nfit)
        else:
            pca_whiten_transform_host(ctx, xp, mp, cp, sp, op, nr, nf, nc, nfit)
    _pca_whiten_finite(op, output_count)
    return PythonObject(0)


def pca_whiten_transform_binding(
    x_addr: PythonObject, mean_addr: PythonObject,
    components_addr: PythonObject, singular_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _pca_whiten_apply(
        x_addr, mean_addr, components_addr, singular_addr, out_addr, params, False)


def pca_whiten_inverse_transform_binding(
    scores_addr: PythonObject, components_addr: PythonObject,
    singular_addr: PythonObject, mean_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    return _pca_whiten_apply(
        scores_addr, mean_addr, components_addr, singular_addr, out_addr, params, True)


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
    """`qnFit` (DEVIATIONS 546-549). params: n_rows, n_features,
    n_classes, penalty_l1, penalty_l2, grad_tol, change_tol, max_iter,
    linesearch_max_iter, lbfgs_memory, fit_intercept, penalty_normalized,
    has_sample_weight -- cuML's `qn_params` in its field order -- and,
    since lane/logistic-multiclass (2026-09-14), an OPTIONAL 14th, the loss
    id: QN_LOSS_LOGISTIC (0) with `n_classes == 2`, the value a 13-field
    call gets, or QN_LOSS_SOFTMAX (2) with `n_classes > 2`, the coef buffer
    then `n_classes * (n_features + fit_intercept)` floats. Returns
    num_iters; info[0] = objective, info[1] = retcode."""
    if len(params) != 13 and len(params) != 14:
        raise Error("qn_fit: params must carry the 13 qn_params fields, plus an optional 14th, the loss id")
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
    var loss = Int(py=params[13]) if len(params) == 14 else 0
    var iters = 0
    with GILReleased(Python()):
        var ctx = DeviceContext()
        iters = qn_fit_host(
            ctx, xp, yp, wp, ip, nr, nf, nc, l1, l2, grad_tol, change_tol,
            max_iter, ls_max, mem, fit_intercept, normalized, has_sw, loss,
        )
    return PythonObject(iters)


def qn_decision_function_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`qnDecisionFunction`: scores = X w + b. params: n_rows, n_features,
    fit_intercept and, since lane/logistic-multiclass (2026-09-14), an
    OPTIONAL 4th, n_classes: absent or below 3 is the binary shape (out is
    n_rows floats, the contract every 3-field caller has), `n_classes > 2`
    the softmax shape (out is `n_rows * n_classes` floats, row-major)."""
    if len(params) != 3 and len(params) != 4:
        raise Error("qn_decision_function: params must contain n_rows, n_features, fit_intercept and an optional n_classes")
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=coef_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var fi = Int(py=params[2]) != 0
    var nc = Int(py=params[3]) if len(params) == 4 else 1
    with GILReleased(Python()):
        var ctx = DeviceContext()
        qn_decision_function_host(ctx, xp, cp, op, nr, nf, fi, nc)
    return PythonObject(0)


def qn_predict_binary_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Binary `qn_predict`: int64 0/1 codes under strict score > 0."""
    if len(params) != 3:
        raise Error("qn_predict_binary: params must contain n_rows, n_features, fit_intercept")
    var nr = Int(py=params[0])
    var nf = Int(py=params[1])
    var fi = Int(py=params[2]) != 0
    var op = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=out_addr))
    with GILReleased(Python()):
        var ctx = DeviceContext()
        qn_predict_binary_host(
            ctx, _f32_ptr(Int(py=x_addr)), _f32_ptr(Int(py=coef_addr)),
            op, nr, nf, fi,
        )
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


def qn_softmax_binding(
    scores_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The multinomial predict_proba link on the host through
    identical_exp64 (lane/logistic-multiclass, 2026-09-14,
    `qn_softmax_host`): scores is float32 (n_rows, n_classes) row-major,
    out is float64 (n_rows, n_classes). params: n_rows, n_classes."""
    if len(params) != 2:
        raise Error("qn_softmax: params must contain n_rows, n_classes")
    var sp = _f32_ptr(Int(py=scores_addr))
    var op = _f64_ptr(Int(py=out_addr))
    var nr = Int(py=params[0])
    var nc = Int(py=params[1])
    if nc < 3:
        raise Error("qn_softmax: n_classes must be at least 3; the binary link is qn_sigmoid")
    with GILReleased(Python()):
        qn_softmax_host(sp, op, nr, nc)
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
    refuses every unimplemented one BY NAME (kde/estimator.mojo). Returns
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
    # DEVIATION 2660: X and the queries are validated and staged from the
    # caller's memory (no `List` copies); the weights, `n_train` values
    # that `kde_fit_validate` and `host_sum_weights` read, are still one
    # owned copy. The scores are written to `out_addr` from the pinned
    # download buffer. Same values in, same bits out.
    var weights = List[Float32]()
    if has_weights:
        var wp = _f32_ptr(Int(py=weights_addr))
        weights = read_f32(Int(wp), max(0, n_train))
    with GILReleased(Python()):
        kde_score_samples_host_ptr(
            tp, n_train, qp, n_query, n_features, bandwidth, kname,
            mname, weights, has_weights, op,
        )
    return PythonObject(n_query)


def kde_fit_prepare_binding(
    train_addr: PythonObject,
    weights_addr: PythonObject,
    params: PythonObject,
    kernel: PythonObject,
    metric: PythonObject,
) raises -> PythonObject:
    """Validate and upload a KDE fit set once (DEVIATION 3003); returns the
    handle `kde_score_samples_resident` scores through. `params`, in this
    order (mirrored in `python/mojolearn/density.py`):

        0  n_train
        1  n_features
        2  bandwidth   (float)
        3  has_weights (0/1; weights_addr is read only when 1)

    Refuses, by name, everything `kde_score_samples` refuses about the fit
    set. Release with `kde_fit_release`."""
    if len(params) != 4:
        raise Error(
            "kde_fit_prepare: params must contain 4 values, got "
            + String(len(params))
        )
    var tp = _f32_ptr(Int(py=train_addr))
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var bandwidth = Float32(Float64(py=params[2]))
    var has_weights = Int(py=params[3]) != 0
    var kname = String(py=kernel)
    var mname = String(py=metric)
    var weights = List[Float32]()
    if has_weights:
        var wp = _f32_ptr(Int(py=weights_addr))
        weights = read_f32(Int(wp), max(0, n_train))
    var handle: Int
    with GILReleased(Python()):
        handle = kde_fit_prepare(tp, n_train, n_features, bandwidth, kname, mname, weights, has_weights)
    return PythonObject(handle)


def kde_fit_release_binding(handle: PythonObject) raises -> PythonObject:
    """Drop a resident KDE fit set (DEVIATION 3003)."""
    kde_fit_release(Int(py=handle))
    return PythonObject(None)


def kde_score_samples_resident_binding(
    handle: PythonObject,
    query_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    kernel: PythonObject,
    metric: PythonObject,
) raises -> PythonObject:
    """`kde_score_samples` over a resident fit set (DEVIATION 3003).
    `params`: [n_query, n_features, bandwidth]. Writes `n_query` float32
    to `out_addr`; returns n_query."""
    if len(params) != 3:
        raise Error(
            "kde_score_samples_resident: params must contain 3 values, got "
            + String(len(params))
        )
    var h = Int(py=handle)
    var qp = _f32_ptr(Int(py=query_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var n_query = Int(py=params[0])
    var n_features = Int(py=params[1])
    var bandwidth = Float32(Float64(py=params[2]))
    var kname = String(py=kernel)
    var mname = String(py=metric)
    with GILReleased(Python()):
        kde_score_samples_resident(h, qp, n_query, n_features, bandwidth, kname, mname, op)
    return PythonObject(n_query)


def estimators_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL, 2 DETERMINISTIC. The same shape as `svm_numeric_mode`. This
    binding had no read-back until 2026-09-10, so
    `NumericModeMixin.numeric_mode_used()` fell through to the module path
    hint, which for the FAST tier is the package directory itself and read
    back as the word 'mojolearn' rather than a tier."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


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
        m.def_function[dbscan_parallel_available_binding]("dbscan_parallel_available")
        m.def_function[gram_parallel_available_binding]("gram_parallel_available")
        m.def_function[gram_parallel_available_binding]("gram_outputs_parallel_available")
        m.def_function[gram_parallel_available_binding]("qr_parallel_available")
        m.def_function[glm_parallel_available_binding]("glm_parallel_available")
        m.def_function[estimators_vendor_binding]("estimators_vendor")
        m.def_function[estimators_numeric_mode_binding]("estimators_numeric_mode")
        m.def_function[dbscan_fit_binding]("dbscan_fit")
        m.def_function[dbscan_fit_core_binding]("dbscan_fit_core")
        m.def_function[labeled_reference_predict_binding]("labeled_reference_predict")
        m.def_function[kde_score_samples_binding]("kde_score_samples")
        m.def_function[kde_fit_prepare_binding]("kde_fit_prepare")
        m.def_function[kde_fit_release_binding]("kde_fit_release")
        m.def_function[kde_score_samples_resident_binding]("kde_score_samples_resident")
        m.def_function[pca_fit_binding]("pca_fit")
        m.def_function[pca_fit_full_binding]("pca_fit_full")
        m.def_function[pca_transform_binding]("pca_transform")
        m.def_function[pca_whiten_transform_binding]("pca_whiten_transform")
        m.def_function[pca_whiten_inverse_transform_binding]("pca_whiten_inverse_transform")
        m.def_function[tsvd_fit_binding]("tsvd_fit")
        m.def_function[tsvd_transform_binding]("tsvd_transform")
        m.def_function[inverse_transform_binding]("inverse_transform")
        m.def_function[ols_fit_binding]("ols_fit")
        m.def_function[ols_predict_binding]("ols_predict")
        m.def_function[ridge_fit_binding]("ridge_fit")
        m.def_function[qn_fit_binding]("qn_fit")
        m.def_function[qn_decision_function_binding]("qn_decision_function")
        m.def_function[qn_predict_binary_binding]("qn_predict_binary")
        m.def_function[qn_sigmoid_binding]("qn_sigmoid")
        m.def_function[qn_softmax_binding]("qn_softmax")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_estimators: ", e))


def gram_parallel_available_binding() raises -> PythonObject:
    return PythonObject(1)


def glm_parallel_available_binding() raises -> PythonObject:
    return PythonObject(1)


def dbscan_parallel_available_binding() raises -> PythonObject:
    return PythonObject(1)
