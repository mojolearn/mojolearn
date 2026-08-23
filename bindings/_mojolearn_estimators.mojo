"""CPython boundary for the verified DBSCAN, PCA, tSVD and OLS kernels.

Kept in a separate extension so the independently changing primary binding
does not become a merge point. Arrays cross as borrowed NumPy addresses; all
device buffers and contexts live for one call and no pointer is retained.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from dbscan.estimator import dbscan_fit
from decomposition.estimator import (
    inverse_transform_host,
    pca_fit_host,
    pca_transform_host,
    tsvd_fit_host,
    tsvd_transform_host,
)
from glm.estimator import ols_fit_host, ols_predict_host


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


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


@export
def PyInit__mojolearn_estimators() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_estimators")
        m.def_function[dbscan_fit_binding]("dbscan_fit")
        m.def_function[pca_fit_binding]("pca_fit")
        m.def_function[pca_transform_binding]("pca_transform")
        m.def_function[tsvd_fit_binding]("tsvd_fit")
        m.def_function[tsvd_transform_binding]("tsvd_transform")
        m.def_function[inverse_transform_binding]("inverse_transform")
        m.def_function[ols_fit_binding]("ols_fit")
        m.def_function[ols_predict_binding]("ols_predict")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_estimators: ", e))
