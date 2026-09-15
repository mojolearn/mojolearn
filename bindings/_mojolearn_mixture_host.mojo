# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_mixture` family: GaussianMixture (CPU
training for the workstream D estimators, 2026-09-15; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md sections 1.1 and 3.2).

HOST ONLY. No DeviceContext, no kernel launch. The fit and the four scoring
entries are `mixture/host/gmm_host_oracle.mojo`, the device path of
`mixture/estimator.mojo` restated on the host over the gemm profile's
normative answer, the Cholesky host profile and the k-means host oracle. So
every model array, `n_iter_`, `converged_` and `lower_bound_` are meant to
be the GPU columns' bytes.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES, with the GPU binding's
address and params contract word for word (`bindings/_mojolearn_mixture.mojo`,
mirrored in `python/mojolearn/mixture.py`), so `GaussianMixture` runs
unchanged on a CPU-only install through `_backend._HOST_MODULES`
(`"_mojolearn_mixture": "_mojolearn_mixture_host"`): `gmm_fit` (7 addresses,
9 params), `gmm_score_samples`, `gmm_predict_proba`, `gmm_predict` and
`gmm_score_bic_aic` (7 addresses, 6 params each), `mixture_vendor` answering
"cpu" and `mixture_numeric_mode`. ABSENT, and so refused BY NAME through
`_HostBinding`: `gmm_parallel_available`, the multi-GPU driver's probe.

The sabotage arm (`mixture_host_sabotage`) is
`gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`): every GEMM leaf walks descending.
"""
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
from checks.numerics import GLOBAL_NUMERIC_MODE
from gemm.host.gemm_oracle import GEMM_ORACLE_HOST_SABOTAGE
from mixture.checks.sample import gmm_sample_host
from mixture.host.gmm_host_oracle import (
    GmmHostParams,
    gmmh_covariance_type_from_name,
    gmmh_fit,
    gmmh_init_params_from_name,
    gmmh_predict,
    gmmh_predict_proba,
    gmmh_score_bic_aic,
    gmmh_score_samples,
)


def mixture_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mixture_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def mixture_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "mixture host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_mixture_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def mixture_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every GEMM leaf descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(GEMM_ORACLE_HOST_SABOTAGE)


def mixture_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def mixture_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def _gmm_fit_run(
    x: List[Float32],
    n: Int,
    d: Int,
    params: GmmHostParams,
    wp: MutPointer[Float32, MutUntrackedOrigin],
    mp: MutPointer[Float32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    var model = gmmh_fit(x, n, d, params)
    var k = model.n_components
    for i in range(k):
        wp.unsafe_store(i, model.weights[i])
        lp.unsafe_store(i, model.log_det_chol[i])
    for i in range(k * d):
        mp.unsafe_store(i, model.means[i])
    for i in range(k * d * d):
        cp.unsafe_store(i, model.covariances[i])
        pp.unsafe_store(i, model.precisions_cholesky[i])
    sp.unsafe_store(0, Float64(model.n_iter))
    sp.unsafe_store(1, Float64(1 if model.converged else 0))
    sp.unsafe_store(2, Float64(model.lower_bound))
    var n_iter = model.n_iter
    _ = model^
    return n_iter


def gmm_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`GaussianMixture(...).fit(X)` on the host. Returns `n_iter_`.
    `addrs`: 0 x, 1 weights_out, 2 means_out, 3 covariances_out,
    4 precisions_chol_out, 5 log_det_chol_out, 6 scalars_out (n_iter,
    converged, lower_bound). `params`: 0 n, 1 d, 2 n_components,
    3 covariance_type (a string), 4 tol, 5 reg_covar, 6 max_iter,
    7 init_params (a string), 8 random_state."""
    if len(addrs) != 7:
        raise Error(
            "gmm_fit: addrs must contain 7 addresses (x, weights_out,"
            " means_out, covariances_out, precisions_chol_out,"
            " log_det_chol_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 9:
        raise Error(
            "gmm_fit: params must contain 9 values (n, d, n_components,"
            " covariance_type, tol, reg_covar, max_iter, init_params,"
            " random_state), got "
            + String(len(params))
        )
    var wp = f32_ptr(Int(py=addrs[1]))
    var mp = f32_ptr(Int(py=addrs[2]))
    var cp = f32_ptr(Int(py=addrs[3]))
    var pp = f32_ptr(Int(py=addrs[4]))
    var lp = f32_ptr(Int(py=addrs[5]))
    var sp = f64_ptr(Int(py=addrs[6]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var gp = GmmHostParams(
        Int(py=params[2]),
        gmmh_covariance_type_from_name(String(py=params[3])),
        Float32(Float64(py=params[4])),
        Float32(Float64(py=params[5])),
        Int(py=params[6]),
        gmmh_init_params_from_name(String(py=params[7])),
        UInt64(Int(py=params[8])),
    )
    var x = read_f32(Int(py=addrs[0]), max(0, n * d))
    var n_iter = 0
    with GILReleased(Python()):
        n_iter = _gmm_fit_run(x, n, d, gp, wp, mp, cp, pp, lp, sp)
    _ = x^
    return PythonObject(n_iter)


@fieldwise_init
struct _Model(Movable):
    var k: Int
    var d: Int
    var n: Int
    var weights: List[Float32]
    var means: List[Float32]
    var precisions: List[Float32]
    var log_det: List[Float32]
    var x: List[Float32]


def _rebuild_model(
    addrs: PythonObject, params: PythonObject, what: String
) raises -> _Model:
    """The scoring entries' shared prefix, the GPU binding's
    `_rebuild_model`: `addrs[0..4]` weights, means, covariances,
    precisions_chol, log_det_chol; `addrs[5]` x; `params` k, d, n_iter,
    converged, lower_bound, n."""
    if len(addrs) != 7:
        raise Error(
            what
            + ": addrs must contain 7 addresses (weights, means,"
            " covariances, precisions_chol, log_det_chol, x, out), got "
            + String(len(addrs))
        )
    if len(params) != 6:
        raise Error(
            what
            + ": params must contain 6 values (k, d, n_iter, converged,"
            " lower_bound, n), got "
            + String(len(params))
        )
    var k = Int(py=params[0])
    var d = Int(py=params[1])
    var n = Int(py=params[5])
    var weights = read_f32(Int(py=addrs[0]), max(0, k))
    var means = read_f32(Int(py=addrs[1]), max(0, k * d))
    var precisions = read_f32(Int(py=addrs[3]), max(0, k * d * d))
    var log_det = read_f32(Int(py=addrs[4]), max(0, k))
    var x = read_f32(Int(py=addrs[5]), max(0, n * d))
    return _Model(k, d, n, weights^, means^, precisions^, log_det^, x^)


def gmm_score_samples_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`score_samples(X)`: `addrs[6]` is out (n float32). Returns 0."""
    var m = _rebuild_model(addrs, params, String("gmm_score_samples"))
    var op = f32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        var s = gmmh_score_samples(
            m.weights, m.means, m.precisions, m.log_det, m.k, m.d, m.x, m.n
        )
        for i in range(m.n):
            op.unsafe_store(i, s[i])
        _ = s^
    _ = m^
    return PythonObject(0)


def gmm_predict_proba_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`predict_proba(X)`: `addrs[6]` is out (n * k float32). Returns 0."""
    var m = _rebuild_model(addrs, params, String("gmm_predict_proba"))
    var op = f32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        var r = gmmh_predict_proba(
            m.weights, m.means, m.precisions, m.log_det, m.k, m.d, m.x, m.n
        )
        for i in range(m.n * m.k):
            op.unsafe_store(i, r[i])
        _ = r^
    _ = m^
    return PythonObject(0)


def gmm_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`predict(X)`: `addrs[6]` is out (n int32). Returns 0."""
    var m = _rebuild_model(addrs, params, String("gmm_predict"))
    var op = i32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        var labels = gmmh_predict(
            m.weights, m.means, m.precisions, m.log_det, m.k, m.d, m.x, m.n
        )
        for i in range(m.n):
            op.unsafe_store(i, labels[i])
        _ = labels^
    _ = m^
    return PythonObject(0)


def gmm_sample_binding(
    addrs: PythonObject, params: PythonObject, sample_params: PythonObject
) raises -> PythonObject:
    """`sample(n_samples)` on the host, the GPU binding's name, arity and
    lists: `addrs[0..4]` the model, `addrs[5]` X out (n_samples * d
    float32), `addrs[6]` y out (n_samples int32); `params` k, d, n_iter,
    converged, lower_bound, 0; `sample_params` n_samples, random_state low
    32 bits, high 32 bits. The arithmetic is
    `mixture/checks/sample.mojo::gmm_sample_host`, the device kernel row for
    row. Returns n_samples."""
    if len(addrs) != 7 or len(params) != 6 or len(sample_params) != 3:
        raise Error(
            "gmm_sample: needs 7 addresses, 6 params and 3 sample_params, got "
            + String(len(addrs))
            + ", "
            + String(len(params))
            + ", "
            + String(len(sample_params))
        )
    var k = Int(py=params[0])
    var d = Int(py=params[1])
    var n = Int(py=sample_params[0])
    var seed = (UInt64(Int(py=sample_params[2])) << 32) | UInt64(
        Int(py=sample_params[1])
    )
    var weights = read_f32(Int(py=addrs[0]), max(0, k))
    var means = read_f32(Int(py=addrs[1]), max(0, k * d))
    var precisions = read_f32(Int(py=addrs[3]), max(0, k * d * d))
    var xp = f32_ptr(Int(py=addrs[5]))
    var yp = i32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        var labels = List[Int32](length=max(0, n), fill=Int32(0))
        var x = gmm_sample_host(weights, means, precisions, k, d, n, seed, labels)
        for i in range(n * d):
            xp.unsafe_store(i, x[i])
        for i in range(n):
            yp.unsafe_store(i, labels[i])
    return PythonObject(n)


def gmm_score_bic_aic_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`score(X)`, `bic(X)` and `aic(X)`: `addrs[6]` is out (3 float64).
    Returns 0."""
    var m = _rebuild_model(addrs, params, String("gmm_score_bic_aic"))
    var sp = f64_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        var r = gmmh_score_bic_aic(
            m.weights, m.means, m.precisions, m.log_det, m.k, m.d, m.x, m.n
        )
        sp.unsafe_store(0, Float64(r.score))
        sp.unsafe_store(1, Float64(r.bic))
        sp.unsafe_store(2, Float64(r.aic))
    _ = m^
    return PythonObject(0)


@export
def PyInit__mojolearn_mixture_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_mixture_host")
        module.def_function[mixture_host_numeric_mode_binding]("mixture_host_numeric_mode")
        module.def_function[mixture_host_vendor_binding]("mixture_host_vendor")
        module.def_function[mixture_host_column_binding]("mixture_host_column")
        module.def_function[mixture_host_sabotage_binding]("mixture_host_sabotage")
        module.def_function[mixture_vendor_binding]("mixture_vendor")
        module.def_function[mixture_numeric_mode_binding]("mixture_numeric_mode")
        module.def_function[gmm_fit_binding]("gmm_fit")
        module.def_function[gmm_score_samples_binding]("gmm_score_samples")
        module.def_function[gmm_predict_proba_binding]("gmm_predict_proba")
        module.def_function[gmm_predict_binding]("gmm_predict")
        module.def_function[gmm_score_bic_aic_binding]("gmm_score_bic_aic")
        module.def_function[gmm_sample_binding]("gmm_sample")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_mixture_host: ", e))
