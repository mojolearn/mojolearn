# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the Gaussian mixture lane (workstream D, 2026-09-14).

A separate extension module, for `bindings/_mojolearn_gp.mojo`'s reason.
`mixture/estimator.mojo` is reached and nothing is re-decided: the
covariance_type and init_params STRINGS cross as strings and are decoded
by `covariance_type_from_name` / `init_params_from_name` on the Mojo side,
so every by-name refusal there ('tied', 'diag', 'spherical', 'k-means++',
'random_from_data') fires from Python with its own sentence. Every
parameter refusal (`gmm_validate_params`, DEVIATION 1738), every data
refusal (`gmm_validate_data`) and the collapsed-component raise
(DEVIATION 1723) live one layer down.

THE ABI IS THE GP'S: two length-checked lists per entry point, orders
written out here and mirrored in `python/mojolearn/mixture.py`.

THE MODEL CROSSES AS ITS ARRAYS. `GaussianMixtureModel` is a
`fieldwise_init` struct of host lists and scalars; fit writes every field
into caller-sized buffers and the scoring entries rebuild the struct from
the same buffers. `n_iter`, `converged` and `lower_bound` cross too, because
the estimator's header says a model that does not carry them cannot be
compared with another model.

THE GIL is released around every device call, and nothing inside a
`GILReleased` block touches a `PythonObject`.
"""

from std.os import abort
from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, copy_f32, read_f32
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from mixture.estimator import (
    GaussianMixtureModel,
    GmmParams,
    covariance_type_from_name,
    gaussian_mixture_aic,
    gaussian_mixture_bic,
    gaussian_mixture_fit,
    gaussian_mixture_predict,
    gaussian_mixture_predict_proba,
    gaussian_mixture_sample,
    gaussian_mixture_score,
    gaussian_mixture_score_samples,
    init_params_from_name,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    return f64_ptr(addr)


def mixture_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL, 2
    DETERMINISTIC (the GP's shape, for its reason)."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mixture_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


def gmm_parallel_available() raises -> PythonObject:
    """1: every E-step (fit and scoring) reads MOJOLEARN_GMM_DEVICE_COUNT and
    row-shards through mixture/multi_gpu.mojo::gmm_e_step_dispatch."""
    return PythonObject(1)


def _gmm_fit_run(
    x: List[Float32],
    n: Int,
    d: Int,
    params: GmmParams,
    wp: MutPointer[Float32, MutUntrackedOrigin],
    mp: MutPointer[Float32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    var model = gaussian_mixture_fit(x, n, d, params)
    var k = model.n_components
    copy_f32(model.weights.unsafe_ptr(), wp, k)
    copy_f32(model.means.unsafe_ptr(), mp, k * d)
    copy_f32(model.covariances.unsafe_ptr(), cp, k * d * d)
    copy_f32(model.precisions_cholesky.unsafe_ptr(), pp, k * d * d)
    copy_f32(model.log_det_chol.unsafe_ptr(), lp, k)
    sp.unsafe_store(0, Float64(model.n_iter))
    var conv = 0
    if model.converged:
        conv = 1
    sp.unsafe_store(1, Float64(conv))
    sp.unsafe_store(2, Float64(model.lower_bound))
    return model.n_iter


def gmm_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`GaussianMixture(...).fit(X)` (`gaussian_mixture_fit`). Returns
    `n_iter_`. A collapsed component RAISES (DEVIATION 1723).

    `addrs`, in this exact order:

        0  x                      n * d float32, row-major, read
        1  weights_out            k float32, WRITTEN
        2  means_out              k * d float32, WRITTEN
        3  covariances_out        k * d * d float32, WRITTEN (reg_covar on
                                   the diagonal, as scikit-learn's)
        4  precisions_chol_out    k * d * d float32, WRITTEN (upper
                                   triangular per component)
        5  log_det_chol_out       k float32, WRITTEN (DEVIATION 1726)
        6  scalars_out            3 float64, WRITTEN: n_iter, converged
                                   (0/1), lower_bound

    `params`, in this exact order:

        0  n
        1  d
        2  n_components           k
        3  covariance_type        a STRING, decoded on the Mojo side
        4  tol                    (float)
        5  reg_covar              (float)
        6  max_iter
        7  init_params            a STRING, decoded on the Mojo side
        8  random_state
    """
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
    var xp = _f32_ptr(Int(py=addrs[0]))
    var wp = _f32_ptr(Int(py=addrs[1]))
    var mp = _f32_ptr(Int(py=addrs[2]))
    var cp = _f32_ptr(Int(py=addrs[3]))
    var pp = _f32_ptr(Int(py=addrs[4]))
    var lp = _f32_ptr(Int(py=addrs[5]))
    var sp = _f64_ptr(Int(py=addrs[6]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var gp = GmmParams.default()
    gp.n_components = Int(py=params[2])
    gp.covariance_type = covariance_type_from_name(String(py=params[3]))
    gp.tol = Float32(Float64(py=params[4]))
    gp.reg_covar = Float32(Float64(py=params[5]))
    gp.max_iter = Int(py=params[6])
    gp.init_params = init_params_from_name(String(py=params[7]))
    gp.random_state = UInt64(Int(py=params[8]))
    var x = read_f32(Int(xp), max(0, n * d))
    var n_iter = 0
    with GILReleased(Python()):
        n_iter = _gmm_fit_run(x, n, d, gp, wp, mp, cp, pp, lp, sp)
    return PythonObject(n_iter)


def _rebuild_model(
    addrs: PythonObject, params: PythonObject, what: String
) raises -> GaussianMixtureModel:
    """The model from the scoring entries' shared prefix.

    `addrs[0..4]`: weights (k), means (k*d), covariances (k*d*d),
    precisions_chol (k*d*d), log_det_chol (k), all read.
    `params[0..5]`: k, d, n_iter, converged (0/1), lower_bound, n (rows of
    the x that follows).
    """
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
    var n_iter = Int(py=params[2])
    var converged = Int(py=params[3]) != 0
    var lower_bound = Float32(Float64(py=params[4]))
    var weights = read_f32(Int(py=addrs[0]), max(0, k))
    var means = read_f32(Int(py=addrs[1]), max(0, k * d))
    var covariances = read_f32(Int(py=addrs[2]), max(0, k * d * d))
    var precisions = read_f32(Int(py=addrs[3]), max(0, k * d * d))
    var log_det = read_f32(Int(py=addrs[4]), max(0, k))
    return GaussianMixtureModel(
        k,
        d,
        weights^,
        means^,
        covariances^,
        precisions^,
        log_det^,
        n_iter,
        converged,
        lower_bound,
    )


def _gmm_score_samples_run(
    model: GaussianMixtureModel,
    x: List[Float32],
    n: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var s = gaussian_mixture_score_samples(model, x, n)
    copy_f32(s.unsafe_ptr(), op, n)


def gmm_score_samples_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`score_samples(X)`: one log likelihood per row. `addrs[5]` is x
    (n * d, read), `addrs[6]` is out (n float32, WRITTEN); the prefix is
    `_rebuild_model`'s. Returns 0."""
    var model = _rebuild_model(addrs, params, String("gmm_score_samples"))
    var n = Int(py=params[5])
    var x = read_f32(Int(py=addrs[5]), max(0, n * model.n_features))
    var op = _f32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        _gmm_score_samples_run(model, x, n, op)
    return PythonObject(0)


def _gmm_predict_proba_run(
    model: GaussianMixtureModel,
    x: List[Float32],
    n: Int,
    op: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var r = gaussian_mixture_predict_proba(model, x, n)
    copy_f32(r.unsafe_ptr(), op, n * model.n_components)


def gmm_predict_proba_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`predict_proba(X)`: `exp(log_resp)`, `n * k`. `addrs[5]` is x,
    `addrs[6]` is out (n * k float32, WRITTEN). Returns 0."""
    var model = _rebuild_model(addrs, params, String("gmm_predict_proba"))
    var n = Int(py=params[5])
    var x = read_f32(Int(py=addrs[5]), max(0, n * model.n_features))
    var op = _f32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        _gmm_predict_proba_run(model, x, n, op)
    return PythonObject(0)


def _gmm_predict_run(
    model: GaussianMixtureModel,
    x: List[Float32],
    n: Int,
    op: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    var labels = gaussian_mixture_predict(model, x, n)
    for i in range(n):
        op.unsafe_store(i, labels[i])


def gmm_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`predict(X)`: argmax over the weighted log probabilities, ties to
    the lowest component. `addrs[5]` is x, `addrs[6]` is out (n int32,
    WRITTEN). Returns 0."""
    var model = _rebuild_model(addrs, params, String("gmm_predict"))
    var n = Int(py=params[5])
    var x = read_f32(Int(py=addrs[5]), max(0, n * model.n_features))
    var op = _i32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        _gmm_predict_run(model, x, n, op)
    return PythonObject(0)


def _gmm_sample_run(
    model: GaussianMixtureModel,
    n: Int,
    seed: UInt64,
    xp: MutPointer[Float32, MutUntrackedOrigin],
    yp: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    var labels = List[Int32](length=max(0, n), fill=Int32(0))
    var x = gaussian_mixture_sample(model, n, seed, labels)
    copy_f32(x.unsafe_ptr(), xp, n * model.n_features)
    for i in range(n):
        yp.unsafe_store(i, labels[i])


def gmm_sample_binding(
    addrs: PythonObject, params: PythonObject, sample_params: PythonObject
) raises -> PythonObject:
    """`sample(n_samples)`: the model prefix is `_rebuild_model`'s with
    `params[5]` (n) = 0; `addrs[5]` is X out (n_samples * d float32,
    WRITTEN), `addrs[6]` y out (n_samples int32, WRITTEN).
    `sample_params` is, in this order: 0 n_samples, 1 random_state's low 32
    bits, 2 its high 32 bits. Returns n_samples."""
    var model = _rebuild_model(addrs, params, String("gmm_sample"))
    if len(sample_params) != 3:
        raise Error(
            "gmm_sample: sample_params must contain 3 values (n_samples,"
            " random_state low, random_state high), got "
            + String(len(sample_params))
        )
    var n = Int(py=sample_params[0])
    var seed = (UInt64(Int(py=sample_params[2])) << 32) | UInt64(
        Int(py=sample_params[1])
    )
    var xp = _f32_ptr(Int(py=addrs[5]))
    var yp = _i32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        _gmm_sample_run(model, n, seed, xp, yp)
    return PythonObject(n)


def _gmm_score_bic_aic_run(
    model: GaussianMixtureModel,
    x: List[Float32],
    n: Int,
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    sp.unsafe_store(0, Float64(gaussian_mixture_score(model, x, n)))
    sp.unsafe_store(1, Float64(gaussian_mixture_bic(model, x, n)))
    sp.unsafe_store(2, Float64(gaussian_mixture_aic(model, x, n)))


def gmm_score_bic_aic_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`score(X)`, `bic(X)` and `aic(X)` in one call: `addrs[5]` is x,
    `addrs[6]` is out (3 float64, WRITTEN: score, bic, aic). Each is a
    float32 on the Mojo side (the host ascending fold through `ftz` and
    `identical_div`) widened exactly. Returns 0."""
    var model = _rebuild_model(addrs, params, String("gmm_score_bic_aic"))
    var n = Int(py=params[5])
    var x = read_f32(Int(py=addrs[5]), max(0, n * model.n_features))
    var sp = _f64_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        _gmm_score_bic_aic_run(model, x, n, sp)
    return PythonObject(0)


@export
def PyInit__mojolearn_mixture() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_mixture")
        m.def_function[gmm_parallel_available]("gmm_parallel_available")
        m.def_function[mixture_vendor_binding]("mixture_vendor")
        m.def_function[mixture_numeric_mode_binding]("mixture_numeric_mode")
        m.def_function[gmm_fit_binding]("gmm_fit")
        m.def_function[gmm_score_samples_binding]("gmm_score_samples")
        m.def_function[gmm_predict_proba_binding]("gmm_predict_proba")
        m.def_function[gmm_predict_binding]("gmm_predict")
        m.def_function[gmm_score_bic_aic_binding]("gmm_score_bic_aic")
        m.def_function[gmm_sample_binding]("gmm_sample")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_mixture: ", e))
