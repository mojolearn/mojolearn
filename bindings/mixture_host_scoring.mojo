# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The four GaussianMixture scoring entries of the mixture host bindings,
written once (the neighbors and density inference lane, 2026-09-15).

`bindings/_mojolearn_mixture_host.mojo` (the reference binding: fit and
scoring) and `bindings/_mojolearn_mixture_infer_host.mojo` (the inference
binding a wheel ships: scoring only, no `gmmh_fit`) both register these
functions, so the two binaries answer score_samples, predict_proba, predict,
score, bic and aic through one source. The address and params contract is
the GPU binding's (`bindings/_mojolearn_mixture.mojo`), word for word.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from mixture.host.gmm_host_oracle import (
    gmmh_predict,
    gmmh_predict_proba,
    gmmh_score_bic_aic,
    gmmh_score_samples,
)


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
