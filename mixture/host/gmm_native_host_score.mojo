# SPDX-License-Identifier: Apache-2.0
"""The host GaussianMixture scoring body of the typed transport
(`mixture_native_api_v1`), shared by the host binding's descriptor
(mixture/host/gmm_native_host.mojo) and the inference-only binding's
(mixture/host/gmm_native_infer_host.mojo), which compiles no fit."""
from std.memory import unsafe_memcpy
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from core.mixture_native_api import (
    MIXTURE_NATIVE_SCORE_SAMPLES, MIXTURE_NATIVE_PREDICT_PROBA, MIXTURE_NATIVE_PREDICT,
)
from mixture.host.gmm_host_oracle import gmmh_score_samples, gmmh_predict_proba, gmmh_predict, gmmh_score_bic_aic


def gmm_native_host_score(operation: Int, weights: Int, means: Int, covariances: Int, precisions: Int, log_det: Int,
                          source: Int, destination: Int, components: Int, features: Int, n_iter: Int,
                          converged: Bool, lower_bound: Float64, rows: Int) raises:
    # bindings/mixture_host_scoring.mojo's `_model_load`, then its bodies.
    var k = components
    var d = features
    var n = rows
    var w = read_f32(weights, max(0, k))
    var m = read_f32(means, max(0, k * d))
    var p = read_f32(precisions, max(0, k * d * d))
    var l = read_f32(log_det, max(0, k))
    var x = read_f32(source, max(0, n * d))
    if operation == MIXTURE_NATIVE_SCORE_SAMPLES:
        var s = gmmh_score_samples(w, m, p, l, k, d, x, n)
        unsafe_memcpy(dest=f32_ptr(destination), src=s.unsafe_ptr(), count=max(0, n))
        _ = s^
    elif operation == MIXTURE_NATIVE_PREDICT_PROBA:
        var r = gmmh_predict_proba(w, m, p, l, k, d, x, n)
        unsafe_memcpy(dest=f32_ptr(destination), src=r.unsafe_ptr(), count=max(0, n * k))
        _ = r^
    elif operation == MIXTURE_NATIVE_PREDICT:
        var labels = gmmh_predict(w, m, p, l, k, d, x, n)
        unsafe_memcpy(dest=i32_ptr(destination), src=labels.unsafe_ptr(), count=max(0, n))
        _ = labels^
    else:
        var r = gmmh_score_bic_aic(w, m, p, l, k, d, x, n)
        var sp = f64_ptr(destination)
        sp.unsafe_store(0, Float64(r.score))
        sp.unsafe_store(1, Float64(r.bic))
        sp.unsafe_store(2, Float64(r.aic))
    _ = w^
    _ = m^
    _ = p^
    _ = l^
    _ = x^
