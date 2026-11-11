# SPDX-License-Identifier: Apache-2.0
"""The host binding's GaussianMixture descriptor (`mixture_native_api_v1` of
`_mojolearn_mixture_host`): the bodies `gmm_fit_binding` and
bindings/mixture_host_scoring.mojo run, reached with typed words."""
from std.memory import unsafe_memcpy
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from core.mixture_native_api import (
    MIXTURE_NATIVE_ABI, MIXTURE_NATIVE_SCORE_SAMPLES, MIXTURE_NATIVE_PREDICT_PROBA, MIXTURE_NATIVE_PREDICT,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from mixture.gmm_native_dispatch import GmmNativeBackend, gmm_native_dispatch
from mixture.host.gmm_host_oracle import (
    GmmHostParams, gmmh_covariance_type_from_name, gmmh_init_params_from_name,
    gmmh_score_samples, gmmh_predict_proba, gmmh_predict, gmmh_score_bic_aic,
)
from mixture.host.gmm_host_entries import _gmm_fit_run


struct GmmNativeHostBackend(GmmNativeBackend):
    @staticmethod
    def mode() -> Int:
        return GLOBAL_NUMERIC_MODE

    @staticmethod
    def capabilities() -> Int:
        return 0

    @staticmethod
    def fit(source: Int, rows: Int, features: Int, components: Int, covariance_type: String,
            tol: Float64, reg_covar: Float64, max_iter: Int, init_params: String, random_state: Int,
            weights: Int, means: Int, covariances: Int, precisions: Int, log_det: Int, scalars: Int,
            labels: Int) raises -> Int:
        if labels != 0:
            raise Error("GaussianMixture native ABI: the host binding has no fit_predict")
        # gmm_fit_binding's GmmHostParams, field for field and in its order.
        var gp = GmmHostParams(
            components,
            gmmh_covariance_type_from_name(covariance_type),
            Float32(tol),
            Float32(reg_covar),
            max_iter,
            gmmh_init_params_from_name(init_params),
            UInt64(random_state),
        )
        var x = read_f32(source, max(0, rows * features))
        var n_iter = _gmm_fit_run(x, rows, features, gp, f32_ptr(weights), f32_ptr(means), f32_ptr(covariances),
                                  f32_ptr(precisions), f32_ptr(log_det), f64_ptr(scalars))
        _ = x^
        return n_iter

    @staticmethod
    def score(operation: Int, weights: Int, means: Int, covariances: Int, precisions: Int, log_det: Int,
              source: Int, destination: Int, components: Int, features: Int, n_iter: Int, converged: Bool,
              lower_bound: Float64, rows: Int) raises:
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


def gmm_native_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return gmm_native_dispatch[GmmNativeHostBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct GmmNativeHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MIXTURE_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(gmm_native_host_v1)


comptime GMM_NATIVE_HOST_API = SharedSlot[GmmNativeHostAPI,
    ("MojolearnGmmNativeHostIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnGmmNativeHostFastV1")]
