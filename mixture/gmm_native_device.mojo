# SPDX-License-Identifier: Apache-2.0
"""The GPU binding's GaussianMixture descriptor (`mixture_native_api_v1` of
`_mojolearn_mixture`): the bodies `gmm_fit_binding` and the scoring
bindings run (mixture/gmm_entries.mojo), reached with typed words."""
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from core.mixture_native_api import (
    MIXTURE_NATIVE_ABI, MIXTURE_CAP_FIT_PREDICT, MIXTURE_NATIVE_SCORE_SAMPLES, MIXTURE_NATIVE_PREDICT_PROBA,
    MIXTURE_NATIVE_PREDICT,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from mixture.gmm_native_dispatch import GmmNativeBackend, gmm_native_dispatch
from mixture.estimator import GaussianMixtureModel, GmmParams, covariance_type_from_name, init_params_from_name
from mixture.gmm_entries import (
    _gmm_fit_run, _gmm_score_samples_run, _gmm_predict_proba_run, _gmm_predict_run, _gmm_score_bic_aic_run,
)


struct GmmNativeDeviceBackend(GmmNativeBackend):
    @staticmethod
    def mode() -> Int:
        return GLOBAL_NUMERIC_MODE

    @staticmethod
    def capabilities() -> Int:
        return MIXTURE_CAP_FIT_PREDICT

    @staticmethod
    def fit(source: Int, rows: Int, features: Int, components: Int, covariance_type: String,
            tol: Float64, reg_covar: Float64, max_iter: Int, init_params: String, random_state: Int,
            weights: Int, means: Int, covariances: Int, precisions: Int, log_det: Int, scalars: Int,
            labels: Int) raises -> Int:
        # gmm_fit_binding's GmmParams, field for field and in its order.
        var gp = GmmParams.default()
        gp.n_components = components
        gp.covariance_type = covariance_type_from_name(covariance_type)
        gp.tol = Float32(tol)
        gp.reg_covar = Float32(reg_covar)
        gp.max_iter = max_iter
        gp.init_params = init_params_from_name(init_params)
        gp.random_state = UInt64(random_state)
        var x = read_f32(source, max(0, rows * features))
        var n_iter = _gmm_fit_run(x, rows, features, gp, f32_ptr(weights), f32_ptr(means), f32_ptr(covariances),
                                  f32_ptr(precisions), f32_ptr(log_det), f64_ptr(scalars), labels)
        _ = x^
        return n_iter

    @staticmethod
    def score(operation: Int, weights: Int, means: Int, covariances: Int, precisions: Int, log_det: Int,
              source: Int, destination: Int, components: Int, features: Int, n_iter: Int, converged: Bool,
              lower_bound: Float64, rows: Int) raises:
        # The GPU binding's `_rebuild_model`: covariances_ is read by no
        # scoring kernel and is not copied (p3 rank 296).
        var k = components
        var d = features
        var model = GaussianMixtureModel(k, d, read_f32(weights, max(0, k)), read_f32(means, max(0, k * d)),
                                         List[Float32](), read_f32(precisions, max(0, k * d * d)),
                                         read_f32(log_det, max(0, k)), n_iter, converged, Float32(lower_bound))
        var x = read_f32(source, max(0, rows * d))
        if operation == MIXTURE_NATIVE_SCORE_SAMPLES:
            _gmm_score_samples_run(model, x, rows, f32_ptr(destination))
        elif operation == MIXTURE_NATIVE_PREDICT_PROBA:
            _gmm_predict_proba_run(model, x, rows, f32_ptr(destination))
        elif operation == MIXTURE_NATIVE_PREDICT:
            _gmm_predict_run(model, x, rows, i32_ptr(destination))
        else:
            _gmm_score_bic_aic_run(model, x, rows, f64_ptr(destination))
        _ = x^
        _ = model^


def gmm_native_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return gmm_native_dispatch[GmmNativeDeviceBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct GmmNativeDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MIXTURE_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(gmm_native_device_v1)


comptime GMM_NATIVE_DEVICE_API = SharedSlot[GmmNativeDeviceAPI,
    ("MojolearnGmmNativeDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnGmmNativeDeviceFastV1")]
