# SPDX-License-Identifier: Apache-2.0
"""The host binding's GaussianMixture descriptor (`mixture_native_api_v1` of
`_mojolearn_mixture_host`): the host fit body and the shared host scoring
body (mixture/host/gmm_native_host_score.mojo), reached with typed words."""
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.hostptr import f32_ptr, f64_ptr, read_f32
from core.mixture_native_api import MIXTURE_NATIVE_ABI
from checks.numerics import GLOBAL_NUMERIC_MODE
from mixture.gmm_native_dispatch import GmmNativeBackend, gmm_native_dispatch
from mixture.host.gmm_host_oracle import GmmHostParams, gmmh_covariance_type_from_name, gmmh_init_params_from_name
from mixture.host.gmm_native_host_score import gmm_native_host_score
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
        gmm_native_host_score(operation, weights, means, covariances, precisions, log_det, source, destination,
                              components, features, n_iter, converged, lower_bound, rows)


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
