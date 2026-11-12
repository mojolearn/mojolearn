# SPDX-License-Identifier: Apache-2.0
"""The inference-only mixture binding's GaussianMixture descriptor
(`mixture_native_api_v1` of `_mojolearn_mixture_infer_host`): the host
scoring body (mixture/host/gmm_native_host_score.mojo) reached with typed
words, and no fit (this binding compiles none). The mixture door
(bindings/mixture_native.mojo) scores a saved model (`mojolearn.host_model`)
through it; before it existed the door found no typed transport on this
binding and every host-model GaussianMixture score refused."""
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.mixture_native_api import MIXTURE_NATIVE_ABI
from checks.numerics import GLOBAL_NUMERIC_MODE
from mixture.gmm_native_dispatch import GmmNativeBackend, gmm_native_dispatch
from mixture.host.gmm_native_host_score import gmm_native_host_score


struct GmmNativeInferHostBackend(GmmNativeBackend):
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
        raise Error("GaussianMixture native ABI: the inference-only mixture binding has no fit")

    @staticmethod
    def score(operation: Int, weights: Int, means: Int, covariances: Int, precisions: Int, log_det: Int,
              source: Int, destination: Int, components: Int, features: Int, n_iter: Int, converged: Bool,
              lower_bound: Float64, rows: Int) raises:
        gmm_native_host_score(operation, weights, means, covariances, precisions, log_det, source, destination,
                              components, features, n_iter, converged, lower_bound, rows)


def gmm_native_infer_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return gmm_native_dispatch[GmmNativeInferHostBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct GmmNativeInferHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MIXTURE_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(gmm_native_infer_host_v1)


# Its own slot name: the reference host binding and this one may load in one
# process, and one name published twice refuses at init.
comptime GMM_NATIVE_INFER_HOST_API = SharedSlot[GmmNativeInferHostAPI,
    ("MojolearnGmmNativeInferHostIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnGmmNativeInferHostFastV1")]
