# SPDX-License-Identifier: Apache-2.0
"""Typed transport for the default GaussianMixture entries (`gmm_fit`,
`gmm_fit_predict`, `gmm_score_samples`, `gmm_predict_proba`, `gmm_predict`,
`gmm_score_bic_aic`) of `_mojolearn_mixture` and `_mojolearn_mixture_host`.

The mixture door (bindings/mixture_native.mojo) used to build two Python
lists per call (addresses, params), call the binding's Python entry and let
it unpack every word with `Int(py=)` / `Float64(py=)`. This descriptor
carries the same words as plain integers and floats; the two names
(`covariance_type`, `init_params`) cross as NUL-terminated UTF-8 addresses
and are decoded by the same `*_from_name` functions, so every by-name
refusal is unchanged. Same bodies, same words: the bits do not move."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime MIXTURE_NATIVE_ABI = "mojolearn.mixture.native.v1"
comptime MIXTURE_NATIVE_COUNT = 7
comptime MIXTURE_NATIVE_MODE = 0
comptime MIXTURE_NATIVE_CAPABILITIES = 1
comptime MIXTURE_NATIVE_FIT = 2
comptime MIXTURE_NATIVE_SCORE_SAMPLES = 3
comptime MIXTURE_NATIVE_PREDICT_PROBA = 4
comptime MIXTURE_NATIVE_PREDICT = 5
comptime MIXTURE_NATIVE_SCORE_BIC_AIC = 6
# Capability bit 0: the fit writes predict(X) itself (`gmm_fit_predict`,
# row N401; the GPU binding only).
comptime MIXTURE_CAP_FIT_PREDICT = 1


@fieldwise_init
struct GmmFitRequest(Copyable, Movable):
    """`gmm_fit` / `gmm_fit_predict`: addresses of x and the six caller-sized
    outputs (the order of the deleted Python `gmm_fit` entry), the
    labels output (0: none) and the two option names."""
    var source: Int
    var weights: Int
    var means: Int
    var covariances: Int
    var precisions: Int
    var log_det: Int
    var scalars: Int
    var labels: Int
    var covariance_type: Int
    var init_params: Int
    var rows: Int
    var features: Int
    var components: Int
    var max_iter: Int
    var random_state: Int
    var tol: Float64
    var reg_covar: Float64


@fieldwise_init
struct GmmScoreRequest(Copyable, Movable):
    """The scoring entries' shared prefix (`_rebuild_model`) plus x, the
    output and the row count."""
    var weights: Int
    var means: Int
    var covariances: Int
    var precisions: Int
    var log_det: Int
    var source: Int
    var destination: Int
    var components: Int
    var features: Int
    var n_iter: Int
    var converged: Bool
    var rows: Int
    var lower_bound: Float64


@fieldwise_init
struct GmmDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def fit(self, r: GmmFitRequest) raises -> Int:
        """Returns `n_iter_`."""
        var a: Array[Int, 10] = [r.source, r.weights, r.means, r.covariances, r.precisions, r.log_det,
                                 r.scalars, r.labels, r.covariance_type, r.init_params]
        var p: Array[Int, 5] = [r.rows, r.features, r.components, r.max_iter, r.random_state]
        var f: Array[Float64, 2] = [r.tol, r.reg_covar]
        return native_descriptor_call_fixed(self.entry, MIXTURE_NATIVE_FIT, a, p, f)

    def score(self, operation: Int, r: GmmScoreRequest) raises:
        var a: Array[Int, 7] = [r.weights, r.means, r.covariances, r.precisions, r.log_det,
                                r.source, r.destination]
        var p: Array[Int, 5] = [r.components, r.features, r.n_iter, Int(r.converged), r.rows]
        var f: Array[Float64, 1] = [r.lower_bound]
        _ = native_descriptor_call_fixed(self.entry, operation, a, p, f)
