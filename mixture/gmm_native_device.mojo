# SPDX-License-Identifier: Apache-2.0
"""The GPU binding's GaussianMixture descriptor (`mixture_native_api_v1` of
`_mojolearn_mixture`): the fit and scoring bodies (mixture/gmm_entries.mojo)
the deleted Python `gmm_*` entries ran, reached with typed words."""
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32
from core.mixture_native_api import (
    MIXTURE_NATIVE_ABI, MIXTURE_CAP_FIT_PREDICT, MIXTURE_NATIVE_SCORE_SAMPLES, MIXTURE_NATIVE_PREDICT_PROBA,
    MIXTURE_NATIVE_PREDICT, MIXTURE_NATIVE_SCORE_BIC_AIC, MIXTURE_CAP_DEV_STORE, MIXTURE_NATIVE_FIT_STORE,
    MIXTURE_NATIVE_SCORE_STORE,
)
from core.native_descriptor_call import native_descriptor_report
from mixture.gmm_entries import _gmm_fit_device_run, _gmm_score_device_run
from mixture.mixture_store import mixture_store_ctx, mixture_store_buffer
from checks.numerics import GLOBAL_NUMERIC_MODE
from mixture.gmm_native_dispatch import GmmNativeBackend, gmm_native_dispatch
from core.native_string import unsafe_string_from_utf8_cstr
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
        # MIXTURE_CAP_DEV_STORE (lane vdf-scope-cluster2b): the resident store
        # (mixture/mixture_store.mojo) and the two resident forms below.
        return MIXTURE_CAP_FIT_PREDICT | MIXTURE_CAP_DEV_STORE

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


def _w(address: Int, index: Int) -> Int:
    return Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def _f(address: Int, index: Int) -> Float64:
    return Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def _gmm_store_op(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises -> Int:
    """THE RESIDENT FORMS (MIXTURE_CAP_DEV_STORE, lane vdf-scope-cluster2b): the fit and the scores over slot
    `a[0]` / `a[5]` of the image's resident store, on the store's (= the kernels') context. The slot's buffer is
    held (one reference, copied out under the store's lock) for the whole run, so a concurrent free cannot drop
    it; no word of X visits the host."""
    var ctx = mixture_store_ctx()
    if operation == MIXTURE_NATIVE_FIT_STORE:
        if aa == 0 or pp == 0 or ff == 0 or na != 10 or np != 5 or nf != 2:
            raise Error("GaussianMixture native ABI: invalid resident fit request")
        var rows = _w(pp, 0)
        var features = _w(pp, 1)
        # gmm_fit_binding's GmmParams, field for field and in its order (the host-address fit's).
        var gp = GmmParams.default()
        gp.n_components = _w(pp, 2)
        gp.covariance_type = covariance_type_from_name(_gmm_name(_w(aa, 8)))
        gp.tol = Float32(_f(ff, 0))
        gp.reg_covar = Float32(_f(ff, 1))
        gp.max_iter = _w(pp, 3)
        gp.init_params = init_params_from_name(_gmm_name(_w(aa, 9)))
        gp.random_state = UInt64(_w(pp, 4))
        var dx = mixture_store_buffer(_w(aa, 0), rows * features)
        var n_iter = _gmm_fit_device_run(ctx, dx, rows, features, gp, f32_ptr(_w(aa, 1)), f32_ptr(_w(aa, 2)),
                                         f32_ptr(_w(aa, 3)), f32_ptr(_w(aa, 4)), f32_ptr(_w(aa, 5)),
                                         f64_ptr(_w(aa, 6)), _w(aa, 7))
        _ = dx^
        _ = ctx^  # DEVIATION 1946: the context handle goes LAST
        return n_iter
    if aa == 0 or pp == 0 or ff == 0 or na != 7 or np != 6 or nf != 1 or _w(aa, 6) == 0:
        raise Error("GaussianMixture native ABI: invalid resident scoring request")
    var which = _w(pp, 5)
    if which < MIXTURE_NATIVE_SCORE_SAMPLES or which > MIXTURE_NATIVE_SCORE_BIC_AIC:
        raise Error("GaussianMixture native ABI: unknown resident scoring operation " + String(which))
    var k = _w(pp, 0)
    var d = _w(pp, 1)
    var rows = _w(pp, 4)
    # The GPU binding's `_rebuild_model`, as `score` below (covariances_ not copied).
    var model = GaussianMixtureModel(k, d, read_f32(_w(aa, 0), max(0, k)), read_f32(_w(aa, 1), max(0, k * d)),
                                     List[Float32](), read_f32(_w(aa, 3), max(0, k * d * d)),
                                     read_f32(_w(aa, 4), max(0, k)), _w(pp, 2), _w(pp, 3) != 0, Float32(_f(ff, 0)))
    var dx = mixture_store_buffer(_w(aa, 5), rows * d)
    _gmm_score_device_run(which - MIXTURE_NATIVE_SCORE_SAMPLES, model, ctx, dx, rows, _w(aa, 6))
    _ = dx^
    _ = model^
    _ = ctx^  # DEVIATION 1946: the context handle goes LAST
    return 0


def _gmm_name(address: Int) raises -> String:
    if address == 0:
        raise Error("GaussianMixture native ABI: missing option name")
    return unsafe_string_from_utf8_cstr(Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address))


def gmm_native_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    if operation == MIXTURE_NATIVE_FIT_STORE or operation == MIXTURE_NATIVE_SCORE_STORE:
        try:
            if result == 0:
                raise Error("GaussianMixture native ABI: missing result span")
            var answer = _gmm_store_op(operation, aa, na, pp, np, ff, nf)
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
            return 0
        except problem:
            native_descriptor_report(problem, context, sink)
            return 1
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
