# SPDX-License-Identifier: Apache-2.0
# PACK part and the operators of the GaussianMixture
# (python/mojolearn/mixture.py) and BayesianGaussianMixture
# (python/mojolearn/_expansion_cluster.py) door: fit, fit_predict, predict,
# predict_proba, score_samples, score, bic and aic, one native call per public
# method, reached from bindings/family_native_door.mojo for every
# `family_mix_*` operation.
#
# The door was bindings/mixture_public_native_pack.mojo (one file). UNPACK
# (bindings/mixture_public_native_unpack.mojo) reads the model and the
# arguments into typed values in the Python methods' refusal order; RUN
# (mixture/mixture_door_run.mojo, no Python import) holds the number
# refusals, the entry words and the bic / aic arithmetic; the numerical work
# is the typed provider call (`GmmDriver`, `BgmmDriver`: EM iterations and
# n_init restarts run inside one call per fit). This file allocates the
# public outputs, writes the fitted attributes with the names, shapes and
# dtypes the Python set, and dispatches the operations. Same entries, same
# inputs, same order: the bits do not move.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_family_door import admit_family
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setattr, py_fields, py_setitem, py_item
from bindings.native_loaded_module_door import native_loaded_module
from bindings.family_raw_attr_unpack import family_raw_has
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.family_common_unpack import FAMILY_F32P
from bindings.family_common_pack import family_array
from bindings.classical_validate_unpack import classical_check_x, classical_estimator_name
from bindings.classical_validate_run import classical_check_width
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.native_array_view_pack import native_array_ndarray_view
from bindings.svm_mixture_native_door import svm_mixture_operation
from providers.python.estimator_binding_pyapi_door import estimator_binding
from bindings.cluster_side_records_unpack import (
    cluster_is_subclass,
    cluster_record_put,
    cluster_record_drop,
)
from bindings.cluster_side_records_pack import cluster_record_get
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from core.bgmm_native_api import BgmmResultSlot
from core.mixture_native_api import (
    MIXTURE_NATIVE_CAPABILITIES, MIXTURE_CAP_FIT_PREDICT,
    MIXTURE_NATIVE_SCORE_SAMPLES, MIXTURE_NATIVE_PREDICT_PROBA, MIXTURE_NATIVE_PREDICT, MIXTURE_NATIVE_SCORE_BIC_AIC,
)
from bindings.boundary_read_door import pack_room
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.estimator_tabular_boundary_pyapi_unpack import _target, _refusal
from bindings.py_optional import py_or_none
from bindings.operation_names_mix import (
    MIX_OPN_FAMILY_MIX_GMM_FIT, MIX_OPN_FAMILY_MIX_GMM_FIT_PREDICT, MIX_OPN_FAMILY_MIX_GMM_SCORE_SAMPLES,
    MIX_OPN_FAMILY_MIX_GMM_PREDICT_PROBA, MIX_OPN_FAMILY_MIX_GMM_PREDICT, MIX_OPN_FAMILY_MIX_GMM_BIC,
    MIX_OPN_FAMILY_MIX_GMM_AIC, MIX_OPN_FAMILY_MIX_BGMM_FIT, MIX_OPN_FAMILY_MIX_BGMM_FIT_PREDICT,
    MIX_OPN_FAMILY_MIX_BGMM_PREDICT, MIX_OPN_FAMILY_MIX_BGMM_PREDICT_PROBA, MIX_OPN_FAMILY_MIX_BGMM_SCORE_SAMPLES,
    MIX_OPN_FAMILY_MIX_BGMM_SCORE, MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT,
    MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE,
    MIX_OPN_FAMILY_MIX_GMM_PUBLIC_INITIALIZE, MIX_OPN_FAMILY_MIX_GMM_PUBLIC_BINDING, MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SAMPLE,
)
from bindings.operation_names_svm import SVM_OPN_FAMILY_SVM_GMM_SAMPLE
from bindings.svm_keys_unpack import svm_keys
from providers.python.device_route_boundary_door import device_route_clear
from mixture.mixture_door_run import (
    bgmm_entry_floats, bgmm_entry_ints, gmm_fit_request, mixture_by_type_shape, gmm_information,
    gmm_public_fitting, gmm_public_query, gmm_public_which, gmm_public_method, gmm_public_plain,
)
from bindings.mixture_public_native_unpack import (
    _GMM_ADMIT_SLOT, _GMM_OPERANDS_SLOT, BgmmEntryFit,
    _get, _get_or, _x_finite, _check_mode, _gmm_needs_ext, _gmm_connection, _gmm_score_call, _gmm_ext_score,
    _bgmm_fit_call, _bgmm_score, _gmm_public_admit,
    gmm_default_refuse_options, gmm_fit_settings, gmm_fit_random_state, gmm_ext_fit_stage, bgmm_fit_stage,
)


# ------------------------------------------------------------------ helpers
def _pack_f(values: List[Float32], var shape: List[Int], context: PythonObject) raises -> PythonObject:
    """A result block of the BGMM entry as an owned C-order float32 Array:
    ONE allocation (`pack_room`) and one memcpy of the entry's words.
    Replaces `_from_flat` over the CPython `x_cluster_call` result, which
    copied each block twice (the exporter's array.array, then
    `array.array(code, flat)`)."""
    var room = pack_room("<f4", shape^, context)
    if len(values):
        unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=room.address), src=values.unsafe_ptr(), count=len(values))
    _ = values
    var result = room.array
    _ = room^
    return result


def _pack_i(values: List[Int32], var shape: List[Int], context: PythonObject) raises -> PythonObject:
    """`_pack_f` for an int32 result block."""
    var room = pack_room("<i4", shape^, context)
    if len(values):
        unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=room.address),
                      src=values.unsafe_ptr(), count=len(values))
    _ = values
    var result = room.array
    _ = room^
    return result


def _state(blocks: List[List[Float32]], order: List[Int], zeros_at: Int, zeros: Int,
           context: PythonObject) raises -> PythonObject:
    """The warm-start state: the entry's float blocks `order` end to end
    (with `zeros` zeros in place of block position `zeros_at` when that block
    is empty, and for every -1 in `order`, an inserted block) as one float32
    Array, the bytes the former `list(...) + ...` concatenation cast to
    float32. ONE allocation of the final size, one memcpy per block (it was
    an array.array grown block by block, then wrapped)."""
    var total = 0
    for j in range(len(order)):  # small-loop(blocks: the eight fitted parameter blocks)
        if order[j] < 0 or (j == zeros_at and len(blocks[order[j]]) == 0):
            total += zeros
        else:
            total += len(blocks[order[j]])
    var room = pack_room("<f4", [total], context)  # zero-filled
    var dst = room.address
    var offset = 0
    for j in range(len(order)):  # small-loop(blocks: the eight fitted parameter blocks)
        if order[j] < 0 or (j == zeros_at and len(blocks[order[j]]) == 0):
            offset += zeros  # pack_room is zero-filled
        else:
            var size = len(blocks[order[j]])
            if size:
                unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=dst + 4 * offset),
                              src=blocks[order[j]].unsafe_ptr(), count=size)
            offset += size
    _ = blocks
    var result = room.array
    _ = room^  # the Array stays alive through the copies above
    return result


def _by_type(flat: List[Float32], k: Int, d: Int, ct: Int, context: PythonObject) raises -> PythonObject:
    """`BayesianGaussianMixture._by_type`: the entry's by-type values (full
    k x d x d, tied d x d, diag k x d, spherical k) under the covariance
    type's shape."""
    return _pack_f(flat, mixture_by_type_shape(ct, k, d), context)


# ----------------------------------------------------- GaussianMixture, default
def _gmm_default_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                     context: PythonObject, fit_predict: Bool) raises -> PythonObject:
    """The default fit (`gmm_fit`); returns (the float32 X it read, labels).

    `fit_predict` and a binding that exports `gmm_fit_predict` (the GPU
    binding; row N401): the fit writes predict(X) of the fitted model into
    `labels` (an int32 (n,) Array, the array `predict` returns) from its own
    device buffers, in the same crossing; X is not uploaded again. Else
    `labels` is None and the caller predicts as before."""
    py_setattr(model, "_ext", PythonObject(None))
    gmm_default_refuse_options(model)
    var x = admit_family(source, 2, "X", context).array
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var settings = gmm_fit_settings(model)
    var k = settings.k
    var weights = family_array(Python.tuple(k), "<f4", context)
    var means = family_array(Python.tuple(k, d), "<f4", context)
    var covariances = family_array(Python.tuple(k, d, d), "<f4", context)
    var precisions = family_array(Python.tuple(k, d, d), "<f4", context)
    var log_det = family_array(Python.tuple(k), "<f4", context)
    # n_iter, converged, lower_bound: a Mojo buffer (was a Python <f8 Array).
    var scalars = List[Float64](length=3, fill=0.0)
    _check_mode(binding, model, context)
    var connection = _gmm_connection(binding)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var labels = Optional[PythonObject]()
    if fit_predict and (connection.driver.metadata(MIXTURE_NATIVE_CAPABILITIES) & MIXTURE_CAP_FIT_PREDICT) != 0:
        labels = family_array(Python.tuple(n), "<i4", context)
    # The two names cross as NUL-terminated UTF-8 and are decoded by the
    # binding's own covariance_type_from_name / init_params_from_name.
    var covariance_name = settings.covariance_name
    var init_name = settings.init_name
    var random_state = gmm_fit_random_state(model)
    var request = gmm_fit_request(
        Int(py=classical_address(x)), Int(py=classical_address(weights)), Int(py=classical_address(means)),
        Int(py=classical_address(covariances)), Int(py=classical_address(precisions)),
        Int(py=classical_address(log_det)), Int(scalars.unsafe_ptr()),
        Int(py=classical_address(labels.value())) if Bool(labels) else 0,
        Int(covariance_name.as_c_string_slice().unsafe_ptr()), Int(init_name.as_c_string_slice().unsafe_ptr()),
        n, d, k, settings.max_iter, random_state, settings.tol, settings.reg_covar)
    var n_iter = 0
    try:
        with GILReleased(Python()):
            n_iter = connection.driver.fit(request)
    finally:
        # Every address above stays owned through the native call.
        _ = connection.provider
        _ = x
        _ = weights
        _ = means
        _ = covariances
        _ = precisions
        _ = log_det
        _ = scalars
        _ = labels
        _ = covariance_name
        _ = init_name
    py_setattr(model, "n_features_in_", PythonObject(d))
    py_setattr(model, "weights_", weights)
    py_setattr(model, "means_", means)
    py_setattr(model, "covariances_", covariances)
    py_setattr(model, "precisions_cholesky_", precisions)
    py_setattr(model, "log_det_chol_", log_det)
    py_setattr(model, "n_iter_", PythonObject(n_iter))
    py_setattr(model, "converged_", PythonObject(Int(scalars[1]) != 0))
    py_setattr(model, "lower_bound_", PythonObject(scalars[2]))
    _ = scalars^
    return Python.tuple(x, py_or_none(labels))


# ------------------------------------------- GaussianMixture, option parity
def _gmm_ext_fit(model: PythonObject, xc_binding: PythonObject, source: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._gmm_ext_fit`: scikit-learn's GaussianMixture on
    the cluster lane's mixture driver (ENTRY_BGMM, plain mode); returns the
    float32 X it read."""
    var stage = gmm_ext_fit_stage(model, source, context)
    var result = _bgmm_entry_fit(xc_binding, stage)
    var nk = svm_keys()
    var x = stage.x
    var key = stage.key
    var d = stage.d
    var k = stage.k
    var ct = stage.ct
    ref f = result.storage[0].f
    ref s = result.storage[0].s
    py_setattr(model, "n_features_in_", PythonObject(d))
    py_setattr(model, "weights_", _pack_f(f[0], [k], context))
    py_setattr(model, "means_", _pack_f(f[1], [k, d], context))
    py_setattr(model, "covariances_", _by_type(f[13], k, d, ct, context))
    py_setattr(model, "precisions_cholesky_", _by_type(f[14], k, d, ct, context))
    py_setattr(model, "log_det_chol_", _pack_f(f[12], [k], context))   # x_cluster bgmm_entry
    # int(), float() and bool() of the entry's float64 scalars, natively.
    py_setattr(model, "n_iter_", PythonObject(Int(s[1])))
    py_setattr(model, "converged_", PythonObject(s[2] != 0.0))
    py_setattr(model, "lower_bound_", PythonObject(s[0]))
    # The fit's own state: the full per-component Cholesky block and log
    # constants, and the state a warm_start refit resumes from. The x_cluster
    # image is not state: a query takes it from the estimator's binding route
    # (its device and mode), so a fitted model handed to an estimator on the
    # other device is scored by that estimator's own image, and the state
    # holds arrays and scalars only (round 2, lane dw-neighbors-cluster: it
    # stored the module, validated it against a retained-module list on every
    # admission and kept that list alive in the import context).
    var ext = Python.dict()
    ext[nk[].k_key] = key
    ext[nk[].k_pchol] = _pack_f(f[3], [k, d, d], context)
    ext[nk[].k_consts] = _pack_f(f[8], [k], context)
    var order: List[Int] = [11, 4, -1, 6, 1, 7, 2, 3]
    ext[nk[].k_state] = _state(f, order, -1, k, context)
    _ = result^
    _ = stage^
    py_setattr(model, "_ext", ext)
    return x


def _gmm_bic_aic(model: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject, admitted: Bool,
                 key: PythonObject) raises -> List[Float64]:
    """`GaussianMixture._score_bic_aic`: (score, bic, aic). `admitted`:
    `source` is the X the public door admitted (classical_check_x: 2-D,
    non-empty, finite, float32 in C order), used as is (lane perf2-cluster:
    X was admitted twice per query). `key`: the public query's record key."""
    if _get_or(model, "_ext", PythonObject(None)) is not None:
        # `_gmm_ext_bic_aic`: scikit-learn's `_n_parameters`, `bic` and
        # `aic`; the mean log-likelihood is the binding's float-float fold
        # over the rows divided by n.
        var x = source if admitted else _x_finite(source, "X", context)
        var n = Int(py=x.shape[0])
        var got = _gmm_ext_score(model, x, context)
        var score = got.storage[0].s[0] / Float64(n)
        _ = got^
        var means = _get(model, "means_")
        var k = Int(py=means.shape[0])
        var d = Int(py=means.shape[1])
        var nk = svm_keys()
        var ct = Int(py=py_item(context, nk[].k_family_mix_ctypes)[_get(model, "covariance_type")])
        _ = x^
        return gmm_information(score, n, k, d, ct)
    var x = source if admitted else admit_family(source, 2, "X", context).array
    # The entry's three float64 words land in a Mojo buffer (it was a Python
    # `<f8` array allocated per call and read back word by word).
    var result = List[Float64](length=3, fill=0.0)
    _gmm_score_call(model, binding, x, MIXTURE_NATIVE_SCORE_BIC_AIC, "score", Int(result.unsafe_ptr()), context,
                    key)
    _ = x^
    return result^


def _gmm_infer(model: PythonObject, binding: PythonObject, source: PythonObject, which: Int,
               context: PythonObject, admitted: Bool,
               key: PythonObject) raises -> PythonObject:
    """score_samples (0), predict_proba (1) or predict (2). `admitted` and
    `key` as in `_gmm_bic_aic`."""
    var ext = _get_or(model, "_ext", PythonObject(None))
    if ext is not None:
        var xe = source if admitted else _x_finite(source, "X", context)
        return _gmm_ext_labels(model, xe, which, context)
    var x = source if admitted else admit_family(source, 2, "X", context).array
    return _gmm_default_infer(model, binding, x, which, context, key)


def _gmm_ext_labels(model: PythonObject, x: PythonObject, which: Int, context: PythonObject) raises -> PythonObject:
    var n = Int(py=x.shape[0])
    var got = _gmm_ext_score(model, x, context)
    var k = Int(py=_get(model, "means_").shape[0])
    if which == 0:
        return _pack_f(got.storage[0].f[1], [n], context)
    if which == 1:
        return _pack_f(got.storage[0].f[2], [n, k], context)
    return _pack_i(got.storage[0].i[0], [n], context)


def _gmm_default_infer(model: PythonObject, binding: PythonObject, x: PythonObject, which: Int,
                       context: PythonObject, key: PythonObject) raises -> PythonObject:
    var n = Int(py=x.shape[0])
    if which == 0:
        var out = family_array(Python.tuple(n), "<f4", context)
        _gmm_score_call(model, binding, x, MIXTURE_NATIVE_SCORE_SAMPLES, "score_samples", Int(py=classical_address(out)), context, key)
        _ = out  # the output stays owned through the native call above
        return out
    if which == 1:
        var k = 0
        var held = cluster_record_get(_GMM_OPERANDS_SLOT, model, key)
        if held is not None and held[0] is binding:
            k = Int(py=held[7])
        elif family_raw_has(model, "weights_"):
            k = Int(py=_get(model, "weights_").shape[0])
        var out = family_array(Python.tuple(n, k), "<f4", context)
        _gmm_score_call(model, binding, x, MIXTURE_NATIVE_PREDICT_PROBA, "predict_proba", Int(py=classical_address(out)), context, key)
        _ = out  # the output stays owned through the native call above
        return out
    var out = family_array(Python.tuple(n), "<i4", context)
    _gmm_score_call(model, binding, x, MIXTURE_NATIVE_PREDICT, "predict", Int(py=classical_address(out)), context, key)
    _ = out  # the output stays owned through the native call above
    return out


def _gmm_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GaussianMixture.fit`: args (binding, x_cluster binding, X); returns
    the float32 X it read."""
    if _gmm_needs_ext(model):
        return _gmm_ext_fit(model, args[1], args[2], context)
    return _gmm_default_fit(model, args[0], args[2], context, False)[0]


# ------------------------------------------------- BayesianGaussianMixture
def _bgmm_entry_fit(binding: PythonObject, stage: BgmmEntryFit) raises -> BgmmResultSlot:
    """ENTRY_BGMM for one unpacked fit: RUN lays out the entry's words, the
    typed provider runs them (GIL released inside `_bgmm_fit_call`)."""
    var floats = bgmm_entry_floats(stage.prior_weight, stage.prior_mean_precision, stage.prior_freedom,
                                   stage.reg_covar, stage.tol, stage.lower)
    var ints = bgmm_entry_ints(stage.n, stage.d, stage.k, stage.process, stage.max_iter, stage.n_init,
                               stage.init_code, stage.seed, stage.mean_prior, stage.cov_mode, stage.ct,
                               stage.warm, stage.plain, stage.flag_w, stage.flag_m, stage.flag_p)
    return _bgmm_fit_call(binding, stage.x, stage.aux, ints^, floats^)


def _bgmm_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
              context: PythonObject) raises -> PythonObject:
    """`BayesianGaussianMixture._fit`: ENTRY_BGMM; returns the labels."""
    var stage = bgmm_fit_stage(model, source, context)
    var result = _bgmm_entry_fit(binding, stage)
    var key = stage.key
    var n = stage.n
    var d = stage.d
    var k = stage.k
    var ct = stage.ct
    var process = stage.process
    ref f = result.storage[0].f
    ref s = result.storage[0].s
    py_setattr(model, "weights_", _pack_f(f[0], [k], context))
    py_setattr(model, "means_", _pack_f(f[1], [k, d], context))
    py_setattr(model, "_full_pchol", _pack_f(f[3], [k, d, d], context))
    # f[13], f[14], f[15]: the binding's by-type forms (bgmm_entry)
    py_setattr(model, "covariances_", _by_type(f[13], k, d, ct, context))
    py_setattr(model, "precisions_cholesky_", _by_type(f[14], k, d, ct, context))
    if process:
        py_setattr(model, "weight_concentration_",
                   Python.tuple(_pack_f(f[4], [k], context), _pack_f(f[5], [k], context)))
    else:
        py_setattr(model, "weight_concentration_", _pack_f(f[4], [k], context))
    py_setattr(model, "mean_precision_", _pack_f(f[6], [k], context))
    if ct == 1:
        py_setattr(model, "degrees_of_freedom_", PythonObject(Float64(f[7][0])))
    else:
        py_setattr(model, "degrees_of_freedom_", _pack_f(f[7], [k], context))
    py_setattr(model, "_log_consts", _pack_f(f[8], [k], context))
    py_setattr(model, "mean_prior_", _pack_f(f[9], [d], context))
    if ct == 2:
        py_setattr(model, "covariance_prior_", _pack_f(f[15], [d], context))
    elif ct == 3:
        py_setattr(model, "covariance_prior_", PythonObject(Float64(f[15][0])))
    else:
        py_setattr(model, "covariance_prior_", _pack_f(f[15], [d, d], context))
    # int(), float() and bool() of the entry's float64 scalars, natively.
    py_setattr(model, "lower_bound_", PythonObject(s[0]))
    py_setattr(model, "n_iter_", PythonObject(Int(s[1])))
    py_setattr(model, "converged_", PythonObject(s[2] != 0.0))
    # the state a warm_start refit resumes from (full per-component
    # matrices); an empty f[5] (Dirichlet distribution) is k zeros
    var order: List[Int] = [11, 4, 5, 6, 1, 7, 2, 3]
    py_setattr(model, "_warm", Python.tuple(key, _state(f, order, 2, k, context)))
    py_setattr(model, "weight_concentration_prior_", PythonObject(s[3]))
    py_setattr(model, "mean_precision_prior_", PythonObject(s[4]))
    py_setattr(model, "degrees_of_freedom_prior_", PythonObject(s[5]))
    py_setattr(model, "n_features_in_", PythonObject(d))
    var labels = _pack_i(result.storage[0].i[0], [n], context)
    _ = result^
    _ = stage^
    return labels


# ------------------------------------------------------------------- door
def mixture_operation(model: PythonObject, operation: Int, args: PythonObject,
         context: PythonObject, admitted: Bool = False) raises -> PythonObject:
    """args[0] is the resolved binding; GaussianMixture fit and fit_predict
    also take the `_mojolearn_x_cluster` binding as args[1]; X is last.
    `admitted`: the public cluster door (providers/python/
    cluster_public_lifecycle_pyapi.mojo) admitted args[1] with
    classical_check_x; the BayesianGaussianMixture queries use it as is.
    `operation` is the leaf code of bindings/operation_names_mix.mojo
    (lane bw2-svm; the public block is last: one range test)."""
    classical_connect_numpy(context)
    if operation >= MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT:
        return _gmm_public_operation(model, operation, args, context)
    if operation == MIX_OPN_FAMILY_MIX_GMM_FIT:
        _ = _gmm_fit(model, args, context)
        return model
    if operation == MIX_OPN_FAMILY_MIX_GMM_FIT_PREDICT:
        # `fit(X).predict(X)` on the float32 X the fit read; on the GPU
        # binding the fit returns those labels itself (N401).
        if not _gmm_needs_ext(model):
            var held = _gmm_default_fit(model, args[0], args[2], context, True)
            if held[1] is not None:
                return held[1]
            return _gmm_default_infer(model, args[0], held[0], 2, context, PythonObject(None))
        var x = _gmm_fit(model, args, context)
        if _get_or(model, "_ext", PythonObject(None)) is not None:
            return _gmm_ext_labels(model, x, 2, context)
        return _gmm_default_infer(model, args[0], x, 2, context, PythonObject(None))
    if operation == MIX_OPN_FAMILY_MIX_GMM_SCORE_SAMPLES:
        return _gmm_infer(model, args[0], args[1], 0, context, False, PythonObject(None))
    if operation == MIX_OPN_FAMILY_MIX_GMM_PREDICT_PROBA:
        return _gmm_infer(model, args[0], args[1], 1, context, False, PythonObject(None))
    if operation == MIX_OPN_FAMILY_MIX_GMM_PREDICT:
        return _gmm_infer(model, args[0], args[1], 2, context, False, PythonObject(None))
    if operation == MIX_OPN_FAMILY_MIX_GMM_BIC:
        return PythonObject(_gmm_bic_aic(model, args[0], args[1], context, False, PythonObject(None))[1])
    if operation == MIX_OPN_FAMILY_MIX_GMM_AIC:
        return PythonObject(_gmm_bic_aic(model, args[0], args[1], context, False, PythonObject(None))[2])
    if operation == MIX_OPN_FAMILY_MIX_BGMM_FIT:
        _ = _bgmm_fit(model, args[0], args[1], context)
        return model
    if operation == MIX_OPN_FAMILY_MIX_BGMM_FIT_PREDICT:
        return _bgmm_fit(model, args[0], args[1], context)
    if operation == MIX_OPN_FAMILY_MIX_BGMM_PREDICT:
        # the labels come from the device score (`bodies.argmax_row`: the
        # first largest log responsibility, the lowest index on a tie)
        var n = 0
        var k = 0
        var got = _bgmm_score(model, args[0], args[1], 0, context, admitted, n, k)
        return _pack_i(got.storage[0].i[0], [n], context)
    if operation == MIX_OPN_FAMILY_MIX_BGMM_PREDICT_PROBA:
        # the exp of every log responsibility in the binding (bodies.exp_cell)
        var n = 0
        var k = 0
        var got = _bgmm_score(model, args[0], args[1], 1, context, admitted, n, k)
        return _pack_f(got.storage[0].f[2], [n, k], context)
    if operation == MIX_OPN_FAMILY_MIX_BGMM_SCORE_SAMPLES:
        var n = 0
        var k = 0
        var got = _bgmm_score(model, args[0], args[1], 0, context, admitted, n, k)
        return _pack_f(got.storage[0].f[1], [n], context)
    if operation == MIX_OPN_FAMILY_MIX_BGMM_SCORE:
        # the float-float fold of log_prob_norm in the binding
        var n = 0
        var k = 0
        var got = _bgmm_score(model, args[0], args[1], 2, context, admitted, n, k)
        return PythonObject(got.storage[0].s[0] / Float64(n))
    raise Error("RuntimeError: unknown family operation " + String(operation))


# Public GaussianMixture lifecycle. The old Python _mode wrappers selected
# bindings, validated X/y, tracked fitted state and converted returned Arrays.
# This envelope preserves those responsibilities inside the native operation.
def _gmm_public_initialize(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = svm_keys()
    var names = py_item(context, nk[].k_gmm_public_parameters)
    var b = native_loaded_module("builtins")
    if py_type(names) is not b.tuple or py_type(values) is not b.tuple:
        raise Error("TypeError: GaussianMixture requires its connected constructor tuples")
    var fields = py_fields(model)
    # `device` is the constructor's last value. It is not one of the
    # parameters every operation admits again (`gmm_public_parameters`): an
    # instance without the field runs this installation's device. Its check
    # (unknown value, 'gpu' without a GPU set, 'cpu' for a class that has no
    # routed CPU form) is the shared constructor door's: the public
    # constructor of this class is the full-reset native constructor
    # (bindings/full_reset_constructor_native_door.mojo), which admits the
    # device before any field changes and drops the estimator's route.
    var count = len(names)
    if len(values) != count + 1:
        raise Error("TypeError: GaussianMixture constructor metadata differs")
    var device = values[count]
    var replacements = Python.list()
    for i in range(len(names)):  # driver-loop: fixed public constructor parameter schema
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: GaussianMixture parameter names require exact strings")
        var value = values[i]
        if String(names[i]) == "numeric_mode" and value is None:
            value = _get_or(model, "numeric_mode", PythonObject(None))
        if py_contains(fields, names[i]) and fields[names[i]] is not value:
            native_release_admit(fields[names[i]], context)
        _ = py_call(replacements.append, value)
    for i in range(len(names)):  # driver-loop: publish only after every replaced owner passed admission
        py_setitem(fields, names[i], replacements[i])
    py_setitem(fields, PythonObject("device"), device)
    device_route_clear(model)
    return PythonObject(None)


def _gmm_public_output(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = svm_keys()
    if py_get(context, nk[].k_ndarray) is None:
        return value
    if py_type(value) is py_item(context, nk[].k_array_class):
        return native_array_ndarray_view(value, context)
    if py_type(value) is py_type(Python.tuple()):
        var out = Python.list()
        for i in range(len(value)):  # driver-loop: returned sample arrays, not their numerical elements
            _ = py_call(out.append, _gmm_public_output(value[i], context))
        return py_call(native_loaded_module("builtins").tuple, out)
    return value


def _gmm_public_operation(model: PythonObject, operation: Int, args: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    if py_type(args) is not py_type(Python.tuple()):
        raise Error("TypeError: GaussianMixture public arguments require a native tuple")
    # Passive subclasses of the connected owner are accepted: CPython's MRO
    # test (bindings/cluster_side_records.mojo), not a Python-level walk.
    var nk = svm_keys()
    if not cluster_is_subclass(py_type(model), py_item(context, nk[].k_gmm_public_class)):
        raise Error("TypeError: GaussianMixture requires its connected native class")
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_INITIALIZE:
        return _gmm_public_initialize(model, args, context)
    var fitting = gmm_public_fitting(operation)
    # The queries that read the fitted model and an admitted X.
    var query = gmm_public_query(operation)
    var admitted = _gmm_public_admit(model, context, fitting, query)
    var fields = admitted[0]
    var key = admitted[1]
    var recorded = Bool(py=admitted[2])
    if fitting:
        # A fit replaces the fitted state: release the old records now.
        cluster_record_drop(_GMM_ADMIT_SLOT, model)
        cluster_record_drop(_GMM_OPERANDS_SLOT, model)
    # The estimator's binding route (the shared estimator door, slot
    # `device_bindings_v1`): one record read and pointer compares on a hit.
    # Removed here: the per-call `estimator_binding(model, "_mojolearn_mixture", ...)`
    # and `estimator_binding(model, "_mojolearn_x_cluster", ...)` of this door
    # (mode String, offerings and connections reads, the process `_CPU_ONLY`
    # question, the live registry read) run once per estimator, at resolution.
    var binding = estimator_binding(model, "_mojolearn_mixture", context, False)
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_BINDING:
        _check_mode(binding, model, context)
        return binding
    var outside = estimator_outside()
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SAMPLE:
        host_inference_scalar(args[0], context)
        var result = svm_mixture_operation(model, SVM_OPN_FAMILY_SVM_GMM_SAMPLE, Python.tuple(binding, args[0]), context)
        return _gmm_public_output(result, context) if outside else result
    var guarded = fitting or query
    if guarded and not fitting:
        classical_require_fitted(model, False, gmm_public_method(operation), context)
    var source = args[0]
    var name = classical_estimator_name(model)
    var x = classical_check_x(source, name, context) if guarded else source
    var rows = Int(py=x.shape[0]) if guarded else -1
    var columns = Int(py=x.shape[1]) if guarded else -1
    if fitting:
        _target(args[1], rows, name, context)
    elif guarded:
        var fitted_width = py_get(fields, nk[].k_n_features_in_)
        if fitted_width is not None:
            classical_check_width(columns, Int(py=fitted_width), name)
        estimator_feature_check(model, fields, source, context)
    var prepared = PythonObject(None)
    if fitting:
        var cluster_binding = estimator_binding(model, "_mojolearn_x_cluster", context)
        # The admission probe names the entry the fit calls (the typed BGMM
        # transport; the door no longer calls x_cluster_call).
        _ = neural_descriptor_entry(cluster_binding, "bgmm_native_api_v1")
        prepared = Python.tuple(binding, cluster_binding, x)
    else:
        prepared = Python.tuple(binding, x)
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    try:
        if query:
            result = _gmm_public_query(model, operation, binding, x, context, key)
        else:
            result = mixture_operation(model, gmm_public_plain(operation), prepared, context)
    except problem:
        if guarded:
            var refused = _refusal(problem, fitting, depth > 0, rows, columns, name)
            raise refused^
        raise problem
    finally:
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        estimator_fit_complete(model, source, False, context)
    if query and not recorded:
        # The full admission and the query passed for this key.
        cluster_record_put(_GMM_ADMIT_SLOT, model, key, PythonObject(True))
    return _gmm_public_output(result, context) if outside else result


def _gmm_public_query(model: PythonObject, operation: Int, binding: PythonObject, x: PythonObject,
                      context: PythonObject, key: PythonObject) raises -> PythonObject:
    """predict, predict_proba, score_samples and score on the X the public
    door admitted (classical_check_x: 2-D, non-empty, finite, float32 in C
    order): `mixture_operation`'s routes for those operations, without
    admitting X a second time (lane perf2-cluster)."""
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE:
        return PythonObject(_gmm_bic_aic(model, binding, x, context, True, key)[0])
    return _gmm_infer(model, binding, x, gmm_public_which(operation), context, True, key)
