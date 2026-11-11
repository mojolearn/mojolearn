# SPDX-License-Identifier: Apache-2.0
# TSNE's public methods, one native call each (python/mojolearn/_expansion_ann.py:
# `fit`, `fit_transform`). The shell hands over input/settings; this native
# owner selects x_ann and estimators bindings from one captured numeric mode;
# this owner makes the Python method's refusals (same wording, same order),
# builds the 'pca' start with the call `PCA(n_components=k).fit_transform`
# made (`pca_fit_owned`, the same arguments), runs `x_ann_tsne_fit` with the
# argument lists it always took and stores the fitted attributes. The same
# entries on the same operands: the bits do not move.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_setitem
from bindings.classical_arrays import classical_address, classical_word
from bindings.classical_attributes import classical_connect_numpy
from providers.python.estimator_binding_pyapi import estimator_binding_mode
from bindings.manifold_settings_native import ManifoldSettings
from bindings.manifold_scalar_native import manifold_number
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from bindings.classical_validate import classical_view
from bindings.manifold_lifecycle_native import _cast
from bindings.spectral_umap_routes import spectral_umap_entry as family_entry, spectral_umap_optional_entry
from x_ann.mani_tsne_run import (
    TSNE_INIT_ARRAY, tsne_check_samples, tsne_perplexity_positive, tsne_perplexity_refusal,
    tsne_perplexity_below_samples, tsne_perplexity_samples_refusal, tsne_auto_learning_rate,
    tsne_post_factor_valid, tsne_post_factor_refusal, tsne_exploration_iterations, tsne_init_mode,
    tsne_check_pca_components, tsne_init_shape_matches, tsne_init_shape_refusal, tsne_check_input_status,
    tsne_fit_tests_input,
)
from bindings.family_common import (
    family_repr, family_str, family_is_str, family_is_int, family_type_name, family_call, family_array,
    family_field,
)


def _tsne_int(name: String, value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ann_int("TSNE", name, value)`."""
    if not family_is_int(value):
        raise Error("TypeError: mojolearn TSNE: " + name + " must be an int, got " + family_type_name(value))
    return manifold_number(value, True, context)


def _tsne_pca_start(estimators: PythonObject, x_address: Int, n: Int, d: Int, k: Int,
                    context: PythonObject) raises -> PythonObject:
    """`PCA(n_components=k).fit_transform(x)` (svd_solver='auto', the
    covariance arm, no whitening) as the float32 start: `PCA._fit(x, True)`'s
    refusal and its one `pca_fit_owned` call."""
    tsne_check_pca_components(k, d)
    var fit_owned = spectral_umap_optional_entry(estimators, "pca_fit_owned", context)
    if fit_owned is None:
        raise Error("NotImplementedError: mojolearn PCA: this build of _mojolearn_estimators does not "
                    + "export pca_fit_owned; rebuild it with bindings/build_estimators.sh")
    var components = family_array(Python.tuple(k, d), "<f4", context)
    var mean = family_array(Python.tuple(d), "<f4", context)
    var variance = family_array(Python.tuple(k), "<f4", context)
    var ratio = family_array(Python.tuple(k), "<f4", context)
    var singular = family_array(Python.tuple(k), "<f4", context)
    var z = family_array(Python.tuple(n, k), "<f4", context)
    # ORDER MATCHES PCA._fit: x, components, mean, variance, ratio,
    # singular, out; n, d, nc, dense 0, keep_mode 0, whiten 0, frac 0.0.
    # Lane a8d: the trailing 1 asks pca_fit_owned for X's finite status word
    # over its own upload (no host scan in the door); a flagged X is refused
    # here, before the eigensolver's result is read.
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(k), PythonObject(0),
                             PythonObject(0), PythonObject(0), PythonObject(Float64(0.0)), PythonObject(1))
    var fitted = family_call(fit_owned, PythonObject(x_address), classical_address(components), classical_address(mean),
                    classical_address(variance), classical_address(ratio), classical_address(singular),
                    classical_address(z), params)
    _ = mean  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = singular  # keep the Python object alive through the native call above
    _ = ratio  # keep the Python object alive through the native call above
    _ = variance  # keep the Python object alive through the native call above
    _ = z  # keep the Python object alive through the native call above
    tsne_check_input_status(Int(py=fitted[2]))
    _ = components^
    _ = mean^
    _ = variance^
    _ = ratio^
    _ = singular^
    # the kept count is every fitted component (keep_mode 0); the start is
    # read from the projection's first n * keep words either way
    return z


def _tsne_fit(model: PythonObject, settings: ManifoldSettings, args: PythonObject, snapshot: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    """`TSNE.fit(X)`; returns the embedding."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var op = native_loaded_module("operator")
    var binding = estimator_binding_mode(settings.mode, "_mojolearn_x_ann", context) if args[0] is None else args[0]
    var estimators = estimator_binding_mode(settings.mode, "_mojolearn_estimators", context) if args[1] is None else args[1]
    var fields = settings.values
    # TSNE's public input refusal for every init mode, tested on the device
    # (lane a8d): the PCA start's pca_fit_owned returns X's finite status
    # word over its upload, before its eigensolver's result is used; the
    # other starts ask x_ann_tsne_fit for it (params[12]). No host scan.
    # TOMBSTONE: mani_tsne_native.mojo door host scan of X (host_all_finite_f32) deleted 2026-11-09 by lane a8d-device-x-decomp-manifold; code recoverable at 0ce88411d3.
    # Restore: experiments/removed/A8D_DECOMP_MANIFOLD_HOST_FINITE.patch.
    var x = snapshot.keep
    var x_address = snapshot.source.addr
    var n = snapshot.source.rows
    var d = snapshot.source.cols
    tsne_check_samples(n, d)
    var components = family_field(fields, "n_components")
    if (_tsne_is_bool(components)
            or not Bool(py=py_call(op.contains, Python.tuple(PythonObject(1), PythonObject(2)), components))):
        raise Error("mojolearn TSNE: n_components must be 1 or 2, got " + family_repr(components))
    var k = Int(py=manifold_number(components, True, context))
    var max_iter = _tsne_int("max_iter", family_field(fields, "max_iter"), context)
    var seed = _tsne_int("random_state", family_field(fields, "random_state"), context)
    var perplexity_param = family_field(fields, "perplexity")
    var perplexity = Float64(py=manifold_number(perplexity_param, False, context))
    # scikit-learn's rules: perplexity > 0 (its parameter constraint) and
    # perplexity < n_samples (`_check_params_vs_input`), in its words
    if not tsne_perplexity_positive(perplexity):
        raise tsne_perplexity_refusal(family_repr(perplexity_param))
    if not tsne_perplexity_below_samples(perplexity, n):
        raise tsne_perplexity_samples_refusal(family_str(perplexity_param), n)
    var exag = Float64(py=manifold_number(family_field(fields, "early_exaggeration"), False, context))
    var rate = family_field(fields, "learning_rate")
    var lr: Float64
    if Bool(py=py_call(b.bool, py_call(op.eq, rate, PythonObject("auto")))):
        lr = tsne_auto_learning_rate(n, exag)
    else:
        lr = Float64(py=manifold_number(rate, False, context))
    var post_param = family_field(fields, "post_exaggeration_learning_rate_factor")
    var post_factor = Float64(py=manifold_number(post_param, False, context))
    if not tsne_post_factor_valid(post_factor):
        raise tsne_post_factor_refusal(family_repr(post_param))
    var exploration = tsne_exploration_iterations(Int(py=max_iter))
    var init = family_field(fields, "init")
    var y0: PythonObject
    var y0_address = 0
    var initial_snapshot = Optional[ArrayMetadata]()
    var mode = 0
    var low = PythonObject(0)
    var high = PythonObject(0)
    if family_is_str(init):
        mode = tsne_init_mode(String(py=init))
    if mode != TSNE_INIT_ARRAY:
        # the binding builds the start (x_ann/tsne_init.mojo): mode 1
        # random from random_state, mode 2 the PCA embedding scaled
        if tsne_fit_tests_input(mode):
            y0 = family_array(Python.tuple(2), "<f4", context)
        else:
            y0 = _tsne_pca_start(estimators, x_address, n, d, k, context)
        var s64 = py_call(op.and_, seed, py_call(op.sub, py_call(op.lshift, PythonObject(1), PythonObject(64)),
                                                 PythonObject(1)))
        low = py_call(op.and_, s64, PythonObject(0xFFFFFFFF))
        high = py_call(op.rshift, s64, PythonObject(32))
        y0_address = Int(py=classical_address(y0))
    else:
        # the caller's start array (n, k) float32 (`TSNE._init`)
        if family_is_str(init):
            raise Error("mojolearn TSNE: init must be 'pca', 'random' or an array, got " + family_repr(init))
        var view = classical_view(init, context, "init")
        var shape = array_shape_once(view, context)
        var captured = array_metadata_once(view, shape^, "init", context,
                                           allow_non_native=True, allow_unsupported=True)
        if not tsne_init_shape_matches(captured.shape, n, k):
            raise tsne_init_shape_refusal(family_repr(captured.python_shape()), n, k)
        var prepared = _cast(captured^, "init", context)
        y0 = prepared.keep
        y0_address = prepared.source.addr
        initial_snapshot = Optional(prepared^)
        _ = view
    var embedding = family_array(Python.tuple(n, k), "<f4", context)
    var kl = family_array(Python.tuple(1), "<f4", context)
    var fit = family_entry(binding, "x_ann_tsne_fit", context)
    # x, y0, y_out, kl_out; n, d, max_iter, exploration_iters, perplexity,
    # early_exaggeration, learning_rate, post_exaggeration_learning_rate_factor,
    # init mode, random_state low, high, n_components
    var tested = family_call(fit, Python.list(PythonObject(x_address), PythonObject(y0_address), classical_address(embedding),
                                     classical_address(kl)),
                    Python.list(PythonObject(n), PythonObject(d), max_iter, PythonObject(exploration),
                                PythonObject(perplexity), PythonObject(exag), PythonObject(lr),
                                PythonObject(post_factor), PythonObject(mode), low, high, PythonObject(k),
                                PythonObject(1 if tsne_fit_tests_input(mode) else 0)))
    _ = embedding  # keep the Python object alive through the native call above
    if tsne_fit_tests_input(mode):
        tsne_check_input_status(Int(py=tested[1]))
    _ = x  # keep the Python object alive through the native call above
    _ = kl  # keep the Python object alive through the native call above
    _ = snapshot
    _ = initial_snapshot
    _ = x^
    _ = y0^
    py_setitem(settings.fields, "embedding_", embedding)
    py_setitem(settings.fields, "kl_divergence_", PythonObject(classical_word(kl)))
    py_setitem(settings.fields, "n_iter_", max_iter)
    py_setitem(settings.fields, "learning_rate_", PythonObject(lr))
    py_setitem(settings.fields, "n_features_in_", PythonObject(d))
    return embedding


def _tsne_is_bool(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(native_loaded_module("builtins").isinstance, value,
                           native_loaded_module("builtins").bool))


def tsne_operation(model: PythonObject, settings: ManifoldSettings, operation: String, args: PythonObject,
                   snapshot: ArrayMetadata, context: PythonObject) raises -> PythonObject:
    """TSNE: args = (x_ann binding, estimators binding, X)."""
    if operation == "family_mani_tsne_fit":
        _ = _tsne_fit(model, settings, args, snapshot, context)
        return model
    if operation == "family_mani_tsne_fit_transform":
        return _tsne_fit(model, settings, args, snapshot, context)
    raise Error("RuntimeError: unknown TSNE operation " + operation)
