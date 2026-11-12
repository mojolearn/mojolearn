# SPDX-License-Identifier: Apache-2.0
# TSNE's public methods, one native call each (python/mojolearn/_expansion_ann.py:
# `fit`, `fit_transform`). The shell hands over input/settings; this native
# owner selects x_ann and estimators bindings from one captured numeric mode;
# this owner makes the Python method's refusals (same wording, same order),
# builds the 'pca' start with the call `PCA(n_components=k).fit_transform`
# made (`pca_fit_owned`, the same arguments), runs `x_ann_tsne_fit` with the
# words it always took and stores the fitted attributes. The same bodies on
# the same operands: the bits do not move. Both are native table calls (lane
# calltable-cn-ann): the PCA owned-fit table (core/decomp_entry_table_pca.mojo)
# and the ANN operations table (core/neighbors_entry_table_ann.mojo), each
# image's key read once from its module attribute at UNPACK, the call made
# with the GIL released and no CPython in between.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_ops_key_ann import ann_ops_key, decomp_pca_ops_key
from core.neighbors_entry_table_ann import ann_ops_entry, ann_tsne_fit_call
from core.decomp_entry_table_pca import pca_ops_entry, pca_fit_owned_call
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_setitem
from bindings.classical_arrays import classical_address, classical_word
from bindings.classical_attributes import classical_connect_numpy
from providers.python.estimator_binding_pyapi import estimator_binding_mode
from bindings.manifold_settings_native import ManifoldSettings
from bindings.manifold_scalar_native import manifold_number
from providers.python.array_metadata_pyapi import ArrayMetadata, ArrayPin, array_shape_once, array_metadata_once, array_python_shape
from bindings.classical_validate import classical_view
from bindings.manifold_lifecycle_native import _cast
from x_ann.mani_tsne_run import (
    TSNE_INIT_ARRAY, tsne_check_samples, tsne_perplexity_positive, tsne_perplexity_refusal,
    tsne_perplexity_below_samples, tsne_perplexity_samples_refusal, tsne_auto_learning_rate,
    tsne_post_factor_valid, tsne_post_factor_refusal, tsne_exploration_iterations, tsne_init_mode,
    tsne_check_pca_components, tsne_init_shape_matches, tsne_init_shape_refusal, tsne_check_input_status,
    tsne_fit_tests_input,
)
from bindings.family_common import (
    family_repr, family_str, family_is_str, family_is_int, family_type_name, family_array,
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
    var entry = pca_ops_entry(decomp_pca_ops_key(estimators))
    var components = family_array(Python.tuple(k, d), "<f4", context)
    var mean = family_array(Python.tuple(d), "<f4", context)
    var variance = family_array(Python.tuple(k), "<f4", context)
    var ratio = family_array(Python.tuple(k), "<f4", context)
    var singular = family_array(Python.tuple(k), "<f4", context)
    var z = family_array(Python.tuple(n, k), "<f4", context)
    # ORDER MATCHES PCA._fit: x, components, mean, variance, ratio,
    # singular, out; n, d, nc, dense 0, keep_mode 0, whiten 0, frac 0.0.
    # Lane a8d: the status word's address (the export's trailing finite flag
    # 1) asks the fit for X's finite status word over its own upload (no
    # host scan in the door); a flagged X is refused here, before the
    # eigensolver's result is read.
    var noise = List[Float64](length=1, fill=Float64(0.0))
    var status = List[Int32](length=1, fill=Int32(0))
    var components_address = Int(py=classical_address(components))
    var mean_address = Int(py=classical_address(mean))
    var variance_address = Int(py=classical_address(variance))
    var ratio_address = Int(py=classical_address(ratio))
    var singular_address = Int(py=classical_address(singular))
    var z_address = Int(py=classical_address(z))
    var noise_address = Int(noise.unsafe_ptr())
    var status_address = Int(status.unsafe_ptr())
    with GILReleased(Python()):
        _ = pca_fit_owned_call(entry, x_address, components_address, mean_address, variance_address,
                               ratio_address, singular_address, z_address, noise_address, status_address,
                               n, d, k, False, 0, False, Float64(0.0))
    _ = mean  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = singular  # keep the Python object alive through the native call above
    _ = ratio  # keep the Python object alive through the native call above
    _ = variance  # keep the Python object alive through the native call above
    _ = z  # keep the Python object alive through the native call above
    var x_status = Int(status[0])
    _ = status^  # written through its address; keep it alive past the read
    _ = noise^
    tsne_check_input_status(x_status)
    _ = components^
    _ = mean^
    _ = variance^
    _ = ratio^
    _ = singular^
    # the kept count is every fitted component (keep_mode 0); the start is
    # read from the projection's first n * keep words either way
    return z


def _tsne_fit(model: PythonObject, settings: ManifoldSettings, args: PythonObject, snapshot: ArrayMetadata,
              pin: ArrayPin, context: PythonObject) raises -> PythonObject:
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
    var x = pin.keep
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
        var refusal = tsne_perplexity_refusal(family_repr(perplexity_param))
        raise refusal^
    if not tsne_perplexity_below_samples(perplexity, n):
        var refusal = tsne_perplexity_samples_refusal(family_str(perplexity_param), n)
        raise refusal^
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
        var refusal = tsne_post_factor_refusal(family_repr(post_param))
        raise refusal^
    var exploration = tsne_exploration_iterations(Int(py=max_iter))
    var init = family_field(fields, "init")
    var y0: PythonObject
    var y0_address = 0
    var initial_snapshot = Optional[ArrayMetadata]()
    var init_pin = ArrayPin()  # the door's pin of a caller-given start array
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
        var captured = array_metadata_once(view, shape^, "init", context, init_pin,
                                           allow_non_native=True, allow_unsupported=True)
        if not tsne_init_shape_matches(captured.shape, n, k):
            var refusal = tsne_init_shape_refusal(family_repr(array_python_shape(captured)), n, k)
            raise refusal^
        var prepared = _cast(captured^, init_pin, "init", context)
        y0 = init_pin.keep
        y0_address = prepared.source.addr
        initial_snapshot = Optional(prepared^)
        _ = view
    var embedding = family_array(Python.tuple(n, k), "<f4", context)
    var kl = family_array(Python.tuple(1), "<f4", context)
    var fit = ann_ops_entry(ann_ops_key(binding))
    var embedding_address = Int(py=classical_address(embedding))
    var kl_address = Int(py=classical_address(kl))
    var iteration_words = Int(py=max_iter)
    var seed_low = Int(py=low)
    var seed_high = Int(py=high)
    var finite_flag = 1 if tsne_fit_tests_input(mode) else 0
    var tested = 0
    # x, y0, y_out, kl_out; n, d, max_iter, exploration_iters, init mode,
    # random_state low, high, n_components, finite flag; perplexity,
    # early_exaggeration, learning_rate, post_exaggeration_learning_rate_factor
    with GILReleased(Python()):
        tested = ann_tsne_fit_call(fit, x_address, y0_address, embedding_address, kl_address, n, d,
                                   iteration_words, exploration, perplexity, exag, lr, post_factor, mode,
                                   seed_low, seed_high, k, finite_flag)
    _ = embedding  # keep the Python object alive through the native call above
    if tsne_fit_tests_input(mode):
        tsne_check_input_status(tested)
    _ = x  # keep the Python object alive through the native call above
    _ = kl  # keep the Python object alive through the native call above
    _ = snapshot
    _ = initial_snapshot
    _ = init_pin^
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
                   snapshot: ArrayMetadata, pin: ArrayPin, context: PythonObject) raises -> PythonObject:
    """TSNE: args = (x_ann binding, estimators binding, X)."""
    if operation == "family_mani_tsne_fit":
        _ = _tsne_fit(model, settings, args, snapshot, pin, context)
        return model
    if operation == "family_mani_tsne_fit_transform":
        return _tsne_fit(model, settings, args, snapshot, pin, context)
    raise Error("RuntimeError: unknown TSNE operation " + operation)
