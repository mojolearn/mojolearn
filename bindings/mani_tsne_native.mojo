# SPDX-License-Identifier: Apache-2.0
# TSNE's public methods, one native call each (python/mojolearn/_expansion_ann.py:
# `fit`, `fit_transform`). The shell hands over the x_ann binding `_bind`
# resolved and the `_mojolearn_estimators` binding of the same numeric mode;
# this owner makes the Python method's refusals (same wording, same order),
# builds the 'pca' start with the call `PCA(n_components=k).fit_transform`
# made (`pca_fit_owned`, the same arguments), runs `x_ann_tsne_fit` with the
# argument lists it always took and stores the fitted attributes. The same
# entries on the same operands: the bits do not move.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_setitem
from bindings.classical_arrays import classical_address, classical_word
from bindings.classical_attributes import classical_connect_numpy
from providers.python.estimator_binding_pyapi import estimator_binding_mode
from bindings.manifold_settings_native import ManifoldSettings
from bindings.spectral_umap_routes import spectral_umap_entry as family_entry, spectral_umap_optional_entry
from core.family_cast_impl import family_all_finite_f32
from bindings.family_common import (
    family_repr, family_str, family_is_str, family_is_int, family_type_name, family_call,
    family_array, family_f32_snapshot, family_field, family_view, family_cast_view, family_shape_repr,
    FAMILY_F32P,
)

comptime _TSNE_EXPLORATION_MAX_ITER = 250


def _tsne_int(name: String, value: PythonObject) raises -> PythonObject:
    """`_ann_int("TSNE", name, value)`."""
    if not family_is_int(value):
        raise Error("TypeError: mojolearn TSNE: " + name + " must be an int, got " + family_type_name(value))
    return family_call(Python.import_module("builtins").int, value)


def _tsne_pca_start(estimators: PythonObject, x_address: Int, n: Int, d: Int, k: Int,
                    context: PythonObject) raises -> PythonObject:
    """`PCA(n_components=k).fit_transform(x)` (svd_solver='auto', the
    covariance arm, no whitening) as the float32 start: `PCA._fit(x, True)`'s
    refusal and its one `pca_fit_owned` call."""
    if k < 1 or k > d:
        raise Error("mojolearn n_components must be in [1, " + String(d) + "], got " + String(k))
    var b = Python.import_module("builtins")
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
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(k), PythonObject(0),
                             PythonObject(0), PythonObject(0), PythonObject(Float64(0.0)))
    _ = family_call(fit_owned, PythonObject(x_address), classical_address(components), classical_address(mean),
                    classical_address(variance), classical_address(ratio), classical_address(singular),
                    classical_address(z), params)
    _ = mean  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = singular  # keep the Python object alive through the native call above
    _ = ratio  # keep the Python object alive through the native call above
    _ = variance  # keep the Python object alive through the native call above
    _ = z  # keep the Python object alive through the native call above
    _ = components^
    _ = mean^
    _ = variance^
    _ = ratio^
    _ = singular^
    # the kept count is every fitted component (keep_mode 0); the start is
    # read from the projection's first n * keep words either way
    return z


def _tsne_fit(model: PythonObject, settings: ManifoldSettings, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`TSNE.fit(X)`; returns the embedding."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    var op = Python.import_module("operator")
    var binding = estimator_binding_mode(settings.mode, "_mojolearn_x_ann", context) if args[0] is None else args[0]
    var estimators = estimator_binding_mode(settings.mode, "_mojolearn_estimators", context) if args[1] is None else args[1]
    var fields = settings.values
    # Reject nonfinite coordinates before PCA initialization reaches its
    # eigensolver; preserve TSNE's public input refusal for every init mode.
    var snapshot = family_f32_snapshot(args[2], 2, "X", context)
    var x = snapshot.keep
    var x_address = snapshot.source.addr
    var n = snapshot.source.rows
    var d = snapshot.source.cols
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=x_address), snapshot.count):
        raise Error("Input X contains NaN or infinity")
    if n < 2:
        # scikit-learn's TSNE validates X with ensure_min_samples=2
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of 2 is required by TSNE.")
    var components = family_field(fields, "n_components")
    if (_tsne_is_bool(components)
            or not Bool(py=py_call(op.contains, Python.tuple(PythonObject(1), PythonObject(2)), components))):
        raise Error("mojolearn TSNE: n_components must be 1 or 2, got " + family_repr(components))
    var k = Int(py=family_call(b.int, components))
    var max_iter = _tsne_int("max_iter", family_field(fields, "max_iter"))
    var seed = _tsne_int("random_state", family_field(fields, "random_state"))
    var perplexity_param = family_field(fields, "perplexity")
    var perplexity = Float64(py=family_call(b.float, perplexity_param))
    # scikit-learn's rules: perplexity > 0 (its parameter constraint) and
    # perplexity < n_samples (`_check_params_vs_input`), in its words
    if not perplexity > 0.0:
        raise Error("The 'perplexity' parameter of TSNE must be a float in the range (0.0, inf). Got "
                    + family_repr(perplexity_param) + " instead.")
    if perplexity >= Float64(n):
        raise Error("perplexity (" + family_str(perplexity_param) + ") must be less than n_samples ("
                    + String(n) + ")")
    var exag = Float64(py=family_call(b.float, family_field(fields, "early_exaggeration")))
    var rate = family_field(fields, "learning_rate")
    var lr: Float64
    if Bool(py=py_call(b.bool, py_call(op.eq, rate, PythonObject("auto")))):
        # max(n / exag / 4.0, 50.0), Python's order (the first unless the second is larger)
        var auto_rate = Float64(n) / exag / 4.0
        lr = Float64(50.0) if 50.0 > auto_rate else auto_rate
    else:
        lr = Float64(py=family_call(b.float, rate))
    var post_param = family_field(fields, "post_exaggeration_learning_rate_factor")
    var post_factor = Float64(py=family_call(b.float, post_param))
    # `0.0 < factor < inf`: positive, and not infinite or NaN (inf - inf is NaN)
    if not (0.0 < post_factor and post_factor - post_factor == 0.0):
        raise Error("mojolearn TSNE: post_exaggeration_learning_rate_factor must be positive and finite, got "
                    + family_repr(post_param))
    var iterations = Int(py=max_iter)
    var exploration = min(_TSNE_EXPLORATION_MAX_ITER, iterations)
    var init = family_field(fields, "init")
    var y0: PythonObject
    var mode = 0
    var low = PythonObject(0)
    var high = PythonObject(0)
    var builtin_start = family_is_str(init) and (String(py=init) == "random" or String(py=init) == "pca")
    if builtin_start:
        # the binding builds the start (x_ann/tsne_init.mojo): mode 1
        # random from random_state, mode 2 the PCA embedding scaled
        if String(py=init) == "random":
            y0 = family_array(Python.tuple(2), "<f4", context)
            mode = 1
        else:
            y0 = _tsne_pca_start(estimators, x_address, n, d, k, context)
            mode = 2
        var s64 = py_call(op.and_, seed, py_call(op.sub, py_call(op.lshift, PythonObject(1), PythonObject(64)),
                                                 PythonObject(1)))
        low = py_call(op.and_, s64, PythonObject(0xFFFFFFFF))
        high = py_call(op.rshift, s64, PythonObject(32))
    else:
        # the caller's start array (n, k) float32 (`TSNE._init`)
        if family_is_str(init):
            raise Error("mojolearn TSNE: init must be 'pca', 'random' or an array, got " + family_repr(init))
        var view = family_view(init, -1, "init", context)
        if Int(py=view.ndim) != 2 or Int(py=view.shape[0]) != n or Int(py=view.shape[1]) != k:
            raise Error("mojolearn TSNE: an init array must have shape (" + String(n) + ", " + String(k)
                        + "), got " + family_shape_repr(view))
        y0 = family_cast_view(view, "init", context)
        _ = view^
    var embedding = family_array(Python.tuple(n, k), "<f4", context)
    var kl = family_array(Python.tuple(1), "<f4", context)
    var fit = family_entry(binding, "x_ann_tsne_fit", context)
    # x, y0, y_out, kl_out; n, d, max_iter, exploration_iters, perplexity,
    # early_exaggeration, learning_rate, post_exaggeration_learning_rate_factor,
    # init mode, random_state low, high, n_components
    _ = family_call(fit, Python.list(PythonObject(x_address), classical_address(y0), classical_address(embedding),
                                     classical_address(kl)),
                    Python.list(PythonObject(n), PythonObject(d), max_iter, PythonObject(exploration),
                                PythonObject(perplexity), PythonObject(exag), PythonObject(lr),
                                PythonObject(post_factor), PythonObject(mode), low, high, PythonObject(k)))
    _ = embedding  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = kl  # keep the Python object alive through the native call above
    _ = snapshot
    _ = x^
    _ = y0^
    py_setitem(settings.fields, "embedding_", embedding)
    py_setitem(settings.fields, "kl_divergence_", PythonObject(classical_word(kl)))
    py_setitem(settings.fields, "n_iter_", max_iter)
    py_setitem(settings.fields, "learning_rate_", PythonObject(lr))
    py_setitem(settings.fields, "n_features_in_", PythonObject(d))
    return embedding


def _tsne_is_bool(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(Python.import_module("builtins").isinstance, value,
                           Python.import_module("builtins").bool))


def tsne_operation(model: PythonObject, settings: ManifoldSettings, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """TSNE: args = (x_ann binding, estimators binding, X)."""
    if operation == "family_mani_tsne_fit":
        _ = _tsne_fit(model, settings, args, context)
        return model
    if operation == "family_mani_tsne_fit_transform":
        return _tsne_fit(model, settings, args, context)
    raise Error("RuntimeError: unknown TSNE operation " + operation)
