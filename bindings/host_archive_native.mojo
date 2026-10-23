# SPDX-License-Identifier: Apache-2.0
# Native saved-model reconstruction. The format table is import-time ABI
# metadata, never Python loaders, constructors, or callbacks. The public
# router is switched only after every existing format has a native builder.
from std.python import Python, PythonObject
from std.memory import bitcast
from std.math import isfinite
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains, py_setattr
from bindings.trees_public_iforest import iforest_validate
from bindings.sequential_description import sequential_fields
from bindings.classical_archive_read import classical_archive_read, classical_archive_build, _archive_exact as _raw_archive_exact, _archive_string, _archive_read_repr, _archive_builtin_repr
from bindings.classical_arrays import classical_i64, classical_f64, classical_wrap_array, classical_array, classical_address, _store_code
from core.host_archive_impl import host_archive_copy_f32, host_archive_unique_i32, host_archive_mean_f64, host_archive_reorder_f32, host_archive_reorder_bytes, host_archive_copy_words, host_archive_support_bound, host_archive_svc_gather
from bindings.scaler_lifecycle_native import scaler_archive_build
from core.host_archive_gbdt import host_archive_gbdt_text


def host_archive_typed(shape: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    # Archive output is always Array storage; obtain the width from its fixed
    # native typecode, including int32 (the linear allocator only supports f4/f8/i8).
    var count = 1
    for i in range(len(shape)):  # driver-loop: native output shape metadata
        var dimension = Int(py=shape[i])
        if dimension < 0:
            raise Error("saved model array dimensions must be nonnegative")
        count *= dimension
    var store = py_call(context["array_type"], _store_code(dtype))
    var raw = py_call(Python.import_module("builtins").bytes, count * Int(py=store.itemsize))
    _ = py_call(store.frombytes, raw)
    return classical_wrap_array(store, shape, dtype, "C", context)


def _archive_exact(arrays: PythonObject, name: String, dtype: String, context: PythonObject) raises -> PythonObject:
    var value = _raw_archive_exact(arrays, name, dtype, context)
    if String(value.order) == "C" or Int(py=value.ndim) < 2:
        return value
    var dimensions = List[Int]()
    var strides = List[Int]()
    for i in range(Int(py=value.ndim)):  # driver-loop: saved array shape metadata
        dimensions.append(Int(py=value.shape[i]))
        strides.append(Int(py=value.strides[i]))
    var result = host_archive_typed(value.shape, dtype, context)
    host_archive_reorder_bytes(Int(py=classical_address(value)), dimensions, strides,
        Int(py=value.size), Int(py=value._mv.itemsize), Int(py=classical_address(result)))
    _ = value  # keep the Python object alive through the native call above
    _ = result  # keep the Python object alive through the native call above
    return result


def host_archive_count(arrays: PythonObject, name: String, dtype: String, count: Int,
                       context: PythonObject, vector: Bool = False) raises -> PythonObject:
    var value = _archive_exact(arrays, name, dtype, context)
    if Int(py=value.size) != count:
        raise Error("model " + name + " must hold " + String(count) + " fields")
    if vector and Int(py=value.ndim) != 1:
        raise Error("model " + name + " must be a one-dimensional field vector")
    return value


def host_archive_shape(arrays: PythonObject, name: String, dtype: String, shape: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    var value = _archive_exact(arrays, name, dtype, context)
    if value.shape != shape:
        raise Error("model " + name + " shape does not match saved metadata")
    return value


def host_archive_check_layout(cls: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: native archive classes require an ordinary native class layout")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var found_dictionary = False
    var found_finalizer = False
    for i in range(len(mro)):  # driver-loop: connected class ancestry metadata
        var fields = py_call(b.type.__dict__["__dict__"].__get__, mro[i])
        if not found_finalizer and py_contains(fields, "__del__"):
            if (fields["__del__"] is not py_get(context, "host_archive_finalizer_descriptor")
                    and fields["__del__"] is not py_get(context, "host_archive_peer_finalizer_descriptor")):
                raise Error("TypeError: Python finalizer hooks are not accepted by native archive loading")
            found_finalizer = True
        if not found_dictionary and py_contains(fields, "__dict__"):
            if py_type(fields["__dict__"]) is not context["host_archive_dict_descriptor_type"]:
                raise Error("TypeError: Python dictionary descriptors are not accepted by native archive loading")
            found_dictionary = True
    if not found_dictionary:
        raise Error("TypeError: native archive loading requires a native instance dictionary")


def host_archive_allocate(cls: PythonObject, context: PythonObject) raises -> PythonObject:
    host_archive_check_layout(cls, context)
    var b = Python.import_module("builtins")
    var model = py_call(b.object.__new__, cls)
    return model


def host_archive_object(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: native archive classes require an ordinary native class layout")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var saved = _archive_string(arrays, "estimator", context)
    var known = False
    for i in range(len(mro)):  # driver-loop: connected class ancestry metadata
        known = known or Bool(py=mro[i].__name__ == saved)
    if not known:
        raise Error("model archive estimator does not match its connected class")
    return host_archive_allocate(cls, context)


def host_archive_mode(arrays: PythonObject, fields: PythonObject, context: PythonObject) raises:
    var mode = _archive_string(arrays, "numeric_mode", context)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("mojolearn: invalid saved numeric_mode " + _archive_builtin_repr(mode))
    py_setitem(fields, "numeric_mode", mode)


def host_archive_where(path: PythonObject, context: PythonObject) raises -> String:
    """`mojolearn: {path!r}`, the head of the loaders' refusals."""
    return "mojolearn: " + _archive_read_repr(path, context)


def host_archive_sized(arrays: PythonObject, name: String, dtype: String, count: Int,
                       message: String, context: PythonObject) raises -> PythonObject:
    """An exact-dtype member of `count` elements; any other size raises `message`."""
    var value = _archive_exact(arrays, name, dtype, context)
    if Int(py=value.size) != count:
        raise Error(message)
    return value


def host_archive_meta(arrays: PythonObject, name: String, count: Int, path: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """The int64 `meta` vector, refused as the loaders refused it:
    `{path!r} meta holds {n} fields, {count} are needed`."""
    var value = _archive_exact(arrays, name, "<i8", context)
    if Int(py=value.size) != count:
        raise Error(host_archive_where(path, context) + " " + name + " holds " + String(Int(py=value.size))
                    + " fields, " + String(count) + " are needed")
    return value


def host_archive_matrix(arrays: PythonObject, name: String, dtype: String, rows: Int, cols: Int,
                        path: PythonObject, context: PythonObject) raises -> PythonObject:
    """A 2-D member of shape (rows, cols), refused as
    `{path!r} {name} shape {shape} is not ({rows}, {cols})`."""
    var value = _archive_exact(arrays, name, dtype, context)
    if Int(py=value.ndim) != 2 or Int(py=value.shape[0]) != rows or Int(py=value.shape[1]) != cols:
        raise Error(host_archive_where(path, context) + " " + name + " shape " + _archive_builtin_repr(value.shape)
                    + " is not (" + String(rows) + ", " + String(cols) + ")")
    return value


def host_archive_decomposition(cls: PythonObject, arrays: PythonObject, pca: Bool, path: PythonObject,
                               context: PythonObject) raises -> PythonObject:
    # The refusals keep PCA.load's and TruncatedSVD.load's wording.
    var meta = host_archive_meta(arrays, "meta", 4 if pca else 2, path, context)
    var nc = classical_i64(meta, 0)
    var nf = classical_i64(meta, 1)
    var where = host_archive_where(path, context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    var solver = _archive_string(arrays, "svd_solver" if pca else "algorithm", context)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "n_components", nc)
    py_setitem(fields, "n_components_", nc)
    py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "tol", 0.0)
    py_setitem(fields, "n_oversamples", 10)
    py_setitem(fields, "power_iteration_normalizer", "auto")
    py_setitem(fields, "random_state", PythonObject(None))
    py_setitem(fields, "components_", host_archive_matrix(arrays, "components", "<f4", nc, nf, path, context))
    if pca:
        py_setitem(fields, "copy", True)
        py_setitem(fields, "whiten", classical_i64(meta, 3) != 0)
        py_setitem(fields, "svd_solver", solver)
        py_setitem(fields, "iterated_power", "auto")
        py_setitem(fields, "mean_", host_archive_sized(arrays, "mean", "<f4", nf, where + " mean_ does not match n_features_in_", context))
        py_setitem(fields, "singular_values_", host_archive_sized(arrays, "singular_values", "<f4", nc, where + " singular_values_ does not match n_components_", context))
        py_setitem(fields, "explained_variance_", host_archive_sized(arrays, "explained_variance", "<f4", nc, where + " explained_variance_ does not match n_components_", context))
        py_setitem(fields, "explained_variance_ratio_", host_archive_sized(arrays, "explained_variance_ratio", "<f4", nc, where + " explained_variance_ratio_ does not match n_components_", context))
        var noise = host_archive_sized(arrays, "noise_variance", "<f8", 1, where + " noise_variance must hold one value", context)
        py_setitem(fields, "noise_variance_", classical_f64(noise, 0))
        py_setitem(fields, "n_samples_", classical_i64(meta, 2))
    else:
        py_setitem(fields, "singular_values_", host_archive_sized(arrays, "singular_values", "<f4", nc, where + " singular_values does not match n_components_", context))
        py_setitem(fields, "algorithm", solver)
        py_setitem(fields, "n_iter", 5)
    return model


def host_archive_held(arrays: PythonObject, name: String, dtype: String, count: Int, path: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """An exact-dtype field vector of `count` elements, refused as the
    loaders refused it: `{path!r} {name} holds {n} fields, {count} are needed`."""
    var value = _archive_exact(arrays, name, dtype, context)
    if Int(py=value.size) != count:
        raise Error(host_archive_where(path, context) + " " + name + " holds " + String(Int(py=value.size))
                    + " fields, " + String(count) + " are needed")
    return value


def host_archive_cd(cls: PythonObject, arrays: PythonObject, lasso: Bool, path: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    # The refusals keep ElasticNet.load's and Lasso.load's wording and order;
    # the fields follow the constructor's, then the loader's, assignments.
    var where = host_archive_where(path, context)
    var meta = host_archive_meta(arrays, "meta", 4, path, context)
    var hyper = host_archive_held(arrays, "hyper", "<f8", 3, path, context)
    var ratio = classical_f64(hyper, 1)
    if lasso and ratio != 1.0:
        raise Error(where + " carries l1_ratio " + _archive_builtin_repr(PythonObject(ratio)) + ", a Lasso is 1.0")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "alpha", classical_f64(hyper, 0))
    py_setitem(fields, "l1_ratio", ratio)
    py_setitem(fields, "fit_intercept", classical_i64(meta, 1) != 0)
    py_setitem(fields, "max_iter", classical_i64(meta, 2))
    py_setitem(fields, "tol", classical_f64(hyper, 2))
    py_setitem(fields, "solver", "auto")
    py_setitem(fields, "selection", "cyclic")
    py_setitem(fields, "precompute", False)
    py_setitem(fields, "copy_X", True)
    py_setitem(fields, "warm_start", False)
    py_setitem(fields, "positive", False)
    py_setitem(fields, "random_state", PythonObject(None))
    host_archive_mode(arrays, fields, context)
    var nf = classical_i64(meta, 0)
    py_setitem(fields, "n_features_in_", nf)
    var coef = _archive_exact(arrays, "coef", "<f4", context)
    if Int(py=coef.ndim) != 1 or Int(py=coef.size) != nf:
        raise Error(where + " coef does not match n_features_in_")
    py_setitem(fields, "coef_", coef)
    var intercept = host_archive_sized(arrays, "intercept", "<f8", 1, where + " intercept must hold one value", context)
    py_setitem(fields, "intercept_", classical_f64(intercept, 0))
    py_setitem(fields, "n_iter_", classical_i64(meta, 3))
    return model


def host_archive_fields(arrays: PythonObject, name: String, dtype: String, count: Int, path: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """An exact-dtype field vector refused as the kernel-method loaders
    refused it: `{path!r} {name} holds {n} fields, {count} are needed`."""
    var value = _archive_exact(arrays, name, dtype, context)
    if Int(py=value.size) != count:
        raise Error(host_archive_where(path, context) + " " + name + " holds " + String(Int(py=value.size))
                    + " fields, " + String(count) + " are needed")
    return value


def host_archive_tuple_shape(arrays: PythonObject, name: String, dtype: String, shape: PythonObject,
                             path: PythonObject, context: PythonObject) raises -> PythonObject:
    """A member whose shape is exactly `shape` (a tuple), refused as
    `{path!r} {name} shape {shape} is not {want}`."""
    var b = Python.import_module("builtins")
    var value = _archive_exact(arrays, name, dtype, context)
    var got = py_call(b.tuple, value.shape)
    if not Bool(py=got == shape):
        raise Error(host_archive_where(path, context) + " " + name + " shape " + _archive_builtin_repr(got)
                    + " is not " + _archive_builtin_repr(shape))
    return value


def host_archive_kernel(cls: PythonObject, arrays: PythonObject, kind: Int, path: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # kind: 0 KernelRidge, 1 Nystroem, 2 RBFSampler. These are model
    # schemas, unrelated to benchmark shapes or device execution routes.
    # The refusals keep KernelRidge.load's, Nystroem.load's and
    # RBFSampler.load's wording and order: meta, hyper, numeric_mode, then
    # the arrays in the order the loaders read them.
    var meta = host_archive_fields(arrays, "meta", "<i8", 7 if kind == 0 else (6 if kind == 1 else 3), path, context)
    var hyper = host_archive_fields(arrays, "hyper", "<f8", 2 if kind == 1 else 3, path, context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    var gamma = classical_f64(hyper, 0)
    py_setitem(fields, "gamma", gamma)
    if kind == 2:
        var d = classical_i64(meta, 0)
        var q = classical_i64(meta, 1)
        var seed = classical_i64(meta, 2)
        py_setitem(fields, "n_components", q)
        py_setitem(fields, "random_state", seed)
        py_setitem(fields, "n_features_in_", d)
        py_setitem(fields, "random_weights_", host_archive_tuple_shape(arrays, "random_weights", "<f4", Python.tuple(d, q), path, context))
        py_setitem(fields, "random_offset_", host_archive_tuple_shape(arrays, "random_offset", "<f4", Python.tuple(q), path, context))
        py_setitem(fields, "sigma_", classical_f64(hyper, 1))
        py_setitem(fields, "scale_", classical_f64(hyper, 2))
        py_setitem(fields, "_params", Python.tuple(gamma, seed))
        return model
    var n = classical_i64(meta, 0)
    var d = classical_i64(meta, 1)
    var kernel = classical_i64(meta, 3 if kind == 0 else 2)
    var degree = classical_i64(meta, 4 if kind == 0 else 3)
    var coef0 = classical_f64(hyper, 1)
    py_setitem(fields, "kernel", kernel)
    py_setitem(fields, "degree", degree)
    py_setitem(fields, "coef0", coef0)
    py_setitem(fields, "n_features_in_", d)
    if kind == 0:
        var t = classical_i64(meta, 2)
        var alpha = classical_f64(hyper, 2)
        var squeeze = classical_i64(meta, 6) != 0
        py_setitem(fields, "alpha", alpha)
        py_setitem(fields, "X_fit_", host_archive_tuple_shape(arrays, "x_fit", "<f4", Python.tuple(n, d), path, context))
        var dual = host_archive_tuple_shape(arrays, "dual", "<f4", Python.tuple(n * t), path, context)
        if not squeeze:
            dual = classical_wrap_array(dual._store, Python.tuple(n, t), "<f4", "C", context)
        py_setitem(fields, "n_targets_", t)
        py_setitem(fields, "_squeeze", squeeze)
        py_setitem(fields, "dual_coef_", dual)
        py_setitem(fields, "info_", classical_i64(meta, 5))
        py_setitem(fields, "_kernel_params", Python.tuple(kernel, degree, gamma, coef0, alpha))
    else:
        var seed = classical_i64(meta, 4)
        py_setitem(fields, "n_components", n)
        py_setitem(fields, "random_state", seed)
        py_setitem(fields, "components_", host_archive_tuple_shape(arrays, "components", "<f4", Python.tuple(n, d), path, context))
        py_setitem(fields, "component_indices_", host_archive_tuple_shape(arrays, "component_indices", "<i4", Python.tuple(n), path, context))
        py_setitem(fields, "normalization_", host_archive_tuple_shape(arrays, "normalization", "<f4", Python.tuple(n, n), path, context))
        py_setitem(fields, "eigenvalues_", host_archive_tuple_shape(arrays, "eigenvalues", "<f4", Python.tuple(n), path, context))
        py_setitem(fields, "eigenvectors_", host_archive_tuple_shape(arrays, "eigenvectors", "<f4", Python.tuple(n, n), path, context))
        py_setitem(fields, "sweeps_", classical_i64(meta, 5))
        py_setitem(fields, "_kernel_params", Python.tuple(kernel, degree, gamma, coef0, seed))
    return model


def host_archive_minmax(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 3, context, True)
    var bounds = host_archive_count(arrays, "feature_range", "<f8", 2, context, True)
    var d = classical_i64(meta, 0)
    var mode = _archive_string(arrays, "numeric_mode", context)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("invalid saved numeric_mode")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    var low = classical_f64(bounds, 0)
    var high = classical_f64(bounds, 1)
    py_setitem(fields, "feature_range", Python.tuple(low, high))
    py_setitem(fields, "feature_range_", Python.tuple(low, high))
    py_setitem(fields, "clip", classical_i64(meta, 2) != 0)
    py_setitem(fields, "clip_", classical_i64(meta, 2) != 0)
    py_setitem(fields, "copy", True)
    py_setitem(fields, "numeric_mode", PythonObject(None))
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "n_samples_seen_", classical_i64(meta, 1))
    var names = Python.tuple("data_min", "data_max", "data_range", "scale", "min")
    for i in range(len(names)):  # driver-loop: five saved scaler statistics
        var name = String(names[i])
        py_setitem(fields, name + "_", host_archive_shape(arrays, name, "<f4", Python.tuple(d), context))
    return model


def host_archive_cholesky(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    # The refusals keep Cholesky.load's wording and order: numeric_mode,
    # the three field vectors' dtypes, their sizes, then the factor.
    var b = Python.import_module("builtins")
    var where = host_archive_where(path, context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    var scalars = _archive_exact(arrays, "scalars", "<f8", context)
    var jp = _archive_exact(arrays, "jitter_param", "<f8", context)
    if Int(py=meta.size) != 3 or Int(py=scalars.size) != 2 or Int(py=jp.size) != 1:
        raise Error(where + " holds meta " + String(Int(py=meta.size)) + ", scalars " + String(Int(py=scalars.size))
                    + " and jitter_param " + String(Int(py=jp.size)) + " fields; 3, 2 and 1 are needed")
    var n = classical_i64(meta, 0)
    var factor = _archive_exact(arrays, "L", "<f4", context)
    var got = py_call(b.tuple, factor.shape)
    if n < 1 or not Bool(py=got == Python.tuple(n, n)):
        raise Error(where + " L has shape " + _archive_builtin_repr(got) + ", the factor is "
                    + String(n) + " x " + String(n))
    var jitter = classical_f64(jp, 0)
    py_setitem(fields, "jitter", PythonObject(None) if jitter == -1.0 else PythonObject(jitter))
    py_setitem(fields, "n_", n)
    py_setitem(fields, "L_", factor)
    py_setitem(fields, "info_", classical_i64(meta, 1))
    py_setitem(fields, "nb_", classical_i64(meta, 2))
    py_setitem(fields, "_logdet", classical_f64(scalars, 0))
    py_setitem(fields, "jitter_", classical_f64(scalars, 1))
    return model


def host_archive_embedding(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 3, context, True)
    var v = classical_i64(meta, 0)
    var d = classical_i64(meta, 1)
    var pad = classical_i64(meta, 2)
    if v < 1 or d < 1:
        raise Error("Embedding dimensions must be positive")
    if pad != -1 and (pad < -v or pad >= v):
        raise Error("Embedding padding_idx is outside the vocabulary")
    var plan = _archive_string(arrays, "plan", context)
    if String(plan) != "auto" and String(plan) != "scan" and String(plan) != "sort":
        raise Error("Embedding plan must be auto, scan or sort")
    var saved = host_archive_shape(arrays, "weight", "<f4", Python.tuple(v, d), context)
    var weight = classical_array(v, d, context)
    host_archive_copy_f32(Int(py=classical_address(saved)), Int(py=saved.strides[0]),
                          Int(py=saved.strides[1]), v, d, Int(py=classical_address(weight)))
    _ = saved  # keep the Python object alive through the native call above
    _ = weight  # keep the Python object alive through the native call above
    _ = saved^
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "num_embeddings", v)
    py_setitem(fields, "embedding_dim", d)
    py_setitem(fields, "padding_idx", PythonObject(None) if pad == -1 else PythonObject(v + pad if pad < 0 else pad))
    py_setitem(fields, "max_norm", PythonObject(None))
    py_setitem(fields, "norm_type", 2.0)
    py_setitem(fields, "scale_grad_by_freq", False)
    py_setitem(fields, "sparse", False)
    py_setitem(fields, "plan", plan)
    py_setitem(fields, "weight", weight)
    return model


def host_archive_sorted(table: PythonObject) raises -> String:
    """repr(sorted(table)): the list of names a loader's refusal shows."""
    var b = Python.import_module("builtins")
    return String(py_call(b.repr, py_call(b.sorted, table)))


def host_archive_kmeans(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # The refusals keep KMeans.load's wording and order.
    var where = host_archive_where(path, context)
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    var reals = _archive_exact(arrays, "reals", "<f8", context)
    if Int(py=meta.size) != 9 or Int(py=reals.size) != 5:
        raise Error(where + " holds " + String(Int(py=meta.size)) + " meta fields and " + String(Int(py=reals.size))
                    + " reals, 9 and 5 are needed")
    var k = classical_i64(meta, 0)
    var d = classical_i64(meta, 1)
    var n = classical_i64(meta, 2)
    if k < 1 or d < 1:
        raise Error(where + " holds an empty model (" + String(k) + " centers of " + String(d) + " features)")
    var metric = _archive_string(arrays, "metric", context)
    var start = _archive_string(arrays, "init", context)
    var metrics = context["host_archive_kmeans_metrics"]
    var starts = context["host_archive_kmeans_starts"]
    if not py_contains(metrics, metric):
        raise Error(where + " metric " + _archive_builtin_repr(metric) + " is not one KMeans serves ("
                    + host_archive_sorted(metrics) + ")")
    var metric_code = Int(py=metrics[metric])
    if metric_code != classical_i64(meta, 4):
        raise Error(where + " names metric " + _archive_builtin_repr(metric) + " (code " + String(metric_code)
                    + ") but its meta holds code " + String(classical_i64(meta, 4)))
    if not py_contains(starts, start):
        raise Error(where + " init " + _archive_builtin_repr(start) + " is not one KMeans serves ("
                    + host_archive_sorted(starts) + ")")
    var start_code = Int(py=starts[start])
    if start_code != classical_i64(meta, 5):
        raise Error(where + " names init " + _archive_builtin_repr(start) + " (code " + String(start_code)
                    + ") but its meta holds code " + String(classical_i64(meta, 5)))
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "n_clusters", k)
    py_setitem(fields, "init", start)
    py_setitem(fields, "n_init", classical_i64(meta, 7))
    py_setitem(fields, "max_iter", classical_i64(meta, 6))
    py_setitem(fields, "tol", classical_f64(reals, 3))
    py_setitem(fields, "random_state", classical_i64(meta, 8))
    py_setitem(fields, "init_centroids", PythonObject(None))
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "oversampling_factor", classical_f64(reals, 4))
    host_archive_mode(arrays, fields, context)
    var centers = _archive_exact(arrays, "centers", "<f4", context)
    if Int(py=centers.ndim) != 2 or Int(py=centers.shape[0]) != k or Int(py=centers.shape[1]) != d:
        raise Error(where + " centers shape " + _archive_builtin_repr(centers.shape) + " is not (" + String(k)
                    + ", " + String(d) + "); the centroid count and the dimensionality disagree")
    var labels = _archive_exact(arrays, "labels", "<i4", context)
    if Int(py=labels.ndim) != 1 or Int(py=labels.size) != n:
        raise Error(where + " labels hold " + String(Int(py=labels.size)) + " values, the fit had " + String(n) + " rows")
    py_setitem(fields, "cluster_centers_", centers)
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "n_iter_", classical_i64(meta, 3))
    py_setitem(fields, "inertia_", classical_f64(reals, 0))
    py_setitem(fields, "sum_scale_", classical_f64(reals, 1))
    py_setitem(fields, "weight_scale_", classical_f64(reals, 2))
    return model


def host_archive_vector(arrays: PythonObject, name: String, count: Int, where: String,
                        context: PythonObject) raises -> PythonObject:
    """A 1-D float32 member of `count` values, refused as
    `{path!r} {name} holds {n} values, {count} are needed`."""
    var value = _archive_exact(arrays, name, "<f4", context)
    if Int(py=value.ndim) != 1 or Int(py=value.size) != count:
        raise Error(where + " " + name + " holds " + String(Int(py=value.size)) + " values, "
                    + String(count) + " are needed")
    return value


def host_archive_mixture(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    # The refusals keep GaussianMixture.load's wording and order.
    var where = host_archive_where(path, context)
    var meta = _archive_exact(arrays, "ints", "<i8", context)
    var reals = _archive_exact(arrays, "reals", "<f8", context)
    if Int(py=meta.size) != 6 or Int(py=reals.size) != 3:
        raise Error(where + " holds " + String(Int(py=meta.size)) + " ints and " + String(Int(py=reals.size))
                    + " reals, 6 and 3 are needed")
    var k = classical_i64(meta, 0)
    var d = classical_i64(meta, 1)
    if k < 1 or d < 1:
        raise Error(where + " holds an empty model")
    var covariance_type = _archive_string(arrays, "covariance_type", context)
    var init_params = _archive_string(arrays, "init_params", context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "n_components", k)
    py_setitem(fields, "covariance_type", covariance_type)
    py_setitem(fields, "tol", classical_f64(reals, 1))
    py_setitem(fields, "reg_covar", classical_f64(reals, 2))
    py_setitem(fields, "max_iter", classical_i64(meta, 4))
    py_setitem(fields, "init_params", init_params)
    py_setitem(fields, "random_state", classical_i64(meta, 5))
    py_setitem(fields, "n_init", 1)
    py_setitem(fields, "warm_start", False)
    py_setitem(fields, "means_init", PythonObject(None))
    py_setitem(fields, "weights_init", PythonObject(None))
    py_setitem(fields, "precisions_init", PythonObject(None))
    host_archive_mode(arrays, fields, context)
    var weights = host_archive_vector(arrays, "weights", k, where, context)
    var means = host_archive_vector(arrays, "means", k * d, where, context)
    var covariance = host_archive_vector(arrays, "covariances", k * d * d, where, context)
    var precision = host_archive_vector(arrays, "precisions_cholesky", k * d * d, where, context)
    var log_det = host_archive_vector(arrays, "log_det_chol", k, where, context)
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "weights_", weights)
    py_setitem(fields, "means_", classical_wrap_array(means._store, Python.tuple(k, d), "<f4", "C", context))
    py_setitem(fields, "covariances_", classical_wrap_array(covariance._store, Python.tuple(k, d, d), "<f4", "C", context))
    py_setitem(fields, "precisions_cholesky_", classical_wrap_array(precision._store, Python.tuple(k, d, d), "<f4", "C", context))
    py_setitem(fields, "log_det_chol_", log_det)
    py_setitem(fields, "n_iter_", classical_i64(meta, 2))
    py_setitem(fields, "converged_", classical_i64(meta, 3) != 0)
    py_setitem(fields, "lower_bound_", classical_f64(reals, 0))
    return model


def host_archive_density(cls: PythonObject, arrays: PythonObject, kde: Bool, path: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    if not kde:
        return host_archive_dbscan(cls, arrays, path, context)
    var meta = host_archive_count(arrays, "meta", "<i8", 3, context, True)
    var parameter = host_archive_count(arrays, "bandwidth", "<f8", 1, context, True)
    var nf = classical_i64(meta, 0)
    var n = classical_i64(meta, 1)
    var metric = _archive_string(arrays, "metric", context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "n_samples_fit_", n)
    var bandwidth = classical_f64(parameter, 0)
    if not bandwidth > 0.0:
        raise Error("mojolearn KernelDensity: bandwidth must be positive, got " + host_archive_float_repr(bandwidth))
    py_setitem(fields, "bandwidth", bandwidth)
    py_setitem(fields, "kernel", _archive_string(arrays, "kernel", context))
    py_setitem(fields, "algorithm", "auto")
    py_setitem(fields, "atol", 0)
    py_setitem(fields, "rtol", 0)
    py_setitem(fields, "breadth_first", True)
    py_setitem(fields, "leaf_size", 40)
    py_setitem(fields, "metric_params", PythonObject(None))
    py_setitem(fields, "_x", host_archive_shape(arrays, "x", "<f4", Python.tuple(n, nf), context))
    if classical_i64(meta, 2) != 0:
        py_setitem(fields, "_w", host_archive_shape(arrays, "weights", "<f4", Python.tuple(n), context))
    else:
        if py_contains(arrays, "weights"):
            raise Error("KernelDensity archive carries weights its meta says it lacks")
        py_setitem(fields, "_w", PythonObject(None))
    return model


def host_archive_dbscan(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """DBSCAN.load: the refusals keep the Python loader's wording and order."""
    var where = host_archive_where(path, context)
    var meta = host_archive_meta(arrays, "meta", 3, path, context)
    var eps = host_archive_sized(arrays, "eps", "<f8", 1, where + " eps must hold one value", context)
    var nf = classical_i64(meta, 0)
    var n = classical_i64(meta, 1)
    var metric = _archive_string(arrays, "metric", context)
    var algorithm = _archive_string(arrays, "algorithm", context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "eps", classical_f64(eps, 0))
    py_setitem(fields, "min_samples", classical_i64(meta, 2))
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "algorithm", algorithm)
    py_setitem(fields, "max_mbytes_per_batch", PythonObject(None))
    py_setitem(fields, "max_iterations", PythonObject(None))
    py_setitem(fields, "prediction_data", True)
    host_archive_mode(arrays, fields, context)
    if not py_contains(context["host_archive_dbscan_metrics"], metric):
        raise Error(where + " metric " + _archive_builtin_repr(metric) + " is not one DBSCAN serves")
    var components = _archive_exact(arrays, "components", "<f4", context)
    var indices = _archive_exact(arrays, "core_sample_indices", "<i4", context)
    var core_labels = _archive_exact(arrays, "core_labels", "<i4", context)
    var labels = _archive_exact(arrays, "labels", "<i4", context)
    var nc = Int(py=indices.size)
    if Int(py=components.ndim) != 2 or Int(py=components.shape[0]) != nc or Int(py=components.shape[1]) != nf:
        raise Error(where + " components shape " + _archive_builtin_repr(components.shape)
                    + " is not (" + String(nc) + ", " + String(nf) + ")")
    if Int(py=core_labels.size) != nc or Int(py=labels.size) != n:
        raise Error(where + " core labels or labels do not match the saved counts")
    py_setitem(fields, "components_", components)
    py_setitem(fields, "core_sample_indices_", indices)
    py_setitem(fields, "_core_labels", core_labels)
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "n_samples_fit_", n)
    return model


def host_archive_agglomerative(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                               context: PythonObject) raises -> PythonObject:
    # The refusals keep AgglomerativeClustering.load's wording and order
    # (the metric refusal is its constructor check's).
    var where = host_archive_where(path, context)
    var meta = host_archive_meta(arrays, "meta", 3, path, context)
    var k = classical_i64(meta, 0)
    var nf = classical_i64(meta, 1)
    var n = classical_i64(meta, 2)
    var metric = _archive_string(arrays, "metric", context)
    var metrics = context["host_archive_agglo_metrics"]
    if not py_contains(metrics, metric):
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: metric=" + _archive_builtin_repr(metric)
                    + " is refused by name; one of " + host_archive_sorted(metrics) + ". A callable metric would "
                    + "compute the dissimilarities outside every identity column. See x_cluster/NOT_IMPLEMENTED.tsv")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "n_clusters", k)
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "connectivity", PythonObject(None))
    py_setitem(fields, "linkage", "single")
    py_setitem(fields, "c", 15)
    py_setitem(fields, "memory", PythonObject(None))
    py_setitem(fields, "compute_full_tree", "auto")
    py_setitem(fields, "distance_threshold", PythonObject(None))
    py_setitem(fields, "compute_distances", False)
    py_setitem(fields, "prediction_data", True)
    py_setitem(fields, "_connectivity", "pairwise")
    py_setitem(fields, "_compute_distances", False)
    host_archive_mode(arrays, fields, context)
    var x = _archive_exact(arrays, "x", "<f4", context)
    var labels = _archive_exact(arrays, "labels", "<i4", context)
    if Int(py=x.ndim) != 2 or Int(py=x.shape[0]) != n or Int(py=x.shape[1]) != nf:
        raise Error(where + " x shape " + _archive_builtin_repr(x.shape) + " is not (" + String(n) + ", " + String(nf) + ")")
    if Int(py=labels.size) != n:
        raise Error(where + " labels do not match n_leaves_")
    py_setitem(fields, "_fit_X", x)
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "n_clusters_", k)
    py_setitem(fields, "n_leaves_", n)
    py_setitem(fields, "n_features_in_", nf)
    return model


def host_archive_class_values(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Archive decoding produces exact strings, lists, and Arrays. Read their
    # native storage directly; Array.tolist/__iter__ and arbitrary label
    # protocols never participate in loading.
    var b = Python.import_module("builtins")
    if py_type(value) is b.str:
        return Python.list(value)
    var result = Python.list()
    if py_type(value) is b.list or py_type(value) is b.tuple:
        for i in range(len(value)):  # driver-loop: saved class label nesting
            var nested = host_archive_class_values(value[i], context)
            _ = py_call(result.extend, nested)
        return result
    if py_type(value) is not context["array_class"]:
        raise Error("TypeError: saved class labels must be native archive values")
    var rank = Int(py=value.ndim)
    var count = Int(py=value.size)
    var width = Int(py=value.itemsize)
    var shape = value.shape
    var strides = value.strides
    var storage = value._mv
    var half = String(value.dtype) == "<f2"
    for i in range(count):  # driver-loop: saved distinct class label metadata
        var remaining = i
        var offset = 0
        for dimension in range(rank - 1, -1, -1):  # driver-loop: saved class label shape metadata
            var length = Int(py=shape[dimension])
            offset += (remaining % length) * Int(py=strides[dimension])
            remaining //= length
        var item = storage[offset // width]
        if half:
            item = PythonObject(Float64(bitcast[DType.float16](UInt16(Int(py=item)))))
        _ = py_call(result.append, item)
    return result


def host_archive_logistic(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    # The refusals keep LogisticRegression.load's wording and order; the
    # fields follow the constructor's, then the loader's, assignments.
    var where = host_archive_where(path, context)
    var meta = host_archive_meta(arrays, "meta", 2, path, context)
    var nf = classical_i64(meta, 0)
    var intercept = classical_i64(meta, 1) != 0
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "penalty", "l2")
    py_setitem(fields, "tol", 0.0001)
    py_setitem(fields, "C", 1.0)
    py_setitem(fields, "fit_intercept", intercept)
    py_setitem(fields, "class_weight", PythonObject(None))
    py_setitem(fields, "max_iter", 1000)
    py_setitem(fields, "linesearch_max_iter", 50)
    py_setitem(fields, "l1_ratio", PythonObject(None))
    py_setitem(fields, "solver", "qn")
    host_archive_mode(arrays, fields, context)
    if not py_contains(arrays, "classes"):
        raise Error(where + " has no classes member")
    var classes = host_archive_class_values(arrays["classes"], context)
    if len(classes) < 2:
        raise Error(where + " must carry at least two classes")
    var targets = 1 if len(classes) == 2 else len(classes)
    var w = _archive_exact(arrays, "w", "<f4", context)
    if nf < 0 or Int(py=w.ndim) != 1 or Int(py=w.size) != (nf + (1 if intercept else 0)) * targets:
        raise Error(where + " w does not match n_features_in_, fit_intercept and the class count")
    var coef = host_archive_typed(Python.tuple(targets, nf), "<f4", context)
    var bias = classical_array(targets, 0, context)
    host_archive_copy_f32(Int(py=classical_address(w)), 4, targets * 4, targets, nf, Int(py=classical_address(coef)))
    _ = coef  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    if intercept:
        host_archive_copy_f32(Int(py=classical_address(w)) + targets * nf * 4, 4, 4,
                              targets, 1, Int(py=classical_address(bias)))
        _ = w  # keep the Python object alive through the native call above
        _ = bias  # keep the Python object alive through the native call above
    py_setitem(fields, "classes_", classes)
    py_setitem(fields, "_w", w)
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", bias)
    py_setitem(fields, "n_features_in_", nf)
    return model


def _arima_order(values: PythonObject, label: String) raises:
    """ARIMA `_as_order`'s refusal of a negative saved order entry."""
    for i in range(len(values)):  # driver-loop: over the fixed entries of one order tuple
        if Int(py=values[i]) < 0:
            raise Error("ValueError: mojolearn ARIMA: " + label + " entries must be non-negative integers, got "
                        + _archive_builtin_repr(values))


def host_archive_arima(cls: PythonObject, arrays: PythonObject, exog: Bool, path: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    # The refusals keep ARIMA.load's wording and order: the meta size and
    # format checks, the constructor's validation, the trend's intercept,
    # the saved mode, then each array's shape.
    var where = host_archive_where(path, context)
    var meta = host_archive_meta(arrays, "meta", 13 if exog else 12, path, context)
    var p = classical_i64(meta, 0)
    var d = classical_i64(meta, 1)
    var q = classical_i64(meta, 2)
    var sp = classical_i64(meta, 3)
    var sd = classical_i64(meta, 4)
    var sq = classical_i64(meta, 5)
    var period = classical_i64(meta, 6)
    var k = classical_i64(meta, 7)
    var batch = classical_i64(meta, 8)
    var n = classical_i64(meta, 9)
    var nx = classical_i64(meta, 12) if exog else 0
    if exog and nx < 1:
        raise Error(where + " is 'mojolearn-arima-2' with n_exog=" + String(nx))
    if not exog and py_contains(arrays, "exog"):
        raise Error(where + " is 'mojolearn-arima-1' and holds an exog member")
    var trend = _archive_string(arrays, "trend", context)
    var trend_arg = PythonObject(None) if String(trend) == "none" else trend
    var method_saved = _archive_string(arrays, "method", context)
    # The constructor's validation, in its order.
    var order = Python.tuple(p, d, q)
    var seasonal_order = Python.tuple(sp, sd, sq, period)
    _arima_order(order, "order")
    _arima_order(seasonal_order, "seasonal_order")
    var expected_k = 0 if d + sd else 1
    if trend_arg is not None:
        var lower = String(py=py_call(trend.lower))
        if lower == "n" or lower == "c":
            expected_k = 1 if lower == "c" else 0
        elif lower == "t" or lower == "ct":
            raise Error("NotImplementedError: mojolearn ARIMA: trend=" + _archive_builtin_repr(trend)
                        + " is refused. cuML has no time trend (its only switch is fit_intercept), so there is no "
                        + "reference to hold one to; a time trend is a regressor, so pass the time index as an exog "
                        + "column to fit, predict and forecast. This class carries trend=None, 'n' and 'c'; 'c' is "
                        + "cuML's fit_intercept=True")
        else:
            raise Error("NotImplementedError: mojolearn ARIMA: trend=" + _archive_builtin_repr(trend)
                        + " is refused. This class carries None (statsmodels' rule: 'c' when d + D == 0, "
                        + "otherwise 'n'), 'n' and 'c'. A polynomial trend specification is a list of regressors: "
                        + "pass them as exog")
    var method = py_call(method_saved.lower)
    if String(method) != "ml" and String(method) != "css" and String(method) != "css-ml":
        raise Error("ValueError: mojolearn ARIMA: unknown method " + _archive_builtin_repr(method_saved)
                    + "; cuML's three are ['css', 'css-ml', 'ml'] and this implementation offers 'ml'")
    if k != expected_k:
        raise Error(where + " records k=" + String(k) + ", its trend " + _archive_builtin_repr(trend)
                    + " resolves to " + String(expected_k))
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "order", order)
    py_setitem(fields, "seasonal_order", seasonal_order)
    py_setitem(fields, "trend", trend_arg)
    py_setitem(fields, "k_", expected_k)
    py_setitem(fields, "method", method)
    py_setitem(fields, "maxiter", classical_i64(meta, 10))
    py_setitem(fields, "verbose", False)
    py_setitem(fields, "output_type", PythonObject(None))
    py_setitem(fields, "n_exog_", 0)
    py_setitem(fields, "complexity_", p + q + sp + sq + expected_k + 1)
    host_archive_mode(arrays, fields, context)
    var complexity = p + q + sp + sq + expected_k + 1 + nx
    var loaded = Python.dict()
    py_setitem(loaded, "_y", host_archive_series_shape(arrays, "y", "<f4", Python.tuple(batch, n), path, context))
    var coefficients = Python.tuple("params", "x", "x0")
    for i in range(len(coefficients)):  # driver-loop: three saved coefficient states
        var name = String(coefficients[i])
        py_setitem(loaded, name + "_", host_archive_series_shape(arrays, name, "<f4", Python.tuple(batch, complexity), path, context))
    py_setitem(loaded, "fx_", host_archive_series_shape(arrays, "fx", "<f4", Python.tuple(batch), path, context))
    py_setitem(loaded, "n_iter_", host_archive_series_shape(arrays, "n_iter", "<i4", Python.tuple(batch), path, context))
    py_setitem(loaded, "retcode_", host_archive_series_shape(arrays, "retcode", "<i4", Python.tuple(batch), path, context))
    var scores = Python.tuple("llf", "aic", "bic")
    for i in range(len(scores)):  # driver-loop: three saved fit criterion arrays
        var name = String(scores[i])
        py_setitem(loaded, name + "_", host_archive_series_shape(arrays, name, "<f8", Python.tuple(batch), path, context))
    py_setitem(loaded, "_exog", host_archive_series_shape(arrays, "exog", "<f4", Python.tuple(batch, n, nx), path, context) if exog else PythonObject(None))
    py_setitem(fields, "batch_size_", batch)
    py_setitem(fields, "n_obs_", n)
    py_setitem(fields, "input_copied_", classical_i64(meta, 11) != 0)
    py_setitem(fields, "n_exog_", nx)
    py_setitem(fields, "complexity_", complexity)
    var keys = Python.tuple("_exog", "_y", "params_", "x_", "x0_", "fx_", "n_iter_", "retcode_", "llf_", "aic_", "bic_")
    for i in range(len(keys)):  # driver-loop: eleven saved fitted fields
        py_setitem(fields, keys[i], loaded[keys[i]])
    return model


def host_archive_series_shape(arrays: PythonObject, name: String, dtype: String, shape: PythonObject,
                              path: PythonObject, context: PythonObject) raises -> PythonObject:
    """An exact-dtype member of `shape`, refused as the series loaders
    refused it: `{path!r} {name} has shape {shape}, not {expected}`."""
    var value = _archive_exact(arrays, name, dtype, context)
    var got = py_call(Python.import_module("builtins").tuple, value.shape)
    if not Bool(py=got == shape):
        raise Error(host_archive_where(path, context) + " " + name + " has shape " + _archive_builtin_repr(got)
                    + ", not " + _archive_builtin_repr(shape))
    return value


def host_archive_holtwinters(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    # The refusals keep ExponentialSmoothing.load's wording and order.
    var where = host_archive_where(path, context)
    var meta = host_archive_meta(arrays, "meta", 4, path, context)
    var n = classical_i64(meta, 0)
    var batch = classical_i64(meta, 1)
    var period = classical_i64(meta, 2)
    var start = classical_i64(meta, 3)
    var eps = _archive_exact(arrays, "eps", "<f8", context)
    if Int(py=eps.size) != 1:
        raise Error(where + " eps holds " + String(Int(py=eps.size)) + " values, 1 is needed")
    var initialization = _archive_string(arrays, "initialization_method", context) if py_contains(arrays, "initialization_method") else PythonObject("heuristic")
    var seasonal = _archive_string(arrays, "seasonal", context)
    # The constructor's refusals.
    if String(seasonal) != "additive" and String(seasonal) != "add" and String(seasonal) != "multiplicative" and String(seasonal) != "mul":
        raise Error("ValueError: mojolearn ExponentialSmoothing: seasonal=" + _archive_builtin_repr(seasonal)
                    + " is refused; it must be 'additive'/'add' or 'multiplicative'/'mul' (holtwinters.pyx:197)")
    if String(initialization) != "estimated" and String(initialization) != "heuristic" and String(initialization) != "cuml":
        raise Error("ValueError: mojolearn ExponentialSmoothing: initialization_method=" + _archive_builtin_repr(initialization)
                    + " is refused; it must be 'estimated' (the default), 'heuristic' or 'cuml'")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "endog", PythonObject(None))
    py_setitem(fields, "initialization_method", initialization)
    py_setitem(fields, "seasonal", seasonal)
    py_setitem(fields, "seasonal_periods", period)
    py_setitem(fields, "start_periods", start)
    py_setitem(fields, "ts_num", batch)
    py_setitem(fields, "eps", classical_f64(eps, 0))
    py_setitem(fields, "fit_executed_flag", False)
    host_archive_mode(arrays, fields, context)
    if batch < 1 or n <= period:
        raise Error(where + " records n=" + String(n) + ", ts_num=" + String(batch) + ", seasonal_periods=" + String(period))
    var rows = n - period
    var cl = rows * batch
    var comps = host_archive_series_shape(arrays, "components", "<f4", Python.tuple(3 * cl), path, context)
    var stats = Python.tuple("sse", "alpha", "beta", "gamma")
    var loaded = Python.dict()
    for i in range(len(stats)):  # driver-loop: four saved per-series fit statistics
        var name = String(stats[i])
        py_setitem(loaded, name + "_", host_archive_series_shape(arrays, name, "<f4", Python.tuple(batch), path, context))
    py_setitem(loaded, "n_iter_", host_archive_series_shape(arrays, "n_iter", "<i4", Python.tuple(batch), path, context))
    py_setitem(loaded, "criterion_", host_archive_series_shape(arrays, "criterion", "<i4", Python.tuple(batch), path, context))
    # `_set_fitted`: the packed components and their time-major blocks.
    py_setitem(fields, "n", n)
    py_setitem(fields, "_components_len", cl)
    py_setitem(fields, "_comps", comps)
    py_setitem(fields, "_comp_layout", Python.tuple(cl, batch, rows))
    var time_major = Python.dict()
    var names = Python.tuple("level", "trend", "season")
    for i in range(len(names)):  # driver-loop: three named saved component blocks
        var copy = host_archive_typed(Python.tuple(rows, batch), "<f4", context)
        host_archive_copy_f32(Int(py=classical_address(comps)) + i * cl * 4, batch * 4, 4,
                              rows, batch, Int(py=classical_address(copy)))
        _ = comps  # keep the Python object alive through the native call above
        _ = copy  # keep the Python object alive through the native call above
        py_setitem(time_major, names[i], copy)
    py_setitem(fields, "_time_major", time_major)
    var keys = Python.tuple("sse_", "alpha_", "beta_", "gamma_", "n_iter_", "criterion_")
    for i in range(len(keys)):  # driver-loop: six saved per-series fit statistics
        py_setitem(fields, keys[i], loaded[keys[i]])
    py_setitem(fields, "fit_executed_flag", True)
    return model


def host_archive_ivf(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 8, context, True)
    var n = classical_i64(meta, 0)
    var d = classical_i64(meta, 1)
    var lists = classical_i64(meta, 2)
    var metric = classical_i64(meta, 3)
    if metric != 0 and metric != 1:
        raise Error("unsupported saved IVF metric code")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "n_lists", lists)
    py_setitem(fields, "n_probes", classical_i64(meta, 4))
    py_setitem(fields, "n_neighbors", classical_i64(meta, 5))
    py_setitem(fields, "kmeans_n_iters", classical_i64(meta, 6))
    py_setitem(fields, "metric", "sqeuclidean" if metric == 0 else "euclidean")
    py_setitem(fields, "random_state", classical_i64(meta, 7))
    py_setitem(fields, "centers_", host_archive_shape(arrays, "centers", "<f4", Python.tuple(lists, d), context))
    py_setitem(fields, "center_norms_", host_archive_shape(arrays, "center_norms", "<f4", Python.tuple(lists), context))
    py_setitem(fields, "list_offsets_", host_archive_shape(arrays, "list_offsets", "<i4", Python.tuple(lists + 1), context))
    py_setitem(fields, "list_indices_", host_archive_shape(arrays, "list_indices", "<i4", Python.tuple(n), context))
    py_setitem(fields, "list_data_", host_archive_shape(arrays, "list_data", "<f4", Python.tuple(n, d), context))
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "n_rows_", n)
    py_setitem(fields, "n_lists_", lists)
    py_setitem(fields, "metric_code_", metric)
    return model


def host_archive_ann(cls: PythonObject, arrays: PythonObject, fmt: String, path: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`_AnnSaved.load` of the x_ann indexes (IVFPQIndex, CagraIndex,
    IVFSQIndex, IVFRaBitQIndex): the saved int constructor parameters and
    fitted ints, the numeric mode and the index arrays at their saved
    dtypes, with the Python loader's refusals. `host_archive_ann_layouts`
    maps the format to (parameter names, fitted names, (attribute, member,
    dtype) arrays, the other constructor defaults, the parameters saved as
    -1 for None)."""
    var b = Python.import_module("builtins")
    var layout = context["host_archive_ann_layouts"][fmt]
    var names = layout[0]
    var fitted_names = layout[1]
    var saved = layout[2]
    var defaults = layout[3]
    var none_below = layout[4]
    var params = _archive_exact(arrays, "params", "<i8", context)
    var fitted = _archive_exact(arrays, "fitted", "<i8", context)
    if Int(py=params.size) != len(names) or Int(py=fitted.size) != len(fitted_names):
        raise Error(host_archive_where(path, context) + " does not hold " + String(cls.__name__) + "'s parameters")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    var words = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(params)))
    _ = params  # keep the Python object alive through the native call above
    for i in range(len(names)):  # driver-loop: over the class's saved constructor parameters
        var value = Int(words[i])
        _ = params  # keep the Python object alive through the native call above
        if py_contains(none_below, names[i]) and value < 0:
            py_setitem(fields, names[i], PythonObject(None))
        else:
            py_setitem(fields, names[i], PythonObject(value))
    var keys = py_call(b.list, defaults)
    for i in range(len(keys)):  # driver-loop: over the class's other constructor parameters
        py_setitem(fields, keys[i], defaults[keys[i]])
    host_archive_mode(arrays, fields, context)
    var counts = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(fitted)))
    _ = fitted  # keep the Python object alive through the native call above
    for i in range(len(fitted_names)):  # driver-loop: over the class's saved fitted sizes
        py_setitem(fields, fitted_names[i], PythonObject(Int(counts[i])))
        _ = fitted  # keep the Python object alive through the native call above
    for i in range(len(saved)):  # driver-loop: over the class's saved index arrays
        var entry = saved[i]
        py_setitem(fields, entry[0], _archive_exact(arrays, String(entry[1]), String(entry[2]), context))
    _ = params^
    _ = fitted^
    return model


def host_archive_hdbscan(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """HDBSCAN.load: the refusals keep the Python loader's wording and order."""
    var where = host_archive_where(path, context)
    var meta = _archive_exact(arrays, "ints", "<i8", context)
    var reals = _archive_exact(arrays, "reals", "<f8", context)
    if Int(py=meta.size) != 11 or Int(py=reals.size) != 2:
        raise Error(where + " holds " + String(Int(py=meta.size)) + " ints and " + String(Int(py=reals.size))
                    + " reals, 11 and 2 are needed")
    var d = classical_i64(meta, 0)
    var selected = classical_i64(meta, 1)
    var condensed = classical_i64(meta, 4)
    var edges = classical_i64(meta, 5)
    var exemplars = classical_i64(meta, 10)
    var metric = _archive_string(arrays, "metric", context)
    var method = _archive_string(arrays, "cluster_selection_method", context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "min_cluster_size", classical_i64(meta, 6))
    var samples = classical_i64(meta, 7)
    py_setitem(fields, "min_samples", PythonObject(None) if samples == -1 else PythonObject(samples))
    py_setitem(fields, "cluster_selection_epsilon", classical_f64(reals, 1))
    py_setitem(fields, "max_cluster_size", classical_i64(meta, 8))
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "alpha", classical_f64(reals, 0))
    py_setitem(fields, "cluster_selection_method", method)
    py_setitem(fields, "allow_single_cluster", classical_i64(meta, 9) != 0)
    py_setitem(fields, "prediction_data", True)
    host_archive_mode(arrays, fields, context)
    var raw = _archive_exact(arrays, "raw_data", "<f4", context)
    if Int(py=raw.ndim) != 2 or Int(py=raw.shape[1]) != d or Int(py=raw.shape[0]) < 2:
        raise Error(where + " raw_data has shape " + _archive_builtin_repr(raw.shape) + ", ints say "
                    + String(d) + " features")
    var n = Int(py=raw.shape[0])
    var names = Python.tuple("core_distances", "labels", "parents", "lambdas", "deaths", "selected_clusters",
                             "index_into_children", "exemplar_idx", "exemplar_label_offsets")
    var dtypes = Python.tuple("<f4", "<i4", "<i4", "<f4", "<f4", "<i4", "<i4", "<i4", "<i4")
    var sizes = Python.tuple(n, n, edges, edges, condensed, max(selected, 1), edges + 1, max(exemplars, 1), selected + 1)
    var got = Python.dict()
    for i in range(len(names)):  # driver-loop: nine saved fitted and prediction-data arrays
        var name = String(names[i])
        var size = Int(py=sizes[i])
        var value = _archive_exact(arrays, name, String(dtypes[i]), context)
        if Int(py=value.ndim) != 1 or Int(py=value.size) != size:
            raise Error(where + " " + name + " holds " + String(Int(py=value.size)) + " values, "
                        + String(size) + " are needed")
        py_setitem(got, names[i], value)
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "labels_", got["labels"])
    py_setitem(fields, "core_distances_", got["core_distances"])
    py_setitem(fields, "n_clusters_", selected)
    py_setitem(fields, "n_outliers_", classical_i64(meta, 2))
    py_setitem(fields, "n_boruvka_rounds_", classical_i64(meta, 3))
    py_setitem(fields, "n_condensed_clusters_", condensed)
    py_setitem(fields, "_raw_data", raw)
    var prediction = Python.dict(n_edges=edges, n_condensed=condensed, n_exemplars=exemplars)
    for i in range(2, len(names)):  # driver-loop: seven saved prediction-data arrays
        py_setitem(prediction, names[i], got[names[i]])
    py_setitem(fields, "_prediction_data", prediction)
    return model


def host_archive_neighbor_metric(metric: PythonObject, p: Float64, rbc: Bool,
                                  context: PythonObject) raises -> PythonObject:
    var table = context["host_archive_rbc_metrics" if rbc else "host_archive_neighbor_metrics"]
    var key = py_call(metric.lower)
    if not py_contains(table, key):
        raise Error("saved neighbor metric is unsupported by its search algorithm")
    var code = Int(py=table[key])
    var arg = Float64(2.0)
    if code == 9:
        if not isfinite(p) or (p < 1.0 if rbc else p < 1.1754943508222875e-38):
            raise Error("saved Minkowski p is outside the supported search algorithm range")
        arg = p
    elif p != 2.0:
        raise Error("saved p is read only by the Minkowski metric")
    return Python.tuple(code, arg)


def host_archive_float_repr(value: Float64) raises -> String:
    """repr(value) of a Python float, as the loaders' f-strings print it."""
    return _archive_builtin_repr(PythonObject(value))


def host_archive_rbc_metric(cls_name: String, metric: PythonObject, p: Float64,
                            context: PythonObject) raises -> PythonObject:
    """`host_archive_rbc_metric_object` of a saved float p."""
    return host_archive_rbc_metric_object(cls_name, metric, PythonObject(p), context)


def host_archive_rbc_metric_object(cls_name: String, metric: PythonObject, p: PythonObject,
                                   context: PythonObject) raises -> PythonObject:
    """`neighbors._resolve_rbc_metric(cls_name, metric, p)` for a saved
    metric name: `(metric_value, metric_arg)`, or the same refusal, by name
    and in the same order."""
    var b = Python.import_module("builtins")
    var head = "mojolearn " + cls_name + ": "
    var shown = _archive_builtin_repr(metric)
    if py_type(metric) is not b.str:
        raise Error(head + "metric must be a name, got " + shown)
    var key = py_call(metric.lower)
    var name = String(key)
    var table = context["host_archive_rbc_metrics"]
    if name == "sqeuclidean":
        raise Error(head + "metric='sqeuclidean' is REFUSED on the "
            + "random ball cover. The pruning here IS THE TRIANGLE "
            + "INEQUALITY on the landmark radii "
            + "(neighbors/impl/ball_cover/common.mojo, DEVIATION "
            + "565), and SQUARED Euclidean distance does not satisfy it even "
            + "though Euclidean distance does: on three collinear points at "
            + "unit spacing it gives 4 against 1 + 1. A cover built on it "
            + "would prune away true neighbours silently. Ask for "
            + "metric='euclidean' and square the returned distances, which "
            + "is the same answer and is exact.")
    if name == "cosine":
        raise Error(head + "metric='cosine' is REFUSED on the "
            + "random ball cover. The pruning here IS THE TRIANGLE "
            + "INEQUALITY on the landmark radii, and `1 - cos` does not "
            + "satisfy it: three unit vectors at 0, 60 and 120 degrees give "
            + "1.5 against 0.5 + 0.5. It also fails identity of "
            + "indiscernibles, d(x, 2x) = 0, so a landmark radius stops "
            + "bounding its ball. A cover built on it would prune away true "
            + "neighbours silently rather than return them slowly. Use "
            + "NearestNeighbors with algorithm='brute', which needs no "
            + "inequality and honors every implemented metric.")
    if py_contains(context["host_archive_unsupported_metrics"], key):
        raise Error(head + "metric=" + shown + " is in cuML's "
            + "VALID_METRICS['brute'] but is NOT IMPLEMENTED "
            + "(neighbors/NOT_IMPLEMENTED.tsv)")
    if py_contains(context["host_archive_neighbor_metrics"], key) and not py_contains(table, key):
        raise Error(head + "metric=" + shown + " is computed by brute "
            + "force only (NearestNeighbors(algorithm='brute')). The random "
            + "ball cover's pruning is the triangle inequality on its "
            + "landmark radii and its kernels carry the Minkowski family; "
            + "cuML's own VALID_METRICS['rbc'] is {euclidean, haversine, l2}")
    if not py_contains(table, key):
        var admitted = py_call(PythonObject(", ").join, py_call(b.sorted, table))
        raise Error(head + "unknown metric " + shown + ". The random "
            + "ball cover admits: " + String(admitted))
    var value = Int(py=table[key])
    var arg = Float64(2.0)
    var shown_p = _archive_builtin_repr(p)
    if value != 9:
        # `_refuse_inert_p`: only metric='minkowski'/'lp' reads p; a bool,
        # or a p float() refuses, is not the default.
        var is_default = False
        if py_type(p) is not b.bool:
            try:
                is_default = Float64(py=py_call(b.float, p)) == 2.0
            except:
                is_default = False
        if not is_default:
            raise Error(head + "p is read only by metric='minkowski' "
                + "(or 'lp'); metric=" + shown + " never reads it, so p=" + shown_p + " would "
                + "change nothing. Leave p at its default (2) or ask for "
                + "metric='minkowski'.")
    else:
        arg = Float64(py=py_call(b.float, p))
        # `arg != arg or arg == float("inf")`: NaN or +infinity.
        if arg != arg or (arg > 0.0 and not isfinite(arg)):
            raise Error(head + "metric='minkowski' needs a finite "
                + "p >= 1, got " + shown_p + ". p = infinity is metric='chebyshev', "
                + "which IS admitted here.")
        if arg < 1.0:
            raise Error(head + "metric='minkowski' with p=" + shown_p + " is "
                + "REFUSED on the random ball cover. The pruning here IS THE "
                + "TRIANGLE INEQUALITY on the landmark radii "
                + "(neighbors/impl/ball_cover/common.mojo, "
                + "DEVIATION 564), and |x|^p is not subadditive below p = 1, "
                + "so Lp at p < 1 is not a metric at all: at p = 1/2 the "
                + "points (0,0), (1,0), (1,1) give a direct distance of 4 "
                + "against a two-leg path of 1 + 1. p >= 1 is admitted and "
                + "p = 1 is Manhattan. Use NearestNeighbors with "
                + "algorithm='brute' for p < 1, which is exact brute force "
                + "and needs no inequality.")
    return Python.tuple(value, arg)


def host_archive_radius_refusals(fields: PythonObject, cls_name: String, context: PythonObject) raises:
    """`RadiusNeighbors._check_refusals` over the instance fields (at `fit`
    and at `load`): the metric (which sets `_metric_value`/`_metric_arg`),
    then the algorithm, then the radius, in the Python method's wording."""
    var b = Python.import_module("builtins")
    var metric = fields["metric"]
    var algorithm = fields["algorithm"]
    var radius = fields["radius"]
    var resolved = host_archive_rbc_metric_object(cls_name, metric, fields["p"], context)
    py_setitem(fields, "_metric_value", resolved[0])
    py_setitem(fields, "_metric_arg", resolved[1])
    if not (py_type(algorithm) is b.str and (String(algorithm) == "auto" or String(algorithm) == "rbc")):
        raise Error("mojolearn RadiusNeighbors: algorithm=" + _archive_builtin_repr(algorithm) + " is "
            + "refused. The index here is cuVS's RANDOM BALL COVER, which "
            + "is none of scikit-learn's three: it is not 'brute' (there "
            + "is an index and it prunes), and it is neither 'ball_tree' "
            + "nor 'kd_tree' (it is a one-level cover over sqrt(n) "
            + "landmarks, not a tree). Naming any of them would describe "
            + "the wrong algorithm, so 'auto' and 'rbc' -- the "
            + "algorithm's own name -- are the two accepted. The "
            + "results are exact either way")
    var value = Float64(py=py_call(b.float, radius))
    if not isfinite(value) or value <= 0:
        raise Error("mojolearn RadiusNeighbors: radius=" + _archive_builtin_repr(radius) + " is "
            + "refused; it must be positive and finite")


def host_archive_radius(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """RadiusNeighbors.load: the object, attributes and refusals (wording
    and order) of the Python loader it replaced."""
    var where = host_archive_where(path, context)
    var meta = host_archive_meta(arrays, "meta", 2, path, context)
    var saved_radius = _archive_exact(arrays, "radius", "<f8", context)
    var saved_p = _archive_exact(arrays, "p", "<f8", context)
    if Int(py=saved_radius.size) != 1 or Int(py=saved_p.size) != 1:
        raise Error(where + " radius and p must hold one value each")
    var radius = classical_f64(saved_radius, 0)
    var metric = _archive_string(arrays, "metric", context)
    var algorithm = _archive_string(arrays, "algorithm", context)
    var p = classical_f64(saved_p, 0)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    # The constructor's attributes, in its order.
    py_setitem(fields, "radius", radius)
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "algorithm", algorithm)
    py_setitem(fields, "p", p)
    py_setitem(fields, "_index", PythonObject(None))
    py_setitem(fields, "_metric_value", PythonObject(None))
    py_setitem(fields, "_metric_arg", PythonObject(None))
    host_archive_mode(arrays, fields, context)
    host_archive_radius_refusals(fields, String(cls.__name__), context)
    var nf = classical_i64(meta, 0)
    var ns = classical_i64(meta, 1)
    var index = _archive_exact(arrays, "index", "<f4", context)
    if Int(py=index.ndim) != 2 or Int(py=index.shape[0]) != ns or Int(py=index.shape[1]) != nf or ns < 1 or nf < 1:
        raise Error(where + " index has shape " + _archive_builtin_repr(index.shape) + ", meta says ("
            + String(ns) + ", " + String(nf) + ")")
    py_setitem(fields, "_index", index)
    py_setitem(fields, "n_samples_fit_", ns)
    py_setitem(fields, "n_features_in_", nf)
    return model


def host_archive_neighbors(cls: PythonObject, arrays: PythonObject, kind: Int,
                           context: PythonObject) raises -> PythonObject:
    # kind: 0 nearest, 1 classifier, 2 regressor (RadiusNeighbors is
    # host_archive_radius).
    var meta = host_archive_count(arrays, "meta", "<i8", 6, context, True)
    var parameter = host_archive_count(arrays, "p", "<f8", 1, context, True)
    var p = classical_f64(parameter, 0)
    var nf = classical_i64(meta, 0)
    var n = classical_i64(meta, 1)
    var metric = _archive_string(arrays, "metric", context)
    var algorithm = _archive_string(arrays, "algorithm", context)
    var rbc = String(algorithm) == "rbc"
    if String(algorithm) != "auto" and String(algorithm) != "brute" and String(algorithm) != "rbc":
        raise Error("saved neighbors require brute, auto or rbc search")
    if rbc and (kind == 1 or kind == 2):
        raise Error("saved neighbor classifiers and regressors require brute search")
    _ = host_archive_neighbor_metric(metric, p, rbc, context)
    var weights = PythonObject(None)
    if kind == 1 or kind == 2:
        weights = _archive_string(arrays, "weights", context)
        if String(weights) != "uniform" and String(weights) != "distance":
            raise Error("saved neighbor weights must be uniform or distance")
        if String(weights) == "distance" and String(py=py_call(metric.lower)) == "inner_product":
            raise Error("distance weights require a distance rather than inner product similarity")
    if n < 1 or nf < 1:
        raise Error("saved neighbor index must be nonempty")
    var index = host_archive_shape(arrays, "index", "<f4", Python.tuple(n, nf), context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "algorithm", algorithm)
    py_setitem(fields, "p", p)
    py_setitem(fields, "_index", index)
    py_setitem(fields, "n_samples_fit_", n)
    py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "n_neighbors", classical_i64(meta, 2))
    py_setitem(fields, "query_tile", classical_i64(meta, 3))
    if kind == 0:
        return model
    py_setitem(fields, "weights", weights)
    var outputs = classical_i64(meta, 5)
    var y = host_archive_shape(arrays, "y_cols", "<i4" if kind == 1 else "<f4", Python.tuple(outputs, n), context)
    py_setitem(fields, "_y_cols", y)
    py_setitem(fields, "outputs_2d_", classical_i64(meta, 4) != 0)
    if kind == 1:
        var counts = host_archive_count(arrays, "class_counts", "<i8", outputs, context, True)
        var saved_classes = _archive_exact(arrays, "classes", "<i8", context)
        if Int(py=saved_classes.ndim) != 1:
            raise Error("saved neighbor classes must be a one-dimensional vector")
        var available = Int(py=saved_classes.size)
        var restored = Python.list()
        var off = 0
        for output in range(outputs):  # driver-loop: saved classifier output columns
            var count = classical_i64(counts, output)
            if count < 0 or count > available - off:
                raise Error("neighbor class_counts does not match saved classes")
            var unique = host_archive_unique_i32(Int(py=classical_address(y)) + output * Int(py=y.strides[0]),
                                                 Int(py=y.strides[1]), n)
            _ = y  # keep the Python object alive through the native call above
            if len(unique) != count:
                raise Error("saved neighbor classes do not match distinct labels")
            var labels = Python.list()
            for i in range(count):  # driver-loop: sorted distinct class label metadata
                var label = Int(unique[i])
                if classical_i64(saved_classes, off + i) != label:
                    raise Error("saved neighbor classes are not the sorted unique labels")
                _ = py_call(labels.append, label)
            _ = py_call(restored.append, labels)
            off += count
        if off != available:
            raise Error("neighbor class_counts does not consume saved classes")
        py_setitem(fields, "_classes_list", restored)
    return model


def host_archive_iforest(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "ints", "<i8", 11, context, True)
    var reals = host_archive_count(arrays, "reals", "<f8", 4, context, True)
    var trees = classical_i64(meta, 0)
    var samples_mode = classical_i64(meta, 1)
    var samples_int = classical_i64(meta, 2)
    var depth = classical_i64(meta, 3)
    var features_mode = classical_i64(meta, 4)
    var features_int = classical_i64(meta, 5)
    var bootstrap = classical_i64(meta, 6)
    var seed = classical_i64(meta, 7)
    var auto_contamination = classical_i64(meta, 8)
    var nf = classical_i64(meta, 9)
    var samples_frac = classical_f64(reals, 0)
    var features_frac = classical_f64(reals, 1)
    var contamination = classical_f64(reals, 2)
    if samples_mode < 0 or samples_mode > 2 or features_mode < 0 or features_mode > 1 or bootstrap < 0 or bootstrap > 1 or auto_contamination < 0 or auto_contamination > 1:
        raise Error("IsolationForest archive carries an unknown knob mode")
    if trees < 1:
        raise Error("IsolationForest n_estimators must be at least one")
    if samples_mode == 1 and samples_int <= 0:
        raise Error("IsolationForest integer max_samples must be positive")
    if samples_mode == 2 and not (samples_frac > 0.0 and samples_frac <= 1.0):
        raise Error("IsolationForest fractional max_samples must be in (0, 1]")
    if depth != -1 and depth < 1:
        raise Error("IsolationForest max_depth must be None or positive")
    if features_mode == 1 and features_int < 1:
        raise Error("IsolationForest integer max_features must be positive")
    if features_mode == 0 and not (features_frac > 0.0 and features_frac <= 1.0):
        raise Error("IsolationForest fractional max_features must be in (0, 1]")
    if auto_contamination == 0 and not (contamination > 0.0 and contamination <= 0.5):
        raise Error("IsolationForest contamination must be auto or in (0, .5]")
    if seed < 0 or seed > 4294967295:
        raise Error("IsolationForest random_state must fit uint32")
    var x = _archive_exact(arrays, "x", "<f4", context)
    if Int(py=x.ndim) != 2 or Int(py=x.shape[1]) != nf or Int(py=x.shape[0]) < 1:
        raise Error("IsolationForest saved training shape does not match its features")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "n_estimators", trees)
    py_setitem(fields, "max_samples", PythonObject("auto") if samples_mode == 0 else (PythonObject(samples_int) if samples_mode == 1 else PythonObject(samples_frac)))
    py_setitem(fields, "max_depth", PythonObject(None) if depth == -1 else PythonObject(depth))
    py_setitem(fields, "max_features", PythonObject(features_int) if features_mode == 1 else PythonObject(features_frac))
    py_setitem(fields, "bootstrap", bootstrap != 0)
    py_setitem(fields, "contamination", PythonObject("auto") if auto_contamination != 0 else PythonObject(contamination))
    py_setitem(fields, "random_state", seed)
    py_setitem(fields, "warm_start", False)
    py_setitem(fields, "n_jobs", PythonObject(None))
    py_setitem(fields, "verbose", False)
    py_setitem(fields, "output_type", PythonObject(None))
    py_setitem(fields, "_max_samples_mode", samples_mode)
    py_setitem(fields, "_max_samples_int", samples_int if samples_mode == 1 else 256)
    py_setitem(fields, "_max_samples_frac", samples_frac if samples_mode == 2 else Float64(1.0))
    py_setitem(fields, "_max_depth", depth)
    py_setitem(fields, "_max_features_mode", features_mode)
    py_setitem(fields, "_max_features_int", features_int if features_mode == 1 else 0)
    py_setitem(fields, "_max_features_frac", Float64(1.0) if features_mode == 1 else features_frac)
    py_setitem(fields, "_contamination_auto", auto_contamination != 0)
    py_setitem(fields, "_contamination", Float64(0.0) if auto_contamination != 0 else contamination)
    py_setitem(fields, "_n_estimators", trees)
    py_setitem(fields, "_bootstrap", bootstrap != 0)
    py_setitem(fields, "_seed", seed)
    py_setitem(fields, "_x", x)
    py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "offset_", classical_f64(reals, 3))
    py_setitem(fields, "max_samples_", classical_i64(meta, 10))
    return model


def host_archive_public_iforest(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                                context: PythonObject) raises -> PythonObject:
    """`IsolationForest.load(path)`: the public loader's refusals in its
    order and wording; the saved knobs pass through the constructor's checks
    again (`iforest_validate`, the native `_validate_params`), then the saved
    mode, the training matrix and the fitted scalars."""
    var b = Python.import_module("builtins")
    var where = host_archive_where(path, context)
    var ints = _archive_exact(arrays, "ints", "<i8", context)
    var reals = _archive_exact(arrays, "reals", "<f8", context)
    if Int(py=ints.size) != 11 or Int(py=reals.size) != 4:
        raise Error(where + " holds " + String(Int(py=ints.size)) + " ints and " + String(Int(py=reals.size))
                    + " reals, 11 and 4 are needed")
    var trees = classical_i64(ints, 0)
    var samples_mode = classical_i64(ints, 1)
    var samples_int = classical_i64(ints, 2)
    var depth = classical_i64(ints, 3)
    var features_mode = classical_i64(ints, 4)
    var features_int = classical_i64(ints, 5)
    var bootstrap = classical_i64(ints, 6)
    var seed = classical_i64(ints, 7)
    var auto_contamination = classical_i64(ints, 8)
    var nf = classical_i64(ints, 9)
    var resolved = classical_i64(ints, 10)
    var samples_frac = classical_f64(reals, 0)
    var features_frac = classical_f64(reals, 1)
    var contamination = classical_f64(reals, 2)
    var offset = classical_f64(reals, 3)
    if (samples_mode < 0 or samples_mode > 2 or features_mode < 0 or features_mode > 1
            or auto_contamination < 0 or auto_contamination > 1 or bootstrap < 0 or bootstrap > 1):
        raise Error(where + " carries an unknown knob mode")
    var model = host_archive_allocate(cls, context)
    var fields = sequential_fields(model)
    # What `cls(n_estimators=..., ...)` stores: the eleven parameters, then
    # the numeric mode the constructor wrapper adds.
    py_setitem(fields, "n_estimators", trees)
    py_setitem(fields, "max_samples", PythonObject("auto") if samples_mode == 0 else (PythonObject(samples_int) if samples_mode == 1 else PythonObject(samples_frac)))
    py_setitem(fields, "max_depth", PythonObject(None) if depth == -1 else PythonObject(depth))
    py_setitem(fields, "max_features", PythonObject(features_int) if features_mode == 1 else PythonObject(features_frac))
    py_setitem(fields, "bootstrap", PythonObject(bootstrap != 0))
    py_setitem(fields, "contamination", PythonObject("auto") if auto_contamination != 0 else PythonObject(contamination))
    py_setitem(fields, "random_state", seed)
    py_setitem(fields, "warm_start", PythonObject(False))
    py_setitem(fields, "n_jobs", PythonObject(None))
    py_setitem(fields, "verbose", PythonObject(False))
    py_setitem(fields, "output_type", PythonObject(None))
    py_setitem(fields, "numeric_mode", PythonObject(None))
    iforest_validate(model, fields, context)
    host_archive_mode(arrays, fields, context)
    var x = _archive_exact(arrays, "x", "<f4", context)
    if Int(py=x.ndim) != 2 or Int(py=x.shape[1]) != nf or Int(py=x.shape[0]) < 1:
        raise Error(where + " x has shape " + _archive_builtin_repr(py_call(b.tuple, x.shape)) + ", meta says "
                    + String(nf) + " features")
    py_setitem(fields, "_x", x)
    py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "offset_", offset)
    py_setitem(fields, "max_samples_", resolved)
    return model


def host_archive_spectral(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """SpectralClustering.load: the refusals keep the Python loader's
    wording and order, `_validate_params` included."""
    var where = host_archive_where(path, context)
    var meta = host_archive_meta(arrays, "meta", 8, path, context)
    var clusters = classical_i64(meta, 0)
    var k = classical_i64(meta, 1)
    var neighbors = classical_i64(meta, 2)
    var starts = classical_i64(meta, 3)
    var seed = classical_i64(meta, 4)
    var seed_none = classical_i64(meta, 5) != 0
    var n = classical_i64(meta, 6)
    var nf = classical_i64(meta, 7)
    var tolerance = host_archive_sized(arrays, "eigen_tol", "<f8", 1, where + " eigen_tol must hold one value", context)
    var tol = classical_f64(tolerance, 0)
    var affinity = _archive_string(arrays, "affinity", context)
    var name = String(affinity)
    # SpectralClustering._validate_params, on the values the file holds.
    var affinities = context["host_archive_spectral_affinities"]
    if not py_contains(affinities, affinity):
        raise Error("mojolearn SpectralClustering: affinity=" + _archive_builtin_repr(affinity)
                    + " is refused; it must be one of "
                    + _archive_builtin_repr(py_call(Python.import_module("builtins").list, affinities))
                    + ". A callable or another kernel name builds the affinity outside every identity column.")
    if name == "rbf" or name == "precomputed_nearest_neighbors":
        raise Error("NotImplementedError: mojolearn SpectralClustering: prediction_data with affinity="
                    + _archive_builtin_repr(affinity) + " is refused; the Nystrom predict (DEVIATION 2860) carries the "
                    + "'nearest_neighbors' and 'precomputed' affinities only")
    if not tol > 0.0:
        raise Error("mojolearn SpectralClustering: eigen_tol must be positive, got "
                    + _archive_builtin_repr(PythonObject(tol)) + " (DEVIATION 890; see the 'auto' note)")
    if not seed_none and (seed < 0 or seed >= 4294967296):
        raise Error("mojolearn SpectralClustering: random_state must satisfy 0 <= random_state < 2**32, got "
                    + String(seed) + " (cuML's check_random_seed refuses the same range)")
    if clusters < 1:
        raise Error("mojolearn SpectralClustering: n_clusters must be at least 1")
    if starts < 1:
        raise Error("mojolearn SpectralClustering: n_init must be at least 1")
    if neighbors < 1:
        raise Error("mojolearn SpectralClustering: n_neighbors must be at least 1")
    if k < 1:
        raise Error("mojolearn SpectralClustering: n_components must be at least 1, or None to follow n_clusters")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "n_clusters", clusters)
    py_setitem(fields, "n_components", k)
    py_setitem(fields, "random_state", PythonObject(None) if seed_none else PythonObject(seed))
    py_setitem(fields, "n_neighbors", neighbors)
    py_setitem(fields, "n_init", starts)
    py_setitem(fields, "eigen_tol", tol)
    py_setitem(fields, "affinity", affinity)
    py_setitem(fields, "gamma", PythonObject(None))
    py_setitem(fields, "assign_labels", "kmeans")
    py_setitem(fields, "prediction_data", True)
    py_setitem(fields, "eigen_solver", PythonObject(None))
    py_setitem(fields, "degree", PythonObject(None))
    py_setitem(fields, "coef0", PythonObject(None))
    py_setitem(fields, "kernel_params", PythonObject(None))
    py_setitem(fields, "n_jobs", PythonObject(None))
    py_setitem(fields, "verbose", False)
    py_setitem(fields, "_n_clusters", clusters)
    py_setitem(fields, "_n_neighbors", neighbors)
    py_setitem(fields, "_n_init", starts)
    py_setitem(fields, "_eigen_tol", tol)
    py_setitem(fields, "_gamma", PythonObject(None))
    py_setitem(fields, "_seed", 0 if seed_none else seed)
    host_archive_mode(arrays, fields, context)
    var values = _archive_exact(arrays, "eigenvalues", "<f4", context)
    var vectors = _archive_exact(arrays, "eigenvectors", "<f4", context)
    var diag = _archive_exact(arrays, "diag", "<f4", context)
    var centroids = _archive_exact(arrays, "centroids", "<f4", context)
    var labels = _archive_exact(arrays, "labels", "<i4", context)
    if (Int(py=values.size) != k or Int(py=vectors.ndim) != 2 or Int(py=vectors.shape[0]) != n
            or Int(py=vectors.shape[1]) != k or Int(py=diag.size) != n or Int(py=centroids.ndim) != 2
            or Int(py=centroids.shape[0]) != clusters or Int(py=centroids.shape[1]) != k or Int(py=labels.size) != n):
        raise Error(where + " prediction data shapes do not match n_train=" + String(n) + ", n_components="
                    + String(k) + ", n_clusters=" + String(clusters))
    py_setitem(fields, "_fit_X", PythonObject(None))
    if name == "nearest_neighbors":
        py_setitem(fields, "_fit_X", host_archive_matrix(arrays, "x", "<f4", n, nf, path, context))
        py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "_pd_eigenvalues", values)
    py_setitem(fields, "_pd_eigenvectors", vectors)
    py_setitem(fields, "_pd_diag", diag)
    py_setitem(fields, "_pd_centroids", centroids)
    py_setitem(fields, "labels_", labels)
    py_setitem(fields, "n_components_", k)
    return model


def _umap_whole(value: Int, name: String, minimum: Int, maximum: Int) raises -> Int:
    """UMAP `_integer`'s range refusal for a saved integer control."""
    if value < minimum or value > maximum:
        raise Error("ValueError: UMAP " + name + " must be in [" + String(minimum) + ", " + String(maximum) + "]")
    return value


def _umap_real(value: Float64, name: String) raises -> Float64:
    """UMAP `_scalar`: a finite float32 value, rounded once to binary32."""
    if not isfinite(value) or abs(value) > 3.4028234663852886e38:
        raise Error("ValueError: UMAP " + name + " must be a finite float32 value")
    return Float64(Float32(value))


def host_archive_umap(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    # The refusals keep UMAP.load's wording and order: the vector sizes,
    # then the constructor's validation of the saved controls, the saved
    # mode, the config round trip, then the training and embedding shapes.
    var builtins = Python.import_module("builtins")
    var where = host_archive_where(path, context)
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    var controls = _archive_exact(arrays, "controls", "<f4", context)
    if Int(py=meta.size) != 7 or Int(py=controls.size) != 6:
        raise Error(where + " holds " + String(Int(py=meta.size)) + " meta and " + String(Int(py=controls.size))
                    + " control fields; 7 and 6 are needed")
    var neighbors = classical_i64(meta, 0)
    var components = classical_i64(meta, 1)
    var epochs = classical_i64(meta, 2)
    var seed = classical_i64(meta, 3)
    var negatives = classical_i64(meta, 4)
    var automatic_epochs = classical_i64(meta, 5) != 0
    var values = controls._mv
    var min_dist = Float64(py=values[0])
    var spread = Float64(py=values[1])
    var mix = Float64(py=values[2])
    var connectivity = Float64(py=values[3])
    var rate = Float64(py=values[4])
    var repulsion = Float64(py=values[5])
    var initialization = _archive_string(arrays, "init", context)
    var extras_code = Float64(-1.0)
    var power_saved = Float64(2.0)
    var a_saved = Float64(0.0)
    var b_saved = Float64(0.0)
    if py_contains(arrays, "extras"):
        var extras = _archive_exact(arrays, "extras", "<f8", context)
        if Int(py=extras.size) != 4:
            raise Error(where + " holds " + String(Int(py=extras.size)) + " extras; 4 are needed")
        extras_code = classical_f64(extras, 0)
        power_saved = classical_f64(extras, 1)
        a_saved = classical_f64(extras, 2)
        b_saved = classical_f64(extras, 3)
    var metric = _archive_string(arrays, "metric", context)
    # int(extras[0]), as the loader's constructor arguments evaluated it.
    if extras_code != extras_code:
        raise Error("ValueError: cannot convert float NaN to integer")
    if not isfinite(extras_code):
        raise Error("OverflowError: cannot convert float infinity to integer")
    var saved_code = py_call(builtins.int, PythonObject(extras_code))
    var minkowski_kwds = Bool(py=saved_code == PythonObject(9))
    if String(initialization) == "array":
        initialization = PythonObject("spectral")
    var have_a = a_saved > 0
    var have_b = b_saved > 0
    # The constructor's validation (`UMAP._parameters`, then `_extras`).
    _ = _umap_whole(neighbors, "n_neighbors", 2, 9223372036854775807)
    _ = _umap_whole(components, "n_components", 1, 32)
    var config_epochs = 0 if automatic_epochs else _umap_whole(epochs, "n_epochs", 1, 9223372036854775807)
    _ = _umap_whole(seed, "random_state", 0, 9223372036854775807)
    min_dist = _umap_real(min_dist, "min_dist")
    spread = _umap_real(spread, "spread")
    mix = _umap_real(mix, "set_op_mix_ratio")
    connectivity = _umap_real(connectivity, "local_connectivity")
    rate = _umap_real(rate, "learning_rate")
    repulsion = _umap_real(repulsion, "repulsion_strength")
    _ = _umap_whole(negatives, "negative_sample_rate", 0, 2147483647)
    if rate <= 0:
        raise Error("ValueError: UMAP learning_rate must be positive")
    if repulsion < 0:
        raise Error("ValueError: UMAP repulsion_strength must be nonnegative")
    if not (min_dist >= 0 and min_dist <= spread) or spread <= 0:
        raise Error("ValueError: UMAP requires 0 <= min_dist <= spread and spread > 0")
    if not (mix >= 0 and mix <= 1):
        raise Error("ValueError: UMAP set_op_mix_ratio must be in [0, 1]")
    if connectivity < 0:
        raise Error("ValueError: UMAP local_connectivity must be >= 0")
    var metrics = context["host_archive_umap_metrics"]
    if not py_contains(metrics, metric):
        raise Error("ValueError: UMAP metric=" + _archive_builtin_repr(metric) + " is not supported; the kNN computes "
                    + _archive_builtin_repr(py_call(builtins.sorted, metrics))
                    + " (a callable, and every other name, is refused by name)")
    var code = Int(py=metrics[metric])
    var power = Float64(2.0)
    if code == 9:
        power = _umap_real(power_saved if minkowski_kwds else Float64(2.0), "metric_kwds['p']")
        if not power > 0:
            raise Error("ValueError: UMAP minkowski p must be positive")
    elif minkowski_kwds:
        raise Error("ValueError: UMAP metric_kwds ['p'] are not used by metric=" + _archive_builtin_repr(metric))
    if have_a != have_b:
        raise Error("ValueError: UMAP a and b must be given together")
    var a = Float64(0.0)
    var b = Float64(0.0)
    if have_a:
        a = _umap_real(a_saved, "a")
        b = _umap_real(b_saved, "b")
        if not (a > 0 and b > 0):
            raise Error("ValueError: UMAP a and b must be positive")
    var init_text = String(initialization)
    if init_text != "spectral" and init_text != "random" and init_text != "pca":
        raise Error("ValueError: UMAP init=" + _archive_builtin_repr(initialization) + " must be 'spectral', 'random', 'pca' or an array")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    var config = Python.tuple(neighbors, components, config_epochs, min_dist, spread, mix, connectivity, seed,
                              rate, repulsion, negatives, code, power, a, b)
    var saved = Python.tuple(neighbors, components, epochs, Float64(py=values[0]), Float64(py=values[1]),
                             Float64(py=values[2]), Float64(py=values[3]), seed, Float64(py=values[4]),
                             Float64(py=values[5]), negatives, saved_code, power_saved, a_saved, b_saved)
    if Bool(py=config != saved):
        raise Error(where + " controls " + _archive_builtin_repr(saved) + " do not survive validation as "
                    + _archive_builtin_repr(config))
    var training = _archive_exact(arrays, "training", "<f4", context)
    var embedding = _archive_exact(arrays, "embedding", "<f4", context)
    if Int(py=training.ndim) != 2 or Int(py=embedding.ndim) != 2 or Int(py=embedding.shape[0]) != Int(py=training.shape[0]) or Int(py=embedding.shape[1]) != components:
        raise Error(where + " training " + _archive_builtin_repr(py_call(builtins.tuple, training.shape)) + " and embedding "
                    + _archive_builtin_repr(py_call(builtins.tuple, embedding.shape)) + " do not agree with n_components="
                    + String(components))
    py_setitem(fields, "n_neighbors", neighbors)
    py_setitem(fields, "n_components", components)
    py_setitem(fields, "min_dist", Float64(py=values[0]))
    py_setitem(fields, "spread", Float64(py=values[1]))
    py_setitem(fields, "n_epochs", PythonObject(None) if automatic_epochs else PythonObject(epochs))
    py_setitem(fields, "random_state", seed)
    py_setitem(fields, "set_op_mix_ratio", Float64(py=values[2]))
    py_setitem(fields, "local_connectivity", Float64(py=values[3]))
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "init", initialization)
    py_setitem(fields, "learning_rate", Float64(py=values[4]))
    py_setitem(fields, "repulsion_strength", Float64(py=values[5]))
    py_setitem(fields, "negative_sample_rate", negatives)
    py_setitem(fields, "a", PythonObject(a_saved) if have_a else PythonObject(None))
    py_setitem(fields, "b", PythonObject(b_saved) if have_b else PythonObject(None))
    py_setitem(fields, "metric_kwds", Python.dict(p=power_saved) if minkowski_kwds else PythonObject(None))
    py_setitem(fields, "target_n_neighbors", -1)
    py_setitem(fields, "target_metric", "categorical")
    py_setitem(fields, "target_weight", 0.5)
    py_setitem(fields, "densmap", False)
    py_setitem(fields, "output_metric", "euclidean")
    py_setitem(fields, "_transform_training", training)
    py_setitem(fields, "_transform_embedding", embedding)
    py_setitem(fields, "_transform_config", config)
    py_setitem(fields, "_transform_mode", fields["numeric_mode"])
    var store = py_call(context["array_type"], "f")
    _ = py_call(store.frombytes, py_call(embedding._mv.cast, "B"))
    py_setitem(fields, "embedding_", classical_wrap_array(store, embedding.shape, "<f4", String(embedding.order), context))
    py_setitem(fields, "n_features_in_", Int(py=training.shape[1]))
    py_setitem(fields, "input_copied_", classical_i64(meta, 6) != 0)
    return model


def host_archive_reshape_f32(array: PythonObject, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    return host_archive_reshape_words(array, shape, "<f4", context)


def host_archive_reshape_words(array: PythonObject, shape: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    if String(array.order) == "C":
        return classical_wrap_array(array._store, shape, dtype, "C", context)
    var dimensions = List[Int]()
    var strides = List[Int]()
    for i in range(Int(py=array.ndim)):  # driver-loop: saved array shape metadata
        dimensions.append(Int(py=array.shape[i]))
        strides.append(Int(py=array.strides[i]))
    var result = host_archive_typed(shape, dtype, context)
    host_archive_reorder_f32(Int(py=classical_address(array)), dimensions, strides, Int(py=array.size), Int(py=classical_address(result)))
    _ = result  # keep the Python object alive through the native call above
    return result


def host_archive_gp_row(array: PythonObject, row: Int, cols: Int, shape: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    var result = host_archive_typed(shape, "<f4", context)
    host_archive_copy_f32(Int(py=classical_address(array)) + row * Int(py=array.strides[0]), 0,
                          Int(py=array.strides[1]), 1, cols, Int(py=classical_address(result)))
    _ = result  # keep the Python object alive through the native call above
    return result


def host_archive_gpr(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    # The refusals keep GaussianProcessRegressor.load's wording and order; the
    # fields follow the constructor's, _validate_params's, then the loader's
    # assignments.
    var where = host_archive_where(path, context)
    var ints = _archive_exact(arrays, "ints", "<i8", context)
    var reals = _archive_exact(arrays, "reals", "<f8", context)
    if Int(py=ints.size) != 4 or Int(py=reals.size) != 4:
        raise Error(where + " holds " + String(Int(py=ints.size)) + " ints and " + String(Int(py=reals.size))
                    + " reals, 4 and 4 are needed")
    var d = classical_i64(ints, 0)
    var info = classical_i64(ints, 1)
    var norm = classical_i64(ints, 2) != 0
    var n_ls = classical_i64(ints, 3)
    var x = _archive_exact(arrays, "X_train", "<f4", context)
    if Int(py=x.ndim) != 2 or Int(py=x.shape[1]) != d or Int(py=x.shape[0]) < 1:
        raise Error(where + " X_train has shape " + _archive_builtin_repr(x.shape) + ", ints say " + String(d) + " features")
    var n = Int(py=x.shape[0])
    var factor = _archive_exact(arrays, "L", "<f4", context)
    var dual = _archive_exact(arrays, "alpha", "<f4", context)
    if Int(py=factor.size) != n * n or Int(py=dual.ndim) != 1 or Int(py=dual.size) != n:
        raise Error(where + " L or alpha does not match " + String(n) + " training rows")
    var kinds = _archive_exact(arrays, "kinds", "<i4", context)
    var params = _archive_exact(arrays, "params", "<f8", context)
    var lengths = _archive_exact(arrays, "ls_len", "<i4", context)
    var scales = _archive_exact(arrays, "ls", "<f8", context)
    var count = Int(py=kinds.size)
    if count != Int(py=params.size) or count != Int(py=lengths.size) or count < 1:
        raise Error(where + " carries a malformed kernel spec")
    var kind_list = py_call(kinds._mv.tolist)
    var param_list = py_call(params._mv.tolist)
    var length_list = py_call(lengths._mv.tolist)
    var table = py_call(scales._mv.tolist)
    var total = 0
    for i in range(count):  # driver-loop: saved postfix kernel node metadata
        total += Int(py=length_list[i])
    if total != n_ls or Int(py=scales.size) != max(n_ls, 1):
        raise Error(where + " length scale table does not match its node lengths")
    var nodes = Python.list()
    var offset = 0
    for i in range(count):  # driver-loop: saved postfix kernel node metadata
        var length = Int(py=length_list[i])
        var node_scales = Python.list()
        for j in range(length):  # driver-loop: saved kernel node parameter metadata
            _ = py_call(node_scales.append, table[offset + j])
        _ = py_call(nodes.append, Python.tuple(kind_list[i], param_list[i], node_scales))
        offset += length
    var kernel = host_archive_allocate(context["host_archive_gp_saved_kernel"], context)
    py_setitem(sequential_fields(kernel), "_SavedKernel__nodes", nodes)
    var alpha = classical_f64(reals, 0)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "kernel", kernel)
    py_setitem(fields, "alpha", alpha)
    py_setitem(fields, "optimizer", PythonObject(None))
    py_setitem(fields, "n_restarts_optimizer", 0)
    py_setitem(fields, "normalize_y", norm)
    py_setitem(fields, "copy_X_train", True)
    py_setitem(fields, "n_targets", PythonObject(None))
    py_setitem(fields, "random_state", PythonObject(None))
    py_setitem(fields, "numeric_mode", PythonObject(None))
    py_setitem(fields, "_kernel", kernel)
    py_setitem(fields, "_alpha", alpha)
    py_setitem(fields, "_optimizer", PythonObject(None))
    py_setitem(fields, "_n_restarts_optimizer", 0)
    py_setitem(fields, "_normalize_y", norm)
    py_setitem(fields, "_random_state", PythonObject(None))
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "X_train_", x)
    py_setitem(fields, "kernel_", kernel)
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "L_", host_archive_reshape_f32(factor, Python.tuple(n, n), context))
    py_setitem(fields, "alpha_", dual)
    py_setitem(fields, "info_", info)
    py_setitem(fields, "normalize_y_", norm)
    py_setitem(fields, "_y_train_mean", classical_f64(reals, 1))
    py_setitem(fields, "_y_train_std", classical_f64(reals, 2))
    py_setitem(fields, "log_marginal_likelihood_value_", classical_f64(reals, 3))
    return model


def host_archive_gpc_rows(arrays: PythonObject, name: String, dtype: String, count: Int, width: Int,
                          path: PythonObject, context: PythonObject) raises -> PythonObject:
    """GaussianProcessClassifier.load's `rows`: a (count, width) block, or a
    (count,) vector when width is 0, refused as `{path!r} {name} has shape {shape}`."""
    var value = _archive_exact(arrays, name, dtype, context)
    var shape = Python.tuple(count, width) if width != 0 else Python.tuple(count)
    if Bool(py=value.shape != shape):
        raise Error(host_archive_where(path, context) + " " + name + " has shape " + _archive_builtin_repr(value.shape))
    return value


def host_archive_gpc(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    # The refusals keep GaussianProcessClassifier.load's wording and order;
    # the fields follow the constructor's, _validate_params's, then the
    # loader's and _set_fitted's assignments.
    var where = host_archive_where(path, context)
    var mode = Python.dict()
    host_archive_mode(arrays, mode, context)
    var meta = host_archive_meta(arrays, "meta", 6, path, context)
    var n_train = classical_i64(meta, 0)
    var n_features = classical_i64(meta, 1)
    var n_est = classical_i64(meta, 2)
    var max_iter = classical_i64(meta, 3)
    var n_ls = classical_i64(meta, 4)
    var n_classes = classical_i64(meta, 5)
    var kind_list = py_call(_archive_exact(arrays, "kinds", "<i4", context)._mv.tolist)
    var param_list = py_call(_archive_exact(arrays, "kparams", "<f4", context)._mv.tolist)
    var length_list = py_call(_archive_exact(arrays, "ls_len", "<i4", context)._mv.tolist)
    var table = py_call(_archive_exact(arrays, "ls", "<f4", context)._mv.tolist)
    var count = len(kind_list)
    var total = 0
    for i in range(len(length_list)):  # driver-loop: saved postfix kernel node metadata
        total += Int(py=length_list[i])
    if len(param_list) != count or len(length_list) != count or total != n_ls:
        raise Error(where + " kernel arrays disagree with each other")
    var b = Python.import_module("builtins")
    var nodes = Python.list()
    var offset = 0
    for i in range(count):  # driver-loop: saved postfix kernel node metadata
        var length = Int(py=length_list[i])
        # table[offset:offset + length], Python's slice of the saved table.
        var part = table[py_call(b.slice, offset, offset + length)]
        var node_scales = Python.list()
        for j in range(len(part)):  # driver-loop: saved kernel node parameter metadata
            _ = py_call(node_scales.append, py_call(b.float, part[j]))
        _ = py_call(nodes.append, Python.tuple(py_call(b.int, kind_list[i]), py_call(b.float, param_list[i]), node_scales))
        offset += length
    var kernel = host_archive_allocate(context["host_archive_gpc_saved_kernel"], context)
    py_setitem(sequential_fields(kernel), "_node_list", nodes)
    if max_iter < 1:
        raise Error("mojolearn GaussianProcessClassifier: max_iter_predict must be at least 1, got " + String(max_iter))
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "kernel", kernel)
    py_setitem(fields, "optimizer", PythonObject(None))
    py_setitem(fields, "n_restarts_optimizer", 0)
    py_setitem(fields, "max_iter_predict", max_iter)
    py_setitem(fields, "warm_start", False)
    py_setitem(fields, "copy_X_train", True)
    py_setitem(fields, "random_state", PythonObject(None))
    py_setitem(fields, "multi_class", "one_vs_rest")
    py_setitem(fields, "n_jobs", PythonObject(None))
    py_setitem(fields, "numeric_mode", mode["numeric_mode"])
    py_setitem(fields, "_kernel", kernel)
    py_setitem(fields, "_max_iter_predict", max_iter)
    var x = _archive_exact(arrays, "x", "<f4", context)
    if Bool(py=x.shape != Python.tuple(n_train, n_features)):
        raise Error(where + " x has shape " + _archive_builtin_repr(x.shape) + ", meta says ("
                    + String(n_train) + ", " + String(n_features) + ")")
    if not py_contains(arrays, "classes"):
        raise Error(where + " has no classes member")
    var classes = host_archive_class_values(arrays["classes"], context)
    var want = 1 if n_classes == 2 else n_classes
    if len(classes) != n_classes or n_classes < 2 or n_est != want:
        raise Error(where + " holds " + String(len(classes)) + " classes and " + String(n_est)
                    + " binary fits for n_classes " + String(n_classes))
    var n = n_train
    var ys = host_archive_gpc_rows(arrays, "y", "<f4", n_est, n, path, context)
    var factors = host_archive_gpc_rows(arrays, "L", "<f4", n_est, n * n, path, context)
    var pis = host_archive_gpc_rows(arrays, "pi", "<f4", n_est, n, path, context)
    var wsrs = host_archive_gpc_rows(arrays, "wsr", "<f4", n_est, n, path, context)
    var likelihoods = host_archive_gpc_rows(arrays, "lml", "<f8", n_est, 0, path, context)
    var iterations = host_archive_gpc_rows(arrays, "n_iter", "<i8", n_est, 0, path, context)
    var panels = host_archive_gpc_rows(arrays, "nb", "<i8", n_est, 0, path, context)
    var fits = Python.list()
    var counts = Python.list()
    for i in range(n_est):  # driver-loop: saved per-class Laplace state reconstruction
        var fit = host_archive_allocate(context["host_archive_binary_laplace"], context)
        var state = sequential_fields(fit)
        py_setitem(state, "y_train_", host_archive_gp_row(ys, i, n, Python.tuple(n), context))
        py_setitem(state, "L_", host_archive_gp_row(factors, i, n * n, Python.tuple(n, n), context))
        py_setitem(state, "pi_", host_archive_gp_row(pis, i, n, Python.tuple(n), context))
        py_setitem(state, "W_sr_", host_archive_gp_row(wsrs, i, n, Python.tuple(n), context))
        py_setitem(state, "log_marginal_likelihood_value_", classical_f64(likelihoods, i))
        py_setitem(state, "n_iter_", classical_i64(iterations, i))
        py_setitem(state, "nb_", classical_i64(panels, i))
        _ = py_call(fits.append, fit)
        _ = py_call(counts.append, classical_i64(iterations, i))
    py_setitem(fields, "X_train_", x)
    py_setitem(fields, "n_features_in_", n_features)
    py_setitem(fields, "classes_", classes)
    py_setitem(fields, "n_classes_", n_classes)
    py_setitem(fields, "estimators_", fits)
    py_setitem(fields, "kernel_", kernel)
    if n_classes == 2:
        var only = sequential_fields(fits[0])
        var aliases = Python.tuple("y_train_", "L_", "pi_", "W_sr_", "n_iter_", "log_marginal_likelihood_value_")
        for j in range(len(aliases)):  # driver-loop: public binary fit aliases
            py_setitem(fields, aliases[j], only[aliases[j]])
    else:
        py_setitem(fields, "n_iter_", counts)
        py_setitem(fields, "log_marginal_likelihood_value_", host_archive_mean_f64(Int(py=classical_address(likelihoods)), n_est))
        _ = likelihoods  # keep the Python object alive through the native call above
    return model


def host_archive_slice_words(array: PythonObject, offset: Int, count: Int, shape: PythonObject,
                             dtype: String, context: PythonObject) raises -> PythonObject:
    if offset < 0 or count < 0 or offset > Int(py=array.size) or count > Int(py=array.size) - offset:
        raise Error("saved model block lies outside its archive array")
    var result = host_archive_typed(shape, dtype, context)
    host_archive_copy_words(Int(py=classical_address(array)) + offset * 4, count, Int(py=classical_address(result)))
    _ = result  # keep the Python object alive through the native call above
    return result


def host_archive_support_check(support: PythonObject, count: Int, columns: Int, path: PythonObject,
                               context: PythonObject) raises:
    """`_svm_impl._check_support_bounds`: a saved kernel='precomputed'
    support index must name one of the model's `columns` training columns;
    the first one outside is named."""
    var words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(support)))
    for i in range(count):  # driver-loop: load-time validation of the saved support indices
        var row = Int(words[i])
        if row < 0 or row >= columns:
            raise Error(host_archive_where(path, context) + " kernel='precomputed' support index " + String(row)
                        + " is outside the model's " + String(columns) + " columns")


def host_archive_svm_check(fields: PythonObject, classifier: Bool, context: PythonObject) raises:
    """`SVC._validate_params` / `SVR._validate_params` on the constructor
    values a load restored: the same refusals, in the same order, then the
    checked `_hp_*` working values."""
    var b = Python.import_module("builtins")
    var head = String("mojolearn SVC: ") if classifier else String("mojolearn SVR: ")
    var kernels = context["host_archive_svm_kernels"]
    var refused = context["host_archive_svm_refused" if classifier else "host_archive_svr_refused"]
    var kernel = fields["kernel"]
    if py_type(kernel) is not b.str:
        raise Error(head + "kernel is a name")
    var lower = py_call(kernel.lower)
    var k = kernel if Bool(py=kernel == lower) else lower
    var name = String(k)
    if py_contains(refused, k):
        raise Error("NotImplementedError: " + head + "kernel=" + _archive_builtin_repr(kernel) + " is refused; "
                    + String(refused[k]))
    if not py_contains(kernels, k):
        raise Error(head + "kernel=" + _archive_builtin_repr(kernel) + " is not a kernel name; "
                    + "this implementation carries " + _archive_builtin_repr(py_call(b.sorted, kernels)) + " and refuses "
                    + "the others by name (" + _archive_builtin_repr(py_call(b.sorted, refused)) + ")")
    var gamma = fields["gamma"]
    if py_type(gamma) is b.str:
        var folded = py_call(gamma.lower)
        var g = gamma if Bool(py=gamma == folded) else folded
        if String(g) != "auto" and String(g) != "scale":
            raise Error(head + "gamma=" + _archive_builtin_repr(gamma) + " is not a name; it is "
                        + "'auto', 'scale' or a float")
        gamma = g
    else:
        var value = Float64(py=gamma)
        gamma = PythonObject(value)
        if not isfinite(value) or value < 0.0:
            raise Error(head + "gamma must be finite and >= 0 for the RBF "
                        + "kernel, got " + host_archive_float_repr(value) + " (DEVIATION 636; scikit-learn's own "
                        + "constraint is gamma >= 0)")
    var degree = fields["degree"]
    var coef0 = fields["coef0"]
    var shown_degree = _archive_builtin_repr(degree)
    var polynomial = name == "poly"
    var tanh = name == "sigmoid"
    if polynomial:
        degree = py_call(b.int, degree)
        if Bool(py=degree < 0) or Bool(py=degree > 32):
            raise Error(head + "degree must be in [0, 32], got " + _archive_builtin_repr(degree) + " (DEVIATION 1663)")
    elif tanh and classifier:
        if Bool(py=degree != 3):
            raise Error("NotImplementedError: " + head + "degree=" + shown_degree + " is refused with kernel='sigmoid'; "
                        + "it is read only by kernel='poly'")
    elif Bool(py=degree != 3):
        raise Error("NotImplementedError: " + head + "degree=" + shown_degree + " is refused with kernel="
                    + _archive_builtin_repr(k) + "; it is read only by kernel='poly'. Passing it with another kernel "
                    + "would be a parameter accepted and ignored")
    if polynomial or tanh:
        var real = Float64(py=coef0)
        coef0 = PythonObject(real)
        if not isfinite(real):
            raise Error(head + "coef0 must be finite, got " + host_archive_float_repr(real) + " (DEVIATION 636)")
    elif Bool(py=coef0 != 0.0):
        raise Error("NotImplementedError: " + head + "coef0=" + _archive_builtin_repr(coef0) + " is refused with kernel="
                    + _archive_builtin_repr(k) + "; it is read only by kernel='poly' and kernel='sigmoid'")
    var penalty = Float64(py=fields["C"])
    if not isfinite(penalty):
        raise Error(head + "C must be finite, got " + host_archive_float_repr(penalty) + " (DEVIATION 636)")
    if penalty <= 0.0:
        raise Error(head + "C must be positive, got " + host_archive_float_repr(penalty))
    var tolerance = Float64(py=fields["tol"])
    if not isfinite(tolerance):
        raise Error(head + "tol must be finite, got " + host_archive_float_repr(tolerance) + " (DEVIATION 636)")
    if tolerance <= 0.0:
        raise Error(head + "tol must be positive, got " + host_archive_float_repr(tolerance))
    var epsilon = Float64(0.0)
    if not classifier:
        # Epsilon is converted, not refused: the Mojo fit's own refusals
        # stay the ones a caller reaches.
        epsilon = Float64(py=fields["epsilon"])
    var cache = Float64(py=fields["cache_size"])
    if not (cache > 0.0) or not isfinite(cache):
        raise Error(head + "cache_size is the prediction buffer here "
                    + "(DEVIATION 871) and must be a positive finite MiB, got " + host_archive_float_repr(cache))
    var maximum = Int(py=fields["max_iter"])
    if maximum == 0 or maximum < -1:
        raise Error(head + "max_iter is a positive cap or -1 for no "
                    + "limit (cuML's default), got " + String(maximum))
    var unchanged = Int(py=fields["nochange_steps"])
    if unchanged < 0:
        raise Error(head + "nochange_steps cannot be negative, got " + String(unchanged))
    if classifier:
        py_setitem(fields, "_hp_C", penalty)
    py_setitem(fields, "_hp_kernel", k)
    py_setitem(fields, "_hp_degree", degree)
    py_setitem(fields, "_hp_gamma", gamma)
    py_setitem(fields, "_hp_coef0", coef0)
    py_setitem(fields, "_hp_tol", tolerance)
    if not classifier:
        py_setitem(fields, "_hp_C", penalty)
        py_setitem(fields, "_hp_epsilon", epsilon)
    py_setitem(fields, "_hp_cache_size", cache)
    py_setitem(fields, "_hp_max_iter", maximum)
    py_setitem(fields, "_hp_nochange_steps", unchanged)
    if classifier:
        py_setitem(fields, "_hp_random_state", PythonObject(None))
        py_setitem(fields, "_hp_probability", False)
        py_setitem(fields, "_hp_break_ties", False)


def host_archive_svc_layout(support: PythonObject, dual: PythonObject, meta: PythonObject, k: Int, rows: Int,
                            cap: Int, out_support: PythonObject, out_counts: PythonObject,
                            out_dual: PythonObject) raises -> Int:
    """scikit-learn's one-vs-one attributes from the saved pair machines,
    the rules of `svm/host/svc_ovo_host.mojo::ovo_layout_host` (the layout
    `SVC._set_ovo` runs at fit): a support row's class is its pair side (i
    when its dual is negative, j otherwise); `support_` is class-major,
    ascending row within a class; `dual_coef_[j - 1 if side == i else i][col]
    = -d`, every other cell 0.0. Integer placement only. Returns n_SV."""
    var sp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(support)))
    var dp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(dual)))
    var support_words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out_support)))
    var count_words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out_counts)))
    var dual_words = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out_dual)))
    var side_of = List[Int](length=max(1, rows), fill=-1)
    var offset = 0
    var q = 0
    for i in range(k):  # driver-loop: saved one-vs-one class machines
        for j in range(i + 1, k):  # driver-loop: saved one-vs-one class machines
            var m = classical_i64(meta, 4 * q + 2)
            for e in range(m):  # driver-loop: load-time layout of one saved pair's support rows
                var r = Int(sp[offset + e])
                if r < 0 or r >= rows:
                    raise Error("saved SVC support row is outside its layout bound")
                side_of[r] = i if dp[offset + e] < Float32(0.0) else j
            offset += m
            q += 1
    var column_of = List[Int](length=max(1, rows), fill=-1)
    var n_sv = 0
    for c in range(k):  # driver-loop: saved classes, class-major support order
        var start = n_sv
        for r in range(rows):  # driver-loop: load-time layout over the saved support row bound
            if side_of[r] == c:
                if n_sv >= cap:
                    raise Error("saved SVC support rows exceed their layout bound")
                column_of[r] = n_sv
                support_words[n_sv] = Int32(r)
                n_sv += 1
        count_words[c] = Int32(n_sv - start)
    for t in range((k - 1) * n_sv):  # driver-loop: load-time zero fill of the saved dual layout
        dual_words[t] = Float32(0.0)
    offset = 0
    q = 0
    for i in range(k):  # driver-loop: saved one-vs-one class machines
        for j in range(i + 1, k):  # driver-loop: saved one-vs-one class machines
            var m = classical_i64(meta, 4 * q + 2)
            for e in range(m):  # driver-loop: load-time layout of one saved pair's dual coefficients
                var d = dp[offset + e]
                var row = (j - 1) if d < Float32(0.0) else i
                dual_words[row * n_sv + column_of[Int(sp[offset + e])]] = -d
            offset += m
            q += 1
    return n_sv


def host_archive_svm_ovo(fields: PythonObject, arrays: PythonObject, features: Int, path: PythonObject,
                        context: PythonObject) raises:
    """`SVC._load_ovo`: the per-pair machines, then the public attributes
    `_set_ovo` builds, with its refusals and messages."""
    var where = host_archive_where(path, context)
    var classes = host_archive_class_values(arrays["classes"], context)
    var k = len(classes)
    var pair_count = k * (k - 1) // 2
    var meta = _archive_exact(arrays, "pair_meta", "<i8", context)
    if k < 3 or Int(py=meta.size) != 4 * pair_count:
        raise Error(where + " pair_meta does not hold " + String(pair_count) + " pairs")
    var bias = _archive_exact(arrays, "pair_b", "<f4", context)
    var raw_support = _archive_exact(arrays, "pair_support", "<i4", context)
    var support = host_archive_reshape_words(raw_support, Python.tuple(Int(py=raw_support.size)), "<i4", context)
    _ = raw_support  # keep the Python object alive through the native call above
    if String(fields["_hp_kernel"]) == "precomputed":
        host_archive_support_check(support, Int(py=raw_support.size), features, path, context)
    var raw_dual = _archive_exact(arrays, "pair_dual", "<f4", context)
    var raw_vectors = _archive_exact(arrays, "pair_sv", "<f4", context)
    var total = 0
    for q in range(pair_count):  # driver-loop: saved one-vs-one class machines
        total += classical_i64(meta, 4 * q + 2)
    if Int(py=bias.size) != pair_count or Int(py=raw_support.size) != total or Int(py=raw_dual.size) != total or Int(py=raw_vectors.size) != total * features:
        raise Error(where + " pair arrays do not match pair_meta")
    var dual = host_archive_reshape_f32(raw_dual, Python.tuple(total), context)
    var vectors = host_archive_reshape_f32(raw_vectors, Python.tuple(total, features), context)
    var bias_flat = host_archive_reshape_f32(bias, Python.tuple(pair_count), context)
    var pairs = Python.list()
    var out_bias = host_archive_typed(Python.tuple(pair_count), "<f4", context)
    var out_iterations = host_archive_typed(Python.tuple(pair_count), "<i4", context)
    var bias_words = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out_bias)))
    _ = out_bias  # keep the Python object alive through the native call above
    var saved_bias = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(bias_flat)))
    _ = bias_flat  # keep the Python object alive through the native call above
    var iteration_words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out_iterations)))
    _ = out_iterations  # keep the Python object alive through the native call above
    var offset = 0
    var q = 0
    for i in range(k):  # driver-loop: saved one-vs-one class machines
        for j in range(i + 1, k):  # driver-loop: saved one-vs-one class machines
            var pi = classical_i64(meta, 4 * q)
            var pj = classical_i64(meta, 4 * q + 1)
            var n = classical_i64(meta, 4 * q + 2)
            var iterations = classical_i64(meta, 4 * q + 3)
            if pi != i or pj != j:
                raise Error(where + " pair " + String(q) + " is (" + String(pi) + ", " + String(pj) + "), not ("
                            + String(i) + ", " + String(j) + ")")
            if n < 0 or n > total - offset:
                raise Error(where + " pair arrays do not match pair_meta")
            if iterations < -2147483648 or iterations > 2147483647:
                raise Error("OverflowError: saved SVC pair iteration count does not fit int32")
            var pair = Python.dict()
            var pd = host_archive_slice_words(dual, offset, n, Python.tuple(1, n), "<f4", context)
            _ = dual  # keep the Python object alive through the native call above
            var pv = host_archive_slice_words(vectors, offset * features, n * features, Python.tuple(n, features), "<f4", context)
            _ = vectors  # keep the Python object alive through the native call above
            var ps = host_archive_slice_words(support, offset, n, Python.tuple(n), "<i4", context)
            _ = support  # keep the Python object alive through the native call above
            _ = raw_support  # keep the Python object alive through the native call above
            var bias_value = Float64(saved_bias[q])
            _ = bias_flat  # keep the Python object alive through the native call above
            py_setitem(pair, "i", i)
            py_setitem(pair, "j", j)
            py_setitem(pair, "dual", pd)
            _ = dual  # keep the Python object alive through the native call above
            py_setitem(pair, "sv", pv)
            _ = vectors  # keep the Python object alive through the native call above
            py_setitem(pair, "support", ps)
            _ = support  # keep the Python object alive through the native call above
            py_setitem(pair, "b", bias_value)
            py_setitem(pair, "n_iter", iterations)
            _ = py_call(pairs.append, pair)
            bias_words[q] = -Float32(bias_value)
            _ = out_bias  # keep the Python object alive through the native call above
            iteration_words[q] = Int32(iterations)
            _ = out_iterations  # keep the Python object alive through the native call above
            offset += n
            q += 1
    var shape = String(_archive_string(arrays, "shape", context))
    var ties = _archive_exact(arrays, "break_ties", "<i8", context)
    if Int(py=ties.size) < 1:
        raise Error("mojolearn: " + _archive_read_repr(path, context) + " break_ties holds no value")
    var ovr = shape != "ovo"
    py_setitem(fields, "decision_function_shape", "ovr" if ovr else "ovo")
    py_setitem(fields, "break_ties", classical_i64(ties, 0) != 0 and ovr)
    py_setitem(fields, "_hp_break_ties", fields["break_ties"])
    # `_set_ovo`: the layout `fit` builds, with the row bound max(support) + 1.
    var rows = host_archive_support_bound(Int(py=classical_address(support)), total, -1)
    _ = support  # keep the Python object alive through the native call above
    _ = raw_support  # keep the Python object alive through the native call above
    var cap = min(rows, total)
    var public_support = host_archive_typed(Python.tuple(max(1, cap)), "<i4", context)
    var public_counts = host_archive_typed(Python.tuple(k), "<i4", context)
    var public_dual = host_archive_typed(Python.tuple(max(1, (k - 1) * cap)), "<f4", context)
    var n_sv = host_archive_svc_layout(support, dual, meta, k, rows, cap, public_support, public_counts, public_dual)
    _ = raw_support  # keep the Python object alive through the native call above
    if n_sv < 0 or n_sv > cap:
        raise Error("saved SVC native layout returned an invalid row count")
    var result_support = host_archive_slice_words(public_support, 0, n_sv, Python.tuple(n_sv), "<i4", context)
    _ = public_support  # keep the Python object alive through the native call above
    var result_dual = host_archive_slice_words(public_dual, 0, (k - 1) * n_sv, Python.tuple(k - 1, n_sv), "<f4", context)
    _ = public_dual  # keep the Python object alive through the native call above
    var result_vectors = host_archive_typed(Python.tuple(n_sv, features), "<f4", context)
    host_archive_svc_gather(Int(py=classical_address(support)), total, Int(py=classical_address(vectors)), features,
                            Int(py=classical_address(result_support)), n_sv, Int(py=classical_address(result_vectors)))
    _ = vectors  # keep the Python object alive through the native call above
    _ = support  # keep the Python object alive through the native call above
    _ = raw_support  # keep the Python object alive through the native call above
    _ = result_vectors  # keep the Python object alive through the native call above
    _ = result_support  # keep the Python object alive through the native call above
    _ = public_support  # keep the Python object alive through the native call above
    py_setitem(fields, "classes_", classes)
    py_setitem(fields, "n_features_in_", features)
    py_setitem(fields, "_pairs", pairs)
    py_setitem(fields, "support_", result_support)
    _ = public_support  # keep the Python object alive through the native call above
    py_setitem(fields, "support_vectors_", result_vectors)
    py_setitem(fields, "dual_coef_", result_dual)
    _ = public_dual  # keep the Python object alive through the native call above
    py_setitem(fields, "intercept_", out_bias)
    py_setitem(fields, "n_support_", public_counts)
    py_setitem(fields, "n_iter_", out_iterations)


def host_archive_svm(cls: PythonObject, arrays: PythonObject, classifier: Bool, path: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """SVC.load / SVR.load: the object, attributes and refusals (wording
    and order) of the Python loaders they replaced."""
    var b = Python.import_module("builtins")
    var where = host_archive_where(path, context)
    var meta = host_archive_meta(arrays, "meta", 6 if classifier else 5, path, context)
    var hyper = host_archive_held(arrays, "hyper", "<f8", 4 if classifier else 5, path, context)
    var gamma_setting = _archive_string(arrays, "gamma", context)
    var kernel = _archive_string(arrays, "kernel", context)
    var raw_kernel = String(kernel)
    var degree = PythonObject(classical_i64(meta, 5)) if classifier else PythonObject(3)
    var coef = PythonObject(0.0)
    if raw_kernel == "poly" or raw_kernel == "sigmoid":
        if classifier:
            var saved = _archive_exact(arrays, "coef0", "<f8", context)
            if Int(py=saved.size) != 1:
                raise Error(where + " coef0 must hold one float64")
            coef = PythonObject(classical_f64(saved, 0))
        else:
            var saved = _archive_exact(arrays, "kernel_params", "<f8", context)
            if Int(py=saved.size) != 2:
                raise Error(where + " kernel_params must hold degree and coef0")
            degree = py_call(b.int, PythonObject(classical_f64(saved, 0)))
            coef = PythonObject(classical_f64(saved, 1))
    var gamma = gamma_setting
    if String(gamma_setting) != "auto" and String(gamma_setting) != "scale":
        gamma = py_call(b.float, gamma_setting)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    # The constructor's attributes, in its order.
    if classifier:
        var names = Python.tuple("C", "kernel", "degree", "gamma", "coef0", "tol", "cache_size", "max_iter",
                                 "nochange_steps", "verbose", "output_type", "random_state", "class_weight",
                                 "decision_function_shape", "probability", "break_ties")
        var values = Python.tuple(classical_f64(hyper, 0), kernel, degree, gamma, coef, classical_f64(hyper, 1),
                                  classical_f64(hyper, 2), classical_i64(meta, 3), classical_i64(meta, 4), PythonObject(False),
                                  PythonObject(None), PythonObject(None), PythonObject(None), PythonObject("ovr"),
                                  PythonObject(False), PythonObject(False))
        for i in range(len(names)):  # driver-loop: the fixed SVC constructor parameters
            py_setitem(fields, names[i], values[i])
    else:
        var names = Python.tuple("kernel", "degree", "gamma", "coef0", "tol", "C", "epsilon", "cache_size",
                                 "verbose", "max_iter", "nochange_steps", "output_type")
        var values = Python.tuple(kernel, degree, gamma, coef, classical_f64(hyper, 2), classical_f64(hyper, 0),
                                  classical_f64(hyper, 1), classical_f64(hyper, 3), PythonObject(False), classical_i64(meta, 3),
                                  classical_i64(meta, 4), PythonObject(None))
        for i in range(len(names)):  # driver-loop: the fixed SVR constructor parameters
            py_setitem(fields, names[i], values[i])
    host_archive_mode(arrays, fields, context)
    host_archive_svm_check(fields, classifier, context)
    var nf = classical_i64(meta, 0)
    var n = classical_i64(meta, 1)
    if classifier:
        if py_contains(arrays, "prob_ab"):
            var probability = _archive_exact(arrays, "prob_ab", "<f8", context)
            if Int(py=probability.ndim) != 1:
                raise Error(where + " prob_ab must be a vector")
            var count = Int(py=probability.size)
            var pairs = Python.list()
            var a = host_archive_typed(Python.tuple((count + 1) // 2), "<f8", context)
            var bb = host_archive_typed(Python.tuple(count // 2), "<f8", context)
            var ap = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(a)))
            _ = a  # keep the Python object alive through the native call above
            var bp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(bb)))
            _ = bb  # keep the Python object alive through the native call above
            for i in range(count // 2):  # driver-loop: saved per-pair calibration coefficients
                ap[i] = classical_f64(probability, 2 * i)
                _ = a  # keep the Python object alive through the native call above
                bp[i] = classical_f64(probability, 2 * i + 1)
                _ = bb  # keep the Python object alive through the native call above
                _ = py_call(pairs.append, Python.tuple(ap[i], bp[i]))
                _ = bb  # keep the Python object alive through the native call above
                _ = a  # keep the Python object alive through the native call above
            if count % 2:
                ap[count // 2] = classical_f64(probability, count - 1)
                _ = a  # keep the Python object alive through the native call above
            py_setitem(fields, "probability", True)
            py_setitem(fields, "_hp_probability", True)
            py_setitem(fields, "_prob_ab", pairs)
            py_setitem(fields, "probA_", a)
            py_setitem(fields, "probB_", bb)
        if py_contains(arrays, "pair_meta"):
            host_archive_svm_ovo(fields, arrays, nf, path, context)
            py_setitem(fields, "_gamma", classical_f64(hyper, 3))
            py_setitem(fields, "_label0", 0.0)
            py_setitem(fields, "_label1", 1.0)
            return model
    var dual = host_archive_matrix(arrays, "dual_coef", "<f4", 1, n, path, context)
    var vectors = host_archive_matrix(arrays, "support_vectors", "<f4", n, nf, path, context)
    var support = _archive_exact(arrays, "support", "<i4", context)
    if Int(py=support.ndim) != 1 or Int(py=support.size) != n:
        raise Error(where + " support does not match n_support_")
    if raw_kernel == "precomputed":
        host_archive_support_check(support, n, nf, path, context)
    var intercept = _archive_exact(arrays, "intercept", "<f4", context)
    if Int(py=intercept.ndim) != 1 or Int(py=intercept.size) != 1:
        raise Error(where + " intercept must hold one float32")
    var labels = PythonObject(None)
    if classifier:
        labels = _archive_exact(arrays, "labels", "<f8", context)
        if Int(py=labels.size) != 2:
            raise Error(where + " labels must hold the two solver labels")
        var classes = host_archive_class_values(arrays["classes"], context)
        if len(classes) != 2:
            raise Error(where + " must carry exactly two classes")
        py_setitem(fields, "classes_", classes)
    py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "n_support_", n)
    py_setitem(fields, "n_iter_", classical_i64(meta, 2))
    py_setitem(fields, "intercept_", intercept)
    py_setitem(fields, "support_", support)
    py_setitem(fields, "support_vectors_", vectors)
    py_setitem(fields, "dual_coef_", dual)
    py_setitem(fields, "_gamma", classical_f64(hyper, 3 if classifier else 4))
    if classifier:
        py_setitem(fields, "_label0", classical_f64(labels, 0))
        py_setitem(fields, "_label1", classical_f64(labels, 1))
    return model


def host_archive_forest(cls: PythonObject, arrays: PythonObject, fmt: String,
                        context: PythonObject) raises -> PythonObject:
    var estimator = _archive_string(arrays, "estimator", context)
    var metadata = context["host_archive_forest_estimators"][estimator]
    var groves = fmt.endswith("-parallel-groves-1")
    if groves:
        var mode = _archive_string(arrays, "numeric_mode", context)
        if String(py=py_call(py_call(mode.strip).lower)) != "identical":
            raise Error("parallel_groves archives must be saved under numeric_mode='identical'")
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    if Int(py=meta.ndim) != 1 or Int(py=meta.size) < 3:
        raise Error("saved forest metadata must hold at least three fields")
    var features = classical_i64(meta, 0)
    var trees = classical_i64(meta, 1)
    var outputs = classical_i64(meta, 2)
    if features < 1 or trees < 1 or outputs < 1:
        raise Error("saved forest features, trees and outputs must be positive")
    var colid = _archive_exact(arrays, "colid", "<i4", context)
    var nodes = Int(py=colid.size)
    if Int(py=colid.ndim) != 1 or String(colid.order) != "C" or nodes < trees:
        raise Error("saved forest column array does not match its tree count")
    var offsets = host_archive_count(arrays, "offsets", "<i4", trees + 1, context, True)
    var thresholds = host_archive_count(arrays, "quesval", "<f4", nodes, context, True)
    var left = host_archive_count(arrays, "left_child", "<i4", nodes, context, True)
    var leaves = _archive_exact(arrays, "leaves", "<f4", context)
    if Int(py=leaves.ndim) != 1 or Int(py=leaves.size) // outputs != nodes or Int(py=leaves.size) % outputs != 0:
        raise Error("saved forest leaves do not match its nodes and outputs")
    if Int(py=offsets._mv[0]) != 0 or Int(py=offsets._mv[trees]) != nodes:
        raise Error("saved forest offsets do not cover its node array")
    var classifier = Bool(py=metadata[1])
    var classes = PythonObject(None)
    if py_contains(arrays, "classes"):
        classes = host_archive_class_values(arrays["classes"], context)
    if classifier and (classes is None or len(classes) != outputs):
        raise Error("saved forest classifier must carry num_outputs classes")
    if not classifier and outputs != 1:
        raise Error("saved forest regressor must carry one output")
    var module = context["host_archive_forest_binding"]
    if module is None:
        raise Error("ImportError: " + String(context["host_archive_forest_binding_error"]))
    if groves and not Bool(py=context["host_archive_forest_has_groves"]):
        raise Error("RuntimeError: forest host binding predates the parallel_groves engine")
    var model = host_archive_allocate(cls, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "estimator", estimator)
    py_setitem(fields, "family", metadata[0])
    py_setitem(fields, "is_classifier", classifier)
    py_setitem(fields, "device", _archive_string(arrays, "device", context) if py_contains(arrays, "device") else PythonObject(None))
    py_setitem(fields, "_inference_engine", "parallel_groves" if groves else "sequential")
    py_setitem(fields, "_groves", PythonObject(None))
    py_setitem(fields, "_offsets", offsets)
    py_setitem(fields, "_colid", colid)
    py_setitem(fields, "_quesval", thresholds)
    py_setitem(fields, "_left_child", left)
    py_setitem(fields, "_leaves", leaves)
    py_setitem(fields, "n_features_in_", features)
    py_setitem(fields, "_n_trees", trees)
    py_setitem(fields, "_num_outputs", outputs)
    py_setitem(fields, "_binding", module)
    if classifier:
        py_setitem(fields, "classes_", classes)
        py_setitem(fields, "n_classes_", len(classes))
    return model


def host_archive_meta_word(meta: PythonObject, index: Int) raises -> Int:
    """`int(meta[index])` of a 1-D int64 member, refused past its end as
    `Array.__getitem__` refused it."""
    var size = Int(py=meta.size)
    if index >= size:
        raise Error("IndexError: mojolearn: index " + String(index) + " out of range for axis 0 of size " + String(size))
    return classical_i64(meta, index)


def host_archive_public_forest(cls: PythonObject, arrays: PythonObject, fmt: String, extra: Bool,
                               path: PythonObject, context: PythonObject) raises -> PythonObject:
    """`RandomForest*.load` / `ExtraTrees*.load` / `DecisionTree*.load`:
    the public estimator exactly as the Python loader built it from
    `cls.__new__(cls)`, with its refusals in its order. `extra` reads the
    three extra-trees meta fields."""
    var saved = _archive_string(arrays, "estimator", context)
    if String(py=saved) != String(py=cls.__name__):
        raise Error(host_archive_where(path, context) + " was saved by " + String(saved) + ", not " + String(cls.__name__))
    var model = host_archive_allocate(cls, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "device", _archive_string(arrays, "device", context))
    # The 0.8.55 `ForestProtocol._restore_inference_metadata`.
    if fmt.endswith("-parallel-groves-1"):
        py_setitem(fields, "inference_engine", "parallel_groves")
        py_setitem(fields, "inference_engine_", "parallel_groves")
        var mode = py_call(py_call(_archive_string(arrays, "numeric_mode", context).strip).lower)
        if String(py=mode) != "fast" and String(py=mode) != "identical":
            raise Error("numeric_mode must be fast, identical or None")
        py_setitem(fields, "numeric_mode", mode)
        py_setitem(fields, "_fit_numeric_mode", mode)
    else:
        py_setitem(fields, "inference_engine", "sequential")
        py_setitem(fields, "inference_engine_", "sequential")
    py_setitem(fields, "_offsets", _archive_exact(arrays, "offsets", "<i4", context))
    py_setitem(fields, "_colid", _archive_exact(arrays, "colid", "<i4", context))
    py_setitem(fields, "_quesval", _archive_exact(arrays, "quesval", "<f4", context))
    py_setitem(fields, "_left_child", _archive_exact(arrays, "left_child", "<i4", context))
    py_setitem(fields, "_leaves", _archive_exact(arrays, "leaves", "<f4", context))
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    py_setitem(fields, "n_features_in_", host_archive_meta_word(meta, 0))
    py_setitem(fields, "_n_trees", host_archive_meta_word(meta, 1))
    py_setitem(fields, "_num_outputs", host_archive_meta_word(meta, 2))
    if extra:
        py_setitem(fields, "depth_cap_bound_", host_archive_meta_word(meta, 3) != 0)
        py_setitem(fields, "max_depth_resolved_", host_archive_meta_word(meta, 4))
        py_setitem(fields, "max_features_", host_archive_meta_word(meta, 5))
    if py_contains(arrays, "classes"):
        var classes = host_archive_class_values(arrays["classes"], context)
        py_setitem(fields, "classes_", classes)
        py_setitem(fields, "n_classes_", len(classes))
    # The former `_mojolearn_after_fit` after a load: the five model arrays
    # frozen (the loaded Arrays are the archive's own fresh storage) and an
    # empty resident snapshot holder.
    var model_names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    for i in range(5):  # small-loop(model_names): the five model array names
        py_setattr(fields[model_names[i]], "_readonly", PythonObject(True))
    py_setitem(fields, "_resident_holder", Python.dict())
    return model


def host_archive_gbdt_state(text: PythonObject, context: PythonObject) raises -> PythonObject:
    var parsed = host_archive_gbdt_text(String(text))
    var state = Python.dict()
    for entry in parsed.metadata.items():  # driver-loop: native parsed model dimensions
        py_setitem(state, entry.key, entry.value)
    for entry in parsed.integers.items():  # driver-loop: native parsed integer layout arrays
        var array = host_archive_typed(Python.tuple(len(entry.value)), "<i4", context)
        var out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(array)))
        _ = array  # keep the Python object alive through the native call above
        for i in range(len(entry.value)):  # driver-loop: publish native parsed buffer words
            out[i] = entry.value[i]
            _ = array  # keep the Python object alive through the native call above
        py_setitem(state, entry.key, array)
    for entry in parsed.floats.items():  # driver-loop: native parsed floating layout arrays
        var array = host_archive_typed(Python.tuple(len(entry.value)), "<f4", context)
        var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(array)))
        _ = array  # keep the Python object alive through the native call above
        for i in range(len(entry.value)):  # driver-loop: publish native parsed buffer words
            out[i] = entry.value[i]
            _ = array  # keep the Python object alive through the native call above
        py_setitem(state, entry.key, array)
    py_setitem(state, "bias", parsed.bias)
    py_setitem(state, "non_symmetric", parsed.non_symmetric)
    return state


def host_archive_gbdt(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var estimator = _archive_string(arrays, "estimator", context)
    var loss = _archive_string(arrays, "loss", context)
    var losses = context["host_archive_gbdt_estimators"][estimator]
    if losses is not None and not py_contains(losses, loss):
        raise Error("mojolearn: a " + String(estimator) + " archive with loss " + String(loss) + "; that class saves " + String(py=py_call(PythonObject(", ").join, losses)) + " only, the file is corrupt")
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    if Int(py=meta.ndim) != 1 or Int(py=meta.size) < 3:
        raise Error("saved gradient boosting metadata needs at least three fields")
    var bias = _archive_exact(arrays, "bias", "<f8", context)
    if Int(py=bias.size) < 1:
        raise Error("saved gradient boosting bias is empty")
    var stored = _archive_exact(arrays, "model", "<u1", context)
    var raw = py_call(Python.import_module("builtins").bytes, stored._mv)
    var text = py_call(raw.decode, "utf-8")
    var state = host_archive_gbdt_state(text, context)
    if classical_i64(meta, 0) != Int(py=state["n_input_features"]) or classical_i64(meta, 1) != Int(py=state["dim"]):
        raise Error("saved gradient boosting metadata disagrees with its model text")
    if classical_f64(bias, 0) != Float64(py=state["bias"]):
        raise Error("saved gradient boosting bias disagrees with its model text")
    var module = context["host_archive_forest_binding"]
    if module is None:
        raise Error("ImportError: " + String(context["host_archive_forest_binding_error"]))
    var mode = PythonObject(None)
    if py_contains(arrays, "numeric_mode"):
        mode = _archive_string(arrays, "numeric_mode", context)
        if String(mode) != "fast" and String(mode) != "identical":
            raise Error("invalid saved numeric_mode")
    var model = host_archive_allocate(cls, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "estimator", estimator)
    py_setitem(fields, "loss", loss)
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "model_", text)
    py_setitem(fields, "_arrays", state)
    py_setitem(fields, "n_features_in_", state["n_input_features"])
    py_setitem(fields, "n_model_columns_", state["n_features"])
    py_setitem(fields, "approx_dim_", state["dim"])
    py_setitem(fields, "bias_", state["bias"])
    py_setitem(fields, "n_classes_", PythonObject(None) if classical_i64(meta, 2) < 0 else PythonObject(classical_i64(meta, 2)))
    py_setitem(fields, "is_classifier", py_contains(context["host_archive_gbdt_classification_losses"], loss))
    py_setitem(fields, "has_proba", py_contains(context["host_archive_gbdt_proba_losses"], loss))
    py_setitem(fields, "_binding", module)
    return model


def _archive_one_string(value: PythonObject) raises -> PythonObject:
    """`ensemble._one_string`: the single str a loaded member holds, or
    None."""
    var b = Python.import_module("builtins")
    var item = value
    while py_type(item) is b.list or py_type(item) is b.tuple:  # driver-loop: over the nesting depth of a stored string field
        if len(item) != 1:
            return PythonObject(None)
        item = item[0]
    if py_type(item) is b.str:
        return item
    return PythonObject(None)


def host_archive_public_gbdt(cls: PythonObject, arrays: PythonObject, path: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """`GradientBoosting.load` (and OrderedRMSE / FeatureFreq, which
    inherit it): the model exactly as the Python loader built it from
    `cls.__new__(cls)`, with its refusals in its order. The output
    dimension comes from the model text through the GBDT binding's
    `gbdt_model_dim` (or, for a CTR text on the CPU binding, the native
    text reader), as the loader read it."""
    var b = Python.import_module("builtins")
    var saved = _archive_string(arrays, "estimator", context)
    if String(py=saved) != String(py=cls.__name__):
        raise Error(host_archive_where(path, context) + " was saved by " + String(saved) + ", not " + String(cls.__name__))
    var model = host_archive_allocate(cls, context)
    var fields = sequential_fields(model)
    py_setitem(fields, "numeric_mode", PythonObject(None))
    if py_contains(arrays, "numeric_mode"):
        var mode = _archive_one_string(arrays["numeric_mode"])
        if mode is None:
            raise Error("mojolearn: saved numeric_mode must be one string")
        if String(mode) != "fast" and String(mode) != "identical":
            raise Error("mojolearn: invalid saved numeric_mode " + _archive_builtin_repr(mode))
        py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "loss", _archive_string(arrays, "loss", context))
    var stored = _archive_exact(arrays, "model", "<u1", context)
    var text = py_call(py_call(b.bytes, stored._mv).decode, "utf-8")
    py_setitem(fields, "model_", text)
    var meta = _archive_exact(arrays, "meta", "<i8", context)
    py_setitem(fields, "n_features_in_", host_archive_meta_word(meta, 0))
    # `obj._bind("_mojolearn_gbdt")`: the saved mode, else the process default.
    var mode = fields["numeric_mode"]
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    var tables = py_get(context, "host_archive_gbdt_bindings")
    if tables is None:
        raise Error("ImportError: mojolearn: the GBDT binding is not connected to the archive owner")
    var table = py_get(tables, PythonObject(String(py_call(py_call(mode.strip).lower))))
    if table is None:
        raise Error("numeric_mode must be fast, identical or None")
    if py_type(table) is b.str:
        raise Error(String(table))
    var t = String(text)
    var ctr = ("\nctr_columns " in t or t.startswith("ctr_columns ") or "\nctr_table " in t or t.startswith("ctr_table ")
               or "\ntensor_ctr_registry " in t or t.startswith("tensor_ctr_registry "))
    var vendor = py_get(table, "gbdt_vendor")
    var cpu = vendor is not None and String(py_call(b.str, py_call(vendor))) == "cpu"
    var dim = 0
    if ctr and cpu:
        dim = Int(py=host_archive_gbdt_state(text, context)["dim"])
    else:
        var entry = py_get(table, "gbdt_model_dim")
        if entry is None:
            raise Error("AttributeError: the GBDT binding has no gbdt_model_dim")
        dim = Int(py=py_call(entry, text))
    py_setitem(fields, "approx_dim_", PythonObject(dim))
    var saved_dim = host_archive_meta_word(meta, 1)
    if dim != saved_dim:
        raise Error(host_archive_where(path, context) + " stores approx_dim " + String(saved_dim) + " but "
                    + "its model text holds " + String(dim) + "; the file is "
                    + "corrupt")
    var classes = host_archive_meta_word(meta, 2)
    py_setitem(fields, "n_classes_", PythonObject(None) if classes < 0 else PythonObject(classes))
    var best = host_archive_meta_word(meta, 3)
    py_setitem(fields, "best_iteration_", PythonObject(None) if best < 0 else PythonObject(best))
    py_setitem(fields, "stopped_early_", PythonObject(host_archive_meta_word(meta, 4) != 0))
    var bias = _archive_exact(arrays, "bias", "<f8", context)
    if Int(py=bias.size) < 1:
        raise Error("IndexError: list index out of range")
    py_setitem(fields, "bias_", PythonObject(classical_f64(bias, 0)))
    py_setitem(fields, "loss_curve_", PythonObject(None))
    py_setitem(fields, "test_loss_curve_", PythonObject(None))
    return model


def host_archive_public_owner(path: PythonObject, arrays: PythonObject, cls: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    """`Class.load(path)` of a public estimator: the format tag and the
    saving estimator are checked with the refusals `_serialize.read_npz` and
    `linear_model._check_saved_by` gave, and the nearest connected ancestor
    (the class whose schema the file holds) is returned."""
    var b = Python.import_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: native archive loading requires an ordinary class")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var connected = context["archive_class_formats"]
    var owner = PythonObject(None)
    for i in range(len(mro)):  # driver-loop: loading class ancestry metadata
        if py_contains(connected, mro[i]):
            owner = mro[i]
            break
    if owner is None:
        raise Error("TypeError: " + String(cls.__name__) + " has no connected saved format")
    var accepted = connected[owner]
    var tag = PythonObject("")
    if py_contains(arrays, "format"):
        tag = _archive_string(arrays, "format", context)
    if not py_contains(accepted[0], tag):
        raise Error("mojolearn: " + _archive_read_repr(path, context) + " holds model format "
                    + _archive_builtin_repr(tag) + ", this loader reads " + _archive_builtin_repr(accepted[1]))
    var saved = _archive_string(arrays, "estimator", context)
    var known = False
    for i in range(len(mro)):  # driver-loop: loading class ancestry names
        known = known or Bool(py=mro[i].__name__ == saved)
    if not known:
        raise Error("mojolearn: " + _archive_read_repr(path, context) + " was saved by "
                    + String(saved) + ", not " + String(cls.__name__))
    return owner


def host_archive_build(path: PythonObject, context: PythonObject,
                       requested_class: PythonObject = PythonObject(None), classical_only: Bool = False) raises -> PythonObject:
    # Read the archive once and select its native schema from the fixed
    # import-time registry. No class constructor or load method is invoked.
    var arrays = classical_archive_read(path, context)
    # A public class's own `load` (the context _native_archive_bridge
    # connects) keys the schema on the class, not on the saved name: a
    # subclass loads its base's file, as the Python loaders did.
    var public_owner = PythonObject(None)
    if requested_class is not None and py_get(context, "archive_class_formats") is not None:
        public_owner = host_archive_public_owner(path, arrays, requested_class, context)
    var fmt = String(_archive_string(arrays, "format", context))
    if classical_only and not py_contains(context["host_archive_classical_formats"], fmt):
        raise Error("model archive format does not name a classical host model")
    var estimator = _archive_string(arrays, "estimator", context)
    if public_owner is not None:
        estimator = public_owner.__name__
    # Historical explicit CPU saves used their Host* class name. Admit only
    # these exact aliases within their matching unchanged archive schemas.
    if fmt == "mojolearn-kde-1" and String(estimator) == "HostKernelDensity":
        estimator = PythonObject("KernelDensity")
    elif fmt == "mojolearn-iforest-1" and String(estimator) == "HostIsolationForest":
        estimator = PythonObject("IsolationForest")
    var formats = context["host_archive_formats"]
    var classes = py_get(formats, fmt)
    if classes is None:
        raise Error("native host archive builder has no supported schema for " + fmt)
    var cls = py_get(classes, estimator)
    if cls is None:
        raise Error("mojolearn: " + String(py_call(Python.import_module("builtins").repr, path)) + " was saved by " + String(estimator) + ", which " + fmt + " does not hold")
    if public_owner is not None:
        cls = requested_class
    elif requested_class is not None:
        var b = Python.import_module("builtins")
        if py_type(requested_class) is not b.type:
            raise Error("TypeError: native archive loading requires an ordinary class")
        var mro = py_call(b.type.__dict__["__mro__"].__get__, requested_class)
        var matched = False
        for i in range(len(mro)):  # driver-loop: requested loading class ancestry
            matched = matched or mro[i] is cls
        if not matched:
            raise Error("model archive was saved by a different estimator")
        cls = requested_class
    var model = PythonObject(None)
    if fmt == "mojolearn-linear-1":
        # Admit the class layout before the existing linear builder allocates.
        var probe = host_archive_object(cls, arrays, context)
        model = classical_archive_build(cls, arrays, path, String(estimator) == "Ridge", context)
    elif fmt == "mojolearn-pca-1" or fmt == "mojolearn-tsvd-1":
        model = host_archive_decomposition(cls, arrays, fmt == "mojolearn-pca-1", path, context)
    elif fmt == "mojolearn-cd-1":
        model = host_archive_cd(cls, arrays, String(estimator) == "Lasso", path, context)
    elif fmt == "mojolearn-kernel-ridge-1" or fmt == "mojolearn-nystroem-1" or fmt == "mojolearn-rbf-sampler-1":
        model = host_archive_kernel(cls, arrays, 0 if fmt == "mojolearn-kernel-ridge-1" else (1 if fmt == "mojolearn-nystroem-1" else 2), path, context)
    elif fmt == "mojolearn-scaler-1" and String(estimator) == "MinMaxScaler":
        model = host_archive_minmax(cls, arrays, context)
    elif fmt == "mojolearn-scaler-1" and String(estimator) == "StandardScaler":
        model = scaler_archive_build(cls, arrays, path, context)
    elif fmt == "mojolearn-cholesky-1":
        model = host_archive_cholesky(cls, arrays, path, context)
    elif fmt == "mojolearn-embedding-1":
        model = host_archive_embedding(cls, arrays, context)
    elif fmt == "mojolearn-kmeans-1":
        model = host_archive_kmeans(cls, arrays, path, context)
    elif fmt == "mojolearn-gmm-1":
        model = host_archive_mixture(cls, arrays, path, context)
    elif fmt == "mojolearn-kde-1" or fmt == "mojolearn-dbscan-1":
        model = host_archive_density(cls, arrays, fmt == "mojolearn-kde-1", path, context)
    elif fmt == "mojolearn-agglomerative-1":
        model = host_archive_agglomerative(cls, arrays, path, context)
    elif fmt == "mojolearn-logistic-1":
        model = host_archive_logistic(cls, arrays, path, context)
    elif fmt == "mojolearn-arima-1" or fmt == "mojolearn-arima-2":
        model = host_archive_arima(cls, arrays, fmt == "mojolearn-arima-2", path, context)
    elif fmt == "mojolearn-holtwinters-1":
        model = host_archive_holtwinters(cls, arrays, path, context)
    elif fmt == "mojolearn-ivf-flat-1":
        model = host_archive_ivf(cls, arrays, context)
    elif py_contains(py_get(context, "host_archive_ann_layouts", Python.dict()), fmt):
        model = host_archive_ann(cls, arrays, fmt, path, context)
    elif fmt == "mojolearn-hdbscan-2":
        model = host_archive_hdbscan(cls, arrays, path, context)
    elif fmt == "mojolearn-radius-1":
        model = host_archive_radius(cls, arrays, path, context)
    elif fmt == "mojolearn-knn-1":
        model = host_archive_neighbors(cls, arrays, 0 if String(estimator) == "NearestNeighbors" else (1 if String(estimator) == "KNeighborsClassifier" else 2), context)
        # Native equivalent of the original loader's _mojolearn_after_fit:
        # a newly restored CPU neighbor model owns an empty query holder. A
        # public class's own `load` also set `used_query_tile_ = None`.
        var holder = Python.dict()
        if public_owner is not None:
            py_setitem(holder, "used_query_tile_", PythonObject(None))
        py_setitem(sequential_fields(model), "_resident_holder", holder)
    elif fmt == "mojolearn-iforest-1" and public_owner is not None:
        model = host_archive_public_iforest(cls, arrays, path, context)
    elif fmt == "mojolearn-iforest-1":
        model = host_archive_iforest(cls, arrays, context)
    elif fmt == "mojolearn-spectral-1":
        model = host_archive_spectral(cls, arrays, path, context)
    elif fmt == "mojolearn-umap-1":
        model = host_archive_umap(cls, arrays, path, context)
    elif fmt == "mojolearn-gp-1":
        model = host_archive_gpr(cls, arrays, path, context)
    elif fmt == "mojolearn.gpc.v1":
        model = host_archive_gpc(cls, arrays, path, context)
    elif fmt == "mojolearn-svc-1" or fmt == "mojolearn-svr-1":
        model = host_archive_svm(cls, arrays, fmt == "mojolearn-svc-1", path, context)
    elif public_owner is not None and (fmt.startswith("mojolearn-randomforest-1") or fmt.startswith("mojolearn-extratrees-1")):
        model = host_archive_public_forest(cls, arrays, fmt, fmt.startswith("mojolearn-extratrees-1"), path, context)
    elif public_owner is not None and fmt == "mojolearn-gbdt-1":
        model = host_archive_public_gbdt(cls, arrays, path, context)
    elif fmt == "mojolearn-gbdt-1":
        model = host_archive_gbdt(cls, arrays, context)
    elif py_contains(context["host_archive_forest_formats"], fmt):
        model = host_archive_forest(cls, arrays, fmt, context)
    else:
        raise Error("native host archive schema has not been implemented: " + fmt + "/" + String(estimator))
    if py_contains(context["host_archive_fitted_estimators"], estimator):
        py_setitem(sequential_fields(model), "_mojolearn_fitted", True)
    if requested_class is None:
        var fields = sequential_fields(model)
        if fmt == "mojolearn-kernel-ridge-1" or fmt == "mojolearn-nystroem-1":
            var kernel = Int(py=fields["_kernel_params"][0])
            if kernel != 0 and kernel != 1 and kernel != 2 and kernel != 3 and kernel != 5:
                raise Error("ImportError: unsupported saved kernel code " + String(kernel))
        if fmt == "mojolearn-logistic-1" and len(fields["classes_"]) > 2:
            if String(fields["numeric_mode"]) != "identical":
                raise Error("saved multiclass LogisticRegression runs identical only on the host")
            if not Bool(py=context["host_archive_has_softmax"]):
                raise Error("ImportError: host LogisticRegression binding does not export qn_softmax")
        py_setitem(fields, "estimator", estimator)
    return model
