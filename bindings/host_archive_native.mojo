# SPDX-License-Identifier: Apache-2.0
# Native saved-model reconstruction. The format table is import-time ABI
# metadata, never Python loaders, constructors, or callbacks. The public
# router is switched only after every existing format has a native builder.
from std.python import Python, PythonObject
from std.memory import bitcast
from std.math import isfinite
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.sequential_description import sequential_fields
from bindings.classical_archive_read import classical_archive_read, classical_archive_build, _archive_exact, _archive_string
from bindings.classical_arrays import classical_i64, classical_f64, classical_wrap_array, classical_array, classical_typed, classical_address
from core.host_archive_impl import host_archive_copy_f32, host_archive_unique_i32
from bindings.scaler_lifecycle_native import scaler_archive_build


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


def host_archive_object(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: native archive classes require an ordinary native class layout")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var saved = _archive_string(arrays, "estimator", context)
    var known = False
    var found_dictionary = False
    var found_finalizer = False
    for i in range(len(mro)):  # driver-loop: connected class ancestry metadata
        var fields = py_call(b.type.__dict__["__dict__"].__get__, mro[i])
        if not found_finalizer and py_contains(fields, "__del__"):
            if fields["__del__"] is not py_get(context, "host_archive_finalizer_descriptor"):
                raise Error("TypeError: Python finalizer hooks are not accepted by native archive loading")
            found_finalizer = True
        if not found_dictionary and py_contains(fields, "__dict__"):
            if py_type(fields["__dict__"]) is not context["host_archive_dict_descriptor_type"]:
                raise Error("TypeError: Python dictionary descriptors are not accepted by native archive loading")
            found_dictionary = True
        known = known or Bool(py=mro[i].__name__ == saved)
    if not known:
        raise Error("model archive estimator does not match its connected class")
    if not found_dictionary:
        raise Error("TypeError: native archive loading requires a native instance dictionary")
    var model = py_call(b.object.__new__, cls)
    return model


def host_archive_mode(arrays: PythonObject, fields: PythonObject, context: PythonObject) raises:
    var mode = _archive_string(arrays, "numeric_mode", context)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("invalid saved numeric_mode")
    py_setitem(fields, "numeric_mode", mode)


def host_archive_decomposition(cls: PythonObject, arrays: PythonObject, pca: Bool,
                               context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 4 if pca else 2, context, True)
    var nc = classical_i64(meta, 0)
    var nf = classical_i64(meta, 1)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "n_components", nc)
    py_setitem(fields, "n_components_", nc)
    py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "tol", 0.0)
    py_setitem(fields, "n_oversamples", 10)
    py_setitem(fields, "power_iteration_normalizer", "auto")
    py_setitem(fields, "random_state", PythonObject(None))
    py_setitem(fields, "components_", host_archive_shape(arrays, "components", "<f4", Python.tuple(nc, nf), context))
    py_setitem(fields, "singular_values_", host_archive_count(arrays, "singular_values", "<f4", nc, context))
    if pca:
        py_setitem(fields, "copy", True)
        py_setitem(fields, "whiten", classical_i64(meta, 3) != 0)
        py_setitem(fields, "svd_solver", _archive_string(arrays, "svd_solver", context))
        py_setitem(fields, "iterated_power", "auto")
        py_setitem(fields, "mean_", host_archive_count(arrays, "mean", "<f4", nf, context))
        py_setitem(fields, "explained_variance_", host_archive_count(arrays, "explained_variance", "<f4", nc, context))
        py_setitem(fields, "explained_variance_ratio_", host_archive_count(arrays, "explained_variance_ratio", "<f4", nc, context))
        var noise = host_archive_count(arrays, "noise_variance", "<f8", 1, context, True)
        py_setitem(fields, "noise_variance_", classical_f64(noise, 0))
        py_setitem(fields, "n_samples_", classical_i64(meta, 2))
    else:
        py_setitem(fields, "algorithm", _archive_string(arrays, "algorithm", context))
        py_setitem(fields, "n_iter", 5)
    return model


def host_archive_cd(cls: PythonObject, arrays: PythonObject, lasso: Bool,
                    context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 4, context, True)
    var hyper = host_archive_count(arrays, "hyper", "<f8", 3, context, True)
    var ratio = classical_f64(hyper, 1)
    if lasso and ratio != 1.0:
        raise Error("a saved Lasso must carry l1_ratio 1.0")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
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
    var nf = classical_i64(meta, 0)
    py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "coef_", host_archive_shape(arrays, "coef", "<f4", Python.tuple(nf), context))
    var intercept = host_archive_count(arrays, "intercept", "<f8", 1, context, True)
    py_setitem(fields, "intercept_", classical_f64(intercept, 0))
    py_setitem(fields, "n_iter_", classical_i64(meta, 3))
    return model


def host_archive_kernel(cls: PythonObject, arrays: PythonObject, kind: Int,
                        context: PythonObject) raises -> PythonObject:
    # kind: 0 KernelRidge, 1 Nystroem, 2 RBFSampler. These are model
    # schemas, unrelated to benchmark shapes or device execution routes.
    var meta = host_archive_count(arrays, "meta", "<i8", 7 if kind == 0 else (6 if kind == 1 else 3), context, True)
    var hyper = host_archive_count(arrays, "hyper", "<f8", 2 if kind == 1 else 3, context, True)
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
        py_setitem(fields, "random_weights_", host_archive_shape(arrays, "random_weights", "<f4", Python.tuple(d, q), context))
        py_setitem(fields, "random_offset_", host_archive_shape(arrays, "random_offset", "<f4", Python.tuple(q), context))
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
        var dual = host_archive_shape(arrays, "dual", "<f4", Python.tuple(n * t), context)
        if not squeeze:
            dual = classical_wrap_array(dual._store, Python.tuple(n, t), "<f4", "C", context)
        py_setitem(fields, "alpha", alpha)
        py_setitem(fields, "X_fit_", host_archive_shape(arrays, "x_fit", "<f4", Python.tuple(n, d), context))
        py_setitem(fields, "n_targets_", t)
        py_setitem(fields, "_squeeze", squeeze)
        py_setitem(fields, "dual_coef_", dual)
        py_setitem(fields, "info_", classical_i64(meta, 5))
        py_setitem(fields, "_kernel_params", Python.tuple(kernel, degree, gamma, coef0, alpha))
    else:
        var seed = classical_i64(meta, 4)
        py_setitem(fields, "n_components", n)
        py_setitem(fields, "random_state", seed)
        py_setitem(fields, "components_", host_archive_shape(arrays, "components", "<f4", Python.tuple(n, d), context))
        py_setitem(fields, "component_indices_", host_archive_shape(arrays, "component_indices", "<i4", Python.tuple(n), context))
        py_setitem(fields, "normalization_", host_archive_shape(arrays, "normalization", "<f4", Python.tuple(n, n), context))
        py_setitem(fields, "eigenvalues_", host_archive_shape(arrays, "eigenvalues", "<f4", Python.tuple(n), context))
        py_setitem(fields, "eigenvectors_", host_archive_shape(arrays, "eigenvectors", "<f4", Python.tuple(n, n), context))
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


def host_archive_cholesky(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 3, context, True)
    var scalars = host_archive_count(arrays, "scalars", "<f8", 2, context, True)
    var jp = host_archive_count(arrays, "jitter_param", "<f8", 1, context, True)
    var n = classical_i64(meta, 0)
    if n < 1:
        raise Error("model Cholesky factor dimension must be positive")
    var factor = host_archive_shape(arrays, "L", "<f4", Python.tuple(n, n), context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
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


def host_archive_kmeans(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 9, context, True)
    var reals = host_archive_count(arrays, "reals", "<f8", 5, context, True)
    var k = classical_i64(meta, 0)
    var d = classical_i64(meta, 1)
    var n = classical_i64(meta, 2)
    if k < 1 or d < 1:
        raise Error("saved KMeans holds an empty model")
    var metric = _archive_string(arrays, "metric", context)
    var start = _archive_string(arrays, "init", context)
    var metrics = context["host_archive_kmeans_metrics"]
    var starts = context["host_archive_kmeans_starts"]
    if not py_contains(metrics, metric) or Int(py=metrics[metric]) != classical_i64(meta, 4):
        raise Error("saved KMeans metric and code do not match")
    if not py_contains(starts, start) or Int(py=starts[start]) != classical_i64(meta, 5):
        raise Error("saved KMeans init and code do not match")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "n_clusters", k)
    py_setitem(fields, "init", start)
    py_setitem(fields, "n_init", classical_i64(meta, 7))
    py_setitem(fields, "max_iter", classical_i64(meta, 6))
    py_setitem(fields, "tol", classical_f64(reals, 3))
    py_setitem(fields, "random_state", classical_i64(meta, 8))
    py_setitem(fields, "init_centroids", PythonObject(None))
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "oversampling_factor", classical_f64(reals, 4))
    py_setitem(fields, "cluster_centers_", host_archive_shape(arrays, "centers", "<f4", Python.tuple(k, d), context))
    py_setitem(fields, "labels_", host_archive_shape(arrays, "labels", "<i4", Python.tuple(n), context))
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "n_iter_", classical_i64(meta, 3))
    py_setitem(fields, "inertia_", classical_f64(reals, 0))
    py_setitem(fields, "sum_scale_", classical_f64(reals, 1))
    py_setitem(fields, "weight_scale_", classical_f64(reals, 2))
    return model


def host_archive_mixture(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "ints", "<i8", 6, context, True)
    var reals = host_archive_count(arrays, "reals", "<f8", 3, context, True)
    var k = classical_i64(meta, 0)
    var d = classical_i64(meta, 1)
    if k < 1 or d < 1:
        raise Error("saved GaussianMixture holds an empty model")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "n_components", k)
    py_setitem(fields, "covariance_type", _archive_string(arrays, "covariance_type", context))
    py_setitem(fields, "tol", classical_f64(reals, 1))
    py_setitem(fields, "reg_covar", classical_f64(reals, 2))
    py_setitem(fields, "max_iter", classical_i64(meta, 4))
    py_setitem(fields, "init_params", _archive_string(arrays, "init_params", context))
    py_setitem(fields, "random_state", classical_i64(meta, 5))
    py_setitem(fields, "n_init", 1)
    py_setitem(fields, "warm_start", False)
    py_setitem(fields, "means_init", PythonObject(None))
    py_setitem(fields, "weights_init", PythonObject(None))
    py_setitem(fields, "precisions_init", PythonObject(None))
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "weights_", host_archive_shape(arrays, "weights", "<f4", Python.tuple(k), context))
    var means = host_archive_shape(arrays, "means", "<f4", Python.tuple(k * d), context)
    var covariance = host_archive_shape(arrays, "covariances", "<f4", Python.tuple(k * d * d), context)
    var precision = host_archive_shape(arrays, "precisions_cholesky", "<f4", Python.tuple(k * d * d), context)
    py_setitem(fields, "means_", classical_wrap_array(means._store, Python.tuple(k, d), "<f4", "C", context))
    py_setitem(fields, "covariances_", classical_wrap_array(covariance._store, Python.tuple(k, d, d), "<f4", "C", context))
    py_setitem(fields, "precisions_cholesky_", classical_wrap_array(precision._store, Python.tuple(k, d, d), "<f4", "C", context))
    py_setitem(fields, "log_det_chol_", host_archive_shape(arrays, "log_det_chol", "<f4", Python.tuple(k), context))
    py_setitem(fields, "n_iter_", classical_i64(meta, 2))
    py_setitem(fields, "converged_", classical_i64(meta, 3) != 0)
    py_setitem(fields, "lower_bound_", classical_f64(reals, 0))
    return model


def host_archive_density(cls: PythonObject, arrays: PythonObject, kde: Bool,
                         context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 3, context, True)
    var parameter = host_archive_count(arrays, "bandwidth" if kde else "eps", "<f8", 1, context, True)
    var nf = classical_i64(meta, 0)
    var n = classical_i64(meta, 1)
    var metric = _archive_string(arrays, "metric", context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "n_samples_fit_", n)
    if kde:
        var bandwidth = classical_f64(parameter, 0)
        if not bandwidth > 0.0:
            raise Error("KernelDensity bandwidth must be positive")
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
    else:
        if not py_contains(context["host_archive_dbscan_metrics"], metric):
            raise Error("saved DBSCAN metric is unsupported")
        py_setitem(fields, "eps", classical_f64(parameter, 0))
        py_setitem(fields, "min_samples", classical_i64(meta, 2))
        py_setitem(fields, "algorithm", _archive_string(arrays, "algorithm", context))
        py_setitem(fields, "max_mbytes_per_batch", PythonObject(None))
        py_setitem(fields, "max_iterations", PythonObject(None))
        py_setitem(fields, "prediction_data", True)
        var indices = _archive_exact(arrays, "core_sample_indices", "<i4", context)
        var nc = Int(py=indices.size)
        py_setitem(fields, "components_", host_archive_shape(arrays, "components", "<f4", Python.tuple(nc, nf), context))
        py_setitem(fields, "core_sample_indices_", indices)
        py_setitem(fields, "_core_labels", host_archive_count(arrays, "core_labels", "<i4", nc, context))
        py_setitem(fields, "labels_", host_archive_count(arrays, "labels", "<i4", n, context))
    return model


def host_archive_agglomerative(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 3, context, True)
    var k = classical_i64(meta, 0)
    var nf = classical_i64(meta, 1)
    var n = classical_i64(meta, 2)
    var metric = _archive_string(arrays, "metric", context)
    if not py_contains(context["host_archive_agglo_metrics"], metric):
        raise Error("NotImplementedError: saved AgglomerativeClustering metric is unsupported")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
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
    py_setitem(fields, "_fit_X", host_archive_shape(arrays, "x", "<f4", Python.tuple(n, nf), context))
    py_setitem(fields, "labels_", host_archive_count(arrays, "labels", "<i4", n, context))
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


def host_archive_logistic(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 2, context, True)
    var nf = classical_i64(meta, 0)
    var intercept = classical_i64(meta, 1) != 0
    if nf < 0:
        raise Error("saved LogisticRegression feature count cannot be negative")
    var classes = host_archive_class_values(arrays["classes"], context)
    if len(classes) < 2:
        raise Error("saved LogisticRegression must carry at least two classes")
    var targets = 1 if len(classes) == 2 else len(classes)
    var w = _archive_exact(arrays, "w", "<f4", context)
    var words = Int(py=w.size)
    if Int(py=w.ndim) != 1 or words % targets != 0 or words // targets - (1 if intercept else 0) != nf:
        raise Error("saved LogisticRegression weights do not match features and class count")
    var coef = classical_typed(Python.tuple(targets, nf), "<f4", context)
    var bias = classical_array(targets, 0, context)
    host_archive_copy_f32(Int(py=classical_address(w)), 4, targets * 4, targets, nf, Int(py=classical_address(coef)))
    if intercept:
        host_archive_copy_f32(Int(py=classical_address(w)) + targets * nf * 4, 4, 4,
                              targets, 1, Int(py=classical_address(bias)))
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "penalty", "l2")
    py_setitem(fields, "tol", 0.0001)
    py_setitem(fields, "C", 1.0)
    py_setitem(fields, "fit_intercept", intercept)
    py_setitem(fields, "class_weight", PythonObject(None))
    py_setitem(fields, "max_iter", 1000)
    py_setitem(fields, "linesearch_max_iter", 50)
    py_setitem(fields, "l1_ratio", PythonObject(None))
    py_setitem(fields, "solver", "qn")
    py_setitem(fields, "classes_", classes)
    py_setitem(fields, "_w", w)
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", bias)
    py_setitem(fields, "n_features_in_", nf)
    return model


def host_archive_arima(cls: PythonObject, arrays: PythonObject, exog: Bool,
                       context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 13 if exog else 12, context, True)
    for i in range(7):  # driver-loop: fixed ARIMA order fields
        if classical_i64(meta, i) < 0:
            raise Error("ARIMA order entries must be non-negative integers")
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
        raise Error("ARIMA exogenous format requires at least one exogenous column")
    if not exog and py_contains(arrays, "exog"):
        raise Error("ARIMA archive carries exog under the non-exogenous format")
    var trend = _archive_string(arrays, "trend", context)
    var trend_arg = PythonObject(None) if String(trend) == "none" else trend
    var expected_k = 0 if d + sd else 1
    if trend_arg is not None:
        var lower = String(py=py_call(trend.lower))
        if lower != "n" and lower != "c":
            raise Error("NotImplementedError: ARIMA trend must be None, n or c; use native exogenous data for a time trend")
        expected_k = 1 if lower == "c" else 0
    if k != expected_k:
        raise Error("ARIMA saved intercept count does not match its trend")
    var method = py_call(_archive_string(arrays, "method", context).lower)
    if String(method) != "ml" and String(method) != "css" and String(method) != "css-ml":
        raise Error("unknown saved ARIMA method")
    var complexity = p + q + sp + sq + k + 1 + nx
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "order", Python.tuple(p, d, q))
    py_setitem(fields, "seasonal_order", Python.tuple(sp, sd, sq, period))
    py_setitem(fields, "trend", trend_arg)
    py_setitem(fields, "k_", k)
    py_setitem(fields, "method", method)
    py_setitem(fields, "maxiter", classical_i64(meta, 10))
    py_setitem(fields, "verbose", False)
    py_setitem(fields, "output_type", PythonObject(None))
    py_setitem(fields, "batch_size_", batch)
    py_setitem(fields, "n_obs_", n)
    py_setitem(fields, "input_copied_", classical_i64(meta, 11) != 0)
    py_setitem(fields, "n_exog_", nx)
    py_setitem(fields, "complexity_", complexity)
    py_setitem(fields, "_exog", host_archive_shape(arrays, "exog", "<f4", Python.tuple(batch, n, nx), context) if exog else PythonObject(None))
    py_setitem(fields, "_y", host_archive_shape(arrays, "y", "<f4", Python.tuple(batch, n), context))
    var coefficients = Python.tuple("params", "x", "x0")
    for i in range(len(coefficients)):  # driver-loop: three saved coefficient states
        var name = String(coefficients[i])
        py_setitem(fields, name + "_", host_archive_shape(arrays, name, "<f4", Python.tuple(batch, complexity), context))
    py_setitem(fields, "fx_", host_archive_shape(arrays, "fx", "<f4", Python.tuple(batch), context))
    py_setitem(fields, "n_iter_", host_archive_shape(arrays, "n_iter", "<i4", Python.tuple(batch), context))
    py_setitem(fields, "retcode_", host_archive_shape(arrays, "retcode", "<i4", Python.tuple(batch), context))
    var scores = Python.tuple("llf", "aic", "bic")
    for i in range(len(scores)):  # driver-loop: three saved fit criterion arrays
        var name = String(scores[i])
        py_setitem(fields, name + "_", host_archive_shape(arrays, name, "<f8", Python.tuple(batch), context))
    return model


def host_archive_holtwinters(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 4, context, True)
    var eps = host_archive_count(arrays, "eps", "<f8", 1, context, True)
    var n = classical_i64(meta, 0)
    var batch = classical_i64(meta, 1)
    var period = classical_i64(meta, 2)
    var start = classical_i64(meta, 3)
    var seasonal = _archive_string(arrays, "seasonal", context)
    if String(seasonal) != "additive" and String(seasonal) != "add" and String(seasonal) != "multiplicative" and String(seasonal) != "mul":
        raise Error("unsupported saved ExponentialSmoothing seasonal type")
    var initialization = _archive_string(arrays, "initialization_method", context) if py_contains(arrays, "initialization_method") else PythonObject("heuristic")
    if String(initialization) != "estimated" and String(initialization) != "heuristic" and String(initialization) != "cuml":
        raise Error("unsupported saved ExponentialSmoothing initialization method")
    if batch < 1 or n <= period:
        raise Error("saved ExponentialSmoothing rows and season period are inconsistent")
    var rows = n - period
    var cl = rows * batch
    var comps = host_archive_shape(arrays, "components", "<f4", Python.tuple(3 * cl), context)
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "endog", PythonObject(None))
    py_setitem(fields, "initialization_method", initialization)
    py_setitem(fields, "seasonal", seasonal)
    py_setitem(fields, "seasonal_periods", period)
    py_setitem(fields, "start_periods", start)
    py_setitem(fields, "ts_num", batch)
    py_setitem(fields, "eps", classical_f64(eps, 0))
    py_setitem(fields, "n", n)
    py_setitem(fields, "_components_len", cl)
    py_setitem(fields, "_comps", comps)
    py_setitem(fields, "_comp_layout", Python.tuple(cl, batch, rows))
    var time_major = Python.dict()
    var names = Python.tuple("level", "trend", "season")
    for i in range(len(names)):  # driver-loop: three named saved component blocks
        var copy = classical_typed(Python.tuple(rows, batch), "<f4", context)
        host_archive_copy_f32(Int(py=classical_address(comps)) + i * cl * 4, batch * 4, 4,
                              rows, batch, Int(py=classical_address(copy)))
        py_setitem(time_major, names[i], copy)
    py_setitem(fields, "_time_major", time_major)
    var stats = Python.tuple("sse", "alpha", "beta", "gamma")
    for i in range(len(stats)):  # driver-loop: four saved per-series fit statistics
        var name = String(stats[i])
        py_setitem(fields, name + "_", host_archive_shape(arrays, name, "<f4", Python.tuple(batch), context))
    py_setitem(fields, "n_iter_", host_archive_shape(arrays, "n_iter", "<i4", Python.tuple(batch), context))
    py_setitem(fields, "criterion_", host_archive_shape(arrays, "criterion", "<i4", Python.tuple(batch), context))
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


def host_archive_hdbscan(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "ints", "<i8", 11, context, True)
    var reals = host_archive_count(arrays, "reals", "<f8", 2, context, True)
    var d = classical_i64(meta, 0)
    var selected = classical_i64(meta, 1)
    var condensed = classical_i64(meta, 4)
    var edges = classical_i64(meta, 5)
    var exemplars = classical_i64(meta, 10)
    var raw = _archive_exact(arrays, "raw_data", "<f4", context)
    if Int(py=raw.ndim) != 2 or Int(py=raw.shape[1]) != d or Int(py=raw.shape[0]) < 2:
        raise Error("HDBSCAN raw_data shape does not match saved features")
    var n = Int(py=raw.shape[0])
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "min_cluster_size", classical_i64(meta, 6))
    var samples = classical_i64(meta, 7)
    py_setitem(fields, "min_samples", PythonObject(None) if samples == -1 else PythonObject(samples))
    py_setitem(fields, "cluster_selection_epsilon", classical_f64(reals, 1))
    py_setitem(fields, "max_cluster_size", classical_i64(meta, 8))
    py_setitem(fields, "metric", _archive_string(arrays, "metric", context))
    py_setitem(fields, "alpha", classical_f64(reals, 0))
    py_setitem(fields, "cluster_selection_method", _archive_string(arrays, "cluster_selection_method", context))
    py_setitem(fields, "allow_single_cluster", classical_i64(meta, 9) != 0)
    py_setitem(fields, "prediction_data", True)
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "labels_", host_archive_shape(arrays, "labels", "<i4", Python.tuple(n), context))
    py_setitem(fields, "core_distances_", host_archive_shape(arrays, "core_distances", "<f4", Python.tuple(n), context))
    py_setitem(fields, "n_clusters_", selected)
    py_setitem(fields, "n_outliers_", classical_i64(meta, 2))
    py_setitem(fields, "n_boruvka_rounds_", classical_i64(meta, 3))
    py_setitem(fields, "n_condensed_clusters_", condensed)
    py_setitem(fields, "_raw_data", raw)
    var prediction = Python.dict(n_edges=edges, n_condensed=condensed, n_exemplars=exemplars)
    var names = Python.tuple("parents", "lambdas", "deaths", "selected_clusters", "index_into_children", "exemplar_idx", "exemplar_label_offsets")
    var dtypes = Python.tuple("<i4", "<f4", "<f4", "<i4", "<i4", "<i4", "<i4")
    var sizes = Python.tuple(edges, edges, condensed, max(selected, 1), edges + 1, max(exemplars, 1), selected + 1)
    for i in range(len(names)):  # driver-loop: seven saved prediction-data arrays
        py_setitem(prediction, names[i], host_archive_shape(arrays, String(names[i]), String(dtypes[i]), Python.tuple(sizes[i]), context))
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


def host_archive_neighbors(cls: PythonObject, arrays: PythonObject, kind: Int,
                           context: PythonObject) raises -> PythonObject:
    # kind: 0 nearest, 1 classifier, 2 regressor, 3 radius.
    var radius = kind == 3
    var meta = host_archive_count(arrays, "meta", "<i8", 2 if radius else 6, context, True)
    var parameter = host_archive_count(arrays, "p", "<f8", 1, context, True)
    var p = classical_f64(parameter, 0)
    var nf = classical_i64(meta, 0)
    var n = classical_i64(meta, 1)
    var metric = _archive_string(arrays, "metric", context)
    var algorithm = _archive_string(arrays, "algorithm", context)
    var rbc = radius or String(algorithm) == "rbc"
    if radius:
        if String(algorithm) != "auto" and String(algorithm) != "rbc":
            raise Error("RadiusNeighbors requires auto or rbc search")
    elif String(algorithm) != "auto" and String(algorithm) != "brute" and String(algorithm) != "rbc":
        raise Error("saved neighbors require brute, auto or rbc search")
    if rbc and (kind == 1 or kind == 2):
        raise Error("saved neighbor classifiers and regressors require brute search")
    var resolved = host_archive_neighbor_metric(metric, p, rbc, context)
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
    if radius:
        var saved_radius = host_archive_count(arrays, "radius", "<f8", 1, context, True)
        var r = classical_f64(saved_radius, 0)
        if not isfinite(r) or r <= 0:
            raise Error("RadiusNeighbors radius must be positive and finite")
        py_setitem(fields, "radius", r)
        py_setitem(fields, "_metric_value", resolved[0])
        py_setitem(fields, "_metric_arg", resolved[1])
        return model
    py_setitem(fields, "n_neighbors", classical_i64(meta, 2))
    py_setitem(fields, "query_tile", classical_i64(meta, 3))
    py_setitem(fields, "used_query_tile_", PythonObject(None))
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


def host_archive_spectral(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 8, context, True)
    var tolerance = host_archive_count(arrays, "eigen_tol", "<f8", 1, context, True)
    var clusters = classical_i64(meta, 0)
    var k = classical_i64(meta, 1)
    var neighbors = classical_i64(meta, 2)
    var starts = classical_i64(meta, 3)
    var seed = classical_i64(meta, 4)
    var seed_none = classical_i64(meta, 5) != 0
    var n = classical_i64(meta, 6)
    var nf = classical_i64(meta, 7)
    var tol = classical_f64(tolerance, 0)
    var affinity = _archive_string(arrays, "affinity", context)
    if String(affinity) == "rbf" or String(affinity) == "precomputed_nearest_neighbors":
        raise Error("NotImplementedError: saved spectral prediction data requires nearest_neighbors or precomputed affinity")
    if String(affinity) != "nearest_neighbors" and String(affinity) != "precomputed":
        raise Error("unsupported saved SpectralClustering affinity")
    if not tol > 0.0:
        raise Error("SpectralClustering eigen_tol must be positive")
    if not seed_none and (seed < 0 or seed >= 4294967296):
        raise Error("SpectralClustering random_state must fit uint32")
    if clusters < 1 or k < 1 or neighbors < 1 or starts < 1:
        raise Error("saved spectral cluster, component, neighbor and start counts must be positive")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
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
    py_setitem(fields, "_pd_eigenvalues", host_archive_count(arrays, "eigenvalues", "<f4", k, context))
    py_setitem(fields, "_pd_eigenvectors", host_archive_shape(arrays, "eigenvectors", "<f4", Python.tuple(n, k), context))
    py_setitem(fields, "_pd_diag", host_archive_count(arrays, "diag", "<f4", n, context))
    py_setitem(fields, "_pd_centroids", host_archive_shape(arrays, "centroids", "<f4", Python.tuple(clusters, k), context))
    py_setitem(fields, "labels_", host_archive_count(arrays, "labels", "<i4", n, context))
    py_setitem(fields, "_fit_X", host_archive_shape(arrays, "x", "<f4", Python.tuple(n, nf), context) if String(affinity) == "nearest_neighbors" else PythonObject(None))
    if String(affinity) == "nearest_neighbors":
        py_setitem(fields, "n_features_in_", nf)
    py_setitem(fields, "n_components_", k)
    return model


def host_archive_umap(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var meta = host_archive_count(arrays, "meta", "<i8", 7, context, True)
    var controls = host_archive_count(arrays, "controls", "<f4", 6, context, True)
    var neighbors = classical_i64(meta, 0)
    var components = classical_i64(meta, 1)
    var epochs = classical_i64(meta, 2)
    var seed = classical_i64(meta, 3)
    var negatives = classical_i64(meta, 4)
    var automatic_epochs = classical_i64(meta, 5) != 0
    if neighbors < 2 or components < 1 or components > 32 or seed < 0 or negatives < 0 or negatives > 2147483647:
        raise Error("saved UMAP integer parameters are outside the supported ranges")
    if (automatic_epochs and epochs != 0) or (not automatic_epochs and epochs < 1):
        raise Error("saved UMAP epochs do not match the automatic-epochs flag")
    var values = controls._mv
    for i in range(6):  # driver-loop: six saved scalar configuration fields
        if not isfinite(Float64(py=values[i])):
            raise Error("saved UMAP controls must be finite float32 values")
    var min_dist = Float64(py=values[0])
    var spread = Float64(py=values[1])
    var mix = Float64(py=values[2])
    var connectivity = Float64(py=values[3])
    var rate = Float64(py=values[4])
    var repulsion = Float64(py=values[5])
    if rate <= 0 or repulsion < 0 or min_dist < 0 or min_dist > spread or spread <= 0 or mix < 0 or mix > 1 or connectivity < 0:
        raise Error("saved UMAP scalar configuration is inconsistent")
    var metric = _archive_string(arrays, "metric", context)
    var metrics = context["host_archive_umap_metrics"]
    if not py_contains(metrics, metric):
        raise Error("unsupported saved UMAP metric")
    var code = Int(py=metrics[metric])
    var saved_code = -1
    var power = Float64(2.0)
    var a = Float64(0.0)
    var b = Float64(0.0)
    if py_contains(arrays, "extras"):
        var extras = host_archive_count(arrays, "extras", "<f8", 4, context, True)
        var value = classical_f64(extras, 0)
        if not isfinite(value) or value < -9223372036854775808.0 or value >= 9223372036854775808.0:
            raise Error("invalid saved UMAP metric code")
        saved_code = Int(value)
        power = classical_f64(extras, 1)
        a = classical_f64(extras, 2)
        b = classical_f64(extras, 3)
    if code != saved_code:
        raise Error("saved UMAP metric code does not match its metric")
    if code == 9:
        if not isfinite(power) or power <= 0 or power > 3.4028234663852886e38 or Float64(Float32(power)) != power:
            raise Error("saved UMAP Minkowski power must survive float32 validation")
    elif power != 2.0:
        raise Error("saved UMAP metric does not read its nondefault power")
    var have_ab = a > 0 and b > 0
    if have_ab:
        if not isfinite(a) or not isfinite(b) or a > 3.4028234663852886e38 or b > 3.4028234663852886e38 or Float64(Float32(a)) != a or Float64(Float32(b)) != b:
            raise Error("saved UMAP a and b must survive float32 validation")
    elif a != 0.0 or b != 0.0:
        raise Error("saved UMAP a and b must be positive together or both absent")
    var initialization = _archive_string(arrays, "init", context)
    if String(initialization) == "array":
        initialization = PythonObject("spectral")
    if String(initialization) != "spectral" and String(initialization) != "random" and String(initialization) != "pca":
        raise Error("unsupported saved UMAP initialization")
    var training = _archive_exact(arrays, "training", "<f4", context)
    var embedding = _archive_exact(arrays, "embedding", "<f4", context)
    if Int(py=training.ndim) != 2 or Int(py=embedding.ndim) != 2 or training.shape[0] != embedding.shape[0] or Int(py=embedding.shape[1]) != components:
        raise Error("saved UMAP training and embedding shapes do not match")
    var model = host_archive_object(cls, arrays, context)
    var fields = sequential_fields(model)
    host_archive_mode(arrays, fields, context)
    py_setitem(fields, "n_neighbors", neighbors)
    py_setitem(fields, "n_components", components)
    py_setitem(fields, "min_dist", min_dist)
    py_setitem(fields, "spread", spread)
    py_setitem(fields, "n_epochs", PythonObject(None) if automatic_epochs else PythonObject(epochs))
    py_setitem(fields, "random_state", seed)
    py_setitem(fields, "set_op_mix_ratio", mix)
    py_setitem(fields, "local_connectivity", connectivity)
    py_setitem(fields, "metric", metric)
    py_setitem(fields, "init", initialization)
    py_setitem(fields, "learning_rate", rate)
    py_setitem(fields, "repulsion_strength", repulsion)
    py_setitem(fields, "negative_sample_rate", negatives)
    py_setitem(fields, "a", PythonObject(a) if have_ab else PythonObject(None))
    py_setitem(fields, "b", PythonObject(b) if have_ab else PythonObject(None))
    py_setitem(fields, "metric_kwds", Python.dict(p=power) if saved_code == 9 else PythonObject(None))
    py_setitem(fields, "target_n_neighbors", -1)
    py_setitem(fields, "target_metric", "categorical")
    py_setitem(fields, "target_weight", 0.5)
    py_setitem(fields, "densmap", False)
    py_setitem(fields, "output_metric", "euclidean")
    py_setitem(fields, "_transform_training", training)
    py_setitem(fields, "_transform_embedding", embedding)
    py_setitem(fields, "_transform_config", Python.tuple(neighbors, components, epochs, min_dist, spread, mix, connectivity, seed, rate, repulsion, negatives, code, power, a if have_ab else Float64(0.0), b if have_ab else Float64(0.0)))
    py_setitem(fields, "_transform_mode", fields["numeric_mode"])
    var store = py_call(context["array_type"], "f")
    _ = py_call(store.frombytes, py_call(embedding._mv.cast, "B"))
    py_setitem(fields, "embedding_", classical_wrap_array(store, embedding.shape, "<f4", String(embedding.order), context))
    py_setitem(fields, "n_features_in_", training.shape[1])
    py_setitem(fields, "input_copied_", classical_i64(meta, 6) != 0)
    return model


def host_archive_build(path: PythonObject, context: PythonObject,
                       requested_class: PythonObject = PythonObject(None)) raises -> PythonObject:
    # This deliberately separate private entry allows each schema to be
    # qualified without introducing a native-to-Python fallback. Public
    # host_model keeps its existing full-format contract until all builders
    # have been implemented and qualified together.
    var arrays = classical_archive_read(path, context)
    var fmt = String(_archive_string(arrays, "format", context))
    var estimator = _archive_string(arrays, "estimator", context)
    var formats = context["host_archive_formats"]
    var classes = py_get(formats, fmt)
    if classes is None:
        raise Error("native host archive builder has no qualified schema for " + fmt)
    var cls = py_get(classes, estimator)
    if cls is None:
        raise Error("model estimator does not match its saved archive format")
    if requested_class is not None:
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
        model = classical_archive_build(cls, arrays, path, String(estimator) == "Ridge", context)
    elif fmt == "mojolearn-pca-1" or fmt == "mojolearn-tsvd-1":
        model = host_archive_decomposition(cls, arrays, fmt == "mojolearn-pca-1", context)
    elif fmt == "mojolearn-cd-1":
        model = host_archive_cd(cls, arrays, String(estimator) == "Lasso", context)
    elif fmt == "mojolearn-kernel-ridge-1" or fmt == "mojolearn-nystroem-1" or fmt == "mojolearn-rbf-sampler-1":
        model = host_archive_kernel(cls, arrays, 0 if fmt == "mojolearn-kernel-ridge-1" else (1 if fmt == "mojolearn-nystroem-1" else 2), context)
    elif fmt == "mojolearn-scaler-1" and String(estimator) == "MinMaxScaler":
        model = host_archive_minmax(cls, arrays, context)
    elif fmt == "mojolearn-scaler-1" and String(estimator) == "StandardScaler":
        model = scaler_archive_build(cls, arrays, path, context)
    elif fmt == "mojolearn-cholesky-1":
        model = host_archive_cholesky(cls, arrays, context)
    elif fmt == "mojolearn-embedding-1":
        model = host_archive_embedding(cls, arrays, context)
    elif fmt == "mojolearn-kmeans-1":
        model = host_archive_kmeans(cls, arrays, context)
    elif fmt == "mojolearn-gmm-1":
        model = host_archive_mixture(cls, arrays, context)
    elif fmt == "mojolearn-kde-1" or fmt == "mojolearn-dbscan-1":
        model = host_archive_density(cls, arrays, fmt == "mojolearn-kde-1", context)
    elif fmt == "mojolearn-agglomerative-1":
        model = host_archive_agglomerative(cls, arrays, context)
    elif fmt == "mojolearn-logistic-1":
        model = host_archive_logistic(cls, arrays, context)
    elif fmt == "mojolearn-arima-1" or fmt == "mojolearn-arima-2":
        model = host_archive_arima(cls, arrays, fmt == "mojolearn-arima-2", context)
    elif fmt == "mojolearn-holtwinters-1":
        model = host_archive_holtwinters(cls, arrays, context)
    elif fmt == "mojolearn-ivf-flat-1":
        model = host_archive_ivf(cls, arrays, context)
    elif fmt == "mojolearn-hdbscan-2":
        model = host_archive_hdbscan(cls, arrays, context)
    elif fmt == "mojolearn-radius-1":
        model = host_archive_neighbors(cls, arrays, 3, context)
    elif fmt == "mojolearn-knn-1":
        model = host_archive_neighbors(cls, arrays, 0 if String(estimator) == "NearestNeighbors" else (1 if String(estimator) == "KNeighborsClassifier" else 2), context)
        # Native equivalent of the original loader's _mojolearn_after_fit:
        # a newly restored CPU neighbor model owns an empty query holder.
        py_setitem(sequential_fields(model), "_resident_holder", Python.dict())
    elif fmt == "mojolearn-iforest-1":
        model = host_archive_iforest(cls, arrays, context)
    elif fmt == "mojolearn-spectral-1":
        model = host_archive_spectral(cls, arrays, context)
    elif fmt == "mojolearn-umap-1":
        model = host_archive_umap(cls, arrays, context)
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
        py_setitem(fields, "estimator", estimator)
    return model
