# SPDX-License-Identifier: Apache-2.0
# Native saved-model reconstruction. The format table is import-time ABI
# metadata, never Python loaders, constructors, or callbacks. The public
# router is switched only after every existing format has a native builder.
from std.python import Python, PythonObject
from std.memory import bitcast
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.sequential_description import sequential_fields
from bindings.classical_archive_read import classical_archive_read, classical_archive_build, _archive_exact, _archive_string
from bindings.classical_arrays import classical_i64, classical_f64, classical_wrap_array, classical_array, classical_typed, classical_address
from core.host_archive_impl import host_archive_copy_f32
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
    for i in range(len(mro)):  # driver-loop: connected class ancestry metadata
        var fields = py_call(b.type.__dict__["__dict__"].__get__, mro[i])
        if py_contains(fields, "__del__"):
            raise Error("TypeError: Python finalizer hooks are not accepted by native archive loading")
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
    else:
        raise Error("native host archive schema has not been implemented: " + fmt + "/" + String(estimator))
    if requested_class is None:
        var fields = sequential_fields(model)
        if fmt == "mojolearn-kernel-ridge-1" or fmt == "mojolearn-nystroem-1":
            var kernel = Int(py=fields["_kernel_params"][0])
            if kernel != 0 and kernel != 1 and kernel != 2 and kernel != 3 and kernel != 5:
                raise Error("ImportError: unsupported saved kernel code " + String(kernel))
        py_setitem(fields, "estimator", estimator)
    return model
