# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024):
# KernelDensity `fit`, `score_samples` and `score` (python/mojolearn/density.py).
# Each `family_xn_kde_*` operation validates, prepares, runs the family's
# compiled kernels and returns, with the former Python method's refusals,
# operations and order, so the bits do not move:
#
#   fit            `_release_resident_fit`, `as_f32_dense_c(X)` (and its
#                  `input_copied_` flag), the finiteness refusal, the
#                  binding's `kde_fast_immutable_fit` snapshot switch, then
#                  `check_sample_weight` (float32, scikit-learn's wording).
#   score_samples  `_scores(X, False)`: the fitted check, the query cast and
#   score          width refusal, the device-resident fit handle
#                  (`_resident_fit_handle`: the same key, the same
#                  `_validate_params` on a changed key, the same release and
#                  `kde_fit_prepare` argument list) and `kde_score_samples` /
#                  `kde_score_samples_resident` with the argument lists
#                  bindings/_mojolearn_estimators.mojo documents; `score` is
#                  the binding's fixed-order total (`want_total`).
#
# The shell hands over the binding `_bind("_mojolearn_estimators")` resolved
# for the model's numeric mode. ReferenceShardedNeighbors (also routed to
# this prefix by the lane's skeleton) is not served here: its dispatch is the
# Python `DevicePool` (python/mojolearn/_parallel_pool.py), whose workers run
# Python handlers.
from providers.python.estimator_binding_pyapi import estimator_arguments
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_begin, neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_contains, py_setitem, py_fields, py_type
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_pandas import classical_admit_value
from bindings.host_inference_scalars import host_inference_scalar
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_str, family_call, family_array, family_as_f32, family_sample_weight,
    family_field, family_is_str,
)
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense, _sparse_fields
from bindings.native_buffer_owner import native_buffer_view
from bindings.classical_labels import _any_contiguous_address
from bindings.neighbors_knn_native import knn_door
from core.family_cast_impl import family_all_finite_f32


def _kde_admit_state(model: PythonObject, context: PythonObject) raises:
    classical_check_layout(model, context)
    var fields = py_fields(model)
    var names = Python.tuple("bandwidth", "algorithm", "kernel", "metric", "atol", "rtol", "breadth_first",
        "leaf_size", "n_features_in_", "n_samples_fit_")
    for i in range(len(names)):  # small-loop(fixed KDE settings and fitted metadata)
        var value = py_get(fields, names[i])
        if value is not None:
            host_inference_scalar(value, context, True)
    var params = py_get(fields, "metric_params")
    if params is not None:
        if py_type(params) is not native_loaded_module("builtins").dict:
            raise Error("TypeError: KernelDensity metric_params must be an exact native dictionary")
        classical_admit_value(params, context)
        var values = py_call(native_loaded_module("builtins").tuple, params.values())
        for i in range(len(values)):  # driver-loop: admitted metric-parameter scalar values
            host_inference_scalar(values[i], context)
    var arrays = Python.tuple("_x", "_w")
    for i in range(len(arrays)):  # small-loop(fitted KDE buffer owners)
        var value = py_get(fields, arrays[i])
        if value is not None:
            classical_admit_value(value, context)


def _kde_sparse_f32(source: PythonObject, format: String) raises -> Bool:
    """Whether `source.toarray()` is already float32 (the matrix's own
    dtype, read from its stored fields as `host_sparse_dense` reads it)."""
    var fields = _sparse_fields(source)
    var dtype = fields["dtype"] if (format == "dok" or format == "lil") else fields["data"].dtype
    return String(dtype.kind) == "f" and Int(py=dtype.itemsize) == 4


def _kde_borrowed(source: PythonObject, x: PythonObject, context: PythonObject) raises -> Bool:
    """`_materialize` borrowed `source` and the cast kept it: the converted
    array reads the source's own memory (an Array, or a contiguous buffer
    export); a list, a strided export or a cast is a copy."""
    var b = native_loaded_module("builtins")
    if py_type(source) is b.list or py_type(source) is b.tuple:
        return False
    var address = Int(py=classical_address(x))
    if py_type(source) is context["array_class"]:
        return address == Int(py=classical_address(source))
    try:
        var view = native_buffer_view(source)
        var same = _any_contiguous_address(view) == address
        _ = view^
        return same
    except:
        return False


def _kde_dense(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_f32_dense_c(source, ndim=2, name="X")`: `(x, copied)`.
    A scipy.sparse matrix is densified exactly first (`toarray()`), and
    `copied` is the conversion's flag for the dense array it was handed, as
    `as_f32_c(obj.toarray())` reported it: a copy unless the matrix holds
    float32 (`toarray()` keeps its dtype)."""
    var sparse = host_sparse_format(source)
    if sparse.byte_length() > 0:
        if not py_contains(context, PythonObject("output_context")):
            py_setitem(context, "output_context", context)
        var dense = family_as_f32(host_sparse_dense(source, sparse, context), 2, "X", context)
        return Python.tuple(dense, PythonObject(not _kde_sparse_f32(source, sparse)))
    var x = family_as_f32(source, 2, "X", context)
    var copied = not _kde_borrowed(source, x, context)
    return Python.tuple(x, PythonObject(copied))


def _kde_copy(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`value.copy()` of a C-order float32 array: a new mojolearn Array."""
    var b = native_loaded_module("builtins")
    var result = family_array(py_call(b.tuple, value.shape), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(value))), count=Int(py=value.size))
    _ = result  # keep the Python object alive through the native call above
    _ = value  # keep the Python object alive through the native call above
    return result


def _kde_truth(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=family_call(native_loaded_module("builtins").bool, value))


def _kde_validate(fields: PythonObject) raises:
    """`KernelDensity._validate_params`: the constructor checks, in the
    Python method's order and wording."""
    var b = native_loaded_module("builtins")
    var op = native_loaded_module("operator")
    var bandwidth = family_field(fields, "bandwidth")
    var algorithm = family_field(fields, "algorithm")
    var kernel = family_field(fields, "kernel")
    var metric = family_field(fields, "metric")
    var atol = family_field(fields, "atol")
    var rtol = family_field(fields, "rtol")
    var breadth_first = family_field(fields, "breadth_first")
    var leaf_size = family_field(fields, "leaf_size")
    var metric_params = family_field(fields, "metric_params")
    if family_is_str(bandwidth):
        raise Error("NotImplementedError: mojolearn KernelDensity: bandwidth='" + family_str(bandwidth)
                    + "' is refused; compute "
                    + "the rule on the host and pass the float (sklearn's scott is "
                    + "n ** (-1/(d+4)), silverman is (n (d+2) / 4) ** (-1/(d+4)))")
    var width = family_call(b.float, bandwidth)
    var bw = Float64(py=width)
    if not (bw > 0.0) or bw != bw:
        raise Error("mojolearn KernelDensity: bandwidth must be positive, got " + family_repr(width))
    if not py_contains(Python.tuple(PythonObject(None), PythonObject("auto")), algorithm):
        raise Error("mojolearn KernelDensity: algorithm=" + family_repr(algorithm) + " is refused; this is "
                    + "cuML's brute-force KDE, there is no tree to choose")
    var knobs = _kde_truth(family_call(op.ne, atol, PythonObject(0)))
    if not knobs:
        knobs = _kde_truth(family_call(op.ne, rtol, PythonObject(0)))
    if not knobs:
        knobs = not _kde_truth(breadth_first)
    if not knobs:
        knobs = _kde_truth(family_call(op.ne, leaf_size, PythonObject(40)))
    if knobs:
        raise Error("mojolearn KernelDensity: atol/rtol/breadth_first/leaf_size "
                    + "are tree-traversal knobs; this is brute force (refused)")
    if not family_is_str(kernel) or not family_is_str(metric):
        raise Error("mojolearn KernelDensity: kernel and metric are names")
    # cuML's rule, `kernel_density.py:302-307`: at most ONE entry, and its
    # VALUE is taken regardless of its key.
    if _kde_truth(metric_params):
        if Int(py=family_call(b.len, metric_params)) != 1:
            raise Error("mojolearn KernelDensity: only metrics with a single "
                        + "argument are supported (cuML raises the same, "
                        + "kernel_density.py:304-306)")
        var values = family_call(b.list, family_call(family_call(b.getattr, metric_params, PythonObject("values"))))
        var p = family_call(b.float, values[0])
        if Float64(py=p) != 2.0:
            raise Error("NotImplementedError: mojolearn KernelDensity: metric_params=" + family_repr(metric_params)
                        + " asks for "
                        + "Minkowski p=" + family_repr(p) + ". The KERNEL computes any finite "
                        + "positive normal p (kde/impl/distance/distance.mojo, "
                        + "DEVIATION 552) and kde/estimator.mojo takes it as "
                        + "`metric_arg`; what is missing is a SLOT IN THE "
                        + "BINDING -- kde_score_samples_binding length-checks "
                        + "its params list at exactly 5 entries "
                        + "(bindings/_mojolearn_estimators.mojo:647, and the same "
                        + "check in bindings/_mojolearn_estimators_host.mojo:213). "
                        + "Pass p=2, "
                        + "or apply the two-line binding change in "
                        + "kde/README.md's HAND-OFF.")


def _kde_release(fields: PythonObject) raises:
    """`KernelDensity._release_resident_fit`: drop the device copy, if one is
    held, through the minting binding's `kde_fit_release`; quiet on a
    binding that cannot be reached any more and on a released handle."""
    var cached = py_get(fields, PythonObject("_resident"), PythonObject(None))
    if cached is None:
        return
    var b = native_loaded_module("builtins")
    if py_type(cached) is not b.tuple or len(cached) != 3 or py_type(cached[1]) is not b.int:
        raise Error("TypeError: KDE resident state must retain its native handle and module")
    var release = knn_door(cached[2], "kde_fit_release")
    py_setitem(fields, "_resident", PythonObject(None))
    try:
        if release is not None:
            _ = py_call(release, cached[1])
    except:
        pass


def _kde_handle(fields: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """`KernelDensity._resident_fit_handle(binding)`: the handle of the
    device-resident fit set, prepared on the first call and reused while the
    key (fit array address and shape, weights address, kernel, metric,
    bandwidth) is unchanged; None where the binding has no
    `kde_fit_prepare`."""
    var b = native_loaded_module("builtins")
    var op = native_loaded_module("operator")
    var prepare = knn_door(binding, "kde_fit_prepare")
    if prepare is None:
        return PythonObject(None)
    var x = family_field(fields, "_x")
    var w = family_field(fields, "_w")
    var w_address = PythonObject(0)
    if w is not None:
        w_address = classical_address(w)
    var key = Python.tuple(classical_address(x), py_call(b.tuple, x.shape), w_address,
                           family_call(b.str, family_field(fields, "kernel")),
                           family_call(b.str, family_field(fields, "metric")),
                           family_call(b.float, family_field(fields, "bandwidth")))
    var cached = py_get(fields, PythonObject("_resident"), PythonObject(None))
    if cached is not None:
        if py_type(cached) is not b.tuple or len(cached) != 3 or py_type(cached[1]) is not b.int:
            raise Error("TypeError: KDE resident state must retain its native handle and module")
        classical_admit_value(cached[0], context)
        _ = knn_door(cached[2], "kde_fit_release")
    if cached is not None and _kde_truth(family_call(op.eq, cached[0], key)):
        return cached[1]
    if cached is not None:
        _kde_validate(fields)
    _kde_release(fields)
    # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_fit_prepare_binding.
    var params = Python.list(family_call(b.int, x.shape[0]), family_call(b.int, x.shape[1]),
                             family_call(b.float, family_field(fields, "bandwidth")),
                             PythonObject(1 if w is not None else 0))
    var handle = family_call(b.int, family_call(prepare, key[0], key[2], params,
                                                family_field(fields, "kernel"), family_field(fields, "metric")))
    _ = x  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    # The minting binding travels with the handle: the release (above, and
    # the native finalizer when the instance is dropped) is its own
    # `kde_fit_release`.
    py_setitem(fields, "_resident", Python.tuple(key, handle, binding))
    return handle


def _kde_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`KernelDensity.fit(X, y=None, sample_weight=None)`."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var binding = args[0]
    # A refit drops the device copy of the previous fit set HERE, as
    # NearestNeighbors.fit does (DEVIATION 2921's key rule).
    _kde_release(fields)
    var dense = _kde_dense(args[1], context)
    var x = dense[0]
    py_setitem(fields, "input_copied_", dense[1])
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), Int(py=x.size)):
        # DEVIATION 604 refuses a non-finite fit set at the first
        # score_samples; refusing it here as well names the input at the
        # call that supplied it, as scikit-learn's fit does.
        raise Error("mojolearn KernelDensity: X contains a NaN or an infinity; "
                    + "refused by name at fit (DEVIATION 604)")
    _ = x  # keep the Python object alive through the native call above
    # Opt-in immutable snapshot: transport-only copy of caller bytes.
    # Residency never relies on external mutable pointer identity.
    var immutable = False
    var probe = knn_door(binding, "kde_fast_immutable_fit")
    if probe is not None:
        immutable = _kde_truth(family_call(probe))
    if immutable:
        x = _kde_copy(x, context)
        py_setitem(fields, "input_copied_", PythonObject(True))
    py_setitem(fields, "_x", x)
    py_setitem(fields, "n_features_in_", x.shape[1])
    py_setitem(fields, "n_samples_fit_", x.shape[0])
    var sample_weight = args[2]
    if sample_weight is not None:
        # scikit-learn's _check_sample_weight (shape, finite, >= 0, not all
        # zero, its wording) by the base binding's reductions
        var w = family_sample_weight(sample_weight, Int(py=x.shape[0]), False, context)
        if immutable:
            w = _kde_copy(w, context)
        py_setitem(fields, "_w", w)
    else:
        py_setitem(fields, "_w", PythonObject(None))


def _kde_scores(model: PythonObject, args: PythonObject, want_total: Bool,
                context: PythonObject) raises -> PythonObject:
    """`KernelDensity._scores(X, want_total)`: score_samples(X), or with
    `want_total` their sum, folded in the binding (on the device on a GPU
    install: `device_sum_f32_fixed`; the host binding's
    `host_sum_f32_fixed`, the same order)."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var binding = args[0]
    if not py_contains(fields, PythonObject("_x")):
        raise Error("mojolearn KernelDensity: call fit() first")
    var dense = _kde_dense(args[1], context)
    var q = dense[0]
    var features = family_field(fields, "n_features_in_")
    if _kde_truth(family_call(native_loaded_module("operator").ne, q.shape[1], features)):
        raise Error("mojolearn KernelDensity: X has " + family_str(q.shape[1]) + " features, "
                    + "fit saw " + family_str(features))
    var out = family_array(Python.tuple(q.shape[0]), "<f4", context)
    var w = family_field(fields, "_w")
    var want = PythonObject(1 if want_total else 0)
    # THE FIT SET STAYS ON THE DEVICE (DEVIATION 3003): the first call
    # validates and uploads it, every later call scores the device copy
    # through `kde_score_samples_resident`, validating and uploading the
    # queries only. A binding without that entry takes the per-call path.
    var handle = _kde_handle(fields, binding, context)
    var got = PythonObject(None)
    if handle is not None:
        var resident = knn_door(binding, "kde_score_samples_resident")
        # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_score_samples_resident_binding.
        got = family_call(resident, handle, classical_address(q), classical_address(out),
                          Python.list(family_call(b.int, q.shape[0]), family_call(b.int, features),
                                      family_call(b.float, family_field(fields, "bandwidth")), want),
                          family_field(fields, "kernel"), family_field(fields, "metric"))
        _ = q  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
    else:
        var x = family_field(fields, "_x")
        var w_address = PythonObject(0)
        if w is not None:
            w_address = classical_address(w)
        var entry = knn_door(binding, "kde_score_samples")
        # ORDER MATCHES bindings/_mojolearn_estimators.mojo::kde_score_samples_binding.
        got = family_call(entry, classical_address(x), classical_address(q), w_address, classical_address(out),
                          Python.list(family_call(b.int, x.shape[0]), family_call(b.int, q.shape[0]),
                                      family_call(b.int, features),
                                      family_call(b.float, family_field(fields, "bandwidth")),
                                      PythonObject(1 if w is not None else 0), want),
                          family_field(fields, "kernel"), family_field(fields, "metric"))
        _ = x  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        _ = q  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
    if want_total:
        return family_call(b.float, got)
    return out


def _neighbors_kde_operation(model: PythonObject, operation: String, var args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    _kde_admit_state(model, context)
    for i in range(1, len(args)):  # small-loop(public input owners before sparse/native preparation)
        if host_sparse_format(args[i]).byte_length() == 0:
            classical_admit_value(args[i], context)
    if operation != "family_xn_kde_release":
        args = estimator_arguments(model, "_mojolearn_estimators", args, context)
    if operation == "family_xn_kde_fit":
        _kde_fit(model, args, context)
        return model
    if operation == "family_xn_kde_score_samples":
        return _kde_scores(model, args, False, context)
    if operation == "family_xn_kde_score":
        return _kde_scores(model, args, True, context)
    if operation == "family_xn_kde_release":
        _kde_release(py_fields(model))
        return PythonObject(None)
    raise Error("RuntimeError: unknown neighbors operation " + operation)


def neighbors_kde_operation(model: PythonObject, operation: String, args: PythonObject,
                                context: PythonObject) raises -> PythonObject:
    if operation == "family_xn_kde_release":
        return _neighbors_kde_operation(model, operation, args, context)
    var call = neighbors_public_begin(model, operation, args, context)
    var result = PythonObject(None)
    try:
        result = _neighbors_kde_operation(model, operation, call.args, context)
    except error:
        raise neighbors_public_refusal(error, call, model)
    finally:
        neighbors_public_end(model, call)
    return neighbors_public_finish(model, operation, result, call, context)
