# SPDX-License-Identifier: Apache-2.0
# KernelRidge `fit`/`predict` and Nystroem `fit`/`transform`/`fit_transform`,
# one native call each (python/mojolearn/kernel_methods.py). This owner selects
# the import-connected binding and makes the public methods'
# refusals in their wording and order (kernel, degree, gamma, coef0, alpha,
# the chi2 kernels' negative input, the precomputed square), cross-checks the
# binary's compile-time mode, allocates the outputs, runs the binding's
# kernel_ridge_fit(_ptr) / kernel_ridge_predict(_ptr) / nystroem_fit(_ptr) /
# nystroem_transform with the argument lists
# `bindings/_mojolearn_kernel_methods.mojo` documents, and stores the fit.
# The same kernels on the same inputs: the bits do not move.
#
# Lane migrate-classical-b2: KernelRidge.predict reads X_fit_ and dual_coef_
# from a device copy minted at the first predict after a fit
# (`kernel_ridge_resident_prepare`, kernel_methods/krr_resident.mojo)
# instead of uploading both per call; Nystroem.fit_transform uploads X once
# (`nystroem_fit_transform_ptr`); RBFSampler.fit / fit_transform are one
# native call, gamma='scale' rounded from the binding's exact limbs in exact
# integer math (core/scale_gamma_exact.mojo). A binding without a door
# takes the entries it has, as before.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_caller_frame import native_frame_globals
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_validate import classical_view
from bindings.classical_warnings import classical_warning
from bindings.metrics_native_support import metric_float, metric_is_real, metric_is_bool
from bindings.host_inference_scalars import host_inference_primitive_number
from core.cast_in import CAST_KIND_SIGNED
from providers.python.estimator_binding_pyapi import estimator_arguments, estimator_binding
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_fit_begin, estimator_fit_end
from providers.python.kernel_public_lifecycle_pyapi import kernel_public_admit, kernel_public_finite, kernel_public_ignored_target, kernel_public_result, kernel_public_column_target, kernel_public_refusal, kernel_public_feature_snapshot, kernel_public_fit_complete
from providers.python.kernel_method_descriptor_pyapi import kernel_method_provider, kernel_method_execute, kernel_method_metadata
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_execute, neural_descriptor_adopt, neural_descriptor_resource_close,
    neural_descriptor_info, neural_descriptor_entry,
)
from core.scale_gamma_exact import scale_gamma_from_limbs, SGX_BAD
from bindings.mcls_cluster import mcls_cluster_operation
from bindings.parallel_plan_door import plan_devices, plan_enter, plan_exit, plan_field, plan_require_identical
from core.device_plan import DEVICE_PLAN_WORKER, DEVICE_PLAN_CHOLESKY
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from bindings.classical_attributes import classical_connect_numpy
from bindings.mode_metadata_native import _fields
from bindings.public_attributes_native import _string_keys
from bindings.family_common import (
    family_repr, family_call,
    family_entry, family_array,
    family_owned_snapshot, family_f32_snapshot, family_snapshot_min, family_sample_weight_snapshot, family_field, family_f32,
)

comptime _KERNEL_PRECOMPUTED = 4
comptime _KERNEL_CHI2 = 7
comptime _KERNEL_ADDITIVE_CHI2 = 8


def _km_warning_location() raises -> PythonObject:
    # Match warnings.warn's current Python caller through the supported frame
    # API; sys._getframe/f_code would invoke user audit hooks during execution.
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyEval_GetFrame")()
    if not raw:
        return PythonObject(None)
    var frame = PythonObject(from_borrowed=raw)
    raw = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetCode")(frame._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var code = PythonObject(from_owned=raw)
    var line = cpy.lib.get_function[Int32]("PyFrame_GetLineNumber")(frame._obj_ptr)
    var module = py_get(native_frame_globals(frame), "__name__", "")
    if py_type(module) is not py_type(PythonObject("")):
        module = PythonObject("")
    return Python.tuple(code.co_filename, Int(line), module)


def _km_int(value: PythonObject, name: String) raises -> Int:
    # Admit the exact scalar before conversion: int subclasses can run __int__.
    if py_type(value) is not py_type(PythonObject(0)):
        raise Error("TypeError: mojolearn kernel methods: " + name + " must be a native int")
    return Int(py=value)


def _km_number(value: PythonObject, name: String, context: PythonObject) raises -> Float64:
    if not metric_is_real(value, context) or metric_is_bool(value, context):
        raise Error("TypeError: mojolearn kernel methods: " + name + " must be a native real number")
    return metric_float(value, context)


def _km_seed(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Native NumPy scalar seeds remain supported without their number slots.
    var word = host_inference_primitive_number(value, context, True)
    return PythonObject(_km_int(word, "random_state"))


def _km_bool(value: PythonObject, name: String) raises -> Bool:
    if py_type(value) is not py_type(PythonObject(True)):
        raise Error("TypeError: mojolearn kernel methods: " + name + " must be a native bool")
    return Bool(py=value)


def _km_tuple(value: PythonObject, count: Int, name: String) raises -> PythonObject:
    if py_type(value) is not py_type(Python.tuple()):
        raise Error("TypeError: mojolearn kernel methods: " + name + " must be a native tuple")
    if len(value) != count:
        raise Error("ValueError: mojolearn kernel methods: " + name + " has an invalid size")
    return value


def _km_params(fields: PythonObject, ridge: Bool, context: PythonObject) raises -> PythonObject:
    var params = _km_tuple(family_field(fields, "_kernel_params"), 5, "_kernel_params")
    var last = PythonObject(_km_number(params[4], "alpha", context)) if ridge else PythonObject(_km_int(params[4], "random_state"))
    return Python.tuple(_km_int(params[0], "kernel"), _km_int(params[1], "degree"),
                        _km_number(params[2], "gamma", context), _km_number(params[3], "coef0", context), last)


def _km_i32_snapshot(value: PythonObject, count: Int, context: PythonObject) raises -> ArrayMetadata:
    var view = classical_view(value, context, "component_indices_")
    var shape = array_shape_once(view, context)
    if len(shape) != 1 or shape[0] != count:
        raise Error("ValueError: component_indices_ must match the fitted component count")
    var result = array_metadata_once(view, shape^, "component_indices_", context)
    if result.source.kind != CAST_KIND_SIGNED or result.source.itemsize != 4 or not result.c_order:
        raise Error("TypeError: component_indices_ must be a contiguous native int32 array")
    return result^


def _km_kernel_code(kernel: PythonObject, where: String, precomputed: Bool, context: PythonObject) raises -> Int:
    if py_type(kernel) is py_type(PythonObject("")):
        var name = String(kernel)
        if name == "precomputed":
            if precomputed:
                return _KERNEL_PRECOMPUTED
            raise Error("mojolearn " + where + ": kernel='precomputed' is refused by name; the "
                        + "lane forms the kernel matrix on the device and takes no matrix "
                        + "from the caller (kernel_methods/NOT_IMPLEMENTED.tsv)")
        if name == "linear":
            return 0
        if name == "poly" or name == "polynomial":
            return 1
        if name == "rbf":
            return 2
        if name == "sigmoid":
            return 3
        if name == "laplacian":
            return 5
        if name == "cosine":
            return 6
        if name == "chi2":
            return 7
        if name == "additive_chi2":
            return 8
        raise Error("mojolearn " + where + ": kernel must be a supported kernel name, got " + name)
    return _km_int(kernel, "kernel")


def _km_real(value: PythonObject, name: String, where: String, context: PythonObject) raises -> PythonObject:
    return PythonObject(_km_number(value, where + "." + name, context))


def _km_gamma(gamma: PythonObject, n_features: Int, where: String, kernel: Int, context: PythonObject) raises -> PythonObject:
    """`kernel_methods._gamma_for`: None is 1 / n_features (1.0 for chi2)."""
    if gamma is None and kernel == _KERNEL_CHI2:
        return PythonObject(1.0)
    if gamma is None:
        return PythonObject(1.0 / Float64(n_features))
    return _km_real(gamma, "gamma", where, context)


def _km_chi2_input(x: ArrayMetadata, kernel: Int, where: String) raises:
    """scikit-learn's chi2 kernels refuse negative input, before any device work."""
    if (kernel == _KERNEL_CHI2 or kernel == _KERNEL_ADDITIVE_CHI2) and x.count > 0 and Float64(family_snapshot_min(x)) < 0.0:
        raise Error("mojolearn " + where + ": X contains negative values (the chi2 kernels need x >= 0)")


def _km_extension(fields: PythonObject, binding: PythonObject, where: String, context: PythonObject) raises -> Tuple[PythonObject, PythonObject, Int]:
    """`_KernelMethodBase._extension`: the binary's own compile-time mode
    must be the one requested."""
    var b = native_loaded_module("builtins")
    var want = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    if want is None:
        want = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(want) is not py_type(PythonObject("")):
        raise Error("TypeError: numeric_mode must be a native string or None")
    want = want.strip().lower()
    var selected = String(want)
    if selected != "identical" and selected != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var native = kernel_method_provider(fields, binding, selected == "identical")
    var getter = PythonObject(None)
    if native is None:
        getter = neural_descriptor_entry(binding, "kernel_methods_numeric_mode", False)
    if native is not None or getter is not None:
        var got = kernel_method_metadata(native, 13) if native is not None else _km_int(family_call(getter), "numeric mode")
        var codes = context["family_km_mode_codes"]
        var wanted = py_get(codes, want, PythonObject(None))
        if wanted is None or Int(py=wanted) != got:
            var owner_name = py_get(py_fields(binding), "__name__", PythonObject("kernel methods binding"))
            if py_type(owner_name) is not b.str:
                raise Error("TypeError: kernel methods binding name must be a native string")
            raise Error("RuntimeError: mojolearn " + where + ": numeric_mode=" + family_repr(want) + " was requested "
                        + "but " + String(owner_name) + " reports compile-time mode code "
                        + String(got) + "; the binary and the directory it sits in disagree, rebuild "
                        + "it with bash bindings/build_kernel_methods.sh")
    var flags = kernel_method_metadata(native, 14) if native is not None else 0
    return (binding, native, flags)


def _km_pointer_entry(binding: PythonObject, pointer: String, name: String, context: PythonObject) raises -> PythonObject:
    """The pointer entry when the binding registers it, else the list entry
    (the same addresses and parameters)."""
    var entry = neural_descriptor_entry(binding, pointer, False)
    if entry is not None:
        return entry
    return family_entry(binding, name, context)


def _km_degree(fields: PythonObject, where: String) raises -> Int:
    return _km_int(family_field(fields, "degree"), where + ".degree")


def _krr_fit(model: PythonObject, args: PythonObject, context: PythonObject, x_meta: ArrayMetadata) raises:
    """`KernelRidge.fit(X, y, sample_weight)`."""
    classical_connect_numpy(context)
    var where = String("KernelRidge")
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var x = x_meta.keep
    var n = x_meta.source.rows
    var d = x_meta.source.cols
    if args[2] is None:
        raise Error("ValueError: KernelRidge requires y to be passed, but the target y is None.")
    var target = family_f32_snapshot(args[2], -1, "y", context)
    kernel_public_finite(target, "y")
    var rank = len(target.shape)
    var t = 1
    var squeeze = True
    if rank == 2:
        t = target.source.cols
        squeeze = kernel_public_column_target(args[2], target, context)
    elif rank != 1:
        raise Error("mojolearn " + where + ": y must be 1-D or 2-D, got " + String(rank) + "-D")
    if target.source.rows != n:
        raise Error("mojolearn " + where + ": y has " + String(target.source.rows) + " rows, X has " + String(n))
    var y = target.keep
    # `_kp`: the kernel, degree, gamma and coef0, in that order.
    var kernel = _km_kernel_code(family_field(fields, "kernel"), where, True, context)
    var degree = _km_degree(fields, where)
    var gamma = _km_gamma(family_field(fields, "gamma"), d, where, kernel, context)
    var coef0 = _km_real(family_field(fields, "coef0"), "coef0", where, context)
    _km_chi2_input(x_meta, kernel, where)
    if kernel == _KERNEL_PRECOMPUTED and d != n:
        raise Error("mojolearn " + where + ": kernel='precomputed' needs the square "
                    + "n x n kernel matrix as X, got " + String(n) + " x " + String(d))
    var alpha = _km_real(family_field(fields, "alpha"), "alpha", where, context)
    var dual = family_array(Python.tuple(n) if squeeze else Python.tuple(n, t), "<f4", context)
    var scalars = family_array(Python.tuple(1), "<f8", context)
    var weights = PythonObject(None)
    var weights_owner = PythonObject(None)
    var weights_address = 0
    if args[3] is not None:
        var snapshot = family_sample_weight_snapshot(args[3], n, False, context, wide=True)
        weights = snapshot.keep
        weights_owner = snapshot.owner
        weights_address = snapshot.source.addr
    var admission = _km_extension(fields, args[0], where, context)
    var binding = admission[0]
    var native = admission[1]
    var flags = admission[2]
    var fit = PythonObject(None)
    if native is None:
        fit = _km_pointer_entry(binding, "kernel_ridge_fit_ptr", "kernel_ridge_fit", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_fit_binding:
    # x, y, dual_out, scalars_out[, sw]; n, d, t, kernel, degree, gamma, coef0, alpha.
    var addresses = Python.list(PythonObject(x_meta.source.addr), PythonObject(target.source.addr), classical_address(dual),
                                classical_address(scalars))
    _ = dual  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = scalars  # keep the Python object alive through the native call above
    if weights is not None:
        _ = py_call(addresses.append, PythonObject(weights_address))
        _ = weights  # keep the Python object alive through the native call above
        _ = scalars  # keep the Python object alive through the native call above
        _ = dual  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(t), PythonObject(kernel),
                             PythonObject(degree), gamma, coef0, alpha)
    var info = kernel_method_execute(native, "kernel_ridge_fit_ptr" if (flags & 1) != 0 else "kernel_ridge_fit", addresses, params) if native is not None else family_call(fit, addresses, params)
    _ = weights  # keep the Python object alive through the native call above
    _ = scalars  # keep the Python object alive through the native call above
    _ = dual  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var kept = family_owned_snapshot(x_meta, context)
    _ = y^
    _ = weights^
    _ = weights_owner^
    _ = target^
    py_setitem(fields, "X_fit_", kept)
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "n_targets_", PythonObject(t))
    py_setitem(fields, "_squeeze", PythonObject(squeeze))
    py_setitem(fields, "dual_coef_", dual)
    py_setitem(fields, "info_", PythonObject(_km_int(info, "info_")))
    py_setitem(fields, "_kernel_params", Python.tuple(PythonObject(kernel), PythonObject(degree), gamma, coef0, alpha))
    _ = x^
    # The last fit's resident arrays are released; the next predict uploads
    # the new ones once (`_km_resident_krr`).
    _km_release(model, "kernel_ridge_resident_release")
    py_setitem(fields, _HOLDER, Python.dict())
    _ = x_meta


def _krr_predict(model: PythonObject, args: PythonObject, context: PythonObject,
                 xq_meta: ArrayMetadata, resident: Bool = True) raises -> PythonObject:
    """`KernelRidge.predict(X)`: K(X, X_fit_) @ dual_coef_. `resident=False`
    (the multi-GPU driver) skips the device copy of the fit: a copy minted on
    a device plan's root would outlive the call on that device. The resident
    and the per-call predict run the same kernels on the same bytes."""
    classical_connect_numpy(context)
    var where = String("KernelRidge")
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    if not py_contains(fields, PythonObject("dual_coef_")):
        raise Error("mojolearn " + where + ": call fit before predict")
    var xq = xq_meta.keep
    var q = xq_meta.source.rows
    var d = xq_meta.source.cols
    var fitted = _km_int(family_field(fields, "n_features_in_"), "n_features_in_")
    if d != fitted:
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit had " + String(fitted))
    var x_fit_obj = family_field(fields, "X_fit_")
    var x_fit_meta = family_f32_snapshot(x_fit_obj, 2, "X_fit_", context)
    var x_fit = x_fit_meta.keep
    var n = x_fit_meta.source.rows
    var t = _km_int(family_field(fields, "n_targets_"), "n_targets_")
    var kp = _km_params(fields, True, context)
    var kernel = Int(py=kp[0])
    _km_chi2_input(xq_meta, kernel, where)
    var dual_obj = family_field(fields, "dual_coef_")
    var dual_meta = family_f32_snapshot(dual_obj, -1, "dual_coef_", context)
    var dual = dual_meta.keep
    if t < 1 or x_fit_meta.source.cols != d or dual_meta.count != n * t:
        raise Error("ValueError: KernelRidge fitted arrays do not match the fitted dimensions")
    var squeeze = _km_bool(family_field(fields, "_squeeze"), "_squeeze")
    var out = family_array(Python.tuple(q) if squeeze else Python.tuple(q, t), "<f4", context)
    var admission = _km_extension(fields, args[0], where, context)
    var binding = admission[0]
    var native = admission[1]
    var flags = admission[2]
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(t), kp[0], kp[1], kp[2], kp[3], kp[4],
                             PythonObject(_km_int(family_field(fields, "info_"), "info_")), PythonObject(q))
    var handle = PythonObject(None)
    if resident:
        handle = _km_resident_krr(model, binding, x_fit_obj, dual_obj, x_fit_meta, dual_meta, n, d, t, context, native, flags)
    if handle is not None:
        # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_predict_resident_binding:
        # handle; x_new, out; the predict's ten params.
        var output_addresses = Python.list(PythonObject(xq_meta.source.addr), classical_address(out))
        if native is not None:
            _ = kernel_method_execute(native, "kernel_ridge_predict_resident", output_addresses, params, Int(py=handle))
        else:
            _ = family_call(family_entry(binding, "kernel_ridge_predict_resident", context), handle, output_addresses, params)
        _ = out  # keep the Python object alive through the native call above
        _ = xq  # keep the Python object alive through the native call above
        _ = x_fit^
        _ = dual^
        _ = xq^
        _ = xq_meta
        _ = x_fit_meta^
        _ = dual_meta^
        return out
    var predict = PythonObject(None)
    if native is None:
        predict = _km_pointer_entry(binding, "kernel_ridge_predict_ptr", "kernel_ridge_predict", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_predict_binding:
    # x_fit, dual, x_new, out; n, d, t, kernel, degree, gamma, coef0, alpha, info, q.
    var addresses = Python.list(PythonObject(x_fit_meta.source.addr), PythonObject(dual_meta.source.addr), PythonObject(xq_meta.source.addr),
                                classical_address(out))
    _ = out  # keep the Python object alive through the native call above
    if native is not None:
        _ = kernel_method_execute(native, "kernel_ridge_predict_ptr" if (flags & 1) != 0 else "kernel_ridge_predict", addresses, params)
    else:
        _ = family_call(predict, addresses, params)
    _ = out  # keep the Python object alive through the native call above
    _ = x_fit^
    _ = dual^
    _ = xq^
    _ = xq_meta
    _ = x_fit_meta^
    _ = dual_meta^
    return out


def _nys_fit(model: PythonObject, args: PythonObject, context: PythonObject,
             x_meta: ArrayMetadata, transform: Bool = False) raises -> PythonObject:
    """`Nystroem.fit(X)`: the basis rows and the normalization. With
    `transform` (`fit_transform`), returns the transform of X: one
    `nystroem_fit_transform_ptr` call (X uploaded once) where the binding
    registers it, else None (the caller transforms)."""
    classical_connect_numpy(context)
    var where = String("Nystroem")
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var x = x_meta.keep
    var n = x_meta.source.rows
    var d = x_meta.source.cols
    var kernel = _km_kernel_code(family_field(fields, "kernel"), where, False, context)
    var degree = _km_degree(fields, where)
    var components_param = family_field(fields, "n_components")
    var q = _km_int(components_param, where + ".n_components")
    if q < 1:
        raise Error("mojolearn " + where + ": n_components must be positive, got " + String(q))
    if q > n:
        # scikit-learn's rule and warning (kernel_approximation.py,
        # Nystroem.fit): n_components becomes n_samples (the full kernel).
        classical_warning("n_components > n_samples. This is not possible.\n"
                          + "n_components was set to n_samples, which results"
                          + " in inefficient evaluation of the full kernel.",
                          native_loaded_module("builtins").UserWarning, context,
                          _km_warning_location())
        q = n
    var gamma = _km_gamma(family_field(fields, "gamma"), d, where, kernel, context)
    _km_chi2_input(x_meta, kernel, where)
    var coef0 = _km_real(family_field(fields, "coef0"), "coef0", where, context)
    var seed = _km_seed(family_field(fields, "random_state"), context)
    var components = family_array(Python.tuple(q, d), "<f4", context)
    var indices = family_array(Python.tuple(q), "<i4", context)
    var normalization = family_array(Python.tuple(q, q), "<f4", context)
    var eigenvalues = family_array(Python.tuple(q), "<f4", context)
    var eigenvectors = family_array(Python.tuple(q, q), "<f4", context)
    var scalars = family_array(Python.tuple(1), "<f8", context)
    var admission = _km_extension(fields, args[0], where, context)
    var binding = admission[0]
    var native = admission[1]
    var flags = admission[2]
    var joint = PythonObject(None)
    if transform:
        if native is not None:
            if (flags & 6) == 6:
                joint = PythonObject(True)
        else:
            joint = _km_door(binding, "nystroem_fit_transform_ptr", context)
    var out = PythonObject(None)
    var fit = joint
    if joint is None and native is None:
        fit = _km_pointer_entry(binding, "nystroem_fit_ptr", "nystroem_fit", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::nystroem_fit_binding:
    # x, components_out, indices_out, normalization_out, eigenvalues_out,
    # eigenvectors_out, scalars_out; n, d, kernel, degree, gamma, coef0, q, seed.
    var addresses = Python.list(PythonObject(x_meta.source.addr), classical_address(components), classical_address(indices),
                                classical_address(normalization), classical_address(eigenvalues),
                                classical_address(eigenvectors), classical_address(scalars))
    _ = indices  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = eigenvalues  # keep the Python object alive through the native call above
    _ = normalization  # keep the Python object alive through the native call above
    _ = scalars  # keep the Python object alive through the native call above
    _ = eigenvectors  # keep the Python object alive through the native call above
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(kernel), PythonObject(degree),
                             gamma, coef0, PythonObject(q), seed)
    if joint is not None:
        # ORDER MATCHES nystroem_fit_transform_ptr_binding: the seven fit
        # addresses, then the n x q transform.
        out = family_array(Python.tuple(n, q), "<f4", context)
        _ = py_call(addresses.append, classical_address(out))
        _ = out  # keep the Python object alive through the native call above
        _ = scalars  # keep the Python object alive through the native call above
        _ = eigenvectors  # keep the Python object alive through the native call above
        _ = eigenvalues  # keep the Python object alive through the native call above
        _ = normalization  # keep the Python object alive through the native call above
        _ = indices  # keep the Python object alive through the native call above
        _ = components  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    var fit_name = "nystroem_fit_transform_ptr" if joint is not None else ("nystroem_fit_ptr" if (flags & 2) != 0 else "nystroem_fit")
    var sweeps = kernel_method_execute(native, fit_name, addresses, params) if native is not None else family_call(fit, addresses, params)
    _ = out  # keep the Python object alive through the native call above
    _ = scalars  # keep the Python object alive through the native call above
    _ = eigenvectors  # keep the Python object alive through the native call above
    _ = eigenvalues  # keep the Python object alive through the native call above
    _ = normalization  # keep the Python object alive through the native call above
    _ = indices  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "components_", components)
    py_setitem(fields, "component_indices_", indices)
    py_setitem(fields, "normalization_", normalization)
    py_setitem(fields, "eigenvalues_", eigenvalues)
    py_setitem(fields, "eigenvectors_", eigenvectors)
    py_setitem(fields, "sweeps_", PythonObject(_km_int(sweeps, "sweeps_")))
    py_setitem(fields, "_kernel_params", Python.tuple(PythonObject(kernel), PythonObject(degree), gamma, coef0, seed))
    if transform and out is None:
        out = family_array(Python.tuple(n, q), "<f4", context)
        var transform_addresses = Python.list(classical_address(components), classical_address(indices),
            classical_address(normalization), classical_address(eigenvalues), classical_address(eigenvectors),
            PythonObject(x_meta.source.addr), classical_address(out))
        var transform_parameters = Python.list(q, d, kernel, degree, gamma, coef0, seed, sweeps, n)
        if native is not None:
            _ = kernel_method_execute(native, "nystroem_transform", transform_addresses, transform_parameters)
        else:
            _ = family_call(family_entry(binding, "nystroem_transform", context), transform_addresses, transform_parameters)
        _ = components
        _ = indices
        _ = normalization
        _ = eigenvalues
        _ = eigenvectors
        _ = out
    _ = x^
    _ = x_meta
    return out


def _nys_transform(model: PythonObject, args: PythonObject, context: PythonObject, x_meta: ArrayMetadata) raises -> PythonObject:
    """`Nystroem.transform(X)`: K(X, components_) @ normalization_.T."""
    classical_connect_numpy(context)
    var where = String("Nystroem")
    var fields = py_fields(model)
    if not py_contains(fields, PythonObject("components_")):
        raise Error("mojolearn " + where + ": call fit before transform")
    var x = x_meta.keep
    var m = x_meta.source.rows
    var d = x_meta.source.cols
    var fitted = _km_int(family_field(fields, "n_features_in_"), "n_features_in_")
    if d != fitted:
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit had " + String(fitted))
    var components_meta = family_f32_snapshot(family_field(fields, "components_"), 2, "components_", context)
    var components = components_meta.keep
    var q = components_meta.source.rows
    var kp = _km_params(fields, False, context)
    _km_chi2_input(x_meta, Int(py=kp[0]), where)
    var out = family_array(Python.tuple(m, q), "<f4", context)
    var indices_meta = _km_i32_snapshot(family_field(fields, "component_indices_"), q, context)
    var indices = indices_meta.keep
    var normalization_meta = family_f32_snapshot(family_field(fields, "normalization_"), 2, "normalization_", context)
    var normalization = normalization_meta.keep
    var eigenvalues_meta = family_f32_snapshot(family_field(fields, "eigenvalues_"), 1, "eigenvalues_", context)
    var eigenvalues = eigenvalues_meta.keep
    var eigenvectors_meta = family_f32_snapshot(family_field(fields, "eigenvectors_"), 2, "eigenvectors_", context)
    var eigenvectors = eigenvectors_meta.keep
    if (components_meta.source.cols != d or normalization_meta.source.rows != q or normalization_meta.source.cols != q
            or eigenvalues_meta.count != q or eigenvectors_meta.source.rows != q or eigenvectors_meta.source.cols != q):
        raise Error("ValueError: Nystroem fitted arrays do not match the fitted dimensions")
    var admission = _km_extension(fields, args[0], where, context)
    var binding = admission[0]
    var native = admission[1]
    var flags = admission[2]
    var transform = PythonObject(None)
    if native is None:
        transform = family_entry(binding, "nystroem_transform", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::nystroem_transform_binding:
    # components, indices, normalization, eigenvalues, eigenvectors, x, out;
    # q, d, kernel, degree, gamma, coef0, seed, sweeps, m.
    var addresses = Python.list(PythonObject(components_meta.source.addr), PythonObject(indices_meta.source.addr),
                                PythonObject(normalization_meta.source.addr), PythonObject(eigenvalues_meta.source.addr),
                                PythonObject(eigenvectors_meta.source.addr), PythonObject(x_meta.source.addr), classical_address(out))
    _ = indices  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = eigenvalues  # keep the Python object alive through the native call above
    _ = normalization  # keep the Python object alive through the native call above
    _ = eigenvectors  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var params = Python.list(PythonObject(q), PythonObject(d), kp[0], kp[1], kp[2], kp[3], kp[4],
                             PythonObject(_km_int(family_field(fields, "sweeps_"), "sweeps_")), PythonObject(m))
    if native is not None:
        _ = kernel_method_execute(native, "nystroem_transform", addresses, params)
    else:
        _ = family_call(transform, addresses, params)
    _ = eigenvectors  # keep the Python object alive through the native call above
    _ = eigenvalues  # keep the Python object alive through the native call above
    _ = normalization  # keep the Python object alive through the native call above
    _ = indices  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = components^
    _ = normalization^
    _ = eigenvalues^
    _ = eigenvectors^
    _ = indices^
    _ = x^
    _ = components_meta^
    _ = indices_meta^
    _ = normalization_meta^
    _ = eigenvalues_meta^
    _ = eigenvectors_meta^
    _ = x_meta
    return out


def _rbf_transform(model: PythonObject, args: PythonObject, context: PythonObject, x_meta: ArrayMetadata) raises -> PythonObject:
    """`RBFSampler.transform(X)`: scale_ * cos(X @ random_weights_ + random_offset_)."""
    classical_connect_numpy(context)
    var where = String("RBFSampler")
    var fields = py_fields(model)
    if not py_contains(fields, PythonObject("random_weights_")):
        raise Error("mojolearn " + where + ": call fit before transform")
    var x = x_meta.keep
    var m = x_meta.source.rows
    var d = x_meta.source.cols
    var fitted = _km_int(family_field(fields, "n_features_in_"), "n_features_in_")
    if d != fitted:
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit had " + String(fitted))
    var weights_meta = family_f32_snapshot(family_field(fields, "random_weights_"), 2, "random_weights_", context)
    var weights = weights_meta.keep
    var q = weights_meta.source.cols
    var params_in = _km_tuple(family_field(fields, "_params"), 2, "_params")
    var params = Python.tuple(_km_number(params_in[0], "gamma", context), _km_int(params_in[1], "random_state"))
    var offset_meta = family_f32_snapshot(family_field(fields, "random_offset_"), 1, "random_offset_", context)
    var offset = offset_meta.keep
    if weights_meta.source.rows != d or offset_meta.count != q:
        raise Error("ValueError: RBFSampler fitted arrays do not match the fitted dimensions")
    var out = family_array(Python.tuple(m, q), "<f4", context)
    var admission = _km_extension(fields, args[0], where, context)
    var binding = admission[0]
    var native = admission[1]
    var flags = admission[2]
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::rbf_sampler_transform_binding:
    # weights, offset, x, out; d, q, gamma, seed, sigma, scale, m.
    var addresses = Python.list(PythonObject(weights_meta.source.addr), PythonObject(offset_meta.source.addr), PythonObject(x_meta.source.addr), classical_address(out))
    var parameters = Python.list(PythonObject(d), PythonObject(q), params[0], params[1], PythonObject(_km_number(family_field(fields, "sigma_"), "sigma_", context)), PythonObject(_km_number(family_field(fields, "scale_"), "scale_", context)), PythonObject(m))
    if native is not None:
        _ = kernel_method_execute(native, "rbf_sampler_transform", addresses, parameters)
    else:
        _ = family_call(family_entry(binding, "rbf_sampler_transform", context), addresses, parameters)
    _ = offset  # keep the Python object alive through the native call above
    _ = weights  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = weights^
    _ = offset^
    _ = x^
    _ = weights_meta^
    _ = offset_meta^
    _ = x_meta
    return out


comptime _HOLDER = "_resident_holder"
comptime _PY2MOJO_SCALE_GAMMA = 1


def _km_door(binding: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """The binding's compiled entry `name`, or None where it registers none
    (`_km_pointer_entry`'s lookup)."""
    return neural_descriptor_entry(binding, name, False)


def _km_release(model: PythonObject, release: String) raises:
    """Drop the resident handle, if held, through the minting binding's
    `release` entry. Quiet on a binding that cannot be reached any more and
    on a handle already released (a copied instance)."""
    var b = native_loaded_module("builtins")
    var holder = py_get(py_fields(model), PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        return
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is None:
        return
    if py_type(held) is not py_type(Python.tuple()):
        raise Error("TypeError: kernel resident ownership must be a native tuple")
    if len(held) != 3 and len(held) != 5:
        raise Error("ValueError: kernel resident ownership has an invalid size")
    py_setitem(holder, "index", PythonObject(None))
    try:
        if len(held) == 5:
            neural_descriptor_resource_close(held[3], [])
            return
        _ = py_call(neural_descriptor_entry(held[2], release), PythonObject(_km_int(held[1], "resident handle")))
    except:
        pass


def _km_resident_krr(model: PythonObject, binding: PythonObject, x_fit_obj: PythonObject, dual_obj: PythonObject,
                     x_fit: ArrayMetadata, dual: ArrayMetadata, n: Int, d: Int, t: Int,
                     context: PythonObject, native: PythonObject, flags: Int) raises -> PythonObject:
    """The handle of the device copy of the fit's X and dual, uploaded at
    the first predict after a fit and reused while `X_fit_` and
    `dual_coef_` are the same objects (the held key keeps references to
    them) and the binding is the minting one; None where the binding has no
    resident door. Only the holder dictionary changes."""
    var prepare = PythonObject(None)
    if native is None:
        prepare = _km_door(binding, "kernel_ridge_resident_prepare", context)
        if prepare is None:
            return PythonObject(None)
    elif (flags & 1) == 0:
        return PythonObject(None)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var holder = py_get(fields, PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        holder = Python.dict()
        py_setitem(fields, _HOLDER, holder)
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is not None:
        if py_type(held) is not py_type(Python.tuple()):
            raise Error("TypeError: kernel resident ownership must be a native tuple")
        if len(held) != 3 and len(held) != 5:
            raise Error("ValueError: kernel resident ownership has an invalid size")
        var key = _km_tuple(held[0], 5, "resident key")
        _ = _km_int(held[1], "resident handle")
        if (key[0] is x_fit_obj and key[1] is dual_obj and held[2] is binding and _km_int(key[2], "resident rows") == n
                and _km_int(key[3], "resident features") == d and _km_int(key[4], "resident targets") == t
                and ((native is None and len(held) == 3) or (native is not None and len(held) == 5 and held[4] is native))):
            return held[1]
        _km_release(model, "kernel_ridge_resident_release")
    var addresses = Python.list(PythonObject(x_fit.source.addr), PythonObject(dual.source.addr))
    var parameters = Python.list(PythonObject(n), PythonObject(d), PythonObject(t))
    var key = Python.tuple(x_fit_obj, dual_obj, PythonObject(n), PythonObject(d), PythonObject(t))
    var handle = kernel_method_execute(native, "kernel_ridge_resident_prepare", addresses, parameters) if native is not None else family_call(prepare, addresses, parameters)
    if native is not None:
        var owner = neural_descriptor_adopt(native, Int(py=handle), 5, [])
        py_setitem(holder, "index", Python.tuple(key, handle, binding, owner, native))
    else:
        py_setitem(holder, "index", Python.tuple(key, handle, binding))
    return handle


def _km_scale_gamma(binding: PythonObject, x: ArrayMetadata, d: Int, context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`_scale_gamma.scale_gamma_x`: the binding's exact limbs of the float32
    cells, the one rounding in exact integer math."""
    var b = native_loaded_module("builtins")
    var flags_entry = PythonObject(None)
    var flags = 0
    if native is not None:
        flags = kernel_method_metadata(native, 15)
    else:
        flags_entry = neural_descriptor_entry(binding, "py2mojo_linear_flags", False)
        if flags_entry is not None:
            flags = _km_int(family_call(flags_entry), "linear flags")
    if (flags & _PY2MOJO_SCALE_GAMMA) == 0:
        raise Error("RuntimeError: mojolearn: this binding has no scale_gamma_limbs (an older binary or a "
                    + "-D MOJOLEARN_PY2MOJO_linear_OFF build); rebuild it")
    var n = x.count
    if n == 0:
        raise Error("gamma='scale' needs at least one cell of X")
    var words = family_array(Python.tuple(SGX_BAD + 1), "<i8", context)
    if native is not None:
        _ = neural_descriptor_execute(native, 16, [x.source.addr, Int(py=classical_address(words))], [n], [0.0])
    else:
        _ = family_call(neural_descriptor_entry(binding, "scale_gamma_limbs"), PythonObject(x.source.addr), PythonObject(n), classical_address(words))
    _ = words  # keep the Python object alive through the native call above
    var wp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(words)))
    _ = words  # keep the Python object alive through the native call above
    var limbs = List[Int64]()
    for i in range(SGX_BAD + 1):  # small-loop(limbs: the 28 fixed words of the binding)
        limbs.append(wp[i])
        _ = words  # keep the Python object alive through the native call above
    _ = words^
    return PythonObject(scale_gamma_from_limbs(limbs, n, d))


def _rbf_fit(model: PythonObject, args: PythonObject, transform: Bool, context: PythonObject, x_meta: ArrayMetadata) raises -> PythonObject:
    """`RBFSampler.fit(X)` (returns the model) or, with `transform`,
    `fit_transform(X)`: one `rbf_sampler_fit_transform_resident` call where
    the binding registers it, else the fit and then the transform."""
    classical_connect_numpy(context)
    var where = String("RBFSampler")
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var x = x_meta.keep
    var m = x_meta.source.rows
    var d = x_meta.source.cols
    var binding = args[0]
    # `_fit_params`: gamma, then n_components, then the seed.
    var gamma_param = family_field(fields, "gamma")
    var gamma: PythonObject
    var extension = PythonObject(None)
    var native = PythonObject(None)
    if py_type(gamma_param) is py_type(PythonObject("")) and String(gamma_param) == "scale":
        var admission = _km_extension(fields, binding, where, context)
        extension = admission[0]
        native = admission[1]
        gamma = _km_scale_gamma(extension, x_meta, d, context, native)
    else:
        gamma = _km_real(gamma_param, "gamma", where, context)
    var components_param = family_field(fields, "n_components")
    var q = _km_int(components_param, where + ".n_components")
    if q < 1:
        # BY NAME, BEFORE ANY BUFFER (the MI300X n_components=0 run, 2026-09-14).
        raise Error("mojolearn " + where + ": n_components must be positive, got " + String(q))
    var seed = _km_seed(family_field(fields, "random_state"), context)
    var weights = family_array(Python.tuple(d, q), "<f4", context)
    var offset = family_array(Python.tuple(q), "<f4", context)
    var scalars = family_array(Python.tuple(2), "<f8", context)
    if extension is None:
        var admission = _km_extension(fields, binding, where, context)
        extension = admission[0]
        native = admission[1]
    var out = PythonObject(None)
    var joint = PythonObject(None)
    if transform and native is None:
        joint = neural_descriptor_entry(extension, "rbf_sampler_fit_transform_resident", False)
    if joint is not None:
        # lane apple-fast-w2-kfeat MOJOLEARN_KM_FAST_RBF_RESIDENT
        # (kernel_methods/rbf_resident.mojo): the draws, the upload, the
        # projection and every download in one binding call and one wait.
        out = family_array(Python.tuple(m, q), "<f4", context)
        # ORDER MATCHES kernel_methods/rbf_resident.mojo::rbf_sampler_fit_transform_binding.
        _ = family_call(joint, Python.list(classical_address(weights), classical_address(offset),
                                           classical_address(scalars), PythonObject(x_meta.source.addr), classical_address(out)),
                        Python.list(PythonObject(d), PythonObject(q), gamma, seed, PythonObject(m)))
        _ = offset  # keep the Python object alive through the native call above
        _ = weights  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        _ = scalars  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    else:
        # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::rbf_sampler_fit_binding:
        # weights_out, offset_out, scalars_out; d, q, gamma, seed.
        var addresses = Python.list(classical_address(weights), classical_address(offset), classical_address(scalars))
        var parameters = Python.list(PythonObject(d), PythonObject(q), gamma, seed)
        if native is not None:
            _ = kernel_method_execute(native, "rbf_sampler_fit", addresses, parameters)
        else:
            _ = family_call(family_entry(extension, "rbf_sampler_fit", context), addresses, parameters)
        _ = scalars  # keep the Python object alive through the native call above
        _ = offset  # keep the Python object alive through the native call above
        _ = weights  # keep the Python object alive through the native call above
    var sp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(scalars)))
    _ = scalars  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "random_weights_", weights)
    py_setitem(fields, "random_offset_", offset)
    var sigma = sp[0]
    var scale = sp[1]
    py_setitem(fields, "sigma_", PythonObject(sigma))
    _ = scalars  # keep the Python object alive through the native call above
    py_setitem(fields, "scale_", PythonObject(scale))
    _ = scalars  # keep the Python object alive through the native call above
    py_setitem(fields, "_params", Python.tuple(gamma, seed))
    _ = scalars^
    if not transform:
        _ = x^
        _ = x_meta
        return model
    if out is None:
        out = family_array(Python.tuple(m, q), "<f4", context)
        var addresses = Python.list(classical_address(weights), classical_address(offset), PythonObject(x_meta.source.addr), classical_address(out))
        var parameters = Python.list(d, q, gamma, seed, sigma, scale, m)
        if native is not None:
            _ = kernel_method_execute(native, "rbf_sampler_transform", addresses, parameters)
        else:
            _ = family_call(family_entry(extension, "rbf_sampler_transform", context), addresses, parameters)
        _ = weights
        _ = offset
        _ = out
    _ = x^
    _ = x_meta
    return out


# ---------------------------------------------------------------------------
# The multi-GPU drivers parallel_classical.fit_kernel_method and
# apply_kernel_method, one native call each (lane/native-parallel-driver-
# 20261024, docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md). They used to send
# the estimator by pickle to ONE cooperative worker process whose visibility
# mask held the selected devices and whose MOJOLEARN_SVM_DEVICE_COUNT and
# MOJOLEARN_CHOLESKY_DEVICE_COUNT held their number. Here the caller's thread
# binds a device plan in the binding (core/device_plan.mojo) around the plain
# owners above: the kernel rows, the factor's trailing rows and the solve's
# columns go to the same owners with the same partition and join order, and
# the root to devices[0]. No bit moves: par-kernel-ridge and par-nystroem
# hash the same arrays (tools/identity_break.py).
# ---------------------------------------------------------------------------


def _km_par_admit(model: PythonObject, context: PythonObject) raises -> Bool:
    """`parallel_classical._admit_kernel_method`, in its order and wording;
    True for a KernelRidge, False for a Nystroem."""
    var classes = context["family_km_par_classes"]
    var cls = py_type(model)
    var ridge = cls is classes[0]
    if not ridge and cls is not classes[1]:
        raise Error("TypeError: requires mojolearn.KernelRidge or Nystroem")
    plan_require_identical(model, "parallel kernel methods require IDENTICAL numeric mode")
    var kernel = plan_field(model, "kernel")
    if py_type(kernel) is py_type(PythonObject("")) and String(kernel) == "laplacian":
        raise Error("ValueError: parallel kernel methods refuse kernel='laplacian' by name: its L1 "
                    + "distance matrix has no distributed row seam in this lane")
    return ridge


comptime _KM_REBUILD = "rebuild kernel methods binding for distributed kernel rows"


def _km_par_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`fit_kernel_method(estimator, X, y, devices=...)`: args (binding, X, y,
    devices), the binding the plain methods use (`_KernelMethodBase._bind`)."""
    var ridge = _km_par_admit(model, context)
    if ridge and args[2] is None:
        raise Error("ValueError: KernelRidge fit requires y")
    var devices = plan_devices(args[3], "km_fit", True, context)
    var binding = args[0]
    var x = family_f32_snapshot(args[1], 2, "X", context, True)
    plan_enter(binding, devices, _KM_REBUILD, DEVICE_PLAN_WORKER | DEVICE_PLAN_CHOLESKY)
    try:
        if ridge:
            _krr_fit(model, Python.tuple(binding, args[1], args[2], PythonObject(None)), context, x)
        else:
            _ = _nys_fit(model, Python.tuple(binding, args[1]), context, x)
    finally:
        plan_exit(binding)
    _ = x^
    return model


def _km_par_apply(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`apply_kernel_method(estimator, X, devices=...)`: args (binding, X,
    devices). The native result (float32 for KernelRidge.predict); the driver
    applies the plain method's public conversion for its own caller."""
    var ridge = _km_par_admit(model, context)
    var devices = plan_devices(args[2], "km_apply", True, context)
    var binding = args[0]
    var x = family_f32_snapshot(args[1], 2, "X", context, True)
    var out = PythonObject(None)
    plan_enter(binding, devices, _KM_REBUILD, DEVICE_PLAN_WORKER | DEVICE_PLAN_CHOLESKY)
    try:
        if ridge:
            out = _krr_predict(model, Python.tuple(binding, args[1]), context, x, False)
        else:
            out = _nys_transform(model, Python.tuple(binding, args[1]), context, x)
    finally:
        plan_exit(binding)
    _ = x^
    return out

def kernel_methods_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    if operation == "family_km_provider_info":
        var provider = py_get(py_fields(model), "_native_kernel_method_provider")
        if provider is None:
            raise Error("kernel methods provider was not admitted")
        return neural_descriptor_info(provider)
    if operation == "family_km_par_fit" or operation == "family_km_par_apply":
        var admitted = estimator_arguments(model, "_mojolearn_kernel_methods", args, context)
        if operation == "family_km_par_fit":
            return _km_par_fit(model, admitted, context)
        return _km_par_apply(model, admitted, context)
    if operation == "family_km_getstate":
        var state = py_call(native_loaded_module("builtins").dict, _fields(model))
        _ = py_call(state.pop, PythonObject(_HOLDER), PythonObject(None))
        _ = py_call(state.pop, "_native_kernel_method_provider", PythonObject(None))
        return state
    if operation == "family_km_setstate":
        var fields = _fields(model)
        if py_type(args[0]) is not native_loaded_module("builtins").dict:
            raise Error("TypeError: kernel state requires an ordinary native dictionary")
        _string_keys(args[0])
        _ = py_call(fields.update, args[0])
        py_setitem(fields, _HOLDER, Python.dict())
        return PythonObject(None)
    var fitting = operation.endswith("fit") or operation.endswith("fit_transform")
    var method = String("transform")
    if operation.endswith("fit_transform"):
        method = "fit_transform"
    elif fitting:
        method = "fit"
    elif operation.endswith("predict"):
        method = "predict"
    elif operation.endswith("score"):
        method = "score"
    var fields = kernel_public_admit(model, method, fitting, args[1], context)
    var x = family_f32_snapshot(args[1], 2, "X", context, True)
    if operation.startswith("family_km_rbf_"):
        kernel_public_finite(x, "X")
    if fitting and not operation.startswith("family_km_krr_"):
        kernel_public_ignored_target(args[2] if len(args) > 2 else PythonObject(None), x.source.rows, context)
    var feature_names = kernel_public_feature_snapshot(args[1], context) if fitting else PythonObject(None)
    var admitted = estimator_arguments(model, "_mojolearn_kernel_methods", args, context)
    var outside = estimator_outside()
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    try:
        result = _kernel_methods_operation(model, operation, admitted, context, x)
    except error:
        var failure = kernel_public_refusal(error, fitting, depth > 0, x.source.rows, x.source.cols, model)
        raise failure^
    finally:
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        kernel_public_fit_complete(model, feature_names)
    _ = feature_names^
    _ = x^
    return kernel_public_result(result, method == "predict", outside, context)


def _kernel_methods_operation(model: PythonObject, operation: String, args: PythonObject,
                              context: PythonObject, x: ArrayMetadata) raises -> PythonObject:
    if operation == "family_km_krr_fit":
        _krr_fit(model, args, context, x)
        return model
    if operation == "family_km_krr_predict":
        return _krr_predict(model, args, context, x)
    if operation == "family_km_krr_score":
        var truth = family_f32(args[2], -1, "y", context)
        var prediction = _krr_predict(model, args, context, x)
        var mode = py_get(py_fields(model), PythonObject("numeric_mode"), PythonObject(None))
        var metrics = estimator_binding(model, "_mojolearn_metrics", context)
        return mcls_cluster_operation("family_mcls_r2", Python.tuple(metrics, truth, prediction, args[3],
                                      PythonObject("uniform_average"), PythonObject(True), mode), context)
    if operation == "family_km_nys_fit":
        _ = _nys_fit(model, args, context, x)
        return model
    if operation == "family_km_nys_transform":
        return _nys_transform(model, args, context, x)
    if operation == "family_km_rbf_transform":
        return _rbf_transform(model, args, context, x)
    if operation == "family_km_nys_fit_transform":
        return _nys_fit(model, args, context, x, True)
    if operation == "family_km_rbf_fit":
        return _rbf_fit(model, args, False, context, x)
    if operation == "family_km_rbf_fit_transform":
        return _rbf_fit(model, args, True, context, x)
    raise Error("RuntimeError: unknown kernel methods operation " + operation)
