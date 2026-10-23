# SPDX-License-Identifier: Apache-2.0
# KernelRidge `fit`/`predict` and Nystroem `fit`/`transform`/`fit_transform`,
# one native call each (python/mojolearn/kernel_methods.py). The shell hands
# over the binding `_bind` resolved; this owner makes the Python methods'
# refusals in their wording and order (kernel, degree, gamma, coef0, alpha,
# the chi2 kernels' negative input, the precomputed square), cross-checks the
# binary's compile-time mode, allocates the outputs, runs the binding's
# kernel_ridge_fit(_ptr) / kernel_ridge_predict(_ptr) / nystroem_fit(_ptr) /
# nystroem_transform with the argument lists
# `bindings/_mojolearn_kernel_methods.mojo` documents, and stores the fit.
# The same kernels on the same inputs: the bits do not move.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    family_repr, family_is_str, family_is_int, family_is_real, family_type_name, family_call,
    family_entry, family_array, family_view, family_cast_view, family_as_f32, family_as_f32_dense,
    family_owned, family_min, family_sample_weight, family_field,
)

comptime _KERNEL_PRECOMPUTED = 4
comptime _KERNEL_CHI2 = 7
comptime _KERNEL_ADDITIVE_CHI2 = 8


def _km_kernel_code(kernel: PythonObject, where: String, precomputed: Bool, context: PythonObject) raises -> Int:
    """`kernel_methods._kernel_code`."""
    var b = Python.import_module("builtins")
    var is_name = family_is_str(kernel)
    if is_name and String(kernel) == "precomputed":
        if precomputed:
            return _KERNEL_PRECOMPUTED
        raise Error("mojolearn " + where + ": kernel='precomputed' is refused by name; the "
                    + "lane forms the kernel matrix on the device and takes no matrix "
                    + "from the caller (kernel_methods/NOT_IMPLEMENTED.tsv)")
    if is_name:
        var table = context["family_km_kernels"]
        if not py_contains(table, kernel):
            raise Error("mojolearn " + where + ": kernel must be one of "
                        + family_repr(py_call(b.sorted, table)) + ", got " + family_repr(kernel))
        return Int(py=table[kernel])
    if not family_is_int(kernel):
        raise Error("TypeError: mojolearn " + where + ": kernel must be a name or a KM_KERNEL code, got "
                    + family_type_name(kernel))
    return Int(py=family_call(b.int, kernel))


def _km_real(value: PythonObject, name: String, where: String) raises -> PythonObject:
    """`kernel_methods._real`: a real number, as a float."""
    if not family_is_real(value):
        raise Error("TypeError: mojolearn " + where + ": " + name + " must be a real number, got "
                    + family_type_name(value))
    return family_call(Python.import_module("builtins").float, value)


def _km_gamma(gamma: PythonObject, n_features: Int, where: String, kernel: Int) raises -> PythonObject:
    """`kernel_methods._gamma_for`: None is 1 / n_features (1.0 for chi2)."""
    if gamma is None and kernel == _KERNEL_CHI2:
        return PythonObject(1.0)
    if gamma is None:
        return PythonObject(1.0 / Float64(n_features))
    return _km_real(gamma, "gamma", where)


def _km_chi2_input(x: PythonObject, kernel: Int, where: String) raises:
    """scikit-learn's chi2 kernels refuse negative input, before any device work."""
    if (kernel == _KERNEL_CHI2 or kernel == _KERNEL_ADDITIVE_CHI2) and Int(py=x.size) > 0 and Float64(family_min(x)) < 0.0:
        raise Error("mojolearn " + where + ": X contains negative values (the chi2 kernels need x >= 0)")


def _km_extension(model: PythonObject, binding: PythonObject, where: String, context: PythonObject) raises -> PythonObject:
    """`_KernelMethodBase._extension`: the binary's own compile-time mode
    must be the one requested."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var want = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    if want is None or not Bool(py=py_call(b.bool, want)):
        want = context["backend_fields"]["_DEFAULT_MODE"]
    var getter = py_call(b.getattr, binding, PythonObject("kernel_methods_numeric_mode"), PythonObject(None))
    if getter is not None:
        var got = Int(py=family_call(b.int, family_call(getter)))
        var codes = context["family_km_mode_codes"]
        var wanted = py_get(codes, want, PythonObject(None))
        if wanted is None or Int(py=wanted) != got:
            raise Error("RuntimeError: mojolearn " + where + ": numeric_mode=" + family_repr(want) + " was requested "
                        + "but " + String(py_call(b.getattr, binding, PythonObject("__name__"))) + " reports compile-time mode code "
                        + String(got) + "; the binary and the directory it sits in disagree, rebuild "
                        + "it with bash bindings/build_kernel_methods.sh")
    return binding


def _km_pointer_entry(binding: PythonObject, pointer: String, name: String, context: PythonObject) raises -> PythonObject:
    """The pointer entry when the binding registers it, else the list entry
    (the same addresses and parameters)."""
    var entry = py_call(Python.import_module("builtins").getattr, binding, PythonObject(pointer), PythonObject(None))
    if entry is not None and Bool(py=py_call(Python.import_module("builtins").bool, entry)):
        if py_type(entry) is not context["family_builtin_type"]:
            raise Error("TypeError: mojolearn: " + pointer + " must be a compiled entry")
        return entry
    return family_entry(binding, name, context)


def _km_degree(fields: PythonObject, where: String, message: String) raises -> Int:
    var degree = family_field(fields, "degree")
    if not family_is_int(degree):
        raise Error("TypeError: " + message)
    return Int(py=family_call(Python.import_module("builtins").int, degree))


def _krr_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`KernelRidge.fit(X, y, sample_weight)`."""
    classical_connect_numpy(context)
    var where = String("KernelRidge")
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var x = family_as_f32_dense(args[1], 2, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var target = family_view(args[2], -1, "y", context)
    var rank = Int(py=target.ndim)
    var t = 1
    var squeeze = True
    if rank == 2:
        t = Int(py=target.shape[1])
        squeeze = False
    elif rank != 1:
        raise Error("mojolearn " + where + ": y must be 1-D or 2-D, got " + String(rank) + "-D")
    if Int(py=target.shape[0]) != n:
        raise Error("mojolearn " + where + ": y has " + String(Int(py=target.shape[0])) + " rows, X has " + String(n))
    var y = family_cast_view(target, "y", context)
    # `_kp`: the kernel, degree, gamma and coef0, in that order.
    var kernel = _km_kernel_code(family_field(fields, "kernel"), where, True, context)
    var degree = _km_degree(fields, where, "mojolearn " + where + ": degree must be an int, got "
                            + family_type_name(family_field(fields, "degree")))
    var gamma = _km_gamma(family_field(fields, "gamma"), d, where, kernel)
    var coef0 = _km_real(family_field(fields, "coef0"), "coef0", where)
    _km_chi2_input(x, kernel, where)
    if kernel == _KERNEL_PRECOMPUTED and d != n:
        raise Error("mojolearn " + where + ": kernel='precomputed' needs the square "
                    + "n x n kernel matrix as X, got " + String(n) + " x " + String(d))
    var alpha = _km_real(family_field(fields, "alpha"), "alpha", where)
    var dual = family_array(Python.tuple(n) if squeeze else Python.tuple(n, t), "<f4", context)
    var scalars = family_array(Python.tuple(1), "<f8", context)
    var weights = PythonObject(None)
    if args[3] is not None:
        weights = family_sample_weight(args[3], n, False, context, wide=True)
    var binding = _km_extension(model, args[0], where, context)
    var fit = _km_pointer_entry(binding, "kernel_ridge_fit_ptr", "kernel_ridge_fit", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_fit_binding:
    # x, y, dual_out, scalars_out[, sw]; n, d, t, kernel, degree, gamma, coef0, alpha.
    var addresses = Python.list(classical_address(x), classical_address(y), classical_address(dual),
                                classical_address(scalars))
    if weights is not None:
        _ = py_call(addresses.append, classical_address(weights))
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(t), PythonObject(kernel),
                             PythonObject(degree), gamma, coef0, alpha)
    var info = family_call(fit, addresses, params)
    var kept = family_owned(x, context)
    _ = y^
    _ = weights^
    _ = target^
    py_setitem(fields, "X_fit_", kept)
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "n_targets_", PythonObject(t))
    py_setitem(fields, "_squeeze", PythonObject(squeeze))
    py_setitem(fields, "dual_coef_", dual)
    py_setitem(fields, "info_", family_call(b.int, info))
    py_setitem(fields, "_kernel_params", Python.tuple(PythonObject(kernel), PythonObject(degree), gamma, coef0, alpha))
    _ = x^


def _krr_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`KernelRidge.predict(X)`: K(X, X_fit_) @ dual_coef_."""
    classical_connect_numpy(context)
    var where = String("KernelRidge")
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    if not py_contains(fields, PythonObject("dual_coef_")):
        raise Error("mojolearn " + where + ": call fit before predict")
    var xq = family_as_f32_dense(args[1], 2, "X", context)
    var q = Int(py=xq.shape[0])
    var d = Int(py=xq.shape[1])
    var fitted = family_field(fields, "n_features_in_")
    if d != Int(py=fitted):
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit had " + String(Int(py=fitted)))
    var x_fit = family_as_f32(family_field(fields, "X_fit_"), 2, "X_fit_", context)
    var n = Int(py=x_fit.shape[0])
    var t = Int(py=family_field(fields, "n_targets_"))
    var kp = family_field(fields, "_kernel_params")
    var kernel = Int(py=kp[0])
    _km_chi2_input(xq, kernel, where)
    var dual = family_as_f32(family_field(fields, "dual_coef_"), -1, "dual_coef_", context)
    var squeeze = Bool(py=py_call(b.bool, family_field(fields, "_squeeze")))
    var out = family_array(Python.tuple(q) if squeeze else Python.tuple(q, t), "<f4", context)
    var binding = _km_extension(model, args[0], where, context)
    var predict = _km_pointer_entry(binding, "kernel_ridge_predict_ptr", "kernel_ridge_predict", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_predict_binding:
    # x_fit, dual, x_new, out; n, d, t, kernel, degree, gamma, coef0, alpha, info, q.
    var addresses = Python.list(classical_address(x_fit), classical_address(dual), classical_address(xq),
                                classical_address(out))
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(t), kp[0], kp[1], kp[2], kp[3], kp[4],
                             family_field(fields, "info_"), PythonObject(q))
    _ = family_call(predict, addresses, params)
    _ = x_fit^
    _ = dual^
    _ = xq^
    return out


def _nys_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`Nystroem.fit(X)`: the basis rows and the normalization."""
    classical_connect_numpy(context)
    var where = String("Nystroem")
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var x = family_as_f32_dense(args[1], 2, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var kernel = _km_kernel_code(family_field(fields, "kernel"), where, False, context)
    var degree = _km_degree(fields, where, "mojolearn " + where + ": degree must be an int")
    var components_param = family_field(fields, "n_components")
    if not family_is_int(components_param):
        raise Error("TypeError: mojolearn " + where + ": n_components must be an int")
    var q = Int(py=family_call(b.int, components_param))
    if q < 1:
        raise Error("mojolearn " + where + ": n_components must be positive, got " + String(q))
    if q > n:
        # scikit-learn's rule and warning (kernel_approximation.py,
        # Nystroem.fit): n_components becomes n_samples (the full kernel).
        _ = family_call(Python.import_module("warnings").warn,
                        PythonObject("n_components > n_samples. This is not possible.\n"
                                     + "n_components was set to n_samples, which results"
                                     + " in inefficient evaluation of the full kernel."))
        q = n
    var gamma = _km_gamma(family_field(fields, "gamma"), d, where, kernel)
    _km_chi2_input(x, kernel, where)
    var coef0 = _km_real(family_field(fields, "coef0"), "coef0", where)
    var seed = family_call(b.int, family_field(fields, "random_state"))
    var components = family_array(Python.tuple(q, d), "<f4", context)
    var indices = family_array(Python.tuple(q), "<i4", context)
    var normalization = family_array(Python.tuple(q, q), "<f4", context)
    var eigenvalues = family_array(Python.tuple(q), "<f4", context)
    var eigenvectors = family_array(Python.tuple(q, q), "<f4", context)
    var scalars = family_array(Python.tuple(1), "<f8", context)
    var binding = _km_extension(model, args[0], where, context)
    var fit = _km_pointer_entry(binding, "nystroem_fit_ptr", "nystroem_fit", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::nystroem_fit_binding:
    # x, components_out, indices_out, normalization_out, eigenvalues_out,
    # eigenvectors_out, scalars_out; n, d, kernel, degree, gamma, coef0, q, seed.
    var addresses = Python.list(classical_address(x), classical_address(components), classical_address(indices),
                                classical_address(normalization), classical_address(eigenvalues),
                                classical_address(eigenvectors), classical_address(scalars))
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(kernel), PythonObject(degree),
                             gamma, coef0, PythonObject(q), seed)
    var sweeps = family_call(fit, addresses, params)
    _ = x^
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "components_", components)
    py_setitem(fields, "component_indices_", indices)
    py_setitem(fields, "normalization_", normalization)
    py_setitem(fields, "eigenvalues_", eigenvalues)
    py_setitem(fields, "eigenvectors_", eigenvectors)
    py_setitem(fields, "sweeps_", family_call(b.int, sweeps))
    py_setitem(fields, "_kernel_params", Python.tuple(PythonObject(kernel), PythonObject(degree), gamma, coef0, seed))


def _nys_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Nystroem.transform(X)`: K(X, components_) @ normalization_.T."""
    classical_connect_numpy(context)
    var where = String("Nystroem")
    var fields = py_fields(model)
    if not py_contains(fields, PythonObject("components_")):
        raise Error("mojolearn " + where + ": call fit before transform")
    var x = family_as_f32_dense(args[1], 2, "X", context)
    var m = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var fitted = family_field(fields, "n_features_in_")
    if d != Int(py=fitted):
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit had " + String(Int(py=fitted)))
    var components = family_as_f32(family_field(fields, "components_"), 2, "components_", context)
    var q = Int(py=components.shape[0])
    var kp = family_field(fields, "_kernel_params")
    _km_chi2_input(x, Int(py=kp[0]), where)
    var out = family_array(Python.tuple(m, q), "<f4", context)
    var indices = family_field(fields, "component_indices_")
    var normalization = family_as_f32(family_field(fields, "normalization_"), 2, "normalization_", context)
    var eigenvalues = family_as_f32(family_field(fields, "eigenvalues_"), 1, "eigenvalues_", context)
    var eigenvectors = family_as_f32(family_field(fields, "eigenvectors_"), 2, "eigenvectors_", context)
    var binding = _km_extension(model, args[0], where, context)
    var transform = family_entry(binding, "nystroem_transform", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::nystroem_transform_binding:
    # components, indices, normalization, eigenvalues, eigenvectors, x, out;
    # q, d, kernel, degree, gamma, coef0, seed, sweeps, m.
    var addresses = Python.list(classical_address(components), classical_address(indices),
                                classical_address(normalization), classical_address(eigenvalues),
                                classical_address(eigenvectors), classical_address(x), classical_address(out))
    var params = Python.list(PythonObject(q), PythonObject(d), kp[0], kp[1], kp[2], kp[3], kp[4],
                             family_field(fields, "sweeps_"), PythonObject(m))
    _ = family_call(transform, addresses, params)
    _ = components^
    _ = normalization^
    _ = eigenvalues^
    _ = eigenvectors^
    _ = indices^
    _ = x^
    return out


def kernel_methods_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """The kernel methods' public operations: args[0] is the resolved binding."""
    if operation == "family_km_krr_fit":
        _krr_fit(model, args, context)
        return model
    if operation == "family_km_krr_predict":
        return _krr_predict(model, args, context)
    if operation == "family_km_nys_fit":
        _nys_fit(model, args, context)
        return model
    if operation == "family_km_nys_transform":
        return _nys_transform(model, args, context)
    if operation == "family_km_nys_fit_transform":
        _nys_fit(model, args, context)
        return _nys_transform(model, args, context)
    raise Error("RuntimeError: unknown kernel methods operation " + operation)
