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
from core.scale_gamma_exact import scale_gamma_from_limbs, SGX_BAD
from bindings.mcls_cluster import mcls_cluster_operation
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    family_repr, family_is_str, family_is_int, family_is_real, family_type_name, family_call,
    family_entry, family_array, family_view, family_cast_view, family_as_f32, family_as_f32_dense,
    family_owned, family_min, family_sample_weight, family_field, family_f32,
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
    # The last fit's resident arrays are released; the next predict uploads
    # the new ones once (`_km_resident_krr`).
    _km_release(model, "kernel_ridge_resident_release")
    py_setitem(fields, _HOLDER, Python.dict())


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
    var x_fit_obj = family_field(fields, "X_fit_")
    var x_fit = family_as_f32(x_fit_obj, 2, "X_fit_", context)
    var n = Int(py=x_fit.shape[0])
    var t = Int(py=family_field(fields, "n_targets_"))
    var kp = family_field(fields, "_kernel_params")
    var kernel = Int(py=kp[0])
    _km_chi2_input(xq, kernel, where)
    var dual_obj = family_field(fields, "dual_coef_")
    var dual = family_as_f32(dual_obj, -1, "dual_coef_", context)
    var squeeze = Bool(py=py_call(b.bool, family_field(fields, "_squeeze")))
    var out = family_array(Python.tuple(q) if squeeze else Python.tuple(q, t), "<f4", context)
    var binding = _km_extension(model, args[0], where, context)
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(t), kp[0], kp[1], kp[2], kp[3], kp[4],
                             family_field(fields, "info_"), PythonObject(q))
    var handle = _km_resident_krr(model, binding, x_fit_obj, dual_obj, x_fit, dual, n, d, t, context)
    if handle is not None:
        # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_predict_resident_binding:
        # handle; x_new, out; the predict's ten params.
        _ = family_call(family_entry(binding, "kernel_ridge_predict_resident", context), handle,
                        Python.list(classical_address(xq), classical_address(out)), params)
        _ = x_fit^
        _ = dual^
        _ = xq^
        return out
    var predict = _km_pointer_entry(binding, "kernel_ridge_predict_ptr", "kernel_ridge_predict", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_predict_binding:
    # x_fit, dual, x_new, out; n, d, t, kernel, degree, gamma, coef0, alpha, info, q.
    var addresses = Python.list(classical_address(x_fit), classical_address(dual), classical_address(xq),
                                classical_address(out))
    _ = family_call(predict, addresses, params)
    _ = x_fit^
    _ = dual^
    _ = xq^
    return out


def _nys_fit(model: PythonObject, args: PythonObject, context: PythonObject,
             transform: Bool = False) raises -> PythonObject:
    """`Nystroem.fit(X)`: the basis rows and the normalization. With
    `transform` (`fit_transform`), returns the transform of X: one
    `nystroem_fit_transform_ptr` call (X uploaded once) where the binding
    registers it, else None (the caller transforms)."""
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
    var joint = PythonObject(None)
    if transform:
        joint = _km_door(binding, "nystroem_fit_transform_ptr", context)
    var out = PythonObject(None)
    var fit = joint
    if joint is None:
        fit = _km_pointer_entry(binding, "nystroem_fit_ptr", "nystroem_fit", context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::nystroem_fit_binding:
    # x, components_out, indices_out, normalization_out, eigenvalues_out,
    # eigenvectors_out, scalars_out; n, d, kernel, degree, gamma, coef0, q, seed.
    var addresses = Python.list(classical_address(x), classical_address(components), classical_address(indices),
                                classical_address(normalization), classical_address(eigenvalues),
                                classical_address(eigenvectors), classical_address(scalars))
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(kernel), PythonObject(degree),
                             gamma, coef0, PythonObject(q), seed)
    if joint is not None:
        # ORDER MATCHES nystroem_fit_transform_ptr_binding: the seven fit
        # addresses, then the n x q transform.
        out = family_array(Python.tuple(n, q), "<f4", context)
        _ = py_call(addresses.append, classical_address(out))
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
    return out


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


def _rbf_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`RBFSampler.transform(X)`: scale_ * cos(X @ random_weights_ + random_offset_)."""
    classical_connect_numpy(context)
    var where = String("RBFSampler")
    var fields = py_fields(model)
    if not py_contains(fields, PythonObject("random_weights_")):
        raise Error("mojolearn " + where + ": call fit before transform")
    var x = family_as_f32_dense(args[1], 2, "X", context)
    var m = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var fitted = family_field(fields, "n_features_in_")
    if d != Int(py=fitted):
        raise Error("mojolearn " + where + ": X has " + String(d) + " features, the fit had " + String(Int(py=fitted)))
    var weights = family_as_f32(family_field(fields, "random_weights_"), 2, "random_weights_", context)
    var q = Int(py=weights.shape[1])
    var params = family_field(fields, "_params")
    var offset = family_field(fields, "random_offset_")
    var out = family_array(Python.tuple(m, q), "<f4", context)
    var binding = _km_extension(model, args[0], where, context)
    # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::rbf_sampler_transform_binding:
    # weights, offset, x, out; d, q, gamma, seed, sigma, scale, m.
    _ = family_call(family_entry(binding, "rbf_sampler_transform", context),
                    Python.list(classical_address(weights), classical_address(offset), classical_address(x),
                                classical_address(out)),
                    Python.list(PythonObject(d), PythonObject(q), params[0], params[1], family_field(fields, "sigma_"),
                                family_field(fields, "scale_"), PythonObject(m)))
    _ = weights^
    _ = offset^
    _ = x^
    return out


comptime _HOLDER = "_resident_holder"
comptime _PY2MOJO_SCALE_GAMMA = 1


def _km_door(binding: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """The binding's compiled entry `name`, or None where it registers none
    (`_km_pointer_entry`'s lookup)."""
    var entry = py_call(Python.import_module("builtins").getattr, binding, PythonObject(name), PythonObject(None))
    if entry is None or not Bool(py=py_call(Python.import_module("builtins").bool, entry)):
        return PythonObject(None)
    if py_type(entry) is not context["family_builtin_type"]:
        raise Error("TypeError: mojolearn: " + name + " must be a compiled entry")
    return entry


def _km_release(model: PythonObject, release: String) raises:
    """Drop the resident handle, if held, through the minting binding's
    `release` entry. Quiet on a binding that cannot be reached any more and
    on a handle already released (a copied instance)."""
    var b = Python.import_module("builtins")
    var holder = py_get(py_fields(model), PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        return
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is None:
        return
    py_setitem(holder, "index", PythonObject(None))
    try:
        _ = py_call(py_call(b.getattr, held[2], PythonObject(release)), held[1])
    except:
        pass


def _km_resident_krr(model: PythonObject, binding: PythonObject, x_fit_obj: PythonObject, dual_obj: PythonObject,
                     x_fit: PythonObject, dual: PythonObject, n: Int, d: Int, t: Int,
                     context: PythonObject) raises -> PythonObject:
    """The handle of the device copy of the fit's X and dual, uploaded at
    the first predict after a fit and reused while `X_fit_` and
    `dual_coef_` are the same objects (the held key keeps references to
    them) and the binding is the minting one; None where the binding has no
    resident door. Only the holder dictionary changes."""
    var prepare = _km_door(binding, "kernel_ridge_resident_prepare", context)
    if prepare is None:
        return PythonObject(None)
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var holder = py_get(fields, PythonObject(_HOLDER), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        holder = Python.dict()
        py_setitem(fields, _HOLDER, holder)
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is not None:
        var key = held[0]
        if (key[0] is x_fit_obj and key[1] is dual_obj and held[2] is binding and Int(py=key[2]) == n
                and Int(py=key[3]) == d and Int(py=key[4]) == t):
            return held[1]
        _km_release(model, "kernel_ridge_resident_release")
    var handle = family_call(prepare, Python.list(classical_address(x_fit), classical_address(dual)),
                             Python.list(PythonObject(n), PythonObject(d), PythonObject(t)))
    py_setitem(holder, "index", Python.tuple(
        Python.tuple(x_fit_obj, dual_obj, PythonObject(n), PythonObject(d), PythonObject(t)), handle, binding))
    return handle


def _km_scale_gamma(binding: PythonObject, x: PythonObject, d: Int, context: PythonObject) raises -> PythonObject:
    """`_scale_gamma.scale_gamma_x`: the binding's exact limbs of the float32
    cells, the one rounding in exact integer math."""
    var b = Python.import_module("builtins")
    var flags_entry = py_call(b.getattr, binding, PythonObject("py2mojo_linear_flags"), PythonObject(None))
    var flags = 0
    if flags_entry is not None:
        flags = Int(py=family_call(b.int, family_call(flags_entry)))
    if (flags & _PY2MOJO_SCALE_GAMMA) == 0:
        raise Error("RuntimeError: mojolearn: this binding has no scale_gamma_limbs (an older binary or a "
                    + "-D MOJOLEARN_PY2MOJO_linear_OFF build); rebuild it")
    var n = Int(py=x.size)
    if n == 0:
        raise Error("gamma='scale' needs at least one cell of X")
    var words = family_array(Python.tuple(SGX_BAD + 1), "<i8", context)
    _ = family_call(family_call(b.getattr, binding, PythonObject("scale_gamma_limbs")), classical_address(x),
                    PythonObject(n), classical_address(words))
    var wp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(words)))
    var limbs = List[Int64]()
    for i in range(SGX_BAD + 1):  # small-loop(limbs: the 28 fixed words of the binding)
        limbs.append(wp[i])
    _ = words^
    return PythonObject(scale_gamma_from_limbs(limbs, n, d))


def _rbf_fit(model: PythonObject, args: PythonObject, transform: Bool, context: PythonObject) raises -> PythonObject:
    """`RBFSampler.fit(X)` (returns the model) or, with `transform`,
    `fit_transform(X)`: one `rbf_sampler_fit_transform_resident` call where
    the binding registers it, else the fit and then the transform."""
    classical_connect_numpy(context)
    var where = String("RBFSampler")
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var x = family_as_f32_dense(args[1], 2, "X", context)
    var m = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var binding = args[0]
    # `_fit_params`: gamma, then n_components, then the seed.
    var gamma_param = family_field(fields, "gamma")
    var gamma: PythonObject
    if family_is_str(gamma_param) and String(gamma_param) == "scale":
        gamma = _km_scale_gamma(_km_extension(model, binding, where, context), x, d, context)
    else:
        gamma = _km_real(gamma_param, "gamma", where)
    var components_param = family_field(fields, "n_components")
    if not family_is_int(components_param):
        raise Error("TypeError: mojolearn " + where + ": n_components must be an int")
    var q = Int(py=family_call(b.int, components_param))
    if q < 1:
        # BY NAME, BEFORE ANY BUFFER (the MI300X n_components=0 run, 2026-09-14).
        raise Error("mojolearn " + where + ": n_components must be positive, got " + String(q))
    var seed = family_call(b.int, family_field(fields, "random_state"))
    var weights = family_array(Python.tuple(d, q), "<f4", context)
    var offset = family_array(Python.tuple(q), "<f4", context)
    var scalars = family_array(Python.tuple(2), "<f8", context)
    var extension = _km_extension(model, binding, where, context)
    var out = PythonObject(None)
    var joint = PythonObject(None)
    if transform:
        joint = py_call(b.getattr, extension, PythonObject("rbf_sampler_fit_transform_resident"), PythonObject(None))
    if joint is not None:
        # lane apple-fast-w2-kfeat MOJOLEARN_KM_FAST_RBF_RESIDENT
        # (kernel_methods/rbf_resident.mojo): the draws, the upload, the
        # projection and every download in one binding call and one wait.
        out = family_array(Python.tuple(m, q), "<f4", context)
        # ORDER MATCHES kernel_methods/rbf_resident.mojo::rbf_sampler_fit_transform_binding.
        _ = family_call(joint, Python.list(classical_address(weights), classical_address(offset),
                                           classical_address(scalars), classical_address(x), classical_address(out)),
                        Python.list(PythonObject(d), PythonObject(q), gamma, seed, PythonObject(m)))
    else:
        # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::rbf_sampler_fit_binding:
        # weights_out, offset_out, scalars_out; d, q, gamma, seed.
        _ = family_call(family_entry(extension, "rbf_sampler_fit", context),
                        Python.list(classical_address(weights), classical_address(offset), classical_address(scalars)),
                        Python.list(PythonObject(d), PythonObject(q), gamma, seed))
    var sp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(scalars)))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "random_weights_", weights)
    py_setitem(fields, "random_offset_", offset)
    py_setitem(fields, "sigma_", PythonObject(sp[0]))
    py_setitem(fields, "scale_", PythonObject(sp[1]))
    py_setitem(fields, "_params", Python.tuple(gamma, seed))
    _ = scalars^
    if not transform:
        _ = x^
        return model
    if out is None:
        _ = x^
        return _rbf_transform(model, args, context)
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
    if operation == "family_km_krr_score":
        # `RegressorScoreMixin.score`: y as float32 (`as_f32_c(y, ndim=None,
        # name="y")`), the native predict, then `metrics.r2_score(y, pred,
        # sample_weight=..., numeric_mode=...)` through the metrics owner;
        # args[4] is the metrics binding `_get_binding` resolved.
        classical_connect_numpy(context)
        var truth = family_f32(args[2], -1, "y", context)
        var prediction = _krr_predict(model, args, context)
        var mode = py_get(py_fields(model), PythonObject("numeric_mode"), PythonObject(None))
        return mcls_cluster_operation("family_mcls_r2", Python.tuple(args[4], truth, prediction, args[3],
                                      PythonObject("uniform_average"), PythonObject(True), mode), context)
    if operation == "family_km_nys_fit":
        _ = _nys_fit(model, args, context)
        return model
    if operation == "family_km_nys_transform":
        return _nys_transform(model, args, context)
    if operation == "family_km_rbf_transform":
        return _rbf_transform(model, args, context)
    if operation == "family_km_nys_fit_transform":
        var joint = _nys_fit(model, args, context, True)
        if joint is not None:
            return joint
        return _nys_transform(model, args, context)
    if operation == "family_km_rbf_fit":
        return _rbf_fit(model, args, False, context)
    if operation == "family_km_rbf_fit_transform":
        return _rbf_fit(model, args, True, context)
    if operation == "family_km_getstate":
        # A pickle or a deepcopy carries no device handle (meaningful only
        # in the minting process); the copy uploads its own at first use.
        var state = py_call(Python.import_module("builtins").dict, py_fields(model))
        _ = py_call(state.pop, PythonObject(_HOLDER), PythonObject(None))
        return state
    if operation == "family_km_setstate":
        var fields = py_fields(model)
        _ = py_call(fields.update, args[0])
        py_setitem(fields, _HOLDER, Python.dict())
        return PythonObject(None)
    raise Error("RuntimeError: unknown kernel methods operation " + operation)
