# SPDX-License-Identifier: Apache-2.0
# GaussianRandomProjection / SparseRandomProjection's native operations
# (dispatched by bindings/decomp_native.mojo): fit's checks (the target
# dimension from johnson_lindenstrauss_min_dim's native bound, the density),
# X converted once, then one compiled driver (x_decomp/rp_owner.mojo:
# x_decomp_rp_fit / x_decomp_rp_project, the kit steps and switch routes the
# Python shell took, the same entries, streams and order, so the same bits),
# and the fitted matrix stored where the `components_` property reads it.
from std.math import sqrt
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_host_f32
from bindings.host_sparse_native import host_sparse_format
from bindings.decomp_inputs import decomp_input
from bindings.decomp_owner_common import (
    decomp_attr, decomp_check, decomp_entries, decomp_eq, decomp_f32, decomp_float, decomp_gemm_check, decomp_int,
    decomp_lt, decomp_le, decomp_seed, decomp_set_mode,
)
from bindings.jl_native import jl_min_dim


def _str(value: PythonObject) raises -> String:
    return String(py_call(Python.import_module("builtins").str, value))


def _components(fields: PythonObject) raises -> PythonObject:
    """The fitted matrix the `components_` property reads (`_rp_components`)."""
    return py_get(fields, "_rp_components")


def rp_fit(model: PythonObject, source: PythonObject, sparse: Bool, context: PythonObject) raises -> PythonObject:
    """`_RandomProjection.fit`; returns the converted X (kept for fit_transform)."""
    var fields = py_fields(model)
    decomp_set_mode(fields, context)
    var entries = decomp_entries(model, fields, Python.tuple("x_decomp_rp_fit"), context)
    var sparse_input = host_sparse_format(source).byte_length() > 0
    var name = String(py_type(model).__name__)
    var x = decomp_input(source, name, context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var requested = fields["n_components"]
    var auto = decomp_eq(requested, PythonObject("auto"))
    var kc: Int
    if auto:
        var eps = fields["eps"]
        var bound = jl_min_dim(PythonObject(n), eps, context)
        if decomp_le(bound, PythonObject(0)):
            raise Error("eps=" + _str(eps) + " and n_samples=" + String(n) + " lead to a target dimension of "
                        + _str(bound) + " which is invalid")
        if decomp_lt(PythonObject(d), bound):
            raise Error("eps=" + _str(eps) + " and n_samples=" + String(n) + " lead to a target dimension of "
                        + _str(bound) + " which is larger than the original space with n_features=" + String(d))
        kc = Int(py=bound)
    else:
        kc = decomp_int(requested)
        if kc <= 0:
            raise Error("n_components must be greater than 0, got " + String(kc))
    var density = Float64(0.0)
    if sparse:
        var given = fields["density"]
        density = 1.0 / sqrt(Float64(d)) if decomp_eq(given, PythonObject("auto")) else decomp_float(given)
        if not (density > 0 and density <= 1):
            raise Error("Expected density in range ]0, 1], got: " + _str(PythonObject(density)))
    var seed = decomp_seed(fields["random_state"])
    var inverse = Bool(py=fields["compute_inverse_components"])
    var components = decomp_f32(Python.tuple(kc, d), context)
    var inverse_m = decomp_f32(Python.tuple(d, kc), context) if inverse else PythonObject(None)
    var bad = Int(py=py_call_typed(entries["x_decomp_rp_fit"], Python.list(
        classical_address(x), n, d, kc, 1 if sparse else 0, seed, 1 if inverse else 0,
        1 if (not auto and not inverse and not sparse_input) else 0, classical_address(components),
        classical_address(inverse_m) if inverse else classical_address(components)), Python.list(density)))
    _ = x  # keep the Python object alive through the native call above
    _ = inverse_m  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    if bad >= 0:
        raise Error("X: input must be finite; NaN/inf are unsupported")
    _ = inverse_m  # keep the Python object alive through the native calls above
    _ = components  # keep the Python object alive through the native calls above
    _ = x  # keep the Python object alive through the native calls above
    if sparse:
        py_setitem(fields, "density_", PythonObject(density))
    py_setitem(fields, "n_components_", PythonObject(kc))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    # the `components_` setter's fields
    if py_contains(fields, "_rp_lazy"):
        _ = py_call(fields.pop, "_rp_lazy")
    py_setitem(fields, "_rp_components", components)
    if inverse:
        py_setitem(fields, "inverse_components_", inverse_m)
    return x


def _check(model: PythonObject, fields: PythonObject) raises:
    if _components(fields) is None:
        raise Error("RuntimeError: " + String(py_type(model).__name__) + " is not fitted; call fit first")


def rp_project(model: PythonObject, fields: PythonObject, source: PythonObject, x: PythonObject, inverse: Bool,
               context: PythonObject) raises -> PythonObject:
    """transform (X components_^T) or inverse_transform (X inverse_components_^T)."""
    var entries = decomp_entries(model, fields, Python.tuple("x_decomp_rp_project"), context)
    var m = Int(py=x.shape[0])
    var width = Int(py=x.shape[1])
    var matrix: PythonObject
    if inverse:
        if not py_contains(fields, "inverse_components_"):
            raise Error("inverse_transform needs compute_inverse_components=True")
        matrix = classical_host_f32(fields["inverse_components_"], 2, "inverse_components_", context)
        decomp_gemm_check(width, Int(py=matrix.shape[1]))
    else:
        var fitted = decomp_int(decomp_attr(model, fields, "n_features_in_"))
        if width != fitted:
            raise Error("X has " + String(width) + " features, but " + String(py_type(model).__name__)
                        + " is expecting " + String(fitted))
        matrix = classical_host_f32(_components(fields), 2, "components_", context)
    var rows = Int(py=matrix.shape[0])
    var out = decomp_f32(Python.tuple(m, rows), context)
    var tiled = not inverse and host_sparse_format(source).byte_length() == 0
    var flagged = Int(py=py_call_typed(entries["x_decomp_rp_project"], Python.list(
        classical_address(x), m, width, classical_address(matrix), rows, 1 if tiled else 0, classical_address(out))))
    _ = out  # keep the Python object alive through the native call above
    _ = matrix^
    if flagged != 0:
        raise Error("X: input must be finite; NaN/inf are unsupported")
    _ = out  # keep the Python object alive through the native calls above
    return out


def rp_query(model: PythonObject, source: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    _check(model, fields)
    var name = String(py_type(model).__name__)
    var x = decomp_input(source, name, context)
    var out = rp_project(model, fields, source, x, inverse, context)
    _ = x^
    return out
