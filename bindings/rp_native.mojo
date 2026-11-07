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
from bindings.native_loaded_module import native_loaded_module
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
from bindings.decomp_kit_native import decomp_resident_cached, decomp_resident_operand, decomp_resident_drop


def _str(value: PythonObject) raises -> String:
    return String(py_call(native_loaded_module("builtins").str, value))


def _components(fields: PythonObject) raises -> PythonObject:
    """The fitted matrix the `components_` property reads (`_rp_components`)."""
    return py_get(fields, "_rp_components")


def rp_fit(model: PythonObject, source: PythonObject, sparse: Bool, context: PythonObject) raises -> PythonObject:
    """`_RandomProjection.fit`; returns the converted X (kept for fit_transform)."""
    var fields = py_fields(model)
    # A new fitted matrix replaces the old: its device copies go now (N350).
    decomp_resident_drop(model)
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


def rp_components(model: PythonObject) raises -> PythonObject:
    """The `components_` getter: the fitted matrix, or AttributeError."""
    var value = _components(py_fields(model))
    if value is None:
        raise Error("AttributeError: components_")
    return value


def rp_set_components(model: PythonObject, value: PythonObject) raises:
    """The `components_` setter: replaces the fitted matrix field."""
    var fields = py_fields(model)
    if py_contains(fields, "_rp_lazy"):
        _ = py_call(fields.pop, "_rp_lazy")
    py_setitem(fields, "_rp_components", value)


def rp_getstate(model: PythonObject) raises -> PythonObject:
    """`__getstate__`: a shallow copy of the instance fields."""
    return py_call(native_loaded_module("builtins").dict, py_fields(model))


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
    var name = String("inverse_components_") if inverse else String("components_")
    var stored: PythonObject
    if inverse:
        if not py_contains(fields, "inverse_components_"):
            raise Error("inverse_transform needs compute_inverse_components=True")
        stored = fields["inverse_components_"]
    else:
        var fitted = decomp_int(decomp_attr(model, fields, "n_features_in_"))
        if width != fitted:
            raise Error("X has " + String(width) + " features, but " + String(py_type(model).__name__)
                        + " is expecting " + String(fitted))
        stored = _components(fields)
    # A fitted matrix already resident as its own frozen words is not
    # admitted again (N350); otherwise the per-call admission, as before.
    var hit = decomp_resident_cached(model, entries, stored, name)
    if hit is not None:
        matrix = stored
    else:
        matrix = classical_host_f32(stored, 2, name, context)
    if inverse:
        decomp_gemm_check(width, Int(py=matrix.shape[1]))
    var rows = Int(py=matrix.shape[0])
    var out = decomp_f32(Python.tuple(m, rows), context)
    var tiled = not inverse and host_sparse_format(source).byte_length() == 0
    var params = Python.list(
        classical_address(x), m, width, classical_address(matrix), rows, 1 if tiled else 0, classical_address(out))
    # The device copy of the fitted matrix (word 7, GPU bindings only; lane
    # ne1-a-decomp-resident, N350): uploaded at the first projection of a
    # fitted state and kept in the model's resident record
    # (bindings/decomp_kit_native.mojo `decomp_resident_*`: the model, its
    # generation word and the matrix object; the read-only rule first). Only
    # when the converted matrix is the stored object itself (a cast copy
    # would be a new object per call); else the per-call upload, as before.
    # The device words are the host words: no bit change.
    var held = hit
    if held is None and matrix is stored:
        held = decomp_resident_operand(model, entries, stored, name, rows, Int(py=matrix.shape[1]))
    if held is not None:
        _ = py_call(params.append, held[5])
    var flagged = Int(py=py_call_typed(entries["x_decomp_rp_project"], params))
    _ = held  # the resident entry (and its device buffer) outlives the native call above
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
