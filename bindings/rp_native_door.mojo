# SPDX-License-Identifier: Apache-2.0
# GaussianRandomProjection / SparseRandomProjection's native operations
# (dispatched by bindings/decomp_native.mojo): fit's checks (the target
# dimension from johnson_lindenstrauss_min_dim's native bound, the density),
# X converted once, then one compiled driver (x_decomp/rp_owner.mojo:
# rp_fit_run / rp_project_run, the kit steps and switch routes the Python
# shell took, the same streams and order, so the same bits), and the fitted
# matrix stored where the `components_` property reads it.
# UNPACK reads the estimator's fields and arrays into one typed request
# (core/decomp_factor_api.mojo); RUN is one call through the operations table
# of the estimator's image (`decomp_entries_routed`: the device image, or the
# CPU image for `device="cpu"`), the GIL released; PACK stores the fitted
# fields. No Python parameter list and no Python-callable driver entry.
from std.math import sqrt
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from std.python._cpython import GILReleased
from bindings.classical_arrays_door import classical_address
from bindings.admitted_input_unpack import admit_x
from bindings.host_sparse_native_door import host_sparse_format
from bindings.decomp_inputs_unpack import decomp_input, decomp_admit
from core.admitted_refusals import admitted_finite_refuse
from bindings.decomp_owner_common_unpack import (
    decomp_attr, decomp_check, decomp_eq, decomp_f32, decomp_float, decomp_int,
    decomp_lt, decomp_le, decomp_seed, decomp_set_mode,
)
from core.decomp_factor_api import RPFitRequest, RPProjectRequest, rp_fit_call, rp_project_call
from x_decomp.decomp_owner_common_run import decomp_gemm_check
from bindings.jl_native_pack import jl_min_dim
from bindings.decomp_kit_native_door import (
    decomp_resident_cached, decomp_resident_operand, decomp_resident_drop, decomp_entries_routed, decomp_entries_ref,
)


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
    var entry = decomp_entries_ref(
        decomp_entries_routed(model, fields, decomp_attr(model, fields, "numeric_mode_"), context))
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
    var request = RPFitRequest(
        Int(py=classical_address(x)), Int(py=classical_address(components)),
        Int(py=classical_address(inverse_m)) if inverse else Int(py=classical_address(components)),
        n, d, kc, sparse, seed, inverse, not auto and not inverse and not sparse_input, density)
    var bad = -1
    with GILReleased(Python()):
        bad = rp_fit_call(entry, request)
    _ = x  # keep the Python object alive through the native call above
    _ = inverse_m  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    if bad >= 0:
        raise Error("X: input must be finite; NaN/inf are unsupported")
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
               context: PythonObject, unchecked: Bool = False) raises -> PythonObject:
    """transform (X components_^T) or inverse_transform (X inverse_components_^T).
    `unchecked` (transform on a dense X only, `rp_query`): the admission
    borrowed X without its host finite scan; the native call tests it (the
    tiled projection's device flag, or one host scan where the tile does not
    run) and a non-finite X is refused here in the admission's wording."""
    var entries = decomp_entries_routed(model, fields, decomp_attr(model, fields, "numeric_mode_"), context)
    var entry = decomp_entries_ref(entries)
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
        matrix = admit_x(stored, 2, name, context)
    if inverse:
        decomp_gemm_check(width, Int(py=matrix.shape[1]))
    var rows = Int(py=matrix.shape[0])
    var out = decomp_f32(Python.tuple(m, rows), context)
    var tiled = not inverse and host_sparse_format(source).byte_length() == 0
    # The device copy of the fitted matrix (its pool id, GPU bindings only; lane
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
    # tiled: 0 = the plain product; 1 = the tiled projection of an X the
    # admission tested; 2 = the tiled projection of an X nothing tested yet.
    var pool = -1
    if held is not None:
        pool = Int(py=held[5])
    var request = RPProjectRequest(
        Int(py=classical_address(x)), Int(py=classical_address(matrix)), Int(py=classical_address(out)),
        m, width, rows, (2 if unchecked else 1) if tiled else 0, pool)
    var flagged = 0
    with GILReleased(Python()):
        flagged = rp_project_call(entry, request)
    _ = x  # keep the Python object alive through the native call above
    _ = held  # the resident entry (and its device buffer) outlives the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = matrix^
    if flagged != 0:
        if unchecked:
            admitted_finite_refuse(flagged, String(py_type(model).__name__))
        raise Error("X: input must be finite; NaN/inf are unsupported")
    _ = out  # keep the Python object alive through the native calls above
    return out


def rp_query(model: PythonObject, source: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    _check(model, fields)
    var name = String(py_type(model).__name__)
    # fx-board-preprocess (2026-11-19): transform on a dense X borrows a
    # float32 C-order X without the host finite scan. The tiled projection
    # already flags a non-finite X on the device copy it uploads (word `flag`
    # of x_decomp/rp_owner.mojo rp_project_run), so the host scan was a second
    # test of the same words: one more pass over all of X per transform (the
    # 0.8.55 `_project` path had none: "no host finiteness pass"). A cast X is
    # still tested in its cast pass (`finite` stays True), a sparse X and
    # inverse_transform keep the host test, and where the tile does not run
    # (the host binding, or a device that declines it) rp_project_run scans
    # the host words once before the product. Same refusal text either way.
    var device_tested = not inverse and host_sparse_format(source).byte_length() == 0
    var admitted = decomp_admit(source, name, context, defer_finite=device_tested)
    var x = admitted.array
    var out = rp_project(model, fields, source, x, inverse, context, device_tested and not admitted.finite)
    _ = x^
    return out
