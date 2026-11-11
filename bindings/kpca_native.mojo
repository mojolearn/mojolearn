# SPDX-License-Identifier: Apache-2.0
"""KernelPCA UNPACK/PACK around typed fit, eigen and query plans."""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.classical_arrays import classical_address, classical_typed_native
from bindings.admitted_input import AdmittedX, admit_f32
from bindings.decomp_inputs import decomp_admit
from bindings.decomp_owner_common import F32P, decomp_attr
from bindings.decomp_scalar_admission import decomp_scalar_admit, decomp_scalar_repr
from bindings.decomp_scalar_compare import decomp_name_admitted
from bindings.decomp_truth_boundary import decomp_truth
from bindings.public_lifecycle_owner import lifecycle_admit_fields
from core.kpca_native_api import KPCAEigenRequest
from core.kpca_native_plan import (
    KPCAKernelOptions, KPCAFitRoute, KPCAEigenRoom, KPCAScratch, KPCADenseFit,
    KPCAFitPrepareRequest, KPCAAlphaRequest, KPCAQueryRequest, KPCA_PRECOMPUTED, kpca_shape,
)
from bindings.kpca_descriptor_boundary import (
    kpca_near_connection, kpca_decomp_connection, kpca_resident,
    kpca_run_prepare, kpca_run_eigen, kpca_run_alpha, kpca_run_fused, kpca_run_centered, kpca_run_finish,
)


def _field(fields: PythonObject, name: String) raises -> PythonObject:
    lifecycle_admit_fields(fields)
    return fields[PythonObject(name)]


def _attr(model: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    lifecycle_admit_fields(fields)
    return decomp_attr(model, fields, name)


def _int(value: PythonObject, context: PythonObject) raises -> Int:
    decomp_scalar_admit(value, context)
    return Int(py=py_call(native_loaded_module("builtins").int, value))


def _float(value: PythonObject, context: PythonObject) raises -> Float64:
    decomp_scalar_admit(value, context)
    return Float64(py=py_call(native_loaded_module("builtins").float, value))


def _name(value: PythonObject, name: String, context: PythonObject) raises -> Bool:
    decomp_scalar_admit(value, context)
    return decomp_name_admitted(value, name)


def _f32_scalar(value: Float64) -> Float64:
    return Float64(Float32(value))


def _f32(rows: Int, cols: Int, context: PythonObject, vector: Bool = False) raises -> PythonObject:
    return classical_typed_native(kpca_shape(rows, cols, vector), "<f4", context["output_context"])


def _address(value: PythonObject) raises -> Int:
    return Int(py=classical_address(value))


def _mode(fields: PythonObject, context: PythonObject) raises -> String:
    lifecycle_admit_fields(fields)
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not py_type(PythonObject("")):
        raise Error("numeric_mode must be fast, identical or None")
    var name = String(mode)
    if name != "identical" and name != "fast":
        name = String(py_call(py_call(mode.strip).lower))
        if name != "identical" and name != "fast":
            raise Error("numeric_mode must be fast, identical or None")
    return name^


def _kernels(context: PythonObject) raises -> PythonObject:
    var table = context["kpca_kernels"]
    lifecycle_admit_fields(table)
    return table


def _kind(kernel: PythonObject, context: PythonObject) raises -> Int:
    # The import-connected name table belongs to UNPACK. Keys and the input
    # scalar are admitted before hashing, and RUN receives only a native code.
    var value = _kernels(context)[kernel]
    if py_type(value) is not py_type(PythonObject(0)):
        raise Error("TypeError: KernelPCA kernel codes require exact native integers")
    return Int(py=value)


def kpca_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var kernel = _field(fields, "kernel")
    if Bool(py=py_call(native_loaded_module("builtins").callable, kernel)):
        raise Error("NotImplementedError: KernelPCA: callable kernels require a native kernel name")
    var precomputed = _name(kernel, "precomputed", context)
    if not py_contains(_kernels(context), kernel) and not precomputed:
        raise Error("NotImplementedError: KernelPCA: kernel=" + decomp_scalar_repr(kernel, context) + " is not implemented")
    if decomp_truth(_field(fields, "fit_inverse_transform"), context):
        raise Error("NotImplementedError: KernelPCA: fit_inverse_transform is not implemented")
    if decomp_truth(_field(fields, "kernel_params"), context):
        raise Error("NotImplementedError: KernelPCA: kernel_params is not implemented")
    var x = decomp_admit(source, "KernelPCA", context)
    if precomputed and x.rows != x.cols:
        raise Error("Precomputed matrix must be a square matrix.")
    var gamma_value = _field(fields, "gamma")
    var gamma = _f32_scalar(1.0 / Float64(x.cols) if gamma_value is None else _float(gamma_value, context))
    py_setitem(fields, "_gamma", PythonObject(gamma))
    var mode = _mode(fields, context)
    var near = kpca_near_connection(mode, context)
    var decomp = kpca_decomp_connection(mode, context)
    var requested_value = _field(fields, "n_components")
    var requested = -1 if requested_value is None else _int(requested_value, context)
    var auto = _name(_field(fields, "eigen_solver"), "auto", context)
    var kind = KPCA_PRECOMPUTED if precomputed else _kind(kernel, context)
    var route = KPCAFitRoute(x.rows, requested, kind, auto, kpca_resident(near))
    var columns = _f32(x.rows, 0, context, True)
    var total = _f32(1, 0, context, True)
    var prepared: Optional[KPCADenseFit] = None
    var centered = _address(columns)  # same unread resident-route dummy
    if not route.resident:
        # Precomputed fit never converts its ignored degree/coef0 fields.
        var degree = 0 if precomputed else _int(_field(fields, "degree"), context)
        var coef0 = 0.0 if precomputed else _f32_scalar(_float(_field(fields, "coef0"), context))
        var options = KPCAKernelOptions(kind, degree, gamma, coef0)
        var prepare = KPCAFitPrepareRequest(x.address, _address(columns), _address(total), x.rows, x.cols, options^)
        prepared = Optional[KPCADenseFit](kpca_run_prepare(near, decomp, prepare))
        centered = prepared.value().address()
    var room = KPCAEigenRoom(x.rows, route.room)
    var remove_zero = decomp_truth(_field(fields, "remove_zero_eig"), context)
    var request = KPCAEigenRequest(centered, x.address, room.values_address(), room.vectors_address(),
        _address(columns), _address(total), route.resident, x.rows, x.cols, requested,
        route.lanczos, remove_zero, gamma)
    try:
        var kept = kpca_run_eigen(decomp, request)
        lifecycle_admit_fields(fields)
        var eigenvalues = _f32(kept, 0, context, True)
        var eigenvectors = _f32(x.rows, kept, context)
        if kept:
            unsafe_memcpy(dest=F32P(unsafe_from_address=_address(eigenvalues)),
                src=F32P(unsafe_from_address=room.values_address()), count=kept)
            unsafe_memcpy(dest=F32P(unsafe_from_address=_address(eigenvectors)),
                src=F32P(unsafe_from_address=room.vectors_address()), count=x.rows * kept)
        lifecycle_admit_fields(fields)
        py_setitem(fields, "eigenvalues_", eigenvalues)
        py_setitem(fields, "eigenvectors_", eigenvectors)
        py_setitem(fields, "_fit_X", x.array)
        py_setitem(fields, "_fit_cols", columns)
        py_setitem(fields, "_fit_all", total)
        py_setitem(fields, "n_features_in_", PythonObject(x.cols))
    finally:
        _ = prepared
        _ = room
        _ = x
        _ = columns
        _ = total
    return model


@fieldwise_init
struct KPCAAlphaInputs(Movable):
    var vectors: AdmittedX
    var values: AdmittedX


def _alpha_inputs(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> KPCAAlphaInputs:
    var vectors = admit_f32(_attr(model, fields, "eigenvectors_"), 2, "eigenvectors_", context)
    var values = admit_f32(_attr(model, fields, "eigenvalues_"), 1, "eigenvalues_", context)
    if values.rows < vectors.cols:
        raise Error("KernelPCA: eigenvalues_ is shorter than the eigenvector count")
    return KPCAAlphaInputs(vectors^, values^)


def kpca_fit_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = kpca_fit(model, source, context)
    var fields = py_fields(model)
    var near = kpca_near_connection(_mode(fields, context), context)
    var alpha = _alpha_inputs(model, fields, context)
    var dst = _f32(alpha.vectors.rows, alpha.vectors.cols, context)
    var request = KPCAAlphaRequest(alpha.vectors.address, alpha.values.address, _address(dst),
        alpha.vectors.rows, alpha.vectors.cols, False)
    try:
        kpca_run_alpha(near, request)
    finally:
        _ = alpha
    return dst


def kpca_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var q = decomp_admit(source, "KernelPCA", context)
    var fitted = admit_f32(_attr(model, fields, "_fit_X"), 2, "_fit_X", context)
    var near = kpca_near_connection(_mode(fields, context), context)
    var alpha = _alpha_inputs(model, fields, context)
    var alphas = KPCAScratch(alpha.vectors.rows, alpha.vectors.cols)
    var count = alpha.vectors.cols
    var scaling = KPCAAlphaRequest(alpha.vectors.address, alpha.values.address, alphas.address(),
        alpha.vectors.rows, count, True)
    try:
        kpca_run_alpha(near, scaling)
    finally:
        _ = alpha
    var kernel = _field(fields, "kernel")
    var precomputed = _name(kernel, "precomputed", context)
    if precomputed and q.cols != fitted.rows:
        raise Error("KernelPCA: a precomputed kernel needs " + String(fitted.rows)
            + " columns (the fitted rows), got " + String(q.cols))
    var columns = admit_f32(_attr(model, fields, "_fit_cols"), 1, "_fit_cols", context)
    var total = admit_f32(_attr(model, fields, "_fit_all"), 1, "_fit_all", context)
    var gamma = _float(_attr(model, fields, "_gamma"), context)
    var coef0 = _float(_field(fields, "coef0"), context)
    var degree = _int(_field(fields, "degree"), context)
    var dst = PythonObject(None)
    try:
        # Rebound public storage must contain every word the existing
        # kernels read. Longer vectors retain their prior prefix behavior.
        if alphas.rows != fitted.rows or columns.rows < fitted.rows or total.rows < 1:
            raise Error("KernelPCA: fitted query buffers do not cover the fitted row count")
        if not precomputed and fitted.cols != q.cols:
            raise Error("KernelPCA: query and fitted input feature counts differ")
        if q.rows != 0 and count != 0 and not decomp_truth(context["kpca_old_items"], context):
            dst = _f32(q.rows, count, context)
            var kind = KPCA_PRECOMPUTED if precomputed else _kind(kernel, context)
            var options = KPCAKernelOptions(kind, degree, gamma, coef0)
            var request = KPCAQueryRequest(q.address, fitted.address, columns.address, total.address,
                alphas.address(), _address(dst), q.rows, fitted.rows, q.cols, count, options^)
            kpca_run_fused(near, request)
        else:
            var kind = KPCA_PRECOMPUTED if precomputed else _kind(kernel, context)
            var options = KPCAKernelOptions(kind, degree, gamma, coef0)
            var request = KPCAQueryRequest(q.address, fitted.address, columns.address, total.address,
                alphas.address(), 0, q.rows, fitted.rows, q.cols, count, options^)
            var centered = kpca_run_centered(near, request)
            dst = _f32(q.rows, count, context)
            kpca_run_finish(near, centered, alphas.address(), _address(dst), count)
            _ = centered
    finally:
        _ = q
        _ = fitted
        _ = columns
        _ = total
        _ = alphas
    return dst
