# SPDX-License-Identifier: Apache-2.0
# KernelPCA's native operations (dispatched by bindings/decomp_native.mojo):
# fit's refusals, X converted once, the kernel matrix and its centering on
# the x_neighbors entries the shell called (xn_kernel, xn_colsum,
# xn_scale_div, xn_kpca_center; the same arguments in the same order), then
# one compiled decomp driver for the eigenpairs (x_decomp/kpca_owner.mojo:
# x_decomp_kpca_eig, with the FAST Apple resident kernel route the binding
# reports); transform's fused (or itemized) x_neighbors chain. Same bits.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_host_f32
from bindings.decomp_inputs import decomp_input
from bindings.decomp_owner_common import (
    F32P, decomp_attr, decomp_eq, decomp_f32, decomp_float, decomp_int, decomp_repr,
)


def _f32_scalar(value: Float64) -> Float64:
    """`_f32_scalar`: a double rounded once to float32."""
    return Float64(Float32(value))


def _mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # NumericModeMixin's `numeric_mode` as `_backend.binding` resolves it.
    var b = Python.import_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    var name = py_call(py_call(mode.strip).lower)
    if String(name) != "identical" and String(name) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    return name


def _near(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """The x_neighbors entries of `mode`, checked against the selected vendor
    and mode."""
    var entries = context["decomp_neighbors"][mode]
    if py_type(entries) is Python.import_module("builtins").str:
        raise Error(String(entries))
    if String(entries["x_neighbors_vendor"]()) != String(context["vendor"]):
        raise Error("RuntimeError: KernelPCA native vendor differs from selected execution")
    if Int(py=entries["x_neighbors_numeric_mode"]()) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: KernelPCA native mode differs from selected execution")
    return entries


def _decomp(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var entries = context["decomp_entries"][mode]
    if py_type(entries) is Python.import_module("builtins").str:
        raise Error(String(entries))
    if py_get(entries, "x_decomp_kpca_eig") is None:
        raise Error("NotImplementedError: mojolearn KernelPCA: this build of _mojolearn_x_decomp does not export "
                    + "x_decomp_kpca_eig; rebuild it with bindings/build_x_decomp.sh")
    return entries


def _op(near: PythonObject, name: String, addrs: PythonObject, ints: PythonObject, floats: PythonObject) raises:
    """`_XNeighbors._op(name, ...)`: one x_neighbors entry."""
    _ = py_call_typed(near["xn_" + name], addrs, ints, floats)


def _kind(kernel: PythonObject, context: PythonObject) raises -> Int:
    return Int(py=context["kpca_kernels"][kernel])


def kpca_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`KernelPCA.fit`."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var kernel = fields["kernel"]
    var precomputed = decomp_eq(kernel, PythonObject("precomputed"))
    if Bool(py=py_call(b.callable, kernel)) or (
            not Bool(py=py_call(Python.import_module("operator").contains, context["kpca_kernels"], kernel))
            and not precomputed):
        raise Error("NotImplementedError: KernelPCA: kernel=" + decomp_repr(kernel) + " is not implemented")
    if Bool(py=fields["fit_inverse_transform"]):
        raise Error("NotImplementedError: KernelPCA: fit_inverse_transform is not implemented")
    if Bool(py=fields["kernel_params"]):
        raise Error("NotImplementedError: KernelPCA: kernel_params is not implemented")
    var x = decomp_input(source, "KernelPCA", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if precomputed and n != d:
        raise Error("Precomputed matrix must be a square matrix.")
    var gamma_value = fields["gamma"]
    var gamma = _f32_scalar(1.0 / Float64(d) if gamma_value is None else decomp_float(gamma_value))
    py_setitem(fields, "_gamma", PythonObject(gamma))
    var mode = _mode(fields, context)
    var near = _near(mode, context)
    var decomp = _decomp(mode, context)
    var requested_value = fields["n_components"]
    var requested = -1 if requested_value is None else decomp_int(requested_value)
    var auto = decomp_eq(fields["eigen_solver"], PythonObject("auto"))
    var flag = py_get(near, "x_neighbors_kpca_resident")
    var resident = (decomp_eq(kernel, PythonObject("rbf")) and auto and n > 200 and requested > 0 and requested < 10
                    and flag is not None and Int(py=py_call(flag)) != 0)
    var cols = decomp_f32(Python.tuple(n), context)
    var all_ = decomp_f32(Python.tuple(1), context)
    var kc = PythonObject(None)
    if not resident:
        var k = x
        if not precomputed:
            k = decomp_f32(Python.tuple(n, n), context)
            _op(near, "kernel", Python.list(classical_address(x), classical_address(x), classical_address(k)),
                Python.list(n, n, d, _kind(kernel, context), decomp_int(fields["degree"])),
                Python.list(_f32_scalar(gamma), _f32_scalar(decomp_float(fields["coef0"]))))
        var colsum = decomp_f32(Python.tuple(n), context)
        _op(near, "colsum", Python.list(classical_address(k), classical_address(colsum)), Python.list(n, n), Python.list())
        _op(near, "scale_div", Python.list(classical_address(colsum), classical_address(cols)), Python.list(n),
            Python.list(Float64(n)))
        # K_fit_all_: the colsum of cols seen as 1 x n, scaled (the shell's shapes)
        var colsum2 = decomp_f32(Python.tuple(n), context)
        _op(near, "colsum", Python.list(classical_address(cols), classical_address(colsum2)), Python.list(1, n),
            Python.list())
        all_ = decomp_f32(Python.tuple(n), context)
        _op(near, "scale_div", Python.list(classical_address(colsum2), classical_address(all_)), Python.list(n),
            Python.list(Float64(n)))
        kc = decomp_f32(Python.tuple(n, n), context)
        _op(near, "kpca_center", Python.list(classical_address(k), classical_address(cols), classical_address(cols),
                                            classical_address(all_), classical_address(kc)), Python.list(n, n),
            Python.list())
        _ = k^
    var c = 0 if requested < 0 else requested
    var lanczos = auto and n > 200 and c > 0 and c < 10
    var room = n if requested < 0 else min(n, requested)
    var values = List[Float32](length=max(n, 1), fill=Float32(0))
    var vectors = List[Float32](length=max(n * room, 1), fill=Float32(0))
    var spare = classical_address(cols)
    var kept = Int(py=py_call_typed(decomp["x_decomp_kpca_eig"], Python.list(
        classical_address(kc) if kc is not None else spare, classical_address(x), 1 if resident else 0, n, d,
        requested, 1 if lanczos else 0, 1 if Bool(py=fields["remove_zero_eig"]) else 0,
        Int(values.unsafe_ptr()), Int(vectors.unsafe_ptr()), classical_address(cols), classical_address(all_)),
        Python.list(gamma)))
    var eigenvalues = decomp_f32(Python.tuple(kept), context)
    var eigenvectors = decomp_f32(Python.tuple(n, kept), context)
    if kept:
        memcpy(dest=F32P(unsafe_from_address=Int(py=classical_address(eigenvalues))),
               src=F32P(unsafe_from_address=Int(values.unsafe_ptr())), count=kept)
        memcpy(dest=F32P(unsafe_from_address=Int(py=classical_address(eigenvectors))),
               src=F32P(unsafe_from_address=Int(vectors.unsafe_ptr())), count=n * kept)
    _ = values^
    _ = vectors^
    _ = kc^
    py_setitem(fields, "eigenvalues_", eigenvalues)
    py_setitem(fields, "eigenvectors_", eigenvectors)
    py_setitem(fields, "_fit_X", x)
    py_setitem(fields, "_fit_cols", cols)
    py_setitem(fields, "_fit_all", all_)
    py_setitem(fields, "n_features_in_", PythonObject(d))
    return model


def _alpha_scale(model: PythonObject, fields: PythonObject, near: PythonObject, divide: Int,
                 context: PythonObject) raises -> PythonObject:
    """`_alpha_scale(divide)`: the eigenvectors scaled by sqrt(eigenvalue)
    (0) or divided by it (1), on `xn_kpca_alpha_scale`."""
    var v = classical_host_f32(decomp_attr(model, fields, "eigenvectors_"), 2, "eigenvectors_", context)
    var w = classical_host_f32(decomp_attr(model, fields, "eigenvalues_"), 1, "eigenvalues_", context)
    var rows = Int(py=v.shape[0])
    var cols = Int(py=v.shape[1])
    var out = decomp_f32(Python.tuple(rows, cols), context)
    _op(near, "kpca_alpha_scale", Python.list(classical_address(v), classical_address(w), classical_address(out)),
        Python.list(rows, cols, divide), Python.list())
    _ = v^
    _ = w^
    return out


def kpca_fit_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`KernelPCA.fit_transform`: fit, then the scaled eigenvectors."""
    _ = kpca_fit(model, source, context)
    var fields = py_fields(model)
    return _alpha_scale(model, fields, _near(_mode(fields, context), context), 0, context)


def kpca_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`KernelPCA.transform`: the fused kernel, centering and product chain
    (`xn_kpca_transform`), or its itemized ops under MOJOLEARN_XN_OLD_ITEMS."""
    var fields = py_fields(model)
    var q = decomp_input(source, "KernelPCA", context)
    var nq = Int(py=q.shape[0])
    var width = Int(py=q.shape[1])
    var fit_x = classical_host_f32(decomp_attr(model, fields, "_fit_X"), 2, "_fit_X", context)
    var nf = Int(py=fit_x.shape[0])
    var near = _near(_mode(fields, context), context)
    var alphas = _alpha_scale(model, fields, near, 1, context)
    var c = Int(py=alphas.shape[1])
    var kernel = fields["kernel"]
    var pre = decomp_eq(kernel, PythonObject("precomputed"))
    if pre and width != nf:
        raise Error("KernelPCA: a precomputed kernel needs " + String(nf) + " columns (the fitted rows), got "
                    + String(width))
    var fit_cols = classical_host_f32(decomp_attr(model, fields, "_fit_cols"), 1, "_fit_cols", context)
    var fit_all = classical_host_f32(decomp_attr(model, fields, "_fit_all"), 1, "_fit_all", context)
    var gamma = decomp_float(decomp_attr(model, fields, "_gamma"))
    var coef0 = decomp_float(fields["coef0"])
    var degree = decomp_int(fields["degree"])
    var out: PythonObject
    if nq != 0 and c != 0 and not Bool(py=context["kpca_old_items"]):
        out = decomp_f32(Python.tuple(nq, c), context)
        _op(near, "kpca_transform", Python.list(
            classical_address(q), classical_address(q) if pre else classical_address(fit_x),
            classical_address(fit_cols), classical_address(fit_all), classical_address(alphas),
            classical_address(out)),
            Python.list(nq, nf, width, c, Int(py=context["kpca_precomputed_kind"]) if pre else _kind(kernel, context),
                        degree),
            Python.list(0.0 if pre else _f32_scalar(gamma), 0.0 if pre else _f32_scalar(coef0), Float64(nf)))
    else:
        var k = q
        if not pre:
            k = decomp_f32(Python.tuple(nq, nf), context)
            _op(near, "kernel", Python.list(classical_address(q), classical_address(fit_x), classical_address(k)),
                Python.list(nq, nf, width, _kind(kernel, context), degree),
                Python.list(_f32_scalar(gamma), _f32_scalar(coef0)))
        var rows = decomp_f32(Python.tuple(nq), context)
        _op(near, "rowsum", Python.list(classical_address(k), classical_address(rows)), Python.list(nq, nf), Python.list())
        var pred = decomp_f32(Python.tuple(nq), context)
        _op(near, "scale_div", Python.list(classical_address(rows), classical_address(pred)), Python.list(nq),
            Python.list(Float64(nf)))
        var kc = decomp_f32(Python.tuple(nq, nf), context)
        _op(near, "kpca_center", Python.list(classical_address(k), classical_address(fit_cols),
                                            classical_address(pred), classical_address(fit_all), classical_address(kc)),
            Python.list(nq, nf), Python.list())
        out = decomp_f32(Python.tuple(nq, c), context)
        _op(near, "matmul", Python.list(classical_address(kc), classical_address(alphas), classical_address(out)),
            Python.list(nq, nf, c), Python.list())
        _ = k^
    _ = q^
    _ = fit_x^
    _ = fit_cols^
    _ = fit_all^
    return out
