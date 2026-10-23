# SPDX-License-Identifier: Apache-2.0
# FastICA's native operations (dispatched by bindings/decomp_native.mojo):
# the parameter checks `_fit` / `_solve` made, X converted once, then one
# compiled driver (x_decomp/ica_owner.mojo: x_decomp_ica_fit / _project, the
# kit steps the Python shell sequenced, the same entries, operands and order,
# so the same bits), and the fitted attributes stored as the shell stored them.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_host_f32
from bindings.decomp_inputs import decomp_input
from bindings.decomp_owner_common import (
    decomp_attr, decomp_check, decomp_entries, decomp_eq, decomp_f32, decomp_float, decomp_gemm_check, decomp_int,
    decomp_le, decomp_min_samples, decomp_repr, decomp_seed, decomp_set_mode,
)


def _whiten_code(whiten: PythonObject) raises -> Int:
    # False -> 0, 'unit-variance' -> 1, 'arbitrary-variance' -> 2, by the
    # shell's `in` / `==` / truthiness tests.
    var allowed = Python.tuple(False, "unit-variance", "arbitrary-variance")
    if not Bool(py=py_call(Python.import_module("operator").contains, allowed, whiten)):
        raise Error("whiten=" + decomp_repr(whiten) + " is not supported")
    if not Bool(py=whiten):
        return 0
    return 1 if decomp_eq(whiten, PythonObject("unit-variance")) else 2


def ica_fit(model: PythonObject, source: PythonObject, project: Bool, context: PythonObject) raises -> PythonObject:
    """`FastICA._fit(X, project)`: self, or the sources with `project`."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    decomp_set_mode(fields, context)
    var entries = decomp_entries(model, fields, Python.tuple("x_decomp_ica_fit"), context)
    var x = decomp_input(source, "FastICA", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    decomp_min_samples(n, d, "FastICA")
    var algorithm = fields["algorithm"]
    if not Bool(py=py_call(Python.import_module("operator").contains, Python.tuple("parallel", "deflation"), algorithm)):
        raise Error("algorithm must be 'parallel' or 'deflation'")
    var whiten = fields["whiten"]
    var code = _whiten_code(whiten)
    var smaller = min(n, d)
    var requested = fields["n_components"]
    var nc = smaller
    if code != 0 and requested is not None:
        nc = min(decomp_int(requested), smaller)
    # `_solve`'s checks, made before the driver runs
    var alpha = PythonObject(1.0)
    var fun = fields["fun"]
    var fun_code: Int
    if decomp_eq(fun, PythonObject("logcosh")):
        var fun_args = fields["fun_args"]
        var options = fun_args if Bool(py=fun_args) else Python.dict()
        alpha = py_call(options.get, "alpha", 1.0)
        if not (decomp_le(PythonObject(1), alpha) and decomp_le(alpha, PythonObject(2))):
            raise Error("alpha must be in [1,2]")
        fun_code = 0
    elif decomp_eq(fun, PythonObject("exp")):
        fun_code = 1
    elif decomp_eq(fun, PythonObject("cube")):
        fun_code = 2
    else:
        raise Error("fun must be 'logcosh', 'exp' or 'cube' (a callable is not carried)")
    var winit = PythonObject(None)
    var seed = 0
    if fields["w_init"] is None:
        seed = decomp_seed(fields["random_state"])
    else:
        winit = decomp_input(fields["w_init"], "FastICA", context)
        if Int(py=winit.shape[0]) != nc or Int(py=winit.shape[1]) != nc:
            raise Error("w_init has invalid shape -- should be (" + String(nc) + ", " + String(nc) + ")")
    var cols = d if code != 0 else nc
    var components = decomp_f32(Python.tuple(nc, cols), context)
    var mixing = decomp_f32(Python.tuple(cols, nc), context)
    var whitening = decomp_f32(Python.tuple(nc, d), context) if code != 0 else PythonObject(None)
    var mean = decomp_f32(Python.tuple(d), context) if code != 0 else PythonObject(None)
    var sources = decomp_f32(Python.tuple(n, nc), context) if project else PythonObject(None)
    var spare = classical_address(components)
    var params = Python.list(
        classical_address(x), n, d, nc, code, classical_address(winit) if winit is not None else PythonObject(0), seed,
        0 if decomp_eq(algorithm, PythonObject("parallel")) else 1, fun_code, decomp_int(fields["max_iter"]),
        1 if project else 0, classical_address(components), classical_address(mixing),
        classical_address(whitening) if code != 0 else spare, classical_address(mean) if code != 0 else spare,
        classical_address(sources) if project else spare)
    var it = py_call_typed(entries["x_decomp_ica_fit"], params,
                           Python.list(decomp_float(alpha), decomp_float(fields["tol"])))
    py_setitem(fields, "n_iter_", PythonObject(Int(py=it)))
    if code != 0:
        py_setitem(fields, "whitening_", whitening)
        py_setitem(fields, "mean_", mean)
    py_setitem(fields, "components_", components)
    py_setitem(fields, "mixing_", mixing)
    py_setitem(fields, "_whiten", whiten)
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = x^
    _ = winit^
    return sources if project else model


def ica_project(model: PythonObject, source: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    """transform: (X - mean_) components_^T; inverse_transform: X mixing_^T (+ mean_)."""
    var fields = py_fields(model)
    decomp_check(model, fields, "components_")
    var entries = decomp_entries(model, fields, Python.tuple("x_decomp_ica_project"), context)
    var x = decomp_input(source, "FastICA", context)
    var n = Int(py=x.shape[0])
    var width = Int(py=x.shape[1])
    var name = String("mixing_") if inverse else String("components_")
    var matrix = classical_host_f32(decomp_attr(model, fields, name), 2, name, context)
    var rows = Int(py=matrix.shape[0])
    decomp_gemm_check(width, Int(py=matrix.shape[1]))
    var whiten = Bool(py=decomp_attr(model, fields, "_whiten"))
    var mean = classical_host_f32(decomp_attr(model, fields, "mean_"), 1, "mean_", context) if whiten else matrix
    var out = decomp_f32(Python.tuple(n, rows), context)
    _ = py_call_typed(entries["x_decomp_ica_project"], Python.list(
        1 if inverse else 0, classical_address(x), n, width, classical_address(matrix), rows,
        classical_address(mean), 1 if whiten else 0, classical_address(out)))
    _ = x^
    _ = matrix^
    _ = mean^
    return out
