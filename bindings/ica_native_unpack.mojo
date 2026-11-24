# SPDX-License-Identifier: Apache-2.0
# FastICA's native operations (dispatched by bindings/decomp_native.mojo):
# the parameter checks `_fit` / `_solve` made, X converted once, then one
# compiled driver (x_decomp/ica_owner.mojo: ica_fit_run / ica_project_run, the
# kit steps the Python shell sequenced, the same operands and order, so the
# same bits), and the fitted attributes stored as the shell stored them.
# UNPACK reads the estimator's fields and arrays into one typed request
# (core/decomp_factor_api.mojo); RUN is one call through the operations table
# of the estimator's image (`decomp_entries_routed`: the device image, or the
# CPU image for `device="cpu"`), the GIL released; PACK stores the fitted
# fields. No Python parameter list and no Python-callable driver entry.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from std.python._cpython import GILReleased
from bindings.classical_arrays_door import classical_address
from bindings.admitted_input_unpack import admit_x
from bindings.decomp_inputs_unpack import decomp_input
from bindings.decomp_owner_common_unpack import (
    decomp_attr, decomp_check, decomp_eq, decomp_f32, decomp_float, decomp_int,
    decomp_le, decomp_repr, decomp_seed, decomp_set_mode,
)
from bindings.decomp_kit_native_door import decomp_entries_routed, decomp_entries_ref
from core.decomp_factor_api import ICAFitRequest, ICAProjectRequest, ica_fit_call, ica_project_call
from core.fit_loop_clock import FitLoop


def _address(array: PythonObject) raises -> Int:
    return Int(py=classical_address(array))
from x_decomp.decomp_owner_common_run import decomp_gemm_check, decomp_min_samples


def _whiten_code(whiten: PythonObject) raises -> Int:
    # False -> 0, 'unit-variance' -> 1, 'arbitrary-variance' -> 2, by the
    # shell's `in` / `==` / truthiness tests.
    var allowed = Python.tuple(False, "unit-variance", "arbitrary-variance")
    if not Bool(py=py_call(native_loaded_module("operator").contains, allowed, whiten)):
        raise Error("whiten=" + decomp_repr(whiten) + " is not supported")
    if not Bool(py=whiten):
        return 0
    return 1 if decomp_eq(whiten, PythonObject("unit-variance")) else 2


def ica_fit(model: PythonObject, source: PythonObject, project: Bool, context: PythonObject) raises -> PythonObject:
    """`FastICA._fit(X, project)`: self, or the sources with `project`."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    decomp_set_mode(fields, context)
    var entry = decomp_entries_ref(
        decomp_entries_routed(model, fields, decomp_attr(model, fields, "numeric_mode_"), context))
    var x = decomp_input(source, "FastICA", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    decomp_min_samples(n, d, "FastICA")
    var algorithm = fields["algorithm"]
    if not Bool(py=py_call(native_loaded_module("operator").contains, Python.tuple("parallel", "deflation"), algorithm)):
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
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var winit = Optional[PythonObject]()
    var seed = 0
    if fields["w_init"] is None:
        seed = decomp_seed(fields["random_state"])
    else:
        winit = decomp_input(fields["w_init"], "FastICA", context)
        if Int(py=winit.value().shape[0]) != nc or Int(py=winit.value().shape[1]) != nc:
            raise Error("w_init has invalid shape -- should be (" + String(nc) + ", " + String(nc) + ")")
    var cols = d if code != 0 else nc
    var components = decomp_f32(Python.tuple(nc, cols), context)
    var mixing = decomp_f32(Python.tuple(cols, nc), context)
    var whitening = decomp_f32(Python.tuple(nc, d), context) if code != 0 else PythonObject(None)
    var mean = decomp_f32(Python.tuple(d), context) if code != 0 else PythonObject(None)
    var sources = decomp_f32(Python.tuple(n, nc), context) if project else PythonObject(None)
    var spare = _address(components)
    # the fixed-point loop's nanoseconds (`fit_loop_`)
    var loop_ns = List[Int](length=1, fill=0)
    var request = ICAFitRequest(
        _address(x), _address(winit.value()) if Bool(winit) else 0, spare, _address(mixing),
        _address(whitening) if code != 0 else spare, _address(mean) if code != 0 else spare,
        _address(sources) if project else spare,
        n, d, nc, code, seed, 0 if decomp_eq(algorithm, PythonObject("parallel")) else 1, fun_code,
        decomp_int(fields["max_iter"]), project, decomp_float(alpha), decomp_float(fields["tol"]),
        Int(loop_ns.unsafe_ptr()), -1)
    var it = 0
    with GILReleased(Python()):
        it = ica_fit_call(entry, request)
    _ = sources  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    _ = whitening  # keep the Python object alive through the native call above
    _ = mixing  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = winit  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", PythonObject(it))
    # `fit_loop_`: (n_iter_, the fixed-point loop's milliseconds), Mojo values packed once per fit
    var loop = FitLoop(it, loop_ns[0], True)
    py_setitem(fields, "fit_loop_", Python.tuple(PythonObject(loop.iterations), PythonObject(loop.milliseconds())))
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
    var entry = decomp_entries_ref(
        decomp_entries_routed(model, fields, decomp_attr(model, fields, "numeric_mode_"), context))
    var x = decomp_input(source, "FastICA", context)
    var n = Int(py=x.shape[0])
    var width = Int(py=x.shape[1])
    var name = String("mixing_") if inverse else String("components_")
    var matrix = admit_x(decomp_attr(model, fields, name), 2, name, context)
    var rows = Int(py=matrix.shape[0])
    decomp_gemm_check(width, Int(py=matrix.shape[1]))
    var whiten = Bool(py=decomp_attr(model, fields, "_whiten"))
    var mean = admit_x(decomp_attr(model, fields, "mean_"), 1, "mean_", context) if whiten else matrix
    var out = decomp_f32(Python.tuple(n, rows), context)
    var request = ICAProjectRequest(_address(x), _address(matrix), _address(mean), _address(out),
                                    inverse, n, width, rows, whiten, -1)
    with GILReleased(Python()):
        ica_project_call(entry, request)
    _ = matrix  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    _ = x^
    _ = matrix^
    _ = mean^
    return out
