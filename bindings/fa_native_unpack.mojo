# SPDX-License-Identifier: Apache-2.0
# FactorAnalysis's native operations (dispatched by bindings/decomp_native.mojo):
# the parameter checks `fit` made, X converted once, then one compiled driver
# (x_decomp/fa_owner.mojo: fa_fit_run / fa_query_run, the kit steps
# the Python shell sequenced, the same operands and order, so the
# same bits), and the fitted attributes stored as the shell stored them.
# UNPACK reads the estimator's fields and arrays into one typed request
# (core/decomp_factor_api.mojo); RUN is one call through the operations table
# of the estimator's image (`decomp_entries_routed`: the device image, or the
# CPU image for `device="cpu"`), the GIL released; PACK stores the fitted
# fields. No Python parameter list and no Python-callable driver entry.
from std.python import Python, PythonObject
from core.fit_loop_clock import FitLoop
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from std.python._cpython import GILReleased
from bindings.classical_arrays_door import classical_address
from bindings.admitted_input_unpack import admit_x
from bindings.decomp_inputs_unpack import decomp_input
from bindings.py_optional import py_or_none
from bindings.decomp_owner_common_unpack import (
    F32P, decomp_attr, decomp_check, decomp_eq, decomp_f32, decomp_float, decomp_int, decomp_set_mode,
)
from bindings.decomp_kit_native_door import decomp_entries_routed, decomp_entries_ref
from core.decomp_factor_api import FAFitRequest, FAQueryRequest, fa_fit_call, fa_query_call

from bindings.fa_native_pack import (
    _psi_init,
)


comptime _TRANSFORM = 0
comptime _COVARIANCE = 1
comptime _PRECISION = 2
comptime _SCORE_SAMPLES = 3
comptime _SCORE = 4


def _address(array: PythonObject) raises -> Int:
    return Int(py=classical_address(array))


def _contains(options: PythonObject, value: PythonObject) raises -> Bool:
    return Bool(py=py_call(native_loaded_module("operator").contains, options, value))


def fa_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`FactorAnalysis.fit`."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    decomp_set_mode(fields, context)
    if not _contains(Python.tuple("lapack", "randomized"), fields["svd_method"]):
        raise Error("svd_method must be 'lapack' or 'randomized'")
    var rotation = fields["rotation"]
    if not _contains(Python.tuple(None, "varimax", "quartimax"), rotation):
        raise Error("rotation must be None, 'varimax' or 'quartimax'")
    var entry = decomp_entries_ref(
        decomp_entries_routed(model, fields, decomp_attr(model, fields, "numeric_mode_"), context))
    var x = decomp_input(source, "FactorAnalysis", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var requested = fields["n_components"]
    var nc = decomp_int(requested) if Bool(py=requested) else d
    var max_iter = decomp_int(fields["max_iter"])
    var psi0 = _psi_init(fields, d, context)
    var components = decomp_f32(Python.tuple(nc, d), context)
    var psi = decomp_f32(Python.tuple(d), context)
    var mean = decomp_f32(Python.tuple(d), context)
    var ll = List[Float64](length=max(max_iter, 1), fill=Float64(0))
    var code = 0
    if rotation is not None:
        code = 1 if decomp_eq(rotation, PythonObject("varimax")) else 2
    # the EM loop's nanoseconds (`fit_loop_`)
    var loop_ns = List[Int](length=1, fill=0)
    var request = FAFitRequest(_address(x), _address(psi0), _address(components), _address(psi), _address(mean),
                               Int(ll.unsafe_ptr()), n, d, nc, max_iter, code, decomp_float(fields["tol"]),
                               Int(loop_ns.unsafe_ptr()), -1)
    var it = 0
    with GILReleased(Python()):
        it = fa_fit_call(entry, request)
    _ = components  # keep the Python object alive through the native call above
    _ = psi0  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    _ = psi  # keep the Python object alive through the native call above
    if it == 0:
        # `_fit_store` with no fitted W: the shell's attribute read failed
        raise Error("AttributeError: 'NoneType' object has no attribute '" + (String("T") if code != 0 else String("out")) + "'")
    _ = mean  # keep the Python object alive through the native calls above
    _ = psi  # keep the Python object alive through the native calls above
    _ = components  # keep the Python object alive through the native calls above
    _ = psi0  # keep the Python object alive through the native calls above
    _ = x  # keep the Python object alive through the native calls above
    var loglike = Python.list()
    for k in range(it):  # driver-loop: over the EM iterations' log-likelihoods
        _ = py_call(loglike.append, PythonObject(ll[k]))
    _ = mean  # keep the Python object alive through the native calls above
    _ = psi  # keep the Python object alive through the native calls above
    _ = components  # keep the Python object alive through the native calls above
    _ = psi0  # keep the Python object alive through the native calls above
    _ = x  # keep the Python object alive through the native calls above
    _ = ll^
    py_setitem(fields, "components_", components)
    py_setitem(fields, "noise_variance_", psi)
    py_setitem(fields, "mean_", mean)
    py_setitem(fields, "loglike_", loglike)
    py_setitem(fields, "n_iter_", PythonObject(it))
    # `fit_loop_`: (n_iter_, the EM loop's milliseconds), Mojo values packed once per fit
    var loop = FitLoop(it, loop_ns[0], True)
    py_setitem(fields, "fit_loop_", Python.tuple(PythonObject(loop.iterations), PythonObject(loop.milliseconds())))
    _ = mean  # keep the Python object alive through the native call above
    _ = psi  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = psi0  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = x^
    _ = psi0^
    return model


def fa_query(model: PythonObject, source: PythonObject, op: Int, context: PythonObject) raises -> PythonObject:
    """transform / get_covariance / get_precision / score_samples / score."""
    var fields = py_fields(model)
    decomp_check(model, fields, "components_")
    var entry = decomp_entries_ref(
        decomp_entries_routed(model, fields, decomp_attr(model, fields, "numeric_mode_"), context))
    var w = admit_x(fields["components_"], 2, "components_", context)
    var nc = Int(py=w.shape[0])
    var d = Int(py=w.shape[1])
    var psi = admit_x(decomp_attr(model, fields, "noise_variance_"), 1, "noise_variance_", context)
    var mean = admit_x(decomp_attr(model, fields, "mean_"), 1, "mean_", context)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var x = Optional[PythonObject]()
    var n = 0
    if op == _TRANSFORM or op == _SCORE_SAMPLES or op == _SCORE:
        x = decomp_input(source, "FactorAnalysis", context)
        n = Int(py=x.value().shape[0])
        var width = Int(py=x.value().shape[1])
        if width != d:
            raise Error("x_decomp: cannot broadcast 1x" + String(d) + " against " + String(n) + "x" + String(width))
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var out = Optional[PythonObject]()
    if op == _TRANSFORM:
        out = decomp_f32(Python.tuple(n, nc), context)
    elif op == _COVARIANCE or op == _PRECISION:
        out = decomp_f32(Python.tuple(d, d), context)
    elif op == _SCORE_SAMPLES:
        out = decomp_f32(Python.tuple(n), context)
    var spare = _address(w)
    var request = FAQueryRequest(op, _address(x.value()) if Bool(x) else spare, spare, _address(psi),
                                 _address(mean), _address(out.value()) if Bool(out) else spare, n, d, nc, -1)
    var value = Float64(0.0)
    with GILReleased(Python()):
        value = fa_query_call(entry, request)
    _ = x  # keep the Python object alive through the native call above
    _ = psi  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    _ = w^
    _ = psi^
    _ = mean^
    _ = x^
    if op == _SCORE:
        var _keep_ret_out = PythonObject(value)
        _ = out  # keep the Python object alive through the native call above
        return _keep_ret_out^
    return py_or_none(out)
