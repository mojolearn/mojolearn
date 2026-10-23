# SPDX-License-Identifier: Apache-2.0
# FactorAnalysis's native operations (dispatched by bindings/decomp_native.mojo):
# the parameter checks `fit` made, X converted once, then one compiled driver
# (x_decomp/fa_owner.mojo: x_decomp_fa_fit / x_decomp_fa_query, the kit steps
# the Python shell sequenced, the same entries, operands and order, so the
# same bits), and the fitted attributes stored as the shell stored them.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_host_f32
from bindings.decomp_inputs import decomp_input
from bindings.decomp_owner_common import (
    F32P, decomp_attr, decomp_check, decomp_entries, decomp_eq, decomp_f32, decomp_float, decomp_int, decomp_set_mode,
)

comptime _TRANSFORM = 0
comptime _COVARIANCE = 1
comptime _PRECISION = 2
comptime _SCORE_SAMPLES = 3
comptime _SCORE = 4


def _contains(options: PythonObject, value: PythonObject) raises -> Bool:
    return Bool(py=py_call(Python.import_module("operator").contains, options, value))


def _psi_init(fields: PythonObject, d: Int, context: PythonObject) raises -> PythonObject:
    """`_psi_init`: ones, or the noise_variance_init values as float32."""
    var b = Python.import_module("builtins")
    var given = fields["noise_variance_init"]
    if given is None:
        var ones = decomp_f32(Python.tuple(d), context)
        var p = F32P(unsafe_from_address=Int(py=classical_address(ones)))
        for j in range(d):  # driver-loop: over the features' initial noise variances (model structure)
            p[j] = 1.0
        return ones
    var values = py_call(b.list, given)
    var count = len(values)
    var psi = decomp_f32(Python.tuple(count), context)
    var p = F32P(unsafe_from_address=Int(py=classical_address(psi)))
    for j in range(count):  # driver-loop: over the caller's noise_variance_init entries (model structure)
        p[j] = Float32(decomp_float(values[j]))
    if count != d:
        raise Error("noise_variance_init dimension does not match the number of features : " + String(count)
                    + " != " + String(d))
    return psi


def fa_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`FactorAnalysis.fit`."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    decomp_set_mode(fields, context)
    if not _contains(Python.tuple("lapack", "randomized"), fields["svd_method"]):
        raise Error("svd_method must be 'lapack' or 'randomized'")
    var rotation = fields["rotation"]
    if not _contains(Python.tuple(None, "varimax", "quartimax"), rotation):
        raise Error("rotation must be None, 'varimax' or 'quartimax'")
    var entries = decomp_entries(model, fields, Python.tuple("x_decomp_fa_fit"), context)
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
    var it = Int(py=py_call_typed(entries["x_decomp_fa_fit"], Python.list(
        classical_address(x), n, d, nc, max_iter, classical_address(psi0), code, classical_address(components),
        classical_address(psi), classical_address(mean), Int(ll.unsafe_ptr())), Python.list(decomp_float(fields["tol"]))))
    if it == 0:
        # `_fit_store` with no fitted W: the shell's attribute read failed
        raise Error("AttributeError: 'NoneType' object has no attribute '" + (String("T") if code != 0 else String("out")) + "'")
    var loglike = Python.list()
    for k in range(it):  # driver-loop: over the EM iterations' log-likelihoods
        _ = py_call(loglike.append, PythonObject(ll[k]))
    _ = ll^
    py_setitem(fields, "components_", components)
    py_setitem(fields, "noise_variance_", psi)
    py_setitem(fields, "mean_", mean)
    py_setitem(fields, "loglike_", loglike)
    py_setitem(fields, "n_iter_", PythonObject(it))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = x^
    _ = psi0^
    return model


def fa_query(model: PythonObject, source: PythonObject, op: Int, context: PythonObject) raises -> PythonObject:
    """transform / get_covariance / get_precision / score_samples / score."""
    var fields = py_fields(model)
    decomp_check(model, fields, "components_")
    var entries = decomp_entries(model, fields, Python.tuple("x_decomp_fa_query"), context)
    var w = classical_host_f32(fields["components_"], 2, "components_", context)
    var nc = Int(py=w.shape[0])
    var d = Int(py=w.shape[1])
    var psi = classical_host_f32(decomp_attr(model, fields, "noise_variance_"), 1, "noise_variance_", context)
    var mean = classical_host_f32(decomp_attr(model, fields, "mean_"), 1, "mean_", context)
    var x = PythonObject(None)
    var n = 0
    if op == _TRANSFORM or op == _SCORE_SAMPLES or op == _SCORE:
        x = decomp_input(source, "FactorAnalysis", context)
        n = Int(py=x.shape[0])
        var width = Int(py=x.shape[1])
        if width != d:
            raise Error("x_decomp: cannot broadcast 1x" + String(d) + " against " + String(n) + "x" + String(width))
    var out = PythonObject(None)
    if op == _TRANSFORM:
        out = decomp_f32(Python.tuple(n, nc), context)
    elif op == _COVARIANCE or op == _PRECISION:
        out = decomp_f32(Python.tuple(d, d), context)
    elif op == _SCORE_SAMPLES:
        out = decomp_f32(Python.tuple(n), context)
    var spare = classical_address(w)
    var value = py_call_typed(entries["x_decomp_fa_query"], Python.list(
        op, classical_address(x) if x is not None else spare, n, d, nc, classical_address(w), classical_address(psi),
        classical_address(mean), classical_address(out) if out is not None else spare))
    _ = w^
    _ = psi^
    _ = mean^
    _ = x^
    if op == _SCORE:
        return PythonObject(Float64(py=value))
    return out
