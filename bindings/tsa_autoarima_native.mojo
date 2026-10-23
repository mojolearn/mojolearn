# SPDX-License-Identifier: Apache-2.0
"""`AutoARIMA.fit` and `AutoARIMA.predict`
(python/mojolearn/_x_sequence_autoarima.py) as one native call each,
through `tsa_native_call` (bindings/tsa_public_native.mojo):

    fit      the argument refusals; for every chosen order group, an ARIMA
             built natively (the constructor's fields for a search order),
             its series gathered by the base binding's `gather_rows_bytes`,
             and either the search's own fit block adopted (SEARCH_REUSE:
             the same gathers and `arima_ic_from_loglike`) or the native
             `ARIMA.fit` (bindings/tsa_arima_native.mojo)
    predict  every group's native `ARIMA.predict`, scattered into one
             (batch_size, width) float32 block by `scatter_rows_bytes`

The same compiled entries with the same argument words, so no bit moves.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_typed
from bindings.tsa_hw_native import tsa_array
from bindings.tsa_arima_native import arima_fit, arima_predict, arima_adopt

comptime _AA = "AutoARIMA"


def _b() raises -> PythonObject:
    return Python.import_module("builtins")


def _new_arima(order: PythonObject, sorder: PythonObject, k: Int, maxiter: Int, context: PythonObject) raises -> PythonObject:
    """`ARIMA(order=..., seasonal_order=..., trend="c" if k else "n",
    maxiter=...)` for a search order: the constructor's fields, written
    natively (a search order is already the normalized integer tuple)."""
    var b = _b()
    var model = py_call(b.object.__new__, context["arima_class"])
    var fields = py_fields(model)
    var o = py_call(b.tuple, order)
    var so = py_call(b.tuple, sorder)
    py_setitem(fields, "order", o)
    py_setitem(fields, "seasonal_order", so)
    py_setitem(fields, "trend", "c" if k else "n")
    py_setitem(fields, "k_", k)
    py_setitem(fields, "method", "ml")
    py_setitem(fields, "maxiter", maxiter)
    py_setitem(fields, "verbose", False)
    py_setitem(fields, "output_type", PythonObject(None))
    py_setitem(fields, "n_exog_", 0)
    var complexity = Int(py=o[0]) + Int(py=o[2]) + Int(py=so[0]) + Int(py=so[2]) + k + 1
    py_setitem(fields, "complexity_", complexity)
    py_setitem(fields, "numeric_mode", PythonObject(None))
    return model


def _take(block: PythonObject, ids: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_take(block, ids)`: block[ids] of a C-contiguous 2-D float32 block, by
    the base binding's byte gather."""
    var count = len(ids)
    var rows = Int(py=block.shape[0])
    var width = Int(py=block.shape[1])
    var out = classical_typed(Python.tuple(count, width), "<f4", context)
    if count:
        _ = py_call(context["base_gather_rows_bytes"], classical_address(block), classical_address(out),
                    classical_address(ids), rows, count, 4 * width)
        _ = out  # keep the Python object alive through the native call above
    return out


def autoarima_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var models = py_get(fields, "models")
    if models is None or len(models) == 0:
        raise Error("RuntimeError: AutoARIMA: call search() before fit()")
    var h = args[0]
    var maxiter = Int(py=py_call(_b().int, args[1]))
    var method = args[2]
    var truncate = args[3]
    if not Bool(py=h == PythonObject(1e-8)) or Bool(py=truncate) or not Bool(py=method == "ml"):
        raise Error("NotImplementedError: AutoARIMA.fit: method 'ml' with the default h only")
    var reuse = py_get(fields, "_reuse")
    var ids_all = fields["_ids"]
    var endog = fields["_endog"]
    var fitted = fields["_fitted"]
    var search_maxiter = py_get(fields, "_search_maxiter")
    for i in range(len(models)):  # driver-loop: over the chosen order groups (at most the grid)
        var spec = models[i]
        var m = _new_arima(spec[0], spec[1], Int(py=spec[2]), maxiter, context)
        var y = _take(endog, ids_all[i], context)
        var adopt = (reuse is not None and len(reuse) > i and reuse[i] is not None
                     and search_maxiter is not None and maxiter == Int(py=search_maxiter))
        if adopt:
            _ = arima_adopt(m, y, reuse[i], context)
        else:
            _ = arima_fit(m, Python.tuple(y, PythonObject(None)), context)
        py_setitem(fitted, i, m)
    return model


def autoarima_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if args[2] is not None:
        raise Error("NotImplementedError: AutoARIMA.predict: prediction intervals (level) are not implemented")
    var n_obs = Int(py=fields["n_obs"])
    var start = Int(py=py_call(_b().int, args[0]))
    var end = n_obs if args[1] is None else Int(py=py_call(_b().int, args[1]))
    var width = end - start
    var fitted = py_get(fields, "_fitted")
    if fitted is None:
        raise Error("RuntimeError: AutoARIMA: call fit() first")
    for i in range(len(fitted)):  # driver-loop: over the chosen order groups (at most the grid)
        if fitted[i] is None:
            raise Error("RuntimeError: AutoARIMA: call fit() first")
    var batch = Int(py=fields["batch_size"])
    var out = classical_typed(Python.tuple(batch, width), "<f4", context)
    var ids_all = fields["_ids"]
    for i in range(len(fitted)):  # driver-loop: over the chosen order groups (at most the grid)
        var part = arima_predict(fitted[i], Python.tuple(PythonObject(start), PythonObject(end), PythonObject(None)), context)
        var ids = ids_all[i]
        if len(ids):
            _ = py_call(context["base_scatter_rows_bytes"], classical_address(part), classical_address(ids), len(ids),
                        4 * width, classical_address(out), batch)
            _ = ids  # keep the Python object alive through the native call above
            _ = part  # keep the Python object alive through the native call above
            _ = out  # keep the Python object alive through the native call above
        _ = part^
    return out
