# SPDX-License-Identifier: Apache-2.0
# AutoARIMA fit owns exact state admission, native per-group fit/reuse and
# staged fitted-model replacement. Search and output have separate providers.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from providers.python.autoarima_owner_pyapi import autoarima_model_class, autoarima_commit
from providers.python.autoarima_lifecycle_pyapi import _fields, _integer, _number, _packed, _buffer
from bindings.classical_pandas import classical_admit_value
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_typed
from bindings.tsa_arima_native import arima_fit, _adopt, _no_exog
from bindings.tsa_hw_native import tsa_array



def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _new_arima(order: PythonObject, sorder: PythonObject, k: Int, maxiter: Int, context: PythonObject) raises -> PythonObject:
    """`ARIMA(order=..., seasonal_order=..., trend="c" if k else "n",
    maxiter=...)` for a search order: the constructor's fields, written
    natively (a search order is already the normalized integer tuple)."""
    var b = _b()
    var model = py_call(b.object.__new__, autoarima_model_class(context))
    var fields = _fields(model, context["arima_class"])
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


def autoarima_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(model, context["autoarima_class"])
    var models = py_get(fields, "models")
    if models is None:
        raise Error("RuntimeError: AutoARIMA: call search() before fit()")
    if py_type(models) is not py_type(Python.list()):
        raise Error("TypeError: AutoARIMA selected models require a native list")
    if len(models) == 0:
        raise Error("RuntimeError: AutoARIMA: call search() before fit()")
    var h = _number(args[0], context)
    if py_type(args[2]) is not py_type(PythonObject("")):
        raise Error("TypeError: AutoARIMA method requires a native string")
    var maxiter = _integer(args[1], context)
    if h != 1e-8 or _integer(args[3], context) != 0 or String(args[2]) != "ml":
        raise Error("NotImplementedError: AutoARIMA.fit: method ml with the default h only")
    var reuse = py_get(fields, "_reuse")
    var ids_all = fields["_ids"]
    if py_type(ids_all) is not py_type(Python.list()) or len(ids_all) != len(models):
        raise Error("TypeError: AutoARIMA model groups require matching native index lists")
    if reuse is not None and (py_type(reuse) is not py_type(Python.list()) or len(reuse) != len(models)):
        raise Error("TypeError: AutoARIMA reuse metadata requires a matching native list")
    var endog = _packed(fields["_endog"], "endog", context)
    if len(endog.shape) != 2:
        raise Error("ValueError: AutoARIMA endog must be a matrix")
    var rows = endog.shape[0]
    var width = endog.shape[1]
    var fitted = Python.list()
    var search_maxiter = py_get(fields, "_search_maxiter")
    var iterations = -1 if search_maxiter is None else _integer(search_maxiter, context)
    for i in range(len(models)):  # driver-loop: over chosen order groups
        var spec = models[i]
        if py_type(spec) is not py_type(Python.tuple()) or len(spec) != 3:
            raise Error("TypeError: AutoARIMA model specification requires a native tuple")
        var order = spec[0]
        var seasonal = spec[1]
        if py_type(order) is not py_type(Python.tuple()) or len(order) != 3 or py_type(seasonal) is not py_type(Python.tuple()) or len(seasonal) != 4:
            raise Error("TypeError: AutoARIMA selected orders require native tuples")
        var ordinary = Python.list()
        var seasonal_words = Python.list()
        for k in range(3):  # small-loop(selected ordinary order words)
            _ = py_call(ordinary.append, _integer(order[k], context))
        for k in range(4):  # small-loop(selected seasonal order words)
            _ = py_call(seasonal_words.append, _integer(seasonal[k], context))
        var m = _new_arima(ordinary, seasonal_words, _integer(spec[2], context), maxiter, context)
        var ids = _packed(ids_all[i], "group indices", context, True)
        if len(ids.shape) != 1:
            raise Error("ValueError: AutoARIMA group indices must be a vector")
        var y = classical_typed(Python.tuple(ids.count, width), "<f4", context)
        _ = py_call_typed(context["base_gather_rows_bytes"], endog.source.addr, classical_address(y), ids.source.addr,
                          rows, ids.count, width * 4)
        var saved = PythonObject(None) if reuse is None else reuse[i]
        if saved is not None and maxiter == iterations:
            _adopt_reuse(m, y, saved, ids.count, width, Int(py=_fields(m, context["arima_class"])["complexity_"]), context)
        else:
            _ = arima_fit(m, Python.tuple(y, PythonObject(None)), context)
        _ = py_call(fitted.append, m)
        _ = ids
        _ = y
    var updates = Python.dict()
    py_setitem(updates, "_fitted", fitted)
    autoarima_commit(fields, updates, context)
    _ = endog
    return model


def _adopt_reuse(model: PythonObject, y: PythonObject, saved: PythonObject, count: Int, n_obs: Int, complexity: Int, context: PythonObject) raises:
    var tuple_type = py_type(Python.tuple())
    if py_type(saved) is not tuple_type or len(saved) != 3:
        raise Error("TypeError: AutoARIMA search reuse requires a native triple")
    var block = saved[0]
    if py_type(block) is not tuple_type or len(block) != 5:
        raise Error("TypeError: AutoARIMA search fit block requires a native tuple")
    # Admission precedes all shape/index reads in the adoption owner. The
    # original arrays remain pinned by saved throughout the native adoption.
    var f = _packed(block[0], "search float block", context)
    var flags = _buffer(block[1], "search flags", context)
    var ll = _packed(block[4], "search likelihoods", context)
    if not flags.c_order or flags.source.itemsize != 4 or flags.source.kind != 1:
        raise Error("TypeError: AutoARIMA search flags require contiguous int32 storage")
    var offsets = block[2]
    if py_type(offsets) is not py_type(Python.list()) and py_type(offsets) is not tuple_type:
        raise Error("TypeError: AutoARIMA search offsets require a native sequence")
    var trial = _integer(saved[1], context)
    var rows = _integer(block[3], context)
    if trial < 0 or trial >= len(offsets) or rows <= 0 or flags.count < 2 * rows * len(offsets) or ll.count < rows * len(offsets):
        raise Error("ValueError: AutoARIMA search reuse dimensions disagree")
    for i in range(len(offsets)):  # driver-loop: over retained candidate block offsets
        var offset = _integer(offsets[i], context)
        if offset < 0 or offset >= f.count:
            raise Error("ValueError: AutoARIMA search block offset is out of range")
    if _integer(offsets[trial], context) + rows * (3 * complexity + 1) > f.count:
        raise Error("ValueError: AutoARIMA selected search block is incomplete")
    var local = _packed(saved[2], "search local indices", context, True)
    if len(local.shape) != 1 or local.count != count:
        raise Error("ValueError: AutoARIMA search reuse local indices disagree")
    # Capture each caller-owned source once, then gather using only these
    # native addresses/dimensions while the snapshots pin original owners.
    var offset = _integer(offsets[trial], context)
    var params = tsa_array("f", "<f4", Python.tuple(count, complexity), count * complexity, 4, context)
    var x = tsa_array("f", "<f4", Python.tuple(count, complexity), count * complexity, 4, context)
    var x0 = tsa_array("f", "<f4", Python.tuple(count, complexity), count * complexity, 4, context)
    var stats = tsa_array("f", "<f4", Python.tuple(2 * count), 2 * count, 4, context)
    var result_flags = tsa_array("i", "<i4", Python.tuple(2 * count), 2 * count, 4, context)
    var gather = context["base_gather_rows_bytes"]
    _ = py_call_typed(gather, f.source.addr + 4 * offset, classical_address(params), local.source.addr,
                      rows, count, 4 * complexity)
    _ = py_call_typed(gather, f.source.addr + 4 * (offset + rows * complexity), classical_address(x), local.source.addr,
                      rows, count, 4 * complexity)
    _ = py_call_typed(gather, f.source.addr + 4 * (offset + 2 * rows * complexity), classical_address(x0), local.source.addr,
                      rows, count, 4 * complexity)
    var sa = Int(py=classical_address(stats))
    var fa = Int(py=classical_address(result_flags))
    _ = py_call_typed(gather, ll.source.addr + 4 * trial * rows, sa, local.source.addr, rows, count, 4)
    _ = py_call_typed(gather, f.source.addr + 4 * (offset + 3 * rows * complexity), sa + 4 * count,
                      local.source.addr, rows, count, 4)
    _ = py_call_typed(gather, flags.source.addr + 8 * rows * trial, fa, local.source.addr, rows, count, 4)
    _ = py_call_typed(gather, flags.source.addr + 4 * (2 * rows * trial + rows), fa + 4 * count,
                      local.source.addr, rows, count, 4)
    _adopt(model, _fields(model, context["arima_class"]), y, False, 2, count, n_obs, _no_exog(context), 0,
           complexity, params, x, x0, stats, result_flags, PythonObject(None), context)
    _ = local
    _ = f
    _ = flags
    _ = ll
