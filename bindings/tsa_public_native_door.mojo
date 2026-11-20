# SPDX-License-Identifier: Apache-2.0
"""`tsa_native_call(model, operation, args, context)`: the one native call of
`kpss_test` and `select_d` (python/mojolearn/_tsa_impl.py), and of
ExponentialSmoothing's fit, predict and score (bindings/tsa_hw_native.mojo).

The series are admitted and laid out series-major (the column-major
float32 form of `(n_obs, n_series)`; a 1-D series as it is), the
refusals are the Python body's, and the `_mojolearn_tsa` entry of the
process default tier runs with the same argument words, so the answers
keep their bits. The results are mojolearn Arrays, as before.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays_door import classical_input, classical_address, classical_cast_kind, classical_wrap_array
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from core.xseq_marshal import tsa_flags_u8
from core.admitted_input_nd import admitted_cast_nd_f32
from bindings.native_array_storage_door import native_scratch_bytes, native_scratch_address
from bindings.tsa_hw_native_door import tsa_array, hw_fit, hw_predict, hw_score, hw_initialize, hw_forecast, hw_component, hw_save
from bindings.tsa_admission_door import tsa_scalar, tsa_truth
from bindings.admitted_input_unpack import admit_view, AdmittedTypedPin
from bindings.admitted_input_run import admitted_view_kind
from bindings.native_loaded_module_door import native_loaded_module
from bindings.tsa_arima_native_door import arima_fit, arima_predict, arima_forecast
from providers.python.arima_lifecycle_pyapi_door import arima_initialize, arima_parameter, arima_repr
from bindings.tsa_autoarima_native_door import autoarima_fit
from providers.python.autoarima_lifecycle_pyapi_door import autoarima_initialize, autoarima_forecast, autoarima_predict_native
from providers.python.autoarima_search_pyapi_door import autoarima_search
from bindings.tsa_keys_unpack import (tsa_keys, TSA_OPN_ARIMA_FIT, TSA_OPN_ARIMA_FORECAST, TSA_OPN_ARIMA_INITIALIZE, TSA_OPN_ARIMA_PARAMETER, TSA_OPN_ARIMA_PREDICT, TSA_OPN_ARIMA_REPR, TSA_OPN_AUTOARIMA_FIT, TSA_OPN_AUTOARIMA_FORECAST,
    TSA_OPN_AUTOARIMA_INITIALIZE, TSA_OPN_AUTOARIMA_PREDICT, TSA_OPN_AUTOARIMA_SEARCH, TSA_OPN_HW_ATTRIBUTE, TSA_OPN_HW_COMPONENT, TSA_OPN_HW_FIT, TSA_OPN_HW_FORECAST,
    TSA_OPN_HW_INITIALIZE, TSA_OPN_HW_PREDICT, TSA_OPN_HW_SAVE, TSA_OPN_HW_SCORE, TSA_OPN_KPSS_TEST, TSA_OPN_SELECT_D)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _array(code: String, typestr: String, count: Int, width: Int, context: PythonObject) raises -> PythonObject:
    """A zero-filled `(count,)` mojolearn Array of `typestr`."""
    return tsa_array(code, typestr, Python.tuple(count), count, width, context)


def _entries(context: PythonObject) raises -> PythonObject:
    """The `_mojolearn_tsa` entries of the process default tier."""
    var nk = tsa_keys()
    var mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var entries = py_item(context, nk[].k_tsa_bindings)[mode]
    if py_type(entries) is _b().str:
        raise Error(String(py=entries))
    return entries


def _series_major(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_series_major(y, name)`: (series-major float32 scratch store, n_obs,
    batch). The store is the kernel's read-only input for this call only
    (never returned or kept), written in full by the one parallel cast, so it
    is uninitialised (`native_scratch_bytes`; REGRESSIONS F3/C3, was a
    zero-filled Array); read its address with `native_scratch_address`."""
    var value_pin = AdmittedTypedPin()
    var value = admit_view(source, name + " y", context, value_pin)
    var kind = admitted_view_kind(value, name + " y")
    var rank = len(value.shape)
    if rank != 1 and rank != 2:
        raise Error("mojolearn " + name + ": y must be 1-D or 2-D (n_obs, n_series)")
    var rows = value.shape[0]
    var cols = 1 if rank == 1 else value.shape[1]
    if rows * cols == 0:
        raise Error("mojolearn " + name + ": y is empty")
    var out = native_scratch_bytes(rows * cols * 4)
    var dims = List[Int]()
    var steps = List[Int]()
    if rank == 2:
        dims.append(cols)
        steps.append(value.strides[1])
    dims.append(rows)
    steps.append(value.strides[0])
    _ = admitted_cast_nd_f32(value.addr, dims, steps, kind[1], kind[0],
                 _F32P(unsafe_from_address=native_scratch_address(out)))
    _ = value_pin^
    _ = out
    return Python.tuple(out, rows, cols)


def tsa_kpss(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """kpss_test(y, d, D, s, pval_threshold, return_statistic)."""
    var nk = tsa_keys()
    var b = _b()
    var prepared = _series_major(args[0], "kpss_test", context)
    var flat = prepared[0]
    var n_obs = Int(py=prepared[1])
    var batch = Int(py=prepared[2])
    var flags = _array("i", "<i4", batch, 4, context)
    var stat = _array("f", "<f4", batch, 4, context)
    _ = py_call_typed(py_item(_entries(context), nk[].k_kpss_test), PythonObject(native_scratch_address(flat)), classical_address(flags), classical_address(stat),
                      Python.list(batch, n_obs, py_call_typed(b.int, tsa_scalar(args[1], context, True)), py_call_typed(b.int, tsa_scalar(args[2], context, True)),
                                  py_call_typed(b.int, tsa_scalar(args[3], context, True)), py_call_typed(b.float, tsa_scalar(args[4], context))))
    _ = stat  # keep the Python object alive through the native call above
    _ = flags  # keep the Python object alive through the native call above
    _ = flat  # keep the Python object alive through the native call above
    # the binding writes 1 (stationary) or 0
    var stationary = _array("B", "<u1", batch, 1, context)
    tsa_flags_u8(_I32P(unsafe_from_address=Int(py=classical_address(flags))),
                 _U8P(unsafe_from_address=Int(py=classical_address(stationary))), batch)
    _ = flags  # keep the Python object alive through the native call above
    _ = stationary  # keep the Python object alive through the native call above
    _ = flat^
    if tsa_truth(args[5], context):
        return Python.tuple(stationary, stat)
    return stationary


def tsa_select_d(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """select_d(y, D, s, d_max, pval_threshold)."""
    var nk = tsa_keys()
    var b = _b()
    var seasonal = Int(py=py_call_typed(b.int, tsa_scalar(args[1], context, True)))
    var limit_value = PythonObject(2 - seasonal) if args[3] is None else py_call_typed(b.int, tsa_scalar(args[3], context, True))
    var prepared = _series_major(args[0], "select_d", context)
    var flat = prepared[0]
    var n_obs = Int(py=prepared[1])
    var batch = Int(py=prepared[2])
    # the documented bound, refused by name on every vendor (the host
    # binding's own refusal; the device selector is never handed one outside it)
    if True:
        var limit = Int(py=py_call_typed(b.int, limit_value))
        if limit < 0 or limit + seasonal > 2:
            raise Error("select_d: d_max must satisfy 0 <= d_max <= 2 - D (d_max=" + String(limit) + ", D="
                        + String(seasonal) + "), refused by name")
    var out = _array("i", "<i4", batch, 4, context)
    _ = py_call_typed(py_item(_entries(context), nk[].k_select_d), PythonObject(native_scratch_address(flat)), classical_address(out),
                      Python.list(batch, n_obs, seasonal, py_call_typed(b.int, tsa_scalar(args[2], context, True)), py_call_typed(b.int, limit_value),
                                  py_call_typed(b.float, tsa_scalar(args[4], context))))
    _ = out  # keep the Python object alive through the native call above
    _ = flat  # keep the Python object alive through the native call above
    _ = flat^
    return out


def tsa_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var nk = tsa_keys()
        # One lookup with the caller's own str (bindings/py_keys.mojo); the
        # former chain copied the name and compared it against up to 22 names.
        var operation = py_op_code(nk[].ops, call[1])
        var context = call[3]
        if operation == TSA_OPN_HW_ATTRIBUTE:
            # p3-cnn-sequence-tsa: `__getattr__` sends every missing name here
            # (hasattr / copy / pickle probes); a name that is not one of the
            # three lazy components is refused before any other work.
            var probe = call[2][0]
            if py_type(probe) is py_type(PythonObject("")):
                var text = String(py=probe)
                if text != "level_" and text != "trend_" and text != "season_":
                    raise Error("AttributeError: ExponentialSmoothing has no such attribute")
        classical_connect_numpy(context)
        if operation == TSA_OPN_KPSS_TEST:
            return tsa_kpss(call[2], context).steal_data()
        if operation == TSA_OPN_SELECT_D:
            return tsa_select_d(call[2], context).steal_data()
        # ExponentialSmoothing (bindings/tsa_hw_native.mojo)
        if operation == TSA_OPN_HW_INITIALIZE:
            return hw_initialize(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_HW_FORECAST:
            return hw_forecast(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_HW_ATTRIBUTE:
            return hw_component(call[0], call[2], context, True).steal_data()
        if operation == TSA_OPN_HW_COMPONENT:
            return hw_component(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_HW_SAVE:
            return hw_save(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_HW_FIT:
            return hw_fit(call[0], context).steal_data()
        if operation == TSA_OPN_HW_PREDICT:
            return hw_predict(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_HW_SCORE:
            return hw_score(call[0], call[2], context).steal_data()
        # ARIMA (bindings/tsa_arima_native.mojo)
        if operation == TSA_OPN_ARIMA_INITIALIZE:
            return arima_initialize(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_ARIMA_PARAMETER:
            return arima_parameter(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_ARIMA_REPR:
            return arima_repr(call[0], context).steal_data()
        if operation == TSA_OPN_ARIMA_FIT:
            var fitted = arima_fit(call[0], call[2], context)
            # fit bookkeeping of the former `_mode` fit wrapper
            py_setitem(py_fields(call[0]), nk[].k__mojolearn_fitted, PythonObject(True))
            return fitted.steal_data()
        if operation == TSA_OPN_ARIMA_FORECAST:
            return arima_forecast(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_ARIMA_PREDICT:
            # NotFittedError, as the former `_mode` predict guard raised it
            classical_require_fitted(call[0], False, "predict", context)
            return arima_predict(call[0], call[2], context).steal_data()
        # AutoARIMA (bindings/tsa_autoarima_native.mojo)
        if operation == TSA_OPN_AUTOARIMA_SEARCH:
            return autoarima_search(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_AUTOARIMA_INITIALIZE:
            return autoarima_initialize(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_AUTOARIMA_FORECAST:
            return autoarima_forecast(call[0], call[2], context).steal_data()
        if operation == TSA_OPN_AUTOARIMA_FIT:
            var fitted = autoarima_fit(call[0], call[2], context)
            # fit bookkeeping of the former `_mode` fit wrapper
            py_setitem(py_fields(call[0]), nk[].k__mojolearn_fitted, PythonObject(True))
            return fitted.steal_data()
        if operation == TSA_OPN_AUTOARIMA_PREDICT:
            # NotFittedError, as the former `_mode` predict guard raised it
            classical_require_fitted(call[0], False, "predict", context)
            return autoarima_predict_native(call[0], call[2], context).steal_data()
        raise Error("RuntimeError: unknown tsa operation " + String(py=call[1]))
    except error:
        var text = String(error)
        if text.startswith("NotFittedError:"):
            try:
                ref cpy = Python().cpython()
                # classical_require_fitted chose the active class (mojolearn's
                # NotFittedError, or its bridge under scikit-learn's); the ABI
                # args tuple owns the context throughout PyErr_SetString.
                var cls = causal_borrowed_item(cpy.PyTuple_GetItem(args_ptr, 3), "active_not_fitted_class")
                return causal_set_exception(error_without_class(error, "NotFittedError"), cls)
            except nested:
                return raise_python_exception(nested)
        if text.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if text.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if text.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if text.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if text.startswith("IndexError:"):
            return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
        if text.startswith("ZeroDivisionError:"):
            return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
        if text.startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
