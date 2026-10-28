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
from bindings.py_balanced import py_call, py_type, py_get
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_input, classical_address, classical_cast_kind, classical_wrap_array
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from core.xseq_marshal import xseq_cast_nd, tsa_flags_u8
from bindings.tsa_hw_native import tsa_array, hw_fit, hw_predict, hw_score, hw_initialize, hw_forecast, hw_component, hw_save
from bindings.tsa_admission import tsa_buffer, tsa_scalar, tsa_truth
from bindings.native_loaded_module import native_loaded_module
from bindings.tsa_arima_native import arima_fit, arima_predict
from bindings.tsa_autoarima_native import autoarima_fit, autoarima_predict
from providers.python.autoarima_lifecycle_pyapi import autoarima_initialize, autoarima_forecast

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
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = context["tsa_bindings"][mode]
    if py_type(entries) is _b().str:
        raise Error(String(py=entries))
    return entries


def _series_major(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_series_major(y, name)`: (series-major float32 Array, n_obs, batch)."""
    var value = tsa_buffer(source, name + " y", context)
    var rank = len(value.shape)
    if rank != 1 and rank != 2:
        raise Error("mojolearn " + name + ": y must be 1-D or 2-D (n_obs, n_series)")
    var rows = value.shape[0]
    var cols = 1 if rank == 1 else value.shape[1]
    if rows * cols == 0:
        raise Error("mojolearn " + name + ": y is empty")
    var out = _array("f", "<f4", rows * cols, 4, context)
    var dims = List[Int]()
    var steps = List[Int]()
    if rank == 2:
        dims.append(cols)
        steps.append(value.strides[1])
    dims.append(rows)
    steps.append(value.strides[0])
    xseq_cast_nd(value.source.addr, dims, steps, value.source.itemsize, value.source.kind, True,
                 _F32P(unsafe_from_address=Int(py=classical_address(out))))
    _ = value
    _ = out
    return Python.tuple(out, rows, cols)


def tsa_kpss(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """kpss_test(y, d, D, s, pval_threshold, return_statistic)."""
    var b = _b()
    var prepared = _series_major(args[0], "kpss_test", context)
    var flat = prepared[0]
    var n_obs = Int(py=prepared[1])
    var batch = Int(py=prepared[2])
    var flags = _array("i", "<i4", batch, 4, context)
    var stat = _array("f", "<f4", batch, 4, context)
    _ = py_call_typed(_entries(context)["kpss_test"], classical_address(flat), classical_address(flags), classical_address(stat),
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
    _ = py_call_typed(_entries(context)["select_d"], classical_address(flat), classical_address(out),
                      Python.list(batch, n_obs, seasonal, py_call_typed(b.int, tsa_scalar(args[2], context, True)), py_call_typed(b.int, limit_value),
                                  py_call_typed(b.float, tsa_scalar(args[4], context))))
    _ = out  # keep the Python object alive through the native call above
    _ = flat  # keep the Python object alive through the native call above
    _ = flat^
    return out


def tsa_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(py=call[1])
        var context = call[3]
        classical_connect_numpy(context)
        if operation == "kpss_test":
            return tsa_kpss(call[2], context).steal_data()
        if operation == "select_d":
            return tsa_select_d(call[2], context).steal_data()
        # ExponentialSmoothing (bindings/tsa_hw_native.mojo)
        if operation == "hw_initialize":
            return hw_initialize(call[0], call[2], context).steal_data()
        if operation == "hw_forecast":
            return hw_forecast(call[0], call[2], context).steal_data()
        if operation == "hw_attribute":
            return hw_component(call[0], call[2], context, True).steal_data()
        if operation == "hw_component":
            return hw_component(call[0], call[2], context).steal_data()
        if operation == "hw_save":
            return hw_save(call[0], call[2], context).steal_data()
        if operation == "hw_fit":
            return hw_fit(call[0], context).steal_data()
        if operation == "hw_predict":
            return hw_predict(call[0], call[2], context).steal_data()
        if operation == "hw_score":
            return hw_score(call[0], call[2], context).steal_data()
        # ARIMA (bindings/tsa_arima_native.mojo)
        if operation == "arima_fit":
            return arima_fit(call[0], call[2], context).steal_data()
        if operation == "arima_predict":
            return arima_predict(call[0], call[2], context).steal_data()
        # AutoARIMA (bindings/tsa_autoarima_native.mojo)
        if operation == "autoarima_initialize":
            return autoarima_initialize(call[0], call[2], context).steal_data()
        if operation == "autoarima_forecast":
            return autoarima_forecast(call[0], call[2], context).steal_data()
        if operation == "autoarima_fit":
            return autoarima_fit(call[0], call[2], context).steal_data()
        if operation == "autoarima_predict":
            return autoarima_predict(call[0], call[2], context).steal_data()
        raise Error("RuntimeError: unknown tsa operation " + operation)
    except error:
        var text = String(error)
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
        if text.startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
