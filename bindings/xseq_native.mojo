# SPDX-License-Identifier: Apache-2.0
"""`xseq_native_call(model, operation, args, context)`: the one native call
of every migrated public operation of the sequence-expansion estimators
(python/mojolearn/_native_xseq_bridge.py connects it at import).

Operations: rnn_fit, rnn_predict, rnn_predict_proba, rnn_hidden_sequence
(bindings/xseq_neural_native.mojo, the recurrent models); mlp_fit,
mlp_predict, mlp_predict_proba (the perceptrons); theta_*, croston_*, ets_*,
garch_fit, var_fit, stl_fit (bindings/xseq_forecast_native.mojo).
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.xseq_neural_native import xseq_rnn, xseq_mlp
from bindings.xseq_forecast_native import xseq_forecast


def _xseq_raise(error: Error, args_ptr: PyObjectPtr) -> PyObjectPtr:
    """The Python exception a native refusal names in its prefix."""
    var text = String(error)
    if text.startswith("LinAlgError:"):
        try:
            ref cpy = Python().cpython()
            # The active ABI args tuple owns the context and the class
            # throughout PyErr_SetString; borrowed without transfer.
            var cls = causal_borrowed_item(cpy.PyTuple_GetItem(args_ptr, 3), "xseq_linalg_error")
            return causal_set_exception(error_without_class(error, "LinAlgError"), cls)
        except nested:
            return raise_python_exception(nested)
    if text.startswith("TypeError:"):
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
    if text.startswith("NotImplementedError:"):
        return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
    if text.startswith("RuntimeError:"):
        return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
    if text.startswith("ImportError:"):
        return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
    if text.startswith("IndexError:"):
        return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
    if text.startswith("AttributeError:"):
        return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
    if text.startswith("OverflowError:"):
        return raise_python_exception(error_without_class(error, "OverflowError"), ExceptionType("PyExc_OverflowError"))
    if text.startswith("ZeroDivisionError:"):
        return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
    return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


def xseq_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(py=call[1])
        var args = call[2]
        var context = call[3]
        var result: PythonObject
        if operation.startswith("rnn_"):
            result = xseq_rnn(model, operation, args, context)
        elif operation.startswith("mlp_"):
            result = xseq_mlp(model, operation, args, context)
        else:
            result = xseq_forecast(model, operation, args, context)
        return result.steal_data()
    except error:
        return _xseq_raise(error, args_ptr)
