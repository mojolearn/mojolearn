# SPDX-License-Identifier: Apache-2.0
# The native owner of `johnson_lindenstrauss_min_dim`: the public function
# is one call into jl_c, which admits and captures scalar inputs, checks
# eps and n_samples natively, and computes scikit-learn's bound
# 4 log(n_samples) / (eps^2 / 2 - eps^3 / 3) with the same binary64
# operations: eps^2 and eps^3 correctly rounded (`_portable_math.powi`,
# DEVIATION 6900: an exact integer power rounded once to nearest-even) and
# the pinned log (`_portable_math.log`, packaging/portable_math/portable_math.c
# `mojolearn_log`, x_metrics/epilogue.mojo `portable_log_c`). So the same
# bits as the shell; CPython publishes the completed truncated integer.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_type
from bindings.py_typed_call import typed_python_error
from bindings.classical_attributes import classical_connect_numpy
from providers.python.jl_scalar_pyapi_unpack import jl_scalar
from x_decomp.jl_run import _cube_rounded, _pinned_log


def jl_min_dim(n_samples: PythonObject, eps: PythonObject, context: PythonObject) raises -> PythonObject:
    # Preserve epsilon-before-count validation and the original arithmetic.
    var e = jl_scalar(eps, True, context)
    var n = jl_scalar(n_samples, False, context)
    var denominator = (e * e) / 2.0 - _cube_rounded(e) / 3.0
    var bound = 4.0 * _pinned_log(n) / denominator
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyLong_FromDouble")(bound)
    if not result:
        raise typed_python_error()
    return PythonObject(from_owned=result)


def jl_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: jl_native_call requires model, operation, arguments and connected context")
        if py_type(call[1]) is not py_type(PythonObject("")) or py_type(call[2]) is not py_type(Python.tuple()) or py_type(call[3]) is not py_type(Python.dict()):
            raise Error("TypeError: native JL requires an operation, argument tuple and connected context")
        var operation = String(call[1])
        var args = call[2]
        if operation != "min_dim" or len(args) != 2:
            raise Error("unknown private johnson_lindenstrauss_min_dim operation")
        var context = call[3]
        classical_connect_numpy(context)
        return jl_min_dim(args[0], args[1], context).steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("OverflowError:"):
            return raise_python_exception(error_without_class(error, "OverflowError"), ExceptionType("PyExc_OverflowError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
