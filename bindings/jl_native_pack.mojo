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
from bindings.py_typed_call import typed_python_error
from bindings.py_cpython_table import cpython_table, CPythonTable
from bindings.classical_attributes_door import classical_connect_numpy
from providers.python.jl_scalar_pyapi_unpack import (
    JlTypeWords, jl_type_words, jl_type_word, jl_builtin_scalar, jl_scalar_word,
)
from x_decomp.jl_run import _cube_rounded, _pinned_log


def _jl_bound(e: Float64, n: Float64) -> Float64:
    # Preserve epsilon-before-count validation (the callers) and the
    # original arithmetic.
    var denominator = (e * e) / 2.0 - _cube_rounded(e) / 3.0
    return 4.0 * _pinned_log(n) / denominator


def _jl_publish(bound: Float64, table: UnsafePointer[CPythonTable, MutAnyOrigin]) raises -> PyObjectPtr:
    # Perf (lane fb-rest, 2026-11-17; algos/jl-min-dim 1.9x vs 0.8.55, one
    # scalar call per point): a finite bound inside the int64 range is
    # published through the import-resolved table's PyLong_FromLongLong; the
    # Int64 conversion truncates toward zero as PyLong_FromDouble does, so
    # the same integer. Before: a symbol lookup (dlsym) of PyLong_FromDouble
    # on every call. NaN, infinity and larger bounds keep PyLong_FromDouble
    # (its own OverflowError/ValueError).
    var result: PyObjectPtr
    if bound > -9.0e18 and bound < 9.0e18:
        result = table[].PyLong_FromLongLong(Int64(bound))
    else:
        ref cpy = Python().cpython()
        result = cpy.lib.get_function[PyObjectPtr]("PyLong_FromDouble")(bound)
    if not result:
        raise typed_python_error()
    return result


def jl_min_dim(n_samples: PythonObject, eps: PythonObject, context: PythonObject) raises -> PythonObject:
    """The bound for already admitted Python scalars (RandomProjection's
    n_components='auto'); the public function goes through `jl_c`."""
    var table = cpython_table()
    var words = jl_type_words()
    var e = jl_scalar_word(eps._obj_ptr, jl_type_word(eps._obj_ptr), True, context._obj_ptr, table, words)
    var n = jl_scalar_word(n_samples._obj_ptr, jl_type_word(n_samples._obj_ptr), False, context._obj_ptr, table, words)
    var result = _jl_publish(_jl_bound(e, n), table)
    _ = n_samples
    _ = eps
    _ = context
    return PythonObject(from_owned=result)


def _jl_min_dim_name(address: Int, size: Int) -> Bool:
    """The operation word is exactly "min_dim" (its UTF-8 bytes)."""
    comptime expected: StaticString = "min_dim"
    if size != expected.byte_length():
        return False
    var text = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    var want = expected.as_bytes()
    for i in range(size):  # small-loop(the seven bytes of the one private operation name)
        if text[i] != want[i]:
            return False
    return True


def jl_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    # Perf (lane fb12-rows, 2026-11-17; algos/jl-min-dim 8.27 -> 15.7 ms on
    # the same box vs 0.8.55, ~10,000 public calls per fit): the call is read
    # through borrowed tuple items and class words (one table and one
    # class-word slot per call), and NumPy is connected only when an argument
    # is not an exact int, bool or float (only that admission reads it).
    # Before: eight PythonObject item reads, a new str, tuple and dict per
    # call to compare classes, a String copy of the operation, the connected
    # NumPy probe (a new dict and key per call) and per-scalar sample objects.
    # Same integers, same refusals and messages.
    try:
        var table = cpython_table()
        var words = jl_type_words()
        # METH_VARARGS: CPython hands over its own exact argument tuple.
        if table[].PyTuple_Size(args_ptr) != 4:
            raise Error("TypeError: jl_native_call requires model, operation, arguments and connected context")
        var operation = table[].PyTuple_GetItem(args_ptr, 1)
        var arguments = table[].PyTuple_GetItem(args_ptr, 2)
        var context = table[].PyTuple_GetItem(args_ptr, 3)
        if (jl_type_word(operation) != words[].str_word or jl_type_word(arguments) != words[].tuple_word
                or jl_type_word(context) != words[].dict_word):
            raise Error("TypeError: native JL requires an operation, argument tuple and connected context")
        var size = 0
        var address = table[].PyUnicode_AsUTF8AndSize(operation, Int(MutPointer(to=size)))
        if address == 0:
            raise typed_python_error()
        if not _jl_min_dim_name(address, size) or table[].PyTuple_Size(arguments) != 2:
            raise Error("unknown private johnson_lindenstrauss_min_dim operation")
        var n_samples = table[].PyTuple_GetItem(arguments, 0)
        var eps = table[].PyTuple_GetItem(arguments, 1)
        var n_cls = jl_type_word(n_samples)
        var eps_cls = jl_type_word(eps)
        if not jl_builtin_scalar(words, n_cls) or not jl_builtin_scalar(words, eps_cls):
            classical_connect_numpy(PythonObject(from_borrowed=context))
        var e = jl_scalar_word(eps, eps_cls, True, context, table, words)
        var n = jl_scalar_word(n_samples, n_cls, False, context, table, words)
        return _jl_publish(_jl_bound(e, n), table)
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("OverflowError:"):
            return raise_python_exception(error_without_class(error, "OverflowError"), ExceptionType("PyExc_OverflowError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
