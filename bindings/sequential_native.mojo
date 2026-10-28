# SPDX-License-Identifier: Apache-2.0
"""Native owner of the existing Sequential model-description interface."""
from std.python import Python, PythonObject
from bindings.native_loaded_module import NativeModules
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_setitem


from bindings.sequential_description import sequential_fields, sequential_append, sequential_add, sequential_compile, sequential_program, sequential_table, sequential_layers, sequential_sample, _shape
from bindings.sequential_execution import sequential_entries, sequential_fit, sequential_forward, sequential_predict, sequential_targets
from bindings.sequential_weights import sequential_parameter_layers, sequential_load_weights
from bindings.sequential_checkpoint import sequential_save, sequential_load_checkpoint
from providers.python.autograd_program_arrays_pyapi import initialize_program_arrays
from bindings.sequential_admission import sequential_admit


def sequential_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        sequential_admit(mods, model, operation, args, context)
        var result = PythonObject(None)
        if operation == "initialize":
            initialize_program_arrays()
        elif operation == "construct":
            var b = Python.import_module("builtins")
            var fields = sequential_fields(model)
            py_setitem(fields, "numeric_mode", args[0])
            py_setitem(fields, "input_shape", PythonObject(None) if args[1] is None else py_call(b.tuple, _shape(mods, args[1], "input_shape", False, context)))
            py_setitem(fields, "_layers", Python.list())
            py_setitem(fields, "_options", PythonObject(None))
        elif operation == "add":
            result = sequential_add(mods, model, args[0], args[1], args[2], context)
        elif operation == "append":
            sequential_append(mods, model, args[0], args[1], args[2], args[3], args[4])
        elif operation == "compile":
            result = sequential_compile(mods, model, args, context)
        elif operation == "layers":
            result = sequential_layers(model, context)
        elif operation == "program":
            result = sequential_program(mods, model, args[0], args[1])
        elif operation == "table":
            result = sequential_table(model)
        elif operation == "sample":
            result = sequential_sample(mods, model, args[0])
        elif operation == "fit":
            result = sequential_fit(mods, model, args, context)
        elif operation == "predict" or operation == "predict_proba":
            result = sequential_predict(mods, model, args[0], operation == "predict_proba", context)
        elif operation == "forward":
            result = sequential_forward(mods, model, args[0], Int(py=args[1]), args[2], sequential_entries(model, context), context)
        elif operation == "targets":
            result = sequential_targets(mods, model, args[0], Int(py=args[1]), Int(py=args[2]), String(args[3]), sequential_entries(model, context), context)
        elif operation == "parameter_layers":
            result = sequential_parameter_layers(model)
        elif operation == "load_weights":
            result = sequential_load_weights(mods, model, args, context)
        elif operation == "save":
            result = sequential_save(mods, model, args[0], Bool(py=args[1]), context)
        elif operation == "load_checkpoint":
            result = sequential_load_checkpoint(mods, model, args[0], args[1], context)
        else:
            raise Error("Sequential: unknown native operation")
        return result.steal_data()
    except error:
        if String(error).startswith("FileNotFoundError:"):
            return raise_python_exception(error_without_class(error, "FileNotFoundError"), ExceptionType("PyExc_FileNotFoundError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
