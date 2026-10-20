# SPDX-License-Identifier: Apache-2.0
"""Native owner of the existing Sequential model-description interface."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception


from bindings.sequential_description import sequential_fields, sequential_append, sequential_add, sequential_compile, sequential_program, sequential_table, sequential_layers, sequential_sample, _shape
from bindings.sequential_execution import sequential_entries, sequential_fit, sequential_forward, sequential_predict, sequential_targets
from bindings.sequential_weights import sequential_parameter_layers, sequential_load_weights
from bindings.sequential_checkpoint import sequential_save, sequential_load_checkpoint
from autograd.program_arrays import initialize_program_arrays


def sequential_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        var result = PythonObject(None)
        if operation == "initialize":
            initialize_program_arrays()
        elif operation == "construct":
            var b = Python.import_module("builtins")
            var fields = sequential_fields(model)
            fields["numeric_mode"] = args[0]
            fields["input_shape"] = PythonObject(None) if args[1] is None else b.tuple(_shape(args[1], "input_shape", False, context))
            fields["_layers"] = Python.list()
            fields["_options"] = PythonObject(None)
        elif operation == "add":
            result = sequential_add(model, args[0], args[1], args[2], context)
        elif operation == "append":
            sequential_append(model, args[0], args[1], args[2], args[3], args[4])
        elif operation == "compile":
            result = sequential_compile(model, args, context)
        elif operation == "layers":
            result = sequential_layers(model, context)
        elif operation == "program":
            result = sequential_program(model, args[0], args[1])
        elif operation == "table":
            result = sequential_table(model)
        elif operation == "sample":
            result = sequential_sample(model, args[0])
        elif operation == "fit":
            result = sequential_fit(model, args, context)
        elif operation == "predict" or operation == "predict_proba":
            result = sequential_predict(model, args[0], operation == "predict_proba", context)
        elif operation == "forward":
            result = sequential_forward(model, args[0], Int(py=args[1]), args[2], sequential_entries(model, context), context)
        elif operation == "targets":
            result = sequential_targets(model, args[0], Int(py=args[1]), Int(py=args[2]), String(args[3]), sequential_entries(model, context), context)
        elif operation == "parameter_layers":
            result = sequential_parameter_layers(model)
        elif operation == "load_weights":
            result = sequential_load_weights(model, args, context)
        elif operation == "save":
            result = sequential_save(model, args[0], Bool(py=args[1]), context)
        elif operation == "load_checkpoint":
            result = sequential_load_checkpoint(model, args[0], args[1], context)
        else:
            raise Error("Sequential: unknown native operation")
        return result.steal_data()
    except error:
        if String(error).startswith("FileNotFoundError:"):
            return raise_python_exception(error, ExceptionType("PyExc_FileNotFoundError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error, ExceptionType("PyExc_ImportError"))
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        return raise_python_exception(error, ExceptionType.ValueError)
