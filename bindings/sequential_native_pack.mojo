# SPDX-License-Identifier: Apache-2.0
"""Native owner of the existing Sequential model-description interface."""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_loaded_module_door import NativeModules
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_setitem


from bindings.sequential_description_unpack import sequential_fields, sequential_append
from bindings.sequential_description_pack import sequential_add, sequential_compile, sequential_program, sequential_table, sequential_layers, _shape
from bindings.sequential_execution_pack import sequential_entries, sequential_fit, sequential_forward, sequential_predict, sequential_targets
from bindings.sequential_weights_pack import sequential_load_weights
from bindings.sequential_checkpoint_door import sequential_save, sequential_load_checkpoint
from providers.python.autograd_program_arrays_pyapi_door import initialize_program_arrays
from bindings.sequential_admission_door import sequential_admit
from bindings.sequential_route_door import sequential_device_admit, sequential_route_clear
from bindings.py_keys import py_op_code
from bindings.sequential_keys_unpack import (
    sequential_keys,
    SEQUENTIAL_OPN_ADD,
    SEQUENTIAL_OPN_APPEND,
    SEQUENTIAL_OPN_COMPILE,
    SEQUENTIAL_OPN_CONSTRUCT,
    SEQUENTIAL_OPN_FIT,
    SEQUENTIAL_OPN_FORWARD,
    SEQUENTIAL_OPN_INITIALIZE,
    SEQUENTIAL_OPN_LAYERS,
    SEQUENTIAL_OPN_LOAD_CHECKPOINT,
    SEQUENTIAL_OPN_LOAD_WEIGHTS,
    SEQUENTIAL_OPN_PREDICT,
    SEQUENTIAL_OPN_PREDICT_PROBA,
    SEQUENTIAL_OPN_PROGRAM,
    SEQUENTIAL_OPN_SAVE,
    SEQUENTIAL_OPN_TABLE,
    SEQUENTIAL_OPN_TARGETS,
)


def sequential_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        # One lookup with the caller's own str (bindings/py_keys.mojo): the
        # former door copied the name into a String and compared it against
        # up to 16 names here and again in the admission.
        var nk = sequential_keys()
        var operation = py_op_code(nk[].ops, call[1])
        var args = call[2]
        var context = call[3]
        sequential_admit(mods, model, operation, args, context)
        var result = PythonObject(None)
        if operation == SEQUENTIAL_OPN_INITIALIZE:
            initialize_program_arrays()
        elif operation == SEQUENTIAL_OPN_CONSTRUCT:
            var b = native_loaded_module("builtins")
            var fields = sequential_fields(model)
            # `device` is checked where it is given, before any field changes
            # (bindings/sequential_route_door.mojo); the default None reads nothing.
            sequential_device_admit(args[2], context)
            sequential_route_clear(model)
            py_setitem(fields, nk[].k_numeric_mode, args[0])
            py_setitem(fields, nk[].k_device, args[2])
            py_setitem(fields, nk[].k_input_shape, PythonObject(None) if args[1] is None else py_call(b.tuple, _shape(mods, args[1], "input_shape", False, context)))
            py_setitem(fields, nk[].k__layers, Python.list())
            py_setitem(fields, nk[].k__options, PythonObject(None))
        elif operation == SEQUENTIAL_OPN_ADD:
            result = sequential_add(mods, model, args[0], args[1], args[2], context)
        elif operation == SEQUENTIAL_OPN_APPEND:
            sequential_append(mods, model, args[0], args[1], args[2], args[3], args[4])
        elif operation == SEQUENTIAL_OPN_COMPILE:
            result = sequential_compile(mods, model, args, context)
        elif operation == SEQUENTIAL_OPN_LAYERS:
            result = sequential_layers(model, context)
        elif operation == SEQUENTIAL_OPN_PROGRAM:
            result = sequential_program(mods, model, args[0], args[1])
        elif operation == SEQUENTIAL_OPN_TABLE:
            result = sequential_table(model)
        elif operation == SEQUENTIAL_OPN_FIT:
            result = sequential_fit(mods, model, args, context)
        elif operation == SEQUENTIAL_OPN_PREDICT or operation == SEQUENTIAL_OPN_PREDICT_PROBA:
            result = sequential_predict(mods, model, args[0], operation == SEQUENTIAL_OPN_PREDICT_PROBA, context)
        elif operation == SEQUENTIAL_OPN_FORWARD:
            result = sequential_forward(mods, model, args[0], Int(py=args[1]), args[2], sequential_entries(model, context), context)
        elif operation == SEQUENTIAL_OPN_TARGETS:
            result = sequential_targets(mods, model, args[0], Int(py=args[1]), Int(py=args[2]), String(args[3]), sequential_entries(model, context), context)
        elif operation == SEQUENTIAL_OPN_LOAD_WEIGHTS:
            result = sequential_load_weights(mods, model, args, context)
        elif operation == SEQUENTIAL_OPN_SAVE:
            result = sequential_save(mods, model, args[0], Bool(py=args[1]), context)
        elif operation == SEQUENTIAL_OPN_LOAD_CHECKPOINT:
            result = sequential_load_checkpoint(mods, model, args[0], args[1], args[2], context)
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
