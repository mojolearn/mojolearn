# Incremental verification group over the exact changed production functions.
# The retained original tree entry serves unrelated operations. It must be a
# native built-in: this verification dispatcher cannot invoke Python callbacks.
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder, ExceptionType, raise_python_exception
from std.python._cpython import PyObjectPtr
from bindings.py_typed_call import py_call_typed
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type
from bindings.neural_block_public import neural_block_c
from bindings.trees_constructor_native import forest_construct
from bindings.sequence_device_native import sequence_device_operation, sequence_device_finalize
from bindings.optimizer_admission import opt_guards
from bindings.optimizer_public_door import door_ready


def trees_dispatch(model: PythonObject, operation: PythonObject, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    if String(operation) == "trees_forest_construct":
        return forest_construct(model, args, context)
    var original = context["verifier_original_tree_entry"]
    if py_type(original) is not Python.import_module("types").BuiltinFunctionType:
        raise Error("verification requires the original compiled tree entry")
    return py_call_typed(original, model, operation, args, context)


def device_dispatch(operation: PythonObject, owner: PythonObject, values: PythonObject,
                    mode: PythonObject, context: PythonObject) raises -> PythonObject:
    opt_guards(context)
    if String(operation) == "seqdev_finalize":
        return sequence_device_finalize(owner, values, context)
    door_ready()
    if py_type(values) is not Python.import_module("builtins").tuple:
        raise Error("TypeError: optimizer requires native argument tuples")
    return sequence_device_operation(String(operation), owner, values, mode, context)


def dispatch_c[device: Bool](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = PythonObject(None)
        comptime if device:
            result = device_dispatch(args[0], args[1], args[2], args[3], args[4])
        else:
            result = trees_dispatch(args[0], args[1], args[2], args[3])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


@export
def PyInit__mojolearn_verifier_repairs() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_verifier_repairs")
        module.def_py_c_function(neural_block_c[0], "neural_block_init")
        module.def_py_c_function(neural_block_c[4], "neural_block_set_params")
        module.def_py_c_function(dispatch_c[True], "sequence_device_native")
        module.def_py_c_function(dispatch_c[False], "trees_public_operation")
        return module.finalize()
    except:
        return PythonObject()
