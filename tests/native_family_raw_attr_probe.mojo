from std.os import abort
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder
from bindings.family_raw_attr import family_raw_get_or
from bindings.optimizer_public_words import optimizer_python_error

def read_c(self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var value = family_raw_get_or(args[0], String(args[1]), args[2])
        return value.steal_data()
    except error:
        return optimizer_python_error(error)

@export
def PyInit_raw_attr_probe() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("raw_attr_probe")
        module.def_py_c_function(read_c, "read")
        return module.finalize()
    except error:
        abort(String(error))
