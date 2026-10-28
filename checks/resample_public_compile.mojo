# Compile-only fixture; never loaded or executed.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.linalg_ops_resample import rs_operation

def resample_public_compile(operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return rs_operation(String(operation), args, context)

@export
def PyInit__mojolearn_resample_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_resample_public_compile")
        m.def_function[resample_public_compile]("resample_public_operation")
        return m.finalize()
    except error:
        abort(String(error))
