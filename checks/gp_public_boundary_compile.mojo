# Compile only; no runtime execution or numerical identity evidence.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.gp_native import gp_operation

def compile_entry(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return gp_operation(model, String(operation), args, context)

@export
def PyInit__mojolearn_gp_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_gp_public_compile")
        m.def_function[compile_entry]("gp_operation")
        return m.finalize()
    except error:
        abort(String(error))
