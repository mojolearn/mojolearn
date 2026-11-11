# Compile only: no runtime or cross-vendor identity evidence.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_public_native import trees_public_operation

def compile_entry(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return trees_public_operation(model, String(operation), args, context)

@export
def PyInit__mojolearn_iforest_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_iforest_public_compile")
        m.def_function[compile_entry]("trees_operation")
        return m.finalize()
    except error:
        abort(String(error))
