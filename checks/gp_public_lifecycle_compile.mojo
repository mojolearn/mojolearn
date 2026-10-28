# Compilation fixture only: the public GP lifecycle and both numerical owners.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from providers.python.gp_public_lifecycle_pyapi import gp_public_operation

def lifecycle(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return gp_public_operation(model, String(operation), args, context)

@export
def PyInit__mojolearn_gp_lifecycle_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_gp_lifecycle_compile")
        m.def_function[lifecycle]("gp_public_operation")
        return m.finalize()
    except error:
        abort(String(error))
