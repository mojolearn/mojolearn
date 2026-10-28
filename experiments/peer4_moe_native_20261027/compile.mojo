# Compile-only public MoE owner fixture. Never loaded or executed.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.sequence_layers_native import sequence_layers_operation

def operation(op: PythonObject, owner: PythonObject, values: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    return sequence_layers_operation(String(op), owner, values, mode, context)

@export
def PyInit__mojolearn_moe_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_moe_public_compile")
        m.def_function[operation]("sequence_layers_operation")
        return m.finalize()
    except error:
        abort(String(error))
