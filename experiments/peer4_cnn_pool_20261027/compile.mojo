# Compile-only entry: all pooling owner branches are reachable; never executed.
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from bindings.xcnn_pool_native import xcnn_pool_layers

def pool_compile(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return xcnn_pool_layers(model, String(py=operation), args, context)

@export
def PyInit__mojolearn_pool_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_pool_compile")
        m.def_function[pool_compile]("pool_call")
        return m.finalize()
    except error:
        abort(String(error))
