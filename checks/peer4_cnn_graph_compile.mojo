# Compile-only native graph public owner; never loaded or executed.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.xcnn_graph_native import xcnn_graph_layers

def graph_public_call(layer: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return xcnn_graph_layers(layer, String(operation), args, context)

@export
def PyInit__mojolearn_cnn_graph_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_cnn_graph_compile")
        m.def_function[graph_public_call]("graph_public_call")
        return m.finalize()
    except error:
        abort(String(error))
