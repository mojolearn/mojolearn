# Compile only: the full native public Embedding owner and lifetime entry.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.embedding_layer_native import emblayer_operation, embedding_layer_lifetime_c


def embedding_compile(operation: PythonObject, owner: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    return emblayer_operation(String(py=operation), owner, values, context)

@export
def PyInit__mojolearn_embedding_owner_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_embedding_owner_compile")
        m.def_function[embedding_compile]("embedding_call")
        m.def_py_c_function(embedding_layer_lifetime_c, "embedding_lifetime")
        return m.finalize()
    except error:
        abort(String(error))
