# Compile-only follow-up for the frozen auxiliary owner source.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_public_auxiliary import forest_shape, embedding_apply, tree_export_state

def shape_call(model: PythonObject, operation: PythonObject,
               context: PythonObject) raises -> PythonObject:
    return forest_shape(model, String(operation), context)

def state_call(model: PythonObject, operation: PythonObject) raises -> PythonObject:
    return tree_export_state(model, String(operation))

@export
def PyInit__mojolearn_tree_auxiliary_owner_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tree_auxiliary_owner_compile")
        module.def_function[shape_call]("shape")
        module.def_function[embedding_apply]("apply")
        module.def_function[state_call]("state")
        return module.finalize()
    except error:
        abort(String(error))
