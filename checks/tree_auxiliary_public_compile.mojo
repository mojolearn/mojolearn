# Compile-only fixture for the public tree auxiliary dispatcher and its owners.
# No module loading, runtime execution or identity evidence is claimed.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_public_native import trees_public_operation

def public_tree_call(model: PythonObject, operation: PythonObject, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    return trees_public_operation(model, String(operation), args, context)

@export
def PyInit__mojolearn_tree_auxiliary_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tree_auxiliary_compile")
        module.def_function[public_tree_call]("trees_public_operation")
        return module.finalize()
    except error:
        abort(String(error))
