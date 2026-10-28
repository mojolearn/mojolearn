# Compile-only fixture. No extension loading or runtime execution.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_public_state import forest_restore_state, tree_is_fitted

def fitted_call(model: PythonObject, forest: PythonObject) raises -> PythonObject:
    return tree_is_fitted(model, Bool(py=forest))

@export
def PyInit__mojolearn_tree_lifecycle_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tree_lifecycle_compile")
        module.def_function[forest_restore_state]("restore")
        module.def_function[fitted_call]("is_fitted")
        return module.finalize()
    except error:
        abort(String(error))
