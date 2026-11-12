# Compile-only entry coverage: never imported or executed by the controller.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_explainer_native import trees_explainer_operation

def operation(model: PythonObject, name: PythonObject, args: PythonObject,
              context: PythonObject) raises -> PythonObject:
    return trees_explainer_operation(model, String(name), args, context)

@export
def PyInit__mojolearn_tree_explainer_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tree_explainer_compile")
        module.def_function[operation]("operation")
        return module.finalize()
    except error:
        abort(String(error))
