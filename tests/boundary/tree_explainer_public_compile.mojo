# Compile-only entry coverage: never imported or executed by the controller.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_explainer_native_door import trees_explainer_operation
from bindings.native_operation_codes import OP_LEAF_NONE
from bindings.native_operation_names_unpack import operation_lookup, operation_code

def operation(model: PythonObject, name: PythonObject, args: PythonObject,
              context: PythonObject) raises -> PythonObject:
    # The explainer owner takes the leaf code (lane bw3-trees).
    var names = operation_lookup()
    var code = operation_code(names, name)
    var leaf = names[].table.leaves[code] if code >= 0 else OP_LEAF_NONE
    return trees_explainer_operation(model, leaf, args, context)

@export
def PyInit__mojolearn_tree_explainer_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tree_explainer_compile")
        module.def_function[operation]("operation")
        return module.finalize()
    except error:
        abort(String(error))
