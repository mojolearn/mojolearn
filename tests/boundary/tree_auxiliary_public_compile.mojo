# Compile-only fixture for the public tree auxiliary dispatcher and its owners.
# No module loading, runtime execution or identity evidence is claimed.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_public_native_door import trees_public_operation
from bindings.native_operation_codes import OP_LEAF_NONE
from bindings.native_operation_names_unpack import operation_lookup, operation_code

def public_tree_call(model: PythonObject, operation: PythonObject, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    # The tree door passes the name's leaf code (lane bw3-trees).
    var names = operation_lookup()
    var code = operation_code(names, operation)
    var leaf = names[].table.leaves[code] if code >= 0 else OP_LEAF_NONE
    return trees_public_operation(model, leaf, String(operation), args, context)

@export
def PyInit__mojolearn_tree_auxiliary_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tree_auxiliary_compile")
        module.def_function[public_tree_call]("trees_public_operation")
        return module.finalize()
    except error:
        abort(String(error))
