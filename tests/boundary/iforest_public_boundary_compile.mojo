# Compile only: no runtime or cross-vendor identity evidence.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_public_native_door import trees_public_operation
from bindings.native_operation_codes import OP_LEAF_NONE
from bindings.native_operation_names_unpack import operation_lookup, operation_code

def compile_entry(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # The tree door passes the name's leaf code (lane bw3-trees).
    var names = operation_lookup()
    var code = operation_code(names, operation)
    var leaf = names[].table.leaves[code] if code >= 0 else OP_LEAF_NONE
    return trees_public_operation(model, leaf, String(operation), args, context)

@export
def PyInit__mojolearn_iforest_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_iforest_public_compile")
        m.def_function[compile_entry]("trees_operation")
        return m.finalize()
    except error:
        abort(String(error))
