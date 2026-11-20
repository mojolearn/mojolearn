from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_public_native_door import trees_public_operation
from bindings.native_operation_codes import OP_LEAF_NONE
from bindings.native_operation_names_unpack import operation_lookup, operation_code
from bindings.public_attributes_native_door import _plan, _lookup
from bindings.mode_metadata_native_door import _fields

def operation(model: PythonObject, name: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # The tree door passes the name's leaf code (lane bw3-trees).
    var names = operation_lookup()
    var code = operation_code(names, name)
    var leaf = names[].table.leaves[code] if code >= 0 else OP_LEAF_NONE
    return trees_public_operation(model, leaf, String(name), args, context)

def attribute(model: PythonObject, name: PythonObject, context: PythonObject) raises -> PythonObject:
    return _lookup(model, name, _fields(model), _plan(model, context), context)

@export
def PyInit__mojolearn_tree_boost_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tree_boost_compile")
        module.def_function[operation]("operation")
        module.def_function[attribute]("attribute")
        return module.finalize()
    except error:
        abort(String(error))
