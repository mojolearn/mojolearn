from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_public_native import trees_public_operation
from bindings.public_attributes_native import _plan, _lookup
from bindings.mode_metadata_native import _fields

def operation(model: PythonObject, name: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return trees_public_operation(model, String(name), args, context)

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
