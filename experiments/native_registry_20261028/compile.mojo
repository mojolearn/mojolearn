from bindings.native_loaded_module import native_loaded_module
from bindings.native_prepared_owner import prepared_namespace_admitted
from providers.python.neural_descriptor_pyapi import neural_descriptor_connect
from providers.python.x_neighbors_admission_pyapi import xn_admit_checked
from std.python import PythonObject
@export
def check(value: PythonObject) raises -> PythonObject:
    var module = native_loaded_module("builtins")
    return xn_admit_checked(value)
from bindings.native_prepared_owner import prepared_class_matches
from bindings.native_loaded_module import NativeModules
@export
def check_class(cls: PythonObject, stamp: PythonObject) raises -> Bool:
    var mods = NativeModules()
    return prepared_class_matches(mods, cls, List[PythonObject](), stamp)
from providers.python.classical_family_admission_pyapi import classical_family_admit
from bindings.classical_attributes import classical_check_layout
@export
def check_layout(model: PythonObject, context: PythonObject) raises:
    classical_check_layout(model, context)
    _ = classical_family_admit(model)
