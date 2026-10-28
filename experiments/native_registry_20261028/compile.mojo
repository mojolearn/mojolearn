from bindings.native_loaded_module import native_loaded_module
from bindings.native_prepared_owner import prepared_namespace_admitted
from providers.python.neural_descriptor_pyapi import neural_descriptor_connect
from providers.python.x_neighbors_admission_pyapi import xn_admit_checked
from std.python import PythonObject
@export
def check(value: PythonObject) raises -> PythonObject:
    var module = native_loaded_module("builtins")
    return xn_admit_checked(value)
