# Focused compilation of the changed Samba public/configuration native doors.
# Never execute this fixture; native module setup and transitive runtime pending.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from bindings.samba_public_native import samba_public_c
from bindings.samba_config_native import samba_config_c
from bindings.samba_admission import samba_admit_value, _samba_metadata

@export
def compile_samba_public(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return samba_public_c(py_self, args)

@export
def compile_samba_config(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return samba_config_c(py_self, args)

@export
def compile_shared_value_walk(value: PythonObject, admission: PythonObject) raises:
    samba_admit_value(value, admission)

@export
def compile_shared_metadata_walk(value: PythonObject, admission: PythonObject) raises:
    _samba_metadata(value, admission)
