# Compile the actual public stream-read C entry and its legacy/resident/native routes.
from std.python._cpython import PyObjectPtr
from bindings.causal_lm_public import causal_public_c

@export
def legacy_read_plan_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[5](py_self, args)

@export
def resident_alias_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[10](py_self, args)
