# Both runtime-toggle arms instantiate in these actual public C entry closures.
from std.python._cpython import PyObjectPtr
from bindings.causal_lm_public import causal_public_c

@export
def sampling_read_ab_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[5](py_self, args)

@export
def sampling_generate_ab_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[7](py_self, args)
