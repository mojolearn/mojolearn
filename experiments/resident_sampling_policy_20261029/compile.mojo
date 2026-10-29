# Focused compilation of actual public entries affected by sampling policy.
# This instantiates runtime opt-in/off branches; it is not runtime verification.
from std.python._cpython import PyObjectPtr
from bindings.causal_lm_public import causal_public_c

@export
def sampling_stream_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[4](py_self, args)

@export
def sampling_read_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[5](py_self, args)

@export
def sampling_generate_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[7](py_self, args)

@export
def sampling_snapshot_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[11](py_self, args)

@export
def sampling_resume_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return causal_public_c[12](py_self, args)
