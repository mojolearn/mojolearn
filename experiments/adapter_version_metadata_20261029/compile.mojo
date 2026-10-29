# Compile the actual C entry used by shipped adapter metadata through x_sequence.
from std.python._cpython import PyObjectPtr
from providers.python.autograd_import_frontend_pyapi import ops_import_model_c


@export
def adapter_metadata_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return ops_import_model_c(py_self, packed)
