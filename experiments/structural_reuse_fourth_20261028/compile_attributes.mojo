# Compile only: specialized and generic attribute entries, never execute.
from std.python._cpython import PyObjectPtr
from bindings.public_attributes_native import public_attribute_c, public_attribute_generic_c, public_attribute_generic_slot_c
@export
def attributes(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_c(a, b)
@export
def generic(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_generic_c(a, b)
@export
def generic_slot(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_generic_slot_c(a, b)
