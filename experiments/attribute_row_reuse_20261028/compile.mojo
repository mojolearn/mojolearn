# Compile only: native attribute and shared fitted/parameter metadata entry bodies.
from std.python._cpython import PyObjectPtr
from bindings.public_attributes_native import public_attribute_c, public_attribute_generic_c, public_attribute_generic_slot_c
from bindings.classical_attributes import classical_attribute_descriptor_c
from bindings.mode_metadata_native import mode_metadata_c

@export
def compile_attribute(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_c(a, b)

@export
def compile_generic(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_generic_c(a, b)

@export
def compile_metadata(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return mode_metadata_c(a, b)

@export
def compile_generic_slot(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return public_attribute_generic_slot_c(a, b)

@export
def compile_descriptor_factory(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return classical_attribute_descriptor_c(a, b)
