# Compile only: dispatcher plus independent bytes/bytearray UTF-8 helper.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from bindings.tokenizer_public_native import tokenizer_public_c, tokenizer_decode_utf8
@export
def tokenizer(a: PyObjectPtr, b: PyObjectPtr) abi("C") -> PyObjectPtr:
    return tokenizer_public_c(a, b)
@export
def utf8(raw: PythonObject, errors: PythonObject) raises -> PythonObject:
    return tokenizer_decode_utf8(raw, errors)
