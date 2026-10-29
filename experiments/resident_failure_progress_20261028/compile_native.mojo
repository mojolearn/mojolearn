# Compile-only native resident execution closure. Deliberately no provider,
# boundary, PythonObject, CPython or Python import in this root or its closure.
from transformer.native_lm_resident_dispatch import transformer_lm_resident_operation

@export
def compile_resident_generation_native(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    return transformer_lm_resident_operation(op, a, p, f)
