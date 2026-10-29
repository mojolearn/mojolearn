# Separate compile-only provider/publication boundary. Never run this fixture.
from bindings.causal_lm_public import causal_lm_generate_binding, causal_lm_read_binding
from bindings._mojolearn_transformer import causal_lm_resident_read_version_binding
from std.python import PythonObject

@export
def compile_resident_generation_boundary(model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    return causal_lm_generate_binding(model, value, maximum, options)

@export
def compile_resident_generation_stream(handle: PythonObject, count: PythonObject) raises -> PythonObject:
    return causal_lm_read_binding(handle, count)

@export
def compile_resident_generation_capability() -> PythonObject:
    return causal_lm_resident_read_version_binding()
