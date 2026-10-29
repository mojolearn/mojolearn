# Focused Samba configuration/registry compilation; never executed.
from std.python import PythonObject
from bindings.samba_config_native import samba_registry, samba_block_shapes
from bindings.samba_state_native import _samba_registry

@export
def compile_registry(config: PythonObject) raises -> PythonObject:
    return samba_registry(config)

@export
def compile_block_shapes(config: PythonObject, kind: String) raises -> PythonObject:
    return samba_block_shapes(config, kind)

@export
def compile_flat_registry(config: PythonObject) raises -> PythonObject:
    return _samba_registry(config)
