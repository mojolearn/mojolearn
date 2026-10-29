# Compile-only full shared public forward/step boundary, including Mamba2Block.step.
from std.python import PythonObject
from bindings.neural_block_public import neural_block_forward_binding


@export
def mamba_public_step_compile(block: PythonObject, value: PythonObject, state: PythonObject,
                             step: PythonObject, lengths: PythonObject, context: PythonObject) raises -> PythonObject:
    return neural_block_forward_binding(block, value, state, step, lengths, context)
