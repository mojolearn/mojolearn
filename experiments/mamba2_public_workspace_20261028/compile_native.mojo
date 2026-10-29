# Compile-only actual native GPU resource dispatch, including create/close/step.
from core.native_descriptor_abi import NativeDescriptorError
from mamba.native_workspace_dispatch import mamba2_workspace_v1
from bindings._mojolearn_mamba import mamba2_workspace_api_binding
from std.python import PythonObject


@export
def compile_mamba2_workspace(op: Int, aa: Int, na: Int, pp: Int, np: Int,
                            ff: Int, nf: Int, result: Int, context: Int,
                            sink: NativeDescriptorError) abi("C") -> Int:
    return mamba2_workspace_v1(op, aa, na, pp, np, ff, nf, result, context, sink)


@export
def compile_mamba2_workspace_factory() raises -> PythonObject:
    return mamba2_workspace_api_binding()
