# SPDX-License-Identifier: Apache-2.0
# Private named-operation ABI for sibling native modules. These are not public
# Torch/MAX per-operation adapters. All 64-bit metadata vectors are borrowed
# until return; the provider holds buffer/module owners and the GIL boundary.
# Arguments: operation, addresses/count, integers/count, Float64 scalars/count,
# result address, native error context and compiled native UTF-8 error sink.
comptime NativeDescriptorError = def(Int, Int, Int) thin abi("C") -> NoneType
comptime NativeDescriptorCall = def(
    Int, Int, Int, Int, Int, Int, Int, Int, Int, NativeDescriptorError,
) thin abi("C") -> Int

comptime PRIMITIVE_ABI_NAME = "mojolearn.training.primitives.v1"
comptime TRANSFORMER_BLOCK_ABI_NAME = "mojolearn.transformer.block.v1"
comptime MAMBA_STATE_ABI_NAME = "mojolearn.mamba.state.v1"
comptime MAMBA_BLOCK_ABI_NAME = "mojolearn.mamba.block.v1"
comptime PRIM_EMBED_FORWARD = 0
comptime PRIM_EMBED_BACKWARD = 1
comptime PRIM_NORM_FORWARD = 2
comptime PRIM_NORM_BACKWARD = 3
comptime PRIM_LINEAR_FORWARD = 4
comptime PRIM_LINEAR_BACKWARD = 5
comptime MAMBA1_FORWARD = 0
comptime MAMBA1_DECODE = 1
comptime MAMBA2_FORWARD = 2
comptime MAMBA2_DECODE = 3
comptime MAMBA3_FORWARD = 4
comptime MAMBA3_DECODE = 5
