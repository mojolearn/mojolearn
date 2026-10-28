# SPDX-License-Identifier: Apache-2.0
# Compile-only GPU arithmetic/table closure. Never execute this raw ABI probe.
# The coordinator selects one supported target and its identical-mode flags.
from core.native_descriptor_abi import NativeDescriptorError
from transformer.native_block_dispatch import transformer_block_v1


@export
def peer3_transformer_session_tape_gpu_compile(
    op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError,
) abi("C") -> Int:
    return transformer_block_v1(op, aa, na, pp, np, ff, nf, result, context, sink)
