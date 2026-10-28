# SPDX-License-Identifier: Apache-2.0
# Compile-only CPU arithmetic/table closure, not just provider pointer calls.
# Never execute this raw ABI probe. The coordinator supplies identical flags.
from core.native_descriptor_abi import NativeDescriptorError
from transformer.host.native_block_dispatch import transformer_host_block_v1


@export
def peer3_transformer_session_tape_host_compile(
    op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError,
) abi("C") -> Int:
    return transformer_host_block_v1(op, aa, na, pp, np, ff, nf, result, context, sink)
