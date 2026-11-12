# SPDX-License-Identifier: Apache-2.0
# The session forward's execution owner (block table op 11,
# transformer/native_block_dispatch.mojo). Numerical implementation remains
# _transformer_run_session, unchanged.
# Lane transformer-session-native (2026-11-12): the generation session ABI
# (`transformer_session_forward_v1`, TransformerNativeAPI, TRANSFORMER_NATIVE_API,
# transformer/session_abi.mojo + .h, transformer/native_session_call.mojo and the
# `transformer_native_api_v1` entry) is deleted: it took a raw session address
# from a Python-allocated owner; the session is a registry handle of the block
# table now, whose op 11 runs this same function with the same words.
from transformer.block_options import BlockOptions
from transformer.transformer_binding_impl import (
    TransformerSession, _transformer_run_session,
)


def transformer_session_forward_native(
    mut owner: TransformerSession, a: List[Int], p: List[Int], opts: BlockOptions,
) raises -> Int:
    # Caller validates metadata and reserves owner.busy before releasing its
    # interpreter GIL.
    try:
        return _transformer_run_session(owner, a,
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], opts)
    except error:
        # Preserve original drain/clear on computational failure. The caller
        # releases busy only after reacquiring its GIL (or native lock).
        owner.clear()
        raise error
