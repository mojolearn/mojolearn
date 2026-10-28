# SPDX-License-Identifier: Apache-2.0
# Shared execution owner for the Python door and native generation ABI.
# Numerical implementation remains _transformer_run_session, unchanged.
from std.ffi import _Global
from transformer.session_abi import (
    GENERATION_SESSION_ABI_NAME, GenerationErrorSink, GenerationSessionForward,
)
from transformer.block_options import BLOCK_OPTION_ADDRS, BLOCK_OPTION_PARAMS, BlockOptions
from transformer.transformer_binding_impl import (
    TransformerSession, _check_base_addrs, _transformer_run_session,
)


def transformer_session_forward_native(
    mut owner: TransformerSession, a: List[Int], p: List[Int], opts: BlockOptions,
) raises -> Int:
    # Caller validates metadata and reserves owner.busy before releasing its
    # interpreter GIL. Both doors share this execution/cleanup implementation.
    try:
        return _transformer_run_session(owner, a,
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], opts)
    except error:
        # Preserve original drain/clear on computational failure. The caller
        # releases busy only after reacquiring its GIL (or native lock).
        owner.clear()
        raise error


def transformer_session_forward_v1(
    phase: Int, session: Int, addresses: Int, n_addresses: Int, parameters: Int, n_parameters: Int,
    result_address: Int, error_context: Int, error_sink: GenerationErrorSink,
) abi("C") -> Int:
    try:
        if session == 0 or result_address == 0:
            raise Error("transformer native ABI: null session or result")
        var owner = MutPointer[TransformerSession, MutUntrackedOrigin](unsafe_from_address=session)
        if phase == 0:
            if owner[].busy or owner[].closed:
                raise Error("transformer: session is busy or closed")
            owner[].busy = True
            return 0
        if phase == 2:
            owner[].busy = False
            return 0
        if phase != 1 or not owner[].busy or owner[].closed:
            raise Error("transformer native ABI: execution requires a reserved open session")
        if n_addresses != 13 and n_addresses != 13 + BLOCK_OPTION_ADDRS:
            raise Error("transformer_forward: addrs must contain 13 addresses, or 13 + "
                + String(BLOCK_OPTION_ADDRS) + " with the block options tail, got " + String(n_addresses))
        if addresses == 0:
            raise Error("transformer native ABI: null address vector")
        var ap = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=addresses)
        var a = List[Int]()
        for i in range(n_addresses):  # small-loop(validated 13 or 24 native address words)
            a.append(ap[i])
        while len(a) < 13 + BLOCK_OPTION_ADDRS:  # small-loop(pad absent option address tail)
            a.append(0)
        # Match Python door refusal order: address checks precede scalars.
        _check_base_addrs(a, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], String("transformer_forward"))
        if n_parameters != 10 and n_parameters != 10 + BLOCK_OPTION_PARAMS:
            raise Error("transformer_forward: params must contain 10 values, or 10 + "
                + String(BLOCK_OPTION_PARAMS) + " with the block options tail, got " + String(n_parameters))
        if parameters == 0:
            raise Error("transformer native ABI: null parameter vector")
        var pp = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=parameters)
        var p = List[Int]()
        for i in range(n_parameters):  # small-loop(validated 10 or 27 native scalar words)
            p.append(pp[i])
        var opts = BlockOptions.from_params(p, 10)
        var result = transformer_session_forward_native(owner[], a, p, opts)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result_address)[0] = result
        return 0
    except problem:
        # Reservation is released only by phase 2 after the provider has
        # reacquired its GIL. A refusal never unlocks another caller's work.
        var message = String(problem)
        var bytes = message.as_bytes()
        # Only a compiled native sink crosses this ABI. It copies bytes into
        # caller-owned native storage while message remains alive, without
        # truncating an exception or freeing an allocation in another image.
        error_sink(error_context, Int(bytes.unsafe_ptr()), len(bytes))
        _ = bytes
        _ = message
        return 1


struct TransformerNativeAPI(Defaultable, Movable):
    var name: String
    var entries: List[GenerationSessionForward]

    def __init__(out self):
        self.name = String(GENERATION_SESSION_ABI_NAME)
        self.entries = List[GenerationSessionForward]()
        self.entries.append(transformer_session_forward_v1)


comptime TRANSFORMER_NATIVE_API = _Global[
    StorageType=TransformerNativeAPI, name="MojoTransformerSessionNativeAPIv1",
    init_fn=TransformerNativeAPI.__init__,
]
