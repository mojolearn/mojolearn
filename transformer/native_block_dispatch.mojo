# SPDX-License-Identifier: Apache-2.0
# Private sibling ABI for the existing fresh/backward Transformer operations.
# Common descriptor transport; no Python/provider imports or changed kernels.
from checks.numerics import GLOBAL_NUMERIC_MODE
from transformer.native_lm_resident_dispatch import transformer_lm_resident_operation
from transformer.native_decode_dispatch import transformer_decode_operation
from transformer.session_native_dispatch import transformer_session_forward_native
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, TRANSFORMER_BLOCK_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from transformer.block_options import BLOCK_OPTION_ADDRS, BLOCK_OPTION_PARAMS, BlockOptions
from transformer.transformer_binding_impl import (
    TransformerSession, _check_base_addrs, _transformer_run_session,
    _transformer_run_session_backward, _transformer_run, _transformer_backward_run,
    _transformer_run_session_int15, INT15_ENTRY_ADDRS, INT15_ENTRY_PARAMS,
)
from core.shared_slot import SharedSlot
from core.state_registry import StateRegistry
from core.neural_entry_table import NEURAL_BLOCK_OPERATIONS, NEURAL_BLOCK_SESSION_CREATE, NEURAL_BLOCK_SESSION_CLOSE
from core.transformer_block_session import transformer_block_session_word, transformer_block_session_of

# 0 reserve, 1 release (caller GIL/lock held); 2 session fresh, 3 session
# backward; 4 ephemeral fresh, 5 ephemeral backward; 11 session forward, 35
# session int15 forward, 13 session close, 36 session create. Session
# operations lead their integer vector with the session's registry handle word
# (lane transformer-session-native, 2026-11-12: the session lives in BLOCK_SESSIONS
# below, never in a Python object; core/transformer_block_session.mojo). All
# spans borrowed.
# TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
# Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
# (standalone tape ops 6..10 and 14, session invalidate-tape op 12 and the
# session-ticket ops 36..40 are refused as unknown operations; op 36 is now
# session create.)

# The image's block sessions, named per tier: an identical and a fast image in
# one process never share session storage.
comptime BLOCK_SESSIONS = StateRegistry[TransformerSession,
    ("MojolearnTransformerBlockSessionsIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnTransformerBlockSessionsFastV1")]


def _transformer_block_session_operation(op: Int, session: Int, a: List[Int], words: List[Int]) raises -> Int:
    # `session` is the address of a leased registry owner; the caller holds
    # the lease through this call.
    var owner = MutPointer[TransformerSession, MutUntrackedOrigin](unsafe_from_address=session)
    if op == 0:
        if owner[].busy or owner[].closed:
            raise Error("transformer: session is busy or closed")
        owner[].busy = True
        return 0
    if op == 1:
        owner[].busy = False
        return 0
    if not owner[].busy or owner[].closed:
        raise Error("transformer block native ABI: execution requires a reserved open session")
    var p = List[Int](capacity=len(words))
    for i in range(1, len(words)):  # small-loop(checked session operation scalar span)
        p.append(words[i])
    if op == 35:
        if len(a) != INT15_ENTRY_ADDRS or len(p) != INT15_ENTRY_PARAMS:
            raise Error("transformer_session_forward_int15: invalid native descriptor arity")
        _check_base_addrs(a, [0, 1, 2, 10, 11, 12], String("transformer_session_forward_int15"))
        var opts = BlockOptions.from_params(p, 10)
        try:
            return _transformer_run_session_int15(owner[], a,
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], opts)
        except problem:
            owner[].clear()
            raise problem
    if op == 11:
        if len(a) != 13 and len(a) != 13 + BLOCK_OPTION_ADDRS:
            raise Error("transformer_forward: invalid native address span")
        var expanded = a.copy()
        while len(expanded) < 13 + BLOCK_OPTION_ADDRS:  # small-loop(pad absent optional address words)
            expanded.append(0)
        _check_base_addrs(expanded, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], String("transformer_forward"))
        if len(p) != 10 and len(p) != 10 + BLOCK_OPTION_PARAMS:
            raise Error("transformer_forward: invalid native scalar span")
        var opts = BlockOptions.from_params(p, 10)
        return transformer_session_forward_native(owner[], expanded, p, opts)
    if op == 2:
        var name = "transformer_session_forward_fresh"
        if len(a) != 13 and len(a) != 13 + BLOCK_OPTION_ADDRS:
            raise Error(name + ": addrs must contain 13 addresses, or 13 + " + String(BLOCK_OPTION_ADDRS)
                + " with the block options tail, got " + String(len(a)))
        var expanded = a.copy()
        while len(expanded) < 13 + BLOCK_OPTION_ADDRS:  # small-loop(pad existing optional block address tail)
            expanded.append(0)
        _check_base_addrs(expanded, [0, 1, 2, 3, 4, 5, 6, 8, 9, 12], name)
        if len(p) != 10 and len(p) != 10 + BLOCK_OPTION_PARAMS:
            raise Error(name + ": params must contain 10 values, or 10 + " + String(BLOCK_OPTION_PARAMS)
                + " with the block options tail, got " + String(len(p)))
        var opts = BlockOptions.from_params(p, 10)
        try:
            return _transformer_run_session[True](owner[], expanded,
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], opts)
        except problem:
            owner[].clear()
            raise problem
    # op 3: session backward.
    if len(a) != 21 or len(p) != 8:
        raise Error("transformer backward: expected 21 addresses and 8 scalars (B,"
            " L, d_model, n_heads, n_kv_heads, head_dim, intermediate, window)")
    for i in range(21):  # small-loop(original backward buffer slots)
        if a[i] == 0:
            raise Error("transformer backward: null buffer address at slot " + String(i))
    try:
        _transformer_run_session_backward(owner[], a, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    except problem:
        owner[].clear()
        raise problem
    return 0


def _transformer_block_ephemeral(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    # Ops 4 (fresh forward) and 5 (backward): no session.
    if op == 4:
        var name = "transformer_forward_fresh"
        if len(a) != 11 and len(a) != 11 + BLOCK_OPTION_ADDRS:
            raise Error(name + ": addrs must contain 11 addresses, or 11 + " + String(BLOCK_OPTION_ADDRS)
                + " with the block options tail, got " + String(len(a)))
        var raw = a.copy()
        while len(raw) < 11 + BLOCK_OPTION_ADDRS:  # small-loop(pad existing optional block address tail)
            raw.append(0)
        if len(p) != 8 and len(p) != 8 + BLOCK_OPTION_PARAMS:
            raise Error(name + ": params must contain 8 values, or 8 + " + String(BLOCK_OPTION_PARAMS)
                + " with the block options tail, got " + String(len(p)))
        var opts = BlockOptions.from_params(p, 8)
        var expanded = List[Int]()
        for i in range(10):  # small-loop(original ten input/weight address slots)
            expanded.append(raw[i])
        expanded.append(0)
        expanded.append(0)
        expanded.append(raw[10])
        for i in range(BLOCK_OPTION_ADDRS):  # small-loop(original optional address tail)
            expanded.append(raw[11 + i])
        _check_base_addrs(expanded, [0, 1, 2, 3, 4, 5, 6, 8, 9], name)
        if expanded[12] == 0:
            raise Error("transformer_forward_fresh: null output address")
        if p[0] <= 0 or p[1] <= 0:
            raise Error("transformer_forward_fresh: B and L must be positive")
        return _transformer_run[True](expanded,
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[1], 0, p[7], opts)
    if op != 5:
        raise Error("transformer block native ABI: unknown operation")
    if len(a) != 21 or len(p) != 8:
        raise Error("transformer backward: expected 21 addresses and 8 scalars (B,"
            " L, d_model, n_heads, n_kv_heads, head_dim, intermediate, window)")
    for i in range(21):  # small-loop(original backward buffer slots)
        if a[i] == 0:
            raise Error("transformer backward: null buffer address at slot " + String(i))
    if p[0] <= 0 or p[1] <= 0 or p[2] <= 0:
        raise Error("transformer backward: B, L and d_model must be positive")
    _transformer_backward_run(a, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    return 0


def transformer_block_operation(op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    if op >= 25 and op < 33:
        return transformer_lm_resident_operation(op, a, words, f)
    if len(f) != 0:
        raise Error("transformer block: unexpected floating scalars")
    if op >= 15 and op < 25:
        return transformer_decode_operation(op - 15, a, words)
    if op == NEURAL_BLOCK_SESSION_CREATE:
        if len(a) != 0 or len(words) != 0:
            raise Error("transformer session: create takes no arguments")
        # A fresh session: creation performs no GPU operation.
        return transformer_block_session_word(BLOCK_SESSIONS.insert(TransformerSession()))
    if op == NEURAL_BLOCK_SESSION_CLOSE:
        if len(words) != 1 or len(a) != 0:
            raise Error("transformer session administration: invalid native descriptor")
        var handle = transformer_block_session_of(words[0])
        var lease = BLOCK_SESSIONS.acquire(handle)
        var owner = MutPointer[TransformerSession, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=lease[])))
        if owner[].busy:
            _ = lease
            # Refuses a busy session, as the Python close did; the handle stays.
            raise Error("transformer: session is busy or closed")
        owner[].closed = True
        try:
            owner[].clear()
        except problem:
            _ = lease
            raise problem
        owner[].ctx = None
        _ = lease
        BLOCK_SESSIONS.release(handle)
        return 0
    if (op >= 6 and op < 11) or op == 12 or op == 14:
        raise Error("transformer block native ABI: unknown operation")
    if op < 4 or op == 11 or op == 35:
        if len(words) == 0:
            raise Error("transformer block native ABI: null session")
        # The lease keeps the owner alive through the call even if another
        # thread closes the handle meanwhile (a close of a busy session
        # refuses).
        var lease = BLOCK_SESSIONS.acquire(transformer_block_session_of(words[0]))
        var result = 0
        try:
            result = _transformer_block_session_operation(op, Int(UnsafePointer(to=lease[])), a, words)
        except problem:
            _ = lease
            raise problem
        _ = lease
        return result
    return _transformer_block_ephemeral(op, a, words)


def transformer_block_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                         result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= NEURAL_BLOCK_OPERATIONS or nf < 0 or nf > 1 or na < 0 or na > 45 or np < 0 or np > 28:
            raise Error("transformer block native ABI: unknown operation or scalar span")
        if result == 0:
            raise Error("transformer block native ABI: null result")
        if op < 2 and (na != 0 or np != 1):
            raise Error("transformer block native ABI: reservation expects only one session handle")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        if nf != 0:
            if ff == 0:
                raise Error("transformer block native ABI: null floating scalar span")
            f.append(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)[0])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = transformer_block_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeTransformerBlockAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(TRANSFORMER_BLOCK_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(transformer_block_v1)


comptime NATIVE_TRANSFORMER_BLOCK_API = SharedSlot[NativeTransformerBlockAPI, ("MojolearnTransformerBlockNativeAPIIdenticalV3" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnTransformerBlockNativeAPIFastV3")]
