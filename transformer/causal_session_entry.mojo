# SPDX-License-Identifier: Apache-2.0
"""`_mojolearn_transformer`'s CausalLM session entry (lane
causal-lm-sessions-native, 2026-11-11), family CAUSAL_SESSIONS_TRANSFORMER of
`core/causal_lm_entry_table_sessions.mojo`.

The resident generation sessions live in this image's state registries; a
caller holds handle words only. Every operation acquires its leases, then runs
the existing session dispatch on the leased owners
(`transformer/native_decode_dispatch.mojo`,
`transformer/native_lm_resident_dispatch.mojo`) with the same reservation,
failure and cache semantics the Python session entries had. The leases keep
every owner alive through the call even if another thread releases its
handle. Numerical session bodies are untouched.

The registries are named per tier: an identical and a fast image in one
process never share session storage.
"""
from std.memory import ArcPointer, alloc
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.state_registry import StateRegistry
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.native_session_group import NativeSessionGroup
from core.shared_slot import SharedSlot
from core.causal_lm_entry_table_sessions import (
    causal_session_handle_word, causal_session_handle_of, CAUSAL_SESSIONS_SCHEMA, CAUSAL_SESSIONS_OPERATIONS,
    CAUSAL_SESSION_DECODE_CREATE, CAUSAL_SESSION_DECODE_OPEN, CAUSAL_SESSION_DECODE_OPEN15,
    CAUSAL_SESSION_DECODE_EXPORT, CAUSAL_SESSION_DECODE_LOAD, CAUSAL_SESSION_DECODE_CLOSE,
    CAUSAL_SESSION_LM_CREATE, CAUSAL_SESSION_LM_OPEN, CAUSAL_SESSION_LM_OPEN15, CAUSAL_SESSION_LM_RUN,
    CAUSAL_SESSION_LM_FORWARD, CAUSAL_SESSION_LM_CLOSE, CAUSAL_SESSION_PROGRAM_READ, CAUSAL_SESSION_PROGRAM_RELEASE,
    CAUSAL_SESSION_DECODE_PIN, CAUSAL_SESSION_DECODE_UNPIN, CAUSAL_SESSION_DECODE_PINNED,
)
from transformer.transformer_binding_impl import TransformerDecodeSession, CausalLMSession
from transformer.native_decode_dispatch import transformer_decode_operation, transformer_decode_reserved_run
from transformer.native_lm_resident_dispatch import transformer_lm_resident_operation
from transformer.causal_program_entry import causal_program_operation

comptime DECODE_SESSIONS = StateRegistry[TransformerDecodeSession,
    ("MojolearnCausalDecodeSessionsIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnCausalDecodeSessionsFastV1")]
comptime LM_SESSIONS = StateRegistry[CausalLMSession,
    ("MojolearnCausalLMSessionsIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnCausalLMSessionsFastV1")]
comptime TRANSFORMER_PROGRAMS_NAME = ("MojolearnCausalProgramsTransformerIdenticalV1" if GLOBAL_NUMERIC_MODE == 1
    else "MojolearnCausalProgramsTransformerFastV1")

# Codes of the existing dispatch the operations reuse
# (transformer_decode_operation, transformer_lm_resident_operation).
comptime _DECODE_RESERVE_CLOSED = 0
comptime _DECODE_RESERVE_OPEN = 1
comptime _DECODE_UNRESERVE = 2
comptime _DECODE_OPEN = 3
comptime _DECODE_OPEN15 = 4
comptime _DECODE_EXPORT = 7
comptime _DECODE_LOAD = 8
comptime _DECODE_CLOSE = 9
comptime _LM_RESERVE = 25
comptime _LM_UNRESERVE = 26
comptime _LM_OPEN = 27
comptime _LM_OPEN15 = 28
comptime _LM_CLOSE = 29
comptime _GROUP_RESERVE = 30
comptime _GROUP_RELEASE = 31
comptime _GROUP_EXECUTE = 32

#: A door's lease on one decode session (CAUSAL_SESSION_DECODE_PIN): a heap
#: box holding a copy of the registry's Arc, so the session stays alive while
#: the door's handle holds the pin word, whatever the registry does.
comptime _DecodePin = UnsafePointer[ArcPointer[TransformerDecodeSession], MutUntrackedOrigin]


def _decode_address(lease: ArcPointer[TransformerDecodeSession]) -> Int:
    return Int(UnsafePointer(to=lease[]))


def _lm_address(lease: ArcPointer[CausalLMSession]) -> Int:
    return Int(UnsafePointer(to=lease[]))


def _decode_reserved(lease: ArcPointer[TransformerDecodeSession], reserve: Int, op: Int,
    a: List[Int], scalars: List[Int]) raises -> Int:
    # Reserve, run, unreserve: the Python entry's busy protocol. A failed
    # operation leaves the owner as the dispatch left it (released after a
    # failed open, unusable after a failed export or load) and not busy.
    var owner: List[Int] = [_decode_address(lease)]
    _ = transformer_decode_operation(reserve, List[Int](), owner)
    var full = owner.copy()
    for i in range(len(scalars)):  # driver-loop: admitted native decode scalar span
        full.append(scalars[i])
    var result = 0
    try:
        result = transformer_decode_operation(op, a, full)
    except problem:
        _ = transformer_decode_operation(_DECODE_UNRESERVE, List[Int](), owner)
        _ = lease
        raise problem
    _ = transformer_decode_operation(_DECODE_UNRESERVE, List[Int](), owner)
    _ = lease
    return result


def _lm_group(operation: Int, lease: ArcPointer[CausalLMSession], words: List[Int],
    a: List[Int]) raises -> Int:
    # words: [h, n, layer handles..., scalars...]. Every layer lease is
    # held through the grouped reservation, execution and release.
    if len(words) < 2 or words[1] < 1 or len(words) < 2 + words[1]:
        raise Error("causal lm resident: invalid layer handle span")
    var n = words[1]
    var layers = List[ArcPointer[TransformerDecodeSession]](capacity=n)
    var addresses = List[Int](capacity=n)
    for i in range(n):  # driver-loop: over decoder layers of the model
        layers.append(DECODE_SESSIONS.acquire(causal_session_handle_of(words[2 + i])))
        addresses.append(_decode_address(layers[i]))
    var scalars = List[Int]()
    for i in range(2 + n, len(words)):  # driver-loop: admitted group scalar span
        scalars.append(words[i])
    var group = NativeSessionGroup(_lm_address(lease), operation, addresses^, a.copy(), scalars^)
    var descriptor: List[Int] = [Int(UnsafePointer(to=group))]
    _ = transformer_lm_resident_operation(_GROUP_RESERVE, List[Int](), descriptor, List[Float64]())
    var result = 0
    try:
        result = transformer_lm_resident_operation(_GROUP_EXECUTE, List[Int](), descriptor, List[Float64]())
    except problem:
        _ = transformer_lm_resident_operation(_GROUP_RELEASE, List[Int](), descriptor, List[Float64]())
        _ = group^
        _ = layers^
        _ = lease
        raise problem
    _ = transformer_lm_resident_operation(_GROUP_RELEASE, List[Int](), descriptor, List[Float64]())
    _ = group^
    _ = layers^
    _ = lease
    return result


def causal_session_operation(op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    if op == CAUSAL_SESSION_PROGRAM_READ or op == CAUSAL_SESSION_PROGRAM_RELEASE:
        return causal_program_operation[TRANSFORMER_PROGRAMS_NAME](op, a, words, f)
    if op == CAUSAL_SESSION_DECODE_CREATE or op == CAUSAL_SESSION_LM_CREATE:
        if len(a) != 0 or len(words) != 0 or len(f) != 0:
            raise Error("causal lm sessions: create takes no arguments")
        # A closed session: creation performs no GPU operation.
        if op == CAUSAL_SESSION_DECODE_CREATE:
            return causal_session_handle_word(DECODE_SESSIONS.insert(TransformerDecodeSession()))
        return causal_session_handle_word(LM_SESSIONS.insert(CausalLMSession()))
    if len(words) == 0:
        raise Error("causal lm sessions: missing session handle")
    var handle = causal_session_handle_of(words[0])
    var scalars = List[Int]()
    for i in range(1, len(words)):  # driver-loop: admitted session scalar span
        scalars.append(words[i])
    if op >= CAUSAL_SESSION_DECODE_OPEN and op <= CAUSAL_SESSION_DECODE_LOAD:
        if len(f) != 0:
            raise Error("causal lm sessions: unexpected floating scalars")
        var lease = DECODE_SESSIONS.acquire(handle)
        if op == CAUSAL_SESSION_DECODE_OPEN:
            return _decode_reserved(lease, _DECODE_RESERVE_CLOSED, _DECODE_OPEN, a, scalars)
        if op == CAUSAL_SESSION_DECODE_OPEN15:
            return _decode_reserved(lease, _DECODE_RESERVE_CLOSED, _DECODE_OPEN15, a, scalars)
        if op == CAUSAL_SESSION_DECODE_EXPORT:
            return _decode_reserved(lease, _DECODE_RESERVE_OPEN, _DECODE_EXPORT, a, scalars)
        return _decode_reserved(lease, _DECODE_RESERVE_OPEN, _DECODE_LOAD, a, scalars)
    if op == CAUSAL_SESSION_DECODE_PIN:
        # Once per door handle (at its open): the lease the door's per-token
        # calls run on (`_decode_pinned`), dropped by UNPIN when the door's
        # handle is destroyed.
        if len(a) != 0 or len(scalars) != 0 or len(f) != 0:
            raise Error("causal lm sessions: invalid pin descriptor")
        var box = alloc[ArcPointer[TransformerDecodeSession]](1)
        box.unsafe_write(DECODE_SESSIONS.acquire(handle))
        return Int(box)
    if op == CAUSAL_SESSION_DECODE_CLOSE:
        if len(a) != 0 or len(scalars) != 0 or len(f) != 0:
            raise Error("causal lm sessions: invalid close descriptor")
        var lease = DECODE_SESSIONS.acquire(handle)
        var owner: List[Int] = [_decode_address(lease)]
        # Refuses a busy session, as the Python close did; the handle stays.
        _ = transformer_decode_operation(_DECODE_CLOSE, List[Int](), owner)
        _ = lease
        DECODE_SESSIONS.release(handle)
        return 0
    var lm = LM_SESSIONS.acquire(handle)
    if op == CAUSAL_SESSION_LM_OPEN or op == CAUSAL_SESSION_LM_OPEN15:
        if len(scalars) != 3 or len(f) != 1:
            raise Error("causal lm session open: expected B, V, D and eps")
        var owner: List[Int] = [_lm_address(lm)]
        _ = transformer_lm_resident_operation(_LM_RESERVE, List[Int](), owner, List[Float64]())
        var p = owner.copy()
        for i in range(3):  # small-loop(B, V, D)
            p.append(scalars[i])
        try:
            _ = transformer_lm_resident_operation(_LM_OPEN15 if op == CAUSAL_SESSION_LM_OPEN15 else _LM_OPEN, a, p, f)
        except problem:
            _ = transformer_lm_resident_operation(_LM_UNRESERVE, List[Int](), owner, List[Float64]())
            _ = lm
            raise problem
        _ = transformer_lm_resident_operation(_LM_UNRESERVE, List[Int](), owner, List[Float64]())
        _ = lm
        return 0
    if op == CAUSAL_SESSION_LM_RUN:
        if len(f) != 0:
            raise Error("causal lm sessions: unexpected floating scalars")
        return _lm_group(1, lm, words, a)
    if op == CAUSAL_SESSION_LM_FORWARD:
        if len(f) != 0:
            raise Error("causal lm sessions: unexpected floating scalars")
        return _lm_group(0, lm, words, a)
    if op == CAUSAL_SESSION_LM_CLOSE:
        if len(a) != 0 or len(scalars) != 0 or len(f) != 0:
            raise Error("causal lm sessions: invalid close descriptor")
        var owner: List[Int] = [_lm_address(lm)]
        # Refuses a busy session, as the Python close did; the handle stays.
        _ = transformer_lm_resident_operation(_LM_CLOSE, List[Int](), owner, List[Float64]())
        _ = lm
        LM_SESSIONS.release(handle)
        return 0
    raise Error("causal lm sessions: unknown operation")


def _decode_unpin(pin: Int) raises:
    if pin == 0:
        raise Error("causal lm sessions: null decode pin")
    var box = _DecodePin(unsafe_from_address=pin)
    box.unsafe_deinit_pointee()
    box.free()


def _decode_pinned(pin: Int, kind: Int, count: Int, cached: Int, x: Int, y: Int) raises -> Int:
    # The door's per-token step (and its decode / forward) on the session its
    # pin leases: typed words, no registry lookup, no lock, no List.
    if pin == 0:
        raise Error("causal lm sessions: null decode pin")
    var box = _DecodePin(unsafe_from_address=pin)
    return transformer_decode_reserved_run(box[][], kind, x, y, count, cached)


def causal_sessions_transformer_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                                   result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op == CAUSAL_SESSION_DECODE_PINNED or op == CAUSAL_SESSION_DECODE_UNPIN:
            # Hot-path audit 2026-11-13: read the caller's envelopes in place
            # (two address words, four or one scalar words), no native List.
            if result == 0:
                raise Error("causal lm sessions native ABI: null result")
            var words = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
            if op == CAUSAL_SESSION_DECODE_UNPIN:
                if na != 0 or np != 1 or nf != 0 or pp == 0:
                    raise Error("causal lm sessions: invalid unpin descriptor")
                _decode_unpin(words[0])
                MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = 0
                return 0
            if na != 2 or np != 4 or nf != 0 or aa == 0 or pp == 0:
                raise Error("causal lm sessions: a pinned decode call takes [x, y] and [pin, kind, count, cached]")
            var buffers = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = _decode_pinned(
                words[0], words[1], words[2], words[3], buffers[0], buffers[1])
            return 0
        if op < 0 or op >= CAUSAL_SESSIONS_OPERATIONS or na < 0 or na > 4096 or np < 0 or np > 4096 or nf < 0 or nf > 64:
            raise Error("causal lm sessions native ABI: unknown operation or span")
        if result == 0:
            raise Error("causal lm sessions native ABI: null result")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64](capacity=nf)
        if nf != 0:
            if ff == 0:
                raise Error("causal lm sessions native ABI: null floating scalar span")
            var src = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
            for i in range(nf):  # driver-loop: admitted floating scalar span
                f.append(src[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = causal_session_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeCausalSessionsAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(CAUSAL_SESSIONS_SCHEMA)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(causal_sessions_transformer_v1)


comptime NATIVE_CAUSAL_SESSIONS_API = SharedSlot[NativeCausalSessionsAPI,
    ("MojolearnCausalSessionsTransformerAPIIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnCausalSessionsTransformerAPIFastV1")]
