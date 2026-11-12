# SPDX-License-Identifier: Apache-2.0
"""`_mojolearn_training`'s Samba program entry (lane samba-sessions-native,
2026-11-12): the table `core/samba_entry_table_programs.mojo`.

The stack programs (`SambaStackProgram`, training/samba_program.mojo) live in
this image's state registry, named per tier (an identical and a fast image in
one process never share program storage). A caller holds a handle word only.
Every operation acquires its lease, reserves the program (the former Python
entries' `busy` / `usable` protocol, now under a native lock word because the
caller releases the GIL around the call), runs the SAME program methods in
the SAME order the nine Python entries ran them, and unreserves. The lease
keeps the program alive through the call even if another thread releases its
handle. Numerical bodies are untouched.

Moved here from the Python entries of bindings/_mojolearn_training.mojo
(`samba_program_open/shape/forward_binding`, `samba_decode_open/step/tokens/
export/load_binding`, `_samba_program_new`, `_decode_run`), which this lane
deleted. Each operation is one program call: nothing to split.
"""
from std.memory import ArcPointer
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.hostptr import f32_ptr, i32_ptr
from core.neural_context import neural_ctx
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.state_registry import StateRegistry
from core.samba_entry_table_programs import (
    samba_program_handle_word, samba_program_handle_of, SAMBA_PROGRAMS_SCHEMA,
    SAMBA_PROGRAM_OPEN, SAMBA_PROGRAM_STATUS, SAMBA_PROGRAM_FORWARD, SAMBA_PROGRAM_DECODE_STEP,
    SAMBA_PROGRAM_DECODE_TOKENS, SAMBA_PROGRAM_DECODE_EXPORT, SAMBA_PROGRAM_DECODE_LOAD, SAMBA_PROGRAM_RELEASE,
)
from training.native_primitive_dispatch import PRIMITIVE_CONTEXT
from training.samba_program import SambaStackProgram, SAMBA_PROGRAM_ON

comptime SAMBA_PROGRAMS = StateRegistry[SambaStackProgram,
    ("MojolearnSambaProgramsIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnSambaProgramsFastV1")]


struct SambaProgramGate(Defaultable, Movable):
    """The lock word that makes the busy check-and-set atomic: the former
    entries tested and set `busy` under the GIL."""

    var lock: List[Int64]

    def __init__(out self):
        self.lock = List[Int64](length=1, fill=0)


comptime SAMBA_PROGRAM_GATE = SharedSlot[SambaProgramGate,
    ("MojolearnSambaProgramGateIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnSambaProgramGateFastV1")]


def _gate() raises -> MutPointer[Int64, MutUntrackedOrigin]:
    var gate = SAMBA_PROGRAM_GATE.get_or_create_ptr()
    return rebind[MutPointer[Int64, MutUntrackedOrigin]](gate[].lock.unsafe_ptr())


def _owner(lease: ArcPointer[SambaStackProgram]) -> MutPointer[SambaStackProgram, MutUntrackedOrigin]:
    return MutPointer[SambaStackProgram, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=lease[])))


def _reserve(owner: MutPointer[SambaStackProgram, MutUntrackedOrigin], name: String, need_usable: Bool,
             lost: String) raises:
    var lock = _gate()
    shared_slot_lock(lock)
    try:
        if owner[].busy:
            var holder = String("program") if lost == "" else String("session")
            raise Error(name + ": " + holder + " busy")
        if need_usable and not owner[].usable:
            var why = String(": the program was lost after a failed call") if lost == "" else lost
            raise Error(name + why)
        owner[].busy = True
    finally:
        shared_slot_unlock(lock)


def _unreserve(owner: MutPointer[SambaStackProgram, MutUntrackedOrigin], lose: Bool, restore: Bool) raises:
    var lock = _gate()
    shared_slot_lock(lock)
    owner[].busy = False
    if lose:
        owner[].usable = False
    if restore:
        owner[].usable = True
    shared_slot_unlock(lock)


def _nonnull(words: List[Int], begin: Int, name: String) raises -> List[Int]:
    var kept = List[Int](capacity=len(words) - begin)
    for i in range(begin, len(words)):  # small-loop(state addresses of one program call): argument words, not data
        if words[i] == 0:
            raise Error(name + ": null state address at slot " + String(i - begin))
        kept.append(words[i])
    return kept^


def _tail(words: List[Int], begin: Int) -> List[Int]:
    var kept = List[Int](capacity=len(words) - begin)
    for i in range(begin, len(words)):  # small-loop(cursor words of one program call): argument words, not data
        kept.append(words[i])
    return kept^


def _write_words(address: Int, words: List[Int], capacity: Int, name: String) raises:
    if address == 0:
        raise Error(name + ": null cursor output address")
    if len(words) > capacity:
        raise Error(name + ": the cursor output holds " + String(capacity) + " words, the program wrote "
                    + String(len(words)))
    var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(len(words)):  # small-loop(two cursor words per layer): result words, not data
        dst[i] = words[i]


def _program_new(a: List[Int], p: List[Int], f: List[Float64], name: String) raises -> SambaStackProgram:
    # p = [n_layers, kinds..., vocab, d_model, n_heads, n_kv_heads, head_dim,
    # intermediate, tie, n_total, batch_size, max_tokens, cursors...]
    if len(p) < 1 or len(f) != 1 or len(a) < 1:
        raise Error(name + ": params must contain the layer codes, 10 shape words and norm_eps")
    var n = p[0]
    if n < 1 or len(p) < 11 + n:
        raise Error(name + ": params must contain " + String(n) + " layer codes and 10 shape words")
    var kinds = List[Int](capacity=n)
    for i in range(n):  # driver-loop: over layers; small-loop(one kind code per layer): reads codes, not data
        kinds.append(p[1 + i])
    var s = 1 + n
    if a[0] == 0:
        raise Error(name + ": null registry address")
    var ctx = neural_ctx[PRIMITIVE_CONTEXT]()
    return SambaStackProgram(
        ctx, kinds, p[s], p[s + 1], p[s + 2], p[s + 3], p[s + 4], p[s + 5], p[s + 6] != 0, Float32(f[0]),
        p[s + 7], p[s + 8], p[s + 9], f32_ptr(a[0]),
    )


def _open(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    var stateful = len(a) > 1
    var name = String("samba_decode_open") if stateful else String("samba_program_open")
    var program = _program_new(a, p, f, name)
    if stateful:
        # A decode session: the state loaded once here (the former
        # samba_decode_open_binding).
        var pieces = _nonnull(a, 1, name)
        var cursors = _tail(p, 11 + p[0])
        program.load_state(pieces, cursors)
        program.finish()
    elif len(p) != 11 + p[0]:
        raise Error(name + ": cursor words are given without state pieces")
    return samba_program_handle_word(SAMBA_PROGRAMS.insert(program^))


def _forward(owner: MutPointer[SambaStackProgram, MutUntrackedOrigin], a: List[Int], p: List[Int]) raises -> Int:
    # The former samba_program_forward_binding.
    var name = String("samba_program_forward")
    if len(a) < 4 or len(p) < 4:
        raise Error(name + ": addresses must hold [registry, ids, logits, cursors_out] and params [h, b, l, n_reports]")
    var n_reports = p[3]
    if n_reports < 0 or 4 + n_reports > len(a):
        raise Error(name + ": invalid report address count")
    var reports = List[Int](capacity=n_reports)
    for i in range(n_reports):  # driver-loop: over the Mamba-3 layers' report addresses
        reports.append(a[4 + i])
    var stateful = 4 + n_reports < len(a)
    var sp = _nonnull(a, 4 + n_reports, name) if stateful else List[Int]()
    var sc = _tail(p, 4) if stateful else List[Int]()
    if not stateful and len(p) != 4:
        raise Error(name + ": cursor words are given without state pieces")
    var registry = f32_ptr(a[0])
    var ids = i32_ptr(a[1])
    var logits = f32_ptr(a[2])
    var b = p[1]
    var l = p[2]
    if b != owner[].b:
        raise Error(name + ": the program holds " + String(owner[].b) + " rows, the call has " + String(b))
    if stateful and a[3] == 0:
        raise Error(name + ": null cursor output address")
    _reserve(owner, name, True, "")
    var exported = List[Int]()
    try:
        owner[].refresh_weights(registry)
        if stateful:
            owner[].load_state(sp, sc)
        else:
            owner[].zero_state()
        owner[].run(ids, logits, l)
        if len(reports) > 0:
            owner[].enqueue_reports(reports)
        if stateful:
            exported = owner[].enqueue_export(sp)
        owner[].finish()
    except error:
        _unreserve(owner, True, False)
        raise error
    _unreserve(owner, False, False)
    if stateful:
        _write_words(a[3], exported, len(sc), name)
    return 0


def _decode_run(owner: MutPointer[SambaStackProgram, MutUntrackedOrigin], a: List[Int], p: List[Int],
                tokens: Bool) raises -> Int:
    # The former _decode_run of bindings/_mojolearn_training.mojo.
    var name = String("samba_decode_tokens") if tokens else String("samba_decode_step")
    var want = 3 if tokens else 2
    if len(a) != 2:
        raise Error(name + ": addresses must contain 2 entries, got " + String(len(a)))
    if len(p) != want:
        raise Error(name + ": params must contain " + String(want - 1) + " values, got " + String(len(p) - 1))
    var ids = i32_ptr(a[0])
    var logits = f32_ptr(a[1])
    if p[1] != owner[].b:
        raise Error(name + ": the session holds " + String(owner[].b) + " rows, the call has " + String(p[1]))
    var n = p[2] if tokens else 1
    _reserve(owner, name, True, ": the session was lost after a failed call; close it and open a new one")
    try:
        if tokens:
            owner[].decode(ids, logits, n)
        else:
            owner[].run(ids, logits, 1)
            owner[].finish()
    except error:
        _unreserve(owner, True, False)
        raise error
    _unreserve(owner, False, False)
    return owner[].first_position()


def _export(owner: MutPointer[SambaStackProgram, MutUntrackedOrigin], a: List[Int], p: List[Int]) raises -> Int:
    # The former samba_decode_export_binding.
    var name = String("samba_decode_export")
    if len(a) < 1 or len(p) != 2:
        raise Error(name + ": addresses must hold [cursors_out, pieces...] and params [h, capacity]")
    var pieces = _nonnull(a, 1, name)
    _reserve(owner, name, False, ": session")
    var cursors = List[Int]()
    try:
        cursors = owner[].export_state(pieces)
    except error:
        _unreserve(owner, False, False)
        raise error
    _unreserve(owner, False, False)
    _write_words(a[0], cursors, p[1], name)
    return len(cursors)


def _load(owner: MutPointer[SambaStackProgram, MutUntrackedOrigin], a: List[Int], p: List[Int]) raises -> Int:
    # The former samba_decode_load_binding: a successful load makes a lost
    # session usable.
    var name = String("samba_decode_load")
    var pieces = _nonnull(a, 0, name)
    var cursors = _tail(p, 1)
    _reserve(owner, name, False, ": session")
    try:
        owner[].load_state(pieces, cursors)
        owner[].finish()
    except error:
        _unreserve(owner, True, False)
        raise error
    _unreserve(owner, False, True)
    return 0


def samba_program_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    comptime if not SAMBA_PROGRAM_ON:
        raise Error("samba programs: a host build has no device program")
    else:
        if op == SAMBA_PROGRAM_OPEN:
            return _open(a, p, f)
        if len(p) < 1:
            raise Error("samba programs: no program handle")
        var handle = samba_program_handle_of(p[0])
        if op == SAMBA_PROGRAM_RELEASE:
            SAMBA_PROGRAMS.release(handle)
            return 0
        var lease = SAMBA_PROGRAMS.acquire(handle)
        var owner = _owner(lease)
        var result = 0
        try:
            if op == SAMBA_PROGRAM_STATUS:
                result = 1 if owner[].usable else 0
            elif op == SAMBA_PROGRAM_FORWARD:
                result = _forward(owner, a, p)
            elif op == SAMBA_PROGRAM_DECODE_STEP:
                result = _decode_run(owner, a, p, False)
            elif op == SAMBA_PROGRAM_DECODE_TOKENS:
                result = _decode_run(owner, a, p, True)
            elif op == SAMBA_PROGRAM_DECODE_EXPORT:
                result = _export(owner, a, p)
            elif op == SAMBA_PROGRAM_DECODE_LOAD:
                result = _load(owner, a, p)
            else:
                raise Error("samba programs: unknown operation")
        finally:
            # The lease keeps the program alive through the call.
            _ = lease
        return result


def samba_programs_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                      result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if na < 0 or na > 4096 or np < 0 or np > 4096 or nf < 0 or nf > 64:
            raise Error("samba programs native ABI: invalid span")
        if result == 0:
            raise Error("samba programs native ABI: null result")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64](capacity=nf)
        if nf != 0:
            if ff == 0:
                raise Error("samba programs native ABI: null floating scalar span")
            var src = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
            for i in range(nf):  # driver-loop: admitted floating scalar span
                f.append(src[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = samba_program_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeSambaProgramsAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SAMBA_PROGRAMS_SCHEMA)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(samba_programs_v1)


comptime NATIVE_SAMBA_PROGRAMS_API = SharedSlot[NativeSambaProgramsAPI,
    ("MojolearnSambaProgramsAPIIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnSambaProgramsAPIFastV1")]
