# SPDX-License-Identifier: Apache-2.0
# Named block operations on the common native descriptor transport. All math,
# transfer, reports and retained stages remain in the existing Mamba owners.
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from mamba.native_decode_dispatch import mamba_decode_operation
from mamba.native_tape_dispatch import mamba_tape_operation
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, MAMBA_BLOCK_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from mamba.impl.modules.afn_defines import IDN_MAMBA3_REPORTS_ON_REQUEST
from mamba.checks.mamba3_fixture import Mamba3Dims
from mamba.impl.mamba_block_program import mamba_block_call_abi
from mamba.impl.mamba_shell_impl import (
    Mamba3PrefillSession, _mamba1_backward_run, _mamba2_backward_run,
    _mamba3_backward_run, _m3_prefill_run,
    _m3_prefill_backward_run, _M3_BWD_NONFINITE_DY, NN40_OWNED_WEIGHTS,
    _m3_install_weights_run, _m3_export_weights_run, _mamba3_run_owned,
)
from core.shared_slot import SharedSlot
from core.state_registry import StateRegistry
from core.neural_entry_table import (
    NEURAL_MAMBA_OPERATIONS, NEURAL_MAMBA3_SESSION_CREATE, NEURAL_MAMBA3_SESSION_INFO,
    NEURAL_MAMBA3_SESSION_CAPS, NEURAL_MAMBA3_SESSION_CLOSE,
    MAMBA3_CAP_SESSION_PREFILL, MAMBA3_CAP_REPORTS_OPTIONAL, MAMBA3_CAP_OWNED_WEIGHTS,
)
from core.mamba3_prefill_session import mamba3_prefill_session_word, mamba3_prefill_session_of

# 0/1 reserve/release under the provider GIL; 2/3/4 Mamba1/2/3 VJP;
# 5 fresh Mamba3 prefill; 6/7 session prefill/VJP; 8/9 owned prefill/VJP.
# 10..14 immutable tape reserve/release/forward/backward/close.
# 15 owned snapshot install; 16 owned carried-state forward/decode;
# 17 owned export; 18 session close; 19 tape reset; 20..43 decode;
# 44 session create, 45 session info, 46 capabilities.
# Session operations lead their words with the session's registry handle word
# (lane mamba3-prefill-handle, 2026-11-12: the session lives in
# PREFILL_SESSIONS below, never in a Python object;
# core/mamba3_prefill_session.mojo).
comptime MAMBA_BLOCK_OPERATIONS = NEURAL_MAMBA_OPERATIONS

# The image's prefill sessions, named per tier: an identical and a fast image
# in one process never share session storage.
comptime PREFILL_SESSIONS = StateRegistry[Mamba3PrefillSession,
    ("MojolearnMamba3PrefillSessionsIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnMamba3PrefillSessionsFastV1")]


def mamba3_prefill_caps() -> Int:
    """The image's fixed capability bits (NEURAL_MAMBA3_SESSION_CAPS): the
    build condition under which the image registered its former session
    Python entries (`mamba3_prefill_session_*`, `mamba3_owned_weights_enabled`)."""
    comptime if not (GLOBAL_NUMERIC_MODE <= NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL"]()):
        return 0
    var caps = MAMBA3_CAP_SESSION_PREFILL
    comptime if IDN_MAMBA3_REPORTS_ON_REQUEST:
        caps |= MAMBA3_CAP_REPORTS_OPTIONAL
    comptime if NN40_OWNED_WEIGHTS:
        caps |= MAMBA3_CAP_OWNED_WEIGHTS
    return caps


def _mamba3_session_close(word: Int) raises -> Int:
    var handle = mamba3_prefill_session_of(word)
    var lease = PREFILL_SESSIONS.acquire(handle)
    var closing = MutPointer[Mamba3PrefillSession, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=lease[])))
    if closing[].busy:
        _ = lease
        raise Error("mamba3 prefill session: busy")
    closing[].usable = False
    closing[].release()
    _ = lease
    PREFILL_SESSIONS.release(handle)
    return 0


def _mamba3_session_info(word: Int, a: List[Int]) raises -> Int:
    if len(a) != 1 or a[0] == 0:
        raise Error("mamba3 prefill session info: expected one output address")
    var lease = PREFILL_SESSIONS.acquire(mamba3_prefill_session_of(word))
    var owner = MutPointer[Mamba3PrefillSession, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=lease[])))
    var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[0])
    dst[0] = owner[].weight_uploads
    dst[1] = owner[].weight_recopies
    dst[2] = owner[].weight_reuses
    dst[3] = owner[].backward_reuses
    dst[4] = owner[].backward_recomputes
    dst[5] = 1 if owner[].stages_valid else 0
    _ = lease
    return 0


def _mamba3_session_operation(op: Int, session: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    # `session` is the address of a leased registry owner; the caller holds
    # the lease through this call.
    var owner = MutPointer[Mamba3PrefillSession, MutUntrackedOrigin](unsafe_from_address=session)
    if op == 0:
        if owner[].busy:
            raise Error("mamba3 prefill session: busy")
        if not owner[].usable:
            raise Error("mamba3 prefill session: lost after a failed call")
        owner[].busy = True
        return 0
    if op == 1:
        owner[].busy = False
        return 0
    if not owner[].busy or not owner[].usable:
        raise Error("Mamba block native ABI: execution requires a reserved usable session")
    var p = List[Int](capacity=len(words))
    for i in range(1, len(words)):  # small-loop(admitted native session scalar span)
        p.append(words[i])
    if op == 15:
        comptime if not NN40_OWNED_WEIGHTS:
            raise Error("mamba3 owned weights: NN40 arm is disabled")
        if len(a) != 9 or len(p) != 1 or len(f) != 0:
            raise Error("mamba3 owned weights: expected nine weight addresses and d_model")
        var dims = Mamba3Dims.of(p[0])
        var expanded: List[Int] = [0]
        for i in range(9):  # small-loop(nine admitted owned weight addresses)
            if a[i] == 0:
                raise Error("mamba3 owned weights: null weight address")
            expanded.append(a[i])
        # Original copy-on-install helper commits the generation only after a
        # complete upload. A failure preserves the previous usable snapshot.
        _m3_install_weights_run(owner[], expanded, dims, p[0])
        return owner[].weight_generation
    if op == 17:
        if len(a) != 9 or len(p) != 1 or len(f) != 0:
            raise Error("mamba3 owned export: expected nine writable buffers and generation")
        if not owner[].owned_weights or not owner[].w or p[0] != owner[].weight_generation:
            raise Error("mamba3 owned export: missing, unusable or stale generation")
        for i in range(9):  # small-loop(nine admitted owned export addresses)
            if a[i] == 0:
                raise Error("mamba3 owned export: null output address")
        var dm = owner[].dm
        _m3_export_weights_run(owner[], a, dm, Mamba3Dims.of(dm))
        return p[0]
    if op == 16:
        if len(a) != 25 or len(p) != 7 or len(f) != 0:
            raise Error("mamba3_state_owned: expected 25 addresses and seven scalars")
        if not owner[].owned_weights or p[5] != owner[].weight_generation:
            raise Error("mamba3 owned forward: stale or unavailable weight generation")
        if p[6] != 0 and p[6] != 1:
            raise Error("mamba3_state_owned: step must be 0 or 1")
        if p[6] == 1 and p[1] != 1:
            raise Error("mamba3_state_owned: a decode step is one token (L = 1)")
        # The existing owner validates counters and dimensions and preserves
        # partial-state/error behavior. No second implementation of recurrence.
        return _mamba3_run_owned(owner[], a, p[0], p[1], p[2], p[3], p[4])
    if op == 6 or op == 8:
        var owned = op == 8
        if len(a) != (6 if owned else 15) or len(p) != (4 if owned else 3) or len(f) != 0:
            raise Error("mamba3 prefill: invalid native descriptor arity")
        if p[0] <= 0 or p[1] <= 0:
            raise Error("mamba3 prefill: B and L must be positive")
        var expanded = List[Int](length=25, fill=0)
        if owned:
            expanded[0] = a[0]
            for i in range(5):  # small-loop(output and four report addresses)
                expanded[20 + i] = a[1 + i]
        else:
            for i in range(10):  # small-loop(input and nine weight addresses)
                expanded[i] = a[i]
            for i in range(5):  # small-loop(output and four report addresses)
                expanded[20 + i] = a[10 + i]
        if owned:
            if not owner[].owned_weights or p[3] != owner[].weight_generation:
                raise Error("mamba3 owned forward: stale or unavailable weight generation")
            if p[2] != owner[].dm:
                raise Error("mamba3 owned forward: invalid shape or configuration")
        for i in range(10):  # small-loop(input and optional caller-owned weights)
            if (i == 0 or not owned) and expanded[i] == 0:
                raise Error("mamba3 prefill: null input/weight address")
        if expanded[20] == 0:
            raise Error("mamba3 prefill: null output address")
        comptime if not IDN_MAMBA3_REPORTS_ON_REQUEST:
            for i in range(4):  # small-loop(four required public reports)
                if expanded[21 + i] == 0:
                    raise Error("mamba3 prefill: report address required")
        try:
            return _m3_prefill_run(owner[], expanded, p[0], p[1], p[2], owned)
        except problem:
            owner[].usable = False
            owner[].release()
            raise problem
    if op != 7 and op != 9:
        raise Error("Mamba block native ABI: unknown session operation")
    # Session backward keeps the finite-grad refusal recoverable exactly as
    # its old public entry; owned backward preserves its existing invalidation.
    var owned = op == 9
    if len(a) != (12 if owned else 21) or len(p) != (4 if owned else 3) or len(f) != 0:
        raise Error("mamba3 prefill backward: invalid native descriptor arity")
    if owned and (not owner[].owned_weights or p[3] != owner[].weight_generation):
        raise Error("mamba3 owned backward: stale or unavailable weight generation")
    if p[0] <= 0 or p[1] <= 0 or p[2] <= 0 or (owned and p[2] != owner[].dm):
        raise Error("mamba3 prefill backward: invalid shape or configuration")
    var expanded = List[Int](length=21, fill=0)
    if owned:
        expanded[0] = a[0]
        for i in range(1, 12):  # small-loop(gradient input/output addresses)
            expanded[9 + i] = a[i]
    else:
        expanded = a.copy()
    for i in range(len(a)):  # small-loop(12 or 21 native VJP addresses)
        if a[i] == 0:
            raise Error("mamba3 prefill backward: null buffer address at slot " + String(i))
    try:
        _m3_prefill_backward_run(owner[], expanded, p[0], p[1], p[2], owned)
    except problem:
        if not owned and String(problem).startswith(_M3_BWD_NONFINITE_DY):
            raise problem
        owner[].usable = False
        owner[].release()
        raise problem
    return 0


def mamba_block_operation(op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    if op < 0 or op >= MAMBA_BLOCK_OPERATIONS:
        raise Error("Mamba block native ABI: unknown operation")
    if op == NEURAL_MAMBA3_SESSION_CREATE:
        if len(a) != 0 or len(words) != 0 or len(f) != 0:
            raise Error("mamba3 prefill session: create takes no arguments")
        # A fresh session: creation performs no GPU operation.
        return mamba3_prefill_session_word(PREFILL_SESSIONS.insert(Mamba3PrefillSession()))
    if op == NEURAL_MAMBA3_SESSION_CAPS:
        if len(a) != 0 or len(words) != 0 or len(f) != 0:
            raise Error("mamba3 prefill session: capabilities take no arguments")
        return mamba3_prefill_caps()
    if op == NEURAL_MAMBA3_SESSION_INFO:
        if len(words) != 1 or len(f) != 0:
            raise Error("mamba3 prefill session info: invalid native descriptor")
        return _mamba3_session_info(words[0], a)
    if op >= 20:
        return mamba_decode_operation(1 + (op - 20) // 8, (op - 20) % 8, a, words, f)
    if op == 19:
        if len(f) != 0:
            raise Error("mamba3 private tape reset: unexpected floats")
        return mamba_tape_operation(5, a, words)
    if op == NEURAL_MAMBA3_SESSION_CLOSE:
        if len(words) != 1 or len(a) != 0 or len(f) != 0:
            raise Error("mamba3 prefill session close: invalid native descriptor")
        return _mamba3_session_close(words[0])
    if op >= 10 and op < 15:
        if len(f) != 0:
            raise Error("mamba3 forward tape: unexpected float scalars")
        return mamba_tape_operation(op - 10, a, words)
    if op < 2 or op >= 6:
        if len(words) == 0:
            raise Error("Mamba block native ABI: null session")
        # The lease keeps the owner alive through the call even if another
        # thread closes the handle meanwhile (a close of a busy session
        # refuses).
        var lease = PREFILL_SESSIONS.acquire(mamba3_prefill_session_of(words[0]))
        var result = 0
        try:
            result = _mamba3_session_operation(op, Int(UnsafePointer(to=lease[])), a, words, f)
        except problem:
            _ = lease
            raise problem
        _ = lease
        return result
    if op >= 2 and op <= 4:
        var p = words.copy()
        var family = op - 1
        var count = 23 if family == 1 else 21
        if len(a) != count or len(p) != 3 or len(f) != (2 if family == 2 else 0):
            raise Error("mamba" + String(family) + " backward: invalid native descriptor arity")
        if p[0] <= 0 or p[1] <= 0 or p[2] <= 0:
            raise Error("mamba" + String(family) + " backward: B, L and d_model must be positive")
        for i in range(count):  # small-loop(21 or 23 VJP buffer addresses)
            if a[i] == 0:
                raise Error("mamba" + String(family) + " backward: null buffer address at slot " + String(i))
        if family == 1:
            _mamba1_backward_run(a, p[0], p[1], p[2])
        elif family == 2:
            _mamba2_backward_run(a, p[0], p[1], p[2], Float32(f[0]), Float32(f[1]))
        else:
            _mamba3_backward_run(a, p[0], p[1], p[2])
        return 0
    # op 5: the fresh (session-free) Mamba3 prefill.
    if len(a) != 15 or len(words) != 3 or len(f) != 0:
        raise Error("mamba3 prefill: invalid native descriptor arity")
    if words[0] <= 0 or words[1] <= 0:
        raise Error("mamba3 prefill: B and L must be positive")
    var expanded = List[Int](length=25, fill=0)
    for i in range(10):  # small-loop(input and nine weight addresses)
        expanded[i] = a[i]
    for i in range(5):  # small-loop(output and four report addresses)
        expanded[20 + i] = a[10 + i]
    return mamba_block_call_abi(3, expanded, words[0], words[1], words[2], 0, 0, Float32(0), Float32(0), True)


def mamba_block_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                   result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= MAMBA_BLOCK_OPERATIONS or na < 0 or na > 25 or np < 0 or np > 8 or nf < 0 or nf > 2:
            raise Error("Mamba block native ABI: invalid operation or descriptor span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        if nf != 0 and ff == 0:
            raise Error("Mamba block native ABI: null float span")
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(zero or two admitted float scalars)
            f.append(fp[i])
        if result == 0:
            raise Error("Mamba block native ABI: null result")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = mamba_block_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeMambaBlockAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MAMBA_BLOCK_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(mamba_block_v1)


comptime NATIVE_MAMBA_BLOCK_API = SharedSlot[NativeMambaBlockAPI, ("MojolearnMambaBlockNativeAPIIdenticalV3" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnMambaBlockNativeAPIFastV3")]
