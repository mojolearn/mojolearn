# SPDX-License-Identifier: Apache-2.0
# CPU training siblings use the common provider ABI and their original host
# numerical owners. No GPU implementation or Python dispatch enters this file.
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, MAMBA_BLOCK_ABI_NAME, MAMBA_STATE_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from mamba.host.mamba_host_shell_impl import (
    _mamba1_run, _mamba2_run, _mamba1_backward_run, _mamba2_backward_run, _mamba3_backward_run, _mamba3_run,
)
from core.shared_slot import SharedSlot
from core.neural_entry_table import NEURAL_MAMBA_OPERATIONS, NEURAL_MAMBA3_SESSION_CAPS

# TOMBSTONE: MOJOLEARN_NN40_OWNED_WEIGHTS (leftover: the CPU training image's owned-snapshot prefill sessions, HOST_PREFILL_SESSIONS with ops 0/1 reserve/release, 8/9, 15/16/17, 18 close, 44 create and 45 info; the host image has no prefill session, Mamba3StateRoute.has_sessions is False for it) deleted 2026-11-25 by lane/owned-weights-deletion; code recoverable at 097a876282.


def mamba3_host_prefill_caps() -> Int:
    """The host image's capability bits: no prefill session, every report
    materialized. Callers read them on the block's training entry whatever
    its column (bindings/neural_block_public_door.mojo backward,
    providers/python/samba_execution_pyapi_door.mojo)."""
    return 0


def mamba_host_block_operation(op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    if op == NEURAL_MAMBA3_SESSION_CAPS:
        if len(a) != 0 or len(words) != 0 or len(f) != 0:
            raise Error("mamba host provider: capabilities take no arguments")
        return mamba3_host_prefill_caps()
    if op == 2 or op == 3 or op == 4:
        if len(a) != (23 if op == 2 else 21) or len(words) != 3 or len(f) != (2 if op == 3 else 0):
            raise Error("Mamba host backward: invalid descriptor arity")
        for address in a:  # driver-loop: admitted fixed gradient address vector
            if address == 0:
                raise Error("Mamba host backward: null buffer address")
        if words[0] <= 0 or words[1] <= 0 or words[2] <= 0:
            raise Error("Mamba host backward: B, L and d_model must be positive")
        if op == 2:
            _mamba1_backward_run(a, words[0], words[1], words[2])
        elif op == 3:
            _mamba2_backward_run(a, words[0], words[1], words[2], Float32(f[0]), Float32(f[1]))
        else:
            _mamba3_backward_run(a, words[0], words[1], words[2])
        return 0
    if op == 5:
        if len(a) != 15 or len(words) != 3 or len(f) != 0:
            raise Error("mamba3_forward_fresh: expected 15 addresses and B, L, d_model")
        if words[0] < 1 or words[1] < 1:
            raise Error("mamba3_forward_fresh: B and L must be positive")
        var expanded = List[Int](length=25, fill=0)
        for i in range(10):  # small-loop(input plus nine weights)
            expanded[i] = a[i]
        for i in range(10, 15):  # small-loop(output plus four reports)
            expanded[i + 10] = a[i]
        return _mamba3_run(expanded, words[0], words[1], words[2], 0, 0, True)
    raise Error("Mamba host provider: unavailable operation")


def mamba_host_block_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                          result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= NEURAL_MAMBA_OPERATIONS or na < 0 or na > 25 or np < 0 or np > 28 or nf < 0 or nf > 2 or result == 0:
            raise Error("mamba host provider: invalid descriptor span or operation")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        if nf != 0 and ff == 0:
            raise Error("mamba host provider: null float span")
        var raw = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var f = List[Float64]()
        for i in range(nf):  # small-loop(at most two admitted floating scalars)
            f.append(raw[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = mamba_host_block_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct MambaHostBlockAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MAMBA_BLOCK_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(mamba_host_block_v1)


comptime NATIVE_MAMBA_HOST_BLOCK_API = SharedSlot[MambaHostBlockAPI, ("MojolearnMambaHostBlockAPIIdenticalV3" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnMambaHostBlockAPIFastV3")]


def mamba_host_state_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    var family = op // 2
    var step = (op & 1) != 0
    var counts: List[Int] = [14, 16, 25]
    if len(a) != counts[family] or len(p) != (2 if step else 3) + family or len(f) != (2 if family == 1 else 0):
        raise Error("Mamba host state: invalid descriptor arity")
    for address in a:  # driver-loop: original host buffer address validation
        if address == 0:
            raise Error("Mamba host state: null buffer address")
    var length = 1 if step else p[1]
    var width = p[1] if step else p[2]
    var tail = 2 if step else 3
    if family == 0:
        _mamba1_run(a, p[0], length, width, 0)
        return 0
    if family == 1:
        return _mamba2_run(a, p[0], length, width, p[tail], Float32(f[0]), Float32(f[1]))
    return _mamba3_run(a, p[0], length, width, p[tail], p[tail + 1], False)


def mamba_host_state_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                          result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= 6 or na < 0 or na > 25 or np < 0 or np > 28 or nf < 0 or nf > 2 or result == 0:
            raise Error("mamba host provider: invalid descriptor span or operation")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        if nf != 0 and ff == 0:
            raise Error("mamba host provider: null float span")
        var raw = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var f = List[Float64]()
        for i in range(nf):  # small-loop(at most two admitted floating scalars)
            f.append(raw[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = mamba_host_state_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct MambaHostStateAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MAMBA_STATE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(mamba_host_state_v1)


comptime NATIVE_MAMBA_HOST_STATE_API = SharedSlot[MambaHostStateAPI, ("MojolearnMambaHostStateAPIIdenticalV3" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnMambaHostStateAPIFastV3")]
