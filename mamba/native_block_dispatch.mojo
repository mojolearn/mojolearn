# SPDX-License-Identifier: Apache-2.0
# Named block operations on the common native descriptor transport. All math,
# transfer, reports and retained stages remain in the existing Mamba owners.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, MAMBA_BLOCK_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from mamba.impl.modules.afn_defines import IDN_MAMBA3_REPORTS_ON_REQUEST
from mamba.impl.mamba_shell_impl import (
    Mamba3PrefillSession, _mamba1_backward_run, _mamba2_backward_run,
    _mamba3_backward_run, _mamba3_run, _m3_prefill_run,
    _m3_prefill_backward_run, _M3_BWD_NONFINITE_DY,
)

# 0/1 reserve/release under the provider GIL; 2/3/4 Mamba1/2/3 VJP;
# 5 fresh Mamba3 prefill; 6/7 session prefill/VJP; 8/9 owned prefill/VJP.
# Session execution prepends an admitted opaque owner handle to its words.
comptime MAMBA_BLOCK_OPERATIONS = 10


def mamba_block_operation(op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    if op < 0 or op >= MAMBA_BLOCK_OPERATIONS:
        raise Error("Mamba block native ABI: unknown operation")
    var p = List[Int]()
    var handle = 0
    if op < 2 or op >= 6:
        if len(words) == 0 or words[0] == 0:
            raise Error("Mamba block native ABI: null session")
        handle = words[0]
        var owner = MutPointer[Mamba3PrefillSession, MutUntrackedOrigin](unsafe_from_address=handle)
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
        for i in range(1, len(words)):  # small-loop(admitted native session scalar span)
            p.append(words[i])
    else:
        p = words.copy()
    if op >= 2 and op <= 4:
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
    if op == 5 or op == 6 or op == 8:
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
        if op == 5:
            return _mamba3_run[True](expanded, p[0], p[1], p[2], 0, 0)
        var owner = MutPointer[Mamba3PrefillSession, MutUntrackedOrigin](unsafe_from_address=handle)
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
    # Session backward keeps the finite-grad refusal recoverable exactly as
    # its old public entry; owned backward preserves its existing invalidation.
    var owned = op == 9
    if len(a) != (12 if owned else 21) or len(p) != (4 if owned else 3) or len(f) != 0:
        raise Error("mamba3 prefill backward: invalid native descriptor arity")
    var owner = MutPointer[Mamba3PrefillSession, MutUntrackedOrigin](unsafe_from_address=handle)
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


def mamba_block_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                   result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= MAMBA_BLOCK_OPERATIONS or na < 0 or na > 23 or np < 0 or np > 5 or nf < 0 or nf > 2:
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


comptime NATIVE_MAMBA_BLOCK_API = _Global[StorageType=NativeMambaBlockAPI,
    name="MojoMambaBlockNativeAPIv1", init_fn=NativeMambaBlockAPI.__init__]
