# SPDX-License-Identifier: Apache-2.0
# CPU training siblings use the common provider ABI and their original host
# numerical owners. No GPU implementation or Python dispatch enters this file.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, MAMBA_BLOCK_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from mamba.host.mamba_host_shell_impl import (
    _mamba1_backward_run, _mamba2_backward_run, _mamba3_backward_run, _mamba3_run,
    M3OwnedHostSession, NN40_OWNED_WEIGHTS, _m3_owned_install_run,
    _m3_owned_forward_run, _m3_owned_backward_run, _m3_owned_state_run,
    _m3_owned_export_run, _nn40_host_require,
)


def mamba_host_block_operation(op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    if op == 0 or op == 1 or op == 18:
        if len(words) != 1 or words[0] == 0 or len(a) != 0 or len(f) != 0:
            raise Error("mamba3 owned host: invalid lifecycle descriptor")
        var owner = MutPointer[M3OwnedHostSession, MutUntrackedOrigin](unsafe_from_address=words[0])
        if op == 1:
            owner[].busy = False
        elif op == 18:
            if owner[].busy:
                raise Error("mamba3 owned host: session busy")
            owner[].weights = None
            owner[].usable = False
        else:
            if owner[].busy or not owner[].usable:
                raise Error("mamba3 owned host: session busy or unusable")
            owner[].busy = True
        return 0
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
    if op != 8 and op != 9 and op != 15 and op != 16 and op != 17:
        raise Error("Mamba host provider: unavailable operation")
    if len(words) < 2 or words[0] == 0 or len(f) != 0:
        raise Error("mamba3 owned host: invalid execution descriptor")
    var owner = MutPointer[M3OwnedHostSession, MutUntrackedOrigin](unsafe_from_address=words[0])
    if not owner[].busy or not owner[].usable:
        raise Error("mamba3 owned host: execution requires a reserved usable owner")
    var p = List[Int]()
    for i in range(1, len(words)):  # small-loop(admitted operation scalar vector)
        p.append(words[i])
    var want_a = 9 if op == 15 or op == 17 else (25 if op == 16 else (6 if op == 8 else 12))
    var want_p = 1 if op == 15 or op == 17 else (7 if op == 16 else 4)
    if len(a) != want_a or len(p) != want_p:
        raise Error("mamba3 owned host: invalid operation arity")
    for address in a:  # driver-loop: preserve the original non-null buffer admission
        if address == 0:
            raise Error("mamba3 owned host: null buffer address")
    if op == 15:
        comptime if not NN40_OWNED_WEIGHTS:
            raise Error("mamba3 owned host: NN40 arm disabled")
        var weights: List[Int] = [0]
        weights.extend(a.copy())
        _m3_owned_install_run(owner[], weights, p[0])
        return owner[].generation
    if op == 17:
        if not owner[].weights:
            raise Error("mamba3 owned export: no snapshot")
        var dims = owner[].weights.value().dims.copy()
        _nn40_host_require(owner[], dims.d_model, p[0], True)
        _m3_owned_export_run(owner[], a, dims.d_model, dims)
        return p[0]
    _nn40_host_require(owner[], p[2], p[5] if op == 16 else p[3], True)
    if op == 16:
        if p[6] != 0 and p[6] != 1:
            raise Error("mamba3_state_owned: step must be 0 or 1")
        if p[6] == 1 and p[1] != 1:
            raise Error("mamba3_state_owned: a decode step is one token (L = 1)")
        var result = _m3_owned_state_run(owner[], a, p[0], p[1], p[2], p[3], p[4])
        owner[].weight_reuses += 1
        return result
    if p[0] <= 0 or p[1] <= 0:
        raise Error("mamba3 owned host: positive B and L required")
    try:
        if op == 8:
            _m3_owned_forward_run(owner[], a, p[0], p[1], p[2])
        else:
            _m3_owned_backward_run(owner[], a, p[0], p[1], p[2])
    except problem:
        owner[].usable = False
        raise problem
    owner[].weight_reuses += 1
    if op == 9:
        owner[].backward_recomputes += 1
    return 0


def mamba_host_block_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                          result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= 44 or na < 0 or na > 25 or np < 0 or np > 28 or nf < 0 or nf > 2 or result == 0:
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


comptime NATIVE_MAMBA_HOST_BLOCK_API = _Global[StorageType=MambaHostBlockAPI,
    name="MojoMambaHostBlockAPIv1", init_fn=MambaHostBlockAPI.__init__]
