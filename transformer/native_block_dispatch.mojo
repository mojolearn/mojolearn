# SPDX-License-Identifier: Apache-2.0
# Private sibling ABI for the existing fresh/backward Transformer operations.
# Common descriptor transport; no Python/provider imports or changed kernels.
from std.ffi import _Global
from transformer.native_lm_resident_dispatch import transformer_lm_resident_operation
from transformer.native_decode_dispatch import transformer_decode_operation
from transformer.native_tape_dispatch import transformer_tape_operation
from transformer.session_native_dispatch import transformer_session_forward_native
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, TRANSFORMER_BLOCK_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from transformer.block_options import BLOCK_OPTION_ADDRS, BLOCK_OPTION_PARAMS, BlockOptions
from transformer.transformer_binding_impl import (
    TransformerSession, _check_base_addrs, _transformer_run_session,
    _transformer_run_session_backward, _transformer_run, _transformer_backward_run,
)

# 0 reserve, 1 release (caller GIL/lock held); 2 session fresh, 3 session
# backward; 4 ephemeral fresh, 5 ephemeral backward. Session operations prepend
# Tape operations 6..10 reserve/release/forward/backward/close their own owner.
# one checked opaque owner handle to their integer vector. All spans borrowed.


def transformer_block_operation(op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    if op >= 25:
        return transformer_lm_resident_operation(op, a, words, f)
    if len(f) != 0:
        raise Error("transformer block: unexpected floating scalars")
    if op >= 15:
        return transformer_decode_operation(op - 15, a, words)
    if op == 14:
        return transformer_tape_operation(5, a, words)
    if op == 12 or op == 13:
        if len(words) != 1 or words[0] == 0 or len(a) != 0:
            raise Error("transformer session administration: invalid native descriptor")
        var owner = MutPointer[TransformerSession, MutUntrackedOrigin](unsafe_from_address=words[0])
        if owner[].busy or (op == 12 and owner[].closed):
            raise Error("transformer: session is busy or closed")
        if op == 12:
            owner[].invalidate_tape()
        else:
            owner[].closed = True
            owner[].clear()
            owner[].ctx = None
        return 0
    if op >= 6 and op < 11:
        return transformer_tape_operation(op - 6, a, words)
    var p = List[Int]()
    var session = 0
    if op < 4 or op == 11:
        if len(words) == 0 or words[0] == 0:
            raise Error("transformer block native ABI: null session")
        session = words[0]
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
        for i in range(1, len(words)):  # small-loop(checked session operation scalar span)
            p.append(words[i])
    else:
        p = words.copy()
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
        var owner = MutPointer[TransformerSession, MutUntrackedOrigin](unsafe_from_address=session)
        return transformer_session_forward_native(owner[], expanded, p, opts)
    var result = 0
    if op == 2 or op == 4:
        var base_a = 13 if op == 2 else 11
        var base_p = 10 if op == 2 else 8
        var name = "transformer_session_forward_fresh" if op == 2 else "transformer_forward_fresh"
        if len(a) != base_a and len(a) != base_a + BLOCK_OPTION_ADDRS:
            raise Error(name + ": addrs must contain " + String(base_a) + " addresses, or "
                + String(base_a) + " + " + String(BLOCK_OPTION_ADDRS)
                + " with the block options tail, got " + String(len(a)))
        var raw = a.copy()
        while len(raw) < base_a + BLOCK_OPTION_ADDRS:  # small-loop(pad existing optional block address tail)
            raw.append(0)
        var expanded = List[Int]()
        if op == 2:
            expanded = raw.copy()
            _check_base_addrs(expanded, [0, 1, 2, 3, 4, 5, 6, 8, 9, 12], name)
        if len(p) != base_p and len(p) != base_p + BLOCK_OPTION_PARAMS:
            raise Error(name + ": params must contain " + String(base_p) + " values, or "
                + String(base_p) + " + " + String(BLOCK_OPTION_PARAMS)
                + " with the block options tail, got " + String(len(p)))
        var opts = BlockOptions.from_params(p, base_p)
        if op == 2:
            var owner = MutPointer[TransformerSession, MutUntrackedOrigin](unsafe_from_address=session)
            try:
                result = _transformer_run_session[True](owner[], expanded,
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], opts)
            except problem:
                owner[].clear()
                raise problem
        else:
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
            result = _transformer_run[True](expanded,
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[1], 0, p[7], opts)
    elif op == 3 or op == 5:
        if len(a) != 21 or len(p) != 8:
            raise Error("transformer backward: expected 21 addresses and 8 scalars (B,"
                " L, d_model, n_heads, n_kv_heads, head_dim, intermediate, window)")
        for i in range(21):  # small-loop(original backward buffer slots)
            if a[i] == 0:
                raise Error("transformer backward: null buffer address at slot " + String(i))
        if op == 3:
            var owner = MutPointer[TransformerSession, MutUntrackedOrigin](unsafe_from_address=session)
            try:
                _transformer_run_session_backward(owner[], a,
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
            except problem:
                owner[].clear()
                raise problem
        else:
            if p[0] <= 0 or p[1] <= 0 or p[2] <= 0:
                raise Error("transformer backward: B, L and d_model must be positive")
            _transformer_backward_run(a, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    else:
        raise Error("transformer block native ABI: unknown operation")
    return result


def transformer_block_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                         result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= 35 or nf < 0 or nf > 1 or na < 0 or na > 45 or np < 0 or np > 28:
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


comptime NATIVE_TRANSFORMER_BLOCK_API = _Global[StorageType=NativeTransformerBlockAPI,
    name="MojoTransformerBlockNativeAPIv1", init_fn=NativeTransformerBlockAPI.__init__]
