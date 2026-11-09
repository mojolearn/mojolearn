# SPDX-License-Identifier: Apache-2.0
# Resident Transformer operations behind the common block provider. The typed
# numerical owner/functions and cache invalidation semantics are unchanged.
from transformer.block_options import BLOCK_OPTION_ADDRS, BLOCK_OPTION_PARAMS, BlockOptions
from transformer.transformer_binding_impl import (
    TransformerDecodeSession, _require_session_open, _check_base_addrs,
    _session_open_run, _session_open_run_int15, _session_export_run, _session_load_run,
)


def transformer_decode_operation(op: Int, a: List[Int], words: List[Int]) raises -> Int:
    if len(words) == 0 or words[0] == 0:
        raise Error("transformer decode: null session owner")
    var owner = MutPointer[TransformerDecodeSession, MutUntrackedOrigin](unsafe_from_address=words[0])
    var p = List[Int]()
    for i in range(1, len(words)):  # driver-loop: admitted native decode scalar span
        p.append(words[i])
    if op <= 2 or op == 9:
        if len(a) != 0 or len(p) != 0:
            raise Error("transformer decode lifecycle: invalid descriptor")
        if op == 9:
            owner[].close()
            return 0
        if op == 2:
            owner[].busy = False
            return 0
        if op == 0:
            if owner[].busy or owner[].is_open():
                raise Error("transformer decode session: busy or already open")
        else:
            _require_session_open(owner[])
        owner[].busy = True
        return 0
    if not owner[].busy:
        raise Error("transformer decode operation requires a reserved session")
    if op == 3 or op == 4:
        var packed = op == 4
        if (packed and (len(a) != 11 + BLOCK_OPTION_ADDRS + 21 or len(p) != 9 + BLOCK_OPTION_PARAMS)) or (not packed and ((len(a) != 11 and len(a) != 11 + BLOCK_OPTION_ADDRS) or (len(p) != 9 and len(p) != 9 + BLOCK_OPTION_PARAMS))):
            raise Error("transformer decode open: invalid address or scalar span")
        var expanded: List[Int] = [0]
        for i in range(11):  # small-loop(eleven original open address slots)
            expanded.append(a[i])
        expanded.append(0)
        for i in range(11, len(a)):  # driver-loop: admitted optional/packed weight address tail
            expanded.append(a[i])
        while len(expanded) < 13 + BLOCK_OPTION_ADDRS:  # small-loop(absent optional tail slots)
            expanded.append(0)
        # Separate assignments: a ternary over list literals of different lengths does not type-check.
        var base_slots: List[Int]
        if packed:
            base_slots = [1, 2, 10, 11]
        else:
            base_slots = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
        _check_base_addrs(expanded, base_slots, String("transformer_decode_session_open"))
        var opts = BlockOptions.from_params(p, 9)
        if p[0] <= 0:
            raise Error("transformer_decode_session_open: B must be positive")
        owner[].usable = False
        var result = 0
        try:
            if packed:
                result = _session_open_run_int15(owner[], expanded, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], opts)
            else:
                result = _session_open_run(owner[], expanded, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], opts)
        except problem:
            owner[].release()
            raise problem
        owner[].usable = True
        return result
    if op == 5 or op == 6:
        # Lane x4: the descriptor step (5) and forward (6) served only the
        # public decode session's provider arm, which now runs the attention
        # step op through its binding entries; the CausalLM layers use open,
        # export, load and close only.
        raise Error("transformer decode: the descriptor step and forward were removed; use the session entries")
    if len(a) != 2 or a[0] == 0 or a[1] == 0:
        raise Error("transformer decode: expected two nonnull buffer addresses")
    if op == 8 and len(p) != 1:
        raise Error("transformer decode: expected cached_tokens")
    if op == 7 and len(p) != 0:
        raise Error("transformer decode export: unexpected scalars")
    try:
        if op == 7:
            return _session_export_run(owner[], a[0], a[1])
        if op == 8:
            return _session_load_run(owner[], a[0], a[1], p[0])
        raise Error("transformer decode: unknown operation")
    except problem:
        owner[].usable = False
        raise problem
