# SPDX-License-Identifier: Apache-2.0
"""Typed generation steps behind the CausalLM shell (bindings/causal_lm_public.mojo).

Every loop over batch rows, tokens or bytes lives here, over native pointers
and lists. The shell only unpacks Python objects and calls these functions.
"""
from std.memory import memcpy
from bindings.generation_policy import generation_pick


def causal_lm_rows_done(finished: List[Bool]) -> Bool:
    for i in range(len(finished)):
        if not finished[i]:
            return False
    return True


def causal_lm_join_resident(
    dst: MutPointer[Int32, MutUntrackedOrigin], prompt: MutPointer[Int32, MutUntrackedOrigin],
    generated: MutPointer[Int32, MutUntrackedOrigin], b: Int, l: Int, n: Int,
):
    for r in range(b):
        memcpy(dest=dst + r * (l + n), src=prompt + r * l, count=l)
        memcpy(dest=dst + r * (l + n) + l, src=generated + r * n, count=n)


def causal_lm_join_continuation(
    dest: MutPointer[Int32, MutUntrackedOrigin], prompt: MutPointer[Int32, MutUntrackedOrigin],
    tail: MutPointer[Int32, MutUntrackedOrigin], b: Int, l: Int, n: Int,
):
    for r in range(b):
        memcpy(dest=dest + r * (l + n), src=prompt + r * l, count=l)
        if n > 0:
            memcpy(dest=dest + r * (l + n) + l, src=tail + r * n, count=n)


def causal_lm_pick_step(
    raw: MutPointer[Float32, MutUntrackedOrigin], dst: MutPointer[Int32, MutUntrackedOrigin],
    b: Int, l: Int, v: Int, mut finished: List[Bool], mut lengths: List[Int], eos: List[Int],
    pad: Int, greedy: Bool, temperature: Float32, top_k: Int, top_p: Float64, seed: UInt64,
    produced: Int, mut tokens: List[Int32],
) raises:
    for r in range(b):
        var pick = pad
        if not finished[r]:
            var row = List[Float32](capacity=v)
            for j in range(v):
                row.append(raw[(r * l + l - 1) * v + j])
            pick = generation_pick(row, greedy, temperature, top_k, top_p, seed, r, produced)
            lengths[r] += 1
            for i in range(len(eos)):
                if pick == eos[i]:
                    finished[r] = True
        dst[r] = Int32(pick)
        tokens.append(Int32(pick))


def causal_lm_tokens_row_major(
    tokens: List[Int32], dst: MutPointer[Int32, MutUntrackedOrigin], b: Int, wrote: Int,
):
    for r in range(b):
        for t in range(wrote):
            dst[r * wrote + t] = tokens[t * b + r]


def causal_lm_drop_eos(raw: MutPointer[Int32, MutUntrackedOrigin], n: Int, eos: List[Int]) -> List[Int]:
    var values = List[Int]()
    for i in range(n):
        var special = False
        for j in range(len(eos)):
            special = special or Int(raw[i]) == eos[j]
        if not special:
            values.append(Int(raw[i]))
    return values^


def causal_lm_utf8_cut(pending: MutPointer[UInt8, MutUntrackedOrigin], size: Int, done: Bool) -> Int:
    var cut = size
    # Preserve an incomplete trailing UTF-8 sequence across tokens. Invalid
    # complete sequences are handled by the normal CPython UTF-8 decoder.
    if not done and size > 0:
        var lead = size - 1
        while lead > max(-1, size - 4) and (Int(pending[lead]) & 0xC0) == 0x80:
            lead -= 1
        if lead >= 0:
            var first = Int(pending[lead])
            var need = 1
            if first >= 0xC2 and first <= 0xDF:
                need = 2
            elif first >= 0xE0 and first <= 0xEF:
                need = 3
            elif first >= 0xF0 and first <= 0xF4:
                need = 4
            if size - lead < need:
                cut = lead
    return cut


def causal_lm_store_ids(
    ptr: MutPointer[Int32, MutUntrackedOrigin], ids: MutPointer[Int64, MutUntrackedOrigin], n: Int,
):
    for i in range(n):
        ptr[i] = Int32(Int(ids[i]))


def causal_lm_flags_as_ints(finished: List[Bool]) -> List[Int]:
    var out = List[Int](capacity=len(finished))
    for i in range(len(finished)):
        out.append(1 if finished[i] else 0)
    return out^


def causal_lm_restore_rows(
    lengths_src: MutPointer[Int64, MutUntrackedOrigin], finished_src: MutPointer[Int8, MutUntrackedOrigin],
    b: Int, produced: Int, mut lengths: List[Int], mut finished: List[Bool],
) raises:
    for i in range(b):
        lengths[i] = Int(lengths_src[i])
        finished[i] = finished_src[i] != 0
        if lengths[i] < 0 or lengths[i] > produced:
            raise Error("generation: invalid checkpoint row length")
