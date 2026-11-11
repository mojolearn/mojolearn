# SPDX-License-Identifier: Apache-2.0
"""The CausalLM decode program behind a native registry handle (lane
causal-lm-sessions-native, 2026-11-11). Shared by the transformer and Mamba
images' session entries; each image names its own registry, so a program is
always created, read and destroyed by one image's code.

The read is `causal_lm_program_read_v1`'s body with plain words in place of
the Python owner and lists: reuse the held program when it serves these
weights (`causal_program_matches`), else open a new one; run one read; write
the result words to the caller's output span. The numerical program is
untouched.
"""
from core.state_registry import StateRegistry
from core.causal_lm_entry_table_sessions import (
    causal_session_handle_word, causal_session_handle_of,
    CAUSAL_SESSION_PROGRAM_READ, CAUSAL_SESSION_PROGRAM_RELEASE,
)
from transformer.causal_lm_program import (
    CausalLMProgram, causal_program_open, causal_program_matches, causal_program_read,
)


def causal_program_operation[name: StaticString](op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    """PROGRAM_READ: words [held handle or 0, output address, output
    capacity, p...]. Writes [count, result words...] to the output and
    returns the handle that served the read (the held one when reused, else a
    new one the caller now owns; the caller releases the held one when it is
    replaced). PROGRAM_RELEASE: words [handle]."""
    comptime PROGRAMS = StateRegistry[CausalLMProgram, name]
    if op == CAUSAL_SESSION_PROGRAM_RELEASE:
        if len(words) != 1 or len(a) != 0 or len(f) != 0:
            raise Error("causal lm program: invalid release descriptor")
        PROGRAMS.release(causal_session_handle_of(words[0]))
        return 0
    if op != CAUSAL_SESSION_PROGRAM_READ:
        raise Error("causal lm program: unknown operation")
    if len(words) < 3 or words[1] == 0 or words[2] < 5:
        raise Error("causal lm program: invalid read descriptor")
    var held = words[0]
    var output = words[1]
    var capacity = words[2]
    var p = List[Int](capacity=len(words) - 3)
    for i in range(3, len(words)):  # driver-loop: over descriptor words (argument unpacking, no data)
        p.append(words[i])
    var fi = List[Float32](capacity=len(f))
    for i in range(len(f)):  # driver-loop: over descriptor floats (argument unpacking, no data)
        fi.append(Float32(f[i]))
    var reuse = False
    if held != 0:
        var prior = PROGRAMS.acquire(causal_session_handle_of(held))
        reuse = causal_program_matches(prior[], a, p, fi)
        _ = prior
    var handle = held
    if not reuse:
        handle = causal_session_handle_word(PROGRAMS.insert(causal_program_open(a, p, fi)))
    var ret = List[Int]()
    try:
        var lease = PROGRAMS.acquire(causal_session_handle_of(handle))
        ret = causal_program_read(lease[], a, p, fi)
        _ = lease
        if len(ret) + 1 > capacity:
            raise Error("causal lm program: result exceeds the output span")
    except problem:
        if not reuse:
            # A program opened for this read dies with it, as the former
            # Python owner did when the caller discarded it.
            PROGRAMS.release(causal_session_handle_of(handle))
        raise problem
    var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=output)
    dst[0] = len(ret)
    for i in range(len(ret)):  # driver-loop: over result words (no data)
        dst[i + 1] = ret[i]
    return handle
