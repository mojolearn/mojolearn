# SPDX-License-Identifier: Apache-2.0
"""`_mojolearn_mamba`'s CausalLM session entry (lane causal-lm-sessions-native,
2026-11-11), family CAUSAL_SESSIONS_MAMBA of
`core/causal_lm_entry_table_sessions.mojo`: the decode program operations
only (a Mamba model has no resident Transformer sessions). The program lives
in this image's own registry, named per tier.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot
from core.causal_lm_entry_table_sessions import (
    CAUSAL_SESSIONS_SCHEMA, CAUSAL_SESSIONS_OPERATIONS, CAUSAL_SESSION_PROGRAM_READ, CAUSAL_SESSION_PROGRAM_RELEASE,
)
from transformer.causal_program_entry import causal_program_operation

comptime MAMBA_PROGRAMS_NAME = ("MojolearnCausalProgramsMambaIdenticalV1" if GLOBAL_NUMERIC_MODE == 1
    else "MojolearnCausalProgramsMambaFastV1")


def causal_sessions_mamba_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                             result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op != CAUSAL_SESSION_PROGRAM_READ and op != CAUSAL_SESSION_PROGRAM_RELEASE:
            raise Error("causal lm sessions: a Mamba image serves only the decode program")
        if na < 0 or na > 4096 or np < 0 or np > 4096 or nf < 0 or nf > 64:
            raise Error("causal lm sessions native ABI: invalid span")
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
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = causal_program_operation[MAMBA_PROGRAMS_NAME](op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeCausalSessionsMambaAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(CAUSAL_SESSIONS_SCHEMA)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(causal_sessions_mamba_v1)


comptime NATIVE_CAUSAL_SESSIONS_MAMBA_API = SharedSlot[NativeCausalSessionsMambaAPI,
    ("MojolearnCausalSessionsMambaAPIIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnCausalSessionsMambaAPIFastV1")]
