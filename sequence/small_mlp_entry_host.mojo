# SPDX-License-Identifier: Apache-2.0
"""The `small_mlp_program` descriptor entry of `_mojolearn_x_sequence_host` and its publication in
the neural-blocks entry table (core/neural_entry_table.mojo family
NEURAL_TABLE_MLP_PROGRAM; lane calltable-neural-blocks, 2026-11-11).

The same calls the Python entry `small_mlp_program` makes
(providers/python/small_mlp_program_pyapi.mojo), on plain spans: OPEN and
SESSIONS on a fresh executor, every other operation on the open program the
integer span's handle names. No arithmetic here; publication is metadata
with no independent work to split.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.neural_entry_table import (
    NeuralEntryKey, neural_table_publish, NEURAL_TABLE_MLP_PROGRAM,
)
from core.shared_slot import SharedSlot
from sequence.exec import HostExec
from autograd.ops_api import OPS_HOST
from autograd.small_mlp_program import (
    SMALL_MLP_PROGRAM_OPEN, SMALL_MLP_PROGRAM_SESSIONS, small_mlp_program_open, small_mlp_program_run,
    small_mlp_sessions_forward,
)
from autograd.small_mlp_program_entry import SmallMlpEntryCall, small_mlp_entry_result, small_mlp_entry_report


def small_mlp_program_host_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                               result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var call = SmallMlpEntryCall(op, aa, na, pp, np, ff, nf, result)
        var value: Int
        if op == SMALL_MLP_PROGRAM_OPEN:
            value = small_mlp_program_open[HostExec, OPS_HOST](HostExec(), call.a, call.p, call.f)
        elif op == SMALL_MLP_PROGRAM_SESSIONS:
            value = small_mlp_sessions_forward[HostExec, OPS_HOST](HostExec(), call.a, call.p)
        else:
            value = small_mlp_program_run[HostExec, OPS_HOST](op, call.a, call.p, List[Float32]())
        small_mlp_entry_result(result, value)
        return 0
    except problem:
        small_mlp_entry_report(problem, context, sink)
        return 1


struct SmallMlpProgramHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(small_mlp_program_host_v1)


comptime NATIVE_HOST_SMALL_MLP_PROGRAM_API = SharedSlot[SmallMlpProgramHostAPI, ("MojolearnSmallMlpProgramHostAPIIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnSmallMlpProgramHostAPIFastV1")]


def publish_neural_small_mlp_host() raises:
    """Module init of `_mojolearn_x_sequence_host`: publish under (tier, cpu)."""
    var api = NATIVE_HOST_SMALL_MLP_PROGRAM_API.get_or_create_ptr()
    neural_table_publish[NEURAL_TABLE_MLP_PROGRAM](NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), Int(api), api[].entries[0])
