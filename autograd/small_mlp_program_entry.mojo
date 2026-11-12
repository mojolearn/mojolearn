# SPDX-License-Identifier: Apache-2.0
"""Span admission shared by the two `small_mlp_program` descriptor entries
(sequence/small_mlp_entry_device.mojo, sequence/small_mlp_entry_host.mojo),
published in core/neural_entry_table.mojo family NEURAL_TABLE_MLP_PROGRAM
(lane calltable-neural-blocks, 2026-11-11).

A call is the descriptor shape of core/native_descriptor_abi.mojo:
operation (a SMALL_MLP_PROGRAM_* code), the address span, the integer span
and the Float64 span, in the order bindings/small_mlp_native.mojo used for
the former Python entry `small_mlp_program(op, addresses, ints, floats)`.
OPEN reads five Float64 optimizer settings, converted to Float32 exactly as
the Python unpack converted them; every other operation takes no floats.
The caller owns every span until the synchronous call returns.
"""
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.neural_entry_table import NEURAL_MLP_PROGRAM_OPERATIONS


struct SmallMlpEntryCall(Movable):
    var a: List[Int]
    var p: List[Int]
    var f: List[Float32]

    def __init__(out self, op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int) raises:
        if op < 0 or op >= NEURAL_MLP_PROGRAM_OPERATIONS or result == 0:
            raise Error("small MLP program: unknown operation or null result")
        if nf < 0 or nf > 5 or (nf > 0 and ff == 0):
            raise Error("small MLP program: null or invalid float span")
        self.a = native_descriptor_words(aa, na)
        self.p = native_descriptor_words(pp, np)
        self.f = List[Float32]()
        var raw = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(at most five optimizer settings)
            self.f.append(Float32(raw[i]))


def small_mlp_entry_result(result: Int, value: Int):
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value


def small_mlp_entry_report(problem: Error, context: Int, sink: NativeDescriptorError):
    native_descriptor_report(problem, context, sink)
