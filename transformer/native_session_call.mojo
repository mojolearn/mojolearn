# SPDX-License-Identifier: Apache-2.0
# Native execution boundary. No provider, Python object, GIL, capsule, Python
# helper, or framework import belongs here; tools/test_generation_native_abi.py
# checks this module's local import closure. Providers retain external owners
# and synchronize producers before handing over these borrowed native vectors.
from std.sys import is_defined
from core.native_string import unsafe_string_from_utf8_cstr
from transformer.session_abi import GenerationSessionForward
from core.native_call_output import NativeCallOutput


def _generation_native_error(context: Int, address: Int, size: Int) abi("C") -> NoneType:
    var out = MutPointer[List[UInt8], MutUntrackedOrigin](unsafe_from_address=context)
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(size):  # driver-loop: copy complete native error text
        out[].append(src[i])
    return None


@no_inline
def _generation_forward_into(
    phase: Int, forward: GenerationSessionForward, handle: Int, addresses: List[Int], params: List[Int],
    mut output: NativeCallOutput,
) raises -> Int:
    # Out of line, so `output` is real storage of the caller's frame for the
    # whole foreign call (the shape of core/native_descriptor_call.mojo).
    var error_context = Int(MutPointer(to=output.error))
    var status = forward(phase, handle, Int(addresses.unsafe_ptr()), len(addresses),
        Int(params.unsafe_ptr()), len(params), Int(MutPointer(to=output.result)), error_context, _generation_native_error)
    _ = addresses
    _ = params
    if status != 0:
        output.error.append(0)
        raise Error(unsafe_string_from_utf8_cstr(output.error.unsafe_ptr()))
    return output.result


def generation_session_forward(
    phase: Int, forward: GenerationSessionForward, handle: Int, addresses: List[Int], params: List[Int],
) raises -> Int:
    # round2-fixes-2 (perf audit 3 left open): the output words live on this
    # frame and reach the foreign call through the out-of-line helper's `mut`
    # reference, so no heap List per generation forward, as the entry tables
    # do (lane entry-table-zero-alloc). The former one-element heap List
    # (Mojo 1.0.0/Apple: the inline Array result stayed zero after the ABI
    # call and repeated public forwards lost their cache position; the bounded
    # native and public regressions retain that evidence) stays behind the
    # same opt-in define, MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT, until the stack
    # form runs on Apple (the one Apple run owed for the entry tables).
    comptime if is_defined["MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT"]():
        var held = List[NativeCallOutput]()
        held.append(NativeCallOutput())
        var held_result = _generation_forward_into(phase, forward, handle, addresses, params, held[0])
        _ = held
        return held_result
    else:
        var output = NativeCallOutput()
        var result = _generation_forward_into(phase, forward, handle, addresses, params, output)
        _ = output
        return result
