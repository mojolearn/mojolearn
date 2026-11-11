# SPDX-License-Identifier: Apache-2.0
"""Synchronous descriptor calls with fixed, caller-local request envelopes."""
from std.sys import is_defined
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_call_output import NativeCallOutput
from core.native_descriptor_call import native_descriptor_call_spans


def native_descriptor_call_fixed[NA: Int, NP: Int, NF: Int](
    entry: NativeDescriptorCall, operation: Int,
    addresses: Array[Int, NA], ints: Array[Int, NP],
    floats: Array[Float64, NF],
) raises -> Int:
    # Request envelopes are borrowed inline storage, never retained by the
    # synchronous entry. Empty spans have no address. The output words live
    # on this frame, reached by reference through the out-of-line span call
    # (lane entry-table-zero-alloc; the former heap owner is the opt-in
    # MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT, see core/native_descriptor_call.mojo).
    var aa = 0
    var pp = 0
    var ff = 0
    comptime if NA > 0:
        aa = Int(addresses.unsafe_ptr())
    comptime if NP > 0:
        pp = Int(ints.unsafe_ptr())
    comptime if NF > 0:
        ff = Int(floats.unsafe_ptr())
    comptime if is_defined["MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT"]():
        var held = List[NativeCallOutput]()
        held.append(NativeCallOutput())
        try:
            return native_descriptor_call_spans(entry, operation,
                aa, NA, pp, NP, ff, NF, held[0])
        finally:
            _ = addresses
            _ = ints
            _ = floats
            _ = held
    var output = NativeCallOutput()
    try:
        # Result/error reset and propagation stay in the existing shared call.
        return native_descriptor_call_spans(entry, operation,
            aa, NA, pp, NP, ff, NF, output)
    finally:
        # Integer addresses carry no lifetime information. These uses also
        # cover refusal unwinding, nested entries and concurrent callers.
        _ = addresses
        _ = ints
        _ = floats
        _ = output
