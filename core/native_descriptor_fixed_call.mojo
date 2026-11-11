# SPDX-License-Identifier: Apache-2.0
"""Synchronous descriptor calls with fixed, caller-local request envelopes."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_call_output import NativeCallOutput
from core.native_descriptor_call import native_descriptor_call_spans


def native_descriptor_call_fixed[NA: Int, NP: Int, NF: Int](
    entry: NativeDescriptorCall, operation: Int,
    addresses: Array[Int, NA], ints: Array[Int, NP],
    floats: Array[Float64, NF],
) raises -> Int:
    # Request envelopes are borrowed inline storage, never retained by the
    # synchronous entry. Empty spans have no address. Keep the proven heap
    # output owner: integer-address writable outputs require it on our ABI.
    var aa = 0
    var pp = 0
    var ff = 0
    comptime if NA > 0:
        aa = Int(addresses.unsafe_ptr())
    comptime if NP > 0:
        pp = Int(ints.unsafe_ptr())
    comptime if NF > 0:
        ff = Int(floats.unsafe_ptr())
    var output = List[NativeCallOutput]()
    output.append(NativeCallOutput())
    try:
        # Result/error reset and propagation stay in the existing shared call.
        return native_descriptor_call_spans(entry, operation,
            aa, NA, pp, NP, ff, NF, output[0])
    finally:
        # Integer addresses carry no lifetime information. These uses also
        # cover refusal unwinding, nested entries and concurrent callers.
        _ = addresses
        _ = ints
        _ = floats
        _ = output
