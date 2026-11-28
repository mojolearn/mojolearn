# SPDX-License-Identifier: Apache-2.0
"""IEEE float32 to an unsigned radix key, shared by host and device sorts."""
from std.memory import bitcast


@always_inline
def sort_key_bits(v: Float32, descending: Bool) -> UInt32:
    """The sortable key of a float32: unsigned order is ascending float
    order (svm/checks/device_select.mojo `twiddle_keys_kernel`'s map:
    negatives bit-inverted, non-negatives with the sign bit set); inverted
    for a descending order."""
    var bits = bitcast[DType.uint32](v)
    var key: UInt32
    if bits & UInt32(0x80000000) != UInt32(0):
        key = ~bits
    else:
        key = bits | UInt32(0x80000000)
    if descending:
        key = ~key
    return key
