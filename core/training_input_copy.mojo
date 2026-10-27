# SPDX-License-Identifier: Apache-2.0
# Native byte-preserving admission copy; no arithmetic conversion is performed.
from std.memory import memcpy
from core.host_helpers_impl import all_finite_f32


def training_copy_f32(source: Int, target: Int, shape: List[Int], strides: List[Int], name: String) raises:
    var count = 1
    var contiguous = True
    var stride = 4
    for i in range(len(shape)):
        var axis = len(shape) - 1 - i
        if strides[axis] != stride:
            contiguous = False
        count *= shape[axis]
        stride *= shape[axis]
    var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=target)
    if contiguous:
        # Ordinary buffers may start at an unaligned byte offset. Copy bytes
        # without inventing an alignment guarantee for the admitted source.
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=target),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source), count=count * 4)
    else:
        for i in range(count):  # cpu-route: byte-preserving strided input admission
            var index = i
            var address = source
            for k in range(len(shape)):
                var axis = len(shape) - 1 - k
                address += (index % shape[axis]) * strides[axis]
                index //= shape[axis]
            var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
            dst[i] = UInt32(bytes[0]) | (UInt32(bytes[1]) << 8) | (UInt32(bytes[2]) << 16) | (UInt32(bytes[3]) << 24)
    # Same native predicate as the original Array.all_finite boundary.
    if all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=target), count) == 0:
        raise Error(name + " must be finite float32")
