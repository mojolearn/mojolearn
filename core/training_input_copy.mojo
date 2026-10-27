# SPDX-License-Identifier: Apache-2.0
# Native byte-preserving admission copy; no arithmetic conversion is performed.
from std.memory import memcpy


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
        memcpy(dest=dst, src=MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=source), count=count)
    else:
        for i in range(count):  # cpu-route: byte-preserving strided input admission
            var index = i
            var address = source
            for k in range(len(shape)):
                var axis = len(shape) - 1 - k
                address += (index % shape[axis]) * strides[axis]
                index //= shape[axis]
            dst[i] = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)[0]
    for i in range(count):  # cpu-route: existing finite-input admission on copied words
        if (dst[i] & UInt32(0x7f800000)) == UInt32(0x7f800000):
            raise Error(name + " must be finite float32")
