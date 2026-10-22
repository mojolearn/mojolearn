# SPDX-License-Identifier: Apache-2.0
# Native input casts only; sequence optimizer update kernels are unchanged.
# NumPy integer -> float32 inputs round directly. Do not route via float64:
# int64 values just above a float32 midpoint can double-round differently.
from core.cast_in import cast_f32_bits


def sequence_optimizer_cast(addr: Int, shape: List[Int], strides: List[Int], item: Int,
                            kind: Int, native: Bool, dst: Int, count: Int, broadcast: Bool):
    var output = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=dst)
    for i in range(count):
        var index = 0 if broadcast else i
        var offset = 0
        for axis in range(len(shape)):
            var k = len(shape) - axis - 1
            if shape[k] > 0:
                offset += (index % shape[k]) * strides[k]
                index //= shape[k]
        var source = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr + offset)
        var raw = UInt64(0)
        for byte in range(item):
            raw |= UInt64(source[byte if native else item - 1 - byte]) << UInt64(byte * 8)
        output[i] = cast_f32_bits(raw, kind, item)
