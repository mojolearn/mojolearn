# SPDX-License-Identifier: Apache-2.0
"""Typed half-precision conversions moved from bindings/sequential_inputs.mojo."""
from std.memory import bitcast
from gemm.f16_widen import f16_bits_to_f32


def sequence_widen_halves(src_addr: Int, dst_addr: Int, count: Int):
    """Half bit words to float32, one per element (the shell extends its value list from the result)."""
    var src = Pointer[UInt16, ImmUntrackedOrigin](unsafe_from_address=src_addr)
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst_addr)
    for k in range(count):
        dst[k] = f16_bits_to_f32(src[k])


def sequence_narrow_halves(src_addr: Int, dst_addr: Int, count: Int):
    """Exact half scalars, given as binary64, serialized as half bit words."""
    var src = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=src_addr)
    var dst = MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=dst_addr)
    for k in range(count):
        dst[k] = bitcast[DType.uint16](Float16(src[k]))
