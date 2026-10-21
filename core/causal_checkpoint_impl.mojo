# SPDX-License-Identifier: Apache-2.0
"""Typed half-precision widening moved word for word from bindings/causal_checkpoint.mojo."""
from std.memory import bitcast
from gemm.f16_widen import f16_bits_to_f32


def checkpoint_widen_halves(
    source: MutPointer[UInt16, MutUntrackedOrigin], dest: MutPointer[Float32, MutUntrackedOrigin],
    count: Int, dtype: String,
):
    for i in range(count):
        dest[i] = f16_bits_to_f32(source[i]) if dtype == "F16" else bitcast[DType.float32](UInt32(source[i]) << 16)
