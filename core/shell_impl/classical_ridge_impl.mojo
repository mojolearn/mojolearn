# SPDX-License-Identifier: Apache-2.0
"""Typed output widening moved word for word from bindings/classical_ridge.mojo."""


def classical_widen_f32_f64(src_addr: Int, dst_addr: Int, size: Int):
    """Exact widening of a completed float32 prediction buffer to float64."""
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=src_addr)
    var dst = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=dst_addr)
    for k in range(size):  # native output marshaling: exact widening of the completed prediction buffer
        dst[k] = Float64(src[k])
