# SPDX-License-Identifier: Apache-2.0
"""Typed host copy moved word for word from bindings/hostptr.mojo."""


@always_inline
def copy_f32[src_origin: Origin, dst_origin: MutOrigin, //](
    src_ptr: Pointer[Float32, src_origin],
    dst_ptr: MutPointer[Float32, dst_origin], n: Int,
):
    """Hostpass B2 SIMD-8 copy, including the non-multiple-of-eight tail."""
    var i = 0
    var body = n - n % 8
    while i < body:
        dst_ptr.unsafe_store[width=8](i, src_ptr.unsafe_load[width=8](i))
        i += 8
    while i < n:
        dst_ptr.unsafe_store(i, src_ptr.unsafe_load(i))
        i += 1
