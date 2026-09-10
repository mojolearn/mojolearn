# SPDX-License-Identifier: Apache-2.0
"""DEVIATION 2486: shared host copies, without floating-point arithmetic.

Counts are nonnegative numbers of elements. Callers validate spans and keep
both allocations alive for the entire call. Copies require disjoint spans (or exactly the same
span); partial overlap is not supported. No pointer is retained.
"""
from std.memory import memcpy


def f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float64 buffer address")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def u32_ptr(addr: Int) raises -> MutPointer[UInt32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null uint32 buffer address")
    return MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=addr)


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


def read_f32(addr: Int, n: Int) raises -> List[Float32]:
    """An owned copy using the existing Transformer memcpy implementation."""
    var src = f32_ptr(addr)
    if n < 0:
        raise Error("mojolearn: negative float32 copy length")
    var out = List[Float32](length=n, fill=Float32(0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=src, count=n)
    return out^


def read_i32(addr: Int, n: Int) raises -> List[Int32]:
    """Owned integer copy for metric labels; no float conversion."""
    var src = i32_ptr(addr)
    if n < 0:
        raise Error("mojolearn: negative int32 copy length")
    var out = List[Int32](length=n, fill=Int32(0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=src, count=n)
    return out^
