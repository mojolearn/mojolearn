# SPDX-License-Identifier: Apache-2.0
"""Typed element work behind the parallel byte LM boundary
(`bindings/parallel_byte_native.mojo`): span arithmetic, the word copy of an
admitted input array, and the value checks on admitted state. No Python here;
the boundary resolves addresses and calls these."""
from std.math import isfinite
from std.memory import memcpy


def _parallel_span(dims: List[Int]) raises -> Int:
    var count = 1
    for d in dims:
        if d < 0 or (d != 0 and count > 2147483647 // d):
            raise Error("Byte-LM shape exceeds signed 32-bit indexing")
        count *= d
    return count


def parallel_copy_words(src_addr: Int, dst_addr: Int, dims: List[Int], n: Int, fortran_order: Bool):
    """Copy `n` four-byte words from `src_addr` to `dst_addr`, reordering a
    column-major source of shape `dims` into row-major when `fortran_order`."""
    var src = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=src_addr)
    var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=dst_addr)
    if fortran_order:
        var strides = List[Int]()
        var stride = 1
        for axis in range(len(dims)):
            strides.append(stride)
            stride *= dims[axis]
        for i in range(n):
            var remaining = i
            var source = 0
            for reverse in range(len(dims)):
                var axis = len(dims) - reverse - 1
                source += (remaining % dims[axis]) * strides[axis]
                remaining //= dims[axis]
            dst[i] = src[source]
    else:
        memcpy(dest=dst, src=src, count=n)


def parallel_require_finite_f32(addr: Int, n: Int) raises:
    var fp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(n):
        if not isfinite(fp[i]):
            raise Error("parallel input requires finite float32 values")


def parallel_require_nonnegative_f32(addr: Int, n: Int) raises:
    var vp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(n):
        if vp[i] < 0:
            raise Error("Byte-LM requires nonnegative second moments")


def parallel_require_binary_i32(addr: Int, n: Int) raises:
    var fp = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(n):
        if fp[i] < 0 or fp[i] > 1:
            raise Error("Byte-LM requires binary optimizer flags")
