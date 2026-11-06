# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Vector passes over admitted float32 words for the neighbor and cluster
public doors (p3-neighbors-cluster, 2026-11-06).

The public begins scanned every X word for NaN/infinity, and widened float32
results to float64, one scalar at a time. These do the same eight lanes at a
time (the `core/staged_download.mojo::_first_nonfinite` and
`core/host_helpers_impl.mojo::cast_f64_to_f32` forms), with the scalar tail.
The finite test reads only exponent bits and the widening is exact, so no
result or refusal changes. No host threads: the GPU bindings import these.
"""
from std.memory import bitcast

comptime _P3NC_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _P3NC_F64P = MutPointer[Float64, MutUntrackedOrigin]


def p3nc_all_finite_f32(p: _P3NC_F32P, n: Int) -> Bool:
    """True when none of the `n` float32 words at `p` is a NaN or an
    infinity (exponent bits all ones)."""
    comptime W = 8
    var exp = SIMD[DType.uint32, W](0x7F800000)
    var i = 0
    var body = n - n % W
    while i < body:
        var bits = bitcast[DType.uint32, W](p.unsafe_load[width=W](i)) & exp
        if bits.eq(exp).reduce_or():
            return False
        i += W
    while i < n:
        if (bitcast[DType.uint32](p.unsafe_load(i)) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            return False
        i += 1
    return True


def p3nc_widen_f32_f64(sp: _P3NC_F32P, dp: _P3NC_F64P, n: Int):
    """`dp[i] = Float64(sp[i])` for `i` in `[0, n)` (exact)."""
    var i = 0
    var body = n - (n % 8)
    while i < body:
        var v = sp.unsafe_load[width=8](i)
        dp.unsafe_store[width=8](i, v.cast[DType.float64]())
        i += 8
    while i < n:
        dp.unsafe_store(i, sp.unsafe_load(i).cast[DType.float64]())
        i += 1
