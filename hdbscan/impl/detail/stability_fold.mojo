# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU-safe half of `hdbscan/impl/detail/stabilities.mojo` (lane
rehearsal-suite-green, 2026-10-08): the stability order keys, the fold width
and `cluster_stability_kernel`'s sum restated for the host column, moved
verbatim so the host oracle imports no `std.gpu` / `max.gpu` module. The
device file imports every name back.
"""
from std.memory import bitcast

from checks.numerics import ftz, identical_mul_add
from hierarchy.checks.edge_order import WEIGHT_KEY_NAN


@always_inline
def stability_order_key_bits(b: UInt32) -> Int32:
    """`hierarchy/checks/edge_order.mojo::weight_order_key` of the float
    whose bits are `b`, THE SAME MAP, spelled on the bits alone (the gfx942
    banner above `births_init_kernel`).

    The NaN test is `(b & 0x7FFFFFFF) > 0x7F800000`: exponent all ones and
    a nonzero mantissa, which is exactly the set `w != w` selects, for
    either sign. The rest is `weight_order_key`'s three integer operations.
    `check_stability_key_is_edge_order` in `hdbscan/checks/
    hdbscan_check.mojo` sweeps this against `weight_order_key` over every
    exponent, both zeros, both infinities and several NaN patterns on the
    host, so the two cannot drift.
    """
    if (b & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
        return WEIGHT_KEY_NAN
    var k: UInt32
    if (b & UInt32(0x80000000)) != UInt32(0):
        k = ~b
    else:
        k = b | UInt32(0x80000000)
    return bitcast[DType.int32](k ^ UInt32(0x80000000))


@always_inline
def stability_order_key(w: Float32) -> Int32:
    """`stability_order_key_bits` of `w`'s bits, for the host gate."""
    return stability_order_key_bits(bitcast[DType.uint32](w))


@always_inline
def stability_order_unkey_bits(k: Int32) -> UInt32:
    """The exact inverse of `stability_order_key_bits` away from NaN, as
    bits: `hierarchy/checks/edge_order.mojo::weight_order_unkey` without its
    final float cast. `check_stability_key_is_edge_order` asserts the round
    trip on every non-NaN pattern it sweeps."""
    var u = bitcast[DType.uint32](k) ^ UInt32(0x80000000)
    if (u & UInt32(0x80000000)) != UInt32(0):
        return u ^ UInt32(0x80000000)
    return ~u


comptime STAB_FOLD = 256
"""THE FOLD WIDTH, a fixed part of the stability sum's order (DEVIATION
1603): one block of `STAB_FOLD` threads per cluster, thread t folding the
segment positions t, t + STAB_FOLD, ... in that order, then a pairwise tree
over the `STAB_FOLD` partials (stride 128, 64, ..., 1). The host column
(`hdbscan_host_oracle.mojo::hdbh_stabilities`) and the check oracle fold in
the same order; changing this number changes the bits everywhere."""


def stability_fold_host(
    lambdas: List[Float32],
    sizes: List[Int32],
    lo: Int,
    hi: Int,
    birth: Float32,
) -> Float32:
    """`cluster_stability_kernel`'s sum on the host, in its order: the
    `STAB_FOLD` strided partials, then the pairwise tree. The CPU column
    (`hdbh_stabilities`) calls this."""
    var part = List[Float32](length=STAB_FOLD, fill=Float32(0.0))
    var n_seg = hi - lo
    for t in range(STAB_FOLD):
        var acc = Float32(0.0)
        var k = t
        while k < n_seg:
            var i = lo + k
            var term = ftz(lambdas[i] - birth)
            var size_f = sizes[i].cast[DType.float32]()
            acc = ftz(identical_mul_add(term, size_f, acc))
            k += STAB_FOLD
        part[t] = acc
    var s = STAB_FOLD // 2
    while s > 0:
        for t in range(s):
            part[t] = ftz(identical_mul_add(Float32(1.0), part[t + s], part[t]))
        s //= 2
    return part[0]
