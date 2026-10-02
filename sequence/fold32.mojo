# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PARALLEL FOLD ORDER of the sequence lane's series fits (lane
hr2-kpca-seq, 2026-10-02): a sum over items i = 0 .. n - 1 is 32 strided
slots (slot s folds items s, s + 32, s + 64, ... in ascending order, each
with the caller's operation), then the slots pairwise, offsets 16, 8, 4, 2,
1 (s[l] = s[l] + s[l + off] for l < off), s[0] the result.

One warp runs it as 32 lanes (a 64-lane AMD wavefront: both halves the same
32, every exchange inside its half) and an xor butterfly, where lane l adds
its partner's value: a + b and b + a are the same word, so every lane ends
with s[0]'s bits. The host column and the one-thread kernel run the same
slots and pairs in a loop. Every vendor and the CPU column: one order."""
from std.gpu.primitives.warp import shuffle_xor

from sequence.ops import add

comptime FOLD_L = 32


@always_inline
def tree32(mut s: InlineArray[Float32, FOLD_L]) -> Float32:
    """The slots' pairwise tree, serially (the host column's statements)."""
    comptime for e in range(5):
        comptime off = FOLD_L >> (e + 1)
        comptime for l in range(off):
            s[l] = add(s[l], s[l + off])
    return s[0]


@always_inline
def tree32_warp(v: Float32) -> Float32:
    """The same tree on a warp: lane l (mod 32) holds slot l; every lane
    returns the result."""
    var x = v
    comptime for e in range(5):
        comptime off = FOLD_L >> (e + 1)
        x = add(x, shuffle_xor(x, UInt32(off)))
    return x
