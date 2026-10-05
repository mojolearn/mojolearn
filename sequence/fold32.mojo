# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PARALLEL FOLD ORDER of the sequence lane's series fits (from
lane/hr2-kpca-seq e99030f6f, 2026-10-02; brought onto the team kernels by
lane fix-t1-seq): a sum over items i = 0 .. n - 1 is 32 strided slots (slot
s folds items s, s + 32, s + 64, ... in ascending order from 0.0, each with
the caller's operation), then the slots pairwise, offsets 16, 8, 4, 2, 1
(s[l] = s[l] + s[l + off] for l < off), s[0] the result.

The device team computes the slots on up to 32 threads and every thread runs
the tree on the stored slots; the host column runs the same slots and pairs
in a loop. Every vendor and the CPU column: one order."""
from sequence.ops import add

comptime FOLD_L = 32


@always_inline
def tree32(mut s: InlineArray[Float32, FOLD_L]) -> Float32:
    """The slots' pairwise tree, serially (every column's statements)."""
    comptime for e in range(5):
        comptime off = FOLD_L >> (e + 1)
        comptime for l in range(off):
            s[l] = add(s[l], s[l + off])
    return s[0]
