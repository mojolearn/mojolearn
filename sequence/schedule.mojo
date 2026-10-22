# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The minibatch schedule of the sequence lane's trainers (lane
cgr4-py-compute: it was numpy in `_x_sequence_rnn.py`): every epoch's row
order, one Fisher-Yates permutation per epoch from one splitmix64 stream
(seeded by the caller), or the identity without shuffle. Integers only; the
same Mojo in both bindings, so the same order on every column."""
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]


@always_inline
def splitmix64(mut s: UInt64) -> UInt64:
    s += UInt64(0x9E3779B97F4A7C15)
    var z = s
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def fill_epoch_order(op: I32P, n: Int, shuffle: Bool, mut s: UInt64):
    """op[0:n] = 0..n-1, then (shuffle) one Fisher-Yates permutation drawn
    from the splitmix64 stream `s` (advanced in place)."""
    for i in range(n):
        op[i] = Int32(i)
    if shuffle:
        var i = n - 1
        while i > 0:
            var j = Int(splitmix64(s) % UInt64(i + 1))
            var t = op[i]
            op[i] = op[j]
            op[j] = t
            i -= 1

