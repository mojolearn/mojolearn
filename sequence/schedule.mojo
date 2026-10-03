# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The minibatch schedule of the sequence lane's trainers (lane
cgr4-py-compute: it was numpy in `_x_sequence_rnn.py`): every epoch's row
order, one Fisher-Yates permutation per epoch from one splitmix64 stream
(seeded by the caller), or the identity without shuffle, and each step's
(offset, count) pair into it. Integers only; the same Mojo in both
bindings, so the same schedule on every column."""
from std.python import PythonObject

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


def epoch_schedule_py(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [order (epochs * n) int32 out, steps (2 * epochs * ceil(n /
    bs)) int32 out]; ip = [n, epochs, bs, shuffle, seed_lo, seed_hi].
    Returns the number of steps."""
    if len(addrs) != 2 or len(ip) != 6:
        raise Error("epoch_schedule: requires 2 addresses and 6 integer parameters")
    var n = Int(py=ip[0])
    var epochs = Int(py=ip[1])
    var bs = Int(py=ip[2])
    var shuffle = Int(py=ip[3]) != 0
    var s = (UInt64(Int(py=ip[5])) << 32) | UInt64(Int(py=ip[4]))
    if n < 1 or epochs < 0 or bs < 1 or epochs * n >= 2147483647:
        raise Error("epoch_schedule: needs n >= 1, epochs >= 0, bs >= 1, epochs * n < 2^31 - 1")
    var op = I32P(unsafe_from_address=Int(py=addrs[0]))
    var sp = I32P(unsafe_from_address=Int(py=addrs[1]))
    var per = (n + bs - 1) // bs
    for e in range(epochs):
        var base = e * n
        fill_epoch_order(op + base, n, shuffle, s)
        for k in range(per):
            var start = k * bs
            sp[2 * (e * per + k)] = Int32(base + start)
            sp[2 * (e * per + k) + 1] = Int32(min(bs, n - start))
    return PythonObject(epochs * per)
