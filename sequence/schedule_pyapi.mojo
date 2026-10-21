# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entry of the sequence trainers' minibatch schedule
(`sequence/schedule.mojo`): PythonObject arguments to typed values and the
typed result back. Nothing else."""
from std.python import PythonObject

from sequence.schedule import I32P, epoch_schedule


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
    var seed = (UInt64(Int(py=ip[5])) << 32) | UInt64(Int(py=ip[4]))
    var op = I32P(unsafe_from_address=Int(py=addrs[0]))
    var sp = I32P(unsafe_from_address=Int(py=addrs[1]))
    return PythonObject(epoch_schedule(op, sp, n, epochs, bs, shuffle, seed))
