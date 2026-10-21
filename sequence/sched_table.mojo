# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_SCHED_FAST_TABLE (FAST + Apple default since rab13-schedtable,
rollback -D MOJOLEARN_SCHED_FAST_TABLE_OFF): a block
of ExponentialLR values base_lr gamma^e, e = e0 .. e0 + n - 1, for
`python/mojolearn/_x_sequence_sched.py` to serve one list index per
`lr_at` call. CPU-ONLY ROUTE by nature (a learning-rate schedule is one
scalar per optimizer step; the opponent is CPU torch, and the schedule was
host Python by contract, DEVIATION 5540).

THE SAME BITS AS THE CONTRACT. y = base_lr * gamma ** e in float64 is within
a few float64 ulps of the exact rational value (pow is correctly rounded to
< 1 ulp on the hosts we ship, the product adds 1/2 ulp). The value is kept
only when y (1 - 1e-12) and y (1 + 1e-12), an interval about 4,500 float64
ulps wide each side that therefore holds the exact value, round to the SAME
normal finite float32: rounding is monotone, so the exact value rounds there
too. Anything else (the exact value near a float32 rounding boundary, below
the smallest normal where the contract flushes, or overflowing) is written
as NaN and the Python layer decides it on its exact path."""
from std.math import isfinite, nan

comptime F64P = MutPointer[Float64, MutUntrackedOrigin]
#: the smallest normal float32, 2^-126
comptime F32_MIN_NORMAL = Float32(1.1754943508222875e-38)
#: the relative half-width of the enclosure around the float64 value
comptime SCHED_REL = 1.0e-12


def sched_exp_block(dst: F64P, base: Float64, gamma: Float64, e0: Int, n: Int) raises -> Int:
    """Writes dst[i] = the contract's float32 of base_lr gamma^(e0 + i) as a
    float64, or NaN where undecided, for i < n (base_lr = base, both > 0
    and finite; e0 >= 0, n >= 0). Returns the number of NaN entries. The
    Python entry (`sched_exp_block_py`) is `sequence/sched_table_pyapi.mojo`'s."""
    if not (base > 0.0 and gamma > 0.0 and isfinite(base) and isfinite(gamma)) or e0 < 0 or n < 0:
        raise Error("sched_exp_block: base_lr, gamma > 0 finite, e0 >= 0, n >= 0")
    var undecided = 0
    for i in range(n):
        var y = base * (gamma ** Float64(e0 + i))
        var flo = (y * (1.0 - SCHED_REL)).cast[DType.float32]()
        var fhi = (y * (1.0 + SCHED_REL)).cast[DType.float32]()
        if flo == fhi and flo >= F32_MIN_NORMAL and isfinite(flo):
            dst[i] = flo.cast[DType.float64]()
        else:
            dst[i] = nan[DType.float64]()
            undecided += 1
    return undecided
