"""MiniBatchKMeans' center update as cells (lane/neural-pass133, 2026-10-02).

`x_cluster/minibatch.mojo::minibatch_step`'s unit-weight `update_center_dense`
for one center (the host ops' loop) and for one (center, feature) word (the
device kernel's thread), the same statements: wsum counts the center's rows
in batch order; a center with rows gets `c * w`, then `+ x` for its rows in
batch order, then `* (1 / (w + wsum))`; its count becomes `w + wsum`."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from checks.numerics import ftz, identical_div, identical_mul
from checks.soft_f64 import sf64_from_int, sf64_lt, sf64_mul
from x_cluster.bodies import splitmix_at

comptime FPtr = MutPointer[Float32, MutAnyOrigin]
comptime IPtr = MutPointer[Int32, MutAnyOrigin]


@always_inline
def mb_center_wsum(pl: IPtr, batch: Int, j: Int) -> Float32:
    var wsum = Float32(0)
    for t in range(batch):
        if Int(pl[t]) == j:
            wsum = ftz(wsum + Float32(1))
    return wsum


@always_inline
def mb_center_word(pb: FPtr, batch: Int, pl: IPtr, cjf: Float32, wj: Float32, wsum: Float32, j: Int, f: Int,
                   d: Int) -> Float32:
    """The center word (j, f) after the batch (wsum > 0)."""
    var acc = ftz(identical_mul(cjf, wj))
    for t in range(batch):
        if Int(pl[t]) == j:
            acc = ftz(acc + ftz(pb[t * d + f]))
    var alpha = ftz(identical_div(Float32(1), ftz(wj + wsum)))
    return ftz(identical_mul(acc, alpha))


@always_inline
def mb_center_update(pb: FPtr, batch: Int, pl: IPtr, pc: FPtr, pw: FPtr, j: Int, d: Int):
    var wsum = mb_center_wsum(pl, batch, j)
    if wsum > Float32(0):
        var wj = pw[j]
        for f in range(d):
            pc[j * d + f] = mb_center_word(pb, batch, pl, pc[j * d + f], wj, wsum, j, f, d)
        pw[j] = ftz(wj + wsum)


# 2^-53 as a binary64 bit pattern: `SplitMix64.unit`'s scale.
comptime MB_SF64_2M53 = UInt64(0x3CA0000000000000)


@always_inline
def mb_cum_word(cum: IPtr, i: Int) -> UInt64:
    """The bit pattern of cumulative weight i: words 2 i (low) and 2 i + 1."""
    return UInt64(UInt32(cum[2 * i])) | (UInt64(UInt32(cum[2 * i + 1])) << 32)


@always_inline
def mb_wdraw_cell(idx: IPtr, cum: IPtr, n: Int, state: UInt64, t: Int):
    """idx[t] = `common.weighted_draw(cum_w, rng)`'s draw t + 1 of the
    splitmix64 stream at `state` (lane n3): `unit() * cum[n - 1]` and the
    side-left search in soft binary64 (`checks/soft_f64`: integer
    instructions, correctly rounded), so the device and the host column
    write the host loop's indices on every vendor. `cum` holds the ascending
    Float64 cumulative weights as 2 n int32 words (`mb_cum_word`)."""
    var u = splitmix_at(state, UInt64(t + 1)) >> 11
    var unit = sf64_mul(sf64_from_int(Int(u)), MB_SF64_2M53)
    var v = sf64_mul(unit, mb_cum_word(cum, n - 1))
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        # cum[mid] <= v (both non-negative, finite)
        if not sf64_lt(v, mb_cum_word(cum, mid)):
            lo = mid + 1
        else:
            hi = mid
    idx[t] = Int32(lo if lo < n else n - 1)
