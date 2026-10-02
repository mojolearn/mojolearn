# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TruncatedSVD's explained-variance tail, one definition for the device
(`tsvd_finish_kernel`, one block) and the host column
(`decomposition/host/pca_oracle.mojo::tsvd_explained_finish`), from
cpu-gpu-cleanup c-decomp (2026-10-02; the GPU binding downloaded the column
variances and finished on the host in Float64, which Metal has not).

The total variance: TSVD_FIN_TPB slots, slot t the float-float sum of the
column variances [t c, (t + 1) c) ascending (c = ceil(n_features /
TSVD_FIN_TPB)), the slots joined by a fixed halving tree (slot t takes slot
t + half), rounded once to float32. Each ratio is one float32 division (0
when the total is not positive). The same statements on both columns."""
from x_linear.ff import FF, ff_add, ff_add_f, ff_f32
from checks.numerics import ftz

comptime TSVD_FIN_TPB = 256
comptime _P = MutPointer[Float32, MutAnyOrigin]


@always_inline
def tsvd_slot_sum(var_x: _P, nf: Int, t: Int) -> FF:
    var c = (nf + TSVD_FIN_TPB - 1) // TSVD_FIN_TPB
    var lo = t * c
    var hi = min(lo + c, nf)
    var acc = FF(Float32(0), Float32(0))
    for i in range(lo, hi):
        acc = ff_add_f(acc, var_x.unsafe_load(i))
    return acc


@always_inline
def tsvd_ratio(var_t: _P, i: Int, full: Float32) -> Float32:
    return ftz(var_t.unsafe_load(i) / full) if full > Float32(0.0) else Float32(0.0)


def tsvd_finish_host(var_t: _P, nc: Int, var_x: _P, nf: Int, explained: _P, ratio: _P):
    """The host column: the slots, the same tree, the same divisions."""
    var hi = List[Float32](length=TSVD_FIN_TPB, fill=Float32(0))
    var lo = List[Float32](length=TSVD_FIN_TPB, fill=Float32(0))
    for t in range(TSVD_FIN_TPB):
        var s = tsvd_slot_sum(var_x, nf, t)
        hi[t] = s.hi
        lo[t] = s.lo
    var half = TSVD_FIN_TPB // 2
    while half > 0:
        for t in range(half):
            var s = ff_add(FF(hi[t], lo[t]), FF(hi[t + half], lo[t + half]))
            hi[t] = s.hi
            lo[t] = s.lo
        half //= 2
    var full = ff_f32(FF(hi[0], lo[0]))
    for i in range(nc):
        explained.unsafe_store(i, var_t.unsafe_load(i))
        ratio.unsafe_store(i, tsvd_ratio(var_t, i, full))
