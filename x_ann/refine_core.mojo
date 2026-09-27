# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""refine: exact re-ranking of candidate neighbors (lane/algos-ann, pass 1).

Reference: cuVS `cpp/src/neighbors/refine/refine_device.cuh` /
`refine.cuh` (squared L2 of each candidate, then select_k). Here one query
per cell: the ascending fused square sum per candidate (an id < 0 is a
padding slot and is skipped, as their kOutOfBoundsRecord is), and a sorted
insertion under (distance, id), so equal distances resolve by id whatever
order the candidates arrive in."""

from checks.numerics import ftz, identical_mul_add
from x_ann.ivf_pq_core import F32P, I32P, pq_inf, pq_insert


@always_inline
def refine_cell(
    qi: Int, x: F32P, n: Int, d: Int, queries: F32P, cand: I32P, k0: Int, k: Int,
    out_d: F32P, out_i: I32P,
):
    var base = qi * k
    for s in range(k):
        out_d.unsafe_store(base + s, pq_inf())
        out_i.unsafe_store(base + s, Int32(-1))
    for t in range(k0):
        var v = Int(cand.unsafe_load(qi * k0 + t))
        if v < 0 or v >= n:
            continue
        # a repeated candidate is scored once: skip it if an earlier slot names it
        var dup = False
        for u in range(t):
            if Int(cand.unsafe_load(qi * k0 + u)) == v:
                dup = True
                break
        if dup:
            continue
        var acc = Float32(0.0)
        for c in range(d):
            var diff = ftz(ftz(queries.unsafe_load(qi * d + c)) - ftz(x.unsafe_load(v * d + c)))
            acc = ftz(identical_mul_add(diff, diff, acc))
        pq_insert(k, base, acc, Int32(v), out_d, out_i)
