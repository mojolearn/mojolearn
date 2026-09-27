# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-SQ: 8-bit scalar quantization of the IVF residuals, per-cell
arithmetic shared by the GPU and the CPU (lane/algos-ann, pass 1).

Reference: cuVS `cpp/src/neighbors/ivf_sq/ivf_sq_build.cuh` (per-dimension
vmin / vmax of the residuals, a 5% margin, delta = (range + 2 margin) / 255,
code = clamp(roundf((v - vmin) / delta), 0, 255), :183-250 and :484-525) and
`ivf_sq_search.cuh` (decode and the L2 scan).

THE FIXED-ORDER DESIGN
  * vmin / vmax: one cell per dimension scanning rows ascending with a
    strict compare, so a -0.0 / 0.0 tie keeps the first row's sign (their
    reduction's order is the device's).
  * roundf is restated as trunc plus an exact half test (x - trunc(x) is
    exact in float32), no libm call.
  * the decoded value is one pinned fused step vmin + code * delta; the scan
    distance is the ascending fused square sum; top-k under (distance, id).
"""

from std.math import trunc
from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add
from x_ann.ivf_pq_core import F32P, I32P, pq_coarse_dist, pq_inf, pq_insert


@always_inline
def sq_range_cell(c: Int, r: F32P, n: Int, dim: Int, vmin: F32P, delta: F32P):
    var lo = ftz(r.unsafe_load(c))
    var hi = lo
    for i in range(1, n):
        var v = ftz(r.unsafe_load(i * dim + c))
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    var rng = ftz(hi - lo)
    var margin = ftz(identical_mul(rng, Float32(0.05)))
    if rng > Float32(0.0):
        delta.unsafe_store(c, ftz(identical_div(ftz(rng + ftz(identical_mul(Float32(2.0), margin))), Float32(255.0))))
    else:
        delta.unsafe_store(c, Float32(1.0))
    vmin.unsafe_store(c, ftz(lo - margin))


@always_inline
def sq_encode_cell(e: Int, r: F32P, dim: Int, vmin: F32P, delta: F32P, codes: I32P):
    var c = e % dim
    var dv = delta.unsafe_load(c)
    var x = ftz(identical_div(ftz(ftz(r.unsafe_load(e)) - vmin.unsafe_load(c)), dv))
    var code = Float32(0.0)
    if x > Float32(0.0):
        var t = trunc(x)
        code = t + Float32(1.0) if ftz(x - t) >= Float32(0.5) else t
        if code > Float32(255.0):
            code = Float32(255.0)
    codes.unsafe_store(e, Int32(Int(code)))


@always_inline
def sq_search_cell(
    qi: Int, queries: F32P, dim: Int, centers: F32P, n_lists: Int, offsets: I32P,
    list_indices: I32P, codes: I32P, vmin: F32P, delta: F32P, k: Int, n_probes: Int,
    mask: I32P, out_d: F32P, out_i: I32P, out_n: I32P,
):
    """The IVF-PQ probe walk with the SQ scan; `mask[row] == 0` removes a row
    (the sample filter)."""
    var base = qi * k
    var q_off = qi * dim
    for s in range(k):
        out_d.unsafe_store(base + s, pq_inf())
        out_i.unsafe_store(base + s, Int32(-1))
    var prev_d = Float32(0.0)
    var prev_l = -1
    var n_cand = 0
    for _ in range(n_probes):
        var best_l = -1
        var best_d = Float32(0.0)
        for l in range(n_lists):
            var d = pq_coarse_dist(queries, q_off, centers, l, dim)
            var after = prev_l < 0 or d > prev_d or (d == prev_d and l > prev_l)
            if after and (best_l < 0 or d < best_d or (d == best_d and l < best_l)):
                best_l = l
                best_d = d
        if best_l < 0:
            break
        prev_l = best_l
        prev_d = best_d
        for slot in range(Int(offsets.unsafe_load(best_l)), Int(offsets.unsafe_load(best_l + 1))):
            var row = Int(list_indices.unsafe_load(slot))
            if mask.unsafe_load(row) == 0:
                continue
            var acc = Float32(0.0)
            for c in range(dim):
                var qr = ftz(ftz(queries.unsafe_load(q_off + c)) - ftz(centers.unsafe_load(best_l * dim + c)))
                var dec = ftz(identical_mul_add(
                    Float32(Int(codes.unsafe_load(row * dim + c))), delta.unsafe_load(c), vmin.unsafe_load(c)
                ))
                var diff = ftz(qr - dec)
                acc = ftz(identical_mul_add(diff, diff, acc))
            pq_insert(k, base, acc, Int32(row), out_d, out_i)
            n_cand += 1
    out_n.unsafe_store(qi, Int32(n_cand))
