# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The spectral routes' dense-to-COO scan over FLOAT32 and the precomputed
kNN connectivity affinity (lane/py-dn-kern, 2026-09-28). Host code only:
compares, integer bookkeeping and the exact values 0, 0.5 and 1. Shared by
the base binding (`bindings/_mojolearn.mojo`) and its CPU route
(`bindings/_mojolearn_core_host.mojo`), so both columns run these bodies.

`nonzero_f32_count` / `nonzero_f32_fill` are DEVIATION 2489's float64 scan
over a float32 matrix: the test is `v != 0.0` (-0.0 a zero, NaN kept), row
major, one row and column per kept value and the value copied. They replace
`_spectral_impl._DenseCOO`, a Python n^2 loop that read each float32 cell as
a Python float (an exact widening) and made the same test, so the triples are
the same bytes.

`knn_affinity_f32` is `SpectralEmbedding._precomputed_knn_affinity`'s body:
each row keeps its k smallest candidate distances ordered by (value, column),
the Python `sorted` of `(v, j)` tuples (-0.0 == 0.0, so a tie goes to the
lower column), marks C[i, j] = 1, and writes A = 0.5 (C + C^T), which only
takes the values 0, 0.5 and 1. A candidate's sort key is its float32 bit
pattern (monotonic over the nonnegative values the route admits; -0.0 is
keyed as +0.0) above its column, one integer sort per row. It lives in
core/dense_coo_host.mojo (the CPU route) and core/dense_coo_device.mojo (the
GPU binding) since cpu-gpu-cleanup c-core."""

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]


def nonzero_f32_count(sp: F32P, count: Int) -> Int:
    var nz = 0
    for i in range(count):
        if sp.unsafe_load(i) != Float32(0):
            nz += 1
    return nz


def nonzero_f32_fill(sp: F32P, nr: Int, nc: Int, rp: I32P, cp: I32P, vp: F32P, cap: Int) raises -> Int:
    """Row-major triples of the entries with `v != 0.0`; raises when more
    than `cap` would be written (a stale count)."""
    var k = 0
    for r in range(nr):
        var base = r * nc
        for c in range(nc):
            var v = sp.unsafe_load(base + c)
            if v != Float32(0):
                if k >= cap:
                    raise Error(
                        "nonzero_f32_fill: more than " + String(cap)
                        + " nonzero entries; count first with nonzero_f32_count"
                    )
                rp.unsafe_store(k, Int32(r))
                cp.unsafe_store(k, Int32(c))
                vp.unsafe_store(k, v)
                k += 1
    return k
