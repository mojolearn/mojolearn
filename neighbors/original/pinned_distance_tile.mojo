# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The tiled k-NN arm's distances, computed where we can see the order.

DEVIATION 505 (IDENTITY_PATHS row 24). Reached only under
`NUMERIC_IDENTICAL`.

NOT A PORT. cuVS's `tiled_brute_force_knn` gets this tile from
`cuvs::distance::pairwise_distance`, which is cuBLAS underneath
(`knn_brute_force.cuh:172-183`), and this tree mirrors that call with MAX's
`linalg.matmul` through `core/gemm.mojo::gemm_nt`. Under `NUMERIC_FAST` that
is the right thing and stays: a library with no source is a library we do
not port.

WHY `IDENTICAL` CANNOT USE IT
------------------------------
A vendor matmul chooses its own tile shape and its own k-split, per vendor
and per shape, and a k-split IS a summation order. Nothing in this
repository can pin it, read it, or check it -- `VENDOR_LIBRARIES.md` is
about which calls we may make, not about what they do to the last bit. Two
GPUs running `linalg.matmul` on the same inputs are entitled to two
different `z`, and the whole expanded identity is built on `z`.

cuVS is in the same position and worse: their default distance GEMM runs at
`CUBLAS_COMPUTE_32F_FAST_TF32` (`unfused_distance_nn.cuh:196`), ten mantissa
bits, so their float32 k-NN is not float32 and is not reproducible across
NVIDIA GPU MODELS either. Our tiled arm inherits their DESIGN, not their
irreproducibility, and this file is where the two part company.

WHAT THIS KERNEL IS
-------------------
One thread per output cell. Each thread walks the feature axis ASCENDING and
accumulates through `identical_mul_add`, so the summation order is a pure
function of `k` and nothing else -- not the grid, not the block, not the
device, not the shape. Then the expanded epilogue, the same one
`core/expand_distances.mojo` applies, folded in so the tile is written once.

It is deliberately the SIMPLEST correct shape rather than a fast one:

- no shared-memory staging, so no page count to pin;
- no register tile, so no `AccRowsPerTh` to keep in step with a policy;
- no split of the k axis, so nothing to fold in a chosen order.

Every one of those would be a second thing to pin. `IDENTICAL` is the mode
that buys reproducibility with speed, and the price is stated in the lane
file rather than hidden: this reads `k` floats per cell from global memory
where the vendor matmul reads them once per tile.

THE FAST ARM'S BITS DO NOT MOVE. Nothing here is reachable unless
`GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL`; `tiled_brute_force_knn` keeps
calling `gemm_nt` plus `expand_distances_kernel` in the default build.
"""

from std.gpu import block_dim, block_idx, thread_idx

from original.numerics import ftz, identical_mul_add, identical_sqrt


comptime PINNED_TILE_TPB = 256


def pinned_distance_tile_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    n_features_in: Int32,
    is_sqrt_in: Int32,
):
    """`z[i][j] = ||q_i||^2 + ||y_j||^2 - 2 q_i . y_j`, clamped at zero.

    The dot product is accumulated in ONE thread over the whole feature
    axis, ascending, so it has one order everywhere. The norms are the ones
    `compute_norms` already produced, which is theirs
    (`knn_brute_force.cuh:110-146`, hoisted out of the tile loop).

    The clamp is theirs too (`unfused_distance_nn.cuh:80-81`): GEMM
    round-off makes a point sitting on its own neighbour come out slightly
    negative and `sqrt` of that is NaN. It is kept even though this arm has
    no GEMM, because a cancellation can go negative without one.
    """
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var d = Int(n_features_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n_rows * n_cols:
        return

    var row = idx // n_cols
    var col = idx % n_cols

    var acc = Float32(0.0)
    for f in range(d):
        var qv = ftz(q.unsafe_load(row * d + f))
        var yv = ftz(y.unsafe_load(col * d + f))
        acc = ftz(identical_mul_add(qv, yv, acc))

    var dist = ftz(
        identical_mul_add(
            Float32(-2.0),
            acc,
            ftz(ftz(q_norm.unsafe_load(row)) + ftz(y_norm.unsafe_load(col))),
        )
    )
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    if is_sqrt_in != 0:
        # DEVIATION 550 (2026-08-23): `identical_sqrt`, not the stdlib
        # `sqrt`. This tile is the IDENTICAL arm, and the stdlib sqrt is
        # NVIDIA's approximate PTX sqrt (DEVIATION 258: 180,714 of 2^20
        # inputs off by one ulp) -- E1U's knn card at 8660400 agreed on
        # index_norm and query_norm and diverged at out_dist on the H100,
        # which is exactly one unrouted sqrt after the norms. Apple's
        # native sqrt is correctly rounded, so this moves no Apple bit.
        dist = ftz(identical_sqrt(dist))
    z.unsafe_store(idx, dist)
