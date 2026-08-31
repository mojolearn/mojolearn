# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Turn a GEMM output into distances, in place.

NOT A PORT of a file. In cuVS this is an EPILOGUE fused into the distance
call (`distance/detail/`), and in the unfused path it lives inside
`reduce_min_kernel` because the reduction consumes each element as it is
formed. Brute-force k-NN cannot do that: it needs every distance to survive
so a top-k can run over them, so the epilogue has to become its own pass.

The arithmetic is the same expanded identity and the same clamp
(`unfused_distance_nn.cuh:80-81`), including the reason the clamp exists:
GEMM round-off makes a point sitting on its own neighbor come out slightly
negative, and `sqrt` of that is NaN.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt

from mojo_only.numerics import ftz, identical_mul_add


def expand_distances_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    is_sqrt_in: Int32,
):
    """`z[i][j] <- ||x_i||^2 + ||y_j||^2 - 2 z[i][j]`, clamped at zero."""
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows_in) * n_cols:
        return

    var row = idx // n_cols
    var col = idx % n_cols
    # ONE pinned multiply-add, IDENTITY_PATHS row 9, and the seams flushed
    # for row 10. Under FAST `(-2) * z + (xn + yn)` is bit-for-bit the
    # subtraction it replaces. This kernel is on the TILED k-NN arm, whose
    # `z` comes from a vendor matmul -- the arithmetic here is pinned so
    # that the ONLY unpinned thing left on that arm is the product itself,
    # which is what its refusal under IDENTICAL names.
    var d = ftz(
        identical_mul_add(
            Float32(-2.0),
            ftz(z.unsafe_load(idx)),
            ftz(ftz(x_norm.unsafe_load(row)) + ftz(y_norm.unsafe_load(col))),
        )
    )
    if d <= Float32(0.0):
        d = Float32(0.0)
    if is_sqrt_in != 0:
        d = ftz(sqrt(d))
    z.unsafe_store(idx, d)
