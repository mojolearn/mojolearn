# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The pinned distance tile with two pins BROKEN ON PURPOSE, for the check.

NOT A PORT, NOT REACHED by any driver or by the Python-facing entry. A copy
of `neighbors/checks/pinned_distance_tile.mojo`'s kernel (DEVIATION 505)
with two sabotage arms that `linkage_check.mojo` selects and nothing else
can: the copy exists because that file belongs to the neighbors lane and a
sabotage arm does not belong in a production kernel. The un-sabotaged arm
of THIS kernel is never launched: `connectivities.mojo` calls the real one
when `sabotage == LINK_SAB_NONE` and this one otherwise, so the production
bits never depend on this file.

ARMS
  LINK_SAB_ROTATE_CONTRACTION  the feature loop starts at `block_idx % d`
                               and wraps; a different summation order per
                               block, which a pinned order forbids.
  LINK_SAB_STD_SQRT            `std.math.sqrt` at the seam instead of
                               `identical_sqrt`. Apple's is correctly
                               rounded, so on this device the bits MAY NOT
                               move; the check reports either way.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt

from hierarchy.checks.edge_order import (
    LINK_SAB_ROTATE_CONTRACTION,
    LINK_SAB_STD_SQRT,
)
from checks.numerics import ftz, identical_mul_add, identical_sqrt


def sabotage_distance_tile_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    n_features_in: Int32,
    is_sqrt_in: Int32,
    sabotage: Int32,
):
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var d = Int(n_features_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n_rows * n_cols:
        return
    var row = idx // n_cols
    var col = idx % n_cols

    var start = 0
    if sabotage == LINK_SAB_ROTATE_CONTRACTION and d > 0:
        start = Int(block_idx.x) % d

    var acc = Float32(0.0)
    for t in range(d):
        var f = start + t
        if f >= d:
            f -= d
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
        if sabotage == LINK_SAB_STD_SQRT:
            dist = ftz(sqrt(dist))
        else:
            dist = ftz(identical_sqrt(dist))
    z.unsafe_store(idx, dist)
