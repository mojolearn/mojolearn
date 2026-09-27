# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the GPU and CPU host bindings; product, not only a check.
"""Unit rows for the cosine metric (DEVIATION 5113, the cluster lane).

cuML's DBSCAN(metric='cosine') divides every row by its L2 norm in place
(`dbscan.cuh`, `rowNorm` then `matrixVectorOp`) and runs the L2 eps
neighborhood against `2 * eps`, because for unit rows
`|a - b|^2 = 2 - 2 cos(a, b) = 2 * cosine_distance(a, b)`.

Here the scaling is HOST code that both bindings compile from this one file,
before anything reaches a device, so the unit rows carry the same bits on
every column: the squared norm is one ascending fold of pinned products
(`identical_mul`), the root `identical_sqrt`, each quotient `identical_div`,
every intermediate flushed (`ftz`).

A row whose norm is zero has no direction, and its cosine distance is
undefined: cuML divides by zero and carries NaN into the kernel. It is
refused BY NAME. A non-finite row is refused as well.
"""

from std.math import isfinite

from checks.numerics import ftz, identical_div, identical_mul, identical_sqrt


def cosine_unit_rows(
    src: List[Float32], n_rows: Int, n_cols: Int, what: String
) raises -> List[Float32]:
    """`src` (n_rows x n_cols, row-major) with every row scaled to unit L2
    length. `what` names the caller in the refusal."""
    var out = List[Float32](length=n_rows * n_cols, fill=Float32(0))
    for i in range(n_rows):
        var base = i * n_cols
        var n2 = Float32(0)
        for f in range(n_cols):
            var v = ftz(src[base + f])
            n2 = ftz(n2 + ftz(identical_mul(v, v)))
        if not (n2 > Float32(0)) or not isfinite(n2):
            raise Error(
                what + ": metric='cosine' needs every row to have a positive,"
                " finite norm; row " + String(i) + " has squared norm "
                + String(n2) + ". The cosine distance of a zero row is"
                " undefined (cuML divides by zero and carries NaN), so it is"
                " refused by name"
            )
        var norm = identical_sqrt(n2)
        for f in range(n_cols):
            out[base + f] = ftz(identical_div(ftz(src[base + f]), norm))
    return out^
