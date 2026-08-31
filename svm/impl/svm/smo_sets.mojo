# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`in_upper` / `in_lower`: which KKT set a training vector is in.

PORT OF `cuml/cpp/src/svm/smo_sets.cuh` at cuML v26.08.00, both functions
character for character. The comments are theirs: the long forms
`(0 < a && a < C) || (y == 1 && a == 0) || (y == -1 && a == C)` collapse to
the two-clause forms because `a` is always clipped into `[0, C]`.

Host and device: these are called from the block solve, from `ws_util`'s
flag kernels, from `results`, and from the host oracle, and there is ONE
spelling so that every caller agrees about the sets.
"""


@always_inline
def in_upper(a: Float32, y: Float32, C: Float32) -> Bool:
    """`(y < 0 && a > 0) || (y > 0 && a < C)`."""
    return (y < Float32(0.0) and a > Float32(0.0)) or (
        y > Float32(0.0) and a < C
    )


@always_inline
def in_lower(a: Float32, y: Float32, C: Float32) -> Bool:
    """`(y < 0 && a < C) || (y > 0 && a > 0)`."""
    return (y < Float32(0.0) and a < C) or (
        y > Float32(0.0) and a > Float32(0.0)
    )
