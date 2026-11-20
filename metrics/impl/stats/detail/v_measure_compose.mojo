# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The V-measure from a homogeneity and a completeness (RAFT
`v_measure.cuh` :27-50, scikit-learn `homogeneity_completeness_v_measure`):
`0.0` when both are zero, else `(1 + beta) h c / (beta h + c)`. Host
Float64, spelled exactly as `v_measure` and the host oracle spelled it (two
multiplies, a multiply then an add, a division), so every caller that
composes it from the same h and c gets the same bits. No GPU imports: the
device and the host column both use it."""


def v_measure_from(homogeneity: Float64, completeness: Float64, beta: Float64) -> Float64:
    if completeness + homogeneity == 0.0:
        return 0.0
    var num = (1.0 + beta) * homogeneity * completeness
    var den = beta * homogeneity + completeness
    return num / den
