# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Clustering-score units (scikit-learn 1.9 sklearn/metrics/cluster/_unsupervised.py):
the distance of each row to its cluster's centroid, for the Calinski-Harabasz
within-cluster dispersion (squared) and the Davies-Bouldin intra-cluster
distance (its root). The centroids themselves are group.mojo's per-cluster
PairSums, divided on the host."""
from x_metrics.common import FP, IP, PairSum, p, ld, st, ldi
from checks.numerics import ftz, identical_mul, identical_sqrt


def row_centroid_dist_unit(t: Int, f: FP, q: IP):
    """q = [X, d, LAB, C, OUT, root]; t = row. OUT[t] = sum over columns,
    ascending, of (X[t, c] - C[LAB[t], c])^2 (PairSum), its square root when
    root = 1 (correctly rounded, IDENTITY_PATHS row 10's pin)."""
    var X = p(q, 0)
    var d = p(q, 1)
    var g = ldi(f, p(q, 2) + t)
    var C = p(q, 3) + g * d
    var acc = PairSum()
    for c in range(d):
        var v = ftz(ld(f, X + t * d + c) - ld(f, C + c))
        acc.add(identical_mul(v, v))
    var r = acc.result()
    if p(q, 5) == 1:
        r = identical_sqrt(r)
    st(f, p(q, 4) + t, r)
