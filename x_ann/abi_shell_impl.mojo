# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed checks and sizes the x_ann bindings ran inline (lane thin-core,
2026-10-21: moved word for word, no Python import, no bit change)."""
from x_ann.ivf_pq_core import I32P


def rabitq_padded_dim(dim: Int) -> Int:
    """The smallest power of two >= dim (1 when dim <= 1), the RaBitQ code width."""
    var D = 1
    while D < dim:
        D *= 2
    return D


def cagra_check_graph(g: I32P, n: Int, deg: Int) raises:
    """Refuse a graph that names a row outside the n-row dataset."""
    for e in range(n * deg):
        if Int(g.unsafe_load(e)) < 0 or Int(g.unsafe_load(e)) >= n:
            raise Error("CAGRA search: the graph names a row outside the dataset")
