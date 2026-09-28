# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA on the host: `x_ann/cagra_device.mojo`'s launches as loops over the
SAME cells, in the same order."""

from x_ann.tsne_core import ts_knn_cell
from x_ann.cagra_core import cagra_prune, cagra_reverse_merge, cg_search_cell
from x_ann.host.ivf_pq_host import fp, ip


def cagra_build_host(x_in: List[Float32], n: Int, d: Int, kdeg: Int, deg: Int) raises -> List[Int32]:
    var x = x_in.copy()
    var nd = List[Float32](length=n * kdeg, fill=Float32(0.0))
    var ni = List[Int32](length=n * kdeg, fill=Int32(0))
    for i in range(n):
        ts_knn_cell(i, fp(x), n, d, kdeg, fp(nd), ip(ni))
    var pruned = cagra_prune(n, kdeg, ni, deg)
    _ = x^
    _ = nd^
    return cagra_reverse_merge(n, deg, pruned)


def cagra_search_host(
    x_in: List[Float32], n: Int, d: Int, graph_in: List[Int32], deg: Int, queries_in: List[Float32], m: Int,
    k: Int, L: Int, width: Int, max_iter: Int, n_seeds: Int,
    mut out_d: List[Float32], mut out_i: List[Int32],
) raises:
    var words = (n + 31) // 32
    var x = x_in.copy()
    var graph = graph_in.copy()
    var queries = queries_in.copy()
    var bd = List[Float32](length=m * L, fill=Float32(0.0))
    var bi = List[Int32](length=m * L, fill=Int32(0))
    var bx = List[Int32](length=m * L, fill=Int32(0))
    var vis = List[Int32](length=m * words, fill=Int32(0))
    out_d = List[Float32](length=m * k, fill=Float32(0.0))
    out_i = List[Int32](length=m * k, fill=Int32(0))
    for q in range(m):
        cg_search_cell(q, fp(queries), fp(x), n, d, ip(graph), deg, k, L, width, max_iter, n_seeds,
                       fp(bd), ip(bi), ip(bx), ip(vis), words, fp(out_d), ip(out_i), q)
    _ = x^
    _ = graph^
    _ = queries^
