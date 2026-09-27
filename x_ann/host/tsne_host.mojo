# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE on the host: `x_ann/tsne_device.mojo`'s launches as loops over the
SAME cells (`x_ann/tsne_core.mojo`), in the same order."""

from checks.numerics import ftz, identical_log
from x_ann.tsne_core import (
    F32P, I32P, ts_kl_cell, ts_knn_cell, ts_perplexity_cell, ts_repulse_cell, ts_step_cell,
    ts_sum_cell, tsne_nn, tsne_symmetrize, tsne_validate,
)
from x_ann.host.ivf_pq_host import fp, ip


def tsne_fit_host(
    x_in: List[Float32], n: Int, d: Int, y0: List[Float32], perplexity: Float32, exaggeration: Float32,
    learning_rate: Float32, max_iter: Int, exploration: Int, mut y_out: List[Float32], mut kl_out: Float32,
) raises:
    tsne_validate(n, d, perplexity, max_iter, exploration)
    var nn = tsne_nn(n, perplexity)
    var x = x_in.copy()
    var nn_d = List[Float32](length=n * nn, fill=Float32(0.0))
    var nn_i = List[Int32](length=n * nn, fill=Int32(0))
    var p_cond = List[Float32](length=n * nn, fill=Float32(0.0))
    var log_perp = identical_log(perplexity)
    for i in range(n):
        ts_knn_cell(i, fp(x), n, d, nn, fp(nn_d), ip(nn_i))
    for i in range(n):
        ts_perplexity_cell(i, fp(nn_d), nn, log_perp, fp(p_cond))
    var indptr = List[Int32]()
    var indices = List[Int32]()
    var values = List[Float32]()
    tsne_symmetrize(n, nn, nn_i, p_cond, indptr, indices, values)

    var ya = y0.copy()
    var yb = y0.copy()
    var upd = List[Float32](length=2 * n, fill=Float32(0.0))
    var gains = List[Float32](length=2 * n, fill=Float32(1.0))
    var rz = List[Float32](length=n, fill=Float32(0.0))
    var rep = List[Float32](length=2 * n, fill=Float32(0.0))
    var z = List[Float32](length=1, fill=Float32(0.0))
    var kl = List[Float32](length=n, fill=Float32(0.0))
    for it in range(max_iter):
        var ex = exaggeration if it < exploration else Float32(1.0)
        var mom = Float32(0.5) if it < exploration else Float32(0.8)
        var ycur = fp(ya) if it % 2 == 0 else fp(yb)
        var ynext = fp(yb) if it % 2 == 0 else fp(ya)
        for i in range(n):
            ts_repulse_cell(i, ycur, n, fp(rz), fp(rep))
        ts_sum_cell(fp(rz), n, fp(z))
        for e in range(2 * n):
            ts_step_cell(e, ycur, ynext, ip(indptr), ip(indices), fp(values), fp(rep), fp(z),
                         fp(upd), fp(gains), ex, mom, learning_rate)
    var yfin = fp(ya) if max_iter % 2 == 0 else fp(yb)
    for i in range(n):
        ts_repulse_cell(i, yfin, n, fp(rz), fp(rep))
    ts_sum_cell(fp(rz), n, fp(z))
    for i in range(n):
        ts_kl_cell(i, yfin, ip(indptr), ip(indices), fp(values), fp(z), fp(kl))
    if max_iter % 2 == 0:
        y_out = ya.copy()
    else:
        y_out = yb.copy()
    var total = Float32(0.0)
    for i in range(n):
        total = total + kl[i]
    kl_out = total
    _ = x^
    _ = nn_d^
