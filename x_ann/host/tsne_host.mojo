# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE on the host: `x_ann/tsne_device.mojo`'s launches and schedule as
loops over the SAME cells (`x_ann/tsne_core.mojo`), in the same order."""

from checks.numerics import ftz, identical_log, identical_sqrt
from x_ann.tsne_core import (
    F32P, I32P, ts_fold, ts_kl_cell, ts_knn_cell, ts_perplexity_cell, ts_repulse_cell, ts_sq_fold,
    ts_step_cell, ts_sum_cell, tsne_nn, tsne_symmetrize, tsne_validate,
)
from x_ann.host.ivf_pq_host import fp, ip

comptime TS_CHECK_EVERY = 50
comptime TS_EXPLORATION_PATIENCE = 250


def tsne_fit_host(
    x_in: List[Float32], n: Int, d: Int, nc: Int, y0: List[Float32], perplexity: Float32, exaggeration: Float32,
    learning_rate: Float32, max_iter: Int, exploration: Int, exact: Bool, n_iter_without_progress: Int,
    min_grad_norm: Float32, mut y_out: List[Float32], mut kl_out: Float32, mut n_iter_out: Int,
) raises:
    tsne_validate(n, d, nc, perplexity, max_iter, exploration)
    var nn = tsne_nn(n, perplexity, exact)
    var dof = nc - 1 if nc > 1 else 1
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
    var upd = List[Float32](length=nc * n, fill=Float32(0.0))
    var gains = List[Float32](length=nc * n, fill=Float32(1.0))
    var gbuf = List[Float32](length=nc * n, fill=Float32(0.0))
    var rz = List[Float32](length=n, fill=Float32(0.0))
    var rep = List[Float32](length=nc * n, fill=Float32(0.0))
    var z = List[Float32](length=1, fill=Float32(0.0))
    var kl = List[Float32](length=n, fill=Float32(0.0))
    var it = 0
    var last = -1
    var in_a = True
    var kl_last = Float32(0.0)
    for phase in range(2):
        var end = exploration if phase == 0 else max_iter
        if it >= end:
            continue
        if phase == 1:
            upd = List[Float32](length=nc * n, fill=Float32(0.0))
            gains = List[Float32](length=nc * n, fill=Float32(1.0))
        var ex = exaggeration if phase == 0 else Float32(1.0)
        var mom = Float32(0.5) if phase == 0 else Float32(0.8)
        var patience = TS_EXPLORATION_PATIENCE if phase == 0 else n_iter_without_progress
        var best_err = Float32(3.4028235e38)
        var best_iter = it
        for i in range(it, end):
            var check = (i + 1) % TS_CHECK_EVERY == 0 or i == end - 1
            var ycur = fp(ya) if in_a else fp(yb)
            var ynext = fp(yb) if in_a else fp(ya)
            for e in range(nc * n):
                ts_repulse_cell(e, ycur, n, nc, dof, fp(rz), fp(rep))
            ts_sum_cell(fp(rz), n, fp(z))
            var err = Float32(0.0)
            if check:
                for r in range(n):
                    ts_kl_cell(r, ycur, nc, dof, ip(indptr), ip(indices), fp(values), fp(z), ex, fp(kl))
                err = ts_fold(kl)
            for e in range(nc * n):
                ts_step_cell(e, ycur, ynext, nc, dof, ip(indptr), ip(indices), fp(values), fp(rep), fp(z),
                             fp(upd), fp(gains), fp(gbuf), ex, mom, learning_rate)
            var gnorm = Float32(0.0)
            if check:
                gnorm = ftz(identical_sqrt(ts_sq_fold(gbuf)))
            in_a = not in_a
            last = i
            if check:
                kl_last = err
            if (i + 1) % TS_CHECK_EVERY == 0:
                if err < best_err:
                    best_err = err
                    best_iter = i
                elif i - best_iter > patience:
                    break
                if gnorm <= min_grad_norm:
                    break
        it = last + 1
    if in_a:
        y_out = ya.copy()
    else:
        y_out = yb.copy()
    kl_out = kl_last
    n_iter_out = last
    _ = x^
    _ = nn_d^
