# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ on the host: `x_ann/ivf_pq_device.mojo`'s launches as loops over
the SAME cell functions (`x_ann/ivf_pq_core.mojo`), in the same launch order.
The coarse quantizer is the same Lloyd cells over whole rows."""

from std.sys.compile import is_defined

from x_ann.refine_core import refine_cell
from x_ann.ivf_rabitq_core import rq_encode_cell, rq_pow2, rq_scale, rq_search_cell
from x_ann.ivf_sq_core import sq_encode_cell, sq_range_cell, sq_search_cell
from x_ann.ivf_pq_core import (
    F32P, I32P, IvfPqIndex, pq_assign_cell, pq_init_cell, pq_lists_from_labels,
    pq_len_of, pq_residual_cell, pq_search_cell, pq_update_cell, pq_validate,
)

comptime X_ANN_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def fp(mut l: List[Float32]) -> F32P:
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def ip(mut l: List[Int32]) -> I32P:
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _coarse_host(
    mut x: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    mut centers: List[Float32], mut labels: List[Int32],
):
    centers = List[Float32](length=n_lists * dim, fill=Float32(0.0))
    labels = List[Int32](length=n, fill=Int32(0))
    for e in range(n_lists * dim):
        pq_init_cell(e, fp(x), n, dim, dim, n_lists, seed, fp(centers))
    for _ in range(kmeans_n_iters):
        for e in range(n):
            pq_assign_cell(e, fp(x), fp(centers), 1, dim, dim, n_lists, ip(labels))
        for e in range(n_lists * dim):
            pq_update_cell(e, fp(x), ip(labels), n, 1, dim, dim, n_lists, fp(centers))
    for e in range(n):
        pq_assign_cell(e, fp(x), fp(centers), 1, dim, dim, n_lists, ip(labels))


def ivf_pq_build_host(
    x_in: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    pq_dim: Int, pq_bits: Int, pq_iters: Int,
) raises -> IvfPqIndex:
    pq_validate(n, dim, n_lists, pq_dim, pq_bits, pq_iters)
    var pq_len = pq_len_of(dim, pq_dim)
    var rot_dim = pq_len * pq_dim
    var n_codes = 1 << pq_bits
    var x = x_in.copy()
    var centers = List[Float32]()
    var labels = List[Int32]()
    _coarse_host(x, n, dim, n_lists, kmeans_n_iters, seed, centers, labels)
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    pq_lists_from_labels(labels, n, n_lists, offsets, list_indices)

    var r = List[Float32](length=n * rot_dim, fill=Float32(0.0))
    var cb_count = pq_dim * n_codes * pq_len
    var cb = List[Float32](length=cb_count, fill=Float32(0.0))
    var codes = List[Int32](length=n * pq_dim, fill=Int32(0))
    for e in range(n * rot_dim):
        pq_residual_cell(e, fp(x), fp(centers), ip(labels), dim, rot_dim, fp(r))
    for e in range(cb_count):
        pq_init_cell(e, fp(r), n, rot_dim, pq_len, n_codes, seed, fp(cb))
    for _ in range(pq_iters):
        for e in range(n * pq_dim):
            pq_assign_cell(e, fp(r), fp(cb), pq_dim, rot_dim, pq_len, n_codes, ip(codes))
        for e in range(cb_count):
            pq_update_cell(e, fp(r), ip(codes), n, pq_dim, rot_dim, pq_len, n_codes, fp(cb))
    for e in range(n * pq_dim):
        pq_assign_cell(e, fp(r), fp(cb), pq_dim, rot_dim, pq_len, n_codes, ip(codes))
    comptime if X_ANN_HOST_SABOTAGE:
        codes[0] = Int32((Int(codes[0]) + 1) % n_codes)
    _ = x^
    _ = r^
    return IvfPqIndex(n_lists, dim, n, pq_dim, pq_len, n_codes, centers^, offsets^, list_indices^, cb^, codes^)


def ivf_pq_search_host(
    centers_in: List[Float32], offsets_in: List[Int32], list_indices_in: List[Int32],
    codebooks_in: List[Float32], codes_in: List[Int32], mask_in: List[Int32], n_lists: Int, dim: Int, pq_dim: Int,
    pq_bits: Int, queries_in: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
) raises:
    var pq_len = pq_len_of(dim, pq_dim)
    var n_codes = 1 << pq_bits
    var centers = centers_in.copy()
    var offsets = offsets_in.copy()
    var list_indices = list_indices_in.copy()
    var cb = codebooks_in.copy()
    var codes = codes_in.copy()
    var queries = queries_in.copy()
    var mask = mask_in.copy()
    out_d = List[Float32](length=m * k, fill=Float32(0.0))
    out_i = List[Int32](length=m * k, fill=Int32(0))
    out_n = List[Int32](length=m, fill=Int32(0))
    for q in range(m):
        pq_search_cell(
            q, fp(queries), dim, fp(centers), n_lists, ip(offsets), ip(list_indices), ip(codes),
            fp(cb), pq_dim, pq_len, n_codes, k, n_probes, ip(mask), fp(out_d), ip(out_i), ip(out_n),
        )
    _ = centers^
    _ = offsets^
    _ = list_indices^
    _ = cb^
    _ = codes^
    _ = queries^


def ivf_sq_build_host(
    x_in: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    mut centers: List[Float32], mut offsets: List[Int32], mut list_indices: List[Int32],
    mut vmin: List[Float32], mut delta: List[Float32], mut codes: List[Int32],
) raises:
    pq_validate(n, dim, n_lists, 1, 1, 0)
    var x = x_in.copy()
    var labels = List[Int32]()
    _coarse_host(x, n, dim, n_lists, kmeans_n_iters, seed, centers, labels)
    pq_lists_from_labels(labels, n, n_lists, offsets, list_indices)
    var r = List[Float32](length=n * dim, fill=Float32(0.0))
    for e in range(n * dim):
        pq_residual_cell(e, fp(x), fp(centers), ip(labels), dim, dim, fp(r))
    vmin = List[Float32](length=dim, fill=Float32(0.0))
    delta = List[Float32](length=dim, fill=Float32(0.0))
    codes = List[Int32](length=n * dim, fill=Int32(0))
    for c in range(dim):
        sq_range_cell(c, fp(r), n, dim, fp(vmin), fp(delta))
    for e in range(n * dim):
        sq_encode_cell(e, fp(r), dim, fp(vmin), fp(delta), ip(codes))
    _ = x^
    _ = r^


def ivf_sq_search_host(
    centers_in: List[Float32], offsets_in: List[Int32], list_indices_in: List[Int32], vmin_in: List[Float32],
    delta_in: List[Float32], codes_in: List[Int32], mask_in: List[Int32], n_lists: Int, dim: Int,
    queries_in: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
) raises:
    var centers = centers_in.copy()
    var offsets = offsets_in.copy()
    var list_indices = list_indices_in.copy()
    var vmin = vmin_in.copy()
    var delta = delta_in.copy()
    var codes = codes_in.copy()
    var mask = mask_in.copy()
    var queries = queries_in.copy()
    out_d = List[Float32](length=m * k, fill=Float32(0.0))
    out_i = List[Int32](length=m * k, fill=Int32(0))
    out_n = List[Int32](length=m, fill=Int32(0))
    for q in range(m):
        sq_search_cell(q, fp(queries), dim, fp(centers), n_lists, ip(offsets), ip(list_indices), ip(codes),
                       fp(vmin), fp(delta), k, n_probes, ip(mask), fp(out_d), ip(out_i), ip(out_n))
    _ = centers^
    _ = offsets^
    _ = list_indices^
    _ = vmin^
    _ = delta^
    _ = codes^
    _ = mask^
    _ = queries^


def refine_host(
    x_in: List[Float32], n: Int, d: Int, queries_in: List[Float32], m: Int, cand_in: List[Int32], k0: Int, k: Int,
    mut out_d: List[Float32], mut out_i: List[Int32],
) raises:
    var x = x_in.copy()
    var queries = queries_in.copy()
    var cand = cand_in.copy()
    out_d = List[Float32](length=m * k, fill=Float32(0.0))
    out_i = List[Int32](length=m * k, fill=Int32(0))
    for q in range(m):
        refine_cell(q, fp(x), n, d, fp(queries), ip(cand), k0, k, fp(out_d), ip(out_i))
    _ = x^
    _ = queries^
    _ = cand^


def ivf_rabitq_build_host(
    x_in: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    mut centers: List[Float32], mut offsets: List[Int32], mut list_indices: List[Int32],
    mut codes: List[Int32], mut norms: List[Float32], mut ips: List[Float32],
) raises:
    pq_validate(n, dim, n_lists, 1, 1, 0)
    var D = rq_pow2(dim)
    var words = (D + 31) // 32
    var scale = rq_scale(D)
    var x = x_in.copy()
    var labels = List[Int32]()
    _coarse_host(x, n, dim, n_lists, kmeans_n_iters, seed, centers, labels)
    pq_lists_from_labels(labels, n, n_lists, offsets, list_indices)
    var ws = List[Float32](length=n * D, fill=Float32(0.0))
    codes = List[Int32](length=n * words, fill=Int32(0))
    norms = List[Float32](length=n, fill=Float32(0.0))
    ips = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        rq_encode_cell(i, fp(x), fp(centers), ip(labels), dim, D, seed, scale, fp(ws), words, ip(codes),
                       fp(norms), fp(ips))
    _ = x^
    _ = ws^


def ivf_rabitq_search_host(
    centers_in: List[Float32], offsets_in: List[Int32], list_indices_in: List[Int32], codes_in: List[Int32],
    norms_in: List[Float32], ips_in: List[Float32], mask_in: List[Int32], n_lists: Int, dim: Int, seed: Int,
    queries_in: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
) raises:
    var D = rq_pow2(dim)
    var words = (D + 31) // 32
    var scale = rq_scale(D)
    var centers = centers_in.copy()
    var offsets = offsets_in.copy()
    var list_indices = list_indices_in.copy()
    var codes = codes_in.copy()
    var norms = norms_in.copy()
    var ips = ips_in.copy()
    var mask = mask_in.copy()
    var queries = queries_in.copy()
    var ws = List[Float32](length=m * D, fill=Float32(0.0))
    out_d = List[Float32](length=m * k, fill=Float32(0.0))
    out_i = List[Int32](length=m * k, fill=Int32(0))
    out_n = List[Int32](length=m, fill=Int32(0))
    for q in range(m):
        rq_search_cell(q, fp(queries), dim, fp(centers), n_lists, ip(offsets), ip(list_indices), ip(codes),
                       fp(norms), fp(ips), D, words, seed, scale, k, n_probes, ip(mask), fp(ws), fp(out_d),
                       ip(out_i), ip(out_n))
    _ = centers^
    _ = offsets^
    _ = list_indices^
    _ = codes^
    _ = norms^
    _ = ips^
    _ = mask^
    _ = queries^
    _ = ws^
