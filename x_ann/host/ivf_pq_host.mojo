# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ on the host: `x_ann/ivf_pq_device.mojo`'s launches as loops over
the SAME cell functions (`x_ann/ivf_pq_core.mojo`), in the same launch order.
The coarse quantizer is the same Lloyd cells over whole rows."""

from std.sys.compile import is_defined

from cluster.host.kmeans_oracle import host_kmeans_fit
from cluster.impl.kmeans_params import INIT_KMEANS_PLUS_PLUS, METRIC_L2_EXPANDED
from ivf.host.ivf_host import host_ivf_build
from x_ann.refine_core import refine_cell
from x_ann.ivf_rabitq_core import rq_encode_cell, rq_pow2, rq_scale, rq_search_cell
from x_ann.ivf_sq_core import sq_encode_cell, sq_range_cell, sq_search_cell
from x_ann.ivf_pq_core import (
    F32P, I32P, IvfPqIndex, pq_assign_cell, pq_labels_from_lists,
    pq_len_of, pq_residual_cell, pq_search_cell, pq_validate,
)

comptime X_ANN_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def fp(mut l: List[Float32]) -> F32P:
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def ip(mut l: List[Int32]) -> I32P:
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _coarse_host(
    x: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    mut centers: List[Float32], mut offsets: List[Int32], mut list_indices: List[Int32],
    mut labels: List[Int32],
) raises:
    """`ivf/host/ivf_host.mojo::host_ivf_build`, the host twin of the device
    driver's IVF-Flat coarse build."""
    var flat = host_ivf_build(x, n, dim, n_lists, kmeans_n_iters, METRIC_L2_EXPANDED, UInt64(seed))
    centers = flat.centers.copy()
    offsets = flat.offsets.copy()
    list_indices = List[Int32](capacity=n)
    for s in range(n):
        list_indices.append(Int32(Int(flat.list_indices[s])))
    labels = pq_labels_from_lists(offsets, list_indices, n_lists, n)


def _codebooks_host(
    r: List[Float32], n: Int, rot_dim: Int, pq_dim: Int, pq_len: Int, n_codes: Int, pq_iters: Int, seed: Int,
) raises -> List[Float32]:
    """`host_kmeans_fit` per subspace, the twin of the device `kmeans_fit`."""
    var codebooks = List[Float32](capacity=pq_dim * n_codes * pq_len)
    for j in range(pq_dim):
        var sub = List[Float32](capacity=n * pq_len)
        for i in range(n):
            for t in range(pq_len):
                sub.append(r[i * rot_dim + j * pq_len + t])
        var cb = List[Float32](length=n_codes * pq_len, fill=Float32(0.0))
        var lab = List[UInt32](length=n, fill=UInt32(0))
        _ = host_kmeans_fit(
            sub, n, pq_len, n_codes, cb, lab, List[Float32](), 0, pq_iters, Float64(1e-4), UInt64(seed), 1,
            INIT_KMEANS_PLUS_PLUS, METRIC_L2_EXPANDED, Float64(2.0),
        )
        for e in range(n_codes * pq_len):
            codebooks.append(cb[e])
    return codebooks^


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
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var labels = List[Int32]()
    _coarse_host(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels)
    var r = List[Float32](length=n * rot_dim, fill=Float32(0.0))
    for e in range(n * rot_dim):
        pq_residual_cell(e, fp(x), fp(centers), ip(labels), dim, rot_dim, fp(r))
    var cb = _codebooks_host(r, n, rot_dim, pq_dim, pq_len, n_codes, pq_iters, seed)
    var codes = List[Int32](length=n * pq_dim, fill=Int32(0))
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
    pq_validate(n, dim, n_lists, 1, 1, 1)
    var x = x_in.copy()
    var labels = List[Int32]()
    _coarse_host(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels)
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
    pq_validate(n, dim, n_lists, 1, 1, 1)
    var D = rq_pow2(dim)
    var words = (D + 31) // 32
    var scale = rq_scale(D)
    var x = x_in.copy()
    var labels = List[Int32]()
    _coarse_host(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels)
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
