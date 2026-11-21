# SPDX-License-Identifier: Apache-2.0
"""The fit operations of the CPU column of the ANN operations table
(core/neighbors_entry_table_ann.mojo, operations 10 and 13 to 15; lane
dw-neighbors, 2026-11-21): the typed bodies of the host binding's build
exports (bindings/_mojolearn_x_ann_host.mojo `x_ann_ivf_pq_build`,
`x_ann_ivf_sq_build`, `x_ann_ivf_rabitq_build`, `x_ann_cagra_build`), moved
here whole: the same host drivers on the same words, so the bits are
unchanged. The exports now UNPACK their lists and call these bodies; the
table entry calls them too. Nothing here touches Python.

Called with the GIL RELEASED around the whole body. The exports released it
around the build only: the row copy in and the copies out of the result
lists ran with it held and no longer do (they read and write the caller's
pinned arrays, which the door keeps alive through the call). The host
drivers' signatures (a row list in, result lists out) are not changed here.

The CPU column keeps no resident index: operations 11 and 12 are refused by
the dispatcher (x_ann/ops_entry_host.mojo).

Each build is one call of a host driver that is parallel inside; a fit is
not split further here.
"""
from core.hostptr import f32_ptr, read_f32
from x_ann.abi import out_f32_to, out_i32_to
from x_ann.ivf_pq_core import F32P
from x_ann.host.cagra_host import cagra_build_host
from x_ann.host.ivf_pq_host import ivf_pq_build_host, ivf_sq_build_host, ivf_rabitq_build_host


def ann_ivf_pq_build_host_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_pq_build`: a x, centers, offsets, list_indices, codebooks,
    codes; p n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits,
    pq_kmeans_n_iters. Returns n."""
    if len(a) != 6 or len(p) != 8:
        raise Error("x_ann_ivf_pq_build: need 6 addresses and 8 parameters")
    var n = p[0]
    var dim = p[1]
    var x = read_f32(a[0], n * dim)
    var index = ivf_pq_build_host(x, n, dim, p[2], p[3], p[4], p[5], p[6], p[7])
    out_f32_to(index.centers, a[1])
    out_i32_to(index.offsets, a[2])
    out_i32_to(index.list_indices, a[3])
    out_f32_to(index.codebooks, a[4])
    out_i32_to(index.codes, a[5])
    return n


def ann_ivf_sq_build_host_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_sq_build`: a x, centers, offsets, list_indices, vmin,
    delta, codes; p n, dim, n_lists, kmeans_n_iters, seed[,
    train_rows_per_list] (0 or absent keeps the default training rows).
    Returns n."""
    if len(a) != 7 or len(p) < 5:
        raise Error("x_ann_ivf_sq_build: need 7 addresses and 5 or 6 parameters")
    var n = p[0]
    var dim = p[1]
    var train_rows_per_list = p[5] if len(p) > 5 else 0
    var x = read_f32(a[0], n * dim)
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var vmin = List[Float32]()
    var delta = List[Float32]()
    var codes = List[Int32]()
    ivf_sq_build_host(
        x, n, dim, p[2], p[3], p[4], centers, offsets, list_indices, vmin, delta, codes,
        train_rows_per_list=train_rows_per_list,
    )
    out_f32_to(centers, a[1])
    out_i32_to(offsets, a[2])
    out_i32_to(list_indices, a[3])
    out_f32_to(vmin, a[4])
    out_f32_to(delta, a[5])
    out_i32_to(codes, a[6])
    return n


def ann_ivf_rabitq_build_host_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_rabitq_build`: a x, centers, offsets, list_indices, codes
    (n x words int32 bit words), norms, ips; p n, dim, n_lists,
    kmeans_n_iters, seed. Returns n."""
    if len(a) != 7 or len(p) != 5:
        raise Error("x_ann_ivf_rabitq_build: need 7 addresses and 5 parameters")
    var n = p[0]
    var dim = p[1]
    var x = read_f32(a[0], n * dim)
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var codes = List[Int32]()
    var norms = List[Float32]()
    var ips = List[Float32]()
    ivf_rabitq_build_host(x, n, dim, p[2], p[3], p[4], centers, offsets, list_indices, codes, norms, ips)
    out_f32_to(centers, a[1])
    out_i32_to(offsets, a[2])
    out_i32_to(list_indices, a[3])
    out_i32_to(codes, a[4])
    out_f32_to(norms, a[5])
    out_f32_to(ips, a[6])
    return n


def ann_cagra_build_host_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_cagra_build`: a x, graph_out; p n, d,
    intermediate_graph_degree, graph_degree. Returns n. The rows are read
    in place."""
    if len(a) != 2 or len(p) != 4:
        raise Error("x_ann_cagra_build: need 2 addresses and 4 parameters")
    var n = p[0]
    var d = p[1]
    var kdeg = p[2]
    var deg = p[3]
    if n < 2 or d <= 0 or kdeg < 1 or kdeg > n - 1 or deg < 1 or deg > kdeg:
        raise Error("CAGRA: need 1 <= graph_degree <= intermediate_graph_degree <= n - 1")
    var x = rebind[F32P](f32_ptr(a[0]))
    var g = cagra_build_host(x, n, d, kdeg, deg)
    out_i32_to(g, a[1])
    return n
