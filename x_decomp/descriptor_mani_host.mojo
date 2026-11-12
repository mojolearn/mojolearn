# SPDX-License-Identifier: Apache-2.0
# The manifold kit's table operations on the host image (lane entangled-mkit,
# 2026-11-12): the bodies of the host binding's `x_decomp_lle_iterate`,
# `_mds_fit`, `_lanczos`, `_eigh_tridiag` and `_graph_*` doors
# (providers/python/x_decomp_{lle_iter,mds,lanczos_host,graph_host}_pyapi.mojo)
# on typed words, with the same checks; the shared ones in
# x_decomp/descriptor_mani_ops.mojo on HostExec.
from x_decomp.host import HostExec
from x_decomp.cells import F32Ptr
from x_decomp.lle_iter import LleArgs, lle_iterate_run
from x_decomp.mds import MdsArgs, mds_fit_check, mds_fit_run
from x_decomp.lanczos_host import eigh_tridiag_host_run, lanczos_run
from x_decomp.graph_host import (
    graph_dijkstra_run, graph_join_run, graph_knn_dense_run, graph_knn_run, graph_lle_iw_run,
    graph_radius_geo_run, graph_radius_run, host_components,
)
from x_decomp.descriptor_ops import xd_f32, xd_float
from x_decomp.descriptor_schema import xd_dim
from x_decomp.descriptor_mani_ops import xd_mani_shared_op, xd_is_mani_shared_op, xd_f64_out, xd_graph_dim
from x_decomp.descriptor_mani_schema import (
    XD_LLE_ITERATE, XD_MDS_FIT, XD_LANCZOS, XD_EIGH_TRIDIAG, XD_GRAPH_KNN, XD_GRAPH_KNN_DENSE, XD_GRAPH_RADIUS,
    XD_GRAPH_RADIUS_GEO, XD_GRAPH_LLE_IW, XD_GRAPH_COMPONENTS, XD_GRAPH_JOIN, XD_GRAPH_DIJKSTRA,
)


def xd_lle_args(p: List[Int], floats: Int) raises -> LleArgs:
    """`_lle_args`: p = [n, n1, nc, p, max_iter, seed]; f = [floor, sub_tol, stall_tol]."""
    return LleArgs(xd_dim(p[0]), xd_dim(p[1]), xd_dim(p[2]), xd_dim(p[3]), p[4], p[5], xd_float(floats, 0),
                   xd_float(floats, 1), xd_float(floats, 2))


def xd_mds_args(p: List[Int], floats: Int) raises -> MdsArgs:
    """`_mds_args`: p = [n, nc, n_starts, max_iter, metric, norm]; f = [eps]."""
    return MdsArgs(p[3], p[4] != 0, p[5] != 0, xd_float(floats, 0))


def xd_mani_host_op(op: Int, a: List[Int], p: List[Int], floats: Int) raises -> Int:
    if xd_is_mani_shared_op(op):
        return xd_mani_shared_op[HostExec](op, a, p, floats)
    if op == XD_LLE_ITERATE:
        var args = xd_lle_args(p, floats)
        var e = 0.0
        var st = lle_iterate_run[HostExec](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f32(a[3]), xd_f32(a[4]),
                                           xd_f32(a[5]), xd_f32(a[6]), xd_f32(a[7]), args, e)
        xd_f64_out(a[8])[0] = e
        return st
    if op == XD_MDS_FIT:
        var n = xd_dim(p[0])
        var nc = xd_dim(p[1])
        var ns = xd_dim(p[2])
        mds_fit_check(n, nc, ns)
        var args = xd_mds_args(p, floats)
        var st = 0.0
        var it = mds_fit_run[HostExec](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), n, nc, ns, args, st)
        xd_f64_out(a[3])[0] = st
        return it
    if op == XD_LANCZOS:
        # lanczos_py: the words unchecked, as the door read them
        return lanczos_run(F32Ptr(unsafe_from_address=a[0]), F32Ptr(unsafe_from_address=a[1]),
                           F32Ptr(unsafe_from_address=a[2]), p[0], p[1], p[2], p[3])
    if op == XD_EIGH_TRIDIAG:
        return eigh_tridiag_host_run(F32Ptr(unsafe_from_address=a[0]), F32Ptr(unsafe_from_address=a[1]),
                                     F32Ptr(unsafe_from_address=a[2]), p[0], p[1])
    if op == XD_GRAPH_KNN:
        var n = xd_graph_dim(p[0])
        var m = xd_graph_dim(p[1])
        var nn = xd_graph_dim(p[2])
        graph_knn_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), n, m, nn, p[3])
        return n
    if op == XD_GRAPH_KNN_DENSE:
        var n = xd_graph_dim(p[0])
        graph_knn_dense_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), n, xd_graph_dim(p[1]))
        return n
    if op == XD_GRAPH_RADIUS:
        var n = xd_graph_dim(p[0])
        graph_radius_run(xd_f32(a[0]), xd_f32(a[1]), n, xd_float(floats, 0))
        return n
    if op == XD_GRAPH_RADIUS_GEO:
        var nq = xd_graph_dim(p[0])
        graph_radius_geo_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), nq, xd_graph_dim(p[1]), xd_float(floats, 0))
        return nq
    if op == XD_GRAPH_LLE_IW:
        var n = xd_graph_dim(p[0])
        graph_lle_iw_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), n, xd_graph_dim(p[1]))
        return n
    if op == XD_GRAPH_COMPONENTS:
        var n = xd_graph_dim(p[0])
        return host_components(xd_f32(a[0]), n, xd_f32(a[1]))
    if op == XD_GRAPH_JOIN:
        var n = xd_graph_dim(p[0])
        var count = xd_graph_dim(p[1])
        graph_join_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), n, count)
        return count
    if op == XD_GRAPH_DIJKSTRA:
        var n = xd_graph_dim(p[0])
        graph_dijkstra_run(xd_f32(a[0]), xd_f32(a[1]), n)
        return n
    raise Error("x_decomp native ABI: the host binding has no manifold operation " + String(op))
