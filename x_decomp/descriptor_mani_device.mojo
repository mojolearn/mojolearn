# SPDX-License-Identifier: Apache-2.0
# The manifold kit's table operations on the GPU image (lane entangled-mkit,
# 2026-11-12): the bodies of the GPU binding's `x_decomp_lle_iterate`
# (lle_iterate_dev_py), `_mds_fit` (mds_fit_dev_py), `_dev_trisolve`,
# `_dev_dsum_sq`, `_dev_lanczos`, `_dev_eigh_tridiag`, `_dev_graph_*` and the
# Apple FAST ordered LLE cells on typed words, with the same checks; the
# shared ones in x_decomp/descriptor_mani_ops.mojo on DevExec.
from gemm.afn_apple_fast import AFN_GEMM_APPLE
from x_decomp.device import DevExec
from x_decomp.resident import dev_trisolve_run
from x_decomp.select_dev import dev_dsum_sq_run
from x_decomp.lanczos_dev import dev_eigh_tridiag_run, dev_lanczos_run
from x_decomp.lle_iter_dev import lle_gemm_ordered_run, lle_iterate_dev_run, lle_lu_ordered_run
from x_decomp.mds import mds_fit_check
from x_decomp.mds_dev import mds_fit_dev_run
from x_decomp.graph_device import (
    dev_graph_components_run, dev_graph_dijkstra_run, dev_graph_join_run, dev_graph_knn_dense_run,
    dev_graph_knn_run, dev_graph_lle_iw_run, dev_graph_radius_geo_run, dev_graph_radius_run,
)
from x_decomp.descriptor_ops import xd_f32, xd_i32, xd_float
from x_decomp.descriptor_schema import xd_dim
from x_decomp.descriptor_mani_ops import xd_mani_shared_op, xd_is_mani_shared_op, xd_f64_out
from x_decomp.descriptor_mani_host import xd_lle_args, xd_mds_args
from x_decomp.descriptor_mani_schema import (
    XD_LLE_ITERATE, XD_MDS_FIT, XD_DEV_TRISOLVE, XD_DEV_DSUM_SQ, XD_DEV_LANCZOS, XD_DEV_EIGH_TRIDIAG,
    XD_DEV_GRAPH_KNN, XD_DEV_GRAPH_KNN_DENSE, XD_DEV_GRAPH_RADIUS, XD_DEV_GRAPH_RADIUS_GEO, XD_DEV_GRAPH_LLE_IW,
    XD_DEV_GRAPH_COMPONENTS, XD_DEV_GRAPH_JOIN, XD_DEV_GRAPH_DIJKSTRA, XD_LLE_GEMM_ORDERED, XD_LLE_LU_ORDERED,
)


def xd_mani_device_op(op: Int, a: List[Int], p: List[Int], floats: Int) raises -> Int:
    if xd_is_mani_shared_op(op):
        return xd_mani_shared_op[DevExec](op, a, p, floats)
    if op == XD_LLE_ITERATE:
        var args = xd_lle_args(p, floats)
        var e = 0.0
        var st = lle_iterate_dev_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f32(a[3]), xd_f32(a[4]),
                                     xd_f32(a[5]), xd_f32(a[6]), xd_f32(a[7]), args, e)
        xd_f64_out(a[8])[0] = e
        return st
    if op == XD_MDS_FIT:
        var n = xd_dim(p[0])
        var nc = xd_dim(p[1])
        var ns = xd_dim(p[2])
        mds_fit_check(n, nc, ns)
        var args = xd_mds_args(p, floats)
        var best_s = 0.0
        var best_it = mds_fit_dev_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), n, nc, ns, args, best_s)
        xd_f64_out(a[3])[0] = best_s
        return best_it
    if op == XD_DEV_TRISOLVE:
        return dev_trisolve_run(p[0], p[1], p[2], p[3], xd_dim(p[4]), xd_dim(p[5]), xd_dim(p[6]))
    if op == XD_DEV_DSUM_SQ:
        return dev_dsum_sq_run(p[0], p[1], xd_dim(p[2]))
    if op == XD_DEV_LANCZOS:
        return dev_lanczos_run(p[0], p[1], p[2], xd_dim(p[3]), xd_dim(p[4]), xd_dim(p[5]), xd_dim(p[6]))
    if op == XD_DEV_EIGH_TRIDIAG:
        return dev_eigh_tridiag_run(p[0], p[1], p[2], xd_dim(p[3]), xd_dim(p[4]))
    if op == XD_DEV_GRAPH_KNN:
        return dev_graph_knn_run(p[0], p[1], p[2], xd_dim(p[3]), xd_dim(p[4]), xd_dim(p[5]), p[6])
    if op == XD_DEV_GRAPH_KNN_DENSE:
        return dev_graph_knn_dense_run(p[0], p[1], p[2], xd_dim(p[3]), xd_dim(p[4]))
    if op == XD_DEV_GRAPH_RADIUS:
        return dev_graph_radius_run(p[0], p[1], xd_float(floats, 0), xd_dim(p[2]))
    if op == XD_DEV_GRAPH_RADIUS_GEO:
        return dev_graph_radius_geo_run(p[0], p[1], p[2], xd_float(floats, 0), xd_dim(p[3]), xd_dim(p[4]))
    if op == XD_DEV_GRAPH_LLE_IW:
        return dev_graph_lle_iw_run(p[0], p[1], p[2], xd_dim(p[3]), xd_dim(p[4]))
    if op == XD_DEV_GRAPH_COMPONENTS:
        return dev_graph_components_run(p[0], p[1], xd_dim(p[2]))
    if op == XD_DEV_GRAPH_JOIN:
        return dev_graph_join_run(p[0], p[1], p[2], xd_dim(p[3]), xd_dim(p[4]))
    if op == XD_DEV_GRAPH_DIJKSTRA:
        return dev_graph_dijkstra_run(p[0], p[1], xd_dim(p[2]))
    if op == XD_LLE_GEMM_ORDERED:
        comptime if AFN_GEMM_APPLE:
            lle_gemm_ordered_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_dim(p[0]), xd_dim(p[1]),
                                 xd_dim(p[2]), xd_dim(p[3]) != 0, xd_dim(p[4]) != 0)
            return 0
        raise Error("x_decomp native ABI: this binding has no x_decomp_lle_gemm_ordered")
    if op == XD_LLE_LU_ORDERED:
        comptime if AFN_GEMM_APPLE:
            lle_lu_ordered_run(xd_f32(a[0]), xd_i32(a[1]), xd_f32(a[2]), xd_dim(p[0]))
            return 0
        raise Error("x_decomp native ABI: this binding has no x_decomp_lle_lu_ordered")
    raise Error("x_decomp native ABI: the GPU binding has no manifold operation " + String(op))
