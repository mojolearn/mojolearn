# SPDX-License-Identifier: Apache-2.0
# The manifold kit's operations of the x_decomp kernel-call table (lane
# entangled-mkit, 2026-11-12): the `x_decomp_*` Python entries the manifold kit
# (was bindings/mani_kit.mojo `MKit.fn(name)` + `family_call`) still called by
# name, appended as table operations so the typed kit
# (bindings/mani_kit_run.mojo) calls the image's dispatch with plain words.
# Each operation runs the SAME typed body its `*_py` door ran, with the same
# integer checks and the same float32 rounding: no numeric change.
#
# Numbered from 120 so the sibling lane (entangled-decompkit, which appends to
# the 70s) and this lane never take the same id; xd_schema refuses the gap.
# Float64 results travel through an address (the doors returned a Python
# float): `dsum_sq`, LLE's last subspace change and MDS's stress.

# host-address operations on both images (generic over the image's executor)
comptime XD_TRISOLVE = 120          # a: [lu, idx, b, dst] i: [n, nrhs, trans]
comptime XD_DSUM_SQ = 121           # a: [x, dst(f64)] i: [n]
comptime XD_BARYCENTER = 122        # a: [x, y, nbr, w, flags] i: [n, ny, d, k] f: [reg]
comptime XD_LLE_APPLY = 123         # a: [w, idx, e, dst] i: [nq, nf, nn, nc]
comptime XD_LLE_LOCAL = 124         # a: [x, idx, b] i: [method, n, d, nn, nc] f: [tol]
# host-address operations whose body differs per image (host: HostExec body;
# GPU image: the `*_dev_*` body the GPU binding registered under the same name)
comptime XD_LLE_ITERATE = 125       # a: [f0, lu, pm, im, z, x, y, s, e(f64)] i: [n, n1, nc, p, max_iter, seed] f: [floor, sub_tol, stall_tol] -> status
comptime XD_MDS_FIT = 126           # a: [dis, starts, y, stress(f64)] i: [n, nc, ns, max_iter, metric, norm] f: [eps] -> n_iter
# host image only (the host binding's entries)
comptime XD_LANCZOS = 127           # a: [a, q, ab] i: [n, j0, j1, cap]
comptime XD_EIGH_TRIDIAG = 128      # a: [ab, w, z] i: [j, cap]
comptime XD_GRAPH_KNN = 129         # a: [d, idx, dst] i: [n, m, nn, excl]
comptime XD_GRAPH_KNN_DENSE = 130   # a: [idx, w, dst] i: [n, nn]
comptime XD_GRAPH_RADIUS = 131      # a: [d, dst] i: [n] f: [r]
comptime XD_GRAPH_RADIUS_GEO = 132  # a: [dq, d, g] i: [nq, n] f: [r]
comptime XD_GRAPH_LLE_IW = 133      # a: [idx, wb, dst] i: [n, nn]
comptime XD_GRAPH_COMPONENTS = 134  # a: [w, comp] i: [n] -> count
comptime XD_GRAPH_JOIN = 135        # a: [w, d, comp] i: [n, count]
comptime XD_GRAPH_DIJKSTRA = 136    # a: [w, dist] i: [n]
# GPU image only (resident pool ids)
comptime XD_DEV_TRISOLVE = 137      # i: [lu, idx, src, dst, n, nrhs, trans]
comptime XD_DEV_DSUM_SQ = 138       # i: [a, dst, n]
comptime XD_DEV_LANCZOS = 139       # i: [a, q, ab, n, j0, j1, cap]
comptime XD_DEV_EIGH_TRIDIAG = 140  # i: [ab, w, z, j, cap]
comptime XD_DEV_GRAPH_KNN = 141     # i: [d, idx, dst, n, m, nn, excl]
comptime XD_DEV_GRAPH_KNN_DENSE = 142  # i: [idx, w, dst, n, nn]
comptime XD_DEV_GRAPH_RADIUS = 143  # i: [d, dst, n] f: [r]
comptime XD_DEV_GRAPH_RADIUS_GEO = 144  # i: [dq, d, g, nq, n] f: [r]
comptime XD_DEV_GRAPH_LLE_IW = 145  # i: [idx, wb, dst, n, nn]
comptime XD_DEV_GRAPH_COMPONENTS = 146  # i: [w, comp, n] -> count
comptime XD_DEV_GRAPH_JOIN = 147    # i: [w, d, comp, n, count]
comptime XD_DEV_GRAPH_DIJKSTRA = 148  # i: [w, dist, n]
# the Apple FAST image's ordered LLE cells (AFN_GEMM_APPLE; XD_CAP_LLE_ORDERED)
comptime XD_LLE_GEMM_ORDERED = 149  # a: [a, b, dst] i: [m, k, n, ta, tb]
comptime XD_LLE_LU_ORDERED = 150    # a: [a, piv(int32), info] i: [n]
comptime XD_MANI_OPERATION_END = 151

# Capability bits the kit read by probing the module namespace for an entry
# (`opt_dev("x_decomp_dev_lu")`, `opt_dev("x_decomp_dev_eigh_to")`,
# `enable_lle_precision`'s ordered entries).
comptime XD_CAP_DEV_LU = 8          # IDN_LU_RESIDENT: XD_DEV_LU / XD_DEV_LU_AUX
comptime XD_CAP_DEV_EIGH = 16       # IDN_EIGH_RESIDENT: XD_DEV_EIGH_TO
comptime XD_CAP_LLE_ORDERED = 32    # AFN_GEMM_APPLE: XD_LLE_GEMM_ORDERED / XD_LLE_LU_ORDERED


def xd_is_mani_op(op: Int) -> Bool:
    return op >= XD_TRISOLVE and op < XD_MANI_OPERATION_END


def xd_mani_schema(op: Int) raises -> Tuple[Int, Int, Int]:
    """(addresses, integers, floats) of each manifold operation, exactly."""
    if op == XD_TRISOLVE:
        return (4, 3, 0)
    if op == XD_LLE_APPLY:
        return (4, 4, 0)
    if op == XD_DSUM_SQ:
        return (2, 1, 0)
    if op == XD_BARYCENTER:
        return (5, 4, 1)
    if op == XD_LLE_LOCAL:
        return (3, 5, 1)
    if op == XD_LLE_ITERATE:
        return (9, 6, 3)
    if op == XD_MDS_FIT:
        return (4, 6, 1)
    if op == XD_LANCZOS:
        return (3, 4, 0)
    if op == XD_EIGH_TRIDIAG:
        return (3, 2, 0)
    if op == XD_GRAPH_KNN:
        return (3, 4, 0)
    if op == XD_GRAPH_KNN_DENSE or op == XD_GRAPH_LLE_IW:
        return (3, 2, 0)
    if op == XD_GRAPH_RADIUS:
        return (2, 1, 1)
    if op == XD_GRAPH_RADIUS_GEO:
        return (3, 2, 1)
    if op == XD_GRAPH_COMPONENTS or op == XD_GRAPH_DIJKSTRA:
        return (2, 1, 0)
    if op == XD_GRAPH_JOIN:
        return (3, 2, 0)
    if op == XD_DEV_TRISOLVE or op == XD_DEV_LANCZOS or op == XD_DEV_GRAPH_KNN:
        return (0, 7, 0)
    if op == XD_DEV_DSUM_SQ or op == XD_DEV_GRAPH_COMPONENTS or op == XD_DEV_GRAPH_DIJKSTRA:
        return (0, 3, 0)
    if op == XD_DEV_EIGH_TRIDIAG or op == XD_DEV_GRAPH_KNN_DENSE or op == XD_DEV_GRAPH_LLE_IW or op == XD_DEV_GRAPH_JOIN:
        return (0, 5, 0)
    if op == XD_DEV_GRAPH_RADIUS:
        return (0, 3, 1)
    if op == XD_DEV_GRAPH_RADIUS_GEO:
        return (0, 5, 1)
    if op == XD_LLE_GEMM_ORDERED:
        return (3, 5, 0)
    if op == XD_LLE_LU_ORDERED:
        return (3, 1, 0)
    raise Error("x_decomp native ABI: unknown operation " + String(op))
