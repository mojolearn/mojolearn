# SPDX-License-Identifier: Apache-2.0
# The manifold kit's table operations (x_decomp/descriptor_mani_schema.mojo)
# that run on both images, generic over the image's executor, plus the f64
# output cell they share. Each branch is the body of the matching door in
# providers/python/x_decomp_api_pyapi.mojo (`trisolve_py`, `dsum_sq_py`,
# `barycenter_rows_py`, `lle_apply_py`, `lle_local_py`) on typed words: the
# same checks, the same float32 rounding of the scalars, the same typed entry
# (lane entangled-mkit, 2026-11-12). The typed kit's door releases the GIL
# around the whole RUN, as each door released it around its call.
from x_decomp.api import trisolve_run, lle_apply_run, lle_local_run, dsum_sq_run
from x_decomp.exec_trait import Exec
from x_decomp.descriptor_ops import xd_f32, xd_float
from x_decomp.descriptor_schema import xd_dim
from x_decomp.descriptor_mani_schema import XD_TRISOLVE, XD_DSUM_SQ, XD_BARYCENTER, XD_LLE_APPLY, XD_LLE_LOCAL


def xd_f64_out(address: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    """A caller's one-word binary64 result cell (was the door's Python float)."""
    if address == 0:
        raise Error("x_decomp: null float64 result address")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)


def xd_graph_dim(v: Int) raises -> Int:
    """The host graph doors' `_ni` bound, with their sentence."""
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: graph dimension out of range")
    return v


def xd_is_mani_shared_op(op: Int) -> Bool:
    return op >= XD_TRISOLVE and op <= XD_LLE_LOCAL


def xd_mani_shared_op[E: Exec](op: Int, a: List[Int], p: List[Int], floats: Int) raises -> Int:
    if op == XD_TRISOLVE:
        var n = xd_dim(p[0])
        trisolve_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f32(a[3]), n, xd_dim(p[1]), xd_dim(p[2]))
        return n
    if op == XD_DSUM_SQ:
        # dsum_sq_py: the count unchecked, as the door passed it
        xd_f64_out(a[1])[0] = dsum_sq_run(a[0], p[0])
        return p[0]
    if op == XD_BARYCENTER:
        var n = xd_dim(p[0])
        var ny = xd_dim(p[1])
        var d = xd_dim(p[2])
        var k = xd_dim(p[3])
        E.barycenter_rows(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f32(a[3]), xd_f32(a[4]), n, ny, d, k,
                          Float32(xd_float(floats, 0)))
        return n
    if op == XD_LLE_APPLY:
        var nq = xd_dim(p[0])
        lle_apply_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f32(a[3]), nq, xd_dim(p[1]), xd_dim(p[2]),
                         xd_dim(p[3]))
        return nq
    if op == XD_LLE_LOCAL:
        var method = xd_dim(p[0])
        var n = xd_dim(p[1])
        lle_local_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), method, n, xd_dim(p[2]), xd_dim(p[3]),
                         xd_dim(p[4]), Float32(xd_float(floats, 0)))
        return n
    raise Error("x_decomp native ABI: not a shared manifold operation " + String(op))
