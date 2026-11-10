# SPDX-License-Identifier: Apache-2.0
# The host-address operations of the x_decomp kernel-call table
# (x_decomp/descriptor_schema.mojo), generic over the image's executor:
# x_decomp/descriptor_device.mojo instantiates them on DevExec (the GPU
# binding's `x_decomp_ew` etc. are `ew_py[DevExec]`), x_decomp/descriptor_host.mojo
# on HostExec. Each branch is the body of the matching door in
# providers/python/x_decomp_api_pyapi.mojo (`ew_py`, `gemm_py`, `colsum_py`,
# `rowsum_py`, `sqdist_py`, `move_py`) on typed words instead of PythonObjects:
# the same dimension checks, the same float32 rounding of the scalar, the same
# typed entry. The caller (bindings/kit_call.mojo) releases the GIL around
# these, as the doors did.
from x_decomp.api import ew_run, gemm_run, sqdist_run, move_run, idn_flags_run, tsqr_r_run, ols_tsqr_r_run, order_f_run, svd_run, rank_above_run, geqrf_run, orgqr_run
from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.descriptor_schema import (
    XD_EW, XD_GEMM, XD_COLSUM, XD_ROWSUM, XD_SQDIST, XD_MOVE, xd_dim,
    XD_FLAGS, XD_TSQR_R, XD_OLS_TSQR_R, XD_ORDER, XD_SVD, XD_RANK, XD_ORTH_DIAG, XD_GEQRF, XD_ORGQR,
)


def xd_f32(address: Int) raises -> F32Ptr:
    """`_f` of the doors: a host float32 buffer, never null."""
    if address == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=address)


def xd_float(floats: Int, i: Int) raises -> Float64:
    if floats == 0:
        raise Error("x_decomp native ABI: null float span")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=floats)[i]


def xd_is_address_op(op: Int) -> Bool:
    return (op >= XD_EW and op <= XD_MOVE) or op == XD_FLAGS or op == XD_TSQR_R or op == XD_OLS_TSQR_R or op == XD_ORDER or op == XD_SVD or op == XD_RANK or op == XD_ORTH_DIAG or op == XD_GEQRF or op == XD_ORGQR


def xd_address_op[E: Exec](op: Int, a: List[Int], p: List[Int], floats: Int) raises -> Int:
    """One host-address operation; returns what the door returned."""
    if op == XD_EW:
        # p = [op, count, d, lb, bm, lc, cm], as x_decomp_ew
        var count = xd_dim(p[1])
        var d = xd_dim(p[2])
        var lb = xd_dim(p[3])
        var lc = xd_dim(p[5])
        var sv = Float32(xd_float(floats, 0))
        ew_run[E](p[0], xd_f32(a[0]), xd_f32(a[1]), lb, p[4], xd_f32(a[2]), lc, p[6], xd_f32(a[3]), count, d, sv)
        return count
    if op == XD_GEMM:
        return gemm_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_dim(p[0]), xd_dim(p[1]), xd_dim(p[2]),
                           p[3] != 0, p[4] != 0)
    if op == XD_COLSUM:
        var n = xd_dim(p[0])
        var d = xd_dim(p[1])
        E.colsum(xd_f32(a[0]), xd_f32(a[1]), n, d)
        return d
    if op == XD_ROWSUM:
        var n = xd_dim(p[0])
        var d = xd_dim(p[1])
        E.rowsum(xd_f32(a[0]), xd_f32(a[1]), n, d)
        return n
    if op == XD_SQDIST:
        # the Minkowski p rounded once to float32, as sqdist_py rounds it
        var pw = Float32(xd_float(floats, 0))
        return sqdist_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_dim(p[0]), xd_dim(p[1]), xd_dim(p[2]),
                             xd_dim(p[3]), pw)
    if op == XD_MOVE:
        var count = xd_dim(p[1])
        move_run(p[0], xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), count, xd_dim(p[2]), xd_dim(p[3]), xd_dim(p[4]),
                 xd_dim(p[5]), xd_dim(p[6]), xd_dim(p[7]), xd_dim(p[8]), xd_dim(p[9]))
        return count
    if op == XD_FLAGS:
        return idn_flags_run()
    if op == XD_TSQR_R:
        return tsqr_r_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_dim(p[0]), xd_dim(p[1]), xd_dim(p[2]), p[3] != 0)
    if op == XD_OLS_TSQR_R:
        return ols_tsqr_r_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f32(a[3]), a[4], xd_dim(p[0]), xd_dim(p[1]))
    if op == XD_ORDER:
        order_f_run(a[0], xd_dim(p[0]), a[1])
        return p[0]
    if op == XD_SVD:
        svd_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_dim(p[0]), xd_dim(p[1]))
        return p[1]
    if op == XD_RANK:
        return rank_above_run(a[0], xd_dim(p[0]), xd_float(floats, 0))
    if op == XD_ORTH_DIAG:
        E.orth_diag(xd_f32(a[0]), xd_dim(p[0]), xd_dim(p[1]), xd_f32(a[1]))
        return p[1]
    if op == XD_GEQRF:
        return geqrf_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_dim(p[0]), xd_dim(p[1]))
    if op == XD_ORGQR:
        orgqr_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_dim(p[0]), xd_dim(p[1]), xd_dim(p[2]), xd_dim(p[3]))
        return p[3]
    raise Error("x_decomp native ABI: not a host-address operation " + String(op))
