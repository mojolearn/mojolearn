# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of the native dense linear algebra functions (randomized_svd,
solve, lstsq, lu_factor, lu_solve; bindings/linalg_decomp_native.mojo is
UNPACK and PACK) over the typed kit (bindings/decomp_kit_run.mojo). Same
x_decomp operations, same words, same order: the bits do not move. Buffers
the door owns (admitted inputs, the public output Arrays the device factor
writes) arrive as addresses the door keeps alive. No Python (lane
entangled-decompkit, 2026-11-12).

solve and lu_solve keep a Python decision between two RUN pieces where the
public behaviour needs one (the singular-matrix warning before the solve,
and the right-hand side's admission, which depends on the factor's route)."""
from bindings.decomp_kit_run import DKit, DK_ADD, DK_SUB, DK_SQ, DK_SELECT, DK_RECIP, DK_MUL
from x_decomp.descriptor_schema import (
    XD_RANDOMIZED_SVD, XD_DEV_LU_SOLVE, XD_LU_GESV, XD_DEV_LU_FACTOR, XD_TSQR_R,
)


def ld_rsvd_run(kit: DKit, addresses: List[Int], params: List[Int]) raises:
    # The one compiled randomized_svd driver (x_decomp/randomized_svd_owner.mojo).
    _ = kit.xd(XD_RANDOMIZED_SVD, addresses, params, List[Float64]())


def ld_gesv_run(mut kit: DKit, a: Int, bm: Int) raises -> Tuple[Int, Int]:
    # gesv: one entry, the factor and pivots resident: (x, the singular diagonal number).
    var x = kit.copy_host(bm)
    var info = kit.zeros(1, 1)
    var addresses: List[Int] = [kit.at(a), kit.at(x), kit.at(info)]
    _ = kit.xd(XD_LU_GESV, addresses, [kit.nrows(a), kit.ncols(bm)], List[Float64]())
    kit.wrote_host(x)
    return (x, Int(kit.word(info)))


def ld_dev_lu_factor_run(kit: DKit, a_address: Int, lu_address: Int, piv_address: Int, info_address: Int,
                         da: Int, dlu: Int, dp: Int, n: Int) raises -> Int:
    # `_lu_factor_resident`: A up once, the factor on the device, LU and piv
    # down into the door's Arrays; < 0 when A is not finite.
    var addresses: List[Int] = [a_address, lu_address, piv_address, info_address]
    return kit.xd(XD_DEV_LU_FACTOR, addresses, [da, dlu, dp, n], List[Float64]())


def ld_device_buffers(mut kit: DKit, n: Int) raises -> Tuple[Int, Int, Int]:
    # The device A, LU and piv of solve's resident factor (the kit's own
    # buffers, returned to the pool when the kit closes).
    var da = kit.device_output(n, n)
    var dlu = kit.device_output(n, n)
    var dp = kit.device_output(n, 1)
    return (kit.matrices[da].device, kit.matrices[dlu].device, kit.matrices[dp].device)


def ld_resident_solve_run(kit: DKit, b_address: Int, out_address: Int, da: Int, dlu: Int, dp: Int, n: Int,
                          nrhs: Int, refine: Bool) raises:
    # `_lu_solve_resident`: B up, the solve on the device factor, X down.
    var addresses: List[Int] = [b_address, out_address]
    _ = kit.xd(XD_DEV_LU_SOLVE, addresses, [dlu, dp, da, n, nrhs, 1 if refine else 0], List[Float64]())


def ld_lu_solve_run(mut kit: DKit, l: Int, piv_address: Int, bm: Int, trans: Int, qfix_a: Int) raises -> Int:
    # `_Kit.lu_solve(lu, piv, B, trans)`, then LU_QFIX's one refinement step
    # x += LU^-1 (B - A x) when `qfix_a` (the float32 A) is given.
    var x = kit.lu_solve_addr(l, piv_address, bm, trans)
    if qfix_a >= 0:
        var residual = kit.lu_resid(qfix_a, x, bm)
        var step = kit.lu_solve_addr(l, piv_address, residual, 0)
        x = kit.ew(DK_ADD, x, step)
    return x


def ld_cut(mut kit: DKit, s: Int, rcond: Float64) raises -> Float64:
    # `_f32(k.word(S, 0) * rcond)`
    return Float64(Float32(kit.word(s, 0) * rcond))


def ld_pseudo_solve(mut kit: DKit, u: Int, s: Int, vt: Int, rhs: Int, cut: Float64) raises -> Int:
    # Vt^T ((U^T B) * (1 / S kept)^T): the values at or below cut dropped.
    var zero = kit.zeros(1, 1)
    var kept = kit.ew(DK_SELECT, s, s, zero, cut)
    var inverse = kit.ew(DK_RECIP, kept)
    var utb = kit.mm(u, rhs, ta=True)
    var inverse_t = kit.transpose(inverse)
    var scaled = kit.ew(DK_MUL, utb, inverse_t)
    return kit.mm(vt, scaled, ta=True)


struct LstsqOut(Copyable, Movable):
    """lstsq's (solution, residues (-1: empty), rank, singular values)."""
    var x: Int
    var resid: Int
    var rank: Int
    var s: Int

    def __init__(out self, x: Int, resid: Int, rank: Int, s: Int):
        self.x = x
        self.resid = resid
        self.rank = rank
        self.s = s


comptime LINALG_TS_MAX_N = 512


def ld_lstsq_run(mut kit: DKit, a: Int, bm: Int, rcond: Float64) raises -> LstsqOut:
    """lstsq after admission: `_tsqr_lstsq_core` (the blocked TSQR of [a | b],
    the SVD of its small R) when the shape allows it, else the QR +
    one-sided Jacobi SVD of a."""
    var m = kit.nrows(a)
    var nn = kit.ncols(a)
    var nrhs = kit.ncols(bm)
    if nn >= 1 and nrhs >= 1 and nn + nrhs <= LINALG_TS_MAX_N and m >= nn + nrhs:
        var n = nn + nrhs
        var ra = kit.zeros(n, n)
        var addresses: List[Int] = [kit.at(a), kit.at(bm), kit.at(ra)]
        _ = kit.xd(XD_TSQR_R, addresses, [m, nn, nrhs, 0], List[Float64]())
        kit.wrote_host(ra)
        var top = kit.rows_of(ra, 0, nn)
        var r = kit.cols_of(top, 0, nn)
        var c = kit.cols_of(top, nn, n)
        var usv = kit.svd_tall(r)
        var cut = ld_cut(kit, usv[1], rcond)
        var rank = kit.count_gt(usv[1], cut)
        var x = ld_pseudo_solve(kit, usv[0], usv[1], usv[2], c, cut)
        var resid = -1
        if rank == nn and m > nn:
            var tail = kit.rows_of(ra, nn, n)
            var last = kit.cols_of(tail, nn, n)
            var squares = kit.ew(DK_SQ, last)
            resid = kit.colsum(squares)
        return LstsqOut(x, resid, rank, usv[1])
    var r = min(m, nn)
    var usv = kit.thin_svd(a, r, True)
    var cut = ld_cut(kit, usv[1], rcond) if r > 0 else Float64(0.0)
    var rank = kit.count_gt(usv[1], cut)
    var x = ld_pseudo_solve(kit, usv[0], usv[1], usv[2], bm, cut)
    var resid = -1
    if rank == nn and m > nn:
        var fitted = kit.mm(a, x)
        var diff = kit.ew(DK_SUB, bm, fitted)
        var squares = kit.ew(DK_SQ, diff)
        resid = kit.colsum(squares)
    return LstsqOut(x, resid, rank, usv[1])
