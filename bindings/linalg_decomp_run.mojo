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
from std.memory import unsafe_memcpy
from bindings.decomp_kit_run import DKit, DK_ADD, DK_SUB, DK_SQ, DK_SELECT, DK_RECIP, DK_MUL
from x_decomp.descriptor_schema import (
    XD_RANDOMIZED_SVD, XD_DEV_LU_SOLVE, XD_LU_GESV, XD_DEV_LU_FACTOR, XD_TSQR_R, XD_DEV_TSQR_R, XD_DEV_LU_GESV,
    XD_DEV_LU_FACTOR_IN, XD_DEV_LU_SOLVE_IN, XD_DEV_FROM_TRAIN, XD_DEV_TO_TRAIN, XD_DEV_FINITE,
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


def ld_dev_gesv_run(mut kit: DKit, a: Int, bm: Int) raises -> Tuple[Int, Int]:
    """`ld_gesv_run` of device operands (lane vdf-linalg7): XD_DEV_LU_GESV reads A and B from their pool ids and
    writes X into a new one (`DevExec.lu_gesv`'s launches on device copies of the same words); only info (one word)
    comes to the host. (x, the singular diagonal number)."""
    var n = kit.nrows(a)
    var nrhs = kit.ncols(bm)
    var x = kit.device_output(n, nrhs)
    var info = kit.zeros(1, 1)
    var addresses: List[Int] = [kit.at(info)]
    _ = kit.xd(XD_DEV_LU_GESV, addresses, [kit.device(a), kit.device(bm), kit.matrices[x].device, n, nrhs],
               List[Float64]())
    return (x, Int(kit.word(info)))


def ld_dev_lu_pair_run(mut kit: DKit, a: Int, bm: Int, refine: Bool) raises -> Tuple[Int, Int, Int, Int]:
    """solve's resident LU pair route (`_lu_factor_resident`, then `_lu_solve_resident`) of device operands (lane
    vdf-linalg7): the factor reads A where the door adopted it (XD_DEV_LU_FACTOR_IN: nothing goes up, only info
    comes down) and the solve reads B and writes X on pool ids (XD_DEV_LU_SOLVE_IN). The same launches on the same
    words as the host operand's XD_DEV_LU_FACTOR / XD_DEV_LU_SOLVE. Returns (x, singular, the LU id, the pivots id):
    x is -1 when the factor is singular (the door then warns and solves on the factor as the host route does)."""
    var n = kit.nrows(a)
    var nrhs = kit.ncols(bm)
    var buffers = ld_device_buffers(kit, n)
    var info = kit.zeros(1, 1)
    var da = kit.device(a)
    var addresses: List[Int] = [kit.at(info)]
    _ = kit.xd(XD_DEV_LU_FACTOR_IN, addresses, [da, buffers[1], buffers[2], n], List[Float64]())
    var singular = Int(kit.word(info))
    if singular != 0:
        return (-1, singular, buffers[1], buffers[2])
    var x = kit.device_output(n, nrhs)
    _ = kit.xd(XD_DEV_LU_SOLVE_IN, List[Int](), [buffers[1], buffers[2], da, kit.device(bm), kit.matrices[x].device,
               n, nrhs, 1 if refine else 0], List[Float64]())
    return (x, 0, buffers[1], buffers[2])


def ld_dev_lu_factor_train_run(mut kit: DKit, a_handle: Int, da: Int, dlu: Int, dp: Int, n: Int) raises -> Int:
    """`lu_factor` of a device operand on the resident pair route (lane vdf-tail-b): training handle `a_handle`'s
    n x n words copied device to device into the pair's own pool id `da` (XD_DEV_FROM_TRAIN), tested finite there
    (XD_DEV_FINITE, refused in the host admission's words), then XD_DEV_LU_FACTOR_IN into `dlu` / `dp`: the host
    operand's XD_DEV_LU_FACTOR on the same words, with nothing going up and only info (one word) coming down.
    Returns the singular diagonal number (0: regular)."""
    var cells = n * n
    _ = kit.xd(XD_DEV_FROM_TRAIN, List[Int](), [a_handle, da, cells], List[Float64]())
    if kit.xd(XD_DEV_FINITE, List[Int](), [da, cells], List[Float64]()) != 0:
        raise Error("a: input must be finite; NaN/inf are unsupported")
    var info = kit.zeros(1, 1)
    var addresses: List[Int] = [kit.at(info)]
    _ = kit.xd(XD_DEV_LU_FACTOR_IN, addresses, [da, dlu, dp, n], List[Float64]())
    return Int(kit.word(info))


def ld_dev_lu_solve_pair_run(mut kit: DKit, da: Int, dlu: Int, dp: Int, bm: Int, refine: Bool) raises -> Int:
    """`lu_solve` of a device right-hand side on a resident pair (lane vdf-tail-b): XD_DEV_LU_SOLVE_IN reads the
    pair's LU, pivots and A (pool ids) and B (kit matrix bm, adopted) in place and writes X into a new pool matrix:
    the host operand's XD_DEV_LU_SOLVE on the same words. Returns x."""
    var n = kit.nrows(bm)
    var nrhs = kit.ncols(bm)
    var x = kit.device_output(n, nrhs)
    _ = kit.xd(XD_DEV_LU_SOLVE_IN, List[Int](), [dlu, dp, da, kit.device(bm), kit.matrices[x].device, n, nrhs,
               1 if refine else 0], List[Float64]())
    return x


def ld_pool_to_train(kit: DKit, id: Int, handle: Int, words: Int) raises:
    """PACK pool id `id`'s first `words` 4-byte words into training handle `handle` (one device copy; an int32 array
    is the same words)."""
    _ = kit.xd(XD_DEV_TO_TRAIN, List[Int](), [id, handle, words], List[Float64]())


def ld_train_pivots_down(mut kit: DKit, handle: Int, n: Int) raises -> Int:
    """The n int32 pivots of training handle `handle` in a kit int store (its index): one device copy into a pool id
    this kit owns, then n words down. The kit's own solve (`ld_lu_solve_run`) reads its pivots on the host, as for a
    host pair."""
    var id = kit.dev_alloc(n)
    kit.allocations.append(id)
    _ = kit.xd(XD_DEV_FROM_TRAIN, List[Int](), [handle, id, n], List[Float64]())
    var store = kit.istore(n)
    kit.dev_download(id, kit.iaddr(store), n)
    return store


def ld_pivots_to_train(mut kit: DKit, store: Int, handle: Int, n: Int) raises:
    """PACK the n int32 pivots of kit int store `store` (the kit's own host factor) into training handle `handle`: the
    words go into a kit matrix of n words and up once (`to_train`), bit for bit."""
    var m = kit.zeros(n, 1)
    unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=kit.at(m)),
                  src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=kit.iaddr(store)), count=n)
    kit.to_train(m, handle)


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
        var ra: Int
        if not kit._has_host(a) and not kit._has_host(bm) and kit.matrices[a].device >= 0 and kit.matrices[bm].device >= 0:
            # lane vdf-linalg7: device operands (adopted training arrays): the same TSQR of [a | b] on their pool ids
            # (XD_DEV_TSQR_R packs them on the device where the host route uploads them), R written into a pool id
            # (the NaN refusal is a device scan) and kept there as a device matrix: the slices below are device
            # moves (copies: the same words) and the small SVD reads R in place, so R never crosses the host.
            ra = kit.device_output(n, n)
            _ = kit.xd(XD_DEV_TSQR_R, [0], [kit.matrices[a].device, m, nn, 0, kit.matrices[ra].device,
                       kit.matrices[bm].device, nrhs], List[Float64]())
        else:
            ra = kit.zeros(n, n)
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
