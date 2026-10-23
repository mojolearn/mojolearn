# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's `DecompDev` (x_decomp/decomp_owner_kit.mojo): the
resident eigh / LU where the build registers `x_decomp_dev_eigh` /
`x_decomp_dev_lu` (IDN_EIGH_RESIDENT, IDN_LU_RESIDENT), and each solver
the GPU body the binding registers under that name (`ica_solve_dev_py`, ...)."""
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.resident import IDN_EIGH_RESIDENT, IDN_LU_RESIDENT, dev_eigh_run, dev_lu_run
from x_decomp.ica_dev import ica_solve_dev_run
from x_decomp.nmf import NmfArgs
from x_decomp.nmf_dev import nmf_nndsvd_dev_run, nmf_solve_dev_run
from x_decomp.fa_em import FaArgs
from x_decomp.fa_em_dev import fa_em_main_dev_run
from x_decomp.rotation_dev import ortho_rotation_dev_run
from x_decomp.resident import IDN_QR_R_RESIDENT, dev_lda_bound_run, dev_lda_rows_run, dev_qr_r_run, dev_sqdist_run
from x_decomp.lanczos_dev import dev_lanczos_run
from x_decomp.resident import (
    GRP_CLS2_ANY, GRP_CLS2_DEVSCAN, GRP_CLS2_LAZY, GRP_CLS2_NOSCAN, GRP_FAST_FUSED, GRP_FAST_SRP_STRAT,
    dev_project_run, grp_fit_fused_run,
)
from x_decomp.lda_fit import LdaArgs
from x_decomp.lda_fit_dev import lda_fit_dev_run
from x_decomp.kit_device import lda_online_dev_run
from x_decomp.fa_fast import (
    FA_FAST_APPLE, FA_GRAM_DF, FA_GRAM_ONCE, FA_ITER_DEVICE, FA_TRANSFORM_FUSED, fa_em_run, fa_gram_run,
    fa_transform_run,
)


struct DevDecomp(DecompDev):
    @staticmethod
    def has_eigh() -> Bool:
        return IDN_EIGH_RESIDENT

    @staticmethod
    def eigh(a: Int, w: F32Ptr, v: F32Ptr, n: Int, uplo: Int) raises:
        _ = dev_eigh_run(a, w, v, n, uplo)

    @staticmethod
    def has_lu() -> Bool:
        return IDN_LU_RESIDENT

    @staticmethod
    def lu(a: Int, lu: Int, piv: I32Ptr, info: F32Ptr, n: Int) raises:
        _ = dev_lu_run(a, lu, piv, info, n)

    @staticmethod
    def ica_solve(
        px: F32Ptr, pw: F32Ptr, nc: Int, xr: Int, xc: Int, algo: Int, fun: Int, max_iter: Int, alpha: Float64,
        tol: Float64,
    ) raises -> Int:
        return ica_solve_dev_run(px, pw, nc, xr, xc, algo, fun, max_iter, alpha, tol)

    @staticmethod
    def nmf_solve(px: F32Ptr, pw: F32Ptr, ph: F32Ptr, n: Int, d: Int, nc: Int, solver: Int, a: NmfArgs) raises -> Int:
        return nmf_solve_dev_run(px, pw, ph, n, d, nc, solver, a)

    @staticmethod
    def nmf_nndsvd(pu: F32Ptr, ps: F32Ptr, pv: F32Ptr, pw: F32Ptr, ph: F32Ptr, n: Int, d: Int, r: Int, nc: Int) raises -> Int:
        return nmf_nndsvd_dev_run(pu, ps, pv, pw, ph, n, d, r, nc)

    @staticmethod
    def has_qr_r() -> Bool:
        return IDN_QR_R_RESIDENT

    @staticmethod
    def qr_r(a: Int, r: F32Ptr, m: Int, n: Int, consume: Bool) raises -> Int:
        return dev_qr_r_run(a, r, m, n, consume)

    @staticmethod
    def fa_em_main(pa: F32Ptr, pp: F32Ptr, pw: F32Ptr, pl: MutPointer[Float64, MutUntrackedOrigin], ar: Int,
                   args: FaArgs) raises -> Int:
        return fa_em_main_dev_run(pa, pp, pw, pl, ar, args)

    @staticmethod
    def ortho_rotation(pc: F32Ptr, po: F32Ptr, nrow: Int, ncol: Int, vm: Bool, max_iter: Int, tol: Float64) raises:
        ortho_rotation_dev_run(pc, po, nrow, ncol, vm, max_iter, tol)

    @staticmethod
    def fa_flags() -> Int:
        # `x_decomp_fa_defines` is registered only under FA_FAST_APPLE
        var f = 0
        comptime if FA_FAST_APPLE:
            comptime if FA_GRAM_ONCE:
                f |= 1
            comptime if FA_ITER_DEVICE:
                f |= 2
            comptime if FA_TRANSFORM_FUSED:
                f |= 4
            comptime if FA_GRAM_DF:
                f |= 8
        return f

    @staticmethod
    def fa_gram(x: Int, mean: Int, g: Int, var_: Int, n: Int, d: Int) raises:
        _ = fa_gram_run(x, mean, g, var_, n, d)

    @staticmethod
    def fa_em(g: Int, psi_src: F32Ptr, w_dst: F32Ptr, psi_dst: F32Ptr, ll_dst: MutPointer[Float64, MutAnyOrigin],
              d: Int, nc: Int, n: Int, max_iter: Int, tol: Float64) raises -> Int:
        return fa_em_run(g, psi_src, w_dst, psi_dst, ll_dst, d, nc, n, max_iter, tol)

    @staticmethod
    def fa_transform(x: Int, mean: Int, pm: Int, dst: Int, n: Int, d: Int, nc: Int) raises:
        _ = fa_transform_run(x, mean, pm, dst, n, d, nc)

    @staticmethod
    def lda_rows(x: Int, ew: Int, d: Int, e: Int, n: Int, k: Int, v: Int, max_iter: Int, prior: Float64,
                 tol: Float64) raises:
        _ = dev_lda_rows_run(x, ew, d, e, n, k, v, max_iter, prior, tol)

    @staticmethod
    def lda_bound(x: Int, ddt: Int, dcomp: Int, dst: Int, n: Int, k: Int, v: Int, floor: Float64) raises:
        _ = dev_lda_bound_run(x, ddt, dcomp, dst, n, k, v, floor)

    @staticmethod
    def lda_fit(px: F32Ptr, pc: F32Ptr, pe: F32Ptr, n: Int, v: Int, nc: Int, a: LdaArgs, mut draw: Int,
                mut nbi: Int) raises -> Int:
        return lda_fit_dev_run(px, pc, pe, n, v, nc, a, draw, nbi)

    @staticmethod
    def lda_online(px: F32Ptr, pc: F32Ptr, pe: F32Ptr, n: Int, v: Int, nc: Int, bs: Int, mdi: Int, seed: Int,
                   mut draw: Int, mut nbi: Int, fv: List[Float64]) raises:
        lda_online_dev_run(px, pc, pe, n, v, nc, bs, mdi, seed, draw, nbi, fv)

    @staticmethod
    def sqdist(a: Int, b: Int, dst: Int, na: Int, nb: Int, d: Int) raises:
        _ = dev_sqdist_run(a, b, dst, na, nb, d, 0, 2.0)

    @staticmethod
    def lanczos_dev(ia: Int, iq: Int, iab: Int, n: Int, j0: Int, j1: Int, cap: Int) raises:
        _ = dev_lanczos_run(ia, iq, iab, n, j0, j1, cap)

    @staticmethod
    def lanczos_host(pa: F32Ptr, pq: F32Ptr, pab: F32Ptr, n: Int, j0: Int, j1: Int, cap: Int) raises:
        raise Error("x_decomp: the GPU binding runs Lanczos on resident matrices")

    @staticmethod
    def grp_flags() -> Int:
        # `grp_cls2_py`'s bits; `x_decomp_grp_cls2` is registered only under GRP_CLS2_ANY
        var f = 0
        comptime if GRP_CLS2_ANY:
            comptime if GRP_CLS2_NOSCAN:
                f |= 1
            comptime if GRP_CLS2_DEVSCAN:
                f |= 2
            comptime if GRP_CLS2_LAZY:
                f |= 4
            comptime if GRP_FAST_FUSED:
                f |= 8
            comptime if GRP_FAST_SRP_STRAT:
                f |= 16
        return f

    @staticmethod
    def grp_fit_fused(src: F32Ptr, cnt: Int, dst: Int, host_out: F32Ptr, count: Int, seed: Int, mode: Int, sd: Int,
                      sgrid: Int, sc: Float64, thr: Float64) raises -> Int:
        comptime if GRP_FAST_FUSED:
            return grp_fit_fused_run(src, cnt, dst, host_out, count, seed, mode, sd, sgrid, sc, thr)
        raise Error("x_decomp: this build has no fused random-projection fit")

    @staticmethod
    def dev_project(a: Int, b: Int, c: Int, flag: Int, m: Int, k: Int, n: Int) raises -> Int:
        return dev_project_run(a, b, c, flag, m, k, n)
