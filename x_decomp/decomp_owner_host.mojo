# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host binding's `DecompDev` (x_decomp/decomp_owner_kit.mojo): no
resident matrices; each solver is the host body the host binding registers
under that name (`ica_solve_py[HostExec]`, ...)."""
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.host import HostExec
from x_decomp.ica import ica_solve_run
from x_decomp.nmf import NmfArgs, nmf_nndsvd_run, nmf_solve_run
from x_decomp.fa_em import FaArgs, fa_em_main_run
from x_decomp.rotation import ortho_rotation_run
from x_decomp.lda_fit import LdaArgs, lda_fit_run
from x_decomp.api import lda_online_run
from x_decomp.lanczos_host import lanczos_run


struct HostDecomp(DecompDev):
    @staticmethod
    def has_eigh() -> Bool:
        return False

    @staticmethod
    def eigh(a: Int, w: F32Ptr, v: F32Ptr, n: Int, uplo: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def has_lu() -> Bool:
        return False

    @staticmethod
    def lu(a: Int, lu: Int, piv: I32Ptr, info: F32Ptr, n: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def ica_solve(
        px: F32Ptr, pw: F32Ptr, nc: Int, xr: Int, xc: Int, algo: Int, fun: Int, max_iter: Int, alpha: Float64,
        tol: Float64,
    ) raises -> Int:
        return ica_solve_run[HostExec](px, pw, nc, xr, xc, algo, fun, max_iter, alpha, tol)

    @staticmethod
    def nmf_solve(px: F32Ptr, pw: F32Ptr, ph: F32Ptr, n: Int, d: Int, nc: Int, solver: Int, a: NmfArgs) raises -> Int:
        return nmf_solve_run[HostExec](px, pw, ph, n, d, nc, solver, a)

    @staticmethod
    def nmf_nndsvd(pu: F32Ptr, ps: F32Ptr, pv: F32Ptr, pw: F32Ptr, ph: F32Ptr, n: Int, d: Int, r: Int, nc: Int) raises -> Int:
        return nmf_nndsvd_run[HostExec](pu, ps, pv, pw, ph, n, d, r, nc)

    @staticmethod
    def has_qr_r() -> Bool:
        return False

    @staticmethod
    def qr_r(a: Int, r: F32Ptr, m: Int, n: Int, consume: Bool) raises -> Int:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def fa_em_main(pa: F32Ptr, pp: F32Ptr, pw: F32Ptr, pl: MutPointer[Float64, MutUntrackedOrigin], ar: Int,
                   args: FaArgs) raises -> Int:
        return fa_em_main_run[HostExec](pa, pp, pw, pl, ar, args)

    @staticmethod
    def ortho_rotation(pc: F32Ptr, po: F32Ptr, nrow: Int, ncol: Int, vm: Bool, max_iter: Int, tol: Float64) raises:
        ortho_rotation_run[HostExec](pc, po, nrow, ncol, vm, max_iter, tol)

    @staticmethod
    def fa_flags() -> Int:
        return 0

    @staticmethod
    def fa_gram(x: Int, mean: Int, g: Int, var_: Int, n: Int, d: Int) raises:
        raise Error("x_decomp: no FAST Apple FactorAnalysis on the host binding")

    @staticmethod
    def fa_em(g: Int, psi_src: F32Ptr, w_dst: F32Ptr, psi_dst: F32Ptr, ll_dst: MutPointer[Float64, MutAnyOrigin],
              d: Int, nc: Int, n: Int, max_iter: Int, tol: Float64) raises -> Int:
        raise Error("x_decomp: no FAST Apple FactorAnalysis on the host binding")

    @staticmethod
    def fa_transform(x: Int, mean: Int, pm: Int, dst: Int, n: Int, d: Int, nc: Int) raises:
        raise Error("x_decomp: no FAST Apple FactorAnalysis on the host binding")

    @staticmethod
    def lda_rows(x: Int, ew: Int, d: Int, e: Int, n: Int, k: Int, v: Int, max_iter: Int, prior: Float64,
                 tol: Float64) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def lda_bound(x: Int, ddt: Int, dcomp: Int, dst: Int, n: Int, k: Int, v: Int, floor: Float64) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def lda_fit(px: F32Ptr, pc: F32Ptr, pe: F32Ptr, n: Int, v: Int, nc: Int, a: LdaArgs, mut draw: Int,
                mut nbi: Int) raises -> Int:
        return lda_fit_run[HostExec](px, pc, pe, n, v, nc, a, draw, nbi)

    @staticmethod
    def lda_online(px: F32Ptr, pc: F32Ptr, pe: F32Ptr, n: Int, v: Int, nc: Int, bs: Int, mdi: Int, seed: Int,
                   mut draw: Int, mut nbi: Int, fv: List[Float64]) raises:
        var got = lda_online_run[HostExec](px, pc, pe, n, v, nc, bs, mdi, seed, draw, nbi, fv)
        draw = got[0]
        nbi = got[1]

    @staticmethod
    def sqdist(a: Int, b: Int, dst: Int, na: Int, nb: Int, d: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def lanczos_dev(ia: Int, iq: Int, iab: Int, n: Int, j0: Int, j1: Int, cap: Int) raises:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def lanczos_host(pa: F32Ptr, pq: F32Ptr, pab: F32Ptr, n: Int, j0: Int, j1: Int, cap: Int) raises:
        _ = lanczos_run(pa, pq, pab, n, j0, j1, cap)

    @staticmethod
    def grp_flags() -> Int:
        return 0

    @staticmethod
    def grp_fit_fused(src: F32Ptr, cnt: Int, dst: Int, host_out: F32Ptr, count: Int, seed: Int, mode: Int, sd: Int,
                      sgrid: Int, sc: Float64, thr: Float64) raises -> Int:
        raise Error("x_decomp: no resident matrices on the host binding")

    @staticmethod
    def dev_project(a: Int, b: Int, c: Int, flag: Int, m: Int, k: Int, n: Int) raises -> Int:
        raise Error("x_decomp: no resident matrices on the host binding")
