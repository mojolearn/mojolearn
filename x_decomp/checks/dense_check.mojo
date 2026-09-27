# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5307-5309 of the decomp lane: the LU pivot and its tie
(5307), the substitution folds of getrs and of the Cholesky (5308), and the
two-pass modified Gram-Schmidt (5309).

    tools/with_identical_mode.sh pixi run mojo run -I . x_decomp/checks/dense_check.mojo
"""
from core.identity_trace import IdentityTrace
from x_decomp.checks.oracles import oracle_chol, oracle_lu, oracle_lu_solve, oracle_orth
from x_decomp.checks.seam_util import (
    count_diff_f32,
    count_diff_i32,
    iptr,
    lcg_unit,
    ptr,
    require_separates,
    same,
    seam_fixture,
    zeros,
)
from x_decomp.device import DevExec
from x_decomp.host import HostExec


def tie_matrix(n: Int) -> List[Float32]:
    """Small integers with EXACT pivot ties in every column (|a| = 2 twice),
    so the pivot row is the tie-break's choice, and a well-scaled rest."""
    var a = List[Float32](capacity=n * n)
    var st = UInt64(77)
    for i in range(n):
        for j in range(n):
            var u = lcg_unit(st)
            a.append(Float32(Int(u * Float32(5))) - Float32(2))
    for j in range(n):
        a[((j + 1) % n) * n + j] = Float32(2)
        a[((j + 3) % n) * n + j] = Float32(-2)
    return a^


def spd(n: Int) -> List[Float32]:
    var x = seam_fixture(n + 5, n, 11)
    for t in range(len(x)):
        if abs(x[t]) > Float32(100):
            x[t] = x[t] * Float32(1e-6)
    var g = List[Float32](length=n * n, fill=Float32(0))
    for i in range(n):
        for j in range(n):
            var acc = Float32(0)
            for r in range(n + 5):
                acc = acc + x[r * n + i] * x[r * n + j]
            g[i * n + j] = acc
        g[i * n + i] = g[i * n + i] + Float32(n)
    for i in range(n):
        for j in range(i + 1, n):
            g[j * n + i] = g[i * n + j]
    return g^


def main() raises:
    var tr = IdentityTrace()
    tr.header("x_decomp dense_check (DEVIATIONS 5307-5309)")
    # ---- 5307 the pivot
    var n = 9
    var a = tie_matrix(n)
    var got_o = oracle_lu(a, n)
    var alt_o = oracle_lu(a, n, 1)
    require_separates("5307 pivot tie (lowest row vs last)", count_diff_i32(got_o[1], alt_o[1]))
    var lu_d = a.copy()
    var piv_d = List[Int32](length=n, fill=Int32(0))
    var info = zeros(1)
    DevExec.lu(ptr(lu_d), iptr(piv_d), ptr(info), n)
    same("5307 lu pivots device", count_diff_i32(piv_d, got_o[1]))
    same("5307 lu factors device", count_diff_f32(lu_d, got_o[0]))
    var lu_h = a.copy()
    var piv_h = List[Int32](length=n, fill=Int32(0))
    HostExec.lu(ptr(lu_h), iptr(piv_h), ptr(info), n)
    same("5307 lu pivots host", count_diff_i32(piv_h, got_o[1]))
    same("5307 lu factors host", count_diff_f32(lu_h, got_o[0]))
    tr.record_list_f32("x_decomp.lu", lu_d)
    # ---- 5308 getrs and the Cholesky
    var nrhs = 5
    var bsrc = seam_fixture(n, nrhs, 13)
    for t in range(len(bsrc)):
        if abs(bsrc[t]) > Float32(100):
            bsrc[t] = bsrc[t] * Float32(1e-6)
    var want = oracle_lu_solve(got_o[0], got_o[1], bsrc, n, nrhs)
    require_separates("5308 getrs substitution order", count_diff_f32(want, oracle_lu_solve(got_o[0], got_o[1], bsrc, n, nrhs, 1)))
    var xd = bsrc.copy()
    DevExec.lu_solve(ptr(got_o[0]), iptr(got_o[1]), ptr(xd), n, nrhs)
    same("5308 lu_solve device", count_diff_f32(xd, want))
    var xh = bsrc.copy()
    HostExec.lu_solve(ptr(got_o[0]), iptr(got_o[1]), ptr(xh), n, nrhs)
    same("5308 lu_solve host", count_diff_f32(xh, want))
    tr.record_list_f32("x_decomp.lu_solve", xd)
    var ns = 11
    var g = spd(ns)
    var wl = oracle_chol(g, ns)
    require_separates("5308 Cholesky fold order", count_diff_f32(wl, oracle_chol(g, ns, 1)))
    var cd = g.copy()
    DevExec.chol(ptr(cd), ptr(info), ns)
    same("5308 chol device", count_diff_f32(cd, wl))
    var ch = g.copy()
    HostExec.chol(ptr(ch), ptr(info), ns)
    same("5308 chol host", count_diff_f32(ch, wl))
    tr.record_list_f32("x_decomp.chol", cd)
    # ---- 5309 MGS2
    var mm = 40
    var l = 6
    var q = seam_fixture(mm, l, 17)
    for t in range(mm):
        # nearly dependent columns: column 1 = column 0 + a small perturbation
        q[t * l + 1] = q[t * l + 0] + Float32(1e-3) * q[t * l + 2]
    var wq = oracle_orth(q, mm, l)
    require_separates("5309 orth passes (two vs one)", count_diff_f32(wq, oracle_orth(q, mm, l, 1)))
    var qd = q.copy()
    DevExec.orth(ptr(qd), mm, l)
    same("5309 orth device", count_diff_f32(qd, wq))
    var qh = q.copy()
    HostExec.orth(ptr(qh), mm, l)
    same("5309 orth host", count_diff_f32(qh, wq))
    tr.record_list_f32("x_decomp.orth", qd)
    print("PASS x_decomp dense_check")
