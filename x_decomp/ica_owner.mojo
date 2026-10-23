# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`FastICA.fit` / `fit_transform` / `transform` / `inverse_transform` as ONE
native operation each (lane mca2-decomp-a, 2026-10-22): the steps
`_expansion_decomp.FastICA._fit`, `_solve`, `transform` and
`inverse_transform` sequenced from Python with the decomp kit, on `RKit`
(x_decomp/decomp_owner_kit.mojo): the same entries' typed bodies, operands,
broadcast modes, float32 scalars and binary64 scalar arithmetic, the
fixed-point iteration the binding's `x_decomp_ica_solve` runs
(`DecompDev.ica_solve`, its host or GPU body)."""
from std.math import sqrt

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev, RKit
from x_decomp.decomp_owner_kit import (
    DecompDev, OP_ADD, OP_DIV, OP_MAXS, OP_MUL, OP_SCALE, OP_SQDIFF, OP_SQRT, OP_SUB,
    kit_colmean, kit_eigh, kit_pinv_rows, kit_rand, kit_reversed_columns, kit_rows,
)

#: `_F32_EPS * 10`: the eigenvalue floor before the whitening's sqrt
comptime _EV_FLOOR = 1.1920928955078125e-07 * 10
#: `k.rand(nc, nc, seed, 20, 1)`: the initial unmixing matrix's stream and kind
comptime _WINIT_STREAM = 20
comptime _WINIT_KIND = 1


def ica_fit_run[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    x: F32Ptr, n: Int, d: Int, nc: Int, whiten: Int, winit: F32Ptr, has_winit: Bool, seed: Int,
    algo: Int, fun: Int, max_iter: Int, alpha: Float64, tol: Float64, project: Bool,
    components_out: F32Ptr, mixing_out: F32Ptr, whitening_out: F32Ptr, mean_out: F32Ptr, sources_out: F32Ptr,
) raises -> Int:
    """`_fit(X, project)`. whiten: 0 False, 1 'unit-variance', 2
    'arbitrary-variance'. Writes components (nc x d, or the unmixing matrix
    without whitening), mixing (its pseudo-inverse), whitening (nc x d) and
    mean (d) when whitening, and with `project` the sources (n x nc).
    Returns n_iter_."""
    var kit = RKit[E, D, RES]()
    try:
        var m = kit.add(Int(x), -1, n, d)
        var xc: Int
        var x1: Int
        var k_mat = -1
        var mean = -1
        if whiten != 0:
            mean = kit_colmean(kit, m)
            xc = kit.ew(OP_SUB, m, mean)
            var gram = kit.mm(xc, xc, ta=True)
            var eig = kit_eigh[E, D, RES, X](kit, gram)
            var ev = kit_reversed_columns(kit, eig[0])
            ev = kit.ew(OP_MAXS, ev, scalar=_EV_FLOOR)
            var sv = kit.ew(OP_SQRT, ev)
            var u = kit_reversed_columns(kit, eig[1])
            var first = kit_rows(kit, u, 0, 1)
            var signs = kit.signs(first)
            u = kit.ew(OP_MUL, u, signs)
            var scaled = kit.ew(OP_DIV, u, sv)
            var scaled_t = kit.transpose(scaled)
            k_mat = kit_rows(kit, scaled_t, 0, nc)
            var projected = kit.mm(k_mat, xc, tb=True)
            x1 = kit.ew(OP_SCALE, projected, scalar=sqrt(Float64(n)))
        else:
            xc = m
            x1 = kit.transpose(m)
        var w_init: Int
        if not has_winit:
            w_init = kit_rand(kit, nc, nc, seed, _WINIT_STREAM, _WINIT_KIND)
        else:
            w_init = kit.add(Int(winit), -1, nc, nc)
        # `_solve`: W = Winit.copy(), solved in place on X1's host words
        var w = kit.copy_host(w_init)
        var it = X.ica_solve(kit.ptr(x1), kit.ptr(w), nc, kit.r(x1), kit.c(x1), algo, fun, max_iter, alpha, tol)
        var s: Int
        var comp: Int
        if whiten != 0:
            var wk = kit.mm(w, k_mat)
            s = kit.mm(xc, wk, tb=True)
            if whiten == 1:
                var smean = kit_colmean(kit, s)
                var spread = kit.ew(OP_SQDIFF, s, smean)
                var sums = kit.sum(spread, True)
                var var_ = kit.ew(OP_SCALE, sums, scalar=1.0 / Float64(n))
                var std = kit.ew(OP_SQRT, var_)
                s = kit.ew(OP_DIV, s, std)
                var std_t = kit.transpose(std)
                w = kit.ew(OP_DIV, w, std_t)
            comp = kit.mm(w, k_mat)
            kit.copy_out(k_mat, whitening_out, nc * d)
            kit.copy_out(mean, mean_out, d)
        else:
            s = kit.mm(m, w, tb=True)
            comp = w
        var comp_count = kit.r(comp) * kit.c(comp)
        kit.copy_out(comp, components_out, comp_count)
        var mixing = kit_pinv_rows[E, D, RES, X](kit, comp)
        var mixing_count = kit.r(mixing) * kit.c(mixing)
        kit.copy_out(mixing, mixing_out, mixing_count)
        if project:
            var s_count = kit.r(s) * kit.c(s)
            kit.copy_out(s, sources_out, s_count)
        kit.close()
        return it
    except error:
        kit.close()
        raise error^


def ica_project_run[E: Exec, D: RDev, RES: Bool](
    x: F32Ptr, n: Int, width: Int, inverse: Bool, matrix: F32Ptr, rows: Int, mean: F32Ptr, whiten: Bool,
    out: F32Ptr,
) raises:
    """transform: (X - mean_) components_^T (rows x width components); or
    inverse_transform: X mixing_^T (+ mean_), mixing_ rows x width."""
    var kit = RKit[E, D, RES]()
    try:
        var m = kit.add(Int(x), -1, n, width)
        var b = kit.add(Int(matrix), -1, rows, width)
        var y: Int
        if not inverse:
            if whiten:
                var mu = kit.add(Int(mean), -1, 1, width)
                m = kit.ew(OP_SUB, m, mu)
            y = kit.mm(m, b, tb=True)
        else:
            y = kit.mm(m, b, tb=True)
            if whiten:
                var mu = kit.add(Int(mean), -1, 1, rows)
                y = kit.ew(OP_ADD, y, mu)
        var y_count = kit.r(y) * kit.c(y)
        kit.copy_out(y, out, y_count)
        kit.close()
    except error:
        kit.close()
        raise error^
