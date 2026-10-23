# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`GaussianRandomProjection` / `SparseRandomProjection` fit, transform and
inverse_transform as ONE native operation each (lane mca2-decomp-a,
2026-10-22): the steps `_expansion_decomp._RandomProjection.fit`,
`_fused_fit`, `_make`, `_rp_rand`, `_pinv_rows`, `transform` / `_project`
and `inverse_transform` sequenced from Python with the decomp kit, on `RKit`
(x_decomp/decomp_owner_kit.mojo): the same entries' typed bodies, Philox
streams and kinds, float32 scalars, the route the binding's switches chose
(`DecompDev.grp_flags`: the FAST Apple fused fit and SRP strata), the
resident draws and the tiled projection where the Python kit used them.
The LAZY switch's deferred download is the same words downloaded at fit
(the state between calls is the published matrix)."""
from std.math import sqrt

from x_decomp.api import rand_run
from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev, RKit
from x_decomp.decomp_owner_kit import DecompDev, OP_SCALE, OP_SELECT, kit_pinv_rows

#: the projection's rejection threshold offset (`2.0 ** -25`)
comptime _SRP_EPS: Float64 = 2.9802322387695312e-08


def _rp_rand[E: Exec, D: RDev, RES: Bool](
    mut kit: RKit[E, D, RES], rows: Int, cols: Int, seed: Int, stream: Int, kind: Int
) raises -> Int:
    """`_rp_rand(k, rows, cols, seed, stream, kind, dev)`: drawn into a device
    matrix on a GPU binding (every size), else on the host."""
    var count = rows * cols
    var s = seed & 0xFFFFFFFF
    var st = stream & 0xFFFFFFFF
    comptime if RES:
        if count:
            var out = kit.device_output(rows, cols)
            D.rand(kit.mats[out].dev, count, s, st, kind)
            return out
    var out = kit.zeros(rows, cols)
    if count:
        rand_run[E](kit.ptr(out), count, s, st, kind)
    return out


def srp_grid_count(thr: Float64) -> Int:
    """`_srp_grid_count(thr)`: the uniforms k 2^-24 at or below float32(thr)."""
    var t32 = Float64(Float32(thr))
    if t32 < 0:
        return 0
    return min(Int(t32 * 16777216.0) + 1, 16777216)


def rp_fit_run[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    x: F32Ptr, n: Int, d: Int, kc: Int, sparse: Bool, density: Float64, seed: Int, inverse: Bool, fused_ok: Bool,
    components_out: F32Ptr, inverse_out: F32Ptr,
) raises -> Int:
    """The projection matrix (kc x d) and, with `inverse`, its
    pseudo-inverse (d x kc). `fused_ok`: n_components is an int, no inverse
    and a dense X, so a FUSED build takes `_fused_fit` (X scanned for a
    non-finite value there). Returns the first non-finite index the fused
    scan found (or -1)."""
    var flags = X.grp_flags()
    var kit = RKit[E, D, RES]()
    try:
        var fused = False
        comptime if RES:
            fused = (flags & 8) != 0 and fused_ok
        if fused:
            var mode = 0
            var sc = 1.0 / sqrt(Float64(kc))
            var thr = 0.0
            var sd = 1
            var sgrid = 0
            if sparse:
                mode = 2 if density == 1.0 else 1
                sc = sqrt(1.0 / density) / sqrt(Float64(kc))
                thr = density - _SRP_EPS
                if mode == 1 and (flags & 16) != 0:
                    mode = 3
                    sd = d
                    sgrid = srp_grid_count(thr)
            var c = kit.device_output(kc, d)
            var bad = X.grp_fit_fused(x, n * d, kit.mats[c].dev, components_out, kc * d, seed & 0xFFFFFFFF, mode,
                                      sd, sgrid, sc, thr)
            kit.close()
            return bad
        var comps: Int
        if not sparse:
            var drawn = _rp_rand(kit, kc, d, seed, 1, 1)
            comps = kit.ew(OP_SCALE, drawn, scalar=1.0 / sqrt(Float64(kc)))
        else:
            var u = _rp_rand(kit, kc, d, seed, 2, 0)
            var signs = _rp_rand(kit, kc, d, seed, 3, 2)
            var sgn = kit.ew(OP_SCALE, signs, scalar=sqrt(1.0 / density) / sqrt(Float64(kc)))
            if density == 1.0:
                comps = sgn
            else:
                # u < density keeps the signed value, else 0 (select: x > s -> y else z)
                var zero = kit.zeros(1, 1)
                comps = kit.ew(OP_SELECT, u, zero, sgn, scalar=density - _SRP_EPS)
        kit.copy_out(comps, components_out, kc * d)
        if inverse:
            var inv = kit_pinv_rows[E, D, RES, X](kit, comps)
            kit.copy_out(inv, inverse_out, d * kc)
        kit.close()
        return -1
    except error:
        kit.close()
        raise error^


def rp_project_run[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    x: F32Ptr, m: Int, d: Int, matrix: F32Ptr, rows: Int, tiled: Bool, out: F32Ptr,
) raises -> Int:
    """X (m x d) times matrix^T (matrix rows x d) into out (m x rows):
    with `tiled` (transform on a dense X) the tiled projection on a GPU
    binding (`_project`), else `_Kit.mm(M, B, tb=True)`. Returns 1 when the
    tiled projection flagged a non-finite X, else 0."""
    var kit = RKit[E, D, RES]()
    try:
        var a = kit.add(Int(x), -1, m, d)
        var b = kit.add(Int(matrix), -1, rows, d)
        comptime if RES:
            if tiled:
                var c = kit.device_output(m, rows)
                var flag = kit.device_output(1, 1)
                var da = kit.device(a)
                var db = kit.device(b)
                if X.dev_project(da, db, kit.mats[c].dev, kit.mats[flag].dev, m, d, rows) >= 0:
                    kit.copy_out(c, out, m * rows)
                    var bad = kit.word(flag)
                    kit.close()
                    return 1 if bad != 0 else 0
        var y = kit.mm(a, b, tb=True)
        kit.copy_out(y, out, m * rows)
        kit.close()
        return 0
    except error:
        kit.close()
        raise error^
