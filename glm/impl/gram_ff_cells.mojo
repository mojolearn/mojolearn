# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane fg-linear L3 (default ON 2026-10-09, -D MOJOLEARN_IDN_GRAM_FF_FALLBACK_OFF restores the eig fallback): the
float-float cells of the resident Gram solve's second chance. Shared by the
device kernels (glm/impl/gram_solve.mojo) and the host column
(glm/host/gram_solve_host.mojo); no GPU import.

When the float32 Gram fails its trust gate (cond(S (G + alpha I) S) above
~2^12), a Ridge fit re-forms the centered Gram G and cross c in float-float
(x_linear/ff.mojo: every operation the lane's IDENTICAL float32 arithmetic in
the classical error-free compositions, ~48 bits), equilibrates by the same
powers of two, factors by a float-float Cholesky and solves, all on the
device, no Jacobi. The design is read once more (one pass over X).

THE ORDER (the bits). Rows are cut into GFF_LEAVES leaves of
ceil(n / GFF_LEAVES) rows (a fixed count, not a data shape: 64 leaves keep
the partial buffer at 64 (d^2 + d) float pairs and give 64 d^2 / 2 parallel
cells); each leaf cell folds its rows ascending; the leaves fold ascending.
The Cholesky, the forward and the backward solve run one column step at a
time, each cell's inner sum ascending (`chol_serial`'s order, the fp32 route's
statements in float-float). The host column calls the same cells in the same
order, so device and host agree word for word.
"""
from checks.numerics import ftz, identical_mul
from checks.soft_f64 import sf64_from_f32, sf64_sub, sf64_to_f32
from x_linear.ff import FF, ff_add, ff_add_f, ff_centered, ff_div, ff_f32, ff_ld, ff_mul, ff_mul_f, ff_sqrt, ff_st, ff_sub
from x_linear.ops import FP, ld, st

#: The leaf count of the float-float Gram (module docstring).
comptime GFF_LEAVES = 64

#: A float-float pivot square below this fraction of its equilibrated
#: diagonal is not trusted: float-float carries ~48 bits, so the gate keeps
#: >= 24 bits after the cancellation, the float32 gate's 12-of-24 rule
#: (GS_TRUST_GATE = 2^-12) restated at twice the width.
comptime GFF_TRUST_GATE = Float32(5.9604644775390625e-08)  # 2^-24


@always_inline
def gff_leaf_rows(n: Int) -> Int:
    var r = (n + GFF_LEAVES - 1) // GFF_LEAVES
    return r if r > 0 else 1


@always_inline
def gff_cells(d: Int) -> Int:
    """Partial cells per leaf: the d x d Gram (upper triangle formed, the
    lower left zero) then the d cross cells."""
    return d * d + d


@always_inline
def gff_mean_split(m64: UInt64) -> FF:
    """A binary64 mean as hi + lo float32 words (the narrowing rounded to
    nearest, the remainder exact in binary64, then narrowed), by the soft
    binary64 seams so every column forms the same two words."""
    var hi = ftz(sf64_to_f32(m64))
    var lo = ftz(sf64_to_f32(sf64_sub(m64, sf64_from_f32(hi))))
    return FF(hi, lo)


def gff_leaf_cell(x: FP, y: FP, mh: FP, ml: FP, n: Int, d: Int, leaf: Int, q: Int) -> FF:
    """Leaf `leaf`'s float-float partial of cell q: q < d^2 is Gram cell
    (i, j) = sum (x_ri - mu_i)(x_rj - mu_j) (zero when i > j), q >= d^2 the
    cross cell j = sum (x_rj - mu_j)(y_r - ybar). Rows ascending."""
    var lr = gff_leaf_rows(n)
    var r0 = leaf * lr
    var r1 = r0 + lr
    if r1 > n:
        r1 = n
    var acc = FF(Float32(0), Float32(0))
    if q < d * d:
        var i = q // d
        var j = q - i * d
        if i > j:
            return acc
        var mi = ff_ld(mh, ml, i)
        var mj = ff_ld(mh, ml, j)
        for r in range(r0, r1):
            var ci = ff_centered(ld(x, r * d + i), mi)
            var cj = ff_centered(ld(x, r * d + j), mj)
            acc = ff_add(acc, ff_mul(ci, cj))
    else:
        var j = q - d * d
        var mj = ff_ld(mh, ml, j)
        var my = ff_ld(mh, ml, d)
        for r in range(r0, r1):
            var cj = ff_centered(ld(x, r * d + j), mj)
            var cy = ff_centered(ld(y, r), my)
            acc = ff_add(acc, ff_mul(cj, cy))
    return acc


def gff_fold_cell(ph: FP, pl: FP, gh: FP, gl: FP, ch: FP, cl: FP, d: Int, q: Int):
    """Cell q's leaves folded ascending; Gram cells (i <= j) stored at (i, j)
    and (j, i), cross cells into c. Cells i > j store nothing."""
    var t = gff_cells(d)
    if q < d * d:
        var i = q // d
        var j = q - i * d
        if i > j:
            return
        var acc = FF(Float32(0), Float32(0))
        for leaf in range(GFF_LEAVES):
            acc = ff_add(acc, ff_ld(ph, pl, leaf * t + q))
        ff_st(gh, gl, i * d + j, acc)
        ff_st(gh, gl, j * d + i, acc)
    else:
        var acc = FF(Float32(0), Float32(0))
        for leaf in range(GFF_LEAVES):
            acc = ff_add(acc, ff_ld(ph, pl, leaf * t + q))
        ff_st(ch, cl, q - d * d, acc)


@always_inline
def gff_equilibrated_cell(gh: FP, gl: FP, i: Int, j: Int, d: Int, alpha: Float32, si: Float32, sj: Float32) -> FF:
    """A[i, j] = s_i (G[i, j] + alpha [i == j]) s_j in float-float; the
    scales are exact powers of two (`ols_equilibration_scale` of the float32
    rounding of the regularized diagonal, the fp32 route's scales)."""
    var v = ff_ld(gh, gl, i * d + j)
    if i == j:
        v = ff_add_f(v, ftz(alpha))
    return ff_mul_f(ff_mul_f(v, si), sj)


@always_inline
def gff_regularized_diag_f32(gh: FP, gl: FP, i: Int, d: Int, alpha: Float32) -> Float32:
    """The float32 value the equilibration scale is taken of: (G_ii + alpha)
    rounded once."""
    return ff_f32(ff_add_f(ff_ld(gh, gl, i * d + i), ftz(alpha)))


def gff_chol_cell(ah: FP, al: FP, adh: FP, adl: FP, info: FP, i: Int, j: Int, d: Int):
    """Column step j, row i >= j, of the float-float Cholesky: every row
    re-forms the pivot chain from the untouched equilibrated diagonal copy
    (adh, adl), inner sums ascending; row j stores the pivot root (and
    raises info on a non-positive pivot), rows i > j store L[i, j] and zero
    the upper cell (j, i). Within a step no cell is written that another row
    of the step reads, so the device runs the rows in parallel and the host
    in order with the same words."""
    var piv = ff_ld(adh, adl, j)
    for p in range(j):
        var l = ff_ld(ah, al, j * d + p)
        piv = ff_sub(piv, ff_mul(l, l))
    if not (piv.hi > Float32(0)):
        if i == j:
            st(info, 0, Float32(1))
        piv = FF(Float32(1), Float32(0))
    var r = ff_sqrt(piv)
    if i == j:
        ff_st(ah, al, j * d + j, r)
        return
    var s = ff_ld(ah, al, i * d + j)
    for p in range(j):
        s = ff_sub(s, ff_mul(ff_ld(ah, al, i * d + p), ff_ld(ah, al, j * d + p)))
    ff_st(ah, al, i * d + j, ff_div(s, r))
    st(ah, j * d + i, Float32(0))
    st(al, j * d + i, Float32(0))


@always_inline
def gff_pivot_trusted(lh_jj: Float32, adh_j: Float32) -> Bool:
    """`l_jj^2 >= GFF_TRUST_GATE * a_jj` on the hi words."""
    return not (ftz(identical_mul(lh_jj, lh_jj)) < ftz(identical_mul(ftz(adh_j), GFF_TRUST_GATE)))


def gff_forward_cell(lh: FP, ll: FP, bh: FP, bl: FP, zh: FP, zl: FP, i: Int, j: Int, d: Int):
    """Forward solve L z = b, column step j, row i >= j."""
    var zj = ff_div(ff_ld(bh, bl, j), ff_ld(lh, ll, j * d + j))
    if i == j:
        ff_st(zh, zl, j, zj)
    else:
        ff_st(bh, bl, i, ff_sub(ff_ld(bh, bl, i), ff_mul(ff_ld(lh, ll, i * d + j), zj)))


def gff_backward_cell(lh: FP, ll: FP, zh: FP, zl: FP, wh: FP, wl: FP, i: Int, j: Int, d: Int):
    """Backward solve L^T w = z, column step j (descending), row i <= j."""
    var wj = ff_div(ff_ld(zh, zl, j), ff_ld(lh, ll, j * d + j))
    if i == j:
        ff_st(wh, wl, j, wj)
    else:
        ff_st(zh, zl, i, ff_sub(ff_ld(zh, zl, i), ff_mul(ff_ld(lh, ll, j * d + i), wj)))


@always_inline
def gff_coef(wh: FP, wl: FP, sv: FP, i: Int) -> Float32:
    """coef_i = s_i w'_i (exact power-of-two scale), rounded once to float32."""
    return ff_f32(ff_mul_f(ff_ld(wh, wl, i), ld(sv, i)))
