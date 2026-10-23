# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`TruncatedSVD.fit`'s two decomp-kit routes as ONE native operation each
(lane mca2-decomp-a, 2026-10-22), the steps `_expansion_decomp.
_randomized_decompose(center=False)` and `_linalg_impl._tsvd_tsqr_components`
sequenced from Python with the decomp kit (`_Kit`), one x_decomp call each.

SAME CALLS, SAME ORDER, SAME OPERANDS. Both drivers run on the randomized
PCA kit (`RKit`, x_decomp/pca_randomized.mojo), which keeps every matrix
where the Python kit kept it and reaches each step through the typed body
of the entry the Python kit called (`x_decomp_ew`, `x_decomp_dev_colsum`,
`x_decomp_dev_svd`, ...), with the same broadcast modes, float32 scalars,
Philox seed and stream, power iterations, orth rank guard, paddings and
`svd_flip` sign rule. The caller's X is read in place and never written.

`tsvd_randomized_fit_run`: algorithm='randomized'; `_rsvd_core` is
`pca_randomized_core` (shared with PCA), then svd_flip(u_based_decision=
False) and scikit-learn TruncatedSVD's explained variance (np.var of
U * S per column, ddof 0, against the summed column variances of X).

`tsvd_tsqr_components_run`: the FAST TSVD_QFIX route (x_decomp/qfix.mojo
bit 2): R of the blocked TSQR of X (`tsqr_r_run`, nothing kept), the
one-sided Jacobi SVD of R sorted descending (`RKit.tall_svd`, `_Kit.svd`),
the first rows of Vt each signed so its largest-|.| entry is positive."""
from x_decomp.api import tsqr_r_run
from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev, RKit, pca_randomized_core

# x_decomp/cells.mojo op codes (`_OP` in _expansion_decomp.py)
comptime _OP_MUL = 2
comptime _OP_DIV = 3
comptime _OP_SCALE = 14
comptime _OP_SQDIFF = 21


def tsvd_randomized_fit_run[E: Exec, D: RDev, RES: Bool](
    x: F32Ptr, components_out: F32Ptr, singular_out: F32Ptr, variance_out: F32Ptr, ratio_out: F32Ptr,
    rows: Int, cols: Int, components: Int, oversamples: Int, iterations: Int, auto_iterations: Bool, seed: Int,
) raises:
    """`_randomized_decompose(X, components, center=False, ...)`: the four
    public arrays written (components x cols, then components each).
    `auto_iterations`: n_iter='auto' (scikit-learn's size-ratio rule)."""
    if components < 1 or components > min(rows, cols):
        raise Error("n_components must be in [1, min(n_samples, n_features)]")
    var power = iterations
    if auto_iterations:
        # `_rsvd_core`'s n_iter='auto', by the problem's size ratio (not a
        # dimension threshold)
        power = 7 if Float64(components) < 0.1 * Float64(min(rows, cols)) else 4
    var kit = RKit[E, D, RES]()
    try:
        var original = kit.add(Int(x), -1, rows, cols)
        # k.colmean(M): the column sums scaled by 1 / rows
        var sums = kit.sum(original, True)
        var mean = kit.ew(_OP_SCALE, sums, scalar=1.0 / Float64(rows))
        var decomposed = pca_randomized_core(kit, original, components, oversamples, power, seed)
        var u = decomposed[0]
        var s = decomposed[1]
        var vt = decomposed[2]
        # svd_flip(u_based_decision=False): each row of Vt, U's columns follow
        var signs = kit.absmax_signs(vt, False)
        vt = kit.ew(_OP_MUL, vt, signs)
        var signst = kit.transpose(signs)
        u = kit.ew(_OP_MUL, u, signst)
        # Xt = U * S; ev = colsum((Xt - colmean(Xt))^2) / n
        var xt = kit.ew(_OP_MUL, u, s)
        var xt_sums = kit.sum(xt, True)
        var xt_rows = kit.r(xt)
        var xt_mean = kit.ew(_OP_SCALE, xt_sums, scalar=1.0 / Float64(xt_rows))
        var spread = kit.ew(_OP_SQDIFF, xt, xt_mean)
        var spread_sums = kit.sum(spread, True)
        var variance = kit.ew(_OP_SCALE, spread_sums, scalar=1.0 / Float64(rows))
        # full = total(colsum((M - mean)^2) / n); ratio = ev / full
        var data_spread = kit.ew(_OP_SQDIFF, original, mean)
        var data_sums = kit.sum(data_spread, True)
        var data_variance = kit.ew(_OP_SCALE, data_sums, scalar=1.0 / Float64(rows))
        var full_value = kit.total(data_variance)
        # the float32 total read back is exact: the 1 x 1 operand holds it
        var full = kit.constant(Float32(full_value))
        var ratio = kit.ew(_OP_DIV, variance, full)
        kit.copy_out(vt, components_out, components * cols)
        kit.copy_out(s, singular_out, components)
        kit.copy_out(variance, variance_out, components)
        kit.copy_out(ratio, ratio_out, components)
        kit.close()
    except error:
        kit.close()
        raise error^


def tsvd_tsqr_components_run[E: Exec, D: RDev, RES: Bool](
    x: F32Ptr, components_out: F32Ptr, singular_out: F32Ptr, rows: Int, cols: Int, components: Int,
) raises:
    """`_tsvd_tsqr_components(x, components, mode)` on a TSQR shape: the
    components (components x cols) and singular values (components)."""
    if components < 1 or components > cols:
        raise Error("x_decomp: TruncatedSVD TSQR route needs 1 <= n_components <= n_features")
    var r_words = List[Float32](length=cols * cols, fill=Float32(0))
    var r_ptr = r_words.unsafe_ptr()
    # ORDER MATCHES `_tsqr_r`: (a, b, r_out), params (m, d, nrhs 0, keep 0)
    _ = tsqr_r_run[E](x, x, F32Ptr(unsafe_from_address=Int(r_ptr)), rows, cols, 0, False)
    var kit = RKit[E, D, RES]()
    try:
        var r = kit.add(Int(r_ptr), -1, cols, cols)
        var decomposed = kit.tall_svd(r)
        var s = decomposed[0]
        var vt = decomposed[1]
        # Vt.rows(0, components), signed by its rows' largest-|.| entries
        var kept_rows = kit.indices(0, components)
        var v = kit.take_rows(vt, kept_rows)
        var signs = kit.absmax_signs(v, False)
        v = kit.ew(_OP_MUL, v, signs)
        var kept_s = kit.columns(s, 0, components)
        kit.copy_out(v, components_out, components * cols)
        kit.copy_out(kept_s, singular_out, components)
        kit.close()
    except error:
        kit.close()
        raise error^
    _ = r_words^
