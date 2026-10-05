# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The scalar steps of one Lanczos step (lane cpu2-l8-decomp, 2026-10-04),
shared by the device route (x_decomp/lanczos_dev.mojo) and its host twin
(x_decomp/lanczos_host.mojo) so the two give the same words: every value
through the cells' flush-to-zero arithmetic (x_decomp/cells.mojo `add`,
`sub`, `mul`, `div0`, `sqrt0`, correctly rounded under IDENTICAL), the
products through the kit's gemm. No device code: both bindings import it."""
from x_decomp.cells import add, div0, mul, sqrt0, sub
from checks.numerics import ftz


@always_inline
def lz_alpha(prev: Float32, c: Float32, second: Bool) -> Float32:
    """alpha_j after a Gram-Schmidt pass: c[j] on the first pass, the
    running alpha plus c[j] on the second (float32, one rounding)."""
    if second:
        return add(prev, c)
    return ftz(c)


@always_inline
def lz_w(w: Float32, t: Float32) -> Float32:
    """w - Q^T c, one value."""
    return sub(w, t)


@always_inline
def lz_beta(dot: Float32) -> Float32:
    """beta_j = sqrt(max(w . w, 0)) (0 for a NaN dot)."""
    return sqrt0(dot)


@always_inline
def lz_inv(beta: Float32, alpha: Float32) -> Float32:
    """1 / beta_j, or 0 at a breakdown (beta <= 1e-30 max(1, |alpha|)),
    which the host finds in the betas and stops at."""
    var a = abs(ftz(alpha))
    var m = a if a > Float32(1) else Float32(1)
    if beta > mul(Float32(1e-30), m):
        return div0(Float32(1), beta)
    return Float32(0)


@always_inline
def lz_q(w: Float32, s: Float32) -> Float32:
    """q_{j+1} = w / beta as w times 1 / beta (the kit's `scale` cell)."""
    return mul(w, s)
