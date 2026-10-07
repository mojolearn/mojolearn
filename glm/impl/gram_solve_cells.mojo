# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The scalar cells of MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE shared by the
device (glm/impl/gram_solve.mojo) and the host column
(glm/host/gram_solve_host.mojo). No GPU import: the CPU binding imports
this file. The power-of-two equilibration scale is passed in by the caller
(`ols_equilibration_scale` on the device, its documented copy
`host_equilibration_scale` on the host: the same bits)."""
from checks.numerics import ftz, identical_mul
from x_decomp.cells import F32Ptr

#: x_linear/ridge.mojo RIDGE_FF_GATE: a pivot square below this fraction of
#: its (equilibrated) diagonal is not trusted in float32.
comptime GS_TRUST_GATE = Float32(0.000244140625)  # 2^-12


@always_inline
def gs_regularized_diag(g_jj: Float32, alpha: Float32) -> Float32:
    """G[j, j] + alpha, the value the equilibration scale is taken of."""
    return ftz(ftz(g_jj) + ftz(alpha))


@always_inline
def gs_equilibrated_cell(g: F32Ptr, i: Int, j: Int, d: Int, alpha: Float32, si: Float32, sj: Float32) -> Float32:
    """A[i, j] = s_i (G[i, j] + alpha [i == j]) s_j; the scales are exact
    powers of two, so the only roundings are the sum and the flushes."""
    var v = ftz(g.unsafe_load(i * d + j))
    if i == j:
        v = ftz(v + ftz(alpha))
    return ftz(identical_mul(ftz(identical_mul(si, v)), sj))


@always_inline
def gs_scaled_rhs(c: F32Ptr, i: Int, si: Float32) -> Float32:
    """b_i = s_i c_i."""
    return ftz(identical_mul(si, ftz(c.unsafe_load(i))))


@always_inline
def gs_pivot_trusted(l_jj: Float32, a_jj: Float32) -> Bool:
    """`l_jj^2 >= GS_TRUST_GATE * a_jj` (x_linear/ridge.mojo `chol_trusted`'s
    test, on the equilibrated diagonal)."""
    return not (ftz(identical_mul(l_jj, l_jj)) < ftz(identical_mul(ftz(a_jj), GS_TRUST_GATE)))
