# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The diffusion map's own arithmetic (lane sc-graph-1, 2026-11-28; Scanpy
`tl.diffmap`, `Neighbors.compute_transitions` with `density_normalize`, then
`compute_eigen` on the symmetric transitions), shared word for word by the
device arm (`spectral/impl/preprocessing/detail/diffmap.mojo`) and the host
column (`spectral/host/spectral_oracle.mojo`), so the two agree bit for bit.
No GPU import: the host binding compiles this file.

Scanpy forms K = Q W Q with Q = diag(1 / q), q the degrees of W (the
density normalization), then T = Z K Z with Z = diag(1 / sqrt(rowsum K)),
and takes the top eigenpairs of T. T is I - L_sym(K), the identity minus
the normalized Laplacian of K, so its eigenvectors are the Laplacian's and
its eigenvalues 1 - lambda: the spectral embedding's Laplacian and Lanczos
on K are the eigen solver (the largest eigenpairs of -L_sym(K), Ritz value
r = -lambda, eigenvalue of T = 1 + r)."""
from checks.numerics import ftz, identical_div, identical_mul


@always_inline
def diffmap_density_value(w: Float32, qr: Float32, qc: Float32) -> Float32:
    """K's entry w / (q_r q_c): one pinned product, one pinned quotient,
    each flushed. A zero denominator (an explicit zero weight on an
    isolated node) gives 0, not Scanpy's NaN."""
    var den = ftz(identical_mul(qr, qc))
    if den == Float32(0):
        return Float32(0)
    return ftz(identical_div(w, den))


@always_inline
def diffmap_eigenvalue(ritz: Float32) -> Float32:
    """The eigenvalue of T from the Ritz value of -L_sym(K): 1 + r, flushed."""
    return ftz(Float32(1) + ritz)
