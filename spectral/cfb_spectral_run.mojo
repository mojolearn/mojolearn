# SPDX-License-Identifier: Apache-2.0
"""Typed shape checks and dense-entry refusal codes of the spectral clustering and embedding doors (RUN).
The door (`bindings/cfb_spectral_door.mojo`) admits the inputs and publishes the results."""

# The dense entries' refusal codes (moved with _sc_refusal).
comptime _DENSE_EMPTY = -1
comptime _DENSE_NONFINITE = -2
comptime _DENSE_NEGATIVE = -3


def spectral_lanczos_fits(eigenpairs: Int, n: Int) -> Bool:
    """Whether the Lanczos solver takes `eigenpairs` of an n x n Laplacian:
    its basis needs `eigenpairs + 1 < ncv <= n`
    (spectral/impl/sparse/solver/detail/lanczos.mojo `lanczos_smallest`, the
    host oracle's twin in spectral/host/spectral_oracle.mojo), so `n` must
    exceed `eigenpairs + 1`. The ncv choice (`compute_eigenpairs_keep_device`)
    takes the full space, ncv = n, whenever that holds. THE ONE SHAPE RULE of
    the class refusals and the device routes' decline (lane gs-fix-6,
    2026-11-26): they tested `eigenpairs < n`, so n = eigenpairs + 1 (4 rows,
    n_components 2 with drop_first) passed them and the solver refused in its
    internal words, and the device affinity route
    (bindings/cfb_spectral_door.mojo `_se_device_affinity`) ran its kernel
    instead of declining to the former route's order. Size reasoning only: the
    same rule for every n, k and device."""
    return eigenpairs >= 1 and eigenpairs + 1 < n


def _sc_check_shape(clusters: Int, n: Int, k: Int) raises:
    if clusters > n:
        raise Error("mojolearn SpectralClustering: n_clusters=" + String(clusters) + " "
                    + "exceeds n_samples=" + String(n))
    if not spectral_lanczos_fits(k, n):
        raise Error("mojolearn SpectralClustering: n_components=" + String(k) + " must be "
                    + "below n_samples - 1=" + String(n - 1) + "; the Lanczos requires "
                    + "1 <= n_components and n_components + 1 < n_samples (its basis "
                    + "holds n_components + 1 vectors and more), got n_samples=" + String(n))


def _sc_refusal(n_out: Int, who: String) raises:
    """The dense entries' refusal codes, in the COO route's wording."""
    if n_out == _DENSE_EMPTY:
        raise Error("mojolearn " + who + ": the precomputed affinity "
                    + "matrix has no nonzero entries")
    if n_out == _DENSE_NONFINITE:
        if who == "SpectralClustering":
            raise Error("mojolearn SpectralClustering: the precomputed affinity "
                        + "matrix has a non-finite entry (the implemented path refuses "
                        + "it by name, because sqrt of a non-finite degree is a "
                        + "NaN and no NaN may reach a recorded value)")
        raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                    + "matrix has a non-finite entry")
    if n_out == _DENSE_NEGATIVE:
        if who == "SpectralClustering":
            raise Error("mojolearn SpectralClustering: the precomputed affinity "
                        + "matrix has a negative entry (refused by name: sqrt of a "
                        + "negative degree is a NaN in cuVS too)")
        raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                    + "matrix has a negative entry (refused by name: sqrt of a "
                    + "negative degree is a NaN)")


def _se_check_shape(components: Int, n: Int, n_lanczos: Int) raises:
    if not spectral_lanczos_fits(n_lanczos, n):
        raise Error("mojolearn SpectralEmbedding: n_components=" + String(components) + " "
                    + "needs " + String(n_lanczos) + " eigenpairs, which must be below "
                    + "n_samples - 1=" + String(n - 1) + " (the Lanczos basis holds one more "
                    + "vector than the eigenpairs and needs room past it), got n_samples=" + String(n))
