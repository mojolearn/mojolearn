# SPDX-License-Identifier: Apache-2.0
"""Typed shape checks and dense-entry refusal codes of the spectral clustering and embedding doors (RUN).
The door (`bindings/cfb_spectral_door.mojo`) admits the inputs and publishes the results."""

# The dense entries' refusal codes (moved with _sc_refusal).
comptime _DENSE_EMPTY = -1
comptime _DENSE_NONFINITE = -2
comptime _DENSE_NEGATIVE = -3


def spectral_eigenpairs_fit(eigenpairs: Int, n: Int) -> Bool:
    """The fitted embedding needs fewer eigenpairs than rows, including the
    trivial eigenvector if the caller drops it."""
    return 1 <= eigenpairs and eigenpairs < n


def _sc_check_shape(clusters: Int, n: Int, k: Int) raises:
    if clusters > n:
        raise Error("mojolearn SpectralClustering: n_clusters=" + String(clusters) + " "
                    + "exceeds n_samples=" + String(n))
    if not spectral_eigenpairs_fit(k, n):
        raise Error("mojolearn SpectralClustering: n_components=" + String(k) + " must be "
                    + "at least 1 and below n_samples=" + String(n))


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
    if not spectral_eigenpairs_fit(n_lanczos, n):
        raise Error("mojolearn SpectralEmbedding: n_components=" + String(components) + " "
                    + "needs " + String(n_lanczos) + " eigenpairs, which must be below "
                    + "n_samples=" + String(n))
