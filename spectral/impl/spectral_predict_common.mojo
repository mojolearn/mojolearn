# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What the spectral predict's device pass and its host column share: the
kept state, the result, the constants and the refusals
(`spectral/host/spectral_predict_host.mojo` has the rule). No host compute
and no host-module import, so a GPU binding can take these without reaching
the host column (lane cgfin-c-cluster, 2026-10-02)."""


#: DEVIATION 2860: a used column with `|1 + theta_c|` below this is refused.
comptime SPECTRAL_PREDICT_MIN_ABS_EIGENVALUE = Float32(1e-3)

#: The weight of a query's edge to one of its k nearest training rows: the
#: fit's `0.5 * (1.0 + 0.0)` for an edge with no reverse edge.
comptime SPECTRAL_PREDICT_ONE_WAY_EDGE = Float32(0.5)

comptime SPECTRAL_AFFINITY_NEAREST_NEIGHBORS = 0
comptime SPECTRAL_AFFINITY_PRECOMPUTED = 1


struct SpectralPredictionState(Movable):
    """What predict needs from the fit; see the module docstring."""

    var eigenvalues: List[Float32]
    var eigenvectors: List[Float32]
    var diag: List[Float32]
    var centroids: List[Float32]

    def __init__(out self):
        self.eigenvalues = List[Float32]()
        self.eigenvectors = List[Float32]()
        self.diag = List[Float32]()
        self.centroids = List[Float32]()


@fieldwise_init
struct SpectralPrediction(Movable):
    var labels: List[Int32]
    var embedding: List[Float32]


def spectral_predict_validate(
    n_train: Int,
    n_queries: Int,
    n_features: Int,
    n_components: Int,
    n_clusters: Int,
    n_neighbors: Int,
    affinity: Int,
) raises:
    """The refusals both bindings raise, in one order and one wording."""
    if affinity != SPECTRAL_AFFINITY_NEAREST_NEIGHBORS and affinity != SPECTRAL_AFFINITY_PRECOMPUTED:
        raise Error(
            "spectral_predict: affinity must be 0 (nearest_neighbors) or 1"
            " (precomputed), got " + String(affinity)
        )
    if n_train < 2:
        raise Error("spectral_predict: the fitted model holds " + String(n_train) + " training rows")
    if n_queries < 1:
        raise Error("spectral_predict: X has no rows; refused by name")
    if n_components < 1 or n_components >= n_train:
        raise Error(
            "spectral_predict: n_components=" + String(n_components)
            + " must satisfy 1 <= n_components < n_train=" + String(n_train)
        )
    if n_clusters < 1 or n_clusters > n_train:
        raise Error("spectral_predict: n_clusters=" + String(n_clusters) + " is outside [1, n_train]")
    if affinity == SPECTRAL_AFFINITY_NEAREST_NEIGHBORS:
        if n_features < 1:
            raise Error("spectral_predict: X has no features; refused by name")
        if n_neighbors < 1 or n_neighbors > n_train:
            raise Error(
                "spectral_predict: n_neighbors=" + String(n_neighbors)
                + " must satisfy 1 <= n_neighbors <= n_train=" + String(n_train)
            )


def spectral_predict_check_state(
    state: SpectralPredictionState, n_train: Int, n_components: Int, n_clusters: Int
) raises:
    if (
        len(state.eigenvalues) != n_components
        or len(state.eigenvectors) != n_train * n_components
        or len(state.diag) != n_train
        or len(state.centroids) != n_clusters * n_components
    ):
        raise Error(
            "spectral_predict: the prediction data does not match n_train="
            + String(n_train) + ", n_components=" + String(n_components)
            + ", n_clusters=" + String(n_clusters) + "; refused by name"
        )
