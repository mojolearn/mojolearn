# SPDX-License-Identifier: Apache-2.0
# RUN part of the x_cluster fit door (MiniBatchKMeans fit and partial_fit,
# BisectingKMeans, MeanShift, OPTICS, AffinityPropagation): typed Mojo only
# (no Python imports). UNPACK is bindings/cluster_fit_public_native_unpack.mojo,
# PACK and the operator are bindings/cluster_fit_public_native_pack.mojo.
# Every refusal that is a comparison on numbers is here, on Mojo values, with
# the public wording; UNPACK calls it at the place the compare stood, so the
# refusal order is unchanged. The numerical work is the typed driver call of
# core/kmeans_variants_api.mojo, core/cluster_centroid_fit_api.mojo and
# core/optics_fit_api.mojo; this file builds their requests. Every
# iteration, batch, bisection and seed loop runs inside the entry.
from core.kmeans_variants_api import VariantsSpan
from core.cluster_centroid_fit_api import MeanShiftFitRequest


# ------------------------------------------------------------------ shared
def cluster_rows_cover(k: Int, n: Int) raises:
    """`n_clusters` is in [1, n_samples]."""
    if k < 1 or k > n:
        raise Error("n_samples=" + String(n) + " should be >= n_clusters=" + String(k) + ".")


# --------------------------------------------------------------- MeanShift
def meanshift_seed_features(seed_features: Int, d: Int) raises:
    """Explicit seeds have X's feature count."""
    if seed_features != d:
        raise Error("seeds have " + String(seed_features) + " features, X has " + String(d))


def meanshift_bandwidth_refused(bandwidth: Float64) -> Bool:
    """An explicit bandwidth that is not > 0 (a NaN is refused too)."""
    return not bandwidth > 0


def meanshift_fit_request(source: VariantsSpan, seeds: VariantsSpan, n: Int, d: Int, n_seeds: Int,
                          cluster_all: Bool, max_iter: Int, bin_seeding: Bool, min_bin_freq: Int,
                          bandwidth: Float64) -> MeanShiftFitRequest:
    """ORDER MATCHES x_cluster/entries.mojo::meanshift_entry: n, d, n_seeds,
    cluster_all, max_iter, bin_seeding, min_bin_freq; the bandwidth (0.0:
    estimate it)."""
    return MeanShiftFitRequest(source, seeds, n, d, n_seeds, cluster_all, max_iter, bin_seeding,
                               min_bin_freq, bandwidth)
