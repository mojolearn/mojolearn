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
from core.kmeans_variants_api import VariantsSpan, BisectFitRequest, MiniBatchFitRequest, MiniBatchPartialRequest
from core.kmeans_variants_plan import MiniBatchStreamState
from std.math import isinf
from core.cluster_centroid_fit_api import MeanShiftFitRequest, AffinityFitRequest
from core.optics_fit_api import OpticsMetric, OpticsFitRequest


# ------------------------------------------------------------------ shared
def cluster_rows_cover(k: Int, n: Int) raises:
    """`n_clusters` is in [1, n_samples]."""
    if k < 1 or k > n:
        raise Error("n_samples=" + String(n) + " should be >= n_clusters=" + String(k) + ".")


def cluster_n_init_positive(n_init: Int) raises:
    if n_init < 1:
        raise Error("n_init must be >= 1")


# --------------------------------------------------------- MiniBatchKMeans
def minibatch_auto_n_init(has_init: Bool, init_random: Bool) -> Int:
    """n_init='auto': three restarts for init='random', one for
    'k-means++' or explicit centers."""
    return 3 if (not has_init and init_random) else 1


def minibatch_fit_request(source: VariantsSpan, auxiliary: VariantsSpan, n: Int, d: Int, k: Int,
                          max_iter: Int, batch_size: Int, max_no_improvement: Int, init_size: Int,
                          n_init: Int, has_init: Bool, seed: Int, init_random: Bool, has_weights: Bool,
                          scan_input: Bool, tolerance: Float64, reassignment_ratio: Float64) -> MiniBatchFitRequest:
    """The mini-batch fit's typed request (core/kmeans_variants_api.mojo
    `minibatch_fit` lays out its words). `auxiliary` holds the explicit
    initial centers, then the sample weights, end to end; the random init
    flag is set only when no centers were given."""
    return MiniBatchFitRequest(source, auxiliary, n, d, k, max_iter, batch_size, max_no_improvement, init_size,
                               n_init, has_init, seed, not has_init and init_random, has_weights, scan_input,
                               tolerance, reassignment_ratio)


def minibatch_partial_batch(batch_size: Int, n: Int) -> Int:
    """The effective batch of a first `partial_fit`: no more than X's rows."""
    return min(batch_size, n)


def minibatch_partial_init_size(explicit: Bool, init_size: Int, batch: Int, k: Int, n: Int) -> Int:
    """The init sample of a first `partial_fit`: the given `init_size`, or
    three batches; three per cluster when that is under `k`; no more than
    X's rows."""
    var size = init_size if explicit else 3 * batch
    if size < k:
        size = 3 * k
    return min(size, n)


def minibatch_seed_high(seed: Int) -> Int:
    """The random stream's high word of a 63-bit seed."""
    return seed >> 32


def minibatch_seed_low(seed: Int) -> Int:
    return seed & 0xFFFFFFFF


def minibatch_partial_request(source: VariantsSpan, auxiliary: VariantsSpan, n: Int, d: Int, k: Int,
                              first: Bool, state: MiniBatchStreamState, has_weights: Bool,
                              reassignment_ratio: Float64) -> MiniBatchPartialRequest:
    """One `partial_fit` step's typed request (core/kmeans_variants_api.mojo
    `minibatch_partial` lays out its words). `auxiliary` holds the explicit
    initial centers (first step) or the centers and counts (later steps),
    then the sample weights."""
    return MiniBatchPartialRequest(source, auxiliary, n, d, k, first, state.init_mode, state.init_size,
                                   state.batch_size, state.since_reassignment, state.state_hi, state.state_lo,
                                   has_weights, reassignment_ratio)


# ------------------------------------------------------------------ door
def cluster_fit_a_unknown_operation(operation: String) raises:
    """A `family_cfa_*` name the operation table does not hold (the shells
    pass table names only): the text the name chain raised."""
    raise Error("RuntimeError: unknown family operation " + operation)


def cluster_fit_unknown_code(operation: Int) raises:
    raise Error("RuntimeError: unknown family operation code " + String(operation))


# --------------------------------------------------------- BisectingKMeans
def bisect_fit_request(source: VariantsSpan, weights: VariantsSpan, n: Int, d: Int, k: Int, n_init: Int,
                       random_init: Bool, max_iter: Int, seed: Int, largest_cluster: Bool,
                       tolerance: Float64) -> BisectFitRequest:
    """The bisecting fit's typed request (core/kmeans_variants_api.mojo
    `bisect_fit` lays out its words)."""
    return BisectFitRequest(source, weights, n, d, k, n_init, random_init, max_iter, seed, largest_cluster,
                            tolerance)


def bisect_tree_rows(words: Int) -> Int:
    """The bisection tree's node count: three int32 words per node."""
    return words // 3


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


# ----------------------------------------------------- AffinityPropagation
def affinity_damping_refused(damping: Float64) -> Bool:
    """A damping outside [0.5, 1) (a NaN is refused too)."""
    return not (0.5 <= damping and damping < 1)


def affinity_iterations_positive(value: Int) raises:
    """max_iter, then convergence_iter: each at least 1."""
    if value < 1:
        raise Error("max_iter and convergence_iter must be >= 1")


def affinity_similarity_square(precomputed: Bool, n: Int, d: Int) raises:
    if precomputed and n != d:
        raise Error("The matrix of similarities must be a square array. Got (" + String(n) + ", " + String(d)
                    + ") instead.")


def affinity_preference_rows(count: Int, n: Int) raises:
    if count != n:
        raise Error("preference must be a scalar or have one value per sample")


def affinity_fit_request(source: VariantsSpan, preference: VariantsSpan, n: Int, d: Int, precomputed: Bool,
                         mode: Int, max_iter: Int, convergence_iter: Int, seed: Int, damping: Float64,
                         scalar: Float64) -> AffinityFitRequest:
    """ORDER MATCHES x_cluster/entries.mojo::affinity_entry: n, d,
    precomputed, preference mode (0 median, 1 scalar, 2 one per sample),
    max_iter, convergence_iter, seed; damping, the scalar preference."""
    return AffinityFitRequest(source, preference, n, d, precomputed, mode, max_iter, convergence_iter, seed,
                              damping, scalar)


# ------------------------------------------------------------------ OPTICS
# OPTICS metric codes (x_cluster/optics.mojo).
comptime OPTICS_PRECOMPUTED = 5


def optics_minkowski_refused(p: Float64) -> Bool:
    """A minkowski power that is not >= 1 (a NaN is refused too)."""
    return not p >= 1


def optics_minkowski_metric(p: Float64) -> OpticsMetric:
    """`OPTICS._metric_code` for minkowski: (code, p); p=2 keeps the
    squared-distance euclidean route (-1), p=1 is manhattan, +inf chebyshev."""
    if p == 2:
        return OpticsMetric(-1, 2.0)
    if p == 1:
        return OpticsMetric(1, 1.0)
    if isinf(p) and p > 0:
        return OpticsMetric(2, p)
    return OpticsMetric(3, p)


def optics_euclidean_metric() -> OpticsMetric:
    """'euclidean' and 'l2': the squared-distance route."""
    return OpticsMetric(-1, 2.0)


def optics_eps_refused(dbscan: Bool, eps: Float64, max_eps: Float64) -> Bool:
    """cluster_method='dbscan' with eps above max_eps."""
    return dbscan and eps > max_eps


def optics_xi_range(xi: Float64) raises:
    if not (xi >= 0 and xi <= 1):
        raise Error("xi must be in [0, 1]")


def optics_precomputed_square(metric: Int, n: Int, d: Int) raises:
    if metric == OPTICS_PRECOMPUTED and n != d:
        raise Error("metric='precomputed' needs a square distance matrix, got (" + String(n) + ", " + String(d) + ")")


def optics_fit_request(source: VariantsSpan, n: Int, d: Int, min_samples: Int, min_cluster_size: Int,
                       is_xi: Bool, predecessor_correction: Bool, metric: OpticsMetric, max_eps: Float64,
                       xi: Float64, eps: Float64) -> OpticsFitRequest:
    """ORDER MATCHES x_cluster/entries.mojo::optics_entry: n, d, min_samples,
    min_cluster_size, method (0 xi), predecessor_correction, metric;
    max_eps, xi, eps, p."""
    return OpticsFitRequest(source, n, d, min_samples, min_cluster_size, 0 if is_xi else 1,
                            predecessor_correction, metric.copy(), max_eps, xi, eps)


def optics_hierarchy_rows(words: Int) -> Int:
    """The xi hierarchy's (start, end) row count."""
    return words // 2
