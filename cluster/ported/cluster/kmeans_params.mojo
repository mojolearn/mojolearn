"""k-means hyper-parameters, with cuVS's names and cuVS's defaults.

PORT OF `cuvs::cluster::kmeans::params` (`cpp/include/cuvs/cluster/
kmeans.hpp:28-121`) at cuVS `94c2819`. Transliterated. Do not improve.

Every default here is theirs, including the ones that look arbitrary, for the
same reason `gbdt/options/catboost_options.mojo` keeps CatBoost's: if a
measurement comes out badly we have to be able to say it is their
configuration that is slow and not our taste in defaults.

Two of these defaults decide which algorithm actually runs and are worth
reading before changing:

- `init = INIT_KMEANS_PLUS_PLUS` together with `oversampling_factor = 2.0`
  selects SCALABLE k-means++ (k-means||), not the classic sequential one.
  `oversampling_factor == 0` is overloaded as an algorithm switch to the
  classic variant (`detail/kmeans.cuh:910-915`). It is not a tuning knob with
  a neutral value; zero means "different algorithm".
- `n_init = 1`, where scikit-learn's default is 10. Seeded restarts are the
  single biggest quality lever in k-means, so an accuracy comparison against
  scikit-learn that leaves both at their defaults is comparing one restart
  against ten and is not a comparison of the implementations.
"""


# `params::InitMethod` (`kmeans.hpp:44-60`). Mojo has no scoped enum, so
# these are the codes, in their declaration order.
comptime INIT_KMEANS_PLUS_PLUS = 0
comptime INIT_RANDOM = 1
comptime INIT_ARRAY = 2


# `cuvs::distance::DistanceType`, only the members this port implements.
#
# DEVIATION. cuVS's `kmeans_fit` does NOT restrict the metric: its
# `RAFT_EXPECTS` block (`detail/kmeans.cuh:825-835`) checks sample_weight,
# n_clusters, tol, oversampling_factor and the centroid extents, and nothing
# else. Any metric with a `pairwise_distance` reaches their unfused arm
# (`kmeans_common.cuh:450-491`). `validate()` below raises on anything that is
# not L2Expanded or L2SqrtExpanded because only their FUSED arm
# (`kmeans_common.cuh:430-449`, `is_fused`) is ported here, and running a
# metric through a path that was never written for it is worse than refusing
# it. The refusal is ours; do not attribute it to them.
comptime METRIC_L2_EXPANDED = 0
comptime METRIC_L2_SQRT_EXPANDED = 1
comptime METRIC_COSINE_EXPANDED = 2


def init_method_name(init: Int) -> String:
    if init == INIT_KMEANS_PLUS_PLUS:
        return String("KMeansPlusPlus")
    if init == INIT_RANDOM:
        return String("Random")
    return String("Array")


@fieldwise_init
struct KMeansParams(Copyable, ImplicitlyCopyable, Movable):
    """`cuvs::cluster::kmeans::params`, field for field, default for default.

    `verbosity` is dropped: it selects a rapids_logger level and we have no
    logger. That is the only field of theirs with no counterpart here.
    """

    var metric: Int
    var n_clusters: Int
    var init: Int
    var max_iter: Int
    var tol: Float64
    var seed: UInt64
    var n_init: Int
    var oversampling_factor: Float64
    var batch_samples: Int
    var batch_centroids: Int
    var inertia_check: Bool

    @staticmethod
    def default() -> Self:
        """Their defaults, from the field initializers in `kmeans.hpp:28-121`.

        `inertia_check = False` (`kmeans.hpp:120`) is the one to read twice.
        It gates a WHOLE STEP of the Lloyd iteration, not a diagnostic: with
        it off, their loop never computes the cluster cost and never applies
        the `delta > 1 - tol` ratio test, and the only convergence criterion
        is the centroid shift (`detail/kmeans.cuh:468-492`). Turning it on
        adds a full-length reduction and a second stopping rule, so a fit run
        with it on can stop on a different iteration than the default fit.
        """
        return Self(
            metric=METRIC_L2_EXPANDED,
            n_clusters=8,
            init=INIT_KMEANS_PLUS_PLUS,
            max_iter=300,
            tol=1e-4,
            seed=0,
            n_init=1,
            oversampling_factor=2.0,
            batch_samples=1 << 15,
            batch_centroids=0,
            inertia_check=False,
        )

    def uses_scalable_plus_plus(self) -> Bool:
        """`detail/kmeans.cuh:910-915`. Zero picks classic sequential k-means++.
        """
        return self.init == INIT_KMEANS_PLUS_PLUS and (
            self.oversampling_factor != 0.0
        )

    def needs_row_norms(self) -> Bool:
        """The `rowNorm` gate at `detail/kmeans.cuh:395-399`.

        The expanded metrics need `||x||^2` per row because they compute
        `||x||^2 + ||c||^2 - 2 x.c` from a matrix product rather than a
        difference. That identity is the reason a GEMM can do the work, and
        also the reason the result can come out slightly negative and has to
        be clamped downstream.
        """
        return (
            self.metric == METRIC_L2_EXPANDED
            or self.metric == METRIC_L2_SQRT_EXPANDED
        )

    def validate(self) raises:
        """`RAFT_EXPECTS` block at `detail/kmeans.cuh:825-835`.

        Theirs checks n_clusters, tol and oversampling_factor exactly as
        below. The metric refusal is OURS and is explained at the metric
        codes above.
        """
        if (
            self.metric != METRIC_L2_EXPANDED
            and self.metric != METRIC_L2_SQRT_EXPANDED
        ):
            raise Error(
                "kmeans only supports L2Expanded or L2SqrtExpanded distance"
                " metrics."
            )
        if self.n_clusters <= 0:
            raise Error("invalid parameter (n_clusters<=0)")
        if self.tol <= 0.0:
            raise Error("invalid parameter (tol<=0)")
        if self.oversampling_factor < 0.0:
            raise Error("invalid parameter (oversampling_factor<0)")


def get_data_batch_size(batch_samples: Int, n_samples: Int) -> Int:
    """`getDataBatchSize`, `detail/kmeans_common.cuh:176-181`.

    Zero means "do not tile", which is spelled as "tile by the whole thing".

    Note their own dispatch bypasses this on the fused arm:
    `dataBatchSize = is_fused ? n_samples : getDataBatchSize(...)`
    (`kmeans_common.cuh:380`), and `kmeans_fit` logs a debug line saying so
    (`detail/kmeans.cuh:837-846`). `batch_samples` therefore has NO effect on
    the L2Expanded path in cuVS, and it has none here either.
    """
    var min_val = min(batch_samples, n_samples)
    return n_samples if min_val == 0 else min_val


def get_centroids_batch_size(batch_centroids: Int, n_local_clusters: Int) -> Int:
    """`getCentroidsBatchSize`, `detail/kmeans_common.cuh:183-188`."""
    var min_val = min(batch_centroids, n_local_clusters)
    return n_local_clusters if min_val == 0 else min_val
