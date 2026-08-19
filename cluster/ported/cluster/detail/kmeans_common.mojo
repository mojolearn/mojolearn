"""The pieces every k-means path shares: batching, convergence, sampling.

PORT OF `cuvs/src/cluster/detail/kmeans_common.cuh` at cuVS `94c2819`.
Partial. Do not improve.

What is ported here is the part of that file that is DECISION rather than
plumbing. Most of its bulk is RAFT and CUB glue (`cub::DeviceHistogram`,
`cub::DeviceSelect::If`, `cub::DeviceReduce`, `thrust::for_each_n`); see
`PORTED_MAP.tsv` for which of those became `mojo_only/` files and which are
simply not ported yet.

The decisions that ARE theirs and are copied exactly:

- which arm of `minClusterAndDistanceCompute` runs, which is a pure metric
  test and nothing else (`is_fused`, `kmeans_common.cuh:378-379`),
- the convergence test, which runs ON THE HOST, on a scalar copied back
  after the loop body's single stream sync (`detail/kmeans.cuh:461-497`),
- that the cost/ratio half of that test is GATED OFF by default
  (`params.inertia_check = false`, `cuvs/cluster/kmeans.hpp:120`), so the
  default fit stops on the centroid shift alone,
- the k-means|| sampling probability.
"""

from cluster.ported.cluster.kmeans_params import (
    METRIC_COSINE_EXPANDED,
    METRIC_L2_EXPANDED,
    METRIC_L2_SQRT_EXPANDED,
)


def is_fused(metric: Int) -> Bool:
    """`is_fused`, `kmeans_common.cuh:378-379`. Their ENTIRE arm selector.

        bool is_fused = metric == L2Expanded || metric == L2SqrtExpanded;

    There is no compute-capability term, no size threshold, and no CUTLASS
    fallback rule in it. `minClusterAndDistanceCompute` takes the fused arm
    (`:430-449`, `fusedDistanceNNMinReduce`) for those two metrics and the
    tile-and-`coalescedReduction` arm (`:450-491`) for everything else, and
    that is the whole dispatch.

    It has one visible consequence their own code calls out: on the fused arm
    `dataBatchSize` is forced to `n_samples` (`:380`), so `batch_samples` is
    inert for the metrics k-means actually uses.
    """
    return metric == METRIC_L2_EXPANDED or metric == METRIC_L2_SQRT_EXPANDED


def sampling_probability(
    distance_to_nearest: Float64,
    cluster_cost: Float64,
    oversampling_factor: Float64,
    n_clusters: Int,
) -> Float64:
    """`SamplingOp::operator()` at `kmeans_common.cuh:73-81`.

        prob_x = (oversampling_factor * n_clusters * a.value) / cluster_cost

    compared against a uniform draw per point. This is k-means|| step 3: an
    entire OVERSAMPLED round of centers is drawn in one parallel pass, where
    classic k-means++ draws exactly one center per sequential pass. That
    difference is the whole reason the scalable variant exists, and it is why
    `oversampling_factor` is an algorithm switch rather than a knob.
    """
    if cluster_cost == 0.0:
        return 0.0
    return (
        oversampling_factor * Float64(n_clusters) * distance_to_nearest
    ) / cluster_cost


def check_convergence(
    clustering_cost: Float64,
    prior_clustering_cost: Float64,
    sqrd_norm_error: Float64,
    tol: Float64,
    n_iter: Int,
    inertia_check: Bool,
) -> Bool:
    """The stopping rule of `kmeans_fit_main`, `detail/kmeans.cuh:466-492`.

    There is NO `check_convergence` function in cuVS. The rule is written
    inline in the fit loop and it looks like this:

        bool done = false;
        if (params.inertia_check) {                       // :468
          ... computeClusterCost -> clusterCostD          // :470-476
          DataT curClusteringCost = clusterCostD.value(stream);   // :478
          ASSERT(curClusteringCost != 0.0, ...);          // :480
          if (n_iter[0] > 1) {                            // :484
            DataT delta = curClusteringCost / priorClusteringCost;
            if (delta > 1 - params.tol) done = true;
          }
          priorClusteringCost = curClusteringCost;
        }
        raft::resource::sync_stream(handle, stream);      // :491
        if (sqrdNormError < params.tol) done = true;      // :492

    Four details that are easy to "fix" and must not be:

    1. **`inertia_check` is FALSE by default** (`cuvs/cluster/kmeans.hpp:120`).
       The ratio test and the reduction that feeds it do not run in a default
       cuVS fit at all. The default stopping rule is the centroid shift and
       only the centroid shift.
    2. The cost test is a RATIO, not a difference, so `tol` is relative for
       one test and absolute for the other. `1e-4` means two different things
       in the same rule.
    3. `n_iter > 1` skips the ratio test on the first iteration, because
       there is no prior cost yet. The centroid-shift test is NOT skipped, so
       a fit whose first iteration moves nothing stops immediately.
    4. Either test alone stops the fit. They are not combined.

    All of this runs on the HOST, in the same iteration, after the loop
    body's one `sync_stream`, and the loop breaks immediately (`:494-497`).
    The caller advances `prior_clustering_cost` itself, as theirs does at
    `:488`.
    """
    var done = False
    if inertia_check and clustering_cost != 0.0 and n_iter > 1:
        var delta = clustering_cost / prior_clustering_cost
        if delta > 1.0 - tol:
            done = True
    if sqrd_norm_error < tol:
        done = True
    return done


def metric_is_sqrt(metric: Int) -> Bool:
    """Whether the reduction takes a square root before writing.

    `kmeans_common.cuh:444` passes `metric != DistanceType::L2Expanded` as
    `fusedDistanceNNMinReduce`'s `sqrt` argument, so L2Expanded keeps SQUARED
    distances and everything else does not. Inertia is therefore a sum of
    squares under the default metric, which is what makes it comparable to
    scikit-learn's `inertia_`.
    """
    return metric != METRIC_L2_EXPANDED


def centroid_norms_take_sqrt(metric: Int) -> Bool:
    """The centroid `rowNorm` at `kmeans_common.cuh:385-389`.

    Theirs takes no square root there (`rowNorm<L2Norm, true>`, where the
    `true` is row-major, not sqrt), which is right for the L2 branches. Cosine
    is ours: it needs `sqrt` of the centroid norms because its branch DIVIDES
    by `||x|| ||y||`; the L2 branches subtract `2 x.c` from squared norms and
    must not.
    """
    return metric == METRIC_COSINE_EXPANDED
