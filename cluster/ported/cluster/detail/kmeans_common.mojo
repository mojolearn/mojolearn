"""The pieces every k-means path shares: batching, convergence, sampling.

PORT OF `cuvs/src/cluster/detail/kmeans_common.cuh` at cuVS `2140532c`.
Partial. Do not improve.

What is ported here is the part of that file that is DECISION rather than
plumbing. Most of its bulk is RAFT and CUB glue (`cub::DeviceHistogram`,
`cub::DeviceSelect::If`, `cub::DeviceReduce`, `thrust::for_each_n`), which
has no counterpart to transliterate; see `PORTED_MAP.tsv` for which of those
became `mojo_only/` files and which are simply not ported yet.

The decisions that ARE theirs and are copied exactly:

- when the fused path is chosen over the unfused one, and why that is a
  hardware question rather than an algorithm question,
- the two convergence tests and the fact that BOTH are checked every
  iteration and either one alone stops the fit,
- that convergence is evaluated ON DEVICE and the flag reaches the host one
  iteration late, so a fit runs one iteration further than a host-side test
  would,
- the k-means|| sampling probability.
"""

from std.gpu import block_idx, thread_idx

from cluster.ported.cluster.kmeans_params import (
    KMeansParams,
    METRIC_COSINE_EXPANDED,
    METRIC_L2_EXPANDED,
    METRIC_L2_SQRT_EXPANDED,
)


# `use_fused` at `kmeans_common.cuh:60-82`. Their compute-capability codes.
comptime SM_AMPERE_OR_EARLIER = 8
comptime SM_HOPPER = 9
comptime SM_BLACKWELL = 10


def use_fused(sm_major: Int, m: Int, n: Int) -> Bool:
    """`use_fused`, copied including the 4096 threshold.

    Recorded here rather than dropped, even though this tree ports only the
    unfused kernel, because it is the evidence that the unfused path is not a
    fallback. Their own rule sends Blackwell and later to unfused
    unconditionally, and sends Hopper there for anything smaller than 4096.

    We have no counterpart to a CUTLASS tensor-core GEMM, so on every backend
    this tree targets the answer is False. That is a REACH fact and belongs
    in `UNWIRED.md`, not a silent constant.
    """
    if sm_major <= SM_AMPERE_OR_EARLIER:
        return True
    if sm_major == SM_HOPPER and (m >= 4096 or n >= 4096):
        return True
    if sm_major >= SM_BLACKWELL:
        return False
    return False


def sampling_probability(
    distance_to_nearest: Float64,
    cluster_cost: Float64,
    oversampling_factor: Float64,
    n_clusters: Int,
) -> Float64:
    """`SamplingOp::operator()` at `kmeans_common.cuh:100-106`.

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
) -> Bool:
    """`check_convergence` at `kmeans_common.cuh:637-660`, copied exactly.

        if (cur_cost != 0 && n_iter > 1) {
            delta = cur_cost / prior_clustering_cost;
            if (delta > 1 - tol) done = 1;
        }
        if (norm_err < tol) done = 1;

    Three details that are easy to "fix" and must not be:

    1. The cost test is a RATIO, not a difference, so `tol` is relative for
       one test and absolute for the other. `1e-4` means two different things
       in the same function.
    2. `n_iter > 1` skips the ratio test on the first iteration, because
       there is no prior cost yet. The centroid-shift test is NOT skipped, so
       a fit whose first iteration moves nothing stops immediately.
    3. Either test alone stops the fit. They are not combined.
    """
    var done = False
    if clustering_cost != 0.0 and n_iter > 1:
        var delta = clustering_cost / prior_clustering_cost
        if delta > 1.0 - tol:
            done = True
    if sqrd_norm_error < tol:
        done = True
    return done


def metric_is_sqrt(metric: Int) -> Bool:
    """Whether the reduction takes a square root before writing.

    `minClusterDistanceCompute.cu:71` passes
    `metric != DistanceType::L2Expanded`, so L2Expanded keeps SQUARED
    distances and everything else does not. Inertia is therefore a sum of
    squares under the default metric, which is what makes it comparable to
    scikit-learn's `inertia_`.
    """
    return metric != METRIC_L2_EXPANDED


def centroid_norms_take_sqrt(metric: Int) -> Bool:
    """`minClusterDistanceCompute.cu:44-49`.

    Cosine takes `sqrt` of the centroid norms because its branch DIVIDES by
    `||x|| ||y||`; the L2 branches subtract `2 x.c` from squared norms and
    must not.
    """
    return metric == METRIC_COSINE_EXPANDED


def check_convergence_kernel(
    done_flag: MutPointer[Int32, MutAnyOrigin],
    prior_clustering_cost: MutPointer[Float32, MutAnyOrigin],
    clustering_cost: MutPointer[Float32, MutAnyOrigin],
    sqrd_norm_error: MutPointer[Float32, MutAnyOrigin],
    tol_in: Float32,
    n_iter_in: Int32,
):
    """`check_convergence` (`kmeans_common.cuh:637-660`) AS A DEVICE FUNCTION,
    which is what it is in their source.

    The host version above is kept because it documents the three details in
    prose and because a host-side checker is what a bring-up harness wants.
    **This is the one the fit uses**, and porting it as a kernel is not an
    optimization: cuVS runs this on the device
    (`detail/kmeans.cuh:920-930`, a `map_offset` over a single element) and
    reads only the resulting FLAG back. A host-side test was our artifact and
    it cost a full drain plus two transfers per iteration for a number the
    host never needed except to decide.

    Note it also ADVANCES `prior_clustering_cost` in the same call, which is
    theirs and is why the flag and the prior cost cannot be split into two
    kernels without reintroducing an ordering hazard.
    """
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return

    var cur_cost = clustering_cost.unsafe_load(0)
    var norm_err = sqrd_norm_error.unsafe_load(0)
    var done = Int32(0)

    if cur_cost != Float32(0.0) and Int(n_iter_in) > 1:
        var delta = cur_cost / prior_clustering_cost.unsafe_load(0)
        if delta > Float32(1.0) - tol_in:
            done = Int32(1)
    if norm_err < tol_in:
        done = Int32(1)

    prior_clustering_cost.unsafe_store(0, cur_cost)
    done_flag.unsafe_store(0, done)
