"""Initialization and the Lloyd iteration.

PORT OF `cuvs/src/cluster/detail/kmeans.cuh` at cuVS `94c2819`. Partial.
Do not improve.

Their file is 1242 lines and most of it is host bookkeeping for cases this
tree does not have: multi-GPU partitioned fits, `kmeans_transform`, and the
`kmeans_auto_find_k` driver. What is ported is the algorithm: the two
initializations and the iteration.

THE ITERATION, THEIR ORDER, WHICH IS NOT THE TEXTBOOK ORDER
-----------------------------------------------------------
The textbook writes assign, update, test. `kmeans_fit_main`
(`detail/kmeans.cuh:407-497`) is

    assign  ->  accumulate  ->  finalize  ->  measure shift  ->  COPY BACK
            ->  sync  ->  test on the host  ->  break

and the copy-back sitting where it does (`:464-465`, a device-to-device
`raft::copy`) is what makes the shift measurement free: the shift is computed
between the buffer that was just consumed and the one just produced
(`mapThenSumReduce` with `sqdiff_op`, `:454-459`), before either is recycled,
so no third copy of the centroids ever exists.

THE TEST RUNS ON THE HOST AND STOPS THE FIT IN THE SAME ITERATION
----------------------------------------------------------------
`:461-462` copies the shift scalar to the host, `:491` is the loop body's ONE
`raft::resource::sync_stream`, `:492` is `if (sqrdNormError < params.tol)
done = true;`, and `:494-497` breaks. There is no device-side convergence
kernel and no flag in flight.

An earlier version of this port ran the test in a one-thread kernel, read the
flag one iteration late, and attributed both to cuVS. Neither is in their
source: `check_convergence` does not exist anywhere in cuVS, cuML or RAFT.
That version paid for both control planes at once -- it still synced every
iteration -- and it always ran one Lloyd iteration more than cuVS would.

WHAT THE DEFAULT FIT DOES NOT DO
--------------------------------
`params.inertia_check` is FALSE by default (`cuvs/cluster/kmeans.hpp:120`).
With it off, the loop never reduces the cluster cost and never applies the
`delta > 1 - tol` ratio test (`:468-489` is all inside that `if`). The cost
is computed ONCE per restart, after the loop, over a fresh assignment against
the final centroids (`:500-537`) -- so the reported inertia belongs to the
centroids that are returned, not to the ones from the last iteration.

WHAT `n_init` ACTUALLY COSTS
----------------------------
`n_init` restarts are FULLY SEQUENTIAL here, as they are in cuVS: the whole
fit runs, its inertia is compared, and the best is kept (`:888-949`). Nothing
about restarts is inherently serial, and they are the most obviously
parallel thing in k-means, so this is the first place to look when the
control plane becomes the cost. Copied as-is because it is theirs.
"""

from std.math import ceil, log
from max.gpu.host import DeviceBuffer, DeviceContext

from core.gemm import gemm_nt
from cluster.mojo_only.plus_plus import (
    PLUS_PLUS_TPB,
    adopt_candidate_min_kernel,
    binary_search_kernel,
    candidate_cost_kernel,
    chunk_sums_kernel,
    gather_rows_kernel,
    scan_chunk_offsets_kernel,
    write_inclusive_scan_kernel,
)
from cluster.mojo_only.reduce_by_key import (
    REDUCE_BY_KEY_TPB,
    SUM_MODE_PLAIN,
    SUM_MODE_PRODUCT,
    SUM_MODE_SQDIFF,
    copy_f32_kernel,
    finish_sum_kernel,
    finalize_centroids_kernel,
    launch_accumulate_centroid_sums,
    launch_accumulate_weight_per_cluster,
    sum_partials_kernel,
    zero_i32_kernel,
)
from core.row_norms import NORM_TPB, row_norm_kernel
from cluster.ported.cluster.detail.kmeans_common import (
    check_convergence,
)
from cluster.ported.cluster.detail.min_cluster_distance_compute import (
    compute_centroid_norms,
    min_cluster_and_distance_compute,
)
from cluster.ported.cluster.kmeans_params import (
    INIT_ARRAY,
    INIT_KMEANS_PLUS_PLUS,
    INIT_RANDOM,
    KMeansParams,
    get_centroids_batch_size,
    get_data_batch_size,
)


@fieldwise_init
struct HostRng(Copyable, Movable):
    """A documented LCG, because cuVS's `std::mt19937` is not reproducible here.

    DEVIATION (PORTING.md 17). Their host RNG picks the first k-means++
    centroid (`detail/kmeans.cuh:152-157`) and the per-restart seeds
    (`:885`, `:890`), and
    their device RNG (`raft::random::discrete`) draws the k-means||
    candidates. Neither has a Mojo counterpart that would produce the same
    stream, so matching cuVS draw for draw is impossible and pretending
    otherwise would be worse than saying so.

    What IS preserved is the property that matters for validation: the same
    seed gives the same fit, so a disagreement with scikit-learn is an
    algorithm disagreement and not a coin flip. Cross-implementation
    comparison has to be over several seeds, which is what
    `cluster/tools/sklearn_reference.py` does.
    """

    var state: UInt64

    def next_u64(mut self) -> UInt64:
        # splitmix64, chosen because it is short enough to audit here.
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def next_index(mut self, n: Int) -> Int:
        return Int(self.next_u64() % UInt64(n))

    def next_unit(mut self) -> Float64:
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)


@fieldwise_init
struct FitResult(Copyable, ImplicitlyCopyable, Movable):
    var inertia: Float64
    var n_iter: Int


def _sum_device(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut partials: DeviceBuffer[DType.float32],
    mut out_scalar: DeviceBuffer[DType.float32],
    n: Int,
    mode: Int,
) raises:
    """Two enqueued stages into a DEVICE scalar, with the map folded in.

    `mode` selects the map, which cuVS keeps inside the reduction rather than
    materializing: `SUM_MODE_SQDIFF` is `raft::linalg::mapThenSumReduce` with
    `raft::sqdiff_op` (`detail/kmeans.cuh:454-459`), `SUM_MODE_PLAIN` and
    `SUM_MODE_PRODUCT` are `computeClusterCost` with `raft::value_op` over an
    unweighted and a weight-multiplied distance vector respectively
    (`:470-476` and `:516-535`).

    The result lands in a DEVICE scalar. Whether it then comes to the host is
    the caller's business and is where cuVS's control plane is decided: the
    shift scalar does cross every iteration (`:462`), the cost scalar crosses
    only when `inertia_check` is on, and once more after the loop.
    """
    var blocks = (n + REDUCE_BY_KEY_TPB - 1) // REDUCE_BY_KEY_TPB
    if blocks > 256:
        blocks = 256
    if blocks < 1:
        blocks = 1

    ctx.enqueue_function[sum_partials_kernel](
        partials.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(n),
        Int32(mode),
        grid_dim=(blocks, 1, 1),
        block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
    )
    ctx.enqueue_function[finish_sum_kernel](
        out_scalar.unsafe_ptr(),
        partials.unsafe_ptr(),
        Int32(blocks),
        grid_dim=(1, 1, 1),
        block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
    )


def init_random(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    mut rng: HostRng,
) raises:
    """`initRandom` -> `shuffleAndGather` -> `raft::matrix::sample_rows`.

    Theirs permutes an index vector on device and gathers the first k rows.
    Ours draws k indices on the host and issues k row copies, which is the
    same distribution and a different mechanism, chosen because k is at most
    a few hundred and a device permutation of `n_samples` to select `k` of
    them is the more expensive way to spend a launch.

    DEVIATION: theirs samples WITHOUT replacement by construction (it is a
    permutation). This draws with replacement, so two centroids can start
    identical. That is not benign: duplicate initial centroids give an empty
    cluster on iteration one, and the empty-cluster rule keeps the old
    centroid, so the duplicate persists. Rejected duplicates are the fix and
    are done below.
    """
    var chosen = List[Int]()
    while len(chosen) < n_clusters:
        var candidate = rng.next_index(n_samples)
        var seen = False
        for i in range(len(chosen)):
            if chosen[i] == candidate:
                seen = True
        if not seen:
            chosen.append(candidate)

    for j in range(n_clusters):
        ctx.enqueue_function[copy_f32_kernel](
            centroids.unsafe_ptr().unsafe_offset(j * n_features),
            x.unsafe_ptr().unsafe_offset(chosen[j] * n_features),
            Int32(n_features),
            grid_dim=((n_features + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
    ctx.synchronize()


def kmeans_plus_plus(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_norm: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut centroid_norm: DeviceBuffer[DType.float32],
    mut dist_buf: DeviceBuffer[DType.float32],
    mut min_dist: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.uint32],
    mut partials: DeviceBuffer[DType.float32],
    params: KMeansParams,
    n_samples: Int,
    n_features: Int,
    mut rng: HostRng,
) raises:
    """`kmeansPlusPlus`, the classic sequential variant, copied step for step.

    Their comment block numbers the steps and so does this:

        1  C = one point uniformly at random
        2  while |C| < k
        3      draw n_trials candidates with probability d^2(x, C) / phi
        4      keep the candidate that MINIMIZES the resulting cluster cost
        5  end

    Step 4 is the part people get wrong when reimplementing from the paper.
    Arthur and Vassilvitskii's algorithm draws ONE point and accepts it.
    cuVS, like scikit-learn, draws `n_trials = 2 + ceil(log(k))` and keeps
    the greedily best of them (`detail/kmeans.cuh:109`). That greedy variant
    is strictly better in practice and is what both implementations ship, so
    an accuracy comparison against scikit-learn is comparing two greedy
    k-means++ and not greedy against vanilla.

    This is the `oversampling_factor == 0` arm. The DEFAULT arm is scalable
    k-means|| (`initScalableKMeansPlusPlus`) and is NOT PORTED; see
    `UNWIRED.md`. `kmeans_fit_main` therefore refuses the default until it
    lands rather than silently substituting this one, because substituting a
    different initialization and reporting the inertia would be exactly the
    kind of quiet deviation this tree exists to avoid.
    """
    var n_trials = 2 + Int(ceil(log(Float64(params.n_clusters))))

    # Step 1.
    var first = rng.next_index(n_samples)
    ctx.enqueue_function[copy_f32_kernel](
        centroids.unsafe_ptr(),
        x.unsafe_ptr().unsafe_offset(first * n_features),
        Int32(n_features),
        grid_dim=((n_features + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    # d^2(x, C) for the single centroid so far.
    compute_centroid_norms(
        ctx, centroids, centroid_norm, 1, n_features, params.metric
    )
    min_cluster_and_distance_compute(
        ctx,
        x,
        x_norm,
        centroids,
        centroid_norm,
        dist_buf,
        labels,
        min_dist,
        n_samples,
        n_features,
        1,
        params.metric,
        params.batch_samples,
        params.batch_centroids,
    )
    ctx.synchronize()

    var candidates = ctx.enqueue_create_buffer[DType.float32](
        n_trials * n_features
    )
    var candidate_norm = ctx.enqueue_create_buffer[DType.float32](n_trials)
    var candidate_z = ctx.enqueue_create_buffer[DType.float32](
        n_samples * n_trials
    )
    var candidate_cost = ctx.enqueue_create_buffer[DType.float32](n_trials)
    var host_cost = ctx.enqueue_create_host_buffer[DType.float32](n_trials)

    # Two-level device draw. `n_chunks` is capped so the serial walk inside a
    # chunk is `n / n_chunks` steps rather than `n`.
    # One chunk per block of threads, and NO cap on the chunk count: the
    # second stage scans the chunk totals with a per-thread slice derived
    # from their number. Capping it is exactly the trapdoor that silently
    # truncated DBSCAN's CSR.
    var chunk = PLUS_PLUS_TPB
    var n_chunks = (n_samples + chunk - 1) // chunk

    var chunk_totals = ctx.enqueue_create_buffer[DType.float32](n_chunks)
    var chunk_offsets = ctx.enqueue_create_buffer[DType.float32](n_chunks)
    var csum = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var sel_index = ctx.enqueue_create_buffer[DType.uint32](n_trials)
    var d_u01 = ctx.enqueue_create_buffer[DType.float32](n_trials)
    var h_u01 = ctx.enqueue_create_host_buffer[DType.float32](n_trials)
    ctx.synchronize()

    var picked = 1
    while picked < params.n_clusters:
        # Step 3. `raft::random::discrete` over the d^2 weights, ON DEVICE.
        # The host contributes only `n_trials` uniforms, which is
        # O(candidates) and is what HOST_AND_DEVICE.md permits.
        for t in range(n_trials):
            h_u01.unsafe_ptr().unsafe_store(t, Float32(rng.next_unit()))
        ctx.enqueue_copy(dst_buf=d_u01, src_ptr=h_u01.unsafe_ptr())

        ctx.enqueue_function[chunk_sums_kernel](
            chunk_totals.unsafe_ptr(),
            min_dist.unsafe_ptr(),
            Int32(n_samples),
            Int32(chunk),
            grid_dim=(n_chunks, 1, 1),
            block_dim=(PLUS_PLUS_TPB, 1, 1),
        )
        ctx.enqueue_function[scan_chunk_offsets_kernel](
            chunk_offsets.unsafe_ptr(),
            chunk_totals.unsafe_ptr(),
            Int32(n_chunks),
            grid_dim=(1, 1, 1),
            block_dim=(PLUS_PLUS_TPB, 1, 1),
        )
        ctx.enqueue_function[write_inclusive_scan_kernel](
            csum.unsafe_ptr(),
            min_dist.unsafe_ptr(),
            chunk_offsets.unsafe_ptr(),
            Int32(n_samples),
            Int32(chunk),
            grid_dim=(n_chunks, 1, 1),
            block_dim=(PLUS_PLUS_TPB, 1, 1),
        )
        ctx.enqueue_function[binary_search_kernel](
            sel_index.unsafe_ptr(),
            csum.unsafe_ptr(),
            d_u01.unsafe_ptr(),
            Int32(n_samples),
            Int32(n_trials),
            grid_dim=(1, 1, 1),
            block_dim=(max(n_trials, 1), 1, 1),
        )
        ctx.enqueue_function[gather_rows_kernel](
            candidates.unsafe_ptr(),
            x.unsafe_ptr(),
            sel_index.unsafe_ptr(),
            Int32(n_features),
            grid_dim=(n_trials, 1, 1),
            block_dim=(PLUS_PLUS_TPB, 1, 1),
        )

        ctx.enqueue_function[row_norm_kernel](
            candidate_norm.unsafe_ptr(),
            candidates.unsafe_ptr(),
            Int32(n_features),
            Int32(0),
            grid_dim=(n_trials, 1, 1),
            block_dim=(NORM_TPB, 1, 1),
        )
        gemm_nt(
            ctx,
            candidate_z,
            x,
            candidates,
            n_samples,
            n_trials,
            n_features,
        )
        ctx.enqueue_function[candidate_cost_kernel](
            candidate_cost.unsafe_ptr(),
            candidate_z.unsafe_ptr(),
            x_norm.unsafe_ptr(),
            candidate_norm.unsafe_ptr(),
            min_dist.unsafe_ptr(),
            Int32(n_samples),
            Int32(n_trials),
            grid_dim=(n_trials, 1, 1),
            block_dim=(PLUS_PLUS_TPB, 1, 1),
        )
        ctx.synchronize()

        # Step 4, the greedy argmin over candidates. THIS READBACK STAYS.
        # It is `n_trials` floats, which is O(candidates), and cuVS brings
        # `bestCandidateIdx` to the host every accepted centroid too, and
        # syncs the stream for it (`detail/kmeans.cuh:242-244`). Host deciding
        # was never the problem. Theirs argmins on the device first
        # (`cub::DeviceReduce::ArgMin`, `:224-240`) and brings back one int
        # where this brings back `n_trials` floats and argmins on the host;
        # both are O(candidates) and one drain.
        ctx.enqueue_copy(dst_ptr=host_cost.unsafe_ptr(), src_buf=candidate_cost)
        ctx.synchronize()
        var best = 0
        var best_cost = Float64(host_cost.unsafe_ptr().unsafe_load(0))
        for t in range(1, n_trials):
            var c = Float64(host_cost.unsafe_ptr().unsafe_load(t))
            if c < best_cost:
                best_cost = c
                best = t

        ctx.enqueue_function[adopt_candidate_min_kernel](
            min_dist.unsafe_ptr(),
            candidate_z.unsafe_ptr(),
            x_norm.unsafe_ptr(),
            candidate_norm.unsafe_ptr(),
            Int32(n_samples),
            Int32(n_trials),
            Int32(best),
            grid_dim=((n_samples + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[copy_f32_kernel](
            centroids.unsafe_ptr().unsafe_offset(picked * n_features),
            candidates.unsafe_ptr().unsafe_offset(best * n_features),
            Int32(n_features),
            grid_dim=((n_features + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        picked += 1


def kmeans_fit_main(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.uint32],
    params: KMeansParams,
    n_samples: Int,
    n_features: Int,
    sum_scale: Float32,
    weight_scale: Float32,
) raises -> FitResult:
    """`kmeans_fit` plus `kmeans_fit_main`, `detail/kmeans.cuh:352-542` and
    `:811-951`, flattened into one function because the split in theirs exists
    to give the multi-GPU driver a re-entry point this tree does not have.

    `sum_scale` and `weight_scale` come from `mojo_only/fixed_point.mojo` and
    are the caller's responsibility because the bound they satisfy is a
    property of the DATA, not of the fit: see `cluster/mojo_only/
    reduce_by_key.mojo` for why a global bound is enough for every cluster.

    `centroids` is in-out. On `INIT_ARRAY` it is read as the starting set,
    which is theirs (`:916-924`), and on return it holds the best restart.
    """
    params.validate()
    if params.init == INIT_KMEANS_PLUS_PLUS and params.uses_scalable_plus_plus():
        raise Error(
            "scalable k-means++ (oversampling_factor > 0) is NOT PORTED; see"
            " UNWIRED.md. Set oversampling_factor=0 for classic k-means++ or"
            " init=Random."
        )

    var n_clusters = params.n_clusters
    var cd = n_clusters * n_features

    var data_batch = get_data_batch_size(params.batch_samples, n_samples)
    var centroid_batch = get_centroids_batch_size(
        params.batch_centroids, n_clusters
    )

    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var centroid_norm = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    var dist_buf = ctx.enqueue_create_buffer[DType.float32](
        data_batch * centroid_batch
    )
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var sums_i32 = ctx.enqueue_create_buffer[DType.int32](cd)
    var weight_i32 = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var cur_centroids = ctx.enqueue_create_buffer[DType.float32](cd)
    var new_centroids = ctx.enqueue_create_buffer[DType.float32](cd)
    var partials = ctx.enqueue_create_buffer[DType.float32](256)
    # A distinct one-element buffer so `sum_partials_kernel` never receives the
    # same buffer as both of its mutable arguments (PORTING.md 24). Its
    # contents are never read on the modes that pass it.
    var ones = ctx.enqueue_create_buffer[DType.float32](1)
    var d_cost = ctx.enqueue_create_buffer[DType.float32](1)
    var d_shift = ctx.enqueue_create_buffer[DType.float32](1)
    var h_shift = ctx.enqueue_create_host_buffer[DType.float32](1)
    var h_cost = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.synchronize()

    # X's norms, ONCE for the whole fit (`:394-399`). This is the single
    # biggest reason the assignment step is cheap: the sample side of the
    # expanded identity never has to be recomputed, only the centroid side.
    if params.needs_row_norms():
        ctx.enqueue_function[row_norm_kernel](
            x_norm.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(n_features),
            Int32(0),
            grid_dim=(n_samples, 1, 1),
            block_dim=(NORM_TPB, 1, 1),
        )
        ctx.synchronize()

    var rng = HostRng(params.seed)
    var best_inertia = Float64(1.0e308)
    var best_iter = 0

    # `detail/kmeans.cuh:877-883`. An explicit initial centroid set makes
    # every restart identical, so theirs collapses `n_init` to 1 rather than
    # running the same fit repeatedly. It is a debug log there and a silent
    # correction; it is the same correction here.
    var n_init = params.n_init
    if params.init == INIT_ARRAY:
        n_init = 1

    for _seed_iter in range(n_init):
        if params.init == INIT_ARRAY:
            ctx.enqueue_function[copy_f32_kernel](
                cur_centroids.unsafe_ptr(),
                centroids.unsafe_ptr(),
                Int32(cd),
                grid_dim=((cd + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )
            ctx.synchronize()
        elif params.init == INIT_RANDOM:
            init_random(
                ctx, x, cur_centroids, n_samples, n_features, n_clusters, rng
            )
        else:
            kmeans_plus_plus(
                ctx,
                x,
                x_norm,
                cur_centroids,
                centroid_norm,
                dist_buf,
                min_dist,
                labels,
                partials,
                params,
                n_samples,
                n_features,
                rng,
            )

        # `for (n_iter[0] = 1; n_iter[0] <= params.max_iter; ++n_iter[0])`,
        # `detail/kmeans.cuh:407`. On a break `n_current_iter` is the
        # iteration that converged; on exhaustion theirs leaves it at
        # `max_iter + 1`, which is why their own log line subtracts one
        # (`:540`). Not clamped, because theirs is not.
        var prior_cost = Float64(0.0)
        var n_current_iter = params.max_iter + 1
        var it = 1
        while it <= params.max_iter:
            ctx.enqueue_function[zero_i32_kernel](
                sums_i32.unsafe_ptr(),
                Int32(cd),
                grid_dim=((cd + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )
            ctx.enqueue_function[zero_i32_kernel](
                weight_i32.unsafe_ptr(),
                Int32(n_clusters),
                grid_dim=((n_clusters + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )

            compute_centroid_norms(
                ctx,
                cur_centroids,
                centroid_norm,
                n_clusters,
                n_features,
                params.metric,
            )
            min_cluster_and_distance_compute(
                ctx,
                x,
                x_norm,
                cur_centroids,
                centroid_norm,
                dist_buf,
                labels,
                min_dist,
                n_samples,
                n_features,
                n_clusters,
                params.metric,
                params.batch_samples,
                params.batch_centroids,
            )

            # `update_centroids`' two reductions, `detail/kmeans.cuh:300-318`.
            # The dispatch (privatized block-local partials when they fit
            # and the input is large, direct scatter-add otherwise), the
            # grids (from the hardware matrix, replacing a magic 1024-block
            # cap that lived here) and the bit-identity argument between the
            # arms all live in `cluster/mojo_only/reduce_by_key.mojo`.
            launch_accumulate_centroid_sums(
                ctx,
                sums_i32,
                x,
                labels,
                weights,
                n_samples,
                n_features,
                n_clusters,
                sum_scale,
            )
            launch_accumulate_weight_per_cluster(
                ctx,
                weight_i32,
                labels,
                weights,
                n_samples,
                n_clusters,
                weight_scale,
            )

            ctx.enqueue_function[finalize_centroids_kernel](
                new_centroids.unsafe_ptr(),
                cur_centroids.unsafe_ptr(),
                sums_i32.unsafe_ptr(),
                weight_i32.unsafe_ptr(),
                Int32(n_clusters),
                Int32(n_features),
                sum_scale,
                weight_scale,
                grid_dim=((cd + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )

            # The shift, `:453-459`. `mapThenSumReduce` with `sqdiff_op`: the
            # squared difference is applied INSIDE the reduction, between the
            # buffer about to be overwritten and the one just produced, so
            # nothing of size `n_clusters * n_features` is materialized.
            _sum_device(
                ctx,
                cur_centroids,
                new_centroids,
                partials,
                d_shift,
                cd,
                SUM_MODE_SQDIFF,
            )
            # `:461-462`, the scalar starts its trip to the host.
            ctx.enqueue_copy(dst_ptr=h_shift.unsafe_ptr(), src_buf=d_shift)

            # `:464-465`, the copy back over the working set, AFTER the shift
            # has been measured against it.
            ctx.enqueue_function[copy_f32_kernel](
                cur_centroids.unsafe_ptr(),
                new_centroids.unsafe_ptr(),
                Int32(cd),
                grid_dim=((cd + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )

            # `:468-489`, the whole cost half of the stopping rule, and it is
            # OFF by default. Note their in-loop cost is UNWEIGHTED
            # (`raft::value_op` straight over `minClusterAndDistance`); only
            # the post-loop inertia at `:516-535` multiplies by the weights.
            var cur_cost = Float64(0.0)
            if params.inertia_check:
                _sum_device(
                    ctx,
                    min_dist,
                    ones,
                    partials,
                    d_cost,
                    n_samples,
                    SUM_MODE_PLAIN,
                )
                ctx.enqueue_copy(dst_ptr=h_cost.unsafe_ptr(), src_buf=d_cost)
                # `clusterCostD.value(stream)` at `:478` is `rmm`'s blocking
                # scalar read: a copy AND a stream sync. Turning
                # `inertia_check` on therefore costs a SECOND drain per
                # iteration, which is a reason it is off by default.
                ctx.synchronize()
                cur_cost = Float64(h_cost.unsafe_ptr().unsafe_load(0))
                # `ASSERT` at `:480-482`, message included.
                if cur_cost == 0.0:
                    raise Error(
                        "Too few points and centroids being found is getting"
                        " 0 cost from centers"
                    )

            # `:491`. THE one stream sync of the loop body.
            ctx.synchronize()
            var shift = Float64(h_shift.unsafe_ptr().unsafe_load(0))

            # `:492`, on the host, in this iteration.
            var done = check_convergence(
                cur_cost, prior_cost, shift, params.tol, it, params.inertia_check
            )
            if params.inertia_check:
                prior_cost = cur_cost  # `:488`

            # `:494-497`.
            if done:
                n_current_iter = it
                break
            it += 1

        # `:500-537`. The fit's inertia is ONE fresh assignment against the
        # FINAL centroids, weighted, after the loop -- not the last
        # iteration's cost, which belongs to the centroids from before the
        # final update. This also leaves `labels` consistent with the
        # centroids that are returned.
        compute_centroid_norms(
            ctx,
            cur_centroids,
            centroid_norm,
            n_clusters,
            n_features,
            params.metric,
        )
        min_cluster_and_distance_compute(
            ctx,
            x,
            x_norm,
            cur_centroids,
            centroid_norm,
            dist_buf,
            labels,
            min_dist,
            n_samples,
            n_features,
            n_clusters,
            params.metric,
            params.batch_samples,
            params.batch_centroids,
        )
        _sum_device(
            ctx, min_dist, weights, partials, d_cost, n_samples,
            SUM_MODE_PRODUCT,
        )
        ctx.enqueue_copy(dst_ptr=h_cost.unsafe_ptr(), src_buf=d_cost)
        ctx.synchronize()
        var iter_cost = Float64(h_cost.unsafe_ptr().unsafe_load(0))

        # `:938-943`, and `n_iter` is NOT clamped there.
        if iter_cost < best_inertia:
            best_inertia = iter_cost
            best_iter = n_current_iter
            ctx.enqueue_function[copy_f32_kernel](
                centroids.unsafe_ptr(),
                cur_centroids.unsafe_ptr(),
                Int32(cd),
                grid_dim=((cd + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )
            ctx.synchronize()

    return FitResult(best_inertia, best_iter)
