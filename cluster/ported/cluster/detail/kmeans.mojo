"""Initialization and the Lloyd iteration.

PORT OF `cuvs/src/cluster/detail/kmeans.cuh` at cuVS `2140532c`. Partial.
Do not improve.

Their file is 1220 lines and most of it is host bookkeeping for cases this
tree does not have: host-resident input with a pinned batch iterator,
multi-GPU partitioned fits, and `kmeans_transform`. What is ported is the
algorithm: the two initializations and the iteration.

THE ITERATION, THEIR ORDER, WHICH IS NOT THE TEXTBOOK ORDER
-----------------------------------------------------------
The textbook writes assign, update, test. Theirs is

    assign  ->  accumulate  ->  finalize  ->  measure shift  ->  SWAP  ->  test

and the swap sitting where it does (`detail/kmeans.cuh:907`) is what makes
the shift measurement free: the shift is computed between the buffer that
was just consumed and the one just produced, before either is recycled, so
no third copy of the centroids ever exists.

The test itself runs on DEVICE in their code and its result reaches the host
one iteration late (`:817-825`, the `n_current_iter > 1` guard at the TOP of
the loop). They do that to keep the stream unblocked. This port tests on the
host, which is a deviation with a visible consequence: a fit here can report
one fewer iteration than cuVS's on identical data, because they always run
the iteration during which the flag is in flight. See PORTING.md 15. It does
not change the centroids of a converged fit; it changes `n_iter` and it
changes what a fit does when `max_iter` binds.

WHAT `n_init` ACTUALLY COSTS
----------------------------
`n_init` restarts are FULLY SEQUENTIAL here, as they are in cuVS: the whole
fit runs, its inertia is compared, and the best is kept (`:966-971`). Nothing
about restarts is inherently serial, and they are the most obviously
parallel thing in k-means, so this is the first place to look when the
control plane becomes the cost. Copied as-is because it is theirs.
"""

from std.math import ceil, log
from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.mojo_only.gemm import GEMM_TILE, gemm_nt_kernel
from cluster.mojo_only.plus_plus import (
    PLUS_PLUS_TPB,
    adopt_candidate_min_kernel,
    candidate_cost_kernel,
)
from cluster.mojo_only.reduce_by_key import (
    REDUCE_BY_KEY_TPB,
    accumulate_centroid_sums_kernel,
    accumulate_weight_per_cluster_kernel,
    centroid_shift_kernel,
    copy_f32_kernel,
    finalize_centroids_kernel,
    sum_partials_kernel,
    zero_i32_kernel,
)
from cluster.mojo_only.row_norms import NORM_TPB, row_norm_kernel
from cluster.ported.cluster.detail.kmeans_common import check_convergence
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
    centroid and the per-restart seeds (`detail/kmeans.cuh:148`, `:794`), and
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
    n: Int,
    use_b: Bool,
) raises -> Float64:
    """One block-reduced pass plus a host sum over the block partials."""
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
        Int32(1 if use_b else 0),
        grid_dim=(blocks, 1, 1),
        block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
    )
    ctx.synchronize()

    var host = ctx.enqueue_create_host_buffer[DType.float32](blocks)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=partials)
    ctx.synchronize()

    var total = Float64(0.0)
    for i in range(blocks):
        total += Float64(host.unsafe_ptr().unsafe_load(i))
    return total


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
    the greedily best of them (`detail/kmeans.cuh:112`). That greedy variant
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
    var host_dist = ctx.enqueue_create_host_buffer[DType.float32](n_samples)
    var host_cost = ctx.enqueue_create_host_buffer[DType.float32](n_trials)
    ctx.synchronize()

    var picked = 1
    while picked < params.n_clusters:
        # Step 3. `raft::random::discrete` over the d^2 weights. Done on the
        # host because the weights have to come back anyway to normalize.
        ctx.enqueue_copy(dst_ptr=host_dist.unsafe_ptr(), src_buf=min_dist)
        ctx.synchronize()

        var total = Float64(0.0)
        for i in range(n_samples):
            total += Float64(host_dist.unsafe_ptr().unsafe_load(i))
        if total <= 0.0:
            # Every point already sits on a centroid. Their `discrete` would
            # draw uniformly; do the same rather than dividing by zero.
            total = 0.0

        for t in range(n_trials):
            var target = rng.next_unit() * total
            var acc = Float64(0.0)
            var pick = n_samples - 1
            if total <= 0.0:
                pick = rng.next_index(n_samples)
            else:
                for i in range(n_samples):
                    acc += Float64(host_dist.unsafe_ptr().unsafe_load(i))
                    if acc >= target:
                        pick = i
                        break
            ctx.enqueue_function[copy_f32_kernel](
                candidates.unsafe_ptr().unsafe_offset(t * n_features),
                x.unsafe_ptr().unsafe_offset(pick * n_features),
                Int32(n_features),
                grid_dim=((n_features + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )
        ctx.synchronize()

        ctx.enqueue_function[row_norm_kernel](
            candidate_norm.unsafe_ptr(),
            candidates.unsafe_ptr(),
            Int32(n_features),
            Int32(0),
            grid_dim=(n_trials, 1, 1),
            block_dim=(NORM_TPB, 1, 1),
        )
        ctx.enqueue_function[gemm_nt_kernel](
            candidate_z.unsafe_ptr(),
            x.unsafe_ptr(),
            candidates.unsafe_ptr(),
            Int32(n_samples),
            Int32(n_trials),
            Int32(n_features),
            grid_dim=(
                (n_trials + GEMM_TILE - 1) // GEMM_TILE,
                (n_samples + GEMM_TILE - 1) // GEMM_TILE,
                1,
            ),
            block_dim=(GEMM_TILE, GEMM_TILE, 1),
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

        # Step 4, the greedy argmin over candidates.
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
    """`kmeans_fit`, device-resident arm, `detail/kmeans.cuh:556-984`.

    `sum_scale` and `weight_scale` come from `mojo_only/fixed_point.mojo` and
    are the caller's responsibility because the bound they satisfy is a
    property of the DATA, not of the fit: see `cluster/mojo_only/
    reduce_by_key.mojo` for why a global bound is enough for every cluster.

    `centroids` is in-out. On `INIT_ARRAY` it is read as the starting set,
    which is theirs (`:641-644`), and on return it holds the best restart.
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
    var shift_cells = ctx.enqueue_create_buffer[DType.float32](cd)
    var partials = ctx.enqueue_create_buffer[DType.float32](256)
    var ones = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()

    # X's norms, ONCE for the whole fit (`:786-790`). This is the single
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

    for _seed_iter in range(params.n_init):
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

        var prior_cost = Float64(0.0)
        var iter_cost = Float64(0.0)
        var n_current_iter = 0

        for it in range(1, params.max_iter + 1):
            n_current_iter = it

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

            var acc_blocks = min(256, max(1, n_samples // 256))
            ctx.enqueue_function[accumulate_centroid_sums_kernel](
                sums_i32.unsafe_ptr(),
                x.unsafe_ptr(),
                labels.unsafe_ptr(),
                weights.unsafe_ptr(),
                Int32(n_samples),
                Int32(n_features),
                sum_scale,
                grid_dim=(acc_blocks, 1, 1),
                block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
            )
            ctx.enqueue_function[accumulate_weight_per_cluster_kernel](
                weight_i32.unsafe_ptr(),
                labels.unsafe_ptr(),
                weights.unsafe_ptr(),
                Int32(n_samples),
                weight_scale,
                grid_dim=((n_samples + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
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
            ctx.enqueue_function[centroid_shift_kernel](
                shift_cells.unsafe_ptr(),
                cur_centroids.unsafe_ptr(),
                new_centroids.unsafe_ptr(),
                Int32(cd),
                grid_dim=((cd + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )
            ctx.synchronize()

            iter_cost = _sum_device(
                ctx, min_dist, weights, partials, n_samples, True
            )
            var shift = _sum_device(
                ctx, shift_cells, ones, partials, cd, False
            )

            # THE SWAP, `:907`, after the shift and before the test.
            ctx.enqueue_function[copy_f32_kernel](
                cur_centroids.unsafe_ptr(),
                new_centroids.unsafe_ptr(),
                Int32(cd),
                grid_dim=((cd + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )
            ctx.synchronize()

            if check_convergence(
                iter_cost, prior_cost, shift, params.tol, it
            ):
                prior_cost = iter_cost
                break
            prior_cost = iter_cost

        if iter_cost < best_inertia:
            best_inertia = iter_cost
            best_iter = min(n_current_iter, params.max_iter)
            ctx.enqueue_function[copy_f32_kernel](
                centroids.unsafe_ptr(),
                cur_centroids.unsafe_ptr(),
                Int32(cd),
                grid_dim=((cd + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )
            ctx.synchronize()

    return FitResult(best_inertia, best_iter)
