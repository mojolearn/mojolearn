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
    accumulate_centroid_sums_kernel,
    accumulate_weight_per_cluster_kernel,
    centroid_shift_kernel,
    copy_f32_kernel,
    finish_sum_kernel,
    finalize_centroids_kernel,
    sum_partials_kernel,
    zero_i32_kernel,
)
from core.row_norms import NORM_TPB, row_norm_kernel
from cluster.ported.cluster.detail.kmeans_common import (
    check_convergence_kernel,
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
    mut out_scalar: DeviceBuffer[DType.float32],
    n: Int,
    use_b: Bool,
) raises:
    """Two enqueued stages into a DEVICE scalar. No drain, no transfer.

    This used to sum the block partials in a host loop, which cost a drain
    and a transfer per call and was called TWICE per Lloyd iteration, for two
    numbers the host wanted only in order to decide something the device can
    decide. cuVS does not do that (`detail/kmeans.cuh:920-930`), so it was
    our artifact rather than their design. See `HOST_AND_DEVICE.md` on why
    that distinction decides whether a wait may be removed.
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
        Int32(1 if use_b else 0),
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
        # `bestCandidateIdx` to the host every accepted centroid too
        # (`detail/kmeans.cuh:224`). Host deciding was never the problem.
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
    var d_cost = ctx.enqueue_create_buffer[DType.float32](1)
    var d_shift = ctx.enqueue_create_buffer[DType.float32](1)
    var d_prior_cost = ctx.enqueue_create_buffer[DType.float32](1)
    var d_done = ctx.enqueue_create_buffer[DType.int32](1)
    var h_done = ctx.enqueue_create_host_buffer[DType.int32](1)
    var h_cost = ctx.enqueue_create_host_buffer[DType.float32](1)
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

        var n_current_iter = 0
        h_done.unsafe_ptr().unsafe_store(0, Int32(0))

        for it in range(1, params.max_iter + 1):
            # `detail/kmeans.cuh:817-825`. The flag is read at the TOP of the
            # NEXT iteration, so the fit always runs one iteration with the
            # answer in flight and `n_current_iter` is decremented when it
            # fires. That is theirs, decrement included, and this port used
            # to test on the host instead and report one fewer iteration.
            if it > 1:
                ctx.synchronize()
                if h_done.unsafe_ptr().unsafe_load(0) != Int32(0):
                    n_current_iter = it - 1
                    break
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

            # Sized in CELLS now, not rows: `accumulate_centroid_sums_kernel`
            # grid-strides `n_samples * n_features` the way RAFT's
            # `reduce_rows_by_key.cuh:292` does, rather than striding rows
            # and features separately. A grid-stride loop is correct for any
            # grid size, so this is a throughput knob and not a correctness
            # one, but leaving it row-shaped would under-occupy the device
            # whenever `n_features` is large.
            var acc_cells = n_samples * n_features
            var acc_blocks = min(
                1024, max(1, (acc_cells + REDUCE_BY_KEY_TPB - 1) // REDUCE_BY_KEY_TPB)
            )
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

            _sum_device(
                ctx, min_dist, weights, partials, d_cost, n_samples, True
            )
            _sum_device(
                ctx, shift_cells, ones, partials, d_shift, cd, False
            )

            # THE SWAP, `:907`, after the shift and before the test.
            ctx.enqueue_function[copy_f32_kernel](
                cur_centroids.unsafe_ptr(),
                new_centroids.unsafe_ptr(),
                Int32(cd),
                grid_dim=((cd + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )

            # THE TEST, on device, exactly as theirs. It also advances
            # `d_prior_cost`, which is why it cannot be split in two.
            ctx.enqueue_function[check_convergence_kernel](
                d_done.unsafe_ptr(),
                d_prior_cost.unsafe_ptr(),
                d_cost.unsafe_ptr(),
                d_shift.unsafe_ptr(),
                Float32(params.tol),
                Int32(it),
                grid_dim=(1, 1, 1),
                block_dim=(1, 1, 1),
            )
            ctx.enqueue_copy(dst_ptr=h_done.unsafe_ptr(), src_buf=d_done)

        # The fit's inertia. ONE transfer per restart, not one per iteration.
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=h_cost.unsafe_ptr(), src_buf=d_cost)
        ctx.synchronize()
        var iter_cost = Float64(h_cost.unsafe_ptr().unsafe_load(0))

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
