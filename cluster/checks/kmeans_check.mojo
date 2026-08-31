# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The first code that LAUNCHES the k-means port, and the sabotage that proves it.

NO CUVS COUNTERPART. Their tests are gtest fixtures against a CPU reference;
this is the harness this tree's own rule demands, which is a different thing.

**A kernel is not ported until it has been enqueued** (`PORTING.md 9`).
Everything under `cluster/` compiled cleanly for two commits while having
never run, and compiling is not evidence.

WHY THERE ARE TWO KINDS OF CHECK HERE, AND WHY THE SECOND IS THE IMPORTANT ONE
------------------------------------------------------------------------------
`check_kmeans_fit` asks "is the answer right". `check_reach_by_sabotage` asks
"is the answer right FOR THE REASON WE THINK". Those come apart, and in this
project they have come apart repeatedly: mojotrees shipped four stages that
were built, tested, documented and unreachable, and on 2026-08-19 an env-path
reach check passed for its packed-bins arm while nothing read the buffer.

A correct answer is not evidence that a kernel ran. It is evidence that
SOMETHING produced the right numbers, and a default, a stale buffer or a
no-op can do that. So the reach test corrupts a specific input and asserts
the answer MOVES, and it asserts the SHAPE of the movement, which is stronger
than asserting movement at all.

WHY THE FIXTURE IS NOT UNIFORM
------------------------------
Planted clusters are given DISTINGUISHABLE centers and rows are assigned to
them ROUND ROBIN rather than in contiguous blocks. Both matter.

Uniform or symmetric fixtures hide permutation bugs: if every cluster looks
the same, a check on totals passes while every point sits in the wrong
cluster. Contiguous membership hides partition bugs the same way, because a
broken assignment that happens to preserve row order still lands each block
in one cluster.

So the check verifies that the recovered centroids are a PERMUTATION of the
planted ones, each matched exactly once, and that every row's label follows
its planted membership through that permutation.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from cluster.impl.distance.fused_distance_nn.simt_kernel import (
    FUSED_NORMAL_KBLK,
    FUSED_NORMAL_TC,
    FUSED_NORMAL_TR,
    fused_distance_nn_kernel,
    fused_is_skinny,
    fused_smem_bytes,
    fused_veclen_for,
)
from neighbors.impl.distance.detail.pairwise_distance_base import (
    launch_config_generator,
)
from cluster.impl.cluster.detail.min_cluster_distance_compute import (
    compute_centroid_norms,
    min_cluster_and_distance_compute,
    min_cluster_and_distance_compute_unfused,
)
from cluster.checks.reduce_by_key import (
    PRIVATE_ACC_CELLS,
    PRIVATE_MIN_WORK,
    REDUCE_BY_KEY_TPB,
    accumulate_centroid_sums_kernel,
    accumulate_grid_blocks,
    accumulate_weight_per_cluster_kernel,
    launch_accumulate_centroid_sums,
    launch_accumulate_weight_per_cluster,
    zero_i32_kernel,
)
from cluster.impl.cluster.kmeans import fit
from cluster.checks.plus_plus import (
    PLUS_PLUS_TPB,
    chunk_sums_kernel,
    scan_chunk_offsets_kernel,
    write_inclusive_scan_kernel,
)
from cluster.checks.scalable_init import (
    sample_flags_kernel,
    scalable_keep,
    scalable_uniform,
    select_scatter_kernel,
)
from cluster.impl.cluster.kmeans_params import (
    INIT_KMEANS_PLUS_PLUS,
    INIT_ARRAY,
    KMeansParams,
    METRIC_L2_EXPANDED,
)
from core.row_norms import NORM_TPB, row_norm_kernel
from checks.fixed_point import choose_scale
from checks.kernel_matrix import (
    TARGET_COLUMN,
    VENDOR_TF32_PRODUCT_REL_BOUND,
    column_name,
    vendor_fp32_matmul_is_lossy,
    vendor_fp32_matmul_precision_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.memory import bitcast


comptime CHECK_ROWS = 512
comptime CHECK_FEATURES = 4
comptime CHECK_CLUSTERS = 4


def _planted_center(cluster: Int, feature: Int) -> Float32:
    """Well separated and distinct in EVERY feature.

    Separation is 100 per cluster against a jitter of at most 1, so the
    correct assignment is not in doubt and a failure is a wiring failure
    rather than a hard clustering problem. This harness is not measuring
    clustering quality.
    """
    return Float32(100 * (cluster + 1) + feature)


def _jitter(row: Int, feature: Int) -> Float32:
    """A hashed offset, so no two rows are identical.

    Deliberately not a constant and not uniform: identical rows inside a
    cluster would let a kernel that reads the wrong row still produce the
    right centroid.
    """
    var h = (row * 2654435761 + feature * 40503) % 2039
    return Float32(h) / Float32(2039) - Float32(0.5)


def _fill_fixture(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
) raises -> Float64:
    """Upload the fixture. Returns the per-feature magnitude bound."""
    var hx = ctx.enqueue_create_host_buffer[DType.float32](
        CHECK_ROWS * CHECK_FEATURES
    )
    var hw = ctx.enqueue_create_host_buffer[DType.float32](CHECK_ROWS)

    var worst = Float64(0.0)
    for f in range(CHECK_FEATURES):
        var column = Float64(0.0)
        for i in range(CHECK_ROWS):
            var c = i % CHECK_CLUSTERS  # ROUND ROBIN, see the module docstring
            var v = _planted_center(c, f) + _jitter(i, f)
            hx.unsafe_ptr().unsafe_store(i * CHECK_FEATURES + f, v)
            column += Float64(abs(v))
        if column > worst:
            worst = column

    for i in range(CHECK_ROWS):
        hw.unsafe_ptr().unsafe_store(i, Float32(1.0))

    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()
    return worst


def _assign(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_norm: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut centroid_norm: DeviceBuffer[DType.float32],
    mut dist_buf: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.uint32],
    mut min_dist: DeviceBuffer[DType.float32],
    params: KMeansParams,
) raises:
    """One assignment pass, re-run between sabotages.

    A free function rather than a closure: Mojo 1.0 cannot infer a capture
    convention for a `DeviceContext` captured by a nested `def`.
    """
    min_cluster_and_distance_compute(
        ctx,
        x,
        x_norm,
        centroids,
        centroid_norm,
        dist_buf,
        labels,
        min_dist,
        CHECK_ROWS,
        CHECK_FEATURES,
        CHECK_CLUSTERS,
        METRIC_L2_EXPANDED,
        params.batch_samples,
        params.batch_centroids,
    )
    ctx.synchronize()


def check_kmeans_fit() raises:
    """512 rows, 4 features, 4 planted clusters, round robin membership.

    Initialized with `INIT_ARRAY` at centers pushed 30 off the planted ones,
    which is far enough that Lloyd has to move them and near enough that the
    correct basin is unambiguous. `INIT_ARRAY` rather than a random or
    k-means++ init because this check is about the ITERATION, and a check
    that also depends on the draw cannot say which half failed.
    """
    var ctx = DeviceContext()
    var cd = CHECK_CLUSTERS * CHECK_FEATURES

    var x = ctx.enqueue_create_buffer[DType.float32](
        CHECK_ROWS * CHECK_FEATURES
    )
    var weights = ctx.enqueue_create_buffer[DType.float32](CHECK_ROWS)
    var centroids = ctx.enqueue_create_buffer[DType.float32](cd)
    var labels = ctx.enqueue_create_buffer[DType.uint32](CHECK_ROWS)
    ctx.synchronize()

    var magnitude = _fill_fixture(ctx, x, weights)
    var sum_scale = Float32(choose_scale(magnitude))
    var weight_scale = Float32(choose_scale(Float64(CHECK_ROWS)))

    var hc = ctx.enqueue_create_host_buffer[DType.float32](cd)
    for c in range(CHECK_CLUSTERS):
        for f in range(CHECK_FEATURES):
            hc.unsafe_ptr().unsafe_store(
                c * CHECK_FEATURES + f,
                _planted_center(c, f) + Float32(30.0),
            )
    ctx.enqueue_copy(dst_buf=centroids, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()

    var params = KMeansParams.default()
    params.n_clusters = CHECK_CLUSTERS
    params.init = INIT_ARRAY
    params.metric = METRIC_L2_EXPANDED
    params.n_init = 1
    params.max_iter = 50

    var result = fit(
        ctx,
        x,
        weights,
        centroids,
        labels,
        params,
        CHECK_ROWS,
        CHECK_FEATURES,
        sum_scale,
        weight_scale,
    )

    var out_c = ctx.enqueue_create_host_buffer[DType.float32](cd)
    var out_l = ctx.enqueue_create_host_buffer[DType.uint32](CHECK_ROWS)
    ctx.enqueue_copy(dst_ptr=out_c.unsafe_ptr(), src_buf=centroids)
    ctx.enqueue_copy(dst_ptr=out_l.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    # --- the recovered centroids must be a PERMUTATION of the planted ones.
    # Matched exactly once each, which is what rules out a collapse onto one
    # cluster. A per-centroid nearest-match test alone would pass for that.
    var mapping = List[Int]()
    var used = List[Int]()
    for _c in range(CHECK_CLUSTERS):
        used.append(0)

    for slot in range(CHECK_CLUSTERS):
        var best = -1
        var best_err = Float64(1.0e30)
        for planted in range(CHECK_CLUSTERS):
            var err = Float64(0.0)
            for f in range(CHECK_FEATURES):
                var d = Float64(
                    out_c.unsafe_ptr().unsafe_load(slot * CHECK_FEATURES + f)
                    - _planted_center(planted, f)
                )
                err += d * d
            if err < best_err:
                best_err = err
                best = planted
        if best_err > 1.0:
            raise Error(
                "centroid " + String(slot) + " is not near any planted center,"
                " squared error " + String(best_err)
            )
        if used[best] != 0:
            raise Error(
                "two centroids matched planted center " + String(best)
                + ": the fit COLLAPSED, which an inertia check would miss"
            )
        used[best] = 1
        mapping.append(best)

    # --- every row's label must follow its planted membership -------------
    var wrong = 0
    for i in range(CHECK_ROWS):
        var slot = Int(out_l.unsafe_ptr().unsafe_load(i))
        if slot < 0 or slot >= CHECK_CLUSTERS:
            raise Error("label out of range at row " + String(i))
        if mapping[slot] != i % CHECK_CLUSTERS:
            wrong += 1
    if wrong != 0:
        raise Error(
            String(wrong) + " of " + String(CHECK_ROWS)
            + " rows landed in the wrong cluster"
        )

    # --- inertia is the sum of squared jitters, which is KNOWN ------------
    var expected = Float64(0.0)
    for i in range(CHECK_ROWS):
        for f in range(CHECK_FEATURES):
            var c = i % CHECK_CLUSTERS
            var mean = Float64(0.0)
            var count = 0
            for j in range(CHECK_ROWS):
                if j % CHECK_CLUSTERS == c:
                    mean += Float64(_jitter(j, f))
                    count += 1
            mean /= Float64(count)
            var d = Float64(_jitter(i, f)) - mean
            expected += d * d

    var rel = abs(result.inertia - expected) / expected
    if rel > 0.02:
        raise Error(
            "inertia " + String(result.inertia) + " against expected "
            + String(expected) + ", relative " + String(rel)
        )

    print(
        "check_kmeans_fit OK: 4/4 centroids matched as a permutation, 0/"
        + String(CHECK_ROWS)
        + " rows misassigned, inertia "
        + String(result.inertia)
        + " vs expected "
        + String(expected)
        + " (rel "
        + String(rel)
        + "), "
        + String(result.n_iter)
        + " iterations"
    )


def check_reach_by_sabotage() raises:
    """Corrupt one input and assert the answer moves IN THE PREDICTED SHAPE.

    Two sabotages, and the pair is stronger than either alone because they
    predict DIFFERENT movements:

    1. **Centroid norms.** In the expanded identity these differ per centroid,
       so corrupting them must change WHICH centroid wins. Labels must move.
    2. **Sample norms.** These are a constant per ROW, added identically to
       every centroid's distance, so PERTURBING them changes the distance
       VALUE and cannot change the argmin. Distances must move and labels
       must NOT.

       The perturbation has to be a positive OFFSET, not a replacement.
       Their clamp at `distance/detail/distance_ops/l2_exp.cuh:132` is not
       order-preserving:
       drive the distances negative and all of them flatten to exactly 0.0,
       the tie-break hands every row to centroid 0, and the labels move. The
       first version of this check replaced the norms outright, failed with
       384 of 512 labels moved, and the kernel was right.

    A pipeline that passes both is reading each buffer for the reason the
    algorithm says. A no-op passes neither. A pipeline that merely "moves"
    under any corruption passes the first and fails the second, which is the
    case a movement-only reach check would wave through.

    The corruption is applied AFTER `compute_centroid_norms` and the reduction
    is re-run alone, so nothing recomputes over the sabotage.
    """
    var ctx = DeviceContext()

    var x = ctx.enqueue_create_buffer[DType.float32](
        CHECK_ROWS * CHECK_FEATURES
    )
    var weights = ctx.enqueue_create_buffer[DType.float32](CHECK_ROWS)
    var x_norm = ctx.enqueue_create_buffer[DType.float32](CHECK_ROWS)
    var centroids = ctx.enqueue_create_buffer[DType.float32](
        CHECK_CLUSTERS * CHECK_FEATURES
    )
    var centroid_norm = ctx.enqueue_create_buffer[DType.float32](
        CHECK_CLUSTERS
    )
    var dist_buf = ctx.enqueue_create_buffer[DType.float32](
        CHECK_ROWS * CHECK_CLUSTERS
    )
    var labels = ctx.enqueue_create_buffer[DType.uint32](CHECK_ROWS)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](CHECK_ROWS)
    ctx.synchronize()

    _ = _fill_fixture(ctx, x, weights)

    var hc = ctx.enqueue_create_host_buffer[DType.float32](
        CHECK_CLUSTERS * CHECK_FEATURES
    )
    for c in range(CHECK_CLUSTERS):
        for f in range(CHECK_FEATURES):
            hc.unsafe_ptr().unsafe_store(
                c * CHECK_FEATURES + f, _planted_center(c, f)
            )
    ctx.enqueue_copy(dst_buf=centroids, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[row_norm_kernel](
        x_norm.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(CHECK_FEATURES),
        Int32(0),
        grid_dim=(CHECK_ROWS, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )
    compute_centroid_norms(
        ctx,
        centroids,
        centroid_norm,
        CHECK_CLUSTERS,
        CHECK_FEATURES,
        METRIC_L2_EXPANDED,
    )
    ctx.synchronize()

    var base_l = ctx.enqueue_create_host_buffer[DType.uint32](CHECK_ROWS)
    var base_d = ctx.enqueue_create_host_buffer[DType.float32](CHECK_ROWS)
    var sab_l = ctx.enqueue_create_host_buffer[DType.uint32](CHECK_ROWS)
    var sab_d = ctx.enqueue_create_host_buffer[DType.float32](CHECK_ROWS)

    var params = KMeansParams.default()

    _assign(
        ctx, x, x_norm, centroids, centroid_norm, dist_buf, labels, min_dist,
        params,
    )
    ctx.enqueue_copy(dst_ptr=base_l.unsafe_ptr(), src_buf=labels)
    ctx.enqueue_copy(dst_ptr=base_d.unsafe_ptr(), src_buf=min_dist)
    ctx.synchronize()

    # --- SABOTAGE 1: centroid norms. Labels MUST move. -------------------
    var hn = ctx.enqueue_create_host_buffer[DType.float32](CHECK_CLUSTERS)
    for c in range(CHECK_CLUSTERS):
        # Reversed magnitudes, so the cheapest centroid becomes the dearest.
        hn.unsafe_ptr().unsafe_store(
            c, Float32(1.0e6) * Float32(CHECK_CLUSTERS - c)
        )
    ctx.enqueue_copy(dst_buf=centroid_norm, src_ptr=hn.unsafe_ptr())
    ctx.synchronize()

    _assign(
        ctx, x, x_norm, centroids, centroid_norm, dist_buf, labels, min_dist,
        params,
    )
    ctx.enqueue_copy(dst_ptr=sab_l.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    var moved = 0
    for i in range(CHECK_ROWS):
        if sab_l.unsafe_ptr().unsafe_load(i) != base_l.unsafe_ptr().unsafe_load(
            i
        ):
            moved += 1
    if moved == 0:
        raise Error(
            "SABOTAGE 1 FAILED TO REGISTER: corrupting centroid_norm changed"
            " no label. The reduction is not reading it, so the assignment"
            " step is doing something other than what it says."
        )

    # --- restore, then SABOTAGE 2: sample norms. Labels must NOT move. ----
    compute_centroid_norms(
        ctx,
        centroids,
        centroid_norm,
        CHECK_CLUSTERS,
        CHECK_FEATURES,
        METRIC_L2_EXPANDED,
    )
    # ADDS to the true norms rather than replacing them. The first version of
    # this sabotage REPLACED them with small values, which drove every
    # expanded distance negative, and their clamp at
    # `distance_ops/l2_exp.cuh:132` then flattened all four centroids to
    # exactly 0.0 so the tie-break handed every row to centroid 0. It failed
    # with 384 labels moved and the kernel was RIGHT.
    #
    # **Worth keeping as a fact about their kernel, not just about this
    # test.** The invariant "a per-row constant cannot change the argmin"
    # holds only while the distances stay positive. The clamp is not
    # order-preserving once anything drives them below zero, so it is safe
    # for the round-off it exists for and is NOT safe as a general guard.
    var hxn = ctx.enqueue_create_host_buffer[DType.float32](CHECK_ROWS)
    ctx.enqueue_copy(dst_ptr=hxn.unsafe_ptr(), src_buf=x_norm)
    ctx.synchronize()
    for i in range(CHECK_ROWS):
        hxn.unsafe_ptr().unsafe_store(
            i,
            hxn.unsafe_ptr().unsafe_load(i) + Float32(1000.0) * Float32(i + 1),
        )
    ctx.enqueue_copy(dst_buf=x_norm, src_ptr=hxn.unsafe_ptr())
    ctx.synchronize()

    _assign(
        ctx, x, x_norm, centroids, centroid_norm, dist_buf, labels, min_dist,
        params,
    )
    ctx.enqueue_copy(dst_ptr=sab_l.unsafe_ptr(), src_buf=labels)
    ctx.enqueue_copy(dst_ptr=sab_d.unsafe_ptr(), src_buf=min_dist)
    ctx.synchronize()

    var label_moved = 0
    var dist_moved = 0
    for i in range(CHECK_ROWS):
        if sab_l.unsafe_ptr().unsafe_load(i) != base_l.unsafe_ptr().unsafe_load(
            i
        ):
            label_moved += 1
        if sab_d.unsafe_ptr().unsafe_load(i) != base_d.unsafe_ptr().unsafe_load(
            i
        ):
            dist_moved += 1

    if dist_moved == 0:
        raise Error(
            "SABOTAGE 2 FAILED TO REGISTER: corrupting x_norm changed no"
            " distance. The reduction is not reading it."
        )
    if label_moved != 0:
        raise Error(
            "SABOTAGE 2 CHANGED "
            + String(label_moved)
            + " LABELS. A per-row constant cannot change an argmin, so the"
            " reduction is not forming the distance the way the expanded"
            " identity says."
        )

    print(
        "check_reach_by_sabotage OK: centroid_norm moved "
        + String(moved)
        + "/"
        + String(CHECK_ROWS)
        + " labels; x_norm moved "
        + String(dist_moved)
        + " distances and 0 labels, which is the predicted shape"
    )


def check_device_inclusive_scan() raises:
    """The three-stage device scan, alone, against a host scan.

    `cub::DeviceScan::InclusiveSum` has no shipped GPU counterpart
    (`nn.cumsum` is CPU-only), so k-means++ builds one out of
    `block.prefix_sum`. That construction is exactly where a fixed
    per-thread slice silently truncates, which is what happened in DBSCAN's
    CSR and went unnoticed because every fixture was smaller than the cap.

    So this runs at 20,000 elements, well past one block's worth, and checks
    every entry rather than the total. A conservation check would pass on a
    scan that got the placement wrong.
    """
    var ctx = DeviceContext()
    var n = 20000
    var chunk = PLUS_PLUS_TPB
    var n_chunks = (n + chunk - 1) // chunk

    var a = ctx.enqueue_create_buffer[DType.float32](n)
    var totals = ctx.enqueue_create_buffer[DType.float32](n_chunks)
    var offsets = ctx.enqueue_create_buffer[DType.float32](n_chunks)
    var csum = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()

    var ha = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        ha.unsafe_ptr().unsafe_store(i, Float32((i % 17) + 1))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[chunk_sums_kernel](
        totals.unsafe_ptr(), a.unsafe_ptr(), Int32(n), Int32(chunk),
        grid_dim=(n_chunks, 1, 1), block_dim=(PLUS_PLUS_TPB, 1, 1),
    )
    ctx.enqueue_function[scan_chunk_offsets_kernel](
        offsets.unsafe_ptr(), totals.unsafe_ptr(), Int32(n_chunks),
        grid_dim=(1, 1, 1), block_dim=(PLUS_PLUS_TPB, 1, 1),
    )
    ctx.enqueue_function[write_inclusive_scan_kernel](
        csum.unsafe_ptr(), a.unsafe_ptr(), offsets.unsafe_ptr(),
        Int32(n), Int32(chunk),
        grid_dim=(n_chunks, 1, 1), block_dim=(PLUS_PLUS_TPB, 1, 1),
    )
    ctx.synchronize()

    var hc = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=csum)
    ctx.synchronize()

    var running = Float64(0.0)
    var worst = Float64(0.0)
    for i in range(n):
        running += Float64(ha.unsafe_ptr().unsafe_load(i))
        var d = abs(Float64(hc.unsafe_ptr().unsafe_load(i)) - running) / running
        if d > worst:
            worst = d
    if worst > 1.0e-4:
        raise Error(
            "device inclusive scan disagrees with the host scan, worst"
            " relative error " + String(worst) + " at n = " + String(n)
        )
    print(
        "check_device_inclusive_scan OK: " + String(n)
        + " entries, worst relative error " + String(worst)
        + ", past one block's worth"
    )


def check_kmeans_plus_plus_init() raises:
    """Run a fit through the k-means++ path, which nothing else reaches.

    Every other check here uses `INIT_ARRAY` on purpose, so that a failure
    cannot hide in the draw. The cost of that is that the entire
    initialization stayed UNREACHED, which `UNWIRED.md` has said since the
    section was written and which is now false only because of this.

    It exercises the scan, the binary search, the gather, the candidate cost
    and the adopt step. The assertion is deliberately weak on WHICH centroids
    come back, because that depends on the draw, and strong on the thing that
    must hold regardless: with blobs this well separated, k-means++ followed
    by Lloyd has to recover them as a permutation.
    """
    var ctx = DeviceContext()
    var cd = CHECK_CLUSTERS * CHECK_FEATURES

    var x = ctx.enqueue_create_buffer[DType.float32](
        CHECK_ROWS * CHECK_FEATURES
    )
    var weights = ctx.enqueue_create_buffer[DType.float32](CHECK_ROWS)
    var centroids = ctx.enqueue_create_buffer[DType.float32](cd)
    var labels = ctx.enqueue_create_buffer[DType.uint32](CHECK_ROWS)
    ctx.synchronize()

    var magnitude = _fill_fixture(ctx, x, weights)
    var sum_scale = Float32(choose_scale(magnitude))
    var weight_scale = Float32(choose_scale(Float64(CHECK_ROWS)))

    var params = KMeansParams.default()
    params.n_clusters = CHECK_CLUSTERS
    params.init = INIT_KMEANS_PLUS_PLUS
    params.oversampling_factor = 0.0  # classic k-means++, the ported arm
    params.metric = METRIC_L2_EXPANDED
    params.n_init = 1
    params.max_iter = 50
    params.seed = 12345

    var result = fit(
        ctx, x, weights, centroids, labels, params,
        CHECK_ROWS, CHECK_FEATURES, sum_scale, weight_scale,
    )

    var out_c = ctx.enqueue_create_host_buffer[DType.float32](cd)
    ctx.enqueue_copy(dst_ptr=out_c.unsafe_ptr(), src_buf=centroids)
    ctx.synchronize()

    var used = List[Int]()
    for _c in range(CHECK_CLUSTERS):
        used.append(0)
    for slot in range(CHECK_CLUSTERS):
        var best = -1
        var best_err = Float64(1.0e30)
        for planted in range(CHECK_CLUSTERS):
            var err = Float64(0.0)
            for f in range(CHECK_FEATURES):
                var d = Float64(
                    out_c.unsafe_ptr().unsafe_load(slot * CHECK_FEATURES + f)
                    - _planted_center(planted, f)
                )
                err += d * d
            if err < best_err:
                best_err = err
                best = planted
        if best_err > 1.0:
            raise Error(
                "k-means++ init: centroid " + String(slot)
                + " is not near any planted center, squared error "
                + String(best_err)
            )
        if used[best] != 0:
            raise Error(
                "k-means++ init: two centroids matched planted center "
                + String(best) + ", the fit collapsed"
            )
        used[best] = 1

    print(
        "check_kmeans_plus_plus_init OK: 4/4 centroids recovered as a"
        " permutation through the k-means++ path, inertia "
        + String(result.inertia)
        + ", " + String(result.n_iter) + " iterations"
    )


def check_fused_reduction_across_lanes() raises:
    """Exercise the fused kernel's CROSS-LANE merge, which nothing else did.

    **This check exists because a sabotage came back clean and that was the
    check's fault, not the code's.** Skipping a partner lane in the fused
    butterfly changed nothing, because at `CHECK_CLUSTERS = 4` almost every
    lane held nothing but `FUSED_MAX`: with column ownership
    `lane = col % AccThCols` (their strided assignment,
    `contractions.cuh:102`), columns 0..3 occupy lanes 0..3 and the other
    twelve lanes of the group are empty. The merge was near-degenerate and
    a broken merge could not reliably show.

    That is the same failure the repository already has a name for: a fixture
    whose structure hides the thing under test. Here it hid a whole reduction
    stage rather than a permutation.

    So: 40 clusters, spreading real values over ALL 16 lanes of the
    `AccThCols = 16` group, checked against a host argmin. Now a broken
    cross-lane merge has to show.

    The kernel is instantiated DIRECTLY at the NORMAL policy
    (`Policy4x4<float, 4>`): the launcher's own selection would route
    `k = 4` features to the skinny 8-wide policy (`fused_is_skinny`), and
    this check is about the 16-wide merge. The skinny merge is exercised
    through the launcher by `check_kmeans_fit` and the policy selection
    itself by `check_fused_policy_dispatch`.
    """
    var ctx = DeviceContext()
    var n = 512
    var d = 4
    var k = 40

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var c = ctx.enqueue_create_buffer[DType.float32](k * d)
    var xn = ctx.enqueue_create_buffer[DType.float32](n)
    var cn = ctx.enqueue_create_buffer[DType.float32](k)
    var okey = ctx.enqueue_create_buffer[DType.uint32](n)
    var oval = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(
                i * d + f, Float32(_jitter(i, f) * 10.0)
            )
    for j in range(k):
        for f in range(d):
            hc.unsafe_ptr().unsafe_store(
                j * d + f, Float32(_jitter(j + 9001, f) * 10.0)
            )
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[row_norm_kernel](
        xn.unsafe_ptr(), x.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(n, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    ctx.enqueue_function[row_norm_kernel](
        cn.unsafe_ptr(), c.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(k, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    ctx.synchronize()
    comptime lanes_kern = fused_distance_nn_kernel[
        4, FUSED_NORMAL_KBLK, FUSED_NORMAL_TR, FUSED_NORMAL_TC
    ]
    comptime lanes_mblk = 4 * FUSED_NORMAL_TR
    comptime lanes_threads = FUSED_NORMAL_TR * FUSED_NORMAL_TC
    ctx.enqueue_function[lanes_kern](
        okey.unsafe_ptr(), oval.unsafe_ptr(), x.unsafe_ptr(), c.unsafe_ptr(),
        xn.unsafe_ptr(), cn.unsafe_ptr(),
        Int32(n), Int32(k), Int32(d), Int32(0),
        grid_dim=(1, (n + lanes_mblk - 1) // lanes_mblk, 1),
        block_dim=(lanes_threads, 1, 1),
    )
    ctx.synchronize()

    var hk = ctx.enqueue_create_host_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_ptr=hk.unsafe_ptr(), src_buf=okey)
    ctx.synchronize()

    var wrong = 0
    var spread = List[Int]()
    for _t in range(16):
        spread.append(0)
    for i in range(n):
        var best = 0
        var best_d = Float64(1.0e30)
        for j in range(k):
            var dd = Float64(0.0)
            for f in range(d):
                var diff = Float64(
                    hx.unsafe_ptr().unsafe_load(i * d + f)
                ) - Float64(hc.unsafe_ptr().unsafe_load(j * d + f))
                dd += diff * diff
            if dd < best_d:
                best_d = dd
                best = j
        var got = Int(hk.unsafe_ptr().unsafe_load(i))
        if got != best:
            wrong += 1
        # The lane that owns column `best` is `best % AccThCols` (strided
        # ownership, `contractions.cuh:102`).
        spread[best % 16] = 1
    var lanes = 0
    for t in range(16):
        lanes += spread[t]

    if lanes < 4:
        raise Error(
            "the fixture only used " + String(lanes) + " owner lanes; it"
            " cannot exercise a cross-lane merge"
        )
    if wrong != 0:
        raise Error(
            String(wrong) + " of " + String(n)
            + " assignments disagree with a host argmin over "
            + String(k) + " clusters"
        )
    print(
        "check_fused_reduction_across_lanes OK: " + String(n)
        + " rows x " + String(k) + " clusters match a host argmin, winners"
        " spread over " + String(lanes) + " owner lanes"
    )


# --- assignment-arm dispatch ------------------------------------------------

comptime ARM_ROWS = 512
comptime ARM_FEATURES = 32
comptime ARM_CLUSTERS = 64
comptime ARM_SENTINEL = Float32(-777.25)

#: The fp32 budget an assignment arm's min_dist is held to, relative to the
#: MAGNITUDE of the expanded form's operands (`xn + yn + 2|x.c|`), the same
#: number `checks/gram_splitk_check.mojo` holds the Gram product to and
#: for the same reason: it covers fp32 accumulation-order spread, and a
#: vendor product on a lossy column is allowed `VENDOR_TF32_PRODUCT_REL_BOUND`
#: on top of it (DEVIATION 529).
comptime ARM_FP32_REL_BUDGET = Float64(1.0e-5)

#: Output poison: a NaN value and a key no cluster index can be.
comptime ARM_POISON_KEY = UInt32(0xDEADBEEF)


def _poison_outputs(
    ctx: DeviceContext,
    mut keys: DeviceBuffer[DType.uint32],
    mut values: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """Fill an assignment arm's outputs with poison before its launch, so a
    row it never writes reads back as poison rather than as whatever the
    allocator left there (DEVIATION 529)."""
    var hk = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    var nan_bits = UInt32(0x7FC0BEEF)
    for i in range(n):
        hk.unsafe_ptr().unsafe_store(i, ARM_POISON_KEY)
        hv.unsafe_ptr().unsafe_store(i, bitcast[DType.float32](nan_bits))
    ctx.enqueue_copy(dst_buf=keys, src_ptr=hk.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=values, src_ptr=hv.unsafe_ptr())
    ctx.synchronize()


def _surviving_poison(
    keys: HostBuffer[DType.uint32], values: HostBuffer[DType.float32], n: Int
) -> Int:
    """Rows where either output still carries the poison. A NaN value is
    counted as poison whatever its payload: no arm may return one."""
    var count = 0
    for i in range(n):
        var v = values.unsafe_ptr().unsafe_load(i)
        if keys.unsafe_ptr().unsafe_load(i) == ARM_POISON_KEY or v != v:
            count += 1
    return count


def _round_to_tf32(x: Float32) -> Float32:
    """Host model of a tensor core's operand conversion: the fp32 mantissa
    rounded to 10 explicit bits, nearest-even. Used by the self-sabotage in
    `check_assignment_arms_match_oracle`."""
    var bits = bitcast[DType.uint32](x)
    var lsb = (bits >> 13) & 1
    bits = (bits + 0xFFF + lsb) & ~UInt32(0x1FFF)
    return bitcast[DType.float32](bits)


def check_assignment_arm_dispatch() raises:
    """Prove WHICH assignment arm the fit's entry point takes, by a sentinel
    visible only through one arm.

    cuVS's selector has two arms and one distinguishing side effect: the
    unfused arm MATERIALIZES a `ns x nc` distance tile into
    `L2NormBuf_OR_DistBuf` (`kmeans_common.cuh:450-491`); the fused arm
    never writes a tile at all (`:430-449` -> `fusedDistanceNNMinReduce`).
    So `dist_buf` is the sentinel carrier: fill it with a poison value, run
    the entry `fit` calls (`detail/kmeans.mojo`, the Lloyd loop), and the
    poison SURVIVING is the fused arm's signature, while the poison being
    OVERWRITTEN is the unfused arm's.

    The shape matches the fit's 4M x 32, k=64 bench arm in every
    dispatch-relevant dimension -- d=32, k=64, metric L2Expanded, default
    batch parameters -- scaled down only in rows, which appear in neither
    their selector (`is_fused`, `kmeans_common.cuh:378-379`, a metric test
    with nothing else in it) nor ours.

    The check sabotages ITSELF: after the fused entry leaves the poison
    intact, the unfused arm is run on the same inputs and MUST destroy it.
    Without that half, a sentinel that nothing could ever overwrite (wrong
    buffer, wrong size) would pass the first half forever.

    Both arms must also AGREE here: same labels bitwise (planted separation
    is 100 per cluster against jitter <= 1, so no near-ties; both arms
    reduce with `raft::argmin_op`'s total order, PORTING.md 14), and
    distances within a tolerance -- NOT bitwise, because the two arms sum
    the dot product in different orders, exactly as upstream's two arms do.

    **THE DISTANCE TOLERANCE IS PER COLUMN, AND THAT IS DEVIATION 529
    (2026-08-23).** On the H100 leg of E2 this check FAILED under FAST with
    "fused and unfused min_dist diverge, worst relative 51968000000.0"
    after every fused-arm check before it (including a 40-cluster host
    argmin across 16 owner lanes) had passed, and the labels agreed. The
    number decodes: the old comparison was `|a - b| / max(|a|, 1e-6)` with
    `a` the FUSED value, so 5.2e10 means the fused arm returned ~0 (the
    clamped expanded form of a point 100x the jitter from its own centroid,
    correct to fp32 at magnitude 1.3e9) and the UNFUSED arm returned
    ~51968 -- 4e-5 of the magnitude, 400 ulps, which fp32 accumulation of
    32 terms cannot produce and a TF32 tensor-core product does. The
    unfused arm is `gemm_nt` = MAX 26.5.0's `linalg.matmul`, and on NVIDIA
    that is TF32 BY DEFAULT with no compilable opt-out before Blackwell
    (`checks/kernel_matrix.mojo::column_vendor_fp32_matmul_is_tf32`,
    with the MAX source lines). **So the fused arm -- the one the fit
    ships -- was RIGHT, and the arm it was being diffed against was the
    lossy one.** The fix is the judge, not the kernel: both arms are now
    judged against a Float64 host oracle with a MAGNITUDE-relative budget
    (the expanded form `xn + yn - 2 x.c` is a cancellation at this
    fixture's scale, so a value-relative comparison was never a sound
    measure of either arm), the fused arm held to the fp32 budget on EVERY
    column, the unfused arm to the fp32 budget where the vendor product is
    exact and to the TF32 bound where it is not, the line printed naming
    which. The outputs of both arms are POISONED before the launch (NaN
    values, 0xDEADBEEF keys) and every row must be overwritten, which is
    the gate for the uninitialized-output class this failure was first
    read as. `check_assignment_arms_match_oracle` below is the
    well-conditioned twin of this comparison, where the distances are
    resolvable and the budget has teeth on this box.
    """
    var ctx = DeviceContext()
    var n = ARM_ROWS
    var d = ARM_FEATURES
    var k = ARM_CLUSTERS
    var unfused_lossy = vendor_fp32_matmul_is_lossy(
        TARGET_COLUMN, ctx.compute_capability()
    )
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        # The unfused arm's product is `pinned_gemm_nt_kernel` under
        # IDENTICAL (DEVIATION 526), fp32 on every column.
        unfused_lossy = False

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var c = ctx.enqueue_create_buffer[DType.float32](k * d)
    var xn = ctx.enqueue_create_buffer[DType.float32](n)
    var cn = ctx.enqueue_create_buffer[DType.float32](k)
    # Sized exactly as `fit` sizes it: data_batch x centroid_batch at the
    # default (0, 0) batch parameters = n * k.
    var dist_buf = ctx.enqueue_create_buffer[DType.float32](n * k)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k * d)
    for i in range(n):
        var m = i % k  # planted membership, round robin over 64 clusters
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(
                i * d + f,
                Float32(100 * (m + 1) + f) + _jitter(i, f),
            )
    for j in range(k):
        for f in range(d):
            hc.unsafe_ptr().unsafe_store(j * d + f, Float32(100 * (j + 1) + f))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[row_norm_kernel](
        xn.unsafe_ptr(), x.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(n, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    compute_centroid_norms(ctx, c, cn, k, d, METRIC_L2_EXPANDED)
    ctx.synchronize()

    # --- poison the tile buffer -------------------------------------------
    var hpoison = ctx.enqueue_create_host_buffer[DType.float32](n * k)
    for i in range(n * k):
        hpoison.unsafe_ptr().unsafe_store(i, ARM_SENTINEL)
    ctx.enqueue_copy(dst_buf=dist_buf, src_ptr=hpoison.unsafe_ptr())
    ctx.synchronize()

    # --- poison both arms' OUTPUTS too (DEVIATION 529): a row the kernel
    # never writes must read back as poison, not as whatever the allocator
    # left there -- which Metal tends to zero and CUDA does not.
    _poison_outputs(ctx, labels, min_dist, n)

    # --- the entry the fit calls -------------------------------------------
    min_cluster_and_distance_compute(
        ctx, x, xn, c, cn, dist_buf, labels, min_dist,
        n, d, k, METRIC_L2_EXPANDED, 0, 0,
    )
    ctx.synchronize()

    var hd = ctx.enqueue_create_host_buffer[DType.float32](n * k)
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hm = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=dist_buf)
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=labels)
    ctx.enqueue_copy(dst_ptr=hm.unsafe_ptr(), src_buf=min_dist)
    ctx.synchronize()

    var overwritten = 0
    for i in range(n * k):
        if hd.unsafe_ptr().unsafe_load(i) != ARM_SENTINEL:
            overwritten += 1
    if overwritten != 0:
        raise Error(
            "ASSIGNMENT TOOK THE UNFUSED ARM: the fit's entry point"
            " materialized " + String(overwritten) + " cells of a distance"
            " tile. cuVS's dispatch takes the FUSED arm for L2Expanded"
            " (kmeans_common.cuh:378-379, :430-449); ours no longer does."
        )

    var mislabeled = 0
    for i in range(n):
        if Int(hl.unsafe_ptr().unsafe_load(i)) != i % k:
            mislabeled += 1
    if mislabeled != 0:
        raise Error(
            "fused arm mislabeled " + String(mislabeled) + " of " + String(n)
            + " rows on a fixture separated by 100x the jitter"
        )
    var fused_poison = _surviving_poison(hl, hm, n)
    if fused_poison != 0:
        raise Error(
            "fused arm left " + String(fused_poison) + " of " + String(n)
            + " output rows UNWRITTEN (poison survived): the kernel is"
            " reading or skipping rows it must write"
        )

    # --- the sabotage half: the unfused arm MUST destroy the sentinel ------
    var labels_u = ctx.enqueue_create_buffer[DType.uint32](n)
    var min_dist_u = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    _poison_outputs(ctx, labels_u, min_dist_u, n)
    min_cluster_and_distance_compute_unfused(
        ctx, x, xn, c, cn, dist_buf, labels_u, min_dist_u,
        n, d, k, METRIC_L2_EXPANDED, 0, 0,
    )
    ctx.synchronize()
    var hdu = ctx.enqueue_create_host_buffer[DType.float32](n * k)
    var hlu = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hmu = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=hdu.unsafe_ptr(), src_buf=dist_buf)
    ctx.enqueue_copy(dst_ptr=hlu.unsafe_ptr(), src_buf=labels_u)
    ctx.enqueue_copy(dst_ptr=hmu.unsafe_ptr(), src_buf=min_dist_u)
    ctx.synchronize()

    var destroyed = 0
    for i in range(n * k):
        if hdu.unsafe_ptr().unsafe_load(i) != ARM_SENTINEL:
            destroyed += 1
    if destroyed == 0:
        raise Error(
            "SENTINEL CANNOT REGISTER: the unfused arm left the poison"
            " intact, so the sentinel is watching the wrong buffer and the"
            " fused-arm conclusion above is worthless."
        )

    var unfused_poison = _surviving_poison(hlu, hmu, n)
    if unfused_poison != 0:
        raise Error(
            "unfused arm left " + String(unfused_poison) + " of " + String(n)
            + " output rows UNWRITTEN (poison survived)"
        )

    var label_diff = 0
    for i in range(n):
        if hlu.unsafe_ptr().unsafe_load(i) != hl.unsafe_ptr().unsafe_load(i):
            label_diff += 1
    if label_diff != 0:
        raise Error(
            "the two arms disagree on " + String(label_diff)
            + " labels; they reduce with the same total order and must not"
        )

    # --- the distances, each arm against the Float64 oracle with a
    # MAGNITUDE-relative budget (docstring, DEVIATION 529) --------------
    var fused_budget = ARM_FP32_REL_BUDGET
    var unfused_budget = ARM_FP32_REL_BUDGET
    if unfused_lossy:
        unfused_budget += VENDOR_TF32_PRODUCT_REL_BOUND
    var worst_fused = Float64(0.0)
    var worst_unfused = Float64(0.0)
    var bit_equal = 0
    for i in range(n):
        var m = i % k
        var true_d = Float64(0.0)
        var xn64 = Float64(0.0)
        var cn64 = Float64(0.0)
        var dot = Float64(0.0)
        for f in range(d):
            var xv = Float64(hx.unsafe_ptr().unsafe_load(i * d + f))
            var cv = Float64(hc.unsafe_ptr().unsafe_load(m * d + f))
            true_d += (xv - cv) * (xv - cv)
            xn64 += xv * xv
            cn64 += cv * cv
            dot += xv * cv
        # The operands of the expanded form the kernels evaluate: their
        # magnitude is what an fp32 (or TF32) error is relative to.
        var mag = xn64 + cn64 + 2.0 * abs(dot)
        var a = Float64(hm.unsafe_ptr().unsafe_load(i))
        var b = Float64(hmu.unsafe_ptr().unsafe_load(i))
        if hm.unsafe_ptr().unsafe_load(i) == hmu.unsafe_ptr().unsafe_load(i):
            bit_equal += 1
        var rel_a = abs(a - true_d) / mag
        var rel_b = abs(b - true_d) / mag
        if rel_a > worst_fused:
            worst_fused = rel_a
        if rel_b > worst_unfused:
            worst_unfused = rel_b
    if worst_fused > fused_budget:
        raise Error(
            "FUSED min_dist misses the Float64 oracle: worst "
            + String(worst_fused)
            + " of the expanded form's magnitude, budget "
            + String(fused_budget)
            + " (fp32; this arm is ours and is fp32 on every column)"
        )
    if worst_unfused > unfused_budget:
        raise Error(
            "UNFUSED min_dist misses the Float64 oracle: worst "
            + String(worst_unfused)
            + " of the expanded form's magnitude, budget "
            + String(unfused_budget)
            + " (this arm's product is "
            + vendor_fp32_matmul_precision_name(
                TARGET_COLUMN, ctx.compute_capability()
            )
            + " on column "
            + column_name(TARGET_COLUMN)
            + ")"
        )

    print(
        "check_assignment_arm_dispatch OK: fused arm proved (0/"
        + String(n * k)
        + " tile cells written; unfused sabotage overwrote "
        + String(destroyed)
        + "); no output poison survived either arm; arms agree on all"
        " labels; min_dist vs the Float64 oracle, magnitude-relative:"
        " fused worst "
        + String(worst_fused)
        + " (budget "
        + String(fused_budget)
        + ", fp32), unfused worst "
        + String(worst_unfused)
        + " (budget "
        + String(unfused_budget)
        + ", "
        + (
            "the TF32 bound -- DEVIATION 529, this column's vendor product is "
            + vendor_fp32_matmul_precision_name(
                TARGET_COLUMN, ctx.compute_capability()
            )
            if unfused_lossy
            else "fp32"
        )
        + "); "
        + String(bit_equal)
        + "/"
        + String(n)
        + " rows bit-equal across arms"
    )


# --- privatized accumulation -------------------------------------------------

comptime ACC_ROWS = 512
comptime ACC_FEATURES = 32
comptime ACC_CLUSTERS = 64
comptime ACC_SCALE = Float32(1024.0)


def _acc_label(i: Int) -> Int:
    """Hashed, SCATTERED, and deliberately SKEWED labels.

    Not uniform and not round robin: a check whose expected value is the
    same in every cell verifies the total and nothing about placement (this
    repository's uniform-fixture rule). The hash scatters rows across all 64
    clusters in no spatial pattern, and every third row is forced onto
    cluster 7 so one cell family carries ~3x the contention of the rest.
    """
    var h = (i * 2654435761) % 1000003
    if h % 3 == 0:
        return 7
    return h % ACC_CLUSTERS


def _privatized_sums_dropped_flush_kernel[
    veclen: Int
](
    sums_i32: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    labels: MutPointer[UInt32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_features_in: Int32,
    n_clusters_in: Int32,
    scale_in: Float32,
):
    """CHECK-LOCAL SABOTAGE COPY of `accumulate_centroid_sums_privatized_
    kernel`, identical except that BLOCK 0 SKIPS ITS FLUSH (the `veclen`
    read parameter is mirrored so the copy stays identical to the shipped
    body -- PORTING.md 46). It exists only
    so `check_privatized_accumulate` can prove the per-block flush is
    load-bearing: a privatized kernel whose flush silently vanished would
    otherwise be indistinguishable from a slow day. Never shipped, never
    dispatched.
    """
    var n_rows = Int(n_rows_in)
    var n_features = Int(n_features_in)
    var used = Int(n_clusters_in) * n_features
    var total = n_rows * n_features
    var tid = Int(thread_idx.x)

    var priv = stack_allocation[
        PRIVATE_ACC_CELLS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var z = tid
    while z < used:
        priv.unsafe_store(z, Int32(0))
        z += Int(block_dim.x)
    barrier()

    var chunks = total // veclen
    var gid = Int(block_idx.x) * Int(block_dim.x) + tid
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while gid < chunks:
        var base = gid * veclen
        var row = base // n_features
        var f = base - row * n_features
        var label = Int(labels.unsafe_load(row))
        var w = weights.unsafe_load(row)
        var xv = x.unsafe_load[width=veclen](base)
        comptime for j in range(veclen):
            var q = Int32(xv[j] * w * scale_in)
            _ = Atomic.fetch_add(
                priv.unsafe_offset(label * n_features + f + j), q
            )
        gid += stride
    barrier()

    if Int(block_idx.x) == 0:
        return  # THE SABOTAGE: this block's partial never reaches global.

    var cc = tid
    while cc < used:
        var v = priv.unsafe_load(cc)
        if v != Int32(0):
            _ = Atomic.fetch_add(sums_i32.unsafe_offset(cc), v)
        cc += Int(block_dim.x)


def _zero_and_wait(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32], n: Int
) raises:
    ctx.enqueue_function[zero_i32_kernel](
        buf.unsafe_ptr(),
        Int32(n),
        grid_dim=((n + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )


# --- both assignment arms against a resolvable oracle -----------------------

comptime ORACLE_ROWS = 512
comptime ORACLE_FEATURES = 32
comptime ORACLE_CLUSTERS = 64


def _oracle_val(i: Int, salt: Int) -> Float32:
    """A hashed value in [-1, 1): splitmix64 bits, so adjacent cells land
    nowhere near each other and no two rows are alike."""
    var z = UInt64(i + 1) * 0x9E3779B97F4A7C15 + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(Int(z % 2000001) - 1000000) * Float32(1.0e-6)


def _run_both_arms_on(
    ctx: DeviceContext,
    hx: HostBuffer[DType.float32],
    hc: HostBuffer[DType.float32],
    n: Int,
    d: Int,
    k: Int,
    mut hl: HostBuffer[DType.uint32],
    mut hm: HostBuffer[DType.float32],
    mut hlu: HostBuffer[DType.uint32],
    mut hmu: HostBuffer[DType.float32],
) raises:
    """Upload one fixture, compute norms the way the fit does, run the
    fused entry and the unfused arm on it with poisoned outputs, and read
    both back. Shared by the oracle check and its self-sabotage."""
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var c = ctx.enqueue_create_buffer[DType.float32](k * d)
    var xn = ctx.enqueue_create_buffer[DType.float32](n)
    var cn = ctx.enqueue_create_buffer[DType.float32](k)
    var dist_buf = ctx.enqueue_create_buffer[DType.float32](n * k)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n)
    var labels_u = ctx.enqueue_create_buffer[DType.uint32](n)
    var min_dist_u = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()
    ctx.enqueue_function[row_norm_kernel](
        xn.unsafe_ptr(), x.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(n, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    compute_centroid_norms(ctx, c, cn, k, d, METRIC_L2_EXPANDED)
    ctx.synchronize()
    _poison_outputs(ctx, labels, min_dist, n)
    _poison_outputs(ctx, labels_u, min_dist_u, n)
    min_cluster_and_distance_compute(
        ctx, x, xn, c, cn, dist_buf, labels, min_dist,
        n, d, k, METRIC_L2_EXPANDED, 0, 0,
    )
    ctx.synchronize()
    min_cluster_and_distance_compute_unfused(
        ctx, x, xn, c, cn, dist_buf, labels_u, min_dist_u,
        n, d, k, METRIC_L2_EXPANDED, 0, 0,
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=labels)
    ctx.enqueue_copy(dst_ptr=hm.unsafe_ptr(), src_buf=min_dist)
    ctx.enqueue_copy(dst_ptr=hlu.unsafe_ptr(), src_buf=labels_u)
    ctx.enqueue_copy(dst_ptr=hmu.unsafe_ptr(), src_buf=min_dist_u)
    ctx.synchronize()


struct _ArmVerdict(Copyable, Movable):
    """One arm judged against the oracle: worst magnitude-relative miss of
    `min_dist`, how many labels are not the oracle argmin, and how many of
    those sit on a gap the arm's budget could legitimately re-decide."""
    var worst_rel: Float64
    var wrong_labels: Int
    var excused_labels: Int
    var poison: Int

    def __init__(out self):
        self.worst_rel = 0.0
        self.wrong_labels = 0
        self.excused_labels = 0
        self.poison = 0


def _judge_arm(
    hx: HostBuffer[DType.float32],
    hc: HostBuffer[DType.float32],
    hl: HostBuffer[DType.uint32],
    hm: HostBuffer[DType.float32],
    n: Int,
    d: Int,
    k: Int,
    budget: Float64,
) -> _ArmVerdict:
    """Per row: the Float64 true squared distance to EVERY centroid, the
    argmin with the lowest-index tie rule, and the expanded form's operand
    magnitude at the winner. The arm's value must be within `mag x budget`
    of the true minimum; its label must be the argmin unless the runner-up
    is within that same margin (then the arm's arithmetic may legitimately
    have re-decided it, and the row is EXCUSED, counted separately)."""
    var v = _ArmVerdict()
    for i in range(n):
        var xn64 = Float64(0.0)
        for f in range(d):
            var xv = Float64(hx.unsafe_ptr().unsafe_load(i * d + f))
            xn64 += xv * xv
        var best_j = 0
        var best_d = Float64(1.0e300)
        var second_d = Float64(1.0e300)
        var best_mag = Float64(0.0)
        for j in range(k):
            var dd = Float64(0.0)
            var cn64 = Float64(0.0)
            var dot = Float64(0.0)
            for f in range(d):
                var xv = Float64(hx.unsafe_ptr().unsafe_load(i * d + f))
                var cv = Float64(hc.unsafe_ptr().unsafe_load(j * d + f))
                dd += (xv - cv) * (xv - cv)
                cn64 += cv * cv
                dot += xv * cv
            if dd < best_d:
                second_d = best_d
                best_d = dd
                best_j = j
                best_mag = xn64 + cn64 + 2.0 * abs(dot)
            elif dd < second_d:
                second_d = dd
        var got_v = hm.unsafe_ptr().unsafe_load(i)
        var got_k = hl.unsafe_ptr().unsafe_load(i)
        if got_k == ARM_POISON_KEY or got_v != got_v:
            v.poison += 1
            continue
        var rel = abs(Float64(got_v) - best_d) / best_mag
        if rel > v.worst_rel:
            v.worst_rel = rel
        if Int(got_k) != best_j:
            if second_d - best_d <= best_mag * budget:
                v.excused_labels += 1
            else:
                v.wrong_labels += 1
    return v^


def check_assignment_arms_match_oracle() raises:
    """BOTH assignment arms against a Float64 oracle on a fixture whose
    distances are RESOLVABLE, with the per-column budget of DEVIATION 529,
    and a self-sabotage that proves the budget has teeth on this box.

    `check_assignment_arm_dispatch`'s fixture plants centers 100 apart so
    its labels are beyond doubt, and at that scale the expanded form
    `xn + yn - 2 x.c` is a cancellation: the true distance is a few units
    and the operands are 1e9, so the clamped result is 0 on every correct
    arm and a value comparison there has nothing to measure (on the Apple
    M4 both arms return 0.0 bit for bit, and the H100 leg's 51968 came
    from a TF32 product, not from a wrong kernel). This check is the one
    where the numbers mean something: x and the centroids are hashed in
    [-1, 1), so the true minimum is ~10 and the operand magnitude ~30, and
    an fp32 arm resolves the value to ~1e-7 of the magnitude where a TF32
    one lands ~1e-4.

    The dispatch shape is the fit's (d=32, k=64: normal policy, veclen 4,
    the vendor matmul at n_clusters=64) so the arms run are the arms that
    ship. The argmin is the ORACLE'S (lowest index on a tie), not a
    planted label, and a label that disagrees is wrong unless the
    runner-up sits inside the arm's own budget, in which case it is
    counted as EXCUSED and printed -- that is what a TF32 product does to
    a near-tie and the count is the measurement of it.

    THE SELF-SABOTAGE IS THE PROOF THE GATE WOULD FAIL ON APPLE WITH THE
    BUG PRESENT. The same fixture is rounded to ten mantissa bits on the
    host -- a tensor core's operand conversion, modelled exactly -- and
    run through the FUSED arm, then judged against the UNROUNDED oracle.
    That product is TF32-class by construction, on every column, without
    a tensor core; the fp32 budget must REJECT it and the TF32 bound must
    ADMIT it. If the first half fails, the fp32 budget cannot see the
    class of error the H100 leg surfaced and the per-column split is
    decoration; if the second fails, the TF32 bound is too tight for what
    it names. Run on the fused arm deliberately: its arithmetic is known
    fp32 on every column, so the ONLY lossy step is the planted one.
    """
    var ctx = DeviceContext()
    var n = ORACLE_ROWS
    var d = ORACLE_FEATURES
    var k = ORACLE_CLUSTERS
    var cc = ctx.compute_capability()
    var unfused_lossy = vendor_fp32_matmul_is_lossy(TARGET_COLUMN, cc)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        unfused_lossy = False  # pinned_gemm_nt_kernel, DEVIATION 526

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k * d)
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hm = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hlu = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hmu = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n * d):
        hx.unsafe_ptr().unsafe_store(i, _oracle_val(i, 11))
    for i in range(k * d):
        hc.unsafe_ptr().unsafe_store(i, _oracle_val(i, 23))

    _run_both_arms_on(ctx, hx, hc, n, d, k, hl, hm, hlu, hmu)

    var fused_budget = ARM_FP32_REL_BUDGET
    var unfused_budget = ARM_FP32_REL_BUDGET
    if unfused_lossy:
        unfused_budget += VENDOR_TF32_PRODUCT_REL_BOUND
    var vf = _judge_arm(hx, hc, hl, hm, n, d, k, fused_budget)
    var vu = _judge_arm(hx, hc, hlu, hmu, n, d, k, unfused_budget)
    if vf.poison != 0 or vu.poison != 0:
        raise Error(
            "output poison survived: fused " + String(vf.poison)
            + " rows, unfused " + String(vu.poison) + " rows of " + String(n)
        )
    if vf.worst_rel > fused_budget:
        raise Error(
            "FUSED arm misses the Float64 oracle on the resolvable fixture:"
            " worst " + String(vf.worst_rel) + " of the operand magnitude,"
            " budget " + String(fused_budget) + " (fp32, every column)"
        )
    if vf.wrong_labels != 0:
        raise Error(
            "FUSED arm returned " + String(vf.wrong_labels) + " labels that"
            " are not the oracle argmin and not within the fp32 budget of"
            " it"
        )
    if vu.worst_rel > unfused_budget:
        raise Error(
            "UNFUSED arm misses the Float64 oracle on the resolvable"
            " fixture: worst " + String(vu.worst_rel) + " of the operand"
            " magnitude, budget " + String(unfused_budget) + " (product is "
            + vendor_fp32_matmul_precision_name(TARGET_COLUMN, cc)
            + " on column " + column_name(TARGET_COLUMN) + ")"
        )
    if vu.wrong_labels != 0:
        raise Error(
            "UNFUSED arm returned " + String(vu.wrong_labels) + " labels"
            " that are not the oracle argmin and not within its budget ("
            + String(unfused_budget) + ") of it"
        )

    # --- the self-sabotage: a TF32-class product through the fused arm --
    var hx_t = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hc_t = ctx.enqueue_create_host_buffer[DType.float32](k * d)
    var hl_t = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hm_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hlu_t = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hmu_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    var moved = 0
    for i in range(n * d):
        var v0 = hx.unsafe_ptr().unsafe_load(i)
        var t = _round_to_tf32(v0)
        if t != v0:
            moved += 1
        hx_t.unsafe_ptr().unsafe_store(i, t)
    for i in range(k * d):
        hc_t.unsafe_ptr().unsafe_store(
            i, _round_to_tf32(hc.unsafe_ptr().unsafe_load(i))
        )
    if moved < (n * d) // 2:
        raise Error(
            "the TF32 rounding moved only " + String(moved) + " of "
            + String(n * d) + " operands; the sabotage is not a sabotage"
        )
    _run_both_arms_on(ctx, hx_t, hc_t, n, d, k, hl_t, hm_t, hlu_t, hmu_t)
    # Judged against the UNROUNDED fixture's oracle.
    var tf32_bound = ARM_FP32_REL_BUDGET + VENDOR_TF32_PRODUCT_REL_BOUND
    var vs = _judge_arm(hx, hc, hl_t, hm_t, n, d, k, tf32_bound)
    if vs.worst_rel <= ARM_FP32_REL_BUDGET:
        raise Error(
            "SABOTAGE NOT SEEN: the fused arm on TF32-rounded operands"
            " passed the fp32 budget (worst " + String(vs.worst_rel)
            + " <= " + String(ARM_FP32_REL_BUDGET) + "), so the budget"
            " cannot see a TF32-class product and DEVIATION 529's"
            " per-column split is decoration"
        )
    if vs.worst_rel > tf32_bound:
        raise Error(
            "the TF32 bound is too tight for what it names: a TF32-rounded"
            " product lands " + String(vs.worst_rel) + " > "
            + String(tf32_bound)
        )
    if vs.wrong_labels != 0:
        raise Error(
            "the TF32-rounded product moved " + String(vs.wrong_labels)
            + " labels by more than the TF32 bound explains"
        )

    print(
        "check_assignment_arms_match_oracle OK: " + String(n) + " rows x "
        + String(k) + " clusters x " + String(d) + " features hashed in"
        " [-1, 1), both arms vs a Float64 oracle (argmin and value),"
        " no poison survived; fused worst " + String(vf.worst_rel)
        + " of the operand magnitude (budget " + String(fused_budget)
        + ", fp32), " + String(vf.excused_labels) + " near-tie labels"
        " excused; unfused worst " + String(vu.worst_rel) + " (budget "
        + String(unfused_budget) + ", "
        + (
            "the TF32 bound, DEVIATION 529 -- this column's vendor product is "
            + vendor_fp32_matmul_precision_name(TARGET_COLUMN, cc)
            if unfused_lossy else "fp32"
        )
        + "), " + String(vu.excused_labels) + " near-tie labels excused."
        " SABOTAGE: the fused arm on operands rounded to 10 mantissa bits ("
        + String(moved) + " of " + String(n * d) + " moved) lands worst "
        + String(vs.worst_rel) + " with " + String(vs.excused_labels)
        + " labels re-decided -- REJECTED by the fp32 budget, ADMITTED by"
        " the TF32 bound " + String(tf32_bound) + ", so the two budgets"
        " separate a TF32-class arm from an fp32 one on this box (column "
        + column_name(TARGET_COLUMN) + ", compute capability " + String(cc)
        + ")"
    )


def check_privatized_accumulate() raises:
    """The privatized accumulate against the direct one: bit-identical Int32
    totals on a hashed, scattered, skewed fixture; a dropped-flush sabotage
    that must move them; and a run-twice determinism assertion.

    Bit-identity is the DESIGN CLAIM of the privatized arm (integer adds
    re-associated per block are the same multiset of adds), so it is
    asserted exactly -- every cell, `!=` on Int32 -- with no tolerance.
    The shape (512 x 32, k=64) satisfies the dispatch guard the same way
    the fit's 4M x 32 k=64 bench arm does: cells 2048 <= PRIVATE_ACC_CELLS,
    work 16384 >= PRIVATE_MIN_WORK. That is asserted too, so the fixture
    cannot silently drift onto the other arm.
    """
    var ctx = DeviceContext()
    var n = ACC_ROWS
    var d = ACC_FEATURES
    var k = ACC_CLUSTERS
    var cd = k * d

    if not (cd <= PRIVATE_ACC_CELLS and n * d >= PRIVATE_MIN_WORK):
        raise Error(
            "fixture no longer selects the privatized arm; the check is"
            " testing nothing"
        )

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n)
    var weights = ctx.enqueue_create_buffer[DType.float32](n)
    var sums = ctx.enqueue_create_buffer[DType.int32](cd)
    var wsum = ctx.enqueue_create_buffer[DType.int32](k)
    ctx.synchronize()

    # The read-width arm this fixture pins (PORTING.md 46): d=32 on real
    # device buffers must take the veclen=4 instantiation. The dispatch
    # routes on this same function, so run B below exercises the VECTOR
    # arm, and the bitwise comparison against the scalar direct oracle is
    # what proves that arm visits every cell exactly once -- on hashed
    # scattered data a misindexed or dropped lane cannot cancel.
    var vl = fused_veclen_for(d, Int(x.unsafe_ptr()), Int(x.unsafe_ptr()))
    if vl != 4:
        raise Error(
            "fixture no longer selects the veclen=4 read arm (got "
            + String(vl) + "); the vector load path would go untested"
        )

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        hl.unsafe_ptr().unsafe_store(i, UInt32(_acc_label(i)))
        hw.unsafe_ptr().unsafe_store(i, Float32(1.0) + Float32(i % 5) * 0.25)
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _jitter(i, f) * 10.0)
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=labels, src_ptr=hl.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var h_direct_s = ctx.enqueue_create_host_buffer[DType.int32](cd)
    var h_direct_w = ctx.enqueue_create_host_buffer[DType.int32](k)
    var h_priv_s = ctx.enqueue_create_host_buffer[DType.int32](cd)
    var h_priv_w = ctx.enqueue_create_host_buffer[DType.int32](k)
    var h_again_s = ctx.enqueue_create_host_buffer[DType.int32](cd)
    var h_sab_s = ctx.enqueue_create_host_buffer[DType.int32](cd)

    # --- run A: the DIRECT kernels, as the oracle -------------------------
    _zero_and_wait(ctx, sums, cd)
    _zero_and_wait(ctx, wsum, k)
    ctx.enqueue_function[accumulate_centroid_sums_kernel](
        sums.unsafe_ptr(), x.unsafe_ptr(), labels.unsafe_ptr(),
        weights.unsafe_ptr(), Int32(n), Int32(d), ACC_SCALE,
        grid_dim=(accumulate_grid_blocks(n * d, 0), 1, 1),
        block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
    )
    ctx.enqueue_function[accumulate_weight_per_cluster_kernel](
        wsum.unsafe_ptr(), labels.unsafe_ptr(), weights.unsafe_ptr(),
        Int32(n), ACC_SCALE,
        grid_dim=((n + REDUCE_BY_KEY_TPB - 1) // REDUCE_BY_KEY_TPB, 1, 1),
        block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=h_direct_s.unsafe_ptr(), src_buf=sums)
    ctx.enqueue_copy(dst_ptr=h_direct_w.unsafe_ptr(), src_buf=wsum)
    ctx.synchronize()

    # --- run B: the shipped dispatch, which must take the privatized arm --
    _zero_and_wait(ctx, sums, cd)
    _zero_and_wait(ctx, wsum, k)
    launch_accumulate_centroid_sums(
        ctx, sums, x, labels, weights, n, d, k, ACC_SCALE
    )
    launch_accumulate_weight_per_cluster(
        ctx, wsum, labels, weights, n, k, ACC_SCALE
    )
    ctx.enqueue_copy(dst_ptr=h_priv_s.unsafe_ptr(), src_buf=sums)
    ctx.enqueue_copy(dst_ptr=h_priv_w.unsafe_ptr(), src_buf=wsum)
    ctx.synchronize()

    var s_diff = 0
    for i in range(cd):
        if h_priv_s.unsafe_ptr().unsafe_load(i) != h_direct_s.unsafe_ptr().unsafe_load(i):
            s_diff += 1
    var w_diff = 0
    for i in range(k):
        if h_priv_w.unsafe_ptr().unsafe_load(i) != h_direct_w.unsafe_ptr().unsafe_load(i):
            w_diff += 1
    if s_diff != 0 or w_diff != 0:
        raise Error(
            "privatized totals are NOT bit-identical to the direct kernel's:"
            " " + String(s_diff) + " of " + String(cd) + " sum cells and "
            + String(w_diff) + " of " + String(k) + " weight cells differ."
            " Integer adds cannot do that, so one arm is visiting the wrong"
            " cells."
        )

    # --- determinism: the same launch twice must match bitwise ------------
    _zero_and_wait(ctx, sums, cd)
    launch_accumulate_centroid_sums(
        ctx, sums, x, labels, weights, n, d, k, ACC_SCALE
    )
    ctx.enqueue_copy(dst_ptr=h_again_s.unsafe_ptr(), src_buf=sums)
    ctx.synchronize()
    for i in range(cd):
        if h_again_s.unsafe_ptr().unsafe_load(i) != h_priv_s.unsafe_ptr().unsafe_load(i):
            raise Error(
                "two identical privatized runs disagree at cell " + String(i)
                + ": the accumulator is not order-independent after all"
            )

    # --- sabotage: drop block 0's flush; the totals MUST move -------------
    _zero_and_wait(ctx, sums, cd)
    comptime sab_kern = _privatized_sums_dropped_flush_kernel[4]
    ctx.enqueue_function[sab_kern](
        sums.unsafe_ptr(), x.unsafe_ptr(), labels.unsafe_ptr(),
        weights.unsafe_ptr(), Int32(n), Int32(d), Int32(k), ACC_SCALE,
        grid_dim=(
            accumulate_grid_blocks((n * d) // 4, PRIVATE_ACC_CELLS * 4), 1, 1
        ),
        block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=h_sab_s.unsafe_ptr(), src_buf=sums)
    ctx.synchronize()
    var moved = 0
    for i in range(cd):
        if h_sab_s.unsafe_ptr().unsafe_load(i) != h_priv_s.unsafe_ptr().unsafe_load(i):
            moved += 1
    if moved == 0:
        raise Error(
            "SABOTAGE FAILED TO REGISTER: dropping one block's flush moved"
            " no total, so either the flush is not load-bearing or every"
            " partial in block 0 was zero and the fixture is too weak."
        )

    print(
        "check_privatized_accumulate OK: "
        + String(cd)
        + " sum cells + "
        + String(k)
        + " weight cells bit-identical direct vs privatized, run-twice"
        " bitwise equal, dropped flush moved "
        + String(moved)
        + " cells, veclen=4 read arm pinned"
    )


def _accumulate_arm_correct(d: Int, expect_veclen: Int) raises:
    """Run the REAL accumulate dispatch at a feature count whose read-width
    selection lands on `expect_veclen`, and hold its totals bitwise to the
    scalar direct-kernel oracle on the hashed, scattered, skewed fixture.

    Mirrors `_policy_arm_correct`'s method: (1) the selection function --
    the same `fused_veclen_for` the dispatch routes on -- must return
    `expect_veclen` on the real buffer, so the arm named is the arm run;
    (2) the fixture must satisfy the privatized guard, so the dispatch
    cannot silently fall to the direct arm and compare the oracle against
    itself; (3) the oracle totals must be nonzero somewhere, so a dispatch
    that silently launched nothing cannot pass. Bit-identity is the design
    claim, so every cell compares with `!=`, no tolerance.
    """
    var ctx = DeviceContext()
    var n = ACC_ROWS
    var k = ACC_CLUSTERS
    var cd = k * d

    if not (cd <= PRIVATE_ACC_CELLS and n * d >= PRIVATE_MIN_WORK):
        raise Error(
            "arm fixture d=" + String(d) + " does not select the privatized"
            " arm; the check would compare the oracle to itself"
        )

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n)
    var weights = ctx.enqueue_create_buffer[DType.float32](n)
    var sums = ctx.enqueue_create_buffer[DType.int32](cd)
    var wsum = ctx.enqueue_create_buffer[DType.int32](k)
    ctx.synchronize()

    var vl = fused_veclen_for(d, Int(x.unsafe_ptr()), Int(x.unsafe_ptr()))
    if vl != expect_veclen:
        raise Error(
            "fused_veclen_for(d=" + String(d) + ") on the real buffer"
            " returned " + String(vl) + ", expected "
            + String(expect_veclen)
            + " -- the intended read arm is not the one the dispatch takes"
        )

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        hl.unsafe_ptr().unsafe_store(i, UInt32(_acc_label(i)))
        hw.unsafe_ptr().unsafe_store(i, Float32(1.0) + Float32(i % 5) * 0.25)
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _jitter(i, f) * 10.0)
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=labels, src_ptr=hl.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var h_direct_s = ctx.enqueue_create_host_buffer[DType.int32](cd)
    var h_direct_w = ctx.enqueue_create_host_buffer[DType.int32](k)
    var h_arm_s = ctx.enqueue_create_host_buffer[DType.int32](cd)
    var h_arm_w = ctx.enqueue_create_host_buffer[DType.int32](k)

    _zero_and_wait(ctx, sums, cd)
    _zero_and_wait(ctx, wsum, k)
    ctx.enqueue_function[accumulate_centroid_sums_kernel](
        sums.unsafe_ptr(), x.unsafe_ptr(), labels.unsafe_ptr(),
        weights.unsafe_ptr(), Int32(n), Int32(d), ACC_SCALE,
        grid_dim=(accumulate_grid_blocks(n * d, 0), 1, 1),
        block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
    )
    ctx.enqueue_function[accumulate_weight_per_cluster_kernel](
        wsum.unsafe_ptr(), labels.unsafe_ptr(), weights.unsafe_ptr(),
        Int32(n), ACC_SCALE,
        grid_dim=((n + REDUCE_BY_KEY_TPB - 1) // REDUCE_BY_KEY_TPB, 1, 1),
        block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=h_direct_s.unsafe_ptr(), src_buf=sums)
    ctx.enqueue_copy(dst_ptr=h_direct_w.unsafe_ptr(), src_buf=wsum)
    ctx.synchronize()

    var nonzero = 0
    for i in range(cd):
        if h_direct_s.unsafe_ptr().unsafe_load(i) != Int32(0):
            nonzero += 1
    if nonzero == 0:
        raise Error(
            "oracle produced all-zero sums at d=" + String(d)
            + "; the fixture is too weak to catch a no-op arm"
        )

    _zero_and_wait(ctx, sums, cd)
    _zero_and_wait(ctx, wsum, k)
    launch_accumulate_centroid_sums(
        ctx, sums, x, labels, weights, n, d, k, ACC_SCALE
    )
    launch_accumulate_weight_per_cluster(
        ctx, wsum, labels, weights, n, k, ACC_SCALE
    )
    ctx.enqueue_copy(dst_ptr=h_arm_s.unsafe_ptr(), src_buf=sums)
    ctx.enqueue_copy(dst_ptr=h_arm_w.unsafe_ptr(), src_buf=wsum)
    ctx.synchronize()

    var s_diff = 0
    for i in range(cd):
        if h_arm_s.unsafe_ptr().unsafe_load(i) != h_direct_s.unsafe_ptr().unsafe_load(i):
            s_diff += 1
    var w_diff = 0
    for i in range(k):
        if h_arm_w.unsafe_ptr().unsafe_load(i) != h_direct_w.unsafe_ptr().unsafe_load(i):
            w_diff += 1
    if s_diff != 0 or w_diff != 0:
        raise Error(
            "veclen=" + String(expect_veclen) + " read arm at d=" + String(d)
            + " is NOT bit-identical to the direct oracle: " + String(s_diff)
            + " of " + String(cd) + " sum cells and " + String(w_diff)
            + " of " + String(k) + " weight cells differ"
        )
    print(
        "  veclen=" + String(expect_veclen) + " accumulate arm at d="
        + String(d) + ": " + String(cd) + " sum cells + " + String(k)
        + " weight cells bit-identical to the direct oracle ("
        + String(nonzero) + " nonzero)"
    )


def check_accumulate_veclen_dispatch() raises:
    """Pin the accumulate read-width dispatch's NON-DEFAULT arms where the
    selection ladder takes them (PORTING.md 46): d=33 rejects both vector
    widths (132 bytes per row is neither a 16- nor an 8-byte multiple), and
    d=34 takes the 2-wide arm. The default veclen=4 arm is pinned inside
    `check_privatized_accumulate`, on the same fixture that proves the
    privatized structure. An arm nothing exercises proves nothing, so both
    are run for real through the shipped dispatch against the scalar
    direct-kernel oracle on scattered hashed data.
    """
    _accumulate_arm_correct(33, 1)
    _accumulate_arm_correct(34, 2)
    print(
        "check_accumulate_veclen_dispatch OK: scalar read arm"
        " reached-and-correct at d=33, 2-wide arm at d=34, 4-wide arm"
        " pinned in check_privatized_accumulate"
    )


# --- fused policy selection --------------------------------------------------

#: Row tiles of Mblk=64 BEYOND the launcher's grid.y cap that the policy
#: fixture carries, so the kernel's m grid-stride loop must cover them. The
#: fixture used to be a constant 8256 rows = 129 tiles, "more than the M4's
#: minGridSize of 120 blocks" -- true of one column and false of the
#: others: `launchConfigGenerator`'s cap is `numSMs x blocksPerSM`, 120 on
#: the Apple column and an order of magnitude more on a 132-SM H100 or a
#: 304-CU MI300X, where 129 tiles fit under the cap and the check refused
#: itself with "fixture too small" (found by the `-D MOJOLEARN_COLUMN_NVIDIA`
#: build on the Mac while closing DEVIATION 529, and it would have been the
#: H100 leg's next stop in this file). The row count is now DERIVED from
#: the launcher's own cap at runtime, `_policy_rows`, so the inequality the
#: check verifies is constructed on every column rather than assumed.
comptime POL_TILES_PAST_CAP = 9
comptime POL_CLUSTERS = 40


def _policy_rows(expect_veclen: Int) raises -> Int:
    """`(grid.y cap + POL_TILES_PAST_CAP) x Mblk` rows for the normal
    policy at this veclen. The cap is read from `launch_config_generator`
    itself by asking it about a row count no device's cap exceeds."""
    comptime mblk = 4 * FUSED_NORMAL_TR
    var probe = launch_config_generator(
        1 << 30,
        POL_CLUSTERS,
        mblk,
        4 * FUSED_NORMAL_TC,
        FUSED_NORMAL_TR * FUSED_NORMAL_TC,
        fused_smem_bytes(False, expect_veclen),
    )
    return (probe[1] + POL_TILES_PAST_CAP) * mblk


def _planted_gap_within(
    hx: HostBuffer[DType.float32],
    hc: HostBuffer[DType.float32],
    i: Int,
    d: Int,
    k: Int,
    budget: Float64,
) -> Bool:
    """Float64 oracle for ONE row of the planted fixture: True iff the gap
    between the nearest and second-nearest centroid is within `budget` of
    the expanded form's operand magnitude at the winner -- the margin a
    product with that budget may legitimately re-decide."""
    var xn64 = Float64(0.0)
    for f in range(d):
        var xv = Float64(hx.unsafe_ptr().unsafe_load(i * d + f))
        xn64 += xv * xv
    var best_d = Float64(1.0e300)
    var second_d = Float64(1.0e300)
    var best_mag = Float64(0.0)
    for j in range(k):
        var dd = Float64(0.0)
        var cn64 = Float64(0.0)
        var dot = Float64(0.0)
        for f in range(d):
            var xv = Float64(hx.unsafe_ptr().unsafe_load(i * d + f))
            var cv = Float64(hc.unsafe_ptr().unsafe_load(j * d + f))
            dd += (xv - cv) * (xv - cv)
            cn64 += cv * cv
            dot += xv * cv
        if dd < best_d:
            second_d = best_d
            best_d = dd
            best_mag = xn64 + cn64 + 2.0 * abs(dot)
        elif dd < second_d:
            second_d = dd
    return second_d - best_d <= best_mag * budget


def _policy_arm_correct(d: Int, expect_veclen: Int) raises:
    """Run the REAL launcher at a feature count that selects the given
    veclen arm, and require (1) the selection function -- the same one the
    launcher dispatches on -- returns `expect_veclen` for the real buffers,
    (2) every planted label is recovered, including rows beyond
    `grid.y * Mblk`, which only the m grid-stride loop can reach, and
    (3) the unfused arm agrees bitwise on every label (both arms reduce
    with `raft::argmin_op`'s total order and the planted separation is
    ~330,000 squared against a float32 error orders smaller, so equality is
    the expectation, not luck)."""
    var ctx = DeviceContext()
    var n = _policy_rows(expect_veclen)
    var k = POL_CLUSTERS

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var c = ctx.enqueue_create_buffer[DType.float32](k * d)
    var xn = ctx.enqueue_create_buffer[DType.float32](n)
    var cn = ctx.enqueue_create_buffer[DType.float32](k)
    var dist_buf = ctx.enqueue_create_buffer[DType.float32](n * k)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n)
    var labels_u = ctx.enqueue_create_buffer[DType.uint32](n)
    var min_dist_u = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()

    # The selection the launcher will make, asserted BEFORE the run so a
    # failure names the arm rather than the symptom.
    var vl = fused_veclen_for(d, Int(x.unsafe_ptr()), Int(c.unsafe_ptr()))
    if vl != expect_veclen:
        raise Error(
            "fused_veclen_for(k=" + String(d) + ") on real buffers returned "
            + String(vl) + ", expected " + String(expect_veclen)
            + " -- the intended arm is not the one the launcher takes"
        )
    if fused_is_skinny(d):
        raise Error(
            "fixture d=" + String(d) + " unexpectedly selects the skinny"
            " policy; this helper pins the normal-policy arms"
        )

    # The fixture must actually exercise the m grid-stride: the launcher's
    # grid.y (their launchConfigGenerator, this column's inputs) must be
    # SMALLER than the number of row tiles. `_policy_rows` constructed that;
    # this is the independent verification of it.
    var y_chunks = (n + 4 * FUSED_NORMAL_TR - 1) // (4 * FUSED_NORMAL_TR)
    var cfg = launch_config_generator(
        n,
        k,
        4 * FUSED_NORMAL_TR,
        4 * FUSED_NORMAL_TC,
        FUSED_NORMAL_TR * FUSED_NORMAL_TC,
        fused_smem_bytes(False, expect_veclen),
    )
    if cfg[1] >= y_chunks:
        raise Error(
            "fixture too small: grid.y " + String(cfg[1]) + " covers all "
            + String(y_chunks) + " row tiles, so the m grid-stride loop"
            " would go unexercised"
        )

    # Planted membership round robin over the clusters, hashed jitter so no
    # two rows are identical (uniform data hides placement bugs).
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k * d)
    for i in range(n):
        var mem = i % k
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(
                i * d + f, Float32(100 * (mem + 1) + f) + _jitter(i, f)
            )
    for j in range(k):
        for f in range(d):
            hc.unsafe_ptr().unsafe_store(j * d + f, Float32(100 * (j + 1) + f))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[row_norm_kernel](
        xn.unsafe_ptr(), x.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(n, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    compute_centroid_norms(ctx, c, cn, k, d, METRIC_L2_EXPANDED)
    ctx.synchronize()

    min_cluster_and_distance_compute(
        ctx, x, xn, c, cn, dist_buf, labels, min_dist,
        n, d, k, METRIC_L2_EXPANDED, 0, 0,
    )
    min_cluster_and_distance_compute_unfused(
        ctx, x, xn, c, cn, dist_buf, labels_u, min_dist_u,
        n, d, k, METRIC_L2_EXPANDED, 0, 0,
    )
    ctx.synchronize()

    var hl = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hlu = ctx.enqueue_create_host_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=labels)
    ctx.enqueue_copy(dst_ptr=hlu.unsafe_ptr(), src_buf=labels_u)
    ctx.synchronize()

    # The unfused arm's product is the vendor matmul: on a lossy column
    # (DEVIATION 529) a label it moves is excused only if the row's oracle
    # gap between best and runner-up is inside the TF32 bound of the
    # operand magnitude -- the same rule `_judge_arm` applies.
    var unfused_budget = ARM_FP32_REL_BUDGET
    if vendor_fp32_matmul_is_lossy(TARGET_COLUMN, ctx.compute_capability()):
        unfused_budget += VENDOR_TF32_PRODUCT_REL_BOUND
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        unfused_budget = ARM_FP32_REL_BUDGET
    var mislabeled = 0
    var tail_checked = 0
    var arm_diff = 0
    var arm_diff_excused = 0
    var stride_floor = cfg[1] * 4 * FUSED_NORMAL_TR
    for i in range(n):
        if Int(hl.unsafe_ptr().unsafe_load(i)) != i % k:
            mislabeled += 1
        if i >= stride_floor:
            tail_checked += 1
        if hl.unsafe_ptr().unsafe_load(i) != hlu.unsafe_ptr().unsafe_load(i):
            if _planted_gap_within(hx, hc, i, d, k, unfused_budget):
                arm_diff_excused += 1
            else:
                arm_diff += 1
    if mislabeled != 0:
        raise Error(
            "veclen=" + String(expect_veclen) + " arm mislabeled "
            + String(mislabeled) + " of " + String(n) + " rows at d="
            + String(d)
        )
    if tail_checked == 0:
        raise Error(
            "no row beyond grid.y * Mblk existed; the grid-stride tail was"
            " not checked"
        )
    if arm_diff != 0:
        raise Error(
            "fused and unfused labels differ on " + String(arm_diff)
            + " rows at d=" + String(d) + " beyond what the unfused arm's"
            " budget (" + String(unfused_budget) + " of the magnitude)"
            " could re-decide"
        )
    print(
        "  veclen=" + String(expect_veclen) + " arm at d=" + String(d)
        + ": " + String(n) + " labels correct (grid.y " + String(cfg[1])
        + " of " + String(y_chunks) + " tiles, " + String(tail_checked)
        + " rows past the resident grid), fused == unfused"
        + (
            "" if arm_diff_excused == 0
            else " except " + String(arm_diff_excused) + " near-tie rows"
            " the lossy vendor product re-decided inside its budget"
        )
    )


def check_fused_policy_dispatch() raises:
    """Pin the policy/veclen selection (`fused_distance_nn-inl.cuh:102-233`)
    and prove the non-default arms are CORRECT where their selection takes
    them.

    The selection values are pinned against hand-transcribed upstream
    behavior; the launcher dispatches on the SAME two functions, so a pin
    here is a pin on the arm the bench takes. A policy change nothing
    exercises proves nothing, so the scalar (k=33) and 2-wide (k=34) arms
    are then run for real through the launcher, at a row count that forces
    the m grid-stride loop, against planted labels and the unfused arm.
    """
    # --- the selection computation, at pinned inputs ----------------------
    # 16-byte arm: 4*k % 16 == 0 and both pointers 16-aligned -> veclen 4.
    if fused_veclen_for(32, 0, 0) != 4:
        raise Error("k=32 aligned must take veclen 4 (-inl.cuh:110)")
    if fused_veclen_for(4, 0, 0) != 4:
        raise Error("k=4 aligned must take veclen 4: 16 bytes per row")
    # Misalignment demotes: a 4-mod-16 pointer fails both vector tests.
    if fused_veclen_for(32, 4, 0) != 1:
        raise Error("a 4-byte-misaligned x must fall to the scalar arm")
    # 8-byte arm: 4*k % 8 == 0 with 8-aligned pointers -> veclen 2.
    if fused_veclen_for(32, 8, 8) != 2:
        raise Error("8-aligned pointers at k=32 must take veclen 2")
    if fused_veclen_for(34, 0, 0) != 2:
        raise Error("k=34 (136 bytes) must take veclen 2 (-inl.cuh:158)")
    # Scalar arm: odd k.
    if fused_veclen_for(33, 0, 0) != 1:
        raise Error("k=33 (132 bytes) must take veclen 1 (-inl.cuh:210)")
    # Skinny selection is `k < 32` (-inl.cuh:105), nothing else.
    if not fused_is_skinny(31) or fused_is_skinny(32):
        raise Error("is_skinny must be exactly k < 32")

    # --- the bench alignment takes the vectorized arm on REAL buffers -----
    var ctx = DeviceContext()
    var probe_x = ctx.enqueue_create_buffer[DType.float32](64 * 32)
    var probe_c = ctx.enqueue_create_buffer[DType.float32](64 * 32)
    ctx.synchronize()
    var bench_vl = fused_veclen_for(
        32, Int(probe_x.unsafe_ptr()), Int(probe_c.unsafe_ptr())
    )
    if bench_vl != 4:
        raise Error(
            "THE BENCH SHAPE (k=32) DOES NOT TAKE THE VECTORIZED ARM on"
            " real device buffers: veclen came back " + String(bench_vl)
            + ". Either the allocator stopped 16-byte-aligning or the"
            " selection regressed."
        )

    # --- the fallback arms, run for real ----------------------------------
    _policy_arm_correct(33, 1)
    _policy_arm_correct(34, 2)

    print(
        "check_fused_policy_dispatch OK: selection pinned (4/2/1 by k and"
        " alignment, skinny at k<32), bench alignment k=32 takes veclen 4"
        " on real buffers, scalar and 2-wide arms correct through the"
        " launcher with the m grid-stride exercised"
    )


# --- scalable k-means|| ------------------------------------------------------

comptime SCAL_N = 4000


def _scal_cost(i: Int) -> Float32:
    """Hashed, scattered, POSITIVE planted candidate costs.

    Not uniform on purpose: a uniform cost vector gives every point the same
    selection probability and a kernel that reads the wrong element still
    draws from the right distribution. Hashed values make the selected SET
    depend on placement."""
    return Float32((i * 2654435761 + 13) % 2039) / Float32(2039.0) * Float32(
        3.0
    ) + Float32(0.01)


def _scal_preflagged(i: Int) -> Bool:
    """Every 97th sample is already a centroid, so the exclusion half of
    `SamplingOp` (`!flag[a.key]`) has real work to refuse."""
    return i % 97 == 0


def _run_scalable_selection(
    ctx: DeviceContext,
    mut min_dist: DeviceBuffer[DType.float32],
    mut is_centroid: DeviceBuffer[DType.int32],
    mut flags: DeviceBuffer[DType.float32],
    mut chunk_totals: DeviceBuffer[DType.float32],
    mut chunk_offsets: DeviceBuffer[DType.float32],
    mut csum: DeviceBuffer[DType.float32],
    n: Int,
    psi: Float32,
    lk: Float32,
    seed_lo: Int32,
    seed_hi: Int32,
) raises -> Int:
    """One k-means|| sampling round exactly as the shipped init launches it:
    flags, the three-stage scan, and the ONE-float count readback. A free
    function because Mojo 1.0 cannot infer a capture convention for a
    `DeviceContext` in a nested `def`."""
    var chunk = PLUS_PLUS_TPB
    var n_chunks = (n + chunk - 1) // chunk
    ctx.enqueue_function[sample_flags_kernel](
        flags.unsafe_ptr(),
        min_dist.unsafe_ptr(),
        is_centroid.unsafe_ptr(),
        Int32(n),
        psi,
        lk,
        seed_lo,
        seed_hi,
        grid_dim=((n + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[chunk_sums_kernel](
        chunk_totals.unsafe_ptr(), flags.unsafe_ptr(), Int32(n), Int32(chunk),
        grid_dim=(n_chunks, 1, 1), block_dim=(PLUS_PLUS_TPB, 1, 1),
    )
    ctx.enqueue_function[scan_chunk_offsets_kernel](
        chunk_offsets.unsafe_ptr(), chunk_totals.unsafe_ptr(),
        Int32(n_chunks),
        grid_dim=(1, 1, 1), block_dim=(PLUS_PLUS_TPB, 1, 1),
    )
    ctx.enqueue_function[write_inclusive_scan_kernel](
        csum.unsafe_ptr(), flags.unsafe_ptr(), chunk_offsets.unsafe_ptr(),
        Int32(n), Int32(chunk),
        grid_dim=(n_chunks, 1, 1), block_dim=(PLUS_PLUS_TPB, 1, 1),
    )
    var tail = csum.create_sub_buffer[DType.float32](n - 1, 1)
    var h_count = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_ptr=h_count.unsafe_ptr(), src_buf=tail)
    ctx.synchronize()
    return Int(h_count.unsafe_ptr().unsafe_load(0))


def check_scalable_sampling_selection() raises:
    """The k-means|| round machinery against an exact host replay, then a
    cost sabotage that must move the selection in the predicted shape.

    `scalable_uniform` and `scalable_keep` are plain `def`s, so the host can
    run THE SAME predicate the device runs and the expected selection is an
    exact set, not a distribution: same count, same indices, same order
    (`DeviceSelect::If` stability), flags updated exactly on selected union
    preflagged. Note the replay shares the predicate CODE with the kernel on
    purpose -- this check pins the selection/compaction machinery around the
    predicate, and the fit-level checks pin the algorithm's output against
    independent oracles.

    THE SABOTAGE: zero the planted cost of every third sample and rerun with
    the SAME seed and the SAME psi. `prob_x = lk * 0 / psi = 0` and the
    comparison is strict, so a zero-cost sample can NEVER be drawn: the
    predicted movement is exactly the baseline selection minus the zeroed
    rows, nothing else -- every other sample's uniform and probability are
    unchanged. A selection that ignores `min_dist` passes the baseline count
    by luck at best and cannot pass both halves.
    """
    var ctx = DeviceContext()
    var n = SCAL_N
    var chunk = PLUS_PLUS_TPB
    var n_chunks = (n + chunk - 1) // chunk

    var min_dist = ctx.enqueue_create_buffer[DType.float32](n)
    var is_centroid = ctx.enqueue_create_buffer[DType.int32](n)
    var flags = ctx.enqueue_create_buffer[DType.float32](n)
    var chunk_totals = ctx.enqueue_create_buffer[DType.float32](n_chunks)
    var chunk_offsets = ctx.enqueue_create_buffer[DType.float32](n_chunks)
    var csum = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()

    var hd = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hp = ctx.enqueue_create_host_buffer[DType.int32](n)
    var psi64 = Float64(0.0)
    for i in range(n):
        hd.unsafe_ptr().unsafe_store(i, _scal_cost(i))
        hp.unsafe_ptr().unsafe_store(
            i, Int32(1) if _scal_preflagged(i) else Int32(0)
        )
        psi64 += Float64(_scal_cost(i))
    ctx.enqueue_copy(dst_buf=min_dist, src_ptr=hd.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=is_centroid, src_ptr=hp.unsafe_ptr())
    ctx.synchronize()

    var psi = Float32(psi64)
    var lk = Float32(16.0)  # oversampling_factor 2.0 at k = 8
    var round_seed = UInt64(0x9E3779B97F4A7C15) ^ UInt64(20260820)
    var seed_lo = round_seed.cast[DType.uint32]().cast[DType.int32]()
    var seed_hi = (round_seed >> 32).cast[DType.uint32]().cast[DType.int32]()

    # --- the exact expected selection, replayed on the host ----------------
    var expected = List[Int]()
    for i in range(n):
        var u = scalable_uniform(seed_lo, seed_hi, i)
        if not _scal_preflagged(i) and scalable_keep(
            _scal_cost(i), psi, lk, u
        ):
            expected.append(i)
    if len(expected) == 0:
        raise Error("fixture too weak: the host replay selects nothing")
    var expected_div3 = 0
    for e in range(len(expected)):
        if expected[e] % 3 == 0:
            expected_div3 += 1
    if expected_div3 == 0:
        raise Error(
            "fixture too weak: no baseline selection at i % 3 == 0, so the"
            " sabotage below could not register"
        )

    var n_sel = _run_scalable_selection(
        ctx, min_dist, is_centroid, flags, chunk_totals, chunk_offsets, csum,
        n, psi, lk, seed_lo, seed_hi,
    )
    if n_sel != len(expected):
        raise Error(
            "device selected " + String(n_sel) + " where the host replay of"
            " the same predicate selects " + String(len(expected))
        )
    var sel_index = ctx.enqueue_create_buffer[DType.uint32](n_sel)
    ctx.synchronize()
    ctx.enqueue_function[select_scatter_kernel](
        sel_index.unsafe_ptr(), is_centroid.unsafe_ptr(), flags.unsafe_ptr(),
        csum.unsafe_ptr(), Int32(n),
        grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    var h_sel = ctx.enqueue_create_host_buffer[DType.uint32](n_sel)
    var h_flag = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_ptr=h_sel.unsafe_ptr(), src_buf=sel_index)
    ctx.enqueue_copy(dst_ptr=h_flag.unsafe_ptr(), src_buf=is_centroid)
    ctx.synchronize()

    for e in range(len(expected)):
        if Int(h_sel.unsafe_ptr().unsafe_load(e)) != expected[e]:
            raise Error(
                "selection slot " + String(e) + " holds "
                + String(Int(h_sel.unsafe_ptr().unsafe_load(e)))
                + ", host replay expects " + String(expected[e])
                + " -- order or placement is wrong"
            )
    # Flags must now be exactly preflagged union selected.
    var cursor = 0
    for i in range(n):
        var want = Int32(0)
        if _scal_preflagged(i):
            want = Int32(1)
        elif cursor < len(expected) and expected[cursor] == i:
            want = Int32(1)
            cursor += 1
        if h_flag.unsafe_ptr().unsafe_load(i) != want:
            raise Error(
                "isSampleCentroid wrong at " + String(i)
                + ": the scatter's flag update (their thrust::for_each_n,"
                " kmeans_common.cuh:270-276) is not doing what it says"
            )

    # --- SABOTAGE: zero every third cost; those rows must vanish -----------
    for i in range(n):
        if i % 3 == 0:
            hd.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=min_dist, src_ptr=hd.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=is_centroid, src_ptr=hp.unsafe_ptr())  # reset
    ctx.synchronize()

    var sab_expected = List[Int]()
    for e in range(len(expected)):
        if expected[e] % 3 != 0:
            sab_expected.append(expected[e])

    var sab_sel = _run_scalable_selection(
        ctx, min_dist, is_centroid, flags, chunk_totals, chunk_offsets, csum,
        n, psi, lk, seed_lo, seed_hi,
    )
    if sab_sel == n_sel:
        raise Error(
            "SABOTAGE FAILED TO REGISTER: zeroing " + String(n // 3)
            + " costs left the selected count at " + String(n_sel)
            + ", so the sampling is not reading the cost vector"
        )
    if sab_sel != len(sab_expected):
        raise Error(
            "sabotaged selection count " + String(sab_sel)
            + " is not the predicted baseline-minus-zeroed "
            + String(len(sab_expected))
        )
    var sab_index = ctx.enqueue_create_buffer[DType.uint32](sab_sel)
    ctx.synchronize()
    ctx.enqueue_function[select_scatter_kernel](
        sab_index.unsafe_ptr(), is_centroid.unsafe_ptr(), flags.unsafe_ptr(),
        csum.unsafe_ptr(), Int32(n),
        grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    var h_sab = ctx.enqueue_create_host_buffer[DType.uint32](sab_sel)
    ctx.enqueue_copy(dst_ptr=h_sab.unsafe_ptr(), src_buf=sab_index)
    ctx.synchronize()
    for e in range(sab_sel):
        var got = Int(h_sab.unsafe_ptr().unsafe_load(e))
        if got % 3 == 0:
            raise Error(
                "a zero-cost sample " + String(got) + " was selected: the"
                " strict `prob_x > rnd` comparison is not being applied to"
                " the cost"
            )
        if got != sab_expected[e]:
            raise Error(
                "sabotaged selection diverged from the predicted set at slot "
                + String(e)
            )

    print(
        "check_scalable_sampling_selection OK: " + String(n_sel) + "/"
        + String(n) + " drawn exactly as the host replay predicts (order,"
        " flags, count); zeroing 1/3 of the costs moved the selection to"
        " exactly baseline-minus-zeroed (" + String(sab_sel) + "), "
        + String(expected_div3) + " rows vanished as predicted"
    )


def check_scalable_kmeans_plus_plus_init() raises:
    """The DEFAULT configuration -- the one that raised until k-means||
    landed -- run end to end, twice.

    `oversampling_factor` is left at cuVS's default 2.0, which selects
    `initScalableKMeansPlusPlus` (`detail/kmeans.cuh:910-915`); both arms of
    that selection predicate are pinned here first. The fit must recover the
    planted centers as a permutation, its inertia must match the analytic
    expectation, and a second identical run must reproduce the inertia
    BITWISE (every reduction in the path is either fixed-shape or exact
    integer arithmetic, so determinism is a design claim, not luck).

    THE TOLERANCE IS FROM A MEASURED ORACLE SPREAD, NOT A GUESS: sklearn
    1.9.0 `KMeans(n_clusters=4, init='k-means++', n_init=1, max_iter=50)` on
    this exact fixture returns inertia 171.193649 on ALL of seeds 0..9 --
    seed-to-seed spread ZERO -- so the whole 2% budget is our float32 /
    fixed-point arithmetic, which the classic-init path measures at 0.0029
    relative on the same fixture.
    """
    var ctx = DeviceContext()
    var cd = CHECK_CLUSTERS * CHECK_FEATURES

    var x = ctx.enqueue_create_buffer[DType.float32](
        CHECK_ROWS * CHECK_FEATURES
    )
    var weights = ctx.enqueue_create_buffer[DType.float32](CHECK_ROWS)
    var centroids = ctx.enqueue_create_buffer[DType.float32](cd)
    var labels = ctx.enqueue_create_buffer[DType.uint32](CHECK_ROWS)
    ctx.synchronize()

    var magnitude = _fill_fixture(ctx, x, weights)
    var sum_scale = Float32(choose_scale(magnitude))
    var weight_scale = Float32(choose_scale(Float64(CHECK_ROWS)))

    var params = KMeansParams.default()
    params.n_clusters = CHECK_CLUSTERS
    params.metric = METRIC_L2_EXPANDED
    params.max_iter = 50
    params.seed = 20260820
    # oversampling_factor stays at the DEFAULT 2.0. Pin both arms of their
    # selection predicate (`:910-915`) before running one of them.
    if not params.uses_scalable_plus_plus():
        raise Error(
            "oversampling_factor=2.0 must select scalable k-means||"
        )
    var classic = params
    classic.oversampling_factor = 0.0
    if classic.uses_scalable_plus_plus():
        raise Error(
            "oversampling_factor=0 must select classic k-means++, exactly"
            " as their `if (iter_params.oversampling_factor == 0)`"
        )

    var result = fit(
        ctx, x, weights, centroids, labels, params,
        CHECK_ROWS, CHECK_FEATURES, sum_scale, weight_scale,
    )

    var out_c = ctx.enqueue_create_host_buffer[DType.float32](cd)
    var out_l = ctx.enqueue_create_host_buffer[DType.uint32](CHECK_ROWS)
    ctx.enqueue_copy(dst_ptr=out_c.unsafe_ptr(), src_buf=centroids)
    ctx.enqueue_copy(dst_ptr=out_l.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    # --- permutation recovery, matched exactly once each ------------------
    var used = List[Int]()
    for _c in range(CHECK_CLUSTERS):
        used.append(0)
    var mapping = List[Int]()
    for slot in range(CHECK_CLUSTERS):
        var best = -1
        var best_err = Float64(1.0e30)
        for planted in range(CHECK_CLUSTERS):
            var err = Float64(0.0)
            for f in range(CHECK_FEATURES):
                var dd = Float64(
                    out_c.unsafe_ptr().unsafe_load(slot * CHECK_FEATURES + f)
                    - _planted_center(planted, f)
                )
                err += dd * dd
            if err < best_err:
                best_err = err
                best = planted
        if best_err > 1.0:
            raise Error(
                "k-means|| init: centroid " + String(slot)
                + " is not near any planted center, squared error "
                + String(best_err)
            )
        if used[best] != 0:
            raise Error(
                "k-means|| init: two centroids matched planted center "
                + String(best) + ", the fit collapsed"
            )
        used[best] = 1
        mapping.append(best)

    var wrong = 0
    for i in range(CHECK_ROWS):
        var slot = Int(out_l.unsafe_ptr().unsafe_load(i))
        if slot < 0 or slot >= CHECK_CLUSTERS:
            raise Error("label out of range at row " + String(i))
        if mapping[slot] != i % CHECK_CLUSTERS:
            wrong += 1
    if wrong != 0:
        raise Error(
            String(wrong) + " of " + String(CHECK_ROWS)
            + " rows landed in the wrong cluster through k-means||"
        )

    # --- inertia against the analytic expectation -------------------------
    var expected = Float64(0.0)
    for i in range(CHECK_ROWS):
        for f in range(CHECK_FEATURES):
            var c = i % CHECK_CLUSTERS
            var mean = Float64(0.0)
            var count = 0
            for j in range(CHECK_ROWS):
                if j % CHECK_CLUSTERS == c:
                    mean += Float64(_jitter(j, f))
                    count += 1
            mean /= Float64(count)
            var dd = Float64(_jitter(i, f)) - mean
            expected += dd * dd
    var rel = abs(result.inertia - expected) / expected
    if rel > 0.02:
        raise Error(
            "k-means|| inertia " + String(result.inertia)
            + " against expected " + String(expected) + ", relative "
            + String(rel) + " (sklearn oracle spread on this fixture is 0,"
            " so this is arithmetic error, not draw variance)"
        )

    # --- determinism: same seed twice, bitwise-equal inertia --------------
    var again = fit(
        ctx, x, weights, centroids, labels, params,
        CHECK_ROWS, CHECK_FEATURES, sum_scale, weight_scale,
    )
    if again.inertia != result.inertia:
        raise Error(
            "two identical k-means|| fits disagree: " + String(result.inertia)
            + " then " + String(again.inertia)
            + " -- something in the init is not a function of the seed"
        )

    print(
        "check_scalable_kmeans_plus_plus_init OK: DEFAULT config"
        " (oversampling_factor=2.0) ran end to end, 4/4 centroids as a"
        " permutation, 0/" + String(CHECK_ROWS) + " rows misassigned,"
        " inertia " + String(result.inertia) + " vs expected "
        + String(expected) + " (rel " + String(rel)
        + "), run-twice bitwise equal, " + String(result.n_iter)
        + " iterations"
    )


comptime SUPP_CLUSTERS = 8


def check_scalable_supplement_branch() raises:
    """Prove `oversampling_factor`'s VALUE reaches the sampling, and reach
    the fewer-than-k supplement arm (`detail/kmeans.cuh:755-777`), which no
    healthy run takes.

    At `oversampling_factor = 1e-9` the per-point probability is ~1e-12
    against 24-bit uniforms with a STRICT comparison, so no round can draw a
    candidate: the init reaches the selection tail with exactly the one
    step-1 centroid and MUST take the random-supplement arm.

    The fixture plants EIGHT clusters, not the usual four, and that is the
    load-bearing choice: the first version of this check used the 4-cluster
    fixture and the supplement arm's four starting rows happened to cover
    all four basins, so both arms converged to the SAME optimum and their
    inertias tied BITWISE -- legitimately. At k = 8, seven uniform rows
    cover the seven remaining planted clusters with probability
    7!/7^7 ~= 0.6%, so the predicted shape is a starved fit that lands in a
    WORSE basin: inertia strictly greater than the default arm's (a missed
    planted cluster costs on the order of the 100-per-cluster separation
    squared, orders above the jitter inertia). Recovery is deliberately NOT
    asserted -- duplicate coverage with the old-centroid empty-cluster rule
    is cuVS's behavior, not a defect. Also held: the fit completes (this
    exact config raised before the port), labels stay in range, and a
    second run reproduces the inertia bitwise.
    """
    var ctx = DeviceContext()
    var k = SUPP_CLUSTERS
    var d = CHECK_FEATURES
    var cd = k * d

    var x = ctx.enqueue_create_buffer[DType.float32](CHECK_ROWS * d)
    var weights = ctx.enqueue_create_buffer[DType.float32](CHECK_ROWS)
    var centroids = ctx.enqueue_create_buffer[DType.float32](cd)
    var labels = ctx.enqueue_create_buffer[DType.uint32](CHECK_ROWS)
    ctx.synchronize()

    # Eight planted clusters, round robin, hashed jitter -- the same
    # construction as `_fill_fixture` at a different k.
    var hx = ctx.enqueue_create_host_buffer[DType.float32](CHECK_ROWS * d)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](CHECK_ROWS)
    var magnitude = Float64(0.0)
    for f in range(d):
        var column = Float64(0.0)
        for i in range(CHECK_ROWS):
            var c = i % k
            var v = Float32(100 * (c + 1) + f) + _jitter(i, f)
            hx.unsafe_ptr().unsafe_store(i * d + f, v)
            column += Float64(abs(v))
        if column > magnitude:
            magnitude = column
    for i in range(CHECK_ROWS):
        hw.unsafe_ptr().unsafe_store(i, Float32(1.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()
    var sum_scale = Float32(choose_scale(magnitude))
    var weight_scale = Float32(choose_scale(Float64(CHECK_ROWS)))

    var params = KMeansParams.default()
    params.n_clusters = k
    params.metric = METRIC_L2_EXPANDED
    params.max_iter = 50
    params.seed = 20260820

    var base = fit(
        ctx, x, weights, centroids, labels, params,
        CHECK_ROWS, d, sum_scale, weight_scale,
    )

    var starved = params
    starved.oversampling_factor = 1.0e-9
    if not starved.uses_scalable_plus_plus():
        raise Error("a tiny positive oversampling_factor must stay scalable")

    var r1 = fit(
        ctx, x, weights, centroids, labels, starved,
        CHECK_ROWS, d, sum_scale, weight_scale,
    )
    var out_l = ctx.enqueue_create_host_buffer[DType.uint32](CHECK_ROWS)
    ctx.enqueue_copy(dst_ptr=out_l.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()
    for i in range(CHECK_ROWS):
        if Int(out_l.unsafe_ptr().unsafe_load(i)) >= k:
            raise Error(
                "label out of range at row " + String(i)
                + " through the supplement arm"
            )
    if not (r1.inertia > 0.0 and r1.inertia < 1.0e30):
        raise Error(
            "supplement-arm inertia is not a finite positive number: "
            + String(r1.inertia)
        )

    var r2 = fit(
        ctx, x, weights, centroids, labels, starved,
        CHECK_ROWS, d, sum_scale, weight_scale,
    )
    if r2.inertia != r1.inertia:
        raise Error(
            "two identical supplement-arm fits disagree: "
            + String(r1.inertia) + " then " + String(r2.inertia)
        )
    if not (r1.inertia > base.inertia):
        raise Error(
            "oversampling_factor 1e-9 gave inertia " + String(r1.inertia)
            + " against the default arm's " + String(base.inertia)
            + ": a starved init that does no worse than the full one means"
            " the factor's value is not reaching the sampling probability"
            " (or seven random rows covered seven basins, p ~= 0.6% --"
            " reseed the check if the fixture ever changes)"
        )

    print(
        "check_scalable_supplement_branch OK: oversampling_factor=1e-9"
        " starves every round, the < k supplement arm ran and landed in a"
        " worse basin as predicted (inertia " + String(r1.inertia)
        + " vs default-arm " + String(base.inertia)
        + "), run-twice bitwise equal, labels in range"
    )
