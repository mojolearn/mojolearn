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

from max.gpu.host import DeviceBuffer, DeviceContext

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from cluster.ported.distance.fused_distance_nn.simt_kernel import (
    FUSED_NORMAL_KBLK,
    FUSED_NORMAL_TC,
    FUSED_NORMAL_TR,
    fused_distance_nn_kernel,
    fused_is_skinny,
    fused_smem_bytes,
    fused_veclen_for,
)
from neighbors.ported.distance.detail.pairwise_distance_base import (
    launch_config_generator,
)
from cluster.ported.cluster.detail.min_cluster_distance_compute import (
    compute_centroid_norms,
    min_cluster_and_distance_compute,
    min_cluster_and_distance_compute_unfused,
)
from cluster.mojo_only.reduce_by_key import (
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
from cluster.ported.cluster.kmeans import fit
from cluster.mojo_only.plus_plus import (
    PLUS_PLUS_TPB,
    chunk_sums_kernel,
    scan_chunk_offsets_kernel,
    write_inclusive_scan_kernel,
)
from cluster.ported.cluster.kmeans_params import (
    INIT_KMEANS_PLUS_PLUS,
    INIT_ARRAY,
    KMeansParams,
    METRIC_L2_EXPANDED,
)
from core.row_norms import NORM_TPB, row_norm_kernel
from mojo_only.fixed_point import choose_scale


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
    distances within float tolerance -- NOT bitwise, because the two arms
    sum the dot product in different orders, exactly as upstream's two arms
    do.
    """
    var ctx = DeviceContext()
    var n = ARM_ROWS
    var d = ARM_FEATURES
    var k = ARM_CLUSTERS

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

    # --- the sabotage half: the unfused arm MUST destroy the sentinel ------
    var labels_u = ctx.enqueue_create_buffer[DType.uint32](n)
    var min_dist_u = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
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

    var label_diff = 0
    var worst_rel = Float64(0.0)
    for i in range(n):
        if hlu.unsafe_ptr().unsafe_load(i) != hl.unsafe_ptr().unsafe_load(i):
            label_diff += 1
        var a = Float64(hm.unsafe_ptr().unsafe_load(i))
        var b = Float64(hmu.unsafe_ptr().unsafe_load(i))
        var denom = abs(a) if abs(a) > 1.0e-6 else 1.0e-6
        var rel = abs(a - b) / denom
        if rel > worst_rel:
            worst_rel = rel
    if label_diff != 0:
        raise Error(
            "the two arms disagree on " + String(label_diff)
            + " labels; they reduce with the same total order and must not"
        )
    if worst_rel > 1.0e-4:
        raise Error(
            "fused and unfused min_dist diverge, worst relative "
            + String(worst_rel)
        )

    print(
        "check_assignment_arm_dispatch OK: fused arm proved (0/"
        + String(n * k)
        + " tile cells written; unfused sabotage overwrote "
        + String(destroyed)
        + "); arms agree on all labels, min_dist worst rel "
        + String(worst_rel)
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


def _privatized_sums_dropped_flush_kernel(
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
    kernel`, identical except that BLOCK 0 SKIPS ITS FLUSH. It exists only
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

    var gid = Int(block_idx.x) * Int(block_dim.x) + tid
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while gid < total:
        var row = gid // n_features
        var f = gid - row * n_features
        var label = Int(labels.unsafe_load(row))
        var w = weights.unsafe_load(row)
        var q = Int32(x.unsafe_load(gid) * w * scale_in)
        _ = Atomic.fetch_add(priv.unsafe_offset(label * n_features + f), q)
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
    ctx.enqueue_function[_privatized_sums_dropped_flush_kernel](
        sums.unsafe_ptr(), x.unsafe_ptr(), labels.unsafe_ptr(),
        weights.unsafe_ptr(), Int32(n), Int32(d), Int32(k), ACC_SCALE,
        grid_dim=(
            accumulate_grid_blocks(n * d, PRIVATE_ACC_CELLS * 4), 1, 1
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
        + " cells"
    )


# --- fused policy selection --------------------------------------------------

comptime POL_ROWS = 8256  # 129 row tiles of Mblk=64: more than the M4's
#                           minGridSize of 120 blocks at 256 threads, so the
#                           launcher's grid.y is capped BELOW the tile count
#                           and the kernel's m grid-stride loop must cover
#                           the rest. The check verifies that inequality at
#                           runtime rather than trusting this comment.
comptime POL_CLUSTERS = 40


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
    var n = POL_ROWS
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
    # grid.y (their launchConfigGenerator, M4 inputs) must be SMALLER than
    # the number of row tiles.
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

    var mislabeled = 0
    var tail_checked = 0
    var arm_diff = 0
    var stride_floor = cfg[1] * 4 * FUSED_NORMAL_TR
    for i in range(n):
        if Int(hl.unsafe_ptr().unsafe_load(i)) != i % k:
            mislabeled += 1
        if i >= stride_floor:
            tail_checked += 1
        if hl.unsafe_ptr().unsafe_load(i) != hlu.unsafe_ptr().unsafe_load(i):
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
            + " rows at d=" + String(d)
        )
    print(
        "  veclen=" + String(expect_veclen) + " arm at d=" + String(d)
        + ": " + String(n) + " labels correct (grid.y " + String(cfg[1])
        + " of " + String(y_chunks) + " tiles, " + String(tail_checked)
        + " rows past the resident grid), fused == unfused"
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
