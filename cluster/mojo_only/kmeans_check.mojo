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

from cluster.ported.cluster.detail.min_cluster_distance_compute import (
    compute_centroid_norms,
    min_cluster_and_distance_compute,
)
from cluster.ported.cluster.kmeans import fit
from cluster.ported.cluster.kmeans_params import (
    INIT_ARRAY,
    KMeansParams,
    METRIC_L2_EXPANDED,
)
from cluster.mojo_only.row_norms import NORM_TPB, row_norm_kernel
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
       Their clamp at `unfused_distance_nn.cuh:81` is not order-preserving:
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
    # `unfused_distance_nn.cuh:81` then flattened all four centroids to
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
