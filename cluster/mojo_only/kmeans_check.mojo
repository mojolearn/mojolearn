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

from cluster.ported.distance.fused_distance_nn.simt_kernel import (
    fused_distance_nn_kernel,
)
from core.gemm import GEMM_MBLK, GEMM_THREADS
from cluster.ported.cluster.detail.min_cluster_distance_compute import (
    compute_centroid_norms,
    min_cluster_and_distance_compute,
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
    butterfly changed nothing, because at `CHECK_CLUSTERS = 4` every valid
    column lands in ONE lane group: `GEMM_ACC_COLS_PER_TH = 4` columns per
    thread means `tc = col // 4`, so columns 0..3 are all `tc == 0` and the
    other fifteen lanes hold nothing but `FUSED_MAX`. The merge was
    degenerate and a broken merge could not show.

    That is the same failure the repository already has a name for: a fixture
    whose structure hides the thing under test. Here it hid a whole reduction
    stage rather than a permutation.

    So: 40 clusters, spreading real values over ten lane groups, checked
    against a host argmin. Now a broken cross-lane merge has to show.
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
    ctx.enqueue_function[fused_distance_nn_kernel](
        okey.unsafe_ptr(), oval.unsafe_ptr(), x.unsafe_ptr(), c.unsafe_ptr(),
        xn.unsafe_ptr(), cn.unsafe_ptr(),
        Int32(n), Int32(k), Int32(d), Int32(0),
        grid_dim=(1, (n + GEMM_MBLK - 1) // GEMM_MBLK, 1),
        block_dim=(GEMM_THREADS, 1, 1),
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
        spread[best // 4] = 1
    var lanes = 0
    for t in range(16):
        lanes += spread[t]

    if lanes < 4:
        raise Error(
            "the fixture only used " + String(lanes) + " lane groups; it"
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
        " spread over " + String(lanes) + " lane groups"
    )
