# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The callable surface over the k-means fit.

**Why this file exists.** `cluster/impl/kmeans.mojo` already has
`fit`, `predict` and `fit_predict`, faithfully mirroring
`cuvs/src/cluster/kmeans.cuh`. None of them is callable by anyone outside this
repository, for two reasons that are both this file's job to fix:

  1. They take `DeviceBuffer`s. A caller holding a numpy array has no way to
     produce one, so every argument has to be allocated, uploaded and read
     back by somebody, and until now that somebody was always a check.
  2. **They demand `sum_scale` and `weight_scale`**, the fixed-point
     accumulator multipliers. Those are not tuning knobs a user could guess.
     Pass one too small and the centroid sums lose resolution; too large and
     the Int32 accumulator overflows and the answer is silently wrong. Every
     existing caller is a check that computed them from a fixture it
     generated itself.

Nothing here is an implementation; that is `cluster/impl/`, whose reference
is cuVS. This file is host-side policy cuVS has no counterpart
for, in the same category as `checks/`. It follows
`neighbors/estimator.mojo`, which is the first file of this kind, including
its convention that data crosses as raw pointers plus lengths so a CPython
extension can pass buffer addresses straight through.

THE POLICY CHOICES
------------------

1. **THE SCALE IS COMPUTED FROM THE CALLER'S DATA, ON THE DEVICE.**
   `choose_scale` needs `sum over rows of abs(value)` for the plane being
   accumulated, and the bound that matters is the worst column. `plan_sum_scale`
   folds every column on the device in one fixed order (blocked chunks, then a
   halving tree; `cluster/impl/sum_scale_plan.mojo`), the host column restates
   the same fold, and `KMeansFitResult` reports the scale that was chosen.

   The alternative was a fixed scale, which is what an implementation does
   when it does not want to admit this cost. A fixed scale is a silent wrong
   answer on data whose magnitude it did not anticipate.

2. **`row_count` IS PASSED TO `choose_scale`, AND THE CHECKS DO NOT DO THAT.**
   `checks/fixed_point.mojo:55-70` documents that stating the row count
   sharpens the overflow bound from a blanket three-bit headroom to an exact
   allowance, buying a scale 4x finer, and records that the blanket scale's
   dither noise cost 1.6% train mse on the boosting side at 254 borders. The
   k-means checks call `choose_scale(magnitude)` with no row count and
   therefore take the blanket bound. This file passes it. That is a
   DELIBERATE DIFFERENCE from what the checks exercise, it can only make the
   scale finer and never weaker, and `check_kmeans_fit_scale_policy` is what
   holds it.

3. **`fit_predict`, NOT `fit`, AND THE EXTRA PASS IS THE POINT.**
   `fit` leaves `labels` holding the assignment from the LAST iteration,
   which belongs to the centroids from BEFORE that iteration's update.
   Returning those is an off-by-one-iteration bug that is invisible in every
   aggregate metric. Neither cuVS nor scikit-learn does it; both run one more
   full assignment against the final centroids, and so does this. The cost is
   one extra assignment pass, and it is not optional here because a
   caller-facing `fit` that returns stale labels is wrong.

4. **DEFAULTS ARE cuVS'S, NOT scikit-learn's**, and the two differ where it
   would change results. `KMeansParams.default()` mirrors `kmeans.hpp:28-121`:
   `max_iter=300`, `tol=1e-4`, `n_init=1`, `oversampling_factor=2.0`,
   `init=INIT_KMEANS_PLUS_PLUS`, `metric=METRIC_L2_EXPANDED`,
   `inertia_check=False`. scikit-learn's `n_init` default is 10, not 1, so a
   caller comparing against scikit-learn out of the box is comparing one
   restart against ten. Stated here rather than quietly matched, because
   changing it would make our number look better and would not be cuVS's
   behaviour.

   `inertia_check=False` is the one to read twice: with it off the Lloyd loop
   never computes the IN-LOOP cluster cost and the only convergence
   criterion is the centroid shift. The returned `inertia` is a different
   quantity and is ALWAYS formed: cuVS's post-loop assignment against the
   final centroids (`detail/kmeans.cuh:516-535`, `fit`'s `best_inertia`).
   This paragraph used to say "a returned `inertia` of 0.0 means it was
   never computed"; that was false (corrected 2026-08-23).

5. **WEIGHTS ARE EXPLICIT, NOT OPTIONAL-BY-NULL.** `n_weights` is a required
   argument: 0 means unit weights and `weights_ptr` is never read, anything
   else must equal `n_samples`. Mojo 1.0 has no default-constructible
   `MutPointer` and this repository has no null-pointer idiom, so a
   defaulted-to-null parameter was not available. Requiring the count is
   better than inventing one: a caller cannot accidentally get unit weights
   by passing a pointer that happened to be wrong, and the mismatch case
   raises instead of reading past the end.

WHAT IS NOT HERE YET, NAMED SO IT IS NOT MISTAKEN FOR DONE
----------------------------------------------------------

- `predict` against new data with an already-fitted model. `predict` exists in
  the implemented layer and wants a caller-facing wrapper of its own; it is a
  different call and is not wired here.
- `n_init > 1` and `INIT_ARRAY` are exercised at this boundary since
  2026-08-23 by `tools/e2u_matrix_fit.py` (`kmeans_k8_ninit3`,
  `kmeans_k8_array`: both move the answer against the baseline, and the
  restart card tags `restart01.*` appear). No Mojo check covers them.
- Metrics other than `METRIC_L2_EXPANDED`. No Mojo check covers them. The
  Python surface routes `metric` since 2026-09-14 (workstream D):
  `METRIC_L2_SQRT_EXPANDED` is the `kmeans-sqrt` identity lane and
  `python/mojolearn/tests/test_kmeans_metric_surface.py` (labels against the
  argmin to the returned centers, DEVIATION 2716). Those two are now the
  WHOLE set: the cosine metric was deleted on 2026-09-18
  (lane/kmeans-cosine-capability), and an unsupported metric is refused by
  name. This sentence used to say the surface did not expose `metric` at
  all.
- The CPython extension EXISTS (`bindings/_mojolearn.mojo::kmeans_fit_binding`,
  `python/mojolearn/cluster.py`); this sentence used to say it did not.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from cluster.impl.detail.kmeans_common import (
    centroid_norms_take_sqrt,
    metric_is_sqrt,
)
from cluster.impl.detail.kmeans_transform import (
    TRANSFORM_TPB,
    kmeans_transform_kernel,
)
from cluster.impl.detail.min_cluster_distance_compute import (
    compute_centroid_norms,
)
from cluster.impl.kmeans import fit_predict, predict
from core.device_zero import enqueue_fill
from core.row_norms import NORM_TPB, row_norm_kernel
from cluster.impl.kmeans_params import (
    INIT_ARRAY,
    INIT_KMEANS_PLUS_PLUS,
    KMeansParams,
    METRIC_L2_EXPANDED,
    weighted_sum_scale_cap,
)
from cluster.impl.sum_scale_plan import (
    SUM_SCALE_CHUNK,
    SUM_SCALE_TPB,
    sum_scale_from_columns,
    sum_scale_inf,
    sum_scale_is_finite,
    sum_scale_n_chunks,
    sum_scale_term,
)
from checks.fixed_point import choose_scale
from core.pinned_reduce import halving_block_sum


@fieldwise_init
struct KMeansFitResult(Copyable, ImplicitlyCopyable, Movable):
    """What the fit did, including the choices it made for the caller.

    `sum_scale` and `weight_scale` are reported rather than kept private
    because they are the two numbers a wrong answer here would come from, and
    a caller reproducing a result needs them.
    """

    var inertia: Float64
    """The weighted cost against the FINAL centroids, from cuVS's post-loop
    assignment (`detail/kmeans.cuh:516-535`); always formed. The in-loop
    cost that `inertia_check=False` turns off is a different number and
    never leaves the fit."""

    var n_iter: Int
    var sum_scale: Float64
    var weight_scale: Float64


def abs_chunk_sums_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_features_in: Int32,
):
    """Stage 1 of the scale fold (`cluster/impl/sum_scale_plan.mojo`):
    thread `(chunk b, feature f)` folds `sum_scale_term(x[r, f])` over the
    chunk's rows in row order; a non-finite term makes the sum +inf."""
    var n_rows = Int(n_rows_in)
    var n_features = Int(n_features_in)
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var b = gid // n_features
    var f = gid - b * n_features
    var r0 = b * SUM_SCALE_CHUNK
    if r0 >= n_rows:
        return
    var r1 = r0 + SUM_SCALE_CHUNK
    if r1 > n_rows:
        r1 = n_rows
    var acc = Float32(0.0)
    var bad = False
    for row in range(r0, r1):
        var a = sum_scale_term(x.unsafe_load(row * n_features + f))
        if not sum_scale_is_finite(a):
            bad = True
        acc = acc + a
    if bad:
        acc = sum_scale_inf()
    dst.unsafe_store(gid, acc)


def abs_fold_chunks_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    chunk_sums: MutPointer[Float32, MutAnyOrigin],
    n_chunks_in: Int32,
    n_features_in: Int32,
):
    """Stage 2: block `f` folds feature `f`'s chunk sums, thread `t` the
    chunks `t, t + TPB, ...` in order, then the halving tree. A non-finite
    chunk sum makes the column +inf."""
    var n_chunks = Int(n_chunks_in)
    var n_features = Int(n_features_in)
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var acc = Float32(0.0)
    var bad = Float32(0.0)
    var b = t
    while b < n_chunks:
        var v = chunk_sums.unsafe_load(b * n_features + f)
        if sum_scale_is_finite(v):
            acc = acc + v
        else:
            bad = Float32(1.0)
        b += SUM_SCALE_TPB
    var total = halving_block_sum[SUM_SCALE_TPB](acc)
    var n_bad = halving_block_sum[SUM_SCALE_TPB](bad)
    if t == 0:
        if n_bad > Float32(0.0) or not sum_scale_is_finite(total):
            total = sum_scale_inf()
        dst.unsafe_store(f, total)


def plan_sum_scale(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
) raises -> Float64:
    """The fixed-point multiplier for the design `x` already on the device,
    from the worst column (`cluster/impl/sum_scale_plan.mojo`: the fold, its
    order and the bound). Two grid-wide launches and one `n_features` read;
    the host column restates it in `cluster/host/kmeans_oracle.mojo::
    host_plan_sum_scale`, add for add."""
    var n_chunks = sum_scale_n_chunks(n_samples)
    var chunk_sums = ctx.enqueue_create_buffer[DType.float32](
        n_chunks * n_features
    )
    var col = ctx.enqueue_create_buffer[DType.float32](n_features)
    var threads = n_chunks * n_features
    ctx.enqueue_function[abs_chunk_sums_kernel](
        chunk_sums.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_samples),
        Int32(n_features),
        grid_dim=((threads + SUM_SCALE_TPB - 1) // SUM_SCALE_TPB, 1, 1),
        block_dim=(SUM_SCALE_TPB, 1, 1),
    )
    ctx.enqueue_function[abs_fold_chunks_kernel](
        col.unsafe_ptr(),
        chunk_sums.unsafe_ptr(),
        Int32(n_chunks),
        Int32(n_features),
        grid_dim=(n_features, 1, 1),
        block_dim=(SUM_SCALE_TPB, 1, 1),
    )
    var cols = List[Float32](length=n_features, fill=Float32(0.0))
    ctx.enqueue_copy(dst_ptr=cols.unsafe_ptr(), src_buf=col)
    ctx.synchronize()
    _ = chunk_sums^
    _ = col^
    return sum_scale_from_columns(cols, n_samples)


def kmeans_fit(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    out_centroids_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_labels_ptr: MutPointer[UInt32, MutUntrackedOrigin],
    weights_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_weights: Int,
    max_iter: Int = 300,
    tol: Float64 = 1e-4,
    seed: UInt64 = 0,
    n_init: Int = 1,
    init: Int = INIT_KMEANS_PLUS_PLUS,
    metric: Int = METRIC_L2_EXPANDED,
    requested_sum_scale: Float64 = 0.0,
    oversampling_factor: Float64 = 2.0,
) raises -> KMeansFitResult:
    """Fit k-means on host-resident row-major data. See THE POLICY CHOICES.

    `x_ptr` is `n_samples x n_features`, row-major, float32.
    `out_centroids_ptr` is `n_clusters x n_features` and is WRITTEN; when
    `init == INIT_ARRAY` it is also READ first, as the starting centroids.
    `out_labels_ptr` is `n_samples` and is written with the assignment
    against the FINAL centroids.

    `n_weights` is 0 for unit weights, in which case `weights_ptr` is never
    read and any pointer will do; otherwise it must equal `n_samples`.
    `requested_sum_scale` of 0.0 means compute it from the data on the
    device (`plan_sum_scale`); a positive value is used as given.

    `oversampling_factor` is cuVS's (`kmeans.hpp`, default 2.0) and is an
    ALGORITHM SWITCH, not a knob: `0.0` selects the classic sequential
    k-means++ seeding, anything positive the scalable k-means|| one
    (`KMeansParams.uses_scalable_plus_plus`, `detail/kmeans.cuh:910-915`).
    Routed from Python since 2026-09-14 (workstream D); the default is
    unchanged, so every recorded cell keeps its bits.
    """
    if n_samples < 1 or n_features < 1 or n_clusters < 1:
        raise Error(
            "kmeans_fit needs n_samples, n_features and n_clusters >= 1: got "
            + String(n_samples)
            + ", "
            + String(n_features)
            + ", "
            + String(n_clusters)
        )
    if n_weights != 0 and n_weights != n_samples:
        raise Error(
            "kmeans_fit needs n_weights == 0 (unit weights) or == n_samples:"
            " got "
            + String(n_weights)
            + " for "
            + String(n_samples)
            + " samples"
        )
    if n_clusters > n_samples:
        raise Error(
            "kmeans_fit cannot form more clusters than samples: "
            + String(n_clusters)
            + " > "
            + String(n_samples)
        )

    var sum_scale = requested_sum_scale
    # THE WEIGHT BOUND IS NOT ALWAYS n_samples. Unit weights sum to exactly
    # that, but caller-supplied weights can sum to anything, and using
    # n_samples for them would understate the bound and overflow the
    # accumulator. So the supplied case is summed, at the cost of one more
    # host pass over n_samples values -- cheap beside the n_samples *
    # n_features pass `plan_sum_scale` already pays.
    #
    # Lane cpu3-core (2026-10-04): supplied weights take their bound on the
    # device, after the upload below: `plan_sum_scale` over the weights as one
    # column (the same fixed-order fold, widened by its error bound, then
    # `choose_scale(bound, n_samples)`), no host pass. Unit weights keep the
    # exact `n_samples`. The host column (`host_kmeans_fit`) takes the same
    # fold (`host_plan_sum_scale(weights, n, 1)`).
    var weight_scale = choose_scale(Float64(n_samples), n_samples)

    var cd = n_clusters * n_features
    var x = ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var centroids = ctx.enqueue_create_buffer[DType.float32](cd)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n_samples)
    # `fit_predict` needs these two for its final assignment pass. They are
    # workspace, never read by the caller.
    #
    # **AND `x_norm` MUST BE COMPUTED HERE, WHICH IS NOT OBVIOUS.**
    # `fit_predict` does NOT compute it. It forwards whatever the caller
    # passed straight into `predict`, whose docstring explains that the
    # caller supplies it so a predict against already-fitted data does not
    # recompute -- but `fit_predict`'s own docstring says only "fit, then a
    # FRESH assignment" and names no precondition. Nothing in this repository
    # called `fit_predict` before this file, so nothing had ever discovered
    # that. Passing it uninitialized merges clusters: measured on the first
    # run of `check_kmeans_fit_recovers_planted`, which reported planted
    # clusters 0 and 1 collapsed into one label.
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n_samples)
    ctx.synchronize()

    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)

    # The scale from the design on the device (`plan_sum_scale`, one
    # fixed-order fold on every vendor and in the host column).
    if sum_scale <= 0.0:
        sum_scale = plan_sum_scale(ctx, x, n_samples, n_features)
    # DEVIATION 5112: weights above one outgrow the unweighted bound
    # (`weighted_sum_scale_cap`, cluster/impl/kmeans_params.mojo)
    if n_weights != 0 and requested_sum_scale <= 0.0:
        var cap = weighted_sum_scale_cap(x_ptr, weights_ptr, n_samples, n_features)
        if cap < sum_scale:
            sum_scale = cap

    # DEVIATION 2672 (2026-09-11, linear-cluster-istella): NO HOST WEIGHT
    # VECTOR. The fit used to allocate an `n_samples` pinned host buffer,
    # store 1.0 (or copy the caller's weights) into it one row at a time on
    # one thread, and upload it. Unit weights are now the device fill of
    # the same value (`core/device_zero.enqueue_fill`, an exact 1.0 in every
    # cell), and supplied weights upload from the caller's memory directly,
    # as `x` does. The bytes on the device are the bytes the loop wrote, so
    # nothing downstream can move.
    if n_weights != 0:
        ctx.enqueue_copy(dst_buf=weights, src_ptr=weights_ptr)
        weight_scale = plan_sum_scale(ctx, weights, n_samples, 1)
    else:
        enqueue_fill[DType.float32](ctx, weights, Float32(1.0))

    # INIT_ARRAY is the one init that READS this buffer. The others overwrite
    # it, so uploading unconditionally would be wasted traffic on the common
    # path rather than merely harmless.
    if init == INIT_ARRAY:
        ctx.enqueue_copy(dst_buf=centroids, src_ptr=out_centroids_ptr)
    # One block per row, matching every other launch of this kernel.
    #
    # DEVIATION 2716 (2026-09-14, kmeans-sqrt): THE NORMS ARE SQUARED FOR
    # BOTH L2 METRICS. This flag used to follow `metric_is_sqrt`, so under
    # L2SqrtExpanded the final assignment in `fit_predict` computed
    # `||x|| + ||c||^2 - 2 x.c`: the row constant was wrong, the value went
    # negative wherever `||x||^2` exceeded `||x||`, the clamp made it 0, and
    # the tie went to the lowest key. Measured on the M4 at 1eea14f80:
    # 9,675 of 20,000 `labels_` on the `wide` fixture and 4 on `base` were
    # not the argmin to the returned centers (0 under L2Expanded), on every
    # column alike, so the record read IDENTICAL on a wrong answer. NO
    # IDENTITY CHECK CAN CATCH THAT; only an argmin check can. cuVS takes
    # `raft::linalg::norm<L2Norm>` (squared) for L2Expanded AND
    # L2SqrtExpanded (`detail/kmeans.cuh:141-144`, `:1082-1085`), and so
    # does the fit's own norm (`detail/kmeans.mojo`, `Int32(0)`); the root
    # belongs to the reduction's output alone (`metric_is_sqrt`).
    # `inertia_` never read this buffer (it is the fit's own final pass), so
    # only `labels_` moves.
    #
    # `centroid_norms_take_sqrt` now returns False for EVERY metric, because
    # the one metric that divided by the norms (cosine) was deleted on
    # 2026-09-18 (lane/kmeans-cosine-capability). The call is kept rather
    # than folded to `Int32(0)` so that this measurement stays on the path
    # of anyone who goes looking for the flag. Do not simplify it away.
    return _kmeans_fit_tail(
        ctx, x, x_norm, weights, centroids, labels, min_dist, n_samples, n_features,
        n_clusters, out_centroids_ptr, out_labels_ptr, max_iter, tol, seed, n_init, init,
        metric, oversampling_factor, sum_scale, weight_scale,
    )



def _kmeans_fit_tail(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_norm: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.uint32],
    mut min_dist: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    out_centroids_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_labels_ptr: MutPointer[UInt32, MutUntrackedOrigin],
    max_iter: Int,
    tol: Float64,
    seed: UInt64,
    n_init: Int,
    init: Int,
    metric: Int,
    oversampling_factor: Float64,
    sum_scale: Float64,
    weight_scale: Float64,
    copy_labels: Bool = True,
) raises -> KMeansFitResult:
    """`kmeans_fit` from the uploaded design on: the row norms, the fit, the
    read-back (shared with `kmeans_fit_rows`). `copy_labels` False leaves
    the labels in `labels` on the device (`out_labels_ptr` is not written)."""
    var take_sqrt = Int32(0)
    if centroid_norms_take_sqrt(metric):
        take_sqrt = Int32(1)
    ctx.enqueue_function[row_norm_kernel](
        x_norm.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_features),
        take_sqrt,
        grid_dim=(n_samples, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )
    ctx.synchronize()

    var params = KMeansParams.default()
    params.n_clusters = n_clusters
    params.init = init
    params.metric = metric
    params.max_iter = max_iter
    params.tol = tol
    params.seed = seed
    params.n_init = n_init
    params.oversampling_factor = oversampling_factor

    var result = fit_predict(
        ctx,
        x,
        x_norm,
        weights,
        centroids,
        labels,
        min_dist,
        params,
        n_samples,
        n_features,
        Float32(sum_scale),
        Float32(weight_scale),
    )

    # DEVIATION 2672: THE RESULTS LAND IN THE CALLER'S MEMORY DIRECTLY. This
    # used to allocate a pinned host buffer for each output (at 4,000,000
    # rows the label buffer alone is 16 MB of pinned memory per fit), copy
    # the device into them, and then copy them value by value on one thread
    # into the caller's arrays. The copy engine writes the caller's pages
    # instead, which is the same bytes and two host passes fewer.
    ctx.enqueue_copy(dst_ptr=out_centroids_ptr, src_buf=centroids)
    if copy_labels:
        ctx.enqueue_copy(dst_ptr=out_labels_ptr, src_buf=labels)
    ctx.synchronize()

    return KMeansFitResult(
        result.inertia, result.n_iter, sum_scale, weight_scale
    )


def kmeans_fit_rows(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    host_x: MutPointer[Float32, MutUntrackedOrigin],
    rows: List[Int],
    n_features: Int,
    n_clusters: Int,
    out_centroids_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_labels_ptr: MutPointer[UInt32, MutUntrackedOrigin],
    max_iter: Int = 300,
    tol: Float64 = 1e-4,
    seed: UInt64 = 0,
    n_init: Int = 1,
    init: Int = INIT_KMEANS_PLUS_PLUS,
    metric: Int = METRIC_L2_EXPANDED,
    oversampling_factor: Float64 = 2.0,
) raises -> KMeansFitResult:
    """`kmeans_fit` with unit weights on rows `rows` of the host matrix
    `host_x`, whose gathered copy (rows in that order) is ALREADY on the
    device in `x` (lane/neural-pass108: BisectingKMeans gathers each
    cluster on the device instead of copying and uploading it per split).
    The same device scale over the gathered copy (`plan_sum_scale`) and the
    same tail, so the same words; `host_x` is not read."""
    var n_samples = len(rows)
    if n_samples < 1 or n_features < 1 or n_clusters < 1 or n_clusters > n_samples:
        raise Error("kmeans_fit_rows: bad shape " + String(n_samples) + " x " + String(n_features)
                    + " for " + String(n_clusters) + " clusters")
    var sum_scale = plan_sum_scale(ctx, x, n_samples, n_features)
    var weight_scale = choose_scale(Float64(n_samples), n_samples)
    var cd = n_clusters * n_features
    var weights = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var centroids = ctx.enqueue_create_buffer[DType.float32](cd)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n_samples)
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n_samples)
    enqueue_fill[DType.float32](ctx, weights, Float32(1.0))
    return _kmeans_fit_tail(
        ctx, x, x_norm, weights, centroids, labels, min_dist, n_samples, n_features,
        n_clusters, out_centroids_ptr, out_labels_ptr, max_iter, tol, seed, n_init, init,
        metric, oversampling_factor, sum_scale, weight_scale,
    )


def kmeans_fit_rows_resident(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    out_centroids_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mut labels: DeviceBuffer[DType.uint32],
    max_iter: Int = 300,
    tol: Float64 = 1e-4,
    seed: UInt64 = 0,
    n_init: Int = 1,
    init: Int = INIT_KMEANS_PLUS_PLUS,
    metric: Int = METRIC_L2_EXPANDED,
    oversampling_factor: Float64 = 2.0,
) raises -> KMeansFitResult:
    """`kmeans_fit_rows` (the same scale, the same tail, so the same words)
    on the `n_samples` gathered rows in `x`, with the labels left on the
    device in `labels` (n_samples words) for a caller that consumes them
    there (lane cpu3-core: the FAST bisecting split keeps row membership on
    the device). Only the centroids come back."""
    if n_samples < 1 or n_features < 1 or n_clusters < 1 or n_clusters > n_samples:
        raise Error("kmeans_fit_rows_resident: bad shape " + String(n_samples) + " x " + String(n_features)
                    + " for " + String(n_clusters) + " clusters")
    var sum_scale = plan_sum_scale(ctx, x, n_samples, n_features)
    var weight_scale = choose_scale(Float64(n_samples), n_samples)
    var cd = n_clusters * n_features
    var weights = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var centroids = ctx.enqueue_create_buffer[DType.float32](cd)
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n_samples)
    enqueue_fill[DType.float32](ctx, weights, Float32(1.0))
    # the tail does not write `out_labels_ptr` with copy_labels False; any
    # valid host pointer of the type stands in (never dereferenced)
    var no_host_labels = out_centroids_ptr.bitcast[UInt32]()
    return _kmeans_fit_tail(
        ctx, x, x_norm, weights, centroids, labels, min_dist, n_samples, n_features,
        n_clusters, out_centroids_ptr, no_host_labels, max_iter, tol, seed, n_init, init,
        metric, oversampling_factor, sum_scale, weight_scale, copy_labels=False,
    )


def kmeans_predict(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    centroids_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_labels_ptr: MutPointer[UInt32, MutUntrackedOrigin],
    metric: Int = METRIC_L2_EXPANDED,
) raises:
    """The nearest centroid of every row: cuML's `KMeans.predict`, which is
    `_predict_labels_inertia` keeping the labels (`kmeans.pyx:1071-1082`),
    one cuVS assignment pass (`kmeans_predict`, `detail/kmeans.cuh`).

    THE SAME PASS AS `kmeans_fit`'s FINAL ASSIGNMENT, statement for
    statement: the row norms from `row_norm_kernel` with the same
    `centroid_norms_take_sqrt` flag (squared for both metrics, DEVIATION
    2716), then `cluster/impl/kmeans.mojo::predict`, the call `fit_predict`
    makes. So on the training rows and the fitted centroids the labels are
    `labels_` by construction, and the tie rule (the lowest centroid index
    on an equal distance) is the fused kernel's. An unsupported metric is
    refused by name exactly as the fit refuses it
    (`KMeansParams.validate`). `x_ptr` is `n_samples x
    n_features` row-major float32, `centroids_ptr` `n_clusters x
    n_features`; `out_labels_ptr` is written.
    """
    if n_samples < 1 or n_features < 1 or n_clusters < 1:
        raise Error(
            "kmeans_predict needs n_samples, n_features and n_clusters >= 1: got "
            + String(n_samples)
            + ", "
            + String(n_features)
            + ", "
            + String(n_clusters)
        )
    var params = KMeansParams.default()
    params.n_clusters = n_clusters
    params.metric = metric
    params.validate()

    var x = ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    var centroids = ctx.enqueue_create_buffer[DType.float32](
        n_clusters * n_features
    )
    var labels = ctx.enqueue_create_buffer[DType.uint32](n_samples)
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=centroids, src_ptr=centroids_ptr)
    kmeans_predict_device(
        ctx, x, n_samples, n_features, n_clusters, centroids, labels, metric
    )
    ctx.enqueue_copy(dst_ptr=out_labels_ptr, src_buf=labels)
    ctx.synchronize()


def kmeans_predict_device(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    mut centroids: DeviceBuffer[DType.float32],
    mut labels: DeviceBuffer[DType.uint32],
    metric: Int = METRIC_L2_EXPANDED,
) raises:
    """`kmeans_predict` on rows and centroids already on the device (the
    same pass, statement for statement); `labels` is written on the
    device. For callers whose rows a kernel just produced (the spectral
    Nystrom predict), so they do not leave the device in between."""
    if n_samples < 1 or n_features < 1 or n_clusters < 1:
        raise Error(
            "kmeans_predict needs n_samples, n_features and n_clusters >= 1: got "
            + String(n_samples)
            + ", "
            + String(n_features)
            + ", "
            + String(n_clusters)
        )
    var params = KMeansParams.default()
    params.n_clusters = n_clusters
    params.metric = metric
    params.validate()
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var take_sqrt = Int32(0)
    if centroid_norms_take_sqrt(metric):
        take_sqrt = Int32(1)
    ctx.enqueue_function[row_norm_kernel](
        x_norm.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_features),
        take_sqrt,
        grid_dim=(n_samples, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )
    predict(
        ctx,
        x,
        x_norm,
        centroids,
        labels,
        min_dist,
        params,
        n_samples,
        n_features,
    )
    ctx.synchronize()
    _ = x_norm^
    _ = min_dist^


def kmeans_transform(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    centroids_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    metric: Int = METRIC_L2_EXPANDED,
) raises:
    """The distance from every row to every centroid under the model's
    metric: cuML's `KMeans.transform`, cuVS `kmeans_transform`
    (`detail/kmeans.cuh:1178-1219`). `out_ptr` is written row-major
    `n_samples x n_clusters` float32: squared distances for
    `METRIC_L2_EXPANDED`, their roots for `METRIC_L2_SQRT_EXPANDED`.

    The norms are `kmeans_predict`'s (`row_norm_kernel` with the same
    `centroid_norms_take_sqrt` flag, then `compute_centroid_norms`), and the
    cell is the fused kernel's epilog
    (`cluster/impl/detail/kmeans_transform.mojo`), so the cell at
    `kmeans_predict`'s label is the row minimum. An unsupported metric is
    refused by name as the fit refuses it (`KMeansParams.validate`).
    """
    if n_samples < 1 or n_features < 1 or n_clusters < 1:
        raise Error(
            "kmeans_transform needs n_samples, n_features and n_clusters >= 1: got "
            + String(n_samples)
            + ", "
            + String(n_features)
            + ", "
            + String(n_clusters)
        )
    var params = KMeansParams.default()
    params.n_clusters = n_clusters
    params.metric = metric
    params.validate()

    var x = ctx.enqueue_create_buffer[DType.float32](n_samples * n_features)
    var centroids = ctx.enqueue_create_buffer[DType.float32](
        n_clusters * n_features
    )
    var x_norm = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var centroid_norm = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    var out = ctx.enqueue_create_buffer[DType.float32](n_samples * n_clusters)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.enqueue_copy(dst_buf=centroids, src_ptr=centroids_ptr)
    var take_sqrt = Int32(0)
    if centroid_norms_take_sqrt(metric):
        take_sqrt = Int32(1)
    ctx.enqueue_function[row_norm_kernel](
        x_norm.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_features),
        take_sqrt,
        grid_dim=(n_samples, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )
    compute_centroid_norms(
        ctx, centroids, centroid_norm, n_clusters, n_features, metric
    )
    ctx.synchronize()
    var cells = n_samples * n_clusters
    var is_sqrt = Int32(0)
    if metric_is_sqrt(metric):
        is_sqrt = Int32(1)
    ctx.enqueue_function[kmeans_transform_kernel](
        out.unsafe_ptr(),
        x.unsafe_ptr(),
        centroids.unsafe_ptr(),
        x_norm.unsafe_ptr(),
        centroid_norm.unsafe_ptr(),
        Int32(n_samples),
        Int32(n_clusters),
        Int32(n_features),
        is_sqrt,
        grid_dim=((cells + TRANSFORM_TPB - 1) // TRANSFORM_TPB, 1, 1),
        block_dim=(TRANSFORM_TPB, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=out)
    ctx.synchronize()
