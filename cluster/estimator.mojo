# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The callable surface over the k-means fit.

**Why this file exists.** `cluster/ported/cluster/kmeans.mojo` already has
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

Nothing here is a port. `cluster/ported/` mirrors cuVS and is governed by
COPY, DO NOT IMPROVE; this file is host-side policy cuVS has no counterpart
for, in the same category as `mojo_only/`. It follows
`neighbors/estimator.mojo`, which is the first file of this kind, including
its convention that data crosses as raw pointers plus lengths so a CPython
extension can pass buffer addresses straight through.

THE POLICY CHOICES
------------------

1. **THE SCALE IS COMPUTED FROM THE CALLER'S DATA, AND IT COSTS A HOST PASS.**
   `choose_scale` needs `sum over rows of abs(value)` for the plane being
   accumulated, and the bound that matters is the worst column. So this file
   walks all `n_samples * n_features` values on the host before anything is
   uploaded. At the benchmark's 4,000,000 x 32 that is 128 million reads.
   It is not free and it is not hidden: `KMeansFitResult` reports the scale
   that was chosen, and `plan_sum_scale` is exposed separately so a caller
   who already knows their data's bound can compute it once and reuse it.

   The alternative was a fixed scale, which is what an implementation does
   when it does not want to admit this cost. A fixed scale is a silent wrong
   answer on data whose magnitude it did not anticipate.

2. **`row_count` IS PASSED TO `choose_scale`, AND THE CHECKS DO NOT DO THAT.**
   `mojo_only/fixed_point.mojo:55-70` documents that stating the row count
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
  the ported layer and wants a caller-facing wrapper of its own; it is a
  different call and is not wired here.
- `n_init > 1` and `INIT_ARRAY` are exercised at this boundary since
  2026-08-23 by `tools/e2u_matrix_fit.py` (`kmeans_k8_ninit3`,
  `kmeans_k8_array`: both move the answer against the baseline, and the
  restart card tags `restart01.*` appear). No Mojo check covers them.
- Metrics other than `METRIC_L2_EXPANDED`. `METRIC_L2_SQRT_EXPANDED` and
  `METRIC_COSINE_EXPANDED` pass through untested from here, and the Python
  surface does not expose `metric` at all.
- The CPython extension EXISTS (`bindings/_mojolearn.mojo::kmeans_fit_binding`,
  `python/mojolearn/cluster.py`); this sentence used to say it did not.
"""

from max.gpu.host import DeviceContext

from cluster.ported.cluster.detail.kmeans_common import metric_is_sqrt
from cluster.ported.cluster.kmeans import fit_predict
from core.row_norms import NORM_TPB, row_norm_kernel
from cluster.ported.cluster.kmeans_params import (
    INIT_ARRAY,
    INIT_KMEANS_PLUS_PLUS,
    KMeansParams,
    METRIC_L2_EXPANDED,
)
from mojo_only.fixed_point import choose_scale


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


def plan_sum_scale(
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_samples: Int,
    n_features: Int,
) raises -> Float64:
    """The fixed-point multiplier for this data, from the worst column.

    `choose_scale` bounds a partial sum over any SUBSET of rows, so the bound
    it needs is `sum over rows of abs(value)` for the plane being accumulated.
    The centroid accumulation forms one such sum per feature, so the binding
    constraint is the largest of them and that is what is returned.

    Separated out so a caller can pay the host pass once across several fits
    of the same data, and so a check can assert the policy without running a
    fit. This mirrors `plan_query_tile` in `neighbors/estimator.mojo` for the
    same two reasons.
    """
    var worst = Float64(0.0)
    for f in range(n_features):
        var column = Float64(0.0)
        for r in range(n_samples):
            var v = x_ptr.unsafe_load(r * n_features + f)
            column += Float64(abs(v))
        if column > worst:
            worst = column
    return choose_scale(worst, n_samples)


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
) raises -> KMeansFitResult:
    """Fit k-means on host-resident row-major data. See THE POLICY CHOICES.

    `x_ptr` is `n_samples x n_features`, row-major, float32.
    `out_centroids_ptr` is `n_clusters x n_features` and is WRITTEN; when
    `init == INIT_ARRAY` it is also READ first, as the starting centroids.
    `out_labels_ptr` is `n_samples` and is written with the assignment
    against the FINAL centroids.

    `n_weights` is 0 for unit weights, in which case `weights_ptr` is never
    read and any pointer will do; otherwise it must equal `n_samples`.
    `requested_sum_scale` of 0.0 means compute it from the data; pass a value
    from `plan_sum_scale` to skip the host pass.
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
    if sum_scale <= 0.0:
        sum_scale = plan_sum_scale(x_ptr, n_samples, n_features)
    # THE WEIGHT BOUND IS NOT ALWAYS n_samples. Unit weights sum to exactly
    # that, but caller-supplied weights can sum to anything, and using
    # n_samples for them would understate the bound and overflow the
    # accumulator. So the supplied case is summed, at the cost of one more
    # host pass over n_samples values -- cheap beside the n_samples *
    # n_features pass `plan_sum_scale` already pays.
    var weight_bound = Float64(n_samples)
    if n_weights != 0:
        weight_bound = Float64(0.0)
        for r in range(n_samples):
            weight_bound += Float64(abs(weights_ptr.unsafe_load(r)))
    var weight_scale = choose_scale(weight_bound, n_samples)

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

    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_samples)
    ctx.synchronize()
    if n_weights != 0:
        for r in range(n_samples):
            hw.unsafe_ptr().unsafe_store(r, weights_ptr.unsafe_load(r))
    else:
        for r in range(n_samples):
            hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())

    # INIT_ARRAY is the one init that READS this buffer. The others overwrite
    # it, so uploading unconditionally would be wasted traffic on the common
    # path rather than merely harmless.
    if init == INIT_ARRAY:
        ctx.enqueue_copy(dst_buf=centroids, src_ptr=out_centroids_ptr)
    # One block per row, matching every other launch of this kernel. The
    # sqrt flag follows the metric, exactly as the fit path's own norm does:
    # a squared-distance metric wants squared norms.
    var take_sqrt = Int32(0)
    if metric_is_sqrt(metric):
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

    var hc = ctx.enqueue_create_host_buffer[DType.float32](cd)
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](n_samples)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=centroids)
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    for i in range(cd):
        out_centroids_ptr.unsafe_store(i, hc.unsafe_ptr().unsafe_load(i))
    for i in range(n_samples):
        out_labels_ptr.unsafe_store(i, hl.unsafe_ptr().unsafe_load(i))

    return KMeansFitResult(
        result.inertia, result.n_iter, sum_scale, weight_scale
    )
