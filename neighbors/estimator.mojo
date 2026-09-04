# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The callable surface over the brute-force k-NN kernel.

**Why this file exists.** Before it, `neighbors/` held a verified kernel, a
benchmark that timed it, and checks that proved it correct -- and no way for
anyone outside this repository to call any of it. `neighbors/__init__.mojo`
was empty and every entry point under `neighbors/` was a `*_main.mojo` driver
or a `checks/*_check.mojo` verifier. Five algorithms measured, zero
reachable.

Nothing here is a port. `neighbors/gbdt/` mirrors cuVS and is governed by
COPY, DO NOT IMPROVE; this file is host-side policy that cuVS does not have a
counterpart for, in the same category as `checks/`. Every choice it makes
that a caller could observe is named in THE POLICY CHOICES below rather than
left implicit.

**The boundary shape is deliberate.** Data crosses as raw pointers plus
lengths, matching the convention mojotrees' `bindings/_mojotrees.mojo` already
uses, so the eventual CPython extension passes buffer addresses straight
through without a second representation being invented in between.

THE POLICY CHOICES
------------------

1. `query_tile` DEFAULTS TO 256, which is the value
   `bench/bench_main.mojo:72` was measured at. It is not a tuned number and
   it is not claimed to be optimal; it is the number the published 1.51x
   describes. Changing it moves off the measured configuration, so
   `knn_search` reports the tile it actually used rather than letting a
   caller assume.

2. THE WORKSPACE IS CAPPED AND THE CAP CAN LOWER THE TILE.
   `tiled_brute_force_knn` needs a `query_tile x n_index` distance tile, which
   at the benchmark's 400,000-point index is already 409 MB. Left alone it
   grows without bound: a 4,000,000-point index at tile 256 would ask for
   4.1 GB on a 16 GB machine. So the tile is lowered until the tile fits
   `WORKSPACE_BUDGET_BYTES`. **The budget is set so the benchmark shape is
   untouched** -- 409 MB is under it -- and the lowering only begins above
   roughly a 500,000-point index. When it fires, the configuration is no
   longer the measured one and `used_query_tile` says so.

3. `return_sqrt` DEFAULTS TO TRUE, and the benchmark ran with it FALSE.
   scikit-learn's `kneighbors` returns Euclidean distances; the benchmark
   timed squared distances because that is what the check compares. The
   default here follows scikit-learn because this is the caller-facing file,
   and the difference is one `sqrt` per returned element -- `n_queries * k`
   of them, not `n_queries * n_index`. It is not on the hot path and it does
   not disturb the ordering, because `sqrt` is monotone. A caller who wants
   the benchmark's exact arithmetic passes `return_sqrt=False`.

4. THE `knn_method` DISPATCH IS NOT OVERRIDDEN. It is left at
   `KNN_METHOD_AUTO`, which is the shipped default under DEVIATION 36 and the
   arm every published k-NN number describes. This file does not get an
   opinion about which kernel runs.

WHAT IS NOT HERE YET, NAMED SO IT IS NOT MISTAKEN FOR DONE
----------------------------------------------------------

- `radius_neighbors` EXISTS since 2026-08-31: `radius_neighbors_count` /
  `radius_neighbors_fill` below, over the ball cover, bound as
  `_mojolearn.radius_neighbors_count` / `.radius_neighbors_fill` and exported
  as `mojolearn.RadiusNeighbors`. This bullet used to name it as absent. What
  is still owed is a FITTED DEVICE HANDLE, so the index is built once rather
  than once per boundary call; the RADIUS NEIGHBOURS banner below says where
  that cost is paid.
- Metrics other than expanded L2 EXIST since 2026-09-01:
  `impl/distance/detail/distance_ops.mojo::metric_distance_kernel` computes
  L2Expanded, L2SqrtExpanded, CosineExpanded, L1, L2Unexpanded,
  L2SqrtUnexpanded, Linf and LpUnexpanded, and `_METRIC_TABLE` in
  `python/mojolearn/neighbors.py` routes twelve spellings onto them. This
  bullet said cosine and L1 were "a port, not a flag" and that stopped being
  true when the metric lane landed. What is still refused there is the six
  names in cuML's `VALID_METRICS['brute']` that no kernel here computes;
  `NOT_IMPLEMENTED.tsv` lists them.
- k-NN OVER AN INDEX EXISTS since 2026-09-01: `rbc_knn_search` at the bottom
  of this file, over the same random ball cover the radius surface uses. It
  is EXACT, not approximate, and it is the honest answer to a request for
  `algorithm='kd_tree'`; `neighbors/impl/neighbors/ball_cover/knn.mojo`
  carries the bounds, their proofs and DEVIATIONS 558 to 567. What is still
  owed there is the same fitted device handle the radius bullet owes, and a
  `bench/` row: no arm of that lane has measured SPEED, only exactness and
  the pruning ratio `rbc_knn_search` returns.
- `KNeighborsClassifier` / `KNeighborsRegressor` EXIST since 2026-08-23:
  `knn_classifier_predict` / `knn_regressor_predict` below, over the cuML
  port in `neighbors/impl/knn/knn.mojo` and
  `neighbors/impl/selection/knn.mojo`, bound as `_mojolearn.knn_classify`
  / `.knn_regress` and exported as `mojolearn.KNeighborsClassifier` /
  `.KNeighborsRegressor`. This bullet used to name them as absent.
- The CPython extension EXISTS: `bindings/_mojolearn.mojo::knn_search_binding`
  and `python/mojolearn/neighbors.py` (`mojolearn.NearestNeighbors`). This
  bullet used to say nothing here was importable from Python; that stopped
  being true when the binding landed and the sentence outlived it.

THE REPRODUCIBILITY LIMITATION, WHICH IS REAL AND IS NOT HIDDEN
---------------------------------------------------------------

`archive/plans/UNWIRED.md:371`: RAFT places k-NN output with `atomicAdd` and has no index
tie-break, so under the DEFAULT build **which of several equidistant
neighbours is returned is not reproducible**. Distances are stable; the
identity of a tied neighbour is not. Any caller building a bit-identity
claim on the FAST build must know that. It is not hypothetical: with 40
identical queries, the fused arm has been observed returning a different
neighbour for row 16 than for row 0, in one process on one device
(`check_knn_fused_tie_set_is_geometry_invariant`).

**UNDER `NUMERIC_IDENTICAL` THIS IS CLOSED, 2026-08-23** (IDENTITY_PATHS
row 11, DEVIATIONS 500/501/502). The tiled arm's selector runs over a
64-bit `(distance, index)` composite key, so the tie class does not exist
and the lowest indices win by arithmetic; its output slots come from a rank
rather than an atomic arrival. The fused arm's `grid_x` is pinned to 1, so
no mutex merge decides anything, and the ARM ITSELF is pinned to cuVS's own
dispatch -- because the two arms break a tie differently and an AUTO that
chooses by SHAPE would make the answer depend on how many queries the
caller passed.

The host sort below is unchanged and still does its own job: it fixes the
ORDER of the set, which is a different property from WHICH set.
"""

from core.identity_trace import IdentityTrace
from max.gpu.host import DeviceBuffer, DeviceContext

from neighbors.impl.knn.knn import (
    knn_class_proba,
    knn_classify,
    knn_regress,
)
from neighbors.impl.neighbors.detail.knn_brute_force import (
    KNN_METHOD_AUTO,
    METRIC_FROM_IS_SQRT,
    brute_force_knn_impl,
    compute_norms_for_metric,
    resolve_metric,
)
from neighbors.impl.selection.distance_weights import (
    WEIGHTS_DISTANCE,
    WEIGHTS_UNIFORM,
    host_distance_weights,
)
from neighbors.impl.distance.detail.distance_ops import (
    DIST_COSINE_EXPANDED,
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_EXPANDED,
    DIST_LINF,
    DIST_LP_UNEXPANDED,
    cosine_zero_norm_row_ptr,
    metric_uses_norms,
    metric_value_name,
    validate_metric_arg,
)


def knn_metric_from_name(name: String) raises -> Int:
    """cuML's `NearestNeighbors._build_metric_type`
    (`nearest_neighbors.pyx:520-553`), the rows this tree computes.

    THIS TABLE IS NOT `kde/impl/neighbors/kernel_density.mojo::
    metric_from_name` AND THE DIFFERENCE IS UPSTREAM'S, NOT OURS.
    cuML has TWO metric tables and they disagree on one row:

        `pairwise_distances.pyx:71`   "euclidean" -> L2SqrtUNexpanded
        `nearest_neighbors.pyx:522`   "euclidean" -> L2SqrtEXpanded

    KDE goes through the first (`kernel_density.py:315`) and k-NN through
    the second, so the two lanes genuinely compute Euclidean by two
    different identities and can differ in the last bits. That is THEIR
    design and this port keeps it; a single shared table here would have
    silently corrected one of their call sites.

    `lp` is their alias for `minkowski` (`:532`) and `linf`/`taxicab` are
    their aliases too (`:526`, `:534`). Every name in their
    `VALID_METRICS["brute"]` set (`neighbors/__init__.py:27-48`) that this
    tree does not compute is refused BY NAME, so a caller learns it is
    unported rather than unknown.
    """
    if name == "euclidean" or name == "l2":
        return DIST_L2_SQRT_EXPANDED
    if name == "sqeuclidean":
        return DIST_L2_EXPANDED
    if (
        name == "cityblock"
        or name == "l1"
        or name == "manhattan"
        or name == "taxicab"
    ):
        return DIST_L1
    if name == "chebyshev" or name == "linf":
        return DIST_LINF
    if name == "cosine":
        return DIST_COSINE_EXPANDED
    if name == "minkowski" or name == "lp":
        return DIST_LP_UNEXPANDED
    if (
        name == "canberra"
        or name == "jensenshannon"
        or name == "correlation"
        or name == "inner_product"
        or name == "haversine"
        or name == "braycurtis"
    ):
        raise Error(
            "mojolearn k-NN: metric='"
            + name
            + "' is in cuML's VALID_METRICS['brute'] but is NOT PORTED"
            " (neighbors/NOT_IMPLEMENTED.tsv); ported: euclidean, l2,"
            " sqeuclidean, l1, cityblock, manhattan, taxicab, chebyshev,"
            " linf, cosine, minkowski, lp"
        )
    raise Error("mojolearn k-NN: unknown metric '" + name + "'")
from neighbors.checks.radius_distances import rbc_edge_distances
from neighbors.impl.neighbors.ball_cover.common import (
    RBC_METRIC_DEFAULT,
    rbc_validate_metric,
)
from neighbors.impl.neighbors.ball_cover.knn import (
    RBC_KNN_MAX_K,
    rbc_knn_query,
)
from neighbors.impl.neighbors.ball_cover.ball_cover import (
    rbc_build_index,
    rbc_eps_nn_query_count,
    rbc_eps_nn_query_fill,
    rbc_n_landmarks,
)


comptime DEFAULT_QUERY_TILE = 256
"""`bench/bench_main.mojo:72`. The value the published 1.51x was taken at."""

comptime MIN_QUERY_TILE = 32
"""The floor the workspace cap will not lower past. Below this the tile loop
dominates and the arm stops resembling anything that was measured."""

comptime WORKSPACE_BUDGET_BYTES = 768 * 1024 * 1024
"""Ceiling on the `query_tile x n_index` distance tile ALONE.

Chosen so the benchmark's 400,000-point index at tile 256 (409 MB) is under
it and therefore untouched. This is a policy number on a 16 GB machine, not
a measured one.
"""


def plan_query_tile(n_index: Int, n_queries: Int, requested_tile: Int) -> Int:
    """The tile actually used, after the workspace cap and the query clamp.

    Separated out so a caller can ask what a shape will cost before paying
    for it, and so a check can assert both rules fire where they should
    without allocating anything.
    """
    var tile = requested_tile
    if tile < 1:
        tile = DEFAULT_QUERY_TILE

    var per_row_bytes = n_index * 4
    if per_row_bytes > 0:
        while (
            tile > MIN_QUERY_TILE
            and tile * per_row_bytes > WORKSPACE_BUDGET_BYTES
        ):
            tile = tile // 2
    if tile < MIN_QUERY_TILE:
        tile = MIN_QUERY_TILE

    # THE QUERY CLAMP, AND IT IS NOT DEFENSIVE TIDINESS.
    #
    # A tile wider than the query set is not merely wasteful, it returns
    # WRONG NEIGHBOURS, and the kernel does not guard it. Measured the first
    # time `check_knn_search_matches_host` ran, at tile 256 against 64
    # queries: 196 of 320 indices wrong, worst distance error 0.266, which is
    # a wrong answer rather than a precision artifact. `knn_check.mojo` never
    # saw it because its `KNN_TILE` is 64 and its `KNN_QUERIES` is 64, so
    # tile never exceeded the query count on any path that had been run.
    #
    # This clamp goes LAST, after the floor, because a caller with fewer
    # queries than `MIN_QUERY_TILE` must get their query count and not the
    # floor.
    if tile > n_queries:
        tile = n_queries
    if tile < 1:
        tile = 1
    return tile


def knn_search(
    ctx: DeviceContext,
    index_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_index: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    k: Int,
    out_dist_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_idx_ptr: MutPointer[UInt32, MutUntrackedOrigin],
    return_sqrt: Bool = True,
    requested_query_tile: Int = DEFAULT_QUERY_TILE,
    knn_method: Int = KNN_METHOD_AUTO,
    metric: Int = METRIC_FROM_IS_SQRT,
    metric_arg: Float32 = Float32(2.0),
) raises -> Int:
    """Exact k nearest neighbours, index and queries row-major on the host.

    The whole of the docstring is on `knn_search_traced`, which this calls
    with a trace read from the environment. Split 2026-08-23 for the same
    reason as `ols_fit_traced` (DEVIATION 517): the k-NN classifier and
    regressor run a search AND a vote, and a trace's `seq` must increase by
    one across the whole card (`core/identity_trace.mojo`), so the vote has
    to record into the SAME trace the search did. A second `IdentityTrace()`
    would append a second `seq 0` into the same file, which the differ
    refuses -- the exact defect DEVIATION 518 fixed for k-means++.
    """
    var trace = IdentityTrace()
    return knn_search_traced(
        ctx,
        trace,
        index_ptr,
        n_index,
        queries_ptr,
        n_queries,
        n_features,
        k,
        out_dist_ptr,
        out_idx_ptr,
        return_sqrt,
        requested_query_tile,
        knn_method,
        metric,
        metric_arg,
    )


def knn_search_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    index_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_index: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    k: Int,
    out_dist_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_idx_ptr: MutPointer[UInt32, MutUntrackedOrigin],
    return_sqrt: Bool = True,
    requested_query_tile: Int = DEFAULT_QUERY_TILE,
    knn_method: Int = KNN_METHOD_AUTO,
    metric: Int = METRIC_FROM_IS_SQRT,
    metric_arg: Float32 = Float32(2.0),
) raises -> Int:
    """Exact k nearest neighbours, index and queries row-major on the host.

    THE METRIC, 2026-09-01. `metric` is a cuVS `DistanceType` value and
    `metric_arg` is Minkowski's `p`, which is how cuML spells the pair all
    the way down (`nearest_neighbors.pyx:852-854` passes
    `<DistanceType>_metric` and `<float>self.p` to `brute_force_knn`
    regardless of the metric, because every non-Lp op discards it). Use
    `knn_metric_from_name` above for the string table. The DEFAULT is the
    sentinel `METRIC_FROM_IS_SQRT`, which reproduces this function's
    pre-metric behaviour exactly -- `return_sqrt` choosing between
    `L2SqrtExpanded` and `L2Expanded` -- so the nine existing call sites
    across `metrics/`, `spectral/`, `hdbscan/`, `ivf/`, `bench/` and the
    check files keep their arm and their bits with no edit.

    Both inputs are read as `n x n_features` row-major Float32. Both outputs
    are written as `n_queries x k` row-major, distances and indices in the
    same order, which is scikit-learn's `(distances, indices)` layout.

    Returns THE QUERY TILE THAT RAN, so a caller can record which
    configuration produced a number instead of assuming it was the default.
    That is the same discipline `PORTING_RULES.md` rule 8 asks of the
    benchmark, applied at the boundary where a user can actually see it.

    Raises rather than clamping on every shape the kernel cannot serve. A
    clamp here would return a wrong answer quietly, which is the failure mode
    this repository has paid for repeatedly.
    """
    if n_index <= 0:
        raise Error("knn_search: n_index must be positive, got " + String(n_index))
    if n_queries <= 0:
        raise Error(
            "knn_search: n_queries must be positive, got " + String(n_queries)
        )
    if n_features <= 0:
        raise Error(
            "knn_search: n_features must be positive, got " + String(n_features)
        )
    if k <= 0:
        raise Error("knn_search: k must be positive, got " + String(k))
    if k > n_index:
        # `brute_force_knn_impl` refuses this too, for the reason in its
        # docstring: cuVS's `n < k` fill at `:157-166` is not ported on
        # either arm. Caught here so the message names the caller's numbers.
        raise Error(
            "knn_search: k ("
            + String(k)
            + ") exceeds n_index ("
            + String(n_index)
            + "); the upstream's short-index fill is not ported"
        )

    # THE METRIC, RESOLVED AND VALIDATED BEFORE ANY ALLOCATION.
    var mtr = resolve_metric(metric, return_sqrt)
    validate_metric_arg(mtr, metric_arg)  # DEVIATION 552
    if mtr == DIST_COSINE_EXPANDED:
        # DEVIATION 553: cosine divides by ||x||, and cuVS has no guard
        # (`cosine.cuh:86`, `knn_brute_force.cuh:221`). A zero row would
        # make a whole row of the distance matrix NaN, and the two
        # selectors this lane ships sort a NaN to OPPOSITE ends -- radix's
        # `twiddle_in` key puts it above every finite distance, the FAISS
        # queue's `<` never admits it -- so the same query would lose a
        # real neighbour differently on the two arms, with no error. The
        # module docstring of `neighbors/impl/distance/detail/
        # distance_ops.mojo` carries the full argument.
        var zi = cosine_zero_norm_row_ptr(index_ptr, n_index, n_features)
        if zi >= 0:
            raise Error(
                "knn_search: metric='cosine' but index row "
                + String(zi)
                + " is all zeros; cosine distance divides by ||x|| and is"
                " undefined at the origin (DEVIATION 553)"
            )
        var zq = cosine_zero_norm_row_ptr(queries_ptr, n_queries, n_features)
        if zq >= 0:
            raise Error(
                "knn_search: metric='cosine' but query row "
                + String(zq)
                + " is all zeros; cosine distance divides by ||x|| and is"
                " undefined at the origin (DEVIATION 553)"
            )

    var query_tile = plan_query_tile(n_index, n_queries, requested_query_tile)

    # `scaling_main.mojo`'s sizing. `buf_len` must clear `k` or the fallback
    # selector has nowhere to put a full result row.
    var buf_len = n_index // 8
    if buf_len < k:
        buf_len = k

    var index = ctx.enqueue_create_buffer[DType.float32](n_index * n_features)
    var queries = ctx.enqueue_create_buffer[DType.float32](
        n_queries * n_features
    )
    var index_norm = ctx.enqueue_create_buffer[DType.float32](n_index)
    var query_norm = ctx.enqueue_create_buffer[DType.float32](n_queries)
    var dist_tile = ctx.enqueue_create_buffer[DType.float32](
        query_tile * n_index
    )
    var buf_val = ctx.enqueue_create_buffer[DType.float32](
        query_tile * 2 * buf_len
    )
    var buf_idx = ctx.enqueue_create_buffer[DType.uint32](
        query_tile * 2 * buf_len
    )
    var out_dist = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
    var out_idx = ctx.enqueue_create_buffer[DType.uint32](n_queries * k)
    var out_i32 = ctx.enqueue_create_buffer[DType.int32](n_queries * k)
    ctx.synchronize()

    ctx.enqueue_copy(dst_buf=index, src_ptr=index_ptr)
    ctx.enqueue_copy(dst_buf=queries, src_ptr=queries_ptr)
    ctx.synchronize()

    # `knn_brute_force.cuh:117-140`: WHICH norm, and whether one at all,
    # is the metric's decision. L2 wants the SQUARED norm on both sides,
    # cosine wants the TRUE L2 norm, and every unexpanded metric wants
    # none -- their comment at `:122` says exactly that. This used to be
    # two unconditional `compute_norms(..., False)` calls, which was right
    # while `L2Expanded` was the only metric and is not right now.
    #
    # For a metric with no norm the two buffers are allocated (Mojo's
    # launch refuses a null pointer argument) and NEVER WRITTEN, so they
    # hold whatever the allocator left. `metric_distance_kernel` does not
    # read them on those arms; the identity trace below records them only
    # when they mean something, for the same reason.
    compute_norms_for_metric(ctx, index, index_norm, n_index, n_features, mtr)
    compute_norms_for_metric(
        ctx, queries, query_norm, n_queries, n_features, mtr
    )
    ctx.synchronize()

    # THE STAGE HASHES (`core/identity_trace.mojo`), off unless
    # `MOJOLEARN_IDENTITY_TRACE` is set. FOUR records, and the split
    # between them is the diagnosis: the two norms are the input to every
    # distance, `search.out_dist` is the selection's values, and
    # `search.out_idx` is the one that carries the TIE CLASS. Two runs whose
    # distances agree and whose indices do not have diverged in the
    # selector and nowhere else -- which is precisely row 11's failure mode
    # and is invisible in any comparison of distances alone.
    #
    # The tags name algorithm positions and carry no tile, batch or grid,
    # so they align across machines (rule 2). The QUERY TILE is deliberately
    # absent from them for that reason, even though it is in the return
    # value: it is a memory number, and
    # `check_knn_tiled_is_query_tile_invariant` gates that the answer does
    # not depend on it.
    if trace.enabled:
        trace.header(
            String("knn n_index=") + String(n_index) + " n_queries="
            + String(n_queries) + " d=" + String(n_features) + " k="
            + String(k) + " method=" + String(knn_method) + " sqrt="
            + String(return_sqrt) + " metric=" + metric_value_name(mtr)
            + " metric_arg=" + String(metric_arg)
        )
        # ONLY WHEN THE METRIC HAS THEM. Hashing an unwritten buffer would
        # put allocator garbage into a card and make two runs of the SAME
        # call disagree at a stage the algorithm never reads -- the exact
        # false positive an identity card exists to avoid.
        if metric_uses_norms(mtr):
            trace.record_device(ctx, "knn.index_norm", index_norm, n_index)
            trace.record_device(ctx, "knn.query_norm", query_norm, n_queries)

    brute_force_knn_impl(
        ctx,
        queries,
        query_norm,
        index,
        index_norm,
        dist_tile,
        buf_val,
        buf_idx,
        out_dist,
        out_idx,
        out_i32,
        n_queries,
        n_index,
        n_features,
        k,
        query_tile,
        buf_len,
        return_sqrt,
        False,
        True,
        True,
        knn_method,
        mtr,
        metric_arg,
    )
    ctx.synchronize()

    # Device -> pinned host buffer -> the caller's memory. The second hop is
    # not decoration: `archive/plans/UNWIRED.md:31` records that a pointer from
    # `enqueue_create_host_buffer` is not interchangeable with an arbitrary
    # host pointer on this stack, and the failure is SILENT. Copying through
    # a buffer the runtime made keeps this on the route the checks exercise.
    # THESE TWO ARE PRE-SORT, AND A CARD DIFF MUST READ THEM AS SUCH. The
    # host sort below normalizes an order the two arms genuinely disagree
    # about (the table further down measures TILED at 157 of 320 "wrong"
    # ordered and 0 wrong as a SET), so two machines can differ here for a
    # reason the caller never sees. Measured 2026-08-23 with
    # `tools/check_column_invariance.sh`: under FAST, the APPLE and AMD
    # COLUMNS of one source on one device diverge at `knn.out_dist` while
    # every sorted distance agrees -- the arms were different, the answer
    # was the same multiset. Diagnose from the `.sorted` pair below; these
    # two localize WHICH ARM produced the difference.
    if trace.enabled:
        trace.record_device(ctx, "knn.out_dist", out_dist, n_queries * k)
        trace.record_device(ctx, "knn.out_idx", out_idx, n_queries * k)

    var hd = ctx.enqueue_create_host_buffer[DType.float32](n_queries * k)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * k)
    ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=out_dist)
    ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=out_idx)
    ctx.synchronize()

    # THE SORT, AND IT IS A CORRECTNESS REQUIREMENT RATHER THAN A COURTESY.
    #
    # THE TWO SHIPPED ARMS DISAGREE ABOUT ORDER. Measured 2026-08-20 against
    # a Float64 host brute force, same data, same truth function:
    #
    #     arm      ordered comparison    set comparison
    #     FUSED         0 of 320 wrong    0 wrong
    #     TILED       157 of 320 wrong    0 wrong
    #
    # and the same split holds at `knn_check.check_knn`'s own shape
    # (4096 x 64 x 16, k=8): 385 of 512 wrong ordered, 0 wrong as a set. So
    # `tiled_brute_force_knn` returns the RIGHT k neighbours in an
    # UNSPECIFIED order -- RAFT's radix select does not sort -- while the
    # fused path's `WarpSelect` returns them ascending because a bitonic
    # queue is ordered by construction.
    #
    # `KNN_METHOD_AUTO` chooses between those two arms BY SHAPE. Without this
    # sort, the order of a caller's results would depend on how many queries
    # they happened to pass, which is the worst kind of API: correct on the
    # shape you tested and differently ordered on the one you shipped.
    #
    # scikit-learn's `kneighbors` returns neighbours sorted by ascending
    # distance, so a drop-in has to as well.
    #
    # Cost is `n_queries * k^2` host comparisons -- 400,000 at the benchmark
    # shape against a 756 ms fit, so it does not move the number. The key is
    # (distance, index), a TOTAL order, so the ORDER is reproducible given
    # the set. It cannot repair `archive/plans/UNWIRED.md:371`, which is about WHICH of
    # several equidistant neighbours lands in the set at all.
    for i in range(n_queries):
        var base = i * k
        for a in range(1, k):
            var dv = hd.unsafe_ptr().unsafe_load(base + a)
            var iv = hi.unsafe_ptr().unsafe_load(base + a)
            var b = a - 1
            while b >= 0:
                var db = hd.unsafe_ptr().unsafe_load(base + b)
                var ib = hi.unsafe_ptr().unsafe_load(base + b)
                if db < dv or (db == dv and ib <= iv):
                    break
                hd.unsafe_ptr().unsafe_store(base + b + 1, db)
                hi.unsafe_ptr().unsafe_store(base + b + 1, ib)
                b -= 1
            hd.unsafe_ptr().unsafe_store(base + b + 1, dv)
            hi.unsafe_ptr().unsafe_store(base + b + 1, iv)

    # THE CALLER-VISIBLE STAGES, added 2026-08-23. Everything above the sort
    # is an arm's internal order; THIS is what `kneighbors` returns, and it
    # is what a cross-vendor claim about k-NN is a claim about. Recorded as
    # two tags rather than one because the failure they separate is the
    # whole of IDENTITY_PATHS row 11: two runs whose DISTANCES agree and
    # whose INDICES do not have chosen different members of an equidistant
    # class and have diverged in the selector and nowhere else. That exact
    # signature was observed between two COLUMNS under FAST on 2026-08-23
    # (`bench/results/column_invariance/`), so the pair is not hypothetical
    # bookkeeping.
    if trace.enabled:
        trace.record_host("knn.sorted_dist", hd.unsafe_ptr(), n_queries * k)
        trace.record_host("knn.sorted_idx", hi.unsafe_ptr(), n_queries * k)

    for i in range(n_queries * k):
        out_dist_ptr.unsafe_store(i, hd.unsafe_ptr().unsafe_load(i))
        out_idx_ptr.unsafe_store(i, hi.unsafe_ptr().unsafe_load(i))

    return query_tile


# =====================================================================
# THE k-NN CLASSIFIER AND REGRESSOR, 2026-08-23
#
# Host-side composition of TWO things that already exist: `knn_search_traced`
# above, and the cuML port in `neighbors/impl/knn/knn.mojo` (`ML::knn_classify`,
# `ML::knn_class_proba`, `ML::knn_regress`). cuML's Python does exactly this
# composition -- `kneighbors(X, return_distance=False)` then `knn_classify(...)`
# on the indices (`kneighbors_classifier.pyx:245-285`) -- and the only reason
# it is done in Mojo here rather than in `python/mojolearn/neighbors.py` is
# the IDENTITY TRACE: one card, one `seq` sequence, so the vote's stages sit
# after the search's stages in the same file. See `knn_search`'s docstring.
# THAT IS DEVIATION 544: their Python makes two calls (kneighbors, then
# knn_classify on the indices it got back); ours makes one, and the ported
# `ML::` functions return the unique-label sets (theirs return void) so the
# composition can check them against the wrapper's (policy 7 below).
#
# POLICY, since this file is where policy is named:
#
# 5. THE VOTE AND THE MEAN ARE ONE ARITHMETIC IN BOTH MODES. cuML's kernels
#    are serial per query row in neighbour-slot order (no atomic, no block
#    fold), so FAST has nothing faster to choose and IDENTICAL has nothing to
#    replace; the mode changes only the row-10 flushes (`ftz`) at the seams,
#    which are inert on Apple. DEVIATION 542 in `selection/knn.mojo`.
#
# 6. WHAT `y` IS, AT THE BOUNDARY. `n_outputs` columns, each `n_index` long
#    and CONTIGUOUS -- cuML's `order='F'` `y` (`:197`), which is what makes
#    `y[:, i].ptr` a column pointer there. The wrapper transposes a 2-D `y`
#    into that layout once at `fit`.
#
# 7. THE CLASS SET IS COMPUTED TWICE, ON PURPOSE, AND CHECKED. The wrapper
#    takes `np.unique` per output for `classes_` (the pyx takes `cp.unique`)
#    and sizes `predict_proba`'s columns by it; the Mojo side recomputes the
#    set with the ported `getUniquelabels` (`knn.cu:344`). `knn_classify`
#    returns its counts and THIS function raises if they disagree with the
#    wrapper's, rather than writing past the end of a buffer sized by the
#    other answer.
# =====================================================================


def knn_classifier_predict(
    ctx: DeviceContext,
    index_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_index: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    k: Int,
    y_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_outputs: Int,
    n_classes: List[Int],
    out_labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
    out_proba_ptr: MutPointer[Float32, MutUntrackedOrigin],
    out_uniq_ptr: MutPointer[Int32, MutUntrackedOrigin],
    want_proba: Bool,
    requested_query_tile: Int = DEFAULT_QUERY_TILE,
    metric: Int = METRIC_FROM_IS_SQRT,
    metric_arg: Float32 = Float32(2.0),
    weights: Int = WEIGHTS_UNIFORM,
) raises -> Int:
    """Search, then vote (`want_proba=False`: `predict`) or tally
    (`want_proba=True`: `predict_proba`). Returns the query tile that ran.

    `metric` / `metric_arg` are cuVS `DistanceType` and Minkowski `p`,
    exactly as `knn_search_traced` takes them; `knn_metric_from_name` is
    the string table.

    `weights` is `WEIGHTS_UNIFORM` (cuML's only arm) or `WEIGHTS_DISTANCE`
    (DEVIATION 556, scikit-learn's semantics, `neighbors/impl/selection/
    distance_weights.mojo`).

    POLICY 8, NEW WITH THE WEIGHTED ARM: THE SEARCH'S `return_sqrt`
    DEPENDS ON `weights`. A UNIFORM vote reads only the ORDER of the
    distances and the squared distance is monotone in the distance, so
    this function has always passed `return_sqrt = False` and saved a
    root per cell. A DISTANCE weight reads the VALUE: `1/d^2` is not
    `1/d`, and a weighted vote taken on squared distances is a different
    estimator, not a rounding difference. So the weighted arm asks for the
    rooted distance and the uniform arm does not, and BOTH ARE RIGHT.
    Under a metric with no sqrt to skip (L1, Linf, cosine, Lp) the flag
    is inert -- `resolve_metric` only consults it for the sentinel.

    `y_ptr`: `n_outputs` contiguous int32 columns of `n_index` (policy 6).
    `n_classes[i]`: the wrapper's class count for output `i` (policy 7).
    `out_labels_ptr`: `n_queries x n_outputs` int32 row-major, written when
    `want_proba` is False (the ORIGINAL label values).
    `out_proba_ptr`: the per-output `n_queries x n_classes[i]` float32
    blocks, concatenated in output order, written when `want_proba` is True.
    `out_uniq_ptr`: the sorted unique labels per output, concatenated
    (`sum(n_classes)` int32), ALWAYS written -- the wrapper asserts it equals
    `classes_`, which is the check policy 7 describes made visible.

    One call does one of the two, never both: each records `knn_clf.votes`
    and a tag is unique within a trace. That matches cuML, where `predict`
    and `predict_proba` are two `kneighbors` calls.
    """
    if n_outputs < 1:
        raise Error(
            "knn_classifier_predict: n_outputs must be positive, got "
            + String(n_outputs)
        )
    if len(n_classes) != n_outputs:
        raise Error(
            "knn_classifier_predict: n_classes has "
            + String(len(n_classes))
            + " entries for "
            + String(n_outputs)
            + " outputs"
        )

    var trace = IdentityTrace()
    var h_dist = ctx.enqueue_create_host_buffer[DType.float32](n_queries * k)
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * k)
    ctx.synchronize()
    var weighted = weights == WEIGHTS_DISTANCE
    if weights != WEIGHTS_UNIFORM and not weighted:
        raise Error(
            "knn_classifier_predict: weights value "
            + String(weights)
            + " is neither WEIGHTS_UNIFORM nor WEIGHTS_DISTANCE"
        )
    # `knn_search_traced` refuses k <= 0, k > n_index and the empty shapes
    # by name; nothing here re-derives those refusals.
    var used_tile = knn_search_traced(
        ctx,
        trace,
        index_ptr,
        n_index,
        queries_ptr,
        n_queries,
        n_features,
        k,
        h_dist.unsafe_ptr(),
        h_idx.unsafe_ptr(),
        weighted,  # policy 8: the weighted arm needs the ROOTED distance
        requested_query_tile,
        KNN_METHOD_AUTO,
        metric,
        metric_arg,
    )

    # DEVIATION 554: the weights are computed on the HOST, over the sorted
    # distances the search just wrote, exactly as scikit-learn computes
    # them in numpy over the same matrix. `distance_weights.mojo` carries
    # the reason (the zero test is a per-row any-reduction and the
    # replacement is row-level).
    var d_w = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
    if weighted:
        # HOST LISTS ACROSS THE BOUNDARY, not pointers: `archive/plans/UNWIRED.md:31`
        # records that a pointer from `enqueue_create_host_buffer` is not
        # interchangeable with an arbitrary host pointer on this stack and
        # that the failure is SILENT. The copy is `n_queries * k` floats,
        # the size of the answer the caller is already receiving.
        var d_in = List[Float32](capacity=n_queries * k)
        for i in range(n_queries * k):
            d_in.append(h_dist.unsafe_ptr().unsafe_load(i))
        var wl = host_distance_weights(d_in, n_queries, k)
        var h_w = ctx.enqueue_create_host_buffer[DType.float32](
            n_queries * k
        )
        ctx.synchronize()
        for i in range(n_queries * k):
            h_w.unsafe_ptr().unsafe_store(i, wl[i])
        ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
        ctx.synchronize()
        if trace.enabled:
            trace.record_host(
                "knn_clf.weights", h_w.unsafe_ptr(), n_queries * k
            )
        _ = h_w^

    # The sorted indices go back to the device, where cuML's kernels read
    # them (`knn_indices` in `class_probs_kernel`).
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](n_queries * k)
    var y = List[DeviceBuffer[DType.int32]]()
    for i in range(n_outputs):
        y.append(ctx.enqueue_create_buffer[DType.int32](n_index))
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=h_idx.unsafe_ptr())
    for i in range(n_outputs):
        ctx.enqueue_copy(
            dst_buf=y[i], src_ptr=y_ptr.unsafe_offset(i * n_index)
        )
    ctx.synchronize()

    var uniq: List[List[Int32]]
    if want_proba:
        var probas = List[DeviceBuffer[DType.float32]]()
        for i in range(n_outputs):
            probas.append(
                ctx.enqueue_create_buffer[DType.float32](
                    n_queries * n_classes[i]
                )
            )
        ctx.synchronize()
        uniq = knn_class_proba(
            ctx, trace, probas, d_idx, y, n_index, n_queries, k, d_w,
            weighted,
        )
        _check_class_counts(uniq, n_classes)
        var off = 0
        for i in range(n_outputs):
            var cnt = n_queries * n_classes[i]
            var h = ctx.enqueue_create_host_buffer[DType.float32](cnt)
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=probas[i])
            ctx.synchronize()
            for j in range(cnt):
                out_proba_ptr.unsafe_store(
                    off + j, h.unsafe_ptr().unsafe_load(j)
                )
            off += cnt
            _ = h^
        _ = probas^
    else:
        var labels = ctx.enqueue_create_buffer[DType.int32](
            n_queries * n_outputs
        )
        ctx.synchronize()
        uniq = knn_classify(
            ctx, trace, labels, d_idx, y, n_index, n_queries, k, d_w,
            weighted,
        )
        _check_class_counts(uniq, n_classes)
        var h = ctx.enqueue_create_host_buffer[DType.int32](
            n_queries * n_outputs
        )
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=labels)
        ctx.synchronize()
        for j in range(n_queries * n_outputs):
            out_labels_ptr.unsafe_store(j, h.unsafe_ptr().unsafe_load(j))
        _ = h^
        _ = labels^

    # The unique sets the port computed, handed out so the wrapper can
    # compare them to `classes_` -- the only way a caller can SEE policy
    # 7's agreement rather than trust it.
    var off = 0
    for i in range(n_outputs):
        for j in range(len(uniq[i])):
            out_uniq_ptr.unsafe_store(off + j, uniq[i][j])
        off += len(uniq[i])

    _ = h_dist^
    _ = h_idx^
    _ = d_idx^
    _ = d_w^
    _ = y^
    return used_tile


def _check_class_counts(got: List[List[Int32]], want: List[Int]) raises:
    """Policy 7: the port's `getUniquelabels` count against the wrapper's."""
    for i in range(len(want)):
        if len(got[i]) != want[i]:
            raise Error(
                "knn_classifier_predict: the ported getUniquelabels found "
                + String(len(got[i]))
                + " classes for output "
                + String(i)
                + " and the caller sized its buffers for "
                + String(want[i])
                + "; one of the two class sets is wrong and nothing is "
                + "written"
            )


def knn_regressor_predict(
    ctx: DeviceContext,
    index_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_index: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    k: Int,
    y_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_outputs: Int,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    requested_query_tile: Int = DEFAULT_QUERY_TILE,
    metric: Int = METRIC_FROM_IS_SQRT,
    metric_arg: Float32 = Float32(2.0),
    weights: Int = WEIGHTS_UNIFORM,
) raises -> Int:
    """Search, then the mean of the `k` neighbours' targets per output --
    UNWEIGHTED (cuML's `regress_avg_kernel`) or DISTANCE-WEIGHTED
    (DEVIATION 556, scikit-learn's `sum(y w)/sum(w)`).
    Returns the query tile that ran.

    `y_ptr`: `n_outputs` contiguous float32 columns of `n_index` (policy 6).
    `out_ptr`: `n_queries x n_outputs` float32 row-major.
    `metric` / `metric_arg` / `weights`: as `knn_classifier_predict`,
    including POLICY 8 -- the weighted arm asks the search for the ROOTED
    distance because the weight reads the value and not only the order.
    """
    if n_outputs < 1:
        raise Error(
            "knn_regressor_predict: n_outputs must be positive, got "
            + String(n_outputs)
        )
    var trace = IdentityTrace()
    var h_dist = ctx.enqueue_create_host_buffer[DType.float32](n_queries * k)
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * k)
    ctx.synchronize()
    var weighted = weights == WEIGHTS_DISTANCE
    if weights != WEIGHTS_UNIFORM and not weighted:
        raise Error(
            "knn_regressor_predict: weights value "
            + String(weights)
            + " is neither WEIGHTS_UNIFORM nor WEIGHTS_DISTANCE"
        )
    var used_tile = knn_search_traced(
        ctx,
        trace,
        index_ptr,
        n_index,
        queries_ptr,
        n_queries,
        n_features,
        k,
        h_dist.unsafe_ptr(),
        h_idx.unsafe_ptr(),
        weighted,  # policy 8
        requested_query_tile,
        KNN_METHOD_AUTO,
        metric,
        metric_arg,
    )

    var d_w = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
    if weighted:
        # HOST LISTS ACROSS THE BOUNDARY, not pointers: `archive/plans/UNWIRED.md:31`
        # records that a pointer from `enqueue_create_host_buffer` is not
        # interchangeable with an arbitrary host pointer on this stack and
        # that the failure is SILENT. The copy is `n_queries * k` floats,
        # the size of the answer the caller is already receiving.
        var d_in = List[Float32](capacity=n_queries * k)
        for i in range(n_queries * k):
            d_in.append(h_dist.unsafe_ptr().unsafe_load(i))
        var wl = host_distance_weights(d_in, n_queries, k)
        var h_w = ctx.enqueue_create_host_buffer[DType.float32](
            n_queries * k
        )
        ctx.synchronize()
        for i in range(n_queries * k):
            h_w.unsafe_ptr().unsafe_store(i, wl[i])
        ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
        ctx.synchronize()
        if trace.enabled:
            trace.record_host(
                "knn_reg.weights", h_w.unsafe_ptr(), n_queries * k
            )
        _ = h_w^

    var d_idx = ctx.enqueue_create_buffer[DType.uint32](n_queries * k)
    var y = List[DeviceBuffer[DType.float32]]()
    for i in range(n_outputs):
        y.append(ctx.enqueue_create_buffer[DType.float32](n_index))
    var out = ctx.enqueue_create_buffer[DType.float32](n_queries * n_outputs)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=h_idx.unsafe_ptr())
    for i in range(n_outputs):
        ctx.enqueue_copy(
            dst_buf=y[i], src_ptr=y_ptr.unsafe_offset(i * n_index)
        )
    ctx.synchronize()

    knn_regress(
        ctx, trace, out, d_idx, y, n_index, n_queries, k, d_w, weighted
    )

    var h = ctx.enqueue_create_host_buffer[DType.float32](n_queries * n_outputs)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=out)
    ctx.synchronize()
    for j in range(n_queries * n_outputs):
        out_ptr.unsafe_store(j, h.unsafe_ptr().unsafe_load(j))

    _ = h^
    _ = out^
    _ = d_w^
    _ = h_dist^
    _ = h_idx^
    _ = d_idx^
    _ = y^
    return used_tile


# ---------------------------------------------------------------------------
# RADIUS NEIGHBOURS
#
# The bullet at the top of this file used to read "`radius_neighbors`.
# `ball_cover` already does radius search for DBSCAN and is the obvious
# substrate, but it is a different call and is not wired here." It is wired
# here now, and the bullet has been rewritten rather than left to rot.
#
# THE TWO-CALL PROTOCOL, AND WHY IT IS NOT AN ACCIDENT.
#
# A radius query's output size is not a function of its inputs. `nnz` is
# discovered by running the search, so a caller in a language that allocates
# its own output buffers cannot allocate them before the first call. cuML has
# the same problem and solves it the same way (`algo.cuh:137-162`: `eps_nn`
# with a null `ja` to count, then again with a sized one to fill), and this
# file follows it because the alternative is for Mojo to hand Python a pointer
# it must later be told to free.
#
# THE COST OF THAT CHOICE IS AN INDEX BUILT TWICE, and it is written down here
# rather than hidden. `rbc_build_index` depends only on `(x, m, n_cols,
# n_landmarks, seed)` and never on the query, so DBSCAN builds it once and
# queries it per batch (`dbscan/impl/dbscan/runner.mojo:361`). This surface
# cannot, because nothing survives between the two Python calls. Making it
# survive means a fitted device handle, which this tree does not have anywhere
# yet, and introducing the first one to save a build on the first radius
# surface is the wrong order to do those two things in. The build is
# `O(sqrt(m))` landmarks against `m` points; the query is the part that scales
# with the neighbourhood. A reusable fitted device handle remains future work
# tracked in `ROADMAP.md` when it becomes a project priority.
#
# WHAT IS DELIBERATELY NOT HERE. `max_k` truncation. `rbc_eps_nn_query_max_k`
# keeps the FIRST `max_k` hits in emission order, so a truncated row keeps a
# DIFFERENT SUBSET on a 64-lane column (IDENTITY_PATHS row 61); under
# IDENTICAL it already refuses by name. A surface that returns every neighbour
# inside the radius has no reason to reach for it, so it does not.
# ---------------------------------------------------------------------------


def _rbc_index_and_count(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut queries: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut x_reordered: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut adj_ia: DeviceBuffer[DType.int32],
    mut vd: DeviceBuffer[DType.int32],
    n_index: Int,
    n_queries: Int,
    n_features: Int,
    n_landmarks: Int,
    radius: Float32,
    metric: Int = RBC_METRIC_DEFAULT,
    metric_arg: Float32 = Float32(2.0),
) raises -> Int:
    """Build the ball cover over `x`, then count `queries`' eps neighbours.

    The six index buffers are the caller's because both public entry points
    below need them and the fill path needs them to outlive the count.
    """
    var landmark_ids = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    var slot_cols = ctx.enqueue_create_buffer[DType.int32](n_index)
    var slot_dists = ctx.enqueue_create_buffer[DType.float32](n_index)
    var nearest = ctx.enqueue_create_buffer[DType.int32](n_index)
    var nearest_dist = ctx.enqueue_create_buffer[DType.float32](n_index)
    var counts = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    ctx.synchronize()

    rbc_build_index(
        ctx,
        x,
        r,
        x_reordered,
        landmark_ids,
        slot_cols,
        slot_dists,
        nearest,
        nearest_dist,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        counts,
        n_index,
        n_features,
        n_landmarks,
        UInt64(12345),
        metric,
        metric_arg,
    )

    # `eps` IS THE RADIUS, NOT ITS SQUARE. The kernel squares it internally
    # (`registers.mojo:253`); passing the square here silently widens every
    # neighbourhood, which is the trap `runner.mojo:391-396` warns about.
    var nnz = rbc_eps_nn_query_count(
        ctx,
        x_reordered,
        queries,
        r,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        adj_ia,
        vd,
        n_queries,
        n_features,
        n_landmarks,
        radius,
        metric,
        metric_arg,
    )
    _ = landmark_ids^
    _ = slot_cols^
    _ = slot_dists^
    _ = nearest^
    _ = nearest_dist^
    _ = counts^
    return nnz


def _radius_check_shapes(
    n_index: Int, n_queries: Int, n_features: Int, radius: Float32, who: String
) raises:
    """Every shape the ball cover cannot serve, refused by name.

    Same discipline as `knn_search`: a clamp would return a wrong answer
    quietly, and this repository has paid for that more than once.
    """
    if n_index <= 0:
        raise Error(who + ": n_index must be positive, got " + String(n_index))
    if n_queries <= 0:
        raise Error(
            who + ": n_queries must be positive, got " + String(n_queries)
        )
    if n_features <= 0:
        raise Error(
            who + ": n_features must be positive, got " + String(n_features)
        )
    if not (radius > Float32(0.0)):
        raise Error(
            who
            + ": radius must be positive and finite, got "
            + String(radius)
            + ". A radius of zero returns each query's exact duplicates only,"
            " which the index is not built to answer, and a negative or NaN"
            " radius has no neighbourhood at all."
        )


def radius_neighbors_count(
    ctx: DeviceContext,
    index_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_index: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    radius: Float32,
    out_indptr_ptr: MutPointer[Int32, MutUntrackedOrigin],
    metric: Int = RBC_METRIC_DEFAULT,
    metric_arg: Float32 = Float32(2.0),
) raises -> Int:
    """Pass one: how many neighbours, and where each query's row begins.

    Writes `n_queries + 1` int32 into `out_indptr_ptr`, the CSR row starts,
    and returns the total edge count. The caller sizes its index and distance
    arrays from the return value and calls `radius_neighbors_fill`.
    """
    _radius_check_shapes(
        n_index, n_queries, n_features, radius, "radius_neighbors_count"
    )
    # DEVIATION 564: refuse a non-metric BEFORE anything is allocated or
    # uploaded. The message names the triangle inequality, because a caller
    # told only "refused" will reasonably read it as unported work.
    rbc_validate_metric(metric, metric_arg)
    var n_landmarks = rbc_n_landmarks(n_index)

    var x = ctx.enqueue_create_buffer[DType.float32](n_index * n_features)
    var queries = ctx.enqueue_create_buffer[DType.float32](
        n_queries * n_features
    )
    var r = ctx.enqueue_create_buffer[DType.float32](n_landmarks * n_features)
    var x_reordered = ctx.enqueue_create_buffer[DType.float32](
        n_index * n_features
    )
    var r_indptr = ctx.enqueue_create_buffer[DType.int32](n_landmarks + 1)
    var r_1nn_cols = ctx.enqueue_create_buffer[DType.int32](n_index)
    var r_1nn_dists = ctx.enqueue_create_buffer[DType.float32](n_index)
    var r_radius = ctx.enqueue_create_buffer[DType.float32](n_landmarks)
    var adj_ia = ctx.enqueue_create_buffer[DType.int32](n_queries + 1)
    var vd = ctx.enqueue_create_buffer[DType.int32](n_queries + 1)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=index_ptr)
    ctx.enqueue_copy(dst_buf=queries, src_ptr=queries_ptr)
    ctx.synchronize()

    var nnz = _rbc_index_and_count(
        ctx,
        x,
        queries,
        r,
        x_reordered,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        adj_ia,
        vd,
        n_index,
        n_queries,
        n_features,
        n_landmarks,
        radius,
        metric,
        metric_arg,
    )
    ctx.enqueue_copy(dst_ptr=out_indptr_ptr, src_buf=adj_ia)
    ctx.synchronize()
    _ = x^
    _ = queries^
    _ = r^
    _ = x_reordered^
    _ = r_indptr^
    _ = r_1nn_cols^
    _ = r_1nn_dists^
    _ = r_radius^
    _ = adj_ia^
    _ = vd^
    return nnz


def radius_neighbors_fill(
    ctx: DeviceContext,
    index_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_index: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    radius: Float32,
    out_indptr_ptr: MutPointer[Int32, MutUntrackedOrigin],
    out_idx_ptr: MutPointer[Int32, MutUntrackedOrigin],
    out_dist_ptr: MutPointer[Float32, MutUntrackedOrigin],
    nnz_capacity: Int,
    return_sqrt: Bool = True,
    metric: Int = RBC_METRIC_DEFAULT,
    metric_arg: Float32 = Float32(2.0),
) raises -> Int:
    """Pass two: the columns and the distances, in CSR order.

    `nnz_capacity` is what pass one returned and what the caller allocated.
    A larger count here means the data changed between the two calls, which
    is refused rather than truncated: a short read would be a wrong answer
    that looked like a right one.

    Distances are recomputed from the finished CSR
    (`neighbors/checks/radius_distances.mojo`), never stored by the search
    kernel; that file carries the reasoning and the bit-equality argument.
    """
    _radius_check_shapes(
        n_index, n_queries, n_features, radius, "radius_neighbors_fill"
    )
    rbc_validate_metric(metric, metric_arg)  # DEVIATION 564
    if nnz_capacity < 0:
        raise Error(
            "radius_neighbors_fill: nnz_capacity must not be negative, got "
            + String(nnz_capacity)
        )
    var n_landmarks = rbc_n_landmarks(n_index)

    var x = ctx.enqueue_create_buffer[DType.float32](n_index * n_features)
    var queries = ctx.enqueue_create_buffer[DType.float32](
        n_queries * n_features
    )
    var r = ctx.enqueue_create_buffer[DType.float32](n_landmarks * n_features)
    var x_reordered = ctx.enqueue_create_buffer[DType.float32](
        n_index * n_features
    )
    var r_indptr = ctx.enqueue_create_buffer[DType.int32](n_landmarks + 1)
    var r_1nn_cols = ctx.enqueue_create_buffer[DType.int32](n_index)
    var r_1nn_dists = ctx.enqueue_create_buffer[DType.float32](n_index)
    var r_radius = ctx.enqueue_create_buffer[DType.float32](n_landmarks)
    var adj_ia = ctx.enqueue_create_buffer[DType.int32](n_queries + 1)
    var vd = ctx.enqueue_create_buffer[DType.int32](n_queries + 1)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=index_ptr)
    ctx.enqueue_copy(dst_buf=queries, src_ptr=queries_ptr)
    ctx.synchronize()

    var nnz = _rbc_index_and_count(
        ctx,
        x,
        queries,
        r,
        x_reordered,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        adj_ia,
        vd,
        n_index,
        n_queries,
        n_features,
        n_landmarks,
        radius,
        metric,
        metric_arg,
    )
    if nnz > nnz_capacity:
        raise Error(
            "radius_neighbors_fill: the search found "
            + String(nnz)
            + " edges and the caller allocated for "
            + String(nnz_capacity)
            + ". The two calls saw different data. Re-run"
            " radius_neighbors_count against the arrays this call was given"
            " rather than truncating, which would return a subset that looks"
            " like a complete answer."
        )

    if nnz > 0:
        var adj_ja = ctx.enqueue_create_buffer[DType.int32](nnz)
        var out_dist = ctx.enqueue_create_buffer[DType.float32](nnz)
        ctx.synchronize()
        # Under IDENTICAL this call also canonicalizes each row's column
        # order (DEVIATION 551), which is what makes the CSR below the same
        # bytes on every vendor.
        rbc_eps_nn_query_fill(
            ctx,
            x_reordered,
            queries,
            r,
            r_indptr,
            r_1nn_cols,
            r_1nn_dists,
            r_radius,
            adj_ia,
            adj_ja,
            n_queries,
            n_features,
            n_landmarks,
            radius,
            metric,
            metric_arg,
        )
        # THE SAME METRIC THE SEARCH USED, and it is not optional: the
        # returned distances have to be in the metric the caller asked for,
        # and a mismatch here would hand back Euclidean numbers beside an L1
        # neighbour list -- plausible, wrong, and invisible.
        rbc_edge_distances(
            ctx,
            adj_ia,
            adj_ja,
            queries,
            x,
            out_dist,
            n_queries,
            n_features,
            nnz,
            nnz,
            return_sqrt,
            metric,
            metric_arg,
        )
        ctx.enqueue_copy(dst_ptr=out_idx_ptr, src_buf=adj_ja)
        ctx.enqueue_copy(dst_ptr=out_dist_ptr, src_buf=out_dist)
        ctx.synchronize()
        _ = adj_ja^
        _ = out_dist^

    ctx.enqueue_copy(dst_ptr=out_indptr_ptr, src_buf=adj_ia)
    ctx.synchronize()
    _ = x^
    _ = queries^
    _ = r^
    _ = x_reordered^
    _ = r_indptr^
    _ = r_1nn_cols^
    _ = r_1nn_dists^
    _ = r_radius^
    _ = adj_ia^
    _ = vd^
    return nnz


# ---------------------------------------------------------------------------
# k-NEAREST NEIGHBOURS OVER THE BALL COVER
#
# The INDEXED k-NN entry point. `knn_search` above is exact brute force and
# stays the default; this is the same answer computed by pruning instead of
# by comparing every pair, and it is what a caller asking for a spatial index
# actually wants. `neighbors/impl/neighbors/ball_cover/knn.mojo` carries the
# bounds and their proofs, and `NOT_IMPLEMENTED.tsv`'s kd-tree row records why
# a tree is not the structure that answers that request on a GPU.
#
# ONE CALL, NOT TWO. The radius surface above needs two boundary crossings
# because a radius query's output size is discovered by running it. A k-NN
# query's output size is `n_queries * k` and is known before the call, so
# there is one entry point here and the caller allocates up front.
#
# THE INDEX IS STILL BUILT PER CALL, for the same reason the radius surface
# gives: nothing in this library holds a fitted device handle yet. The build
# is O(m * sqrt(m)) against a query that is the part this exists to make
# sublinear, so a caller doing ONE query batch against a fresh index pays
# roughly what brute force costs and gains nothing; the win is a large query
# batch, and it is a large win only when the cover prunes. `out_n_dists` is
# returned so the caller can SEE how much it pruned rather than assume.
# ---------------------------------------------------------------------------


def rbc_knn_search(
    ctx: DeviceContext,
    index_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_index: Int,
    queries_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_features: Int,
    k: Int,
    out_idx_ptr: MutPointer[Int32, MutUntrackedOrigin],
    out_dist_ptr: MutPointer[Float32, MutUntrackedOrigin],
    metric: Int = RBC_METRIC_DEFAULT,
    metric_arg: Float32 = Float32(2.0),
) raises -> Int:
    """Exact k-NN over a random ball cover built on `index_ptr`.

    Writes `n_queries * k` int32 into `out_idx_ptr` (original index-point
    ids, `-1` for an unfilled slot) and `n_queries * k` float32 into
    `out_dist_ptr` (TRUE distances in `metric`), both row-major and ordered
    by `(comparison-space distance, index)` ascending, which is the total
    order stated at the top of `impl/neighbors/ball_cover/knn.mojo`.

    Returns the TOTAL number of candidate distances the query actually
    computed. Brute force over the same shapes would compute
    `n_queries * n_index`, so the ratio of the two is the pruning this call
    achieved, and it is returned rather than printed because a caller and a
    benchmark both want it.
    """
    _radius_check_shapes(
        n_index, n_queries, n_features, Float32(1.0), "rbc_knn_search"
    )
    rbc_validate_metric(metric, metric_arg)  # DEVIATION 564
    if k < 1:
        raise Error("rbc_knn_search: k must be at least 1, got " + String(k))
    if k > n_index:
        raise Error(
            "rbc_knn_search: k = "
            + String(k)
            + " exceeds the "
            + String(n_index)
            + " points in the index. Refused rather than padded: a padded"
            " answer is indistinguishable from a complete one."
        )
    if k > RBC_KNN_MAX_K:
        raise Error(
            "rbc_knn_search: k = "
            + String(k)
            + " exceeds RBC_KNN_MAX_K = "
            + String(RBC_KNN_MAX_K)
            + ". Use knn_search, whose selector is sized per launch."
        )
    var n_landmarks = rbc_n_landmarks(n_index)

    var x = ctx.enqueue_create_buffer[DType.float32](n_index * n_features)
    var queries = ctx.enqueue_create_buffer[DType.float32](
        n_queries * n_features
    )
    var r = ctx.enqueue_create_buffer[DType.float32](n_landmarks * n_features)
    var x_reordered = ctx.enqueue_create_buffer[DType.float32](
        n_index * n_features
    )
    var landmark_ids = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    var slot_cols = ctx.enqueue_create_buffer[DType.int32](n_index)
    var slot_dists = ctx.enqueue_create_buffer[DType.float32](n_index)
    var nearest = ctx.enqueue_create_buffer[DType.int32](n_index)
    var nearest_dist = ctx.enqueue_create_buffer[DType.float32](n_index)
    var r_indptr = ctx.enqueue_create_buffer[DType.int32](n_landmarks + 1)
    var r_1nn_cols = ctx.enqueue_create_buffer[DType.int32](n_index)
    var r_1nn_dists = ctx.enqueue_create_buffer[DType.float32](n_index)
    var r_radius = ctx.enqueue_create_buffer[DType.float32](n_landmarks)
    var counts = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    var out_inds = ctx.enqueue_create_buffer[DType.int32](n_queries * k)
    var out_dists = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
    var dist_count = ctx.enqueue_create_buffer[DType.int32](n_queries)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=index_ptr)
    ctx.enqueue_copy(dst_buf=queries, src_ptr=queries_ptr)
    ctx.synchronize()

    rbc_build_index(
        ctx,
        x,
        r,
        x_reordered,
        landmark_ids,
        slot_cols,
        slot_dists,
        nearest,
        nearest_dist,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        counts,
        n_index,
        n_features,
        n_landmarks,
        UInt64(12345),
        metric,
        metric_arg,
    )
    rbc_knn_query(
        ctx,
        x_reordered,
        queries,
        r,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        out_inds,
        out_dists,
        dist_count,
        n_queries,
        n_features,
        n_landmarks,
        k,
        metric,
        metric_arg,
    )
    ctx.enqueue_copy(dst_ptr=out_idx_ptr, src_buf=out_inds)
    ctx.enqueue_copy(dst_ptr=out_dist_ptr, src_buf=out_dists)
    ctx.synchronize()

    var hc = ctx.enqueue_create_host_buffer[DType.int32](n_queries)
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=dist_count)
    ctx.synchronize()
    var total = 0
    for i in range(n_queries):
        total += Int(hc.unsafe_ptr().unsafe_load(i))

    _ = x^
    _ = queries^
    _ = r^
    _ = x_reordered^
    _ = landmark_ids^
    _ = slot_cols^
    _ = slot_dists^
    _ = nearest^
    _ = nearest_dist^
    _ = r_indptr^
    _ = r_1nn_cols^
    _ = r_1nn_dists^
    _ = r_radius^
    _ = counts^
    _ = out_inds^
    _ = out_dists^
    _ = dist_count^
    _ = hc^
    return total
