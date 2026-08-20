"""The callable surface over the brute-force k-NN kernel.

**Why this file exists.** Before it, `neighbors/` held a verified kernel, a
benchmark that timed it, and checks that proved it correct -- and no way for
anyone outside this repository to call any of it. `neighbors/__init__.mojo`
was empty and every entry point under `neighbors/` was a `*_main.mojo` driver
or a `mojo_only/*_check.mojo` verifier. Five algorithms measured, zero
reachable.

Nothing here is a port. `neighbors/gbdt/` mirrors cuVS and is governed by
COPY, DO NOT IMPROVE; this file is host-side policy that cuVS does not have a
counterpart for, in the same category as `mojo_only/`. Every choice it makes
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

- `radius_neighbors`. `ball_cover` already does radius search for DBSCAN and
  is the obvious substrate, but it is a different call and is not wired here.
- Metrics other than expanded L2. The ported kernel carries only that arm
  (`knn_brute_force.mojo`'s dispatch note), so cosine and L1 are a port, not
  a flag.
- `KNeighborsClassifier` / `KNeighborsRegressor`. A vote and a mean over what
  `knn_search` already returns.
- **A CPython extension module.** mojotrees' `bindings/` shows the shape and
  it has shipped a wheel, but nothing here is importable from Python yet.

THE REPRODUCIBILITY LIMITATION, WHICH IS REAL AND IS NOT HIDDEN
---------------------------------------------------------------

`UNWIRED.md:371`: RAFT places k-NN output with `atomicAdd` and has no index
tie-break, so **which of several equidistant neighbours is returned is not
reproducible**. Distances are stable; the identity of a tied neighbour is
not. Any caller building a bit-identity claim on top of this must know that,
and an `IDENTICAL` column cannot cover k-NN indices until the upstream is
fixed -- which would be an improvement on it, not a port of it.
"""

from max.gpu.host import DeviceContext

from neighbors.ported.neighbors.detail.knn_brute_force import (
    KNN_METHOD_AUTO,
    brute_force_knn_impl,
    compute_norms,
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
) raises -> Int:
    """Exact k nearest neighbours, index and queries row-major on the host.

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

    # L2 wants the SQUARED norm on both sides; `take_sqrt` here is about the
    # norm, not about the distances the caller gets back. That is the
    # upstream's comment at `knn_brute_force.cuh:117-118`.
    compute_norms(ctx, index, index_norm, n_index, n_features, False)
    compute_norms(ctx, queries, query_norm, n_queries, n_features, False)
    ctx.synchronize()

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
    )
    ctx.synchronize()

    # Device -> pinned host buffer -> the caller's memory. The second hop is
    # not decoration: `UNWIRED.md:31` records that a pointer from
    # `enqueue_create_host_buffer` is not interchangeable with an arbitrary
    # host pointer on this stack, and the failure is SILENT. Copying through
    # a buffer the runtime made keeps this on the route the checks exercise.
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
    # the set. It cannot repair `UNWIRED.md:371`, which is about WHICH of
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

    for i in range(n_queries * k):
        out_dist_ptr.unsafe_store(i, hd.unsafe_ptr().unsafe_load(i))
        out_idx_ptr.unsafe_store(i, hi.unsafe_ptr().unsafe_load(i))

    return query_tile
