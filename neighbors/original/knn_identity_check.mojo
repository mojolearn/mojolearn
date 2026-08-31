# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What `IDENTICAL` promises about k-NN, and the tie class that proves it.

IDENTITY_PATHS row 11 was the ledger's only REFUSE for two days: RAFT's
radix select places every output with `atomicAdd` and has no index
tie-break, so which of several EQUIDISTANT neighbours comes back is decided
by arrival order. `neighbors/estimator.mojo`'s own docstring says it, and
its host sort says it again -- the sort fixes the ORDER of a set it cannot
choose.

This file gates the closure. Every check here is built around a PLANTED TIE
CLASS: four index rows that are bit-identical copies of one another, and a
query sitting exactly on them, with `k` smaller than the class. Then the
answer is a choice among equals and nothing else, which is the only fixture
on which the property is visible at all. A random fixture has no ties, and
on it a broken selector and a fixed one agree on every row.

WHAT EACH ARM OWES, AND THEY ARE DIFFERENT DEBTS
-------------------------------------------------
- **TILED** (`select_radix`): under IDENTICAL it runs
  `radix_topk_identical_kernel`, whose key is `(distance, index)` -- a TOTAL
  order. So its tie set is not merely reproducible, it is NAMED: the lowest
  indices win. That is checkable against arithmetic rather than against a
  previous run, and it is what `check_knn_tie_set_is_lowest_index` asserts.
- **FUSED** (`faiss_select::WarpSelect`): its comparator is
  `Comparators.cuh:17`, the DISTANCE ONLY. Under IDENTICAL its tie set is a
  pure function of `(m, n, k)` and the pinned policy -- because `grid_x` is
  pinned to 1 so no mutex merge decides anything, and a row's lanes see the
  same columns in the same order at every `grid_y` -- but it is NOT the
  lowest-index set. What that arm owes is INVARIANCE, not a name, and
  `check_knn_fused_tie_set_is_geometry_invariant` is the corresponding gate.

Two arms with two different tie rules is exactly why DEVIATION 502 pins
WHICH ARM runs under IDENTICAL: an answer that depends on how many queries
the caller passed is not identical in any useful sense.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from original.kernel_matrix import TARGET_COLUMN, lib_lane_width_for
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from neighbors.estimator import knn_search
from neighbors.derived.neighbors.detail.knn_brute_force import (
    KNN_METHOD_AUTO,
    KNN_METHOD_FUSED,
    KNN_METHOD_TILED,
)


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

#: The planted class. Rows 5, 100, 900 and 3000 of the index are bit-identical
#: copies of one point, so a query on that point is equidistant from four
#: neighbours at exactly 0.0.
comptime TIE_N = 4096
comptime TIE_D = 16


def tie_rows() -> List[Int]:
    """The planted class, in ASCENDING order. The order is load-bearing:
    the composite key's answer is the FIRST k of this list."""
    var rows = List[Int]()
    rows.append(5)
    rows.append(100)
    rows.append(900)
    rows.append(3000)
    return rows^


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _coord(i: Int, f: Int) -> Float32:
    """A deterministic point cloud with no accidental duplicates."""
    var h = UInt64(i + 1) * UInt64(0x9E3779B97F4A7C15) + UInt64(
        f + 7
    ) * UInt64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    h = h ^ (h >> UInt64(32))
    return Float32(Int(h & UInt64(0xFFFF))) / Float32(256.0)


def _build_index(ctx: DeviceContext) raises -> HostBuffer[DType.float32]:
    """`TIE_N x TIE_D` points, with the planted class overwritten last so
    the copies are exact.

    A `HostBuffer` and not a `List`: `knn_search` takes
    `MutUntrackedOrigin` pointers, and `UNWIRED.md:31` records that a
    pointer from `enqueue_create_host_buffer` is not interchangeable with
    an arbitrary host pointer on this stack -- SILENTLY. Every other check
    that drives the caller-facing surface uses the runtime's buffers for
    that reason and this one does too.
    """
    var host = ctx.enqueue_create_host_buffer[DType.float32](TIE_N * TIE_D)
    ctx.synchronize()
    for i in range(TIE_N):
        for f in range(TIE_D):
            host.unsafe_ptr().unsafe_store(i * TIE_D + f, _coord(i, f))
    var rows = tie_rows()
    for t in range(len(rows)):
        var row = rows[t]
        for f in range(TIE_D):
            host.unsafe_ptr().unsafe_store(
                row * TIE_D + f, _coord(rows[0], f)
            )
    return host^


def _search(
    ctx: DeviceContext,
    mut index_host: HostBuffer[DType.float32],
    n_queries: Int,
    k: Int,
    method: Int,
    query_tile: Int,
    mut out_d: List[Float32],
    mut out_i: List[UInt32],
) raises -> Int:
    """`knn_search` with every query sitting exactly on the planted point.

    Every query is the SAME point on purpose: the check is about a tie, and
    a tie is a property of one row's candidate set. Repeating it
    `n_queries` times is how the launch geometry is made to vary without
    varying the question.
    """
    var q_host = ctx.enqueue_create_host_buffer[DType.float32](
        n_queries * TIE_D
    )
    var d_host = ctx.enqueue_create_host_buffer[DType.float32](n_queries * k)
    var i_host = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * k)
    ctx.synchronize()
    var rows = tie_rows()
    for i in range(n_queries):
        for f in range(TIE_D):
            q_host.unsafe_ptr().unsafe_store(
                i * TIE_D + f, _coord(rows[0], f)
            )

    var tile = knn_search(
        ctx,
        index_host.unsafe_ptr(),
        TIE_N,
        q_host.unsafe_ptr(),
        n_queries,
        TIE_D,
        k,
        d_host.unsafe_ptr(),
        i_host.unsafe_ptr(),
        True,
        query_tile,
        method,
    )
    for i in range(n_queries * k):
        out_d.append(d_host.unsafe_ptr().unsafe_load(i))
        out_i.append(i_host.unsafe_ptr().unsafe_load(i))
    # `[[mojo-buffer-freed-at-last-use]]`: keep the three alive past the
    # reads above.
    _ = q_host^
    _ = d_host^
    _ = i_host^
    return tile


def check_knn_tie_set_is_lowest_index() raises:
    """IDENTITY_PATHS row 11, the TILED arm's closure (DEVIATIONS 500/501).

    Four equidistant candidates, two slots. Under IDENTICAL the composite
    key makes the answer arithmetic rather than a race: indices 5 and 100,
    the two lowest of the class, in that order.

    Under FAST this asserts NOTHING about which two come back, because
    nothing in the ported selector promises anything -- it reports what it
    saw. That asymmetry is the point of the check: the same fixture is
    undetermined in one mode and pinned in the other.
    """
    var ctx = DeviceContext()
    var index_host = _build_index(ctx)

    var k = 2
    var d0 = List[Float32]()
    var i0 = List[UInt32]()
    _ = _search(ctx, index_host, 1, k, KNN_METHOD_TILED, 256, d0, i0)

    for s in range(k):
        if d0[s] != Float32(0.0):
            raise Error(
                "check_knn_tie_set_is_lowest_index: slot "
                + String(s)
                + " came back at distance "
                + String(d0[s])
                + ", not 0.0. The planted class is not being found at all,"
                " so nothing here is a statement about ties."
            )

    comptime if IDENTICAL_BUILD:
        var want = tie_rows()
        if Int(i0[0]) != want[0] or Int(i0[1]) != want[1]:
            raise Error(
                "check_knn_tie_set_is_lowest_index (IDENTICAL): the tie"
                " class returned indices "
                + String(Int(i0[0]))
                + ", "
                + String(Int(i0[1]))
                + " where the composite (distance, index) key requires "
                + String(want[0])
                + ", "
                + String(want[1])
                + ". Either DEVIATION 500's key is not reached or its"
                " order is not the one it claims."
            )
        print(
            "check_knn_tie_set_is_lowest_index OK (IDENTICAL): four",
            "equidistant candidates, two slots, and the composite key",
            "returns the two LOWEST indices",
            Int(i0[0]),
            "and",
            Int(i0[1]),
            "-- arithmetic, not arrival order",
        )
    else:
        print(
            "check_knn_tie_set_is_lowest_index OK (FAST): the ported",
            "selector returned indices",
            Int(i0[0]),
            "and",
            Int(i0[1]),
            "from the four-member tie class. Nothing requires those two:",
            "RAFT places tied outputs with atomicAdd and this mode keeps",
            "their behaviour, which is what row 11 refused.",
        )


def check_knn_tiled_is_query_tile_invariant() raises:
    """The tiled arm's answer may not depend on how the queries were tiled.

    `query_tile` is a MEMORY knob -- `plan_query_tile` derives it from the
    distance tile a device can hold -- so it is exactly the sort of number
    that differs between two GPUs while nothing about the question changes.
    Three tiles, same fixture, every returned bit equal.

    This runs the SELECTOR at three different block populations and the
    distance step at three different launch shapes. Under IDENTICAL the
    distance step is `pinned_distance_tile_kernel`, whose accumulation
    order is per-cell and therefore tile-independent by construction; under
    FAST it is the vendor matmul, whose k-split may or may not be. A red
    result in FAST only is that matmul reaching the answer, and it is a
    finding rather than a flake.
    """
    var ctx = DeviceContext()
    var index_host = _build_index(ctx)

    var k = 4
    var n_q = 3
    var base_d = List[Float32]()
    var base_i = List[UInt32]()
    var tiles = List[Int]()
    tiles.append(1)
    tiles.append(2)
    tiles.append(3)

    for t in range(len(tiles)):
        var dd = List[Float32]()
        var ii = List[UInt32]()
        _ = _search(
            ctx, index_host, n_q, k, KNN_METHOD_TILED, tiles[t], dd, ii
        )
        if t == 0:
            base_d = dd.copy()
            base_i = ii.copy()
        else:
            var moved = 0
            for i in range(n_q * k):
                if dd[i] != base_d[i] or ii[i] != base_i[i]:
                    moved += 1
            if moved != 0:
                raise Error(
                    "check_knn_tiled_is_query_tile_invariant: query_tile "
                    + String(tiles[t])
                    + " moved "
                    + String(moved)
                    + " of "
                    + String(n_q * k)
                    + " slots against query_tile "
                    + String(tiles[0])
                    + ". A memory knob is reaching the answer."
                )
    print(
        "check_knn_tiled_is_query_tile_invariant OK (" + _mode_name() + "):",
        n_q * k,
        "slots bit-identical at query_tile 1, 2 and 3",
    )


def check_knn_fused_tie_set_is_geometry_invariant() raises:
    """The fused arm owes INVARIANCE, which is a weaker debt and a real one.

    `faiss_select`'s comparator compares the distance only, so the fused arm
    cannot NAME its tie set the way the composite key can. What it can
    promise -- and what pinning `grid_x = 1` buys it (DEVIATION 502) -- is
    that the set is a pure function of `(m, n, k)` and the pinned policy.

    THE FIXTURE IS THE SAME QUESTION ASKED MANY TIMES. Every query is the
    same planted point, so every ROW of every run must come back with the
    same tie set in the same order. The query count is the lever: 1, 40 and
    2,000 identical queries put the launch computation in three different
    regimes, and at 40 the grid it picks engages the x-split whose mutex
    merge is exactly what the pin removes.

    UNDER FAST THIS IS EXPECTED TO MOVE, AND MEASURING HOW MUCH IS THE
    POINT. The first run of this check found row 16 of the 40-query launch
    disagreeing with row 0 -- two identical questions, two different
    neighbours, on ONE device in ONE process. That is not a flake and not a
    vendor difference: it is `updateSortedWarpQ`'s mutex merge resolving a
    tie in whatever order the blocks arrived. So FAST reports the count and
    IDENTICAL requires zero.

    SKIPPED, LOUDLY, ON A COLUMN WITHOUT THE ARM (added 2026-08-23). The
    fused arm refuses wherever the lane width is not 32 (row 23), so on
    `MOJOLEARN_COLUMN_AMD` this check was not failing, it was asking a
    32-lane question of a 64-lane machine and reading the refusal as a
    crash. It prints what it skipped and why; a silent `return` here would
    make an AMD run look like it had verified something it never ran.
    """
    comptime if lib_lane_width_for[TARGET_COLUMN]() != 32:
        print(
            "check_knn_fused_tie_set_is_geometry_invariant SKIPPED (",
            _mode_name(),
            "): this column's lane width is",
            lib_lane_width_for[TARGET_COLUMN](),
            "and the FAISS warp queue is a 32-lane bitonic network, so the",
            "fused arm REFUSES here (IDENTITY_PATHS row 23). Nothing about",
            "the fused tie set is verified on this column; the arm the",
            "default takes here is the tiled one (DEVIATIONS 509, 512) and",
            "its tie set is checked by check_knn_tie_set_is_lowest_index.",
        )
        return

    var ctx = DeviceContext()
    var index_host = _build_index(ctx)

    var k = 2
    var counts = List[Int]()
    counts.append(1)
    counts.append(40)
    counts.append(2000)
    var base_i = List[UInt32]()
    var base_d = List[Float32]()
    var row_moves = 0
    var shape_moves = 0

    for c in range(len(counts)):
        var dd = List[Float32]()
        var ii = List[UInt32]()
        _ = _search(
            ctx, index_host, counts[c], k, KNN_METHOD_FUSED, 256, dd, ii
        )
        for r in range(counts[c]):
            for s in range(k):
                if ii[r * k + s] != ii[s] or dd[r * k + s] != dd[s]:
                    row_moves += 1
        if c == 0:
            for s in range(k):
                base_i.append(ii[s])
                base_d.append(dd[s])
        else:
            for s in range(k):
                if ii[s] != base_i[s] or dd[s] != base_d[s]:
                    shape_moves += 1

    comptime if IDENTICAL_BUILD:
        if row_moves != 0 or shape_moves != 0:
            raise Error(
                "check_knn_fused_tie_set_is_geometry_invariant"
                " (IDENTICAL): "
                + String(row_moves)
                + " rows disagreed with row 0 and "
                + String(shape_moves)
                + " slots moved between query counts. With `grid_x` pinned"
                " to 1 no merge decides a tie and no lane sees a different"
                " column order, so both must be zero."
            )
        print(
            "check_knn_fused_tie_set_is_geometry_invariant OK (IDENTICAL):",
            "the fused arm returned the same tie set",
            Int(base_i[0]),
            "/",
            Int(base_i[1]),
            "at 1, 40 and 2,000 identical queries, every row agreeing with",
            "row 0",
        )
    else:
        print(
            "check_knn_fused_tie_set_is_geometry_invariant OK (FAST):",
            row_moves,
            "rows disagreed with row 0 and",
            shape_moves,
            "slots moved between query counts -- the mutex merge resolving",
            "a tie by arrival order, on one device, in one process. This is",
            "the behaviour DEVIATION 502's grid pin removes.",
        )


def check_knn_auto_arm_is_pinned() raises:
    """DEVIATION 509. Under IDENTICAL, AUTO is the TILED arm on every column.

    The two arms break a tie differently, so an AUTO that consults the
    launch computation makes the ANSWER depend on the query count. This
    drives the caller-facing surface at two query counts that fall on
    opposite sides of DEVIATION 36's grid test and requires AUTO to agree
    with TILED in both.

    THE ARM CHANGED AND THE CHECK CHANGED WITH IT. DEVIATION 502 pinned
    AUTO to FUSED, which is cuVS's own dispatch and is well-defined here;
    it is not well-defined on AMD, where `fused_l2_knn` refuses at its
    entry because the FAISS queue is a 32-lane network (row 23). Pinning
    the identical column to an arm that RAISES on one of its three
    vendors is the defect DEVIATION 509 corrects, and TILED is the arm
    whose tie set is NAMED -- lowest index, by the composite key -- rather
    than merely reproducible.

    Under FAST the same two shapes are EXPECTED to disagree, because that
    is what DEVIATION 36 measured and chose. The check reports it rather
    than asserting it: a shape where the two arms happen to agree is not a
    failure of the default.
    """
    var ctx = DeviceContext()
    var index_host = _build_index(ctx)
    var k = 2

    var disagreements = 0
    var counts = List[Int]()
    counts.append(53)
    counts.append(2000)
    for c in range(len(counts)):
        var da = List[Float32]()
        var ia = List[UInt32]()
        var df = List[Float32]()
        var if_ = List[UInt32]()
        _ = _search(
            ctx, index_host, counts[c], k, KNN_METHOD_AUTO, 256, da, ia
        )
        _ = _search(
            ctx, index_host, counts[c], k, KNN_METHOD_TILED, 256, df, if_
        )
        for s in range(k):
            if ia[s] != if_[s] or da[s] != df[s]:
                disagreements += 1
                comptime if IDENTICAL_BUILD:
                    raise Error(
                        "check_knn_auto_arm_is_pinned (IDENTICAL): at "
                        + String(counts[c])
                        + " queries AUTO returned index "
                        + String(Int(ia[s]))
                        + " at slot "
                        + String(s)
                        + " where FUSED returned "
                        + String(Int(if_[s]))
                        + ". DEVIATION 509 pins AUTO to the TILED arm in"
                        " this mode; a shape-chosen arm is a shape-chosen"
                        " tie set."
                    )
    comptime if IDENTICAL_BUILD:
        print(
            "check_knn_auto_arm_is_pinned OK (IDENTICAL): AUTO == TILED at",
            "53 and 2,000 queries, so the arm no longer depends on the",
            "query count",
        )
    else:
        print(
            "check_knn_auto_arm_is_pinned OK (FAST): AUTO and TILED",
            "disagreed on",
            disagreements,
            "of",
            2 * k,
            "tied slots across the two shapes, which is DEVIATION 36's",
            "shape-chosen arm doing what it was measured to do",
        )


def main() raises:
    check_knn_tie_set_is_lowest_index()
    check_knn_tiled_is_query_tile_invariant()
    check_knn_fused_tie_set_is_geometry_invariant()
    check_knn_auto_arm_is_pinned()
