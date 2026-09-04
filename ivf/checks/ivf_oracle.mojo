# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host oracles for IVF-FLAT: a float32 serial replay and a float64
reference.

NOT A PORT. cuVS checks IVF-FLAT against a brute force to a RECALL
threshold (`cpp/tests/neighbors/ann_ivf_flat.cuh`), because recall is what
an approximate index is for and one backend is all they ship. We ship
Metal, CUDA and HIP from one source and claim a bit-identical answer, so
the device arm is gated BIT FOR BIT under `IDENTICAL` against the float32
replay below, and the replay is gated against the float64 reference to a
tolerance. Both are written FIRST and gated FIRST.

TWO ORACLES, TWO JOBS
----------------------
`oracle_*_f32`      float32, SERIAL, ASCENDING, through the same helpers
                    the device uses (`identical_mul_add`, `ftz`,
                    `identical_sqrt`), every formula spelled here a SECOND
                    time rather than imported from `ivf/impl/` -- so the
                    gate compares two spellings of one arithmetic and not a
                    function against itself.
`reference_*_f64`   float64, the DIRECT form `sum_f (a_f - b_f)^2` rather
                    than the expanded one, host `std.math`. Its job is
                    tolerance sanity and the known-by-construction claims.
                    The direct form is deliberate: an oracle that used the
                    expanded form would share the cancellation the device
                    arm has and could not catch it.

THE ONE PLACE THE FLOAT32 ORACLE MIRRORS A DEVICE SHAPE
--------------------------------------------------------
The norms. `core/row_norms.mojo::row_norm_kernel` folds with
`pinned_block_sum[NORM_TPB]` -- `NORM_TPB` strided lane partials, then a
halving tree -- NOT a serial sum. `_host_row_norm_halving` replays THAT
shape, exactly as `kde/checks/kde_oracle.mojo` and
`glm/checks/ridge_check.mojo::_host_halving_xty` do for the same
reason: the norms are the k-NN lane's kernel and this lane calls rather
than re-spells it, so the oracle has to replay the shape it calls. Under
`FAST` that fold is `block.sum`, the library's shape, and every comparison
that depends on it is a REPORT.

WHAT THE ORACLE DOES **NOT** REPLAY
-------------------------------------
The COARSE QUANTIZER. There is no host k-means here and there is not going
to be one: `cluster/` owns that fit, `archive/research/UNSUPERVISED_IDENTITY.md` states its
identity status, and a second spelling of Lloyd's algorithm in this file
would be exactly the duplication this lane exists not to commit. The
oracle takes the centroids AS GIVEN and replays everything downstream of
them. That is why `ivf/README.md`'s identity table has hazard 2 as its own
row: the search result is reproducible only if the centroid set is, and
that is a claim this lane GATES (`check_quantizer_is_reproducible`) rather
than proves.
"""

from std.math import sqrt as host_sqrt

from core.row_norms import NORM_TPB
from checks.numerics import ftz, identical_mul_add, identical_sqrt


def _host_row_norm_halving(a: List[Float32], row: Int, d: Int) -> Float32:
    """`row_norm_kernel` at `take_sqrt = 0`, replayed: `NORM_TPB` strided
    lane partials (`acc = ftz(fma(v, v, acc))`), then `pinned_block_sum`'s
    halving tree, then `ftz` of the total."""
    var red = List[Float32]()
    for t in range(NORM_TPB):
        var acc = Float32(0.0)
        var col = t
        while col < d:
            var v = ftz(a[row * d + col])
            acc = ftz(identical_mul_add(v, v, acc))
            col += NORM_TPB
        red.append(acc)
    var step = NORM_TPB // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return ftz(red[0])


def oracle_row_norms(
    a: List[Float32], n_rows: Int, d: Int, take_sqrt: Bool
) -> List[Float32]:
    """`row_norm_kernel` over every row. `take_sqrt` follows the metric,
    exactly as the kernel's own flag does."""
    var out = List[Float32]()
    for r in range(n_rows):
        var total = _host_row_norm_halving(a, r, d)
        if take_sqrt:
            if total <= Float32(0.0):
                total = Float32(0.0)
            total = ftz(identical_sqrt(total))
        out.append(total)
    return out^


def oracle_expanded_distance(
    q: List[Float32],
    qi: Int,
    y: List[Float32],
    yi: Int,
    d: Int,
    q_norm: Float32,
    y_norm: Float32,
    is_sqrt: Bool,
) -> Float32:
    """`pinned_distance_tile_kernel`, one cell, replayed.

    The feature axis ASCENDING through `identical_mul_add` with the running
    partial flushed at every step, then
    `fma(-2, acc, ftz(q_norm + y_norm))`, then their clamp
    (`unfused_distance_nn.cuh:80-81`), then `identical_sqrt` when the
    metric wants the root (DEVIATION 550).

    Under `IDENTICAL` this is the same arithmetic the device performs, in
    the same order, and the comparison is an ASSERTION. Under `FAST` the
    device arm is a vendor matmul plus `core/expand_distances.mojo` and no
    host can replicate it, so it is a REPORT.
    """
    var acc = Float32(0.0)
    for f in range(d):
        var qv = ftz(q[qi * d + f])
        var yv = ftz(y[yi * d + f])
        acc = ftz(identical_mul_add(qv, yv, acc))
    var dist = ftz(
        identical_mul_add(
            Float32(-2.0), acc, ftz(ftz(q_norm) + ftz(y_norm))
        )
    )
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    if is_sqrt:
        dist = ftz(identical_sqrt(dist))
    return dist


def _key_less(
    da: Float32, ia: UInt32, db: Float32, ib: UInt32
) -> Bool:
    """The TOTAL ORDER `(distance, index)`, host side.

    The same order `composite_key` builds on the device
    (`neighbors/checks/select_radix_identical.mojo`): distance first,
    then the index, always toward the LOWER index. No two elements of a row
    compare equal here, because the index half is unique by construction,
    which is what makes "the selected set" and "the selected ORDER" both
    pure functions of the input.
    """
    if da < db:
        return True
    if da > db:
        return False
    return ia < ib


def oracle_select_k(
    dist: List[Float32], idx: List[UInt32], k: Int
) raises -> List[Int32]:
    """The `k` smallest of one row under `_key_less`, ascending.

    A SELECTION SORT, which is `O(n k)` and is the point: the device
    selector is eight radix passes plus a rank pass and this is a
    completely different algorithm reaching the same total order. An oracle
    that re-spelled the radix passes would agree with the device about
    every bug they shared.
    """
    var n = len(dist)
    if k > n:
        raise Error(
            "oracle_select_k: k = "
            + String(k)
            + " over a row of "
            + String(n)
        )
    var taken = List[Bool]()
    for _ in range(n):
        taken.append(False)
    var out = List[Int32]()
    for _ in range(k):
        var best = -1
        for i in range(n):
            if taken[i]:
                continue
            if best < 0 or _key_less(
                dist[i], idx[i], dist[best], idx[best]
            ):
                best = i
        if best < 0:
            raise Error("oracle_select_k: ran out of candidates")
        taken[best] = True
        out.append(Int32(best))
    return out^


@fieldwise_init
struct OracleSearchResult(Movable):
    """`n_queries x k` distances and ORIGINAL indices, plus the per-query
    candidate count, matching `IvfSearchResult`'s three fields so a check
    compares like with like."""

    var distances: List[Float32]
    var indices: List[UInt32]
    var n_candidates: List[Int32]


def oracle_brute_force(
    index: List[Float32],
    n_index: Int,
    queries: List[Float32],
    n_queries: Int,
    d: Int,
    k: Int,
    is_sqrt: Bool,
) raises -> OracleSearchResult:
    """Exact k-NN, float32, the replayed device arithmetic, total order.

    This is what `n_probe == n_lists` must reduce to. It is also the truth
    function every recall REPORT is computed against.
    """
    var qn = oracle_row_norms(queries, n_queries, d, False)
    var yn = oracle_row_norms(index, n_index, d, False)
    var out_d = List[Float32]()
    var out_i = List[UInt32]()
    var counts = List[Int32]()
    for q in range(n_queries):
        var row = List[Float32]()
        var ids = List[UInt32]()
        for j in range(n_index):
            row.append(
                oracle_expanded_distance(
                    queries, q, index, j, d, qn[q], yn[j], is_sqrt
                )
            )
            ids.append(UInt32(j))
        var sel = oracle_select_k(row, ids, k)
        for s in range(k):
            out_d.append(row[Int(sel[s])])
            out_i.append(ids[Int(sel[s])])
        counts.append(Int32(n_index))
    return OracleSearchResult(out_d^, out_i^, counts^)


def oracle_ivf_search(
    centers: List[Float32],
    n_lists: Int,
    list_offsets: List[Int32],
    list_indices: List[UInt32],
    list_data: List[Float32],
    n_rows: Int,
    queries: List[Float32],
    n_queries: Int,
    d: Int,
    k: Int,
    n_probes: Int,
    is_sqrt: Bool,
) raises -> OracleSearchResult:
    """The whole IVF search, float32, serial, from the centroids down.

    Every step spelled a second time: the coarse distances, the probe
    selection under `(distance, LIST ID)`, the ascending-original-index
    merge, the candidate distances, and the final selection under
    `(distance, ORIGINAL INDEX)`.

    THE LAST ORDER IS THE ONE THAT MATTERS AND IT IS DELIBERATELY WRITTEN
    ON THE ORIGINAL INDEX, not on the candidate position. The device keys
    on the position, and the two agree only because `merge_probed_lists`
    makes the position order the original-index order (DEVIATION 1786). So
    the device and this oracle agreeing IS the evidence for that property,
    rather than an assumption baked into both.
    """
    var qn = oracle_row_norms(queries, n_queries, d, False)
    var cn = oracle_row_norms(centers, n_lists, d, False)
    var ln = oracle_row_norms(list_data, n_rows, d, False)

    var out_d = List[Float32]()
    var out_i = List[UInt32]()
    var counts = List[Int32]()

    for q in range(n_queries):
        var crow = List[Float32]()
        var cids = List[UInt32]()
        for l in range(n_lists):
            crow.append(
                oracle_expanded_distance(
                    queries, q, centers, l, d, qn[q], cn[l], is_sqrt
                )
            )
            cids.append(UInt32(l))
        var probes = oracle_select_k(crow, cids, n_probes)

        # The candidate slots, ascending by carried original index. Spelled
        # as "walk every slot, keep it if its list is probed" rather than as
        # a k-way merge, because that is a DIFFERENT algorithm reaching the
        # same order and an oracle's job is to be a different algorithm.
        var probed = List[Bool]()
        for _ in range(n_lists):
            probed.append(False)
        for p in range(n_probes):
            probed[Int(probes[p])] = True

        # Collect every slot of every probed list, then SELECTION SORT them
        # on the carried original index. The production path is a k-way
        # merge of runs it knows are already sorted
        # (`merge_probed_lists`); this makes no such assumption and would
        # therefore still produce the ascending order if the layout's
        # within-list ordering were broken -- which is what makes
        # `IVF_SAB_LIST_ARRIVAL_ORDER` a sabotage the two spellings
        # disagree about rather than one they share.
        var pool = List[Int32]()
        for l in range(n_lists):
            if not probed[l]:
                continue
            var s = Int(list_offsets[l])
            while s < Int(list_offsets[l + 1]):
                pool.append(Int32(s))
                s += 1
        var order = List[Int32]()
        var used = List[Bool]()
        for _ in range(len(pool)):
            used.append(False)
        for _ in range(len(pool)):
            var best = -1
            for t in range(len(pool)):
                if used[t]:
                    continue
                if best < 0 or (
                    list_indices[Int(pool[t])]
                    < list_indices[Int(pool[best])]
                ):
                    best = t
            used[best] = True
            order.append(pool[best])

        var n_cand = len(order)
        counts.append(Int32(n_cand))
        if n_cand < k:
            raise Error(
                "oracle_ivf_search: query "
                + String(q)
                + " has "
                + String(n_cand)
                + " candidates, fewer than k = "
                + String(k)
            )
        var crow2 = List[Float32]()
        var cid2 = List[UInt32]()
        for c in range(n_cand):
            var s = Int(order[c])
            crow2.append(
                oracle_expanded_distance(
                    queries, q, list_data, s, d, qn[q], ln[s], is_sqrt
                )
            )
            cid2.append(list_indices[s])
        var sel = oracle_select_k(crow2, cid2, k)
        for i in range(k):
            out_d.append(crow2[Int(sel[i])])
            out_i.append(cid2[Int(sel[i])])

    return OracleSearchResult(out_d^, out_i^, counts^)


def reference_distance_f64(
    q: List[Float32],
    qi: Int,
    y: List[Float32],
    yi: Int,
    d: Int,
    is_sqrt: Bool,
) -> Float64:
    """`sum_f (q_f - y_f)^2` in Float64, DIRECT form, ascending.

    Not the expanded form. The device arm and the float32 oracle both use
    `||q||^2 + ||y||^2 - 2 q.y`, whose cancellation is exactly what makes a
    point sitting on its own neighbour come out slightly negative
    (`unfused_distance_nn.cuh:80-81`'s clamp exists for it). A reference
    that shared that form would share that error and could not measure it.
    """
    var acc = Float64(0.0)
    for f in range(d):
        var diff = Float64(q[qi * d + f]) - Float64(y[yi * d + f])
        acc += diff * diff
    if is_sqrt:
        return host_sqrt(acc)
    return acc


def reference_nearest_f64(
    index: List[Float32],
    n_index: Int,
    queries: List[Float32],
    n_queries: Int,
    d: Int,
    k: Int,
    is_sqrt: Bool,
) raises -> List[UInt32]:
    """The `k` nearest ORIGINAL indices per query in Float64, ties broken
    toward the lower index.

    The truth function for the known-by-construction claims and for the
    recall REPORT's denominator.
    """
    var out = List[UInt32]()
    for q in range(n_queries):
        var row = List[Float64]()
        for j in range(n_index):
            row.append(
                reference_distance_f64(queries, q, index, j, d, is_sqrt)
            )
        var taken = List[Bool]()
        for _ in range(n_index):
            taken.append(False)
        for _ in range(k):
            var best = -1
            for j in range(n_index):
                if taken[j]:
                    continue
                if best < 0 or row[j] < row[best]:
                    best = j
            taken[best] = True
            out.append(UInt32(best))
    return out^


def recall_against(
    got: List[UInt32], truth: List[UInt32], n_queries: Int, k: Int
) raises -> Float64:
    """`|got ∩ truth| / (n_queries * k)`, per query, as a SET.

    A REPORT and never an assertion. IVF is approximate: a recall below one
    is the algorithm working as designed, and a threshold here would be a
    number nobody measured pretending to be a property. `ivf/README.md`
    says the same thing in the same words and `check_recall_is_reported`
    prints it with the word REPORT on the line.

    Compared as a SET rather than slot by slot, because two exactly
    equidistant neighbours can legitimately swap slots between the two
    computations while the answer is the same one. The ORDER is a separate
    claim and `check_nprobe_equals_nlists_is_brute_force` is what makes it.
    """
    if len(got) < n_queries * k or len(truth) < n_queries * k:
        raise Error(
            "recall_against: expected " + String(n_queries * k)
            + " entries in both arrays"
        )
    var hit = 0
    for q in range(n_queries):
        for a in range(k):
            for b in range(k):
                if got[q * k + a] == truth[q * k + b]:
                    hit += 1
                    break
    return Float64(hit) / Float64(n_queries * k)


def reference_assignment_f64(
    x: List[Float32],
    n_rows: Int,
    centers: List[Float32],
    n_lists: Int,
    d: Int,
) -> List[UInt32]:
    """The argmin assignment in Float64 with `raft::argmin_op`'s order.

    `(value, key)`: strictly smaller value wins, and on an EXACT tie the
    LOWER LIST ID wins -- `cluster/impl/distance/fused_distance_nn/
    simt_kernel.mojo:537`'s `d < val[i] or (d == val[i] and col < key[i])`,
    spelled here in Float64 so the fixture's exact tie is unambiguous.

    THIS DOES NOT REPLACE THE DEVICE ASSIGNMENT AND IS NOT COMPARED TO IT
    BIT FOR BIT. The k-means assignment kernel is `cluster/`'s, its
    identity status is `archive/research/UNSUPERVISED_IDENTITY.md`'s, and the only thing
    this function is used for is stating which list an EXACTLY equidistant
    point must land in.
    """
    var out = List[UInt32]()
    for r in range(n_rows):
        var best = 0
        var best_d = Float64(0.0)
        for l in range(n_lists):
            var dist = reference_distance_f64(x, r, centers, l, d, False)
            if l == 0 or dist < best_d:
                best = l
                best_d = dist
        out.append(UInt32(best))
    return out^
