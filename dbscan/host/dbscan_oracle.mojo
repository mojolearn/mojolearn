# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""DBSCAN TRAINING on the host, for a box with no GPU (workstream E of
docs/lanes/TEMP_claim_surface_plan_2026-09-14.md, the lane dbscan of
docs/lanes/BRIEF_cpu_training_2026-09-13.md section 1.1 "dbscan",
2026-09-14). The brief's census found NO host oracle for this fit; this
file is that oracle, written as a second spelling of the device path.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
and no GPU binding imports this file. The only float arithmetic is the
eps-neighborhood predicate, spelled from the two device arms through
`checks/numerics.mojo` leaves; everything after it is integers.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS.

  `host_eps_dist_sq`       `eps_dist_sq`, `neighbors/impl/ball_cover/
                           common.mojo:88`: `diff = ftz(ftz(a) - ftz(b))`,
                           `acc = ftz(identical_mul_add(diff, diff, acc))`,
                           dims ascending. The ball cover's one distance
                           (`rbc_cmp_dist` at L2SqrtUnexpanded, the DBSCAN
                           metric the index admits).
  `host_rbc_build`         `rbc_build_index`, `ball_cover.mojo:426`:
                           `rbc_n_landmarks` (floor sqrt), `_floyd_sample`
                           at seed 12345 (host integer code, copied), the
                           landmark rows, `rbc_landmark_1nn_kernel` (the
                           nearest landmark under a strict `<` over
                           landmarks ascending, its distance rooted by
                           `identical_sqrt`), the per-landmark member lists
                           ranked by (distance, index) (`rbc_rank_kernel`),
                           the reordered rows, and each landmark's radius,
                           the LAST member's distance.
  `host_rbc_eps_row`       `block_rbc_kernel_eps_csr_pass`, `registers.mojo:
                           204`, one query: the landmark test `dist_sq(q, r_k)
                           <= (eps + radius_k)^2` in groups of RBC_LANES
                           landmarks, ascending; per admitted landmark the
                           member scan from the ragged tail chunk backward
                           in chunks of RBC_LANES members, each member
                           admitted when `dist_sq(q, member) <= eps * eps`,
                           and the exit `cur_r_dist - min_warp_dist > eps`
                           (the chunk's first member's distance, the
                           smallest of the chunk) with `cur_r_dist =
                           identical_sqrt(dist_sq(q, r_k))`, evaluated
                           AFTER the chunk as the kernel evaluates it. The
                           neighbors land in the kernel's write order
                           (chunk order, lane order), which only a weighted
                           fold would read. RBC_LANES is read from the
                           matrix as the kernel reads it
                           (`lib_lane_width_for[TARGET_COLUMN]`, 32 on the
                           CPU column as on Apple and NVIDIA; AMD's 64 is
                           a different chunking of the same exit test, and
                           the three GPU columns agree on this lane).
  `host_brute_eps_row`     `eps_unexp_neigh_kernel`, `dbscan/impl/
                           neighbors/epsilon_neighborhood.mojo`: the
                           BRUTE_FORCE arm, `diff = ftz(x - ftz(y))` (the
                           query value NOT flushed, the index value
                           flushed, the kernel's `_eps_acc`), L2
                           `ftz(identical_mul_add(diff, diff, acc))`, L1
                           `ftz(acc + abs(diff))`, k ascending, admitted
                           when `acc <= thresh` with `thresh =
                           dbscan_metric_threshold` (`Float32(eps * eps)`
                           for L2, `Float32(eps)` for L1).
  `host_weak_cc`           `weak_cc_init_kernel` and `weak_cc_label_kernel`,
                           `dbscan/impl/sparse/detail/csr.mojo`, one batch
                           of every row: a core row starts at `i + 1`, a
                           non-core row at MAX_LABEL; each pass walks the
                           rows in index order and, for each neighbor `j`,
                           propagates `min` from a core row and adopts the
                           label of a core neighbor, until a pass changes
                           nothing. THE PASS COUNT IS SCHEDULE DEPENDENT on
                           the device (every row runs concurrently with
                           atomics); this replay is the one-thread schedule
                           in row order, whose fixed point is the same
                           labels (the minimum core index of each
                           core-connected component, and for a border row
                           the minimum over its core neighbors) and whose
                           count is what `n_iter_` reports on the host. No
                           column hashes the count.
  `host_make_monotonic`    `make_monotonic` and `relabel_for_skl_kernel`:
                           the distinct non-noise labels ranked ascending,
                           each label replaced by its rank, MAX_LABEL to
                           -1.
  `host_dbscan_fit`        `dbscan_fit` (`dbscan/estimator.mojo:118`) and
                           `dbscan_fit_impl_weighted` plus `dbscan_fit`
                           (`dbscan/impl/dbscan.mojo`, `runner.mojo`): the
                           refusals in their words, `sparse_rbc_mode`
                           (RBC unless `n_features > MAX_LABEL // n_rows`),
                           the L1-on-RBC refusal, the brute overflow
                           refusal, ONE BATCH (the device sizes its batch
                           from its own memory; a host has no device
                           memory to size from, and the labels are batch
                           invariant), the vertex degrees, the core mask
                           (`vd >= min_pts`), the propagation, the relabel.

THE WEIGHTED CORE TEST (lane/cpu-training-batch3, 2026-09-14, the
dbscan-weighted lane, which the 136-lane record carries on all three GPU
columns). `sample_weight` changes the answer in one place,
`core_points_weighted_kernel` (`dbscan/impl/corepoints/compute.mojo`):
a row is core when its weighted degree `>= Float32(min_pts)`. The degree is
`host_weighted_degree`, a second spelling of the two device folds
(`dbscan/impl/vertexdeg/algo.mojo`), each WVD_TPB strided partials `acc =
ftz(acc + ftz(w))` then `pinned_block_sum`'s halving tree with no flush
inside and none after:
  the ball cover arm, `weighted_vertex_deg_csr_kernel`, strides CSR
  POSITIONS of the row as `rbc_eps_nn_query_fill` leaves it under
  IDENTICAL, canonicalized to ascending column (DEVIATION 551,
  `neighbors/checks/ball_cover_canonical_order.mojo`), so the host sorts
  the kernel-write-order row first;
  the brute arm, `weighted_vertex_deg_dense_kernel`, strides COLUMNS of the
  dense adjacency, adding where the bit is set.
WVD_TPB is read through `lib_block_size_for[K_LIB_WEIGHTED_VERTEX_DEG]`, a
NUMERIC row that resolves to the identity floor on every column.

NOT RESTATED, REFUSED BY NAME: an explicit `max_mbytes_per_batch` (the host
runs one batch and says so rather than pretending to size a device).

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` (the routed families'
one define) increments the first final label, including noise (-1 to 0).
The earlier min_pts + 1 control missed all-noise fixtures: both thresholds
returned the same labels. This native output corruption changes a measured
label on every nonempty fit. Read back by `estimators_host_sabotage`.

The restatement is a prediction until measured. The CPU identity gate
(`tools/identity_break.py --diff <3 GPU columns> <cpu json> --lanes dbscan
--require-columns 4`) is the measurement.
"""
from std.math import sqrt
from std.sys.compile import is_defined
from std.builtin.sort import sort

from checks.kernel_matrix import (
    K_LIB_WEIGHTED_VERTEX_DEG,
    TARGET_COLUMN,
    lib_block_size_for,
    lib_lane_width_for,
)
from checks.numerics import ftz, identical_mul_add, identical_sqrt


comptime DBSCAN_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: `RBC_LANES` (`registers.mojo:170`), the chunk of the member scan.
comptime RBC_LANES = lib_lane_width_for[TARGET_COLUMN]()

#: `WVD_TPB` (`dbscan/impl/vertexdeg/algo.mojo`), the weighted degree's
#: stride and fold width, through the same accessor.
comptime WVD_TPB = lib_block_size_for[K_LIB_WEIGHTED_VERTEX_DEG, TARGET_COLUMN]()

#: `MAX_LABEL` (`csr.mojo`), `RBC_FLT_MAX` (`common.mojo:85`), the metric
#: and method ids (`epsilon_neighborhood.mojo`, `runner.mojo`), restated
#: because those files import the GPU.
comptime MAX_LABEL = Int32(2147483647)
comptime RBC_FLT_MAX = Float32(3.4028234663852886e38)
comptime DBSCAN_METRIC_L2 = 0
comptime DBSCAN_METRIC_L1 = 1
comptime EPS_NN_BRUTE_FORCE = 0
comptime EPS_NN_RBC = 1
comptime RBC_SEED = UInt64(12345)


def host_eps_dist_sq(
    a: List[Float32], a_off: Int, b: List[Float32], b_off: Int, n_dims: Int,
) -> Float32:
    """`eps_dist_sq`, the ball cover's L2 fold."""
    var sum_sq = Float32(0.0)
    for i in range(n_dims):
        var diff = ftz(ftz(a[a_off + i]) - ftz(b[b_off + i]))
        sum_sq = ftz(identical_mul_add(diff, diff, sum_sq))
    return sum_sq


def host_rbc_n_landmarks(m: Int) -> Int:
    """`rbc_n_landmarks`: floor(sqrt(m)), at least 1."""
    if m <= 1:
        return 1
    var s = Int(sqrt(Float64(m)))
    while (s + 1) * (s + 1) <= m:
        s += 1
    while s * s > m:
        s -= 1
    if s < 1:
        s = 1
    return s


def host_floyd_sample(m: Int, n_landmarks: Int, seed: UInt64) -> List[Int32]:
    """`_floyd_sample`, copied: splitmix64 draws, Floyd's sampling."""
    var picked = List[Int32]()
    var step = UInt64(0)
    for j in range(m - n_landmarks, m):
        var z = seed + step * UInt64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> UInt64(31))
        step += UInt64(1)
        var t = Int32(Int(z % UInt64(j + 1)))
        var hit = False
        for q in range(len(picked)):
            if picked[q] == t:
                hit = True
                break
        if hit:
            picked.append(Int32(j))
        else:
            picked.append(t)
    return picked^


@fieldwise_init
struct RBCHostIndex(Movable):
    """What `rbc_build_index` leaves on the device, on the host."""

    var n_landmarks: Int
    var r: List[Float32]
    var x_reordered: List[Float32]
    var r_indptr: List[Int]
    var r_1nn_cols: List[Int32]
    var r_1nn_dists: List[Float32]
    var r_radius: List[Float32]


def host_rbc_build(x: List[Float32], m: Int, n_cols: Int) -> RBCHostIndex:
    """`rbc_build_index` at the DBSCAN call (seed 12345, L2SqrtUnexpanded)."""
    var n_landmarks = host_rbc_n_landmarks(m)
    var picked = host_floyd_sample(m, n_landmarks, RBC_SEED)
    var r = List[Float32](length=n_landmarks * n_cols, fill=Float32(0.0))
    for k in range(n_landmarks):
        var src = Int(picked[k])
        for c in range(n_cols):
            r[k * n_cols + c] = x[src * n_cols + c]
    # rbc_landmark_1nn_kernel
    var nearest = List[Int](length=m, fill=0)
    var nearest_dist = List[Float32](length=m, fill=Float32(0.0))
    for i in range(m):
        var best = RBC_FLT_MAX
        var best_k = 0
        for k in range(n_landmarks):
            var d = host_eps_dist_sq(x, i * n_cols, r, k * n_cols, n_cols)
            if d < best:
                best = d
                best_k = k
        nearest[i] = best_k
        nearest_dist[i] = identical_sqrt(best)
    # counts, the exclusive scan, the slots
    var counts = List[Int](length=n_landmarks, fill=0)
    for i in range(m):
        counts[nearest[i]] += 1
    var r_indptr = List[Int](length=n_landmarks + 1, fill=0)
    for k in range(n_landmarks):
        r_indptr[k + 1] = r_indptr[k] + counts[k]
    # rbc_rank_kernel: within a landmark, ascending (distance, index). The
    # scatter's slot order is atomic and unspecified; the rank is a total
    # order so the ranked lists do not depend on it.
    var r_1nn_cols = List[Int32](length=m, fill=Int32(0))
    var r_1nn_dists = List[Float32](length=m, fill=Float32(0.0))
    var cursor = List[Int](length=n_landmarks, fill=0)
    var slot_cols = List[Int32](length=m, fill=Int32(0))
    var slot_dists = List[Float32](length=m, fill=Float32(0.0))
    for i in range(m):
        var k = nearest[i]
        var pos = r_indptr[k] + cursor[k]
        cursor[k] += 1
        slot_cols[pos] = Int32(i)
        slot_dists[pos] = nearest_dist[i]
    for k in range(n_landmarks):
        var s = r_indptr[k]
        var e = r_indptr[k + 1]
        for p in range(s, e):
            var dp = slot_dists[p]
            var ip = slot_cols[p]
            var rank = 0
            for q in range(s, e):
                var dq = slot_dists[q]
                if dq < dp:
                    rank += 1
                elif dq == dp and slot_cols[q] < ip:
                    rank += 1
            r_1nn_dists[s + rank] = dp
            r_1nn_cols[s + rank] = ip
    var x_reordered = List[Float32](length=m * n_cols, fill=Float32(0.0))
    for pos in range(m):
        var src = Int(r_1nn_cols[pos])
        for c in range(n_cols):
            x_reordered[pos * n_cols + c] = x[src * n_cols + c]
    var r_radius = List[Float32](length=n_landmarks, fill=Float32(0.0))
    for k in range(n_landmarks):
        var s = r_indptr[k]
        var e = r_indptr[k + 1]
        if e <= s:
            r_radius[k] = Float32(0.0)
        else:
            r_radius[k] = r_1nn_dists[e - 1]
    return RBCHostIndex(
        n_landmarks, r^, x_reordered^, r_indptr^, r_1nn_cols^, r_1nn_dists^,
        r_radius^,
    )


def host_rbc_eps_row(
    idx: RBCHostIndex, x: List[Float32], q: Int, n_cols: Int, eps: Float32,
) -> List[Int32]:
    """`block_rbc_kernel_eps_csr_pass` for one query row, the neighbors in
    the kernel's write order."""
    var out = List[Int32]()
    var x_base = n_cols * q
    var eps_cmp = eps * eps
    var cur_k0 = 0
    while cur_k0 < idx.n_landmarks:
        # The lane group: every admitted landmark of the group, ascending.
        for lane in range(RBC_LANES):
            var cur_k = cur_k0 + lane
            if cur_k >= idx.n_landmarks:
                break
            var lane_r_cmp = host_eps_dist_sq(x, x_base, idx.r, cur_k * n_cols, n_cols)
            var bound = eps + idx.r_radius[cur_k]
            if not (lane_r_cmp <= bound * bound):
                continue
            var r_start = idx.r_indptr[cur_k]
            var r_size = idx.r_indptr[cur_k + 1] - r_start
            var cur_r_dist = identical_sqrt(lane_r_cmp)
            var limit = (r_size // RBC_LANES) * RBC_LANES
            # The ragged tail chunk, lanes ascending.
            var min_warp_dist = cur_r_dist
            if limit < r_size:
                min_warp_dist = idx.r_1nn_dists[r_start + limit]
            for lid in range(RBC_LANES):
                var i = limit + lid
                if i < r_size:
                    var dist = host_eps_dist_sq(
                        x, x_base, idx.x_reordered, (r_start + i) * n_cols, n_cols
                    )
                    if dist <= eps_cmp:
                        out.append(idx.r_1nn_cols[r_start + i])
            var i0 = limit
            if cur_r_dist - min_warp_dist > eps:
                i0 = 0
            while i0 >= RBC_LANES:
                i0 -= RBC_LANES
                var min_warp_dist2 = idx.r_1nn_dists[r_start + i0]
                for lid in range(RBC_LANES):
                    var dist2 = host_eps_dist_sq(
                        x, x_base, idx.x_reordered, (r_start + i0 + lid) * n_cols, n_cols
                    )
                    if dist2 <= eps_cmp:
                        out.append(idx.r_1nn_cols[r_start + i0 + lid])
                if cur_r_dist - min_warp_dist2 > eps:
                    i0 = 0
        cur_k0 += RBC_LANES
    return out^


def host_brute_eps_row(
    x: List[Float32], q: Int, n_rows: Int, n_cols: Int, thresh: Float32, metric: Int,
) -> List[Int32]:
    """`eps_unexp_neigh_kernel` for one query row against every row."""
    var out = List[Int32]()
    var x_base = q * n_cols
    for j in range(n_rows):
        var acc = Float32(0.0)
        var y_base = j * n_cols
        for k in range(n_cols):
            var diff = ftz(x[x_base + k] - ftz(x[y_base + k]))
            if metric == DBSCAN_METRIC_L1:
                acc = ftz(acc + abs(diff))
            else:
                acc = ftz(identical_mul_add(diff, diff, acc))
        if acc <= thresh:
            out.append(Int32(j))
    return out^


def host_metric_threshold(metric: Int, eps: Float64) -> Float32:
    """`dbscan_metric_threshold`."""
    if metric == DBSCAN_METRIC_L1:
        return Float32(eps)
    return Float32(eps * eps)


def host_weak_cc(
    mut labels: List[Int32],
    row_ptr: List[Int],
    col_ind: List[Int32],
    core: List[UInt8],
    n_rows: Int,
    max_iterations: Int,
) raises -> Int:
    """`weak_cc_batched` over one batch of every row, the one-thread
    schedule in row order; returns the pass count."""
    for i in range(n_rows):
        if core[i] != UInt8(0):
            labels[i] = Int32(i + 1)
        else:
            labels[i] = MAX_LABEL
    var passes = 0
    var converged = False
    for _it in range(max_iterations):
        var changed = False
        for gi in range(n_rows):
            var ci = labels[gi]
            var ci_mod = False
            var ci_allow_prop = core[gi] != UInt8(0)
            for p in range(row_ptr[gi], row_ptr[gi + 1]):
                var j = Int(col_ind[p])
                var cj = labels[j]
                var cj_allow_prop = core[j] != UInt8(0)
                if ci < cj and ci_allow_prop:
                    if ci < labels[j]:
                        labels[j] = ci
                    if cj_allow_prop:
                        changed = True
                elif ci > cj and cj_allow_prop:
                    ci = cj
                    ci_mod = True
            if ci_mod:
                if ci < labels[gi]:
                    labels[gi] = ci
                if ci_allow_prop:
                    changed = True
        passes += 1
        if not changed:
            converged = True
            break
    if not converged:
        raise Error(
            "weak_cc_batched: label propagation did not converge in "
            + String(max_iterations)
            + " passes. Under identical a truncated propagation"
            " is refused rather than returned: its labels are a"
            " snapshot of the atomic order on THIS machine, not a"
            " function of the graph. Raise max_iterations."
        )
    return passes


def host_make_monotonic(mut labels: List[Int32], n_rows: Int):
    """`make_monotonic` then `relabel_for_skl_kernel`."""
    var seen = List[Int32](length=n_rows, fill=Int32(0))
    for i in range(n_rows):
        var l = labels[i]
        if l != MAX_LABEL:
            seen[Int(l) - 1] = Int32(1)
    var rank = List[Int32](length=n_rows, fill=Int32(0))
    var running = Int32(0)
    for i in range(n_rows):
        rank[i] = running
        running += seen[i]
    for i in range(n_rows):
        var l = labels[i]
        if l != MAX_LABEL:
            labels[i] = rank[Int(l) - 1] + Int32(1)
    for i in range(n_rows):
        if labels[i] == MAX_LABEL:
            labels[i] = Int32(-1)
        else:
            labels[i] = labels[i] - Int32(1)


def host_weighted_degree(
    row: List[Int32], weights: List[Float32], by_column: Bool, n_rows: Int,
) -> Float32:
    """One row's weighted degree. `row` is the row's neighbor columns in
    ASCENDING order. `by_column` False is `weighted_vertex_deg_csr_kernel`
    (thread `t` strides CSR positions `t, t + WVD_TPB, ...`); True is
    `weighted_vertex_deg_dense_kernel` (thread `t` strides columns `t, t +
    WVD_TPB, ...` over all `n_rows`, adding where the adjacency bit is set).
    Both close on the halving tree `red[t] = red[t] + red[t + step]`."""
    comptime assert (WVD_TPB & (WVD_TPB - 1)) == 0, (
        "dbscan host: the halving tree needs a power-of-two block"
    )
    var red = List[Float32](length=WVD_TPB, fill=Float32(0.0))
    var n = len(row)
    if by_column:
        # Column `c` belongs to thread `c % WVD_TPB`, and each thread walks
        # its columns ascending, so the ascending row distributes exactly.
        for p in range(n):
            var c = Int(row[p])
            var t = c % WVD_TPB
            red[t] = ftz(red[t] + ftz(weights[c]))
    else:
        for t in range(WVD_TPB):
            var acc = Float32(0.0)
            var p = t
            while p < n:
                acc = ftz(acc + ftz(weights[Int(row[p])]))
                p += WVD_TPB
            red[t] = acc
    var step = WVD_TPB // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return red[0]


def host_sorted_row(row: List[Int32]) -> List[Int32]:
    """A CSR row in ascending column order; the columns are unique.

    The weighted RBC path needs this exact order before distributing the
    columns across the pinned reduction's lanes.  Use the standard host sort
    rather than insertion sort: a dense epsilon neighbourhood can contain
    every sample, and doing a quadratic sort independently for every row
    made the CPU product and its identity proof needlessly quadratic on top
    of neighbourhood construction.  Sorting integer column ids performs no
    floating-point arithmetic and therefore cannot move the weighted fold's
    bits.
    """
    var out = row.copy()
    sort(out)
    return out^


@fieldwise_init
struct DBSCANHostFit(Movable):
    var labels: List[Int32]
    var passes: Int
    #: the core mask the propagation read, 1 where the core test held
    #: (lane/inference-transductive-predict, 2026-09-15, for
    #: `DBSCAN(prediction_data=True)`); the device copies its own.
    var core: List[UInt8]


def host_dbscan_fit(
    x: List[Float32],
    n_samples: Int,
    n_features: Int,
    eps: Float64,
    min_samples: Int,
    max_iterations: Int,
    eps_nn_method: Int,
    metric: Int,
    weights: List[Float32] = List[Float32](),
    has_weights: Bool = False,
) raises -> DBSCANHostFit:
    """`dbscan_fit` (`dbscan/estimator.mojo`) without the DeviceContext,
    one batch of every row."""
    if n_samples < 1 or n_features < 1:
        raise Error(
            "dbscan_fit needs n_samples and n_features >= 1: got "
            + String(n_samples)
            + ", "
            + String(n_features)
        )
    if not (eps > 0.0):
        raise Error("dbscan_fit needs eps > 0, got " + String(eps))
    if min_samples < 1:
        raise Error("dbscan_fit needs min_samples >= 1, got " + String(min_samples))
    if eps_nn_method != EPS_NN_RBC and eps_nn_method != EPS_NN_BRUTE_FORCE:
        raise Error(
            "dbscan_fit: eps_nn_method must be EPS_NN_RBC (1) or"
            " EPS_NN_BRUTE_FORCE (0), got " + String(eps_nn_method)
        )
    if metric != DBSCAN_METRIC_L2 and metric != DBSCAN_METRIC_L1:
        raise Error(
            "dbscan_fit: metric must be DBSCAN_METRIC_L2 (0) or"
            " DBSCAN_METRIC_L1 (1), got " + String(metric)
        )
    var cap = max_iterations
    if cap <= 0:
        cap = n_samples + 1

    var n_rows = n_samples
    var sparse_rbc_mode = eps_nn_method == EPS_NN_RBC
    if sparse_rbc_mode and n_features > Int(MAX_LABEL) // n_rows:
        sparse_rbc_mode = False
    if sparse_rbc_mode and metric != DBSCAN_METRIC_L2:
        raise Error(
            "dbscan: metric='manhattan' is served by"
            " the BRUTE_FORCE arm only. The ball cover's landmark radii and"
            " its triangle-inequality bounds are computed as Euclidean"
            " distances in neighbors/impl/ball_cover/ (common.mojo"
            " eps_dist_sq; registers.mojo:280, :422, :547), so an L1 query"
            " needs an L1 index that lane has not built yet. Pass"
            " algorithm='brute'."
        )
    if not sparse_rbc_mode and n_rows * n_rows >= Int(MAX_LABEL):
        raise Error(
            "An overflow occurred with the current choice of precision and"
            " the number of samples. (Max allowed batch size is "
            + String(Int(MAX_LABEL) // n_rows)
            + ", but was "
            + String(n_rows)
            + ")."
        )

    # The eps neighborhood of every row, CSR, in the arm's write order.
    var row_ptr = List[Int](length=n_rows + 1, fill=0)
    var col_ind = List[Int32]()
    var wght_sum = List[Float32]()
    if has_weights and len(weights) != n_rows:
        raise Error(
            "dbscan_fit: sample_weight holds " + String(len(weights))
            + " values, n_samples is " + String(n_rows)
        )
    if sparse_rbc_mode:
        var index = host_rbc_build(x, n_rows, n_features)
        var eps_radius = Float32(eps)
        for q in range(n_rows):
            var row = host_rbc_eps_row(index, x, q, n_features, eps_radius)
            if has_weights:
                wght_sum.append(
                    host_weighted_degree(host_sorted_row(row), weights, False, n_rows)
                )
            for p in range(len(row)):
                col_ind.append(row[p])
            row_ptr[q + 1] = len(col_ind)
        if len(col_ind) > Int(MAX_LABEL):
            raise Error(
                "dbscan: the ball-cover neighbourhood has "
                + String(len(col_ind))
                + " edges in one batch, which does not fit the int32 CSR"
                " this implementation uses. cuML requires int64 labels for RBC"
                " (runner.cuh:143-150) for exactly this reason. Use a"
                " smaller eps, a smaller batch, or the BRUTE_FORCE arm."
            )
    else:
        var thresh = host_metric_threshold(metric, eps)
        for q in range(n_rows):
            var row = host_brute_eps_row(x, q, n_rows, n_features, thresh, metric)
            if has_weights:
                wght_sum.append(host_weighted_degree(row, weights, True, n_rows))
            for p in range(len(row)):
                col_ind.append(row[p])
            row_ptr[q + 1] = len(col_ind)

    # core_points_kernel: vd >= min_pts, vd the row's neighbor count (the
    # row itself included, its distance being zero).
    var core = List[UInt8](length=n_rows, fill=UInt8(0))
    var min_pts = min_samples
    if has_weights:
        # `core_points_weighted_kernel`: the float compared directly.
        for i in range(n_rows):
            core[i] = UInt8(1) if wght_sum[i] >= Float32(min_pts) else UInt8(0)
    else:
        for i in range(n_rows):
            var vd = row_ptr[i + 1] - row_ptr[i]
            core[i] = UInt8(1) if vd >= min_pts else UInt8(0)

    var labels = List[Int32](length=n_rows, fill=MAX_LABEL)
    var passes = host_weak_cc(labels, row_ptr, col_ind, core, n_rows, cap)
    host_make_monotonic(labels, n_rows)
    comptime if DBSCAN_ORACLE_HOST_SABOTAGE:
        # Corrupt an actual native label, even when every row is noise.
        # Production labels are -1 or at most n_rows - 1, so this fits Int32.
        labels[0] += 1
    return DBSCANHostFit(labels^, passes, core^)
