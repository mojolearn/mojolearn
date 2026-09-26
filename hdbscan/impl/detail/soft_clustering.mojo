# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`membership_vector` and `all_points_membership_vectors`: HDBSCAN soft
clustering, in float32, on the device, vendor-agnostic.

Reference: `cuml-v26.08.00/cpp/src/hdbscan/detail/soft_clustering.cuh`
(`dist_membership_vector` `:46-149`, `all_points_outlier_membership_vector`
`:151-207`, `all_points_prob_in_some_cluster` `:209-253`,
`outlier_membership_vector` `:255-322`, `prob_in_some_cluster` `:324-369`,
`all_points_membership_vectors` `:385-482`, `membership_vector`
`:501-627`), `detail/kernels/soft_clustering.cuh` (`merge_height_kernel`,
both overloads, `:14-98`) and `detail/utils.h` (`normalize` `:176-194`,
`softmax` `:206-229`). The Python entries are cuML's `hdbscan.pyx:1180`
(`membership_vector`) and `:1114` (`all_points_membership_vectors`).

THE PASS, one row at a time (a query row, or a training row for all points)
  0. membership_vector only: `approximate_predict`'s nearest mutual
     reachability neighbor and its lambda (`predict.mojo`, the same call).
  1. the distance from the row to every exemplar, the minimum per selected
     cluster, `1 / min` (or `FLT_MAX / n_selected` at zero), normalized
     (`dist_membership_kernel`).
  2. the row's lambda and the death of the cluster its leaf point falls out
     of (`soft_row_prep_kernel`); for a query the lambda is
     `min(prediction_lambda, neighbor_lambda)` (`soft_clustering.cuh:564-573`).
  3. the merge height of the row against every selected cluster, the
     condensed tree walked toward the root (`soft_merge_height_kernel`).
  4. the outlier membership, `death / (death - height)` for a query and
     `exp(-(death + 1e-8) / height)` for a training row, then the softmax
     `exp(x - max(x))` and the normalization (`soft_outlier_kernel`).
  5. the probability of being in some cluster, the height at the row's
     argmax over its `max(lambda, death)` (`soft_prob_kernel`).
  6. the combination, `outlier^2 * sqrt(dist)` for a query and
     `dist * outlier` for a training row, normalized, times step 5
     (`soft_combine_kernel`).
Every kernel is one thread per row, one pass over that row's columns, no
fold across rows and no atomic, so a row's answer cannot see which other
rows were in the call. That is what the identity harness's batch part holds
the public call to.

======================================================================
DEVIATION BLOCK -- DEVIATION 1616. SOFT CLUSTERING IS FLOAT32 THROUGHOUT,
EVERY SEAM PINNED, AND SATURATED WHERE THE REFERENCE OVERFLOWS.
======================================================================
WHAT THEIRS DOES. Four seams promote to FLOAT64 inside a float kernel and
round back to float32 once: `value_t(1.0 / val)` (`soft_clustering.cuh:142`),
`exp(-(vec_in + 1e-8) / mat_in)` (`:200`), `pow(m, 2) * pow(d, 0.5)`
(`:592`) and `max(lambda, death) + 1e-8` followed by a float / double
division (`:362-363`). An Apple GPU has no float64, so a formulation that
runs on every column cannot keep them. The rest is float32 already: the
exemplar distances, `vec / (vec - mat)` with its `1e-8` floor (`:312-314`),
the softmax's `exp(mat - vec)` (`utils.h:226`), the L1 row sums and their
divisions (`utils.h:183-193`) and the products (`:466`, `:479`, `:625`).
Their sums and argmaxes are CUB and RAFT reductions whose fold order and tie
order are not specified.

WHAT OURS DOES. Every step is float32 with the repository's identical
seams: one correctly rounded division through `identical_div` (row 49),
`identical_exp` (row 12's portable `expf`), `identical_sqrt`,
`identical_mul` / `identical_mul_add` (row 9's contraction pin) and `ftz`
(row 10's flush) on every sum, difference and product. The four float64
seams become:
  (a) `1 / val`            identical_div(1, val)
  (b) exp(-(v + 1e-8)/m)   identical_exp(-identical_div(ftz(v + 1e-8f), m))
  (c) m^2 * d^0.5          ftz(m * m) * identical_sqrt(d), two float32
                           products instead of one float64 product
  (d) max(l, death)+1e-8   ftz(max(l, death) + 1e-8f), then identical_div
The `1e-8` is the float32 literal 9.99999994e-9 at all three uses (theirs
at `:313` is that same float32; at `:200` and `:362` it is the double).
PINNED ORDERS. A row sum folds columns ascending, `ftz(acc + v)`. The
exemplar distance is the expanded L2 form theirs uses for L2SqrtExpanded,
`sqrt(max(0, |q|^2 + |e|^2 - 2 q.e))`, each norm and the dot product
folded over features ascending through `identical_mul_add`. The argmax over
merge heights keeps the FIRST (lowest) column on a tie, strict `>`.
SATURATION, WHERE THEIRS OVERFLOWS TO NaN. With duplicated rows a lambda
and a death are FLT_MAX (a zero distance), so theirs computes
`FLT_MAX / 1e-8 = inf` at `:314`, the softmax takes `inf - inf` and the
whole row is NaN; a row with several zero exemplar distances can overflow
its L1 sum to inf. Ours saturates the quotient of `:314`, the quotient
inside `:200` and every row sum at FLT_MAX, so those rows are finite. A row
whose sum is zero (every entry underflowed and was flushed) is left at
zero instead of `0 / 0`. A NaN never reaches an output.
HOW FAR FROM THEIRS. On inputs where theirs is finite, each seam differs
from the float64 value by the float32 rounding of its intermediates: (a)
at most 1 ulp; (b) the rounding of `v + 1e-8` (nothing once `v` is above
about 0.1), one float32 quotient and portable `expf`, a few ulps before the
softmax; (c) the extra rounding of `m * m` and `sqrt(d)`, about 1.5 ulp,
and an entry theirs rounds to a tiny float32 can be flushed to zero here
when `m * m` falls below FLT_MIN; (d) the `1e-8` is lost in float32 unless
the larger lambda is below about 0.2, which the double keeps as a relative
`1e-8 / lambda`; and the fold order of each sum, at most one ulp per
column. A different argmax on an exact merge height tie picks a different
death and can move a whole row's scale; theirs does not specify which it
picks.
MEASURED (2026-09-15,
`bench/results/identity_break/2026-09-15_hdbscan-membership-vector/`).
(1) The seams alone: on the same fitted tree, against a numpy
transcription of their four float64 seams, over 4 fits (blobs and a
duplicated grid, eom and leaf): all_points rows differ by at most 2.0e-7
in their sums and 1.2e-2 in one cell (a cell near 1e-6 that one side
flushes or underflows); membership_vector rows by at most 5.1e-5 (that
transcription also recomputes the neighborhood in float64); the argmax of
no row moves; no row is non-finite. (2) End to end against cuML 26.8.0 on
an H100: of the 4 fits, cuML's labels_ equal ours on one (the duplicated
grid, eom); there cells differ by up to 0.30 and row sums by up to 0.29
with no argmax moved, which (1) bounds as the fit's tie resolution
feeding a different tree, not these seams.
======================================================================
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from core.identity_trace import IdentityTrace
from hdbscan.impl.detail.predict import (
    _download_f32,
    _upload_f32,
    _upload_i32,
    approximate_predict,
)
from hdbscan.impl.prediction_data import (
    PredictionData,
    refuse_nonfinite_queries,
    refuse_soft_clustering_inputs,
)


comptime SOFT_TPB = 256
"""Their `int tpb = 256` template default (`soft_clustering.cuh:151,209`)."""

comptime SOFT_FLOAT32_MAX = Float32(3.4028234663852886e38)
comptime SOFT_EPS = Float32(1e-8)
"""The float32 `1e-8` (9.99999994e-9), DEVIATION 1616 (b), (d)."""

comptime SOFT_MODE_PREDICT: Int32 = 0
"""`membership_vector`: rows are queries."""
comptime SOFT_MODE_ALL_POINTS: Int32 = 1
"""`all_points_membership_vectors`: rows are training points."""


def soft_normalize_row(
    p: MutPointer[Float32, MutAnyOrigin], base: Int, n: Int
):
    """`Utils::normalize` (`utils.h:176-194`) for one row. DEVIATION 1616:
    the L1 sum folds columns ascending, saturates at FLT_MAX, and a zero
    sum leaves the row at zero."""
    var s = Float32(0.0)
    for c in range(n):
        s = ftz(s + p.unsafe_load(base + c))
    if s > SOFT_FLOAT32_MAX:
        s = SOFT_FLOAT32_MAX
    if s == Float32(0.0):
        return
    for c in range(n):
        p.unsafe_store(base + c, identical_div(p.unsafe_load(base + c), s))


def soft_row_norm_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_cols: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """The squared norm of each row, features folded ascending."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows):
        return
    var d = Int(n_cols)
    var acc = Float32(0.0)
    for f in range(d):
        var v = ftz(x.unsafe_load(idx * d + f))
        acc = ftz(identical_mul_add(v, v, acc))
    dst.unsafe_store(idx, acc)


def exemplar_min_dist_kernel(
    rows: MutPointer[Float32, MutAnyOrigin],
    row_norms: MutPointer[Float32, MutAnyOrigin],
    exemplars: MutPointer[Float32, MutAnyOrigin],
    exemplar_norms: MutPointer[Float32, MutAnyOrigin],
    exemplar_label_offsets: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_cols: Int32,
    n_selected: Int32,
    min_dist: MutPointer[Float32, MutAnyOrigin],
):
    """`dist_membership_vector`'s pairwise L2SqrtExpanded distance and
    `reduction_op` (`soft_clustering.cuh:88-118`): the minimum over each
    selected cluster's exemplars. A minimum is order-free; the distance is
    DEVIATION 1616's pinned expanded form."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows):
        return
    var d = Int(n_cols)
    var ns = Int(n_selected)
    var rn = ftz(row_norms.unsafe_load(idx))
    for c in range(ns):
        var lo = Int(exemplar_label_offsets.unsafe_load(c))
        var hi = Int(exemplar_label_offsets.unsafe_load(c + 1))
        var best = SOFT_FLOAT32_MAX
        for j in range(lo, hi):
            var acc = Float32(0.0)
            for f in range(d):
                var qv = ftz(rows.unsafe_load(idx * d + f))
                var ev = ftz(exemplars.unsafe_load(j * d + f))
                acc = ftz(identical_mul_add(qv, ev, acc))
            var en = ftz(exemplar_norms.unsafe_load(j))
            var dist = ftz(identical_mul_add(Float32(-2.0), acc, ftz(rn + en)))
            if dist <= Float32(0.0):
                dist = Float32(0.0)
            dist = ftz(identical_sqrt(dist))
            if dist < best:
                best = dist
        min_dist.unsafe_store(idx * ns + c, best)


def dist_membership_kernel(
    min_dist: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_selected: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`soft_clustering.cuh:135-148`: `val > 0 ? 1 / val : FLT_MAX /
    n_selected`, then `normalize`. DEVIATION 1616 (a)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows):
        return
    var ns = Int(n_selected)
    var base = idx * ns
    for c in range(ns):
        var v = min_dist.unsafe_load(base + c)
        if v > Float32(0.0):
            dst.unsafe_store(base + c, identical_div(Float32(1.0), v))
        else:
            dst.unsafe_store(
                base + c, identical_div(SOFT_FLOAT32_MAX, Float32(ns))
            )
    soft_normalize_row(dst, base, ns)


def soft_row_prep_kernel(
    mode: Int32,
    min_mr_inds: MutPointer[Int32, MutAnyOrigin],
    prediction_lambdas: MutPointer[Float32, MutAnyOrigin],
    parents: MutPointer[Int32, MutAnyOrigin],
    index_into_children: MutPointer[Int32, MutAnyOrigin],
    lambdas: MutPointer[Float32, MutAnyOrigin],
    deaths: MutPointer[Float32, MutAnyOrigin],
    n_leaves: Int32,
    row0: Int32,
    n_rows: Int32,
    row_lambda: MutPointer[Float32, MutAnyOrigin],
    row_death: MutPointer[Float32, MutAnyOrigin],
):
    """For a query (`soft_clustering.cuh:564-573`, `:298-303`): the lambda
    `min(prediction_lambda, lambdas[index_into_children[neighbor]])` and
    the death of the neighbor's parent cluster. For a training row
    (`:187-191`, `:245-246`): the row's own lambda and its parent's death."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows):
        return
    var point: Int
    if mode == SOFT_MODE_PREDICT:
        point = Int(min_mr_inds.unsafe_load(idx))
    else:
        point = Int(row0) + idx
    var edge = Int(index_into_children.unsafe_load(point))
    var lam = lambdas.unsafe_load(edge)
    if mode == SOFT_MODE_PREDICT:
        var pl = prediction_lambdas.unsafe_load(idx)
        if pl < lam:
            lam = pl
    row_lambda.unsafe_store(idx, lam)
    row_death.unsafe_store(
        idx, deaths.unsafe_load(Int(parents.unsafe_load(edge)) - Int(n_leaves))
    )


def soft_merge_height_kernel(
    mode: Int32,
    min_mr_inds: MutPointer[Int32, MutAnyOrigin],
    parents: MutPointer[Int32, MutAnyOrigin],
    index_into_children: MutPointer[Int32, MutAnyOrigin],
    lambdas: MutPointer[Float32, MutAnyOrigin],
    selected_clusters: MutPointer[Int32, MutAnyOrigin],
    row_lambda: MutPointer[Float32, MutAnyOrigin],
    row0: Int32,
    n_rows: Int32,
    n_selected: Int32,
    heights: MutPointer[Float32, MutAnyOrigin],
):
    """`merge_height_kernel`, both overloads (`kernels/soft_clustering.cuh:
    14-98`), one thread per row instead of per cell: integer walks and
    copies, no arithmetic."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows):
        return
    var point: Int
    if mode == SOFT_MODE_PREDICT:
        point = Int(min_mr_inds.unsafe_load(idx))
    else:
        point = Int(row0) + idx
    var ns = Int(n_selected)
    var leaf_parent = parents.unsafe_load(
        Int(index_into_children.unsafe_load(point))
    )
    for c in range(ns):
        var right_cluster = selected_clusters.unsafe_load(c)
        var left_cluster = leaf_parent
        var took_right = False
        var took_left = False
        var last_cluster = Int32(0)
        while left_cluster != right_cluster:
            if left_cluster > right_cluster:
                took_left = True
                last_cluster = left_cluster
                left_cluster = parents.unsafe_load(
                    Int(index_into_children.unsafe_load(Int(left_cluster)))
                )
            else:
                took_right = True
                last_cluster = right_cluster
                right_cluster = parents.unsafe_load(
                    Int(index_into_children.unsafe_load(Int(right_cluster)))
                )
        if took_left and took_right:
            heights.unsafe_store(
                idx * ns + c,
                lambdas.unsafe_load(
                    Int(index_into_children.unsafe_load(Int(last_cluster)))
                ),
            )
        else:
            heights.unsafe_store(idx * ns + c, row_lambda.unsafe_load(idx))


def soft_outlier_kernel(
    mode: Int32,
    heights: MutPointer[Float32, MutAnyOrigin],
    row_death: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_selected: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`outlier_membership_vector`'s `vec / (vec - mat)` (`:305-316`) or
    `all_points_outlier_membership_vector`'s `exp(-(vec + 1e-8) / mat)`
    (`:193-202`), then `Utils::softmax` (`utils.h:206-229`, `exp(x -
    max|x|)`, every entry here non-negative) and `normalize`. DEVIATION
    1616 (b) and the saturation."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows):
        return
    var ns = Int(n_selected)
    var base = idx * ns
    var vec = row_death.unsafe_load(idx)
    for c in range(ns):
        var h = heights.unsafe_load(base + c)
        var o: Float32
        if mode == SOFT_MODE_PREDICT:
            var den = ftz(vec - h)
            if den <= Float32(0.0):
                den = SOFT_EPS
            o = identical_div(vec, den)
            if o > SOFT_FLOAT32_MAX:
                o = SOFT_FLOAT32_MAX
        else:
            var t = identical_div(ftz(vec + SOFT_EPS), h)
            if t > SOFT_FLOAT32_MAX:
                t = SOFT_FLOAT32_MAX
            o = identical_exp(-t)
        dst.unsafe_store(base + c, o)
    var mx = dst.unsafe_load(base)
    for c in range(1, ns):
        var v = dst.unsafe_load(base + c)
        if v > mx:
            mx = v
    for c in range(ns):
        dst.unsafe_store(
            base + c, identical_exp(ftz(dst.unsafe_load(base + c) - mx))
        )
    soft_normalize_row(dst, base, ns)


def soft_prob_kernel(
    mode: Int32,
    heights: MutPointer[Float32, MutAnyOrigin],
    row_lambda: MutPointer[Float32, MutAnyOrigin],
    deaths: MutPointer[Float32, MutAnyOrigin],
    selected_clusters: MutPointer[Int32, MutAnyOrigin],
    n_leaves: Int32,
    n_rows: Int32,
    n_selected: Int32,
    prob: MutPointer[Float32, MutAnyOrigin],
):
    """`prob_in_some_cluster` (`:353-368`) or
    `all_points_prob_in_some_cluster` (`:236-252`): the merge height at the
    row's argmax (first column on a tie) over `max(lambda, death)`, plus
    the float32 `1e-8` for a query. DEVIATION 1616 (d)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows):
        return
    var ns = Int(n_selected)
    var base = idx * ns
    var best = 0
    var bh = heights.unsafe_load(base)
    for c in range(1, ns):
        var h = heights.unsafe_load(base + c)
        if h > bh:
            bh = h
            best = c
    var death = deaths.unsafe_load(
        Int(selected_clusters.unsafe_load(best)) - Int(n_leaves)
    )
    var ml = row_lambda.unsafe_load(idx)
    if ml < death:
        ml = death
    if mode == SOFT_MODE_PREDICT:
        ml = ftz(ml + SOFT_EPS)
    if ml > Float32(0.0):
        prob.unsafe_store(idx, identical_div(bh, ml))
    else:
        prob.unsafe_store(idx, Float32(0.0))


def soft_combine_kernel(
    mode: Int32,
    outlier: MutPointer[Float32, MutAnyOrigin],
    dist_membership: MutPointer[Float32, MutAnyOrigin],
    prob: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_selected: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`combine_op` (`:590-598`, `pow(m, 2) * pow(d, 0.5)`) or the product
    at `:462-467`, then `normalize` and the product with the probability
    of being in some cluster (`:619-626`, `:473-480`). DEVIATION 1616 (c)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows):
        return
    var ns = Int(n_selected)
    var base = idx * ns
    for c in range(ns):
        var m = outlier.unsafe_load(base + c)
        var dm = dist_membership.unsafe_load(base + c)
        var v: Float32
        if mode == SOFT_MODE_PREDICT:
            var m2 = ftz(identical_mul(m, m))
            v = ftz(identical_mul(m2, ftz(identical_sqrt(dm))))
        else:
            v = ftz(identical_mul(dm, m))
        dst.unsafe_store(base + c, v)
    soft_normalize_row(dst, base, ns)
    var p = prob.unsafe_load(idx)
    for c in range(ns):
        dst.unsafe_store(
            base + c, ftz(identical_mul(dst.unsafe_load(base + c), p))
        )


def _soft_pass(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mode: Int32,
    rows: List[Float32],
    n_rows: Int,
    n: Int,
    x_host: List[Float32],
    m: Int,
    parents: List[Int32],
    tree_lambdas: List[Float32],
    pd: PredictionData,
    min_mr_inds: List[Int32],
    prediction_lambdas: List[Float32],
    row0: Int,
    tpb: Int,
) raises -> List[Float32]:
    """Steps 1 to 6 of the module docstring for `n_rows` rows."""
    var ns = pd.n_selected_clusters
    var n_ex = pd.n_exemplars
    var cells = n_rows * ns
    var blocks = (n_rows + tpb - 1) // tpb

    # the exemplars as a dense array (`soft_clustering.cuh:65-75`, copy_rows)
    var ex_host = List[Float32](capacity=n_ex * n)
    for j in range(n_ex):
        var r = Int(pd.exemplar_idx[j])
        for f in range(n):
            ex_host.append(x_host[r * n + f])

    var rows_buf = _upload_f32(ctx, rows, n_rows * n)
    var ex_buf = _upload_f32(ctx, ex_host, n_ex * n)
    var off_buf = _upload_i32(ctx, pd.exemplar_label_offsets, ns + 1)
    var row_norms = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var ex_norms = ctx.enqueue_create_buffer[DType.float32](n_ex)
    var min_dist = ctx.enqueue_create_buffer[DType.float32](cells)
    var dist_mv = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.synchronize()
    ctx.enqueue_function[soft_row_norm_kernel](
        rows_buf.unsafe_ptr(), Int32(n_rows), Int32(n), row_norms.unsafe_ptr(),
        grid_dim=(blocks, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[soft_row_norm_kernel](
        ex_buf.unsafe_ptr(), Int32(n_ex), Int32(n), ex_norms.unsafe_ptr(),
        grid_dim=((n_ex + tpb - 1) // tpb, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_function[exemplar_min_dist_kernel](
        rows_buf.unsafe_ptr(), row_norms.unsafe_ptr(), ex_buf.unsafe_ptr(),
        ex_norms.unsafe_ptr(), off_buf.unsafe_ptr(), Int32(n_rows), Int32(n),
        Int32(ns), min_dist.unsafe_ptr(),
        grid_dim=(blocks, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_function[dist_membership_kernel](
        min_dist.unsafe_ptr(), Int32(n_rows), Int32(ns), dist_mv.unsafe_ptr(),
        grid_dim=(blocks, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()

    # the tree
    var n_edges = pd.n_edges
    var par_buf = _upload_i32(ctx, parents, n_edges)
    var iic_buf = _upload_i32(ctx, pd.index_into_children, n_edges + 1)
    var lam_buf = _upload_f32(ctx, tree_lambdas, n_edges)
    var death_buf = _upload_f32(ctx, pd.deaths, pd.n_clusters)
    var sel_buf = _upload_i32(ctx, pd.selected_clusters, ns)
    var nq_inds = len(min_mr_inds)
    var inds_buf = _upload_i32(ctx, min_mr_inds, nq_inds)
    var pl_buf = _upload_f32(ctx, prediction_lambdas, len(prediction_lambdas))
    var row_lambda = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var row_death = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var heights = ctx.enqueue_create_buffer[DType.float32](cells)
    var outlier = ctx.enqueue_create_buffer[DType.float32](cells)
    var prob = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var out = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.synchronize()
    ctx.enqueue_function[soft_row_prep_kernel](
        mode, inds_buf.unsafe_ptr(), pl_buf.unsafe_ptr(), par_buf.unsafe_ptr(),
        iic_buf.unsafe_ptr(), lam_buf.unsafe_ptr(), death_buf.unsafe_ptr(),
        Int32(pd.n_leaves), Int32(row0), Int32(n_rows),
        row_lambda.unsafe_ptr(), row_death.unsafe_ptr(),
        grid_dim=(blocks, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_function[soft_merge_height_kernel](
        mode, inds_buf.unsafe_ptr(), par_buf.unsafe_ptr(), iic_buf.unsafe_ptr(),
        lam_buf.unsafe_ptr(), sel_buf.unsafe_ptr(), row_lambda.unsafe_ptr(),
        Int32(row0), Int32(n_rows), Int32(ns), heights.unsafe_ptr(),
        grid_dim=(blocks, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_function[soft_outlier_kernel](
        mode, heights.unsafe_ptr(), row_death.unsafe_ptr(), Int32(n_rows),
        Int32(ns), outlier.unsafe_ptr(),
        grid_dim=(blocks, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_function[soft_prob_kernel](
        mode, heights.unsafe_ptr(), row_lambda.unsafe_ptr(),
        death_buf.unsafe_ptr(), sel_buf.unsafe_ptr(), Int32(pd.n_leaves),
        Int32(n_rows), Int32(ns), prob.unsafe_ptr(),
        grid_dim=(blocks, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_function[soft_combine_kernel](
        mode, outlier.unsafe_ptr(), dist_mv.unsafe_ptr(), prob.unsafe_ptr(),
        Int32(n_rows), Int32(ns), out.unsafe_ptr(),
        grid_dim=(blocks, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()

    var h_dist_mv = _download_f32(ctx, dist_mv, cells)
    var h_heights = _download_f32(ctx, heights, cells)
    var h_outlier = _download_f32(ctx, outlier, cells)
    var h_prob = _download_f32(ctx, prob, n_rows)
    var h_out = _download_f32(ctx, out, cells)
    trace.record_list_f32("hdbscan.soft.dist_membership", h_dist_mv)
    trace.record_list_f32("hdbscan.soft.merge_heights", h_heights)
    trace.record_list_f32("hdbscan.soft.outlier_membership", h_outlier)
    trace.record_list_f32("hdbscan.soft.prob_in_some_cluster", h_prob)
    trace.record_list_f32("hdbscan.soft.membership", h_out)

    # [[mojo-buffer-freed-at-last-use]]: every buffer outlives the queue.
    _ = rows_buf^
    _ = ex_buf^
    _ = off_buf^
    _ = row_norms^
    _ = ex_norms^
    _ = min_dist^
    _ = dist_mv^
    _ = par_buf^
    _ = iic_buf^
    _ = lam_buf^
    _ = death_buf^
    _ = sel_buf^
    _ = inds_buf^
    _ = pl_buf^
    _ = row_lambda^
    _ = row_death^
    _ = heights^
    _ = outlier^
    _ = prob^
    _ = h_dist_mv^
    _ = h_heights^
    _ = h_outlier^
    _ = h_prob^
    _ = out^
    return h_out^


def membership_vector(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    x_host: List[Float32],
    m: Int,
    n: Int,
    input_core_dists: List[Float32],
    labels: List[Int32],
    parents: List[Int32],
    tree_lambdas: List[Float32],
    pd: PredictionData,
    queries: List[Float32],
    n_prediction_points: Int,
    min_samples: Int,
    tpb: Int = SOFT_TPB,
) raises -> List[Float32]:
    """`soft_clustering.cuh:501-627`. Returns `n_prediction_points x
    n_selected` row-major. `min_samples` is the estimator's, as
    `approximate_predict` takes it."""
    if n_prediction_points < 1:
        raise Error(
            "hdbscan.membership_vector: points_to_predict has no rows;"
            " refused by name"
        )
    if len(queries) < n_prediction_points * n or len(x_host) < m * n:
        raise Error(
            "hdbscan.membership_vector: a buffer is shorter than its shape;"
            " refused by name"
        )
    refuse_nonfinite_queries(queries, n_prediction_points, n)
    refuse_soft_clustering_inputs(
        parents, tree_lambdas, pd, m, "hdbscan.membership_vector"
    )
    trace.header(
        "hdbscan/impl/detail/soft_clustering.mojo membership_vector n_rows="
        + String(m) + " n_cols=" + String(n) + " n_prediction_points="
        + String(n_prediction_points) + " n_selected="
        + String(pd.n_selected_clusters) + " n_exemplars="
        + String(pd.n_exemplars)
    )
    # `:554-562` _compute_knn_and_nearest_neighbor, through approximate_predict
    var near = approximate_predict(
        ctx, trace, x_host, m, n, input_core_dists, labels, tree_lambdas, pd,
        queries, n_prediction_points, min_samples, tpb,
    )
    return _soft_pass(
        ctx, trace, SOFT_MODE_PREDICT, queries, n_prediction_points, n,
        x_host, m, parents, tree_lambdas, pd, near.min_mr_indices,
        near.prediction_lambdas, 0, tpb,
    )


def all_points_membership_vectors(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    x_host: List[Float32],
    m: Int,
    n: Int,
    parents: List[Int32],
    tree_lambdas: List[Float32],
    pd: PredictionData,
    row0: Int,
    n_rows: Int,
    tpb: Int = SOFT_TPB,
) raises -> List[Float32]:
    """`soft_clustering.cuh:385-482` for training rows `row0 .. row0 +
    n_rows - 1`. Returns `n_rows x n_selected` row-major. The rows are
    independent, so any split of `0 .. m - 1` gives the same bytes."""
    if n_rows < 1 or row0 < 0 or row0 + n_rows > m:
        raise Error(
            "hdbscan.all_points_membership_vectors: rows " + String(row0)
            + " + " + String(n_rows) + " are outside the " + String(m)
            + " training rows; refused by name"
        )
    if len(x_host) < m * n:
        raise Error(
            "hdbscan.all_points_membership_vectors: the training matrix is"
            " shorter than its shape; refused by name"
        )
    refuse_soft_clustering_inputs(
        parents, tree_lambdas, pd, m, "hdbscan.all_points_membership_vectors"
    )
    trace.header(
        "hdbscan/impl/detail/soft_clustering.mojo all_points_membership_vectors"
        " n_rows=" + String(m) + " n_cols=" + String(n) + " row0="
        + String(row0) + " count=" + String(n_rows) + " n_selected="
        + String(pd.n_selected_clusters) + " n_exemplars="
        + String(pd.n_exemplars)
    )
    var rows = List[Float32](capacity=n_rows * n)
    for i in range(n_rows * n):
        rows.append(x_host[row0 * n + i])
    return _soft_pass(
        ctx, trace, SOFT_MODE_ALL_POINTS, rows, n_rows, n, x_host, m, parents,
        tree_lambdas, pd, List[Int32](length=1, fill=Int32(0)),
        List[Float32](length=1, fill=Float32(0.0)), row0, tpb,
    )
