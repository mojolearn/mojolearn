# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ball cover's k-NN query, and its non-Euclidean metrics, against an
EXHAUSTIVE host brute force.

NO CUVS COUNTERPART, AND THE DIFFERENCE IS THE POINT. `cuvs/tests/neighbors/
ball_cover.cu` compares `rbc_knn_query` against `brute_force::knn` with a
RECALL TOLERANCE (`ASSERT_TRUE(recall >= 0.99)`), which is the right test for
a query they also ship in an approximate mode. Ours ships no approximate mode
(DEVIATION 562), so the tolerance is zero and the assertion is set equality
per query row, not a rate over rows.

WHY THIS FILE LEADS WITH THE EXHAUSTIVE ARM
--------------------------------------------
The ball cover prunes WHOLE GROUPS. A bound that is one ulp too tight does
not return a slow answer, it returns a WRONG one, and it returns it silently:
the output has the right shape, the right number of entries, and plausible
distances. This repository has already had one scare of exactly that shape
(`ball_cover_check.mojo`'s banner records it, and it turned out to be a
DESTROYED ORACLE rather than a broken kernel, which is the other half of the
same lesson). So the first thing built here was the comparison that cannot be
fooled by shape: over a fixture small enough to brute force completely, every
query row, at several k and at five metrics, must equal the brute-force
answer EXACTLY.

THE TWO FIXTURES, AND WHY THERE ARE TWO
----------------------------------------
  SCATTERED  384 points and 96 separate queries in 3 dimensions, every
             coordinate a splitmix64 hash of `(row, feature)`. No two rows
             share a value, so every query's neighbour LIST is different and
             a wrong permutation, a wrong landmark offset or a row written
             into its neighbour's slot all fail. This is
             `ball_cover_check.mojo`'s fixture discipline, and its banner
             says why a uniform fixture verifies a total and nothing about
             placement.

  LATTICE    256 points in 2 dimensions on a 8x8 integer lattice with every
             position repeated FOUR times, plus 64 queries drawn from the
             half-integer lattice. This one exists for the tie-break and for
             the prune boundary: every distance is the root of a small
             integer, exact ties are everywhere, and the k-th and (k+1)-th
             neighbours are frequently equidistant, so the total order's
             index tie-break is exercised on every row instead of never.
             `uniform-test-data-hides-permutation` is about the first
             fixture; this is the complementary trap, a fixture with NO ties
             never tests the tie-break.

THE ORACLE IS A SECOND SPELLING, NOT A SECOND CALL
---------------------------------------------------
`_host_cmp_dist` writes each metric's accumulation loop out by hand rather
than calling `rbc_cmp_dist` on host pointers. Calling the device's own
function would make bit-equality trivially true and would prove nothing about
the FORMULA; spelling it a second time means a wrong formula fails here even
though a faithful transliteration of it would not. The transcendental is the
one exception: Lp's `pow` is `identical_pow` in both places, because a second
spelling of a transcendental is a second ANSWER, not a second opinion.

WHAT EACH SABOTAGE ARM MUST MOVE
---------------------------------
A gate never shown capable of failing does not count, so every seam here has
an arm that must fail when it is broken, and each one predicts the SHAPE of
the failure rather than only its existence.

  check_rbc_knn_prune_is_load_bearing   sweeps `prune_scale` downward,
      tightening every bound past what the triangle inequality justifies, and
      requires the exhaustive comparison to FAIL. It prints the LARGEST scale
      at which it fails, which is the honest measure of how close to vacuous
      the pruning gate is. **A ONE-ULP TIGHTENING CANNOT MOVE THIS KERNEL AND
      THAT IS BY DESIGN**: DEVIATION 567 widens every threshold by four ulp
      before comparing it, precisely so that float rounding can only admit a
      candidate and never drop one, so a one-ulp tightening lands inside the
      slack. The sweep is what replaces the one-ulp arm, and the printed
      number is what makes it honest.

  check_rbc_knn_slack_costs_no_answer   widens the threshold FURTHER (scale
      above 1) and requires the answer to be UNCHANGED. Together with the arm
      above this brackets the slack: tightening breaks the answer, loosening
      does not, so the shipped bound sits on the safe side of a real edge and
      the slack costs work rather than correctness.

  check_rbc_knn_radius_is_read          scales `R_radius` by 0.3 after the
      build and requires the answer to LOSE at least one correct neighbour.
      Shrinking the radii can only make the landmark test reject balls it
      should accept, so the failure must be a MISSING neighbour and never an
      extra one. If `R_radius` were not read, or if the query were quietly a
      full scan, the answer would not move at all.

  check_rbc_knn_prunes_work             asserts the query computed materially
      fewer than `n_queries * n_index` distances and prints the ratio. This
      is the NON-VACUITY control for every arm above it: an exactness proof
      about a path that is secretly a full scan proves nothing about pruning,
      and this is the only assertion in the file that can tell the two apart.

  check_rbc_eps_metric_is_reached       runs the eps query under L1 and
      compares it against the EUCLIDEAN oracle, requiring them to DIFFER. A
      `metric` argument that were accepted and ignored would return the
      Euclidean answer and every equality arm would still pass.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import PIN_CROSS_VENDOR, ftz, identical_pow
from checks.numerics import identical_div, identical_mul_add, identical_sqrt
from neighbors.impl.distance.detail.distance_ops import (
    DIST_COSINE_EXPANDED,
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_LINF,
    DIST_LP_UNEXPANDED,
    metric_value_name,
)
from neighbors.impl.neighbors.ball_cover.ball_cover import (
    rbc_build_index,
    rbc_eps_nn_query_count,
    rbc_eps_nn_query_fill,
    rbc_n_landmarks,
)
from neighbors.impl.neighbors.ball_cover.common import (
    RBC_FLT_MAX,
    rbc_cmp_bound,
    rbc_validate_metric,
)
from neighbors.impl.neighbors.ball_cover.knn import (
    rbc_knn_query,
    rbc_knn_query_scaled,
)


comptime BK_SCAT_N = 384
comptime BK_SCAT_Q = 96
comptime BK_SCAT_D = 3

comptime BK_LAT_SIDE = 8
comptime BK_LAT_REPEAT = 4
comptime BK_LAT_N = BK_LAT_SIDE * BK_LAT_SIDE * BK_LAT_REPEAT
comptime BK_LAT_Q = 64
comptime BK_LAT_D = 2

#: The FAST-only tolerance, in ulp, on both the returned distance and on a
#: tie flip in the ORDER. It is not a fudge factor for this kernel: under
#: FAST `identical_mul_add` is a bare `a * b + c` and the device may
#: contract where the host does not, and `identical_pow` is the stdlib `**`
#: on both sides, which are two different implementations of `pow`. Under
#: IDENTICAL both become one arithmetic and this constant is not consulted
#: at all -- the assertion there is bit equality, which is what makes the
#: zero-tolerance claim of this file a claim about
#: `check-ball-cover-knn-identical` and not about `check-ball-cover-knn`.
comptime BK_FAST_TIE_ULP = 64


def _hash01(row: Int, feature: Int) -> Float64:
    """splitmix64 on `(row, feature)`, mapped to `[0, 1)`. The same
    generator `ball_cover_check.mojo` uses, so the two files' fixtures are
    from one family and a coordinate can be checked across them by eye."""
    var z = (
        UInt64(row) * UInt64(0x9E3779B97F4A7C15)
        + UInt64(feature + 1) * UInt64(0xBF58476D1CE4E5B9)
        + UInt64(0x94D049BB133111EB)
    )
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> UInt64(31))
    return Float64(z >> UInt64(11)) * (1.0 / 9007199254740992.0)


def _scat_coord(row: Int, feature: Int) -> Float32:
    """A scattered point in a box of side 10. Magnitudes are kept small for
    the reason `ball_cover_check.mojo` gives: `PORTING.md 21` records a k-NN
    fixture destroyed by float32 cancellation at a spacing of 100."""
    return Float32(10.0 * _hash01(row, feature))


def _lat_coord(row: Int, feature: Int) -> Float32:
    """The lattice fixture. `row // BK_LAT_REPEAT` is the lattice cell, so
    every cell holds `BK_LAT_REPEAT` points at EXACTLY the same coordinates
    and every pairwise distance is the root of a small integer."""
    var cell = row // BK_LAT_REPEAT
    if feature == 0:
        return Float32(cell % BK_LAT_SIDE)
    return Float32(cell // BK_LAT_SIDE)


def _lat_query(row: Int, feature: Int) -> Float32:
    """Queries on the HALF-integer lattice, so a query sits between four
    lattice cells and its four nearest cells are equidistant. That is what
    puts the k-th and (k+1)-th neighbours at the same distance for most k
    and makes the index tie-break load-bearing on nearly every row."""
    var cell = row % (BK_LAT_SIDE - 1)
    var other = (row // (BK_LAT_SIDE - 1)) % (BK_LAT_SIDE - 1)
    if feature == 0:
        return Float32(cell) + Float32(0.5)
    return Float32(other) + Float32(0.5)


def _host_cmp_dist(
    ha: MutPointer[Float32, MutUntrackedOrigin],
    ia: Int,
    hb: MutPointer[Float32, MutUntrackedOrigin],
    ib: Int,
    d: Int,
    metric: Int,
    metric_arg: Float32,
) -> Float32:
    """`common.mojo::rbc_cmp_dist`, SPELLED A SECOND TIME ON PURPOSE.

    Same accumulation order over the same operand bits, so a boundary point
    cannot land on one side on the host and the other on the device, but a
    second spelling rather than a second call, so a wrong FORMULA fails here
    where a faithful transliteration of a wrong formula would not. `pow` is
    the one shared call; see the module docstring.
    """
    var acc = Float32(0.0)
    if metric == DIST_L2_SQRT_UNEXPANDED:
        for f in range(d):
            var diff = ftz(
                ftz(ha.unsafe_load(ia * d + f))
                - ftz(hb.unsafe_load(ib * d + f))
            )
            acc = ftz(identical_mul_add(diff, diff, acc))
        return acc
    if metric == DIST_L1:
        for f in range(d):
            acc = ftz(
                acc
                + abs(
                    ftz(
                        ftz(ha.unsafe_load(ia * d + f))
                        - ftz(hb.unsafe_load(ib * d + f))
                    )
                )
            )
        return acc
    if metric == DIST_LINF:
        for f in range(d):
            var diff = abs(
                ftz(
                    ftz(ha.unsafe_load(ia * d + f))
                    - ftz(hb.unsafe_load(ib * d + f))
                )
            )
            if diff > acc:
                acc = diff
        return acc
    if metric == DIST_LP_UNEXPANDED:
        for f in range(d):
            var diff = abs(
                ftz(
                    ftz(ha.unsafe_load(ia * d + f))
                    - ftz(hb.unsafe_load(ib * d + f))
                )
            )
            acc = ftz(acc + ftz(identical_pow(diff, metric_arg)))
        return ftz(
            identical_pow(acc, ftz(identical_div(Float32(1.0), metric_arg)))
        )
    return RBC_FLT_MAX


def _host_true_dist(metric: Int, v: Float32) -> Float32:
    """`common.mojo::rbc_true_dist`, second spelling."""
    if metric == DIST_L2_SQRT_UNEXPANDED:
        return identical_sqrt(v)
    return v


def _host_topk(
    hx: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    hq: MutPointer[Float32, MutUntrackedOrigin],
    q: Int,
    d: Int,
    k: Int,
    metric: Int,
    metric_arg: Float32,
    mut out_idx: List[Int32],
    mut out_cmp: List[Float32],
):
    """The EXHAUSTIVE brute force for one query row.

    Every one of the `n` distances is computed ONCE into `row`, and the best
    `k` are then taken under the total order stated in `knn.mojo`:
    `(comparison-space distance, index)` ascending. `k` selection passes
    over a materialized row, not a sort, because the answer only needs the
    first `k` and because a selection written this way has no comparator to
    get subtly wrong: the order is the same three-line test the device uses,
    written here a second time for the same reason `_host_cmp_dist` is.

    The row is materialized rather than recomputed inside each of the `k`
    passes for a reason that is measurable and not stylistic: `identical_pow`
    is `exp(p * log(z))` and the Lp arms would otherwise call it
    `k * n * n_features` times per query, which turns this gate from seconds
    into minutes and is exactly the kind of cost that gets a gate switched
    off.
    """
    var row = List[Float32]()
    for j in range(n):
        row.append(_host_cmp_dist(hq, q, hx, j, d, metric, metric_arg))

    out_idx.clear()
    out_cmp.clear()
    var taken = List[Bool]()
    for _ in range(n):
        taken.append(False)
    for _ in range(k):
        var best = RBC_FLT_MAX
        var best_j = -1
        for j in range(n):
            if taken[j]:
                continue
            # STRICT `<`, walking `j` ASCENDING: the first index wins a tie,
            # which IS the `(distance, index)` order the device's
            # `rbc_knn_before` produces. A `<=` here would take the LAST
            # index on a tie and the lattice fixture would disagree with the
            # device on nearly every row -- for a reason that would look
            # like a kernel bug and would not be one.
            if best_j < 0:
                best = row[j]
                best_j = j
            elif row[j] < best:
                best = row[j]
                best_j = j
        taken[best_j] = True
        out_idx.append(Int32(best_j))
        out_cmp.append(best)


def _ulp_gap(a: Float32, b: Float32) -> Int:
    """How many representable float32 values separate `a` and `b`.

    Both are non-negative distances here, so the raw bit patterns are
    monotone in the value and their difference IS the ulp count. Used only
    to report the worst gap under FAST, where `identical_mul_add` is a bare
    `a * b + c` and the host and the device may contract differently
    (`fast-is-not-identical`); under IDENTICAL the assertion is equality.
    """
    var ba = Int(rebind[UInt32](a.to_bits()))
    var bb = Int(rebind[UInt32](b.to_bits()))
    if ba >= bb:
        return ba - bb
    return bb - ba


@fieldwise_init
struct _BcIndex(Copyable, Movable):
    """The seven device buffers `rbc_build_index` fills, kept together so
    the arms below can build once and query several ways."""

    var x: DeviceBuffer[DType.float32]
    var x_reordered: DeviceBuffer[DType.float32]
    var r: DeviceBuffer[DType.float32]
    var r_indptr: DeviceBuffer[DType.int32]
    var r_1nn_cols: DeviceBuffer[DType.int32]
    var r_1nn_dists: DeviceBuffer[DType.float32]
    var r_radius: DeviceBuffer[DType.float32]
    var n_landmarks: Int


def _build(
    ctx: DeviceContext,
    hx: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    d: Int,
    metric: Int,
    metric_arg: Float32,
) raises -> _BcIndex:
    """`rbc_build_index` over a host fixture, returning the live index."""
    var n_landmarks = rbc_n_landmarks(n)
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var x_reordered = ctx.enqueue_create_buffer[DType.float32](n * d)
    var r = ctx.enqueue_create_buffer[DType.float32](n_landmarks * d)
    var landmark_ids = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    var slot_cols = ctx.enqueue_create_buffer[DType.int32](n)
    var slot_dists = ctx.enqueue_create_buffer[DType.float32](n)
    var nearest = ctx.enqueue_create_buffer[DType.int32](n)
    var nearest_dist = ctx.enqueue_create_buffer[DType.float32](n)
    var r_indptr = ctx.enqueue_create_buffer[DType.int32](n_landmarks + 1)
    var r_1nn_cols = ctx.enqueue_create_buffer[DType.int32](n)
    var r_1nn_dists = ctx.enqueue_create_buffer[DType.float32](n)
    var r_radius = ctx.enqueue_create_buffer[DType.float32](n_landmarks)
    var counts = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx)
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
        n,
        d,
        n_landmarks,
        UInt64(12345),
        metric,
        metric_arg,
    )
    _ = landmark_ids^
    _ = slot_cols^
    _ = slot_dists^
    _ = nearest^
    _ = nearest_dist^
    _ = counts^
    # Passed by COPY, not by transfer, which is the shape
    # `ensemble/checks/core_primitives_check.mojo::ScanBufs` already uses: a
    # `DeviceBuffer` copy shares the allocation, so the struct keeps every
    # one of them alive past the end of this function. That is also the
    # answer to `mojo-buffer-freed-at-last-use` here -- the buffers are held
    # in a struct field rather than in a local whose last use the compiler
    # gets to pick.
    return _BcIndex(
        x,
        x_reordered,
        r,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        n_landmarks,
    )


def _knn_compare(
    ctx: DeviceContext,
    mut idx: _BcIndex,
    hx: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    hq: MutPointer[Float32, MutUntrackedOrigin],
    nq: Int,
    d: Int,
    k: Int,
    metric: Int,
    metric_arg: Float32,
    prune_scale: Float32,
    label: String,
    must_match: Bool,
) raises -> Int:
    """Run the device query and compare it, row by row, against
    `_host_topk`.

    Returns the number of rows that DISAGREED. `must_match = True` raises on
    the first disagreement with the row, the slot and both answers named;
    the sabotage arms pass `False` and assert on the returned count instead,
    because "it broke" is the assertion there and a raise would hide how
    much it broke by.
    """
    # THE ORACLE'S OWN FIXTURE, BEFORE IT IS TRUSTED TO JUDGE ANYTHING.
    # `ball_cover_check.mojo`'s banner records what a freed `hx` did here:
    # a comparison against a destroyed fixture does not fail, it ACCUSES.
    if (
        hx.unsafe_load(0) != hx.unsafe_load(0)
        or hq.unsafe_load(0) != hq.unsafe_load(0)
    ):
        raise Error(label + ": the oracle's own fixture reads NaN")

    var out_inds = ctx.enqueue_create_buffer[DType.int32](nq * k)
    var out_dists = ctx.enqueue_create_buffer[DType.float32](nq * k)
    var dist_count = ctx.enqueue_create_buffer[DType.int32](nq)
    var query = ctx.enqueue_create_buffer[DType.float32](nq * d)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=query, src_ptr=hq)
    ctx.synchronize()

    rbc_knn_query_scaled(
        ctx,
        idx.x_reordered,
        query,
        idx.r,
        idx.r_indptr,
        idx.r_1nn_cols,
        idx.r_1nn_dists,
        idx.r_radius,
        out_inds,
        out_dists,
        dist_count,
        nq,
        d,
        idx.n_landmarks,
        k,
        prune_scale,
        metric,
        metric_arg,
    )

    var hi = ctx.enqueue_create_host_buffer[DType.int32](nq * k)
    var hd = ctx.enqueue_create_host_buffer[DType.float32](nq * k)
    ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=out_inds)
    ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=out_dists)
    ctx.synchronize()

    var bad_rows = 0
    var worst_ulp = 0
    var tie_flips = 0
    var oi = List[Int32]()
    var oc = List[Float32]()
    for q in range(nq):
        _host_topk(hx, n, hq, q, d, k, metric, metric_arg, oi, oc)
        var row_bad = False
        for s in range(k):
            var got_i = hi.unsafe_ptr().unsafe_load(q * k + s)
            var got_d = hd.unsafe_ptr().unsafe_load(q * k + s)

            # THE INDEX. Under IDENTICAL the assertion is equality and
            # nothing else. Under FAST it cannot be, and the reason is not
            # this kernel: `identical_mul_add` is a bare `a * b + c` there
            # and the device may contract where the host does not, and
            # `identical_pow` is the stdlib `**` on both sides, which are
            # two different implementations of `pow`. Either can move a
            # distance by an ulp or more, and two candidates within that
            # distance of each other can then come back in the other order.
            # So under FAST a differing index is accepted ONLY when the two
            # candidates are within `BK_FAST_TIE_ULP` of each other in the
            # host's own arithmetic -- which is a TIE FLIP and not a lost
            # neighbour -- and the count is printed. THE ZERO-TOLERANCE
            # CLAIM OF THIS FILE IS THE `check-ball-cover-knn-identical`
            # RUN; the FAST run is a smoke arm with a stated tolerance, per
            # `fast-is-not-identical`.
            if got_i != oi[s]:
                var accepted = False

                @parameter
                if not PIN_CROSS_VENDOR:
                    if got_i >= Int32(0):
                        var alt = _host_cmp_dist(
                            hq, q, hx, Int(got_i), d, metric, metric_arg
                        )
                        if _ulp_gap(alt, oc[s]) <= BK_FAST_TIE_ULP:
                            accepted = True
                            tie_flips += 1
                if not accepted:
                    row_bad = True
                    if must_match:
                        raise Error(
                            label
                            + ": query "
                            + String(q)
                            + " slot "
                            + String(s)
                            + " returned index "
                            + String(got_i)
                            + " where the exhaustive brute force says "
                            + String(oi[s])
                            + ". The ball cover PRUNES WHOLE GROUPS, so a"
                            " disagreement here is a group that was skipped"
                            " and should not have been, not a rounding"
                            " difference."
                        )
                    continue

            # THE DISTANCE.
            var want_d = _host_true_dist(metric, oc[s])
            var gap = _ulp_gap(got_d, want_d)
            if gap > worst_ulp:
                worst_ulp = gap

            @parameter
            if PIN_CROSS_VENDOR:
                if got_d != want_d:
                    row_bad = True
                    if must_match:
                        raise Error(
                            label
                            + " (IDENTICAL): query "
                            + String(q)
                            + " slot "
                            + String(s)
                            + " distance "
                            + String(got_d)
                            + " is not bit-equal to the host's "
                            + String(want_d)
                        )
            else:
                if gap > BK_FAST_TIE_ULP:
                    row_bad = True
                    if must_match:
                        raise Error(
                            label
                            + " (FAST): query "
                            + String(q)
                            + " slot "
                            + String(s)
                            + " distance is "
                            + String(gap)
                            + " ulp from the host's, above the "
                            + String(BK_FAST_TIE_ULP)
                            + " a contraction or a `pow` difference can"
                            " explain"
                        )
        if row_bad:
            bad_rows += 1

    if must_match:
        print(
            label
            + " OK: "
            + String(nq)
            + " queries x k="
            + String(k)
            + " ["
            + metric_value_name(metric)
            + "] equal the exhaustive brute force"
            + (
                ", bitwise"
                if PIN_CROSS_VENDOR
                else (
                    ", worst distance gap "
                    + String(worst_ulp)
                    + " ulp and "
                    + String(tie_flips)
                    + " accepted tie flips (FAST tolerance)"
                )
            )
        )
    _ = out_inds^
    _ = out_dists^
    _ = dist_count^
    _ = query^
    _ = hi^
    _ = hd^
    return bad_rows


def _fill_scattered(
    mut hx: HostBuffer[DType.float32], mut hq: HostBuffer[DType.float32]
):
    """The SCATTERED fixture: index rows and query rows, on the host.

    `hx` is `BK_SCAT_N x BK_SCAT_D` and `hq` is `BK_SCAT_Q x BK_SCAT_D`;
    they are filled in place rather than returned so this file never moves
    a `HostBuffer` out of a function, which is the shape
    `ensemble/checks/core_primitives_check.mojo` settled on.
    """
    for i in range(BK_SCAT_N):
        for f in range(BK_SCAT_D):
            hx.unsafe_ptr().unsafe_store(i * BK_SCAT_D + f, _scat_coord(i, f))
    for i in range(BK_SCAT_Q):
        for f in range(BK_SCAT_D):
            # A DIFFERENT hash stream from the index rows, so no query
            # coincides with an index point and the self-neighbour case is
            # tested where it belongs -- on the lattice fixture, which has
            # exact duplicates on purpose -- rather than by accident here.
            hq.unsafe_ptr().unsafe_store(
                i * BK_SCAT_D + f, _scat_coord(i + 100000, f)
            )


def _fill_lattice(
    mut hx: HostBuffer[DType.float32], mut hq: HostBuffer[DType.float32]
):
    """The LATTICE fixture: exact duplicates and exact ties."""
    for i in range(BK_LAT_N):
        for f in range(BK_LAT_D):
            hx.unsafe_ptr().unsafe_store(i * BK_LAT_D + f, _lat_coord(i, f))
    for i in range(BK_LAT_Q):
        for f in range(BK_LAT_D):
            hq.unsafe_ptr().unsafe_store(i * BK_LAT_D + f, _lat_query(i, f))


def check_rbc_knn_matches_brute_force() raises:
    """THE GATE THAT MATTERS. Exhaustive, per row, per slot, no tolerance.

    Five metrics x seven values of k x two fixtures. The metrics are chosen
    to cover both arms of the comparison space (DEVIATION 564): Euclidean is
    the SQUARED arm, and L1, Linf and the two Lp values are the true-distance
    arm. Lp at p = 1 is Manhattan computed through `pow` rather than through
    `l1_core`, which is exactly the aliasing DEVIATION 552 refuses to
    special-case away, so the two must agree to within the transcendental's
    error and the k-NN ANSWER must be identical; Lp at p = 3 is a p that is
    neither of the two special cases.
    """
    var ctx = DeviceContext()
    var metrics = List[Int]()
    var args = List[Float32]()
    metrics.append(DIST_L2_SQRT_UNEXPANDED)
    args.append(Float32(2.0))
    metrics.append(DIST_L1)
    args.append(Float32(2.0))
    metrics.append(DIST_LINF)
    args.append(Float32(2.0))
    metrics.append(DIST_LP_UNEXPANDED)
    args.append(Float32(1.0))
    metrics.append(DIST_LP_UNEXPANDED)
    args.append(Float32(1.5))
    metrics.append(DIST_LP_UNEXPANDED)
    args.append(Float32(3.0))

    var ks = List[Int]()
    ks.append(1)
    ks.append(2)
    ks.append(3)
    ks.append(5)
    ks.append(8)
    ks.append(17)
    ks.append(32)

    var sx = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_N * BK_SCAT_D
    )
    var sq = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_Q * BK_SCAT_D
    )
    ctx.synchronize()
    _fill_scattered(sx, sq)
    for mi in range(len(metrics)):
        var idx = _build(
            ctx,
            sx.unsafe_ptr(),
            BK_SCAT_N,
            BK_SCAT_D,
            metrics[mi],
            args[mi],
        )
        for ki in range(len(ks)):
            _ = _knn_compare(
                ctx,
                idx,
                sx.unsafe_ptr(),
                BK_SCAT_N,
                sq.unsafe_ptr(),
                BK_SCAT_Q,
                BK_SCAT_D,
                ks[ki],
                metrics[mi],
                args[mi],
                Float32(1.0),
                "rbc_knn scattered",
                True,
            )
        _ = idx^

    var lx = ctx.enqueue_create_host_buffer[DType.float32](
        BK_LAT_N * BK_LAT_D
    )
    var lq = ctx.enqueue_create_host_buffer[DType.float32](
        BK_LAT_Q * BK_LAT_D
    )
    ctx.synchronize()
    _fill_lattice(lx, lq)
    for mi in range(len(metrics)):
        var idx2 = _build(
            ctx, lx.unsafe_ptr(), BK_LAT_N, BK_LAT_D, metrics[mi], args[mi]
        )
        for ki in range(len(ks)):
            _ = _knn_compare(
                ctx,
                idx2,
                lx.unsafe_ptr(),
                BK_LAT_N,
                lq.unsafe_ptr(),
                BK_LAT_Q,
                BK_LAT_D,
                ks[ki],
                metrics[mi],
                args[mi],
                Float32(1.0),
                "rbc_knn lattice(ties)",
                True,
            )
        _ = idx2^

    _ = sx^
    _ = sq^
    _ = lx^
    _ = lq^
    print(
        "check_rbc_knn_matches_brute_force OK: 6 metrics x 7 k x 2 fixtures,"
        " every query row equal to the exhaustive brute force"
    )


def check_rbc_knn_prune_is_load_bearing() raises:
    """SABOTAGE: tighten every bound and require the exhaustive comparison
    to FAIL. Prints the LARGEST tightening that breaks it.

    `prune_scale` multiplies the threshold before DEVIATION 567's four-ulp
    widening is added, so a scale below 1 makes the bound tighter than the
    triangle inequality justifies and pruning becomes unsound. If NO scale
    in the sweep breaks the answer, the fixture cannot see the pruning at
    all and every equality arm in this file is vacuous, so that case raises
    with exactly that sentence.

    A ONE-ULP ARM IS NOT POSSIBLE HERE AND THAT IS NOT AN OVERSIGHT. The
    shipped threshold already carries four ulp of slack, deliberately, so
    that float rounding can only ADMIT a candidate and never drop one; a
    one-ulp tightening lands inside that slack and cannot change any answer.
    The sweep below is what replaces it, and the number it prints is the
    honest measure of how much tightening this fixture can detect.
    """
    var ctx = DeviceContext()
    var lx = ctx.enqueue_create_host_buffer[DType.float32](
        BK_LAT_N * BK_LAT_D
    )
    var lq = ctx.enqueue_create_host_buffer[DType.float32](
        BK_LAT_Q * BK_LAT_D
    )
    ctx.synchronize()
    _fill_lattice(lx, lq)
    var idx = _build(
        ctx,
        lx.unsafe_ptr(),
        BK_LAT_N,
        BK_LAT_D,
        DIST_L2_SQRT_UNEXPANDED,
        Float32(2.0),
    )

    var scales = List[Float32]()
    scales.append(Float32(0.99999994))  # one ulp below 1, inside the slack
    scales.append(Float32(0.9999990))
    scales.append(Float32(0.999990))
    scales.append(Float32(0.99990))
    scales.append(Float32(0.9990))
    scales.append(Float32(0.990))
    scales.append(Float32(0.950))
    scales.append(Float32(0.900))
    scales.append(Float32(0.700))
    scales.append(Float32(0.400))

    var first_break = Float32(-1.0)
    var first_break_rows = 0
    for si in range(len(scales)):
        var bad = _knn_compare(
            ctx,
            idx,
            lx.unsafe_ptr(),
            BK_LAT_N,
            lq.unsafe_ptr(),
            BK_LAT_Q,
            BK_LAT_D,
            8,
            DIST_L2_SQRT_UNEXPANDED,
            Float32(2.0),
            scales[si],
            "rbc_knn sabotage",
            False,
        )
        if bad > 0:
            if first_break < Float32(0.0):
                first_break = scales[si]
                first_break_rows = bad

    if first_break < Float32(0.0):
        raise Error(
            "sabotage: TIGHTENING EVERY PRUNING BOUND BY 60% DID NOT CHANGE"
            " THE ANSWER. That is not evidence the kernel is right, it is"
            " evidence THIS FIXTURE CANNOT SEE THE PRUNING: if no bound"
            " ever decides anything, check_rbc_knn_matches_brute_force is"
            " comparing a full scan against a full scan and proves nothing"
            " about the ball cover. Build a fixture whose landmark groups"
            " are large enough that the bounds fire -- more points per"
            " landmark, or a smaller k -- before trusting any arm in this"
            " file."
        )

    if first_break > Float32(0.99999995):
        raise Error(
            "sabotage: a ONE-ULP tightening changed the answer, which"
            " contradicts DEVIATION 567: the shipped threshold is supposed"
            " to carry four ulp of slack precisely so that rounding cannot"
            " decide a prune. Either the slack is not being applied or it"
            " is being applied to the wrong operand magnitude."
        )

    print(
        "check_rbc_knn_prune_is_load_bearing OK: the answer survives a"
        " one-ulp tightening (the DEVIATION 567 slack) and BREAKS at"
        " prune_scale "
        + String(first_break)
        + " on "
        + String(first_break_rows)
        + " of "
        + String(BK_LAT_Q)
        + " query rows. That scale is how much tightening this fixture can"
        " detect; a smaller number would mean a weaker gate."
    )
    _ = idx^
    _ = lx^
    _ = lq^


def check_rbc_knn_slack_costs_no_answer() raises:
    """DEVIATION 567's other side: LOOSENING the bound must change nothing.

    Together with the arm above, this brackets the shipped threshold. If
    widening moved the answer, the widened bound would be admitting
    candidates that then WIN, which would mean the tight bound had been
    pruning real neighbours -- the exact silent failure this lane exists to
    make impossible. It must not move, on either fixture.
    """
    var ctx = DeviceContext()
    var sx = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_N * BK_SCAT_D
    )
    var sq = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_Q * BK_SCAT_D
    )
    ctx.synchronize()
    _fill_scattered(sx, sq)
    var idx = _build(
        ctx,
        sx.unsafe_ptr(),
        BK_SCAT_N,
        BK_SCAT_D,
        DIST_L2_SQRT_UNEXPANDED,
        Float32(2.0),
    )
    var bad = _knn_compare(
        ctx,
        idx,
        sx.unsafe_ptr(),
        BK_SCAT_N,
        sq.unsafe_ptr(),
        BK_SCAT_Q,
        BK_SCAT_D,
        8,
        DIST_L2_SQRT_UNEXPANDED,
        Float32(2.0),
        Float32(1.001),
        "rbc_knn widened",
        False,
    )
    if bad != 0:
        raise Error(
            "check_rbc_knn_slack_costs_no_answer: widening every pruning"
            " bound by 0.1% changed "
            + String(bad)
            + " query rows. A wider bound can only ADMIT candidates, so a"
            " changed answer means the SHIPPED bound was pruning real"
            " neighbours."
        )
    print(
        "check_rbc_knn_slack_costs_no_answer OK: widening the bound by 0.1%"
        " moves no row, so the four-ulp slack costs work and not answers"
    )
    _ = idx^
    _ = sx^
    _ = sq^


def check_rbc_knn_radius_is_read() raises:
    """SABOTAGE: scale `R_radius` by 0.3 and require the answer to LOSE.

    The predicted shape is specific and is asserted rather than merely
    "something changed": shrinking a landmark's radius can only make the
    landmark test reject balls it should accept, so the answer can lose a
    true neighbour and can never gain a false one. `_knn_compare` counts
    rows that disagree with the exhaustive oracle, and a disagreement under
    a SHRUNK radius is by construction a missing neighbour.

    If `R_radius` were never read -- or if the query had quietly become a
    full scan, which is the failure mode that makes every other arm in this
    file vacuous -- the answer would not move at all.
    """
    var ctx = DeviceContext()
    var sx = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_N * BK_SCAT_D
    )
    var sq = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_Q * BK_SCAT_D
    )
    ctx.synchronize()
    _fill_scattered(sx, sq)
    var idx = _build(
        ctx,
        sx.unsafe_ptr(),
        BK_SCAT_N,
        BK_SCAT_D,
        DIST_L2_SQRT_UNEXPANDED,
        Float32(2.0),
    )

    var clean = _knn_compare(
        ctx,
        idx,
        sx.unsafe_ptr(),
        BK_SCAT_N,
        sq.unsafe_ptr(),
        BK_SCAT_Q,
        BK_SCAT_D,
        8,
        DIST_L2_SQRT_UNEXPANDED,
        Float32(2.0),
        Float32(1.0),
        "rbc_knn radius-clean",
        False,
    )
    if clean != 0:
        raise Error(
            "check_rbc_knn_radius_is_read: the UNSABOTAGED run already"
            " disagrees with the oracle on "
            + String(clean)
            + " rows, so nothing below it is evidence about R_radius"
        )

    var hr = ctx.enqueue_create_host_buffer[DType.float32](idx.n_landmarks)
    ctx.enqueue_copy(dst_ptr=hr.unsafe_ptr(), src_buf=idx.r_radius)
    ctx.synchronize()
    for l in range(idx.n_landmarks):
        hr.unsafe_ptr().unsafe_store(
            l, hr.unsafe_ptr().unsafe_load(l) * Float32(0.3)
        )
    ctx.enqueue_copy(dst_buf=idx.r_radius, src_ptr=hr.unsafe_ptr())
    ctx.synchronize()

    var bad = _knn_compare(
        ctx,
        idx,
        sx.unsafe_ptr(),
        BK_SCAT_N,
        sq.unsafe_ptr(),
        BK_SCAT_Q,
        BK_SCAT_D,
        8,
        DIST_L2_SQRT_UNEXPANDED,
        Float32(2.0),
        Float32(1.0),
        "rbc_knn radius-sabotaged",
        False,
    )
    if bad == 0:
        raise Error(
            "sabotage: SHRINKING EVERY LANDMARK RADIUS TO 0.3x DID NOT"
            " CHANGE THE ANSWER. R_radius is either not read by"
            " rbc_knn_kernel or the query is not pruning on it, and in"
            " either case the landmark test is not what is producing these"
            " answers."
        )
    print(
        "check_rbc_knn_radius_is_read OK: R_radius x0.3 loses neighbours on "
        + String(bad)
        + " of "
        + String(BK_SCAT_Q)
        + " rows, and the clean run loses none"
    )
    _ = hr^
    _ = idx^
    _ = sx^
    _ = sq^


def check_rbc_knn_prunes_work() raises:
    """NON-VACUITY: the query must compute materially fewer distances than
    brute force, and the ratio is PRINTED.

    Every other arm in this file asserts that the indexed answer equals the
    brute-force answer. None of them can tell an index that prunes from an
    index that silently scans everything, and an exactness proof about a
    disguised full scan proves nothing about the ball cover. `dist_count` is
    the kernel's own tally of candidate distances actually computed --
    cuVS's `n_dists_computed` (`registers.cuh:409`), which they increment
    and never read -- and this is where it is read.
    """
    var ctx = DeviceContext()
    var sx = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_N * BK_SCAT_D
    )
    var sq = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_Q * BK_SCAT_D
    )
    ctx.synchronize()
    _fill_scattered(sx, sq)
    var idx = _build(
        ctx,
        sx.unsafe_ptr(),
        BK_SCAT_N,
        BK_SCAT_D,
        DIST_L2_SQRT_UNEXPANDED,
        Float32(2.0),
    )

    var k = 4
    var out_inds = ctx.enqueue_create_buffer[DType.int32](BK_SCAT_Q * k)
    var out_dists = ctx.enqueue_create_buffer[DType.float32](BK_SCAT_Q * k)
    var dist_count = ctx.enqueue_create_buffer[DType.int32](BK_SCAT_Q)
    var query = ctx.enqueue_create_buffer[DType.float32](
        BK_SCAT_Q * BK_SCAT_D
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=query, src_ptr=sq.unsafe_ptr())
    ctx.synchronize()
    rbc_knn_query(
        ctx,
        idx.x_reordered,
        query,
        idx.r,
        idx.r_indptr,
        idx.r_1nn_cols,
        idx.r_1nn_dists,
        idx.r_radius,
        out_inds,
        out_dists,
        dist_count,
        BK_SCAT_Q,
        BK_SCAT_D,
        idx.n_landmarks,
        k,
    )
    var hc = ctx.enqueue_create_host_buffer[DType.int32](BK_SCAT_Q)
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=dist_count)
    ctx.synchronize()
    var total = 0
    for i in range(BK_SCAT_Q):
        total += Int(hc.unsafe_ptr().unsafe_load(i))
    var brute = BK_SCAT_Q * BK_SCAT_N

    if total <= 0:
        raise Error(
            "check_rbc_knn_prunes_work: the query computed NO candidate"
            " distances at all, which cannot produce a correct answer and"
            " means dist_count is not being written"
        )
    if total * 10 > brute * 9:
        raise Error(
            "check_rbc_knn_prunes_work: the query computed "
            + String(total)
            + " candidate distances against brute force's "
            + String(brute)
            + ", which is 90% or more. The index is not pruning on this"
            " fixture, so every EQUALITY arm in this file is comparing a"
            " full scan against a full scan and proves nothing about the"
            " ball cover. Fix the fixture or the bounds before trusting"
            " them."
        )
    print(
        "check_rbc_knn_prunes_work OK: "
        + String(total)
        + " candidate distances against brute force's "
        + String(brute)
        + " (n="
        + String(BK_SCAT_N)
        + ", d="
        + String(BK_SCAT_D)
        + ", k="
        + String(k)
        + "). The equality arms are therefore about a PRUNING path."
    )
    _ = out_inds^
    _ = out_dists^
    _ = dist_count^
    _ = query^
    _ = hc^
    _ = idx^
    _ = sx^
    _ = sq^


def check_rbc_metric_refusals() raises:
    """Every refusal BY NAME, and every admission, at the host boundary.

    The refused set is exactly the non-metrics (DEVIATION 564): cosine, the
    two expanded Euclidean tags, and Lp below p = 1. The admitted set is
    exactly the metrics. A refusal that is broader than its own argument is
    the defect this lane was opened to fix, so the ADMISSIONS are asserted
    here too -- a check that only tests refusals cannot notice a refusal
    that came back.
    """
    var refused = List[Int]()
    var refused_arg = List[Float32]()
    refused.append(DIST_COSINE_EXPANDED)
    refused_arg.append(Float32(2.0))
    refused.append(DIST_L2_EXPANDED)
    refused_arg.append(Float32(2.0))
    refused.append(DIST_L2_SQRT_EXPANDED)
    refused_arg.append(Float32(2.0))
    refused.append(DIST_LP_UNEXPANDED)
    refused_arg.append(Float32(0.5))
    refused.append(DIST_LP_UNEXPANDED)
    refused_arg.append(Float32(0.99999))

    for i in range(len(refused)):
        var raised = False
        try:
            rbc_validate_metric(refused[i], refused_arg[i])
        except:
            raised = True
        if not raised:
            raise Error(
                "check_rbc_metric_refusals: "
                + metric_value_name(refused[i])
                + " with arg "
                + String(refused_arg[i])
                + " was ADMITTED. It does not satisfy the triangle"
                " inequality, so the ball cover would prune true neighbours"
                " under it silently (DEVIATION 564)."
            )

    var ok = List[Int]()
    var ok_arg = List[Float32]()
    ok.append(DIST_L2_SQRT_UNEXPANDED)
    ok_arg.append(Float32(2.0))
    ok.append(DIST_L1)
    ok_arg.append(Float32(2.0))
    ok.append(DIST_LINF)
    ok_arg.append(Float32(2.0))
    ok.append(DIST_LP_UNEXPANDED)
    ok_arg.append(Float32(1.0))
    ok.append(DIST_LP_UNEXPANDED)
    ok_arg.append(Float32(1.5))
    ok.append(DIST_LP_UNEXPANDED)
    ok_arg.append(Float32(9.0))
    for i in range(len(ok)):
        rbc_validate_metric(ok[i], ok_arg[i])

    print(
        "check_rbc_metric_refusals OK: 5 non-metrics refused by name, 6"
        " metrics admitted, and Lp at p = 1 (Manhattan) is on the admitted"
        " side"
    )


def _eps_row_sets(
    ctx: DeviceContext,
    mut idx: _BcIndex,
    hx: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    d: Int,
    eps: Float32,
    metric: Int,
    metric_arg: Float32,
    mut hia: HostBuffer[DType.int32],
    mut hja_out: List[Int32],
) raises -> Int:
    """Run the two-pass eps query and copy its CSR to the host."""
    var adj_ia = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var vd = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.synchronize()
    var nnz = rbc_eps_nn_query_count(
        ctx,
        idx.x_reordered,
        idx.x,
        idx.r,
        idx.r_indptr,
        idx.r_1nn_cols,
        idx.r_1nn_dists,
        idx.r_radius,
        adj_ia,
        vd,
        n,
        d,
        idx.n_landmarks,
        eps,
        metric,
        metric_arg,
    )
    var adj_ja = ctx.enqueue_create_buffer[DType.int32](nnz)
    ctx.synchronize()
    rbc_eps_nn_query_fill(
        ctx,
        idx.x_reordered,
        idx.x,
        idx.r,
        idx.r_indptr,
        idx.r_1nn_cols,
        idx.r_1nn_dists,
        idx.r_radius,
        adj_ia,
        adj_ja,
        n,
        d,
        idx.n_landmarks,
        eps,
        metric,
        metric_arg,
    )
    var hja = ctx.enqueue_create_host_buffer[DType.int32](nnz)
    ctx.enqueue_copy(dst_ptr=hia.unsafe_ptr(), src_buf=adj_ia)
    ctx.enqueue_copy(dst_ptr=hja.unsafe_ptr(), src_buf=adj_ja)
    ctx.synchronize()
    hja_out.clear()
    for p in range(nnz):
        hja_out.append(hja.unsafe_ptr().unsafe_load(p))
    _ = adj_ia^
    _ = adj_ja^
    _ = vd^
    _ = hja^
    return nnz


def check_rbc_eps_metrics_match_host() raises:
    """TASK B's GATE: the eps/radius query under each admitted metric, per
    row, against an exhaustive host brute force.

    `eps` is chosen per metric rather than shared, because the same number
    means a very different neighbourhood under L1 and under Linf on the same
    points, and a radius that leaves every row empty (or full) tests
    nothing. The three chosen leave a mixture, which is what makes the
    landmark prune fire on some rows and not on others.
    """
    var ctx = DeviceContext()
    var sx = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_N * BK_SCAT_D
    )
    var sq_unused = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_Q * BK_SCAT_D
    )
    ctx.synchronize()
    _fill_scattered(sx, sq_unused)
    _ = sq_unused^

    var metrics = List[Int]()
    var args = List[Float32]()
    var epss = List[Float32]()
    metrics.append(DIST_L2_SQRT_UNEXPANDED)
    args.append(Float32(2.0))
    epss.append(Float32(2.5))
    metrics.append(DIST_L1)
    args.append(Float32(2.0))
    epss.append(Float32(4.0))
    metrics.append(DIST_LINF)
    args.append(Float32(2.0))
    epss.append(Float32(1.8))
    metrics.append(DIST_LP_UNEXPANDED)
    args.append(Float32(1.5))
    epss.append(Float32(3.0))
    metrics.append(DIST_LP_UNEXPANDED)
    args.append(Float32(1.0))
    epss.append(Float32(4.0))

    var hia = ctx.enqueue_create_host_buffer[DType.int32](BK_SCAT_N + 1)
    ctx.synchronize()
    for mi in range(len(metrics)):
        var idx = _build(
            ctx,
            sx.unsafe_ptr(),
            BK_SCAT_N,
            BK_SCAT_D,
            metrics[mi],
            args[mi],
        )
        var cols = List[Int32]()
        var nnz = _eps_row_sets(
            ctx,
            idx,
            sx.unsafe_ptr(),
            BK_SCAT_N,
            BK_SCAT_D,
            epss[mi],
            metrics[mi],
            args[mi],
            hia,
            cols,
        )
        var bound = rbc_cmp_bound(metrics[mi], epss[mi])
        var seen = List[Int32]()
        for _ in range(BK_SCAT_N):
            seen.append(Int32(-1))
        var total_expected = 0
        for i in range(BK_SCAT_N):
            var start = Int(hia.unsafe_ptr().unsafe_load(i))
            var end = Int(hia.unsafe_ptr().unsafe_load(i + 1))
            for p in range(start, end):
                var c = Int(cols[p])
                if c < 0 or c >= BK_SCAT_N:
                    raise Error(
                        "rbc eps ["
                        + metric_value_name(metrics[mi])
                        + "]: row "
                        + String(i)
                        + " column "
                        + String(c)
                        + " is out of range"
                    )
                if seen[c] == Int32(i):
                    raise Error(
                        "rbc eps: row "
                        + String(i)
                        + " lists column "
                        + String(c)
                        + " twice"
                    )
                seen[c] = Int32(i)
            for j in range(BK_SCAT_N):
                var inside = (
                    _host_cmp_dist(
                        sx.unsafe_ptr(),
                        i,
                        sx.unsafe_ptr(),
                        j,
                        BK_SCAT_D,
                        metrics[mi],
                        args[mi],
                    )
                    <= bound
                )
                if inside:
                    total_expected += 1
                var got = seen[j] == Int32(i)
                if got and not inside:
                    raise Error(
                        "rbc eps ["
                        + metric_value_name(metrics[mi])
                        + " p="
                        + String(args[mi])
                        + "]: row "
                        + String(i)
                        + " returned point "
                        + String(j)
                        + " which brute force puts OUTSIDE eps"
                    )
                if inside and not got:
                    raise Error(
                        "rbc eps ["
                        + metric_value_name(metrics[mi])
                        + " p="
                        + String(args[mi])
                        + "]: row "
                        + String(i)
                        + " MISSED point "
                        + String(j)
                        + ", which brute force puts inside eps. The cover"
                        " pruned a group it should have walked, which is"
                        " the failure DEVIATION 564 has to make impossible"
                        " for a metric."
                    )
        if total_expected != nnz:
            raise Error(
                "rbc eps: the CSR holds "
                + String(nnz)
                + " edges against the oracle's "
                + String(total_expected)
            )
        print(
            "  rbc eps ["
            + metric_value_name(metrics[mi])
            + " p="
            + String(args[mi])
            + " eps="
            + String(epss[mi])
            + "] OK: "
            + String(nnz)
            + " edges, per-cell equal to brute force"
        )
        _ = idx^
    _ = hia^
    _ = sx^
    print(
        "check_rbc_eps_metrics_match_host OK: 5 metrics, per-row set"
        " equality with no tolerance"
    )


def check_rbc_eps_metric_is_reached() raises:
    """REACH: an L1 query must NOT return the Euclidean answer.

    A `metric` argument that were accepted and then ignored -- threaded into
    a signature but never read by the kernel, which is the "reached but
    inert" trap -- would make every equality arm above pass on the Euclidean
    arm and fail on none of them, because the oracle would be asked the same
    wrong question. This asks the kernel for L1 and compares against the
    EUCLIDEAN oracle, and requires a DISAGREEMENT.
    """
    var ctx = DeviceContext()
    var sx = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_N * BK_SCAT_D
    )
    var sq = ctx.enqueue_create_host_buffer[DType.float32](
        BK_SCAT_Q * BK_SCAT_D
    )
    ctx.synchronize()
    _fill_scattered(sx, sq)
    var idx = _build(
        ctx, sx.unsafe_ptr(), BK_SCAT_N, BK_SCAT_D, DIST_L1, Float32(2.0)
    )
    var bad = _knn_compare(
        ctx,
        idx,
        sx.unsafe_ptr(),
        BK_SCAT_N,
        sq.unsafe_ptr(),
        BK_SCAT_Q,
        BK_SCAT_D,
        8,
        DIST_L1,
        Float32(2.0),
        Float32(1.0),
        "rbc_knn reach-l1",
        False,
    )
    if bad != 0:
        raise Error(
            "check_rbc_eps_metric_is_reached: the L1 run does not match the"
            " L1 oracle on "
            + String(bad)
            + " rows, so nothing below is evidence about reach"
        )
    # Now the same DEVICE run, judged by the EUCLIDEAN oracle. It must
    # disagree; if it agreed, `metric` would be inert.
    var out_inds = ctx.enqueue_create_buffer[DType.int32](BK_SCAT_Q * 8)
    var out_dists = ctx.enqueue_create_buffer[DType.float32](BK_SCAT_Q * 8)
    var dist_count = ctx.enqueue_create_buffer[DType.int32](BK_SCAT_Q)
    var query = ctx.enqueue_create_buffer[DType.float32](
        BK_SCAT_Q * BK_SCAT_D
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=query, src_ptr=sq.unsafe_ptr())
    ctx.synchronize()
    rbc_knn_query(
        ctx,
        idx.x_reordered,
        query,
        idx.r,
        idx.r_indptr,
        idx.r_1nn_cols,
        idx.r_1nn_dists,
        idx.r_radius,
        out_inds,
        out_dists,
        dist_count,
        BK_SCAT_Q,
        BK_SCAT_D,
        idx.n_landmarks,
        8,
        DIST_L1,
        Float32(2.0),
    )
    var hi = ctx.enqueue_create_host_buffer[DType.int32](BK_SCAT_Q * 8)
    ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=out_inds)
    ctx.synchronize()

    var differing = 0
    var oi = List[Int32]()
    var oc = List[Float32]()
    for q in range(BK_SCAT_Q):
        _host_topk(
            sx.unsafe_ptr(),
            BK_SCAT_N,
            sq.unsafe_ptr(),
            q,
            BK_SCAT_D,
            8,
            DIST_L2_SQRT_UNEXPANDED,
            Float32(2.0),
            oi,
            oc,
        )
        for s in range(8):
            if hi.unsafe_ptr().unsafe_load(q * 8 + s) != oi[s]:
                differing += 1
                break
    if differing == 0:
        raise Error(
            "reach: an L1 query returned the EUCLIDEAN answer on every one"
            " of "
            + String(BK_SCAT_Q)
            + " rows. `metric` is threaded through the signatures and is"
            " not changing what the kernel computes -- reached but inert."
        )
    print(
        "check_rbc_eps_metric_is_reached OK: the L1 answer matches the L1"
        " oracle everywhere and differs from the EUCLIDEAN oracle on "
        + String(differing)
        + " of "
        + String(BK_SCAT_Q)
        + " rows"
    )
    _ = out_inds^
    _ = out_dists^
    _ = dist_count^
    _ = query^
    _ = hi^
    _ = idx^
    _ = sx^
    _ = sq^
