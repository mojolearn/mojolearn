# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Random ball cover: the distance functors and the sort comparator.

PORT OF `cuvs/src/neighbors/ball_cover/registers_types.cuh` (the `DistFunc`
family) and `cuvs/src/neighbors/ball_cover/common.cuh` (`NNComp`) at cuVS
`94c2819`. Transliterated. Do not improve.

WHICH FUNCTOR ACTUALLY SHIPS, READ FROM THEIR CODEGEN
-----------------------------------------------------
`src/neighbors/ball_cover/detail/ball_cover/registers_00_generate.py:116,148`
instantiates `rbc_eps_pass` for exactly ONE combination:

    (std::int64_t, float, EuclideanSqFunc)

so the eps path ships **squared** Euclidean and nothing else. Haversine and
the plain `EuclideanFunc` exist in the header and are never instantiated for
the eps query. That is why `eps_dist_sq` below is the only functor here and
why every distance in the query kernels is squared.

**THE BUILD AND THE QUERY USE DIFFERENT ROOTS AND THAT IS NOT A MISTAKE.**
The index stores `R_1nn_dists` and `R_radius` as TRUE Euclidean distances,
because cuML only enables the ball-cover path for `L2SqrtExpanded` and
`L2SqrtUnexpanded` (`cuml/cpp/src/dbscan/runner.cuh:152-153`) and the index
is built with `brute_force` under that metric. The query kernel then works in
squared space and un-squares only the query-to-landmark distance, once per
landmark, at `registers.cuh:523,649,794,926`:

    cur_R_dist = sqrt(shfl(lane_R_dist_sq, k_offset))

Both triangle-inequality tests are therefore in unsquared space
(`d(q,r) <= eps + radius(r)` and `d(q,r) - d(r,y) <= eps`) while the
membership test `dist <= eps*eps` is in squared space. Mixing those up is the
one arithmetic error in this file that would silently drop neighbors.

THE `NNComp` ORDER IS PRODUCED WITHOUT A COMPARATOR SORT
--------------------------------------------------------
`common.cuh:26-37` is a `thrust::sort_by_key` comparator over a zip of
`(landmark_id, distance)`: landmark first, distance second. That exact order
is produced in `ball_cover.mojo` by a counting sort on the landmark followed
by a rank-by-counting on the distance inside each group, with the tie broken
by the original point index. See DEVIATION 3 there, including the measured
reason `nn.argsort` is not an option.
"""


from std.memory import bitcast

from checks.numerics import (
    ftz,
    identical_div,
    identical_mul_add,
    identical_sqrt,
)

# DEVIATION 564's op library. `distance_ops.mojo` is cuVS's own layout --
# ONE `distance_ops/` directory that every consumer reaches -- so the cores
# are CALLED from here rather than respelled, which is what makes an Lp cell
# computed by the ball cover and the same cell computed by
# `metric_distance_kernel` the same bits.
from neighbors.impl.distance.detail.distance_ops import (
    DIST_COSINE_EXPANDED,
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_L2_UNEXPANDED,
    DIST_LINF,
    DIST_LP_UNEXPANDED,
    l1_core,
    linf_core,
    lp_unexp_core,
    lp_unexp_epilog,
    metric_value_name,
)


# `std::numeric_limits<value_idx>::max()` is what `registers.cuh:498,536`
# uses as the "further than anything" sentinel for a FLOAT distance. That is
# int64 max reinterpreted through a float conversion, about 9.2e18, which is
# larger than any real squared distance and is therefore correct but is also
# plainly an accident of their template. Ours is float32's max, which cannot
# overflow the type it is stored in. Same behavior in both comparisons it
# feeds (`dist <= eps2` and `lane_R_dist_sq <= squared(eps + radius)`).
comptime RBC_FLT_MAX = Float32(3.4028234663852886e38)


def eps_dist_sq(
    a: MutPointer[Float32, MutAnyOrigin],
    a_off: Int,
    b: MutPointer[Float32, MutAnyOrigin],
    b_off: Int,
    n_dims: Int,
) -> Float32:
    """`EuclideanSqFunc::operator()`, `registers_types.cuh:62-75`.

    No `sqrt`. Their comment at `registers.cuh:490` is "we omit the sqrt() in
    the inner distance compute", and every membership test downstream squares
    `eps` instead.
    """
    # PINNED under IDENTICAL (IDENTITY_PATHS rows 9 and 10, DEVIATION 506).
    # This is the ball-cover arm's ONLY float accumulation and DBSCAN's rbc
    # path runs through it, so one bit here decides an eps-membership the
    # same way the brute-force kernel's accumulator does. `diff` is a
    # subtraction of nearby coordinates, which is where a denormal appears;
    # `sum_sq += diff * diff` is a multiply-add and therefore the codegen's
    # to contract unless it is pinned.
    var sum_sq = Float32(0.0)
    for i in range(n_dims):
        var diff = ftz(
            ftz(a.unsafe_load(a_off + i)) - ftz(b.unsafe_load(b_off + i))
        )
        sum_sq = ftz(identical_mul_add(diff, diff, sum_sq))
    return sum_sq


# ===========================================================================
# DEVIATION 564 (2026-09-01): THE COVER ADMITS ANY TRUE METRIC, NOT ONLY
# EUCLIDEAN, AND THE BOUNDS MOVE OUT OF SQUARED SPACE TO DO IT.
#
# WHAT THE PRUNE ACTUALLY RESTS ON. Both tests in `registers.mojo` and both
# tests in `knn.mojo` are the TRIANGLE INEQUALITY and nothing else:
#
#     d(q, y) >= d(q, r) - radius(r)          the landmark test
#     d(q, y) >= | d(q, r) - d(r, y) |        the in-group test
#
# Neither uses homogeneity, symmetry of the coordinates, or any property of
# the square root. They hold for EVERY function that satisfies the triangle
# inequality, which is exactly the definition of a metric. So the honest
# scope of this index is "the metrics", and the refusal that stood here
# until today -- Euclidean only -- was narrower than its own argument.
#
# WHAT IS STILL REFUSED, AND WHY EACH ONE IS REFUSED
#   cosine (`1 - cos`)   NOT A METRIC, AND IT FAILS TWICE. The ANGLE between
#                        two vectors is a metric; one minus its cosine is
#                        not. Take three unit vectors in a plane at 0, 60
#                        and 120 degrees: d(0, 60) = 0.5, d(60, 120) = 0.5,
#                        d(0, 120) = 1.5, and 1.5 > 0.5 + 0.5. It also fails
#                        identity of indiscernibles -- d(x, 2x) = 0 for
#                        every non-zero x -- so two distinct points can sit
#                        at distance zero and a landmark radius stops
#                        bounding its ball. cuML refuses it on the same
#                        index (`VALID_METRICS["rbc"]`).
#   Lp at p < 1          NOT A METRIC. |x|^p is not subadditive below p = 1;
#                        the standard counterexample in two dimensions is
#                        (0,0), (1,0), (1,1) at p = 1/2, where the direct
#                        distance is 4 and the two-leg path is 1 + 1 = 2.
#   sqeuclidean          NOT A METRIC, DESPITE THE NAME (DEVIATION 565).
#                        d^2 fails the inequality on three COLLINEAR points
#                        at unit spacing: 4 > 1 + 1. A cover built on it
#                        would prune real neighbors. `L2Expanded` and
#                        `L2SqrtExpanded` are refused here for the same
#                        reason `L2SqrtUnexpanded` is admitted: the tag says
#                        which VALUE comes out, and only the rooted one is a
#                        metric. (The expanded IDENTITY is refused for the
#                        second reason DEVIATION 2 in `ball_cover.mojo`
#                        already gives: the build and the query would then
#                        compute the compared quantities by two different
#                        formulas.)
#
# THE RESTRUCTURING, WHICH IS THE PART THAT IS NOT A RE-POINT. The eps
# kernels are written in SQUARED Euclidean space: `eps_dist_sq` returns
# `d^2`, membership is `dist <= eps*eps`, and the landmark test is
# `d^2 <= (eps + radius)^2`. An Lp radius has no squared form -- there is no
# monotone rewriting of `sum |x-y|^p` that lets a threshold be pre-squared
# and still mean the same thing for p != 2 -- so the comparison cannot merely
# be re-pointed at a new distance function. It is restructured into a
# COMPARISON SPACE:
#
#     rbc_cmp_dist   the value the kernel compares (SQUARED for Euclidean,
#                    the plain metric for everything else)
#     rbc_cmp_bound  a TRUE-distance threshold mapped into that space
#                    (`t * t` for Euclidean, `t` for everything else)
#     rbc_true_dist  a comparison-space value mapped back to a TRUE distance
#                    (`identical_sqrt(v)` for Euclidean, `v` otherwise)
#
# On the Euclidean arm every one of those is the SAME ARITHMETIC THE FILE
# ALREADY PERFORMED, in the same order: `eps * eps` is still `eps * eps`,
# `bound * bound` is still `bound * bound`, and the one `identical_sqrt` per
# landmark is still DEVIATION 550's. No Euclidean bit moves, and
# `check-ball-cover`, `check-radius` and `check-dbscan` are the proof.
#
# WHY THE THREE UNEXPANDED OPS AND NOT A FOURTH SPELLING. The cores below
# are `neighbors/impl/distance/detail/distance_ops.mojo`'s, called rather
# than copied, with the same `ftz` on each loaded operand that
# `metric_distance_kernel` applies. That makes an Lp cell computed here and
# an Lp cell computed by the brute-force metric kernel THE SAME BITS, which
# is what lets `ball_cover_knn_check.mojo` use one as the oracle for the
# other.
# ===========================================================================


#: What every existing caller means and what `dbscan/` has always passed:
#: TRUE Euclidean, computed unexpanded, compared in squared space. It is
#: `L2SqrtUnexpanded` and not `L2SqrtExpanded` because `eps_dist_sq` sums
#: the differences directly; see DEVIATION 2 in `ball_cover.mojo`.
comptime RBC_METRIC_DEFAULT = DIST_L2_SQRT_UNEXPANDED


def rbc_metric_is_admissible(metric: Int) -> Bool:
    """The metrics the cover may be built and queried under.

    The test is "does this tag satisfy the triangle inequality", not "can
    we compute it". `metric_distance_kernel` computes cosine and Lp at any
    p; neither fact admits them here.
    """
    return (
        metric == DIST_L2_SQRT_UNEXPANDED
        or metric == DIST_L1
        or metric == DIST_LINF
        or metric == DIST_LP_UNEXPANDED
    )


def rbc_validate_metric(metric: Int, metric_arg: Float32) raises:
    """Refuse a non-metric BY NAME, with the inequality as the reason.

    Host only, before any upload or launch. Every message says what the
    prune rests on, because a caller who is told only "refused" will
    reasonably assume the work is merely unported.
    """
    if metric == DIST_LP_UNEXPANDED:
        # DEVIATION 552's clauses first: they refuse a `p` that is not a
        # number at all, before this file asks whether it is >= 1.
        if metric_arg != metric_arg:
            raise Error(
                "mojolearn ball cover: metric='minkowski' needs a finite p >="
                " 1, got NaN (DEVIATION 552)"
            )
        if metric_arg == bitcast[DType.float32](UInt32(0x7F800000)):
            raise Error(
                "mojolearn ball cover: metric='minkowski' needs a finite p; p"
                " = infinity is Chebyshev, which is metric='chebyshev' and IS"
                " admitted here (DEVIATION 552)"
            )
        if metric_arg < Float32(1.0):
            raise Error(
                "mojolearn ball cover: metric='minkowski' with p = "
                + String(metric_arg)
                + " is REFUSED. The pruning in this index IS the triangle"
                " inequality on the landmark radii (DEVIATION 564), and"
                " |x|^p is not subadditive below p = 1, so Lp at p < 1 is"
                " not a metric at all: in two dimensions at p = 1/2 the"
                " points (0,0), (1,0), (1,1) give a direct distance of 4"
                " against a two-leg path of 1 + 1. A cover built on it"
                " would prune away true neighbours SILENTLY rather than"
                " return them slowly. p >= 1 is admitted, and p = 1 is"
                " Manhattan. Use NearestNeighbors, which is exact brute"
                " force and needs no inequality."
            )
        return
    if rbc_metric_is_admissible(metric):
        return
    if metric == DIST_COSINE_EXPANDED:
        raise Error(
            "mojolearn ball cover: metric='cosine' is REFUSED. The pruning"
            " in this index IS the triangle inequality on the landmark radii"
            " (DEVIATION 564). The ANGLE between two vectors is a metric;"
            " `1 - cos` of it is not: three unit vectors at 0, 60 and 120"
            " degrees give 1.5 against 0.5 + 0.5. So a cover built on it"
            " would prune away true neighbours SILENTLY rather than return"
            " them slowly."
            " cuML refuses it on the same index for the same reason"
            " (VALID_METRICS['rbc']). Use NearestNeighbors, which is exact"
            " brute force and honors every ported metric."
        )
    if metric == DIST_L2_EXPANDED or metric == DIST_L2_UNEXPANDED:
        raise Error(
            "mojolearn ball cover: metric='sqeuclidean' is REFUSED"
            " (DEVIATION 565). SQUARED Euclidean distance is not a metric"
            " even though Euclidean distance is: on three collinear points"
            " at unit spacing it gives 4 against 1 + 1, so it fails the"
            " triangle inequality the pruning rests on. Ask for"
            " metric='euclidean' and square the returned distances"
            " yourself, which is the same answer and is exact."
        )
    if metric == DIST_L2_SQRT_EXPANDED:
        raise Error(
            "mojolearn ball cover: the EXPANDED Euclidean tag is refused on"
            " this index. The value is a metric but the FORMULA is not the"
            " one the index is built with: DEVIATION 2 in ball_cover.mojo"
            " records that the build and the query must compute the"
            " compared quantities the same way, and the expanded identity"
            " ||a||^2 + ||b||^2 - 2ab loses the closest pair to"
            " cancellation. Ask for metric='euclidean', which routes to"
            " L2SqrtUnexpanded here."
        )
    raise Error(
        "mojolearn ball cover: metric "
        + metric_value_name(metric)
        + " ("
        + String(metric)
        + ") is not a metric this index admits. Admitted:"
        " L2SqrtUnexpanded (euclidean), L1 (manhattan), Linf (chebyshev),"
        " LpUnexpanded at p >= 1 (minkowski). The pruning IS the triangle"
        " inequality (DEVIATION 564)."
    )


@always_inline
def rbc_cmp_dist(
    a: MutPointer[Float32, MutAnyOrigin],
    a_off: Int,
    b: MutPointer[Float32, MutAnyOrigin],
    b_off: Int,
    n_dims: Int,
    metric: Int,
    metric_arg: Float32,
) -> Float32:
    """The value the query kernels COMPARE. Squared for Euclidean, the
    plain metric for the other three. See DEVIATION 564.

    The Euclidean arm is `eps_dist_sq` unchanged, called rather than
    respelled, so the shipped Euclidean path is the same function it has
    always been. The other three arms `ftz` each loaded operand before the
    core exactly as `metric_distance_kernel` does, so a cell computed here
    and the same cell computed by the brute-force metric kernel are the
    same bits.
    """
    if metric == DIST_L2_SQRT_UNEXPANDED:
        return eps_dist_sq(a, a_off, b, b_off, n_dims)

    var acc = Float32(0.0)
    if metric == DIST_L1:
        for i in range(n_dims):
            acc = l1_core(
                acc,
                ftz(a.unsafe_load(a_off + i)),
                ftz(b.unsafe_load(b_off + i)),
            )
        return acc
    if metric == DIST_LINF:
        for i in range(n_dims):
            acc = linf_core(
                acc,
                ftz(a.unsafe_load(a_off + i)),
                ftz(b.unsafe_load(b_off + i)),
            )
        return acc
    if metric == DIST_LP_UNEXPANDED:
        for i in range(n_dims):
            acc = lp_unexp_core(
                acc,
                ftz(a.unsafe_load(a_off + i)),
                ftz(b.unsafe_load(b_off + i)),
                metric_arg,
            )
        # `lp_unexp.cuh:67`: `one_over_p` is formed ONCE. Here that is once
        # per pair rather than once per register tile; it is a pure
        # function of `p`, so it is the same bits either way.
        return lp_unexp_epilog(
            acc, ftz(identical_div(Float32(1.0), metric_arg))
        )

    # An inadmissible tag reached a kernel. Quiet NaN, so a gate cannot
    # mistake a mis-dispatch for a result; the host entries refuse first.
    return bitcast[DType.float32](UInt32(0x7FC00000))


@always_inline
def rbc_cmp_bound(metric: Int, t: Float32) -> Float32:
    """A TRUE-distance threshold, mapped into comparison space.

    `t * t` on the Euclidean arm IS the `eps * eps` and the `bound * bound`
    the kernels already spelled; the identity on the other three is not an
    approximation of a square, it is the fact that those arms never left
    true-distance space.
    """
    if metric == DIST_L2_SQRT_UNEXPANDED:
        return t * t
    return t


@always_inline
def rbc_true_dist(metric: Int, v: Float32) -> Float32:
    """A comparison-space value, mapped back to a TRUE distance.

    DEVIATION 550's `identical_sqrt` on the Euclidean arm, unchanged and in
    the same place: once per landmark, on the query-to-landmark distance.
    """
    if metric == DIST_L2_SQRT_UNEXPANDED:
        return identical_sqrt(v)
    return v
