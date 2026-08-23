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


from mojo_only.numerics import ftz, identical_mul_add


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
