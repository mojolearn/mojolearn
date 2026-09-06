# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Six planted datasets, and every bit of every one of them is accounted for.

NOT A PORT. cuML has no Gaussian mixture model at the pin
(`upstream/cuml-v26.08.00` at `265b9da`, verified: no `gmm` directory, no
`mixture` module, no `GaussianMixture` symbol anywhere except an entry in
`python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-list.yaml`, which is
a list of scikit-learn tests that are FLAKY UNDER `cuml.accel` and is
therefore a record that cuML does NOT accelerate the estimator). So there are
no upstream fixtures to mirror either. `sklearn/mixture/tests/
test_gaussian_mixture.py` samples from `np.random.RandomState`, which is a
host RNG this tree cannot reproduce bit for bit on three vendors, and it
compares at tolerances rather than at bits.

THE CONSTRUCTION, AND WHY IT IS NOT A SAMPLER
----------------------------------------------
`cholesky/checks/cholesky_fixture.mojo`'s rule applies verbatim: a fixture
built by host floating-point arithmetic can hand two machines different
inputs before the first kernel runs (IDENTITY_PATHS row 32). A Gaussian
mixture fixture has a second problem on top of that, and it is worse. **A
SAMPLED fixture has no parameters, only estimates of them.** If the points
come out of a Box-Muller transform then the thing the fit is supposed to
recover is the SAMPLE mean and the SAMPLE covariance of whatever came out,
which nobody wrote down, and `check_recovers_planted_parameters` degenerates
into comparing a fit against a second calculation of the same estimate.

So nothing here is sampled. Every cluster is a fixed OFFSET PATTERN placed at
an integer center, and the pattern is chosen so that

  1. the offsets SUM TO EXACTLY ZERO, so the cluster's sample mean is the
     center exactly, with no rounding anywhere;
  2. the offsets' second-moment matrix is a stated matrix of dyadic
     rationals, so the cluster's maximum-likelihood covariance is known in
     closed form before anything runs;
  3. every coordinate, every product and every partial sum is exactly
     representable in float32, so the construction does not depend on the
     host's rounding, its FMA contraction, its denormal policy or its libm.

That makes "the fit must recover the planted parameters" a statement with a
right-hand side that was written down first. See `GMM_OFFSET_XX`,
`GMM_OFFSET_YY` and `GMM_OFFSET_XY` below for the closed form and
`plant_second_moment` for the proof of the exactness bound.

THE OFFSET PATTERN, EIGHT POINTS, AND WHY THESE EIGHT
------------------------------------------------------
    (+1,  0)  (-1,  0)  ( 0, +2)  ( 0, -2)
    (+.5,+.5) (-.5,-.5) (+1, +1)  (-1, -1)

Sum is exactly `(0, 0)`. The second moments are

    sum x*x = 1 + 1 + 0 + 0 + .25 + .25 + 1 + 1 =  4.5
    sum y*y = 0 + 0 + 4 + 4 + .25 + .25 + 1 + 1 = 10.5
    sum x*y = 0 + 0 + 0 + 0 + .25 + .25 + 1 + 1 =  2.5

so the MLE covariance of a cluster is `[[0.5625, 0.3125], [0.3125,
1.3125]]`, every entry a multiple of `1/16`, determinant `0.640625`, and
**the off-diagonal is not zero**, which is the whole point: a diagonal
planted covariance would be recovered correctly by a `covariance_type="diag"`
implementation and would say nothing about the full-covariance path this lane
exists for. The last four offsets are what make it non-diagonal; the first
four are what make it anisotropic.

`uniform-test-data-hides-permutation`: no two points in any fixture are
equal except where duplicates are PLANTED on purpose (`FIX_COLLAPSE`,
`FIX_DUPLICATES`), and the three clusters carry three different centers, so a
permutation of rows or of components changes the answer.

THE HASH IS A PER-LANE COPY AND THAT IS THE CONVENTION, NOT A NEW IDEA.
`gmm_mix64` is the same three lines as `kde/checks/kde_fixture.mojo:18`,
`cholesky/checks/cholesky_fixture.mojo:88`, `holtwinters/checks/
hw_fixture.mojo:30` and `isolation_forest/checks/if_fixture.mojo:32`, with
the same splitmix64 constants, for the reason `core/pinned_reduce.mojo` gives
about its own duplication: the alternative is a cross-lane import of another
lane's fixture file, and cross-lane dependencies on hot files are how two
sessions collide. Five copies of one hash is a debt and this sentence is the
record of it.
"""

from std.memory import bitcast


#: Three well-separated planted clusters, `d = 2`, eight points each.
#: `exp(-mahalanobis/2)` between clusters underflows float32 to EXACTLY zero
#: at this separation, so the converged responsibilities are exactly `1` and
#: exactly `0` and the recovery claim is not a claim about a small number.
comptime FIX_SEPARATED = 0

#: Two HEAVILY OVERLAPPING clusters, centers `1/2` apart against a spread of
#: order `1`, so no responsibility is near zero or one and the softmax is
#: genuinely soft. The fixture the E-step's arithmetic is actually exercised
#: by; `FIX_SEPARATED` exercises its underflow.
comptime FIX_OVERLAP = 1

#: Three clusters where the third is SIX EXACT DUPLICATES of one point. Its
#: maximum-likelihood covariance is exactly the zero matrix, so at
#: `reg_covar = +0.0` the Cholesky's first pivot is exactly `+0.0` and
#: `potrf_lower` returns `info = 1`. **THE COMPONENT MUST COLLAPSE**, by
#: arithmetic rather than by luck, and it collapses at INITIALIZATION.
comptime FIX_COLLAPSE = 2

#: Two clusters carrying EXACT DUPLICATE POINTS (three copies of one offset
#: in each), with a covariance that stays full rank so the fit runs to
#: completion. Duplicates are where a tie-break that is not total shows up.
comptime FIX_DUPLICATES = 3

#: `d = 1`, eight points, two clusters. Small enough that the log likelihood
#: is derived by hand in `check_log_likelihood_by_hand` from the closed form
#: of a one-dimensional Gaussian, at parameters the check SETS rather than
#: fits.
comptime FIX_ONE_D = 4

#: Signed zeros in the DATA: coordinates that are exactly `-0.0` and exactly
#: `+0.0`, in both clusters, so `X - mu` and `X @ P` see both. IDENTITY_PATHS
#: row 39 territory. The mixed-zero LOGSUMEXP ROW is not here and cannot be:
#: no legal data produces two components with exactly equal weighted log
#: probabilities of opposite-signed zero, so that row is PLANTED directly
#: into the kernel by `check_estep_vs_oracle`, exactly as
#: `kde/checks/kde_check.mojo::check_kde_row39_signed_zero_rowmax` plants
#: its rows.
comptime FIX_SIGNED_ZERO = 5

comptime GMM_FIXTURE_COUNT = 6


#: The eight offsets' second moments, as exact float32 constants. Written out
#: rather than summed at run time so that the number the check compares
#: against was decided by a person reading the pattern, not by the same loop
#: the fixture uses. `sum x*x`, `sum y*y`, `sum x*y` over the eight offsets.
comptime GMM_OFFSET_XX = Float32(4.5)
comptime GMM_OFFSET_YY = Float32(10.5)
comptime GMM_OFFSET_XY = Float32(2.5)

#: Points in one cluster of the eight-offset pattern.
comptime GMM_CLUSTER_POINTS = 8


def gmm_mix64(a: Int, b: Int, salt: Int) -> UInt64:
    """splitmix64 over three integers. This lane's only source of
    randomness, and a pure function of its arguments on every machine.

    It is used for the SIGNED-ZERO fixture's sign pattern and for nothing
    else. The planted clusters are not hashed at all: a hashed coordinate
    would destroy the exact second moment the whole file is built around.
    """
    var z = (
        UInt64(a + 1) * 0x9E3779B97F4A7C15
        + UInt64(b + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return z


def gmm_offset(t: Int, axis: Int) -> Float32:
    """Offset `t` of the eight-point pattern, coordinate `axis` (0 = x).

    Every value is a dyadic rational with at most one fractional bit and
    magnitude at most 2, which is what makes `plant_second_moment`'s bound
    hold. The pattern is written as a branch chain rather than as a table so
    that a reader can check the three sums in the file header against it
    without leaving this function.
    """
    if t == 0:
        return Float32(1.0) if axis == 0 else Float32(0.0)
    if t == 1:
        return Float32(-1.0) if axis == 0 else Float32(0.0)
    if t == 2:
        return Float32(0.0) if axis == 0 else Float32(2.0)
    if t == 3:
        return Float32(0.0) if axis == 0 else Float32(-2.0)
    if t == 4:
        return Float32(0.5)
    if t == 5:
        return Float32(-0.5)
    if t == 6:
        return Float32(1.0)
    return Float32(-1.0)


def plant_second_moment(axis_a: Int, axis_b: Int) -> Float32:
    """`sum over the eight offsets of delta[axis_a] * delta[axis_b]`, EXACT.

    **THE EXACTNESS BOUND, written out so it can be redone.** Each offset
    coordinate has at most 1 fractional bit and magnitude at most 2, so a
    product has at most 2 fractional bits and magnitude at most 4, and the
    sum of eight of them has at most 2 fractional bits and magnitude at most
    32: `2 + ceil(log2(32)) = 7` significant bits, against float32's 24. Every
    partial sum is exact, so the result does not depend on the order the
    terms are added in, on whether the host contracts `a*b+c`, on its
    rounding mode or on its denormal policy. There is nothing left for any of
    those to decide, which is why this function calls neither `ftz` nor
    `identical_mul_add`: spelling it with them would suggest the exactness
    came from the helpers rather than from the pattern.

    A CENTER SHIFTS NOTHING. The offsets sum to exactly zero, so a cluster
    placed at any center has this same second moment about its own mean --
    which is the property `check_recovers_planted_parameters` rests on.
    """
    var acc = Float32(0.0)
    for t in range(GMM_CLUSTER_POINTS):
        acc = acc + gmm_offset(t, axis_a) * gmm_offset(t, axis_b)
    return acc


def planted_covariance(scale_bits: Int) -> List[Float32]:
    """The `2 x 2` maximum-likelihood covariance of ONE eight-point cluster,
    row-major, before `reg_covar`.

    `scale_bits` is `log2` of a power-of-two divisor applied to every entry,
    so a caller can plant a tighter cluster without leaving the dyadic set.
    Pass `0` for the pattern as written.

    The division by `GMM_CLUSTER_POINTS = 8` and by `2^scale_bits` are both
    exact (powers of two), so the returned matrix is exact.
    """
    var div = Float32(GMM_CLUSTER_POINTS)
    for _ in range(scale_bits):
        div = div * Float32(2.0)
    var out = List[Float32]()
    out.append(plant_second_moment(0, 0) / div)
    out.append(plant_second_moment(0, 1) / div)
    out.append(plant_second_moment(1, 0) / div)
    out.append(plant_second_moment(1, 1) / div)
    return out^


def _emit_cluster(
    mut x: List[Float32], cx: Float32, cy: Float32, count: Int
):
    """Append `count` points of the eight-offset pattern around `(cx, cy)`.

    `count` may exceed eight, in which case the pattern REPEATS, which is how
    `FIX_DUPLICATES` gets exact duplicate rows without inventing a second
    construction. Every coordinate is `center + offset`, one exact add.
    """
    for i in range(count):
        var t = i % GMM_CLUSTER_POINTS
        x.append(cx + gmm_offset(t, 0))
        x.append(cy + gmm_offset(t, 1))


def gmm_fixture_n(which: Int) -> Int:
    """Rows in each fixture."""
    if which == FIX_SEPARATED:
        return 3 * GMM_CLUSTER_POINTS
    if which == FIX_OVERLAP:
        return 2 * GMM_CLUSTER_POINTS
    if which == FIX_COLLAPSE:
        return 2 * GMM_CLUSTER_POINTS + 6
    if which == FIX_DUPLICATES:
        return 2 * (GMM_CLUSTER_POINTS + 3)
    if which == FIX_ONE_D:
        return 8
    if which == FIX_SIGNED_ZERO:
        return 16
    return 0


def gmm_fixture_d(which: Int) -> Int:
    """Features in each fixture."""
    if which == FIX_ONE_D:
        return 1
    return 2


def gmm_fixture_k(which: Int) -> Int:
    """The `n_components` each fixture is MEANT to be fitted at.

    A fixture and a component count are one object here, not two: the whole
    argument of `FIX_COLLAPSE` is that `K = 3` on data whose third cluster is
    six copies of one point, and running it at `K = 2` would quietly turn the
    collapse fixture into an ordinary one.
    """
    if which == FIX_SEPARATED:
        return 3
    if which == FIX_COLLAPSE:
        return 3
    return 2


def gmm_fixture_name(which: Int) -> String:
    if which == FIX_SEPARATED:
        return String("SEPARATED")
    if which == FIX_OVERLAP:
        return String("OVERLAP")
    if which == FIX_COLLAPSE:
        return String("COLLAPSE")
    if which == FIX_DUPLICATES:
        return String("DUPLICATES")
    if which == FIX_ONE_D:
        return String("ONE_D")
    if which == FIX_SIGNED_ZERO:
        return String("SIGNED_ZERO")
    return String("UNKNOWN")


#: `FIX_SEPARATED`'s three centers, as integers, `(x, y)` interleaved.
#: Separation 32 against a cluster radius of 2: the squared Mahalanobis
#: distance from a point of one cluster to another cluster's center is above
#: 500, and `identical_exp(-250)` underflows float32 to exactly `+0.0`. That
#: is what makes the converged responsibilities exactly one-hot.
comptime SEP_CX0 = Float32(-16.0)
comptime SEP_CY0 = Float32(-16.0)
comptime SEP_CX1 = Float32(16.0)
comptime SEP_CY1 = Float32(-16.0)
comptime SEP_CX2 = Float32(0.0)
comptime SEP_CY2 = Float32(16.0)

#: `FIX_OVERLAP`'s two centers. Half a unit apart on x, nothing on y, against
#: a planted standard deviation of about 0.75 on x and 1.15 on y. Every
#: responsibility in the fitted model is strictly between 0.02 and 0.98, so
#: no cell of the softmax is decided by an underflow.
comptime OVL_CX0 = Float32(-0.25)
comptime OVL_CY0 = Float32(0.0)
comptime OVL_CX1 = Float32(0.25)
comptime OVL_CY1 = Float32(0.0)

#: `FIX_COLLAPSE`'s duplicated point, and how many copies of it there are.
#: Far from both real clusters, so k-means gives it a component of its own on
#: any tie-break, and every copy is bit-identical, so that component's
#: maximum-likelihood covariance is exactly the zero matrix.
comptime COLLAPSE_PX = Float32(0.0)
comptime COLLAPSE_PY = Float32(16.0)
comptime COLLAPSE_COPIES = 6


def gmm_fixture(which: Int) raises -> List[Float32]:
    """The fixture's `n x d` row-major float32 matrix.

    Every value is an exact sum of an integer center and a dyadic offset, so
    the matrix is a pure function of the constants above on every host.
    `FIX_SIGNED_ZERO` is the one that also PLANTS bit patterns, and it plants
    them by `bitcast` rather than by writing `-0.0` into arithmetic, so that
    the sign of every zero in it is a decision this file made and not one the
    host's parser made.
    """
    var x = List[Float32]()

    if which == FIX_SEPARATED:
        _emit_cluster(x, SEP_CX0, SEP_CY0, GMM_CLUSTER_POINTS)
        _emit_cluster(x, SEP_CX1, SEP_CY1, GMM_CLUSTER_POINTS)
        _emit_cluster(x, SEP_CX2, SEP_CY2, GMM_CLUSTER_POINTS)
        return x^

    if which == FIX_OVERLAP:
        _emit_cluster(x, OVL_CX0, OVL_CY0, GMM_CLUSTER_POINTS)
        _emit_cluster(x, OVL_CX1, OVL_CY1, GMM_CLUSTER_POINTS)
        return x^

    if which == FIX_COLLAPSE:
        _emit_cluster(x, SEP_CX0, SEP_CY0, GMM_CLUSTER_POINTS)
        _emit_cluster(x, SEP_CX1, SEP_CY1, GMM_CLUSTER_POINTS)
        # THE COLLAPSING COMPONENT. Six rows, bit for bit the same row.
        for _ in range(COLLAPSE_COPIES):
            x.append(COLLAPSE_PX)
            x.append(COLLAPSE_PY)
        return x^

    if which == FIX_DUPLICATES:
        # Eleven points per cluster: the eight-offset pattern, then offsets
        # 0, 1 and 2 AGAIN, so rows 8, 9, 10 are bit-identical to rows 0, 1,
        # 2. The second moment is no longer the header's matrix (the pattern
        # is not complete), which is fine: this fixture is not the recovery
        # one, and `check_mstep_vs_oracle` compares against the oracle rather
        # than against a closed form.
        _emit_cluster(x, SEP_CX0, SEP_CY0, GMM_CLUSTER_POINTS + 3)
        _emit_cluster(x, SEP_CX1, SEP_CY1, GMM_CLUSTER_POINTS + 3)
        return x^

    if which == FIX_ONE_D:
        # Eight points, one feature, symmetric about zero, two obvious
        # clusters at -1.25 and +1.25. Every value is a multiple of 1/2.
        var vals: List[Float32] = [
            Float32(-2.0),
            Float32(-1.5),
            Float32(-1.0),
            Float32(-0.5),
            Float32(0.5),
            Float32(1.0),
            Float32(1.5),
            Float32(2.0),
        ]
        for v in vals:
            x.append(v)
        return x^

    if which == FIX_SIGNED_ZERO:
        # Two clusters of eight, centers -8 and +8 on x, and the y
        # coordinates are ZEROS OF ALTERNATING SIGN plus two ordinary values
        # so the cluster is not rank deficient. Both zeros are planted by
        # bits.
        var pos_zero = bitcast[DType.float32](UInt32(0x00000000))
        var neg_zero = bitcast[DType.float32](UInt32(0x80000000))
        for c in range(2):
            var cx = Float32(-8.0) if c == 0 else Float32(8.0)
            for t in range(GMM_CLUSTER_POINTS):
                x.append(cx + gmm_offset(t, 0))
                if t < 4:
                    # Alternating signed zeros, decided by the hash so the
                    # pattern is not a run of one sign that a reader could
                    # mistake for a constant.
                    var h = gmm_mix64(c, t, 17)
                    x.append(neg_zero if (h & 1) == 1 else pos_zero)
                else:
                    x.append(gmm_offset(t, 1))
        return x^

    raise Error(
        "gmm_fixture: unknown fixture id "
        + String(which)
        + "; the six are 0..5, see the comptime constants in"
        " mixture/checks/gmm_fixture.mojo"
    )


def gmm_fixture_planted_center(which: Int, k: Int, axis: Int) raises -> Float32:
    """The planted center of cluster `k`, for the fixtures that have one.

    `check_recovers_planted_parameters` matches the FITTED components against
    these by a stated total order rather than by slot, so this returns the
    plant in PLANT order and the matching is the check's job.
    """
    if which == FIX_SEPARATED:
        if k == 0:
            return SEP_CX0 if axis == 0 else SEP_CY0
        if k == 1:
            return SEP_CX1 if axis == 0 else SEP_CY1
        if k == 2:
            return SEP_CX2 if axis == 0 else SEP_CY2
    if which == FIX_OVERLAP:
        if k == 0:
            return OVL_CX0 if axis == 0 else OVL_CY0
        if k == 1:
            return OVL_CX1 if axis == 0 else OVL_CY1
    raise Error(
        "gmm_fixture_planted_center: fixture "
        + gmm_fixture_name(which)
        + " has no planted center for component "
        + String(k)
        + "; only SEPARATED and OVERLAP do, and only they are used by"
        " check_recovers_planted_parameters"
    )
