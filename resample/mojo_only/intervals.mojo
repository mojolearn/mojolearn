"""What turns a distribution into an answer: percentile, basic, BCa (refused),
the jackknife, the standard error and the permutation p-value.

NO UPSTREAM KERNEL. `scipy/stats/_resampling.py` defines the SEMANTICS of
every quantity below and is read as the ORACLE; its implementation is
`numpy` over a last axis and is never the design source. Each function names
the SciPy expression it reproduces so a reader can diff them line by line.

============ WHERE THE ARITHMETIC RUNS, AND WHY ============

EVERYTHING IN THIS FILE EXCEPT THE JACKKNIFE RUNS ON THE HOST, and that is a
decision rather than a leftover. The inputs are the bootstrap distribution
(already back on the host, because the caller is returned it) and four
scalars. PORTING_RULES 0b-ii is explicit that host work the GPU path needs is
not a "CPU path", and IDENTITY_PATHS row 18 is equally explicit about the one
thing that makes host arithmetic dangerous: a HOST LIBM call, whose last bits
differ between macOS and glibc. So the rule this file keeps is narrow and
checkable -- NOT ONE LIBM CALL. The folds are
`metrics/mojo_only/pinned_sum.mojo::host_tree_sum`, which is the same tree the
device folds; the square roots are `identical_sqrt`; the divisions are
`identical_div`; the comparisons are comparisons. Float32 add, multiply,
divide and sqrt are correctly rounded on every host that runs this, so the
answer is a pure function of the bits.

The JACKKNIFE is on the device because it is `n` statistics over `n`
observations, which is the one quantity here that is not O(R) scalars.

============ THE STANDARD ERROR ============

`scipy.stats.bootstrap` returns
`standard_error = xp.std(theta_hat_b, correction=1, axis=-1)`. Ours is the
same quantity, two-pass, through `host_tree_sum` (DEVIATION 1697 for the
ddof, `statistics.mojo`'s header for why two-pass).

================= DEVIATION BLOCK =================

DEVIATION 1699. **BCa IS REFUSED BY NAME, IN BOTH MODES, AND THE REASON IS ONE
FUNCTION.** `method='BCa'` raises. What survives, is computed, and is recorded
on the card is everything BCa needs EXCEPT its last two lines.

WHAT SciPy DOES (`_resampling.py::_bca_interval`):

    z0_hat  = ndtri(percentile_of_score(theta_hat_b, theta_hat))
    U_i     = (n - 1) * (theta_hat_dot - theta_hat_i)          jackknife
    a_hat   = (1/6) * sum(U^3)/n^3 / (sum(U^2)/n^2)^(3/2)
    z_alpha = ndtri(alpha)
    alpha_1 = ndtr(z0_hat + (z0_hat + z_alpha )/(1 - a_hat*(z0_hat + z_alpha )))
    alpha_2 = ndtr(z0_hat + (z0_hat + z_1alpha)/(1 - a_hat*(z0_hat + z_1alpha)))

WHAT IS IDENTICAL HERE AND IS SHIPPED AS A COMPUTED QUANTITY:

  * `percentile_of_score` is `(count(a < s) + count(a <= s)) / (2B)` --
    INTEGER COUNTING over the bootstrap distribution, then one division. No
    fold order, no rounding until the last op. `bca_bias_percentile`.
  * the jackknife `theta_hat_i` is `n` leave-one-out statistics, each the
    pinned tree of `statistics.mojo` (`jackknife_stat_kernel`), and the
    left-out slot is `+0.0` in the ORIGINAL n-slot chunk layout rather than
    compacted -- pinned and stated below.
  * `a_hat` is two pinned host folds and one `identical_sqrt`
    (`bca_acceleration`). `den^(3/2)` is spelled `den * sqrt(den)`, not
    `pow(den, 1.5)`: two correctly-rounded operations against `identical_pow`'s
    exp/log chain, so it is one fewer construction to certify and strictly
    fewer roundings.

WHAT IS NOT IDENTICAL AND IS THE WHOLE OF THE REFUSAL: **`ndtri`, the inverse
normal CDF.** `mojo_only/numerics.mojo` has `portable_erff` / `identical_erf`
(so `ndtr(x) = (1 + erf(x/sqrt(2)))/2` IS available), and it has no inverse of
anything. SciPy's `ndtri` is Cephes' float64 rational approximation with three
coefficient tables and a branch at `p = 0.135`. Re-deriving it in float32
would be a NEW numeric construction with a new error budget that has never
been measured on any vendor, written by a lane that is not allowed to run a
build. IDENTITY_PATHS' opening rule gives exactly three moves -- PIN, REPLACE,
REFUSE -- and there is no fourth called "write a polynomial and hope".

WHY THE REFUSAL IS IN BOTH MODES AND NOT ONLY UNDER IDENTICAL. A FAST-only BCa
would be a method that silently exists on one build and not the other, which
is how a user comes to believe an interval is reproducible when it is not.
There is no FAST arm to preserve here (nothing was ever shipped), so refusing
in both modes costs nothing and removes the ambiguity. This is a departure
from the usual shape of a REFUSE row and it is deliberate.

THE CLOSURE CONDITION, so this refusal has an owner and an end: land
`portable_ndtri` / `identical_ndtri` in `mojo_only/numerics.mojo` beside
`portable_erff`, gate it the way `check-division` gates `portable_divf` (a
hashed sweep against a float64 route, on every vendor), then delete
`bca_refuse` and turn `check_resample_refusals`' BCa clause into a
`check_bca_interval` against a hand-worked example. `bca_bias_percentile`,
`jackknife_stat_kernel` and `bca_acceleration` are already written and already
gated, so the closure is one function plus two lines.

DEVIATION 1700. THE JACKKNIFE'S FOLD KEEPS THE FULL n-SLOT LAYOUT. Leaving
observation `i` out is spelled as `+0.0` in slot `i` of the ordinary
`chunk_count(n)` chunking, NOT as a compaction to `n - 1` slots. Both are pure
functions and both are legitimate; this one is chosen because it makes the
jackknife's tree the SAME tree as the point estimate's for every `i`, so the
only thing that moves between `theta_hat` and `theta_hat_i` is one term. The
consequence, stated because row 39 requires it: a sample whose values are all
`-0.0` folds to `+0.0` here exactly as it does everywhere else in this lane,
and the left-out `+0.0` adds nothing to any other input. The divisor is
`n - 1`, matching SciPy's `_jackknife_resample`, which really does hand the
statistic an `n - 1` long sample.

DEVIATION 1701. `alternative` NARROWS THE INTERVAL EXACTLY AS SciPy DOES.
`scipy.stats.bootstrap` sets `alpha = (1 - confidence_level)/2` for
`'two-sided'` and `alpha = 1 - confidence_level` for the one-sided arms, then
replaces the unused endpoint with an infinity (`-inf` for `'less'`, `+inf` for
`'greater'`). Transcribed, infinities and all, including the ORDER: the
one-sided endpoint is computed from the same `alpha` machinery and only then
is its partner overwritten.

DEVIATION 1702. THE PERMUTATION p-VALUE CARRIES SciPy'S `gamma` TOLERANCE AND
ITS `+1`. Their expression (`_resampling.py::permutation_test`) is

    eps   = finfo(dtype).eps * 100
    gamma = abs(eps * observed)
    p_greater = (count(null >= observed - gamma) + 1) / (n_resamples + 1)

The `+1` in both places is the standard mid-p correction for a randomised
test ([2], [3] in their docstring) and is applied only when the test is NOT
exhaustive; this lane never enumerates, so it is always applied. `gamma`
exists because two theoretically equal statistics can be numerically distinct;
at float32 `eps*100` is 1.1920929e-05, larger than float64's, and it is
computed from OUR dtype rather than copied from theirs. The two-sided value is
`min(p_less, p_greater) * 2`, clipped to `[0, 1]`, which means THE TWO-SIDED
FLOOR IS `2/(n_resamples+1)` AND NOT `1/(n_resamples+1)`.
`check_permutation_separable` asserts both, separately, because conflating
them is the easy mistake here.
=================================================================
"""

from std.gpu import block_idx, thread_idx

from metrics.mojo_only.pinned_sum import (
    PINNED_SUM_W,
    canonicalize_nan,
    chunk_count,
    host_tree_sum,
    virtual_block_sum,
)
from mojo_only.numerics import ftz, identical_div, identical_mul, identical_sqrt
from resample.mojo_only.statistics import (
    STAT_DIFF_MEANS,
    STAT_MEAN,
    STAT_STD,
    _block_broadcast,
    _mean_of_sum,
    quantile_of_sorted_host,
    stat_name,
)


# ===========================================================================
# The methods and the alternatives
# ===========================================================================

comptime METHOD_PERCENTILE = 0
comptime METHOD_BASIC = 1
comptime METHOD_BCA = 2

comptime ALT_TWO_SIDED = 0
comptime ALT_LESS = 1
comptime ALT_GREATER = 2

#: `np.finfo(np.float32).eps * 100`, DEVIATION 1702. Written as a decimal AND
#: relied on only through comparisons, never round-tripped through a String
#: (`[[mojo-string-float-roundtrip]]`).
comptime PERM_EPS_SCALE = Float32(1.1920928955078125e-05)

comptime NEG_INF = Float32(-1.0) / Float32(0.0)
comptime POS_INF = Float32(1.0) / Float32(0.0)


def method_from_name(name: String) raises -> Int:
    """`scipy.stats.bootstrap`'s `method`. Their spellings, plus their
    case-insensitive `'BCa'`."""
    if name == "percentile":
        return METHOD_PERCENTILE
    if name == "basic":
        return METHOD_BASIC
    if name == "BCa" or name == "bca":
        return METHOD_BCA
    raise Error(
        "bootstrap: unknown method '"
        + name
        + "'. SciPy's three are 'percentile', 'basic' and 'BCa'; this lane"
        " ships the first two and REFUSES the third by name (DEVIATION 1699,"
        " intervals.mojo)."
    )


def method_name(method: Int) -> String:
    if method == METHOD_PERCENTILE:
        return String("percentile")
    if method == METHOD_BASIC:
        return String("basic")
    if method == METHOD_BCA:
        return String("BCa")
    return String("?")


def alternative_from_name(name: String) raises -> Int:
    """SciPy's spellings, hyphen and all."""
    if name == "two-sided":
        return ALT_TWO_SIDED
    if name == "less":
        return ALT_LESS
    if name == "greater":
        return ALT_GREATER
    raise Error(
        "resample: unknown alternative '"
        + name
        + "'. SciPy's three are 'two-sided', 'less' and 'greater' (note the"
        " hyphen; 'twosided' and 'two_sided' are not their spellings and are"
        " not accepted here either)."
    )


def alternative_name(alt: Int) -> String:
    if alt == ALT_TWO_SIDED:
        return String("two-sided")
    if alt == ALT_LESS:
        return String("less")
    if alt == ALT_GREATER:
        return String("greater")
    return String("?")


def bca_refuse() raises:
    """DEVIATION 1699. The whole refusal, in one place, so the wording is the
    same wherever a BCa request arrives."""
    raise Error(
        "bootstrap: NUMERIC_IDENTICAL and NUMERIC_FAST both REFUSE"
        " method='BCa'. Its two endpoints are ndtr(z0 + (z0 + z_alpha)/(1 -"
        " a_hat*(z0 + z_alpha))) with z_alpha = ndtri(alpha), and ndtri --"
        " the INVERSE normal CDF -- has no portable construction in"
        " mojo_only/numerics.mojo. SciPy's is a Cephes float64 rational"
        " approximation; a float32 re-derivation would be a new numeric"
        " construction with an error budget nobody has measured on any"
        " vendor, and IDENTITY_PATHS' opening rule offers PIN, REPLACE and"
        " REFUSE and no fourth move. Everything BCa needs BESIDES ndtri is"
        " built, identical and recorded on the card: the bias percentile"
        " (integer counting, resample.bca.z0p), the leave-one-out jackknife"
        " (the pinned tree, resample.jackknife) and the acceleration"
        " (resample.bca.ahat). To close this refusal: land"
        " portable_ndtri/identical_ndtri in mojo_only/numerics.mojo beside"
        " portable_erff, gate it the way check-division gates portable_divf,"
        " then delete bca_refuse and gate the endpoints against a hand-worked"
        " example. Today: use method='percentile' or method='basic', both of"
        " which are shipped and gated."
    )


# ===========================================================================
# alpha, and the two shipped intervals
# ===========================================================================


def alpha_for(confidence_level: Float32, alternative: Int) -> Float32:
    """`alpha = (1 - confidence_level)/2` two-sided, `1 - confidence_level`
    one-sided (`scipy.stats.bootstrap`, DEVIATION 1701). One subtraction and
    at most one exact halving, so no order question arises."""
    var tail = ftz(Float32(1.0) - confidence_level)
    if alternative == ALT_TWO_SIDED:
        return ftz(tail * Float32(0.5))
    return tail


@fieldwise_init
struct Interval(ImplicitlyCopyable, Movable):
    """Two endpoints, low then high. SciPy's `ConfidenceInterval`."""

    var low: Float32
    var high: Float32


def percentile_interval(
    sorted_dist: List[Float32], n_resamples: Int, alpha: Float32
) -> Interval:
    """`quantile(theta_hat_b, [alpha, 1 - alpha])` over the SORTED bootstrap
    distribution, with `statistics.mojo`'s pinned Hyndman-Fan type 7 rule
    (DEVIATION 1698).

    `1 - alpha` is formed once, here, through `ftz` -- not recomputed at the
    call sites, so the two endpoints can never be taken at two spellings of
    the same number.
    """
    var lo = quantile_of_sorted_host(sorted_dist, 0, n_resamples, alpha)
    var hi = quantile_of_sorted_host(
        sorted_dist, 0, n_resamples, ftz(Float32(1.0) - alpha)
    )
    return Interval(lo, hi)


def basic_interval(
    sorted_dist: List[Float32],
    n_resamples: Int,
    alpha: Float32,
    theta_hat: Float32,
) -> Interval:
    """`ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l` applied to the
    percentile interval (`scipy.stats.bootstrap`, `method == 'basic'`).

    NOTE THE SWAP, which is theirs and is not a typo: the basic (Hall)
    interval reflects the percentile interval through the point estimate, so
    the UPPER percentile becomes the LOWER endpoint. A port that kept the
    order would return an interval whose ends are crossed whenever the
    distribution is skewed, and it would still look plausible.

    `2*theta_hat` is `theta_hat + theta_hat`, exact for every finite
    `theta_hat` below half of FLOAT_MAX, so it is spelled as an addition and
    not as a multiply: one fewer thing that a contraction could reach.
    """
    var p = percentile_interval(sorted_dist, n_resamples, alpha)
    var two_theta = ftz(theta_hat + theta_hat)
    return Interval(ftz(two_theta - p.high), ftz(two_theta - p.low))


def narrow_for_alternative(interval: Interval, alternative: Int) -> Interval:
    """DEVIATION 1701's second half: `'less'` sets the low endpoint to
    `-inf`, `'greater'` sets the high endpoint to `+inf`, `'two-sided'`
    changes nothing. Applied AFTER the interval is computed, exactly as
    SciPy applies it."""
    if alternative == ALT_LESS:
        return Interval(NEG_INF, interval.high)
    if alternative == ALT_GREATER:
        return Interval(interval.low, POS_INF)
    return interval


# ===========================================================================
# The standard error (host, the same tree the device folds)
# ===========================================================================


def distribution_mean(dist: List[Float32], n_resamples: Int) -> Float32:
    """`host_tree_sum / n`, the pinned tree of
    `metrics/mojo_only/pinned_sum.mojo`."""
    return _mean_of_sum(host_tree_sum(dist, n_resamples), n_resamples)


def distribution_standard_error(
    dist: List[Float32], n_resamples: Int
) raises -> Float32:
    """`xp.std(theta_hat_b, correction=1)`: two-pass, ddof = 1, the pinned
    tree both times (DEVIATION 1697).

    Raises at `n_resamples == 1` rather than dividing by zero, because a
    one-replicate bootstrap has no standard error and returning a NaN would
    put a vendor payload on a certified stage (row 39 FACT 2).
    """
    if n_resamples < 2:
        raise Error(
            "bootstrap: the standard error needs at least 2 resamples (it is"
            " the ddof=1 standard deviation of the bootstrap distribution,"
            " scipy.stats.bootstrap's correction=1); got n_resamples="
            + String(n_resamples)
        )
    var m = distribution_mean(dist, n_resamples)
    var sq = List[Float32]()
    for i in range(n_resamples):
        var d = ftz(dist[i] - m)
        sq.append(ftz(identical_mul(d, d)))
    var ssd = host_tree_sum(sq, n_resamples)
    return ftz(
        identical_sqrt(ftz(identical_div(ssd, Float32(n_resamples - 1))))
    )


# ===========================================================================
# The permutation p-value (DEVIATION 1702)
# ===========================================================================


@fieldwise_init
struct PValue(ImplicitlyCopyable, Movable):
    """The p-value plus the two counts it was computed from.

    The counts are carried out of the function ON PURPOSE.
    `check_permutation_separable` has to distinguish "the floor, because
    nothing in the null reached the observed statistic" from "the floor plus
    one, because one drawn permutation happened to reproduce the observed
    split" -- which is a legal accident of the fixture and not a defect --
    and it cannot do that from a rounded float.
    """

    var p: Float32
    var count_less: Int
    var count_greater: Int


def permutation_pvalue(
    null_dist: List[Float32],
    n_resamples: Int,
    observed: Float32,
    alternative: Int,
) -> PValue:
    """SciPy's `permutation_test` p-value, transcribed.

        gamma     = abs(eps * observed)                eps = float32 eps * 100
        p_less    = (count(null <= observed + gamma) + 1) / (n + 1)
        p_greater = (count(null >= observed - gamma) + 1) / (n + 1)
        two_sided = min(p_less, p_greater) * 2, clipped to [0, 1]

    `adjustment = 1` unconditionally: their `exact_test` arm enumerates all
    partitions and this lane never does (`_all_partitions_concatenated` is
    not ported; see resample/UNPORTED.tsv).

    THE COMPARISONS ARE COMPARISONS, so nothing here can move between
    vendors. The two divisions are `identical_div` because a vendor may
    substitute a fast reciprocal (IDENTITY_PATHS row 49) and `n + 1` is not
    a power of two. The `min` is over two values that are `+0.0` only if
    both counts are `-1`, which cannot happen, so row 39's `max(+0,-0)`
    hazard has no site here -- and it is written as a strict compare anyway,
    never as `min()`.
    """
    var gamma = abs(ftz(identical_mul(PERM_EPS_SCALE, observed)))
    var hi_bound = ftz(observed + gamma)
    var lo_bound = ftz(observed - gamma)
    var n_less = 0
    var n_greater = 0
    for i in range(n_resamples):
        var v = null_dist[i]
        if v <= hi_bound:
            n_less += 1
        if v >= lo_bound:
            n_greater += 1
    var denom = Float32(n_resamples + 1)
    var p_less = ftz(identical_div(Float32(n_less + 1), denom))
    var p_greater = ftz(identical_div(Float32(n_greater + 1), denom))
    var p: Float32
    if alternative == ALT_LESS:
        p = p_less
    elif alternative == ALT_GREATER:
        p = p_greater
    else:
        var smaller = p_greater
        if p_less < p_greater:
            smaller = p_less
        p = ftz(smaller + smaller)
        if p > Float32(1.0):
            p = Float32(1.0)
    return PValue(p, n_less, n_greater)


# ===========================================================================
# BCa's identical half: the bias percentile, the jackknife, the acceleration
#
# Computed, recorded and gated. The ENDPOINTS are refused (DEVIATION 1699).
# ===========================================================================


def bca_bias_percentile(
    sorted_dist: List[Float32], n_resamples: Int, theta_hat: Float32
) -> Float32:
    """`_percentile_of_score(theta_hat_b, theta_hat)`, their `'mean'` kind:

        (count(a < s) + count(a <= s)) / (2 * B)

    INTEGER COUNTING, then one division. The counts do not care that the
    input is sorted -- it is passed sorted only because the caller has it
    that way -- so no order question, no fold, and the two `+-0.0` spellings
    of a zero statistic compare equal to each other in BOTH counts, which is
    the correct behaviour and is why this returns the same number for either
    zero sign.
    """
    var below = 0
    var at_or_below = 0
    for i in range(n_resamples):
        var v = sorted_dist[i]
        if v < theta_hat:
            below += 1
        if v <= theta_hat:
            at_or_below += 1
    return ftz(
        identical_div(
            Float32(below + at_or_below), Float32(2 * n_resamples)
        )
    )


def jackknife_stat_kernel[stat: Int, tpb: Int](
    theta_i: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    n_features_in: Int32,
):
    """`theta_hat_i` for every leave-one-out sample: one block per left-out
    observation, DEVIATION 1700.

    `scipy.stats._resampling::_jackknife_resample` builds an `(n, n-1)`
    boolean mask and hands the statistic each row. Ours never materialises
    the row: slot `i` of the ordinary `chunk_count(n)` chunking holds `+0.0`
    for block `i`, and the divisor is `n - 1`. Same multiset, same divisor,
    one tree for every `i`.

    Three arms, matching what BCa is ever asked for here: `mean`, `std`
    (ddof=1) and the paired `diff_means`. The order statistics have no
    jackknife arm because BCa is refused anyway; when DEVIATION 1699 closes,
    they are the first thing to add and they need the sort path.
    """
    comptime assert (
        stat == STAT_MEAN or stat == STAT_STD or stat == STAT_DIFF_MEANS
    ), "jackknife_stat_kernel: mean, std and diff_means are the arms"
    comptime lanes = PINNED_SUM_W // tpb
    var left_out = Int(block_idx.x)
    var n = Int(n_in)
    if left_out >= n:
        return
    var tid = Int(thread_idx.x)
    var d = Int(n_features_in)
    var chunks = chunk_count(n)

    var s0 = Float32(0.0)
    var s1 = Float32(0.0)
    for c in range(chunks):
        var v0 = SIMD[DType.float32, lanes](0.0)
        var v1 = SIMD[DType.float32, lanes](0.0)
        comptime for lane in range(lanes):
            var i = c * PINNED_SUM_W + tid + lane * tpb
            if i < n and i != left_out:
                v0[lane] = ftz(x.unsafe_load(i * d))
                comptime if stat == STAT_DIFF_MEANS:
                    v1[lane] = ftz(x.unsafe_load(i * d + 1))
        var t0 = virtual_block_sum[tpb](v0)
        var t1 = virtual_block_sum[tpb](v1)
        if tid == 0:
            s0 = ftz(s0 + t0)
            s1 = ftz(s1 + t1)

    var m = n - 1
    var value = Float32(0.0)
    comptime if stat == STAT_MEAN:
        if tid == 0:
            value = _mean_of_sum(s0, m)
    comptime if stat == STAT_DIFF_MEANS:
        if tid == 0:
            value = ftz(_mean_of_sum(s0, m) - _mean_of_sum(s1, m))
    comptime if stat == STAT_STD:
        var mb = _block_broadcast[tpb](_mean_of_sum(s0, m))
        var ssd = Float32(0.0)
        for c2 in range(chunks):
            var vs = SIMD[DType.float32, lanes](0.0)
            comptime for lane in range(lanes):
                var i2 = c2 * PINNED_SUM_W + tid + lane * tpb
                if i2 < n and i2 != left_out:
                    var dv = ftz(ftz(x.unsafe_load(i2 * d)) - mb)
                    vs[lane] = ftz(identical_mul(dv, dv))
            var ts = virtual_block_sum[tpb](vs)
            if tid == 0:
                ssd = ftz(ssd + ts)
        if tid == 0:
            value = ftz(identical_sqrt(ftz(identical_div(ssd, Float32(m - 1)))))

    if tid == 0:
        theta_i.unsafe_store(left_out, canonicalize_nan(value))


def bca_acceleration(theta_jack: List[Float32], n: Int) raises -> Float32:
    """`a_hat = (1/6) * (sum U^3 / n^3) / (sum U^2 / n^2)^(3/2)` with
    `U_i = (n - 1) * (theta_dot - theta_i)` (`_bca_interval`).

    The two sums are `host_tree_sum`, the same tree as everything else.
    `den^(3/2)` is `den * sqrt(den)` and not `identical_pow(den, 1.5)`: the
    two-operation spelling is exact for the sqrt and correctly rounded for
    the multiply, where `pow` is an exp of a log and carries that whole
    construction's error. `1/6` is spelled as a division by 6, not a
    multiplication by an inexact 0.1666..., so the constant contributes no
    rounding of its own.

    Raises when `den == 0`, i.e. every jackknife statistic is identical (a
    constant sample). SciPy returns NaN and warns; we refuse, for row 39
    FACT 2's reason -- `resample.bca.ahat` is a recorded stage and a
    computed NaN there would carry the vendor's payload.
    """
    if n < 2:
        raise Error(
            "bootstrap: the BCa acceleration needs at least 2 observations;"
            " got n=" + String(n)
        )
    var dot = _mean_of_sum(host_tree_sum(theta_jack, n), n)
    var u2 = List[Float32]()
    var u3 = List[Float32]()
    var scale = Float32(n - 1)
    for i in range(n):
        var u = ftz(identical_mul(scale, ftz(dot - theta_jack[i])))
        var sq = ftz(identical_mul(u, u))
        u2.append(sq)
        u3.append(ftz(identical_mul(sq, u)))
    var nf = Float32(n)
    var n2 = ftz(identical_mul(nf, nf))
    var n3 = ftz(identical_mul(n2, nf))
    var num = ftz(identical_div(host_tree_sum(u3, n), n3))
    var den = ftz(identical_div(host_tree_sum(u2, n), n2))
    if den == Float32(0.0):
        raise Error(
            "bootstrap: the BCa acceleration is 0/0 -- every leave-one-out"
            " statistic is identical, so the sample is constant under this"
            " statistic. SciPy returns NaN and raises DegenerateDataWarning;"
            " this lane refuses, because resample.bca.ahat is a recorded card"
            " stage and a computed NaN carries the vendor's payload"
            " (IDENTITY_PATHS row 39 FACT 2)."
        )
    var den32 = ftz(identical_mul(den, ftz(identical_sqrt(den))))
    return ftz(identical_div(ftz(identical_div(num, den32)), Float32(6.0)))


def bca_arm_name(stat: Int) -> String:
    """Which jackknife arm a statistic would use if DEVIATION 1699 closed.
    Used by the refusal's diagnostics and by the card header."""
    if stat == STAT_MEAN or stat == STAT_STD or stat == STAT_DIFF_MEANS:
        return stat_name(stat)
    return String("none (order statistic; needs the sort path)")
