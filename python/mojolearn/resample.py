# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Maintainer notes (moved from the module docstring): workstream D, 2026-09-14. This module is the
# Python door of `resample/estimator.mojo` through `bindings/_mojolearn_resample.mojo`;
# `resample/README.md` carries the SciPy parameter mapping. The point estimate, interval, standard
# error and p-value are host scalars over the same pinned tree
# (`metrics/checks/pinned_sum.mojo::host_tree_sum`), with no libm call. BCa: DEVIATION 1699, closed by
# DEVIATION 5410. Mojo-host refusals: `n_resamples` outside the positions the index map can address
# (`validate_positions`), a pooled size the permutation map cannot address, and the
# `n_resamples * n` sort-cell ceiling. No published speed number for this lane.
"""Resampling statistics: bootstrap intervals, permutation tests, Monte Carlo
integration and scikit-learn style row resampling.

The public functions are :func:`bootstrap`, :func:`permutation_test`,
:func:`monte_carlo_integrate`, :func:`resample` and :func:`resample_indices`.
Their parameters mean what SciPy's (or scikit-learn's) parameters of the same
name mean. The random draws and the per-replicate statistics run on the
device, one block per replicate; the point estimate, interval, standard error
and p-value are then computed from those replicates in compiled code.

Inputs are converted to float32. Unknown statistic, method, alternative or
integrand names raise ``ValueError``, as does a sample that is not 1-D or
2-D, a non-finite value, a ``confidence_level`` outside (0, 1), a statistic
that reads a column the sample does not have, ``'std'`` on fewer than two
rows, or an ``n_resamples`` too large to address.

``r_first`` (and ``i_first`` for Monte Carlo) selects where a run starts in
the replicate stream: replicates ``[r_first, r_first + n_resamples)`` are
bit-identical to the same slice of one whole run, so a long run can be split
into batches.

Every function takes ``numeric_mode`` (``'identical'``
or ``'fast'`` on Apple); ``None`` uses the package default.

Examples
--------
>>> import numpy as np, mojolearn
>>> x = np.random.default_rng(0).random(1000, dtype=np.float32)
>>> res = mojolearn.resample.bootstrap(x, statistic="mean", n_resamples=999)
>>> low, high = res.confidence_interval
"""
from collections import namedtuple

from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty

_MODE_CODE = {"fast": 0, "identical": 1}

# Codes are `resample/checks/statistics.mojo`'s STAT_* codes.
#: Statistic names accepted by :func:`bootstrap` and :func:`permutation_test`, with their integer codes.
STATISTICS = {"mean": 0, "std": 1, "quantile": 2, "pearson": 3, "diff_means": 4, "trimmed_mean": 5}
# Codes are `resample/checks/intervals.mojo`'s METHOD_* codes.
#: Confidence-interval methods accepted by :func:`bootstrap` (``method`` is case-insensitive).
METHODS = {"percentile": 0, "basic": 1, "bca": 2}
# Codes are `resample/checks/intervals.mojo`'s ALT_* codes.
#: Alternative hypotheses accepted by :func:`bootstrap` and :func:`permutation_test`.
ALTERNATIVES = {"two-sided": 0, "less": 1, "greater": 2}
# Codes are `resample/checks/statistics.mojo`'s MC_F_* codes.
#: Integrands accepted by :func:`monte_carlo_integrate`.
INTEGRANDS = {"const": 0, "sum": 1, "product": 2}


# Sep 22 pip smoke: `res.confidence_interval.low`, SciPy's spelling, raised AttributeError on the
# plain tuple, hence the namedtuple.
#: SciPy's ``ConfidenceInterval``: a 2-tuple ``(low, high)`` that also has ``.low`` and ``.high``.
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])


class BootstrapResult:
    """Result of :func:`bootstrap`, SciPy's ``BootstrapResult`` with extra fields.

    Attributes
    ----------
    point_estimate : float
        The statistic on the original sample.
    distribution : Array of float32, shape (n_resamples,)
        The bootstrap replicates, in replicate order.
    sorted_distribution : Array of float32, shape (n_resamples,)
        The same replicates sorted ascending.
    standard_error : float
        Standard deviation of the bootstrap distribution.
    confidence_interval : ConfidenceInterval
        ``(low, high)``; also available as ``.low`` and ``.high``.
    order_low, order_high : int
        Positions in ``sorted_distribution`` used for the interval ends
        (for ``method='BCa'``, the positions at its adjusted levels).
    """

    def __init__(self, point_estimate, distribution, sorted_distribution,
                 standard_error, low, high, order_low, order_high):
        self.point_estimate = point_estimate
        self.distribution = distribution
        self.sorted_distribution = sorted_distribution
        self.standard_error = standard_error
        self.confidence_interval = ConfidenceInterval(low, high)
        self.order_low = order_low
        self.order_high = order_high


class PermutationTestResult:
    """Result of :func:`permutation_test`, SciPy's ``PermutationTestResult`` with extra fields.

    Attributes
    ----------
    statistic : float
        The statistic on the observed samples.
    null_distribution : Array of float32, shape (n_resamples,)
        The statistic on each permutation.
    pvalue : float
        The p-value for the requested ``alternative``.
    count_less, count_greater : int
        The two counts the p-value was computed from: null values on the
        low side and on the high side of the observed statistic.
    """

    def __init__(self, statistic, null_distribution, pvalue, count_less, count_greater):
        self.statistic = statistic
        self.null_distribution = null_distribution
        self.pvalue = pvalue
        self.count_less = count_less
        self.count_greater = count_greater


class MonteCarloResult:
    """Result of :func:`monte_carlo_integrate`.

    Attributes
    ----------
    integral : float
        The estimate, ``volume * mean``.
    mean : float
        Mean of the integrand over the draws.
    volume : float
        Area of the integration box.
    closed_form : float
        The exact integral of the same integrand over the same box, so the
        estimate's error can be read without a second call.
    """

    def __init__(self, integral, mean, volume, closed_form):
        self.integral = integral
        self.mean = mean
        self.volume = volume
        self.closed_form = closed_form


def _extension(numeric_mode=None):
    mod = _backend.binding("_mojolearn_resample", numeric_mode)
    want = numeric_mode or _backend.default_mode()
    fn = getattr(mod, "resample_numeric_mode", None)
    if fn is not None:
        got = int(fn())
        if got != _MODE_CODE.get(want):
            raise RuntimeError(
                f"mojolearn resample: numeric_mode={want!r} was requested but "
                f"{mod.__name__} reports compile-time mode code {got}; rebuild it "
                "with bash bindings/build_resample.sh"
            )
    return mod


def _code(table, value, name, where):
    if isinstance(value, str):
        if value not in table:
            raise ValueError(
                f"mojolearn {where}: {name} must be one of {sorted(table)}, got {value!r}"
            )
        return table[value]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"mojolearn {where}: {name} must be a name or its code")
    return int(value)


def _real(v, name, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"mojolearn {where}: {name} must be a real number, got {type(v).__name__}")
    return float(v)


def _int(v, name, where):
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"mojolearn {where}: {name} must be an int, got {type(v).__name__}")
    return int(v)


# BCa ships for mean, std and diff_means (DEVIATION 1699, closed by DEVIATION 5410); refused by name on the
# Mojo host for other statistics and a degenerate interval. Unpaired: sample 0 by the one-sample map,
# sample 1 by its own.
def bootstrap(data, statistic="mean", n_resamples=9999, confidence_level=0.95,
              method="percentile", alternative="two-sided", random_state=0,
              q_or_prop=0.5, r_first=0, numeric_mode=None, paired=True):
    """Bootstrap confidence interval for a statistic.

    Follows ``scipy.stats.bootstrap((data,), statistic, n_resamples=...,
    rng=random_state, method=..., confidence_level=..., alternative=...)``.
    The statistic is chosen by name from a compiled set rather than passed
    as a callable. The resampling and the per-replicate statistic run on
    the device.

    Parameters
    ----------
    data : array-like of shape (n,) or (n, 2), or a pair (x, y)
        The sample, converted to float32. A two-column sample keeps its
        row pairing (SciPy's ``paired=True``), which ``'pearson'`` and
        ``'diff_means'`` read. With ``paired=False``, a pair of 1-D samples
        of any lengths.
    statistic : str or int, default='mean'
        ``'mean'``, ``'std'``, ``'quantile'``, ``'pearson'``, ``'diff_means'``
        or ``'trimmed_mean'``, or its code in :data:`STATISTICS`.
    n_resamples : int, default=9999
        Number of bootstrap replicates.
    confidence_level : float, default=0.95
        Confidence level of the interval, in (0, 1).
    method : {'percentile', 'basic', 'BCa'}, default='percentile'
        Interval method, case-insensitive as in SciPy. ``'BCa'`` is
        available for ``'mean'``, ``'std'`` and ``'diff_means'`` and raises
        ``ValueError`` for the other statistics or a degenerate interval.
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Which side(s) of the interval to bound.
    random_state : int, default=0
        Seed of the replicate stream.
    q_or_prop : float, default=0.5
        ``q`` for ``'quantile'`` and ``proportiontocut`` for
        ``'trimmed_mean'``; ignored otherwise.
    r_first : int, default=0
        Index of the first replicate. Replicates ``[r_first, r_first +
        n_resamples)`` equal the same slice of one whole run, bit for bit.
    numeric_mode : {'identical', 'fast'} or None, default=None
        Numeric mode (``'fast'`` is Apple only); ``None`` uses the package
        default.
    paired : bool, default=True
        ``False`` with ``data=(x, y)`` and ``statistic='diff_means'`` runs
        SciPy's unpaired two-sample bootstrap, each sample resampled
        independently; any other statistic raises ``ValueError``. A single
        sample ignores ``paired``, as SciPy does.

    Returns
    -------
    BootstrapResult

    Raises
    ------
    ValueError
        For an unknown name, a sample that is not 1-D or 2-D, a non-finite
        value, a ``confidence_level`` outside (0, 1), ``'std'`` on fewer
        than two rows, or an unsupported ``method`` / ``statistic``
        combination.
    TypeError
        For a parameter of the wrong type.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> x = np.random.default_rng(0).random(1000, dtype=np.float32)
    >>> res = mojolearn.resample.bootstrap(x, statistic="mean", method="BCa")
    >>> low, high = res.confidence_interval
    """
    where = "bootstrap"
    if not paired and isinstance(data, (tuple, list)):
        return _bootstrap_unpaired(data, statistic, n_resamples, confidence_level, method,
                                   alternative, random_state, r_first, numeric_mode)
    x, _ = as_f32_c(data, ndim=None, name="data")
    if x.ndim == 1:
        n, d = x.shape[0], 1
    elif x.ndim == 2:
        n, d = x.shape
    else:
        raise ValueError(f"mojolearn {where}: data must be 1-D or 2-D, got {x.ndim}-D")
    stat = _code(STATISTICS, statistic, "statistic", where)
    meth = _code(METHODS, method.lower() if isinstance(method, str) else method, "method", where)
    alt = _code(ALTERNATIVES, alternative, "alternative", where)
    r = _int(n_resamples, "n_resamples", where)
    rf = _int(r_first, "r_first", where)
    seed = _int(random_state, "random_state", where)
    cl = _real(confidence_level, "confidence_level", where)
    qp = _real(q_or_prop, "q_or_prop", where)
    flat = x.reshape((n * d,))
    dist = empty((max(r, 0),), "<f4")
    sdist = empty((max(r, 0),), "<f4")
    scalars = empty((6,), "<f8")
    _extension(numeric_mode).bootstrap(
        # ORDER MATCHES bindings/_mojolearn_resample.mojo::bootstrap_binding.
        # x, distribution_out, sorted_out, scalars_out
        [addr_ro(flat, name="data"), addr(dist, name="distribution"), addr(sdist, name="sorted_distribution"), addr(scalars, name="scalars")],
        # n, d, statistic, n_resamples, seed, method, confidence_level, alternative, q_or_prop, r_first
        [n, d, stat, r, seed, meth, cl, alt, qp, rf],
    )
    return BootstrapResult(float(scalars[0]), dist, sdist, float(scalars[1]),
                           float(scalars[2]), float(scalars[3]), int(scalars[4]), int(scalars[5]))


def _bootstrap_unpaired(data, statistic, n_resamples, confidence_level, method,
                        alternative, random_state, r_first, numeric_mode):
    where = "bootstrap(paired=False)"
    if len(data) != 2:
        raise ValueError(f"mojolearn {where}: data must be two samples (x, y), got {len(data)}")
    if statistic != "diff_means":
        raise ValueError(
            f"mojolearn {where}: statistic {statistic!r} is refused by name: the unpaired bootstrap's"
            " two-sample statistic is 'diff_means' (a one-sample statistic has no pairing; pearson needs pairs)")
    xx, _ = as_f32_c(data[0], ndim=1, name="x")
    yy, _ = as_f32_c(data[1], ndim=1, name="y")
    meth = _code(METHODS, method.lower() if isinstance(method, str) else method, "method", where)
    alt = _code(ALTERNATIVES, alternative, "alternative", where)
    r = _int(n_resamples, "n_resamples", where)
    dist = empty((max(r, 0),), "<f4")
    sdist = empty((max(r, 0),), "<f4")
    scalars = empty((6,), "<f8")
    _extension(numeric_mode).bootstrap_unpaired(
        # ORDER MATCHES bindings/_mojolearn_resample.mojo::bootstrap_unpaired_binding.
        [addr_ro(xx, name="x"), addr_ro(yy, name="y"), addr(dist, name="distribution"),
         addr(sdist, name="sorted_distribution"), addr(scalars, name="scalars")],
        # n_x, n_y, n_resamples, seed, method, confidence_level, alternative, r_first
        [xx.shape[0], yy.shape[0], r, _int(random_state, "random_state", where), meth,
         _real(confidence_level, "confidence_level", where), alt, _int(r_first, "r_first", where)],
    )
    return BootstrapResult(float(scalars[0]), dist, sdist, float(scalars[1]),
                           float(scalars[2]), float(scalars[3]), int(scalars[4]), int(scalars[5]))


# The null is never exhaustive (DEVIATION 1702). 'pairings' is refused: for the implemented statistics
# its null leaves the statistic unchanged, and pearson has no permutation arm (resample/NOT_IMPLEMENTED.tsv).
def permutation_test(x, y=None, statistic="diff_means", n_resamples=9999,
                     alternative="two-sided", random_state=0, r_first=0, numeric_mode=None,
                     permutation_type="independent"):
    """Permutation test of a statistic.

    Follows ``scipy.stats.permutation_test((x, y), statistic,
    permutation_type=..., n_resamples=..., rng=random_state,
    alternative=...)``, with the statistic chosen by name. Unlike SciPy,
    the null distribution is always sampled with ``n_resamples``
    permutations, never enumerated exhaustively; the p-value is
    conservative.

    Parameters
    ----------
    x : array-like of shape (n_x,)
        First sample, converted to float32.
    y : array-like of shape (n_y,) or None, default=None
        Second sample. Required for ``permutation_type='independent'``.
        With ``'samples'``, ``y`` is the paired second sample, or ``None``
        for a one-sample sign-flip test.
    statistic : str or int, default='diff_means'
        The statistic, by name or by its code in :data:`STATISTICS`. With
        ``permutation_type='samples'`` it must be ``'diff_means'`` for two
        paired samples and ``'mean'`` for one sample.
    n_resamples : int, default=9999
        Number of permutations.
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        The alternative hypothesis.
    random_state : int, default=0
        Seed of the permutation stream.
    r_first : int, default=0
        Index of the first permutation; a slice of permutations equals the
        same slice of one whole run, bit for bit.
    numeric_mode : {'identical', 'fast'} or None, default=None
        Numeric mode (``'fast'`` is Apple only); ``None`` uses the package
        default.
    permutation_type : {'independent', 'samples'}, default='independent'
        ``'independent'`` reassigns pooled observations between ``x`` and
        ``y``. ``'samples'`` swaps the two observations of each pair at
        random (two paired samples), or flips the sign of each observation
        (one sample), as in SciPy. ``'pairings'`` raises ``ValueError``.

    Returns
    -------
    PermutationTestResult

    Raises
    ------
    ValueError
        For an unknown name, a missing ``y``, a refused
        ``permutation_type`` or ``statistic`` combination, or a non-finite
        value.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> x = rng.random(200, dtype=np.float32)
    >>> y = rng.random(300, dtype=np.float32) + 0.1
    >>> res = mojolearn.resample.permutation_test(x, y, n_resamples=999)
    >>> res.pvalue  # doctest: +SKIP
    """
    where = "permutation_test"
    ptype = permutation_type.lower() if isinstance(permutation_type, str) else permutation_type
    if ptype == "samples":
        return _permutation_samples(x, y, statistic, n_resamples, alternative, random_state,
                                    r_first, numeric_mode)
    if ptype == "pairings":
        raise ValueError(
            f"mojolearn {where}: permutation_type='pairings' is refused by name: its null permutes every"
            " sample's observation order, which for the implemented statistics (mean, std, diff_means)"
            " leaves the statistic unchanged -- the null is the observed value R times; pearson, the"
            " statistic it exists for, has no permutation arm (resample/NOT_IMPLEMENTED.tsv)")
    if ptype != "independent":
        raise ValueError(f"mojolearn {where}: permutation_type must be 'independent', 'samples' or"
                         f" 'pairings', got {permutation_type!r}")
    if y is None:
        raise ValueError(f"mojolearn {where}: permutation_type='independent' needs two samples x and y")
    xx, _ = as_f32_c(x, ndim=1, name="x")
    yy, _ = as_f32_c(y, ndim=1, name="y")
    stat = _code(STATISTICS, statistic, "statistic", where)
    alt = _code(ALTERNATIVES, alternative, "alternative", where)
    r = _int(n_resamples, "n_resamples", where)
    rf = _int(r_first, "r_first", where)
    seed = _int(random_state, "random_state", where)
    null = empty((max(r, 0),), "<f4")
    scalars = empty((4,), "<f8")
    _extension(numeric_mode).permutation_test(
        # ORDER MATCHES bindings/_mojolearn_resample.mojo::permutation_test_binding.
        # x, y, null_out, scalars_out
        [addr_ro(xx, name="x"), addr_ro(yy, name="y"), addr(null, name="null_distribution"), addr(scalars, name="scalars")],
        # n_x, n_y, statistic, n_resamples, seed, alternative, r_first
        [xx.shape[0], yy.shape[0], stat, r, seed, alt, rf],
    )
    return PermutationTestResult(float(scalars[0]), null, float(scalars[1]), int(scalars[2]), int(scalars[3]))


def _permutation_samples(x, y, statistic, n_resamples, alternative, random_state, r_first, numeric_mode):
    """permutation_type='samples': (x, y) paired, diff_means, each pair's two
    observations traded by a fair coin; (x,) alone (y None), mean, the sign of
    each observation flipped (SciPy's one-sample convention)."""
    where = "permutation_test(permutation_type='samples')"
    xx, _ = as_f32_c(x, ndim=1, name="x")
    want = "mean" if y is None else "diff_means"
    if statistic != want:
        raise ValueError(
            f"mojolearn {where}: statistic {statistic!r} is refused by name; the implemented arm for "
            f"{'one sample' if y is None else 'two paired samples'} is {want!r}")
    if y is None:
        yy, ny = xx, 0
    else:
        yy, _ = as_f32_c(y, ndim=1, name="y")
        ny = yy.shape[0]
    r = _int(n_resamples, "n_resamples", where)
    null = empty((max(r, 0),), "<f4")
    scalars = empty((4,), "<f8")
    _extension(numeric_mode).permutation_samples(
        # ORDER MATCHES bindings/_mojolearn_resample.mojo::permutation_samples_binding.
        [addr_ro(xx, name="x"), addr_ro(yy, name="y"), addr(null, name="null_distribution"), addr(scalars, name="scalars")],
        # n, n_y, n_resamples, seed, alternative, r_first
        [xx.shape[0], ny, r, _int(random_state, "random_state", where),
         _code(ALTERNATIVES, alternative, "alternative", where), _int(r_first, "r_first", where)],
    )
    return PermutationTestResult(float(scalars[0]), null, float(scalars[1]), int(scalars[2]), int(scalars[3]))


# Three compiled integrand arms; the closed form is hand-derived. The box has MC_DIMS = 2 dimensions.
def monte_carlo_integrate(integrand, lower, upper, n_samples, random_state=0, i_first=0, numeric_mode=None):
    """Monte Carlo estimate of a 2-D integral over a box.

    Computes ``volume * mean(f(x_i))`` over ``n_samples`` uniform draws
    from the two-dimensional box ``[lower, upper)``. The exact value of the
    same integral is returned beside the estimate.

    Parameters
    ----------
    integrand : {'const', 'sum', 'product'} or int
        ``'const'`` (f = 1), ``'sum'`` (x0 + x1) or ``'product'``
        (x0 * x1), or its code in :data:`INTEGRANDS`.
    lower, upper : sequence of 2 floats
        Lower and upper corners of the box; ``ValueError`` if either does
        not hold exactly two values.
    n_samples : int
        Number of uniform draws.
    random_state : int, default=0
        Seed of the draw stream.
    i_first : int, default=0
        Index of the first draw, so a long run can be split into batches.
    numeric_mode : {'identical', 'fast'} or None, default=None
        Numeric mode (``'fast'`` is Apple only); ``None`` uses the package
        default.

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    >>> import mojolearn
    >>> res = mojolearn.resample.monte_carlo_integrate("product", [0, 0], [1, 2], 100000)
    >>> round(res.closed_form, 6)
    1.0
    """
    where = "monte_carlo_integrate"
    f_id = _code(INTEGRANDS, integrand, "integrand", where)
    lo = Array.from_list([_real(v, "lower", where) for v in lower], "<f4")  # glue: validates the two lower bounds
    hi = Array.from_list([_real(v, "upper", where) for v in upper], "<f4")  # glue: validates the two upper bounds
    if lo.shape != (2,) or hi.shape != (2,):
        raise ValueError(
            f"mojolearn {where}: lower and upper must each hold 2 values "
            f"(MC_DIMS), got {lo.shape[0]} and {hi.shape[0]}"
        )
    scalars = empty((4,), "<f8")
    _extension(numeric_mode).monte_carlo_integrate(
        # ORDER MATCHES bindings/_mojolearn_resample.mojo::monte_carlo_integrate_binding.
        # lower, upper, scalars_out
        [addr_ro(lo, name="lower"), addr_ro(hi, name="upper"), addr(scalars, name="scalars")],
        # f_id, n_samples, seed, i_first
        [f_id, _int(n_samples, "n_samples", where), _int(random_state, "random_state", where), _int(i_first, "i_first", where)],
    )
    return MonteCarloResult(float(scalars[0]), float(scalars[1]), float(scalars[2]), float(scalars[3]))


# replace=True: position i draws a row by the Philox position map (kind 6); replace=False keeps the first
# n_samples positions of a keyed total order (kind 7, `rng.permutation(n)[:n_samples]`'s positional
# spelling). Integer only, so the same indices on every vendor and on the CPU.
def resample_indices(n, n_samples=None, replace=True, random_state=0, numeric_mode=None):
    """Row indices for a scikit-learn style resample.

    Returns the indices :func:`resample` gathers. With ``replace=True``
    each position draws a row independently; with ``replace=False`` the
    result is the first ``n_samples`` entries of a random permutation of
    ``range(n)``. The draw is integer only, so the same seed gives the same
    indices on every device. The indices differ from scikit-learn's for the
    same ``random_state``.

    Parameters
    ----------
    n : int
        Number of rows to draw from.
    n_samples : int or None, default=None
        Number of indices; ``None`` means ``n``. With ``replace=False`` it
        must be at most ``n``.
    replace : bool, default=True
        Draw with replacement.
    random_state : int, default=0
        Seed.
    numeric_mode : {'identical', 'fast'} or None, default=None
        Numeric mode (``'fast'`` is Apple only); ``None`` uses the package
        default.

    Returns
    -------
    Array of int32, shape (n_samples,)

    Examples
    --------
    >>> import mojolearn
    >>> idx = mojolearn.resample.resample_indices(1000, n_samples=100, replace=False)
    """
    where = "resample"
    n = _int(n, "n", where)
    count = n if n_samples is None else _int(n_samples, "n_samples", where)
    idx = empty((max(count, 0),), "<i4")
    _extension(numeric_mode).resample_indices(
        # ORDER MATCHES bindings/_mojolearn_resample.mojo::resample_indices_binding.
        [addr(idx, name="indices")],
        [n, count, 1 if replace else 0, _int(random_state, "random_state", where)],
    )
    return idx


def _take(a, idx):
    if hasattr(a, "__array__") and hasattr(a, "shape"):
        from ._optional_numpy import require_numpy
        np = require_numpy('resample')
        return np.asarray(a)[np.asarray(idx, dtype=np.intp)]
    return [a[int(i)] for i in idx]  # cpu-route: Python list input taken by index [py-data-loop]


def _gpu_gather(arrays, n, count, seed, numeric_mode, replace=True):
    """Return owned results or None for unchanged public fallback."""
    # C11 IDENTICAL admission is the compiled binding's explicit opt-in;
    # metadata chooses a binding, all draws and gathered data stay in Mojo.
    if count <= 0 or n <= 0:
        return None
    mod = _extension(numeric_mode)
    enabled = getattr(mod, "resample_gpu_gather_enabled", None)
    if enabled is None or not int(enabled()):
        return None
    from ._optional_numpy import require_numpy
    try:
        np = require_numpy("resample")
    except ImportError:
        return None
    # Never coerce input: preserve existing list/Array/strided/dtype behavior.
    if any(not isinstance(x, np.ndarray) or x.dtype != np.float32
           or x.ndim not in (1, 2) or not x.flags.c_contiguous
           or (x.ndim == 2 and not 0 < x.shape[1] <= 2147483647)
           for x in arrays):  # glue: inspect dtype rank and strides of each argument
        return None
    outputs = [np.empty((count,) + x.shape[1:], dtype=np.float32) for x in arrays]  # glue: allocate one caller-owned output per argument
    addresses = []
    widths = []
    for x, output in zip(arrays, outputs):  # glue: one native span per argument
        addresses.extend((addr_ro(x, name="array"), addr(output, name="resampled")))
        widths.append(1 if x.ndim == 1 else x.shape[1])
    route = mod.resample_gather_gpu if replace else getattr(mod, "resample_permutation_gather_gpu", None)
    if route is not None and int(route(addresses, [n, count, seed] + widths)):
        return outputs
    return None


def _narrow_gather(arrays, n, count, seed, numeric_mode):
    """`-D MOJOLEARN_RESAMPLE_FAST_GATHER_NARROW` (resample/estimator.mojo
    `resample_gather_narrow`): the indices drawn once on the device, the
    float32 C-contiguous arrays whose row is at most the binding's byte
    bound gathered there, the rest by `_take` with the same indices. None
    when the build lacks the define (main's route then runs)."""
    if count <= 0 or n <= 0 or not fast_defines(numeric_mode) & FAST_DEFINE_GATHER_NARROW:
        return None
    mod = _extension(numeric_mode)
    fn = getattr(mod, "resample_gather_narrow", None)
    if fn is None:
        return None
    from ._optional_numpy import require_numpy
    try:
        np = require_numpy("resample")
    except ImportError:
        return None
    bound = int(mod.resample_gather_narrow_bytes())
    idx = empty((count,), "<i4")
    outs, addresses, widths = [], [addr(idx, name="indices")], []
    for x in arrays:  # glue: one native span per argument (dtype, rank and stride checks)
        w = (1 if x.ndim == 1 else x.shape[1]) if isinstance(x, np.ndarray) and x.ndim in (1, 2) else 0
        if not (w > 0 and 4 * w <= bound and x.dtype == np.float32 and x.flags.c_contiguous):
            w = 0
        out = np.empty((count,) + x.shape[1:], dtype=np.float32) if w else None
        outs.append(out)
        addresses.extend((addr_ro(x, name="array"), addr(out, name="resampled")) if w else (0, 0))
        widths.append(w)
    done = int(fn(addresses, [n, count, seed] + widths))
    if done < 0:
        return None
    return [outs[a] if (done >> a) & 1 else _take(x, idx) for a, x in enumerate(arrays)]  # glue: one result per argument


#: Bits of the resample binding's `resample_fast_defines()` mask
#: (resample/estimator.mojo): FAST + Apple candidate defines, default OFF.
FAST_DEFINE_CV_SLICE = 32
FAST_DEFINE_CV_TRUST_FOLDS = 64
FAST_DEFINE_GATHER_NARROW = 128


def fast_defines(numeric_mode=None):
    """The FAST + Apple candidate defines the resample binding was built
    with (`-D MOJOLEARN_RESAMPLE_FAST_*`, `-D MOJOLEARN_CV_FAST_*`), as the
    binding's bit mask; 0 off the FAST tier, on a build without the
    function, or when the binding cannot be loaded. No environment read."""
    if (numeric_mode or _backend.default_mode()) != "fast":
        return 0
    try:
        mod = _extension(numeric_mode)
    except Exception:
        return 0
    fn = getattr(mod, "resample_fast_defines", None)
    return 0 if fn is None else int(fn())


# stratify and sample_weight are refused by name (resample/NOT_IMPLEMENTED.tsv).
def resample(*arrays, replace=True, n_samples=None, random_state=0, stratify=None,
             sample_weight=None, numeric_mode=None):
    """Resample arrays consistently along the first axis.

    Follows ``sklearn.utils.resample(*arrays, replace=..., n_samples=...,
    random_state=...)``: every array is indexed by the same rows, from
    :func:`resample_indices`. One array returns the resampled array,
    several return a list, as in scikit-learn. The rows drawn differ from
    scikit-learn's for the same ``random_state``.

    Parameters
    ----------
    *arrays : sequence of array-like
        Arrays with the same first dimension. NumPy arrays are indexed as
        arrays; other sequences return lists.
    replace : bool, default=True
        Draw with replacement.
    n_samples : int or None, default=None
        Number of rows to draw; ``None`` means the length of the arrays.
        With ``replace=False`` it must be at most that length.
    random_state : int, default=0
        Seed.
    stratify : None
        Not supported; any other value raises ``ValueError``.
    sample_weight : None
        Not supported; any other value raises ``ValueError``.
    numeric_mode : {'identical', 'fast'} or None, default=None
        Numeric mode (``'fast'`` is Apple only); ``None`` uses the package
        default.

    Returns
    -------
    array or list of arrays
        ``None`` when no arrays are given.

    Raises
    ------
    ValueError
        If the arrays have different lengths, or ``stratify`` or
        ``sample_weight`` is given.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> y = np.arange(1000, dtype=np.float32)
    >>> Xb, yb = mojolearn.resample.resample(X, y, n_samples=500)
    """
    where = "resample"
    if stratify is not None:
        raise ValueError(f"mojolearn {where}: stratify= is refused by name (resample/NOT_IMPLEMENTED.tsv): "
                         "scikit-learn's per-class allocation (_approximate_mode) breaks ties with its RNG stream")
    if sample_weight is not None:
        raise ValueError(f"mojolearn {where}: sample_weight= is refused by name (resample/NOT_IMPLEMENTED.tsv): "
                         "a weighted draw is an inverse-CDF lookup over a float cumulative sum not yet pinned")
    if not arrays:
        return None
    n = len(arrays[0])
    for a in arrays[1:]:  # glue: checks each argument array length
        if len(a) != n:
            raise ValueError(f"mojolearn {where}: Found input variables with inconsistent numbers of samples: "
                             f"{[len(x) for x in arrays]}")
    count = n if n_samples is None else _int(n_samples, "n_samples", where)
    if not replace:
        if count < 0 or count > n:
            # Preserve existing refusal before requesting native buffers.
            resample_indices(n, n_samples, replace, random_state, numeric_mode)
        gathered = _gpu_gather(arrays, n, count, _int(random_state, "random_state", where), numeric_mode, replace=False)
        if gathered is not None:
            return gathered[0] if len(gathered) == 1 else gathered
    if replace:
        gathered = _gpu_gather(arrays, n, count, _int(random_state, "random_state", where), numeric_mode)
        if gathered is not None:
            return gathered[0] if len(gathered) == 1 else gathered
        gathered = _narrow_gather(arrays, n, count, _int(random_state, "random_state", where), numeric_mode)
        if gathered is not None:
            return gathered[0] if len(gathered) == 1 else gathered
    idx = resample_indices(n, n_samples, replace, random_state, numeric_mode)
    out = [_take(a, idx) for a in arrays]  # glue: dispatches one gather per argument array
    return out[0] if len(out) == 1 else out


__all__ = ["bootstrap", "permutation_test", "monte_carlo_integrate", "resample", "resample_indices",
           "BootstrapResult", "PermutationTestResult", "MonteCarloResult",
           "STATISTICS", "METHODS", "ALTERNATIVES", "INTEGRANDS"]
