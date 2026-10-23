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
from ._buffer import addr, addr_ro, as_f32_c, empty  # noqa: F401 (kept for importers)

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
    return _NATIVE(None, "family_rs_bootstrap", (_backend.binding("_mojolearn_resample", numeric_mode), data, statistic, n_resamples, confidence_level,
                   method, alternative, random_state, q_or_prop, r_first, numeric_mode, paired), _NATIVE_CONTEXT)


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
    return _NATIVE(None, "family_rs_permutation_test", (_backend.binding("_mojolearn_resample", numeric_mode), x, y, statistic, n_resamples, alternative,
                   random_state, r_first, numeric_mode, permutation_type), _NATIVE_CONTEXT)


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
    return _NATIVE(None, "family_rs_monte_carlo_integrate", (_backend.binding("_mojolearn_resample", numeric_mode), integrand, lower, upper, n_samples,
                   random_state, i_first, numeric_mode), _NATIVE_CONTEXT)


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
    return _NATIVE(None, "family_rs_resample_indices", (_backend.binding("_mojolearn_resample", numeric_mode), n, n_samples, replace, random_state,
                   numeric_mode), _NATIVE_CONTEXT)


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
    return _NATIVE(None, "family_rs_resample", (_backend.binding("_mojolearn_resample", numeric_mode), arrays, replace, n_samples, random_state, stratify,
                   sample_weight, numeric_mode), _NATIVE_CONTEXT)


# The compiled owner of this module's public functions
# (bindings/linalg_ops_resample.mojo), connected once at import: each public
# function is one native call, handed the resample binding of its numeric mode.
from . import _native_family_bridge  # noqa: E402
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
_native_family_bridge.tables(family_rs_state=globals())

__all__ = ["bootstrap", "permutation_test", "monte_carlo_integrate", "resample", "resample_indices",
           "BootstrapResult", "PermutationTestResult", "MonteCarloResult",
           "STATISTICS", "METHODS", "ALTERNATIVES", "INTEGRANDS"]
