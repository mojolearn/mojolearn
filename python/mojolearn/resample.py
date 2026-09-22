# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Resampling on the GPU: `bootstrap`, `permutation_test`,
`monte_carlo_integrate` (workstream D, 2026-09-14).

The Python door of `resample/estimator.mojo` through
`bindings/_mojolearn_resample.mojo`. Every parameter means what SciPy's
parameter of that name means, or is named differently; `resample/README.md`
carries the mapping. The draws and the per-replicate folds run on the
device, one block per replicate; the point estimate, the interval, the
standard error and the p-value are host scalars over the same pinned tree
(`metrics/checks/pinned_sum.mojo::host_tree_sum`), with no libm call.

WHAT IS REFUSED, AND WHERE. Here by name: an unknown statistic, method,
alternative or integrand SPELLING, a sample that is not 1-D or 2-D, and
`method='bca'` (DEVIATION 1699, refused on the Mojo host too). On the Mojo
host by name: `n_resamples` outside the positions the index map can
address (`validate_positions`), a pooled size the permutation map cannot
address, a non-finite cell, a `confidence_level` outside (0, 1), a
statistic that reads a column the sample does not have, `std` on fewer
than two rows, and the `n_resamples * n` sort-cell ceiling.

`r_first` and `i_first` are the batch-invariance handles and part of the
surface: replicates `[r_first, r_first + n_resamples)` of one run are
bit-identical to the corresponding slice of a whole run.

NO SPEED CLAIM. The lane has no published number and this door adds none.
"""
from collections import namedtuple

from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty

_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

#: `resample/checks/statistics.mojo`'s STAT_* codes.
STATISTICS = {"mean": 0, "std": 1, "quantile": 2, "pearson": 3, "diff_means": 4, "trimmed_mean": 5}
#: `resample/checks/intervals.mojo`'s METHOD_* codes.
METHODS = {"percentile": 0, "basic": 1, "bca": 2}
#: `resample/checks/intervals.mojo`'s ALT_* codes.
ALTERNATIVES = {"two-sided": 0, "less": 1, "greater": 2}
#: `resample/checks/statistics.mojo`'s MC_F_* codes.
INTEGRANDS = {"const": 0, "sum": 1, "product": 2}


#: SciPy's `ConfidenceInterval`: a 2-tuple that also answers `.low` and
#: `.high` (Sep 22 pip smoke: `res.confidence_interval.low`, SciPy's
#: spelling, raised AttributeError on the plain tuple).
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])


class BootstrapResult:
    """SciPy's `BootstrapResult` plus the sorted distribution and the two
    order-statistic positions the interval used."""

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
    """SciPy's `PermutationTestResult` plus the two counts the p-value was
    computed from (`PValue`)."""

    def __init__(self, statistic, null_distribution, pvalue, count_less, count_greater):
        self.statistic = statistic
        self.null_distribution = null_distribution
        self.pvalue = pvalue
        self.count_less = count_less
        self.count_greater = count_greater


class MonteCarloResult:
    """`integral = volume * mean`, plus the hand-derived closed form of
    the same integrand so the error is visible without a second call."""

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


def bootstrap(data, statistic="mean", n_resamples=9999, confidence_level=0.95,
              method="percentile", alternative="two-sided", random_state=0,
              q_or_prop=0.5, r_first=0, numeric_mode=None):
    """`scipy.stats.bootstrap((data,), statistic, n_resamples=...,
    rng=random_state, method=..., confidence_level=..., alternative=...)`.

    `data` is `(n,)` or `(n, 2)` float32; a two-column sample keeps its row
    pairing (SciPy's `paired=True`), which `pearson` and `diff_means` read.
    `q_or_prop` is `q` for `quantile` and `proportiontocut` for
    `trimmed_mean`, unread otherwise. `method='bca'` is refused by name
    (DEVIATION 1699).
    """
    where = "bootstrap"
    x, _ = as_f32_c(data, ndim=None, name="data")
    if x.ndim == 1:
        n, d = x.shape[0], 1
    elif x.ndim == 2:
        n, d = x.shape
    else:
        raise ValueError(f"mojolearn {where}: data must be 1-D or 2-D, got {x.ndim}-D")
    stat = _code(STATISTICS, statistic, "statistic", where)
    meth = _code(METHODS, method, "method", where)
    if meth == METHODS["bca"]:
        raise ValueError(
            f"mojolearn {where}: method='bca' is refused by name (DEVIATION 1699); "
            "resample/README.md carries the reason. Use 'percentile' or 'basic'."
        )
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


def permutation_test(x, y, statistic="diff_means", n_resamples=9999,
                     alternative="two-sided", random_state=0, r_first=0, numeric_mode=None):
    """`scipy.stats.permutation_test((x, y), statistic,
    permutation_type='independent', n_resamples=..., rng=random_state,
    alternative=...)`. `x` and `y` are 1-D float32. The null is never
    exhaustive (DEVIATION 1702); the p-value is conservative."""
    where = "permutation_test"
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


def monte_carlo_integrate(integrand, lower, upper, n_samples, random_state=0, i_first=0, numeric_mode=None):
    """`volume * mean(f(x_i))` over `n_samples` uniform draws from the
    two-dimensional box `[lower, upper)`. `integrand` is one of the three
    compiled arms, 'const' (f = 1), 'sum' (x0 + x1) or 'product'
    (x0 * x1); the closed form of the same integrand comes back beside
    the estimate."""
    where = "monte_carlo_integrate"
    f_id = _code(INTEGRANDS, integrand, "integrand", where)
    lo = Array.from_list([_real(v, "lower", where) for v in lower], "<f4")
    hi = Array.from_list([_real(v, "upper", where) for v in upper], "<f4")
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


__all__ = ["bootstrap", "permutation_test", "monte_carlo_integrate",
           "BootstrapResult", "PermutationTestResult", "MonteCarloResult",
           "STATISTICS", "METHODS", "ALTERNATIVES", "INTEGRANDS"]
