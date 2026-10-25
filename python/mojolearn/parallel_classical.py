# SPDX-License-Identifier: Apache-2.0
"""Fit classical estimators with their heaviest work spread across several GPUs.

Each function takes an unfitted (or, for prediction, a fitted) mojolearn
estimator, runs its fit or prediction on the GPUs listed in ``devices`` and
returns the same result, bit for bit, as the estimator's own one-GPU method.
The estimator must use the ``'identical'`` numeric mode (or the default).

Example
-------
>>> import numpy as np
>>> from mojolearn import KMeans
>>> from mojolearn.parallel_classical import fit_kmeans
>>> X = np.random.default_rng(0).standard_normal((100000, 16)).astype(np.float32)
>>> km = fit_kmeans(KMeans(n_clusters=8, random_state=3), X, devices=(0, 1))
>>> km.cluster_centers_.shape
(8, 16)
"""
# Cooperative classical estimators with explicit per-algorithm partitions.
__all__ = [
    'fit_logistic',
    'fit_exponential_smoothing',
    'fit_gram_estimator',
    'fit_arima',
    'fit_kmeans',
    'fit_coordinate_descent',
    'fit_svm',
    'predict_svm',
    'fit_gaussian_process',
    'predict_gaussian_process',
    'fit_dbscan',
    'fit_gaussian_mixture',
    'predict_gaussian_mixture',
    'bootstrap',
    'permutation_test',
    'monte_carlo_integrate',
    'fit_hdbscan',
    'fit_cholesky',
    'solve_cholesky',
    'fit_kernel_method',
    'apply_kernel_method',
    'transform_rbf_sampler',
]

from . import parallel_forecasting as _pf
from . import _native_family_bridge, _native_parallel_public

_FAMILY, _FAMILY_CONTEXT = _native_family_bridge.connect()
_native_parallel_public.connect(_FAMILY_CONTEXT)


def fit_logistic(estimator, X, y, *, devices=(0,)):
    """Fit a ``LogisticRegression`` with its gradient columns spread across GPUs.

    Parameters
    ----------
    estimator : LogisticRegression
        An unfitted estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Class labels.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_logistic', (X, y, devices)), _FAMILY_CONTEXT)


def fit_exponential_smoothing(estimator, *, devices=(0,), series_per_shard=1):
    """Fit an ``ExponentialSmoothing`` model over many series, groups of series on separate GPUs.

    Parameters
    ----------
    estimator : ExponentialSmoothing
        An unfitted model built on its series (``ts_num`` of them).
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    series_per_shard : int, default=1
        Number of series fitted by each GPU task.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    # One native call (`fit_exponential_smoothing`): whole series in
    # `series_per_shard` ranges, each shard fitted by the plain native fit in
    # a worker process per device, the packed components, statistics and
    # flags placed at each shard's series offset and stored as the plain fit
    # stores them.
    return _pf._native_parallel_forecast('fit_exponential_smoothing', (estimator, devices, series_per_shard))


def fit_gram_estimator(estimator, X, y=None, *, devices=(0,), sample_weight=None):
    """Fit a linear model or decomposition with its Gram-matrix blocks spread across GPUs.

    Parameters
    ----------
    estimator : LinearRegression, Ridge, PCA or TruncatedSVD
        An unfitted estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Targets (``LinearRegression`` and ``Ridge``); omit for ``PCA`` and ``TruncatedSVD``.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    sample_weight : array-like of shape (n_samples,), optional
        Per-row weights, as in ``estimator.fit``.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_gram_estimator', (X, y, devices, sample_weight)), _FAMILY_CONTEXT)


def fit_arima(estimator, y, *, devices=(0,), series_per_shard=1, exog=None):
    """Fit an ``ARIMA`` model over many series, groups of series on separate GPUs.

    Each GPU receives only its own series, so the whole batch need not fit on
    one GPU.

    Parameters
    ----------
    estimator : ARIMA
        An unfitted model.
    y : array-like of shape (n_series, n_obs)
        The series, one per row.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    series_per_shard : int, default=1
        Number of series fitted by each GPU task.
    exog : None
        Exogenous regressors are not supported here; use ``ARIMA.fit(y, exog)``.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    # One native call (bindings/parallel_forecast_native.mojo `fit_arima`):
    # whole series in `series_per_shard` ranges, each shard fitted by the
    # plain native fit in a worker process per device (shard i on device
    # i % len(devices)), every fitted array placed at its series offset.
    return _pf._native_parallel_forecast('fit_arima', (estimator, y, devices, series_per_shard, exog))


def fit_kmeans(estimator, X, *, devices=(0,), sample_weight=None):
    """Fit ``KMeans`` with the cluster assignment of row blocks spread across GPUs.

    Parameters
    ----------
    estimator : KMeans
        An unfitted estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    sample_weight : array-like of shape (n_samples,), optional
        Per-row weights.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_kmeans', (X, devices, sample_weight)), _FAMILY_CONTEXT)


def fit_coordinate_descent(estimator, X, y, *, devices=(0,)):
    """Fit ``Lasso`` or ``ElasticNet`` with the coordinate updates' dot products spread across GPUs.

    Parameters
    ----------
    estimator : Lasso or ElasticNet
        An unfitted estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Targets.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_coordinate_descent', (X, y, devices)), _FAMILY_CONTEXT)


def fit_svm(estimator, X, y, *, devices=(0,), sample_weight=None):
    """Fit ``SVC`` or ``SVR`` with kernel rows computed across GPUs.

    Parameters
    ----------
    estimator : SVC or SVR
        An unfitted estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Class labels (``SVC``) or targets (``SVR``).
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    sample_weight : array-like of shape (n_samples,), optional
        Per-row weights scaling each row's ``C``, as in ``estimator.fit``.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_svm', (X, y, devices, sample_weight)), _FAMILY_CONTEXT)


def predict_svm(estimator, X, *, devices=(0,), method='predict'):
    """Predict with a fitted ``SVC`` or ``SVR``, kernel rows computed across GPUs.

    Parameters
    ----------
    estimator : SVC or SVR
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Rows to predict.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    method : {'predict', 'decision_function'}, default='predict'
        ``'decision_function'`` is available for ``SVC`` only.

    Returns
    -------
    The same value ``estimator.<method>(X)`` returns.
    """
    return _FAMILY(estimator, 'family_par_public', ('predict_svm', (X, devices, method)), _FAMILY_CONTEXT)


def fit_gaussian_process(estimator, X, y, *, devices=(0,)):
    """Fit a ``GaussianProcessRegressor`` with covariance rows computed across GPUs.

    Parameters
    ----------
    estimator : GaussianProcessRegressor
        An unfitted estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Targets.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_gaussian_process', (X, y, devices)), _FAMILY_CONTEXT)


def predict_gaussian_process(estimator, X, *, devices=(0,), return_std=False):
    """Predict with a fitted ``GaussianProcessRegressor``, cross-covariance rows computed across GPUs.

    Parameters
    ----------
    estimator : GaussianProcessRegressor
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Rows to predict.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    return_std : bool, default=False
        Also return the predictive standard deviation.

    Returns
    -------
    mean, or (mean, std)
        As ``estimator.predict(X, return_std=return_std)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('predict_gaussian_process', (X, devices, return_std)), _FAMILY_CONTEXT)


def fit_dbscan(estimator, X, *, devices=(0,), sample_weight=None):
    """Fit ``DBSCAN`` with neighborhood rows computed across GPUs.

    Parameters
    ----------
    estimator : DBSCAN
        An unfitted estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    sample_weight : array-like of shape (n_samples,), optional
        Per-row weights.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_dbscan', (X, devices, sample_weight)), _FAMILY_CONTEXT)


def fit_gaussian_mixture(estimator, X, *, devices=(0,)):
    """Fit a ``GaussianMixture`` with every E-step's row blocks spread across GPUs.

    Parameters
    ----------
    estimator : GaussianMixture
        An unfitted estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_gaussian_mixture', (X, devices)), _FAMILY_CONTEXT)


def predict_gaussian_mixture(estimator, X, *, devices=(0,), method='predict'):
    """Score rows with a fitted ``GaussianMixture``, row blocks spread across GPUs.

    Parameters
    ----------
    estimator : GaussianMixture
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Rows to score.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    method : {'predict', 'predict_proba', 'score_samples'}, default='predict'
        The estimator method to run.

    Returns
    -------
    The same value ``estimator.<method>(X)`` returns.
    """
    return _FAMILY(estimator, 'family_par_public', ('predict_gaussian_mixture', (X, devices, method)), _FAMILY_CONTEXT)


def bootstrap(data, *, devices=(0,), statistic="mean", n_resamples=9999, confidence_level=0.95,
              method="percentile", alternative="two-sided", random_state=0, q_or_prop=0.5, r_first=0,
              numeric_mode=None, paired=True):
    """:func:`mojolearn.resample.bootstrap` with the replicates spread across GPUs.

    Parameters
    ----------
    data : array-like
        The sample, as in :func:`mojolearn.resample.bootstrap`.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    statistic, n_resamples, confidence_level, method, alternative, random_state, q_or_prop, r_first, numeric_mode, paired
        As in :func:`mojolearn.resample.bootstrap`.

    Returns
    -------
    BootstrapResult
        Equal to ``mojolearn.resample.bootstrap(data, ...)`` with the same arguments.
    """
    return _FAMILY(None, 'family_par_public', ('bootstrap', (data, statistic, n_resamples, confidence_level, method, alternative, random_state, q_or_prop, r_first, numeric_mode, paired, devices)), _FAMILY_CONTEXT)


def permutation_test(x, y, *, devices=(0,), statistic="diff_means", n_resamples=9999, alternative="two-sided",
                     random_state=0, r_first=0, numeric_mode=None, permutation_type="independent"):
    """:func:`mojolearn.resample.permutation_test` with the permutations spread across GPUs.

    Parameters
    ----------
    x, y : array-like
        The two samples.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    statistic, n_resamples, alternative, random_state, r_first, numeric_mode, permutation_type
        As in :func:`mojolearn.resample.permutation_test`.

    Returns
    -------
    PermutationTestResult
        Equal to ``mojolearn.resample.permutation_test(x, y, ...)`` with the same arguments.
    """
    return _FAMILY(None, 'family_par_public', ('permutation_test', (x, y, statistic, n_resamples, alternative, random_state, r_first, numeric_mode, permutation_type, devices)), _FAMILY_CONTEXT)


def monte_carlo_integrate(integrand, lower, upper, n_samples, *, devices=(0,), random_state=0, i_first=0,
                          numeric_mode=None):
    """:func:`mojolearn.resample.monte_carlo_integrate` with the samples spread across GPUs.

    Parameters
    ----------
    integrand, lower, upper, n_samples
        As in :func:`mojolearn.resample.monte_carlo_integrate`.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    random_state, i_first, numeric_mode
        As in :func:`mojolearn.resample.monte_carlo_integrate`.

    Returns
    -------
    MonteCarloResult
        Equal to the one-GPU result.
    """
    return _FAMILY(None, 'family_par_public', ('monte_carlo_integrate', (integrand, lower, upper, n_samples, random_state, i_first, numeric_mode, devices)), _FAMILY_CONTEXT)


def fit_hdbscan(estimator, X, *, devices=(0,)):
    """Fit ``HDBSCAN`` with the core-distance and pairwise-distance rows computed across GPUs.

    Parameters
    ----------
    estimator : HDBSCAN
        An unfitted estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_hdbscan', (X, devices)), _FAMILY_CONTEXT)


def fit_cholesky(estimator, A, *, devices=(0,)):
    """Factor a symmetric positive-definite matrix with ``Cholesky``, the trailing updates spread across GPUs.

    Parameters
    ----------
    estimator : Cholesky
        An unfitted estimator.
    A : array-like of shape (n, n)
        The matrix, converted to float32.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_cholesky', (A, devices)), _FAMILY_CONTEXT)


def solve_cholesky(estimator, B, *, devices=(0,)):
    """Solve ``A x = B`` with a fitted ``Cholesky``, right-hand-side columns spread across GPUs.

    Parameters
    ----------
    estimator : Cholesky
        A fitted estimator.
    B : array-like of shape (n,) or (n, n_rhs)
        Right-hand sides. A single right-hand side runs on one GPU.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    Array
        Equal to ``estimator.solve(B)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('solve_cholesky', (B, devices)), _FAMILY_CONTEXT)


def fit_kernel_method(estimator, X, y=None, *, devices=(0,)):
    """Fit ``KernelRidge`` or ``Nystroem`` with kernel-matrix rows computed across GPUs.

    The ``'laplacian'`` kernel is not supported.

    Parameters
    ----------
    estimator : KernelRidge or Nystroem
        An unfitted estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets), optional
        Targets; required for ``KernelRidge``.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(...)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('fit_kernel_method', (X, y, devices)), _FAMILY_CONTEXT)


def apply_kernel_method(estimator, X, *, devices=(0,)):
    """``KernelRidge.predict`` or ``Nystroem.transform`` with kernel rows computed across GPUs.

    Parameters
    ----------
    estimator : KernelRidge or Nystroem
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Rows to predict or transform.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.

    Returns
    -------
    Array
        Equal to ``estimator.predict(X)`` or ``estimator.transform(X)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('apply_kernel_method', (X, devices)), _FAMILY_CONTEXT)


def transform_rbf_sampler(estimator, X, *, devices=(0,), rows_per_shard=4096):
    """``RBFSampler.transform`` with blocks of rows on separate GPUs.

    Parameters
    ----------
    estimator : RBFSampler
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Rows to transform.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    rows_per_shard : int, default=4096
        Number of rows in each block.

    Returns
    -------
    Array of shape (n_samples, n_components)
        Equal to ``estimator.transform(X)``.
    """
    return _FAMILY(estimator, 'family_par_public', ('transform_rbf_sampler', (X, devices, rows_per_shard)), _FAMILY_CONTEXT)
