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

import sys

from ._parallel_pool import public_method_result
from ._buffer import as_f32_c
from . import kernel_methods as _km
from . import _cholesky_impl as _chol
from . import _svm_impl as _svm
from . import _gp_impl as _gp
from . import linear_model as _lm
from . import _solver_impl as _cd
from . import decomposition as _dec
from . import cluster as _cl
from . import density as _den
from . import mixture as _mix
from . import hdbscan as _hdb
from . import parallel_forecasting as _pf
from . import _backend
from . import _native_family_bridge

# The classical families' compiled owner: a cooperative driver is one
# `family_par_run` call (bindings/family_native.mojo).
_FAMILY, _FAMILY_CONTEXT = _native_family_bridge.connect()

# The Python-sharded drivers' owner (bindings/parallel_shard_native.mojo).
from . import _native_parallel_shard as _shard  # noqa: E402
from ._native_parallel_shard import _native_parallel_shard  # noqa: E402
_shard.register(context=_FAMILY_CONTEXT, rbf_class=_km.RBFSampler)

#: fit_gram_estimator's plain owners: (entry, operation, context, owner arity).
_GRAM_OWNERS = {
    _lm.LinearRegression: (_lm._native_classical, 'ols_fit', _lm._CLASSICAL, 3),
    _lm.Ridge: (_lm._native_classical, 'ridge_fit', _lm._CLASSICAL, 3),
    _dec.PCA: (_dec._PCA_ENTRY, 'fit', _dec._PCA_CONTEXT, 2),
    _dec.TruncatedSVD: (_dec._TSVD_ENTRY, 'fit', _dec._TSVD_CONTEXT, 2),
}

#: predict_gaussian_mixture's plain owners.
_GMM_PREDICT = {'score_samples': 'family_mix_gmm_score_samples', 'predict_proba': 'family_mix_gmm_predict_proba',
                'predict': 'family_mix_gmm_predict'}

#: The plain owner operations of the SVM drivers.
_SVM_FIT = {_svm.SVC: 'family_svc_svc_fit', _svm.SVR: 'family_svc_svr_fit'}
_SVM_PREDICT = {(_svm.SVC, 'predict'): 'family_svc_svc_predict',
                (_svm.SVC, 'decision_function'): 'family_svc_svc_decision',
                (_svm.SVR, 'predict'): 'family_svc_svr_predict'}

#: The estimators the kernel-method drivers admit, and the plain method whose
#: public conversion `apply_kernel_method` applies.
_KM_CLASSES = (_km.KernelRidge, _km.Nystroem)
_APPLY_METHOD = {_km.KernelRidge: 'predict', _km.Nystroem: 'transform'}


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
    # One native call (bindings/family_native.mojo `family_par_run`, driver
    # `glm_fit`): the plain LogisticRegression fit (bindings/logistic_native.mojo)
    # with its gradient feature columns on the selected devices; the objective
    # and the QN trajectory are unchanged. Data and optimizer state stay on
    # the root.
    if type(estimator) is not _lm.LogisticRegression:
        raise TypeError('fit_logistic requires mojolearn.LogisticRegression')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel logistic regression requires IDENTICAL numeric mode')
    return _FAMILY(estimator, 'family_par_run',
                   ('glm_fit', devices, 'fit', (X, y, None), _lm._LR_CALL, _lm._LR_CONTEXT,
                    (estimator._bind('_mojolearn_estimators'),)), _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `gram_fit`): the plain fit
    # with its pinned Gram chunks (and, for full PCA, its QR panels) on the
    # selected devices. The root still owns preprocessing and solver state;
    # this is computation partitioning, not a fit beyond one GPU's memory.
    if type(estimator) not in _GRAM_OWNERS:
        raise TypeError('fit_gram_estimator requires LinearRegression, Ridge, PCA or TruncatedSVD')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel Gram estimators require IDENTICAL numeric mode')
    data, _ = as_f32_c(X, ndim=2, name='X')
    if data.shape[1] < 1:
        raise ValueError('parallel Gram requires at least one feature')
    if type(estimator) is _dec.PCA and estimator.svd_solver in estimator._DENSE_SOLVERS and data.shape[0] < data.shape[1]:
        raise ValueError('parallel full PCA currently requires the tall TSQR route')
    if sample_weight is not None and type(estimator) in (_dec.PCA, _dec.TruncatedSVD):
        raise TypeError("fit() got an unexpected keyword argument 'sample_weight'")
    entry, operation, context, arity = _GRAM_OWNERS[type(estimator)]
    return _FAMILY(estimator, 'family_par_run',
                   ('gram_fit', devices, operation, (data, y, sample_weight)[:arity], entry, context,
                    _backend.plan_bindings(('_mojolearn_estimators', '_mojolearn_x_linear',
                                            '_mojolearn_x_decomp'))), _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `kmeans_fit`): the plain
    # KMeans fit, in identical mode, with its assignment row tiles on the
    # selected devices. The old worker fitted a fresh KMeans of these seven
    # parameters; the two it did not carry are refused when not their
    # defaults rather than silently replaced.
    if type(estimator) is not _cl.KMeans:
        raise TypeError('fit_kmeans requires mojolearn.KMeans')
    if getattr(estimator, 'numeric_mode', None) not in (None, 'identical'):
        raise ValueError('parallel KMeans requires IDENTICAL numeric mode')
    if getattr(estimator, 'metric', 'euclidean') != 'euclidean':
        raise ValueError("parallel KMeans requires metric='euclidean'")
    if getattr(estimator, 'oversampling_factor', 2.0) != 2.0:
        raise ValueError('parallel KMeans requires the default oversampling_factor')
    # X and sample_weight reach the plain fit, which converts both exactly as
    # the old driver's float32 conversion did.
    X, _ = as_f32_c(X, ndim=2, name='X')
    estimator.numeric_mode = 'identical'
    return _FAMILY(estimator, 'family_par_run',
                   ('kmeans_fit', devices, 'family_kmeans_fit', (estimator._bind(), X, sample_weight), None, None,
                    _backend.plan_bindings(('_mojolearn', '_mojolearn_x_cluster'))), _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `solver_fit`): the plain
    # Lasso/ElasticNet fit with its multi-leaf dots on the selected devices;
    # single-leaf dots, the serial coordinate order and the convergence order
    # stay on the root.
    if type(estimator) not in (_cd.Lasso, _cd.ElasticNet):
        raise TypeError('requires mojolearn.Lasso or ElasticNet')
    if _backend.default_mode() != 'identical':
        raise ValueError('parallel coordinate descent requires IDENTICAL numeric mode')
    return _FAMILY(estimator, 'family_par_run',
                   ('solver_fit', devices, 'cd_fit', (X, y, None), _cd._CD_CALL, _cd._CD_CONTEXT,
                    _backend.plan_bindings(('_mojolearn_solver',))), _FAMILY_CONTEXT)


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
    # One native call (bindings/family_native.mojo `family_par_run`, driver
    # `svm_fit`): the plain SVC/SVR fit with linear/RBF kernel rows on the
    # selected devices. Full data, working-set state and the output kernel
    # tile stay on the root; `sample_weight` is checked and scales each row's
    # C exactly as the plain fit does.
    if type(estimator) not in _SVM_FIT:
        raise TypeError('requires mojolearn.SVC or SVR')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel SVM requires IDENTICAL')
    return _FAMILY(estimator, 'family_par_run',
                   ('svm_fit', devices, _SVM_FIT[type(estimator)],
                    (estimator._bind(_svm._EXT_NAME), X, y, sample_weight), None, None,
                    (estimator._bind(_svm._EXT_NAME),)), _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `svm_predict`): prediction
    # kernel rows on the selected devices, the support-vector fold unchanged.
    if type(estimator) not in _SVM_FIT:
        raise TypeError('requires mojolearn.SVC or SVR')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel SVM requires IDENTICAL')
    if (type(estimator), method) not in _SVM_PREDICT:
        raise ValueError('method must be predict, or decision_function for SVC')
    # what the plain estimator.<method>(X) returns to this caller
    return public_method_result(estimator, method,
                                _FAMILY(estimator, 'family_par_run',
                                        ('svm_predict', devices, _SVM_PREDICT[(type(estimator), method)],
                                         (estimator._bind(_svm._EXT_NAME), X), None, None,
                                         (estimator._bind(_svm._EXT_NAME),)), _FAMILY_CONTEXT),
                                sys._getframe(1))


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
    # One native call (`family_par_run`, driver `gp_fit`): covariance rows on
    # the selected devices; the Cholesky and solve order stay on the root.
    if type(estimator) is not _gp.GaussianProcessRegressor:
        raise TypeError('requires mojolearn.GaussianProcessRegressor')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel Gaussian processes require IDENTICAL')
    return _FAMILY(estimator, 'family_par_run',
                   ('gp_fit', devices, 'family_gp_gpr_fit', (estimator._bind(), X, y), None, None,
                    (estimator._bind(),)), _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `gp_predict`): cross-covariance
    # rows on the selected devices, the prediction and variance folds on the
    # root. The plain predict records `clamped_` / `n_clamped_` itself.
    if type(estimator) is not _gp.GaussianProcessRegressor:
        raise TypeError('requires mojolearn.GaussianProcessRegressor')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel Gaussian processes require IDENTICAL')
    if type(return_std) is not bool:
        raise ValueError('return_std must be bool')
    return public_method_result(estimator, 'predict',
                                _FAMILY(estimator, 'family_par_run',
                                        ('gp_predict', devices, 'family_gp_gpr_predict',
                                         (estimator._bind(), X, return_std, False), None, None,
                                         (estimator._bind(),)), _FAMILY_CONTEXT),
                                sys._getframe(1))


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
    # One native call (`family_par_run`, driver `dbscan_fit`): the plain
    # DBSCAN fit with its neighborhood rows on the selected devices.
    if type(estimator) is not _den.DBSCAN:
        raise TypeError('fit_dbscan requires mojolearn.DBSCAN')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel DBSCAN requires IDENTICAL numeric mode')
    return _FAMILY(estimator, 'family_par_run',
                   ('dbscan_fit', devices, 'family_cfb_dbscan_fit',
                    (estimator._bind('_mojolearn_estimators'), X, sample_weight), None, None,
                    _backend.plan_bindings(('_mojolearn_estimators',))), _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `gmm_fit`): the plain fit with
    # every E-step's rows on the selected devices (and its KMeans
    # initialization's assignment tiles, as the old worker's environment did).
    if type(estimator) is not _mix.GaussianMixture:
        raise TypeError('requires mojolearn.GaussianMixture')
    if getattr(estimator, 'numeric_mode', None) not in (None, 'identical'):
        raise ValueError('parallel GaussianMixture requires IDENTICAL numeric mode')
    X, _ = as_f32_c(X, ndim=2, name='X')
    return _FAMILY(estimator, 'family_par_run',
                   ('gmm_fit', devices, 'family_mix_gmm_fit',
                    (estimator._bind(), estimator._bind('_mojolearn_x_cluster'), X), None, None,
                    _backend.plan_bindings(('_mojolearn_mixture', '_mojolearn_x_cluster', '_mojolearn'))),
                   _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `gmm_predict`): the plain
    # scoring E-step's rows on the selected devices.
    if type(estimator) is not _mix.GaussianMixture:
        raise TypeError('requires mojolearn.GaussianMixture')
    if getattr(estimator, 'numeric_mode', None) not in (None, 'identical'):
        raise ValueError('parallel GaussianMixture requires IDENTICAL numeric mode')
    if method not in _GMM_PREDICT:
        raise ValueError('method must be predict, predict_proba or score_samples')
    if not hasattr(estimator, 'weights_'):
        raise ValueError('GaussianMixture is not fitted')
    X, _ = as_f32_c(X, ndim=2, name='X')
    return public_method_result(estimator, method,
                                _FAMILY(estimator, 'family_par_run',
                                        ('gmm_predict', devices, _GMM_PREDICT[method], (estimator._bind(), X),
                                         None, None, _backend.plan_bindings(('_mojolearn_mixture',))),
                                        _FAMILY_CONTEXT),
                                sys._getframe(1))


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
    # One native call (`family_par_run`, driver `resample`): the plain
    # bootstrap with global replicate ranges on the selected devices.
    # Replicate r is a pure function of (seed, r, data) (the r_first handle),
    # so owners compute contiguous ranges of global replicate IDs and the root
    # assembles the distribution in order, then sorts it and computes the point
    # estimate, interval and standard error exactly as one device does.
    if numeric_mode not in (None, 'identical'):
        raise ValueError('parallel resampling requires IDENTICAL numeric mode')
    return _FAMILY(None, 'family_par_run',
                   ('resample', devices, 'family_rs_bootstrap',
                    (_backend.binding('_mojolearn_resample', 'identical'), data, statistic, n_resamples,
                     confidence_level, method, alternative, random_state, q_or_prop, r_first, 'identical', paired),
                    None, None, _backend.plan_bindings(('_mojolearn_resample',))), _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `resample`): global
    # permutation ranges on the selected devices.
    if numeric_mode not in (None, 'identical'):
        raise ValueError('parallel resampling requires IDENTICAL numeric mode')
    return _FAMILY(None, 'family_par_run',
                   ('resample', devices, 'family_rs_permutation_test',
                    (_backend.binding('_mojolearn_resample', 'identical'), x, y, statistic, n_resamples, alternative,
                     random_state, r_first, 'identical', permutation_type),
                    None, None, _backend.plan_bindings(('_mojolearn_resample',))), _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `resample`): owners draw
    # global sample IDs from aligned PINNED_SUM_W chunks; the root folds the
    # chunk partials in order and forms the integral as one device does.
    if numeric_mode not in (None, 'identical'):
        raise ValueError('parallel resampling requires IDENTICAL numeric mode')
    return _FAMILY(None, 'family_par_run',
                   ('resample', devices, 'family_rs_monte_carlo_integrate',
                    (_backend.binding('_mojolearn_resample', 'identical'), integrand, lower, upper, n_samples,
                     random_state, i_first, 'identical'),
                    None, None, _backend.plan_bindings(('_mojolearn_resample',))), _FAMILY_CONTEXT)


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
    # One native call (`family_par_run`, driver `hdbscan_fit`): the plain
    # HDBSCAN fit with its core-distance neighbor rows and pairwise distance
    # rows on the selected devices.
    if type(estimator) is not _hdb.HDBSCAN:
        raise TypeError('fit_hdbscan requires mojolearn.HDBSCAN')
    if getattr(estimator, 'numeric_mode', None) not in (None, 'identical'):
        raise ValueError('parallel HDBSCAN requires IDENTICAL numeric mode')
    X, _ = as_f32_c(X, ndim=2, name='X')
    return _FAMILY(estimator, 'family_par_run',
                   ('hdbscan_fit', devices, 'family_cfa_hdbscan_fit', (estimator._bind(), X), None, None,
                    _backend.plan_bindings(('_mojolearn_hdbscan', '_mojolearn_solver'))), _FAMILY_CONTEXT)


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
    # One native call (bindings/linalg_native.mojo `family_chol_par_fit`).
    # Each panel's factorization, info read-back, panel solve and subtraction
    # stay on the root, devices[0], in the original panel order; only the rows
    # of each panel's L21 L21^T product move.
    if type(estimator) is not _chol.Cholesky:
        raise TypeError('requires mojolearn.Cholesky')
    return _chol._NATIVE(estimator, 'family_chol_par_fit', (estimator._door_binding(), A, devices),
                         _chol._NATIVE_CONTEXT)


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
    # One native call (`family_chol_par_solve`): whole right-hand-side columns
    # on the selected GPUs; a single right-hand side is sequential in its rows
    # and runs on one GPU.
    if type(estimator) is not _chol.Cholesky:
        raise TypeError('requires mojolearn.Cholesky')
    return _chol._NATIVE(estimator, 'family_chol_par_solve', (estimator._door_binding(), B, devices),
                         _chol._NATIVE_CONTEXT)


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
    # One native call (bindings/kernel_methods_native.mojo `family_km_par_fit`):
    # it admits the estimator, the devices and the installed route, binds the
    # devices to this thread in the binding (core/device_plan.mojo) and runs
    # the plain fit. Rows use the SVM kernel-row seam (original FP32-v1 dot per
    # cell and the original epilogue); KernelRidge's factorization moves whole
    # trailing-update rows and its multi-target solve whole target columns;
    # Nystroem's Jacobi eigensolver stays on the root, devices[0].
    if type(estimator) not in _KM_CLASSES:
        raise TypeError('requires mojolearn.KernelRidge or Nystroem')
    return _km._NATIVE(estimator, 'family_km_par_fit', (estimator._bind(), X, y, devices), _km._NATIVE_CONTEXT)


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
    # One native call (`family_km_par_apply`) returns the native result; the
    # public conversion for this function's caller is the plain method's.
    if type(estimator) not in _KM_CLASSES:
        raise TypeError('requires mojolearn.KernelRidge or Nystroem')
    return public_method_result(estimator, _APPLY_METHOD[type(estimator)],
                                _km._NATIVE(estimator, 'family_km_par_apply', (estimator._bind(), X, devices),
                                            _km._NATIVE_CONTEXT),
                                sys._getframe(1))


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
    # One native call (bindings/parallel_shard_native.mojo
    # `transform_rbf_sampler`): whole query rows in `rows_per_shard` blocks,
    # each transformed by the plain native transform in a worker process per
    # device (block i on device i % len(devices)), the blocks joined by
    # copying bytes in input order. The fitted random weights and offsets are
    # position-mapped draws with global feature and component IDs, so every
    # worker receives the same fitted state; each output row is a pure
    # function of its input row.
    if type(estimator) is not _km.RBFSampler:
        raise TypeError('requires mojolearn.RBFSampler')
    # the joined raw rows; the plain transform's public conversion for our caller
    return public_method_result(estimator, 'transform',
                                _native_parallel_shard('transform_rbf_sampler',
                                                       (estimator, X, devices, rows_per_shard, estimator._bind())),
                                sys._getframe(1))
