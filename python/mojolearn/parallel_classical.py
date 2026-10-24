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

from ._parallel_pool import DevicePool, driver_read_shift, public_method_result
from ._buffer import as_f32_c
from . import kernel_methods as _km
from . import _cholesky_impl as _chol

#: The plain method whose public conversion `apply_kernel_method` applies.
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
    # Partition gradient feature columns; retain the original objective and QN trajectory.
    #
    # Supports the existing binary and multinomial objectives and their admitted
    # penalties. Data and optimizer state still have to fit on the root GPU.
    from .linear_model import LogisticRegression
    if type(estimator) is not LogisticRegression:
        raise TypeError('fit_logistic requires mojolearn.LogisticRegression')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel logistic regression requires IDENTICAL numeric mode')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('glm_fit', estimator, (X, y))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


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
    # Fit Holt-Winters series independently and assemble original component layouts.
    from ._tsa_impl import ExponentialSmoothing
    from . import _backend
    from ._array import _strided_copy
    from ._buffer import empty, addr, addr_ro
    from ._bufcheck import memcopy
    if type(estimator) is not ExponentialSmoothing:
        raise TypeError('requires mojolearn.ExponentialSmoothing')
    if _backend.default_mode() != 'identical':
        raise ValueError('ExponentialSmoothing requires IDENTICAL numeric mode')
    if type(series_per_shard) is not int or series_per_shard < 1:
        raise ValueError('series_per_shard must be a positive integer')
    if estimator.ts_num < 1:
        raise ValueError('at least one series is required')
    data, n = estimator._check_dims(estimator.endog)
    batch = estimator.ts_num
    params = dict(seasonal=estimator.seasonal, seasonal_periods=estimator.seasonal_periods,
                  start_periods=estimator.start_periods, eps=estimator.eps,
                  initialization_method=estimator.initialization_method)
    pool = DevicePool(devices)
    try:
        parts = pool.map([
            ('holtwinters_fit', params,
             (data[start - driver_read_shift(index, start, devices):
                   min(start + series_per_shard, batch) - driver_read_shift(index, start, devices)],))
            for index, start in enumerate(range(0, batch, series_per_shard))])  # glue: one request per device shard
    finally:
        pool.close()
    result = ExponentialSmoothing(estimator.endog, ts_num=batch, **params)
    result.__dict__.update(parts[0].__dict__)
    result.endog, result._data, result.ts_num, result.n = estimator.endog, data, batch, n
    for name in ('level_', 'trend_', 'season_', 'sse_', 'alpha_', 'beta_', 'gamma_', 'n_iter_', 'criterion_'):  # glue: nine fitted attribute names
        arrays = [getattr(part, name) for part in parts]  # glue: one buffer per device shard
        merged = empty((batch,) + arrays[0].shape[1:], arrays[0].dtype)
        cursor = 0
        for array in arrays:  # glue: one memcopy per device shard
            memcopy(addr(merged, name=name) + cursor, addr_ro(array, name=name), array.nbytes)
            cursor += array.nbytes
        setattr(result, name, merged)
    steps = n - estimator.seasonal_periods
    result._components_len = steps * batch
    result._comps = empty((3 * result._components_len,), '<f4')
    destination = addr(result._comps, name='components')
    start = 0
    for part in parts:  # glue: one strided copy per device shard
        # the shard's (3, steps, ts) block into columns [start, start + ts) of
        # the (3, steps, batch) block: one Mojo strided copy (strided_copy_bytes)
        if steps > 0 and part.ts_num > 0:
            _strided_copy(addr_ro(part._comps, name='component shard'), destination, 4,
                          (3, steps, part.ts_num), (steps * part.ts_num, part.ts_num, 1),
                          (steps * batch, batch, 1), 0, start)
        start += part.ts_num
    cl = result._components_len
    result._time_major = {name: result._comps[i * cl:(i + 1) * cl].reshape((steps, batch))
                          for i, name in enumerate(('level', 'trend', 'season'))}  # glue: three component views by name
    result.fit_executed_flag = True
    estimator.__dict__ = result.__dict__.copy()
    return estimator


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
    # Distribute the existing pinned Gram chunks for OLS, Ridge, PCA and SVD.
    #
    # This entry admits the original Gram/eigendecomposition and wide OLS paths.
    # The root still owns preprocessing and solver state; this is computation
    # partitioning, not a fit beyond one GPU's memory capacity.
    from .linear_model import LinearRegression, Ridge
    from .decomposition import PCA, TruncatedSVD
    if type(estimator) not in (LinearRegression, Ridge, PCA, TruncatedSVD):
        raise TypeError('fit_gram_estimator requires LinearRegression, Ridge, PCA or TruncatedSVD')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel Gram estimators require IDENTICAL numeric mode')
    data, _ = as_f32_c(X, ndim=2, name='X')
    rows, columns = data.shape
    if columns < 1:
        raise ValueError('parallel Gram requires at least one feature')
    if type(estimator) is PCA and estimator.svd_solver in estimator._DENSE_SOLVERS and rows < columns:
        raise ValueError('parallel full PCA currently requires the tall TSQR route')
    kwargs = {} if sample_weight is None else dict(sample_weight=sample_weight)
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('gram_fit', estimator, (data, y, kwargs))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


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
    # Fit independent series on GPU workers, retaining each solver's order.
    #
    # A worker receives only its series, so fitting does not require the entire
    # batch on one GPU. A single series must still fit on one GPU. The returned
    # estimator's existing prediction methods are unchanged and use one GPU.
    from ._arima_impl import ARIMA, _series_major
    from ._buffer import empty, addr, addr_ro
    from ._bufcheck import memcopy
    if type(estimator) is not ARIMA:
        raise TypeError('fit_arima requires mojolearn.ARIMA')
    if exog is not None:
        raise NotImplementedError(
            'fit_arima does not shard exogenous regressors; ARIMA.fit(y, exog) '
            'fits them on one device')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel ARIMA requires IDENTICAL numeric mode')
    if type(series_per_shard) is not int or series_per_shard < 1:
        raise ValueError('series_per_shard must be a positive integer')
    data, batch, observations, copied = _series_major(y, 'y')
    # Keep the caller's trend spelling: resolving None to 'c'/'n' preserves
    # k_, but changes the returned configuration and serialized model bytes.
    params = dict(order=estimator.order, seasonal_order=estimator.seasonal_order,
                  trend=estimator.trend, method=estimator.method,
                  maxiter=estimator.maxiter, numeric_mode='identical')
    requests = [('arima_fit', params,
                 (data[start - driver_read_shift(index, start, devices):
                       min(start + series_per_shard, batch) - driver_read_shift(index, start, devices)],))
                for index, start in enumerate(range(0, batch, series_per_shard))]  # glue: one request per device shard
    pool = DevicePool(devices)
    try:
        parts = pool.map(requests)
    finally:
        pool.close()
    result = ARIMA(**params)
    result.__dict__.update(parts[0].__dict__)
    for name in ('params_', 'x_', 'x0_', 'n_iter_', 'retcode_', 'llf_', 'fx_', 'aic_', 'bic_'):  # glue: nine fitted attribute names
        arrays = [getattr(part, name) for part in parts]  # glue: one buffer per device shard
        merged = empty((batch,) + arrays[0].shape[1:], arrays[0].dtype)
        cursor = 0
        for array in arrays:  # glue: one memcopy per device shard
            memcopy(addr(merged, name=name) + cursor, addr_ro(array, name=name), array.nbytes)
            cursor += array.nbytes
        setattr(result, name, merged)
    result.batch_size_, result.n_obs_ = batch, observations
    result._y, result.input_copied_ = data, copied
    estimator.__dict__ = result.__dict__.copy()
    return estimator


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
    # Distribute KMeans row assignments; retain full-data update/reduction order.
    #
    # Initialization, fixed-point centroid sums, restart choice and convergence
    # use the existing implementation. Whole aligned row tiles compute distances
    # concurrently. This first path stages buffers per assignment call.
    from .cluster import KMeans
    if type(estimator) is not KMeans:
        raise TypeError('fit_kmeans requires mojolearn.KMeans')
    params = {name: getattr(estimator, name) for name in (  # glue: copies seven constructor parameters
        'n_clusters', 'init', 'n_init', 'max_iter', 'tol', 'random_state', 'init_centroids')}
    mode = getattr(estimator, 'numeric_mode', None)
    if mode not in (None, 'identical'):
        raise ValueError('parallel KMeans requires IDENTICAL numeric mode')
    params['numeric_mode'] = 'identical'
    X, _ = as_f32_c(X, ndim=2, name='X')
    if sample_weight is not None:
        sample_weight, _ = as_f32_c(sample_weight, ndim=1, name='sample_weight')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('kmeans_fit', params, (X, sample_weight))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


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
    # Distribute original dot leaves; preserve serial coordinate and convergence order.
    #
    # Single-leaf dots remain on the root. Full data and residual state must fit
    # on the root; transport per coordinate can outweigh computation savings.
    from ._solver_impl import Lasso, ElasticNet
    from . import _backend
    if type(estimator) not in (Lasso, ElasticNet):
        raise TypeError('requires mojolearn.Lasso or ElasticNet')
    if _backend.default_mode() != 'identical':
        raise ValueError('parallel coordinate descent requires IDENTICAL numeric mode')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('solver_fit', estimator, (X, y))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


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
    # Distribute linear/RBF kernel rows, retaining the original SVC/SVR solver.
    #
    # ``sample_weight`` (optional, one non-negative weight per row) scales each
    # row's ``C`` exactly as ``SVC.fit`` / ``SVR.fit`` do.
    # Full data, working-set state and output kernel tile remain on the root.
    # A one-row kernel operation has only one active device.
    from ._svm_impl import SVC, SVR
    if type(estimator) not in (SVC, SVR):
        raise TypeError('requires mojolearn.SVC or SVR')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel SVM requires IDENTICAL')
    pool = DevicePool(devices, cooperative=True)
    try:
        # sample_weight goes to the estimator's own fit, which checks it
        # (scikit-learn's rules and wording, _sample_weight.check_sample_weight)
        # and scales each row's C as SVC/SVR do on one device
        result = pool.map([('svm_fit', estimator, (X, y, sample_weight))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


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
    # Distribute prediction kernel rows; retain the original support-vector fold.
    from ._svm_impl import SVC, SVR
    if type(estimator) not in (SVC, SVR):
        raise TypeError('requires mojolearn.SVC or SVR')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel SVM requires IDENTICAL')
    if method not in ('predict', 'decision_function') or (type(estimator) is SVR and method != 'predict'):
        raise ValueError('method must be predict, or decision_function for SVC')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('svm_predict', estimator, (X, method))])[0]
    finally:
        pool.close()
    # what the plain estimator.<method>(X) returns to this caller
    return public_method_result(estimator, method, result, sys._getframe(1))


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
    # Distribute covariance rows; retain the original Cholesky and solve order.
    from ._gp_impl import GaussianProcessRegressor
    if type(estimator) is not GaussianProcessRegressor:
        raise TypeError('requires mojolearn.GaussianProcessRegressor')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel Gaussian processes require IDENTICAL')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('gp_fit', estimator, (X, y))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


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
    # Distribute cross-covariance rows; retain root prediction/variance folds.
    from ._gp_impl import GaussianProcessRegressor
    if type(estimator) is not GaussianProcessRegressor:
        raise TypeError('requires mojolearn.GaussianProcessRegressor')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel Gaussian processes require IDENTICAL')
    if type(return_std) is not bool:
        raise ValueError('return_std must be bool')
    pool = DevicePool(devices, cooperative=True)
    try:
        result, diagnostic = pool.map([('gp_predict', estimator, (X, return_std))])[0]
    finally:
        pool.close()
    if return_std:
        estimator.clamped_, estimator.n_clamped_ = diagnostic
    # the worker's raw result; the plain method's public conversion for our caller
    return public_method_result(estimator, 'predict', result, sys._getframe(1))


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
    # Distribute neighborhood rows; retain root core-point and label merges.
    #
    # Brute L2/L1 and RBC L2 use their original distance/order contracts.
    # Full reference data, the current batch graph and label state still need
    # to fit on the root GPU. This does not pool the reference index.
    from .density import DBSCAN
    if type(estimator) is not DBSCAN:
        raise TypeError('fit_dbscan requires mojolearn.DBSCAN')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel DBSCAN requires IDENTICAL numeric mode')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('dbscan_fit', estimator, (X, sample_weight))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def _admit_gaussian_mixture(estimator):
    from .mixture import GaussianMixture
    if type(estimator) is not GaussianMixture:
        raise TypeError('requires mojolearn.GaussianMixture')
    if getattr(estimator, 'numeric_mode', None) not in (None, 'identical'):
        raise ValueError('parallel GaussianMixture requires IDENTICAL numeric mode')


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
    # Row-shard every E-step; retain the root M-step, Cholesky and convergence test.
    #
    # Whole sample rows run the original E-step on their owners and are copied
    # back into their original positions; the mean log likelihood is folded on
    # the root over the complete gathered rows. The KMeans initialization uses
    # its row-tile assignment driver. Full data, responsibilities and the M-step
    # still have to fit on the root GPU. covariance_type other than 'full' and
    # the refused knobs are refused by the estimator itself, by name.
    _admit_gaussian_mixture(estimator)
    X, _ = as_f32_c(X, ndim=2, name='X')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('gmm_fit', estimator, (X,))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


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
    # Row-shard the scoring E-step of a fitted GaussianMixture.
    #
    # method is 'predict', 'predict_proba' or 'score_samples'; the original
    # per-row argmax and host exponential are unchanged.
    _admit_gaussian_mixture(estimator)
    if method not in ('score_samples', 'predict_proba', 'predict'):
        raise ValueError('method must be predict, predict_proba or score_samples')
    if not hasattr(estimator, 'weights_'):
        raise ValueError('GaussianMixture is not fitted')
    X, _ = as_f32_c(X, ndim=2, name='X')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('gmm_predict', estimator, (method, X))])[0]
    finally:
        pool.close()
    # the worker's raw result; the plain method's public conversion for our caller
    return public_method_result(estimator, method, result, sys._getframe(1))


def _resample_parallel(name, devices, kwargs):
    if kwargs.get('numeric_mode') not in (None, 'identical'):
        raise ValueError('parallel resampling requires IDENTICAL numeric mode')
    kwargs = {k: v for k, v in kwargs.items() if k != 'numeric_mode'}  # glue: drops one keyword argument
    pool = DevicePool(devices, cooperative=True)
    try:
        return pool.map([('resample', None, (name, kwargs))])[0]
    finally:
        pool.close()


def bootstrap(data, *, devices=(0,), **kwargs):
    """:func:`mojolearn.resample.bootstrap` with the replicates spread across GPUs.

    Parameters
    ----------
    data : array-like
        The sample, as in :func:`mojolearn.resample.bootstrap`.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    **kwargs
        Every other argument of :func:`mojolearn.resample.bootstrap`.

    Returns
    -------
    BootstrapResult
        Equal to ``mojolearn.resample.bootstrap(data, **kwargs)``.
    """
    # `mojolearn.resample.bootstrap` with global replicate ranges on the GPUs.
    #
    # Replicate r is a pure function of (seed, r, data) (the r_first handle), so
    # owners compute contiguous ranges of global replicate IDs and the root
    # assembles the distribution in order, then sorts it and computes the point
    # estimate, interval and standard error exactly as one device does.
    return _resample_parallel('bootstrap', devices, dict(kwargs, data=data))


def permutation_test(x, y, *, devices=(0,), **kwargs):
    """:func:`mojolearn.resample.permutation_test` with the permutations spread across GPUs.

    Parameters
    ----------
    x, y : array-like
        The two samples.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    **kwargs
        Every other argument of :func:`mojolearn.resample.permutation_test`.

    Returns
    -------
    PermutationTestResult
        Equal to ``mojolearn.resample.permutation_test(x, y, **kwargs)``.
    """
    # `mojolearn.resample.permutation_test` with global permutation ranges.
    return _resample_parallel('permutation_test', devices, dict(kwargs, x=x, y=y))


def monte_carlo_integrate(integrand, lower, upper, n_samples, *, devices=(0,), **kwargs):
    """:func:`mojolearn.resample.monte_carlo_integrate` with the samples spread across GPUs.

    Parameters
    ----------
    integrand, lower, upper, n_samples
        As in :func:`mojolearn.resample.monte_carlo_integrate`.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    **kwargs
        Every other argument of :func:`mojolearn.resample.monte_carlo_integrate`.

    Returns
    -------
    MonteCarloResult
        Equal to the one-GPU result.
    """
    # `mojolearn.resample.monte_carlo_integrate` with whole sample chunks.
    #
    # Owners draw global sample IDs from aligned PINNED_SUM_W chunks; the root
    # folds the chunk partials in order and forms the integral as one device does.
    return _resample_parallel('monte_carlo_integrate', devices,
                              dict(kwargs, integrand=integrand, lower=lower, upper=upper, n_samples=n_samples))


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
    # HDBSCAN with distributed core-distance k-NN rows and dense distance rows.
    #
    # The k-NN query rows use the neighbors row driver and the m x m pairwise
    # distance matrix uses the hierarchy row driver, each with its original
    # per-row arithmetic and byte gather. The mutual reachability cells, the
    # Boruvka MST, the dendrogram, the condensed hierarchy and cluster selection
    # stay on the root in their original order.
    from .hdbscan import HDBSCAN
    if type(estimator) is not HDBSCAN:
        raise TypeError('fit_hdbscan requires mojolearn.HDBSCAN')
    if getattr(estimator, 'numeric_mode', None) not in (None, 'identical'):
        raise ValueError('parallel HDBSCAN requires IDENTICAL numeric mode')
    X, _ = as_f32_c(X, ndim=2, name='X')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('hdbscan_fit', estimator, (X,))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


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
    return _chol._NATIVE(estimator, 'family_chol_par_fit', (A, devices), _chol._NATIVE_CONTEXT)


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
    return _chol._NATIVE(estimator, 'family_chol_par_solve', (B, devices), _chol._NATIVE_CONTEXT)


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
    return _km._NATIVE(estimator, 'family_km_par_fit', (X, y, devices), _km._NATIVE_CONTEXT)


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
    return public_method_result(estimator, _APPLY_METHOD.get(type(estimator)),
                                _km._NATIVE(estimator, 'family_km_par_apply', (X, devices), _km._NATIVE_CONTEXT),
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
    # RBFSampler.transform over whole query rows on separate GPUs.
    #
    # The fitted random weights and offsets are position-mapped draws with
    # global feature and component IDs, so every worker receives the same
    # fitted state. Each output row is a pure function of its input row
    # (identical GEMM cell contract and a per-cell epilogue); rows are joined
    # by copying bytes in input order.
    from ._array import Array
    from .kernel_methods import RBFSampler
    if type(estimator) is not RBFSampler:
        raise TypeError('requires mojolearn.RBFSampler')
    if getattr(estimator, 'numeric_mode', None) not in (None, 'identical'):
        raise ValueError('parallel RBFSampler requires IDENTICAL numeric mode')
    if not hasattr(estimator, 'random_weights_'):
        raise ValueError('RBFSampler is not fitted')
    if type(rows_per_shard) is not int or rows_per_shard < 1:
        raise ValueError('rows_per_shard must be a positive int')
    X, _ = as_f32_c(X, ndim=2, name='X')
    if X.shape[1] != estimator.n_features_in_:
        raise ValueError('X has %d features, the fit had %d' % (X.shape[1], estimator.n_features_in_))
    rows = X.shape[0]
    if rows == 0:
        return public_method_result(estimator, 'transform', estimator.transform(X), sys._getframe(1))
    shards = [X[start - driver_read_shift(index, start, devices):
                min(start + rows_per_shard, rows) - driver_read_shift(index, start, devices)]
              for index, start in enumerate(range(0, rows, rows_per_shard))]  # glue: one row slice per device shard
    pool = DevicePool(devices)
    try:
        parts = pool.map([('rbf_sampler_rows', estimator, (shard,)) for shard in shards])  # glue: one request per device shard
    finally:
        pool.close()
    q = estimator.random_weights_.shape[1]
    out = Array((rows, q), '<f4')
    start = 0
    for part in parts:  # glue: one block copy per device shard
        part, _ = as_f32_c(part, ndim=2, name="worker result")
        if part.shape[1] != q or start + part.shape[0] > rows:
            raise ValueError("RBFSampler worker returned an invalid result shape")
        out._mv[start * q:(start + part.shape[0]) * q] = part._mv
        start += part.shape[0]
    if start != rows:
        raise ValueError("RBFSampler workers returned an incomplete result")
    # the joined raw rows; the plain transform's public conversion for our caller
    return public_method_result(estimator, 'transform', out, sys._getframe(1))
