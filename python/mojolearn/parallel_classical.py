# SPDX-License-Identifier: Apache-2.0
"""Cooperative classical estimators with explicit per-algorithm partitions."""
from ._parallel_pool import DevicePool, driver_read_shift
from ._buffer import as_f32_c


def fit_logistic(estimator, X, y, *, devices=(0,)):
    """Partition gradient feature columns; retain the original objective and QN trajectory.

    Supports the existing binary and multinomial objectives and their admitted
    penalties. Data and optimizer state still have to fit on the root GPU.
    """
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
    """Fit Holt-Winters series independently and assemble original component layouts."""
    from ._tsa_impl import ExponentialSmoothing
    from . import _backend
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
                  start_periods=estimator.start_periods, eps=estimator.eps)
    pool = DevicePool(devices)
    try:
        parts = pool.map([
            ('holtwinters_fit', params,
             (data[start - driver_read_shift(index, start, devices):
                   min(start + series_per_shard, batch) - driver_read_shift(index, start, devices)],))
            for index, start in enumerate(range(0, batch, series_per_shard))])
    finally:
        pool.close()
    result = ExponentialSmoothing(estimator.endog, ts_num=batch, **params)
    result.__dict__.update(parts[0].__dict__)
    result.endog, result._data, result.ts_num, result.n = estimator.endog, data, batch, n
    for name in ('level_', 'trend_', 'season_', 'sse_', 'alpha_', 'beta_', 'gamma_', 'n_iter_', 'criterion_'):
        arrays = [getattr(part, name) for part in parts]
        merged = empty((batch,) + arrays[0].shape[1:], arrays[0].dtype)
        cursor = 0
        for array in arrays:
            memcopy(addr(merged, name=name) + cursor, addr_ro(array, name=name), array.nbytes)
            cursor += array.nbytes
        setattr(result, name, merged)
    steps = n - estimator.seasonal_periods
    result._components_len = steps * batch
    result._comps = empty((3 * result._components_len,), '<f4')
    destination = addr(result._comps, name='components')
    start = 0
    for part in parts:
        source = addr_ro(part._comps, name='component shard')
        for component in range(3):
            for row in range(steps):
                memcopy(destination + (component * steps * batch + row * batch + start) * 4,
                        source + (component * steps * part.ts_num + row * part.ts_num) * 4,
                        part.ts_num * 4)
        start += part.ts_num
    cl = result._components_len
    result._time_major = {name: result._comps[i * cl:(i + 1) * cl].reshape((steps, batch))
                          for i, name in enumerate(('level', 'trend', 'season'))}
    result.fit_executed_flag = True
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def fit_gram_estimator(estimator, X, y=None, *, devices=(0,), sample_weight=None):
    """Distribute the existing pinned Gram chunks for OLS, Ridge, PCA and SVD.

    This entry admits the original Gram/eigendecomposition and wide OLS paths.
    The root still owns preprocessing and solver state; this is computation
    partitioning, not a fit beyond one GPU's memory capacity.
    """
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
    """Fit independent series on GPU workers, retaining each solver's order.

    A worker receives only its series, so fitting does not require the entire
    batch on one GPU. A single series must still fit on one GPU. The returned
    estimator's existing prediction methods are unchanged and use one GPU.
    """
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
    params = dict(order=estimator.order, seasonal_order=estimator.seasonal_order,
                  trend='c' if estimator.k_ else 'n', method=estimator.method,
                  maxiter=estimator.maxiter, numeric_mode='identical')
    requests = [('arima_fit', params,
                 (data[start - driver_read_shift(index, start, devices):
                       min(start + series_per_shard, batch) - driver_read_shift(index, start, devices)],))
                for index, start in enumerate(range(0, batch, series_per_shard))]
    pool = DevicePool(devices)
    try:
        parts = pool.map(requests)
    finally:
        pool.close()
    result = ARIMA(**params)
    result.__dict__.update(parts[0].__dict__)
    for name in ('params_', 'x_', 'x0_', 'n_iter_', 'retcode_', 'llf_', 'fx_', 'aic_', 'bic_'):
        arrays = [getattr(part, name) for part in parts]
        merged = empty((batch,) + arrays[0].shape[1:], arrays[0].dtype)
        cursor = 0
        for array in arrays:
            memcopy(addr(merged, name=name) + cursor, addr_ro(array, name=name), array.nbytes)
            cursor += array.nbytes
        setattr(result, name, merged)
    result.batch_size_, result.n_obs_ = batch, observations
    result._y, result.input_copied_ = data, copied
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def fit_kmeans(estimator, X, *, devices=(0,), sample_weight=None):
    """Distribute KMeans row assignments; retain full-data update/reduction order.

    Initialization, fixed-point centroid sums, restart choice and convergence
    use the existing implementation. Whole aligned row tiles compute distances
    concurrently. This first path stages buffers per assignment call.
    """
    from .cluster import KMeans
    if type(estimator) is not KMeans:
        raise TypeError('fit_kmeans requires mojolearn.KMeans')
    params = {name: getattr(estimator, name) for name in (
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
    """Distribute original dot leaves; preserve serial coordinate and convergence order.

    Single-leaf dots remain on the root. Full data and residual state must fit
    on the root; transport per coordinate can outweigh computation savings.
    """
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
    """Distribute linear/RBF kernel rows, retaining the original SVC/SVR solver.

    Full data, working-set state and output kernel tile remain on the root.
    A one-row kernel operation has only one active device.
    """
    from ._svm_impl import SVC, SVR
    if type(estimator) not in (SVC, SVR):
        raise TypeError('requires mojolearn.SVC or SVR')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel SVM requires IDENTICAL')
    if sample_weight is not None:
        raise NotImplementedError('SVC/SVR do not implement sample_weight')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('svm_fit', estimator, (X, y))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def predict_svm(estimator, X, *, devices=(0,), method='predict'):
    """Distribute prediction kernel rows; retain the original support-vector fold."""
    from ._svm_impl import SVC, SVR
    if type(estimator) not in (SVC, SVR):
        raise TypeError('requires mojolearn.SVC or SVR')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel SVM requires IDENTICAL')
    if method not in ('predict', 'decision_function') or (type(estimator) is SVR and method != 'predict'):
        raise ValueError('method must be predict, or decision_function for SVC')
    pool = DevicePool(devices, cooperative=True)
    try:
        return pool.map([('svm_predict', estimator, (X, method))])[0]
    finally:
        pool.close()


def fit_gaussian_process(estimator, X, y, *, devices=(0,)):
    """Distribute covariance rows; retain the original Cholesky and solve order."""
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
    """Distribute cross-covariance rows; retain root prediction/variance folds."""
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
    return result


def fit_dbscan(estimator, X, *, devices=(0,), sample_weight=None):
    """Distribute neighborhood rows; retain root core-point and label merges.

    Brute L2/L1 and RBC L2 use their original distance/order contracts.
    Full reference data, the current batch graph and label state still need
    to fit on the root GPU. This does not pool the reference index.
    """
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
    """Row-shard every E-step; retain the root M-step, Cholesky and convergence test.

    Whole sample rows run the original E-step on their owners and are copied
    back into their original positions; the mean log likelihood is folded on
    the root over the complete gathered rows. The KMeans initialization uses
    its row-tile assignment driver. Full data, responsibilities and the M-step
    still have to fit on the root GPU. covariance_type other than 'full' and
    the refused knobs are refused by the estimator itself, by name.
    """
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
    """Row-shard the scoring E-step of a fitted GaussianMixture.

    method is 'predict', 'predict_proba' or 'score_samples'; the original
    per-row argmax and host exponential are unchanged.
    """
    _admit_gaussian_mixture(estimator)
    if method not in ('score_samples', 'predict_proba', 'predict'):
        raise ValueError('method must be predict, predict_proba or score_samples')
    if not hasattr(estimator, 'weights_'):
        raise ValueError('GaussianMixture is not fitted')
    X, _ = as_f32_c(X, ndim=2, name='X')
    pool = DevicePool(devices, cooperative=True)
    try:
        return pool.map([('gmm_predict', estimator, (method, X))])[0]
    finally:
        pool.close()


def _resample_parallel(name, devices, kwargs):
    if kwargs.get('numeric_mode') not in (None, 'identical'):
        raise ValueError('parallel resampling requires IDENTICAL numeric mode')
    kwargs = {k: v for k, v in kwargs.items() if k != 'numeric_mode'}
    pool = DevicePool(devices, cooperative=True)
    try:
        return pool.map([('resample', None, (name, kwargs))])[0]
    finally:
        pool.close()


def bootstrap(data, *, devices=(0,), **kwargs):
    """`mojolearn.resample.bootstrap` with global replicate ranges on the GPUs.

    Replicate r is a pure function of (seed, r, data) (the r_first handle), so
    owners compute contiguous ranges of global replicate IDs and the root
    assembles the distribution in order, then sorts it and computes the point
    estimate, interval and standard error exactly as one device does.
    """
    return _resample_parallel('bootstrap', devices, dict(kwargs, data=data))


def permutation_test(x, y, *, devices=(0,), **kwargs):
    """`mojolearn.resample.permutation_test` with global permutation ranges."""
    return _resample_parallel('permutation_test', devices, dict(kwargs, x=x, y=y))


def monte_carlo_integrate(integrand, lower, upper, n_samples, *, devices=(0,), **kwargs):
    """`mojolearn.resample.monte_carlo_integrate` with whole sample chunks.

    Owners draw global sample IDs from aligned PINNED_SUM_W chunks; the root
    folds the chunk partials in order and forms the integral as one device does.
    """
    return _resample_parallel('monte_carlo_integrate', devices,
                              dict(kwargs, integrand=integrand, lower=lower, upper=upper, n_samples=n_samples))


def fit_hdbscan(estimator, X, *, devices=(0,)):
    """HDBSCAN with distributed core-distance k-NN rows and dense distance rows.

    The k-NN query rows use the neighbors row driver and the m x m pairwise
    distance matrix uses the hierarchy row driver, each with its original
    per-row arithmetic and byte gather. The mutual reachability cells, the
    Boruvka MST, the dendrogram, the condensed hierarchy and cluster selection
    stay on the root in their original order.
    """
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


def _admit_cholesky(estimator):
    from ._cholesky_impl import Cholesky
    if type(estimator) is not Cholesky:
        raise TypeError('requires mojolearn.Cholesky')
    if getattr(estimator, 'numeric_mode', None) not in (None, 'identical'):
        raise ValueError('parallel Cholesky requires IDENTICAL numeric mode')


def fit_cholesky(estimator, A, *, devices=(0,)):
    """Factor with whole trailing-update output rows on the selected GPUs.

    Each panel's factorization, info read-back, panel solve and subtraction
    stay on the root in the original panel order; only the rows of each
    panel's L21 L21^T product move. The full matrix stays on the root.
    """
    _admit_cholesky(estimator)
    A, _ = as_f32_c(A, ndim=2, name='A')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('cholesky_fit', estimator, (A,))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def solve_cholesky(estimator, B, *, devices=(0,)):
    """Solve with whole right-hand-side columns on the selected GPUs.

    A single right-hand side is sequential in its rows and runs on one GPU.
    """
    _admit_cholesky(estimator)
    if not hasattr(estimator, 'L_'):
        raise ValueError('Cholesky is not fitted')
    B, _ = as_f32_c(B, ndim=None, name='B')
    pool = DevicePool(devices, cooperative=True)
    try:
        return pool.map([('cholesky_solve', estimator, (B,))])[0]
    finally:
        pool.close()


def _admit_kernel_method(estimator):
    from .kernel_methods import KernelRidge, Nystroem
    if type(estimator) not in (KernelRidge, Nystroem):
        raise TypeError('requires mojolearn.KernelRidge or Nystroem')
    if getattr(estimator, 'numeric_mode', None) not in (None, 'identical'):
        raise ValueError('parallel kernel methods require IDENTICAL numeric mode')
    if estimator.kernel == 'laplacian':
        raise ValueError("parallel kernel methods refuse kernel='laplacian' by name: its L1 "
                         "distance matrix has no distributed row seam in this lane")


def fit_kernel_method(estimator, X, y=None, *, devices=(0,)):
    """KernelRidge or Nystroem fit with distributed kernel-matrix output rows.

    Rows use the SVM kernel-row seam (original FP32-v1 dot per cell and the
    original epilogue). KernelRidge's factorization moves whole trailing-update
    rows and its multi-target solve whole target columns; Nystroem's Jacobi
    eigensolver stays on the root. 'laplacian' is refused by name.
    """
    _admit_kernel_method(estimator)
    X, _ = as_f32_c(X, ndim=2, name='X')
    from .kernel_methods import KernelRidge
    if type(estimator) is KernelRidge:
        if y is None:
            raise ValueError('KernelRidge fit requires y')
        args = (X, y)
    else:
        args = (X,)
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('km_fit', estimator, args)])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def apply_kernel_method(estimator, X, *, devices=(0,)):
    """KernelRidge.predict or Nystroem.transform with distributed kernel rows."""
    _admit_kernel_method(estimator)
    from .kernel_methods import KernelRidge
    method = 'predict' if type(estimator) is KernelRidge else 'transform'
    X, _ = as_f32_c(X, ndim=2, name='X')
    pool = DevicePool(devices, cooperative=True)
    try:
        return pool.map([('km_apply', estimator, (method, X))])[0]
    finally:
        pool.close()


def transform_rbf_sampler(estimator, X, *, devices=(0,), rows_per_shard=4096):
    """RBFSampler.transform over whole query rows on separate GPUs.

    The fitted random weights and offsets are position-mapped draws with
    global feature and component IDs, so every worker receives the same
    fitted state. Each output row is a pure function of its input row
    (identical GEMM cell contract and a per-cell epilogue); rows are joined
    by copying bytes in input order.
    """
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
        return estimator.transform(X)
    shards = [X[start - driver_read_shift(index, start, devices):
                min(start + rows_per_shard, rows) - driver_read_shift(index, start, devices)]
              for index, start in enumerate(range(0, rows, rows_per_shard))]
    pool = DevicePool(devices)
    try:
        parts = pool.map([('rbf_sampler_rows', estimator, (shard,)) for shard in shards])
    finally:
        pool.close()
    q = estimator.random_weights_.shape[1]
    out = Array((rows, q), '<f4')
    start = 0
    for part in parts:
        part, _ = as_f32_c(part, ndim=2, name="worker result")
        if part.shape[1] != q or start + part.shape[0] > rows:
            raise ValueError("RBFSampler worker returned an invalid result shape")
        out._mv[start * q:(start + part.shape[0]) * q] = part._mv
        start += part.shape[0]
    if start != rows:
        raise ValueError("RBFSampler workers returned an incomplete result")
    return out
