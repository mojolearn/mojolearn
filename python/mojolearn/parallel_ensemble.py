# SPDX-License-Identifier: Apache-2.0
"""Whole-tree data replication: global RNG IDs and canonical prediction order."""
from ._parallel_pool import DevicePool, driver_read_shift
from ._array import Array
from ._buffer import as_f32_c, empty, addr, addr_ro
from ._bufcheck import memcopy


def fit_boosting(estimator, X, y, *, devices=(0,), sample_weight=None, eval_set=None):
    """Partition packed feature groups during greedy or pointwise histograms.

    Boosting rounds, row reductions, quantization scales, split selection and
    leaf estimation retain their original global order. Only publish a full
    successful fitted model. Root state still has to fit on the first GPU.
    """
    from .ensemble import GradientBoosting, GradientBoostingClassifier, GradientBoostingRegressor
    if type(estimator) not in (GradientBoosting, GradientBoostingClassifier, GradientBoostingRegressor):
        raise TypeError('fit_boosting currently requires GradientBoosting, GradientBoostingClassifier or GradientBoostingRegressor')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel boosting requires IDENTICAL numeric mode')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('gbdt_fit', estimator,
            (X, y, dict(sample_weight=sample_weight, eval_set=eval_set)))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def fit_forest(estimator, X, y, *, devices=(0,), trees_per_shard=1):
    """Fit one RandomForest or ExtraTrees model across GPUs using global tree IDs.

    Every shard reads the FULL dataset and original forest seed. Tree ranges
    are concatenated in ID order; floating-point predictions use the existing
    complete-forest predictor. GPU count never selects the reduction order.
    Publishes the fitted estimator only after every worker succeeds.
    """
    from .randomforest import RandomForestClassifier, RandomForestRegressor
    from .extratrees import ExtraTreesClassifier, ExtraTreesRegressor
    if type(estimator) not in (RandomForestClassifier, RandomForestRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor):
        raise TypeError('fit_forest requires a mojolearn RandomForest or ExtraTrees estimator')
    if type(trees_per_shard) is not int or trees_per_shard < 1:
        raise ValueError('trees_per_shard must be a positive integer')
    params = estimator.get_params()
    if params['numeric_mode'] not in (None, 'identical'):
        raise ValueError('parallel forests require IDENTICAL numeric mode')
    params['numeric_mode'] = 'identical'
    X, _ = as_f32_c(X, ndim=2, name='X')
    # Keep classification label encoding identical on every full-data shard.
    if isinstance(estimator, (RandomForestRegressor, ExtraTreesRegressor)):
        y, _ = as_f32_c(y, ndim=1, name='y')
    else:
        from ._labels import flatten_labels
        y = flatten_labels(y)
    count = params['n_estimators']
    if count < 1:
        raise ValueError('n_estimators must be positive')
    requests = []
    for index, start in enumerate(range(0, count, trees_per_shard)):
        shard_params = dict(params, n_estimators=min(trees_per_shard, count - start))
        # `driver_read_shift` is 0 unless the driver sabotage switch is on, and
        # 0 for the first shard whatever the switch says; the shard COUNT is
        # untouched, so only the global tree IDs each shard seeds from move.
        requests.append(('forest_fit', (type(estimator).__name__, shard_params),
                         (X, y, start - driver_read_shift(index, start, devices))))
    pool = DevicePool(devices)
    try:
        parts = pool.map(requests)
    finally:
        pool.close()
    result = type(estimator)(**params)
    result.__dict__.update(parts[0].__dict__)
    offsets = [0]
    for part in parts:
        base = offsets[-1]
        offsets.extend(base + offset for offset in part._offsets.tolist()[1:])
    result._offsets = Array.from_list(offsets, parts[0]._offsets.dtype)
    for name in ('_colid', '_quesval', '_left_child', '_leaves'):
        arrays = [getattr(part, name) for part in parts]
        merged = empty((sum(a.size for a in arrays),), arrays[0].dtype)
        cursor = 0
        for array in arrays:
            memcopy(addr(merged, name=name) + cursor, addr_ro(array, name=name), array.nbytes)
            cursor += array.nbytes
        setattr(result, name, merged)
    # Restore the full constructor/config; worker model metadata is otherwise
    # identical (feature count, class order, criterion and output count).
    result._n_trees = count
    result.n_estimators = count
    result._cfg['n_estimators'] = count
    if hasattr(result, 'depth_cap_bound_'):
        result.depth_cap_bound_ = any(part.depth_cap_bound_ for part in parts)
    result._resident_forest = None
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def fit_isolation_forest(estimator, X, *, devices=(0,), sample_weight=None):
    """Build tree ranges with original global seeds; retain full-data scoring.

    Each GPU receives the full training matrix. Tree scratch is partitioned,
    but the assembled model still lives on the root. Use score_isolation_forest
    for distributed rebuilding during prediction (the current estimator refits
    its forest on every score call).
    """
    from ._iforest_impl import IsolationForest
    if type(estimator) is not IsolationForest:
        raise TypeError('requires mojolearn.IsolationForest')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel IsolationForest requires IDENTICAL')
    if sample_weight is not None:
        raise NotImplementedError('IsolationForest does not support sample_weight')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('iforest_fit', estimator, (X,))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def score_isolation_forest(estimator, X, *, devices=(0,), method='score_samples'):
    """Rebuild trees concurrently and use the original ordered score/threshold."""
    from ._iforest_impl import IsolationForest
    if type(estimator) is not IsolationForest:
        raise TypeError('requires mojolearn.IsolationForest')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel IsolationForest requires IDENTICAL')
    if method not in ('score_samples', 'decision_function', 'predict'):
        raise ValueError('method must be score_samples, decision_function or predict')
    pool = DevicePool(devices, cooperative=True)
    try:
        return pool.map([('iforest_score', estimator, (X, method))])[0]
    finally:
        pool.close()


def fit_ordered_rmse(estimator, X, y, *, permutation, devices=(0,), sample_weight=None):
    """Partition pointwise feature groups; retain the complete ordered folds.

    The original permutation, growing-prefix approximations and leaf updates
    stay on the root. Full training and histogram state still fit one GPU.
    """
    from .ensemble import OrderedRMSE
    if type(estimator) is not OrderedRMSE:
        raise TypeError('requires mojolearn.OrderedRMSE')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel OrderedRMSE requires IDENTICAL numeric mode')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('ordered_rmse_fit', estimator,
            (X, y, dict(permutation=permutation, sample_weight=sample_weight)))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def fit_feature_freq(estimator, X, y, *, devices=(0,), sample_weight=None):
    """Distribute both levels' greedy histograms after original CTR generation.

    Candidate generation, source IDs, level transitions and leaf estimates
    retain their original order. Root candidate/data/model state is replicated.
    """
    from .ensemble import ExperimentalTwoLevelFeatureFreq
    if type(estimator) is not ExperimentalTwoLevelFeatureFreq:
        raise TypeError('requires mojolearn.ExperimentalTwoLevelFeatureFreq')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel FeatureFreq requires IDENTICAL numeric mode')
    pool = DevicePool(devices, cooperative=True)
    try:
        result = pool.map([('gbdt_fit', estimator,
            (X, y, dict(sample_weight=sample_weight)))])[0]
    finally:
        pool.close()
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def _admit_forest_predictor(estimator):
    from .randomforest import RandomForestClassifier, RandomForestRegressor
    from .extratrees import ExtraTreesClassifier, ExtraTreesRegressor
    if type(estimator) not in (RandomForestClassifier, RandomForestRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor):
        raise TypeError('ParallelForestPredictor requires a mojolearn RandomForest or ExtraTrees estimator')
    if not all(hasattr(estimator, name) for name in (
            '_offsets', '_colid', '_quesval', '_left_child', '_leaves',
            '_n_trees', '_num_outputs', 'n_features_in_')):
        raise RuntimeError('ParallelForestPredictor requires a fitted estimator')
    if estimator._effective_mode() != 'identical':
        raise ValueError('ParallelForestPredictor requires IDENTICAL numeric mode')
    if estimator._prediction_engine() != 'parallel_groves':
        raise ValueError("ParallelForestPredictor requires inference_engine='parallel_groves'")
    return isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier))


class ParallelForestPredictor:
    """Own a fitted RF/ET snapshot whose logical groves reside across GPUs.

    Preparation snapshots the estimator immediately. Later source mutations
    or refits do not affect this predictor. Predictions use the existing
    parallel_groves reduction and original estimator label/dtype conversion.
    This pools inference model storage; fit_forest remains a separate API.
    Use as a context manager, or call close to release GPU state and worker.
    """
    def __init__(self, estimator, *, devices=(0, 1)):
        import threading
        self._classifier = _admit_forest_predictor(estimator)
        self.devices = tuple(devices)
        if len(self.devices) > 64:
            raise ValueError('ParallelForestPredictor supports at most 64 devices')
        self._pool = DevicePool(self.devices, cooperative=True)
        self._lock = threading.RLock()
        self._closed = False
        try:
            metadata = self._pool.map([('forest_prepare', estimator, None)])[0]
            self.n_features_in_ = metadata['n_features']
            self.n_estimators_ = metadata['trees']
            if self._classifier:
                self.classes_ = metadata['classes']
                self.n_classes_ = metadata['outputs']
        except BaseException:
            self._closed = True
            self._pool.close()
            raise

    def _predict(self, X, method):
        with self._lock:
            if self._closed:
                raise RuntimeError('pooled forest predictor is closed')
            if method not in ('predict', 'predict_proba'):
                raise ValueError('unsupported pooled forest prediction method')
            if method == 'predict_proba' and not self._classifier:
                raise ValueError('predict_proba requires a classifier')
            X, _ = as_f32_c(X, ndim=2, name='X')
            if X.shape[1] != self.n_features_in_:
                raise ValueError('X has %d features, fit saw %d' %
                                 (X.shape[1], self.n_features_in_))
            try:
                return self._pool.map([('forest_predict', None, (method, X))])[0]
            except BaseException:
                # DevicePool closes on a failed RPC. Never restart a fresh
                # worker that has lost this predictor's prepared snapshot.
                self._closed = True
                self._pool.close()
                raise

    def predict(self, X):
        """Predict using the snapshot's original label decoding/output dtype."""
        return self._predict(X, 'predict')

    def predict_proba(self, X):
        """Classifier probabilities: RF float32, ExtraTrees float64."""
        return self._predict(X, 'predict_proba')

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._pool.map([('forest_release', None, None)])
            finally:
                self._pool.close()

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('pooled forest predictor is closed')
            return self

    def __exit__(self, *exc):
        self.close()
