# SPDX-License-Identifier: Apache-2.0
"""Fit tree ensembles across several GPUs.

Forests split their trees into ranges, one range per GPU; boosting spreads
the feature histograms of every round across the GPUs. The fitted models are
the same, bit for bit, as a one-GPU ``fit``.

Example
-------
>>> import numpy as np
>>> from mojolearn import RandomForestClassifier
>>> from mojolearn.parallel_ensemble import fit_forest
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((10000, 16)).astype(np.float32)
>>> y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
>>> forest = fit_forest(RandomForestClassifier(n_estimators=64, random_state=7), X, y,
...                     devices=(0, 1), trees_per_shard=16)
>>> proba = forest.predict_proba(X[:10])
"""
# Whole-tree data replication: global RNG IDs and canonical prediction order.
__all__ = [
    'fit_boosting',
    'fit_forest',
    'fit_isolation_forest',
    'score_isolation_forest',
    'fit_ordered_rmse',
    'fit_feature_freq',
    'ParallelForestPredictor',
]

import sys

from ._parallel_pool import public_method_result
from ._buffer import _native
from . import _backend, _native_family_bridge
from . import ensemble as _ens
from . import _gbdt_adapters as _gba
from . import _iforest_impl as _ifo
from . import randomforest as _rf
from . import extratrees as _et
from . import _native_parallel_shard as _shard
from ._native_parallel_shard import _native_parallel_shard

# The forests, for the Python-sharded drivers' native owner
# (bindings/parallel_shard_native.mojo): a tree range's fit is the plain
# native fit in a worker process.
_FORESTS = (_rf.RandomForestClassifier, _rf.RandomForestRegressor, _et.ExtraTreesClassifier,
            _et.ExtraTreesRegressor)
_shard.register(forest_classes=_FORESTS,
                forest_regressors=(_rf.RandomForestRegressor, _et.ExtraTreesRegressor),
                forest_by_name={cls.__name__: cls for cls in _FORESTS},  # glue: four fixed class names
                trees_entry=_rf._TREES, trees_context=_rf._TREES_CONTEXT)

# The classical families' compiled owner: a cooperative driver is one
# `family_par_run` call (bindings/family_native.mojo).
_FAMILY, _FAMILY_CONTEXT = _native_family_bridge.connect()

#: fit_boosting's plain owners: (operation, owner arity over
#: (X, y, sample_weight, eval_set, group_id, subgroup_id, pairs, pairs_weight),
#: the plain method's entry and context).
_BOOSTING_FIT = {
    _ens.GradientBoosting: ('trees_gbdt_fit', 8, _ens._TREES, _ens._TREES_CONTEXT),
    _gba.GradientBoostingClassifier: ('trees_gbdt_adapter_classifier_fit', 4, _gba._TREES, _gba._TREES_CONTEXT),
    _gba.GradientBoostingRegressor: ('trees_gbdt_adapter_regressor_fit', 4, _gba._TREES, _gba._TREES_CONTEXT),
}

#: score_isolation_forest's plain owners.
_IFOREST_SCORE = {'score_samples': 'trees_iforest_score_samples',
                  'decision_function': 'trees_iforest_decision_function', 'predict': 'trees_iforest_predict'}


def fit_boosting(estimator, X, y, *, devices=(0,), sample_weight=None, eval_set=None):
    """Fit a gradient-boosting model with its feature histograms spread across GPUs.

    Parameters
    ----------
    estimator : GradientBoosting, GradientBoostingClassifier or GradientBoostingRegressor
        An unfitted estimator using the ``'identical'`` numeric mode (or the default).
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Targets.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    sample_weight : array-like of shape (n_samples,), optional
        Per-row weights, as in ``estimator.fit``.
    eval_set : optional
        Validation data, as in ``estimator.fit``.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(X, y)``.
    """
    # One native call (bindings/family_native.mojo `family_par_run`, driver
    # `gbdt_fit`): the plain fit with every round's feature histograms on the
    # selected devices; trees, splits and leaf values are the one-device fit's.
    if type(estimator) not in _BOOSTING_FIT:
        raise TypeError('fit_boosting currently requires GradientBoosting, GradientBoostingClassifier or '
                        'GradientBoostingRegressor')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel boosting requires IDENTICAL numeric mode')
    operation, arity, entry, context = _BOOSTING_FIT[type(estimator)]
    return _FAMILY(estimator, 'family_par_run',
                   ('gbdt_fit', devices, operation, (X, y, sample_weight, eval_set, None, None, None, None)[:arity],
                    entry, context, _backend.plan_bindings(('_mojolearn_gbdt',))), _FAMILY_CONTEXT)


def fit_forest(estimator, X, y, *, devices=(0,), trees_per_shard=1):
    """Fit a random forest or Extra Trees model with tree ranges on separate GPUs.

    Every GPU reads the whole training set and grows a contiguous range of the
    forest's trees from the forest's own seed, so the trees and predictions do
    not depend on the number of GPUs.

    Parameters
    ----------
    estimator : RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier or ExtraTreesRegressor
        An unfitted estimator using the ``'identical'`` numeric mode (or the default).
    X : array-like of shape (n_samples, n_features)
        Training data, converted to float32.
    y : array-like of shape (n_samples,)
        Class labels or regression targets.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    trees_per_shard : int, default=1
        Number of trees in each range.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(X, y)``.
    """
    # One native call (bindings/parallel_shard_native.mojo `fit_forest`):
    # whole global tree-ID ranges of `trees_per_shard` trees, each a fresh
    # forest of the estimator's parameters fitted by the plain native fit
    # from its first global tree ID in a worker process per device (range i on
    # device i % len(devices)); the trees joined in ID order and their node
    # offsets rebased natively. A tree is a pure function of its global ID,
    # so the joined forest is the one-device forest.
    if type(trees_per_shard) is not int or trees_per_shard < 1:
        raise ValueError('trees_per_shard must be a positive integer')
    return _native_parallel_shard('fit_forest', (estimator, X, y, devices, trees_per_shard,
                                                 _native('msel_rebase_offsets_i32')))


def fit_isolation_forest(estimator, X, *, devices=(0,), sample_weight=None):
    """Fit an ``IsolationForest`` with its trees built on several GPUs.

    Parameters
    ----------
    estimator : IsolationForest
        An unfitted estimator using the ``'identical'`` numeric mode (or the default).
    X : array-like of shape (n_samples, n_features)
        Training data.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    sample_weight : None
        Not supported; passing weights raises ``NotImplementedError``.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(X)``.
    """
    # One native call (`family_par_run`, driver `iforest_fit`): the plain fit
    # with whole tree shards on the selected devices.
    if type(estimator) is not _ifo.IsolationForest:
        raise TypeError('requires mojolearn.IsolationForest')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel IsolationForest requires IDENTICAL')
    if sample_weight is not None:
        raise NotImplementedError('IsolationForest does not support sample_weight')
    return _FAMILY(estimator, 'family_par_run',
                   ('iforest_fit', devices, 'trees_iforest_fit', (X, None), _ifo._TREES, _ifo._TREES_CONTEXT,
                    _backend.plan_bindings(('_mojolearn_svm',))), _FAMILY_CONTEXT)


def score_isolation_forest(estimator, X, *, devices=(0,), method='score_samples'):
    """Score rows with a fitted ``IsolationForest``, its trees evaluated on several GPUs.

    Parameters
    ----------
    estimator : IsolationForest
        An estimator fitted in the ``'identical'`` numeric mode.
    X : array-like of shape (n_samples, n_features)
        Rows to score.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    method : {'score_samples', 'decision_function', 'predict'}, default='score_samples'
        The estimator method to run.

    Returns
    -------
    The same value ``estimator.<method>(X)`` returns.
    """
    # One native call (`family_par_run`, driver `iforest_score`): the plain
    # scoring on the selected devices; the plain method's public conversion.
    if type(estimator) is not _ifo.IsolationForest:
        raise TypeError('requires mojolearn.IsolationForest')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel IsolationForest requires IDENTICAL')
    if method not in _IFOREST_SCORE:
        raise ValueError('method must be score_samples, decision_function or predict')
    return public_method_result(estimator, method,
                                _FAMILY(estimator, 'family_par_run',
                                        ('iforest_score', devices, _IFOREST_SCORE[method], (X,), _ifo._TREES,
                                         _ifo._TREES_CONTEXT, _backend.plan_bindings(('_mojolearn_svm',))),
                                        _FAMILY_CONTEXT),
                                sys._getframe(1))


def fit_ordered_rmse(estimator, X, y, *, permutation, devices=(0,), sample_weight=None):
    """Fit an ``OrderedRMSE`` model with its feature histograms spread across GPUs.

    Parameters
    ----------
    estimator : OrderedRMSE
        An unfitted estimator using the ``'identical'`` numeric mode (or the default).
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Regression targets.
    permutation : array-like of shape (n_samples,)
        The row order of ordered boosting, as in ``estimator.fit``.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    sample_weight : array-like of shape (n_samples,), optional
        Per-row weights.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU
        ``estimator.fit(X, y, permutation=permutation)``.
    """
    # One native call (`family_par_run`, driver `ordered_rmse_fit`): the plain
    # Ordered fit with its pointwise fold histograms on the selected devices.
    if type(estimator) is not _ens.OrderedRMSE:
        raise TypeError('requires mojolearn.OrderedRMSE')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel OrderedRMSE requires IDENTICAL numeric mode')
    return _FAMILY(estimator, 'family_par_run',
                   ('ordered_rmse_fit', devices, 'trees_gbdt_ordered_fit', (X, y, permutation, sample_weight),
                    _ens._TREES, _ens._TREES_CONTEXT, _backend.plan_bindings(('_mojolearn_gbdt',))), _FAMILY_CONTEXT)


def fit_feature_freq(estimator, X, y, *, devices=(0,), sample_weight=None):
    """Fit an ``ExperimentalTwoLevelFeatureFreq`` model with its histograms spread across GPUs.

    Parameters
    ----------
    estimator : ExperimentalTwoLevelFeatureFreq
        An unfitted estimator using the ``'identical'`` numeric mode (or the default).
    X : array-like of shape (n_samples, n_features)
        Training data with the categorical source columns.
    y : array-like of shape (n_samples,)
        Targets.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    sample_weight : array-like of shape (n_samples,), optional
        Per-row weights.

    Returns
    -------
    estimator
        The same estimator, fitted, equal to a one-GPU ``estimator.fit(X, y)``.
    """
    # One native call (`family_par_run`, driver `gbdt_fit`): the plain fit with
    # its feature histograms on the selected devices.
    if type(estimator) is not _ens.ExperimentalTwoLevelFeatureFreq:
        raise TypeError('requires mojolearn.ExperimentalTwoLevelFeatureFreq')
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel FeatureFreq requires IDENTICAL numeric mode')
    return _FAMILY(estimator, 'family_par_run',
                   ('gbdt_fit', devices, 'trees_gbdt_feature_freq_fit', (X, y, sample_weight),
                    _ens._TREES, _ens._TREES_CONTEXT, _backend.plan_bindings(('_mojolearn_gbdt',))), _FAMILY_CONTEXT)


def _admit_forest_predictor(estimator):
    from .randomforest import RandomForestClassifier, RandomForestRegressor
    from .extratrees import ExtraTreesClassifier, ExtraTreesRegressor
    if type(estimator) not in (RandomForestClassifier, RandomForestRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor):
        raise TypeError('ParallelForestPredictor requires a mojolearn RandomForest or ExtraTrees estimator')
    if not all(hasattr(estimator, name) for name in (  # glue: checks the fitted attribute names
            '_offsets', '_colid', '_quesval', '_left_child', '_leaves',
            '_n_trees', '_num_outputs', 'n_features_in_')):
        raise RuntimeError('ParallelForestPredictor requires a fitted estimator')
    if estimator._effective_mode() != 'identical':
        raise ValueError('ParallelForestPredictor requires IDENTICAL numeric mode')
    if estimator._prediction_engine() != 'parallel_groves':
        raise ValueError("ParallelForestPredictor requires inference_engine='parallel_groves'")
    return isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier))


class ParallelForestPredictor:
    """Predict with a fitted forest whose trees are held across several GPUs.

    The forest is copied when the predictor is created; later changes to the
    estimator do not affect it.

    Parameters
    ----------
    estimator : RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier or ExtraTreesRegressor
        A fitted estimator with ``inference_engine='parallel_groves'`` in the
        ``'identical'`` numeric mode.
    devices : tuple of int, default=(0, 1)
        Distinct GPU indices, at most 64.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_estimators_ : int
        Number of trees.
    classes_ : array
        Class labels (classifiers only).

    Use it as a context manager, or call :meth:`close` to release the GPUs.

    Example
    -------
    >>> forest = RandomForestClassifier(n_estimators=40, inference_engine="parallel_groves").fit(X, y)
    >>> with ParallelForestPredictor(forest, devices=(0, 1)) as pool:
    ...     labels = pool.predict(X)
    """
    # Own a fitted RF/ET snapshot whose logical groves reside across GPUs.
    #
    # The snapshot is a private copy of the estimator, taken here (no device
    # state is copied: the copy builds its own resident pool). Each predict is
    # one native call (bindings/family_native.mojo `family_par_run`, driver
    # `forest_predict`): the plain predict of the copy with the device plan
    # bound, so its first predict builds the parallel_groves pool on the
    # selected devices (core/forest_inference_pool.mojo) and later predicts
    # reuse it. Later source mutations or refits do not affect this predictor.
    # This pools inference model storage; fit_forest remains a separate API.
    # Use as a context manager, or call close to release GPU state.
    def __init__(self, estimator, *, devices=(0, 1)):
        import copy
        import threading
        self._classifier = _admit_forest_predictor(estimator)
        self._public_kind = type(estimator)
        self.devices = tuple(devices)
        if len(self.devices) > 64:
            raise ValueError('ParallelForestPredictor supports at most 64 devices')
        if (not self.devices or any(type(i) is not int or i < 0 for i in self.devices)  # glue: device index checks
                or len(set(self.devices)) != len(self.devices)):
            raise ValueError('devices must be distinct nonnegative integer indices')
        self._model = copy.copy(estimator)  # glue: the predictor's own snapshot (device state is not copied)
        self._lock = threading.RLock()
        self._closed = False
        self.n_features_in_ = int(estimator.n_features_in_)
        self.n_estimators_ = int(estimator._n_trees)
        if self._classifier:
            self.classes_ = estimator.classes_
            self.n_classes_ = int(estimator._num_outputs)

    def predict(self, X):
        """Predict labels (classifiers) or values (regressors), as ``estimator.predict(X)``.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError('pooled forest predictor is closed')
            return public_method_result(self._public_kind, 'predict',
                                        _FAMILY(self._model, 'family_par_run',
                                                ('forest_predict', self.devices, 'trees_public_forest_predict',
                                                 (X,), _rf._TREES, _rf._TREES_CONTEXT,
                                                 _backend.plan_bindings(('_mojolearn_rf', '_mojolearn_trees'))),
                                                _FAMILY_CONTEXT),
                                        sys._getframe(1))

    def predict_proba(self, X):
        """Class probabilities, as ``estimator.predict_proba(X)``; classifiers only.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError('pooled forest predictor is closed')
            if not self._classifier:
                raise ValueError('predict_proba requires a classifier')
            return public_method_result(self._public_kind, 'predict_proba',
                                        _FAMILY(self._model, 'family_par_run',
                                                ('forest_predict', self.devices, 'trees_public_forest_proba',
                                                 (X,), _rf._TREES, _rf._TREES_CONTEXT,
                                                 _backend.plan_bindings(('_mojolearn_rf', '_mojolearn_trees'))),
                                                _FAMILY_CONTEXT),
                                        sys._getframe(1))

    def close(self):
        """Release the predictor's GPU state."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            from ._forest_protocol import held_resident_forest
            resident = held_resident_forest(self._model)
            if resident is not None:
                resident._finalizer()
            self._model = None

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('pooled forest predictor is closed')
            return self

    def __exit__(self, *exc):
        self.close()
