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

from . import _native_family_bridge, _native_parallel_public
from . import randomforest as _rf, extratrees as _et
from . import _native_parallel_shard as _shard
from ._native_parallel_shard import _native_parallel_shard

# glue: fixed forest types and compiled entries, connected once at import.
_FORESTS = (_rf.RandomForestClassifier, _rf.RandomForestRegressor,
            _et.ExtraTreesClassifier, _et.ExtraTreesRegressor)
_shard.register(forest_classes=_FORESTS,
                forest_regressors=(_rf.RandomForestRegressor, _et.ExtraTreesRegressor),
                forest_by_name={cls.__name__: cls for cls in _FORESTS},  # glue: four fixed names
                trees_entry=_rf._TREES, trees_context=_rf._TREES_CONTEXT)
_FAMILY, _FAMILY_CONTEXT = _native_family_bridge.connect()
_native_parallel_public.connect(_FAMILY_CONTEXT)


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
    return _FAMILY(estimator, 'family_par_public', ('fit_boosting', (X, y, devices, sample_weight, eval_set)), _FAMILY_CONTEXT)


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
    return _native_parallel_shard('fit_forest', (estimator, X, y, devices, trees_per_shard))


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
    return _FAMILY(estimator, 'family_par_public', ('fit_isolation_forest', (X, devices, sample_weight)), _FAMILY_CONTEXT)


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
    return _FAMILY(estimator, 'family_par_public', ('score_isolation_forest', (X, devices, method)), _FAMILY_CONTEXT)


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
    return _FAMILY(estimator, 'family_par_public', ('fit_ordered_rmse', (X, y, permutation, devices, sample_weight)), _FAMILY_CONTEXT)


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
    return _FAMILY(estimator, 'family_par_public', ('fit_feature_freq', (X, y, devices, sample_weight)), _FAMILY_CONTEXT)




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
        return _FAMILY(self, 'family_par_public', ('predictor_init', (estimator, devices)), _FAMILY_CONTEXT)

    def predict(self, X):
        """Predict labels (classifiers) or values (regressors), as ``estimator.predict(X)``.
        """
        return _FAMILY(self, 'family_par_public', ('predictor_predict', (X,)), _FAMILY_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities, as ``estimator.predict_proba(X)``; classifiers only.
        """
        return _FAMILY(self, 'family_par_public', ('predictor_predict_proba', (X,)), _FAMILY_CONTEXT)

    def close(self):
        """Release the predictor's GPU state."""
        return _FAMILY(self, 'family_par_public', ('predictor_close', ()), _FAMILY_CONTEXT)

    def __enter__(self):
        return _FAMILY(self, 'family_par_public', ('predictor_enter', ()), _FAMILY_CONTEXT)

    def __exit__(self, *exc):
        return _FAMILY(self, 'family_par_public', ('predictor_exit', ()), _FAMILY_CONTEXT)


_native_parallel_public.register_predictor(ParallelForestPredictor, _FAMILY_CONTEXT)
