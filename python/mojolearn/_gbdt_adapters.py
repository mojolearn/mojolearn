# SPDX-License-Identifier: Apache-2.0
"""Bounded sklearn adapters around the existing GPU GBDT learner.

Legacy GradientBoosting.predict remains raw prediction. These adapters support
binary Logloss classification and RMSE regression, numeric features, and pickle
state. Evaluation features must already have the same preprocessing as X;
sklearn Pipeline does not automatically transform a supplied eval_set.
"""
import inspect

import numpy as np

from . import _backend, _metrics_impl as metrics
from ._arrays import _addr, _addr_ro
from .ensemble import GradientBoosting


class _GBDTAdapter:
    _loss = None

    def __init__(self, *, n_estimators=100, max_depth=6, learning_rate=0.03,
                 l2_leaf_reg=3.0, border_count=128, random_state=0,
                 leaf_estimation_method=None, leaf_estimation_iterations=None,
                 bootstrap_type=None, bagging_temperature=1.0, subsample=None,
                 od_type=None, od_pvalue=None, od_wait=None, use_best_model=None,
                 best_model_min_trees=1, score_function=None, nan_mode='Min',
                 random_strength=0.0, use_pointwise_searcher=False,
                 boost_from_average=None, border_build_max_samples=200000,
                 class_weights=None, grow_policy='SymmetricTree', max_leaves=None,
                 min_data_in_leaf=1, min_split_gain=None, min_child_hessian=None,
                 numeric_mode=None):
        values = locals().copy()
        values.pop('self')
        for name, value in values.items():
            setattr(self, name, value)
        self._new_learner(numeric_mode)

    def _new_learner(self, mode):
        if mode is not None and (not isinstance(mode, str) or
                mode.strip().lower() not in ('fast', 'deterministic', 'identical')):
            raise ValueError('numeric_mode must be fast, deterministic, identical or None')
        if self._loss == 'RMSE' and self.class_weights is not None:
            raise NotImplementedError('GradientBoostingRegressor does not support class_weights')
        parameters = self.get_params()
        parameters['numeric_mode'] = mode
        return GradientBoosting(loss=self._loss, **parameters)

    def get_params(self, deep=True):
        if type(self).__init__ is not _GBDTAdapter.__init__:
            raise TypeError('GBDT adapter subclasses with custom constructors need their own parameter protocol')
        names = inspect.signature(_GBDTAdapter.__init__).parameters
        return {name: getattr(self, name) for name in names if name != 'self'}

    def set_params(self, **params):
        if not params:
            return self
        values = self.get_params()
        unknown = sorted(set(params) - values.keys())
        if unknown:
            raise ValueError(f'Invalid {type(self).__name__} parameters: {unknown}')
        values.update(params)
        replacement = type(self)(**values)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self

    def _clear_fit(self):
        for name in list(self.__dict__):
            if name.endswith('_'):
                del self.__dict__[name]

    def _check_fitted(self):
        if not self.__sklearn_is_fitted__():
            try:
                from sklearn.exceptions import NotFittedError
            except ImportError:
                NotFittedError = RuntimeError
            raise NotFittedError(f'{type(self).__name__} is not fitted')

    def __sklearn_is_fitted__(self):
        return getattr(getattr(self, '_learner_', None), 'model_', None) is not None

    def __sklearn_tags__(self):
        from sklearn.utils import Tags, TargetTags, ClassifierTags, RegressorTags
        classifier = self._estimator_type == 'classifier'
        return Tags(estimator_type=self._estimator_type,
                    target_tags=TargetTags(required=True),
                    classifier_tags=ClassifierTags(multi_class=False) if classifier else None,
                    regressor_tags=None if classifier else RegressorTags())

    def _fit_native(self, X, y, sample_weight, eval_set):
        raw_mode = self.numeric_mode if self.numeric_mode is not None else _backend.default_mode()
        learner = self._new_learner(raw_mode)
        mode = raw_mode.strip().lower()
        learner.numeric_mode = mode
        binding = _backend.binding('_mojolearn_gbdt', mode)
        if binding.gbdt_numeric_mode() != {'fast': 0, 'identical': 1, 'deterministic': 2}[mode]:
            raise RuntimeError('GBDT adapter native numeric mode disagrees with requested mode')
        learner.fit(X, y, sample_weight=sample_weight, eval_set=eval_set)
        self._learner_ = learner
        self.numeric_mode_ = mode
        self.n_features_in_ = learner.n_features_in_
        self.model_ = learner.model_
        for name in ('loss_curve_', 'test_loss_curve_', 'best_iteration_', 'stopped_early_'):
            setattr(self, name, getattr(learner, name))
        return self

    def save(self, path):
        raise NotImplementedError('GBDT adapter archives are not implemented; pickle retains adapter state')

    @classmethod
    def load(cls, path):
        raise NotImplementedError('Legacy raw GBDT archives have no adapter vocabulary/parameters')


class GradientBoostingClassifier(_GBDTAdapter):
    """Binary Logloss GPU classifier; original class labels are retained.

    predict returns labels, decision_function returns raw Float32 margins,
    predict_proba returns GPU Float32 [negative, positive] probabilities.
    Positive means the larger sorted class; a zero margin predicts class zero.
    Training weights are supported; weighted scoring is not. eval_set labels
    must belong to the training vocabulary, and its X must already be transformed.
    """
    _loss = 'Logloss'
    _estimator_type = 'classifier'

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self._clear_fit()
        target, kind = metrics._classification_labels(y, 'y')
        classes = sorted(set(target))
        if len(classes) != 2:
            raise ValueError('GradientBoostingClassifier requires exactly two training classes')
        vocabulary = {label: i for i, label in enumerate(classes)}
        encoded = np.asarray([vocabulary[label] for label in target], dtype=np.float32)
        if eval_set is not None:
            if isinstance(eval_set, list):
                if len(eval_set) != 1:
                    raise ValueError('eval_set accepts one (X, y) pair')
                eval_set = eval_set[0]
            if not isinstance(eval_set, tuple) or len(eval_set) != 2:
                raise ValueError('eval_set must be (X_eval, y_eval) or a one-pair list')
            eval_X, eval_y = eval_set
            eval_labels, eval_kind = metrics._classification_labels(eval_y, 'eval_set y')
            if eval_kind != kind or any(label not in vocabulary for label in eval_labels):
                raise ValueError('eval_set contains labels outside the training vocabulary')
            eval_set = (eval_X, np.asarray([vocabulary[label] for label in eval_labels], dtype=np.float32))
        self._fit_native(X, encoded, sample_weight, eval_set)
        if kind == 'string':
            self.classes_ = np.asarray(classes)
        elif min(classes) >= np.iinfo(np.int64).min and max(classes) <= np.iinfo(np.int64).max:
            self.classes_ = np.asarray(classes, dtype=np.int64)
        elif min(classes) >= 0 and max(classes) <= np.iinfo(np.uint64).max:
            self.classes_ = np.asarray(classes, dtype=np.uint64)
        else:
            self.classes_ = np.asarray(classes, dtype=object)
        self.n_classes_ = 2
        self._label_kind_ = kind
        return self

    def decision_function(self, X):
        self._check_fitted()
        return self._learner_.predict(X)

    def _binary_output(self, X, probabilities):
        margins = np.ascontiguousarray(self.decision_function(X), dtype=np.float32)
        if margins.ndim != 1 or not np.all(np.isfinite(margins)):
            raise ValueError('Classifier margins must be finite scalar Float32 values')
        n = len(margins)
        binding = _backend.binding('_mojolearn_gbdt', self.numeric_mode_)
        if probabilities:
            output = np.empty((n, 2), dtype=np.float32)
            wrote = binding.gbdt_binary_probabilities(_addr_ro(margins), _addr(output), [n])
        else:
            output = np.empty(n, dtype=np.int32)
            wrote = binding.gbdt_binary_classes(_addr_ro(margins), _addr(output), [n])
        if wrote != (2 * n if probabilities else n):
            raise RuntimeError('GBDT binary output returned an unexpected row count')
        return output

    def predict_proba(self, X):
        return self._binary_output(X, True)

    def predict(self, X):
        self._check_fitted()
        return self.classes_[self._binary_output(X, False)]

    def score(self, X, y, sample_weight=None):
        self._check_fitted()
        if sample_weight is not None:
            raise NotImplementedError('GBDT adapter score does not support sample_weight')
        target, kind = metrics._classification_labels(y, 'y')
        if kind != self._label_kind_:
            raise TypeError('score labels must have the training label type')
        vocabulary = {label: i for i, label in enumerate(self.classes_.tolist())}
        encoded = np.asarray([vocabulary.get(label, -1) for label in target], dtype=np.int32)
        return metrics.accuracy_score(encoded, self._binary_output(X, False),
                                      numeric_mode=self.numeric_mode_)


class GradientBoostingRegressor(_GBDTAdapter):
    """RMSE GPU regressor with raw Float32 predictions and GPU Float32 R²."""
    _loss = 'RMSE'
    _estimator_type = 'regressor'

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self._clear_fit()
        target = np.asarray(y)
        self._regression_target(target, 'y')
        if eval_set is not None:
            if isinstance(eval_set, list):
                if len(eval_set) != 1:
                    raise ValueError('eval_set accepts one (X, y) pair')
                eval_set = eval_set[0]
            if not isinstance(eval_set, tuple) or len(eval_set) != 2:
                raise ValueError('eval_set must be (X_eval, y_eval) or a one-pair list')
            eval_X, eval_y = eval_set
            eval_target = np.asarray(eval_y)
            self._regression_target(eval_target, 'eval_set y')
            eval_set = (eval_X, eval_target)
        return self._fit_native(X, target, sample_weight, eval_set)

    @staticmethod
    def _regression_target(target, name):
        if target.ndim != 1 or target.size == 0:
            raise ValueError(f'{name} must be nonempty one-dimensional targets')
        if target.dtype != np.dtype('float32'):
            raise TypeError(f'{name} must have dtype float32')
        if not np.all(np.isfinite(target)):
            raise ValueError(f'{name} must be finite')

    def predict(self, X):
        self._check_fitted()
        return self._learner_.predict(X)

    def score(self, X, y, sample_weight=None):
        self._check_fitted()
        if sample_weight is not None:
            raise NotImplementedError('GBDT adapter score does not support sample_weight')
        target = np.asarray(y)
        self._regression_target(target, 'y')
        return metrics.r2_score(target, self.predict(X),
                                numeric_mode=self.numeric_mode_)
