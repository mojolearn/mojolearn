#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded public-native sklearn RF/ET protocol pilot. Run under build lock.

The sklearn scaler is an interoperability fixture, not an IDENTICAL pipeline
qualification. Native model and metric modes are read back separately.
"""
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mojolearn import (ExtraTreesClassifier, ExtraTreesRegressor,
                       RandomForestClassifier, RandomForestRegressor)
from mojolearn import _backend


def main():
    print('sklearn', sklearn.__version__, flush=True)
    X = np.tile(np.array([[-2., 1.], [-1., 0.], [1., 0.], [2., 1.]], np.float32), (16, 1))
    for mode in ('fast', 'deterministic', 'identical'):
        metrics = _backend.binding('_mojolearn_metrics', mode)
        assert metrics.metrics_numeric_mode() == {'fast': 0, 'deterministic': 2, 'identical': 1}[mode]
        for cls in (ExtraTreesClassifier, ExtraTreesRegressor,
                    RandomForestClassifier, RandomForestRegressor):
            classifier = 'Classifier' in cls.__name__
            y = np.where(X[:, 0] > 0, 'positive', 'negative') if classifier else X[:, 0]
            kwargs = dict(n_estimators=2, max_features=1., random_state=7, numeric_mode=mode)
            if 'RandomForest' in cls.__name__:
                kwargs['n_bins'] = 4
            model = cls(**kwargs)
            binding = model._bind()
            prefix = 'rf' if 'RandomForest' in cls.__name__ else 'trees'
            # Older RF bindings expose vendor but not a mode getter. The loader
            # validates available mode metadata and exact tier extension paths.
            getter = getattr(binding, prefix + '_numeric_mode', None)
            if getter is not None:
                assert getter() == {'fast': 0, 'deterministic': 2, 'identical': 1}[mode]
            search = GridSearchCV(Pipeline([('scale', StandardScaler()), ('model', model)]),
                                  {'model__max_depth': [1, 2]}, cv=2, n_jobs=1,
                                  error_score='raise')
            search.fit(X, y)
            best = search.best_estimator_[-1]
            assert best.numeric_mode == mode
            assert not clone(best).__sklearn_is_fitted__()
            score = search.score(X, y)
            assert np.isfinite(score) and score > .5, (cls.__name__, mode, score)
            print(mode, cls.__name__, search.best_params_, 'score', score,
                  'binding', binding.__file__, flush=True)
    print('FOREST PROTOCOL PUBLIC GREEN', flush=True)


if __name__ == '__main__':
    main()
