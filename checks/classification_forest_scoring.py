#!/usr/bin/env python3
"""Small actual GPU forest -> sklearn make_scorer -> GPU classification witness.

Requires sklearn 1.8 and existing RF/ET/metrics artifacts in all three modes.
Run under tools/with_build_lock.sh; host sklearn metrics are independent oracles.
"""
import json

import numpy as np
import sklearn
from sklearn import metrics as reference

from mojolearn import metrics
from mojolearn.randomforest import RandomForestClassifier
from mojolearn.extratrees import ExtraTreesClassifier


def main():
    assert sklearn.__version__.split('.')[:2] == ['1', '8'], sklearn.__version__
    x = np.tile(np.array([[-2., 1.], [-1., 0.], [1., 0.], [2., 1.]], np.float32), (16, 1))
    y = np.where(x[:, 0] > 0, 'positive', 'negative')
    fits = 0
    for mode in ('fast', 'deterministic', 'identical'):
        assert metrics._get_binding(mode).metrics_numeric_mode() == {
            'fast': 0, 'deterministic': 2, 'identical': 1}[mode]
        for cls in (RandomForestClassifier, ExtraTreesClassifier):
            kwargs = dict(n_estimators=2, max_depth=2, max_features=1.,
                          random_state=7, numeric_mode=mode)
            if cls is RandomForestClassifier:
                kwargs['n_bins'] = 4
            model = cls(**kwargs).fit(x, y)
            fits += 1
            prediction = model.predict(x)
            # Score deliberately perturbed reference labels to avoid all-one
            # ratios hiding class mapping or response-method mistakes.
            evaluation_labels = y.copy()
            evaluation_labels[::5] = np.where(y[::5] == 'positive', 'negative', 'positive')
            values = {}
            for name in ('precision_score', 'recall_score', 'f1_score'):
                scorer = reference.make_scorer(getattr(metrics, name), average='macro',
                                               zero_division=0, pos_label='positive', numeric_mode=mode)
                actual = scorer(model, x, evaluation_labels)
                expected = getattr(reference, name)(evaluation_labels, prediction,
                                                     average='macro', zero_division=0)
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
                values[name] = actual
            np.testing.assert_array_equal(
                metrics.confusion_matrix(evaluation_labels, prediction, numeric_mode=mode),
                reference.confusion_matrix(evaluation_labels, prediction))
            print(json.dumps({'mode': mode, 'model': cls.__name__, 'scores': values,
                              'artifact': model._bind().__file__, 'status': 'PASS'}), flush=True)
    print(json.dumps({'status': 'PASS', 'gpu_fits': fits, 'sklearn': sklearn.__version__}))


if __name__ == '__main__':
    main()
