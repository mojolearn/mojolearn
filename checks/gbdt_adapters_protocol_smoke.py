"""Optional sklearn adapter protocol; --gpu adds bounded serial search."""
import argparse
import numpy as np
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags
from mojolearn._gbdt_adapters import GradientBoostingClassifier, GradientBoostingRegressor


def main(gpu=False):
    for cls in (GradientBoostingClassifier, GradientBoostingRegressor):
        raw = np.int64(2)
        model = cls(n_estimators=raw, numeric_mode='identical')
        assert model.get_params()['n_estimators'] is raw
        assert clone(model).get_params() == model.get_params()
        assert not clone(model).__sklearn_is_fitted__()
        assert is_classifier(model) == (cls is GradientBoostingClassifier)
        assert is_regressor(model) == (cls is GradientBoostingRegressor)
        if is_classifier(model):
            assert not get_tags(model).classifier_tags.multi_class
        try:
            model.predict(np.ones((2, 1), np.float32))
        except NotFittedError:
            pass
        else:
            raise AssertionError('unfitted prediction accepted')
        for values in ({'unknown': 1}, {'numeric_mode': 1}, {'grow_policy': 'wrong'}):
            before = model.get_params()
            try:
                model.set_params(**values)
            except (ValueError, NotImplementedError):
                pass
            else:
                raise AssertionError('invalid parameters accepted')
            assert model.get_params() == before
        model.numeric_mode = 1
        try:
            model.fit(np.ones((2, 1), np.float32),
                      np.array([0, 1], np.int32 if cls is GradientBoostingClassifier else np.float32))
        except ValueError:
            pass
        else:
            raise AssertionError('invalid direct mode mutation accepted')
        assert not model.__sklearn_is_fitted__()
    if gpu:
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV
        from mojolearn import StandardScaler
        X = np.tile(np.array([[-2.], [-1.], [1.], [2.]], np.float32), (8, 1))
        for mode in ('fast', 'deterministic', 'identical'):
            for cls in (GradientBoostingClassifier, GradientBoostingRegressor):
                y = np.where(X[:, 0] > 0, 'yes', 'no') if cls is GradientBoostingClassifier else (X[:, 0] > 0).astype(np.float32)
                pipe = Pipeline([('scale', StandardScaler(numeric_mode=mode)),
                                 ('model', cls(n_estimators=2, learning_rate=.5, numeric_mode=mode))])
                search = GridSearchCV(pipe, {'model__max_depth': [1, 2]}, cv=2, n_jobs=1, error_score='raise')
                search.fit(X, y)
                assert search.best_estimator_[-1].numeric_mode_ == mode
                assert search.score(X, y) > .5
                print('PASS GPU Pipeline/GridSearchCV', cls.__name__, mode, flush=True)
    print('PASS GBDT adapter sklearn protocol', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    main(parser.parse_args().gpu)
