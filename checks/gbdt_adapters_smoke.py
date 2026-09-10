"""Focused actual GPU adapters, raw-equivalence and fitted-state smoke."""
import pickle

import numpy as np
from mojolearn import GradientBoosting, _backend, metrics
from mojolearn._gbdt_adapters import GradientBoostingClassifier, GradientBoostingRegressor


def main():
    X = np.tile(np.array([[-2., 0.], [-1., 1.], [1., 1.], [2., 0.]], np.float32), (8, 1))
    codes = (X[:, 0] > 0).astype(np.float32)
    for mode in ('fast', 'deterministic', 'identical'):
        for policy in ('SymmetricTree', 'Depthwise', 'Lossguide'):
            options = dict(n_estimators=2, max_depth=2, learning_rate=.5,
                           random_state=7, numeric_mode=mode, grow_policy=policy,
                           max_leaves=4 if policy == 'Lossguide' else None)
            for cls in (GradientBoostingClassifier, GradientBoostingRegressor):
                classifier = cls is GradientBoostingClassifier
                y = np.where(codes > 0, 'positive', 'negative') if classifier else codes
                model = cls(**options).fit(X, y)
                base = GradientBoosting(loss='Logloss' if classifier else 'RMSE', **options).fit(X, codes)
                assert model.model_ == base.model_
                raw = model.decision_function(X) if classifier else model.predict(X)
                np.testing.assert_array_equal(raw.view(np.uint32), base.predict(X).view(np.uint32))
                score = model.score(X, y)
                assert np.isfinite(score) and score > .5
                restored = pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(restored.predict(X), model.predict(X))
                if classifier:
                    probability = model.predict_proba(X)
                    assert probability.dtype == np.float32 and probability.shape == (len(X), 2)
                    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-7)
                    if policy == 'SymmetricTree':
                        loss = metrics.log_loss(y, probability, labels=model.classes_, numeric_mode=mode)
                        assert np.isfinite(loss) and loss >= 0
                    np.testing.assert_array_equal(restored.predict_proba(X).view(np.uint32), probability.view(np.uint32))
                print('PASS', cls.__name__, mode, policy, score, flush=True)
        # Preserve integers that NumPy's default signed/unsigned promotion
        # would turn into inexact Float64 classes.
        for label_pair in ((-1, 2**64-1), (-2**80, 2**80)):
            y = np.asarray([label_pair[int(c)] for c in codes], dtype=object)
            huge = GradientBoostingClassifier(n_estimators=1, max_depth=2, numeric_mode=mode).fit(X, y)
            assert huge.classes_.tolist() == list(label_pair)
            assert set(huge.predict(X).tolist()).issubset(set(label_pair))
        previous = _backend.default_mode()
        try:
            _backend.set_default_mode(mode)
            fitted = GradientBoostingClassifier(n_estimators=1, max_depth=2).fit(
                X, np.where(codes > 0, 'yes', 'no'),
                eval_set=(X[:8], np.where(codes[:8] > 0, 'yes', 'no')))
            prediction = fitted.predict_proba(X)
            _backend.set_default_mode('fast' if mode != 'fast' else 'identical')
            assert fitted.numeric_mode_ == mode
            np.testing.assert_array_equal(fitted.predict_proba(X).view(np.uint32), prediction.view(np.uint32))
            try:
                fitted.fit(X, np.where(codes > 0, 'yes', 'no'), eval_set=(X[:1], ['unknown']))
            except ValueError:
                pass
            else:
                raise AssertionError('unknown eval label accepted')
            assert not fitted.__sklearn_is_fitted__() and not hasattr(fitted, 'classes_')
        finally:
            _backend.set_default_mode(previous)
    print('PASS focused GPU GBDT adapter smoke', flush=True)


if __name__ == '__main__':
    main()
