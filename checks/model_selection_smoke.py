"""Fold ownership/leakage oracle and optional small real GPU pipeline smoke."""
import argparse
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from mojolearn.model_selection import cross_val_score


class FoldOracle(RegressorMixin, BaseEstimator):
    fits = []

    def fit(self, X, y):
        self.rows_ = X[:, 0].copy()
        self.fits.append(self.rows_)
        return self

    def score(self, X, y):
        assert not np.intersect1d(self.rows_, X[:, 0]).size
        return float(len(X))


def main(gpu):
    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.arange(12, dtype=np.float32)
    original = FoldOracle()
    assert np.array_equal(cross_val_score(original, X, y, cv=3), [4., 4., 4.])
    assert not hasattr(original, 'rows_')
    assert len(FoldOracle.fits) == 3
    bad = [([0, 1], [1, 2]), ([0, 0], [2]), ([-1], [2]), ([12], [2]), ([], [2]), ([0.5], [2])]
    for fold in bad:
        count = len(FoldOracle.fits)
        try:
            cross_val_score(original, X, y, cv=[([0], [1]), fold])
        except ValueError:
            pass
        else:
            raise AssertionError('bad fold accepted')
        assert len(FoldOracle.fits) == count
    assert cross_val_score(original, X, y, cv=2, scoring=lambda m, a, b: -3.).tolist() == [-3., -3.]
    if gpu:
        from mojolearn import StandardScaler, GradientBoostingRegressor
        X = np.tile(np.array([[-2.], [-1.], [1.], [2.]], np.float32), (6, 1))
        y = (X[:, 0] > 0).astype(np.float32)
        for mode in ('fast', 'deterministic', 'identical'):
            pipe = Pipeline([('scale', StandardScaler(numeric_mode=mode)),
                             ('tree', GradientBoostingRegressor(n_estimators=2, max_depth=2,
                                                               learning_rate=.5, numeric_mode=mode))])
            def score(fitted, test_X, test_y):
                assert fitted[-1].numeric_mode_ == mode
                assert fitted[0].n_samples_seen_ == 12
                return fitted.score(test_X, test_y)
            scores = cross_val_score(pipe, X, y, cv=2, scoring=score)
            assert scores.shape == (2,) and np.all(scores > .5)
            assert not hasattr(pipe[0], 'n_samples_seen_')
            print('PASS GPU serial CV', mode, scores, flush=True)
    print('PASS fold isolation and pre-fit index validation', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    main(parser.parse_args().gpu)
