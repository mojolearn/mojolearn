#!/usr/bin/env python3
"""Public GPU binding: analytic weighted logistic Hessians and ABI tails."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
from mojolearn.ensemble import GradientBoosting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('fast', 'deterministic', 'identical'))
    mode = parser.parse_args().mode
    rows = np.arange(256)
    x = (rows % 4).astype(np.float32)[:, None]
    y = (rows // 4 < np.array([8, 16, 48, 56])[rows % 4]).astype(np.float32)
    for policy in ('Depthwise', 'Lossguide'):
        common = dict(numeric_mode=mode, grow_policy=policy, loss='Logloss',
                      score_function='NewtonL2', n_estimators=1, max_depth=2,
                      border_count=3, learning_rate=1., l2_leaf_reg=0.,
                      bootstrap_type='Bayesian', bagging_temperature=0.,
                      boost_from_average=False, class_weights=[2., 2.])
        baseline = GradientBoosting(**common).fit(x, y)
        assert baseline._bind('_mojolearn_gbdt').gbdt_numeric_mode() == {
            'fast': 0, 'identical': 1, 'deterministic': 2}[mode]
        for threshold, count in ((None, 4), (0., 4), (-0., 4), (32., 4),
                                 (32.000000001, 2), (64.000000001, 1)):
            model = GradientBoosting(**common, min_child_hessian=threshold).fit(x, y)
            assert model.get_tree_leaf_counts().tolist() == [count], (policy, threshold)
            if threshold is None or threshold == 0:
                assert model.model_ == baseline.model_
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'hessian.npz'
                model.save(path)
                loaded = GradientBoosting.load(path)
                np.testing.assert_array_equal(loaded.predict_proba(x), model.predict_proba(x))
            print(mode, policy, 'threshold', threshold, 'leaves', count)
        # Curvature must be recomputed after the first tree's +/-2 logits.
        # Each initial child has H=64; at iteration2 H=256*p*(1-p)~=26.9.
        two_trees = {**common, 'max_depth': 1, 'n_estimators': 2,
                     'leaf_estimation_iterations': 1, 'leaf_estimation_method': 'Newton'}
        pure = (rows % 4 >= 2).astype(np.float32)
        updated = GradientBoosting(**two_trees, min_child_hessian=32).fit(x, pure)
        assert updated.get_tree_leaf_counts().tolist() == [2, 1], policy
        print(mode, policy, 'updated curvature leaves', [2, 1])
        # Both counted class weights and both optional growth controls reach
        # native code: the huge gain bound must override a legal Hessian split.
        both = GradientBoosting(**common, min_split_gain=1e10,
                                min_child_hessian=32).fit(x, y)
        assert both.get_tree_leaf_counts().tolist() == [1]
        # Nonconstant stochastic bootstrap scales the Hessian; a bound beyond
        # total available mass must still reject the root without a stale leaf.
        sampled = {**common, 'bagging_temperature': 1., 'random_state': 17}
        sampled_model = GradientBoosting(**sampled, min_child_hessian=1e8).fit(x, y)
        assert sampled_model.get_tree_leaf_counts().tolist() == [1]
    # The builder runs this family coverage itself only in FAST. Exercise
    # equivalent actual launches explicitly in every selected numeric mode.
    grid = np.random.default_rng(0).random((512, 3), dtype=np.float32)
    for loss in ('RMSE', 'Logloss', 'MAE', 'MultiClass', 'MultiClassOneVsAll'):
        target = grid[:, 0]
        if loss == 'Logloss':
            target = (grid[:, 0] > .5).astype(np.float32)
        elif loss.startswith('MultiClass'):
            target = np.minimum((grid[:, 0] * 3).astype(np.int32), 2).astype(np.float32)
        model = GradientBoosting(loss=loss, numeric_mode=mode, n_estimators=2,
                                 max_depth=3, border_count=16).fit(grid, target)
        assert np.isfinite(model.predict(grid)).all()
        if loss == 'Logloss' or loss.startswith('MultiClass'):
            assert np.isfinite(model.predict_proba(grid)).all()
        print(mode, 'existing loss family smoke', loss, 'PASS')
    print('MIN CHILD HESSIAN PUBLIC GREEN', mode)


if __name__ == '__main__':
    main()
