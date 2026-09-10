#!/usr/bin/env python3
"""Installed GBDT binding smoke for the optional split-gain ABI tail."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
from mojolearn.ensemble import GradientBoosting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('fast', 'deterministic', 'identical'), required=True)
    mode = parser.parse_args().mode
    x = (np.arange(256) % 4).astype(np.float32)[:, None]
    y = np.array([0, .25, 4, 4.25], np.float32)[np.arange(256) % 4]
    for policy in ('Depthwise', 'Lossguide'):
        defaults = dict(grow_policy=policy, numeric_mode=mode, n_estimators=1,
                        max_depth=2, border_count=3, learning_rate=1.,
                        l2_leaf_reg=0., score_function='L2', bootstrap_type='No',
                        boost_from_average=False)
        baseline = GradientBoosting(**defaults).fit(x, y)
        assert baseline._bind('_mojolearn_gbdt').gbdt_numeric_mode() == {
            'fast': 0, 'identical': 1, 'deterministic': 2}[mode]
        for threshold, count in ((None, 4), (0., 4), (1.999, 4), (2., 2), (1024., 1)):
            model = GradientBoosting(**defaults, min_split_gain=threshold).fit(x, y)
            assert model.get_tree_leaf_counts().tolist() == [count], (policy, threshold)
            if threshold is None:
                assert model.model_ == baseline.model_
            expected = (y if count == 4 else
                        np.where(x[:, 0] < 2, .125, 4.125) if count == 2 else
                        np.full(256, 2.125))
            np.testing.assert_array_equal(model.predict(x), expected)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'gain.npz'
                model.save(path)
                loaded = GradientBoosting.load(path)
                np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
            print(mode, policy, 'min_split_gain', threshold, 'leaves', count)
        # Counted class weights must remain distinct from the optional tail.
        labels = (np.arange(256) % 2).astype(np.float32)
        common = dict(grow_policy=policy, numeric_mode=mode, loss='Logloss',
                      n_estimators=1, max_depth=2, border_count=3,
                      learning_rate=.3, min_split_gain=1e100)
        plain = GradientBoosting(**common).fit(x, labels)
        weighted = GradientBoosting(**common, class_weights=[1, 3]).fit(x, labels)
        assert plain.get_tree_leaf_counts().tolist() == [1]
        assert weighted.get_tree_leaf_counts().tolist() == [1]
        assert weighted.predict_proba(x)[:, 1].mean() > plain.predict_proba(x)[:, 1].mean()
    print('MIN SPLIT GAIN PUBLIC BINDING GREEN', mode)


if __name__ == '__main__':
    main()
