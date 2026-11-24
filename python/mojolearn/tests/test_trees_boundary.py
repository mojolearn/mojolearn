# SPDX-License-Identifier: Apache-2.0
"""Tree ensembles' public boundary: fitted member parameters."""
import numpy as np

import mojolearn as ml


# ---------------------------------------------------------------- moved from test_meta_members_native.py (vdv-pytest-a)
# The two-fit repeats of DARTRegressor, AdaBoostClassifier and AdaBoostRegressor are the verify lanes trees-dart-reg,
# trees-adaboost-clf and trees-adaboost-reg (tools/identity_lanes/trees.py) on both sides; the DART trees' own
# seeds stay here.


def test_dart_trees_hold_their_own_seeds():
    rng = np.random.default_rng(1)
    X = rng.random((120, 4), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    a = ml.DARTRegressor(n_estimators=8, random_state=0).fit(X, y)
    seeds = [t.get_params()["random_state"] for t in a.trees_]
    assert len(set(seeds)) == len(seeds)
