# SPDX-License-Identifier: Apache-2.0
"""Address-only native calls retain temporary owners until the call finishes."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def _debug_process(body):
    # array.array scratch is allocated through Python's allocator. A fresh
    # process with debug allocation poisons released bytes and checks guards;
    # it exposes a lost owner even when the normal allocator reuses its bytes
    # in a way that happens to leave a plausible numeric value.
    env = dict(os.environ, PYTHONMALLOC="debug", MOJOLEARN_NUMERIC_MODE="identical")
    source = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(body)],
        env=env, text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-6000:]


def test_cluster_class_scratch_survives_native_encoding():
    _debug_process("""
        import math
        import numpy as np
        from mojolearn import metrics

        true = np.array([-3, 7, 7, 100000, -3], dtype=np.int32)
        pred = np.array([2, 2, 2, 1, 1], dtype=np.int32)
        # One pair agrees in both clusterings, with two true pairs and four
        # predicted pairs. This oracle does not call any product helper.
        expected = 1.0 / math.sqrt(8.0)
        for _ in range(12):
            actual = metrics.fowlkes_mallows_score(true, pred)
            assert abs(actual - expected) < 1e-15, (actual, expected)
            assert metrics.fowlkes_mallows_score(pred, true) == actual
            assert metrics.fowlkes_mallows_score(true, true) == 1.0
        np.testing.assert_array_equal(true, [-3, 7, 7, 100000, -3])
        np.testing.assert_array_equal(pred, [2, 2, 2, 1, 1])
    """)


def test_stratified_shuffle_keeps_permutation_arena_and_rows():
    _debug_process("""
        import numpy as np
        from mojolearn.model_selection import StratifiedShuffleSplit

        x = np.arange(288, dtype=np.float32).reshape(96, 3)
        y = np.tile(np.array([10, 20, 30], dtype=np.int64), 32)
        first = None
        for _ in range(8):
            current = []
            for train, test in StratifiedShuffleSplit(3, test_size=.25, random_state=7).split(x, y):
                train, test = np.asarray(train), np.asarray(test)
                assert len(train) == 72 and len(test) == 24
                np.testing.assert_array_equal(np.sort(np.concatenate((train, test))), np.arange(96))
                np.testing.assert_array_equal(np.unique(y[test], return_counts=True)[1], [8, 8, 8])
                current.append((train.copy(), test.copy()))
            assert len(current) == 3
            if first is None:
                first = current
            else:
                for actual, expected in zip(current, first):
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1], expected[1])
    """)


def test_voting_sum_keeps_native_fsum_input():
    _debug_process("""
        import numpy as np
        import mojolearn as ml

        rng = np.random.default_rng(21)
        x = rng.normal(size=(64, 3)).astype(np.float32)
        y = (2 * x[:, 0] - x[:, 1] + .2 * x[:, 2]).astype(np.float32)
        model = ml.VotingRegressor([
            ('a', ml.DecisionTreeRegressor(max_depth=2, random_state=3)),
            ('b', ml.DecisionTreeRegressor(max_depth=4, random_state=5)),
        ], weights=[1., 2.]).fit(x, y)
        a, b = [np.asarray(e.predict(x), dtype=np.float64) for e in model.estimators_]
        expected = (a + 2 * b) / 3
        for _ in range(8):
            np.testing.assert_allclose(model.predict(x), expected, rtol=2e-6, atol=2e-6)
    """)


def test_calibration_keeps_temporary_class_column_indices():
    _debug_process("""
        import numpy as np
        import mojolearn as ml

        rng = np.random.default_rng(15)
        x = rng.normal(size=(96, 4)).astype(np.float32)
        y = np.argmax(x[:, :3], axis=1).astype(np.int64)
        model = ml.CalibratedClassifierCV(
            ml.DecisionTreeClassifier(max_depth=3, random_state=7), cv=3,
        ).fit(x, y)
        first = np.asarray(model.predict_proba(x)).copy()
        assert first.shape == (96, 3) and np.isfinite(first).all()
        np.testing.assert_allclose(first.sum(axis=1), 1, rtol=2e-6, atol=2e-6)
        for _ in range(8):
            np.testing.assert_array_equal(model.predict_proba(x), first)
    """)


def test_forest_and_adaboost_keep_encoded_labels_through_fit():
    _debug_process("""
        import numpy as np
        import mojolearn as ml

        rng = np.random.default_rng(25)
        x = rng.normal(size=(96, 4)).astype(np.float32)
        y = np.argmax(x[:, :3], axis=1).astype(np.int64) * 10 + 5
        for create in (
            lambda: ml.RandomForestClassifier(n_estimators=3, max_depth=3, random_state=7),
            lambda: ml.AdaBoostClassifier(
                ml.DecisionTreeClassifier(max_depth=2), n_estimators=3, random_state=7,
            ),
        ):
            first = None
            for _ in range(4):
                model = create().fit(x, y)
                current = np.asarray(model.predict_proba(x)).copy()
                assert current.shape == (96, 3) and np.isfinite(current).all()
                np.testing.assert_allclose(current.sum(axis=1), 1, rtol=2e-6, atol=2e-6)
                if first is None:
                    first = current
                else:
                    np.testing.assert_array_equal(current, first)
    """)
