# SPDX-License-Identifier: Apache-2.0
"""Native SGD parameter reads retain the prepared class-weight owner."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("n_classes", [2, 3])
def test_unit_class_weights_preserve_weighted_sgd_fit(n_classes):
    # A fresh debug allocator poisons freed array.array storage. Without it,
    # a dropped class-weight Array can still contain plausible old numbers.
    env = dict(os.environ, PYTHONMALLOC="debug", MOJOLEARN_NUMERIC_MODE="identical")
    source = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    body = textwrap.dedent("""
        import sys
        import numpy as np
        from mojolearn import SGDClassifier

        count = int(sys.argv[1])
        labels = np.arange(count, dtype=np.int32) * 10 + 7
        y = np.tile(labels, 32)
        x = np.random.default_rng(53).normal(size=(len(y), 5)).astype(np.float32)
        weights = np.linspace(.25, 2, len(y), dtype=np.float32)
        weights[::11] = 0
        before_x, before_y, before_w = x.copy(), y.copy(), weights.copy()
        options = dict(loss="modified_huber", max_iter=5, tol=None,
                       random_state=3, numeric_mode="identical")
        baseline = SGDClassifier(**options).fit(x, y, sample_weight=weights)
        # Equal class counts make balanced weights exactly one. Explicit
        # unit weights and balanced weights must therefore preserve this
        # fit, including the nonuniform per-sample weights, in both binary
        # and multiclass SGD.
        for class_weight in ("balanced", dict.fromkeys(map(int, labels), 1.0)):
            actual = SGDClassifier(class_weight=class_weight, **options).fit(
                x, y, sample_weight=weights)
            for name in ("coef_", "intercept_"):
                a = np.asarray(getattr(actual, name))
                b = np.asarray(getattr(baseline, name))
                assert np.isfinite(a).all(), (name, class_weight)
                np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
            np.testing.assert_array_equal(actual.classes_, baseline.classes_)
            np.testing.assert_array_equal(
                np.asarray(actual.decision_function(x)).view(np.uint32),
                np.asarray(baseline.decision_function(x)).view(np.uint32))
            assert actual.n_iter_ == baseline.n_iter_
            assert actual.t_ == baseline.t_
        np.testing.assert_array_equal(x, before_x)
        np.testing.assert_array_equal(y, before_y)
        np.testing.assert_array_equal(weights, before_w)
    """)
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", body, str(n_classes)],
        env=env, text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-6000:]
