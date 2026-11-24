# SPDX-License-Identifier: Apache-2.0
"""The metrics programs at the Python boundary: returned types and refusals by words, on every device. Their bits on
each side are the verify lanes."""
import numpy as np
import pytest

from mojolearn import LinearRegression, Ridge

# ---------------------------------------------------------------------------------------------------------------------
# `score` of Ridge and LinearRegression (moved from test_metrics_device_score.py).
CLASSES = [Ridge, LinearRegression]


def data(rows=300, cols=5, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    noise = rng.random(rows, dtype=np.float32) * np.float32(0.25)
    y = (x @ np.arange(1, cols + 1, dtype=np.float32) + noise).astype(np.float32)
    return x, y


@pytest.mark.parametrize("cls", CLASSES)
def test_score_on_the_default_device_is_a_float(cls):
    x, y = data()
    value = cls().fit(x, y).score(x, y)
    assert type(value) is float and 0.0 < value <= 1.0


@pytest.mark.parametrize("cls", CLASSES)
def test_score_refuses_a_target_of_another_length_on_every_device(cls):
    x, y = data()
    for device in (None, "cpu"):
        model = cls(device=device).fit(x, y)
        with pytest.raises(Exception, match="different shapes"):
            model.score(x, y[:-1])
