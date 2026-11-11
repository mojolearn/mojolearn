# SPDX-License-Identifier: Apache-2.0
"""GP score/public-class seams; native suites own numerical kernel oracles."""
import numpy as np
import pytest

from mojolearn._gpc_impl import GaussianProcessClassifier
from mojolearn._cpu_reference import reference_training


@pytest.fixture
def fitted():
    x = np.array([[-2, 0], [-1, 1], [-1, -1], [1, 1], [1, -1], [2, 0]], dtype=np.float32)
    with reference_training():
        model = GaussianProcessClassifier(max_iter_predict=8).fit(x, [0, 0, 0, 1, 1, 1])
    return model, x


@pytest.mark.parametrize("label", [17, np.int64(17), False, 1 << 90, "same", "surrogate\ud800"])
def test_score_collapses_duplicate_rebound_class_values(fitted, label):
    model, x = fitted
    model.classes_ = [label, label]
    assert model.score(x, [label] * len(x)) == 1.0


def test_score_unseen_truth_labels_are_misses(fitted):
    model, x = fitted
    assert model.score(x, [1 << 90] * len(x)) == 0.0


def test_score_does_not_admit_unused_rebound_class_as_metric_label(fitted):
    model, x = fitted
    query = x[:1]
    selected = int(np.asarray(model.predict(query)).item())
    classes = ["unused", "unused"]
    classes[selected] = 17
    model.classes_ = classes
    assert model.score(query, [17]) == 1.0


@pytest.mark.parametrize("truth, name", [([0, 0, 0, 1, 1, 1], "y_pred"), ([0.5] * 6, "y_true")])
def test_score_keeps_truth_before_prediction_float_label_refusal(fitted, truth, name):
    model, x = fitted
    model.classes_ = [0.0, 1.0]
    with pytest.raises(TypeError, match=name + " must contain only strings or only integers"):
        model.score(x, truth)


def test_score_row_count_precedes_metric_label_refusal(fitted):
    model, x = fitted
    with pytest.raises(ValueError, match="y has 1 entries for 6 rows"):
        model.score(x, [0.5])
