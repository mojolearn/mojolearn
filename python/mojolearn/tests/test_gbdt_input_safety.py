# SPDX-License-Identifier: Apache-2.0
"""Input refusals before native fitting; no GPU work is needed."""
import numpy as np
import pytest

from mojolearn.ensemble import GradientBoosting


def guarded(loss="RMSE", **kwargs):
    model = GradientBoosting(loss=loss, **kwargs)
    model._bind = lambda name: pytest.fail("invalid input reached native binding")
    return model


@pytest.mark.parametrize("loss", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("labels", [
    [-1, 0, 1], [0, 0.5, 1], [0, np.nan, 1], [0, np.inf, 1],
    [0, 0, 0], [1, 2, 3], [0, 2, 2], [0, 1, 1e30],
])
def test_multiclass_invalid_labels_never_enter_native(loss, labels):
    with pytest.raises(ValueError, match="class"):
        guarded(loss).fit(np.zeros((3, 2), np.float32), labels)


@pytest.mark.parametrize("loss", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("labels", [[-1], [0.5], [np.nan], [np.inf], [2]])
def test_eval_class_codes_are_bounded_by_training(loss, labels):
    with pytest.raises(ValueError, match="eval_set labels"):
        guarded(loss).fit(np.zeros((3, 2), np.float32), [0, 1, 1],
                          eval_set=(np.zeros((1, 2), np.float32), labels))


@pytest.mark.parametrize("loss", ["MultiClass", "MultiClassOneVsAll"])
def test_class_weight_count_checked_before_native(loss):
    with pytest.raises(ValueError, match="class_weights"):
        guarded(loss, class_weights=[1, 2, 3]).fit(
            np.zeros((3, 2), np.float32), [0, 1, 1])


@pytest.mark.parametrize("weights", [[1, np.nan], [1, np.inf], [-1, 1], [0, 0]])
def test_invalid_sample_weight_never_enters_native(weights):
    with pytest.raises(ValueError, match="sample_weight"):
        guarded().fit(np.zeros((2, 2), np.float32), [0, 1], sample_weight=weights)


@pytest.mark.parametrize("weights", [[1, np.nan], [1, np.inf], [-1, 1]])
def test_nonfinite_class_weights_refused(weights):
    with pytest.raises(ValueError, match="class_weights"):
        GradientBoosting(loss="MultiClass", class_weights=weights)


def test_forbidden_nan_never_enters_native():
    with pytest.raises(ValueError, match="Forbidden"):
        guarded(nan_mode="Forbidden").fit([[0, np.nan], [1, 2]], [0, 1])


@pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
def test_empty_training_shape_never_enters_native(shape):
    with pytest.raises(ValueError):
        guarded().fit(np.zeros(shape, np.float32), np.zeros(shape[0], np.float32))
