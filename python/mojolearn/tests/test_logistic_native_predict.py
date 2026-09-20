# SPDX-License-Identifier: Apache-2.0
"""Binary logistic prediction stays native after its exact score fold."""

import unittest
from unittest.mock import patch

import numpy as np

from mojolearn._array import Array
from mojolearn.linear_model import LogisticRegression


class TestNativeBinaryPredict(unittest.TestCase):
    def test_matches_strict_threshold_without_python_scores(self):
        x = np.array([
            [-2.0, 0.0], [-0.0, 0.0], [0.0, 0.0], [2.0, 0.0],
            [1.0, -1.0], [-1.0, 1.0],
        ], dtype=np.float32)
        model = LogisticRegression(fit_intercept=False)
        model._w = Array.from_buffer(np.array([1.0, -1.0], dtype=np.float32))
        model.classes_ = [-7, 11]
        model.n_features_in_ = 2
        scores = np.asarray(model.decision_function(x)).copy()
        expected = np.where(scores > 0.0, 11, -7).astype(np.int64)

        with patch.object(model, "decision_function",
                          side_effect=AssertionError("predict exported scores")):
            actual = np.asarray(model.predict(x))
        np.testing.assert_array_equal(actual, expected)

    def test_float_and_object_labels_keep_public_types(self):
        x = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
        for classes, expected in (([-1.5, 2.5], [-1.5, -1.5, 2.5]),
                                  (["no", "yes"], ["no", "no", "yes"])):
            with self.subTest(classes=classes):
                model = LogisticRegression(fit_intercept=False)
                model._w = Array.from_buffer(np.array([1.0], dtype=np.float32))
                model.classes_ = classes
                model.n_features_in_ = 1
                actual = model.predict(x)
                np.testing.assert_array_equal(np.asarray(actual), expected)


if __name__ == "__main__":
    unittest.main()
