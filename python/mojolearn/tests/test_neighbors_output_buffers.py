# SPDX-License-Identifier: Apache-2.0
"""The k-NN classifier materializes only the public output it selects."""

import unittest

import numpy as np

import mojolearn.neighbors as neighbors


class TestClassifierOutputBuffers(unittest.TestCase):
    def test_unselected_output_uses_one_element_sentinel(self):
        x = np.arange(96, dtype=np.float32).reshape(24, 4)
        y = np.arange(24, dtype=np.int64) % 3
        model = neighbors.KNeighborsClassifier(n_neighbors=3).fit(x, y)
        q = x[:7]
        expected_labels = np.asarray(model.predict(q)).copy()
        expected_proba = np.asarray(model.predict_proba(q)).copy()

        original_empty = neighbors.empty
        calls = []

        def traced_empty(shape, dtype):
            calls.append((tuple(shape), dtype))
            return original_empty(shape, dtype)

        neighbors.empty = traced_empty
        try:
            labels = model.predict(q)
            self.assertEqual(calls[:3], [((7, 1), "<i4"), ((1,), "<f4"),
                                         ((3,), "<i4")])
            calls.clear()
            proba = model.predict_proba(q)
            self.assertEqual(calls[:3], [((1,), "<i4"), ((21,), "<f4"),
                                         ((3,), "<i4")])
        finally:
            neighbors.empty = original_empty

        np.testing.assert_array_equal(np.asarray(labels), expected_labels)
        np.testing.assert_array_equal(np.asarray(proba), expected_proba)


if __name__ == "__main__":
    unittest.main()
