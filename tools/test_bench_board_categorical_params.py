"""Constructor-only regression: no kernels, opponents, or timing."""
import unittest
from unittest.mock import patch
import numpy as np
import bench_board_algos as b

class CategoricalParams(unittest.TestCase):
    def test_query_only_categories_reach_our_constructor(self):
        data = {"X": np.array([[0, 1], [1, 0]]), "Xq": np.array([[4, 2]])}
        for arm in b.OURS_ARMS:
            with self.subTest(arm=arm), patch.object(b, "_ours_class", return_value=("CategoricalNB", dict)):
                make, _, params = b._est_factory("categorical-nb", arm, data)
                self.assertEqual(params["min_categories"], [5, 3])
                self.assertEqual(make()["min_categories"], [5, 3])
    def test_existing_fit_maximum_is_retained(self):
        data = {"X": np.array([[7, 1], [1, 0]]), "Xq": np.array([[4, 2]])}
        self.assertEqual(b._derived_params("categorical-nb", data, {})["min_categories"], [8, 3])

if __name__ == "__main__": unittest.main()
