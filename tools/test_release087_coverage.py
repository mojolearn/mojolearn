"""Focused regression checks for the release coverage audit and new metric lane."""
import types
import unittest
from unittest.mock import patch

import numpy as np
import identity_break as identity
import verification_matrix as matrix


class CoverageTests(unittest.TestCase):
    def test_public_parallel_surfaces_are_not_hidden_by_root_all(self):
        surface, modules = matrix.public_surface()
        for name in ("model_pool_training.PooledByteLanguageModelTrainer",
                     "offload_training.OffloadedByteLanguageModelTrainer",
                     "parallel_training.ParallelByteLanguageModelTrainer",
                     "parallel_classical.bootstrap"):
            self.assertIn(name, surface)
        self.assertIn("parallel_classical", modules)

    def test_reference_shards_reached_through_helper_are_counted(self):
        names = ("par-reference-knn", "par-reference-knn-reg")
        harness = types.SimpleNamespace(LANES={n: identity.LANES[n] for n in names})
        row = dict(gpu=["apple"], cpu="training", sabotage="declared", batch="part")
        name = "parallel_neighbors_reference.ReferenceShardedNeighbors"
        result = matrix.algorithm_rows(harness, {n: row for n in names},
            {name: ("class", "parallel_neighbors_reference.py", "ReferenceShardedNeighbors")}, set(), {})
        self.assertEqual(result[name]["lanes"], list(names))

    def test_refused_or_moved_cells_are_not_backend_coverage(self):
        cells = {"refused/base": {"verdict": "REFUSED", "hashes": []},
                 "moved/base": {"verdict": "MOVED", "hashes": ["a", "b"]},
                 "empty/base": {"verdict": "STABLE", "hashes": []},
                 "good/base": {"verdict": "STABLE", "hashes": ["a", "a"]}}
        columns = [dict(admit=None, cls=cls, rel=cls, cells=cells)
                   for cls in ("apple", "cpu")]
        self.assertEqual(set(matrix.gpu_coverage(columns)), {"good"})
        self.assertEqual(set(matrix.cpu_recorded(columns)), {"good"})

    def test_module_alias_keeps_qualified_reference(self):
        refs = matrix.lane_references('''def lane(ml):
    from mojolearn import parallel_classical as pc
    return pc.bootstrap([])
''')
        self.assertIn("parallel_classical.bootstrap", refs)
        self.assertNotIn("resample.bootstrap", refs)

    def test_serial_name_cannot_cover_parallel_api(self):
        def serial(ml):
            return ml.resample.bootstrap([])
        harness = types.SimpleNamespace(LANES={"serial": serial})
        row = dict(gpu=["apple"], cpu="training", sabotage="seen(build)", batch="part")
        result = matrix.algorithm_rows(harness, {"serial": row},
            {"parallel_classical.bootstrap": ("function", "parallel_classical.py", "bootstrap")}, set(), {})
        self.assertEqual(result["parallel_classical.bootstrap"]["lanes"], [])

    def test_new_lane_names_global_reduction_not_batch_invariance(self):
        self.assertIn("metrics-homogeneity-completeness", identity.LANES)
        self.assertTrue(identity.BATCH["metrics-homogeneity-completeness"].startswith("n/a:global-contingency"))
        self.assertTrue(identity.BATCH["bpe-trainer"].startswith("n/a:corpus-global"))

    def run_metric(self, bad_order=False, bad_beta=False):
        class Metrics:
            @staticmethod
            def homogeneity_score(a, b):
                return 0.25
            @staticmethod
            def completeness_score(a, b):
                return 0.5
            @staticmethod
            def v_measure_score(a, b, beta=1):
                return (1 + beta) * 0.25 * 0.5 / (beta * 0.25 + 0.5)
            @staticmethod
            def homogeneity_completeness_v_measure(a, b, beta=1):
                h, c = (0.5, 0.25) if bad_order else (0.25, 0.5)
                return h, c, Metrics.v_measure_score(a, b, beta=1 if bad_beta else beta)
        x = np.arange(96, dtype=np.float32).reshape(16, 6) - 40
        return identity.LANES["metrics-homogeneity-completeness"](
            types.SimpleNamespace(metrics=Metrics), x, np.arange(16) % 2, np.zeros(16))

    def run_ordered(self, balanced=False, mutate=False):
        def ordered(parts):
            parts = list(parts)
            def add(a, b):
                return [np.add(x, y, dtype=np.float32) for x, y in zip(a, b)]
            if balanced:
                return add(add(parts[0], parts[1]), add(parts[2], parts[3]))
            result = [a.copy() for a in parts[0]]
            for part in parts[1:]:
                result = add(result, part)
            if mutate:
                parts[0][0][0, 1] = 42
            return result
        module = types.ModuleType("mojolearn.parallel_training")
        module.ordered_sum_gradients = ordered
        with patch.dict("sys.modules", {"mojolearn.parallel_training": module}):
            return identity.LANES["ordered-gradient-sum"](
                None, np.ones((8, 8), dtype=np.float32), None, None)

    def test_ordered_gradient_lane_accepts_left_fold(self):
        self.run_ordered()

    def test_ordered_gradient_lane_detects_balanced_reduction(self):
        with self.assertRaisesRegex(AssertionError, "left fold"):
            self.run_ordered(balanced=True)

    def test_ordered_gradient_lane_detects_input_mutation(self):
        with self.assertRaisesRegex(AssertionError, "mutated"):
            self.run_ordered(mutate=True)

    def test_metric_lane_accepts_consistent_wrapper(self):
        self.run_metric()

    def test_metric_lane_detects_swapped_outputs(self):
        with self.assertRaisesRegex(AssertionError, "order/beta"):
            self.run_metric(bad_order=True)

    def test_metric_lane_detects_ignored_beta(self):
        with self.assertRaisesRegex(AssertionError, "order/beta"):
            self.run_metric(bad_beta=True)


if __name__ == "__main__":
    unittest.main()
