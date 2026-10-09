"""tools/neural_fixtures.py: a fake manifest round-trips, a sha mismatch refuses, a missing set
or entry refuses as infrastructure (never a torch fallback). unittest + numpy; no torch."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _nf():
    spec = importlib.util.spec_from_file_location("neural_fixtures_t", os.path.join(HERE, "neural_fixtures.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class NeuralFixturesTest(unittest.TestCase):
    def setUp(self):
        self.NF = _nf()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        rng = np.random.default_rng(0)
        self.x = rng.standard_normal((4, 8, 6, 6)).astype(np.float32)
        self.dy = rng.standard_normal((4, 8, 6, 6)).astype(np.float32)
        self.state = {"weight": rng.standard_normal((8, 8, 3, 3)).astype(np.float32),
                      "bias": rng.standard_normal(8).astype(np.float32),
                      "bn.num_batches_tracked": np.array(0, dtype=np.int64)}
        ent = self.NF.write_entry(self.dir, "conv2d", "synthetic", "nb4", task="conv2d",
                                  params={"in_channels": 8}, smoke=True,
                                  x=self.x, dy=self.dy, state=self.state)
        self.NF.write_manifest(self.dir, [ent], torch_version="fake", seed=7)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        fx = self.NF.load_for_lane(self.dir, "conv2d", {"task": "conv2d"}, "synthetic", {"_cap": 64})
        self.assertTrue(np.array_equal(fx["x"], self.x))
        self.assertEqual(fx["x"].dtype, np.float32)
        self.assertEqual(sorted(fx["state"]), sorted(self.state))
        for k, v in self.state.items():
            self.assertTrue(np.array_equal(fx["state"][k], v))
            self.assertEqual(fx["state"][k].dtype, v.dtype)
        self.assertTrue(np.array_equal(self.NF.take_dy(fx, self.dy.shape), self.dy))
        with self.assertRaises(self.NF.FixtureMismatch):
            self.NF.take_dy(fx, (1, 2))
        prov = self.NF.provenance(fx)
        self.assertEqual(prov["entry"], "conv2d/synthetic/nb4")
        self.assertEqual(prov["generated_with_torch"], "fake")
        self.assertEqual(self.NF.verify(self.dir), [])
        man = json.load(open(os.path.join(self.dir, self.NF.MANIFEST)))
        self.assertEqual(man["schema"], self.NF.SCHEMA)
        rec = man["entries"]["conv2d/synthetic/nb4"]["files"]["x"]
        self.assertEqual(rec["shape"], [4, 8, 6, 6])
        self.assertEqual(rec["dtype"], "float32")

    def test_sha_mismatch_refuses(self):
        path = os.path.join(self.dir, "conv2d/synthetic/nb4/state/bias.npy")
        a = np.load(path)
        a[0] += 1.0
        np.save(path, a)
        self.assertTrue(self.NF.verify(self.dir))
        with self.assertRaises(self.NF.FixtureMismatch) as cm:
            self.NF.load(self.dir, "conv2d", "synthetic", "nb4")
        self.assertTrue(getattr(cm.exception, "infrastructure", False))
        self.assertIn("sha256", str(cm.exception))

    def test_missing_refuses_as_infrastructure(self):
        with self.assertRaises(self.NF.FixturesMissing) as cm:
            self.NF.load(self.dir, "conv2d", "synthetic", "nb64")      # no full-batch entry
        self.assertIn("neural fixtures missing: run tools/neural_fixtures.py generate", str(cm.exception))
        self.assertTrue(cm.exception.infrastructure)
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(self.NF.FixturesMissing):
                self.NF.load(empty, "conv2d", "synthetic", "nb4")
        with self.assertRaises(self.NF.FixturesMissing):
            self.NF.load_for_lane(None, "conv2d", {"task": "conv2d"}, "synthetic", {})

    def test_variant_and_default_dir(self):
        self.assertEqual(self.NF.variant("conv2d", {}), "nb64")
        self.assertEqual(self.NF.variant("conv2d", {"_cap": 10}), "nb4")
        self.assertEqual(self.NF.variant("gcn", {"X": np.zeros((5, 3))}), "n5")
        old = os.environ.pop(self.NF.ENV_DIR, None)
        try:
            self.assertEqual(self.NF.default_dir("/r/algos-data/rows-full/"),
                             "/r/algos-data/neural-fixtures")
        finally:
            if old is not None:
                os.environ[self.NF.ENV_DIR] = old

    def test_harness_ours_arm_never_imports_torch(self):
        src = open(os.path.join(HERE, "bench_board_algos.py")).read()
        start = src.index("def _build_layer(")
        ours = src[start:src.index("    import torch\n", start)]
        self.assertNotIn("torch.", ours.replace("torch's", ""))
        self.assertNotIn("import torch", ours)
        self.assertNotIn("import torch", open(os.path.join(HERE, "neural_fixtures.py")).read()
                         .split("def generate(")[0])


if __name__ == "__main__":
    sys.exit(unittest.main())
