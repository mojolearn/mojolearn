"""CPU-only receipt checks; no compiled bindings or model execution."""
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_board_state import canonical_hash, model_receipt, scored_receipt


class ReceiptTests(unittest.TestCase):
    def test_arrays_bind_shape_dtype_names_and_bytes(self):
        a = np.arange(4, dtype=np.float32).reshape(2, 2)
        original = canonical_hash({"a": a})
        for changed in ({"a": a.reshape(4)}, {"a": a.view(np.int32)},
                        {"b": a}, {"a": a + 1}):
            self.assertNotEqual(original, canonical_hash(changed))
        self.assertEqual(original, canonical_hash({"a": a.copy(order="F")}))
        self.assertEqual(len(original), 64)

    def test_nested_training_state_and_framing(self):
        self.assertNotEqual(canonical_hash({"ab": "c"}), canonical_hash({"a": "bc"}))
        self.assertEqual(canonical_hash({"b": [1, None], "a": True}),
                         canonical_hash({"a": True, "b": [1, None]}))
        self.assertNotEqual(canonical_hash({"step": 1}), canonical_hash({"step": 2}))
        with self.assertRaises(TypeError):
            canonical_hash(np.array([object()], dtype=object))

    def test_save_hash_excludes_only_declared_device(self):
        class Model:
            __module__ = "mojolearn.testing"
            device = "amd"
            def save(self, path):
                np.savez(path, weights=np.arange(3, dtype=np.float32), device=self.device)
        model = Model()
        a = model_receipt(model)
        model.device = "nvidia"
        b = model_receipt(model)
        self.assertEqual(a["status"], "ok")
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertNotEqual(a["complete_export_sha256"], b["complete_export_sha256"])
        self.assertEqual(a["excluded_metadata"], ["device"])

    def test_unavailable_and_failed_models_never_claim_identity(self):
        class Model:
            __module__ = "mojolearn.testing"
            def save(self, path):
                raise RuntimeError("export refused")
        self.assertEqual(model_receipt(None)["status"], "unavailable")
        record = model_receipt(Model())
        self.assertEqual(record["status"], "error")
        self.assertIn("export refused", record["error"])
        self.assertNotIn("sha256", record)
        self.assertIsNone(scored_receipt(None)["output_sha256"])
        self.assertEqual(scored_receipt({})["output_status"], "unavailable")

    def test_snapshot_reads_once_and_captures_optimizer(self):
        class Model:
            __module__ = "mojolearn.testing"
            reads = 0
            step = 1
            def state_dict(self):
                self.reads += 1
                return {"weights": np.zeros(2, np.float32), "optimizer": {"step": self.step}}
        model = Model()
        a = scored_receipt({"pred": np.ones(2)}, model=model)
        self.assertEqual(model.reads, 1)
        model.step += 1
        b = scored_receipt({"pred": np.ones(2)}, model=model)
        self.assertEqual(a["output_sha256"], b["output_sha256"])
        self.assertNotEqual(a["model"]["sha256"], b["model"]["sha256"])


if __name__ == "__main__":
    unittest.main()
