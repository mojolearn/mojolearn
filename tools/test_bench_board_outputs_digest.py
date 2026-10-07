#!/usr/bin/env python3
"""bench_board_neural.outputs_digest and bench_board's hash fallback to it (pure Python; no race, no build)."""
import hashlib
import importlib.util
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_board_neural as neural  # noqa: E402

_spec = importlib.util.spec_from_file_location("bench_board", os.path.join(HERE, "bench_board.py"))
bb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bb)


class OutputsDigestTests(unittest.TestCase):
    def test_format_and_determinism(self):
        outs = {"losses": np.array([2.5, 2.25], dtype=np.float64), "y": np.arange(6, dtype=np.float32).reshape(2, 3)}
        d = neural.outputs_digest(outs)
        self.assertRegex(d, r"^[0-9a-f]{16}$")
        self.assertEqual(d, neural.outputs_digest({"y": outs["y"].copy(), "losses": outs["losses"].copy()}))
        h = hashlib.sha256()
        for name in ("losses", "y"):
            a = outs[name]
            h.update(("%s|%s|%s;" % (name, a.dtype.str, ",".join(str(s) for s in a.shape))).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        self.assertEqual(d, h.hexdigest()[:16])

    def test_sensitive_to_bits_dtype_shape_and_names(self):
        y = np.arange(6, dtype=np.float32).reshape(2, 3)
        base = neural.outputs_digest({"y": y})
        bumped = y.copy()
        bumped.view(np.uint32)[0, 0] ^= 1          # one ulp
        self.assertNotEqual(base, neural.outputs_digest({"y": bumped}))
        self.assertNotEqual(base, neural.outputs_digest({"y": y.reshape(3, 2)}))
        self.assertNotEqual(base, neural.outputs_digest({"y": y.astype(np.float64)}))
        self.assertNotEqual(base, neural.outputs_digest({"z": y}))
        self.assertEqual(base, neural.outputs_digest({"y": np.asfortranarray(y)}))  # C-order bytes
        self.assertIsNone(neural.outputs_digest({}))

    def test_board_hash_falls_back_to_outputs_digest(self):
        race = bb.plan_races("amd", ["identical"], ["neural"], ["lm-train-step"], neural_shape="small")[0]
        ctx = {"python": "py", "neural_driver": "drv", "vendor": "amd", "rounds": 1, "out": "/o", "round_seconds": 0}
        arm = {"ms": [12.0], "warmup_ms": 13.0, "digests": [None, None], "digest_stable": None,
               "outputs_digest": "0123456789abcdef", "status": "ok", "info": {}}
        cells = bb.classical_cells(ctx, race, {"arms": {"ours": dict(arm)}, "quality": {}})
        ours = [c for c in cells if c["arm"] == "ours"][0]
        self.assertEqual(ours["hash"], "0123456789abcdef")
        arm["digests"] = [None, "fedcba9876543210"]   # a round digest still wins
        cells = bb.classical_cells(ctx, race, {"arms": {"ours": arm}, "quality": {}})
        self.assertEqual([c for c in cells if c["arm"] == "ours"][0]["hash"], "fedcba9876543210")


if __name__ == "__main__":
    unittest.main()
