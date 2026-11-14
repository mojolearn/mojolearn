"""Root-only remote regression gate; authored without execution.

Opt in with MOJOLEARN_NEIGHBORS_EXTENDED_GATE=1 and
MOJOLEARN_EXPECT_VENDOR=cuda|hip. No GPU work occurs without both.
"""
import os
import sys
import unittest

if (os.environ.get("MOJOLEARN_NEIGHBORS_EXTENDED_GATE") != "1"
        or sys.platform != "linux"
        or os.environ.get("MOJOLEARN_EXPECT_VENDOR") not in ("cuda", "hip")):
    raise unittest.SkipTest("Explicit root-only Linux CUDA/HIP opt-in required")

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "MOJOLEARN_CPU_THREADS"):
    os.environ[_name] = "2"

import numpy as np
import mojolearn as ml


class NeighborsWeightedLargeK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ml.vendor() != os.environ["MOJOLEARN_EXPECT_VENDOR"]:
            raise RuntimeError("Native vendor does not match the explicit remote target")
        cls.mode = ml.numeric_mode()
        if cls.mode != os.environ.get("MOJOLEARN_NUMERIC_MODE"):
            raise RuntimeError("Native mode does not match the requested mode")

    def test_k_above_1024_is_refused(self):
        # The vote/selection numerics of this gate (exact-match weight rows,
        # inverse-distance winners, strided rank boundaries) are the Mojo
        # check checks/native_tests/neighbors_weighted_largek.mojo
        # (pixi run check-native-neighbors-weighted-largek-gpu).
        x = (np.arange(1056, dtype=np.int32) // 3).astype(np.float32)[:, None] / 16
        q = np.array([[0], [4]], dtype=np.float32)
        with self.assertRaisesRegex(Exception, "1024"):
            ml.NearestNeighbors(n_neighbors=1025, metric="manhattan").fit(x).kneighbors(q)

    def test_unimplemented_kd_tree_remains_explicit(self):
        with self.assertRaisesRegex(ValueError, "kd_tree"):
            ml.NearestNeighbors(algorithm="kd_tree").fit(np.ones((4, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
