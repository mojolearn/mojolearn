"""CPU-only fixture regression: requires the standard portable-math library."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import algos_lane_check


class GlmTargetFixtureTests(unittest.TestCase):
    def test_base_target_words_match_arm_and_x86_witness(self):
        harness = algos_lane_check.load_harness()
        X, _, y = harness.fixture('base')
        target = harness._linear_pos_target(X[:2000], y[:2000])
        self.assertEqual(hashlib.sha256(target.tobytes()).hexdigest(),
                         '2ebf427c86f79940dac1de89172f1d69d6c613be619c1bfc91e8aa98d5f0774f')
        self.assertTrue((target > 0).all())
        for lane in ('x-glm-poisson', 'x-glm-gamma', 'x-glm-tweedie', 'x-glm-poisson-sw'):
            self.assertEqual(harness.LANE_REVISIONS[lane], 'portable-positive-target-exp-1')
            self.assertIn(lane, harness.NON_SIZE_REVISIONS)


if __name__ == '__main__':
    unittest.main()
