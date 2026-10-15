"""CPU-only fixture regression: requires the standard portable-math library."""
import hashlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import algos_lane_check

PACKAGE = Path(__file__).resolve().parents[2] / 'python/mojolearn'
# packaging/portable_math/stage.py build() writes it here (the CPU runner and
# the wheel stage it before any test runs)
MATH_LIBRARY = PACKAGE / ('.dylibs/libMojolearnMath.dylib' if sys.platform == 'darwin'
                          else '.libs/libMojolearnMath.so')


class GlmTargetFixtureTests(unittest.TestCase):
    def test_base_target_words_match_arm_and_x86_witness(self):
        if not MATH_LIBRARY.is_file():
            self.skipTest(f'portable math library not present: {MATH_LIBRARY} '
                          '(packaging/portable_math/stage.py build())')
        harness = algos_lane_check.load_harness()
        X, _, y = harness.fixture('base')
        # The target needs only mojolearn._portable_math, not a built binding
        # set: a bare package module over the source tree lets it import
        # without the package's binding selection, and is removed afterwards.
        package = types.ModuleType('mojolearn')
        package.__path__ = [str(PACKAGE)]
        with mock.patch.dict(sys.modules, {'mojolearn': package}):
            sys.modules.pop('mojolearn._portable_math', None)
            target = harness._linear_pos_target(X[:2000], y[:2000])
        self.assertEqual(hashlib.sha256(target.tobytes()).hexdigest(),
                         '2ebf427c86f79940dac1de89172f1d69d6c613be619c1bfc91e8aa98d5f0774f')
        self.assertTrue((target > 0).all())
        for lane in ('x-glm-poisson', 'x-glm-gamma', 'x-glm-tweedie', 'x-glm-poisson-sw'):
            self.assertEqual(harness.LANE_REVISIONS[lane], 'portable-positive-target-exp-1')
            self.assertIn(lane, harness.NON_SIZE_REVISIONS)


if __name__ == '__main__':
    unittest.main()
