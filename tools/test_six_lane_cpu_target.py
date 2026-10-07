"""Metadata-only tests: no compiler, binding import, GPU, or estimator execution."""
import unittest
from unittest.mock import patch
from six_lane_ab import nvidia_build_target


class TargetSelectionTests(unittest.TestCase):
    def test_explicit_native_never_probes(self):
        hardware = {}
        with patch('six_lane_ab.subprocess.check_output') as probe:
            self.assertEqual(nvidia_build_target('sm_90', 'native', hardware), 'sm_90')
            probe.assert_not_called()
        self.assertNotIn('gpu', hardware)
        self.assertIn('not detected', hardware['accelerator_selection'])

    def test_invalid_target_refused(self):
        for arch in ['gfx942', 'sm_90;true', 'native', 'sm_', 'sm_90 --emit']:
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                nvidia_build_target(arch, 'native', {})

    def test_default_not_silently_cross_compiled(self):
        with self.assertRaises(ValueError):
            nvidia_build_target('sm_90', 'default', {})

    def test_existing_device_path_preserved(self):
        with patch('six_lane_ab.subprocess.check_output', return_value='H100, GPU-abc, 580.1, 9.0\n'):
            for kind in ['native', 'default']:
                hardware = {}
                self.assertEqual(nvidia_build_target(None, kind, hardware), 'sm_90')
                self.assertIn('gpu', hardware)

    def test_missing_probe_requires_explicit_target(self):
        with patch('six_lane_ab.subprocess.check_output', side_effect=FileNotFoundError()), self.assertRaisesRegex(ValueError, 'CPU-only'):
            nvidia_build_target(None, 'native', {})

    def test_multiple_detected_targets_refused(self):
        with patch('six_lane_ab.subprocess.check_output', return_value='H100, id1, 580, 9.0\nL40, id2, 580, 8.9\n'), self.assertRaises(ValueError):
            nvidia_build_target(None, 'native', {})


if __name__ == '__main__':
    unittest.main()
