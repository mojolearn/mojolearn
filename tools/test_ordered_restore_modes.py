"""Fixture policy tests; no package import, fake native code or GPU required."""
import unittest
from unittest.mock import Mock

from ordered_rmse_surface_check import prepare_restore_default


class RestoreModeCoverage(unittest.TestCase):
    def run_plan(self, selected, installed, expected):
        setter = Mock()
        result = prepare_restore_default('identical', selected, installed, setter, lambda: expected)
        setter.assert_called_once_with(expected)
        return result

    def test_identical_only_still_loads_saved_mode(self):
        row = self.run_plan(['identical'], ['identical'], 'identical')
        self.assertEqual(row['alternate_default_contrast'],
                         {'status': 'NOT_EXERCISED', 'reason': 'alternate_tier_not_installed'})

    def test_requested_but_absent_alternate_not_claimed(self):
        row = self.run_plan(['identical', 'fast'], ['identical'], 'identical')
        self.assertEqual(row['alternate_default_contrast']['status'], 'NOT_EXERCISED')

    def test_installed_unselected_alternate_not_exercised(self):
        row = self.run_plan(['identical'], ['identical', 'fast'], 'identical')
        self.assertEqual(row['alternate_default_contrast']['reason'], 'alternate_tier_not_selected')

    def test_selected_installed_alternate_exercised(self):
        row = self.run_plan(['identical', 'fast'], ['identical', 'fast'], 'fast')
        self.assertEqual(row['alternate_default_contrast'], {'status': 'PASS', 'reason': None})

    def test_bad_selection_or_missing_current_tier_refused(self):
        for selected, installed in (([], ['identical']), (['fast'], ['identical']),
                (['identical', 'invalid'], ['identical']), (['identical'], ['fast']),
                (['identical', 'identical'], ['identical'])):
            with self.subTest(selected=selected, installed=installed), self.assertRaises(AssertionError):
                prepare_restore_default('identical', selected, installed, Mock(), lambda: 'identical')

    def test_present_alternate_load_error_not_downgraded(self):
        with self.assertRaisesRegex(ImportError, 'corrupt image'):
            prepare_restore_default('identical', ['identical', 'fast'], ['identical', 'fast'],
                                    Mock(side_effect=ImportError('corrupt image')), lambda: 'fast')

    def test_wrong_default_readback_refused(self):
        with self.assertRaisesRegex(AssertionError, 'readback'):
            prepare_restore_default('identical', ['identical', 'fast'], ['identical', 'fast'],
                                    Mock(), lambda: 'identical')


if __name__ == '__main__':
    unittest.main()
