"""Missing opponent runs must not repeat retained arms or our measurements."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import tempfile

spec = importlib.util.spec_from_file_location('board', Path(__file__).with_name('bench_board.py'))
board = importlib.util.module_from_spec(spec)
spec.loader.exec_module(board)


class OpponentSelection(unittest.TestCase):
    def test_plan_audit_only_requires_selected_or_retained_opponents(self):
        race = dict(id='r', family='neural', lane='gemm', dataset='synthetic',
                    our_arms={'ours': 'identical'}, arms=['ours', 'torch-gpu', 'missing-lib'])
        for flags, retained, expected in [
            ({}, {}, ['ours']),
            ({}, {'torch-gpu': {}}, ['ours', 'torch-gpu']),
            ({'with_opponents': True}, {}, race['arms']),
            ({'opponents_only': True}, {}, ['torch-gpu', 'missing-lib']),
        ]:
            selected = dict(race)
            if flags.get('opponents_only'):
                selected.update(our_arms={}, arms=['torch-gpu', 'missing-lib'])
            with self.subTest(flags=flags, retained=retained), tempfile.TemporaryDirectory() as tmp:
                evidence = MagicMock()
                evidence.begin.return_value = (Path(tmp), {})
                with patch.object(board, '_load_tool', return_value=evidence), \
                     patch.object(board, 'unsupported_our_arms', return_value={}), \
                     patch.object(board, 'stored_opponents', return_value=retained), \
                     patch.object(board, 'race_data_files', return_value=[]), \
                     patch.object(board, 'race_data_sha', return_value='data'), \
                     patch.object(board, 'race_settings', return_value={}), \
                     patch.object(board, 'preflight_check', return_value={'verdict': 'MATCHED'}) as probe:
                    result = board.audit_plan(dict(out=tmp, box={}, measurement_source={},
                                                   evidence_r2=False, **flags), [selected])
                self.assertEqual(result['verdict'], 'MATCHED')
                self.assertEqual(probe.call_args.args[2], expected)

    def setUp(self):
        self.races = [
            dict(id='one', our_arms={}, opponents=['torch-gpu', 'cuml'], arms=['torch-gpu', 'cuml']),
            dict(id='two', our_arms={}, opponents=['sklearn'], arms=['sklearn']),
        ]

    def test_selects_only_requested_missing_arm_without_mutating_plan(self):
        selected = board.select_opponent_arms(self.races, 'cuml')
        self.assertEqual([r['id'] for r in selected], ['one'])
        self.assertEqual(selected[0]['arms'], ['cuml'])
        self.assertEqual(selected[0]['opponents'], ['cuml'])
        self.assertEqual(self.races[0]['opponents'], ['torch-gpu', 'cuml'])

    def test_refuses_unknown_and_own_arms(self):
        for selection in ('', 'ours', 'typo', 'cuml,typo'):
            with self.subTest(selection=selection), self.assertRaises(SystemExit):
                board.select_opponent_arms(self.races, selection)
        self.races[0]['our_arms'] = {'ours': 'identical'}
        with self.assertRaisesRegex(SystemExit, 'opponents-only'):
            board.select_opponent_arms(self.races, 'cuml')

    def test_multiple_requested_arms_keep_recipe_order(self):
        selected = board.select_opponent_arms(self.races, 'sklearn,cuml,cuml')
        self.assertEqual([r['id'] for r in selected], ['one', 'two'])
        self.assertEqual([r['arms'] for r in selected], [['cuml'], ['sklearn']])


if __name__ == '__main__':
    unittest.main()
