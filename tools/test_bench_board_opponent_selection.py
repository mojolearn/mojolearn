"""Missing opponent runs must not repeat retained arms or our measurements."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('board', Path(__file__).with_name('bench_board.py'))
board = importlib.util.module_from_spec(spec)
spec.loader.exec_module(board)


class OpponentSelection(unittest.TestCase):
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
