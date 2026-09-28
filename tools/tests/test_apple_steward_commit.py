import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('tested_apple_steward', Path(__file__).resolve().parents[1] / 'apple_steward.py')
steward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(steward)


class SubmitCommitTests(unittest.TestCase):
    def test_speed_and_identity_request_use_full_commit(self):
        full = 'abcdef0' + '1' * 33
        for kind in ('speed', 'identity'):
            args = types.SimpleNamespace(commit='abcdef0', kind=kind, lane='fixture', no_coalesce=True)
            with patch.object(steward.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout=full+'\n')) as resolve, patch.object(steward, '_submit') as submit:
                steward.submit(args)
                resolve.assert_called_once()
                self.assertEqual(submit.call_args.args[0].commit, full)
                self.assertTrue(submit.call_args.args[1].endswith(full[:10]))

    def test_unknown_commit_never_ships(self):
        args = types.SimpleNamespace(commit='abcdef0', kind='speed', lane='fixture', no_coalesce=True)
        with patch.object(steward.subprocess, 'run', return_value=types.SimpleNamespace(returncode=1, stdout='')), patch.object(steward, '_submit') as submit:
            with self.assertRaises(SystemExit):
                steward.submit(args)
            submit.assert_not_called()

    def test_identity_coalescing_uses_same_full_commit(self):
        full = 'abcdef0' + '2' * 33
        args = types.SimpleNamespace(commit='abcdef0', kind='identity', lane='fixture', no_coalesce=False, sabotage=None)
        with patch.object(steward.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout=full+'\n')), patch.object(steward, '_coalesce_take', return_value=({}, [], [])) as coalesce, patch.object(steward, '_submit') as submit:
            steward.submit(args)
            self.assertEqual(coalesce.call_args.args[1], full)
            self.assertEqual(submit.call_args.args[0].commit, full)


if __name__ == '__main__':
    unittest.main()
