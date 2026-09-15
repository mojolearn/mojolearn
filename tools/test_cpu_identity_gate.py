"""Negative controls for CPU gate sharding; no bindings or accelerators needed."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cpu_identity_gate_check as gate


def record(lanes=('gemm-pinned', 'kde')):
    return dict(vendor='cpu-test', commit='test', mode='identical', repeats=2,
                complete=True, fixtures={'base': 'fixture', 'odd': 'odd-fixture'},
                host=dict(column='cpu', cpu_model='test', families={
                    'binding': dict(column='cpu', sha256='abc', size=3)}),
                package={}, cells={f'{lane}/{fx}': dict(verdict='STABLE', hashes=['abc', 'abc'])
                                   for lane in lanes for fx in ('base', 'odd')})


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def column(self, value):
        path = self.root / 'column.json'
        path.write_text(json.dumps(value))
        return gate.do_column(SimpleNamespace(json=str(path), covered='gemm-pinned,kde',
                                             binding='binding', commit='test'))

    def test_complete_column_passes(self):
        self.assertEqual(self.column(record()), 0)

    def test_missing_fixture_fails_even_with_complete_flag(self):
        value = record()
        del value['cells']['kde/odd']
        self.assertEqual(self.column(value), 1)
        self.assertIn('covered cell kde/odd is missing', self.output.getvalue())

    def test_refusal_moved_empty_and_incomplete_fail(self):
        for mutation in ('REFUSED', 'MOVED', 'empty', 'incomplete'):
            value = record()
            if mutation == 'empty':
                value['cells'] = {}
            elif mutation == 'incomplete':
                value['complete'] = False
            else:
                value['cells']['kde/base']['verdict'] = mutation
            self.assertEqual(self.column(value), 1, mutation)

    def test_balanced_shards_preserve_all_lanes_once(self):
        lanes = ['kde', 'iforest', 'dbscan-weighted', 'gemm-pinned', 'new-lane']
        shards, _ = gate.shard_lanes(lanes, 3)
        self.assertCountEqual(sum(shards, []), lanes)
        self.assertEqual(len(shards), 3)
        self.assertTrue(all(shards))

    def test_orchestrator_propagates_failure_and_removes_stale_output(self):
        # Fake workers finish immediately; local tests never run compute in parallel.
        for failure in (False, True):
            out = self.root / 'out.json'
            out.write_text('stale success')
            seen = []

            def worker(cmd, stdout, stderr):
                seen.append(cmd)
                path = Path(cmd[cmd.index('--json') + 1])
                path.write_text(json.dumps(record(cmd[cmd.index('--lanes') + 1].split(','))))
                return SimpleNamespace(poll=lambda: 7 if failure else 0)

            def merge(cmd):
                self.assertFalse(out.exists())
                out.write_text('{}')
                return SimpleNamespace(returncode=0)

            args = SimpleNamespace(lanes='gemm-pinned,kde', shards=2, jobs=2,
                                   json=str(out), extra=['--repeats', '1'], heartbeat=120)
            with patch.object(gate.subprocess, 'Popen', side_effect=worker), \
                 patch.object(gate.subprocess, 'run', side_effect=merge), \
                 patch.object(gate.time, 'sleep'):
                self.assertEqual(gate.do_run_column(args), int(failure))
            self.assertEqual(len(seen), 2)
            self.assertTrue(all(cmd[-2:] == ['--repeats', '1'] for cmd in seen))

    def test_merge_rejects_machine_build_fixture_and_commit_changes(self):
        spec = importlib.util.spec_from_file_location('identity_gate_test', Path(__file__).with_name('identity_break.py'))
        identity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(identity)
        left, right = self.root / 'left.json', self.root / 'right.json'
        left.write_text(json.dumps(record(['gemm-pinned'])))
        original = record(['kde'])
        right.write_text(json.dumps(original))
        out = self.root / 'merged.json'
        identity.merge([str(left), str(right)], str(out))
        self.assertEqual(len(json.loads(out.read_text())['cells']), 4)
        for key in ('machine', 'build', 'fixture', 'commit'):
            value = copy.deepcopy(original)
            if key == 'machine':
                value['host']['cpu_model'] = 'other'
            elif key == 'build':
                value['host']['families']['binding']['sha256'] = 'different'
            elif key == 'fixture':
                value['fixtures']['base'] = 'different'
            else:
                value['commit'] = 'other'
            right.write_text(json.dumps(value))
            with self.assertRaisesRegex(SystemExit, 'REFUSING --merge'):
                identity.merge([str(left), str(right)], str(out))


if __name__ == '__main__':
    unittest.main()
