"""Release command orchestration only: fake native work, real shell failures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class InstalledSupplementTests(unittest.TestCase):
    def run_gate(self, failure=''):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / 'tools'
            tools.mkdir()
            shutil.copyfile(Path(__file__).with_name('release_installed_checks.sh'),
                            tools / 'release_installed_checks.sh')
            (tools / 'linux_surface_qualification.sh').write_text('exit 0\n')
            (root / 'commit.txt').write_text('a' * 40 + '\n')
            out = root / 'output'
            venv = out / 'venv/bin'
            venv.mkdir(parents=True)
            python = venv / 'python'
            python.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys
args = sys.argv[1:]
if args[0] == '-m':
    lanes = args[args.index('--lanes') + 1]
    stage = 'umap-column' if lanes == 'umap' else 'earlier-column'
elif 'record' in args:
    stage = 'umap-record'
elif 'check' in args:
    stage = 'umap-cpu-replay'
else:
    stage = 'verifier'
with open(os.environ['CALL_LOG'], 'a') as f:
    f.write(json.dumps(dict(stage=stage, args=args, cwd=os.getcwd(),
                           pythonpath=os.environ.get('PYTHONPATH'),
                           gate_commit=os.environ.get('MOJOLEARN_GATE_COMMIT'))) + '\\n')
sys.exit(17 if stage == os.environ.get('FAIL_STAGE') else 0)
''')
            python.chmod(0o755)
            commands = root / 'commands'
            commands.mkdir()
            # Replace platform-specific affinity/timeout/path commands. The
            # release script still executes env and its own fail-fast shell.
            for name, body in {
                'python3': 'printf "0,1\\n"',
                'taskset': 'exit 0',
                'realpath': '[ "$1" != -m ] || shift\nprintf "%s\\n" "$1"',
                'timeout': 'shift 3\nexec "$@"',
            }.items():
                path = commands / name
                path.write_text('#!/bin/sh\n' + body + '\n')
                path.chmod(0o755)
            log = root / 'calls.jsonl'
            env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
                       CALL_LOG=str(log), FAIL_STAGE=failure, PYTHONPATH='/must-be-removed')
            result = subprocess.run(['bash', str(tools / 'release_installed_checks.sh'),
                'qualify-release-linux3', str(root / 'wheel.whl'), 'digest', 'cuda',
                str(out), str(root / 'proofs'), 'sm_90a'], env=env,
                capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return result, (out / 'exit_code').read_text().strip(), calls

    def test_captures_then_replays_installed_umap_before_success(self):
        result, marker, calls = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(marker, '0')
        self.assertEqual([c['stage'] for c in calls], [
            'earlier-column', 'earlier-column', 'umap-column',
            'umap-record', 'umap-cpu-replay', 'verifier'])
        column, record, replay = calls[2:5]
        self.assertEqual(column['args'][column['args'].index('--repeats') + 1], '2')
        self.assertEqual(column['args'][column['args'].index('--require-backend') + 1], 'cuda')
        for call in (column, record, replay):
            self.assertIsNone(call['pythonpath'])
            self.assertTrue(call['cwd'].endswith('/output'))
            self.assertEqual(call['gate_commit'], 'a' * 40)
        for call in (record, replay):
            self.assertEqual(call['args'][call['args'].index('--package-root') + 1], '')
        self.assertTrue(replay['args'][replay['args'].index('--gpu-column') + 1]
                        .endswith('/umap-column.json'))

    def test_each_supplement_failure_invalidates_surface_success(self):
        for stage in ('earlier-column', 'umap-column', 'umap-record',
                      'umap-cpu-replay', 'verifier'):
            with self.subTest(stage=stage):
                result, marker, calls = self.run_gate(stage)
                self.assertEqual(result.returncode, 17, result.stderr)
                self.assertEqual(marker, '1')
                self.assertEqual(calls[-1]['stage'], stage)


if __name__ == '__main__':
    unittest.main()
