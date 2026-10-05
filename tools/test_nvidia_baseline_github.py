import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ptxgh', ROOT / 'tools/nvidia_baseline_github.py')
gh = importlib.util.module_from_spec(spec); spec.loader.exec_module(gh)
SHA = '8ae4346b1fba67dce59f5a61da0b73c5febd0644'


class Dispatcher(unittest.TestCase):
    def test_source_and_tooling_refs_separate(self):
        cmd = gh.command(SHA, 'feat/ptx-github-build', 2, 'test-baseline')
        self.assertIn('commit=' + SHA, cmd)
        self.assertEqual(cmd[cmd.index('--ref') + 1], 'feat/ptx-github-build')
        self.assertIn('code_format=ptx-baseline', cmd)
        self.assertIn('release-linux-build.yml', cmd)

    def test_invalid_inputs(self):
        for sha, ref, jobs, tag in [('main', 'main', 1, 'x'), (SHA, 'x;echo', 1, 'x'),
                                     (SHA, '../x', 1, 'x'), (SHA, 'main', 3, 'x'),
                                     (SHA, 'main', 1, 'x\ncommand')]:
            with self.subTest(sha=sha, ref=ref, jobs=jobs, tag=tag):
                with self.assertRaises(ValueError): gh.command(sha, ref, jobs, tag)

    def test_dry_run_no_remote_action(self):
        argv = ['ptxgh', SHA, '--tooling-ref', 'feature', '--tag', 'x']
        with patch.object(sys, 'argv', argv), patch.object(gh.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gh.main(), 0); run.assert_not_called()

    def test_explicit_dispatch_preserves_exit(self):
        argv = ['ptxgh', SHA, '--tooling-ref', 'feature', '--tag', 'x', '--dispatch']
        with patch.object(sys, 'argv', argv), patch.object(gh.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gh.main(), 7)
            self.assertEqual(run.call_args.kwargs['timeout'], 60)


class Workflow(unittest.TestCase):
    def test_isolated_jobs_and_retained_source(self):
        # Use BaseLoader so YAML1.1 does not reinterpret the GitHub 'on' key.
        import yaml
        workflow = yaml.load((ROOT / '.github/workflows/release-linux-build.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow['on']), {'workflow_dispatch'})
        self.assertEqual(workflow['on']['workflow_dispatch']['inputs']['code_format']['default'], 'native')
        jobs = workflow['jobs']
        self.assertEqual(set(jobs), {'plan', 'shard', 'assemble', 'baseline'})
        for name in ['plan', 'shard', 'assemble']:
            self.assertIn("inputs.code_format != 'ptx-baseline'", jobs[name]['if'])
        self.assertEqual(jobs['baseline']['if'], "inputs.code_format == 'ptx-baseline'")
        self.assertEqual(jobs['baseline']['timeout-minutes'], '150')
        self.assertEqual(workflow['permissions'], {'contents': 'read'})
        self.assertIn('@sha256:', workflow['env']['IMAGE'])
        steps = jobs['baseline']['steps']
        checkouts = [s['with'] for s in steps if s.get('uses', '').startswith('actions/checkout@')]
        self.assertEqual([s['path'] for s in checkouts], ['tooling', 'src'])
        self.assertEqual(checkouts[0]['sparse-checkout'], 'tools')
        self.assertEqual(checkouts[1]['ref'], '${{ inputs.commit }}')
        self.assertEqual(checkouts[1]['persist-credentials'], 'false')
        build = next(s for s in steps if s.get('name') == 'Build experimental baseline on CPU')
        self.assertEqual(build['env']['SOURCE_COMMIT'], '${{ inputs.commit }}')
        self.assertEqual(build['env']['TOOLING_COMMIT'], '${{ github.sha }}')
        self.assertNotIn('--gpus', build['run'])
        self.assertIn('timeout -k 30 7200 docker run', build['run'])
        upload = next(s for s in steps if s.get('uses', '').startswith('actions/upload-artifact@'))
        self.assertEqual(upload['if'], 'always()')
        self.assertEqual(upload['with']['include-hidden-files'], 'true')
        self.assertEqual(upload['with']['path'], '${{ runner.temp }}/ptx-artifact')
        retention = next(s for s in steps if s.get('name') == 'Retain logs and build artifacts')
        self.assertEqual(retention['if'], 'always()')
        self.assertEqual(build['id'], 'baseline_build')
        self.assertEqual(retention['env']['BUILD_OUTCOME'], '${{ steps.baseline_build.outcome }}')
        self.assertIn('stage_ptx_artifact.py', retention['run'])
        self.assertIn('--require-complete', retention['run'])
        # Parse every newly introduced shell block before any remote dispatch.
        for step in steps:
            if 'run' in step:
                subprocess.run(['bash', '-n'], input=step['run'], text=True, check=True)

    def test_frozen_body_cpu_guards(self):
        path = ROOT / 'tools/gha_nvidia_baseline_box.sh'
        text = path.read_text()
        subprocess.run(['bash', '-n', str(path)], check=True)
        self.assertIn('git diff --quiet HEAD --', text)
        self.assertIn('git -C /tooling diff --quiet HEAD --', text)
        self.assertIn("assert not visible_devices()", text)
        self.assertIn('bash tools/nvidia_baseline_cpu_box.sh "$SOURCE_COMMIT"', text)
        self.assertNotIn('cp /tooling/', text)
        self.assertIn('install --locked -e default', text)
        self.assertIn('6000 "$BUILD_JOBS"', text)
        self.assertIn('source_tree=', text)
        self.assertIn('tooling_commit=', text)


if __name__ == '__main__': unittest.main()
