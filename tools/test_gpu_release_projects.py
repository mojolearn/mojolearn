# SPDX-License-Identifier: Apache-2.0
"""Release project classification and publishing graph regressions."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gpu_release_projects_tested', ROOT / 'tools/gpu_release_projects.py')
projects = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projects)


class ProjectClassification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def wheel(self, profile='core-linux', *, version='1.2.3', name=None, tag='py3-none-manylinux_2_35_x86_64', split_marker=True, gpu_member=None):
        row = (dict(wheel_name='mojolearn', distribution='mojolearn') if profile == 'core-linux'
               else projects.gpu_plugins.package(profile))
        path = self.root / f"{row['wheel_name']}-{version}-{tag}.whl"
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr(f"{row['wheel_name']}-{version}.dist-info/METADATA",
                       f"Metadata-Version: 2.4\nName: {name or row['distribution']}\nVersion: {version}\n")
            if profile == 'core-linux' and split_marker and 'linux' in tag:
                z.writestr(f"mojolearn-{version}.dist-info/{projects.gpu_plugins.CORE_MARKER}",
                           json.dumps(projects.gpu_plugins.core_marker(version)))
            if gpu_member:
                z.writestr(gpu_member, b'kernel')
        return path

    def test_prepared_linux_core_cannot_bypass_permanent_split(self):
        with self.assertRaisesRegex(ValueError, 'split-package marker'):
            projects.classify([self.wheel(split_marker=False)])
        for path in ('mojolearn/cuda/sm_89/x.so', 'mojolearn/cuda_native/sm_89/x.so',
                     'mojolearn/hip_native/gfx942/x.so', 'mojolearn/cuda_ptx/sm_80/x.so',
                     'mojolearn/_mojolearn_rf.so'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'GPU payload member'):
                projects.classify([self.wheel(gpu_member=path)])
        self.assertTrue(projects.classify([self.wheel(gpu_member='mojolearn/host/forest.so')])['core'])
        self.assertTrue(projects.classify([self.wheel(tag='py3-none-macosx_13_0_arm64',
                                                    split_marker=False, gpu_member='mojolearn/_mojolearn_rf.so')])['core'])

    def test_complete_default_and_exact_github_outputs(self):
        wheels = [self.wheel()] + [self.wheel(r['profile']) for r in projects.gpu_plugins.distribution_rows()]
        result = projects.classify(wheels, version='1.2.3', require_complete=True)
        self.assertTrue(result['core'])
        self.assertEqual(result['plugins'], ['amd', 'nvidia'])
        self.assertEqual(result['payloads'], [])
        parsed = {key: json.loads(value) for key, value in
                  (line.split('=', 1) for line in projects.github_outputs(result).splitlines())}
        self.assertEqual(parsed['payloads'], result['payloads'])
        output = self.root / 'output'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            self.assertEqual(projects.main([str(self.root), '--require-complete', '--github-output']), 0)
        self.assertEqual(output.read_text(), projects.github_outputs(result))

    def test_partial_payload_and_aggregate_dispatches(self):
        for profile, role in (('amd', 'plugins'), ('nvidia', 'plugins')):
            wheel = self.wheel(profile)
            result = projects.classify([wheel])
            self.assertFalse(result['core'])
            self.assertEqual(result[role], [profile])
            with self.assertRaisesRegex(ValueError, 'complete Linux release'):
                projects.classify([wheel], require_complete=True)

    def test_macos_only_and_multiple_core_platforms(self):
        mac = self.wheel(tag='py3-none-macosx_13_0_arm64')
        result = projects.classify([mac, self.wheel()])
        self.assertTrue(result['core'])
        self.assertEqual(result['plugins'], [])
        self.assertEqual(result['payloads'], [])

    def test_experimental_unknown_empty_duplicate_and_mixed_versions_fail(self):
        with self.assertRaisesRegex(ValueError, 'experimental'):
            projects.classify([self.wheel('nvidia-ptx80')])
        with self.assertRaisesRegex(ValueError, 'experimental'):
            projects.wheel_prefix('nvidia-ptx80')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            projects.classify([self.root / 'other-1.2.3-py3-none-any.whl'])
        with self.assertRaisesRegex(ValueError, 'no wheels'):
            projects.classify([])
        one = self.wheel('nvidia')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            projects.classify([one, one])
        with self.assertRaisesRegex(ValueError, 'versions disagree'):
            projects.classify([one, self.wheel(version='9.0.0')])
        with self.assertRaisesRegex(ValueError, 'expected version'):
            projects.classify([one], version='9.0.0')

    def test_vendor_budgets_and_architecture_projects_not_released(self):
        self.assertEqual(projects.gpu_plugins.wheel_size_limit('mojolearn-nvidia'), 250 * 1024**2)
        self.assertEqual(projects.gpu_plugins.wheel_size_limit('mojolearn-amd'), 100 * 1024**2)
        self.assertEqual(projects.gpu_plugins.wheel_size_limit('mojolearn'), 100 * 1024**2)
        for profile in ('nvidia-sm89', 'nvidia-sm90', 'amd-gfx942'):
            with self.assertRaisesRegex(ValueError, 'unknown'):
                projects.wheel_prefix(profile)

    def test_names_are_registry_normalized_and_metadata_bound(self):
        self.assertEqual(projects.wheel_prefix('nvidia'), 'mojolearn_nvidia')
        self.assertEqual(projects.wheel_prefix('amd'), 'mojolearn_amd')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            projects.wheel_prefix('../nvidia')
        with self.assertRaisesRegex(ValueError, 'metadata disagree'):
            projects.classify([self.wheel('nvidia', name='mojolearn-amd')])


class PublishingGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = yaml.safe_load((ROOT / '.github/workflows/release-provenance.yml').read_text())
        cls.jobs = cls.workflow['jobs']

    def condition(self, job, *, alpha=False, payloads='[]', plugins='[]', payload_result='skipped',
                  aggregate_result='skipped', cpu=None, cancelled=False):
        stage = 'alpha_stage' if alpha else 'build'
        payload_job = 'publish_alpha_payloads' if alpha else 'publish_payloads'
        aggregate_job = 'publish_alpha_plugins' if alpha else 'publish_plugins'
        outputs = dict(core='true', plugins=plugins, payloads=payloads,
                       linux_plugins=plugins, linux_payloads=payloads)
        needs = SimpleNamespace(**{
            stage: SimpleNamespace(result='success', outputs=SimpleNamespace(**outputs)),
            'cpu_certification': SimpleNamespace(result=cpu or ('skipped' if alpha else 'success')),
            payload_job: SimpleNamespace(result=payload_result),
            aggregate_job: SimpleNamespace(result=aggregate_result)})
        inputs = SimpleNamespace(publish='pypi', alpha_candidate_tag='candidate' if alpha else '',
                                 validation_profile='light' if alpha else 'full')
        expr = self.jobs[job]['if'].replace('&&', ' and ').replace('||', ' or ')
        expr = expr.replace('!cancelled()', 'not cancelled()')
        return eval('(' + expr + ')', {'__builtins__': {}}, dict(needs=needs, inputs=inputs,
                    always=lambda: True, cancelled=lambda: cancelled))

    def test_both_graphs_enforce_payload_then_aggregate_then_core(self):
        for payload, aggregate, core in (('publish_payloads', 'publish_plugins', 'publish'),
                                       ('publish_alpha_payloads', 'publish_alpha_plugins', 'publish_alpha')):
            self.assertIn(payload, self.jobs[aggregate]['needs'])
            self.assertIn(payload, self.jobs[core]['needs'])
            self.assertIn(aggregate, self.jobs[core]['needs'])
            self.assertNotIn(core, self.jobs[payload]['needs'])
            for name in (payload, aggregate):
                self.assertEqual(self.jobs[name]['environment']['name'], '${{ inputs.publish }}-${{ matrix.plugin }}')
                steps = self.jobs[name]['steps']
                run = '\n'.join(step.get('run', '') for step in steps)
                self.assertIn('gpu_release_projects.py --wheel-prefix "$PLUGIN"', run)
                self.assertIn('mv dist/"${WHEEL_PREFIX}"-*.whl upload/', run)
                self.assertIn('gpu_release_projects.py dist', run)
            for name in (aggregate, core):
                steps = self.jobs[name]['steps']
                gate = next(i for i, step in enumerate(steps) if '--plugins-on-index' in step.get('run', ''))
                upload = next(i for i, step in enumerate(steps) if 'gh-action-pypi-publish' in step.get('uses', ''))
                self.assertLess(gate, upload)

    def test_failed_or_unexpectedly_skipped_payload_never_allows_core(self):
        for alpha, core, aggregate in ((False, 'publish', 'publish_plugins'),
                                       (True, 'publish_alpha', 'publish_alpha_plugins')):
            for result in ('failure', 'cancelled', 'skipped'):
                with self.subTest(alpha=alpha, result=result):
                    self.assertFalse(self.condition(core, alpha=alpha, payloads='["nvidia-sm89"]',
                                                    plugins='["nvidia"]', payload_result=result,
                                                    aggregate_result='success'))
                    self.assertFalse(self.condition(aggregate, alpha=alpha, payloads='["nvidia-sm89"]',
                                                    plugins='["nvidia"]', payload_result=result))
            self.assertTrue(self.condition(core, alpha=alpha, payloads='["nvidia-sm89"]', plugins='["nvidia"]',
                                           payload_result='success', aggregate_result='success'))
            self.assertFalse(self.condition(core, alpha=alpha, cancelled=True))

    def test_empty_matrices_allow_macos_and_independent_dispatches(self):
        self.assertTrue(self.condition('publish'))
        self.assertTrue(self.condition('publish_alpha', alpha=True))
        self.assertTrue(self.condition('publish_alpha_plugins', alpha=True, plugins='["nvidia"]'))
        self.assertFalse(self.condition('publish_alpha', alpha=True, cpu='failure'))

    def test_registry_classification_runs_at_both_staging_boundaries(self):
        full = '\n'.join(step.get('run', '') for step in self.jobs['build']['steps'])
        self.assertIn('gpu_release_projects.py "$DIR" --version "$v_toml" --require-complete --github-output', full)
        alpha = '\n'.join(step.get('run', '') for step in self.jobs['alpha_stage']['steps'])
        self.assertIn('gpu_release_projects.py dist --github-output', alpha)
        self.assertIn('linux_payloads', self.jobs['build']['outputs'])
        self.assertIn('payloads', self.jobs['alpha_stage']['outputs'])

    def test_every_workflow_shell_block_parses(self):
        for name, job in self.jobs.items():
            for step in job.get('steps', []):
                script = step.get('run')
                if not script or step.get('shell', 'bash').startswith('python'):
                    continue
                script = re.sub(r'\$\{\{.*?\}\}', 'fixture', script, flags=re.S)
                result = subprocess.run(['bash', '-n'], input=script, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, f"{name}/{step.get('name')}: {result.stderr}")


if __name__ == '__main__':
    unittest.main()
