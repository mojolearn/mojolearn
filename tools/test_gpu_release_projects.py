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

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML not installed: these tests read the GitHub workflow YAML")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gpu_release_projects_tested', ROOT / 'tools/gpu_release_projects.py')
projects = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projects)


class ProjectClassification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def wheel(self, profile='core-linux', *, version='1.2.3', name=None, tag='py3-none-manylinux_2_35_x86_64',
              split_marker=True, gpu_member=None, meta_marker=True):
        gp = projects.gpu_plugins
        if profile == 'core-linux':
            row = dict(wheel_name=gp.CORE_WHEEL_NAME, distribution=gp.CORE_DISTRIBUTION)
        elif profile in ('meta-linux', 'macos'):
            row = dict(wheel_name=gp.META_WHEEL_NAME, distribution=gp.META_DISTRIBUTION)
            if profile == 'macos' and tag == 'py3-none-manylinux_2_35_x86_64':
                tag = 'py3-none-macosx_13_0_arm64'
        else:
            row = gp.package(profile)
        path = self.root / f"{row['wheel_name']}-{version}-{tag}.whl"
        info = f"{row['wheel_name']}-{version}.dist-info"
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr(f"{info}/METADATA",
                       f"Metadata-Version: 2.4\nName: {name or row['distribution']}\nVersion: {version}\n")
            if profile == 'core-linux' and split_marker:
                z.writestr(f"{info}/{gp.CORE_MARKER}", json.dumps(gp.core_marker(version)))
            if profile == 'meta-linux' and meta_marker:
                z.writestr(f"{info}/{gp.PLUGIN_MARKER}", json.dumps(gp.meta_marker(version)))
            if gpu_member:
                z.writestr(gpu_member, b'kernel')
        return path

    def test_core_and_meta_wheel_names(self):
        self.assertEqual(projects.wheel_prefix('core'), 'mojolearn_core')
        self.assertEqual(projects.wheel_prefix('core-linux'), 'mojolearn_core')
        self.assertEqual(projects.wheel_prefix('meta-linux'), 'mojolearn')
        self.assertTrue(self.wheel().name.startswith('mojolearn_core-1.2.3-'))
        self.assertTrue(self.wheel('meta-linux').name.startswith('mojolearn-1.2.3-'))

    def test_prepared_linux_core_cannot_bypass_permanent_split(self):
        with self.assertRaisesRegex(ValueError, 'split-package marker'):
            projects.classify([self.wheel(split_marker=False)])
        for path in ('mojolearn/cuda/sm_89/x.so', 'mojolearn/cuda_native/sm_89/x.so',
                     'mojolearn/hip_native/gfx942/x.so', 'mojolearn/cuda_ptx/sm_80/x.so',
                     'mojolearn/_mojolearn_rf.so'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'GPU payload member'):
                projects.classify([self.wheel(gpu_member=path)])
        result = projects.classify([self.wheel(gpu_member='mojolearn/host/forest.so')])
        self.assertTrue(result['core'])
        self.assertFalse(result['mojolearn'])
        with self.assertRaisesRegex(ValueError, 'Linux wheels only'):
            projects.classify([self.wheel(tag='py3-none-macosx_13_0_arm64', split_marker=False)])

    def test_linux_meta_wheel_is_file_free_and_marked(self):
        result = projects.classify([self.wheel('meta-linux')])
        self.assertTrue(result['mojolearn'])
        self.assertFalse(result['core'])
        self.assertEqual(projects.gpu_plugins.meta_marker('1.2.3')['requires'],
                         ['mojolearn-core==1.2.3', 'mojolearn-nvidia==1.2.3', 'mojolearn-amd==1.2.3'])
        with self.assertRaisesRegex(ValueError, 'aggregate marker'):
            projects.classify([self.wheel('meta-linux', meta_marker=False)])
        with self.assertRaisesRegex(ValueError, 'carries files'):
            projects.classify([self.wheel('meta-linux', gpu_member='mojolearn/__init__.py')])
        # the old combined core (a Linux mojolearn wheel with files and the core marker) is no meta wheel
        old = self.root / 'mojolearn-1.2.3-py3-none-manylinux_2_35_x86_64.whl'
        with zipfile.ZipFile(old, 'w') as z:
            z.writestr('mojolearn-1.2.3.dist-info/METADATA', 'Metadata-Version: 2.4\nName: mojolearn\nVersion: 1.2.3\n')
            z.writestr(f'mojolearn-1.2.3.dist-info/{projects.gpu_plugins.CORE_MARKER}',
                       json.dumps(projects.gpu_plugins.core_marker('1.2.3')))
        with self.assertRaisesRegex(ValueError, 'aggregate marker'):
            projects.classify([old])

    def test_complete_default_and_exact_github_outputs(self):
        wheels = ([self.wheel(), self.wheel('meta-linux')]
                  + [self.wheel(r['profile']) for r in projects.gpu_plugins.distribution_rows()])
        result = projects.classify(wheels, version='1.2.3', require_complete=True)
        self.assertTrue(result['core'])
        self.assertTrue(result['mojolearn'])
        self.assertEqual(result['plugins'], ['amd', 'nvidia'])
        self.assertEqual(result['payloads'], ['nvidia-ptx'])
        parsed = {key: json.loads(value) for key, value in
                  (line.split('=', 1) for line in projects.github_outputs(result).splitlines())}
        self.assertEqual(list(parsed), ['core', 'mojolearn', 'plugins', 'payloads'])
        self.assertEqual(parsed['payloads'], result['payloads'])
        self.assertIs(parsed['mojolearn'], True)
        output = self.root / 'output'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            self.assertEqual(projects.main([str(self.root), '--require-complete', '--github-output']), 0)
        self.assertEqual(output.read_text(), projects.github_outputs(result))

    def test_complete_needs_the_core_and_the_linux_meta_wheel(self):
        packages = [self.wheel(r['profile']) for r in projects.gpu_plugins.distribution_rows()]
        core, meta = self.wheel(), self.wheel('meta-linux')
        for subset in ([core] + packages, [meta] + packages, [core, self.wheel('macos')] + packages):
            with self.subTest(subset=[p.name for p in subset]), \
                    self.assertRaisesRegex(ValueError, 'complete Linux release'):
                projects.classify(subset, require_complete=True)

    def test_partial_payload_and_aggregate_dispatches(self):
        for profile, role in (('amd', 'plugins'), ('nvidia', 'plugins'), ('nvidia-ptx', 'payloads')):
            wheel = self.wheel(profile)
            result = projects.classify([wheel])
            self.assertFalse(result['core'])
            self.assertFalse(result['mojolearn'])
            self.assertEqual(result[role], [profile])
            with self.assertRaisesRegex(ValueError, 'complete Linux release'):
                projects.classify([wheel], require_complete=True)

    def test_macos_only_and_multiple_core_platforms(self):
        result = projects.classify([self.wheel('macos', gpu_member='mojolearn/_mojolearn_rf.so')])
        self.assertTrue(result['mojolearn'])
        self.assertFalse(result['core'])
        result = projects.classify([self.wheel('macos'), self.wheel(), self.wheel('meta-linux')])
        self.assertTrue(result['core'])
        self.assertTrue(result['mojolearn'])
        self.assertEqual(result['plugins'], [])
        self.assertEqual(result['payloads'], [])

    def test_retired_unknown_empty_duplicate_and_mixed_versions_fail(self):
        # The old experimental PTX80 name remains retired; the normal PTX
        # dependency uses mojolearn-nvidia-ptx.
        retired = self.root / 'mojolearn_nvidia_ptx80-1.2.3-py3-none-manylinux_2_35_x86_64.whl'
        retired.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'unknown GPU release project'):
            projects.classify([retired])
        with self.assertRaisesRegex(ValueError, 'unknown GPU release profile'):
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
        # PyPI's per-file limit, 100 MiB for every project: no larger allowance was granted
        # (0.8.37's 134.6 MiB NVIDIA wheel could not publish; gpu_plugins.wheel_size_limit, 66664748c)
        self.assertEqual(projects.gpu_plugins.wheel_size_limit('mojolearn-nvidia'), 100 * 1024**2)
        self.assertEqual(projects.gpu_plugins.wheel_size_limit('mojolearn-amd'), 100 * 1024**2)
        self.assertEqual(projects.gpu_plugins.wheel_size_limit('mojolearn'), 100 * 1024**2)
        self.assertEqual(projects.gpu_plugins.wheel_size_limit('mojolearn-core'), 100 * 1024**2)
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
                  aggregate_result='skipped', core_result='success', core='true', mojolearn='true', cpu=None,
                  cancelled=False):
        stage = 'alpha_stage' if alpha else 'build'
        core_job = 'publish_alpha_core' if alpha else 'publish_core'
        payload_job = 'publish_alpha_payloads' if alpha else 'publish_payloads'
        aggregate_job = 'publish_alpha_plugins' if alpha else 'publish_plugins'
        outputs = dict(core=core, linux_core=core, mojolearn=mojolearn, plugins=plugins, payloads=payloads,
                       linux_plugins=plugins, linux_payloads=payloads)
        needs = SimpleNamespace(**{
            stage: SimpleNamespace(result='success', outputs=SimpleNamespace(**outputs)),
            'cpu_certification': SimpleNamespace(result=cpu or ('skipped' if alpha else 'success')),
            core_job: SimpleNamespace(result=core_result),
            payload_job: SimpleNamespace(result=payload_result),
            aggregate_job: SimpleNamespace(result=aggregate_result)})
        inputs = SimpleNamespace(publish='pypi', alpha_candidate_tag='candidate' if alpha else '',
                                 validation_profile='light' if alpha else 'full')
        expr = self.jobs[job]['if'].replace('&&', ' and ').replace('||', ' or ')
        expr = expr.replace('!cancelled()', 'not cancelled()')
        return eval('(' + expr + ')', {'__builtins__': {}}, dict(needs=needs, inputs=inputs,
                    always=lambda: True, cancelled=lambda: cancelled))

    GRAPHS = (('publish_core', 'publish_payloads', 'publish_plugins', 'publish'),
              ('publish_alpha_core', 'publish_alpha_payloads', 'publish_alpha_plugins', 'publish_alpha'))

    def test_both_graphs_enforce_core_then_payload_then_aggregate_then_mojolearn(self):
        for core, payload, aggregate, meta in self.GRAPHS:
            for later in (payload, aggregate, meta):
                self.assertIn(core, self.jobs[later]['needs'])
            self.assertIn(payload, self.jobs[aggregate]['needs'])
            self.assertIn(payload, self.jobs[meta]['needs'])
            self.assertIn(aggregate, self.jobs[meta]['needs'])
            for earlier, later in ((payload, core), (aggregate, core), (meta, core), (meta, payload),
                                   (meta, aggregate), (aggregate, payload)):
                self.assertNotIn(earlier, self.jobs[later]['needs'])
            self.assertEqual(self.jobs[core]['environment']['name'], '${{ inputs.publish }}-core')
            self.assertEqual(self.jobs[meta]['environment']['name'], '${{ inputs.publish }}')
            core_run = '\n'.join(step.get('run', '') for step in self.jobs[core]['steps'])
            self.assertIn('gpu_release_projects.py --wheel-prefix core', core_run)
            self.assertIn('mv dist/"${WHEEL_PREFIX}"-*.whl upload/', core_run)
            for name in (payload, aggregate):
                self.assertEqual(self.jobs[name]['environment']['name'], '${{ inputs.publish }}-${{ matrix.plugin }}')
                steps = self.jobs[name]['steps']
                run = '\n'.join(step.get('run', '') for step in steps)
                self.assertIn('gpu_release_projects.py --wheel-prefix "$PLUGIN"', run)
                self.assertIn('mv dist/"${WHEEL_PREFIX}"-*.whl upload/', run)
                self.assertIn('gpu_release_projects.py dist', run)
            meta_run = '\n'.join(step.get('run', '') for step in self.jobs[meta]['steps'])
            # mojolearn-* never matches the core's mojolearn_core-* or a vendor's mojolearn_<vendor>-* file
            self.assertIn('mv dist/mojolearn-*.whl upload/', meta_run)
            for name in (aggregate, meta):
                steps = self.jobs[name]['steps']
                gate = next(i for i, step in enumerate(steps) if '--plugins-on-index' in step.get('run', ''))
                upload = next(i for i, step in enumerate(steps) if 'gh-action-pypi-publish' in step.get('uses', ''))
                self.assertLess(gate, upload)

    def test_failed_or_unexpectedly_skipped_core_never_allows_vendors_or_mojolearn(self):
        for alpha, (core, payload, aggregate, meta) in ((False, self.GRAPHS[0]), (True, self.GRAPHS[1])):
            for result in ('failure', 'cancelled', 'skipped'):
                with self.subTest(alpha=alpha, result=result):
                    self.assertFalse(self.condition(payload, alpha=alpha, payloads='["nvidia-ptx"]',
                                                    core_result=result))
                    self.assertFalse(self.condition(aggregate, alpha=alpha, plugins='["nvidia"]',
                                                    core_result=result))
                    self.assertFalse(self.condition(meta, alpha=alpha, core_result=result))
            # a dispatch without a core wheel (core output false) does not wait for it
            self.assertTrue(self.condition(aggregate, alpha=alpha, plugins='["amd"]', core='false',
                                           core_result='skipped'))
            self.assertTrue(self.condition(core, alpha=alpha))
            self.assertFalse(self.condition(core, alpha=alpha, core='false'))

    def test_failed_or_unexpectedly_skipped_payload_never_allows_mojolearn(self):
        for alpha, (_, _, aggregate, meta) in ((False, self.GRAPHS[0]), (True, self.GRAPHS[1])):
            for result in ('failure', 'cancelled', 'skipped'):
                with self.subTest(alpha=alpha, result=result):
                    self.assertFalse(self.condition(meta, alpha=alpha, payloads='["nvidia-ptx"]',
                                                    plugins='["nvidia"]', payload_result=result,
                                                    aggregate_result='success'))
                    self.assertFalse(self.condition(aggregate, alpha=alpha, payloads='["nvidia-ptx"]',
                                                    plugins='["nvidia"]', payload_result=result))
                    self.assertFalse(self.condition(meta, alpha=alpha, payloads='["nvidia-ptx"]',
                                                    plugins='["nvidia"]', payload_result='success',
                                                    aggregate_result=result))
            self.assertTrue(self.condition(meta, alpha=alpha, payloads='["nvidia-ptx"]', plugins='["nvidia"]',
                                           payload_result='success', aggregate_result='success'))
            self.assertFalse(self.condition(meta, alpha=alpha, cancelled=True))

    def test_empty_matrices_allow_macos_and_independent_dispatches(self):
        self.assertTrue(self.condition('publish'))
        self.assertTrue(self.condition('publish', core='false', core_result='skipped'))
        self.assertTrue(self.condition('publish_alpha', alpha=True))
        self.assertTrue(self.condition('publish_alpha_plugins', alpha=True, plugins='["nvidia"]'))
        self.assertFalse(self.condition('publish_alpha', alpha=True, cpu='failure'))
        self.assertFalse(self.condition('publish_alpha_core', alpha=True, cpu='failure'))
        # an alpha candidate without a mojolearn wheel publishes no mojolearn project
        self.assertFalse(self.condition('publish_alpha', alpha=True, mojolearn='false'))

    def test_registry_classification_runs_at_both_staging_boundaries(self):
        full = '\n'.join(step.get('run', '') for step in self.jobs['build']['steps'])
        self.assertIn('gpu_release_projects.py "$DIR" --version "$v_toml" --require-complete --github-output', full)
        alpha = '\n'.join(step.get('run', '') for step in self.jobs['alpha_stage']['steps'])
        self.assertIn('gpu_release_projects.py dist --github-output', alpha)
        for key in ('linux_core', 'linux_plugins', 'linux_payloads'):
            self.assertIn(key, self.jobs['build']['outputs'])
        for key in ('core', 'mojolearn', 'plugins', 'payloads'):
            self.assertIn(key, self.jobs['alpha_stage']['outputs'])

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
