import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import check_light_release as gate


class LightReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.commit = 'a' * 40
        self.manifest = {'files': {}, 'light_smoke': {'source_commit': self.commit, 'receipts': {}}}
        self.reports = {}
        for vendor, platform in [('metal', 'macosx_11_0_arm64'), ('cuda', 'manylinux_2_35_x86_64')]:
            wheel = f'mojolearn-0.8.7-py3-none-{platform}.whl'
            with zipfile.ZipFile(self.root / wheel, 'w') as z:
                z.writestr('mojolearn/identity_columns/COMMIT', self.commit)
            digest = gate.digest(self.root / wheel)
            self.manifest['files'][wheel] = digest
            self.reports[vendor] = dict(status='PASSED', scope='expanded', source_commit=self.commit,
                release_qualified=False, installed={'vendor': vendor}, wheel=wheel, wheel_sha256=digest,
                jobs=[dict(name=name, exit_code=0) for name in sorted(gate.JOBS)], expanded={'scope': 'expanded'})

    def write(self):
        for vendor, report in self.reports.items():
            name = f'light-smoke-{vendor}.json'
            p = self.root / name
            p.write_text(json.dumps(report))
            self.manifest['light_smoke']['receipts'][name] = gate.digest(p)
        (self.root / 'alpha-manifest.json').write_text(json.dumps(self.manifest))

    def test_exact_two_artifacts_pass_only_scoped_smoke(self):
        self.write()
        result = gate.check(self.root, self.commit)
        self.assertEqual(result['status'], 'PASSED_LIGHT_RELEASE')
        self.assertFalse(result['full_numerical_certification'])

    def test_explicit_platform_batch_keeps_its_full_smoke_required(self):
        removed = self.reports.pop('cuda')
        self.manifest['files'].pop(Path(removed['wheel']).name)
        (self.root / Path(removed['wheel']).name).unlink()
        self.write()
        self.assertEqual(gate.check(self.root, self.commit, 'macos')['runtime_vendors'], ['metal'])
        with self.assertRaises(ValueError):
            gate.check(self.root, self.commit)
        with self.assertRaises(ValueError):
            gate.check(self.root, self.commit, 'linux')
        self.reports['metal']['jobs'].pop()
        self.write()
        with self.assertRaisesRegex(ValueError, 'missing required smoke'):
            gate.check(self.root, self.commit, 'macos')

    def test_altered_wheel_is_rejected(self):
        self.write()
        with (self.root / self.reports['cuda']['wheel']).open('ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            gate.check(self.root, self.commit)

    def test_failures_missing_checks_and_false_claims_cannot_pass(self):
        for mutate in (
            lambda r: r.update(status='FAILED'),
            lambda r: r.update(scope='cpu-only'),
            lambda r: r.update(source_commit='b' * 40),
            lambda r: r.update(release_qualified=True),
            lambda r: r.update(installed={'vendor': 'metal'}),
            lambda r: r['jobs'].pop(),
            lambda r: r['jobs'][0].update(exit_code=124),
        ):
            original = json.loads(json.dumps(self.reports['cuda']))
            mutate(self.reports['cuda'])
            self.write()
            with self.assertRaises(ValueError):
                gate.check(self.root, self.commit)
            self.reports['cuda'] = original

    def test_receipts_are_hash_bound_and_source_pin_is_required(self):
        self.write()
        with self.assertRaises(ValueError):
            gate.check(self.root, 'b' * 40)
        (self.root / 'light-smoke-cuda.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'receipt digest'):
            gate.check(self.root, self.commit)

    # ---- THE SPLIT LINUX PACKAGES: one package per publish, each with its own vendor's receipt
    def split(self, publish, vendor):
        """A manifest publishing ONE split Linux package ('core', 'nvidia' or
        'amd'), with a receipt of `vendor` that installed the core and the
        plugin of that vendor."""
        for f in list(self.root.iterdir()):
            f.unlink()
        core = 'mojolearn-0.8.7-py3-none-manylinux_2_35_x86_64.whl'
        with zipfile.ZipFile(self.root / core, 'w') as z:
            z.writestr('mojolearn/identity_columns/COMMIT', self.commit)
            z.writestr('mojolearn-0.8.7.dist-info/gpu_plugins.json', '{}')
        plugin = {'cuda': 'mojolearn_nvidia', 'hip': 'mojolearn_amd'}[vendor] + '-0.8.7-py3-none-manylinux_2_35_x86_64.whl'
        with zipfile.ZipFile(self.root / plugin, 'w') as z:
            z.writestr(f'mojolearn/{vendor}/x/_mojolearn_knn.so', 'inert')
        digests = {w: gate.digest(self.root / w) for w in (core, plugin)}
        published = core if publish == 'core' else plugin
        (self.root / (plugin if publish == 'core' else core)).unlink()
        self.manifest = {'files': {published: digests[published]},
                         'light_smoke': {'source_commit': self.commit, 'receipts': {}}}
        self.reports = {vendor: dict(status='PASSED', scope='expanded', source_commit=self.commit,
            release_qualified=False, installed={'vendor': vendor}, wheel='/box/' + core, wheel_sha256=digests[core],
            plugins=[dict(wheel='/box/' + plugin, wheel_sha256=digests[plugin])],
            jobs=[dict(name=name, exit_code=0) for name in sorted(gate.JOBS)], expanded={'scope': 'expanded'})}
        self.write()
        return published

    def test_split_core_passes_on_either_vendors_receipt(self):
        for vendor in ('cuda', 'hip'):
            with self.subTest(vendor=vendor):
                self.split('core', vendor)
                self.assertEqual(gate.check(self.root, self.commit, 'linux')['runtime_vendors'], [vendor])

    def test_split_plugin_passes_only_on_its_own_vendors_receipt(self):
        for publish, vendor in (('nvidia', 'cuda'), ('amd', 'hip')):
            with self.subTest(publish=publish):
                self.split(publish, vendor)
                self.assertEqual(gate.check(self.root, self.commit, 'linux')['runtime_vendors'], [vendor])
                self.reports[vendor]['installed'] = {'vendor': 'cuda' if vendor == 'hip' else 'hip'}
                self.write()
                with self.assertRaisesRegex(ValueError, 'wrong or duplicate vendor'):
                    gate.check(self.root, self.commit, 'linux')

    def test_split_plugin_bytes_are_bound_to_the_receipt(self):
        published = self.split('nvidia', 'cuda')
        self.assertTrue(published.startswith('mojolearn_nvidia-'))
        self.reports['cuda']['plugins'][0]['wheel_sha256'] = 'f' * 64
        self.write()
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            gate.check(self.root, self.commit, 'linux')
        self.reports['cuda']['plugins'] = []
        self.write()
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            gate.check(self.root, self.commit, 'linux')
        self.split('nvidia', 'cuda')
        self.reports['cuda']['wheel'] = '/box/mojolearn-0.8.6-py3-none-manylinux_2_35_x86_64.whl'
        self.write()
        with self.assertRaisesRegex(ValueError, 'another version'):
            gate.check(self.root, self.commit, 'linux')

    def test_the_combined_linux_wheel_still_needs_the_nvidia_receipt(self):
        self.reports.pop('metal')
        wheel = [w for w in self.manifest['files'] if 'macosx' in w][0]
        self.manifest['files'].pop(wheel)
        (self.root / wheel).unlink()
        self.reports['cuda']['installed'] = {'vendor': 'hip'}
        self.write()
        with self.assertRaisesRegex(ValueError, 'wrong or duplicate vendor'):
            gate.check(self.root, self.commit, 'linux')

    def test_workflow_defaults_to_prepared_light_and_checks_before_both_publish_stages(self):
        import yaml
        workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / '.github/workflows/release-provenance.yml').read_text())
        inputs = workflow.get('on', workflow.get(True))['workflow_dispatch']['inputs']
        self.assertEqual(inputs['validation_profile']['default'], 'light')
        jobs = workflow['jobs']
        self.assertEqual(jobs['alpha_stage']['needs'], 'validate_inputs')
        self.assertEqual(jobs['build']['needs'], 'validate_inputs')
        self.assertIn("validation_profile != 'light'", jobs['build']['if'])
        self.assertIn('exit 2', jobs['validate_inputs']['steps'][0]['run'])
        self.assertIn("validation_profile != 'light'", jobs['cpu_certification']['if'])
        publish = jobs['publish_alpha']['if']
        self.assertIn("needs.alpha_stage.result == 'success'", publish)
        self.assertIn("needs.cpu_certification.result == 'success'", publish)
        for name in ('alpha_stage', 'publish_alpha'):
            self.assertIn('tools/check_light_release.py', '\n'.join(step.get('run', '') for step in jobs[name]['steps']))

    def test_each_pypi_project_is_uploaded_by_its_own_job_and_environment(self):
        """THE SPLIT LINUX PACKAGES: mojolearn in `pypi`/`testpypi` as always;
        mojolearn-nvidia and mojolearn-amd each in `<target>-<plugin>`, after
        the core, from their own packages-dir; the light admission rechecked."""
        import yaml
        workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / '.github/workflows/release-provenance.yml').read_text())
        jobs = workflow['jobs']

        def publish_step(job):
            return next(s for s in jobs[job]['steps'] if 'gh-action-pypi-publish' in s.get('uses', ''))
        for job in ('publish', 'publish_alpha'):
            self.assertEqual(jobs[job]['environment']['name'], '${{ inputs.publish }}')
            self.assertEqual(publish_step(job)['with']['packages-dir'], 'upload/')
            self.assertIn('mv dist/mojolearn-*.whl upload/', '\n'.join(s.get('run', '') for s in jobs[job]['steps']))
        for job, core in (('publish_plugins', 'publish'), ('publish_alpha_plugins', 'publish_alpha')):
            self.assertEqual(jobs[job]['environment']['name'], '${{ inputs.publish }}')
            self.assertEqual(publish_step(job)['with']['packages-dir'], 'upload/')
            self.assertEqual(publish_step(job)['with']['skip-existing'], False)
            self.assertIn(core, jobs[job]['needs'])
            self.assertIn('fromJSON', jobs[job]['strategy']['matrix']['plugin'])
            self.assertIs(jobs[job]['strategy']['fail-fast'], False)
            self.assertEqual(jobs[job]['permissions']['id-token'], 'write')
            self.assertIn("!= '[]'", jobs[job]['if'])
        self.assertIn('tools/check_light_release.py',
                      '\n'.join(s.get('run', '') for s in jobs['publish_alpha_plugins']['steps']))
        self.assertIn("needs.alpha_stage.outputs.core == 'true'", jobs['publish_alpha']['if'])
        self.assertEqual(set(jobs['alpha_stage']['outputs']), {'linux_qualification', 'core', 'plugins'})
        # the matrix value is the plugin's name, so the environments are
        # <target>-nvidia and <target>-amd (docs/RELEASE_CHECKLIST.md 3b)
        build = '\n'.join(s.get('run', '') for s in jobs['build']['steps'])
        self.assertIn('"mojolearn_nvidia-$v_toml-"*) plugins="$plugins nvidia" ;;', build)
        self.assertIn('"mojolearn_amd-$v_toml-"*) plugins="$plugins amd" ;;', build)
        stage = '\n'.join(s.get('run', '') for s in jobs['alpha_stage']['steps'])
        self.assertIn("('mojolearn_nvidia-', 'mojolearn_amd-')", stage)


if __name__ == '__main__':
    unittest.main()
