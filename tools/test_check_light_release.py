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

    def test_workflow_keeps_full_default_and_checks_light_before_both_publish_stages(self):
        import yaml
        workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / '.github/workflows/release-provenance.yml').read_text())
        inputs = workflow.get('on', workflow.get(True))['workflow_dispatch']['inputs']
        self.assertEqual(inputs['validation_profile']['default'], 'full')
        jobs = workflow['jobs']
        self.assertIn("validation_profile != 'light'", jobs['cpu_certification']['if'])
        publish = jobs['publish_alpha']['if']
        self.assertIn("needs.alpha_stage.result == 'success'", publish)
        self.assertIn("needs.cpu_certification.result == 'success'", publish)
        for name in ('alpha_stage', 'publish_alpha'):
            self.assertIn('tools/check_light_release.py', '\n'.join(step.get('run', '') for step in jobs[name]['steps']))


if __name__ == '__main__':
    unittest.main()
