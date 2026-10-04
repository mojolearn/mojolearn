import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

import amd_diagnostic_build as subject


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.commit = 'a' * 40

    def wheels(self, bad_source=False, corrupt=False):
        paths = []
        for index, project in enumerate(sorted(subject.PROJECTS)):
            dist = project.replace('-', '_') + '-0.8.37.dist-info'
            members = {dist + '/METADATA': f'Name: {project}\nVersion: 0.8.37\n'.encode(),
                       dist + '/LINUX_PAYLOAD.json': json.dumps({'source_commit': 'b' * 40 if bad_source and index == 0 else self.commit}).encode()}
            if project == 'mojolearn':
                members['mojolearn/identity_columns/COMMIT'] = self.commit.encode()
            record = io.StringIO()
            writer = csv.writer(record)
            for name, data in members.items():
                sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
                writer.writerow([name, 'sha256=' + sha, len(data)])
            writer.writerow([dist + '/RECORD', '', ''])
            members[dist + '/RECORD'] = record.getvalue().encode()
            if corrupt and index == 0:
                members[dist + '/METADATA'] += b'X-Corrupt: true\n'
            path = self.root / (project.replace('-', '_') + '-0.8.37-py3-none-manylinux_2_35_x86_64.whl')
            with zipfile.ZipFile(path, 'w') as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            paths.append(path)
        return paths

    def test_exact_six_records_verified(self):
        self.assertEqual(len(subject.inventory(self.wheels(), self.commit)), 6)

    def test_wrong_source_refused(self):
        with self.assertRaisesRegex(ValueError, 'source commit'):
            subject.inventory(self.wheels(bad_source=True), self.commit)

    def test_corrupt_record_refused(self):
        with self.assertRaisesRegex(ValueError, 'RECORD mismatch'):
            subject.inventory(self.wheels(corrupt=True), self.commit)

    def test_missing_or_duplicate_project_refused(self):
        paths = self.wheels()
        for selected in (paths[:-1], paths[:-1] + paths[:1]):
            with self.assertRaisesRegex(ValueError, 'six matching'):
                subject.inventory(selected, self.commit)


class LeaseTests(unittest.TestCase):
    def run_lease(self, failure='', unknown=False, mode='rent'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / 'tools'
            tools.mkdir()
            shutil.copy(subject.ROOT / 'tools/amd_diagnostic_lease.sh', tools)
            (tools / 'runpod_pod_lib.sh').write_text('')
            (tools / 'hotaisle_vm_lib.sh').write_text('''
HA_GONE=0
HA_REFUSED=mock
ha_write_deadman() { :; }
ha_write_watchdog() { :; }
ha_rent() { printf '%s\\n' "$*" >> "$OUT/mock-events"; }
ha_ssh() {
 case "$2" in
   *'mkdir -p'*) return 0 ;;
   *'tar czf -'*) echo fetch >> "$OUT/mock-events"; mkdir -p "$TMPD/results"; echo retained > "$TMPD/results/failure.txt"; tar czf - -C "$TMPD" results ;;
   *'cat >'*) cat >/dev/null ;;
   *'sha256sum -c'*) ''' + ('return 7' if failure == 'stage' else 'return 0') + ''' ;;
   *'bash run.sh'*) ''' + ('return 9' if failure == 'body' else 'return 0') + ''' ;;
 esac
}
ha_teardown() { echo teardown >> "$OUT/mock-events"; ''' + ('return 1' if unknown else 'HA_GONE=1; return 0') + '''; }
ha_spend() { echo mock-spend; }
''')
            bundle = root / 'bundle.tgz'
            bundle.write_bytes(b'mock transport')
            out = root / 'out'
            result = subprocess.run(['bash', str(tools / 'amd_diagnostic_lease.sh'), str(bundle), str(out), mode], capture_output=True, text=True)
            events = (out / 'mock-events').read_text()
            receipt = (out / 'teardown.txt').read_text()
            return result.returncode, events, receipt, (out / 'results/failure.txt').exists()

    def test_bounded_lease_and_success_cleanup(self):
        rc, events, receipt, fetched = self.run_lease()
        self.assertEqual(rc, 0)
        self.assertIn('amd-oob-diagnostic 30 300', events)
        self.assertTrue(fetched)
        self.assertIn('destroy_confirmed=1', receipt)

    def test_stage_or_body_failure_always_fetches_and_tears_down(self):
        for phase in ('stage', 'body'):
            with self.subTest(phase=phase):
                rc, events, receipt, fetched = self.run_lease(failure=phase)
                self.assertNotEqual(rc, 0)
                self.assertTrue(fetched)
                self.assertIn('teardown', events)

    def test_uncertain_teardown_fails_successful_body(self):
        rc, events, receipt, fetched = self.run_lease(unknown=True)
        self.assertNotEqual(rc, 0)
        self.assertIn('destroy_confirmed=0', receipt)

    def test_dryrun_never_rents(self):
        rc, events, receipt, fetched = self.run_lease(mode='dry-run')
        self.assertEqual(rc, 0)
        self.assertNotIn('amd-oob-diagnostic', events)
        self.assertFalse(fetched)


if __name__ == '__main__':
    unittest.main()
