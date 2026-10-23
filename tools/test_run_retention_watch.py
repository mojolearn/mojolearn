"""Retention tests: bytes, growing logs, failed uploads, and discovery safety."""
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import subprocess
import contextlib
import io
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('watch', Path(__file__).with_name('run_retention_watch.py'))
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_remote_metadata_tools_do_not_block_output_discovery(self):
        output = io.StringIO()
        with patch.object(watch.subprocess, 'run', side_effect=FileNotFoundError('tool absent')), \
             patch.object(watch.subprocess, 'check_output', return_value=''), \
             patch('glob.glob', return_value=['/root/leg_out']), contextlib.redirect_stdout(output):
            exec(watch.REMOTE, {})
        result = json.loads(output.getvalue())
        self.assertIn('/root/leg_out', result['paths'])
        self.assertEqual(result['source']['exit_code'], 127)

    def test_minimal_image_tar_transport_retains_actual_bytes(self):
        mirror = self.root / 'mirror'; mirror.mkdir()
        def remote(argv, stdout, stderr, timeout):
            with tarfile.open(fileobj=stdout, mode='w:gz') as archive:
                content = b'build still running\n'
                item = tarfile.TarInfo('root/leg_out/cmd.log'); item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
            return subprocess.CompletedProcess(argv, 0)
        with patch.object(watch.subprocess, 'run', side_effect=remote):
            rc = watch.copy_with_tar({'ssh': ['root@host']}, ['/root/leg_out'], mirror, self.root/'copy.log')
        self.assertEqual(rc, 0)
        self.assertEqual((mirror/'files/root/leg_out/cmd.log').read_bytes(), b'build still running\n')
        self.assertFalse(list(mirror.glob('transfer-*.tgz')))

    def test_exact_bytes_and_manifest(self):
        source = self.root / 'run.log'; source.write_bytes(b'passed\nfailed\x00\n')
        index = {}
        directory, entries, errors = watch.make_batch(self.root / 'watch', [source], index)
        self.assertFalse(errors)
        self.assertEqual(index[str(source)], watch.signature(source))
        with tarfile.open(directory / 'evidence.tgz') as archive:
            data = archive.extractfile(entries[0]['member']).read()
            manifest = json.load(archive.extractfile('manifest.json'))
        self.assertEqual(data, source.read_bytes())
        self.assertEqual(manifest['files'][0]['sha256'], watch.hashlib.sha256(data).hexdigest())

    def test_growing_log_retries_without_overwriting_old_batch(self):
        source = self.root / 'run.log'; source.write_text('first\n')
        index = {}
        first, _, _ = watch.make_batch(self.root / 'watch', [source], index)
        original = (first / 'evidence.tgz').read_bytes()
        source.write_text('first\nsecond\n')
        second, entries, _ = watch.make_batch(self.root / 'watch', [source], index)
        self.assertNotEqual(first, second)
        self.assertEqual(original, (first / 'evidence.tgz').read_bytes())
        self.assertEqual(entries[0]['size'], len('first\nsecond\n'))

    def test_changed_during_copy_remains_pending(self):
        source = self.root / 'run.log'; source.write_text('first\n')
        sig = watch.signature(source)
        index = {}
        with patch.object(watch, 'signature', side_effect=[sig, [sig[0]+1, sig[1]+1, sig[2]]]):
            _, entries, _ = watch.make_batch(self.root / 'watch', [source], index)
        self.assertFalse(entries[0]['stable'])
        self.assertNotIn(str(source), index)

    def test_failed_upload_preserves_pending_bytes_then_retries(self):
        source = self.root / 'run.log'; source.write_text('failure evidence\n')
        directory, _, _ = watch.make_batch(self.root / 'watch', [source], {})
        original = (directory / 'evidence.tgz').read_bytes()
        with patch.object(watch, 'run', return_value=1):
            watch.upload_pending(self.root / 'watch')
        receipt = watch.read(directory / 'receipt.json')
        self.assertEqual(receipt['state'], 'pending_upload')
        self.assertEqual(original, (directory / 'evidence.tgz').read_bytes())
        def success(*args, **kwargs):
            watch.save(directory / 'r2-head.json', {'ContentLength': receipt['archive_bytes'],
                'Metadata': {'sha256': receipt['archive_sha256']}})
            return 0
        with patch.object(watch, 'run', side_effect=success):
            watch.upload_pending(self.root / 'watch')
        self.assertEqual(watch.read(directory / 'receipt.json')['state'], 'retained')

    def test_wrong_remote_size_not_accepted(self):
        source = self.root / 'run.log'; source.write_text('some output')
        directory, _, _ = watch.make_batch(self.root / 'watch', [source], {})
        receipt = watch.read(directory / 'receipt.json')
        watch.save(directory / 'r2-head.json', {'ContentLength': 0,
                   'Metadata': {'sha256': receipt['archive_sha256']}})
        with patch.object(watch, 'run', return_value=0):
            watch.upload_pending(self.root / 'watch')
        self.assertEqual(watch.read(directory / 'receipt.json')['state'], 'pending_upload')

    def test_secrets_caches_symlinks_excluded_binaries_kept(self):
        (self.root / 'cache').mkdir(); (self.root / 'cache/result').write_text('cache')
        for name in ['run.log', 'model.so', '.env', 'curlrc', 'key.pem']:
            (self.root / name).write_text('bytes')
        (self.root / 'alias').symlink_to(self.root / 'run.log')
        self.assertEqual({p.name for p in watch.walk_files(self.root)}, {'run.log', 'model.so'})

    def test_box_registration_rotates_identity_skips_released(self):
        path = self.root / 'devpods/test/state.env'; path.parent.mkdir(parents=True)
        path.write_text('PROVIDER=runpod\nPOD_ID=newid\nSSH_TARGET=-p\\ 22\\ root@host\n')
        old = self.root / 'devpods/test.down-20261020/state.env'; old.parent.mkdir(parents=True)
        old.write_text(path.read_text())
        lease = path.parent / 'lease/newid.lease'; lease.parent.mkdir()
        lease.write_text("pod='newid'\ndeadline_epoch=1792700000\n")
        result = watch.boxes(self.root)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['ssh'], ['-p', '22', 'root@host'])
        self.assertEqual(result[0]['name'], 'test-newid')
        self.assertEqual(result[0]['lease_deadline_epoch'], 1792700000)

    def test_old_active_ci_is_followed_and_completion_stays_until_collected(self):
        old = {'databaseId': 12, 'status': 'in_progress'}
        newer = {'databaseId': 99, 'status': 'completed'}
        done = {'databaseId': 12, 'status': 'completed', 'conclusion': 'failure'}
        def query(argv, **kwargs):
            if 'view' in argv:
                self.assertIn('12', argv)
                return json.dumps(done)
            return json.dumps([] if '--status' in argv else [newer])
        with patch.object(watch, 'capture', side_effect=query):
            result = watch.github_observation({'github': [old]})
            self.assertEqual(result, [newer, done])
            self.assertEqual(watch.github_observation({'github': result}), result)

    def test_local_scan_failure_does_not_lose_github_monitoring(self):
        watch.save(self.root / 'observation.json', {'local': {'processes': [], 'open_logs': ['run.log']}})
        with patch.object(watch, 'boxes', return_value=[]), \
             patch.object(watch, 'local_observation', side_effect=subprocess.TimeoutExpired('lsof', 15)), \
             patch.object(watch, 'github_observation', return_value=[{'databaseId': 1}]) as github:
            result = watch.observe(self.root, self.root, self.root)
        github.assert_called_once()
        self.assertIn('local_error', result)
        self.assertEqual(result['local']['open_logs'], ['run.log'])
        self.assertEqual(result['github'], [{'databaseId': 1}])

    def test_ci_artifact_bytes_are_retained_and_download_not_repeated(self):
        workflow = {'databaseId': 123}
        record_path = self.root / 'github/123/receipt.json'
        watch.save(record_path, {'log_exit_code': 0})
        artifact = {'id': 456, 'name': 'generated-program', 'expired': False, 'size_in_bytes': 100}
        def download(argv, stdout, stderr, timeout):
            with watch.zipfile.ZipFile(stdout, 'w') as archive:
                archive.writestr('program.mojo', b'actual source bytes')
            return subprocess.CompletedProcess(argv, 0)
        with patch.object(watch, 'capture', return_value=json.dumps(artifact)), \
             patch.object(watch.subprocess, 'run', side_effect=download) as fetch:
            record = watch.retain_workflow(self.root, workflow)
            self.assertTrue(record['artifacts_complete'])
            watch.retain_workflow(self.root, workflow)
        fetch.assert_called_once()
        with watch.zipfile.ZipFile(self.root / 'github/123/artifacts/456.zip') as archive:
            self.assertEqual(archive.read('program.mojo'), b'actual source bytes')

    def test_expired_artifacts_are_not_claimed_retained(self):
        watch.save(self.root / 'github/123/receipt.json', {'log_exit_code': 0})
        artifact = {'id': 456, 'name': 'lost', 'expired': True, 'size_in_bytes': 100}
        with patch.object(watch, 'capture', return_value=json.dumps(artifact)):
            record = watch.retain_workflow(self.root, {'databaseId': 123})
        self.assertFalse(record['artifacts_complete'])
        self.assertEqual(record['artifacts'][0]['state'], 'expired_unretained')

    def test_health_reports_stale_watchers_and_pending_retention(self):
        watch.save(self.root / 'observation.json', {'observed_at': '2020-01-01T00:00:00+00:00'})
        watch.save(self.root / 'retention.json', {'state': 'attention', 'pending_batches': 1})
        launched = subprocess.CompletedProcess([], 0, stdout='state = not running\nlast exit code = 0', stderr='')
        with patch.object(watch.subprocess, 'run', return_value=launched), \
             patch.object(watch, 'run', return_value=0):
            result = watch.health(self.root, self.root, self.root)
        self.assertEqual(result['state'], 'attention')
        self.assertTrue(any('3 minutes' in a for a in result['alerts']))
        self.assertTrue(any('retention incomplete' in a for a in result['alerts']))
        self.assertTrue((self.root / 'health-events.jsonl').exists())

    def test_health_alerts_on_local_worker_failure_without_inventing_exit_code(self):
        source = self.root / 'worker/STATUS.json'
        watch.save(self.root / 'config.json', {'job_status_files': [str(source)]})
        watch.save(self.root / 'observation.json', {'observed_at': watch.utc()})
        watch.save(self.root / 'retention.json', {'state': 'retained', 'started_at': watch.utc(), 'finished_at': watch.utc()})
        watch.save(source, {'observed_at': watch.utc(), 'jobs': {'build': {'state': 'running'}}})
        launched = subprocess.CompletedProcess([], 0, stdout='state = not running\nlast exit code = 0', stderr='')
        with patch.object(watch.subprocess, 'run', return_value=launched), patch.object(watch, 'run', return_value=0):
            watch.health(self.root, self.root, self.root)
            watch.save(source, {'observed_at': watch.utc(), 'jobs': {'build': {'state': 'lost'}}})
            result = watch.health(self.root, self.root, self.root)
        self.assertTrue(any('Local job build: lost' in a for a in result['alerts']))
        self.assertNotIn('exit_code', result['local_job_changes'][0])

    def test_remote_verification_failure_remains_visible_after_successful_copy(self):
        watch.save(self.root/'observation.json', {'observed_at': watch.utc()})
        watch.save(self.root/'retention.json', {'state': 'retained', 'started_at': watch.utc(),
                   'finished_at': watch.utc(), 'boxes': [{'name': 'amd', 'state': 'copied'}]})
        status = self.root/'mirrors/amd/files/root/out/status.txt'; status.parent.mkdir(parents=True)
        status.write_text('03:18Z BUILD gpu-identical rc=0\n03:19Z VERIFY identical rc=1 RESULT: MISMATCH\n')
        launched = subprocess.CompletedProcess([], 0, stdout='state = not running\nlast exit code = 0', stderr='')
        with patch.object(watch.subprocess, 'run', return_value=launched), patch.object(watch, 'run', return_value=0):
            result = watch.health(self.root, self.root, self.root)
        self.assertEqual(result['state'], 'attention')
        self.assertTrue(any('VERIFY identical failed' in a for a in result['alerts']))
        watch.save(self.root/'history/123.json', watch.read(self.root/'retention.json'))
        watch.save(self.root/'retention.json', {'state': 'collecting', 'started_at': watch.utc(),
                   'last_completed_at': watch.utc(), 'boxes': []})
        with patch.object(watch.subprocess, 'run', return_value=launched), patch.object(watch, 'run', return_value=0):
            collecting = watch.health(self.root, self.root, self.root)
        self.assertTrue(any('VERIFY identical failed' in a for a in collecting['alerts']))


if __name__ == '__main__':
    unittest.main()
