import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('batch', ROOT / 'tools/nvidia_baseline_gpu_batch.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
SHA = 'a' * 40


class Artifacts(unittest.TestCase):
    def make(self, root):
        for name in batch.DISTS:
            with zipfile.ZipFile(root / f'{name}-0.8.37-py3-none-any.whl', 'w') as z:
                if name == 'mojolearn_nvidia_ptx80':
                    data = b'kernel'
                    doc = dict(source_commit=SHA, source_dirty=False, code_format='ptx-baseline',
                               files=[dict(file='identical/x.so', sha256=batch.hashlib.sha256(data).hexdigest())])
                    z.writestr('mojolearn/cuda_ptx/sm_80/PTX_BASELINE.json', json.dumps(doc))
                    z.writestr('mojolearn/cuda_ptx/sm_80/identical/x.so', data)
                else:
                    z.writestr(name + '-0.8.37.dist-info/LINUX_PAYLOAD.json', json.dumps(dict(source_commit=SHA)))
                    if name == 'mojolearn':
                        z.writestr('mojolearn/identity_columns/COMMIT', SHA)

    def test_exact_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.make(root)
            self.assertEqual(len(batch.artifacts(root, SHA)[0]), 7)
            with self.assertRaisesRegex(ValueError, 'source'):
                batch.artifacts(root, 'b' * 40)
            next(root.glob('mojolearn_amd_gfx942-*')).unlink()
            with self.assertRaisesRegex(ValueError, 'Exactly'):
                batch.artifacts(root, SHA)

    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.make(root)
            wheel = next(root.glob('mojolearn_nvidia_ptx80-*'))
            with zipfile.ZipFile(wheel, 'a') as z:
                z.writestr('mojolearn/cuda_ptx/sm_80/identical/x.so', b'changed')
            with self.assertRaisesRegex(ValueError, 'payload'):
                batch.artifacts(root, SHA)

    def test_dryrun_never_calls_transport(self):
        from unittest.mock import patch
        import contextlib, io, sys
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.make(root)
            out = root / 'output'
            argv = ['batch', SHA, '--wheels', str(root), '--out', str(out), '--gpu', 'ada']
            with patch.object(sys, 'argv', argv), patch.object(batch.subprocess, 'check_output', side_effect=[SHA, SHA + '\trefs/heads/candidate']), patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run, patch.object(batch.tempfile, 'TemporaryDirectory') as stage, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(batch.main(), 0)
                stage.assert_not_called()
                self.assertEqual(run.call_count, 1)
                self.assertIn('diff', run.call_args.args[0])
                self.assertFalse(out.exists())

    def test_wrong_installed_manifest_root_refuses_before_rental(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.make(root)
            wheel = next(root.glob('mojolearn_nvidia_ptx80-*'))
            with zipfile.ZipFile(wheel) as z:
                files = {name: z.read(name) for name in z.namelist()}
            with zipfile.ZipFile(wheel, 'w') as z:
                for name, value in files.items():
                    z.writestr(name.replace('mojolearn/', 'other_package/'), value)
            with self.assertRaisesRegex(ValueError, 'registered directory'):
                batch.artifacts(root, SHA)

    def test_changed_staging_copy_never_calls_rental_transport(self):
        from unittest.mock import patch
        import contextlib, io, shutil, sys
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.make(root)
            original_copy = shutil.copyfile
            def changed_copy(source, destination):
                original_copy(source, destination)
                with open(destination, 'ab') as stream:
                    stream.write(b'changed after admission')
            argv = ['batch', SHA, '--wheels', str(root), '--out', str(root / 'out'), '--gpu', 'ada', '--rent']
            with patch.object(sys, 'argv', argv), patch.object(batch.subprocess, 'check_output', side_effect=[SHA, SHA + '\trefs/heads/candidate']), patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run, patch.object(shutil, 'copyfile', side_effect=changed_copy), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'changed while staging'):
                    batch.main()
                self.assertEqual(run.call_count, 1)
                self.assertIn('diff', run.call_args.args[0])

    def test_body_order_and_bound(self):
        body = batch.box_body(SHA)
        self.assertLess(body.index('prototype-comparison.json'), body.index('collect full'))
        self.assertIn('collect full "$role" 2400', body)
        self.assertIn("sysconfig.get_paths()['purelib']", body)
        self.assertIn("/'mojolearn/cuda_ptx/sm_80/PTX_BASELINE.json'", body)
        self.assertNotIn('**/cuda_ptx', body)
        self.assertIn('unset PYTHONPATH MOJOLEARN_CUDA_PATH MOJOLEARN_EXPERIMENTAL_PTX', body)
        self.assertNotIn('collect full', batch.box_body(SHA, False))
        subprocess.run(['bash', '-n'], input=body, text=True, check=True)


MOCK = r'''
RP=https://invalid.test/v1
load_key() { :; }
rp_call() { echo "api:$1:$2" >> "$OUT/calls"; RP_CODE=200; echo '{}' > "$TMPD/rp.body"; }
rp_py() {
 case "$1" in
 id) case "$CASE" in ambiguous|ambiguous_empty) ;; *) echo pod123 ;; esac ;;
 byname) [ "$CASE" = ambiguous_empty ] || echo pod123 ;;
 cost) [ "$CASE" = price ] && echo 99 || echo .74 ;;
 ssh) echo '-p 22 root@fake' ;;
 esac
 return 0
}
write_deadman() { printf 'echo $$ > %s/deadman.pid\nexec sleep 600\n' "$OUT" > "$1/deadman.sh"; echo "$1" > "$OUT/deadman.dir"; }
delete_pod() { echo "delete:$1" >> "$OUT/calls"; }
verify_gone() { echo "verify:$1" >> "$OUT/calls"; [ "$CASE" != unconfirmed ]; }
with_timeout() {
 shift
 echo "transport:$*" >> "$OUT/calls"
 case "$*" in
 *runpod_guard*) [ "$CASE" != arm ] ;;
 *TOKEN_GET*) echo WATCHDOG_ALIVE; echo TOKEN_GET_200 ;;
 *'tar czf - results'*) tar czf - -C "$STAGE" SHA256SUMS ;;
 *'timeout -k 30 6300'*) [ "$CASE" != body ] ;;
 *) return 0 ;;
 esac
}
'''


class Lease(unittest.TestCase):
    def run_case(self, case):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / 'tools').mkdir(); out = root / 'out'; out.mkdir()
            stage = root / 'stage'; stage.mkdir(); (stage / 'SHA256SUMS').write_text('')
            (root / 'tools/runpod_pod_lib.sh').write_text(MOCK)
            script = root / 'tools/nvidia_baseline_gpu_lease.sh'
            script.write_text((ROOT / 'tools/nvidia_baseline_gpu_lease.sh').read_text())
            result = subprocess.run(['bash', str(script), str(stage), str(out), batch.GPUS['ada']],
                                    env={**os.environ, 'CASE': case}, capture_output=True, text=True, timeout=15)
            calls = (out / 'calls').read_text()
            teardown = (out / 'teardown.txt').read_text()
            # A deliberately unconfirmed delete leaves its real mock sleeper
            # armed, exactly as production must. Explicitly clean test fixture.
            if case in ('unconfirmed', 'ambiguous_empty'):
                import signal, shutil
                pid = int((out / 'deadman.pid').read_text())
                os.kill(pid, 0)  # runner must leave it armed when deletion unconfirmed
                os.kill(pid, signal.SIGTERM)
                shutil.rmtree((out / 'deadman.dir').read_text().strip())
            return result, calls, teardown

    def test_success_teardown(self):
        result, calls, teardown = self.run_case('success')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('delete:pod123', calls); self.assertIn('verify:pod123', calls)
        self.assertIn('terminated_verified=1', teardown)

    def test_unconfirmed_teardown_fails_closed(self):
        result, calls, teardown = self.run_case('unconfirmed')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('terminated_verified=0', teardown)
        self.assertIn('deadman remains armed', result.stderr)

    def test_ambiguous_create_with_no_immediate_listing_keeps_deadman(self):
        result, calls, teardown = self.run_case('ambiguous_empty')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('terminated_verified=0', teardown)
        self.assertNotIn('delete:', calls)
        self.assertIn('deadman remains armed', result.stderr)

    def test_failure_paths_teardown(self):
        for case in ['price', 'ambiguous', 'arm', 'body']:
            with self.subTest(case=case):
                result, calls, teardown = self.run_case(case)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('delete:pod123', calls)
                self.assertIn('terminated_verified=1', teardown)
                if case == 'arm':
                    self.assertNotIn('sha256sum -c', calls)


if __name__ == '__main__':
    unittest.main()
