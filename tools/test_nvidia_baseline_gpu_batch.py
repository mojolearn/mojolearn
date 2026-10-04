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

    def test_native_reference_source_and_fixture_scope(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'ref.json'
            doc = dict(commit=SHA, fixtures={k: {} for k in ('base', 'denormal', 'odd')}, cells={'a/base': {}})
            p.write_text(json.dumps(doc))
            self.assertEqual(batch.reference_column(p, SHA), batch.sha(p))
            with self.assertRaisesRegex(ValueError, 'source differs'):
                batch.reference_column(p, 'b' * 40)
            doc['fixtures']['extra'] = {}; p.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'three fixtures'):
                batch.reference_column(p, SHA)

    def test_native_plan_retains_exact_six_hashes_and_reference(self):
        from unittest.mock import patch
        import contextlib, io, sys
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); self.make(root)
            reference=root/'apple.json'
            reference.write_text(json.dumps(dict(commit=SHA,fixtures={k:{} for k in ('base','denormal','odd')},cells={'umap/base':{}})))
            argv=['batch',SHA,'--wheels',str(root),'--out',str(root/'out'),'--gpu','ada',
                  '--native-release-checks','--native-reference-column',str(reference)]
            stdout=io.StringIO()
            with patch.object(sys,'argv',argv), patch.object(batch.subprocess,'check_output',side_effect=[SHA,SHA+'\trefs/heads/candidate']), patch.object(batch.subprocess,'run',return_value=subprocess.CompletedProcess([],0)), contextlib.redirect_stdout(stdout):
                self.assertEqual(batch.main(),0)
            plan=json.loads(stdout.getvalue())
            self.assertEqual(len(plan['native_release_wheels']),6)
            self.assertFalse(any('ptx80' in n for n in plan['native_release_wheels']))
            self.assertEqual(plan['native_reference_sha256'],batch.sha(reference))
            self.assertEqual((plan['work_seconds'],plan['lease_minutes']),(6300,120))
            with patch.object(sys,'argv',argv[:-2]), patch.object(batch.subprocess,'check_output') as calls:
                with self.assertRaisesRegex(ValueError,'reference column'):batch.main()
                calls.assert_not_called()

    def test_canonical_native_stage_is_separate_and_bounded(self):
        body = batch.box_body(SHA, native_release_checks=True)
        self.assertLess(body.index('bash native-release.sh'), body.index('collect prototype'))
        self.assertIn('timeout -k 20 1650 bash native-release.sh || NATIVE_FAILED=1', body)
        self.assertIn('mojolearn_nvidia_ptx80-*) continue', body)
        self.assertIn('--scope expanded --python', body)
        self.assertIn('--fixtures base,denormal,odd', body)
        self.assertIn('--require-columns 2 --lanes "$LANES"', body)
        self.assertIn('--seconds 900 --rss-gib 12 --cores 2', body)
        self.assertIn('collect full "$role" 2400', body)
        self.assertGreater(body.index('test "$NATIVE_FAILED" = 0'), body.index('full-local-comparison'))
        result = subprocess.run(['bash', '-n'], input=body, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_shell_installs_exact_six_and_keeps_receipt_on_later_failure(self):
        import sys
        for fail in ('', 'selftest'):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as td:
                root = Path(td); (root/'wheels').mkdir(); self.make(root/'wheels')
                (root/'venv/bin').mkdir(parents=True); (root/'bin').mkdir()
                (root/'native-reference.json').write_text('{}')
                interpreter = root/'venv/bin/python'
                interpreter.write_text('#!' + sys.executable + '\n' + r'''import json,os,pathlib,subprocess,sys
args=sys.argv[1:]
with open(os.environ['CALLS'],'a') as f:f.write(json.dumps(args)+'\n')
if args[:2]==['-m','venv']:
 p=pathlib.Path(args[2])/'bin';p.mkdir(parents=True);(p/'python').symlink_to(pathlib.Path(sys.argv[0]).resolve())
elif args and args[0].endswith('nvidia_serial_guard.py'):
 sys.exit(subprocess.run(args[args.index('--')+1:]).returncode)
elif args and args[0].endswith('qualify_verifier_wheel.py'):
 out=pathlib.Path(args[args.index('--output')+1]);out.mkdir(parents=True)
 (out/'results.json').write_text(json.dumps({'real_qualifier_arguments':args}))
elif args and args[0].endswith('verify_lanes.py'):
 pathlib.Path(args[args.index('--write-selection')+1]).write_text(json.dumps({'lanes':['umap','ridge']}))
elif args and args[0]=='-c':
 sys.argv=args[1:];exec(args[1])
elif args[:3]==['-m','mojolearn','verify'] and os.environ['FAIL']=='selftest':sys.exit(7)
elif args[:2]==['-m','mojolearn._identity_break']:
 pathlib.Path(args[args.index('--json')+1]).write_text('{}')
''')
                interpreter.chmod(0o755)
                (root/'bin/python3').symlink_to(interpreter)
                timeout = root/'bin/timeout'; timeout.write_text('#!/bin/bash\nshift 3\nexec "$@"\n'); timeout.chmod(0o755)
                result = subprocess.run(['bash', '-c', batch.native_release_body(SHA) + '\nexit "$NATIVE_FAILED"'],
                    cwd=root, env={**os.environ, 'PATH':str(root/'bin')+os.pathsep+os.environ['PATH'],
                                   'CALLS':str(root/'calls.jsonl'), 'FAIL':fail}, capture_output=True, text=True, timeout=20)
                out=root/'results/native-release'
                self.assertEqual(result.returncode, 1 if fail else 0, (out/'body.log').read_text())
                receipt=json.loads((out/'out/results.json').read_text())
                args=receipt['real_qualifier_arguments']
                wheels=[a for a in args if a.endswith('.whl')]
                self.assertEqual(len(wheels),6)
                self.assertFalse(any('ptx80' in a for a in wheels))
                calls=[json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
                installed=[a for a in calls if a[:3]==['-m','pip','install']][0]
                self.assertEqual(set(a for a in installed if a.endswith('.whl')),set(wheels))
                self.assertEqual((out/'body.exit').read_text().strip(),'7' if fail else '0')
                self.assertEqual((out/'column-cuda.json').exists(),not bool(fail))

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

    def test_extra_capture_keeps_strict_collector_and_total_bounds(self):
        body = batch.box_body(SHA, extra_capture=True)
        self.assertIn('--seconds 120 --rss-gib 12 --cores 2', body)
        self.assertIn('source/tools/nvidia_baseline_qualification.py collect', body)
        self.assertIn('MOJOLEARN_CUDA_PATH=ptx-baseline MOJOLEARN_EXPERIMENTAL_PTX=1', body)
        self.assertLess(body.index('extra-wrapper.py'), body.index('collect prototype'))
        self.assertGreater(body.index('test "$EXTRA_FAILED" = 0'), body.index('full-local-comparison.json'))
        subprocess.run(['bash', '-n'], input=body, text=True, check=True)

    def test_full_collectors_capture_the_configuration_witness_themselves(self):
        body = batch.box_body(SHA)
        self.assertIn('cuda-config-witness.py --source ' + SHA, body)
        self.assertIn('--require-collector "results/$scope-$role.json" --wait 300', body)
        self.assertIn('witness=cuda-runtime-config-ptx', body)
        self.assertIn('|| WITNESS_FAILED=1', body)
        self.assertGreater(body.index('test "$WITNESS_FAILED" = 0'), body.index('full-local-comparison.json'))
        # Collectors stay on the frozen payload checkout; only checks use staged tooling.
        self.assertIn('venv/bin/python source/tools/nvidia_baseline_qualification.py collect', body)
        self.assertEqual(body.count('venv/bin/python tooling-check.py check --prototype'), 2)
        self.assertNotIn('source/tools/nvidia_baseline_qualification.py check', body)
        self.assertIn('MOJOLEARN_QUALIFICATION_ROOT="$PWD/source"', body)

    def test_collect_function_runs_witness_only_while_full_collector_is_live(self):
        body = batch.box_body(SHA)
        function = body[body.index('collect() {'):body.index('WITNESS_FAILED=0')]
        for witness_rc, expected in ((0, '0'), (3, '1')):
            with self.subTest(witness_rc=witness_rc), tempfile.TemporaryDirectory() as td:
                root = Path(td); (root/'venv/bin').mkdir(parents=True); (root/'bin').mkdir(); (root/'results').mkdir()
                python = root/'venv/bin/python'
                python.write_text('#!/bin/bash\necho "$MOJOLEARN_CUDA_PATH|$*" >> calls\n'
                                  'case "$*" in *cuda-config-witness.py*) exit ' + str(witness_rc) + ';; esac\n')
                python.chmod(0o755)
                timeout = root/'bin/timeout'; timeout.write_text('#!/bin/bash\nshift 3\nexec "$@"\n'); timeout.chmod(0o755)
                script = ('set -euo pipefail\nMANIFEST=m.json\n' + function + 'WITNESS_FAILED=0\n'
                          'collect prototype baseline 300 --lanes a\ncollect full native-reference 2400\n'
                          'collect full baseline 2400\necho "$WITNESS_FAILED" > failed\n')
                result = subprocess.run(['bash', '-c', script], cwd=root, capture_output=True, text=True, timeout=20,
                                        env={**os.environ, 'PATH': str(root/'bin') + os.pathsep + os.environ['PATH']})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((root/'failed').read_text().strip(), expected)
                calls = (root/'calls').read_text().splitlines()
                witnesses = [c for c in calls if 'cuda-config-witness.py' in c]
                self.assertEqual(len(witnesses), 2)
                self.assertIn('--out results/cuda-runtime-config.json --require-collector results/full-native-reference.json', witnesses[0])
                self.assertIn('--out results/cuda-runtime-config-ptx.json --require-collector results/full-baseline.json', witnesses[1])
                self.assertTrue(all(c.startswith('|') for c in witnesses))  # witness never forces the PTX path
                collectors = [c for c in calls if ' collect ' in c]
                self.assertEqual([c.split('|')[0] for c in collectors], ['ptx-baseline', '', 'ptx-baseline'])

    def test_plan_pins_staged_witness_and_checker(self):
        from unittest.mock import patch
        import contextlib, io, sys
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.make(root)
            argv = ['batch', SHA, '--wheels', str(root), '--out', str(root/'out'), '--gpu', 'ada']
            stdout = io.StringIO()
            with patch.object(sys, 'argv', argv), patch.object(batch.subprocess, 'check_output', side_effect=[SHA, SHA + '\trefs/heads/candidate']), patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), contextlib.redirect_stdout(stdout):
                self.assertEqual(batch.main(), 0)
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan['configuration_witness_sha256'], batch.sha(ROOT / 'tools/cuda_runtime_config_witness.py'))
            self.assertEqual(plan['checker_sha256'], batch.sha(ROOT / 'tools/nvidia_baseline_qualification.py'))

    def test_lease_script_needs_no_ripgrep(self):
        text = (ROOT / 'tools/nvidia_baseline_gpu_lease.sh').read_text()
        self.assertNotRegex(text, r'(^|[ ;&|(])rg ')
        self.assertIn('grep -q WATCHDOG_ALIVE', text)

    def test_native_absent_body_collects_only_forced_ptx(self):
        body = batch.box_body(SHA, extra_capture=True, native_absent=True)
        self.assertNotIn('native-reference', body)
        self.assertNotIn('tooling-check.py check', body)
        self.assertNotIn('native-release.sh', body)
        self.assertEqual(body.count('for role in baseline; do'), 3)
        self.assertIn('collect prototype "$role" 300', body)
        self.assertIn('collect full "$role" 2400', body)
        self.assertIn('extra-wrapper.py', body)
        self.assertIn('cuda-config-witness.py', body)
        self.assertIn('test "$WITNESS_FAILED" = 0', body)
        subprocess.run(['bash', '-n'], input=body, text=True, check=True)
        with self.assertRaisesRegex(ValueError, 'no native stage'):
            batch.box_body(SHA, native_release_checks=True, native_absent=True)
        # Natively supported devices keep both roles and both checks.
        self.assertEqual(batch.box_body(SHA, extra_capture=True).count('for role in native-reference baseline; do'), 3)

    def test_ampere_plan_is_native_absent_and_refuses_native_checks(self):
        from unittest.mock import patch
        import contextlib, io, sys
        self.assertEqual(batch.GPUS['ampere'], 'NVIDIA A100 80GB PCIe')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.make(root)
            argv = ['batch', SHA, '--wheels', str(root), '--out', str(root/'out'), '--gpu', 'ampere']
            stdout = io.StringIO()
            with patch.object(sys, 'argv', argv), patch.object(batch.subprocess, 'check_output', side_effect=[SHA, SHA + '\trefs/heads/candidate']), patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), contextlib.redirect_stdout(stdout):
                self.assertEqual(batch.main(), 0)
            plan = json.loads(stdout.getvalue())
            self.assertTrue(plan['native_absent'])
            self.assertEqual(plan['roles'], ['baseline'])
            self.assertEqual((plan['gpu'], plan['work_seconds'], plan['lease_minutes']), ('NVIDIA A100 80GB PCIe', 6300, 120))
            reference = root/'apple.json'
            reference.write_text(json.dumps(dict(commit=SHA, fixtures={k: {} for k in ('base', 'denormal', 'odd')}, cells={'a/base': {}})))
            with patch.object(sys, 'argv', argv + ['--native-release-checks', '--native-reference-column', str(reference)]), patch.object(batch.subprocess, 'check_output') as calls:
                with self.assertRaisesRegex(ValueError, 'no native stage'):
                    batch.main()
                calls.assert_not_called()

    def test_lease_accepts_a100_with_its_own_price_cap(self):
        text = (ROOT / 'tools/nvidia_baseline_gpu_lease.sh').read_text()
        self.assertIn("'NVIDIA A100 80GB PCIe') ;;", text)
        self.assertIn("1.99 if 'A100' in sys.argv[2] else 3.49", text)

    def test_tooling_split_is_explicit_and_advertised(self):
        from unittest.mock import patch
        import contextlib, io, sys
        tooling = 'b' * 40
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.make(root)
            argv = ['batch', SHA, '--wheels', str(root), '--out', str(root/'out'), '--gpu', 'ada']
            with patch.object(sys, 'argv', argv), patch.object(batch.subprocess, 'check_output', return_value=tooling):
                with self.assertRaisesRegex(ValueError, 'frozen candidate'):
                    batch.main()
            argv += ['--tooling-commit', tooling]
            for advertised, passes in [(SHA+'\trefs/heads/source', False),
                                      (SHA+'\trefs/heads/source\n'+tooling+'\trefs/heads/tools', True)]:
                with self.subTest(advertised=passes), patch.object(sys, 'argv', argv), patch.object(batch.subprocess, 'check_output', side_effect=[tooling,advertised]), patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([],0)), contextlib.redirect_stdout(io.StringIO()) as output:
                    if passes:
                        self.assertEqual(batch.main(),0)
                        plan=json.loads(output.getvalue())
                        self.assertEqual(plan['source_commit'],SHA)
                        self.assertEqual(plan['tooling_commit'],tooling)
                        self.assertEqual(plan['work_seconds'],6300)
                    else:
                        with self.assertRaisesRegex(ValueError,'Push frozen tooling'):
                            batch.main()


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
