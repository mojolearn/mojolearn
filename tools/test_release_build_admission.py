"""Exercise the controller's real retained-build gate with inert ELF stand-ins."""
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from verify_linux_surface_qualification import MODES, expected_bindings

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / 'tools/gemm_remote_leg.sh').read_text().split("<<'RELEASE_ADMIT'\n", 1)[1].split('\nRELEASE_ADMIT', 1)[0]


class BuildAdmissionTests(unittest.TestCase):
    def invoke(self, witness, target, damage=None):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            inventory = [['bindings/example.mojo', 'a' * 64]]
            (out / 'inventory.json').write_text(json.dumps(inventory))
            (out / 'exit_code').write_text('0\n')
            (out / 'results.tsv').write_text(''.join(name + '\t0\n' for name in
                ('resource-prefix-tests', 'physical-source-preflight', 'full46-build', 'build-proof-check')))
            extensions = {}
            for mode in MODES:
                for name in expected_bindings(mode, True):
                    member = f'mojolearn/cuda/{target}/' + ('' if mode == 'fast' else mode + '/') + name + '.so'
                    path = out / 'build/sets' / member.removeprefix('mojolearn/')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    content = ('inert ' + member).encode()
                    path.write_bytes(content)
                    extensions[member] = hashlib.sha256(content).hexdigest()
            proof = dict(complete=True, build_exit=0, source_commit='b' * 40,
                         source_inventory=inventory, extensions=extensions)
            (out / 'build/build-provenance.json').write_text(json.dumps(proof))
            (out / 'preflight.json').write_text(json.dumps(dict(device_architecture=witness,
                vendor='cuda', source_inventory=inventory)))
            if damage == 'bytes':
                path.write_bytes(b'changed')
            if damage == 'inventory':
                (out / 'inventory.json').write_text('[]')
            result = subprocess.run([sys.executable, '-c', SCRIPT, str(out), target, 'b' * 40,
                                     str(out / 'inventory.json')], cwd=ROOT,
                                    capture_output=True, text=True, timeout=10)
            return result.returncode, result.stdout + result.stderr

    def test_hopper_driver_capability_admits_its_architecture_specific_build(self):
        status, output = self.invoke('sm_90', 'sm_90a')
        self.assertEqual(status, 0, output)

    def test_ada_exact_match_admits(self):
        status, output = self.invoke('sm_89', 'sm_89')
        self.assertEqual(status, 0, output)

    def test_other_architectures_and_arbitrary_suffixes_refuse(self):
        for witness, target in [('sm_89', 'sm_90a'), ('sm_89', 'sm_89a'), ('sm_90', 'sm_89')]:
            with self.subTest(witness=witness, target=target):
                status, output = self.invoke(witness, target)
                self.assertNotEqual(status, 0)
                self.assertIn('Physical/source witness mismatch', output)

    def test_changed_bytes_and_sources_still_refuse(self):
        for damage in ('bytes', 'inventory'):
            with self.subTest(damage=damage):
                status, output = self.invoke('sm_90', 'sm_90a', damage)
                self.assertNotEqual(status, 0, output)


class InstalledAdmissionTests(unittest.TestCase):
    def test_installed_route_pins_wheel_and_device_without_build_artifacts(self):
        script = (ROOT / 'tools/gemm_remote_leg.sh').read_text().split(
            "<<'RELEASE_QUALIFY_ADMIT'\n", 1)[1].split('\nRELEASE_QUALIFY_ADMIT', 1)[0]
        for wrong in (None, 'architecture', 'wheel', 'vendor', 'retained_failure'):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                audit = dict(sha256='a' * 64, runtime_architecture='sm_89')
                record = dict(vendor='cuda', wheel_sha256='a' * 64)
                if wrong == 'architecture':
                    audit['runtime_architecture'] = 'sm_90a'
                elif wrong == 'wheel':
                    record['wheel_sha256'] = 'b' * 64
                elif wrong == 'vendor':
                    record['vendor'] = 'hip'
                (out / 'wheel-audit.json').write_text(json.dumps(audit))
                with patch.object(sys, 'argv', ['gate', str(out), 'sm_89', 'a' * 64]), \
                     patch('verify_linux_surface_qualification.retained', return_value=(record, {}),
                           side_effect=ValueError('failed retained evidence') if wrong == 'retained_failure' else None), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if wrong:
                        with self.assertRaises((SystemExit, ValueError)):
                            exec(compile(script, '<installed gate>', 'exec'), {})
                    else:
                        exec(compile(script, '<installed gate>', 'exec'), {})


class PackagePreparationTests(unittest.TestCase):
    def test_index_lock_retries_and_failed_refresh_never_installs(self):
        source = (ROOT / 'tools/do_release061_leg.sh').read_text()
        fragment = source.split('  update_end=', 1)[1].split('\nfi\nexport PATH=', 1)[0]
        fragment = ('update_end=' + fragment).replace('\\$', '$').replace('/root/apt.log', '"$TEST_APT_LOG"')
        for always_fail in ('0', '1'):
            with self.subTest(always_fail=always_fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                programs = {
                    'date': 'p=base/"clock"; n=int(p.read_text())+60 if p.exists() else 60; p.write_text(str(n)); print(n)',
                    'sleep': '',
                    'timeout': 'os.execvp(sys.argv[4],sys.argv[4:])',
                    'apt-get': '''p=base/"calls"; p.open("a").write(" ".join(sys.argv[1:])+"\\n")
if "update" in sys.argv:
 n=p.read_text().count("update")
 sys.exit(1 if os.environ["ALWAYS_FAIL"]=="1" or n==1 else 0)
''',
                }
                for name, body in programs.items():
                    path = root / name
                    path.write_text('#!' + sys.executable + '\nimport os,sys\nfrom pathlib import Path\nbase=Path(os.environ["TEST_DIR"])\n' + body + '\n')
                    path.chmod(0o755)
                env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                           TEST_DIR=str(root), TEST_APT_LOG=str(root / 'apt.log'), ALWAYS_FAIL=always_fail)
                result = subprocess.run(['bash', '-c', 'need=dummy\n' + fragment],
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = (root / 'calls').read_text()
                self.assertEqual(calls.count('update'), 2)
                self.assertEqual('install' in calls, always_fail == '0')
                self.assertIn('APT_EXIT=' + ('124' if always_fail == '1' else '0'), result.stdout)


if __name__ == '__main__':
    unittest.main()
