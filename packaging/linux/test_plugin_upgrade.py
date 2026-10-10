#!/usr/bin/env python3
"""Exercise pip's actual RECORD uninstall with the old vendor ownership layout.

Inert fixture payloads are tagged any so the installer test also runs on macOS.
No shared library is executed and no package index is contacted.
"""
import subprocess
import tempfile
import unittest
import venv
from pathlib import Path

import test_split_wheels as fixtures

pw = fixtures.pw
members = fixtures.members


class PluginUpgrade(unittest.TestCase):
    def test_old_vendor_uninstall_preserves_new_architecture_payloads(self):
        fixtures.SplitWheels.setUpClass()
        self.addCleanup(fixtures.SplitWheels.tearDownClass)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = root / 'environment'
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / 'bin/python'

            def install(wheel):
                result = subprocess.run(
                    [str(python), '-m', 'pip', 'install', '--no-index', '--no-deps',
                     '--disable-pip-version-check', str(wheel)],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 0, result.stdout)

            def write_fixture(name, version, data):
                dist = f'{name}-{version}.dist-info'
                data = {**data, f'{dist}/METADATA': (
                    f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n').encode(),
                    f'{dist}/WHEEL': pw.wheel_file_text('py3-none-any').encode()}
                return pw.write_wheel(root / f'{name}-{version}-py3-none-any.whl', {}, data, dist)

            # Model precisely the legacy vendor RECORD: all CUDA architectures
            # lived under cuda/. Install new payloads before replacing that owner.
            expected = {}
            legacy = {}
            for row in pw.gpu_plugins.PAYLOADS.values():
                if row['vendor'] != 'cuda' or not row['release_enabled']:
                    continue
                data = {n: b for n, b in members(fixtures.SplitWheels.split[row['wheel_name']]).items()
                        if '.dist-info/' not in n}
                expected.update(data)
                # the PTX slot (cuda_ptx/sm_80) too: the legacy owner kept every CUDA set under cuda/
                legacy.update({n.replace('/cuda_native/', '/cuda/').replace('/cuda_ptx/', '/cuda/'): b
                               for n, b in data.items()})
            install(write_fixture('mojolearn_nvidia', '0.0.1', legacy))
            for row in pw.gpu_plugins.PAYLOADS.values():
                if row['vendor'] == 'cuda' and row['release_enabled']:
                    data = {n: b for n, b in expected.items()
                            if pw.gpu_plugins.member_payload(n) == row['profile']}
                    install(write_fixture(row['wheel_name'], '0.0.2', data))
            site = Path(subprocess.check_output(
                [str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
                text=True).strip())
            for path, data in expected.items():
                self.assertEqual((site / path).read_bytes(), data, path)
            self.assertFalse(any((site / 'mojolearn/cuda').rglob('*.so')))


if __name__ == '__main__':
    unittest.main()
