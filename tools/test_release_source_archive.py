#!/usr/bin/env python3
"""Exercise the actual campaign-7 archive selector with tracked fixture files."""
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseSourceArchiveTest(unittest.TestCase):
    def test_archive_carries_portable_math_build_inputs(self):
        script = (ROOT / 'tools/gemm_remote_leg.sh').read_text()
        function = script.split('leg_git_archive() {', 1)[1].split('\nleg_public_source_fetch()', 1)[0]
        shell = 'NVIDIA_CAMPAIGN=7\nPAYLOAD=mamba\nleg_git_archive() {' + function + '\nleg_git_archive HEAD\n'
        wanted = {'packaging/portable_math/stage.py',
                  'packaging/portable_math/portable_math.c',
                  'packaging/portable_math/powers_of_ten.h',
                  'packaging/linux/stage_libs.py', 'core/kernel.mojo'}
        excluded = {'packaging/portable_math/results/old.c',
                    'packaging/portable_math/helper.so', 'bench/results/large.bin'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            for name in wanted | excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture\n')
            subprocess.run(['git', 'add', '.'], cwd=root, check=True)
            subprocess.run(['git', '-c', 'user.name=Archive Test', '-c',
                            'user.email=archive@example.invalid', 'commit', '-qm', 'fixture'],
                           cwd=root, check=True)
            data = subprocess.check_output(['sh', '-c', shell], cwd=root)
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                names = {entry.name for entry in archive if entry.isfile()}
            self.assertEqual(wanted, names)


if __name__ == '__main__':
    unittest.main()
