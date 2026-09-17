#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise output selection and local fetches without credentials or rentals."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
GEMM = (TOOLS / 'gemm_remote_leg.sh').read_text()
RELEASE = (TOOLS / 'do_release061_leg.sh').read_text()


class LegOutputTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='leg output ')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / 'checkout'
        self.repo.mkdir()
        # Preserve the real HOME; override only the task's evidence setting.
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith('MOJOLEARN_')}
        self.env.update(MOJOLEARN_EVIDENCE_ROOT=str(self.root / 'evidence store'),
                        VENDOR='nvidia', PAYLOAD='gemm', MODE='rent',
                        SPEED_FAMILY='classical', NVIDIA_CAMPAIGN='0',
                        COMMIT='a' * 40, LEG_VENDOR='hip', LEG_ARCH='gfx942',
                        LEG_MODE='build')

    def run_shell(self, body, **settings):
        env = dict(self.env, **settings)
        return subprocess.check_output(
            ['/bin/sh', '-eu', '-c', body], cwd=self.repo, env=env, text=True)

    def gemm_selection(self):
        return 'STAMP=' + GEMM.split('\nSTAMP=', 1)[1].split(
            '\n# ---------------------------------------------------------------------------', 1)[0]

    def release_selection(self):
        return 'LEG_EVIDENCE_ROOT=' + RELEASE.split('\nLEG_EVIDENCE_ROOT=', 1)[1].split(
            '\nSTATE=', 1)[0]

    def gemm_path(self, **settings):
        return Path(self.run_shell(self.gemm_selection() + '\nprintf "%s" "$OUT"',
                                   **settings))

    def test_default_home_and_custom_root(self):
        default = self.gemm_path(MOJOLEARN_EVIDENCE_ROOT='')
        self.assertEqual(default.parent, Path.home() / 'mojolearn-evidence/e1g')
        self.assertEqual(self.gemm_path().parent, self.root / 'evidence store/e1g')
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_dry_and_payload_names(self):
        for mode, suffix in [('rent', '-nvidia-mamba'), ('dry', '-nvidia-mamba-dryrun')]:
            with self.subTest(mode=mode):
                self.assertTrue(self.gemm_path(MODE=mode, PAYLOAD='mamba').name.endswith(suffix))

    def test_explicit_output_and_phase8_stay_together(self):
        for output in ['relative output', str(self.root / 'custom output')]:
            with self.subTest(output=output):
                lines = self.run_shell(
                    self.gemm_selection() + '\nprintf "%s\\n%s\\n" "$OUT" "$E1_DEST"',
                    PAYLOAD='phase8', MOJOLEARN_GEMM_LEG_OUT=output).splitlines()
                self.assertEqual(lines[0], output)
                self.assertEqual(Path(lines[1]).parent, Path(output) / 'e1')
                self.assertTrue(Path(lines[1]).name.endswith('-runpod-nvidia'))

    def test_release_source_scoping_and_overrides(self):
        for commit in ['a' * 40, 'b' * 40]:
            output = self.run_shell(self.release_selection() + '\nprintf "%s" "$OUT"',
                                    COMMIT=commit)
            self.assertEqual(Path(output), self.root / 'evidence store/releases' / commit / 'hip-gfx942')
        output = self.run_shell(self.release_selection() + '\nprintf "%s" "$OUT"',
                                MOJOLEARN_RELEASE_RESULTS_ROOT='explicit release')
        self.assertEqual(output, 'explicit release/hip-gfx942')

    def test_qualification_is_separate_from_build(self):
        output = Path(self.run_shell(self.release_selection() + '\nprintf "%s" "$OUT"',
                                     LEG_MODE='qualify'))
        self.assertEqual(output.parent.name, 'qualification')
        self.assertTrue(output.name.startswith('hip-gfx942-'))

    def test_fetch_preserves_complete_output_outside_checkout(self):
        remote = self.root / 'fake remote'
        (remote / 'nested').mkdir(parents=True)
        (remote / 'nested/binary').write_bytes(b'\x7fELF\x00\xff')
        (remote / '.provenance').write_text('source witness\n')
        fetch = 'leg_fetch() {' + GEMM.split('leg_fetch() {', 1)[1].split('\n}\n', 1)[0] + '\n}\n'
        # Substitute only the SSH transport. Execute the actual fetch function.
        transport = '''leg_ssh() {
            case "$1" in
                *'tar czf'*) tar czf - -C "$FAKE_REMOTE" . ;;
                *) return 0 ;;
            esac
        }
        '''
        self.run_shell(self.gemm_selection() + '\n' + transport + fetch + '\nleg_fetch',
                       FAKE_REMOTE=str(remote))
        runs = list((self.root / 'evidence store/e1g').iterdir())
        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0] / 'remote/nested/binary').read_bytes(), b'\x7fELF\x00\xff')
        self.assertEqual((runs[0] / 'remote/.provenance').read_text(), 'source witness\n')
        self.assertEqual(list(self.repo.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
