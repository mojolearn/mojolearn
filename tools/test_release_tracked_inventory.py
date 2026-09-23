"""The release source gates judge TRACKED sources, and tests are not build input.

A throwaway git repository stands in for the Mac checkout. Ignored generated
files (portable-math-build.json, the ignored identity copies under
python/mojolearn/, tokenizer/impl/unicode_table_generated.mojo) and a change
under python/mojolearn/tests/ must not refuse a launch or a pack; an
uncommitted edit to a tracked, shipped file must. File-only, no build.
"""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

import check_linux_release_qualification as gate

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('pack_wheel_tracked', ROOT / 'packaging/linux/pack_wheel.py')
packer = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(packer)
except SystemExit as exc:
    # pack_wheel.py refuses to import below Python 3.11 (tomllib). A SystemExit
    # at collection time is not a test result: under pytest it aborted the
    # WHOLE tools session (INTERNALERROR, 2026-09-23 on 3.10), so nothing
    # after this file ran. It is a skip with the packer's own sentence.
    raise unittest.SkipTest(str(exc))

IGNORED = {
    'portable-math-build.json': '{"local": true}\n',
    'python/mojolearn/_identity_break.py': 'ignored copy\n',
    'python/mojolearn/_identity_trace_diff.py': 'ignored copy\n',
    'tokenizer/impl/unicode_table_generated.mojo': '# generated\n',
}


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def launch_gate(archive, local):
    """The comparison tools/do_release061_leg.sh and gemm_remote_leg.sh run."""
    tracked = set(git(local, 'ls-files', '-z').split('\0'))
    return gate.native_inventory(archive) == [e for e in gate.native_inventory(local) if e[0] in tracked]


class TrackedReleaseInventory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.repo, self.archive = base / 'repo', base / 'archive'
        self.repo.mkdir()
        files = {
            'bindings/build_x.sh': 'echo build\n',
            'kernels/k.mojo': 'fn f(): pass\n',
            'python/mojolearn/api.py': 'VALUE = 1\n',
            'python/mojolearn/tests/test_api.py': 'assert True\n',
            'pixi.toml': '[project]\n',
            '.gitignore': '\n'.join(IGNORED) + '\n',
        }
        for rel, text in files.items():
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / rel).write_text(text)
        git(self.repo, 'init', '-q')
        git(self.repo, 'add', '--', *files)
        git(self.repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'fixture')
        self.commit = git(self.repo, 'rev-parse', 'HEAD')
        self.archive.mkdir()
        tar = subprocess.run(['git', '-C', str(self.repo), 'archive', '--format=tar', self.commit],
                             check=True, capture_output=True).stdout
        subprocess.run(['tar', 'xf', '-', '-C', str(self.archive)], input=tar, check=True)
        # The build box's proof: the archive's inventory.
        self.proof_inventory = gate.native_inventory(self.archive)
        for rel, text in IGNORED.items():
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / rel).write_text(text)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tests_are_not_build_input(self):
        listed = {rel for rel, _ in self.proof_inventory}
        self.assertIn('python/mojolearn/api.py', listed)
        self.assertNotIn('python/mojolearn/tests/test_api.py', listed)

    def test_ignored_files_and_test_edits_do_not_refuse(self):
        self.assertEqual(git(self.repo, 'status', '--porcelain', '--untracked-files=no'), '')
        # The old whole-tree walk counted the ignored files; that was the trip.
        self.assertNotEqual(gate.native_inventory(self.repo), self.proof_inventory)
        (self.repo / 'python/mojolearn/tests/test_api.py').write_text('assert 1 == 1\n')
        self.assertTrue(launch_gate(self.archive, self.repo))
        self.assertEqual(gate.tracked_native_inventory(self.repo), self.proof_inventory)
        # The packer's check: every proof entry against the current checkout.
        for rel, digest in self.proof_inventory:
            self.assertEqual(gate.digest_file(self.repo / rel), digest)

    def test_edit_to_tracked_shipped_file_refuses(self):
        (self.repo / 'python/mojolearn/api.py').write_text('VALUE = 2\n')
        self.assertFalse(launch_gate(self.archive, self.repo))
        self.assertNotEqual(gate.tracked_native_inventory(self.repo), self.proof_inventory)

    def test_deleted_or_staged_native_file_refuses(self):
        (self.repo / 'kernels/k.mojo').unlink()
        self.assertFalse(launch_gate(self.archive, self.repo))
        git(self.repo, 'checkout', '--', 'kernels/k.mojo')
        (self.repo / 'kernels/new.mojo').write_text('fn g(): pass\n')
        self.assertTrue(launch_gate(self.archive, self.repo))  # untracked: not shipped
        git(self.repo, 'add', '--', 'kernels/new.mojo')
        self.assertFalse(launch_gate(self.archive, self.repo))

    def test_packer_binds_shipped_python_to_the_commit(self):
        entries = {'mojolearn/api.py': self.repo / 'python/mojolearn/api.py',
                   'mojolearn/data.json': self.repo / 'portable-math-build.json'}
        packer.require_shipped_python_at_commit(entries, self.commit, self.repo)
        (self.repo / 'python/mojolearn/api.py').write_text('VALUE = 3\n')
        with self.assertRaises(SystemExit):
            packer.require_shipped_python_at_commit(entries, self.commit, self.repo)
        git(self.repo, 'checkout', '--', 'python/mojolearn/api.py')
        # An ignored module in a packaged directory would ship; it is refused.
        entries['mojolearn/_identity_break.py'] = self.repo / 'python/mojolearn/_identity_break.py'
        with self.assertRaises(SystemExit):
            packer.require_shipped_python_at_commit(entries, self.commit, self.repo)

    def test_shell_copy_of_the_snapshot_excludes_tests(self):
        text = (ROOT / 'tools/linux_surface_qualification.sh').read_text()
        self.assertIn("if rel.startswith('python/mojolearn/tests/'):", text)


if __name__ == '__main__':
    unittest.main()
