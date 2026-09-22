"""do_release061_leg.sh --expect-from: the AMD leg's core-host digest comes from
the NVIDIA leg's STAGED set copy, never from a hand-typed value or the unstaged
python/mojolearn/host/ copy (0.8.14 wasted an MI325X rental on that). No
network, no rental: every case below refuses or is checked before preflight."""
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEG = ROOT / 'tools/do_release061_leg.sh'
COMMIT = 'a' * 40


def run(*args, env=None):
    e = dict(os.environ, MOJOLEARN_RELEASE_UBUNTU22='1')
    e.pop('MOJOLEARN_EXPECT_CORE_HOST_SHA256', None)
    e.update(env or {})
    return subprocess.run(['bash', str(LEG), *args], capture_output=True, text=True,
                          timeout=60, env=e)


def nvidia_leg(root, commit=COMMIT, staged=b'staged-core-host', complete=True):
    base = root / 'remote' / 'release-build'
    host = base / 'build' / 'sets' / 'cuda' / 'sm_89' / 'host'
    host.mkdir(parents=True)
    (host / '_mojolearn_core_host.so').write_bytes(staged)
    (base / 'build' / 'build-provenance.json').write_text(json.dumps(
        {'source_commit': commit, 'complete': complete, 'build_exit': 0}))
    # the unstaged copy the 0.8.14 mistake used: present, different, ignored
    unstaged = root / 'python' / 'mojolearn' / 'host'
    unstaged.mkdir(parents=True)
    (unstaged / '_mojolearn_core_host.so').write_bytes(b'unstaged')
    return hashlib.sha256(staged).hexdigest()


class ExpectFromTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_derives_the_staged_copy(self):
        digest = nvidia_leg(self.root)
        wrong = hashlib.sha256(b'unstaged').hexdigest()
        r = run(COMMIT, '/dev/null', '--expect-from', str(self.root),
                env={'MOJOLEARN_EXPECT_CORE_HOST_SHA256': wrong})
        self.assertEqual(r.returncode, 2)
        self.assertIn('disagrees with the staged copy', r.stderr)
        self.assertIn(digest, r.stderr)
        self.assertIn('build/sets/cuda/sm_89/host/_mojolearn_core_host.so', r.stderr)

    def test_matching_hand_value_passes_derivation(self):
        digest = nvidia_leg(self.root)
        r = run(COMMIT, '/nonexistent-token', '--expect-from', str(self.root / 'remote' / 'release-build'),
                env={'MOJOLEARN_EXPECT_CORE_HOST_SHA256': digest})
        # derivation passed; the leg then refuses in preflight (unknown commit)
        self.assertNotIn('--expect-from', r.stdout + r.stderr)
        self.assertIn('REFUSING: commit', r.stdout + r.stderr)

    def test_other_commit_refused(self):
        nvidia_leg(self.root, commit='b' * 40)
        r = run(COMMIT, '/dev/null', '--expect-from', str(self.root))
        self.assertEqual(r.returncode, 2)
        self.assertIn('not ' + COMMIT, r.stdout + r.stderr)

    def test_incomplete_build_refused(self):
        nvidia_leg(self.root, complete=False)
        r = run(COMMIT, '/dev/null', '--expect-from', str(self.root))
        self.assertEqual(r.returncode, 2)
        self.assertIn('not complete', r.stdout + r.stderr)

    def test_bad_hand_value_refused(self):
        r = run(COMMIT, '/dev/null', env={'MOJOLEARN_EXPECT_CORE_HOST_SHA256': 'A' * 64})
        self.assertEqual(r.returncode, 2)
        self.assertIn('STAGED NVIDIA set copy', r.stderr)

    def test_probe_is_advisory_by_default(self):
        leg = LEG.read_text()
        self.assertIn('CORE_HOST_SHA=skip', leg)
        helper = (ROOT / 'tools/release_ubuntu22_build.sh').read_text()
        self.assertIn('core_host_probe=skipped', helper)


if __name__ == '__main__':
    unittest.main()
