"""tools/six_lane_grid_lq_feed.sh against a fake lq (no box is touched).

nv and nv2 (2026-10-08: a second L40S) feed one nvidia lines file at once; they share the nv tag ledger, claim
each tag under a lock right before `lq add`, honour both ledger spellings (`MOJOLEARN_GRID_TAG=<tag>` and the bare
tag), and nv2 rewrites `lq add nv ...` to `lq add nv2 ...`. Depth comes from `lq status <box>`.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

FEED = Path(__file__).resolve().parents[1] / 'six_lane_grid_lq_feed.sh'

FAKE_LQ = """#!/bin/bash
# status <box> -> an empty queue; add -> record the line (slowly, so two feeders overlap) and say queued
d=$(dirname "$0")
case $1 in
  status) [ -n "${2:-}" ] || { echo "status needs a box" >&2; exit 2; }; echo "$2: ";;
  add) sleep 0.2; echo "$*" >> "$d/added.txt"; [ "${FAKE_REFUSE:-}" = "$3" ] && { echo "lq: refused"; exit 2; }; echo "queued x";;
esac
"""


class FeedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        lq = self.d / 'fakelq'
        lq.write_text(FAKE_LQ)
        lq.chmod(0o755)
        self.lines = self.d / 'nvidia.lines'
        self.lines.write_text(''.join(f'lq add nv CMD br t{i} MOJOLEARN_GRID_TAG=g.T{i} cmd\n' for i in range(1, 9)))
        (self.d / 'fed-tags-nv.txt').write_text('MOJOLEARN_GRID_TAG=g.T2\ng.T3\n')   # both spellings

    def tearDown(self):
        self.tmp.cleanup()

    def feed(self, box, env=None):
        e = dict(os.environ, LQ=str(self.d / 'fakelq'), **(env or {}))
        return subprocess.Popen(['bash', str(FEED), box, str(self.lines), '8', '1'], cwd=self.d, env=e,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def added(self):
        p = self.d / 'added.txt'
        return [l.split() for l in p.read_text().splitlines()] if p.exists() else []

    def test_two_nvidia_feeders_queue_each_tag_once(self):
        a, b = self.feed('nv'), self.feed('nv2')
        out = a.communicate(timeout=60)[0] + b.communicate(timeout=60)[0]
        self.assertEqual((a.returncode, b.returncode), (0, 0), out)
        tags = [w[5] for w in self.added()]
        self.assertEqual(sorted(tags), sorted(f'MOJOLEARN_GRID_TAG=g.T{i}' for i in (1, 4, 5, 6, 7, 8)), out)
        self.assertEqual({w[1] for w in self.added()} <= {'nv', 'nv2'}, True)
        ledger = (self.d / 'fed-tags-nv.txt').read_text().split()
        self.assertEqual(len(ledger), len(set(ledger)))
        self.assertFalse((self.d / 'fed-tags-nv2.txt').exists())

    def test_nv2_rewrites_the_box(self):
        p = self.feed('nv2')
        out = p.communicate(timeout=60)[0]
        self.assertEqual(p.returncode, 0, out)
        self.assertTrue(self.added())
        self.assertTrue(all(w[1] == 'nv2' for w in self.added()), self.added())

    def test_refused_line_releases_its_claim(self):
        p = self.feed('nv', {'FAKE_REFUSE': 'CMD'})
        out = p.communicate(timeout=60)[0]
        self.assertEqual(p.returncode, 1, out)
        ledger = (self.d / 'fed-tags-nv.txt').read_text().split()
        self.assertNotIn('MOJOLEARN_GRID_TAG=g.T1', ledger)
        self.assertFalse((self.d / 'fed-tags-nv.txt.lockd').exists())

    def test_unknown_box_refused(self):
        p = self.feed('m2')
        p.communicate(timeout=30)
        self.assertEqual(p.returncode, 2)


if __name__ == '__main__':
    unittest.main()
