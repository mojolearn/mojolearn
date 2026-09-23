"""tools/lm_run_driver.py lands a live segment from THIS attempt's
coordinator: never from a worker's segment.json (no checkpoints) and never
from a `previous-*` directory's arrival verdict. CPU only, synthetic files."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location(
    'lm_run_driver', Path(__file__).resolve().parents[1] / 'lm_run_driver.py')
drv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drv)


def _seg(d, verdict, role=None, ckpts=(), label='x', start='2026-09-23T14:38:00Z', end='2026-09-23T15:05:00Z'):
    d.mkdir(parents=True, exist_ok=True)
    body = dict(verdict=verdict, steps_completed=10, disagreements=[], label=label, utc_start=start, utc_end=end,
                checkpoints=[dict(file=f, sha256='ab' * 32) for f in ckpts])
    if role:
        body['live'] = dict(role=role)
    (d / 'segment.json').write_text(json.dumps(body))


class LiveLanding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.res = Path(self.tmp.name) / 'legs' / 'A-3-live'
        nv = self.res / 'nvidia-H100' / 'remote' / 'lm-segment-A-3'
        _seg(nv / 'segment', 'PASS', 'coordinator', ('ckpt_00000028.blm', 'ckpt_00000030.blm'), 'nvidia-A-3')
        _seg(nv / 'arrival', 'PASS')
        (nv / 'status.txt').write_text('ready\n')
        _seg(self.res / 'amd-1' / 'remote' / 'lm-segment-A-3' / 'segment', 'PASS', 'worker', (), 'amd-A-3')
        old = self.res / 'previous-20260923-103507' / 'nvidia-H100' / 'remote' / 'lm-segment-A-3'
        _seg(old / 'segment', 'FAIL', 'coordinator', start='2026-09-23T13:28:00Z', end='2026-09-23T13:35:00Z')
        _seg(old / 'arrival', 'FAIL')
        # attempt 1's worker directory, fetched then, never moved aside
        _seg(self.res / 'amd' / 'remote' / 'lm-segment-A-3' / 'segment', 'FAIL', 'worker', (), 'amd-A-3',
             start='2026-09-23T00:47:00Z', end='2026-09-23T01:13:43Z')
        (self.res / 'amd' / 'remote' / 'lm-segment-A-3' / 'status.txt').write_text('stale\n')
        self.e = dict(route='A', segment='3', vendor='live', from_ckpt='ckpt_00000020.blm',
                      replay_ckpt='ckpt_00000018.blm', last=30)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_coordinator_of_this_attempt_lands_the_segment(self):
        ledger = drv.Ledger(self.tmp.name)
        ok = drv.land({}, self.e, self.res, self.tmp.name, ledger)
        rec = ledger.landed(self.e)
        self.assertTrue(ok)
        self.assertEqual(rec['arrival'], 'PASS')
        self.assertEqual(sorted(rec['checkpoints']), ['ckpt_00000028.blm', 'ckpt_00000030.blm'])
        self.assertEqual(rec['workers'], {'amd-A-3': 'PASS'})

    def test_a_failing_worker_of_this_attempt_halts(self):
        _seg(self.res / 'amd-1' / 'remote' / 'lm-segment-A-3' / 'segment', 'FAIL', 'worker', (), 'amd-A-3')
        ledger = drv.Ledger(self.tmp.name)
        self.assertFalse(drv.land({}, self.e, self.res, self.tmp.name, ledger))
        self.assertEqual(ledger.landed(self.e)['verdict'], 'FAIL-WORKER')

    def test_a_previous_attempt_is_never_read(self):
        self.assertNotIn('stale', drv._status(self.res))
        paths = [str(p) for p in drv._this_attempt(self.res, 'segment.json')]
        self.assertFalse(any('previous-' in p for p in paths))
        self.assertEqual(drv._worker_verdicts(self.res, '2026-09-23T14:38:00Z'), {'amd-A-3': 'PASS'})


if __name__ == '__main__':
    unittest.main()
