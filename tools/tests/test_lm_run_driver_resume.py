"""tools/lm_run_driver.py resumes a one-box segment that did not land from
its last uploaded checkpoint instead of restarting it. The case is T3's
A/4 (2026-09-25): 1,500 steps from 2400, the box hung during step 2700
after uploading checkpoints 2500 and 2600 and writing chain lines 2401 to
2699. CPU only; every subprocess (the renderer, presigning, curl) is
mocked and nothing is rendered, uploaded or rented."""
import argparse
import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.argv = [sys.argv[0]]
TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location('lm_run_driver', TOOLS / 'lm_run_driver.py')
drv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drv)
import lm_file_evidence as fe  # noqa: E402

RUN = 'runs/t3/x'
SIZE = 1945780169


def _sha(n):
    return hashlib.sha256(b'ckpt %d' % n).hexdigest()


def _state(step):
    return hashlib.sha256(b'state %d' % step).hexdigest()


def _chain(steps, label, linked_from=None):
    """Lines the way lm_segment writes them: canonical JSON, each carrying
    the sha256 of the line before it (None for a box's first line)."""
    out, prev = [], linked_from
    for s in steps:
        row = dict(schema='mojolearn.lm-segment.chain.v1', step=s, route='A', segment='4', label=label,
                   lr_f32_hex='39a646d0', losses_f32_hex=['404192f4', '4047f14a'], state_sha256=_state(s),
                   gradient_sha256=_state(-s), hash_scheme='sliced-sha256-8.v2', seconds=70.2, hash_seconds=2.6,
                   prev=prev)
        line = json.dumps(row, sort_keys=True, separators=(',', ':'))
        out.append(line)
        prev = hashlib.sha256(line.encode()).hexdigest()
    return '\n'.join(out) + '\n'


def _spec(**a4):
    seg4 = dict(segment='4', vendor='amd', steps=1500, **a4)
    return dict(run=RUN, recipe='r.json', recipe_key=RUN + '/recipe.json', tokens_stage='tok', lease_minutes=120,
                dollar_cap=10, nvidia_devices='0,1', amd_devices='0',
                routes=dict(A=[dict(segment='1', vendor='nvidia', steps=1000), dict(segment='2', vendor='nvidia', steps=1000),
                               dict(segment='3', vendor='amd', steps=400), seg4,
                               dict(segment='5', vendor='nvidia', steps=1000)],
                            B=[dict(segment='1', vendor='amd', steps=1000), dict(segment='2', vendor='amd', steps=1000),
                               dict(segment='3', vendor='nvidia', steps=400), dict(segment='4', vendor='nvidia', steps=1500)]))


class FakeRun:
    """subprocess.run: presigning returns a URL naming its key, a one-byte
    GET answers the size held for that key, a PUT copies the file into
    `self.put`, the renderer (and anything else) exits 0."""

    def __init__(self, sizes=None, put_ok=True):
        self.sizes = dict(sizes or {})
        self.put, self.calls, self.put_ok = {}, [], put_ok

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if len(argv) > 2 and str(argv[1]).endswith('dataset_store.sh'):
            verb, key = argv[2], argv[3]
            return mock.Mock(returncode=0, stdout=('https://%s/%s\n' % ('put' if verb == 'presign-put' else 'get', key)))
        if argv[0] == 'curl' and '-T' in argv:
            url = argv[-1]
            if self.put_ok:
                self.put[url.split('https://put/', 1)[1]] = Path(argv[argv.index('-T') + 1]).read_text()
            return mock.Mock(returncode=0 if self.put_ok else 22, stdout='', stderr='')
        if argv[0] == 'curl' and '-r' in argv:
            key = argv[-1].split('https://get/', 1)[1]
            if key not in self.sizes:
                return mock.Mock(returncode=22, stdout='', stderr='404')
            return mock.Mock(returncode=0, stdout='HTTP/1.1 206\r\nContent-Range: bytes 0-0/%d\r\n' % self.sizes[key])
        return mock.Mock(returncode=0, stdout='', stderr='')

    def renders(self):
        return [c for c in self.calls if len(c) > 2 and str(c[1]).endswith('lm_segment_leg.py')]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'out'
        self.out.mkdir()
        self.spec_path = Path(self.tmp.name) / 'spec.json'
        self.ledger = drv.Ledger(self.out)
        for seg, steps in (('1', (0, 998, 1000)), ('2', (1998, 2000)), ('3', (2398, 2400))):
            self.ledger.land(dict(route='A', segment=seg), dict(verdict='PASS', checkpoints={drv._ckpt(n): _sha(n) for n in steps}))
            self.ledger.land(dict(route='B', segment=seg), dict(verdict='PASS', checkpoints={drv._ckpt(n): _sha(n) for n in steps}))
        self.r2 = {'%s/A/4/%s' % (RUN, drv._ckpt(n)): SIZE for n in (2500, 2600)}

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, spec):
        self.spec_path.write_text(json.dumps(spec))
        return drv.load_spec(self.spec_path)

    def entry(self, spec, key):
        return next(e for e in drv.segment_plan(spec) if '%s/%s' % (e['route'], e['segment']) == key)

    def hung(self, name='leg-do-1', chain_to=2699, uploaded=(2500, 2600), manifest=(2500, 2600), arrival='PASS',
             box_sha=True, start=2400, route='A'):
        """The fetched results of a box that was stopped mid-segment: no segment.json."""
        res = self.out / 'legs' / ('%s-4' % route) / name
        box = res / 'remote' / ('lm-segment-%s-4' % route)
        (box / 'segment').mkdir(parents=True)
        (box / 'segment' / 'chain.jsonl').write_text(_chain(range(start + 1, chain_to + 1), 'amd-%s-4' % route))
        (box / 'segment' / 'manifest.tsv').write_text(''.join('%s\t%d\t%s\n' % (drv._ckpt(n), SIZE, _sha(n)) for n in manifest))
        log = ['[02:13:50] segment: route A segment 4 label amd-A-4, global steps %d..3900' % start]
        for n in uploaded:
            log += ['[04:16:20] uploaded %s in 54.0 s' % drv._ckpt(n), '[04:16:20] checkpoint %s' % drv._ckpt(n)]
        (box / 'segment' / 'log.txt').write_text('\n'.join(log) + '\n')
        if box_sha:
            (box / 'checkpoints.sha256').write_text(''.join('%s  /root/gemm_leg_out/lm-segment-A-4/segment/%s\n' % (_sha(n), drv._ckpt(n)) for n in manifest))
        if arrival:
            (box / 'arrival').mkdir()
            (box / 'arrival' / 'segment.json').write_text(json.dumps(dict(verdict=arrival)))
        (box / 'status.txt').write_text('02:13:42 ready\n08:49:18 segment exit=143 secs=23736\n')
        (box / 'in').mkdir()
        (box / 'in' / 'expect_chain.jsonl').write_text('x')
        (res / 'droplet_id.txt').write_text('123456\n')
        return res

    def resumed(self, name='leg-do-1', first=2600, verdict='PASS', agrees=True, route='A'):
        """The fetched results of the resumed box: steps first+1..3900, from_state compared."""
        res = self.out / 'legs' / ('%s-4' % route) / name
        seg = res / 'remote' / ('lm-segment-%s-4' % route) / 'segment'
        seg.mkdir(parents=True)
        (seg / 'chain.jsonl').write_text(_chain(range(first + 1, 3901), 'amd-%s-4' % route))
        ckpts = [n for n in range(first + 1, 3901) if n % 100 == 0 or n == 3898]
        (seg / 'segment.json').write_text(json.dumps(dict(
            verdict=verdict, first_step=first, last_step=3900, last_completed=3900, steps_completed=3900 - first,
            disagreements=[], utc_start='2026-09-25T10:00:00Z',
            from_state=dict(sha256=_state(first), expected=_state(first) if agrees else 'ff' * 32, agrees=agrees),
            checkpoints=[dict(file=drv._ckpt(n), sha256=_sha(n)) for n in ckpts])))
        return res

    def land(self, spec, key, res, fake=None):
        fake = fake or FakeRun(self.r2)
        with mock.patch.object(drv.subprocess, 'run', side_effect=fake):
            return drv.land(spec, self.entry(spec, key), res, self.out, self.ledger), fake

    def render(self, spec, key, fake=None):
        fake = fake or FakeRun(self.r2)
        with mock.patch.object(drv.subprocess, 'run', side_effect=fake):
            drv.render(spec, self.entry(spec, key), self.out, self.ledger)
        return fake.renders()[-1], fake

    def fresh_argv(self, route='A'):
        """A/4's (or B/4's) body exactly as it rendered before resuming existed."""
        src = self.ledger.landed(dict(route='A', segment='3'))['checkpoints']
        arm = 'amd' if route == 'A' else 'nvidia'
        return [sys.executable, str(drv.REPO / 'tools' / 'lm_segment_leg.py'), 'render', '--run', RUN, '--recipe', 'r.json',
                '--recipe-key', RUN + '/recipe.json', '--arm', arm, '--mode', 'one',
                '--devices', '0' if route == 'A' else '0,1', '--tokens-key', 'tok',
                '--route', route, '--segment', '4', '--label', '%s-%s-4' % (arm, route), '--steps', '1500',
                '--from', 'ckpt_00002400.blm', '--seconds', str(8 * 3600), '--boundary', '3900',
                '--from-sha', src['ckpt_00002400.blm'], '--from-key', RUN + '/A/3/ckpt_00002400.blm',
                '--replay', 'ckpt_00002398.blm', '--replay-sha', src['ckpt_00002398.blm'],
                '--replay-key', RUN + '/A/3/ckpt_00002398.blm', '--replay-chain-key', RUN + '/A/3/chain.jsonl',
                *(['--expect-key', RUN + '/A/4/chain.jsonl'] if route != 'A' else []),
                '--out', str(self.out / 'bodies' / ('%s-4.sh' % route))]


class Partial(Base):
    def test_a_hung_attempt_records_2600_and_its_chain(self):
        spec = self.load(_spec())
        res = self.hung()
        ok, fake = self.land(spec, 'A/4', res)
        self.assertIsNone(ok)
        part = self.ledger.partial(dict(route='A', segment='4'))
        self.assertEqual(part['resume_from'], 2600)
        self.assertEqual(part['checkpoints'], {drv._ckpt(2500): _sha(2500), drv._ckpt(2600): _sha(2600)})
        self.assertEqual(part['arrival'], 'PASS')
        self.assertEqual(part['chain_steps'], [2401, 2699])
        self.assertEqual([s for s, _ in drv._chain_lines(part['chain'])], list(range(2401, 2700)))
        a = part['attempts'][0]
        self.assertEqual((a['n'], a['first_step'], a['last_chain_step']), (1, 2400, 2699))
        # the attempt's small files kept in the box's layout, the checkpoints'
        # inputs (in/) left behind, the runner's droplet id beside them
        self.assertTrue((Path(a['dir']) / 'segment' / 'chain.jsonl').exists())
        self.assertTrue((Path(a['dir']) / 'arrival' / 'segment.json').exists())
        self.assertTrue((Path(a['dir']) / 'droplet_id.txt').exists())
        self.assertFalse((Path(a['dir']) / 'in').exists())
        # the results directory moved aside: the next rental fetches into a clean one
        self.assertFalse(res.exists())
        self.assertTrue(Path(a['results']).name.startswith('previous-attempt-1-'))
        # both checkpoints were looked up in R2 by name
        heads = [c[-1] for c in fake.calls if c[0] == 'curl' and '-r' in c]
        self.assertEqual(heads, ['https://get/%s/A/4/ckpt_00002500.blm' % RUN, 'https://get/%s/A/4/ckpt_00002600.blm' % RUN])

    def test_status_shows_the_partial_entry(self):
        spec = self.load(_spec())
        self.land(spec, 'A/4', self.hung())
        buf = io.StringIO()
        with redirect_stdout(buf):
            drv.cmd_status(argparse.Namespace(spec=str(self.spec_path), out=str(self.out)))
        text = buf.getvalue()
        self.assertIn('A/4 amd     pending', text)
        self.assertIn('partial: resume from step 2600; checkpoints landed ckpt_00002500.blm, ckpt_00002600.blm; chain to step 2699', text)

    def test_a_checkpoint_missing_from_r2_is_not_a_resume_point(self):
        spec = self.load(_spec())
        del self.r2['%s/A/4/ckpt_00002600.blm' % RUN]
        self.land(spec, 'A/4', self.hung())
        part = self.ledger.partial(dict(route='A', segment='4'))
        self.assertEqual(part['resume_from'], 2500)
        self.assertIn('R2 has no object', part['attempts'][0]['refused']['ckpt_00002600.blm'])

    def test_a_checkpoint_without_an_upload_line_or_a_later_chain_line_is_not_one(self):
        spec = self.load(_spec())
        # 2600 pinned but never uploaded (the kill came during the PUT)
        self.land(spec, 'A/4', self.hung(uploaded=(2500,)))
        self.assertEqual(self.ledger.partial(dict(route='A', segment='4'))['resume_from'], 2500)

    def test_a_checkpoint_at_the_last_chain_line_leaves_nothing_to_hold_the_resume_to(self):
        spec = self.load(_spec())
        self.land(spec, 'A/4', self.hung(chain_to=2600))
        self.assertEqual(self.ledger.partial(dict(route='A', segment='4'))['resume_from'], 2500)

    def test_a_box_sha_that_disagrees_with_the_manifest_is_refused(self):
        spec = self.load(_spec())
        res = self.hung()
        p = res / 'remote' / 'lm-segment-A-4' / 'checkpoints.sha256'
        p.write_text(p.read_text().replace(_sha(2600), 'ee' * 32))
        self.land(spec, 'A/4', res)
        self.assertEqual(self.ledger.partial(dict(route='A', segment='4'))['resume_from'], 2500)

    def test_no_uploaded_checkpoint_restarts_from_the_start_as_today(self):
        spec = self.load(_spec())
        ok, _ = self.land(spec, 'A/4', self.hung(chain_to=2450, uploaded=(), manifest=()))
        self.assertIsNone(ok)
        self.assertIsNone(self.ledger.partial(dict(route='A', segment='4')))
        argv, fake = self.render(spec, 'A/4')
        self.assertEqual(argv, self.fresh_argv())
        self.assertFalse([c for c in fake.calls if c[0] == 'curl'])

    def test_a_first_segment_keeps_its_uploaded_seed(self):
        spec = self.load(_spec())
        res = self.out / 'legs' / 'A-1' / 'leg-x'
        box = res / 'remote' / 'lm-segment-A-1'
        (box / 'segment').mkdir(parents=True)
        (box / 'segment' / 'chain.jsonl').write_text(_chain(range(1, 151), 'nvidia-A-1'))
        (box / 'segment' / 'manifest.tsv').write_text('%s\t%d\t%s\n' % (drv._ckpt(100), SIZE, _sha(100)))
        (box / 'segment' / 'log.txt').write_text('uploaded ckpt_00000100.blm in 3.0 s\n')
        (box / 'seed.sha256').write_text('%s  /root/x/in/ckpt_00000000.blm\n' % _sha(0))
        (box / 'status.txt').write_text('upload seed exit=0\n')
        self.ledger.data['landed'].pop('A/1')
        self.r2 = {RUN + '/A/1/ckpt_00000100.blm': SIZE}
        self.land(spec, 'A/1', res)
        part = self.ledger.partial(dict(route='A', segment='1'))
        self.assertEqual(part['resume_from'], 100)
        self.assertEqual(part['checkpoints'], {drv._ckpt(0): _sha(0), drv._ckpt(100): _sha(100)})
        argv, _ = self.render(spec, 'A/1')
        self.assertEqual(argv[argv.index('--from') + 1], 'ckpt_00000100.blm')
        self.assertEqual(argv[argv.index('--steps') + 1], '900')

    def test_a_live_segment_is_never_resumed(self):
        spec = _spec()
        spec['routes']['A'][3].update(vendor='live', first='nvidia', shards=[44, 20])
        spec = self.load(spec)
        ok, _ = self.land(spec, 'A/4', self.hung())
        self.assertIsNone(ok)
        self.assertIsNone(self.ledger.partial(dict(route='A', segment='4')))


class Resume(Base):
    def test_the_next_start_renders_1300_steps_from_its_own_2600(self):
        spec = self.load(_spec())
        self.land(spec, 'A/4', self.hung())
        argv, fake = self.render(spec, 'A/4')
        key = RUN + '/A/4/partial.chain.jsonl'
        self.assertEqual(argv, [
            sys.executable, str(drv.REPO / 'tools' / 'lm_segment_leg.py'), 'render', '--run', RUN, '--recipe', 'r.json',
            '--recipe-key', RUN + '/recipe.json', '--arm', 'amd', '--mode', 'one', '--devices', '0', '--tokens-key', 'tok',
            '--route', 'A', '--segment', '4', '--label', 'amd-A-4', '--steps', '1300',
            '--from', 'ckpt_00002600.blm', '--seconds', str(8 * 3600), '--boundary', '3900',
            '--from-sha', _sha(2600), '--from-key', RUN + '/A/4/ckpt_00002600.blm',
            '--expect-key', key, '--out', str(self.out / 'bodies' / 'A-4.sh')])
        # the chain the resumed box is held to went to R2 first: the hung
        # attempt's lines, so steps 2601..2699 are re-derived and compared
        held = [json.loads(l)['step'] for l in fake.put[key].splitlines()]
        self.assertEqual(held, list(range(2401, 2700)))

    def test_route_b_resumes_held_to_route_a(self):
        spec = self.load(_spec())
        # A/4 landed whole; B/4 hung after its own 2500 and 2600
        self.ledger.land(dict(route='A', segment='4'), dict(verdict='PASS', checkpoints={drv._ckpt(3900): _sha(3900)}))
        self.r2 = {'%s/B/4/%s' % (RUN, drv._ckpt(n)): SIZE for n in (2500, 2600)}
        self.land(spec, 'B/4', self.hung(route='B'))
        argv, fake = self.render(spec, 'B/4')
        self.assertEqual(argv[argv.index('--from-key') + 1], RUN + '/B/4/ckpt_00002600.blm')
        self.assertEqual(argv[argv.index('--steps') + 1], '1300')
        self.assertEqual(argv[argv.index('--expect-key') + 1], RUN + '/A/4/chain.jsonl')
        self.assertNotIn('--replay', argv)
        self.assertEqual(fake.put, {})

    def test_resume_false_restarts_byte_equal(self):
        spec = self.load(_spec())
        self.land(spec, 'A/4', self.hung())
        spec = self.load(_spec(resume=False))
        argv, fake = self.render(spec, 'A/4')
        self.assertEqual(argv, self.fresh_argv())
        self.assertEqual(fake.put, {})
        self.assertIn('"resume": false', (self.out / 'driver.log').read_text())

    def test_a_segment_that_never_failed_renders_byte_equal(self):
        spec = self.load(_spec())
        self.assertEqual(self.render(spec, 'A/4')[0], self.fresh_argv())
        self.ledger.land(dict(route='A', segment='4'), dict(verdict='PASS', checkpoints={drv._ckpt(3900): _sha(3900)}))
        self.assertEqual(self.render(spec, 'B/4')[0], self.fresh_argv('B'))


class LandUnion(Base):
    def hang_then_resume(self, **kw):
        spec = self.load(_spec())
        self.land(spec, 'A/4', self.hung())
        return spec, self.land(spec, 'A/4', self.resumed(**kw))

    def test_the_union_lands(self):
        spec, (ok, fake) = self.hang_then_resume()
        self.assertTrue(ok)
        rec = self.ledger.landed(dict(route='A', segment='4'))
        self.assertEqual(rec['verdict'], 'PASS')
        self.assertEqual(rec['resumed_from'], 2600)
        self.assertEqual(rec['steps_completed'], 1500)
        self.assertEqual(rec['arrival'], 'PASS')
        self.assertEqual(sorted(rec['checkpoints']), [drv._ckpt(n) for n in (2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300,
                                                                             3400, 3500, 3600, 3700, 3800, 3898, 3900)])
        self.assertEqual(rec['checkpoints'][drv._ckpt(2600)], _sha(2600))
        self.assertEqual(len(rec['attempts']), 1)
        # the joined chain: the hung box's lines to 2600, then the resumed box's
        key = RUN + '/A/4/chain.union.jsonl'
        self.assertEqual(rec['chain_key'], key)
        rows = [json.loads(l) for l in fake.put[key].splitlines()]
        self.assertEqual([r['step'] for r in rows], list(range(2401, 3901)))
        self.assertIsNone(rows[200]['prev'])  # step 2601: the resumed box's first line, as it wrote it
        # later segments are held to and replay against the joined chain
        argv, _ = self.render(spec, 'B/4')
        self.assertEqual(argv[argv.index('--expect-key') + 1], key)
        argv, _ = self.render(spec, 'A/5')
        self.assertEqual(argv[argv.index('--replay-chain-key') + 1], key)
        self.assertEqual(argv[argv.index('--replay-sha') + 1], _sha(3898))

    def test_the_resumed_state_must_equal_the_chain(self):
        _, (ok, _) = self.hang_then_resume(agrees=False)
        self.assertFalse(ok)
        self.assertEqual(self.ledger.landed(dict(route='A', segment='4'))['verdict'], 'FAIL-RESUME')

    def test_a_resumed_fail_halts(self):
        _, (ok, _) = self.hang_then_resume(verdict='FAIL')
        self.assertFalse(ok)
        self.assertEqual(self.ledger.landed(dict(route='A', segment='4'))['verdict'], 'FAIL')

    def test_a_box_that_started_where_no_partial_says_is_refused(self):
        spec = self.load(_spec())
        ok, _ = self.land(spec, 'A/4', self.resumed(first=2500))
        self.assertFalse(ok)
        self.assertEqual(self.ledger.landed(dict(route='A', segment='4'))['verdict'], 'FAIL-RESUME')

    def test_a_failed_union_upload_blocks_later_segments_by_name(self):
        spec = self.load(_spec())
        self.land(spec, 'A/4', self.hung())
        ok, _ = self.land(spec, 'A/4', self.resumed(), fake=FakeRun(self.r2, put_ok=False))
        self.assertTrue(ok)
        with self.assertRaises(SystemExit) as err:
            self.render(spec, 'B/4')
        self.assertIn('reland', str(err.exception))

    def test_a_second_hang_resumes_from_the_later_checkpoint(self):
        spec = self.load(_spec())
        self.land(spec, 'A/4', self.hung())
        self.r2['%s/A/4/%s' % (RUN, drv._ckpt(2700))] = SIZE
        self.land(spec, 'A/4', self.hung(start=2600, chain_to=2750, uploaded=(2700,), manifest=(2700,), arrival=None))
        part = self.ledger.partial(dict(route='A', segment='4'))
        self.assertEqual(part['resume_from'], 2700)
        self.assertEqual(sorted(part['checkpoints']), [drv._ckpt(n) for n in (2500, 2600, 2700)])
        self.assertEqual(part['arrival'], 'PASS')
        self.assertEqual([s for s, _ in drv._chain_lines(part['chain'], linked=False)], list(range(2401, 2751)))
        self.assertEqual([a['n'] for a in part['attempts']], [1, 2])
        argv, _ = self.render(spec, 'A/4')
        self.assertEqual(argv[argv.index('--steps') + 1], '1200')


class Evidence(Base):
    def test_both_attempts_are_filed(self):
        spec = self.load(_spec())
        self.land(spec, 'A/4', self.hung())
        self.land(spec, 'A/4', self.resumed())
        dest = Path(self.tmp.name) / 'filed'
        rows = fe.file_segment(self.entry(spec, 'A/4'), self.ledger.landed(dict(route='A', segment='4')), dest, 400_000, [])
        att = dest / 'A-4' / 'segment-attempt-1'
        self.assertTrue((att / 'segment' / 'chain.summary.tsv').exists())
        self.assertEqual(len((att / 'segment' / 'chain.summary.tsv').read_text().splitlines()), 300)
        self.assertTrue((att / 'arrival' / 'segment.json').exists())
        self.assertTrue((dest / 'A-4' / 'segment' / 'segment.json').exists())
        self.assertEqual(len(rows), 2)
        self.assertIn('| A/4 (attempt 1, did not land) |', rows[1])
        self.assertIn('2400 to 2699', rows[1])
        self.assertIn('2500, 2600', rows[1])


if __name__ == '__main__':
    unittest.main()
