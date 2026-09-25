"""A one-box segment stopped before its results came home resumes from R2
alone. The case is T3's A/4 on DigitalOcean (2026-09-25 14:15Z): stopped at
step 2659 with checkpoints 2500 and 2600 in R2 but no chain fetched, so the
driver said "no chain came home" and the segment restarted from 2400.

tools/lm_segment.py now PUTs the chain and the manifest as they stand to
`chain.progress.jsonl` and `manifest.progress.tsv` after the first step
past each checkpoint (tools/lm_segment_leg.py mints the two URLs for a
one-box body), and tools/lm_run_driver.py's record_partial reads them from
R2 when no chain came home. CPU only: every subprocess (presigning, curl,
the renderer) is mocked and the trainer is the NumPy stand-in of
test_lm_segment_controls.py; nothing is rendered, uploaded or rented."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.argv = [sys.argv[0]]
HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
sys.path.insert(0, str(TOOLS))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load('test_lm_run_driver_resume_base', HERE / 'test_lm_run_driver_resume.py')
C = _load('test_lm_segment_controls_base', HERE / 'test_lm_segment_controls.py')
drv, seg = R.drv, C.seg
import lm_segment_leg as leg  # noqa: E402


def setUpModule():
    C.setUpModule()


def tearDownModule():
    C.tearDownModule()


class FakeR2(R.FakeRun):
    """FakeRun plus whole-object GETs (`curl -o PATH URL`) served from `objects`."""

    def __init__(self, sizes=None, objects=None, **kw):
        super().__init__(sizes, **kw)
        self.objects = dict(objects or {})

    def __call__(self, argv, **kw):
        if argv and argv[0] == 'curl' and '-o' in argv and '-r' not in argv:
            self.calls.append(list(argv))
            key = argv[-1].split('https://get/', 1)[1]
            if key not in self.objects:
                return mock.Mock(returncode=22, stdout='', stderr='404')
            Path(argv[argv.index('-o') + 1]).write_text(self.objects[key])
            return mock.Mock(returncode=0, stdout='', stderr='')
        return super().__call__(argv, **kw)


def _manifest(steps):
    return ''.join('%s\t%d\t%s\n' % (drv._ckpt(n), R.SIZE, R._sha(n)) for n in steps)


class ResumeFromR2(R.Base):
    """The driver: no chain came home, R2 holds the progress copies."""

    def nothing_home(self):
        """What the runner leaves when the box was stopped before the fetch."""
        res = self.out / 'legs' / 'A-4' / 'leg-do-1'
        res.mkdir(parents=True)
        (res / 'droplet_id.txt').write_text('123456\n')
        return res

    def progress(self, chain_to=2601, manifest=(2500, 2600)):
        """R2 as the stopped box left it: the chain and manifest PUT after step
        2601 (the first step past checkpoint 2600), and both checkpoints."""
        pre = '%s/A/4/' % R.RUN
        return {pre + 'chain.progress.jsonl': R._chain(range(2401, chain_to + 1), 'amd-A-4'),
                pre + 'manifest.progress.tsv': _manifest(manifest)}

    def test_a_stopped_box_resumes_from_2600_by_r2_alone(self):
        spec = self.load(R._spec())
        res = self.nothing_home()
        ok, fake = self.land(spec, 'A/4', res, FakeR2(self.r2, self.progress()))
        self.assertIsNone(ok)
        part = self.ledger.partial(dict(route='A', segment='4'))
        self.assertIsNotNone(part, 'no partial entry: the R2 progress chain was not read')
        self.assertEqual(part['resume_from'], 2600)
        self.assertEqual(part['checkpoints'], {drv._ckpt(2500): R._sha(2500), drv._ckpt(2600): R._sha(2600)})
        self.assertEqual(part['arrival'], 'PASS')
        self.assertEqual(part['chain_steps'], [2401, 2601])
        self.assertEqual(Path(part['chain']).read_text(), self.progress()['%s/A/4/chain.progress.jsonl' % R.RUN])
        a = part['attempts'][0]
        self.assertTrue(a['from_r2'])
        self.assertTrue((Path(a['dir']) / 'segment' / 'chain.jsonl').exists())
        self.assertTrue((Path(a['dir']) / 'droplet_id.txt').exists())
        # the results moved aside; the scratch copy left nothing a later attempt could read as its own
        self.assertFalse(res.exists())
        self.assertEqual(list((self.out / 'legs' / 'A-4').rglob('r2-progress*')), [])
        # the next body resumes from its own 2600, held to the partial chain it uploads
        argv, fake2 = self.render(spec, 'A/4', FakeR2(self.r2))
        self.assertEqual(argv[argv.index('--from') + 1], drv._ckpt(2600))
        self.assertEqual(argv[argv.index('--steps') + 1], '1300')
        self.assertEqual(argv[argv.index('--from-sha') + 1], R._sha(2600))
        self.assertEqual(argv[argv.index('--expect-key') + 1], '%s/A/4/partial.chain.jsonl' % R.RUN)
        self.assertEqual(fake2.put['%s/A/4/partial.chain.jsonl' % R.RUN], Path(part['chain']).read_text())

    def test_a_checkpoint_r2_lacks_does_not_count_even_unchecked(self):
        spec = self.load(R._spec())
        spec['resume_check_r2'] = False    # the log is not there to vouch for the upload: R2 must
        r2 = {k: v for k, v in self.r2.items() if not k.endswith(drv._ckpt(2600))}
        self.land(spec, 'A/4', self.nothing_home(), FakeR2(r2, self.progress()))
        part = self.ledger.partial(dict(route='A', segment='4'))
        self.assertEqual(part['resume_from'], 2500)
        self.assertIn(drv._ckpt(2600), part['attempts'][0]['refused'])

    def test_a_checkpoint_with_no_line_after_it_is_not_the_resume_point(self):
        # the box died during step 2601: the last progress PUT was after step 2501
        self.land(self.load(R._spec()), 'A/4', self.nothing_home(),
                  FakeR2(self.r2, self.progress(chain_to=2501, manifest=(2500,))))
        self.assertEqual(self.ledger.partial(dict(route='A', segment='4'))['resume_from'], 2500)

    def test_no_progress_in_r2_restarts_as_before(self):
        ok, _ = self.land(self.load(R._spec()), 'A/4', self.nothing_home(), FakeR2(self.r2))
        self.assertIsNone(ok)
        self.assertIsNone(self.ledger.partial(dict(route='A', segment='4')))

    def test_a_chain_that_came_home_is_read_first(self):
        # a stale R2 progress chain never overrides the box's own chain
        stale = self.progress(chain_to=2501, manifest=(2500,))
        self.land(self.load(R._spec()), 'A/4', self.hung(), FakeR2(self.r2, stale))
        part = self.ledger.partial(dict(route='A', segment='4'))
        self.assertEqual((part['resume_from'], part['chain_steps']), (2600, [2401, 2699]))
        self.assertNotIn('from_r2', part['attempts'][0])


class ProgressUploads(unittest.TestCase):
    """The box: what lm_segment.py PUTs, and when, with the stand-in trainer."""

    def setUp(self):
        self.h = C.Harness('test_control_none_passes_and_is_stamped')
        self.h.setUp()
        self.addCleanup(self.h.tearDown)
        self.dir = self.h.dir
        self.saved_save = seg.save_checkpoint

        def save(trainer, directory, *, manifest, upload):
            name = seg.checkpoint_name(trainer.step_)
            path = Path(directory) / name
            path.write_bytes(trainer.export_raw()['parameters'].tobytes())
            digest = seg._sha_file(path)
            manifest.pin(name, path.stat().st_size, digest)
            upload(name, path)
            return dict(step=trainer.step_, file=name, bytes=path.stat().st_size, sha256=digest, save_seconds=0.0)
        seg.save_checkpoint = save
        self.addCleanup(setattr, seg, 'save_checkpoint', self.saved_save)

    def run_with(self, name, keys):
        """Steps 4..10 from checkpoint 3 (checkpoint_every 5: checkpoints 5 and 10)."""
        urls = self.dir / (name + '.urls.json')
        urls.write_text(json.dumps({k: 'https://put/' + k for k in keys}))
        puts = []

        def run(argv, **kw):
            if argv[0] == 'curl' and '-T' in argv:
                puts.append((argv[-1].split('https://put/', 1)[1], Path(argv[argv.index('-T') + 1]).read_bytes()))
            return mock.Mock(returncode=0, stdout='', stderr='')
        out = self.dir / name
        with mock.patch.object(seg.subprocess, 'run', side_effect=run):
            rc = seg.main(['run', '--recipe', str(self.h.recipe), '--tokens', str(self.dir), '--from', str(self.dir / 'c.blm'),
                           '--steps', '7', '--out', str(out), '--upload-urls', str(urls)])
        self.assertEqual(rc, 0)
        return puts, out

    def keys(self):
        recipe = seg.load_recipe(self.h.recipe)
        return seg.expected_keys(recipe, 3, 7, None)

    def test_the_chain_to_the_step_after_each_checkpoint_goes_up(self):
        puts, out = self.run_with('p', self.keys() + list(seg.PROGRESS_KEYS))
        names = [n for n, _ in puts]
        self.assertEqual(names, ['ckpt_00000005.blm', 'manifest.progress.tsv', 'chain.progress.jsonl',
                                 'ckpt_00000010.blm', 'chain.jsonl', 'segment.json', 'manifest.tsv'])
        chain = dict(puts)['chain.progress.jsonl'].decode()
        self.assertEqual([json.loads(l)['step'] for l in chain.splitlines()], [4, 5, 6])
        final = (out / 'chain.jsonl').read_bytes()
        self.assertTrue(final.startswith(chain.encode()), 'the progress chain is a prefix of the final chain, byte for byte')
        self.assertEqual(dict(puts)['chain.jsonl'], final)
        self.assertEqual(dict(puts)['manifest.progress.tsv'].decode().split('\t')[0], 'ckpt_00000005.blm')

    def test_bits_and_chain_do_not_change(self):
        with_progress, a = self.run_with('a', self.keys() + list(seg.PROGRESS_KEYS))
        without, b = self.run_with('b', self.keys())
        def untimed(path):   # every field but the wall-clock ones (and `prev`, which hashes them)
            return [{k: v for k, v in json.loads(l).items() if k not in ('seconds', 'hash_seconds', 'prev')}
                    for l in path.read_text().splitlines()]
        self.assertEqual(untimed(a / 'chain.jsonl'), untimed(b / 'chain.jsonl'))
        self.assertEqual(len(untimed(a / 'chain.jsonl')), 7)
        ckpts = lambda d: [r for r in (d / 'manifest.tsv').read_text().splitlines() if r.startswith('ckpt_')]
        self.assertEqual(ckpts(a), ckpts(b))
        self.assertEqual(len(ckpts(a)), 2)
        # a body rendered before the progress keys existed: no refusal, no progress PUT
        self.assertEqual([n for n, _ in without], ['ckpt_00000005.blm', 'ckpt_00000010.blm', 'chain.jsonl',
                                                   'segment.json', 'manifest.tsv'])

    def test_a_failed_progress_put_never_stops_the_segment(self):
        urls = self.dir / 'f.urls.json'
        urls.write_text(json.dumps({k: 'https://put/' + k for k in self.keys() + list(seg.PROGRESS_KEYS)}))

        def run(argv, **kw):
            bad = argv[0] == 'curl' and 'progress' in argv[-1]
            return mock.Mock(returncode=22 if bad else 0, stdout='', stderr='403' if bad else '')
        with mock.patch.object(seg.subprocess, 'run', side_effect=run):
            rc = seg.main(['run', '--recipe', str(self.h.recipe), '--tokens', str(self.dir), '--from', str(self.dir / 'c.blm'),
                           '--steps', '7', '--out', str(self.dir / 'f'), '--upload-urls', str(urls)])
        self.assertEqual(rc, 0)
        self.assertIn('the segment trains on', (self.dir / 'f' / 'log.txt').read_text())


class RenderedUrls(unittest.TestCase):
    """The renderer mints the two progress PUTs for a one-box body only."""

    def render(self, mode, tmp):
        recipe = Path(tmp) / 'recipe.json'
        C._recipe(recipe)
        body = Path(tmp) / ('%s.sh' % mode)
        argv = ['render', '--run', 'runs/t3/x', '--recipe', str(recipe), '--arm', 'amd', '--mode', mode, '--devices', '0',
                '--route', 'A', '--segment', '4', '--label', 'l', '--steps', '7', '--from', 'ckpt_00000003.blm',
                '--from-sha', 'ab' * 32, '--out', str(body)]
        if mode != 'live-worker':
            argv += ['--boundary', '10']
        with mock.patch.object(leg, '_store', side_effect=lambda verb, key, s: 'https://%s/%s' % (verb, key)):
            leg.main(argv)
        return json.loads(body.with_suffix('.ledger.json').read_text())['uploads']

    def test_one_box_body_carries_the_progress_keys(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(set(seg.PROGRESS_KEYS) <= set(self.render('one', tmp)))
            self.assertFalse(set(seg.PROGRESS_KEYS) & set(self.render('live-coordinator', tmp)))
            self.assertFalse(set(seg.PROGRESS_KEYS) & set(self.render('live-worker', tmp)))


if __name__ == '__main__':
    unittest.main()
