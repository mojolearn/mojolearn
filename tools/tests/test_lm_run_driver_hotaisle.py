"""tools/lm_run_driver.py: Hot Aisle as an AMD box source for whole segments.

The hotaisle branch passes the segment's lease and cap (`_hotaisle_lease`),
renders the body again with `hotaisle_devices` (default 0,1), rents the 2x
MI300X VM with one body on both GPUs, reads the leg's busy words, and the AMD
vendor class is per box source, so a DigitalOcean AMD segment and a Hot Aisle
AMD segment start together under --parallel 2. CPU only; every subprocess is
mocked, nothing is rendered or rented."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location(
    'lm_run_driver', Path(__file__).resolve().parents[1] / 'lm_run_driver.py')
drv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drv)


def _spec(b1=None, a2=None, **top):
    spec = dict(run='runs/t3/x', recipe='r.json', recipe_key='runs/t3/x/recipe.json', tokens_stage='tok',
                lease_minutes=120, dollar_cap=10, nvidia_devices='0,1', amd_devices='0', amd_providers=['do', 'hotaisle'],
                routes=dict(A=[dict(segment='1', vendor='nvidia', steps=1000),
                               dict(segment='2', vendor='amd', steps=1000, **(a2 or {}))],
                            B=[dict(segment='1', vendor='amd', steps=1000, **(b1 or {}))]))
    spec.update(top)
    return spec


def _entry(spec, key):
    return next(e for e in drv.segment_plan(spec) if '%s/%s' % (e['route'], e['segment']) == key)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'out'
        self.out.mkdir()
        self.spec_path = Path(self.tmp.name) / 'spec.json'
        self.ledger = drv.Ledger(self.out)
        ck = {'ckpt_%08d.blm' % n: ('%02x' % (n % 251)) * 32 for n in (0, 998, 1000)}
        self.ledger.land(dict(route='A', segment='1'), dict(verdict='PASS', checkpoints=ck))

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, spec):
        self.spec_path.write_text(json.dumps(spec))
        return drv.load_spec(self.spec_path)


class HotaisleRental(Base):
    def _start(self, spec, key, returncode=0, log_text=''):
        """drv._start for one segment; returns every subprocess argv and env."""
        calls = []

        def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, check=None):
            calls.append((cmd, env))
            if stdout is not None and log_text:
                stdout.write(log_text)
                stdout.flush()
            return mock.Mock(returncode=returncode if cmd[0] == 'bash' else 0)

        with mock.patch.object(drv.subprocess, 'run', side_effect=fake_run), mock.patch.object(drv.time, 'sleep'):
            got = drv._start(spec, _entry(spec, key), self.out, self.ledger)
        return got, calls

    def test_b1_on_hotaisle_rents_the_whole_segment_on_both_gpus(self):
        spec = self.load(_spec(b1=dict(provider='hotaisle', lease_minutes=1800, dollar_cap=250)))
        (res, rc), calls = self._start(spec, 'B/1')
        renders = [(c, e) for c, e in calls if 'lm_segment_leg.py' in ' '.join(map(str, c))]
        legs = [(c, e) for c, e in calls if c[0] == 'bash']
        # the spec-wide body, then the body again with the Hot Aisle devices
        self.assertEqual([c[c.index('--devices') + 1] for c, _ in renders], ['0', '0,1'])
        self.assertEqual(renders[1][0][renders[1][0].index('--out') + 1], str(self.out / 'bodies' / 'B-1-hotaisle.sh'))
        (argv, env), = legs
        self.assertEqual(argv, ['bash', 'tools/hotaisle_leg.sh', 'amd', '--rent', '--skip-gates', '--spec', '2gpu',
                                '--one-body', '--segment-lease', '1830', '--dollar-cap', '250'])
        self.assertEqual(env['MOJOLEARN_GEMM_LEG_EXTRA'], str(self.out / 'bodies' / 'B-1-hotaisle.sh'))
        self.assertEqual((env['MOJOLEARN_HOTAISLE_SPEC'], env['MOJOLEARN_HOTAISLE_GPU_ONLY'], env['MOJOLEARN_HOTAISLE_LANE']),
                         ('2gpu', '1', 'lm-B-1'))
        self.assertEqual((env['MOJOLEARN_HOTAISLE_STOCK_WAIT_MINUTES'], env['MOJOLEARN_HOTAISLE_SLOT_WAIT_MINUTES']), ('5', '5'))
        self.assertEqual(env['MOJOLEARN_GPU_ARCHS'], 'gfx942')
        self.assertEqual((res, rc), (self.out / 'legs' / 'B-1' / 'leg-hotaisle-1', 0))

    def test_hotaisle_keys_win(self):
        spec = self.load(_spec(b1=dict(provider='hotaisle', lease_minutes=1800), amd_devices='0,1,2,3',
                               hotaisle_devices='0', hotaisle_extra_minutes=45, hotaisle_dollar_cap=300))
        _, calls = self._start(spec, 'B/1')
        renders = [c for c, _ in calls if 'lm_segment_leg.py' in ' '.join(map(str, c))]
        self.assertEqual([c[c.index('--devices') + 1] for c in renders], ['0,1,2,3', '0'])
        leg = [c for c, _ in calls if c[0] == 'bash'][0]
        self.assertEqual(leg[-4:], ['--segment-lease', '1845', '--dollar-cap', '300'])

    def test_same_devices_render_once(self):
        spec = self.load(_spec(b1=dict(provider='hotaisle'), amd_devices='0,1'))
        _, calls = self._start(spec, 'B/1')
        renders = [c for c, _ in calls if 'lm_segment_leg.py' in ' '.join(map(str, c))]
        self.assertEqual(len(renders), 1)
        env = [e for c, e in calls if c[0] == 'bash'][0]
        self.assertEqual(env['MOJOLEARN_GEMM_LEG_EXTRA'], str(self.out / 'bodies' / 'B-1.sh'))

    def test_a_short_lease_is_the_minimum_hour(self):
        self.assertEqual(drv._hotaisle_lease(dict(lease_minutes=20, hotaisle_extra_minutes=10), {}), ['--minutes', '60'])
        self.assertEqual(drv._hotaisle_lease(dict(lease_minutes=120, dollar_cap=10), dict(dollar_cap=40)),
                         ['--segment-lease', '150', '--dollar-cap', '40'])

    def test_an_explicit_provider_rents_only_there(self):
        spec = self.load(_spec(b1=dict(provider='do')))
        _, calls = self._start(spec, 'B/1')
        self.assertEqual([c[1] for c, _ in calls if c[0] == 'bash'], ['tools/do_extra_leg.sh'])

    def test_busy_hotaisle_walks_on_to_do(self):
        seen = []

        def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, check=None):
            if cmd[0] == 'bash':
                seen.append(cmd[1])
                if cmd[1] == 'tools/hotaisle_leg.sh':
                    stdout.write('REFUSED: the 2gpu 2x MI300X spec showed no stock for 5 minutes. Nothing was created.\n')
                    stdout.flush()
                    return mock.Mock(returncode=3)
            return mock.Mock(returncode=0)

        spec = self.load(_spec(amd_providers=['hotaisle', 'do']))
        with mock.patch.object(drv.subprocess, 'run', side_effect=fake_run):
            got = drv._start(spec, _entry(spec, 'B/1'), self.out, self.ledger)
        self.assertEqual(seen, ['tools/hotaisle_leg.sh', 'tools/do_extra_leg.sh'])
        self.assertEqual(got, (self.out / 'legs' / 'B-1' / 'leg-do-1', 0))

    def test_over_the_cap_skips_hotaisle(self):
        seen = []

        def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, check=None):
            if cmd[0] == 'bash':
                seen.append(cmd[1])
                if cmd[1] == 'tools/hotaisle_leg.sh':
                    stdout.write('segment lease REFUSED: 1830 minutes of the 2x MI300X VM at $5.98/h is up to $182.39, '
                                 'above the --dollar-cap of $10; nothing was created\n')
                    stdout.flush()
                    return mock.Mock(returncode=2)
            return mock.Mock(returncode=0)

        spec = self.load(_spec(amd_providers=['hotaisle', 'do'], lease_minutes=1800))
        with mock.patch.object(drv.subprocess, 'run', side_effect=fake_run):
            got = drv._start(spec, _entry(spec, 'B/1'), self.out, self.ledger)
        self.assertEqual(seen, ['tools/hotaisle_leg.sh', 'tools/do_extra_leg.sh'])
        self.assertEqual(got[1], 0)

    def test_a_hotaisle_failure_that_is_not_busy_is_returned(self):
        spec = self.load(_spec(b1=dict(provider='hotaisle')))
        (res, rc), calls = self._start(spec, 'B/1', returncode=6,
                                       log_text='--one-body on the 2x MI300X VM: rocminfo on the host shows 1 GPU agents, not 2. Deleting.\n')
        self.assertEqual((res, rc), (self.out / 'legs' / 'B-1' / 'leg-hotaisle-1', 6))
        self.assertEqual(len([c for c, _ in calls if c[0] == 'bash']), 1)


class BusyPhrases(unittest.TestCase):
    def test_hotaisle_and_do_words(self):
        for t in ('REFUSED: the 2gpu 2x MI300X spec showed no stock for 5 minutes (last HTTP 200, found, quantity 0).',
                  'REFUSED: no slot freed in 5 minutes. Nothing was created.',
                  'REFUSED: balance $44.65 is below $184.40, the whole lease (1800 min at 598 cents/h = $179.40) plus the $5.00 floor.',
                  'create REFUSED by the API (HTTP 402) and no new VM appears. Nothing is billing.',
                  'REFUSING to create mojolearn-extra-amd: ONE GPU droplet at a time on this account'):
            self.assertTrue(drv._amd_busy(t), t)
        for t in ('segment lease REFUSED: 1830 minutes ... above the --dollar-cap of $10; nothing was created',
                  'THE ON-BOX WATCHDOG COULD NOT BE VERIFIED (pid, second session, ref, GET 200 or description).',
                  'the bundle upload failed'):
            self.assertFalse(drv._amd_busy(t), t)


class VendorClass(Base):
    def test_classes(self):
        spec = self.load(_spec(b1=dict(provider='hotaisle'), a2=dict(provider='do')))
        self.assertEqual(drv._vendor_class(_entry(spec, 'B/1'), spec), {'amd:hotaisle'})
        self.assertEqual(drv._vendor_class(_entry(spec, 'A/2'), spec), {'amd:do'})
        self.assertEqual(drv._vendor_class(_entry(spec, 'A/1'), spec), {'nvidia'})
        self.assertIn('B/1 amd     steps 0..1000 from A/1/ckpt_00000000.blm expect A/1/chain.jsonl  provider hotaisle',
                      self._plan_lines())
        walk = self.load(_spec())
        self.assertEqual(drv._vendor_class(_entry(walk, 'B/1'), walk), {'amd:do', 'amd:hotaisle'})

    def _plan_lines(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            drv.cmd_plan(argparse.Namespace(spec=str(self.spec_path)))
        return buf.getvalue().splitlines()

    def _first_pass(self, spec):
        """One scheduling pass of cmd_run under --parallel 2: which segments it starts."""
        self.load(spec)

        class Stop(Exception):
            pass

        def stop(_):
            raise Stop()

        with mock.patch.object(drv, '_start', return_value=(self.out / 'none', 0)), \
                mock.patch.object(drv.time, 'sleep', side_effect=stop):
            with self.assertRaises(Stop):
                drv.cmd_run(argparse.Namespace(spec=str(self.spec_path), out=str(self.out), parallel=2))
        return [line.split('starting ')[1].split(' ')[0] for line in (self.out / 'driver.log').read_text().splitlines()
                if 'starting ' in line]

    def test_do_and_hotaisle_segments_start_together(self):
        self.assertEqual(self._first_pass(_spec(b1=dict(provider='hotaisle'), a2=dict(provider='do'))), ['A/2', 'B/1'])

    def test_two_segments_on_one_provider_do_not(self):
        self.assertEqual(self._first_pass(_spec(b1=dict(provider='do'), a2=dict(provider='do'))), ['A/2'])

    def test_a_walking_segment_holds_every_provider(self):
        self.assertEqual(self._first_pass(_spec(b1=dict(provider='hotaisle'))), ['A/2'])
        (self.out / 'driver.log').unlink()
        self.assertEqual(self._first_pass(_spec()), ['A/2'])


if __name__ == '__main__':
    unittest.main()
