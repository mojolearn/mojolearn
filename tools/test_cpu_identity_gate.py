"""Negative controls for CPU gate sharding; no bindings or accelerators needed."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cpu_identity_gate_check as gate


def record(lanes=('gemm-pinned', 'kde')):
    return dict(vendor='cpu-test', commit='test', mode='identical', repeats=2,
                complete=True, fixtures={'base': 'fixture', 'odd': 'odd-fixture'},
                host=dict(column='cpu', cpu_model='test', families={
                    'binding': dict(column='cpu', sha256='abc', size=3)}),
                package={}, cells={f'{lane}/{fx}': dict(verdict='STABLE', hashes=['abc', 'abc'])
                                   for lane in lanes for fx in ('base', 'odd')})


def oracle_failure_record(lanes=('gemm-pinned', 'kde')):
    value = record(lanes)
    value['host']['families']['binding']['sabotage'] = True
    next(iter(value['cells'].values())).update(
        verdict='DIVERGENT', oracle_errors=['measured bytes disagree with oracle'] * 2)
    return value


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def column(self, value, sabotage=False):
        path = self.root / 'column.json'
        path.write_text(json.dumps(value))
        return gate.do_column(SimpleNamespace(json=str(path), covered='gemm-pinned,kde',
                                             binding='binding', commit='test', sabotage=sabotage))

    def test_complete_column_passes(self):
        self.assertEqual(self.column(record()), 0)

    def test_native_oracle_failure_is_only_accepted_in_sabotage_arm(self):
        value = oracle_failure_record()
        self.assertEqual(self.column(value), 1)
        self.assertEqual(self.column(value, sabotage=True), 0)
        for mutation in ('readback', 'oracle', 'unstable', 'incomplete', 'one-repeat', 'property'):
            bad = copy.deepcopy(value)
            cell = next(iter(bad['cells'].values()))
            if mutation == 'readback':
                bad['host']['families']['binding']['sabotage'] = False
            elif mutation == 'oracle':
                cell['oracle_errors'] = []
            elif mutation == 'unstable':
                cell['hashes'][1] = 'different'
            elif mutation == 'incomplete':
                bad['complete'] = False
            elif mutation == 'one-repeat':
                bad['repeats'] = 1
            else:
                cell['batch_verdict'] = 'REFUSED'
            self.assertEqual(self.column(bad, sabotage=True), 1, mutation)

    def test_native_property_violation_is_not_a_production_pass(self):
        value = record()
        value['host']['families']['binding']['sabotage'] = True
        next(iter(value['cells'].values())).update(
            rlpair=['RLPAIR_MOVED:measured 0x3ce54c86 vs 0x3ce54c84'] * 2,
            rlpair_verdict='RLPAIR_MOVED')
        self.assertEqual(self.column(value), 1)
        self.assertEqual(self.column(value, sabotage=True), 0)

    def test_oracle_shard_exit_is_not_a_general_exit_code_waiver(self):
        for sabotage, code, kind, expected in ((True, 1, 'oracle', 0),
                                              (False, 1, 'oracle', 1),
                                              (True, 7, 'oracle', 1),
                                              (True, 1, 'stable', 1),
                                              (True, 1, 'missing', 1)):
            def worker(cmd, stdout, stderr):
                lanes = cmd[cmd.index('--lanes') + 1].split(',')
                value = oracle_failure_record(lanes) if kind != 'stable' else record(lanes)
                if kind == 'missing':
                    value['cells'].pop(next(reversed(value['cells'])))
                Path(cmd[cmd.index('--json') + 1]).write_text(json.dumps(value))
                return SimpleNamespace(poll=lambda: code)

            args = SimpleNamespace(lanes='gemm-pinned,kde', shards=2, jobs=2,
                                   json=str(self.root / 'merged.json'), extra=['--repeats', '2'],
                                   heartbeat=120, sabotage=sabotage)
            with patch.object(gate.subprocess, 'Popen', side_effect=worker), \
                 patch.object(gate.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                 patch.object(gate.time, 'sleep'):
                self.assertEqual(gate.do_run_column(args), expected, (sabotage, code, kind))

    def test_stable_training_cannot_hide_a_failed_probe(self):
        for field, verdict in (("infer_verdict", "MOVED"),
                               ("model_verdict", "RELOAD-MOVED"),
                               ("batch_verdict", "BATCH_MOVED"),
                               ("rlpair_verdict", "REFUSED"),
                               ("stepfull_verdict", "MOVED")):
            value = record()
            value['cells']['kde/base'][field] = verdict
            self.assertEqual(self.column(value), 1, (field, verdict))
            self.assertIn(field, self.output.getvalue())

    def test_explicitly_inapplicable_probe_is_allowed(self):
        value = record()
        value['cells']['kde/base']['batch_verdict'] = 'N/A'
        self.assertEqual(self.column(value), 0)

    def test_missing_fixture_fails_even_with_complete_flag(self):
        value = record()
        del value['cells']['kde/odd']
        self.assertEqual(self.column(value), 1)
        self.assertIn('covered cell kde/odd is missing', self.output.getvalue())

    def test_refusal_moved_empty_and_incomplete_fail(self):
        for mutation in ('REFUSED', 'MOVED', 'empty', 'incomplete'):
            value = record()
            if mutation == 'empty':
                value['cells'] = {}
            elif mutation == 'incomplete':
                value['complete'] = False
            else:
                value['cells']['kde/base']['verdict'] = mutation
            self.assertEqual(self.column(value), 1, mutation)

    def test_balanced_shards_preserve_all_lanes_once(self):
        lanes = ['kde', 'iforest', 'dbscan-weighted', 'gemm-pinned', 'new-lane']
        shards, _ = gate.shard_lanes(lanes, 3)
        self.assertCountEqual(sum(shards, []), lanes)
        self.assertEqual(len(shards), 3)
        self.assertTrue(all(shards))

    def test_orchestrator_propagates_failure_and_removes_stale_output(self):
        # Fake workers finish immediately; local tests never run compute in parallel.
        for failure in (False, True):
            out = self.root / 'out.json'
            out.write_text('stale success')
            seen = []

            def worker(cmd, stdout, stderr):
                seen.append(cmd)
                path = Path(cmd[cmd.index('--json') + 1])
                path.write_text(json.dumps(record(cmd[cmd.index('--lanes') + 1].split(','))))
                return SimpleNamespace(poll=lambda: 7 if failure else 0)

            def merge(cmd):
                self.assertFalse(out.exists())
                out.write_text('{}')
                return SimpleNamespace(returncode=0)

            args = SimpleNamespace(lanes='gemm-pinned,kde', shards=2, jobs=2,
                                   json=str(out), extra=['--repeats', '1'], heartbeat=120)
            with patch.object(gate.subprocess, 'Popen', side_effect=worker), \
                 patch.object(gate.subprocess, 'run', side_effect=merge), \
                 patch.object(gate.time, 'sleep'):
                self.assertEqual(gate.do_run_column(args), int(failure))
            self.assertEqual(len(seen), 2)
            self.assertTrue(all(cmd[-2:] == ['--repeats', '1'] for cmd in seen))

    def test_resume_preserves_parts_but_never_reuses_merged_success(self):
        out = self.root / 'resume.json'
        out.write_text('old merged success')
        part = self.root / 'resume.part0.json'
        part.write_text('checkpoint to validate')
        log = self.root / 'resume.part0.log'
        log.write_text('previous attempt\n')
        seen = []

        def worker(cmd, stdout, stderr):
            path = Path(cmd[cmd.index('--json') + 1])
            seen.append(cmd)
            if path == part:
                self.assertEqual(path.read_text(), 'checkpoint to validate')
                self.assertIn('--resume', cmd)
            else:
                self.assertNotIn('--resume', cmd)
            path.write_text(json.dumps(record(cmd[cmd.index('--lanes') + 1].split(','))))
            stdout.write('new attempt\n')
            return SimpleNamespace(poll=lambda: 0)

        def merge(cmd):
            self.assertFalse(out.exists())
            out.write_text('{}')
            return SimpleNamespace(returncode=0)

        args = SimpleNamespace(lanes='gemm-pinned,kde', shards=2, jobs=1,
            json=str(out), extra=['--repeats', '2'], heartbeat=120, resume=True)
        with patch.object(gate.subprocess, 'Popen', side_effect=worker), \
             patch.object(gate.subprocess, 'run', side_effect=merge), \
             patch.object(gate.time, 'sleep'):
            self.assertEqual(gate.do_run_column(args), 0)
        self.assertEqual(len(seen), 2)
        self.assertEqual(log.read_text(), 'previous attempt\nnew attempt\n')

    def test_forwarded_resume_is_rejected_before_deleting_evidence(self):
        out = self.root / 'preserve.json'
        out.write_text('evidence')
        args = SimpleNamespace(extra=['--resume'], json=str(out))
        self.assertEqual(gate.do_run_column(args), 2)
        self.assertEqual(out.read_text(), 'evidence')

    def test_merge_rejects_machine_build_fixture_and_commit_changes(self):
        spec = importlib.util.spec_from_file_location('identity_gate_test', Path(__file__).with_name('identity_break.py'))
        identity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(identity)
        left, right = self.root / 'left.json', self.root / 'right.json'
        left.write_text(json.dumps(record(['gemm-pinned'])))
        original = record(['kde'])
        right.write_text(json.dumps(original))
        out = self.root / 'merged.json'
        identity.merge([str(left), str(right)], str(out))
        self.assertEqual(len(json.loads(out.read_text())['cells']), 4)
        for key in ('machine', 'build', 'fixture', 'commit'):
            value = copy.deepcopy(original)
            if key == 'machine':
                value['host']['cpu_model'] = 'other'
            elif key == 'build':
                value['host']['families']['binding']['sha256'] = 'different'
            elif key == 'fixture':
                value['fixtures']['base'] = 'different'
            else:
                value['commit'] = 'other'
            right.write_text(json.dumps(value))
            with self.assertRaisesRegex(SystemExit, 'REFUSING --merge'):
                identity.merge([str(left), str(right)], str(out))


def load_identity():
    spec = importlib.util.spec_from_file_location('identity_owed_test', Path(__file__).with_name('identity_break.py'))
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    return identity


def cell(train='t', infer=None, batch=None, repeats=2):
    """A STABLE train cell; infer and batch are a hash, 'n/a' or None (key absent)."""
    c = dict(verdict='STABLE', hashes=[train] * repeats, parts=[dict(p=train)] * repeats)
    for part, value in (('infer', infer), ('batch', batch)):
        if value is None:
            continue
        if value == 'n/a':
            c[part], c[f'{part}_verdict'] = ['n/a:no-predict'] * repeats, 'N/A'
        else:
            c[part], c[f'{part}_verdict'] = [value] * repeats, 'STABLE'
    return c


def column(vendor, cells, cpu=False):
    j = dict(vendor=vendor, commit='test', mode='identical', repeats=2, complete=True,
             fixtures={'base': 'b', 'odd': 'o'}, package={}, cells=cells)
    if cpu:
        j['host'] = dict(column='cpu', cpu_model='test', families={})
    return j


class OwedTests(unittest.TestCase):
    """The OWED rule of identity_break --diff --owed-json and its sabotage
    arm, cpu_identity_gate_check.py owed (lane/cpu-gate-owed-cells)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)
        self.identity = load_identity()

    def columns(self):
        """Three GPU records with train hashes and n/a predict cells, plus a
        CPU column that hashes infer and batch: the k-means predict shape."""
        gpu = {f'km/{fx}': cell('t' + fx, 'n/a', 'n/a') for fx in ('base', 'odd')}
        cpu = {f'km/{fx}': cell('t' + fx, 'i' + fx, 'b' + fx) for fx in ('base', 'odd')}
        return dict(apple=column('apple', copy.deepcopy(gpu)), nvidia=column('nvidia', copy.deepcopy(gpu)),
                    amd=column('amd', copy.deepcopy(gpu)), cpu=column('cpu', cpu, cpu=True))

    def diff(self, cols, owed=True, lanes=('km',)):
        paths = []
        for name, j in cols.items():
            p = self.root / f'{name}.json'
            p.write_text(json.dumps(j))
            paths.append(str(p))
        self.owed_path = self.root / 'owed_cells.json'
        if self.owed_path.exists():
            self.owed_path.unlink()
        return self.identity.diff(paths, 4, list(lanes), str(self.owed_path) if owed else None)

    def owed_keys(self):
        return sorted((o['lane'], o['fixture'], o['part'], tuple(o['missing']))
                      for o in json.loads(self.owed_path.read_text())['owed'])

    def test_without_owed_the_four_column_check_fails(self):
        self.assertEqual(self.diff(self.columns(), owed=False), 1)
        self.assertIn('REQUIRE FAIL km/base infer', self.output.getvalue())

    def test_owed_passes_and_lists_exactly_the_missing_cells(self):
        cols = self.columns()
        del cols['amd']['cells']['km/odd']            # a lane cell not in that record
        del cols['nvidia']['cells']['km/base']['batch']      # a part key absent
        del cols['nvidia']['cells']['km/base']['batch_verdict']
        apple = cols['apple']['cells']['km/base']            # Apple has the batch hash, and agrees
        apple['batch'], apple['batch_verdict'] = ['bbase'] * 2, 'STABLE'
        self.assertEqual(self.diff(cols), 0, self.output.getvalue())
        out = self.output.getvalue()
        self.assertIn('summary: IDENTICAL=1, OWED=1', out)   # km/odd train rests on 3
        self.assertIn('| km/base                      | IDENTICAL x4', out)
        want = [('km', 'base', 'batch', ('nvidia', 'amd')),
                ('km', 'base', 'infer', ('apple', 'nvidia', 'amd')),
                ('km', 'odd', 'batch', ('apple', 'nvidia', 'amd')),
                ('km', 'odd', 'infer', ('apple', 'nvidia', 'amd')),
                ('km', 'odd', 'train', ('amd',))]
        self.assertEqual(self.owed_keys(), sorted(want))

    def test_recorded_cell_that_differs_is_still_divergent(self):
        for part in ('train', 'infer'):
            cols = self.columns()
            c = cols['apple']['cells']['km/base']
            if part == 'train':
                c['hashes'] = ['other'] * 2
            else:
                c['infer'], c['infer_verdict'] = ['other'] * 2, 'STABLE'
            self.assertEqual(self.diff(cols), 1, part)
            self.assertRegex(self.output.getvalue(), r'\| km/base +\| (infer +\| )?DIVERGENT')
            self.assertNotIn(('km', 'base', part), [k[:3] for k in self.owed_keys()])

    def test_apple_column_that_agrees_keeps_owed(self):
        cols = self.columns()
        c = cols['apple']['cells']['km/base']
        c['infer'], c['infer_verdict'] = ['ibase'] * 2, 'STABLE'
        self.assertEqual(self.diff(cols), 0)
        self.assertIn(('km', 'base', 'infer', ('nvidia', 'amd')), self.owed_keys())

    def test_cell_missing_from_every_gpu_column_but_unstable_on_cpu_fails(self):
        for mutation in ('moved', 'one-repeat'):
            cols = self.columns()
            c = cols['cpu']['cells']['km/base']
            if mutation == 'moved':
                c['infer'], c['infer_verdict'] = ['ibase', 'other'], 'MOVED'
            else:
                c['infer'] = ['ibase']
            self.assertEqual(self.diff(cols), 1, mutation)
            self.assertNotIn(('km', 'base', 'infer'), [k[:3] for k in self.owed_keys()])

    def test_column_recording_refused_for_a_cpu_hashed_cell_fails(self):
        for where in ('cell', 'part'):
            cols = self.columns()
            c = cols['nvidia']['cells']['km/base']
            if where == 'cell':
                c['verdict'] = 'REFUSED'
            else:
                c['infer'], c['infer_verdict'] = [None, None], 'REFUSED'
            self.assertEqual(self.diff(cols), 1, where)
            self.assertIn('not OWED: column nvidia REFUSED' if where == 'cell' else 'not OWED: column nvidia reads REFUSED',
                          self.output.getvalue())

    def test_owed_needs_require_columns(self):
        with self.assertRaisesRegex(SystemExit, 'needs --require-columns'):
            self.identity.diff([], 0, None, str(self.root / 'x.json'))

    def sabotage_check(self, mutate):
        cols = self.columns()
        self.assertEqual(self.diff(cols), 0)
        prod = self.root / 'cpu.json'
        sab_cells = {k: cell('s' + k, 's-infer' + k, 'BATCH_MOVED:x' + k, repeats=1) for k in cols['cpu']['cells']}
        mutate(sab_cells)
        sab = self.root / 'cpu-sab.json'
        sab.write_text(json.dumps(column('cpu', sab_cells, cpu=True)))
        return gate.do_owed(SimpleNamespace(owed_json=str(self.owed_path), production=str(prod), sabotage=str(sab)))

    def test_owed_cells_that_move_under_sabotage_pass(self):
        self.assertEqual(self.sabotage_check(lambda cells: None), 0)
        self.assertIn('owed verdict OK (4 of 4', self.output.getvalue())

    def test_owed_cell_that_does_not_move_under_sabotage_fails(self):
        def unmoved(cells):
            cells['km/odd']['batch'] = ['bodd']
        self.assertEqual(self.sabotage_check(unmoved), 1)
        self.assertIn('km/odd batch: DID NOT MOVE', self.output.getvalue())

    def test_owed_cell_refused_or_absent_under_sabotage_fails(self):
        for mutation in ('refused', 'absent'):
            def mutate(cells):
                if mutation == 'refused':
                    cells['km/base']['verdict'] = 'REFUSED'
                else:
                    del cells['km/base']
            self.assertEqual(self.sabotage_check(mutate), 1, mutation)
            self.assertIn('km/base infer: the sabotage column has no value', self.output.getvalue())


class BuildListTests(unittest.TestCase):
    """cpu_identity_gate_check.py build-list: the workflow's build lists
    against python/mojolearn/host_surface.py (2026-09-15)."""

    def setUp(self):
        self.manifest = gate.load_manifest()
        self.full = self.manifest['families']()
        self.binding = {f['family']: f['binding'] for f in self.manifest['FAMILIES']}

    def errors(self, families, sabotage=None, readback=None, scope='full'):
        sabotage = families if sabotage is None else sabotage
        readback = [self.binding.get(f, f) for f in families] if readback is None else readback
        return gate.build_list_errors(self.manifest, list(families), list(sabotage), list(readback), scope)

    def test_manifest_lists_pass(self):
        self.assertEqual(self.errors(self.full), [])
        self.assertEqual(self.errors(self.manifest['wheel_families'](), scope='routine'), [])

    def test_the_old_hand_list_fails_on_the_seven_shipped_bindings(self):
        old = ['byte_lm', 'forest', 'tokenizer'] + self.manifest['routed_families']()
        errors = self.errors(old)
        for b in ('_mojolearn_neural_host', '_mojolearn_mixture_infer_host', '_mojolearn_gp_infer_host',
                  '_mojolearn_hdbscan_infer_host', '_mojolearn_embedding_infer_host',
                  '_mojolearn_ivf_search_host', '_mojolearn_forecast_host'):
            self.assertIn(f'the production build leaves out {b}', '\n'.join(errors))
        self.assertTrue(self.errors(old, scope='routine'))

    def test_one_binding_removed_fails(self):
        for scope, families in (('full', self.full), ('routine', self.manifest['wheel_families']())):
            less = [f for f in families if f != 'forecast']
            for where in ('production', 'sabotage'):
                errors = (self.errors(less, sabotage=families, readback=[self.binding[f] for f in families], scope=scope)
                          if where == 'production' else self.errors(families, sabotage=less, scope=scope))
                self.assertIn(f'the {where} build leaves out _mojolearn_forecast_host', '\n'.join(errors), (scope, where))

    def test_readback_must_name_the_build(self):
        readback = [self.binding[f] for f in self.full if f != 'neural']
        self.assertIn('read-back list', '\n'.join(self.errors(self.full, readback=readback)))

    def test_unknown_family_fails(self):
        self.assertIn('not a family the manifest declares', '\n'.join(self.errors(self.full + ['nonesuch'])))

    def test_cli_exit_codes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ok = gate.main(['build-list', '--families', ' '.join(self.full), '--sabotage-families',
                            ' '.join(self.full), '--readback', ','.join(self.binding[f] for f in self.full),
                            '--scope', 'full'])
            less = [f for f in self.full if f != 'ivf_search']
            bad = gate.main(['build-list', '--families', ' '.join(less), '--sabotage-families', ' '.join(less),
                             '--readback', ','.join(self.binding[f] for f in less), '--scope', 'full'])
        self.assertEqual((ok, bad), (0, 1), out.getvalue())
        self.assertIn('leaves out _mojolearn_ivf_search_host', out.getvalue())


class SabotageDefinesTests(unittest.TestCase):
    """The sabotage host set's per-family defines and the CTR table lanes'
    saved models, read from python/mojolearn/host_surface.py
    (lane/cpu-verifier-gaps-7, 2026-09-15)."""

    def setUp(self):
        self.manifest = gate.load_manifest()

    def test_every_family_carries_the_host_define_and_four_carry_their_own(self):
        own = {}
        for family in self.manifest['families']():
            defines = self.manifest['sabotage_build_defines'](family).split()
            self.assertEqual(defines[:2], ['-D', 'MOJOLEARN_HOST_SABOTAGE=1'], family)
            if len(defines) > 2:
                own[family] = defines[2:]
        # The tokenizer carries TWO since lane/laneless-public-classes
        # (2026-09-19): `bpe-trainer` became a covered lane and it never
        # encodes, so the encoder arm left its cell byte-identical
        # (6ed8b49585df3d85 clean and sabotaged, M4, base, --repeats 2). The
        # trainer's own arm moves it (-> f5172d25e6499662), so the gate's set
        # builds with both. A negative control that leaves a covered lane
        # where it found it is not a negative control for that lane.
        # The linalg family joined them on lane/laneless-public-classes
        # (2026-09-19), for the same reason one step further out: the family
        # define moves `gemm_oracle`'s leaf and `gemm_int8_oracle`'s
        # dequantized cell and reaches NOTHING in the four low-bit CONVERSION
        # seams (contract L-1..L-6), so the `lowbit-conversions` cell read the
        # clean hash on all nine fixtures under it (0ff2da2cf430c48a clean and
        # sabotaged, M4, --repeats 2). MOJOLEARN_LOWBIT_CONVERT_SABOTAGE moves
        # it (-> d58be6e94de78728), and the gate's set now builds with both.
        # gbdt carries three since lane/catboost-parity (2026-09-19): the
        # border types' own arm (select_borders' middle border, which no
        # GreedyLogSum lane reaches), Ordered boosting's and the quantile
        # constant's.
        self.assertEqual(own, {'forest': ['-D', 'MOJOLEARN_GBDT_CTR_HOST_SABOTAGE=1'],
                               'gbdt': ['-D', 'MOJOLEARN_BORDER_TYPES_SABOTAGE=1',
                                        '-D', 'MOJOLEARN_ORDERED_SABOTAGE=1',
                                        '-D', 'MOJOLEARN_SAMPLE_QUANTILE_SABOTAGE=1'],
                               'linalg': ['-D', 'MOJOLEARN_LOWBIT_CONVERT_SABOTAGE=1'],
                               'tokenizer': ['-D', 'MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1',
                                             '-D', 'MOJOLEARN_BPE_TRAINER_SABOTAGE=1']})

    def test_lanes_resting_on_their_own_define_are_covered(self):
        covered = self.manifest['covered_lanes']()
        for family in self.manifest['GATE_SABOTAGE_OWN_DEFINES']:
            lanes = self.manifest['family'](family)['training_lanes']
            self.assertTrue(lanes, f'{family} carries an own sabotage define but covers no lane')
            self.assertTrue(set(lanes) <= set(covered), family)

    def test_ctr_models_directory_is_the_lanes_directory(self):
        d = Path(__file__).resolve().parents[1] / self.manifest['GBDT_CTR_MODELS_DIR']
        for lane in self.manifest['GBDT_CTR_MODEL_LANES']:
            self.assertTrue(sorted(d.glob(f'{lane}.*.npz')), f'no saved model for {lane} under {d}')

    def test_ctr_saved_model_cpu_model_part_is_na(self):
        """A CPU column that LOADED the GPU column's file hashes no model
        part: the sabotage arm could never move the file's hash, and the owed
        check would fail on it."""
        text = Path(__file__).with_name('identity_break.py').read_text()
        body = text.split('def _probe_saved_host(', 1)[1].split('\ndef ', 1)[0]
        self.assertIn('"n/a:gpu-saved-file', body)
        self.assertNotIn('_hfile(path)', body)
        self.assertIn('if reload != infer:', body)

def load_classical_gate():
    spec = importlib.util.spec_from_file_location(
        'classical_host_gate', Path(__file__).resolve().parent / 'classical_host_gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SabotageVerdictTests(unittest.TestCase):
    """classical_host_gate's --expect-mismatch rules (lane/ties-sabotage,
    2026-09-15). The fixture shape is the one the old IVF arms left: the
    lane moved on base and stayed EQUAL on ties."""

    def setUp(self):
        self.verdict = load_classical_gate().sabotage_verdict
        self.moved = ['ivf/base', 'radius/base', 'radius/ties']
        self.unmoved = ['ivf/ties']

    def test_every_lane_passes_one_unmoved_fixture(self):
        verdict, code, _ = self.verdict(False, self.moved, self.unmoved, every_lane=True)
        self.assertEqual((verdict, code), ('EXPECTED MISMATCH SEEN', 0))

    def test_every_fixture_fails_one_unmoved_fixture(self):
        verdict, code, lines = self.verdict(False, self.moved, self.unmoved, every_fixture=True)
        self.assertEqual(code, 1)
        self.assertEqual(verdict, 'SABOTAGE NOT CAUGHT ON FIXTURES ivf/ties')
        self.assertIn('check ivf/ties did not move', '\n'.join(lines))

    def test_every_fixture_passes_when_all_move(self):
        verdict, code, lines = self.verdict(False, self.moved + self.unmoved, [], every_fixture=True)
        self.assertEqual((verdict, code, lines), ('EXPECTED MISMATCH SEEN', 0, []))

    def test_lane_rule_only_keeps_the_looser_rule_by_name(self):
        verdict, code, _ = self.verdict(False, self.moved, self.unmoved, every_fixture=True, lane_rule_only=['ivf'])
        self.assertEqual((verdict, code), ('EXPECTED MISMATCH SEEN', 0))
        # the exemption names a lane, and does not excuse a lane that moved nowhere
        verdict, code, _ = self.verdict(False, ['radius/base'], ['ivf/base', 'ivf/ties'],
                                        every_fixture=True, lane_rule_only=['ivf'])
        self.assertEqual((verdict, code), ('SABOTAGE NOT CAUGHT ON LANES ivf', 1))
        verdict, code, _ = self.verdict(False, self.moved, self.unmoved, every_fixture=True, lane_rule_only=['radius'])
        self.assertEqual(code, 1)

    def test_nothing_moved_is_never_a_catch(self):
        verdict, code, _ = self.verdict(True, [], ['ivf/base'], every_fixture=True, lane_rule_only=['ivf'])
        self.assertEqual(code, 1)
        self.assertNotEqual(verdict, 'EXPECTED MISMATCH SEEN')

    def test_gpu_column_disagreement_alone_is_not_a_fault_catch(self):
        # do_check also folds optional GPU column disagreements into
        # verdict_ok. Those cannot prove that this CPU output moved.
        for options in ({}, {'every_lane': True}, {'every_fixture': True}):
            with self.subTest(options=options):
                verdict, code, _ = self.verdict(False, [], ['ivf/base'], **options)
                self.assertEqual(code, 1)
                self.assertNotEqual(verdict, 'EXPECTED MISMATCH SEEN')

    def test_empty_fault_evidence_cannot_pass(self):
        for options in ({}, {'every_lane': True}, {'every_fixture': True}):
            with self.subTest(options=options):
                self.assertEqual(self.verdict(False, [], [], **options)[1], 1)


class ClassicalColumnFaultTests(unittest.TestCase):
    """Exercise the gate with real fixture/reference files and mocked inference."""

    def check(self, *, expect_mismatch, cpu_hash):
        # This suite also runs before any package/native binding is built.
        mojolearn = SimpleNamespace(vendor=lambda: 'cpu')
        host = SimpleNamespace(
            host_model=lambda _: SimpleNamespace(estimator='LinearRegression'),
            binary_path=lambda: 'mock', binary_paths=lambda: [])
        gate = load_classical_gate()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'ols' / 'base'
            directory.mkdir(parents=True)
            (directory / 'model.npz').write_bytes(b'saved model')
            (directory / 'fixture.json').write_text(json.dumps(dict(
                lane='ols', kind='base', probe_rows=256, x_sha256='input')))
            prediction = dict(sha256='prediction', dtype='float32', shape=[256])
            (directory / 'expected.json').write_text(json.dumps(dict(
                status='RECORDED', estimator='LinearRegression', x_sha256='input',
                model_sha256=gate.sha256_bytes(b'saved model'),
                predictions=dict(identity_hash='a' * 16, predict=prediction))))
            column = Path(tmp) / 'gpu.json'
            column.write_text(json.dumps(dict(vendor='cuda', cells={
                'ols/base': dict(infer=['b' * 16, 'b' * 16])})))
            args = SimpleNamespace(package_root=None, fixture_dir=[directory],
                gpu_column=[str(column)], expect_mismatch=expect_mismatch,
                every_lane=False, every_fixture=False, lane_rule_only=[], report=None)
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.dict('sys.modules', {
                    'mojolearn': mojolearn, 'mojolearn._classical_host': host}))
                stack.enter_context(patch.object(gate, 'package_root'))
                stack.enter_context(patch.object(gate, 'identity_tool',
                    return_value=SimpleNamespace(FIXTURES=['base'])))
                stack.enter_context(patch.object(gate, 'held_out', return_value=(None, 'input')))
                stack.enter_context(patch.object(gate, 'digests_for', return_value=dict(
                    identity_hash=cpu_hash, predict=prediction, seconds=0)))
                stack.enter_context(patch.object(gate, 'host_info', return_value={}))
                stack.enter_context(patch.object(gate, 'git_commit', return_value='test'))
                return gate.do_check(args)

    def test_column_disagreement_still_fails_clean_check(self):
        self.assertEqual(self.check(expect_mismatch=False, cpu_hash='a' * 16), 1)

    def test_column_disagreement_does_not_pass_fault_check(self):
        self.assertEqual(self.check(expect_mismatch=True, cpu_hash='a' * 16), 1)

    def test_changed_cpu_output_still_passes_fault_check(self):
        self.assertEqual(self.check(expect_mismatch=True, cpu_hash='c' * 16), 0)


if __name__ == '__main__':
    unittest.main()
