#!/usr/bin/env python3
"""Unit tests for tools/six_lane_grid.py (pure Python, metadata only; no compile, no GPU)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_lane_grid as G  # noqa: E402

GUARD_TEXT = '''
def _check_configuration() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_A"]() and is_defined["MOJOLEARN_B"]()), "A/B exclusive"
    comptime assert not is_defined["MOJOLEARN_RETIRED"](), "retired"
    comptime S = get_defined_int["MOJOLEARN_S",0]()
    comptime assert S == 0 or S == 1 or S == 3, "S arms"
    comptime R = get_defined_int["MOJOLEARN_R",7]()
    comptime assert S != 3 or R == 7, "S3 takes no mask"
    return True
'''


def ctrl(key, define, arms, default='off', kind='switch', **extra):
    arms = {a: [G.norm_define(d) for d in ds] for a, ds in arms.items()}
    return dict(key=key, file='t', define=define, kind=kind, arms=arms, default=default,
                binding='_mojolearn_x', bits_change=False, status='new', **extra)


def algo(aid, controls, groups=(), notes=()):
    return dict(id=aid, controls=list(controls), groups=[list(g) for g in groups], not_reached=[],
                arm_restrict={}, notes=list(notes), workload_bindings=['_mojolearn_x'])


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.g = G.Guards(GUARD_TEXT)

    def test_parses_and_evaluates(self):
        self.assertEqual(self.g.assert_count, 4)
        self.assertEqual(self.g.problems([]), [])
        self.assertEqual(self.g.problems(['MOJOLEARN_A', 'MOJOLEARN_B']), ['A/B exclusive'])
        self.assertEqual(self.g.problems(['MOJOLEARN_RETIRED=1']), ['retired'])
        self.assertEqual(self.g.problems(['MOJOLEARN_S=2']), ['S arms'])
        self.assertEqual(self.g.problems(['MOJOLEARN_S=3', 'MOJOLEARN_R=1']), ['S3 takes no mask'])
        self.assertEqual(self.g.problems(['MOJOLEARN_S=3']), [])

    def test_unknown_guard_shape_raises(self):
        with self.assertRaises(ValueError):
            G.Guards('def _check_configuration() -> Bool:\n    comptime assert foo["X"](), "m"\n    return True\n')

    def test_extra_pairs(self):
        self.g.add_pair('MOJOLEARN_P', 'MOJOLEARN_Q', 'note')
        self.assertTrue(self.g.problems(['MOJOLEARN_P=1', 'MOJOLEARN_Q=1']))
        self.assertFalse(self.g.problems(['MOJOLEARN_P=1']))

    def test_real_guard_file(self):
        g = G.Guards(G.GUARDS.read_text())
        self.assertGreater(g.assert_count, 50)
        self.assertEqual(g.problems([]), [])
        self.assertTrue(g.problems(['MOJOLEARN_IDN_NEURAL_NN01=1']))  # retired
        self.assertTrue(g.problems(['MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=3', 'MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES=1']))
        self.assertTrue(g.problems(['MOJOLEARN_NN34_AFFINE_PREFIX=1', 'MOJOLEARN_IDN_M1_STATE_WINDOW=1']))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.g = G.Guards(GUARD_TEXT)
        self.controls = {
            'a': ctrl('a', 'MOJOLEARN_A', {'off': [], 'on': ['MOJOLEARN_A']}),
            'b': ctrl('b', 'MOJOLEARN_B', {'off': [], 'on': ['MOJOLEARN_B']}),
            's': ctrl('s', 'MOJOLEARN_S', {'off': [], 'one': ['MOJOLEARN_S=1'], 'three': ['MOJOLEARN_S=3']}, kind='arms'),
            'r': ctrl('r', 'MOJOLEARN_R', {'all': [], 'p': ['MOJOLEARN_R=1'], 'h': ['MOJOLEARN_R=2']}, default='all',
                      kind='arms', note='Caller-class scope of s (role tags)'),
            'w': ctrl('w', 'MOJOLEARN_W', {'4': [], '8': ['MOJOLEARN_W=8'], '16': ['MOJOLEARN_W=16']}, default='4', kind='int_sweep'),
            'dead': ctrl('dead', 'MOJOLEARN_DEAD', {'off': [], 'on': ['MOJOLEARN_DEAD']}, note='UNREACHED on the board: serial arm'),
            'off': ctrl('off', 'MOJOLEARN_OFFB', {'off': [], 'on': ['MOJOLEARN_OFFB']}, tags=['off_board']),
        }
        G.harness_problems = lambda defines: []  # isolate from the real CONFLICTS list

    def tearDown(self):
        import importlib
        importlib.reload(G)

    def run_plan(self, a, reach=None, cap=16):
        reach = reach or {k: {a['id']} for k in a['controls']}
        return G.plan_algorithm(a, self.controls, reach, self.g, cap)

    def test_singles_all_on_crosses_and_exclusions(self):
        p = self.run_plan(algo('x:t', ['a', 'b', 's', 'r', 'w', 'dead', 'off'], groups=[['s', 'r'], ['a', 'w']]))
        excluded = {e['control'] for e in p['excluded']}
        self.assertEqual(excluded, {'dead', 'off'})
        tiers = [c['tier'] for c in p['configs']]
        self.assertEqual(tiers[:8], ['single'] * 8)  # a, b, s=one, s=three, r=p(+s), r=h(+s), w=8, w=16
        singles = [c['assignment'] for c in p['configs'] if c['tier'] == 'single']
        self.assertIn({'s': 'one', 'r': 'p'}, singles)  # child carries parent representative arm
        self.assertIn({'w': '8'}, singles)
        self.assertNotIn({'w': '4'}, singles)  # default sweep value is B
        all_on = [c for c in p['configs'] if c['tier'] == 'all_on'][0]
        self.assertNotIn('b', all_on['assignment'])  # a/b exclusive: b dropped
        self.assertEqual(p['dropped_from_all_on'][0]['control'], 'b')
        for c in p['configs']:
            self.assertEqual(self.g.problems(c['defines']), [])
        crosses = [c['assignment'] for c in p['configs'] if c['tier'] == 'cross']
        self.assertNotIn({'s': 'three', 'r': 'p'}, crosses)  # guard: S3 takes no mask
        self.assertTrue(any(i['problem'] == 'S3 takes no mask' for i in G.summarize_invalid(p['invalid'])))
        ids = [c['id'] for c in p['configs']]
        self.assertEqual(len(ids), len(set(ids)))
        defs = [tuple(c['defines']) for c in p['configs']]
        self.assertEqual(len(defs), len(set(defs)))

    def test_cap_defers_in_priority_order(self):
        p = self.run_plan(algo('x:t', ['a', 'b', 's', 'r', 'w'], groups=[['s', 'r'], ['a', 'w']]), cap=3)
        self.assertEqual(len(p['configs']), 3)
        self.assertEqual([c['priority'] for c in p['configs']], [1, 2, 3])
        self.assertTrue(p['deferred'])
        self.assertTrue(any(d['tier'] == 'all_on' for d in p['deferred']))
        self.assertEqual(len(p['deferred_groups']), 1)
        self.assertGreater(p['deferred_groups'][0]['configs'], 0)

    def test_note_restricts_arms(self):
        a = algo('x:t', ['s'])
        a['arm_restrict'] = {'s': ['three']}
        p = self.run_plan(a)
        self.assertEqual([c['assignment'] for c in p['configs']], [{'s': 'three'}])

    def test_packing_respects_reach_and_guards(self):
        def cfg(aid, assign, glob=False, prio=1):
            return dict(id=aid + '.' + G.label(tuple(assign.items())), algorithm=aid, assignment=assign, priority=prio,
                        defines=G.assignment_defines(tuple(assign.items()), self.controls), bindings=['_mojolearn_x'],
                        global_reach=['g'] if glob else [], controls=list(assign))
        reach = {'a': {'x:1'}, 'w': {'x:2'}, 'b': {'x:3'}, 's': {'x:1', 'x:2'}}
        configs = [cfg('x:1', {'a': 'on'}), cfg('x:2', {'w': '8'}), cfg('x:3', {'b': 'on'}), cfg('x:2', {'s': 'one'}, prio=2),
                   cfg('x:3', {'w': '16'}, glob=True, prio=2)]
        packs = G.pack(configs, reach, self.g)
        by = {m['id']: p for p in packs for m in p['members']}
        self.assertIs(by['x:1.a=on'], by['x:2.w=8'])          # disjoint reach -> shared build
        self.assertIsNot(by['x:1.a=on'], by['x:3.b=on'])       # A/B guard exclusion -> separate
        self.assertIsNot(by['x:2.s=one'], by['x:1.a=on'])      # s reaches x:1 -> separate
        self.assertEqual(len(by['x:3.w=16']['members']), 1)    # global reach never packed
        for p in packs:
            self.assertEqual(self.g.problems(p['defines']), [])


class MappingTests(unittest.TestCase):
    INV = {'classical:pca@dataset=taxi': 's', 'classical:pca@dataset=istella': 's',
           'expanded:lda-clf@dataset=taxi': 's', 'expanded:lda-clf@dataset=taxi@input=classification-full-v1': 'v',
           'expanded:lda-clf@lda_outputs@dataset=taxi': 's', 'rf:taxi': 's', 'rf:year': 's', 'oob:rf:taxi': 's',
           'neural:lm-train-step': 's', 'neural:gemm': 's', 'expanded:cnn-clf@dataset=synthetic': 's'}

    def test_map(self):
        self.assertEqual(G.map_workloads('classical:pca', self.INV)[0], ['classical:pca@dataset=istella', 'classical:pca@dataset=taxi'])
        self.assertEqual(G.map_workloads('expanded:lda-clf', self.INV)[0], ['expanded:lda-clf@dataset=taxi@input=classification-full-v1'])
        self.assertIsNone(G.map_workloads('expanded:lda-clf@lda_outputs', self.INV)[0])  # capped original only
        self.assertEqual(G.map_workloads('trees:rf', self.INV)[0], ['rf:taxi', 'rf:year'])
        self.assertEqual(G.map_workloads('gemm:gemm', self.INV)[0], ['neural:gemm'])
        self.assertEqual(G.map_workloads('neural:cnn-clf', self.INV)[0], ['expanded:cnn-clf@dataset=synthetic'])
        self.assertIsNone(G.map_workloads('neural:mamba-forward', self.INV)[0])
        self.assertIsNone(G.map_workloads('classical:pca@svd_solver=full', self.INV)[0])

    def test_expanded_datasets_follow_the_board_lane(self):
        inv = {'expanded:theta@dataset=taxi': 's', 'expanded:theta@dataset=istella': 's',
               'expanded:lu-solve@dataset=taxi': 's', 'expanded:lu-solve@dataset=synthetic': 's',
               'expanded:ridge-cv@dataset=taxi': 's'}
        ids, notes = G.map_workloads('expanded:theta', dict(inv))
        self.assertEqual(ids, ['expanded:theta@dataset=synthetic', 'expanded:theta@dataset=taxi-hourly'])
        self.assertTrue(any('not raced by board lane' in n for n in notes))
        self.assertEqual(G.map_workloads('expanded:lu-solve', dict(inv))[0], ['expanded:lu-solve@dataset=synthetic'])
        self.assertEqual(G.map_workloads('expanded:ridge-cv', dict(inv))[0], ['expanded:ridge-cv@dataset=taxi'])
        self.assertEqual(G.map_workloads('more:arima', {'more:arima@dataset=taxi': 's'})[0], ['more:arima@dataset=synthetic'])


class RealInputTests(unittest.TestCase):
    """End-to-end on the committed grid_controls; checks invariants, not counts."""

    @classmethod
    def setUpClass(cls):
        cls.plan, cls.matrix, cls.build = G.generate()
        cls.guards = G.Guards(G.GUARDS.read_text())
        controls, *_ = G.load_controls()
        for a, b, why in G.source_exclusions(controls):
            cls.guards.add_pair(a, b, why)

    def test_cap_and_guards(self):
        for aid, a in self.plan['algorithms'].items():
            self.assertLessEqual(len(a['configs']), G.CAP, aid)
            for c in a['configs']:
                self.assertEqual(self.guards.problems(c['defines']), [], c['id'])
        for c in self.matrix['configurations']:
            self.assertEqual(self.guards.problems(c['A']['defines']), [], c['id'])
            self.assertEqual(G.harness_problems(c['A']['defines']), [], c['id'])
            self.assertTrue(set(c['grid']['effective_defines']) <= set(c['A']['defines']))
            self.assertEqual(c['B']['defines'], [])

    def test_no_stale_catalog_defines(self):
        retired = {'MOJOLEARN_IDN_NEURAL_NN01', 'MOJOLEARN_NN24_NORM_LANES8', 'MOJOLEARN_C29_TILE', 'MOJOLEARN_C52_PAIR_128'}
        for j in self.build['jobs']:
            self.assertFalse(retired & {d.split('=')[0] for d in j['defines']})
            self.assertIn(G.IDENTICAL_DEFINE, j['defines'])
            self.assertIn(j['vendor'], ('nvidia', 'amd'))

    def test_every_config_has_builds_and_cells(self):
        jobs = {j['key']: j for j in self.build['jobs']}
        cells = {}
        for cell in self.matrix['cells']:
            cells.setdefault(cell['configuration'], set()).add(cell['vendor'])
        for c in self.matrix['configurations']:
            self.assertEqual(cells[c['id']], {'nvidia', 'amd'})
            for vendor in ('nvidia', 'amd'):
                for arm in ('A', 'B'):
                    for key in c['grid']['builds'][vendor][arm]:
                        self.assertIn(dict(configuration=c['id'], arm=arm), jobs[key]['configurations'])
        s = self.plan['summary']
        self.assertLessEqual(s['builds_after_packing'], s['builds_before_packing'])
        self.assertEqual(s['builds_after_packing'], len(self.build['jobs']))
        self.assertEqual(s['aa_pairs'], 2 * s['workloads'])

    def test_nothing_silent(self):
        listed = {k for a in self.plan['algorithms'].values() for k in a['declared_controls']}
        excluded = {e['control'] for e in self.plan['excluded']}
        controls, *_ = G.load_controls()
        self.assertTrue((set(controls) - listed) <= excluded)
        for aid, a in self.plan['algorithms'].items():
            for k in a['declared_controls']:
                if k not in a['eligible_controls']:
                    self.assertIn(k, {e['control'] for e in a['excluded']}, aid)

    def test_queue_accepts_matrix(self):
        import six_lane_ab
        from six_lane_matrix_io import write_matrix
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.json.gz'
            write_matrix(path, self.matrix)
            original = six_lane_ab.check_benchmark
            six_lane_ab.check_benchmark = lambda: None  # format check only; benchmark freeze is the queue owner's gate
            try:
                args = type('A', (), dict(vendor='amd', recipes=None, matrix=path, select=None, output=Path(tmp) / 'q.json'))
                six_lane_ab.queue(args)
            finally:
                six_lane_ab.check_benchmark = original
            q = json.loads((Path(tmp) / 'q.json').read_text())
        self.assertEqual(len(q['jobs']), sum(1 for c in self.matrix['cells'] if c['vendor'] == 'amd'))
        self.assertTrue(all(j['blocked'] for j in q['jobs']))  # no recipes: every job stays blocked


# ---------------------------------------------------------------- FAST mode (Apple M3)
import six_lane_grid_fast as F  # noqa: E402


def fctrl(key, define, on=None, both=(), both_env=None, kind='switch'):
    c = F._ctrl(key, define, on if on is not None else [define], list(both), both_env or {}, ['x'], [], [], 't', [], kind=kind)
    return c


def falgo(aid, controls, groups=(), binding_of=None, family='algos'):
    return dict(id=aid, family=family, lane=aid.split(':', 1)[1], controls=list(controls), groups=[list(g) for g in groups],
                not_reached=[], arm_restrict={}, notes=[], binding_of=binding_of or {k: 'x_prep' for k in controls},
                workloads=[aid + '@dataset=taxi', aid + '@dataset=istella'])


class FastGuardTests(unittest.TestCase):
    def test_env_with_build_define_refused(self):
        g = F.FastGuards(env_names={'MOJOLEARN_E'})
        self.assertTrue(g.problems(['MOJOLEARN_E=1', 'MOJOLEARN_D=1']))
        self.assertEqual(g.problems(['MOJOLEARN_E=1']), [])
        self.assertEqual(g.problems(['MOJOLEARN_D=1']), [])

    def test_one_binding_per_line(self):
        g = F.FastGuards(binding_of={'MOJOLEARN_A': 'arima', 'MOJOLEARN_B': 'tsa'})
        self.assertTrue(g.problems(['MOJOLEARN_A=1', 'MOJOLEARN_B=1']))
        self.assertEqual(g.problems(['MOJOLEARN_A=1']), [])

    def test_card_pairs(self):
        g = F.FastGuards(pairs=[('MOJOLEARN_G05', 'MOJOLEARN_PACK', 'card')])
        self.assertTrue(g.problems(['MOJOLEARN_G05=1', 'MOJOLEARN_PACK=1']))


class FastPlanTests(unittest.TestCase):
    def test_all_on_drops_env_and_crosses_stay_pure(self):
        controls = {k: fctrl(k, 'MOJOLEARN_' + k) for k in ('A', 'B')}
        controls['E'] = fctrl('E', 'MOJOLEARN_E', on=['MOJOLEARN_E=1'], kind='env')
        algo = falgo('algos:lane', ['A', 'B', 'E'], groups=[['A', 'B']])
        reach = {k: {'algos:lane'} for k in controls}
        guards = F.FastGuards(env_names={'MOJOLEARN_E'})
        plan = G.plan_algorithm(algo, controls, reach, guards)
        tiers = [c['tier'] for c in plan['configs']]
        self.assertEqual(tiers.count('single'), 3)
        self.assertEqual([d['control'] for d in plan['dropped_from_all_on']], ['E'])
        for c in plan['configs']:
            self.assertEqual(guards.problems(c['defines']), [], c['id'])

    def test_incumbent_keeps_prerequisite_unless_member(self):
        controls = {'NB': fctrl('NB', 'MOJOLEARN_NB'), 'P03': fctrl('P03', 'MOJOLEARN_P03', on=['MOJOLEARN_NB', 'MOJOLEARN_P03'], both=['MOJOLEARN_NB'])}
        algo = falgo('algos:nb', ['NB', 'P03'])
        reach = {k: {'algos:nb'} for k in controls}
        plan = G.plan_algorithm(algo, controls, reach, F.FastGuards())
        by = {c['tier'] + ':' + ','.join(c['assignment']): F.finish_config(c, algo, controls, set()) for c in plan['configs']}
        self.assertEqual(by['single:P03']['incumbent_defines'], ['MOJOLEARN_NB=1'])
        self.assertEqual(by['single:P03']['candidate_build_defines'], ['MOJOLEARN_NB=1', 'MOJOLEARN_P03=1'])
        self.assertEqual(by['all_on:NB,P03']['incumbent_defines'], [])  # NB is a member: candidate only

    def test_env_config_and_tools(self):
        controls = {'E': fctrl('E', 'MOJOLEARN_E', on=['MOJOLEARN_E=1'], kind='env'), 'T': fctrl('T', 'MOJOLEARN_T')}
        algo = falgo('trees:gbdt-x', ['E', 'T'], binding_of={'E': 'gbdt', 'T': 'gbdt'}, family='trees')
        reach = {k: {'trees:gbdt-x'} for k in controls}
        plan = G.plan_algorithm(algo, controls, reach, F.FastGuards(env_names={'MOJOLEARN_E'}))
        done = {','.join(c['assignment']): F.finish_config(c, algo, controls, {'MOJOLEARN_E'}) for c in plan['configs']}
        self.assertEqual(done['E']['tool'], 'afc_env')
        self.assertEqual(done['E']['candidate_env'], {'MOJOLEARN_E': '1'})
        self.assertEqual(done['T']['tool'], 'aft')

    def test_flags(self):
        self.assertEqual(F.dflags(['MOJOLEARN_A=1', 'MOJOLEARN_K=4']), '-D MOJOLEARN_A -D MOJOLEARN_K=4')
        self.assertEqual(F.envstr({'B': '1', 'A': '2'}), 'A=2 B=1')

    def test_binding_resolution(self):
        c = fctrl('C', 'MOJOLEARN_C')
        c['bindings'], c['paths'] = ['metrics', 'x_decomp'], ['p.mojo']
        orig = F.reaches
        try:
            F.reaches = lambda b, paths: True
            self.assertEqual(F.resolve_binding(c, 'umap')[0], 'metrics')       # LANE_PY_BINDINGS picks one
            self.assertIsNone(F.resolve_binding(c, 'no-such-lane')[0])         # ambiguous: refused, not guessed
            F.reaches = lambda b, paths: b == 'x_decomp'
            self.assertEqual(F.resolve_binding(c, 'no-such-lane')[0], 'x_decomp')
            F.reaches = lambda b, paths: False
            self.assertIsNone(F.resolve_binding(c, 'no-such-lane')[0])
        finally:
            F.reaches = orig
        e = fctrl('E', 'MOJOLEARN_E', on=['MOJOLEARN_E=1'], kind='env')
        e['bindings'] = []
        self.assertEqual(F.resolve_binding(e, 'autoarima')[0], 'arima')


class FastRealInputTests(unittest.TestCase):
    """End-to-end on the committed FAST sources; invariants, not counts."""

    @classmethod
    def setUpClass(cls):
        cls.plan, cls.matrix, cls.build, cls.queue = F.generate_fast()

    def test_no_identical_anything(self):
        defines = [d for j in self.build['jobs'] for d in j['defines'] + list(j['environment'])]
        defines += [d for c in self.matrix['configurations'] for arm in 'AB' for d in c[arm]['defines'] + list(c[arm]['environment'])]
        self.assertFalse([d for d in defines if 'NUMERIC_IDENTICAL' in d])
        self.assertFalse([q['tag'] for q in self.queue if 'NUMERIC_IDENTICAL' in q['command']])
        self.assertTrue(self.plan['verdict_rule']['identity'].startswith('NOT_REQUIRED'))
        for cell in self.matrix['cells']:
            self.assertEqual((cell['vendor'], cell['identity'], cell['mode']), ('apple', 'NOT_REQUIRED', 'fast'))

    def test_queue_lines(self):
        kinds = [q['kind'] for q in self.queue]
        self.assertEqual(kinds, sorted(kinds))  # every A/A ('aa') before every A/B ('ab')
        tags = [q['tag'] for q in self.queue]
        self.assertEqual(len(tags), len(set(tags)))
        for q in self.queue:
            self.assertTrue(q['line'].startswith("lq add m3 CMD lane/apple-fast"), q['line'])
            cmd = q['command']
            if 'aft_ab.sh' in cmd:
                self.assertRegex(cmd, r'aft_ab\.sh \S+ \S+ \S+ 1 "')        # pairs 1
            elif 'afc_ab_def.sh' in cmd:
                self.assertRegex(cmd, r'afc_ab_def\.sh \S+ \S+ \S+ \S+ 1 1 "')  # reps 1, rounds 1
            else:
                self.assertRegex(cmd, r'afc_ab\.sh \S+ \S+ \S+ 1 1 "')
            if q['kind'] == 'ab' and q['tool'] != 'afc_env':
                self.assertIn('main-' + q['binding'] + '/A.so', cmd)  # FAST main restored after the line
        cells = {(c['configuration'], c['workload_id']) for c in self.matrix['cells']}
        self.assertEqual(cells, {(q['configuration'], q['workload_id']) for q in self.queue if q['kind'] == 'ab'})

    def test_cap_guards_and_arms(self):
        for aid, a in self.plan['algorithms'].items():
            self.assertLessEqual(len(a['configs']), G.CAP, aid)
            self.assertEqual(len({c['binding'] for c in a['configs']} - {None}) <= len(a['bindings']), True)
        for c in self.matrix['configurations']:
            self.assertEqual(c['candidate_arm'], 'B')
            self.assertTrue(set(c['A']['defines']) <= set(c['B']['defines']) or c['grid']['kind'] == 'env', c['id'])
            added = {d.split('=')[0] for d in set(c['B']['defines']) - set(c['A']['defines'])}
            self.assertFalse({d for d in added if d.endswith('_OFF')}, c['id'])  # promoted _OFF arms are never the candidate
            self.assertEqual(G.harness_problems(c['B']['defines']), [], c['id'])

    def test_builds(self):
        s = self.plan['summary']
        self.assertEqual(s['builds_after_packing'], len(self.build['jobs']))
        self.assertLessEqual(s['builds_after_packing'], s['builds_before_packing'])
        for j in self.build['jobs']:
            self.assertEqual((j['vendor'], j['mode']), ('apple', 'fast'))

    def test_nothing_silent(self):
        controls, *_ = F.load_all()
        placed = {k for a in self.plan['algorithms'].values() for k in a['declared_controls']}
        excluded = {e['control'] for e in self.plan['excluded']}
        self.assertEqual(set(controls) - placed - excluded, set())

    def test_committed_outputs_fresh(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(F.main_fast(check=True), 0)
            self.assertEqual(G.main(['--check']), 0)  # the IDENTICAL outputs are untouched by FAST mode


class CrossModeTests(unittest.TestCase):
    """--crosses pairwise / pairwise+triples / auto and the --survivors phase 2 on a synthetic group."""

    def setUp(self):
        self.g = G.Guards(GUARD_TEXT)
        self.controls = {
            'p': ctrl('p', 'MOJOLEARN_P', {'off': [], 'on': ['MOJOLEARN_P']}),
            'q': ctrl('q', 'MOJOLEARN_Q', {'off': [], '1': ['MOJOLEARN_Q=1'], '2': ['MOJOLEARN_Q=2']}, kind='arms'),
            't': ctrl('t', 'MOJOLEARN_T', {'off': [], 'a': ['MOJOLEARN_T=1'], 'b': ['MOJOLEARN_T=2'], 'c': ['MOJOLEARN_T=3']}, kind='arms'),
            'u': ctrl('u', 'MOJOLEARN_U', {'off': [], 'on': ['MOJOLEARN_U']}),
        }
        G.harness_problems = lambda defines: []

    def tearDown(self):
        import importlib
        importlib.reload(G)

    def plan(self, controls=('p', 'q', 't'), **kw):
        a = algo('x:t', controls, groups=[['p', 'q', 't']])
        return G.plan_algorithm(a, self.controls, {k: {'x:t'} for k in controls}, self.g, kw.pop('cap', 0), **kw)

    @staticmethod
    def tiers(p):
        out = {}
        for c in p['configs']:
            out[c['tier']] = out.get(c['tier'], 0) + 1
        return out

    def test_pairwise_counts(self):
        # pairs: p x q (1*2) + p x t (1*3) + q x t (2*3) = 11; singles 1 + 2 + 3 = 6; all-on 1
        self.assertEqual(self.tiers(self.plan(crosses='pairwise')), {'single': 6, 'all_on': 1, 'cross': 11})
        # triples: 1*2*3 = 6, after every pair
        p = self.plan(crosses='pairwise+triples')
        self.assertEqual(self.tiers(p), {'single': 6, 'all_on': 1, 'cross': 11, 'triple': 5})  # one triple == all-on
        order = [c['tier'] for c in p['configs']]
        self.assertLess(max(i for i, t in enumerate(order) if t == 'cross'), min(i for i, t in enumerate(order) if t == 'triple'))
        full = self.plan(crosses='full')
        self.assertEqual(self.tiers(full)['cross'], 2 * 3 * 4 - 1 - 6 - 1)  # whole product minus singles, all-on

    def test_pairwise_cap_cuts_configs_not_groups(self):
        p = self.plan(crosses='pairwise', cap=8)  # 6 singles + all-on + 1 pair
        self.assertEqual(len(p['configs']), 8)
        self.assertEqual(p['configs'][-1]['tier'], 'cross')
        self.assertEqual(sum(g['configs'] for g in p['deferred_groups']), 10)
        full = self.plan(crosses='full', cap=8)
        self.assertEqual(self.tiers(full).get('cross', 0), 0)  # group-atomic: the whole group is deferred

    def test_auto_factorial_and_fallback(self):
        p = self.plan(crosses='auto')
        self.assertEqual(p['regime'], 'factorial')
        self.assertEqual(p['factorial_product'], 23)
        self.assertEqual(len(p['configs']), 23)  # every combination exactly once
        self.assertEqual(self.tiers(p), {'single': 6, 'all_on': 1, 'factorial': 16})
        self.assertEqual(len({tuple(c['defines']) for c in p['configs']}), 23)
        q = self.plan(crosses='auto', full_budget=22, cap=16)
        self.assertEqual(q['regime'], 'pairwise')
        self.assertEqual(len(q['configs']), 16)
        self.assertFalse(any(c['tier'] == 'factorial' for c in q['configs']))
        big = self.plan(crosses='auto', cap=16)
        self.assertEqual(len(big['configs']), 23)  # the cap does not apply to factorial algorithms

    def phase1(self):
        def row(cid, assign, verdict, ratio=None):
            return frozenset(assign.items()), dict(configuration=cid, verdict=verdict, tier='single', combined_ratio=ratio)
        return dict([row('s1', {'p': 'on'}, 'FASTER', 0.9), row('s2', {'q': '1'}, 'NEUTRAL', 1.0), row('s3', {'q': '2'}, 'SLOWER', 1.2),
                     row('s4', {'t': 'a'}, 'FASTER', 0.8), row('s5', {'t': 'b'}, 'UNMEASURED'), row('s6', {'u': 'on'}, 'FASTER', 0.7),
                     row('c1', {'p': 'on', 'q': '1'}, 'NEUTRAL')])

    def test_survivors_filter_reuse_and_across(self):
        p = self.plan(controls=('p', 'q', 't', 'u'), crosses='pairwise', phase1=self.phase1())
        states = {(e['control'], e['arm']): e['state'] for e in p['survivors']['arms']}
        self.assertEqual(states, {('p', 'on'): 'survived', ('q', '1'): 'survived', ('q', '2'): 'dropped', ('t', 'a'): 'survived',
                                  ('t', 'b'): 'not_measured', ('t', 'c'): 'not_measured', ('u', 'on'): 'survived'})
        self.assertEqual(p['survivors']['faster'], {'p': ['on'], 't': ['a'], 'u': ['on']})
        got = [(c['tier'], c['assignment']) for c in p['configs']]
        self.assertEqual(got, [('cross', {'p': 'on', 't': 'a'}), ('cross', {'q': '1', 't': 'a'}),
                               ('all_survivors', {'p': 'on', 'q': '1', 't': 'a', 'u': 'on'}),
                               ('cross_across', {'p': 'on', 'u': 'on'}), ('cross_across', {'t': 'a', 'u': 'on'})])
        self.assertEqual([r['phase1_configuration'] for r in p['reused']], ['c1'])  # measured in phase 1: not re-emitted
        self.assertFalse(any(c['tier'] == 'single' for c in p['configs']))
        t = self.plan(controls=('p', 'q', 't', 'u'), crosses='pairwise+triples', phase1=self.phase1())
        self.assertIn(('triple', {'p': 'on', 'q': '1', 't': 'a'}), [(c['tier'], c['assignment']) for c in t['configs']])

    def test_no_survivors_says_why(self):
        p = self.plan(phase1={}, crosses='pairwise')
        self.assertEqual(p['configs'], [])
        self.assertEqual(p['survivors']['counts'], {'not_measured': 6})
        self.assertTrue(all(e['verdict'] == 'NOT_IN_DECISIONS' for e in p['survivors']['arms']))

    def test_default_is_full(self):
        a = algo('x:t', ['p', 'q', 't'], groups=[['p', 'q', 't']])
        reach = {k: {'x:t'} for k in 'pqt'}
        self.assertEqual(G.plan_algorithm(a, self.controls, reach, self.g, 16),
                         G.plan_algorithm(a, self.controls, reach, self.g, 16, crosses='full'))
        self.assertNotIn('regime', G.plan_algorithm(a, self.controls, reach, self.g, 16))


class DefaultOutputsTests(unittest.TestCase):
    def test_identical_and_fast_check_pass_with_defaults(self):
        import contextlib
        import io
        import six_lane_grid_fast as F
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(G.main(['--check']), 0)
            self.assertEqual(G.main(['--mode', 'fast', '--check']), 0)
            self.assertEqual(F.main_fast(check=True), 0)

    def test_extended_options_refused_in_fast_mode(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            G.main(['--mode', 'fast', '--crosses', 'pairwise'])


if __name__ == '__main__':
    unittest.main()
