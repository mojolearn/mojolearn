#!/usr/bin/env python3
"""Unit tests for tools/six_lane_grid_lq.py (pure Python on a synthetic plan and synthetic lq results lines;
nothing is built, raced or sent to a box) and tools/six_lane_grid_lq_feed.sh (against a fake lq)."""
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_lane_grid_decide as D  # noqa: E402
import six_lane_grid_bb as BBW  # noqa: E402
import six_lane_grid_lq as G  # noqa: E402

LANES = {'ridge-cv': ['taxi', 'istella'], 'lasso-cv': ['taxi', 'istella'], 'theta': ['taxi-hourly', 'synthetic'],
         'qda': ['taxi', 'istella'], 'sgd-reg': ['taxi', 'istella']}
BOARD = dict(races={('classical', 'kmeans', 'taxi'), ('trees', 'rf', 'taxi'), ('trees', 'rf', 'istella'),
                    ('trees', 'gbdt-multiclass', 'taxi'), ('neural', 'lm-train-step', 'bytes')},
             neural_data={'lm-train-step': 'bytes'}, tree_task={'gbdt-multiclass': {'taxi': 'taximc'}},
             tree_lanes=['rf', 'gbdt-multiclass'], board_datasets=['taxi', 'istella'])


def wl(algo, ds, extra=''):
    return '%s@dataset=%s%s' % (algo, ds, extra)


def config(cid, algo, pack, defines, workloads, tier='single', assignment=None, priority=1):
    return dict(id=cid, A=dict(defines=defines, environment={}, runtime={}), B=dict(defines=[], environment={}, runtime={}),
                mode='identical', kind='candidate', priority=priority, members=[cid], workloads=workloads, vendors=['nvidia', 'amd'],
                grid=dict(algorithm=algo, assignment=assignment or {'x': 'on'}, tier=tier, pack=pack, packed_extra_defines=[],
                          effective_defines=defines, bindings=['_mojolearn_x_linear']))


def write_plan(d):
    configs = [
        config('G.expanded:ridge-cv.a=on', 'expanded:ridge-cv', 'P001', ['MOJOLEARN_A=1'],
               [wl('expanded:ridge-cv', 'istella'), wl('expanded:ridge-cv', 'taxi')], assignment={'a': 'on'}),
        config('G.expanded:lasso-cv.b=on', 'expanded:lasso-cv', 'P001', ['MOJOLEARN_B=2'],
               [wl('expanded:lasso-cv', 'taxi')], assignment={'b': 'on'}),
        config('G.expanded:ridge-cv.c=on', 'expanded:ridge-cv', 'P002', ['MOJOLEARN_C=1'],
               [wl('expanded:ridge-cv', 'istella'), wl('expanded:ridge-cv', 'taxi')], assignment={'c': 'on'}, priority=2),
        config('G.expanded:qda.d=on', 'expanded:qda', 'P003', ['MOJOLEARN_D=1'],
               [wl('expanded:qda', 'taxi', '@input=classification-full-v1')], assignment={'d': 'on'}),
        config('G.expanded:theta.e=on', 'expanded:theta', 'P004', ['MOJOLEARN_E=1'],
               [wl('expanded:theta', 'taxi')], assignment={'e': 'on'}),
        config('G.classical:kmeans.f=on', 'classical:kmeans', 'P005', ['MOJOLEARN_F=1'],
               [wl('classical:kmeans', 'taxi')], assignment={'f': 'on'}),
        config('G.expanded:sgd-reg@rr.g=on', 'expanded:sgd-reg@rr', 'P006', ['MOJOLEARN_G=1'],
               [wl('expanded:sgd-reg@regression_report', 'taxi')], assignment={'g': 'on'}),
        config('G.rf.h=on', 'trees:rf', 'P007', ['MOJOLEARN_H=1'], ['rf:taxi', 'rf:year'], assignment={'h': 'on'}),
    ]
    packs = {}
    for c in configs:
        p = packs.setdefault(c['grid']['pack'], dict(id=c['grid']['pack'], members=[], defines=[], bindings=['_mojolearn_x_linear'],
                                                     algorithms=[], global_reach=False))
        p['members'].append(c['id'])
        p['defines'] = sorted(set(p['defines']) | set(c['A']['defines']))
    for c in configs:
        c['grid']['packed_extra_defines'] = sorted(set(packs[c['grid']['pack']]['defines']) - set(c['A']['defines']))
    algos = {c['grid']['algorithm']: dict(regime='pairwise' if c['grid']['algorithm'] == 'trees:rf' else 'factorial',
                                          workload_bindings=['_mojolearn_x_linear']) for c in configs}
    algos['expanded:lasso-cv']['regime'] = 'pairwise'
    cells = [dict(configuration=c['id'], workload_id=w, vendor=v) for c in configs for w in c['workloads'] for v in ('nvidia', 'amd')]
    plan = dict(builds=dict(packs=list(packs.values())), algorithms=algos)
    matrix = dict(configurations=configs, cells=cells)
    (d / 'grid-plan.json').write_text(json.dumps(plan))
    with gzip.open(d / 'grid-matrix.json.gz', 'wt') as f:
        json.dump(matrix, f)
    return plan, matrix


def algos_line(jid, vendor, head, tag, lane, ds, ms, quality, digest, defines=None, status='ok'):
    envs = ' MOJOLEARN_GRID_TAG=' + tag + (' MOJOLEARN_BUILD_DEFINES=' + defines if defines else '')
    line = '%s %s main@%s [%s] ALGOS lane=%s dataset=%s arm=ours status=%s median_ms=%s quality=%s digest=%s' % (
        jid, vendor, head, envs, lane, ds, status, ms, json.dumps(quality, sort_keys=True), digest)
    return line[:600]


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        write_plan(self.d)

    def tearDown(self):
        self.tmp.cleanup()

    def render(self, **kw):
        kw.setdefault('lanes', LANES)
        kw.setdefault('board', BOARD)
        return G.render(self.d, kw.pop('vendor', 'nvidia'), **kw)

    def test_lines_packs_and_order(self):
        lines, m = self.render(b_repeats=0)
        a = [x for x in m['lines'] if x['arm'] == 'A']
        # a pack is in the factorial phase when any member is; then priority, then pack number
        self.assertEqual([x['pack'] for x in a], ['P001', 'P003', 'P005', 'P002', 'P007'])
        self.assertEqual([x['regime'] for x in a], ['factorial'] * 4 + ['pairwise'])
        self.assertEqual([x['kind'] for x in a], ['race', 'race', 'cmd', 'race', 'cmd'])
        rid = G.run_id_of(self.d)
        k = [l for l in lines if '.P005.bb ' in l][0]
        self.assertEqual(k, 'lq add nv CMD main %s.P005.bb MOJOLEARN_GRID_TAG=%s.P005.bb MOJOLEARN_BUILD_DEFINES=MOJOLEARN_F=1 '
                            '.pixi/envs/default/bin/python tools/six_lane_grid_bb.py --tag %s.P005.bb --vendor nvidia '
                            '--race classical:kmeans:taxi BUILDS=build,build_x_linear' % (rid, rid, rid))
        self.assertIn('--race trees:rf:taxi BUILDS=', [l for l in lines if '.P007.bb ' in l][0])
        p1 = [l for l in lines if '.P001 ' in l][0]
        self.assertIn('lq add nv RACE main lasso-cv@taxi,ridge-cv@istella,ridge-cv@taxi PAIRS', p1)
        self.assertIn('MOJOLEARN_BUILD_DEFINES=MOJOLEARN_A=1,MOJOLEARN_B=2', p1)
        self.assertIn('BUILDS=build,build_x_linear', p1)
        p2 = [l for l in lines if '.P002 ' in l][0]
        self.assertIn('RACE main ridge-cv istella,taxi MOJOLEARN_GRID_TAG=', p2)
        self.assertTrue(p2.split()[7].startswith('MOJOLEARN_GRID_TAG=' + G.run_id_of(self.d)))
        q = [l for l in lines if '.P003 ' in l][0]
        self.assertIn('qda taxi', q)
        self.assertEqual(m['totals']['input_variant_workloads'], ['expanded:qda@dataset=taxi@input=classification-full-v1'])

    def test_refusals_by_name(self):
        _, m = self.render(b_repeats=0)
        why = {r['workload_id']: r['reason'] for r in m['refused']}
        self.assertIn("races datasets taxi-hourly,synthetic, not 'taxi'", why['expanded:theta@dataset=taxi'])
        self.assertIn('lane variant @regression_report', why['expanded:sgd-reg@regression_report@dataset=taxi'])
        self.assertIn("not driver dataset 'year'", why['rf:year'])
        self.assertEqual(m['totals']['refused_workloads'], 3)

    def test_map_board(self):
        self.assertEqual(G.map_board('gbdt-multiclass:taximc', BOARD)[:3], ('trees', 'gbdt-multiclass', 'taxi'))
        self.assertEqual(G.map_board('neural:lm-train-step', BOARD)[:3], ('neural', 'lm-train-step', 'bytes'))
        self.assertEqual(G.map_board('classical:kmeans@dataset=taxi', BOARD)[:3], ('classical', 'kmeans', 'taxi'))
        for wid, why in (('classical:kmeans@dataset=istella', 'plans no classical/kmeans/istella'),
                         ('gbdt-multiclass:istellamc', 'has no board dataset'), ('neural:gemm', 'not a tools/bench_board.py'),
                         ('more:ridge@dataset=taxi', 'plans no classical2/ridge/taxi'), ('oob:rf:taxi', 'no bench_board route')):
            with self.assertRaises(ValueError) as cm:
                G.map_board(wid, BOARD)
            self.assertIn(why, str(cm.exception))

    def test_b_repeats_spread(self):
        lines, m = self.render(b_repeats=3)
        b = [x for x in m['lines'] if x['arm'] == 'B']
        seen = {}
        for x in b:
            for w in x['workloads']:
                seen.setdefault(w, []).append(x['line'])
        self.assertEqual(len(seen), 6)
        self.assertTrue(all(len(v) == 3 and len(set(v)) == 3 for v in seen.values()))
        for line in lines:
            if '.B0' in line or '.C0' in line:
                self.assertNotIn('MOJOLEARN_BUILD_DEFINES', line)
        self.assertEqual(m['totals']['b_races'], 18)
        self.assertEqual({x['kind'] for x in b if x['tag'].split('.')[1].startswith('C')}, {'cmd'})

    def test_filters_and_budget(self):
        _, m = self.render(b_repeats=1, phase='pairwise')
        self.assertEqual([x['pack'] for x in m['lines'] if x['arm'] == 'A'], ['P001', 'P007'])
        self.assertEqual([x['regime'] for x in m['lines'] if x['arm'] == 'A'], ['pairwise', 'pairwise'])
        self.assertEqual({c['lane'] for x in m['lines'] if x['arm'] == 'A' for c in x['cells']}, {'lasso-cv', 'rf'})
        _, m = self.render(b_repeats=1, only_lanes={'ridge-cv'})
        self.assertEqual({c['lane'] for x in m['lines'] if x['arm'] == 'A' for c in x['cells']}, {'ridge-cv'})
        # first job (P001: 3 A + 3 new workloads x 1 B = 6 races x 19 s = 114 s) fits in 120 s, the next does not
        _, m = self.render(b_repeats=1, budget_hours=120 / 3600.0)
        self.assertEqual([x['pack'] for x in m['lines'] if x['arm'] == 'A'], ['P001'])
        self.assertEqual(m['totals']['cut_a_lines_by_budget'], 4)

    def test_amd_box_and_cli(self):
        out = self.d / 'out' / 'amd.lines'
        lanes = self.d / 'lanes.json'
        lanes.write_text(json.dumps(LANES))
        board = self.d / 'board.json'
        board.write_text(json.dumps(dict(BOARD, races=sorted(BOARD['races']))))
        rc = G.main(['render', '--plan-dir', str(self.d), '--vendor', 'amd', '--out', str(out), '--lanes-json', str(lanes),
                     '--board-json', str(board)])
        self.assertEqual(rc, 0)
        text = out.read_text().splitlines()
        self.assertTrue(text and all(l.startswith(('lq add amd RACE main ', 'lq add amd CMD main ')) for l in text))
        self.assertTrue(all('--vendor amd' in l for l in text if ' CMD ' in l))
        self.assertTrue(Path(str(out) + '.json').exists())
        with self.assertRaises(SystemExit):
            G.main(['render', '--plan-dir', str(self.d), '--vendor', 'amd', '--out', str(out), '--lanes-json', str(lanes),
                    '--board-json', str(board), '--only-lanes', 'no-such-lane'])

    def test_plain_tokens(self):
        with self.assertRaises(ValueError):
            G.lq_line('nv', 'main', [('ridge-cv', 'taxi')], [('MOJOLEARN_BUILD_DEFINES', 'A=$(x)')], [])


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        write_plan(self.d)
        self.rid = G.run_id_of(self.d)

    def tearDown(self):
        self.tmp.cleanup()

    def results(self):
        r = self.rid
        qb = {'r2': 0.9, 'rmse': 1.0}
        nv, amd = [], []
        n = 0
        # incumbent: 3 repeats per workload per vendor; ridge-cv taxi spread 100..104 ms
        for rep, (ms_nv, ms_amd) in enumerate(((100, 50), (102, 51), (104, 52))):
            for lane, ds in (('ridge-cv', 'istella'), ('ridge-cv', 'taxi'), ('lasso-cv', 'taxi'), ('qda', 'taxi')):
                n += 1
                nv.append(algos_line('n%04d' % n, 'nvidia', 'abc1234', '%s.B001r%d' % (r, rep + 1), lane, ds, ms_nv, qb, 'b' * 16))
                amd.append(algos_line('a%04d' % n, 'amd', 'abc1234', '%s.B001r%d' % (r, rep + 1), lane, ds, ms_amd, qb, 'b' * 16))
        # P002 (ridge-cv.c): faster on both vendors, same digest -> FASTER, MATCH, SAME
        nv.append(algos_line('n0100', 'nvidia', 'abc1234', r + '.P002', 'ridge-cv', 'taxi', 80, qb, 'c' * 16, 'MOJOLEARN_C=1'))
        amd.append(algos_line('a0100', 'amd', 'abc1234', r + '.P002', 'ridge-cv', 'taxi', 40, qb, 'c' * 16, 'MOJOLEARN_C=1'))
        # P002 istella: nv slower beyond the floor, amd within it -> NO_VERDICT; digests differ -> MISMATCH
        nv.append(algos_line('n0100', 'nvidia', 'abc1234', r + '.P002', 'ridge-cv', 'istella', 130, qb, 'd' * 16, 'MOJOLEARN_C=1'))
        amd.append(algos_line('a0100', 'amd', 'abc1234', r + '.P002', 'ridge-cv', 'istella', 51, qb, 'e' * 16, 'MOJOLEARN_C=1'))
        # P001: lasso-cv worse r2 on nv -> WORSE; nv line cut at 600 characters (long defines): digest + quality from the log
        long_defs = ','.join('MOJOLEARN_LONG_DEFINE_%03d=1' % i for i in range(30))
        nv.append(algos_line('n0101', 'nvidia', 'abc1234', r + '.P001', 'lasso-cv', 'taxi', 100, {'r2': 0.8, 'rmse': 1.0}, 'f' * 16, long_defs))
        amd.append(algos_line('a0101', 'amd', 'abc1234', r + '.P001', 'lasso-cv', 'taxi', 50, {'r2': 0.9, 'rmse': 1.0}, 'f' * 16, 'MOJOLEARN_A=1'))
        # P003: the qda race failed on amd
        nv.append(algos_line('n0102', 'nvidia', 'abc1234', r + '.P003', 'qda', 'taxi', 100, qb, '1' * 16, 'MOJOLEARN_D=1'))
        amd.append(algos_line('a0102', 'amd', 'abc1234', r + '.P003', 'qda', 'taxi', 'None', {}, 'none', 'MOJOLEARN_D=1', status='error'))
        amd.append('a0103 amd main@abc1234 ridge-cv taxi NO-RESULT builds=0/1 see /root/lq/out/a0103/rc.txt')
        # another campaign's line is ignored
        nv.append(algos_line('n0104', 'nvidia', 'abc1234', 'gdeadbeef.P002', 'ridge-cv', 'taxi', 1, qb, '9' * 16, 'X=1'))
        # CMD route (tools/six_lane_grid_bb.py GRIDBB lines in the job logs): kmeans faster + same hash on both
        # vendors; rf hash differs between vendors; one CMD job printed no GRIDBB line
        def bbl(jid, vendor, tag, fam, lane, ds, ms, h, q=None):
            return '/root/lq/out/%s/%s.log:GRIDBB tag=%s vendor=%s head=abc1234 family=%s lane=%s dataset=%s status=ok ' \
                   'median_ms=%s hash=%s quality=%s' % (jid, tag, tag, vendor, fam, lane, ds, ms, h, json.dumps(q or {'inertia': 5.0}))
        self.bb = {'nvidia': [], 'amd': []}
        for rep_, (mn, ma) in enumerate(((1000, 500), (1010, 505), (1020, 510))):
            for vendor, ms in (('nvidia', mn), ('amd', ma)):
                pre = 'n' if vendor == 'nvidia' else 'a'
                tag = '%s.C001r%d' % (r, rep_ + 1)
                self.bb[vendor].append(bbl('%s03%02d' % (pre, rep_), vendor, tag, 'classical', 'kmeans', 'taxi', ms, 'k' * 16))
                self.bb[vendor].append(bbl('%s03%02d' % (pre, rep_), vendor, tag, 'trees', 'rf', 'taxi', ms, 'r' * 16))
        self.bb['nvidia'].append(bbl('n0310', 'nvidia', r + '.P005.bb', 'classical', 'kmeans', 'taxi', 700, '5' * 16))
        self.bb['amd'].append(bbl('a0310', 'amd', r + '.P005.bb', 'classical', 'kmeans', 'taxi', 350, '5' * 16))
        self.bb['nvidia'].append(bbl('n0311', 'nvidia', r + '.P007.bb', 'trees', 'rf', 'taxi', 1000, '6' * 16))
        self.bb['amd'].append(bbl('a0311', 'amd', r + '.P007.bb', 'trees', 'rf', 'taxi', 505, '7' * 16))
        nv.append('n0310 nvidia main@abc1234 CMD %s.P005.bb rc=0 builds=[build rc=0 ] last: GRIDBB-DONE races=1 ok=1' % r)
        nv.append('n0312 nvidia main@abc1234 CMD %s.P099.bb rc=1 builds=[build rc=1 ] last: error' % r)
        (self.d / 'nv-bb.txt').write_text('\n'.join(self.bb['nvidia']) + '\n')
        (self.d / 'amd-bb.txt').write_text('\n'.join(self.bb['amd']) + '\n')
        (self.d / 'nv-results.txt').write_text('\n'.join(nv) + '\n')
        (self.d / 'amd-results.txt').write_text('\n'.join(amd) + '\n')
        log = self.d / 'nv-out' / 'n0101' / 'race-lasso-cv-taxi-def.log'
        log.parent.mkdir(parents=True)
        log.write_text('2026-10-07T00:00:00Z race lasso-cv taxi arms=ours env=none try=1\n'
                       'ALGOS-ROUND lane=lasso-cv dataset=taxi arm=ours round=0 ms=101.0 infer_ms=None digest=' + 'f' * 64 + '\n'
                       'ALGOS-ROUND lane=lasso-cv dataset=taxi arm=ours round=1 ms=100.0 infer_ms=None digest=' + 'f' * 64 + '\n'
                       'ALGOS lane=lasso-cv dataset=taxi arm=ours status=ok median_ms=100.0 quality={"r2": 0.8, "rmse": 1.0}\n')
        return [self.d / 'nv-results.txt', self.d / 'amd-results.txt'], [self.d / 'nv-out', self.d / 'nv-bb.txt', self.d / 'amd-bb.txt']

    def test_collect_and_decide(self):
        results, logs = self.results()
        out = self.d / 'collected'
        rep = G.collect(results, logs, self.d, out, lanes=LANES, board=BOARD)
        self.assertEqual(rep['ignored']['other_campaign'], 1)
        self.assertEqual(rep['truncated_without_log'], 0)
        self.assertEqual(len(rep['no_result']), 1)
        v = json.loads((out / 'grid-verdicts.json').read_text())
        self.assertEqual(v['schema'], 'mojolearn.six-lane-timing-verdicts/1')
        cases = {(c['configuration'], c['workload_id']): c for c in v['cases']}
        c = cases[('G.expanded:ridge-cv.c=on', 'expanded:ridge-cv@dataset=taxi')]
        self.assertEqual(c['verdict'], 'FASTER')
        self.assertAlmostEqual(c['vendors']['nvidia']['candidate_over_baseline']['scored'], 80 / 102)
        self.assertAlmostEqual(c['vendors']['nvidia']['floor']['scored'], __import__('math').log(104 / 100))
        self.assertEqual(cases[('G.expanded:ridge-cv.c=on', 'expanded:ridge-cv@dataset=istella')]['verdict'], 'NO_VERDICT')
        s = json.loads((out / 'summary.json').read_text())
        self.assertEqual(s['schema'], 'mojolearn.six-lane-comparison/1')
        ids = {(c['configuration_id'], c['workload_id']): c for c in s['cases']}
        self.assertEqual(ids[('G.expanded:ridge-cv.c=on', 'expanded:ridge-cv@dataset=taxi')]['arms']['A']['status'], 'MATCH')
        self.assertEqual(ids[('G.expanded:ridge-cv.c=on', 'expanded:ridge-cv@dataset=taxi')]['arms']['B']['status'], 'MATCH')
        self.assertEqual(ids[('G.expanded:ridge-cv.c=on', 'expanded:ridge-cv@dataset=istella')]['arms']['A']['status'], 'MISMATCH')
        lasso = ids[('G.expanded:lasso-cv.b=on', 'expanded:lasso-cv@dataset=taxi')]
        self.assertEqual(lasso['arms']['A']['status'], 'MATCH')  # nv digest recovered from the race log
        self.assertEqual(ids[('G.expanded:qda.d=on', 'expanded:qda@dataset=taxi@input=classification-full-v1')]['arms']['A']['status'], 'INCOMPLETE')
        q = json.loads((out / 'quality.json').read_text())
        rows = {(r['configuration'], r['workload_id'], r['vendor']): r['candidate_vs_baseline']['verdict'] for r in q['rows']}
        self.assertEqual(rows[('G.expanded:lasso-cv.b=on', 'expanded:lasso-cv@dataset=taxi', 'nvidia')], 'WORSE')
        self.assertEqual(rows[('G.expanded:lasso-cv.b=on', 'expanded:lasso-cv@dataset=taxi', 'amd')], 'SAME')
        self.assertEqual(rows[('G.expanded:qda.d=on', 'expanded:qda@dataset=taxi@input=classification-full-v1', 'amd')], 'FAIL')
        self.assertEqual(rep['coverage']['nvidia']['REFUSED'], 3)
        self.assertEqual(rep['cmd_jobs_without_gridbb'], ['nvidia:n0312'])
        km = cases[('G.classical:kmeans.f=on', 'classical:kmeans@dataset=taxi')]
        self.assertEqual(km['verdict'], 'FASTER')
        self.assertAlmostEqual(km['vendors']['amd']['candidate_over_baseline']['scored'], 350 / 505)
        self.assertEqual(ids[('G.classical:kmeans.f=on', 'classical:kmeans@dataset=taxi')]['arms']['A']['status'], 'MATCH')
        self.assertEqual(ids[('G.classical:kmeans.f=on', 'classical:kmeans@dataset=taxi')]['arms']['B']['status'], 'MATCH')
        self.assertEqual(ids[('G.rf.h=on', 'rf:taxi')]['arms']['A']['status'], 'MISMATCH')
        self.assertEqual(rows[('G.classical:kmeans.f=on', 'classical:kmeans@dataset=taxi', 'nvidia')], 'SAME')
        # decide runs unchanged on the three outputs
        dec_out = self.d / 'decide'
        rc = D.main(['--matrix', str(self.d / 'grid-matrix.json.gz'), '--verdicts', str(out / 'grid-verdicts.json'),
                     '--identity', str(out / 'summary.json'), '--quality', str(out / 'quality.json'), '--out', str(dec_out)])
        self.assertEqual(rc, 0)
        dec = json.loads((dec_out / 'grid-decisions.json').read_text())
        ev = dec['configurations']
        self.assertIn('G.expanded:ridge-cv.c=on', ev)
        rows = {r['workload_id']: r for r in ev['G.expanded:ridge-cv.c=on']['rows']}
        self.assertEqual(rows['expanded:ridge-cv@dataset=taxi']['timing'], 'FASTER')
        self.assertEqual(rows['expanded:ridge-cv@dataset=taxi']['identity'], 'MATCH')
        self.assertEqual(rows['expanded:ridge-cv@dataset=taxi']['quality'], 'SAME')

    def test_grep_dump_logs(self):
        results, logs_ = self.results()
        dump = self.d / 'nv-logs.txt'
        f = '/root/lq/out/n0101/race-lasso-cv-taxi-def.log'
        dump.write_text(f + ':ALGOS-ROUND lane=lasso-cv dataset=taxi arm=ours round=1 ms=100.0 infer_ms=None digest=' + 'f' * 64 + '\n'
                        + f + ':ALGOS lane=lasso-cv dataset=taxi arm=ours status=ok median_ms=100.0 quality={"r2": 0.8}\n')
        rep = G.collect(results, [dump] + logs_[1:], self.d, self.d / 'c2', lanes=LANES, board=BOARD)
        self.assertEqual(rep['truncated_without_log'], 0)
        self.assertEqual(rep['jobs_without_algos'], [])
        rep = G.collect(results, [], self.d, self.d / 'c3', lanes=LANES, board=BOARD)  # lq cut the whole ALGOS text: reported
        self.assertEqual(rep['jobs_without_algos'], ['nvidia:n0101'])

    def test_quality_directions(self):
        self.assertEqual(G.direction('rmse'), 'lower')
        self.assertEqual(G.direction('logloss'), 'lower')
        self.assertEqual(G.direction('trustworthiness_k15'), 'higher')
        self.assertEqual(G.direction('accuracy'), 'higher')
        self.assertIsNone(G.direction('n_clusters'))
        self.assertEqual(G.quality_verdict({'rmse': 1.0 + 1e-9}, {'rmse': 1.0})[0], 'SAME')
        self.assertEqual(G.quality_verdict({'rmse': 1.1}, {'rmse': 1.0})[0], 'WORSE')
        self.assertEqual(G.quality_verdict({'auc': 0.91}, {'auc': 0.9})[0], 'BETTER')
        self.assertEqual(G.quality_verdict({'n_clusters': 3}, {'n_clusters': 4})[0], 'PENDING')


class FeedTests(unittest.TestCase):
    def test_feed_resumes_and_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            fake = t / 'lq'
            fake.write_text('#!/bin/bash\n'
                            'if [ "$1" = status ]; then echo "nv:      2 done       $(cat %s/depth) queued"; echo "amd: 9 lines, done through 9"; exit 0; fi\n'
                            'echo "$*" >> %s/added; echo "queued n0001 (gpu-queue 1)"\n' % (t, t))
            fake.chmod(0o755)
            (t / 'depth').write_text('0')
            lines = t / 'nv.lines'
            lines.write_text('lq add nv RACE main ridge-cv taxi MOJOLEARN_GRID_TAG=g1.P1\nlq add nv RACE main qda taxi MOJOLEARN_GRID_TAG=g1.P2\n')
            feed = Path(__file__).resolve().parent / 'six_lane_grid_lq_feed.sh'
            env = dict(os.environ, LQ=str(fake))
            r = subprocess.run(['bash', str(feed), 'nv', str(lines), '1', '1'], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual((t / 'added').read_text().splitlines(),
                             ['add nv RACE main ridge-cv taxi MOJOLEARN_GRID_TAG=g1.P1', 'add nv RACE main qda taxi MOJOLEARN_GRID_TAG=g1.P2'])
            self.assertEqual((t / 'nv.lines.fed-nv').read_text().strip(), '2')
            r = subprocess.run(['bash', str(feed), 'nv', str(lines), '1', '1'], env=env, capture_output=True, text=True)
            self.assertEqual(len((t / 'added').read_text().splitlines()), 2)  # resumed: nothing re-added
            self.assertIn('fed 2/2', r.stdout)



class BoardWrapperTests(unittest.TestCase):
    def test_groups_and_commands(self):
        races = [('classical', 'kmeans', 'taxi'), ('classical', 'kmeans', 'istella'), ('classical', 'pca', 'taxi'),
                 ('neural', 'lm-train-step', 'bytes'), ('classical2', 'arima', 'synthetic')]
        g = BBW.groups(races)
        self.assertIn(('classical', ['kmeans'], ['istella', 'taxi']), g)
        self.assertIn(('classical', ['pca'], ['taxi']), g)
        self.assertIn(('neural', ['lm-train-step'], ['taxi']), g)
        self.assertIn(('classical2', ['arima'], ['taxi']), g)
        self.assertEqual(len(g), 4)

    def test_summary_lines_parse_back(self):
        import contextlib
        import io
        board = {'races': {'classical/kmeans/taxi/rows=full': {'family': 'classical', 'lane': 'kmeans', 'dataset': 'taxi',
                                                               'cells': [{'arm': 'ours', 'status': 'ok', 'median_ms': 12.5,
                                                                          'hash': 'abcdef0123456789ff', 'quality': {'inertia': 3.0}}]},
                           'neural/lm-train-step/bytes/shape=full': {'family': 'neural', 'lane': 'lm-train-step', 'dataset': 'bytes',
                                                                     'cells': [{'arm': 'ours', 'status': 'REFUSED(x y)'}]}}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = BBW.summarize(board, [('classical', 'kmeans', 'taxi'), ('neural', 'lm-train-step', 'bytes'),
                                       ('trees', 'rf', 'taxi')], 'g1.P1.bb', 'amd', 'abc1234')
        self.assertEqual(ok, 1)
        lines = buf.getvalue().splitlines()
        o = G.parse_gridbb(lines[0], '/root/lq/out/a0001/g1.P1.bb.log', 'a0001')
        self.assertEqual((o['key'], o['median_ms'], o['digest'], o['quality'], o['tag'], o['vendor']),
                         (('classical', 'kmeans', 'taxi'), 12.5, 'abcdef0123456789', {'inertia': 3.0}, 'g1.P1.bb', 'amd'))
        self.assertEqual(G.parse_gridbb(lines[1], 'x', 'a0001')['status'], 'REFUSED(x_y)')
        self.assertEqual(G.parse_gridbb(lines[2], 'x', 'a0001')['status'], 'NO-RECORD')

    def test_dry_run_command(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            BBW.main(['--tag', 't', '--vendor', 'nvidia', '--race', 'trees:rf:taxi', '--dry-run'])
        cmd = buf.getvalue().split()
        for flag in ('--modes identical', '--families trees', '--lanes rf', '--datasets taxi', '--skip-install',
                     '--no-smoke-gate', '--no-infer', '--rows full', '--vendor nvidia'):
            self.assertIn(flag, ' '.join(cmd))


if __name__ == '__main__':
    unittest.main()
