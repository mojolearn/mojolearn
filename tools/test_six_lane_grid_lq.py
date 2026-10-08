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
        self.assertAlmostEqual(c['vendors']['nvidia']['floor']['scored'], max(__import__('math').log(104 / 100), G.FLOOR_MIN))  # the 4% spread is clamped to the 5% minimum
        self.assertEqual(cases[('G.expanded:ridge-cv.c=on', 'expanded:ridge-cv@dataset=istella')]['verdict'], 'SLOWER')  # nv slower beyond its floor, amd within: the average is slower
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


def decisions_doc():
    """grid-decisions.json shape for the synthetic plan (six_lane_grid_decide.decide output, the fields rerun reads)."""
    def row(wid, timing, quality='SAME', identity='MATCH', ratios=None):
        return dict(workload_id=wid, timing=timing, quality=quality, identity=identity, ratios=ratios or {})
    cfg = {
        'G.expanded:ridge-cv.a=on': ({'a': 'on'}, [row(wl('expanded:ridge-cv', 'istella'), 'NO_VERDICT'),
                                                   row(wl('expanded:ridge-cv', 'taxi'), 'INCOMPLETE', 'PENDING', 'INCOMPLETE')]),
        'G.expanded:lasso-cv.b=on': ({'b': 'on'}, [row(wl('expanded:lasso-cv', 'taxi'), 'NO_VERDICT', 'WORSE')]),
        'G.expanded:ridge-cv.c=on': ({'c': 'on'}, [row(wl('expanded:ridge-cv', 'istella'), 'NO_VERDICT'),
                                                   row(wl('expanded:ridge-cv', 'taxi'), 'FASTER')]),
        'G.expanded:qda.d=on': ({'d': 'on'}, [row(wl('expanded:qda', 'taxi', '@input=classification-full-v1'), 'INCOMPLETE', 'FAIL')]),
        'G.expanded:theta.e=on': ({'e': 'on'}, [row(wl('expanded:theta', 'taxi'), 'UNMEASURED', 'PENDING', 'UNMEASURED')]),
        'G.classical:kmeans.f=on': ({'f': 'on'}, [row(wl('classical:kmeans', 'taxi'), 'NO_VERDICT')]),
        'G.expanded:sgd-reg@rr.g=on': ({'g': 'on'}, [row(wl('expanded:sgd-reg@regression_report', 'taxi'), 'UNMEASURED', 'PENDING')]),
        'G.rf.h=on': ({'h': 'on'}, [row('rf:taxi', 'NO_VERDICT'), row('rf:year', 'UNMEASURED', 'PENDING', 'UNMEASURED')]),
    }
    controls = {'a': 'NOT_MEASURED', 'b': 'HOLD_QUALITY', 'c': 'PARTIAL_PROMOTE', 'd': 'HOLD_BROKEN', 'e': 'NOT_MEASURED',
                'f': 'DELETE', 'g': 'NOT_MEASURED', 'h': 'HOLD_BROKEN'}
    return dict(schema=D.SCHEMA, controls={k: dict(control=k, recommendation=v) for k, v in controls.items()},
                configurations={cid: dict(configuration=cid, assignment=a, rows=rows) for cid, (a, rows) in cfg.items()})


class RerunRenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        write_plan(self.d)
        self.dec = self.d / 'decide'
        self.dec.mkdir()
        (self.dec / 'grid-decisions.json').write_text(json.dumps(decisions_doc()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_selection_reasons(self):
        cells, selected, rep = G.undecided_cells(decisions_doc())
        self.assertEqual(selected, ['G.expanded:ridge-cv.a=on', 'G.rf.h=on'])
        self.assertEqual(cells, {('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'istella')),
                                 ('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'taxi')), ('G.rf.h=on', 'rf:taxi')})
        self.assertEqual(rep['skipped_configurations'], {'broken_cell': 1, 'decided_cell': 1, 'flipped_switch': 2, 'unmeasured': 2})
        self.assertEqual(rep['unmeasured_cells_not_rerun'], 1)  # rf:year is first-pass work, not a second point
        self.assertEqual(rep['final_controls'], ['b', 'f'])  # HOLD_BROKEN (d, h) awaits a fix: not final
        _, selected, rep = G.undecided_cells(decisions_doc(), skip_controls=['a'], skip_broken_controls=True)
        self.assertEqual(selected, [])
        self.assertEqual(rep['final_controls'], ['a', 'b', 'd', 'f', 'h'])

    def test_render_rerun_lines(self):
        lines, m = G.render_rerun(self.d, 'nvidia', self.dec, lanes=LANES, board=BOARD)
        rid = G.run_id_of(self.d) + 'r2'
        a = [x for x in m['lines'] if x['arm'] == 'A']
        self.assertEqual([(x['pack'], x['kind']) for x in a], [('P001', 'race'), ('P007', 'cmd')])
        self.assertEqual({(c['configuration'], c['workload_id']) for x in a for c in x['cells']},
                         {('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'istella')),
                          ('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'taxi')), ('G.rf.h=on', 'rf:taxi')})
        p1 = [l for l in lines if ('.P001 ') in l][0]
        self.assertIn('RACE main ridge-cv istella,taxi MOJOLEARN_GRID_TAG=%s.P001 ' % rid, p1)
        self.assertIn('MOJOLEARN_BUILD_DEFINES=MOJOLEARN_A=1,MOJOLEARN_B=2', p1)  # the pack's define set is unchanged
        b = [x for x in m['lines'] if x['arm'] == 'B']
        seen = {}
        for x in b:
            self.assertTrue(x['tag'].startswith(rid + '.'))
            for w in x['workloads']:
                seen[w] = seen.get(w, 0) + 1
        self.assertEqual(seen, {wl('expanded:ridge-cv', 'istella'): 1, wl('expanded:ridge-cv', 'taxi'): 1, 'rf:taxi': 1})
        t = m['totals']
        self.assertEqual((t['run_id'], t['b_repeats'], t['a_races'], t['b_races']), (rid, 1, 3, 3))
        self.assertEqual(t['rerun']['configurations'], 2)
        self.assertIn('plan default', t['measured_note'])
        with self.assertRaises(ValueError):
            G.render_rerun(self.d, 'nvidia', self.dec, rerun_id='x.y', lanes=LANES, board=BOARD)

    def test_measured_hours_from_first_pass(self):
        fp = self.d / 'first'
        fp.mkdir()
        cases = [dict(configuration='G.expanded:ridge-cv.a=on', workload_id=wl('expanded:ridge-cv', 'istella'),
                      vendors=dict(nvidia=dict(a_ms=3600e3, b_ms=7200e3)))]
        (fp / 'grid-verdicts.json').write_text(json.dumps(dict(cases=cases)))
        _, m = G.render_rerun(self.d, 'nvidia', self.dec, first_pass=fp, lanes=LANES, board=BOARD)
        t = m['totals']
        # A istella 1 h + B istella 2 h + four races at the 19 s plan default
        self.assertEqual(t['measured_races'], 2)
        self.assertAlmostEqual(t['measured_race_hours'], round(3 + 4 * 19 / 3600.0, 3))

    def test_cli(self):
        lanes = self.d / 'lanes.json'
        lanes.write_text(json.dumps(LANES))
        board = self.d / 'board.json'
        board.write_text(json.dumps(dict(BOARD, races=sorted(BOARD['races']))))
        out = self.d / 'out' / 'nv.rerun.lines'
        rc = G.main(['render', '--plan-dir', str(self.d), '--vendor', 'nvidia', '--out', str(out), '--lanes-json', str(lanes),
                     '--board-json', str(board), '--rerun-undecided', str(self.dec), '--rerun-id', 'gpass2'])
        self.assertEqual(rc, 0)
        text = out.read_text().splitlines()
        self.assertEqual(len(text), 4)  # 2 A lines + 2 incumbent groups (RACE, CMD) x 1 repeat
        self.assertTrue(all('MOJOLEARN_GRID_TAG=gpass2.' in l for l in text))
        with self.assertRaises(SystemExit):
            G.main(['render', '--plan-dir', str(self.d), '--vendor', 'nvidia', '--out', str(out), '--lanes-json', str(lanes),
                    '--board-json', str(board), '--rerun-id', 'gpass2'])


class MultiPassCollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        write_plan(self.d)
        self.rid = G.run_id_of(self.d)
        self.r2 = self.rid + 'r2'

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        qb = {'r2': 0.9, 'rmse': 1.0}
        nv, amd = [], []
        n = [0]

        def b(run, rep, lane, ds, ms_nv, ms_amd, dig_nv='b' * 16):
            n[0] += 1
            nv.append(algos_line('n%04d' % n[0], 'nvidia', 'abc1234', '%s.B001r%d' % (run, rep), lane, ds, ms_nv, qb, dig_nv))
            amd.append(algos_line('a%04d' % n[0], 'amd', 'abc1234', '%s.B001r%d' % (run, rep), lane, ds, ms_amd, qb, 'b' * 16))

        def a(run, pack, lane, ds, ms_nv, ms_amd, dig_nv, dig_amd, defines='MOJOLEARN_A=1,MOJOLEARN_B=2', q_nv=qb):
            n[0] += 1
            nv.append(algos_line('n%04d' % n[0], 'nvidia', 'abc1234', '%s.%s' % (run, pack), lane, ds, ms_nv, q_nv, dig_nv, defines))
            amd.append(algos_line('a%04d' % n[0], 'amd', 'abc1234', '%s.%s' % (run, pack), lane, ds, ms_amd, qb, dig_amd, defines))
        # pass 1: 3 incumbent repeats (nv 100..104, amd 50..52); pass 2: one repeat, the box ran 10% slower (110 / 55)
        for rep, (mn, ma) in enumerate(((100, 50), (102, 51), (104, 52))):
            for lane, ds in (('ridge-cv', 'istella'), ('ridge-cv', 'taxi')):
                b(self.rid, rep + 1, lane, ds, mn, ma)
        b(self.r2, 1, 'ridge-cv', 'istella', 110, 55)
        b(self.r2, 1, 'ridge-cv', 'taxi', 110, 55, dig_nv='9' * 16)  # incumbent bits changed run to run on nv
        # ridge-cv.a istella: pass 1 0.96x (nv) 0.98x (amd), pass 2 0.92x / 0.94x, digests stable
        a(self.rid, 'P001', 'ridge-cv', 'istella', 0.96 * 102, 0.98 * 51, 'c' * 16, 'c' * 16)
        a(self.r2, 'P001', 'ridge-cv', 'istella', 0.92 * 110, 0.94 * 55, 'c' * 16, 'c' * 16)
        # ridge-cv.a taxi: the nv candidate digest changes between passes -> MISMATCH-RUN; pass 2 quality WORSE on nv
        a(self.rid, 'P001', 'ridge-cv', 'taxi', 102, 51, 'd' * 16, 'd' * 16)
        a(self.r2, 'P001', 'ridge-cv', 'taxi', 110, 55, 'e' * 16, 'd' * 16, q_nv={'r2': 0.8, 'rmse': 1.0})
        (self.d / 'nv.txt').write_text('\n'.join(nv) + '\n')
        (self.d / 'amd.txt').write_text('\n'.join(amd) + '\n')
        return [self.d / 'nv.txt', self.d / 'amd.txt']

    def test_two_passes(self):
        import math
        results = self.write()
        out = self.d / 'c'
        rep = G.collect(results, [], self.d, out, lanes=LANES, board=BOARD, run_id=[self.rid, self.r2])
        self.assertEqual(rep['run_ids'], [self.rid, self.r2])
        self.assertEqual(rep['ignored']['other_campaign'], 0)
        v = json.loads((out / 'grid-verdicts.json').read_text())
        cases = {(c['configuration'], c['workload_id']): c for c in v['cases']}
        c = cases[('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'istella'))]
        row = c['vendors']['nvidia']
        self.assertEqual((row['passes'], row['pass_runs']), (2, [self.rid, self.r2]))
        # each pass over its own incumbent median: log ratios averaged, not pooled medians
        self.assertAlmostEqual(row['log_ratio']['scored'], (math.log(0.96) + math.log(0.92)) / 2)
        self.assertAlmostEqual(row['b_ms'], math.sqrt(102 * 110))
        self.assertEqual(row['b_samples'], 4)
        f = json.loads((out / 'floors.json').read_text())['floors']
        fl = f[wl('expanded:ridge-cv', 'istella') + '|nvidia']
        self.assertEqual((fl['samples'], fl['passes']), (4, [self.rid, self.r2]))  # pooled across passes, one head
        self.assertAlmostEqual(fl['floor']['scored'], math.log(110 / 100))
        self.assertEqual(rep['passes_per_cell'], {'amd:2': 2, 'nvidia:2': 2})
        s = json.loads((out / 'summary.json').read_text())
        ids = {(x['configuration_id'], x['workload_id']): x for x in s['cases']}
        taxi = ids[('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'taxi'))]
        self.assertEqual(taxi['arms']['A']['status'], 'MISMATCH_RUN')
        self.assertEqual(taxi['arms']['B']['status'], 'MISMATCH')  # incumbent digest unstable on nv (pooled passes)
        self.assertEqual(taxi['status'], 'MISMATCH')
        self.assertEqual(ids[('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'istella'))]['status'], 'MATCH')
        self.assertEqual(s['counts']['MISMATCH_RUN'], 0)
        mr = {(x['workload_id'], x['arm'], x['vendor']): x for x in rep['mismatch_run']}
        self.assertEqual(mr[(wl('expanded:ridge-cv', 'taxi'), 'A', 'nvidia')]['digests'], {self.rid: 'd' * 16, self.r2: 'e' * 16})
        self.assertIn((wl('expanded:ridge-cv', 'taxi'), 'B', 'nvidia'), mr)
        q = json.loads((out / 'quality.json').read_text())
        rows = {(r['configuration'], r['workload_id'], r['vendor']): r for r in q['rows']}
        qt = rows[('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'taxi'), 'nvidia')]
        self.assertEqual((qt['candidate_vs_baseline']['verdict'], qt['passes']), ('WORSE', 2))
        self.assertEqual(qt['pass_verdicts'], {self.rid: 'SAME', self.r2: 'WORSE'})
        # one pass only: the second pass is another campaign, and the first pass alone is what collect always gave
        rep1 = G.collect(results, [], self.d, self.d / 'c1', lanes=LANES, board=BOARD, run_id=self.rid)
        self.assertEqual(rep1['ignored']['other_campaign'], 8)  # 2 A + 2 B lines x 2 vendors
        v1 = {(c['configuration'], c['workload_id']): c for c in json.loads((self.d / 'c1' / 'grid-verdicts.json').read_text())['cases']}
        r1 = v1[('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'istella'))]['vendors']['nvidia']
        self.assertAlmostEqual(r1['candidate_over_baseline']['scored'], 0.96)
        self.assertEqual(r1['passes'], 1)
        s1 = {(x['configuration_id'], x['workload_id']): x for x in json.loads((self.d / 'c1' / 'summary.json').read_text())['cases']}
        self.assertEqual(s1[('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'taxi'))]['status'], 'MATCH')
        # CLI: --runs comma list == several --run-id values
        rc = G.main(['collect', '--results'] + [str(x) for x in results] + ['--plan-dir', str(self.d), '--runs',
                     '%s,%s' % (self.rid, self.r2), '--out', str(self.d / 'c2'), '--lanes-json', str(self._lanes()),
                     '--board-json', str(self._board())])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads((self.d / 'c2' / 'grid-verdicts.json').read_text()), v)

    def test_pass_without_its_own_incumbent(self):
        import math
        results = self.write()
        # drop the pass-2 incumbent lines: the pass falls back to the pooled incumbent and says so
        for p in results:
            p.write_text('\n'.join(l for l in p.read_text().splitlines() if '%s.B001' % self.r2 not in l) + '\n')
        G.collect(results, [], self.d, self.d / 'c', lanes=LANES, board=BOARD, run_id=[self.rid, self.r2])
        v = {(c['configuration'], c['workload_id']): c for c in json.loads((self.d / 'c' / 'grid-verdicts.json').read_text())['cases']}
        row = v[('G.expanded:ridge-cv.a=on', wl('expanded:ridge-cv', 'istella'))]['vendors']['nvidia']
        self.assertEqual(row['b_from_other_pass'], [self.r2])
        self.assertAlmostEqual(row['pass_log_ratios'][1], math.log(0.92 * 110 / 102))

    def test_merge_quality(self):
        self.assertEqual(G.merge_quality(['SAME', 'WORSE']), 'WORSE')
        self.assertEqual(G.merge_quality(['SAME', 'FAIL']), 'FAIL')
        self.assertEqual(G.merge_quality(['PENDING', 'SAME']), 'SAME')
        self.assertEqual(G.merge_quality(['BETTER', 'SAME']), 'SAME')
        self.assertEqual(G.merge_quality(['PENDING']), 'PENDING')
        self.assertEqual(G.run_ids_of('a,b', self.d), ['a', 'b'])
        self.assertEqual(G.run_ids_of(None, self.d), [self.rid])

    def _lanes(self):
        p = self.d / 'lanes.json'
        p.write_text(json.dumps(LANES))
        return p

    def _board(self):
        p = self.d / 'board.json'
        p.write_text(json.dumps(dict(BOARD, races=sorted(BOARD['races']))))
        return p


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



NEURAL_BOARD = dict(races={('neural', 'mamba1-forward', 'gaussian'), ('neural', 'mlp-train-step', 'gaussian'),
                           ('neural', 'lm-train-step', 'bytes'), ('classical', 'kmeans', 'taxi')},
                    neural_data={'mamba1-forward': 'gaussian', 'mlp-train-step': 'gaussian', 'lm-train-step': 'bytes'},
                    tree_task={}, tree_lanes=[], board_datasets=['taxi', 'istella'])
GEMM_ENV = 'MOJOLEARN_EXPERIMENT_GEMM_PROFILE=mojolearn.identical.gemm.fp32.ni08-leaf256'


def write_neural_plan(d):
    """A plan whose neural workloads carry no dataset (the board decides it), one per lane, plus a classical one."""
    rows = [('G.neural:mamba1-forward.m1=on', 'neural:mamba1-forward', 'P010', ['MOJOLEARN_IDN_M1_SCAN=3'], ['_mojolearn_mamba']),
            ('G.neural:mlp-train-step.leaf=all256', 'neural:mlp-train-step', 'P011', ['MOJOLEARN_IDN_GEMM_LEAF=3'], ['_mojolearn_training']),
            ('G.classical:kmeans.f=on', 'classical:kmeans', 'P012', ['MOJOLEARN_F=1'], ['_mojolearn_x_linear'])]
    configs, packs, algos = [], [], {}
    for cid, algo, pack, defines, binds in rows:
        wid = algo if algo.startswith('neural:') else algo + '@dataset=taxi'
        c = config(cid, algo, pack, defines, [wid], assignment={cid.split('.')[-1].split('=')[0]: 'on'})
        c['grid']['bindings'] = binds
        configs.append(c)
        packs.append(dict(id=pack, members=[cid], defines=defines, bindings=binds, algorithms=[algo], global_reach=False))
        algos[algo] = dict(regime='factorial', workload_bindings=binds)
    cells = [dict(configuration=c['id'], workload_id=w, vendor=v) for c in configs for w in c['workloads'] for v in ('nvidia', 'amd')]
    (d / 'grid-plan.json').write_text(json.dumps(dict(builds=dict(packs=packs), algorithms=algos)))
    with gzip.open(d / 'grid-matrix.json.gz', 'wt') as f:
        json.dump(dict(configurations=configs, cells=cells), f)


class NotReadyTests(unittest.TestCase):
    """status=not_ready = infrastructure: dropped by collect (never FAIL/BROKEN); redo-not-ready re-tags the lines."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        write_plan(self.d)
        self.rid = G.run_id_of(self.d)

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_redo(self):
        self.assertEqual(G.split_redo('P875n1'), ('P875', 1))
        self.assertEqual(G.split_redo('B022r1n2'), ('B022r1', 2))
        self.assertEqual(G.split_redo('C004r1'), ('C004r1', 0))
        self.assertEqual(G.split_redo('P875'), ('P875', 0))
        self.assertEqual(G.split_redo('P005.bb'), ('P005.bb', 0))

    def write_results(self):
        r, qb = self.rid, {'r2': 0.9, 'rmse': 1.0}
        nv, amd = [], []
        n = [0]

        def add(lst, vendor, tag, lane, ds, ms, digest, status='ok', defines=None):
            n[0] += 1
            lst.append(algos_line('%s%04d' % (vendor[0], n[0]), vendor, 'abc1234', tag, lane, ds,
                                  'None' if status != 'ok' else ms, {} if status != 'ok' else qb,
                                  'none' if status != 'ok' else digest, defines, status=status))
        for rep, (mn, ma) in enumerate(((100, 50), (102, 51), (104, 52))):
            for lane, ds in (('ridge-cv', 'taxi'), ('ridge-cv', 'istella'), ('lasso-cv', 'taxi'), ('qda', 'taxi')):
                tag = '%s.B001r%d' % (r, rep + 1)
                if rep == 0 and lane == 'qda':  # incumbent repeat 1 not_ready on nv: the redo fills it
                    add(nv, 'nvidia', tag, lane, ds, mn, 'b' * 16, status='not_ready')
                    add(nv, 'nvidia', tag + 'n1', lane, ds, mn, 'b' * 16)
                else:
                    add(nv, 'nvidia', tag, lane, ds, mn, 'b' * 16)
                add(amd, 'amd', tag, lane, ds, ma, 'b' * 16)
        # P002 ridge-cv taxi: nv not_ready, the redo (n1) is ok -> used; amd ok first time, its redo is ignored
        add(nv, 'nvidia', r + '.P002', 'ridge-cv', 'taxi', 0, '', status='not_ready', defines='MOJOLEARN_C=1')
        add(nv, 'nvidia', r + '.P002n1', 'ridge-cv', 'taxi', 80, 'c' * 16, defines='MOJOLEARN_C=1')
        add(amd, 'amd', r + '.P002', 'ridge-cv', 'taxi', 40, 'c' * 16, defines='MOJOLEARN_C=1')
        add(amd, 'amd', r + '.P002n1', 'ridge-cv', 'taxi', 400, 'z' * 16, defines='MOJOLEARN_C=1')
        # P003 qda: amd not_ready, never redone -> NOT_READY, not FAIL
        add(nv, 'nvidia', r + '.P003', 'qda', 'taxi', 100, '1' * 16, defines='MOJOLEARN_D=1')
        add(amd, 'amd', r + '.P003', 'qda', 'taxi', 0, '', status='not_ready', defines='MOJOLEARN_D=1')
        # P001 lasso-cv: a real error on amd stays FAIL even when a redo ran ok
        add(nv, 'nvidia', r + '.P001', 'lasso-cv', 'taxi', 100, 'f' * 16, defines='MOJOLEARN_A=1')
        add(amd, 'amd', r + '.P001', 'lasso-cv', 'taxi', 0, '', status='error', defines='MOJOLEARN_A=1')
        add(amd, 'amd', r + '.P001n1', 'lasso-cv', 'taxi', 50, 'f' * 16, defines='MOJOLEARN_A=1')
        (self.d / 'nv-results.txt').write_text('\n'.join(nv) + '\n')
        (self.d / 'amd-results.txt').write_text('\n'.join(amd) + '\n')
        return [self.d / 'nv-results.txt', self.d / 'amd-results.txt']

    def test_collect_not_ready_is_infrastructure(self):
        out = self.d / 'collected'
        rep = G.collect(self.write_results(), [], self.d, out, lanes=LANES, board=BOARD)
        self.assertEqual(rep['ignored']['not_ready'], 3)
        self.assertEqual(rep['ignored']['superseded_redo'], 2)  # amd P002n1 behind an ok original, P001n1 behind an error
        self.assertEqual(rep['not_ready']['a_cells'], 1)
        self.assertEqual(rep['not_ready']['a_cells_by_lane'], {'amd:qda': 1})
        self.assertEqual(rep['coverage']['amd'].get('NOT_READY'), 1)
        q = json.loads((out / 'quality.json').read_text())
        rows = {(r['configuration'], r['vendor']): r['candidate_vs_baseline']['verdict'] for r in q['rows']}
        self.assertNotIn(('G.expanded:qda.d=on', 'amd'), rows)
        self.assertEqual(rows[('G.expanded:lasso-cv.b=on', 'amd')], 'FAIL')  # error is never superseded
        self.assertEqual(rows[('G.expanded:ridge-cv.c=on', 'nvidia')], 'SAME')
        v = json.loads((out / 'grid-verdicts.json').read_text())
        c = {(x['configuration'], x['workload_id']): x for x in v['cases']}[('G.expanded:ridge-cv.c=on', 'expanded:ridge-cv@dataset=taxi')]
        self.assertAlmostEqual(c['vendors']['nvidia']['candidate_over_baseline']['scored'], 80 / 102)
        self.assertAlmostEqual(c['vendors']['amd']['candidate_over_baseline']['scored'], 40 / 51)
        self.assertEqual(c['verdict'], 'FASTER')
        floors = json.loads((out / 'floors.json').read_text())['floors']
        self.assertEqual(floors['expanded:qda@dataset=taxi@input=classification-full-v1|nvidia']['samples'], 3)
        dec_out = self.d / 'decide'
        self.assertEqual(D.main(['--matrix', str(self.d / 'grid-matrix.json.gz'), '--verdicts', str(out / 'grid-verdicts.json'),
                                 '--identity', str(out / 'summary.json'), '--quality', str(out / 'quality.json'),
                                 '--out', str(dec_out)]), 0)
        dec = json.loads((dec_out / 'grid-decisions.json').read_text())
        self.assertNotEqual(dec['controls']['d']['recommendation'], 'HOLD_BROKEN')
        self.assertEqual(dec['controls']['b']['recommendation'], 'HOLD_BROKEN')
        failed = {(f['configuration'], f['workload_id']) for f in dec.get('failed_cells', [])}
        self.assertNotIn('G.expanded:qda.d=on', {c for c, _ in failed})

    def lines_file(self):
        r = self.rid
        pre = ' BUILDS=build,build_x_linear PREBUILT=/root/grid-prebuilt'
        lines = [
            'lq add nv RACE grid/f ridge-cv istella,taxi MOJOLEARN_GRID_TAG=%s.P002 MOJOLEARN_BUILD_DEFINES=MOJOLEARN_C=1%s' % (r, pre),
            'lq add nv RACE grid/f qda taxi MOJOLEARN_GRID_TAG=%s.P003 MOJOLEARN_BUILD_DEFINES=MOJOLEARN_D=1%s' % (r, pre),
            'lq add nv RACE grid/f lasso-cv,qda,ridge-cv istella,taxi MOJOLEARN_GRID_TAG=%s.B001r1%s' % (r, pre),
            'lq add nv RACE grid/f lasso-cv,qda,ridge-cv istella,taxi MOJOLEARN_GRID_TAG=%s.B001r2%s' % (r, pre),
            'lq add nv RACE grid/f ridge-cv taxi MOJOLEARN_GRID_TAG=gdeadbeef.P002%s' % pre,
            'lq add amd RACE grid/f ridge-cv istella,taxi MOJOLEARN_GRID_TAG=%s.P002 MOJOLEARN_BUILD_DEFINES=MOJOLEARN_C=1%s' % (r, pre),
            'lq add amd RACE grid/f qda taxi MOJOLEARN_GRID_TAG=%s.P003 MOJOLEARN_BUILD_DEFINES=MOJOLEARN_D=1%s' % (r, pre),
            'lq add amd RACE grid/f lasso-cv taxi MOJOLEARN_GRID_TAG=%s.P001 MOJOLEARN_BUILD_DEFINES=MOJOLEARN_A=1%s' % (r, pre),
        ]
        f = self.d / 'race.pb.lines'
        f.write_text('\n'.join(lines) + '\n')
        return f, lines

    def test_redo_lines(self):
        f, src = self.lines_file()
        r = self.rid
        nv_res = self.d / 'nv-only.txt'
        nv_res.write_text('\n'.join([
            algos_line('n1', 'nvidia', 'abc1234', r + '.P002', 'ridge-cv', 'taxi', 'None', {}, 'none', status='not_ready'),
            algos_line('n1', 'nvidia', 'abc1234', r + '.P002', 'ridge-cv', 'istella', 90, {}, 'c' * 16),
            algos_line('n2', 'nvidia', 'abc1234', r + '.P003', 'qda', 'taxi', 90, {}, 'c' * 16),
            algos_line('n3', 'nvidia', 'abc1234', r + '.B001r1', 'qda', 'taxi', 'None', {}, 'none', status='not_ready'),
            algos_line('n4', 'nvidia', 'abc1234', r + '.B001r2', 'qda', 'taxi', 'None', {}, 'none', status='error'),
            algos_line('n5', 'nvidia', 'abc1234', 'gdeadbeef.P002', 'ridge-cv', 'taxi', 'None', {}, 'none', status='not_ready'),
            algos_line('a1', 'amd', 'abc1234', r + '.P003', 'qda', 'taxi', 'None', {}, 'none', status='not_ready'),
        ]) + '\n')
        lines, man = G.render_not_ready_redo([f], [nv_res], 'nvidia', r)
        self.assertEqual(lines, [src[0].replace(r + '.P002', r + '.P002n1'), src[2].replace(r + '.B001r1', r + '.B001r1n1')])
        self.assertEqual((man['lines'], man['a_lines'], man['b_lines'], man['not_ready_cells']), (2, 1, 1, 2))
        self.assertEqual(man['not_ready_cells_by_lane'], {'qda': 1, 'ridge-cv': 1})
        lines, man = G.render_not_ready_redo([f], [nv_res], 'amd', r)
        self.assertEqual(lines, [src[6].replace(r + '.P003', r + '.P003n1')])
        # a redo of a redo: P002n1 still not_ready -> rep 2; B001r1n1 ok -> done
        nv_res.write_text(nv_res.read_text() + '\n'.join([
            algos_line('n6', 'nvidia', 'abc1234', r + '.P002n1', 'ridge-cv', 'taxi', 'None', {}, 'none', status='not_ready'),
            algos_line('n7', 'nvidia', 'abc1234', r + '.B001r1n1', 'qda', 'taxi', 70, {}, 'c' * 16),
        ]) + '\n')
        lines, man = G.render_not_ready_redo([f], [nv_res], 'nvidia', r, rep=2)
        self.assertEqual(lines, [src[0].replace(r + '.P002', r + '.P002n2')])
        for line in lines:
            self.assertTrue(all(__import__('re').fullmatch(r'[A-Za-z0-9_.,=@:+/-]+', t) for t in line.split()))

    def test_redo_cli(self):
        f, src = self.lines_file()
        r = self.rid
        res = self.d / 'amd-res.txt'
        res.write_text(algos_line('a1', 'amd', 'abc1234', r + '.P003', 'qda', 'taxi', 'None', {}, 'none', status='not_ready') + '\n')
        out = self.d / 'amd.race.notready-redo.pb.lines'
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = G.main(['redo-not-ready', '--vendor', 'amd', '--lines', str(f), '--results', str(res), '--run-id', r, '--out', str(out)])
        self.assertEqual(rc, 0)
        self.assertEqual(out.read_text().splitlines(), [src[6].replace(r + '.P003', r + '.P003n1')])
        self.assertEqual(json.loads(Path(str(out) + '.json').read_text())['lanes'], ['qda'])


class NeuralWorkloadTests(unittest.TestCase):
    """Grid ge123e6f9: the neural CMD races (mamba*, mlp-train-step, gemm leaf arms) failed on the box for tooling
    reasons; render --only-workloads neural: writes a redo under its own run id, and collect --repair-runs lets it
    replace the failed first pass."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        write_plan(self.d)
        write_neural_plan(self.d)
        self.rid = G.run_id_of(self.d)

    def tearDown(self):
        self.tmp.cleanup()

    def test_neural_line(self):
        lines, man = G.render(self.d, 'nvidia', 'grid/freeze-x', 3, lanes=LANES, board=NEURAL_BOARD, run_id='gtestr1n',
                              prebuilt='/root/grid-prebuilt', only_workloads=['neural:'])
        self.assertTrue(lines and all('kmeans' not in l for l in lines))  # only the neural workloads
        self.assertEqual(man['totals']['only_workloads'], ['neural:'])
        m1 = [l for l in lines if ' gtestr1n.P010.bb ' in l][0]
        # the dataset is the board's (bench_board_neural DATA_OF), never one the plan or the runner invents
        self.assertIn('--race neural:mamba1-forward:gaussian', m1)
        self.assertIn('--pip sklearn=scikit-learn==1.7.2,torch=torch', m1)  # the Mamba conductor imports torch
        self.assertIn('BUILDS=build,build_mamba', m1)
        self.assertTrue(m1.endswith('PREBUILT=/root/grid-prebuilt'))
        self.assertNotIn(GEMM_ENV, m1)
        mlp = [l for l in lines if ' gtestr1n.P011.bb ' in l][0]
        self.assertIn('--race neural:mlp-train-step:gaussian', mlp)
        self.assertIn('BUILDS=build,build_linalg,build_training', mlp)  # SmallMLPTrainer requires the linalg binding
        self.assertIn(' ' + GEMM_ENV + ' ', mlp)  # the all256 leaf binary refuses without its expected profile
        self.assertNotIn('--pip', mlp)
        b = [l for l in lines if '.C00' in l]
        self.assertEqual(len(b), 6)  # two incumbent groups (binding sets mamba / training+linalg) x 3 repeats
        self.assertTrue(any('build_linalg' in l and '--race neural:mlp-train-step:gaussian' in l for l in b))
        self.assertTrue(all(GEMM_ENV not in l and 'MOJOLEARN_BUILD_DEFINES' not in l for l in b))
        self.assertTrue(all(l.split()[5].startswith('gtestr1n.') for l in lines))  # new tags: the fed ledger skips none
        self.assertEqual(G.define_envs(['MOJOLEARN_IDN_GEMM_LEAF=3', 'MOJOLEARN_IDN_ALL_OFF']), [])
        self.assertTrue(G.workload_selected('neural:gemm', ['neural:gemm']))
        self.assertFalse(G.workload_selected('expanded:cnn-clf@dataset=synthetic', ['neural:']))

    def test_cli_only_workloads(self):
        bj = self.d / 'board.json'
        bj.write_text(json.dumps(dict(NEURAL_BOARD, races=sorted(NEURAL_BOARD['races']))))
        lj = self.d / 'lanes.json'
        lj.write_text(json.dumps(LANES))
        out = self.d / 'amd.cmd.neural-redo.pb.lines'
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            rc = G.main(['render', '--plan-dir', str(self.d), '--vendor', 'amd', '--branch', 'grid/freeze-x', '--run-id',
                         self.rid + 'r1n', '--only-workloads', 'neural:', '--prebuilt', '/root/grid-prebuilt',
                         '--lanes-json', str(lj), '--board-json', str(bj), '--out', str(out)])
        self.assertEqual(rc, 0)
        lines = out.read_text().splitlines()
        self.assertTrue(lines and all(l.startswith('lq add amd CMD grid/freeze-x %sr1n.' % self.rid) for l in lines))
        self.assertTrue(all('--vendor amd' in l for l in lines))

    def gridbb(self, jid, vendor, tag, lane, status, ms, h):
        return ('/root/lq/out/%s/%s.log:GRIDBB tag=%s vendor=%s head=abc1234 family=neural lane=%s dataset=gaussian '
                'status=%s median_ms=%s hash=%s quality=%s' % (jid, tag, tag, vendor, lane, status, ms, h,
                                                              json.dumps({'rmse': 0.5} if status == 'ok' else {})))

    def test_collect_maps_back_and_repairs(self):
        first, redo = self.rid, self.rid + 'r1n'
        logs = []
        n = 0
        for vendor, scale in (('nvidia', 1.0), ('amd', 0.5)):
            for lane in ('mamba1-forward', 'mlp-train-step'):
                pack = 'P010' if lane.startswith('mamba') else 'P011'
                n += 1
                logs.append(self.gridbb('x%03d' % n, vendor, '%s.%s.bb' % (first, pack), lane, 'NO-RECORD', 'None', 'none'))
                n += 1
                logs.append(self.gridbb('x%03d' % n, vendor, '%s.%s.bb' % (redo, pack), lane, 'ok', 80 * scale, 'a' * 16))
                for r in (1, 2, 3):
                    n += 1
                    logs.append(self.gridbb('x%03d' % n, vendor, '%s.C001r%d' % (first, r), lane, 'NO-RECORD', 'None', 'none'))
                    n += 1
                    logs.append(self.gridbb('x%03d' % n, vendor, '%s.C001r%d' % (redo, r), lane, 'ok', (100 + r) * scale, 'b' * 16))
        dump = self.d / 'logs.txt'
        dump.write_text('\n'.join(logs) + '\n')
        res = self.d / 'results.txt'
        res.write_text('')
        rep = G.collect([res], [dump], self.d, self.d / 'c1', lanes=LANES, board=NEURAL_BOARD, run_id=[first],
                        repair_runs=[redo])
        self.assertEqual(rep['run_ids'], [first, redo])
        self.assertEqual(rep['ignored']['unknown_cell'], 0)
        v = {(c['configuration'], c['workload_id']): c for c in json.loads((self.d / 'c1' / 'grid-verdicts.json').read_text())['cases']}
        m1 = v[('G.neural:mamba1-forward.m1=on', 'neural:mamba1-forward')]  # the GRIDBB row maps back to the plan's id
        self.assertEqual(m1['verdict'], 'FASTER')
        self.assertAlmostEqual(m1['vendors']['nvidia']['candidate_over_baseline']['scored'], 80 / 102)
        q = {(r['workload_id'], r['vendor']): r for r in json.loads((self.d / 'c1' / 'quality.json').read_text())['rows']}
        row = q[('neural:mlp-train-step', 'amd')]
        self.assertEqual(row['candidate_vs_baseline']['verdict'], 'SAME')
        self.assertEqual(row['superseded_fail_passes'], [first])
        self.assertEqual(row['pass_verdicts'][first], 'FAIL')  # the failure stays as evidence
        ids = {(c['configuration_id'], c['workload_id']): c for c in json.loads((self.d / 'c1' / 'summary.json').read_text())['cases']}
        self.assertEqual(ids[('G.neural:mamba1-forward.m1=on', 'neural:mamba1-forward')]['arms']['A']['status'], 'MATCH')
        # without --repair-runs the failed first pass wins the merge
        G.collect([res], [dump], self.d, self.d / 'c2', lanes=LANES, board=NEURAL_BOARD, run_id=[first, redo])
        q2 = {(r['workload_id'], r['vendor']): r for r in json.loads((self.d / 'c2' / 'quality.json').read_text())['rows']}
        self.assertEqual(q2[('neural:mlp-train-step', 'amd')]['candidate_vs_baseline']['verdict'], 'FAIL')

    def test_runner_installs_torch_for_mamba(self):
        models = BBW.neural_models()
        self.assertEqual(models['mamba1-forward'], 'mamba1')
        self.assertEqual(BBW.pip_extra([('neural', 'mamba2-forward', 'gaussian')], models), [('torch', 'torch')])
        self.assertEqual(BBW.pip_extra([('neural', 'mlp-train-step', 'gaussian'), ('trees', 'rf', 'taxi')], models), [])
        self.assertEqual(BBW.pip_arg([('neural', 'transformer-forward', 'gaussian')], models), BBW.DEFAULT_PIP)
        self.assertIn(('neural', ['mamba1-forward'], ['taxi']), BBW.groups([('neural', 'mamba1-forward', 'gaussian')]))
        self.assertEqual(BBW.race_ids('neural', 'mamba1-forward', 'gaussian'), ['neural/mamba1-forward/gaussian/shape=full'])


if __name__ == '__main__':
    unittest.main()
