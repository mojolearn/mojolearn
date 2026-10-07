#!/usr/bin/env python3
"""Roll the IDENTICAL switch-grid results up into one decision per switch arm.

The grid (tools/six_lane_grid.py) plans configurations per algorithm; the timing tool judges each
configuration x workload cell (FASTER / SLOWER / NO_VERDICT); the comparator decides NVIDIA == AMD
identity per cell; the quality review scores the candidate against the incumbent per cell. None of them
answers the question the grid exists for: which switch should be ON, for which algorithm, and which
arm. This tool does, from those files only. It never fits, builds or reads receipts.

Inputs
  --matrix    experiments/six_lane_integration/grid/grid-matrix.json.gz   (configurations: algorithm, assignment, tier)
  --verdicts  grid-verdicts.json from `six_lane_timing.py verdicts`        (optional; repeatable)
  --identity  summary.json from `six_lane_compare_results.py`              (optional; repeatable)
  --quality   quality-review.json rows (candidate_vs_baseline.verdict)      (optional; repeatable)

Outputs (in --out): grid-decisions.json and GRID_DECISIONS.md.

Rules (CLAUDE.md "Measurement process" and "Experiments"):
  * a cell counts only with a timing verdict, NVIDIA == AMD identity MATCH and quality not WORSE;
  * an algorithm is FASTER for an arm when no workload is SLOWER and at least one is FASTER, NEUTRAL when
    every measured workload is NO_VERDICT, SLOWER when any workload is SLOWER;
  * a switch arm is PROMOTE when every algorithm it reaches is FASTER or NEUTRAL and at least one is FASTER;
    SPLIT when some algorithms are FASTER and others SLOWER (flip it per algorithm, never globally);
    DELETE when it is SLOWER everywhere or NEUTRAL everywhere (noise is a loser);
    HOLD_IDENTITY on any MISMATCH, HOLD_QUALITY on any WORSE; NOT_MEASURED until every reached algorithm
    has evidence (partial evidence is still listed);
  * for a control with several arms, the best arm is the PROMOTE/SPLIT arm with the smallest combined
    (geometric mean over vendors and workloads) candidate/incumbent time ratio;
  * a cross or all-on configuration is an INTERACTION when its verdict is not what its member singles
    predict (SLOWER although every member alone was FASTER or NEUTRAL; FASTER although a member alone was
    SLOWER). Interactions are listed per algorithm and never averaged away;
  * per algorithm, the recommended configuration is the measured FASTER configuration (single, cross or
    all-on) with identity MATCH, quality not WORSE and the smallest combined ratio.
"""
import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'experiments/six_lane_integration/grid/grid-matrix.json.gz'
OUT_DIR = ROOT / 'experiments/six_lane_integration/grid'
SCHEMA = 'mojolearn.six-lane-grid-decisions/1'

TIMING_RANK = {'SLOWER': 0, 'FASTER': 1, 'NO_VERDICT': 2, 'UNMEASURED': 3}
QUALITY_BAD = {'WORSE', 'FAIL', 'REGRESSED'}
QUALITY_OK = {'SAME', 'BETTER', 'IMPROVED', 'EQUAL'}


def load_json(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def load_many(paths):
    return [load_json(p) for p in (paths or [])]


# ---------------------------------------------------------------- evidence indexes

def timing_index(verdict_docs):
    """(configuration, workload_id) -> {verdict, ratios{vendor: scored ratio}}."""
    out = {}
    for doc in verdict_docs:
        for case in doc.get('cases', []):
            ratios = {}
            for vendor, row in (case.get('vendors') or {}).items():
                r = (row.get('candidate_over_baseline') or {}).get('scored')
                if r is not None and math.isfinite(r) and r > 0:
                    ratios[vendor] = r
            out[(case['configuration'], case['workload_id'])] = dict(verdict=case.get('verdict', 'NO_VERDICT'),
                                                                       ratios=ratios, evidence=[v.get('evidence') for v in (case.get('vendors') or {}).values()])
    return out


def identity_index(summary_docs):
    """(configuration, workload_id) -> MATCH | MISMATCH | INCOMPLETE (candidate arm A; arm B is the incumbent)."""
    out = {}
    for doc in summary_docs:
        for case in doc.get('cases', []):
            arms = case.get('arms') or {}
            status = (arms.get('A') or {}).get('status') or case.get('status') or 'INCOMPLETE'
            key = (case.get('configuration_id') or case.get('configuration'), case.get('workload_id'))
            prev = out.get(key)
            # MISMATCH wins over anything; MATCH over INCOMPLETE.
            if prev is None or status == 'MISMATCH' or (status == 'MATCH' and prev == 'INCOMPLETE'):
                out[key] = status
    return out


def quality_index(quality_docs):
    """(configuration, workload_id) -> worst verdict over vendors (WORSE beats PENDING beats SAME/BETTER)."""
    out = {}
    for doc in quality_docs:
        rows = doc.get('rows') or doc.get('cases') or []
        for row in rows:
            cfg = row.get('configuration') or row.get('configuration_id')
            wid = row.get('workload_id')
            if not wid and row.get('lane') and row.get('dataset'):
                wid = ('lane', row['lane'], row['dataset'])
            verdict = ((row.get('candidate_vs_baseline') or {}).get('verdict') or row.get('verdict') or 'PENDING').upper()
            key = (cfg, wid)
            prev = out.get(key)
            if prev is None or verdict in QUALITY_BAD or (prev in QUALITY_OK and verdict not in QUALITY_OK):
                out[key] = verdict
    return out


def quality_for(qidx, cfg, wid):
    direct = qidx.get((cfg, wid))
    if direct is not None:
        return direct
    # quality rows that carry lane + dataset instead of a workload id
    if '@dataset=' in wid:
        algo, ds = wid.split('@dataset=', 1)
        lane = algo.split(':', 1)[-1]
        return qidx.get((cfg, ('lane', lane, ds)), 'PENDING')
    return 'PENDING'


# ---------------------------------------------------------------- roll-up

def geo_mean(values):
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def cell_rows(config, tidx, iidx, qidx):
    rows = []
    for wid in config.get('workloads', []):
        t = tidx.get((config['id'], wid))
        rows.append(dict(workload_id=wid,
                         timing=t['verdict'] if t else 'UNMEASURED',
                         ratios=t['ratios'] if t else {},
                         identity=iidx.get((config['id'], wid), 'UNMEASURED'),
                         quality=quality_for(qidx, config['id'], wid)))
    return rows


def algorithm_verdict(rows):
    """Collapse a configuration's workload rows for one algorithm."""
    timings = [r['timing'] for r in rows]
    identities = [r['identity'] for r in rows]
    qualities = [r['quality'] for r in rows]
    if any(i == 'MISMATCH' for i in identities):
        return 'HOLD_IDENTITY'
    if any(q in QUALITY_BAD for q in qualities):
        return 'HOLD_QUALITY'
    if any(t == 'UNMEASURED' for t in timings):
        return 'UNMEASURED'
    if any(i != 'MATCH' for i in identities):
        return 'IDENTITY_INCOMPLETE'
    if any('SLOWER' == t for t in timings):
        return 'SLOWER'
    if any('FASTER' == t for t in timings):
        return 'FASTER'
    return 'NEUTRAL'


def arm_recommendation(per_algorithm):
    """per_algorithm: {algorithm: verdict}."""
    verdicts = set(per_algorithm.values())
    if not verdicts:
        return 'NOT_MEASURED'
    if 'HOLD_IDENTITY' in verdicts:
        return 'HOLD_IDENTITY'
    if 'HOLD_QUALITY' in verdicts:
        return 'HOLD_QUALITY'
    if 'UNMEASURED' in verdicts or 'IDENTITY_INCOMPLETE' in verdicts:
        return 'NOT_MEASURED'
    if 'SLOWER' in verdicts and 'FASTER' in verdicts:
        return 'SPLIT'
    if 'FASTER' in verdicts:
        return 'PROMOTE'
    return 'DELETE'  # SLOWER everywhere, or noise everywhere


def predicted_from_singles(single_verdicts):
    """What the member singles predict for a cross: SLOWER if any member is SLOWER, else FASTER/NEUTRAL."""
    if any(v == 'SLOWER' for v in single_verdicts):
        return 'SLOWER'
    if all(v in ('FASTER', 'NEUTRAL') for v in single_verdicts) and single_verdicts:
        return 'NOT_SLOWER'
    return 'UNKNOWN'


def decide(matrix, tidx, iidx, qidx):
    configs = [c for c in matrix.get('configurations', []) if (c.get('grid') or {}).get('algorithm')]
    by_algo = defaultdict(list)
    for c in configs:
        by_algo[c['grid']['algorithm']].append(c)

    # per configuration evidence
    evidence = {}
    for c in configs:
        rows = cell_rows(c, tidx, iidx, qidx)
        ratios = [geo_mean(r['ratios'].values()) for r in rows]
        evidence[c['id']] = dict(configuration=c['id'], algorithm=c['grid']['algorithm'], tier=c['grid'].get('tier'),
                                 assignment=c['grid'].get('assignment', {}), rows=rows,
                                 verdict=algorithm_verdict(rows), combined_ratio=geo_mean(ratios))

    # singles per (control, arm) per algorithm
    single_by_arm = defaultdict(dict)  # (control, arm) -> {algorithm: evidence}
    for c in configs:
        g = c['grid']
        if g.get('tier') != 'single':
            continue
        assign = g.get('assignment', {})
        if len(assign) != 1:
            # a child control carries its parent's representative arm; attribute to the non-parent control when unique
            continue
        (ctrl, arm), = assign.items()
        single_by_arm[(ctrl, arm)][g['algorithm']] = evidence[c['id']]

    controls = defaultdict(dict)  # control -> arm -> decision
    for (ctrl, arm), algos in sorted(single_by_arm.items()):
        per_algo = {a: e['verdict'] for a, e in algos.items()}
        rec = arm_recommendation(per_algo)
        controls[ctrl][arm] = dict(recommendation=rec, per_algorithm=per_algo,
                                   combined_ratio=geo_mean([e['combined_ratio'] for e in algos.values()]),
                                   configurations=sorted(e['configuration'] for e in algos.values()),
                                   promote_for=sorted(a for a, v in per_algo.items() if v == 'FASTER') if rec == 'SPLIT' else [],
                                   refuse_for=sorted(a for a, v in per_algo.items() if v == 'SLOWER') if rec == 'SPLIT' else [])

    # best arm per control
    control_summary = {}
    for ctrl, arms in controls.items():
        eligible = [(a, d) for a, d in arms.items() if d['recommendation'] in ('PROMOTE', 'SPLIT') and d['combined_ratio']]
        best = min(eligible, key=lambda ad: ad[1]['combined_ratio'])[0] if eligible else None
        recs = {d['recommendation'] for d in arms.values()}
        if best:
            overall = arms[best]['recommendation']
        elif recs <= {'DELETE'}:
            overall = 'DELETE'
        elif 'HOLD_IDENTITY' in recs:
            overall = 'HOLD_IDENTITY'
        elif 'HOLD_QUALITY' in recs:
            overall = 'HOLD_QUALITY'
        else:
            overall = 'NOT_MEASURED'
        control_summary[ctrl] = dict(control=ctrl, recommendation=overall, best_arm=best, arms=arms,
                                     algorithms=sorted({a for d in arms.values() for a in d['per_algorithm']}))

    # interactions and per-algorithm recommendation
    algorithms = {}
    for algo, cfgs in sorted(by_algo.items()):
        single_verdict = {}
        for c in cfgs:
            g = c['grid']
            if g.get('tier') == 'single' and len(g.get('assignment', {})) == 1:
                (ctrl, arm), = g['assignment'].items()
                single_verdict[(ctrl, arm)] = evidence[c['id']]['verdict']
        interactions = []
        for c in cfgs:
            g = c['grid']
            if g.get('tier') not in ('cross', 'all_on'):
                continue
            ev = evidence[c['id']]
            if ev['verdict'] in ('UNMEASURED', 'IDENTITY_INCOMPLETE'):
                continue
            members = [single_verdict.get((k, v), 'UNMEASURED') for k, v in sorted(g.get('assignment', {}).items())]
            predicted = predicted_from_singles(members)
            flagged = (ev['verdict'] == 'SLOWER' and predicted == 'NOT_SLOWER') or \
                      (ev['verdict'] == 'FASTER' and predicted == 'SLOWER')
            if flagged:
                interactions.append(dict(configuration=c['id'], tier=g['tier'], assignment=g.get('assignment', {}),
                                         verdict=ev['verdict'], member_singles=members, predicted=predicted,
                                         combined_ratio=ev['combined_ratio']))
        measured = [evidence[c['id']] for c in cfgs]
        winners = [e for e in measured if e['verdict'] == 'FASTER' and e['combined_ratio']]
        best = min(winners, key=lambda e: e['combined_ratio']) if winners else None
        counts = defaultdict(int)
        for e in measured:
            counts[e['verdict']] += 1
        algorithms[algo] = dict(algorithm=algo, configurations=len(cfgs), verdict_counts=dict(sorted(counts.items())),
                                recommended=None if not best else dict(configuration=best['configuration'], assignment=best['assignment'],
                                                                         combined_ratio=best['combined_ratio'], tier=best['tier']),
                                interactions=interactions,
                                losers=sorted(e['configuration'] for e in measured if e['verdict'] == 'SLOWER'))

    totals = defaultdict(int)
    for cs in control_summary.values():
        totals[cs['recommendation']] += 1
    return dict(schema=SCHEMA, matrix_schema=matrix.get('schema'), base_main=matrix.get('base_main'),
                counts=dict(controls=len(control_summary), arms=sum(len(c['arms']) for c in control_summary.values()),
                            configurations=len(configs), by_recommendation=dict(sorted(totals.items()))),
                controls=dict(sorted(control_summary.items())), algorithms=algorithms,
                configurations={k: dict(v, rows=v['rows']) for k, v in sorted(evidence.items())},
                rule=dict(cell='timing verdict + NVIDIA==AMD identity MATCH + quality not WORSE',
                          algorithm='SLOWER if any workload SLOWER; FASTER if any FASTER and none SLOWER; NEUTRAL otherwise',
                          arm='PROMOTE: every reached algorithm FASTER or NEUTRAL, at least one FASTER. SPLIT: FASTER and SLOWER both present '
                              '(flip per algorithm). DELETE: SLOWER or NEUTRAL everywhere (noise is a loser). HOLD_*: identity MISMATCH or quality WORSE. '
                              'NOT_MEASURED: any reached algorithm without evidence.',
                          best_arm='PROMOTE/SPLIT arm with the smallest combined candidate/incumbent ratio',
                          interaction='cross or all-on verdict that the member singles do not predict',
                          recommended_configuration='per algorithm: FASTER configuration with the smallest combined ratio'))


# ---------------------------------------------------------------- rendering

def fmt_ratio(r):
    return '-' if r is None else '%.3fx' % r


def render_md(dec):
    L = ['# IDENTICAL switch grid decisions', '',
         'Generated by `tools/six_lane_grid_decide.py` from the grid matrix plus the timing verdicts, NVIDIA==AMD identity and quality files.',
         'A switch is **NOT_MEASURED** until every algorithm it reaches has a cell with a timing verdict, identity MATCH and quality not WORSE.',
         '', '## Totals', '', '| item | count |', '|---|---|']
    L.append('| controls | %d |' % dec['counts']['controls'])
    L.append('| arms | %d |' % dec['counts']['arms'])
    L.append('| configurations | %d |' % dec['counts']['configurations'])
    for k, v in dec['counts']['by_recommendation'].items():
        L.append('| %s | %d |' % (k, v))
    L += ['', '## Rule', '']
    for k, v in dec['rule'].items():
        L.append('- **%s**: %s' % (k, v))
    L += ['', '## Switches', '', '| control | recommendation | best arm | arms (recommendation, combined ratio) | algorithms | split |', '|---|---|---|---|---|---|']
    for ctrl, cs in dec['controls'].items():
        arms = '<br>'.join('%s: %s %s' % (a, d['recommendation'], fmt_ratio(d['combined_ratio'])) for a, d in cs['arms'].items())
        split = ''
        for a, d in cs['arms'].items():
            if d['recommendation'] == 'SPLIT':
                split += '%s: on for %s; off for %s<br>' % (a, ', '.join(d['promote_for']), ', '.join(d['refuse_for']))
        L.append('| %s | %s | %s | %s | %s | %s |' % (ctrl, cs['recommendation'], cs['best_arm'] or '-', arms, ', '.join(cs['algorithms']), split or '-'))
    L += ['', '## Algorithms', '', '| algorithm | configs | verdicts | recommended configuration | interactions | losers |', '|---|---|---|---|---|---|']
    for algo, a in dec['algorithms'].items():
        rec = a['recommended']
        rec_s = '-' if not rec else '%s (%s, %s)' % (', '.join('%s=%s' % kv for kv in sorted(rec['assignment'].items())), rec['tier'], fmt_ratio(rec['combined_ratio']))
        inter = '<br>'.join('%s: %s (singles predicted %s)' % (', '.join('%s=%s' % kv for kv in sorted(i['assignment'].items())), i['verdict'], i['predicted'])
                            for i in a['interactions']) or '-'
        verdicts = ', '.join('%s %d' % kv for kv in a['verdict_counts'].items())
        L.append('| %s | %d | %s | %s | %s | %s |' % (algo, a['configurations'], verdicts, rec_s, inter, '<br>'.join(a['losers']) or '-'))
    L.append('')
    return '\n'.join(L)


def write_outputs(dec, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'grid-decisions.json').write_text(json.dumps(dec, indent=1, sort_keys=True) + '\n')
    (out_dir / 'GRID_DECISIONS.md').write_text(render_md(dec))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--matrix', type=Path, default=MATRIX)
    p.add_argument('--verdicts', type=Path, action='append', help='six_lane_timing.py verdicts output (repeatable)')
    p.add_argument('--identity', type=Path, action='append', help='six_lane_compare_results.py summary.json (repeatable)')
    p.add_argument('--quality', type=Path, action='append', help='quality review rows (repeatable)')
    p.add_argument('--out', type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    dec = decide(load_json(args.matrix), timing_index(load_many(args.verdicts)),
                 identity_index(load_many(args.identity)), quality_index(load_many(args.quality)))
    write_outputs(dec, args.out)
    print(json.dumps(dec['counts']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
