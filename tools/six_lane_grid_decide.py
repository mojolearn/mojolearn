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
  --phase2-matrix  phase-2 grid-matrix.json.gz (six_lane_grid.py --survivors), merged with --matrix so
              phase-2 crosses are judged against the phase-1 singles (optional)

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

--mode fast (Apple FAST grid, tools/six_lane_grid.py --mode fast --vendor apple; CLAUDE.md "FAST mode needs no
identical anything"): the matrix and outputs default to experiments/six_lane_integration/grid-fast/; identity is
NOT_REQUIRED on every cell (no --identity input, no identity hold); one vendor votes (apple, the M3): its own
per-vendor verdict when the verdict file carries one, else the case verdict, and only its ratio enters the combined
ratio; the quality gate is the board quality metric of the candidate (arm B) vs FAST main (arm A), the af_quality
verdict in --quality (WORSE holds the arm). Everything else (algorithm, arm, best arm, interactions, recommended
configuration) is the same rule.
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
FAST_MATRIX = ROOT / 'experiments/six_lane_integration/grid-fast/grid-matrix.json.gz'
FAST_OUT_DIR = ROOT / 'experiments/six_lane_integration/grid-fast'
FAST_VOTERS = ('apple',)

TIMING_RANK = {'SLOWER': 0, 'FASTER': 1, 'NO_VERDICT': 2, 'UNMEASURED': 3}
QUALITY_BAD = {'WORSE', 'REGRESSED'}  # FAIL = the candidate race itself failed: the cell is unmeasured, not a quality loss
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

def timing_index(verdict_docs, voters=None):
    """(configuration, workload_id) -> {verdict, ratios{vendor: scored ratio}}.

    voters (fast mode): only these vendors count; a voter's own `verdict` wins over the case verdict."""
    out = {}
    for doc in verdict_docs:
        for case in doc.get('cases', []):
            ratios = {}
            vendors = {v: row for v, row in (case.get('vendors') or {}).items() if voters is None or v in voters}
            for vendor, row in vendors.items():
                r = (row.get('candidate_over_baseline') or {}).get('scored')
                if r is not None and math.isfinite(r) and r > 0:
                    ratios[vendor] = r
            verdict = case.get('verdict', 'NO_VERDICT')
            if voters is not None:
                own = [row.get('verdict') for row in vendors.values() if row.get('verdict')]
                if own:
                    verdict = own[0]
                elif not vendors and case.get('vendors'):
                    verdict = 'UNMEASURED'  # only non-voting vendors measured this cell
            reasons = ((case.get('phases') or {}).get('scored') or {}).get('reasons') or []
            if verdict == 'NO_VERDICT' and any('within floor' not in r for r in reasons):
                verdict = 'INCOMPLETE'  # a voter has no pair, no floor or no phase: not a neutral result
            # A provisional floor (fewer than 2 incumbent repeats yet) decides only large effects: a neutral or
            # modest result under it is INCOMPLETE until the real floor arrives (no DELETE-as-noise on a guess).
            provisional = any(str((row.get('floor_source') or '')).startswith('provisional') for row in vendors.values())
            if provisional and verdict in ('NO_VERDICT', 'FASTER', 'SLOWER'):
                effects = [abs((row.get('log_ratio') or {}).get('scored') or 0.0) for row in vendors.values()]
                if not effects or min(effects) < math.log(2.0):
                    verdict = 'INCOMPLETE'
            out[(case['configuration'], case['workload_id'])] = dict(verdict=verdict, reasons=reasons,
                                                                       ratios=ratios, evidence=[v.get('evidence') for v in vendors.values()])
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


def cell_rows(config, tidx, iidx, qidx, identity_required=True):
    rows = []
    for wid in config.get('workloads', []):
        t = tidx.get((config['id'], wid))
        rows.append(dict(workload_id=wid,
                         timing=t['verdict'] if t else 'UNMEASURED',
                         ratios=t['ratios'] if t else {},
                         identity=iidx.get((config['id'], wid), 'UNMEASURED') if identity_required else 'NOT_REQUIRED',
                         quality=quality_for(qidx, config['id'], wid)))
    return rows


def algorithm_verdict(rows):
    """Collapse a configuration's workload rows for one algorithm (identity NOT_REQUIRED rows skip identity)."""
    timings = [r['timing'] for r in rows]
    identities = [r['identity'] for r in rows if r['identity'] != 'NOT_REQUIRED']
    qualities = [r['quality'] for r in rows]
    if any(i == 'MISMATCH' for i in identities):
        return 'HOLD_IDENTITY'
    if any(q in QUALITY_BAD for q in qualities):
        return 'HOLD_QUALITY'
    if any(t in ('UNMEASURED', 'INCOMPLETE') for t in timings) or any(q == 'FAIL' for q in qualities):
        return 'UNMEASURED'  # a failed candidate race leaves the cell unmeasured (it is counted in failed_cells)
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


def merge_matrices(phase1, phase2):
    """Phase-1 + phase-2 grid matrices as one (phase-2 reuses phase-1 ids only for the same assignment)."""
    out = dict(phase1)
    by_id = {c['id']: c for c in phase1.get('configurations', [])}
    merged = list(phase1.get('configurations', []))
    for c in phase2.get('configurations', []):
        prev = by_id.get(c['id'])
        if prev is not None:
            if (prev.get('grid') or {}).get('assignment') != (c.get('grid') or {}).get('assignment'):
                raise ValueError('phase-2 configuration ' + c['id'] + ' reuses a phase-1 id with another assignment')
            continue
        merged.append(dict(c, grid=dict(c.get('grid') or {}, phase=2)))
    out['configurations'] = merged
    out['cells'] = list(phase1.get('cells', [])) + [x for x in phase2.get('cells', []) if x.get('configuration') not in by_id]
    out['merged_phase2'] = dict(schema=phase2.get('schema'), configurations=len(merged) - len(by_id))
    return out


def decide(matrix, tidx, iidx, qidx, mode='identical'):
    fast = mode == 'fast'
    configs = [c for c in matrix.get('configurations', []) if (c.get('grid') or {}).get('algorithm')]
    by_algo = defaultdict(list)
    for c in configs:
        by_algo[c['grid']['algorithm']].append(c)

    # per configuration evidence
    evidence = {}
    for c in configs:
        rows = cell_rows(c, tidx, iidx, qidx, identity_required=not fast)
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

    # An algorithm is complete when every configuration that reaches it has a verdict (singles, crosses, factorial).
    # Winners and losers are declared only then: a neutral single is not deleted before its crosses are judged.
    complete = {algo: all(evidence[c['id']]['verdict'] not in ('UNMEASURED', 'IDENTITY_INCOMPLETE') for c in cfgs)
                for algo, cfgs in by_algo.items()}
    controls = defaultdict(dict)  # control -> arm -> decision
    for (ctrl, arm), algos in sorted(single_by_arm.items()):
        per_algo = {a: e['verdict'] for a, e in algos.items()}
        rec = arm_recommendation(per_algo)
        if rec in ('PROMOTE', 'DELETE', 'SPLIT') and not all(complete.get(a, False) for a in per_algo):
            rec = 'PARTIAL_' + rec  # points this way so far; a reached algorithm still has unmeasured configurations
        controls[ctrl][arm] = dict(recommendation=rec, per_algorithm=per_algo,
                                   algorithms_complete={a: complete.get(a, False) for a in per_algo},
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
        elif any(r.startswith('PARTIAL_') for r in recs):
            overall = sorted(r for r in recs if r.startswith('PARTIAL_'))[0]
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
            if g.get('tier') in (None, 'single'):
                continue  # every multi-control tier (cross, all_on, triple, factorial, all_survivors, cross_across)
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
    failed_cells = [dict(configuration=c, workload_id=w) for c, w in sorted({(cfg['configuration'], r['workload_id']) for cfg in evidence.values() for r in cfg['rows'] if r['quality'] == 'FAIL'})]
    rule = dict(cell='timing verdict + NVIDIA==AMD identity MATCH + quality not WORSE',
                algorithm='SLOWER if any workload SLOWER; FASTER if any FASTER and none SLOWER; NEUTRAL otherwise',
                arm='PROMOTE: every reached algorithm FASTER or NEUTRAL, at least one FASTER. SPLIT: FASTER and SLOWER both present '
                    '(flip per algorithm). DELETE: SLOWER or NEUTRAL everywhere (noise is a loser). HOLD_*: identity MISMATCH or quality WORSE. '
                    'NOT_MEASURED: any reached algorithm without evidence. PARTIAL_<rec>: the arm points that way but a reached algorithm still has unmeasured configurations (nothing is flipped or deleted before that algorithm grid is complete). A NO_VERDICT with a missing pair, floor or phase is INCOMPLETE, not neutral.',
                best_arm='PROMOTE/SPLIT arm with the smallest combined candidate/incumbent ratio',
                interaction='cross or all-on verdict that the member singles do not predict',
                recommended_configuration='per algorithm: FASTER configuration with the smallest combined ratio')
    if fast:
        rule.update(cell='M3 timing verdict (beyond the M3 A/A floor) + quality not WORSE vs FAST main (arm A); identity NOT_REQUIRED',
                    identity='NOT_REQUIRED: FAST needs no identical anything (no same bits across vendors, vs arm A, or run to run)',
                    voters='apple (the M3) only',
                    quality='board quality metric of B vs FAST main (A), tools/af_quality.py rel 1e-3 / abs 1e-6 (the af_board_apply gate); WORSE holds',
                    arm=rule['arm'].replace('HOLD_*: identity MISMATCH or quality WORSE', 'HOLD_QUALITY: quality WORSE'),
                    promotion='a PROMOTE arm becomes the FAST default with a *_OFF rollback and an EXPERIMENTS.md row; DELETE arms are removed from the code')
    extra = {}
    if matrix.get('merged_phase2'):
        extra['merged_phase2'] = matrix['merged_phase2']
        rule['phase2'] = ('phase-1 and phase-2 matrices merged: interactions of phase-2 crosses are judged against the '
                          'phase-1 singles of the same algorithm')
    return dict(schema=SCHEMA, mode=mode, matrix_schema=matrix.get('schema'), base_main=matrix.get('base_main'), **extra,
                counts=dict(controls=len(control_summary), arms=sum(len(c['arms']) for c in control_summary.values()),
                            configurations=len(configs), by_recommendation=dict(sorted(totals.items())), failed_cells=len(failed_cells)),
                failed_cells=sorted(failed_cells, key=lambda f: (f['configuration'], f['workload_id'])),
                controls=dict(sorted(control_summary.items())), algorithms=algorithms,
                configurations={k: dict(v, rows=v['rows']) for k, v in sorted(evidence.items())},
                rule=rule)


# ---------------------------------------------------------------- rendering

def fmt_ratio(r):
    return '-' if r is None else '%.3fx' % r


def render_md(dec):
    if dec.get('mode') == 'fast':
        L = ['# FAST switch grid decisions (Apple M3)', '',
             'Generated by `tools/six_lane_grid_decide.py --mode fast` from the FAST grid matrix plus the M3 timing verdicts and quality files.',
             'Identity is NOT_REQUIRED (FAST needs no identical anything). A switch is **NOT_MEASURED** until every lane it reaches has a cell',
             'with an M3 timing verdict and quality not WORSE than FAST main.']
    else:
        L = ['# IDENTICAL switch grid decisions', '',
             'Generated by `tools/six_lane_grid_decide.py` from the grid matrix plus the timing verdicts, NVIDIA==AMD identity and quality files.',
             'A switch is **NOT_MEASURED** until every algorithm it reaches has a cell with a timing verdict, identity MATCH and quality not WORSE.']
    L += [
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
    p.add_argument('--mode', choices=('identical', 'fast'), default='identical',
                   help='fast: Apple FAST grid (identity NOT_REQUIRED, apple the one voter, grid-fast/ defaults)')
    p.add_argument('--matrix', type=Path)
    p.add_argument('--verdicts', type=Path, action='append', help='six_lane_timing.py verdicts output (repeatable)')
    p.add_argument('--identity', type=Path, action='append', help='six_lane_compare_results.py summary.json (repeatable)')
    p.add_argument('--quality', type=Path, action='append', help='quality review rows (repeatable)')
    p.add_argument('--phase2-matrix', type=Path, help='phase-2 grid-matrix.json.gz (tools/six_lane_grid.py --survivors) merged '
                                                       'with --matrix so interactions are judged against the phase-1 singles')
    p.add_argument('--out', type=Path)
    args = p.parse_args(argv)
    fast = args.mode == 'fast'
    if fast and args.identity:
        p.error('--mode fast takes no --identity: FAST needs no identical anything')
    matrix = load_json(args.matrix or (FAST_MATRIX if fast else MATRIX))
    if args.phase2_matrix:
        matrix = merge_matrices(matrix, load_json(args.phase2_matrix))
    args.out = args.out or (FAST_OUT_DIR if fast else OUT_DIR)
    dec = decide(matrix, timing_index(load_many(args.verdicts), voters=FAST_VOTERS if fast else None),
                 identity_index(load_many(args.identity)), quality_index(load_many(args.quality)), mode=args.mode)
    write_outputs(dec, args.out)
    print(json.dumps(dec['counts']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
