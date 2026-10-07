#!/usr/bin/env python3
"""Scored timing, A/A noise floors and A/B timing verdicts for six-lane receipts.

Metadata only: reads retained receipts, never imports an estimator, compiles or
measures. One run per arm (owner rule); the A/A pair is one pair per workload
per box, and its |log(A/B)| is that workload's noise floor on that box.

Scored clock (owner, 2026-10-07): fit/training plus the separate
transform/predict call, each reported separately. Load, preparation, runner
construction, output capture and hashing are outside it. Receipts written
before the worker recorded scored_* fields fall back to fit_or_training_seconds
plus inference_seconds (which still include inference-runner setup and capture
for the classical family); the fallback is labeled in every output.

Verdict rule: a phase gets FASTER/SLOWER only when, on every voting vendor,
|log(A/B)| exceeds that vendor's A/A floor for the same workload AND all voting
vendors agree in direction. Otherwise NO_VERDICT with the reason.
IDENTICAL votes: NVIDIA and AMD. FAST votes: Apple.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

PHASES = ('scored', 'fit', 'inference')
VOTERS = {'identical': ('nvidia', 'amd'), 'fast': ('apple',)}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def scored(timings):
    """{scored, fit, inference, source} seconds from one result's timings."""
    timings = timings or {}
    if _finite(timings.get('scored_seconds')):
        return dict(scored=timings['scored_seconds'], fit=timings.get('scored_fit_seconds'),
                    inference=timings.get('scored_inference_seconds'), source='scored_seconds')
    fit = timings.get('fit_or_training_seconds')
    if not _finite(fit):
        return None
    inference = timings.get('inference_seconds')
    inference = inference if _finite(inference) else 0.0
    return dict(scored=fit + inference, fit=fit, inference=inference,
                source='legacy fit_or_training_seconds + inference_seconds (includes inference setup/capture)')


def receipts(paths):
    """Yield (path, receipt) from receipt files, results.json maps or directories."""
    for raw in paths:
        path = Path(raw)
        files = sorted(path.glob('**/receipt*.json')) if path.is_dir() else [path]
        for file in files:
            data = json.loads(file.read_text())
            if isinstance(data, dict) and 'runs' in data:
                yield str(file), data
            elif isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, dict) and 'runs' in value:
                        yield str(file), value


def pair(receipt):
    """Scored A and B results of one complete receipt, else None."""
    if receipt.get('status') != 'MEASURED_FULL':
        return None
    runs = {r['arm']: r.get('result') for r in receipt.get('runs', [])
            if r.get('phase') == 'scored' and r.get('returncode') == 0 and isinstance(r.get('result'), dict)}
    if set(runs) != {'A', 'B'}:
        return None
    return runs


def log_ratios(a, b):
    """log(A/B) per phase; None where a phase is absent or zero in either arm."""
    sa, sb = scored(a.get('timings')), scored(b.get('timings'))
    if not sa or not sb:
        return None
    out = {}
    for phase in PHASES:
        x, y = sa.get(phase), sb.get(phase)
        out[phase] = math.log(x / y) if _finite(x) and _finite(y) and x > 0 and y > 0 else None
    out['source'] = sa['source'] if sa['source'] == sb['source'] else 'mixed: ' + sa['source'] + ' / ' + sb['source']
    return out


def vendor_of(receipt, runs):
    return runs['A'].get('vendor') or receipt.get('workload', {}).get('vendor')


def floors(paths):
    """A/A noise floors keyed 'workload_id|vendor'. Rejects pairs that are not byte-identical."""
    out, rejected = {}, []
    for path, receipt in receipts(paths):
        runs = pair(receipt)
        job = receipt.get('workload', {})
        if runs is None:
            rejected.append(dict(evidence=path, reason='incomplete scored pair'))
            continue
        if not job.get('aa_noise_floor'):
            rejected.append(dict(evidence=path, reason='not an A/A job'))
            continue
        if runs['A'].get('loaded_artifacts') != runs['B'].get('loaded_artifacts') or not runs['A'].get('loaded_artifacts'):
            rejected.append(dict(evidence=path, reason='A and B did not load byte-identical artifacts'))
            continue
        ratios = log_ratios(runs['A'], runs['B'])
        if ratios is None:
            rejected.append(dict(evidence=path, reason='missing scored timings'))
            continue
        key = job['workload_id'] + '|' + vendor_of(receipt, runs)
        if key in out:
            rejected.append(dict(evidence=path, reason='second A/A pair for ' + key + '; one pair per workload per box'))
            continue
        out[key] = dict(workload_id=job['workload_id'], vendor=vendor_of(receipt, runs), evidence=path,
                        source_sha=receipt.get('source_sha'), timing_source=ratios['source'],
                        floor={p: abs(ratios[p]) if ratios[p] is not None else None for p in PHASES})
    return dict(schema='mojolearn.six-lane-aa-floors/1', floors=out, rejected=rejected,
                policy='One A/A pair (byte-identical builds) per workload per box; floor = |log(A/B)| of the scored clock, per phase.')


def _spread(values):
    """log(high/low) of a sample of the same build: p90/p10 with 4+ samples, max/min below that."""
    vals = sorted(v for v in values if _finite(v) and v > 0)
    if len(vals) < 2:
        return None
    if len(vals) >= 4:
        lo, hi = vals[int(0.1 * (len(vals) - 1))], vals[int(round(0.9 * (len(vals) - 1)))]
    else:
        lo, hi = vals[0], vals[-1]
    return math.log(hi / lo) if lo > 0 else None


def floors_from_pairs(paths, min_samples=2):
    """Noise floors from the incumbent arm (B) of ordinary A/B receipts: B is the same build in every
    cell of a workload, so the grid repeats it for free (Andrew, 2026-10-07: no separate A/A pass).
    Samples are grouped per workload, vendor and incumbent source sha; floor = log spread of the scored
    clock per phase. Fewer than `min_samples` repeats leaves the key without a floor (listed)."""
    samples, meta = {}, {}
    for path, receipt in receipts(paths):
        runs = pair(receipt)
        job = receipt.get('workload', {})
        if runs is None or job.get('aa_noise_floor'):
            continue
        b = runs['B']
        sb = scored(b.get('timings'))
        if not sb:
            continue
        key = job['workload_id'] + '|' + vendor_of(receipt, runs)
        # B is main at the receipt's source sha; its .so may come from a different pack per cell, so group by sha.
        digest = receipt.get('source_sha') or json.dumps(b.get('loaded_artifacts'), sort_keys=True)
        bucket = samples.setdefault(key, {}).setdefault(digest, {p: [] for p in PHASES})
        for p in PHASES:
            if _finite(sb.get(p)):
                bucket[p].append(sb[p])
        meta.setdefault(key, dict(workload_id=job['workload_id'], vendor=vendor_of(receipt, runs), evidence=[],
                                  timing_source=sb['source'], source_sha=receipt.get('source_sha')))['evidence'].append(path)
    out, thin = {}, []
    for key, by_digest in samples.items():
        digest, bucket = max(by_digest.items(), key=lambda kv: len(kv[1]['scored']))  # the build with the most repeats
        n = len(bucket['scored'])
        if n < min_samples:
            thin.append(dict(key=key, samples=n, reason='fewer than %d incumbent repeats' % min_samples))
            continue
        out[key] = dict(meta[key], samples=n, builds_seen=len(by_digest), b_artifacts=digest,
                        floor={p: _spread(bucket[p]) for p in PHASES})
    return dict(schema='mojolearn.six-lane-aa-floors/1', floors=out, rejected=thin,
                policy='Floor from the incumbent arm B repeated across the A/B cells of one workload on one box '
                       '(same build): log(p90/p10) of the scored clock with 4+ samples, log(max/min) below that. '
                       'No separate A/A pass is needed; A/A receipts, if present, are ignored here.')


def judge(per_vendor, voters, phase):
    """per_vendor: {vendor: {'log_ratio': {...}, 'floor': {...} | None}}."""
    reasons, signs = [], []
    for vendor in voters:
        row = per_vendor.get(vendor)
        if row is None:
            reasons.append(vendor + ': no A/B pair')
            continue
        r = row['log_ratio'].get(phase)
        f = (row.get('floor') or {}).get(phase)
        if r is None:
            reasons.append(vendor + ': phase not measured')
        elif f is None:
            reasons.append(vendor + ': no A/A floor')
        elif abs(r) <= f:
            reasons.append(vendor + ': |log ratio| %.4f within floor %.4f' % (abs(r), f))
        else:
            signs.append(1 if r > 0 else -1)
    if reasons:
        return dict(verdict='NO_VERDICT', reasons=reasons)
    if len(set(signs)) != 1:
        return dict(verdict='NO_VERDICT', reasons=['voting vendors disagree in direction'])
    return dict(verdict='SLOWER' if signs[0] > 0 else 'FASTER', reasons=[])


def verdicts(ab_paths, floor_doc):
    """Group A/B receipts by (configuration, workload); judge each phase."""
    table = floor_doc.get('floors', {})
    cases = {}
    for path, receipt in receipts(ab_paths):
        runs = pair(receipt)
        if runs is None:
            continue
        job = receipt.get('workload', {})
        if job.get('aa_noise_floor'):
            continue
        config = (job.get('master_selection') or {}).get('id') or job.get('key')
        vendor = vendor_of(receipt, runs)
        ratios = log_ratios(runs['A'], runs['B'])
        if ratios is None:
            continue
        case = cases.setdefault(config + '|' + job['workload_id'],
                                dict(configuration=config, workload_id=job['workload_id'], mode=receipt.get('mode'), vendors={}))
        floor = table.get(job['workload_id'] + '|' + vendor)
        case['vendors'][vendor] = dict(evidence=path, log_ratio={p: ratios[p] for p in PHASES},
                                       candidate_over_baseline={p: math.exp(ratios[p]) if ratios[p] is not None else None for p in PHASES},
                                       timing_source=ratios['source'], floor=floor['floor'] if floor else None,
                                       floor_evidence=floor['evidence'] if floor else None)
    for case in cases.values():
        voters = VOTERS.get(case['mode'], VOTERS['identical'])
        case['voters'] = list(voters)
        case['phases'] = {p: judge(case['vendors'], voters, p) for p in PHASES}
        case['verdict'] = case['phases']['scored']['verdict']
    return dict(schema='mojolearn.six-lane-timing-verdicts/1', cases=sorted(cases.values(), key=lambda c: (c['configuration'], c['workload_id'])),
                rule='Verdict only when |log(A/B)| exceeds the workload A/A floor on every voting vendor and all agree in direction. A = candidate, B = incumbent; FASTER means A faster.',
                accepted=False, promoted=False)


def aa_job(job, worker):
    """A/A copy of an admitted queue job and its worker recipe: both arms run B's build.

    A and B load the same incumbent package and artifacts, so the builds are
    byte-identical. The queue key changes; the source freeze, dataset and workload
    recipe do not. Execution authorization is not inherited.
    """
    job = copy.deepcopy(job)
    worker = copy.deepcopy(worker)
    key = 'aa-' + hashlib.sha256((job['workload_id'] + '|' + worker['vendor']).encode()).hexdigest()[:17]
    marker = dict(source_key=job['key'], workload_id=job['workload_id'], incumbent_configuration=job['master_selection']['B'])
    # The queue job takes a new key (its own result directory). The worker's
    # job keeps the admitted key: registered-variant validators bind it.
    job['key'] = key
    for target in (job, worker['job']):
        target['aa_noise_floor'] = marker
        target['master_selection'] = dict(target['master_selection'], A=copy.deepcopy(target['master_selection']['B']))
        target['artifact_provenance'] = dict(A=copy.deepcopy(target['artifact_provenance']['B']), B=target['artifact_provenance']['B'])
        target['arms']['A'] = dict(copy.deepcopy(target['arms']['B']), argv=list(target['arms']['A'].get('argv', [])))
    worker['packages'] = dict(A=worker['packages']['B'], B=worker['packages']['B'])
    worker['artifact_provenance'] = job['artifact_provenance']
    worker['execution_authorized'] = False
    return job, worker


def aa_queue(queue_doc, worker_dir, out_path):
    """One A/A job per workload on this box, from an existing admitted queue."""
    jobs, seen, skipped = [], set(), []
    out_path = Path(out_path)
    out_workers = out_path.parent / (out_path.stem + '-workers')
    for job in queue_doc['jobs']:
        if job.get('blocked') or not job['arms']['A'].get('argv'):
            skipped.append(dict(key=job['key'], reason='blocked or no admitted recipe'))
            continue
        if job['workload_id'] in seen:
            continue
        worker_path = Path(worker_dir) / (job['key'] + '.json')
        worker = json.loads(worker_path.read_text())
        new_job, new_worker = aa_job(job, worker)
        new_path = out_workers / (new_job['key'] + '.json')
        for arm in ('A', 'B'):
            argv = list(job['arms'][arm]['argv'])
            argv[argv.index('--recipe') + 1] = str(new_path.resolve())
            new_job['arms'][arm]['argv'] = argv
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(json.dumps(new_worker, indent=2, sort_keys=True) + '\n')
        jobs.append(new_job)
        seen.add(job['workload_id'])
    doc = dict(queue_doc, jobs=jobs, execution_authorized=False, aa_noise_floor=True, aa_skipped=skipped)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='command', required=True)
    q = s.add_parser('aa-queue', help='write an A/A queue (one pair per workload) from an admitted queue')
    q.add_argument('--queue', type=Path, required=True)
    q.add_argument('--workers', type=Path, help='worker recipe directory (default: <queue stem>-workers)')
    q.add_argument('--output', type=Path, required=True)
    f = s.add_parser('floors', help='A/A receipts -> noise floors (or, with --from-pairs, A/B receipts: incumbent-arm repeats)')
    f.add_argument('receipts', nargs='+')
    f.add_argument('--out', type=Path, required=True)
    f.add_argument('--from-pairs', action='store_true',
                   help='derive the floor from arm B repeated across the A/B cells of each workload; no A/A pass needed')
    f.add_argument('--min-samples', type=int, default=2, help='--from-pairs: fewest incumbent repeats that give a floor')
    v = s.add_parser('verdicts', help='A/B receipts + floors -> timing verdicts')
    v.add_argument('receipts', nargs='+')
    v.add_argument('--floors', type=Path, required=True)
    v.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'aa-queue':
        workers = args.workers or args.queue.parent / (args.queue.stem + '-workers')
        doc = aa_queue(json.loads(args.queue.read_text()), workers, args.output)
        print(json.dumps(dict(aa_jobs=len(doc['jobs']), skipped=len(doc['aa_skipped']), output=str(args.output),
                              execution='NOT RUN; later authorization must be recorded in queue and workers')))
        return 0
    if args.command == 'floors':
        doc = floors_from_pairs(args.receipts, args.min_samples) if args.from_pairs else floors(args.receipts)
        print(json.dumps(dict(floors=len(doc['floors']), rejected=len(doc['rejected']), output=str(args.out))))
    else:
        doc = verdicts(args.receipts, json.loads(args.floors.read_text()))
        counts = {}
        for case in doc['cases']:
            counts[case['verdict']] = counts.get(case['verdict'], 0) + 1
        print(json.dumps(dict(cases=len(doc['cases']), verdicts=counts, output=str(args.out))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
