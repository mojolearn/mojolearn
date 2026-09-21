#!/usr/bin/env python3
"""Strict verdict for the two-dataset, four-arm Byte-LM flat-view trial."""
import argparse, json, math, statistics
from pathlib import Path

SHAPE = [1, 2048, 768, 12, 12, 64, 2048, 12, 50257]
DATASETS = {
    'taxi': ('gbm-bench/taxi/taxi_speed.npz', 419757252, '10d5d35f376a5b2ad5c66fabe624ee2caafb58bc5e7516824f801de9aab6cc15'),
    'istella': ('gbm-bench/istella/istella_speed.npz', 2248281826, '31f042376c0b819fe169cbbd998e840567c23dae25c14cb9054b460292f6ffef'),
}
ARMS = {'off': 0, 'param': 1, 'grad': 2, 'both': 3}
ORDERS = {0: ('off', 'param', 'grad', 'both'),
          1: ('both', 'grad', 'param', 'off'),
          2: ('param', 'off', 'both', 'grad')}
SCHEDULE = ('step k row b: dataset bytes at (k*2654435761+b*2246822519) modulo '
            '(size-length-1); uint8 to int32; target shifted one byte')


def load(path): return json.loads(path.read_text())


def check_record(rec, dataset, errors, label, sabotage=False):
    key, size, digest = DATASETS[dataset]
    counts = ({'witness_steps': 2, 'warmup': 0, 'samples': 0, 'total': 2} if sabotage else
              {'witness_steps': 3, 'warmup': 3, 'samples': 9, 'total': 15})
    meta = rec.get('run_metadata') or {}
    final = rec.get('final_witness') or {}
    vendor = {'nvidia': 'cuda', 'amd': 'hip'}.get(rec.get('target_column'))
    checks = (
        (rec.get('schema') == 'mojolearn.lm-flat-views.v1', 'schema'),
        (rec.get('dataset', {}).get('key') == key, 'dataset key'),
        (rec.get('dataset', {}).get('bytes') == size, 'dataset bytes'),
        (rec.get('dataset', {}).get('sha256') == digest, 'dataset sha256'),
        (rec.get('shape') == SHAPE, 'shape'), (rec.get('seed') == 20260921, 'seed'),
        (rec.get('schedule') == SCHEDULE, 'schedule'), (rec.get('counts') == counts, 'counts'),
        (rec.get('completed_steps') == counts['total'], 'completed steps'),
        (len(rec.get('witnesses', [])) == counts['total'], 'witness count'),
        (final.get('completed_steps') == counts['total'] and
         rec.get('final_witness') == (rec.get('witnesses') or [None])[-1], 'final witness'),
        (len(rec.get('seconds', [])) == counts['samples'], 'sample count'),
        (bool(rec.get('commit')), 'commit'),
        (len(rec.get('initial_parameters_sha256', '')) == 64, 'initial parameters'),
        (rec.get('binding', {}).get('bytes', 0) > 0 and len(rec.get('binding', {}).get('sha256', '')) == 64, 'binding'),
        (meta.get('native_vendor') == vendor, 'native vendor'),
        (meta.get('native_numeric_mode') == 1, 'numeric mode'),
        (meta.get('binding_sha256') == rec.get('binding', {}).get('sha256'), 'loaded binding'),
        (bool(meta.get('source_sha256')), 'source inventory'))
    for ok, name in checks:
        if not ok: errors.append('%s: invalid %s' % (label, name))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    runs = {d: {a: [load(p) for p in sorted((args.root/d/a).glob('process*/result.json'))]
                for a in ARMS} for d in DATASETS}
    sabotage = {d: {k: load(args.root/d/(k+'_sabotage')/'process0/result.json')
                    for k in ('param', 'grad')} for d in DATASETS}
    errors, rows, ratios, all_normal, all_records = [], [], {}, [], []
    for dataset in DATASETS:
        group = runs[dataset]
        for arm, word in ARMS.items():
            if len(group[arm]) != 3:
                errors.append('%s/%s has %d processes, expected 3' % (dataset, arm, len(group[arm])))
            for rec in group[arm]:
                label = '%s/%s/process%s' % (dataset, arm, rec.get('process'))
                check_record(rec, dataset, errors, label)
                process = rec.get('process')
                if process not in ORDERS or rec.get('position') != ORDERS[process].index(arm):
                    errors.append(label + ': launch order invalid')
                if rec.get('arm') != word: errors.append(label + ': arm witness invalid')
                witnesses = rec.get('witnesses') or []
                if len(witnesses) < 2 or rec.get('post_swap_witness') != witnesses[1]:
                    errors.append(label + ': post-swap witness invalid')
                all_normal.append(rec)
                all_records.append(rec)
        if not group['off']:
            errors.append(dataset + ': no baseline'); continue
        reference = (group['off'][0].get('witnesses'), group['off'][0].get('final_witness'))
        for arm in ARMS:
            for rec in group[arm]:
                if (rec.get('witnesses'), rec.get('final_witness')) != reference:
                    errors.append('%s/%s/process%s state differs' % (dataset, arm, rec.get('process')))
        base_witnesses = group['off'][0].get('witnesses') or []
        if not base_witnesses:
            errors.append(dataset + ': baseline has no first-step witness')
        base_first = base_witnesses[0] if base_witnesses else {}
        sabotage_fields = {
            'param': ('loss', 'gradients', 'parameters', 'm', 'v'),
            'grad': ('gradients', 'parameters', 'm', 'v'),
        }
        for kind, arm_word in (('param', 7), ('grad', 11)):
            rec = sabotage[dataset][kind]; label = dataset+'/'+kind+'_sabotage'
            check_record(rec, dataset, errors, label, True)
            all_records.append(rec)
            if rec.get('process') != 0 or rec.get('position') != 0:
                errors.append(label + ': process/position invalid')
            if rec.get('arm') != arm_word: errors.append(label + ': arm witness invalid')
            sab_witnesses = rec.get('witnesses') or []
            sab_first = sab_witnesses[0] if sab_witnesses else {}
            for field in sabotage_fields[kind]:
                if sab_first.get(field) == base_first.get(field):
                    errors.append('%s: independent reach did not change %s' % (label, field))
        timing_ok = all(len(group[a]) == 3 and all(
            isinstance(r.get('median_seconds'), (int, float)) and
            isinstance(r.get('min_seconds'), (int, float)) and r.get('min_seconds', 0) > 0 and
            isinstance(r.get('max_seconds'), (int, float))
            for r in group[a]) for a in ARMS)
        if not timing_ok:
            errors.append(dataset + ': timing records malformed')
            ratios[dataset] = {}
            continue
        baseline = statistics.median([r['median_seconds'] for r in group['off']])
        ratios[dataset] = {}
        for arm in ('param', 'grad', 'both'):
            meds = [r['median_seconds'] for r in group[arm]]
            spreads = [r['max_seconds']/r['min_seconds'] for r in group[arm]]
            ratio = statistics.median(meds)/baseline; ratios[dataset][arm] = ratio
            rows.append(dict(dataset=dataset, arm=arm, baseline_median=baseline,
                             candidate_median=statistics.median(meds), ratio=ratio,
                             process_medians=meds, within_process_spreads=spreads,
                             process_median_spread=max(meds)/min(meds),
                             stable_diagnostic=all(v <= 1.10 for v in spreads)))
        if ratios[dataset]['both'] >= 1:
            errors.append('%s/both has no positive median speedup: %.6f' % (dataset, ratios[dataset]['both']))
    if all_records:
        for label, getter in (('commit', lambda r:r.get('commit')),
                              ('target', lambda r:r.get('target_column')),
                              ('source', lambda r:(r.get('run_metadata') or {}).get('source_sha256')),
                              ('initial parameters', lambda r:r.get('initial_parameters_sha256'))):
            if len({json.dumps(getter(r), sort_keys=True) for r in all_records}) != 1:
                errors.append('records disagree on '+label)
    if all_normal:
        hashes = {}
        for arm in ARMS:
            hashes[arm] = {r.get('binding',{}).get('sha256') for d in DATASETS for r in runs[d][arm]}
            if len(hashes[arm]) != 1: errors.append(arm+': binding hashes differ')
        if len({next(iter(v), None) for v in hashes.values()}) != 4:
            errors.append('normal arms do not have four distinct bindings')
        all_binding_hashes = [r.get('binding',{}).get('sha256') for r in all_records]
        if len(set(all_binding_hashes)) != 6:
            errors.append('four normal and two sabotage builds do not have six distinct bindings')
    both_ratios = [ratios.get(d, {}).get('both') for d in DATASETS]
    geomean = (math.sqrt(both_ratios[0] * both_ratios[1])
               if all(isinstance(v, (int, float)) and math.isfinite(v) for v in both_ratios)
               else None)
    verdict = dict(schema='mojolearn.lm-flat-views-verdict.v1', status='PASS' if not errors else 'REJECT',
                   promotion_eligible=not errors, datasets=list(DATASETS), rows=rows,
                   both_geomean_ratio=geomean, errors=errors,
                   stability_policy='diagnostic only; no spread threshold rejects promotion',
                   policy='If NVIDIA and AMD each pass, enable both views for Apple without Apple timing.')
    args.out.write_text(json.dumps(verdict, indent=1, allow_nan=False)+'\n'); print(args.out.read_text(), end='')
    raise SystemExit(0 if not errors else 1)


if __name__ == '__main__': main()
