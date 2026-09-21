#!/usr/bin/env python3
"""Verdict for the two-dataset, four-arm Byte-LM flat-view trial."""
import argparse
import json
import math
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    names = ('off', 'param', 'grad', 'both')
    datasets = ('taxi', 'istella')
    runs = {d: {a: [load(path) for path in sorted((args.root / d / a).glob('process*/result.json'))]
                for a in names} for d in datasets}
    sabotages = {d: load(args.root / d / 'sabotage' / 'result.json') for d in datasets}
    expected = dict(off=0, param=1, grad=2, both=3)
    errors, rows = [], []
    ratios = {}
    for dataset in datasets:
        group = runs[dataset]
        for arm, word in expected.items():
            if len(group[arm]) < 3 or len(group[arm]) % 2 == 0:
                errors.append('%s/%s has %d processes, expected an odd count >=3' %
                              (dataset, arm, len(group[arm])))
            for process, rec in enumerate(group[arm]):
                if rec['arm'] != word:
                    errors.append('%s/%s/process%d witness arm=%r expected=%d' %
                                  (dataset, arm, process, rec['arm'], word))
        if not group['off']:
            errors.append(dataset + ': no baseline processes')
            continue
        reference = group['off'][0]['witnesses']
        for arm in names:
            for process, rec in enumerate(group[arm]):
                if rec['witnesses'] != reference:
                    errors.append('%s/%s/process%d full witnesses differ' %
                                  (dataset, arm, process))
        if sabotages[dataset]['arm'] != 7:
            errors.append(dataset + ': sabotage arm witness is not 7')
        if sabotages[dataset]['witnesses'] == reference[:len(sabotages[dataset]['witnesses'])]:
            errors.append(dataset + ': wrong-offset sabotage did not diverge')
        baseline_medians = [rec['median_seconds'] for rec in group['off']]
        baseline = statistics.median(baseline_medians)
        ratios[dataset] = {}
        for arm in ('param', 'grad', 'both'):
            medians = [rec['median_seconds'] for rec in group[arm]]
            candidate = statistics.median(medians)
            process_spreads = [rec['max_seconds'] / rec['min_seconds']
                               for rec in group[arm]]
            ratio = candidate / baseline
            ratios[dataset][arm] = ratio
            rows.append(dict(dataset=dataset, arm=arm, baseline_median=baseline,
                             candidate_median=candidate, ratio=ratio,
                             process_medians=medians,
                             within_process_spreads=process_spreads,
                             process_median_spread=max(medians) / min(medians),
                             stable_diagnostic=all(x <= 1.10 for x in process_spreads)))
        if ratios[dataset]['both'] >= 1.0:
            errors.append('%s/both has no positive median speedup: %.6f' %
                          (dataset, ratios[dataset]['both']))
    geomean = math.sqrt(ratios['taxi']['both'] * ratios['istella']['both'])
    verdict = dict(schema='mojolearn.lm-flat-views-verdict.v1',
                   status='PASS' if not errors else 'REJECT',
                   promotion_eligible=not errors, datasets=list(datasets), rows=rows,
                   both_geomean_ratio=geomean, errors=errors,
                   stability_policy='diagnostic only; no spread threshold rejects promotion',
                   policy=('If NVIDIA and AMD each pass this exact gate, enable both views for Apple too without a separate Apple timing requirement.'))
    args.out.write_text(json.dumps(verdict, indent=1, allow_nan=False) + '\n')
    print(args.out.read_text(), end='')
    raise SystemExit(0 if not errors else 1)


if __name__ == '__main__':
    main()
