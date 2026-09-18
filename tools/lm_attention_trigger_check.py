#!/usr/bin/env python3
"""Validate per-step status coverage; print counts and deliberately break checks."""
import copy
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def validate(result, minimum):
    rows = result['steps']
    assert len(rows) >= minimum, 'step coverage'
    n = result['shape'][7]
    for step, row in enumerate(rows):
        assert row['step'] == step, 'step sequence'
        a = row['attention']
        for key in ('forward_status', 'backward_status'):
            values = a[key]
            assert len(values) == n, 'layer coverage'
            assert all(v in (-1, 0, 1, 2, 3) for v in values), 'unknown status'
            names = {-1: 'NOT_ATTEMPTED', 0: 'FUSED_RAN', 1: 'FUSED_REFUSED_REGIME', 2: 'FUSED_CORNER'}
            if 3 in values or 'FUSED_SKIPPED_STICKY' in a[key + '_counts']:
                names[3] = 'FUSED_SKIPPED_STICKY'
            assert a[key + '_counts'] == {name: values.count(code) for code, name in names.items()}, 'status counts'
        assert len(a['attn_materialized']) == n, 'materialization coverage'
    return rows


def main():
    root = Path(sys.argv[1])
    r = json.loads((root / 'baseline/result.json').read_text())
    # Watch the same validator reject deliberate corruption before accepting it.
    for name, mutate in (
        ('step coverage', lambda x: x['steps'].__delitem__(slice(699, None))),
        ('layer coverage', lambda x: x['steps'][0]['attention']['forward_status'].pop()),
        ('unknown status', lambda x: x['steps'][0]['attention']['forward_status'].__setitem__(0, 99)),
        ('status counts', lambda x: x['steps'][0]['attention']['forward_status_counts'].__setitem__('FUSED_RAN', 999)),
        ('materialization coverage', lambda x: x['steps'][0]['attention']['attn_materialized'].pop()),
        ('step sequence', lambda x: x['steps'][0].__setitem__('step', 9)),
    ):
        broken = copy.deepcopy(r)
        mutate(broken)
        try:
            validate(broken, 700)
        except AssertionError as e:
            assert str(e) == name, (name, str(e))
            print('EXPECTED FAIL:', name)
        else:
            raise AssertionError('BLIND: ' + name)
    rows = validate(r, 700)
    for arm, expected in (('forced-eager', -1), ('forced-fused', 0)):
        control = json.loads((root / arm / 'result.json').read_text())
        row = validate(control, 1)[0]
        for key in ('forward_status', 'backward_status'):
            values = row['attention'][key]
            assert values == [expected] * control['shape'][7], (arm, key, values)
            print('MATCH:', arm, key, values)
    for key in ('forward_status', 'backward_status'):
        counts = Counter(v for row in rows for v in row['attention'][key])
        print(key, dict(sorted(counts.items())))
        for layer in range(r['shape'][7]):
            counts = Counter(row['attention'][key][layer] for row in rows)
            first = next((row['step'] for row in rows if row['attention'][key][layer] > 0), None)
            print(key, 'layer', layer, 'counts', dict(sorted(counts.items())), 'first_refusal', first)
    for label, window in (('head', rows[1:51]), ('tail', rows[-50:])):
        print(label, 'median_seconds', statistics.median(x['seconds'] for x in window),
              'device_used_mb', [x['device_used_mb'] for x in window if x['device_used_mb'] is not None],
              'eager_bytes', window[-1]['attention']['eager_bytes'],
              'aexp_bytes', window[-1]['attention']['forward_aexp_bytes'])


if __name__ == '__main__':
    main()
