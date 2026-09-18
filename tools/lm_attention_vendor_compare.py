#!/usr/bin/env python3
"""Cross-column 700-step witness: a column's legacy/repaired arms against
the recorded NVIDIA arms (lane/attention-replay-vendors).

    python3 tools/lm_attention_vendor_compare.py <column dir> <nvidia dir> [label]

Each dir holds legacy/result.json and repaired/result.json. Every one of the
700 losses, the six state hashes at steps 0 and 699, and the per-step,
per-layer forward/backward statuses and backward replay sites must be equal
between the columns for the same arm, and the losses/hashes equal across
arms. Each comparison is first shown to REJECT a deliberately corrupted copy.
Prints the matches, not a count.
"""
import copy
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HASHES = ('loss', 'gradients', 'parameters', 'm', 'v', 'flags')
ROUTING = ('forward_status', 'backward_status', 'backward_repair_sites')


def load(root, arm):
    return json.loads((Path(root) / arm / 'result.json').read_text())


def compare(a, b, routing, verbose=False, tag=''):
    assert a['shape'] == b['shape'] and a['seed'] == b['seed'], 'inputs'
    assert a['corpus']['sha256'] == b['corpus']['sha256'], 'corpus'
    assert len(a['steps']) == len(b['steps']) == 700, 'coverage'
    for x, y in zip(a['steps'], b['steps']):
        assert x['step'] == y['step'], 'sequence'
        assert x['loss'] == y['loss'], 'loss'
        if routing:
            for key in ROUTING:
                assert x['attention'][key] == y['attention'][key], key
        if verbose:
            print('MATCH', tag, 'step', x['step'], 'loss', x['loss'],
                  'bwd', ''.join(map(str, y['attention']['backward_status'])) if routing else '',
                  'sites', ''.join(map(str, y['attention']['backward_repair_sites'])) if routing else '')
        if x['step'] in (0, 699):
            for key in HASHES:
                assert x[key] == y[key], key
                if verbose:
                    print('MATCH', tag, 'step', x['step'], key, x[key])


def must_fail(a, b, routing, reason):
    try:
        compare(a, b, routing)
    except AssertionError as e:
        assert str(e) == reason, ('unrelated failure', str(e), reason)
        print('EXPECTED FAIL:', reason)
    else:
        raise AssertionError('BLIND comparator: ' + reason)


def controls(a, b):
    for key in HASHES:
        broken = copy.deepcopy(b)
        broken['steps'][-1][key] = 'deliberately corrupted'
        must_fail(a, broken, True, key)
    broken = copy.deepcopy(b)
    broken['steps'][350]['loss'] = 'deliberately corrupted'
    must_fail(a, broken, True, 'loss')
    for key in ROUTING:
        broken = copy.deepcopy(b)
        row = broken['steps'][123]['attention'][key]
        row[0] = 7
        must_fail(a, broken, True, key)
    broken = copy.deepcopy(b)
    broken['steps'] = broken['steps'][:-1]
    must_fail(a, broken, True, 'coverage')


def summary(name, r):
    fwd = Counter(v for row in r['steps'] for v in row['attention']['forward_status'])
    bwd = Counter(v for row in r['steps'] for v in row['attention']['backward_status'])
    sites = Counter(v for row in r['steps'] for v in row['attention']['backward_repair_sites'])
    tail = statistics.median(x['seconds'] for x in r['steps'][-50:])
    head = statistics.median(x['seconds'] for x in r['steps'][1:51])
    mem = [x['device_used_mb'] for x in r['steps'][-50:] if x['device_used_mb'] is not None]
    print(name, 'forward_status', dict(sorted(fwd.items())), 'backward_status', dict(sorted(bwd.items())),
          'repair_sites', dict(sorted(sites.items())))
    print(name, 'head_median_s', head, 'tail_median_s', tail, 'tail_device_mib', mem,
          'eager_bytes', r['steps'][-1]['attention']['eager_bytes'],
          'aexp_bytes', r['steps'][-1]['attention']['forward_aexp_bytes'])
    return tail


def main():
    col, nv = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else 'column'
    tails = {}
    for arm in ('legacy', 'repaired'):
        a, b = load(nv, arm), load(col, arm)
        controls(a, b)
        compare(a, b, True, True, f'nvidia/{label} {arm}')
        tails[arm] = summary(f'{label} {arm}', b)
        summary(f'nvidia {arm}', a)
    # Across arms on this column: bits equal, routing intentionally different.
    compare(load(col, 'legacy'), load(col, 'repaired'), False, True, f'{label} legacy/repaired')
    print(label, 'late-step speedup legacy/repaired', tails['legacy'] / tails['repaired'])


if __name__ == '__main__':
    main()
