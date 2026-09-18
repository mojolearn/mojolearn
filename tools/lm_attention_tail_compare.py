#!/usr/bin/env python3
"""Print exact witnesses; prove each numerical comparison rejects corruption."""
import copy
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HASHES = ('loss', 'gradients', 'parameters', 'm', 'v', 'flags')


def compare(a, b, verbose=True):
    assert a['shape'] == b['shape'] and a['seed'] == b['seed'], 'inputs'
    assert a['corpus']['sha256'] == b['corpus']['sha256'], 'corpus'
    assert len(a['steps']) == len(b['steps']) == 700, 'coverage'
    for x, y in zip(a['steps'], b['steps']):
        assert x['step'] == y['step'], 'sequence'
        # Probe replaces numeric loss with its hash on exported steps.
        assert x['loss'] == y['loss'], 'loss'
        assert x['attention']['exact_tail_guard'] is False, 'legacy arm'
        assert y['attention']['exact_tail_guard'] is True, 'guarded arm'
        if verbose:
            print('MATCH step', x['step'], 'loss', x['loss'])
        if x['step'] in (0, 699):
            for key in HASHES:
                assert x[key] == y[key], key
                if verbose:
                    print('MATCH step', x['step'], key, x[key])


def main():
    root = Path(sys.argv[1])
    a = json.loads((root / 'legacy/result.json').read_text())
    b = json.loads((root / 'guarded/result.json').read_text())
    for key in HASHES:
        broken = copy.deepcopy(b)
        broken['steps'][-1][key] = 'deliberately corrupted'
        try:
            compare(a, broken, False)
        except AssertionError as e:
            assert str(e) == key, ('unrelated failure', str(e))
            print('EXPECTED FAIL:', key)
        else:
            raise AssertionError('BLIND comparator: ' + key)
    for name, left, right, reason in (('same legacy arm', a, a, 'guarded arm'),
                                     ('same guarded arm', b, b, 'legacy arm')):
        try:
            compare(left, right, False)
        except AssertionError as e:
            assert str(e) == reason, str(e)
            print('EXPECTED FAIL:', name)
        else:
            raise AssertionError('BLIND: ' + name)
    compare(a, b)
    for name, r in (('legacy', a), ('guarded', b)):
        counts = Counter(v for row in r['steps'] for v in row['attention']['backward_status'])
        print(name, 'backward_status', dict(sorted(counts.items())))
        for window, rows in (('head', r['steps'][1:51]), ('tail', r['steps'][-50:])):
            print(name, window, 'median_seconds', statistics.median(x['seconds'] for x in rows),
                  'device_mib', [x['device_used_mb'] for x in rows if x['device_used_mb'] is not None],
                  'eager_bytes', rows[-1]['attention']['eager_bytes'],
                  'aexp_bytes', rows[-1]['attention']['forward_aexp_bytes'])


if __name__ == '__main__':
    main()
