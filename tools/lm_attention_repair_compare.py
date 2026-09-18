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
    assert a['shape'] == b['shape'] == [1,2048,768,12,12,64,2048,12,50257] and a['seed'] == b['seed'], 'inputs'
    assert a['corpus']['sha256'] == b['corpus']['sha256'], 'corpus'
    assert len(a['steps']) == len(b['steps']) == 700, 'coverage'
    for x, y in zip(a['steps'], b['steps']):
        assert x['step'] == y['step'], 'sequence'
        # Probe replaces numeric loss with its hash on exported steps.
        assert x['loss'] == y['loss'], 'loss'
        assert x['attention']['repair_masked_tail'] is False, 'legacy arm'
        assert y['attention']['repair_masked_tail'] is True, 'repaired arm'
        if verbose:
            print('MATCH step', x['step'], 'loss', x['loss'])
        if x['step'] in (0, 699):
            for key in HASHES:
                assert x[key] == y[key], key
                if verbose:
                    print('MATCH step', x['step'], key, x[key])


def default_capacity(r):
    for row in r['steps']:
        a = row['attention']
        assert a['release_eager'] is True and a['sticky_eager'] is False, 'default storage policy'
        assert a['eager_bytes'] == 432, 'eager capacity'
        assert a['forward_aexp_bytes'] == 2415919104, 'aexp capacity'
        assert len(a['backward_repair_sites']) == 12 and all(v in (0, 1, 2, 3) for v in a['backward_repair_sites']), 'repair coverage'


def main():
    root = Path(sys.argv[1])
    a = json.loads((root / 'legacy/result.json').read_text())
    b = json.loads((root / 'repaired/result.json').read_text())
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
    for name, left, right, reason in (('same legacy arm', a, a, 'repaired arm'),
                                     ('same repaired arm', b, b, 'legacy arm')):
        try:
            compare(left, right, False)
        except AssertionError as e:
            assert str(e) == reason, str(e)
            print('EXPECTED FAIL:', name)
        else:
            raise AssertionError('BLIND: ' + name)
    compare(a, b)
    if '--default' in sys.argv:
        for label, key, value in (('default storage policy', 'release_eager', False),
                                  ('eager capacity', 'eager_bytes', 999),
                                  ('aexp capacity', 'forward_aexp_bytes', 999),
                                  ('repair coverage', 'backward_repair_sites', [])):
            broken = copy.deepcopy(b)
            broken['steps'][-1]['attention'][key] = value
            try:
                default_capacity(broken)
            except AssertionError as e:
                assert str(e) == label, str(e)
                print('EXPECTED FAIL:', label)
            else:
                raise AssertionError('BLIND: ' + label)
        default_capacity(b)
    for name, r in (('legacy', a), ('repaired', b)):
        repairs = Counter(v for row in r['steps'] for v in row['attention']['backward_repair_sites'])
        print(name, 'backward_repair_sites', dict(sorted(repairs.items())))
        if name == 'repaired' and not any(v for v in repairs):
            print('INERT: no masked-tail repair executed')
        counts = Counter(v for row in r['steps'] for v in row['attention']['backward_status'])
        print(name, 'backward_status', dict(sorted(counts.items())))
        for window, rows in (('head', r['steps'][1:51]), ('tail', r['steps'][-50:])):
            print(name, window, 'median_seconds', statistics.median(x['seconds'] for x in rows),
                  'device_mib', [x['device_used_mb'] for x in rows if x['device_used_mb'] is not None],
                  'eager_bytes', rows[-1]['attention']['eager_bytes'],
                  'aexp_bytes', rows[-1]['attention']['forward_aexp_bytes'])


if __name__ == '__main__':
    main()
