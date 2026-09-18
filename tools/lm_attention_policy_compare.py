#!/usr/bin/env python3
"""Three-arm policy/lifetime comparison with deliberately failing controls."""
import copy
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from lm_attention_tail_compare import HASHES


def compare(a, b, policy, release, verbose=True):
    assert a['shape'] == b['shape'] == [1,2048,768,12,12,64,2048,12,50257], 'shape'
    assert a['seed'] == b['seed'] and a['corpus']['sha256'] == b['corpus']['sha256'], 'inputs'
    assert len(a['steps']) == len(b['steps']) == 700, 'coverage'
    for i, (x, y) in enumerate(zip(a['steps'], b['steps'])):
        assert x['step'] == y['step'] == i, 'sequence'
        assert x['loss'] == y['loss'], 'loss'
        ax, ay = x['attention'], y['attention']
        assert ax['sticky_eager'] is False and ax['release_eager'] is False, 'baseline arm'
        assert ay['sticky_eager'] is policy and ay['release_eager'] is release, 'candidate arm'
        assert ax['exact_tail_guard'] is ay['exact_tail_guard'] is False, 'tail guard disabled'
        if release:
            assert ay['eager_bytes'] == 432, 'retained eager bytes'
            assert ay['forward_aexp_bytes'] == 2415919104, 'aexp bytes'
        if verbose:
            print('MATCH step', i, 'loss', x['loss'])
        if i in (0, 699):
            for key in HASHES:
                assert x[key] == y[key], key
                if verbose:
                    print('MATCH step', i, key, x[key])


def routing(result):
    previous = [False] * 12
    for row in result['steps']:
        a = row['attention']
        assert len(a['layers_prefer_eager']) == len(a['forward_status']) == len(a['backward_status']) == 12, 'routing layer coverage'
        for layer, preferred in enumerate(a['layers_prefer_eager']):
            f, b = a['forward_status'][layer], a['backward_status'][layer]
            if previous[layer]:
                assert preferred and f == b == -1, 'pre-launch routing'
            elif preferred:
                assert f == 2 or b == 2, 'observed corner trigger'
        previous = a['layers_prefer_eager']
    assert all(previous), 'all layers exercised'


def main():
    root = Path(sys.argv[1])
    a = json.loads((root / 'baseline/result.json').read_text())
    for arm, release in (('sticky', False), ('released', True)):
        b = json.loads((root / arm / 'result.json').read_text())
        for key in HASHES:
            broken = copy.deepcopy(b)
            broken['steps'][-1][key] = 'deliberate corruption'
            try:
                compare(a, broken, True, release, False)
            except AssertionError as e:
                assert str(e) == key, str(e)
                print('EXPECTED FAIL', arm, key)
            else:
                raise AssertionError('BLIND ' + key)
        for label, left, right, reason in (('same baseline', a, a, 'candidate arm'),
                                           ('same candidate', b, b, 'baseline arm')):
            try:
                compare(left, right, True, release, False)
            except AssertionError as e:
                assert str(e) == reason, str(e)
                print('EXPECTED FAIL', arm, label)
            else:
                raise AssertionError('BLIND arm witness')
        print('ARM', arm)
        compare(a, b, True, release)
        if release:
            for field, reason in (('eager_bytes','retained eager bytes'),('forward_aexp_bytes','aexp bytes')):
                broken = copy.deepcopy(b)
                broken['steps'][-1]['attention'][field] = -1
                try:
                    compare(a, broken, True, release, False)
                except AssertionError as e:
                    assert str(e) == reason, str(e)
                    print('EXPECTED FAIL', field)
                else:
                    raise AssertionError('BLIND ' + field)
            assert any(row['attention']['released_eager_bytes'] > 0 for row in b['steps']), 'INERT release'
        routing(b)
        for reason in ('pre-launch routing', 'observed corner trigger', 'routing layer coverage'):
            broken = copy.deepcopy(b)
            if reason == 'pre-launch routing':
                broken['steps'][-1]['attention']['forward_status'][0] = 0
            elif reason == 'observed corner trigger':
                broken['steps'][0]['attention']['layers_prefer_eager'][0] = True
            else:
                broken['steps'][0]['attention']['layers_prefer_eager'].pop()
            try:
                routing(broken)
            except AssertionError as e:
                assert str(e) == reason, str(e)
                print('EXPECTED FAIL', reason)
            else:
                raise AssertionError('BLIND routing checker')
    for arm in ('baseline','sticky','released'):
        r = json.loads((root / arm / 'result.json').read_text())
        rows = r['steps']
        print(arm, 'forward', dict(Counter(v for x in rows for v in x['attention']['forward_status'])),
              'backward', dict(Counter(v for x in rows for v in x['attention']['backward_status'])))
        print(arm, 'head median', statistics.median(x['seconds'] for x in rows[1:51]),
              'tail median', statistics.median(x['seconds'] for x in rows[-50:]),
              'tail MiB', [x['device_used_mb'] for x in rows[-50:] if x['device_used_mb'] is not None],
              'last attention', rows[-1]['attention'])


if __name__ == '__main__':
    main()
