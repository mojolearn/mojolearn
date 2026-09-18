#!/usr/bin/env python3
"""Strict 700-step GEMM operand-flush A/B, with executable negative controls."""
import copy
import json
import math
import statistics
import struct
import sys
from pathlib import Path

SHAPE = [1, 2048, 768, 12, 12, 64, 2048, 12, 50257]
HASHES = ('loss', 'gradients', 'parameters', 'm', 'v', 'flags')


def compare(a, b, verbose=True):
    assert a['shape'] == b['shape'] == SHAPE, 'shape'
    assert a['seed'] == b['seed'] == 20260917, 'seed'
    assert a['corpus'] == b['corpus'] and a['corpus']['sha256'], 'corpus'
    assert a['gemm_stage_ftz'] is False and b['gemm_stage_ftz'] is True, 'dispatch'
    assert a['run_metadata']['binding_sha256'] != b['run_metadata']['binding_sha256'], 'binding'
    assert len(a['steps']) == len(b['steps']) == 700, 'coverage'
    for index, (x, y) in enumerate(zip(a['steps'], b['steps'])):
        assert x['step'] == y['step'] == index, 'sequence'
        assert x['completed_steps'] == y['completed_steps'] == index+1, 'completed'
        if isinstance(x['loss'], str) or isinstance(y['loss'], str):
            assert x['loss'] == y['loss'], 'loss'
        else:
            assert struct.pack('<f', x['loss']) == struct.pack('<f', y['loss']), 'loss'
        assert x['attention'] == y['attention'], 'attention'
        assert x['seconds'] > 0 and y['seconds'] > 0, 'timing'
        if verbose:
            print('MATCH step', index, 'loss', x['loss'], y['loss'])
        if index in (0, 699):
            for key in HASHES:
                assert isinstance(x[key], str) and len(x[key]) == 64 and x[key] == y[key], key
                if verbose:
                    print('MATCH step', index, key, x[key], y[key])
        else:
            assert isinstance(x['loss'], (int, float)) and math.isfinite(x['loss']), 'finite'


def verify(a, b):
    # Each intentional defect must fail for its own reason, not an unrelated one.
    mutations = [(key, lambda d, k=key: d['steps'][-1].__setitem__(k, 'broken')) for key in HASHES]
    mutations += [
        ('loss', lambda d: d['steps'][399].__setitem__('loss', 12345.0)),
        ('dispatch', lambda d: d.__setitem__('gemm_stage_ftz', False)),
        ('coverage', lambda d: d['steps'].pop()),
        ('sequence', lambda d: d['steps'][399].__setitem__('step', 7)),
        ('corpus', lambda d: d['corpus'].__setitem__('sha256', 'broken')),
        ('attention', lambda d: d['steps'][399]['attention'].__setitem__('eager_bytes', -1)),
        ('binding', lambda d: d['run_metadata'].__setitem__('binding_sha256', a['run_metadata']['binding_sha256'])),
    ]
    for reason, mutate in mutations:
        bad = copy.deepcopy(b)
        mutate(bad)
        try:
            compare(a, bad, False)
        except AssertionError as exc:
            assert str(exc) == reason, ('unrelated failure', reason, str(exc))
            print('EXPECTED FAIL', reason)
        else:
            raise AssertionError('BLIND comparator: '+reason)
    compare(a, b)


def main():
    root = Path(sys.argv[1])
    summary = {}
    corpora = sys.argv[2:] or ['enwik8', 'pile_github']
    assert corpora and set(corpora) <= {'enwik8', 'pile_github'}
    for corpus in corpora:
        a, b = [json.loads((root/corpus/arm/'result.json').read_text()) for arm in ('base', 'stage')]
        print('CORPUS', corpus)
        verify(a, b)
        times = [statistics.median(r['seconds'] for r in d['steps'][-200:]) for d in (a, b)]
        summary[corpus] = dict(base_seconds=times[0], stage_seconds=times[1], ratio=times[1]/times[0])
    summary['geomean_ratio'] = math.prod(summary[c]['ratio'] for c in corpora)**(1/len(corpora))
    summary['verdict'] = ('FASTER' if all(summary[c]['ratio'] < 1 for c in corpora) else 'NO FLIP') if len(corpora) == 2 else 'ONE CORPUS ONLY'
    print(json.dumps(summary, indent=2))
    (root/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
