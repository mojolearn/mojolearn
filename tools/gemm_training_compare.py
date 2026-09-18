#!/usr/bin/env python3
"""Strict 700-step GEMM operand-flush A/B, with executable negative controls."""
import argparse
import copy
import json
import math
import statistics
import struct
import sys
from pathlib import Path

SHAPE = [1, 2048, 768, 12, 12, 64, 2048, 12, 50257]
HASHES = ('loss', 'gradients', 'parameters', 'm', 'v', 'flags')


def compare(a, b, verbose=True, dispatch_key='gemm_stage_ftz'):
    assert a['shape'] == b['shape'] == SHAPE, 'shape'
    assert a['seed'] == b['seed'] == 20260917, 'seed'
    assert a['corpus'] == b['corpus'] and a['corpus']['sha256'], 'corpus'
    assert a[dispatch_key] is False and b[dispatch_key] is True, 'dispatch'
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
        assert all(math.isfinite(t) and t > 0 for t in (x['seconds'], y['seconds'])), 'timing'
        if verbose:
            print('MATCH step', index, 'loss', x['loss'], y['loss'])
        if index in (0, 699):
            for key in HASHES:
                assert isinstance(x[key], str) and len(x[key]) == 64 and x[key] == y[key], key
                if verbose:
                    print('MATCH step', index, key, x[key], y[key])
        else:
            assert isinstance(x['loss'], (int, float)) and math.isfinite(x['loss']), 'finite'


def verify(a, b, dispatch_key='gemm_stage_ftz'):
    # Each intentional defect must fail for its own reason, not an unrelated one.
    mutations = [(key, lambda d, k=key: d['steps'][-1].__setitem__(k, 'broken')) for key in HASHES]
    mutations += [
        ('shape', lambda d: d['shape'].__setitem__(0, 4)),
        ('seed', lambda d: d.__setitem__('seed', 123)),
        ('completed', lambda d: d['steps'][399].__setitem__('completed_steps', 7)),
        ('timing', lambda d: d['steps'][399].__setitem__('seconds', -1.0)),
        ('timing', lambda d: d['steps'][399].__setitem__('seconds', float('inf'))),
        ('loss', lambda d: d['steps'][399].__setitem__('loss', 12345.0)),
        ('dispatch', lambda d: d.__setitem__(dispatch_key, False)),
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
            compare(a, bad, False, dispatch_key)
        except AssertionError as exc:
            assert str(exc) == reason, ('unrelated failure', reason, str(exc))
            print('EXPECTED FAIL', reason)
        else:
            raise AssertionError('BLIND comparator: '+reason)
    bad_a, bad_b = copy.deepcopy(a), copy.deepcopy(b)
    for d in (bad_a, bad_b):
        d['steps'][399]['loss'] = float('nan')
    try:
        compare(bad_a, bad_b, False, dispatch_key)
    except AssertionError as exc:
        assert str(exc) == 'finite', ('unrelated failure', str(exc))
        print('EXPECTED FAIL finite')
    else:
        raise AssertionError('BLIND comparator: finite')
    compare(a, b, dispatch_key=dispatch_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('corpora', nargs='*')
    parser.add_argument('--kind', choices=('stage','workspace'), default='stage')
    args = parser.parse_args()
    root = args.root
    key = 'gemm_stage_ftz' if args.kind == 'stage' else 'gemm_reuse_group_ws'
    summary = {}
    corpora = args.corpora or ['enwik8', 'pile_github']
    assert corpora and set(corpora) <= {'enwik8', 'pile_github'}
    for corpus in corpora:
        a, b = [json.loads((root/corpus/arm/'result.json').read_text()) for arm in ('base', 'stage')]
        print('CORPUS', corpus)
        verify(a, b, key)
        times = [statistics.median(r['seconds'] for r in d['steps'][-200:]) for d in (a, b)]
        peaks = [max((r['device_used_mb'] for r in d['steps'] if r['device_used_mb'] is not None), default=None) for d in (a,b)]
        summary[corpus] = dict(base_seconds=times[0], candidate_seconds=times[1], ratio=times[1]/times[0], sampled_device_peak_mib=peaks)
    summary['geomean_ratio'] = math.prod(summary[c]['ratio'] for c in corpora)**(1/len(corpora))
    summary['verdict'] = ('FASTER' if all(summary[c]['ratio'] < 1 for c in corpora) else 'NO FLIP') if len(corpora) == 2 else 'ONE CORPUS ONLY'
    print(json.dumps(summary, indent=2))
    (root/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
