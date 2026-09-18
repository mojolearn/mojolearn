#!/usr/bin/env python3
"""Compare actual NVIDIA/AMD 700-step numeric witnesses, not execution status."""
import argparse
import copy
import json
import math
import struct
from pathlib import Path
from gemm_training_compare import HASHES, SHAPE


def compare(a, b, verbose=True):
    assert a['shape'] == b['shape'] == SHAPE, 'shape'
    assert a['seed'] == b['seed'] == 20260917, 'seed'
    assert a['corpus'] == b['corpus'], 'corpus'
    vendors = [d['run_metadata']['native_vendor'] for d in (a, b)]
    assert set(vendors) == {'cuda', 'hip'}, 'vendors'
    assert len(a['steps']) == len(b['steps']) == 700, 'coverage'
    for index, (x, y) in enumerate(zip(a['steps'], b['steps'])):
        assert x['step'] == y['step'] == index, 'sequence'
        assert x['completed_steps'] == y['completed_steps'] == index+1, 'completed'
        if index in (0, 699):
            for key in HASHES:
                assert isinstance(x[key], str) and len(x[key]) == 64 and x[key] == y[key], key
                if verbose: print('MATCH', index, key, x[key], y[key])
        else:
            assert math.isfinite(x['loss']) and math.isfinite(y['loss']), 'finite'
            assert struct.pack('<f', x['loss']) == struct.pack('<f', y['loss']), 'loss'
            if verbose: print('MATCH', index, 'loss', x['loss'], y['loss'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nvidia', type=Path)
    parser.add_argument('amd', type=Path)
    args = parser.parse_args()
    a, b = [json.loads(p.read_text()) for p in (args.nvidia, args.amd)]
    changes = [(key, lambda d, k=key: d['steps'][-1].__setitem__(k, 'broken')) for key in HASHES]
    changes += [
        ('shape', lambda d: d['shape'].__setitem__(0, 4)),
        ('seed', lambda d: d.__setitem__('seed', 1)),
        ('corpus', lambda d: d['corpus'].__setitem__('sha256', 'broken')),
        ('vendors', lambda d: d['run_metadata'].__setitem__('native_vendor', 'cuda')),
        ('coverage', lambda d: d['steps'].pop()),
        ('sequence', lambda d: d['steps'][399].__setitem__('step', 7)),
        ('completed', lambda d: d['steps'][399].__setitem__('completed_steps', 7)),
        ('loss', lambda d: d['steps'][399].__setitem__('loss', 12345.0)),
        ('finite', lambda d: d['steps'][399].__setitem__('loss', float('nan'))),
    ]
    for reason, mutate in changes:
        bad = copy.deepcopy(b)
        mutate(bad)
        try: compare(a, bad, False)
        except AssertionError as exc:
            assert str(exc) == reason, ('unrelated failure', reason, str(exc))
            print('EXPECTED FAIL', reason)
        else: raise AssertionError('blind cross-vendor gate: '+reason)
    compare(a, b)
    print('MATCH NVIDIA/AMD: 700 losses, six endpoint state hashes; execution paths may differ')


if __name__ == '__main__':
    main()
