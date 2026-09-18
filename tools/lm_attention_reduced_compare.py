#!/usr/bin/env python3
"""Reduced-shape HD64 training witness across columns and arms
(lane/attention-replay-vendors).

    python3 tools/lm_attention_reduced_compare.py label=dir/result.json [label=dir/result.json ...]

The first result is the reference. Every other one must match it at every
step's loss and every recorded witness hash (loss, gradients, parameters, m,
v, flags). Corruption controls fail first. The shape must use head_dim 64
(at head_dim 8 the fused kernels are not instantiated: a blind shape).
Per-result attention routing and replay sites are printed, so a witness whose
replay never ran is reported INERT rather than read as a replay result.
"""
import copy
import json
import sys
from collections import Counter

HASHES = ('loss', 'gradients', 'parameters', 'm', 'v', 'flags')


def compare(a, b, verbose=False, tag=''):
    assert a['shape'] == b['shape'] and a['seed'] == b['seed'], 'inputs'
    assert a['shape'][5] == 64, 'blind head_dim'
    assert a['corpus']['sha256'] == b['corpus']['sha256'], 'corpus'
    assert len(a['steps']) == len(b['steps']) > 0, 'coverage'
    witnessed = 0
    for x, y in zip(a['steps'], b['steps']):
        assert x['step'] == y['step'], 'sequence'
        assert x['loss'] == y['loss'], 'loss'
        if 'gradients' in x or 'gradients' in y:
            witnessed += 1
            for key in HASHES:
                assert x.get(key) == y.get(key), key
                if verbose:
                    print('MATCH', tag, 'step', x['step'], key, x[key])
    assert witnessed >= 2, 'no witnesses'
    if verbose:
        print('MATCH', tag, len(a['steps']), 'losses,', witnessed, 'witnessed steps')


def main():
    runs = []
    for arg in sys.argv[1:]:
        label, path = arg.split('=', 1)
        runs.append((label, json.load(open(path))))
    ref_label, ref = runs[0]
    for key in HASHES:
        broken = copy.deepcopy(ref)
        row = [r for r in broken['steps'] if 'gradients' in r][-1]
        row[key] = 'deliberately corrupted'
        try:
            compare(ref, broken)
        except AssertionError as e:
            assert str(e) == key, str(e)
            print('EXPECTED FAIL:', key)
        else:
            raise AssertionError('BLIND: ' + key)
    blind = copy.deepcopy(ref)
    blind['shape'][5] = 8
    try:
        compare(blind, blind)
    except AssertionError as e:
        assert str(e) == 'blind head_dim', str(e)
        print('EXPECTED FAIL: blind head_dim')
    else:
        raise AssertionError('BLIND: head_dim 8 accepted')
    for label, r in runs:
        att = [s['attention'] for s in r['steps']]
        print(label, 'shape', r['shape'], 'forward', dict(Counter(v for a in att for v in a['forward_status'])),
              'backward', dict(Counter(v for a in att for v in a['backward_status'])),
              'sites', dict(Counter(v for a in att for v in a['backward_repair_sites'])),
              'repair_masked_tail', att[-1]['repair_masked_tail'], 'release_eager', att[-1]['release_eager'],
              'eager_bytes_last', att[-1]['eager_bytes'])
        if label != ref_label:
            compare(ref, r, True, f'{ref_label}/{label}')


if __name__ == '__main__':
    main()
