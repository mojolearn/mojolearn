#!/usr/bin/env python3
"""Validate and deliberately break the shipped B4 2000-step endurance record."""
import copy
import json
import math
import statistics
import sys
from collections import Counter


def validate(r):
    assert r['shape'] == [4,2048,768,12,12,64,2048,12,50257], 'shape'
    assert r['corpus']['sha256'] == '2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8', 'corpus'
    assert len(r['steps']) == 2000, 'coverage'
    for i, row in enumerate(r['steps']):
        assert row['step'] == i and row['completed_steps'] == i + 1, 'sequence'
        assert math.isfinite(row['loss']), 'finite loss'
        a = row['attention']
        assert a['sticky_eager'] is False and a['release_eager'] is True, 'default arm'
        assert a['exact_tail_guard'] is True and a['repair_masked_tail'] is True, 'repair enabled'
        assert len(a['backward_repair_sites']) == 12 and all(v in (0, 1, 2, 3) for v in a['backward_repair_sites']), 'repair coverage'
        assert a['eager_bytes'] == 432, 'eager capacity'
        assert a['forward_aexp_bytes'] == 9663676416, 'aexp capacity'
        assert len(a['forward_status']) == len(a['backward_status']) == 12, 'status coverage'


def main():
    r = json.load(open(sys.argv[1]))
    for name, mutate in (
        ('shape', lambda x: x['shape'].__setitem__(0, 1)),
        ('corpus', lambda x: x['corpus'].__setitem__('sha256', 'wrong')),
        ('coverage', lambda x: x['steps'].pop()),
        ('sequence', lambda x: x['steps'][0].__setitem__('completed_steps', 3)),
        ('finite loss', lambda x: x['steps'][0].__setitem__('loss', float('nan'))),
        ('default arm', lambda x: x['steps'][0]['attention'].__setitem__('release_eager', False)),
        ('repair enabled', lambda x: x['steps'][0]['attention'].__setitem__('repair_masked_tail', False)),
        ('repair coverage', lambda x: x['steps'][0]['attention']['backward_repair_sites'].pop()),
        ('eager capacity', lambda x: x['steps'][0]['attention'].__setitem__('eager_bytes', 999)),
        ('aexp capacity', lambda x: x['steps'][0]['attention'].__setitem__('forward_aexp_bytes', 999)),
        ('status coverage', lambda x: x['steps'][0]['attention']['backward_status'].pop()),
    ):
        broken = copy.deepcopy(r)
        mutate(broken)
        try:
            validate(broken)
        except AssertionError as e:
            assert str(e) == name, ('unrelated failure', str(e))
            print('EXPECTED FAIL:', name)
        else:
            raise AssertionError('BLIND: ' + name)
    validate(r)
    rows = r['steps']
    for row in rows:
        print('MATCH step', row['step'], 'completed', row['completed_steps'],
              'eager_bytes', row['attention']['eager_bytes'], 'finite_loss', row['loss'])
    for key in ('forward_status', 'backward_status', 'backward_repair_sites'):
        print(key, dict(Counter(v for row in rows for v in row['attention'][key])))
    for name, window in (('head', rows[1:51]), ('tail', rows[-200:])):
        seconds = statistics.median(x['seconds'] for x in window)
        print(name, 'median_seconds', seconds, 'tokens_per_second', 8192 / seconds,
              'device_mib', [x['device_used_mb'] for x in window if x['device_used_mb'] is not None])
    print('sampled_device_peak_mib', max(row['device_used_mb'] or 0 for row in rows))
    print('last_attention', rows[-1]['attention'])


if __name__ == '__main__':
    main()
