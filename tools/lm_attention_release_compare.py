#!/usr/bin/env python3
"""Release with automatic fused/eager transitions; all 700 losses and state hashes."""
import copy
import json
import sys
from pathlib import Path
from lm_attention_repair_compare import HASHES, default_capacity


def compare(a, b, verbose=False):
    assert a['shape'] == b['shape'] and a['seed'] == b['seed'] and a['corpus'] == b['corpus'], 'inputs'
    assert len(a['steps']) == len(b['steps']) == 700, 'coverage'
    default_capacity(b)
    assert any(row['attention']['released_eager_bytes'] > 0 for row in b['steps']), 'INERT release'
    for i, (x, y) in enumerate(zip(a['steps'], b['steps'])):
        assert x['step'] == y['step'] == i, 'sequence'
        assert x['attention']['repair_masked_tail'] is y['attention']['repair_masked_tail'] is False, 'legacy arithmetic'
        assert x['attention']['forward_status'] == y['attention']['forward_status'], 'forward routing'
        assert x['attention']['backward_status'] == y['attention']['backward_status'], 'backward routing'
        assert x['loss'] == y['loss'], 'loss'
        if verbose: print('MATCH release step', i, 'loss', x['loss'], 'eager_bytes', y['attention']['eager_bytes'])
        if i in (0, 699):
            for key in HASHES:
                assert x[key] == y[key], key
                if verbose: print('MATCH release step', i, key, x[key])


def main():
    root = Path(sys.argv[1])
    a = json.loads((root/'legacy/result.json').read_text())
    b = json.loads((root/'released_legacy/result.json').read_text())
    for key in HASHES:
        broken = copy.deepcopy(b)
        broken['steps'][-1][key] = 'corrupted'
        try: compare(a, broken)
        except AssertionError as e:
            assert str(e) == key, str(e)
            print('EXPECTED FAIL release', key)
        else: raise AssertionError('BLIND release ' + key)
    for key, label, value in (('eager_bytes', 'eager capacity', 999),
                              ('forward_aexp_bytes', 'aexp capacity', 999),
                              ('release_eager', 'default storage policy', False),
                              ('forward_status', 'forward routing', [-1]*12),
                              ('backward_status', 'backward routing', [-1]*12)):
        broken = copy.deepcopy(b)
        broken['steps'][-1]['attention'][key] = value
        try: compare(a, broken)
        except AssertionError as e:
            assert str(e) == label, str(e)
            print('EXPECTED FAIL release', label)
        else: raise AssertionError('BLIND release ' + label)
    compare(a, b, True)


if __name__ == '__main__': main()
