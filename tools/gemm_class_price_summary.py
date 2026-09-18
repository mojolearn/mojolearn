#!/usr/bin/env python3
"""Cross-build, cross-column digest gate and fixed-size GEMM prices."""
import copy
import json
import re
import statistics
import sys
from pathlib import Path
KINDS = {f'{k}_{p}' for k in ('proj', 'gateup', 'down', 'head') for p in ('fwd', 'dA', 'dB')}


def read(path):
    text = path.read_text()
    rows = re.findall(r'^BITS (\S+) EQUAL cells=(\d+) digest=(\S+)$', text, re.M)
    assert len(rows) == 12 and {x[0] for x in rows} == KINDS, f'incomplete BITS: {path}'
    return {r[0]: r[1:] for r in rows}


def equal(a, b, verbose=True):
    assert set(a) == set(b) == KINDS
    for k in sorted(KINDS):
        assert a[k] == b[k], f'{k} mismatch: {a[k]} != {b[k]}'
        if verbose: print('MATCH', k, a[k], b[k])


if __name__ == '__main__':
    paths = [Path(p) for p in sys.argv[1:]]
    assert paths, 'supply all base/class price logs on both columns'
    baseline = read(paths[0])
    for k in sorted(KINDS):
        bad = copy.deepcopy(baseline); bad[k] = ('0', 'broken')
        try: equal(baseline, bad, False)
        except AssertionError as exc: print('EXPECTED FAIL', str(exc))
        else: raise AssertionError('gate accepted changed digest')
    for p in paths:
        print('FILE', p)
        equal(baseline, read(p))
        for row in p.read_text().splitlines():
            if row.startswith(('STEP ', 'PRICE ')): print(row)
