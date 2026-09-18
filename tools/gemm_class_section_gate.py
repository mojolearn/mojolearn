#!/usr/bin/env python3
"""Require exact allocated ELF section identity, after proving rejection."""
import copy
import json
import sys
from pathlib import Path


def equal(a, b, verbose=True):
    assert '.text' in a and '.rodata' in a
    assert set(a) == set(b), 'section set differs'
    for name in sorted(a):
        assert a[name] == b[name], f'section differs: {name}'
        if verbose: print('MATCH_SECTION', name, a[name], b[name])


a, b = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
for name in sorted(a):
    bad = copy.deepcopy(a)
    bad[name]['sha256'] = 'broken'
    try: equal(a, bad, False)
    except AssertionError as exc: print('EXPECTED FAIL', str(exc))
    else: raise AssertionError('section gate accepted corruption')
equal(a, b)
print('INERT: allocated section contents and sizes match')
