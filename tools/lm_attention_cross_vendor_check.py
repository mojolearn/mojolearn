#!/usr/bin/env python3
"""Compare the 130 explicit native repair/preservation bit matches by site."""
import sys
from pathlib import Path


def matches(path):
    return [line for line in Path(path).read_text().splitlines() if line.startswith('MATCH ')]


def equal(a, b):
    assert len(a) == len(b) == 130, 'coverage'
    for x, y in zip(a, b):
        assert x == y, 'site or bits'


def main():
    a, b = map(matches, sys.argv[1:3])
    for name, broken, reason in (
        ('missing site', b[:-1], 'coverage'),
        ('negative zero changed', b[:-1] + [b[-1].replace('fused=0x80000000', 'fused=0x00000000')], 'site or bits'),
    ):
        try: equal(a, broken)
        except AssertionError as e:
            assert str(e) == reason, str(e)
            print('EXPECTED FAIL cross-vendor', name)
        else: raise AssertionError('BLIND cross-vendor ' + name)
    equal(a, b)
    for x in a: print('NVIDIA/AMD', x)


if __name__ == '__main__': main()
