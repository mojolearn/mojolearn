#!/usr/bin/env python3
"""Fail-closed gate for the AMD post-round class-flush device probe."""
import argparse
import re
import sys
from pathlib import Path


def gate(text, require_shipped_class=False):
    if require_shipped_class:
        pattern = r'^SEAM_PROBE column=amd hwftz=False class_shipped=True lanes=shipped,fma,hwftz,swrtf,class$'
        matches = re.findall(pattern, text, re.M)
        print(f'MATCH production class dispatch actual={matches}')
        if len(matches) != 1:
            raise ValueError('production AMD class dispatch missing or duplicated')
    expected = '62a6b5621e27c707'
    for lane in ('shipped', 'swrtf', 'class'):
        rows = re.findall(rf'^SEAM_HASH lane={lane} fnv1a64=([0-9a-f]+)$', text, re.M)
        print(f'MATCH lane={lane} actual={rows} expected={expected}')
        if rows != [expected]:
            raise ValueError(f'{lane}: missing, duplicate, or wrong hash')
    for pattern in (r'^SEAM_N 262144$', r'^SEAM_MISMATCH shipped/class count=0$',
                    r'^SEAM_BOUNDARY .* shipped=00800000 .* class=00800000$', r'^SEAM_DONE$'):
        matches = re.findall(pattern, text, re.M)
        print(f'MATCH required={pattern} actual={matches}')
        if len(matches) != 1:
            raise ValueError(f'missing/duplicate required evidence: {pattern}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--require-shipped-class', action='store_true')
    args = parser.parse_args()
    try:
        gate(args.log.read_text(), args.require_shipped_class)
    except (ValueError, OSError) as exc:
        print(f'FAIL {exc}')
        sys.exit(1)
    print('PASS class flush equals shipped round-then-flush')
