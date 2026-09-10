#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare every native array byte in matching retained two-step LM captures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

NATIVE = ('ids', 'initial_p', 'initial_m', 'initial_v', 'initial_flags',
          'grad', 'post_p', 'post_m', 'post_v', 'post_flags', 'loss')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--cases', nargs='+', default=['default', 'alternate_gqa'])
    args = parser.parse_args()
    records = []
    for case in args.cases:
        for step in range(2):
            name = f'{case}-step{step}.npz'
            paths = [args.before / name, args.after / name]
            with np.load(paths[0], allow_pickle=False) as old, np.load(paths[1], allow_pickle=False) as new:
                checks = {key: (old[key].dtype == new[key].dtype and old[key].shape == new[key].shape
                                and old[key].tobytes() == new[key].tobytes()) for key in NATIVE}
                records.append(dict(file=name, passed=all(checks.values()), arrays=checks,
                    bytes_compared=sum(old[key].nbytes for key in NATIVE),
                    file_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]))
    passed = all(record['passed'] for record in records)
    print(json.dumps(dict(passed=passed, scope='stored native arrays, not timing or cross-vendor admission',
                          captures=records), indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
