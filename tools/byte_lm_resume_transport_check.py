#!/usr/bin/env python3
"""Check compact transport completion only; never admit numerical identity."""
import argparse
import json
from pathlib import Path


def read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('missing/unsafe/oversized diagnostic: ' + str(path))
    return path.read_text()


def jobs(directory, expected):
    if read(directory / 'exit_code').strip() != '0':
        raise ValueError('campaign failed')
    rows = [line.split('\t') for line in read(directory / 'results.tsv').splitlines()]
    if rows != [[name, '0'] for name in expected]:
        raise ValueError('missing/reordered/refused job')
    for name in expected:
        read(directory / (name + '.log'))
        if not read(directory / (name + '.command.txt')).strip():
            raise ValueError('missing command witness')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--action', required=True, choices=('head64', 'resume128'))
    parser.add_argument('--vendor', required=True, choices=('nvidia', 'amd'))
    parser.add_argument('--baseline-sha', required=True)
    parser.add_argument('--foreign-sha', default='')
    parser.add_argument('--checkpoint-sha', default='')
    args = parser.parse_args()
    root = args.directory
    vendor = dict(nvidia='cuda', amd='hip')[args.vendor]
    setup = root / 'byte-lm-resume-setup'
    # Compact itself owns its command/log witnesses in a separate fresh output.
    if read(setup / 'exit_code').strip() != '0':
        raise ValueError('transport setup failed')
    if read(setup / 'results.tsv').splitlines() != ['venv\t0', 'numpy\t0', 'binding\t0', 'compact\t0']:
        raise ValueError('incomplete setup')
    for name in ('venv', 'numpy', 'binding'):
        read(setup / (name + '.log'))
        read(setup / (name + '.command.txt'))
    out = root / 'byte-lm-resume'
    expected = ['preflight', 'head64', 'verify-head'] if args.action == 'head64' else [
        'preflight', 'resume128', 'verify-resume', 'zero-moments65', 'verify-control']
    jobs(out, expected)
    before = json.loads(read(out / 'preflight.json'))
    for key, value in dict(schema='mojolearn.byte-lm.compact-resume-preflight.v1',
                           vendor=vendor, action=args.action,
                           baseline_handoff_sha256=args.baseline_sha,
                           foreign_handoff_sha256=args.foreign_sha or None,
                           foreign_checkpoint_sha256=args.checkpoint_sha or None).items():
        if before.get(key) != value:
            raise ValueError('preflight pin mismatch: ' + key)
    for name in ['preflight'] + [item for item in expected if item.startswith('verify-')]:
        data = json.loads(read(out / (name + '.json')))
        if data.get('identity_admitted') is not False or data.get('learning_admitted') is not False:
            raise ValueError('compact record improperly claims numerical admission')
    print(json.dumps(dict(status='DIAGNOSTIC_COMPLETE', action=args.action, vendor=vendor,
                          identity_admitted=False, learning_admitted=False,
                          required_next='Root local final comparator against all original and new raw arrays')))


if __name__ == '__main__':
    main()
