#!/usr/bin/env python3
"""Fail closed when a retained feature campaign is missing jobs or arms."""
import json
from pathlib import Path
import math
import re
import sys


def validate_comparison(path, lane):
    path = Path(path)
    result = json.loads((path / f'{lane}-three-arm/results.json').read_text())
    if result['status'] != 'PASSED' or set(result['metadata']) != {'fast', 'identical', 'external'}:
        raise ValueError('Missing admitted three-arm comparison: ' + lane)
    if not result['accuracy'] or not all(r['passed'] for r in result['accuracy']):
        raise ValueError('Missing accuracy admission: ' + lane)
    metadata = result['metadata']
    inputs = {tuple(row['inputs']) for row in metadata.values()}
    if len(inputs) != 1 or any(not re.fullmatch('[0-9a-f]{64}', h) for h in next(iter(inputs))):
        raise ValueError('Input provenance differs: ' + lane)
    for mode in ('fast', 'identical'):
        if metadata[mode]['mode'] != mode or not re.fullmatch('[0-9a-f]{64}', metadata[mode]['binding_sha256']):
            raise ValueError('Missing compiled mode/binary witness: ' + lane)
    rounds = result['args']['rounds']
    if type(rounds) is not int or rounds < 7:
        raise ValueError('Insufficient timed rounds: ' + lane)
    expected = {(arm, r) for arm in metadata for r in range(rounds + 1)}
    records = result['records']
    if len(records) != len(expected) or {(r['arm'], r['round']) for r in records} != expected:
        raise ValueError('Missing/duplicate timed arms: ' + lane)
    for row in records:
        if not math.isfinite(row['ms']) or row['ms'] <= 0 or row['warmup'] is not (row['round'] == 0):
            raise ValueError('Invalid duration/warmup witness: ' + lane)
        if not row['hashes'] or any(not re.fullmatch('[0-9a-f]{64}', h) for h in row['hashes']):
            raise ValueError('Missing output bytes witness: ' + lane)
    if len({tuple(r['hashes']) for r in records if r['arm'] == 'identical'}) != 1:
        raise ValueError('IDENTICAL output hashes changed: ' + lane)
    quality = result['accuracy']
    if len(quality) != len(expected) or {(r['arm'], r['round']) for r in quality} != expected:
        raise ValueError('Missing per-arm quality admission: ' + lane)
    return result


def validate(path):
    path = Path(path)
    required = {'vendor-venv', 'vendor-wheels', 'vendor-freeze', 'vendor-cuda',
                'mamba-corpus', 'mamba23-backward', 'nan-forbidden',
                'mamba2-native-dump', 'mamba3-native-dump',
                'umap-finite-params', 'umap-controls-fast', 'umap-controls-identical',
                'umap-identity', 'compare-gbdt', 'compare-umap'}
    for mode in ('fast', 'identical'):
        required.update(f'build-{binding}-{mode}' for binding in ('gbdt', 'metrics', 'mamba'))
        required.update(f'{surface}-{mode}' for surface in ('gbdt-boundary', 'umap-api', 'mamba-api'))
    rows = [line.split('\t') for line in (path / 'results.tsv').read_text().splitlines()]
    if len(rows) != len(required) or {r[0] for r in rows} != required:
        raise ValueError('Missing or duplicate required campaign jobs')
    if any(len(r) != 2 or r[1] != '0' for r in rows):
        raise ValueError('Campaign contains failed or skipped jobs')
    if (path / 'exit_code').read_text().strip() != '0':
        raise ValueError('Campaign did not exit successfully')
    for lane in ('gbdt', 'umap'):
        validate_comparison(path, lane)
    return {'status': 'PASSED', 'scope': 'NVIDIA source/API fixture qualification and two comparisons; no universal or cross-vendor bitwise certification'}


if __name__ == '__main__':
    print(json.dumps(validate(sys.argv[1]), indent=2))
