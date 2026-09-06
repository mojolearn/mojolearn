#!/usr/bin/env python3
"""Read-only artifact admission; no model execution or cross-vendor claim.

Inventory mode hashes the complete extracted source archive before bootstrap.
The launcher compares local and remote inventories byte-for-byte. Admission
binds retained raw capture/reference bytes to the reported numerical result;
it does not independently rerun the numerical oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

JOBS = (
    'training-venv', 'training-dependencies', 'dependencies', 'dependency-freeze',
    'guard-checks', 'gradient-build', 'gradient-capture', 'gradient-oracle',
    'training-build', 'linalg-build', 'mlp-surface-and-reference',
    'mlp-numerical-edges', 'mlp-continuous', 'mlp-head', 'mlp-resume', 'mlp-compare',
    'retain-training', 'retain-linalg',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    require(path.is_file() and not path.is_symlink(), f'missing regular file: {path}')
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path, maximum=2 * 1024 * 1024):
    require(path.is_file() and not path.is_symlink(), f'missing regular file: {path}')
    with path.open('rb') as handle:
        value = handle.read(maximum + 1)
    require(len(value) <= maximum, f'oversize artifact: {path}')
    return value


def inventory(root):
    files = {}
    for path in sorted(root.rglob('*')):
        # Nothing is excluded: this runs before any environment is installed.
        require(not path.is_symlink(), f'source symlink refused: {path}')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = digest(path)
    require(files, 'empty frozen source inventory')
    return dict(schema='mojolearn.training.source-inventory.v1', files=files)


def admit(root):
    # This import contains only stdlib definitions; no NumPy/Torch work occurs.
    from transformer_training_gradient_oracle import (
        ARRAY_COUNTS, PROFILE, SCHEMA, registry, GRAD_ATOL, GRAD_RTOL,
        LOSS_ATOL, LOSS_RTOL,
    )
    require(read(root / 'exit_code').strip() == b'0', 'campaign did not finish successfully')
    rows = [line.split('\t') for line in read(root / 'results.tsv').decode().splitlines()]
    require(rows == [[name, '0'] for name in JOBS], 'missing, duplicate, reordered, or failed job')
    provenance = read(root / 'provenance.txt').decode().splitlines()
    require('vendor=cuda' in provenance, 'NVIDIA provenance required')
    require(any(re.fullmatch(r'source=[0-9a-f]{40,64}', line) for line in provenance),
            'frozen commit missing')
    for name in JOBS:
        require((root / (name + '.log')).is_file(), f'missing job log: {name}')
    for name in ('mlp-surface-and-reference', 'mlp-numerical-edges'):
        log = read(root / (name + '.log')).decode(errors='replace')
        require(re.search(r'\b[1-9][0-9]* passed\b', log) is not None,
                f'no executed passing tests: {name}')
        require(re.search(r'\b[1-9][0-9]* (?:skipped|failed|errors?|xfailed|xpassed)\b', log) is None,
                f'incomplete test qualification: {name}')

    capture_dir = root / 'gradient-capture'
    manifest = read(capture_dir / 'capture.json')
    capture = json.loads(manifest)
    require(capture.get('schema') == SCHEMA and capture.get('profile') == PROFILE
            and capture.get('registry') == registry() and capture.get('vendor') == 'cuda'
            and capture.get('numeric_mode') == 'identical'
            and capture.get('completed_steps') == 1, 'capture profile mismatch')
    frozen = json.loads(read(root.parent / 'source_inventory.json', 16 * 1024 * 1024))
    sources = capture.get('source_sha256', {})
    require(isinstance(sources, dict) and sources
            and all(frozen.get('files', {}).get(name) == value for name, value in sources.items()),
            'capture source hashes differ from the frozen archive')
    require(set(capture.get('arrays', {})) == set(ARRAY_COUNTS), 'capture arrays incomplete')
    for name, count in ARRAY_COUNTS.items():
        item = capture['arrays'][name]
        require(item.get('file') == name + '.f32' and item.get('count') == count,
                f'capture descriptor mismatch: {name}')
        raw = read(capture_dir / item['file'], count * 4)
        require(len(raw) == count * 4 and hashlib.sha256(raw).hexdigest() == item.get('sha256'),
                f'capture byte mismatch: {name}')
        require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', raw)),
                f'nonfinite capture: {name}')
    ids = capture.get('token_ids', {})
    require(ids.get('file') == 'token_ids.i32' and ids.get('shape') == [2, 9], 'token profile mismatch')
    raw = read(capture_dir / 'token_ids.i32', 72)
    require(len(raw) == 72 and hashlib.sha256(raw).hexdigest() == ids.get('sha256')
            and all(0 <= x[0] < 64 for x in struct.iter_unpack('<i', raw)), 'token bytes mismatch')

    oracle = json.loads(read(root / 'gradient-oracle.json'))
    require(oracle.get('schema') == 'mojolearn.training.gradient-oracle.v1'
            and oracle.get('passed') is True and oracle.get('vendor') == 'cuda'
            and oracle.get('cuda_version') and not oracle.get('hip_version')
            and oracle.get('device'), 'oracle not admitted on NVIDIA')
    require(oracle.get('capture_manifest_sha256') == hashlib.sha256(manifest).hexdigest(),
            'oracle references another capture')
    require(oracle.get('tolerances') == dict(grad_atol=GRAD_ATOL, grad_rtol=GRAD_RTOL,
            loss_atol=LOSS_ATOL, loss_rtol=LOSS_RTOL), 'oracle tolerance changed')

    def passed(item, count):
        return (isinstance(item, dict) and item.get('passed') is True
                and item.get('cells') == count and item.get('failed_cells') == 0)

    require(passed(oracle.get('loss'), 1), 'loss failed')
    gradients = oracle.get('gradients', {})
    require(set(gradients) == {r['name'] for r in registry()}, 'gradient registry mismatch')
    for item in registry():
        require(passed(gradients[item['name']], item['count']), f'gradient failed: {item["name"]}')
    controls = oracle.get('controls', {})
    require(oracle.get('parameters_moved') is True and controls.get('sign_effective') is True
            and controls.get('nonlinear_effective') is True
            and passed(controls.get('nonlinear_forward'), 1), 'oracle controls ineffective')
    references = oracle.get('reference_files', {})
    require(set(references) == {'reference', 'nonlinear-control'}, 'reference files missing')
    for label, item in references.items():
        expected = 'gradient-oracle.json.' + label + '.npz'
        require(item.get('file') == expected and digest(root / expected) == item.get('sha256'),
                f'reference bytes mismatch: {label}')
    mlp = json.loads(read(root / 'mlp-comparison.json'))
    require(mlp.get('schema') == 'small-mlp.comparison.v1' and mlp.get('identity') == 'PASS'
            and mlp.get('learning', {}).get('status') == 'PASS'
            and mlp.get('vendors') == {'left': ['cuda'], 'right': ['cuda', 'cuda']},
            'same-device MLP continuation or learning failed')
    for directory, start, end in (('mlp-continuous', 0, 16), ('mlp-head', 0, 8),
                                  ('mlp-resume', 8, 16)):
        folder = root / directory
        meta = json.loads(read(folder / 'metadata.json', 262144))
        require(meta.get('schema') == 'small-mlp.capture.v1' and meta.get('status') == 'CAPTURED'
                and meta.get('vendor') == 'cuda' and meta.get('numeric_mode') == 'identical'
                and meta.get('training_mode') == 1 and meta.get('linalg_mode') == 'identical'
                and meta.get('start_step') == start and meta.get('end_step') == end,
                f'incomplete MLP capture: {directory}')
        require(meta.get('sources') and all(frozen.get('files', {}).get(name) == value
                for name, value in meta['sources'].items()), 'MLP source inventory mismatch')
        entries = [meta['inputs'], meta['initial_state'], *meta['checkpoints']]
        require([item['step'] for item in meta['steps']] == list(range(start + 1, end + 1)),
                'MLP step sequence incomplete')
        entries.extend(item['archive'] for item in meta['steps'])
        if meta.get('resume_input'):
            entries.append(meta['resume_input'])
        for item in entries:
            name = item['file']
            require(isinstance(name, str) and Path(name).name == name
                    and digest(folder / name) == item.get('file_sha256', item.get('sha256')),
                    f'MLP retained byte mismatch: {directory}')
        for name in ('training', 'linalg'):
            binding = meta['bindings'][name]
            require(binding.get('vendor') == 'cuda' and binding.get('sha256') ==
                    digest(root / 'bindings' / ('_mojolearn_' + name + '.so')),
                    'MLP retained binding mismatch')
    retained = {}
    for name in ('train-gradient-capture', 'bindings/_mojolearn_training.so',
                 'bindings/_mojolearn_linalg.so'):
        require((root / name).stat().st_size > 0, f'empty binary: {name}')
        retained[name] = digest(root / name)
    return dict(schema='mojolearn.training.admission.v1', passed=True,
                scope='NVIDIA integration and fixed-profile FP64 tolerance checks only; '
                      'no cross-vendor, bitwise, or performance certificate',
                jobs=list(JOBS), retained_sha256=retained)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path, nargs='?')
    parser.add_argument('--inventory-root', type=Path)
    args = parser.parse_args()
    if (args.artifacts is None) == (args.inventory_root is None):
        parser.error('select artifacts or --inventory-root')
    try:
        result = inventory(args.inventory_root) if args.inventory_root else admit(args.artifacts)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Training admission refused: {exc}\n')
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
