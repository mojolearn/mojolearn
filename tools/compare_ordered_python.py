#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare retained installed OrderedRMSE IDENTICAL evidence without GPU access."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from verify_linux_surface_qualification import retained, require, sha

PREFIX = 'ORDERED_PYTHON_JSON '
REFUSALS = {'depth', 'objective', 'categorical', 'duplicate_permutation',
            'zero_weight_mass', 'eval_set'}
ARRAYS = {'prediction_bits': 32, 'query_prediction_bits': 8,
          'repeat_prediction_bits': 32}


def check_record(record, vendor):
    require(record.get('schema') == 'mojolearn.ordered_python.v1'
            and record.get('status') == 'PASS', 'Ordered Python gate did not pass')
    require(record.get('requested_mode') == 'identical'
            and type(record.get('native_mode_code')) is int
            and record['native_mode_code'] == 1
            and record.get('native_vendor') == vendor,
            'Ordered Python native mode/vendor differs')
    require(all(type(record.get(k)) is int and record[k] == n
                for k, n in (('rows', 32), ('query_rows', 8), ('trees', 3))),
            'Unexpected ordered fixture dimensions')
    refusals = record.get('refusals')
    require(isinstance(refusals, list) and len(refusals) == 6
            and all(isinstance(r, str) for r in refusals)
            and set(refusals) == REFUSALS, 'Incomplete ordered refusal inventory')
    require(record.get('repeat_exact') is True, 'Ordered repeat identity failed')
    for name, size in ARRAYS.items():
        cells = record.get(name)
        require(isinstance(cells, list) and len(cells) == size,
                'Unexpected ordered array size: ' + name)
        require(all(type(v) is int and 0 <= v <= 0xffffffff
                    and (v & 0x7f800000) != 0x7f800000 for v in cells),
                'Invalid/nonfinite ordered float32 bits: ' + name)
    require(record['prediction_bits'] == record['repeat_prediction_bits'],
            'Ordered repeat raw bits disagree')
    text = record.get('model_text')
    require(isinstance(text, str) and bool(text) and '\x00' not in text,
            'Missing/invalid ordered model text')
    require([line for line in text.splitlines() if line.startswith('trees ')] == ['trees 3'],
            'Ordered model tree count differs')
    require(record.get('model_sha256') == hashlib.sha256(text.encode()).hexdigest(),
            'Ordered model hash disagrees')
    return record


def load(directory):
    directory = Path(directory)
    qualification, _ = retained(directory)
    require(qualification.get('status') == 'PASSED', 'Installed qualification did not pass')
    vendor = qualification.get('vendor')
    require(vendor in ('hip', 'cuda'), 'Ordered comparison requires HIP/CUDA evidence')
    require(isinstance(qualification.get('source_sha256'), str)
            and re.fullmatch(r'[0-9a-f]{64}', qualification['source_sha256']) is not None,
            'Invalid qualification source hash')
    # retained() verifies hashes for the entire evidence inventory. Also bind
    # this particular surface to its installed wheel, mode and native artifact.
    installed = json.loads((directory / 'ordered-rmse-identical.installed.json').read_text())
    require(installed.get('mode') == 'identical' and installed.get('vendor') == vendor
            and installed.get('wheel_sha256') == qualification['wheel_sha256'],
            'Ordered installed provenance differs')
    binding = installed['installed_bindings']['_mojolearn_gbdt']
    require(type(binding.get('mode_code')) is int and binding['mode_code'] == 1,
            'Ordered installed binding mode differs')
    audit = json.loads((directory / 'wheel-audit.json').read_text())
    package = Path(installed['package']).parent
    member = str(Path(binding['path']).relative_to(package))
    require(binding['sha256'] == audit['extension_hashes'].get(member),
            'Ordered installed binding differs from wheel')
    lines = (directory / 'ordered-rmse-identical.log').read_text().splitlines()
    records = [line[len(PREFIX):] for line in lines if line.startswith(PREFIX)]
    require(len(records) == 1, 'Need exactly one ordered JSON evidence record')
    require(lines.count('ORDERED PYTHON SURFACE PASS') == 1,
            'Missing/duplicate ordered completion marker')
    require(lines.count('ORDERED_PYTHON_NATIVE identical ' + vendor) == 1,
            'Missing/duplicate ordered native readback')
    return qualification, check_record(json.loads(records[0]), vendor)


def compare(left, right):
    a, ar = load(left)
    b, br = load(right)
    require({a['vendor'], b['vendor']} == {'hip', 'cuda'},
            'Comparison requires one HIP and one CUDA qualification')
    require(a['source_sha256'] == b['source_sha256'], 'Different native build sources')
    require(a['evidence_sha256']['qualification-sources.json'] ==
            b['evidence_sha256']['qualification-sources.json'],
            'Different qualification sources')
    for name in ('model_text', 'model_sha256', *ARRAYS):
        require(ar[name] == br[name], 'Ordered IDENTICAL mismatch: ' + name)
    return {
        'schema': 'mojolearn.linux.installed-ordered-identity.v1', 'status': 'PASSED',
        'source_sha256': a['source_sha256'], 'mode': 'identical', 'trees': 3,
        'prediction_cells': 32, 'query_cells': 8, 'repeat_cells': 32,
        'model_sha256': ar['model_sha256'],
        'qualification_sha256': [sha(Path(p) / 'qualification.json') for p in (left, right)],
        'wheel_sha256': {r['vendor']: r['wheel_sha256'] for r in (a, b)},
        'scope': 'Installed numeric single-permutation OrderedRMSE fixture; not full CatBoost parity',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.left, args.right)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        result = {'status': 'FAILED', 'reason': str(exc)}
    text = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')
    return 0 if result['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
