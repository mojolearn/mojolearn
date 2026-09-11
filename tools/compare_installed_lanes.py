#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare successful installed UMAP/Ordered lanes without approving a candidate.

Reads each candidate's qualification-normalized directory. Failed unrelated
jobs remain visible in all 24 status rows. This diagnostic cannot release a
wheel and does not replace full qualification admission.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from compare_ordered_python import ARRAYS, PREFIX, check_record
from verify_linux_surface_qualification import BINDINGS, FIXTURES, MODES, SURFACES, check_quality, expected_bindings, expected_jobs, require

TARGETS = {'umap', 'umap-transform', 'umap-quality', 'ordered-rmse'}
EXTENSION = re.compile(r'mojolearn/(cuda|hip)/(sm_[0-9]+a?|gfx[0-9a-f]+)/(?:(deterministic|identical)/)?(_mojolearn[^/]*)\.so')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit_wheel(candidate, qualification):
    audit = read(qualification / 'wheel-audit.json')
    vendor = audit['qualification_vendor']
    require(vendor in ('hip', 'cuda'), 'Unsupported vendor')
    wheels = list((candidate / 'normalized').glob('*.whl'))
    require(len(wheels) == 1 and 'manylinux_' in wheels[0].name, 'Need one normalized manylinux wheel')
    require(digest(wheels[0]) == audit['sha256'], 'Local normalized wheel SHA differs')
    proof_path = candidate / 'build/build-provenance.json'
    proof = read(proof_path)
    require(digest(proof_path) == audit['build_provenance_sha256'], 'Build proof hash differs')
    require(proof.get('schema') == 'mojolearn.linux.build-provenance.v1'
            and proof.get('complete') is True and proof.get('build_exit') == 0
            and proof.get('action') == 'build', 'Incomplete build proof')
    require(re.fullmatch(r'[0-9a-f]{40}', proof.get('source_commit', '')) is not None,
            'Missing build source commit')
    inventory = proof['source_inventory']
    require(isinstance(inventory, list) and inventory
            and len(inventory) == len({p for p, _ in inventory}), 'Invalid source inventory')
    for path, hashed in inventory:
        require(not PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts
                and re.fullmatch(r'[0-9a-f]{64}', hashed) is not None, 'Invalid source inventory entry')
    source_sha = hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()
    require(source_sha == proof['source_sha256'] == audit['source_sha256'], 'Source proof hash differs')
    require(len(proof['extensions']) == 45, 'Need full 45-extension proof')
    wrappers = {path.removeprefix('python/'): hashed for path, hashed in inventory
                if path.startswith('python/mojolearn/') and path.endswith('.py')
                and len(PurePosixPath(path).parts) == 3}
    require(wrappers, 'Missing wrapper source inventory')
    extensions, sets, seen_wrappers = {}, {}, set()
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        require(len(names) == len(set(names)), 'Duplicate wheel member')
        records = [p for p in names if p.endswith('.dist-info/RECORD')]
        require(len(records) == 1, 'Need one wheel RECORD')
        declared = set()
        for name, hashed, size in csv.reader(io.StringIO(wheel.read(records[0]).decode())):
            require(name not in declared, 'Duplicate RECORD row')
            declared.add(name)
            if name == records[0]:
                continue
            with wheel.open(name) as stream:
                actual = hashlib.file_digest(stream, 'sha256').digest()
            encoded = base64.urlsafe_b64encode(actual).rstrip(b'=').decode()
            require(hashed == 'sha256=' + encoded and int(size) == wheel.getinfo(name).file_size,
                    'Wheel RECORD differs: ' + name)
            if name in wrappers:
                require(actual.hex() == wrappers[name], 'Wheel wrapper differs from build source: ' + name)
                seen_wrappers.add(name)
            match = EXTENSION.fullmatch(name)
            if match:
                v, arch, mode, extension = match.groups()
                extensions[name.removeprefix('mojolearn/')] = actual.hex()
                sets.setdefault((v, arch, mode or 'fast'), set()).add(extension)
            elif name.endswith('.so') and '/_mojolearn' in name:
                raise ValueError('Unexpected extension location: ' + name)
        require(declared == set(names), 'Incomplete wheel RECORD')
    require(seen_wrappers == set(wrappers), 'Missing proven Python wrappers in wheel')
    require(extensions == audit['extension_hashes'], 'Wheel/audit extension inventory differs')
    require({'/'.join(k): len(v) for k, v in sets.items()} == audit['sets'], 'Wheel set audit differs')
    require(sorted({v for v, _, _ in sets}) == audit['advertised_vendors'], 'Advertised vendors differ')
    for v, arch in {(v, arch) for v, arch, _ in sets}:
        for mode in MODES:
            require(sets.get((v, arch, mode)) == expected_bindings(mode), 'Incomplete wheel vendor/tier set')
    own = {'mojolearn/' + name: hashed for name, hashed in extensions.items()
           if name.startswith(vendor + '/')}
    require(own == proof['extensions'], 'Wheel/proof vendor extension inventory differs')
    return audit


def status_rows(qualification, audit):
    rows = [line.split('\t') for line in (qualification / 'results.tsv').read_text().splitlines()]
    expected = expected_jobs(audit)
    require(len(rows) == len(expected) and all(len(r) == 3 for r in rows), 'Need every installed exit row')
    require({(s, m) for s, m, _ in rows} == expected, 'Missing/duplicate installed exit rows')
    require(all(re.fullmatch(r'[0-9]+', code) is not None for _, _, code in rows), 'Invalid exit code')
    require(all(code == '0' for surface, _, code in rows if surface in TARGETS), 'Target lane failed')
    return [{'surface': s, 'mode': m, 'exit_code': int(code)} for s, m, code in rows]


def installed(qualification, name, mode, audit):
    record = read(qualification / (name + '-' + mode + '.installed.json'))
    require(record.get('mode') == mode and record.get('vendor') == audit['qualification_vendor']
            and record.get('wheel_sha256') == audit['sha256'], 'Installed provenance differs')
    package = Path(record['package']).parent
    require(package.name == 'mojolearn' and 'site-packages' in package.parts
            and 'venv' in package.parts, 'Installed package path is not isolated')
    bindings = record['installed_bindings']
    require(set(bindings) == expected_bindings(mode), 'Incomplete installed binding readback')
    for name, row in bindings.items():
        member = str(Path(row['path']).relative_to(package))
        require(row['sha256'] == audit['extension_hashes'].get(member), 'Installed binding hash differs')
        match = EXTENSION.fullmatch('mojolearn/' + member)
        require(match is not None and match[1] == audit['qualification_vendor']
                and (match[3] or 'fast') == mode and match[4] == name, 'Installed binding path/tier differs')
        if 'mode_code' in row:
            require(type(row['mode_code']) is int and row['mode_code'] == MODES[mode], 'Installed mode readback differs')
    require(bindings['_mojolearn_gbdt'].get('mode_code') == MODES[mode]
            and bindings['_mojolearn_metrics'].get('mode_code') == MODES[mode],
            'Target native mode readback missing')
    return bindings


def load(candidate):
    candidate = Path(candidate)
    qualification = candidate / 'qualification-normalized'
    audit = audit_wheel(candidate, qualification)
    rows = status_rows(qualification, audit)
    candidate_status = read(candidate / 'candidate-status.json')
    qualification_status = read(qualification / 'qualification.json')
    require(candidate_status.get('status') in ('PASSED', 'FAILED'), 'Missing candidate disposition')
    require(qualification_status.get('status') in ('PASSED', 'FAILED'), 'Missing qualification disposition')
    exit_text = (qualification / 'exit_code').read_text().strip()
    require(re.fullmatch(r'[0-9]+', exit_text) is not None, 'Missing qualification exit marker')
    for surface in TARGETS:
        for mode in MODES:
            if (surface, mode) not in expected_jobs(audit):
                continue  # DEVIATION 2490: identical-only surfaces have no lower-tier job
            require((qualification / (surface + '-' + mode + '.log')).is_file(), 'Missing target log')
            bindings = installed(qualification, surface, mode, audit)
            if surface == 'umap-quality':
                check_quality(read(qualification / (surface + '-' + mode + '.json')), mode,
                              bindings['_mojolearn_metrics']['sha256'])
    lines = (qualification / 'ordered-rmse-identical.log').read_text().splitlines()
    values = [line[len(PREFIX):] for line in lines if line.startswith(PREFIX)]
    require(len(values) == 1 and lines.count('ORDERED PYTHON SURFACE PASS') == 1,
            'Incomplete/duplicate ordered log')
    require(lines.count('ORDERED_PYTHON_NATIVE identical ' + audit['qualification_vendor']) == 1,
            'Ordered native log readback differs')
    ordered = check_record(json.loads(values[0]), audit['qualification_vendor'])
    quality = read(qualification / 'umap-quality-identical.json')
    qualification_sources = read(qualification / 'qualification-sources.json')
    require(isinstance(qualification_sources, dict) and qualification_sources,
            'Missing qualification source inventory')
    summary = {
        'vendor': audit['qualification_vendor'], 'wheel_sha256': audit['sha256'],
        'source_sha256': audit['source_sha256'],
        'qualification_sources_sha256': digest(qualification / 'qualification-sources.json'),
        'candidate_status': candidate_status, 'qualification_status': qualification_status,
        'qualification_exit_code': int(exit_text), 'installed_exit_rows': rows,
        'retained_evidence_sha256': {p.name: digest(p) for p in sorted(qualification.iterdir()) if p.is_file()},
    }
    return summary, quality, ordered


def compare(left, right):
    a, aq, ao = load(left)
    b, bq, bo = load(right)
    require({a['vendor'], b['vendor']} == {'hip', 'cuda'}, 'Need HIP and CUDA candidates')
    for field in ('source_sha256', 'qualification_sources_sha256'):
        require(a[field] == b[field], 'Different source: ' + field)
    left_rows = {r['profile']: r for r in aq['results']}
    right_rows = {r['profile']: r for r in bq['results']}
    fields = ('training_input', 'query_input', 'training_embedding', 'query_embedding',
              'parameters', 'transform_schedule', 'fitted_config', 'fitted_mode')
    for fixture in FIXTURES:
        for field in fields:
            require(left_rows[fixture][field] == right_rows[fixture][field], 'UMAP mismatch: ' + fixture + '/' + field)
    for field in ('model_text', 'model_sha256', *ARRAYS):
        require(ao[field] == bo[field], 'Ordered mismatch: ' + field)
    return {
        'schema': 'mojolearn.linux.installed-lane-comparison.v1', 'lane_status': 'MATCH',
        'overall_release_eligible': False,
        'scope': 'UMAP and numeric OrderedRMSE lane evidence only; full qualification/release admission remains separate',
        'source_sha256': a['source_sha256'], 'umap_fixtures': sorted(FIXTURES),
        'umap_compared_arrays': 24, 'ordered_prediction_cells': 72,
        'ordered_model_sha256': ao['model_sha256'], 'candidates': [a, b],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('amd', type=Path)
    parser.add_argument('nvidia', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.amd, args.nvidia)
    except (ValueError, OSError, KeyError, TypeError, AttributeError, zipfile.BadZipFile) as exc:
        result = {'schema': 'mojolearn.linux.installed-lane-comparison.v1',
                  'lane_status': 'FAILED', 'overall_release_eligible': False, 'reason': str(exc)}
    text = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')
    return 0 if result['lane_status'] == 'MATCH' else 1


if __name__ == '__main__':
    raise SystemExit(main())
