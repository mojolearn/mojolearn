#!/usr/bin/env python3
"""Read-only admission of installed Linux surface evidence; never invokes a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

MODES = {'fast': 0, 'deterministic': 2, 'identical': 1}
BINDINGS = {'_mojolearn' + suffix for suffix in ('', '_gbdt', '_estimators', '_rf', '_trees',
            '_svm', '_solver', '_metrics', '_tsa', '_linalg', '_arima', '_training', '_gp',
            '_mamba', '_transformer')}
SURFACES = ('smoke', 'umap', 'umap-transform', 'umap-quality', 'ordered-rmse', 'mamba', 'transformer', 'arima')
FIXTURES = {
    'cubic128_interleaved64_64': (64, 64, 2),
    'saddle_grid64_cellcenters49': (64, 49, 3),
    'cubic256_interleaved128_128_seed3': (128, 128, 2),
    'cubic256_interleaved128_128_seed41': (128, 128, 2),
    'saddle_grid128_cellcenters105_seed11': (128, 105, 3),
    'saddle_grid128_cellcenters105_seed29': (128, 105, 3),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(root):
    paths = set((root / 'python/mojolearn').rglob('*.py'))
    paths.update((root / 'packaging/linux').glob('*.py'))
    paths.update((root / 'tools').glob('*.py'))
    paths.add(root / 'tools/linux_surface_qualification.sh')
    # The installed Mamba surface gate reads these committed reference operands.
    for case in ('base_b2_l4_d8', 'm2_base_b2_l4_d32', 'm3_base_b2_l4_d32'):
        directory = root / 'mamba/corpus' / case
        paths.update(p for p in directory.rglob('*') if p.is_file())
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def check_bits(value, shape):
    require(value.get('shape') == list(shape), 'Unexpected array shape')
    rows = value.get('uint32', [])
    require(len(rows) == shape[0] and all(len(r) == shape[1] for r in rows), 'Truncated raw bits')
    cells = [v for row in rows for v in row]
    require(all(type(v) is int and 0 <= v <= 0xffffffff and (v & 0x7f800000) != 0x7f800000
                for v in cells), 'Invalid/nonfinite float32 bits')
    digest = hashlib.sha256(struct.pack('<' + 'I' * len(cells), *cells)).hexdigest()
    require(value.get('float32_le_sha256') == digest, 'Raw bits/hash disagree')


def check_quality(record, mode, binding_sha):
    require(record.get('status') == 'PASS' and record.get('profile') == 'expanded'
            and record.get('mode') == mode, 'Quality did not pass requested expanded mode')
    require(record.get('k') == 5 and record.get('thresholds') == {
        'trustworthiness': 0.85, 'retention': 0.35, 'minimum_control_margin': 0.15},
        'Quality contract changed')
    rows = record.get('results', [])
    require(len(rows) == len(FIXTURES) and {r.get('profile') for r in rows} == set(FIXTURES),
            'Missing, duplicate or unexpected quality fixture')
    for row in rows:
        require(row.get('passed') is True and row.get('binding_sha256') == binding_sha
                and row.get('binding_mode_code') == MODES[mode] and row.get('fitted_mode') == mode,
                'Quality installed binding/mode admission differs')
        n, q, d = FIXTURES[row['profile']]
        for name, shape in (('training_input', (n, 3)), ('query_input', (q, 3)),
                            ('training_embedding', (n, d)), ('query_embedding', (q, d))):
            check_bits(row[name], shape)
        require(set(row['controls']) == {'query_embedding_permutation', 'training_embedding_permutation'},
                'Missing quality controls')
        for metric, threshold in (('trustworthiness', 0.85), ('retention', 0.35)):
            measured = row['quality'][metric]
            control = max(c[metric] for c in row['controls'].values())
            require(threshold <= measured <= 1 and 0 <= control <= 1
                    and measured - control >= 0.15
                    and abs(row['control_margins'][metric] - (measured - control)) <= 1e-12,
                    'Quality threshold/control failure')


def verify(root, out):
    frozen = json.loads((out / 'qualification-sources.json').read_text())
    require(frozen == sources(root), 'Qualification source changed')
    audit = json.loads((out / 'wheel-audit.json').read_text())
    require(sha(audit['wheel']) == audit['sha256'], 'Wheel changed during qualification')
    expected = {(s, m) for s in SURFACES for m in MODES}
    rows = [line.split('\t') for line in (out / 'results.tsv').read_text().splitlines()]
    require(len(rows) == len(expected) and all(len(r) == 3 for r in rows), 'Incomplete status inventory')
    require({(s, m) for s, m, _ in rows} == expected and all(status == '0' for _, _, status in rows),
            'Missing, duplicate or failed installed job')
    records = {}
    for surface, mode in sorted(expected):
        name = surface + '-' + mode
        require((out / (name + '.log')).is_file(), 'Missing job log')
        record = json.loads((out / (name + '.installed.json')).read_text())
        require(record.get('mode') == mode and record.get('vendor') == audit['qualification_vendor']
                and record.get('wheel_sha256') == audit['sha256'], 'Installed job provenance mismatch')
        package = Path(record['package']).parent
        require(package.is_relative_to(out / 'venv') and 'site-packages' in package.parts,
                'Package outside isolated installation')
        bindings = record['installed_bindings']
        require(set(bindings) == BINDINGS, 'Incomplete binding inventory')
        for row in bindings.values():
            member = str(Path(row['path']).relative_to(package))
            require(row['sha256'] == audit['extension_hashes'].get(member), 'Binding differs from wheel')
            if 'mode_code' in row:
                require(row['mode_code'] == MODES[mode], 'Incorrect mode readback')
        if surface == 'umap-quality':
            check_quality(json.loads((out / (name + '.json')).read_text()), mode,
                          bindings['_mojolearn_metrics']['sha256'])
        records[name] = sha(out / (name + '.installed.json'))
    return {'schema': 'mojolearn.linux.installed-surfaces.v1', 'status': 'PASSED',
            'vendor': audit['qualification_vendor'], 'wheel_sha256': audit['sha256'],
            'source_sha256': audit['source_sha256'], 'installed_records': records,
            'evidence_sha256': {p.name: sha(p) for p in sorted(out.iterdir())
                               if p.is_file() and p.name not in ('qualification.json', 'exit_code')},
            'scope': '24 installed jobs; UMAP six held-out quality fixtures per mode; not universal identity'}


def retained(out):
    """Validate fetched evidence without dereferencing original remote paths."""
    record = json.loads((out / 'qualification.json').read_text())
    require(record.get('schema') == 'mojolearn.linux.installed-surfaces.v1'
            and record.get('status') == 'PASSED', 'Installed qualification did not pass')
    require((out / 'exit_code').read_text().strip() == '0', 'Qualification exit marker failed')
    evidence = record.get('evidence_sha256', {})
    expected = {(s, m) for s in SURFACES for m in MODES}
    required = {'results.tsv', 'wheel-audit.json', 'qualification-sources.json',
                'installed-dependencies.txt', 'dependency-check.log'}
    required.update(s + '-' + m + suffix for s, m in expected
                    for suffix in ('.log', '.installed.json'))
    required.update('umap-quality-' + m + '.json' for m in MODES)
    require(required <= set(evidence), 'Incomplete retained evidence inventory')
    for name, digest in evidence.items():
        require(Path(name).name == name and name not in ('.', '..'), 'Invalid evidence path')
        require(sha(out / name) == digest, 'Retained evidence hash differs: ' + name)
    rows = [line.split('\t') for line in (out / 'results.tsv').read_text().splitlines()]
    require(len(rows) == len(expected) and all(len(r) == 3 for r in rows), 'Incomplete retained statuses')
    require({(s, m) for s, m, _ in rows} == expected and all(r == '0' for _, _, r in rows),
            'Failed/duplicate retained statuses')
    installed = json.loads((out / 'umap-quality-identical.installed.json').read_text())
    require(installed.get('vendor') == record['vendor'] and installed.get('mode') == 'identical'
            and installed.get('wheel_sha256') == record['wheel_sha256'], 'Retained provenance mismatch')
    quality = json.loads((out / 'umap-quality-identical.json').read_text())
    check_quality(quality, 'identical', installed['installed_bindings']['_mojolearn_metrics']['sha256'])
    return record, quality


def compare(left, right):
    a, qa = retained(left)
    b, qb = retained(right)
    require({a['vendor'], b['vendor']} == {'cuda', 'hip'}, 'Comparison requires CUDA and HIP')
    require(a['source_sha256'] == b['source_sha256'], 'Different native build sources')
    require(a['evidence_sha256']['qualification-sources.json'] ==
            b['evidence_sha256']['qualification-sources.json'], 'Different qualification sources')
    ar = {r['profile']: r for r in qa['results']}
    br = {r['profile']: r for r in qb['results']}
    arrays = ('training_input', 'query_input', 'training_embedding', 'query_embedding')
    for name in FIXTURES:
        for field in (*arrays, 'parameters', 'transform_schedule', 'fitted_config', 'fitted_mode'):
            require(ar[name][field] == br[name][field], 'IDENTICAL mismatch: ' + name + '/' + field)
    return {'schema': 'mojolearn.linux.installed-umap-identity.v1', 'status': 'PASSED',
            'source_sha256': a['source_sha256'], 'fixtures': sorted(FIXTURES),
            'compared_arrays': len(FIXTURES) * len(arrays),
            'qualification_sha256': [sha(p / 'qualification.json') for p in (left, right)],
            'wheel_sha256': {r['vendor']: r['wheel_sha256'] for r in (a, b)},
            'scope': 'Six installed IDENTICAL fit/transform fixtures across AMD/NVIDIA; no other mode identity claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('snapshot', 'verify', 'compare'))
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if args.action == 'compare':
        try:
            print(json.dumps(compare(root, out), indent=2))
            return 0
        except (ValueError, OSError, KeyError, TypeError, struct.error) as exc:
            print(json.dumps({'status': 'FAILED', 'reason': str(exc)}))
            return 1
    if args.action == 'snapshot':
        (out / 'qualification-sources.json').write_text(json.dumps(sources(root), indent=2) + '\n')
        (out / 'qualification.json').write_text('{"status": "INCOMPLETE"}\n')
        return 0
    try:
        result = verify(root, out)
    except (ValueError, OSError, KeyError, TypeError, struct.error) as exc:
        result = {'status': 'FAILED', 'reason': str(exc)}
    (out / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
