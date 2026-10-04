#!/usr/bin/env python3
"""Collect and compare experimental PTX evidence without enabling IDENTICAL.

inventory derives one-GPU scope from the existing applicability registry.
collect runs identity_break in this process and witnesses the files it loaded.
check compares retained columns, including exact hashes and structural N/A.
Receipts are evidence from a trusted collector, not cryptographic attestation.
Even full agreement is limited to the observed hardware/driver combinations;
this tool never edits the runtime admission policy or qualifies a release.

On an installed experimental wheel, with the frozen source checkout present:
  MOJOLEARN_CUDA_PATH=ptx-baseline MOJOLEARN_EXPERIMENTAL_PTX=1 \
  MOJOLEARN_NUMERIC_MODE=identical timeout 1200s python tools/nvidia_baseline_qualification.py \
    collect --manifest SET/PTX_BASELINE.json --lanes kmeans --fixtures base --out baseline.json
Run a separate process with the native wheel and without the two PTX variables:
  MOJOLEARN_NUMERIC_MODE=identical timeout 1200s python tools/nvidia_baseline_qualification.py \
    collect --role native-reference --manifest SET/PTX_BASELINE.json \
    --lanes kmeans --fixtures base --out native.json
  python tools/nvidia_baseline_qualification.py check --prototype \
    --manifest SET/PTX_BASELINE.json --out comparison.json baseline.json native.json
Omit --lanes/--fixtures to collect the whole applicable registry. Cold-cache,
warm-cache, fresh-process and additional driver runs should retain separate
receipts. Comparisons use the existing harness's output fingerprints, not a
proof covering untested inputs or every public algorithm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'mojolearn.nvidia-baseline-run.v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harness_digest():
    paths = [ROOT / 'tools/identity_break.py', *sorted((ROOT / 'tools/identity_lanes').glob('*.py'))]
    return hashlib.sha256(json.dumps([(str(p.relative_to(ROOT)), sha(p)) for p in paths]).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def modules():
    import identity_columns
    h = identity_columns.harness()
    v = identity_columns.vref()
    return h, v


def inventory(h):
    import lane_applicability as applicability
    applicability.use_harness(h)
    scopes = applicability.scopes()
    included, excluded = [], {}
    for name in sorted(h.LANES):
        ok, reason = scopes[name].applicable('nvidia-1gpu')
        if ok:
            included.append(name)
        else:
            excluded[name] = reason
    return dict(schema='mojolearn.nvidia-baseline-inventory.v1',
                harness_lanes=len(h.LANES), lanes=included, excluded=excluded,
                fixtures=list(h.FIXTURES), parts=['train', 'infer', 'model', *h.NARROWABLE_PARTS],
                applicability_gaps=list(applicability._SCOPES_DISAGREE),
                scope='single-GPU applicable identity lanes; physical multi-GPU is separate',
                universal_gpu_support=False, release_qualified=False)


def manifest_files(manifest):
    require(manifest.get('schema') == 'mojolearn.ptx-baseline.v1' and manifest.get('code_format') == 'ptx-baseline'
            and manifest.get('vendor') == 'cuda' and manifest.get('target') == 'sm_80',
            'Wrong baseline manifest format or target')
    require(re.fullmatch('[0-9a-f]{40}', str(manifest.get('source_commit', ''))),
            'Missing frozen source commit')
    require(manifest.get('source_dirty') is False and not manifest.get('errors') and manifest.get('experimental') is True
            and manifest.get('identical_qualified') is False, 'Manifest is not a clean experimental baseline')
    files = {}
    modules_count = 0
    for row in manifest.get('files', []):
        name = row.get('file', '')
        require(name and not Path(name).is_absolute() and '..' not in Path(name).parts
                and name not in files, 'Unsafe or duplicate payload path')
        require(re.fullmatch('[0-9a-f]{64}', str(row.get('sha256', ''))), 'Invalid payload hash')
        for module in row.get('ptx_modules', []):
            require(module.get('target') == 'sm_80'
                    and re.fullmatch('[0-9a-f]{64}', str(module.get('sha256', ''))),
                    'Nonbaseline or unwitnessed PTX module')
            modules_count += 1
        files[name] = row
    require(files and modules_count, 'Empty PTX payload')
    return files


def fixture_witnesses(h, fixtures):
    train, heldout = {}, {}
    for name in fixtures:
        x, yc, yr = h.fixture(name)
        train[name] = dict(X=h._h(x), y_clf=h._h(yc), y_reg=h._h(yr))
        heldout[name] = dict(X=h._h(h.heldout(name)))
    return train, heldout


def column_values(column, h, v, lanes, fixtures, witnesses, source):
    reason = v.admit(column, 'retained-column.json', known_lanes=h.LANES)
    require(reason is None, 'Inadmissible column: ' + str(reason))
    require(v.record_device_class(column, 'retained-column.json')[0] == 'nvidia',
            'Column is not an NVIDIA execution')
    require(column.get('complete') is True and column.get('commit') == source,
            'Incomplete or wrong-source column')
    require(type(column.get('repeats')) is int and column['repeats'] >= 2,
            'At least two repeats are required')
    require(column.get('parts_collected') is not None and not column.get('skipped'),
            'Missing modern part declaration or skipped lanes')
    require(not column.get('package', {}).get('bindings_error'), 'Binding provenance failed')
    expected_cells = {lane + '/' + fixture for lane in lanes for fixture in fixtures}
    require(set(column['cells']) == expected_cells, 'Missing or unexpected lane/fixture coverage')
    train, heldout = witnesses
    require(column.get('fixtures') == train and column.get('heldout') == heldout,
            'Fixture or held-out input bytes differ')
    values = {}
    for lane in lanes:
        for revision in ('LANE_REVISIONS', 'BATCH_REVISIONS'):
            expected = getattr(h, revision).get(lane)
            require(expected is None or column.get(revision.lower(), {}).get(lane) == expected,
                    'Stale ' + revision + ': ' + lane)
        parts = ['train', 'infer', 'model', 'batch', *h.PROPERTY_PARTS]
        if lane in h.RLPAIR:
            parts.append('rlpair')
        for fixture in fixtures:
            key = lane + '/' + fixture
            cell = column['cells'][key]
            require(not any(value for name, value in cell.items() if 'error' in name),
                    'Cell reports an error: ' + key)
            for part in parts:
                value = v._part_value(cell, part, min_repeats=2)
                require(value is not None and not value.startswith('n/a:UNDECLARED'),
                        'Unverified part: ' + key + '/' + part)
                if part == 'batch':
                    protocol = dict(alone=h.BATCH_ALONE, split=list(h.BATCH_SPLIT) + ['n'],
                                    prefix='1,7,full-1', enabled=True)
                elif part == 'rlpair':
                    protocol = h._rlpair_protocol()
                elif part in h.PROPERTY_PARTS:
                    protocol = h._part_protocol(part, h.BATCH_ALONE)
                else:
                    protocol = None
                require(protocol is None or column.get(part + '_protocol') == protocol,
                        'Property protocol differs: ' + part)
                values[key + '/' + part] = value
    return values


def validate_receipt(receipt, column, manifest, manifest_hash, files, harness_hash):
    require(receipt.get('schema') == SCHEMA and receipt.get('source_commit') == manifest['source_commit'],
            'Wrong receipt schema or source')
    require(receipt.get('harness_sha256') == harness_hash and receipt.get('exit_code') == 0,
            'Wrong harness or failed execution')
    require(receipt.get('collector') == 'same-process-loaded-bindings-v1', 'Missing runtime collector witness')
    hardware = receipt.get('hardware', {})
    capability = hardware.get('compute_capability')
    require(hardware.get('uuid') and hardware.get('name') and receipt.get('driver_version')
            and isinstance(capability, list) and len(capability) == 2
            and all(type(x) is int and x >= 0 for x in capability) and capability >= [8, 0],
            'Missing hardware/driver evidence or unsupported capability')
    role = receipt.get('role')
    selection = receipt.get('selection', {})
    expected = 'ptx-baseline' if role == 'baseline' else 'native'
    require(role in ('baseline', 'native-reference') and selection.get('selected') == expected
            and selection.get('requested') == expected and selection.get('native_fallback') is False,
            'Native fallback or missing forced selection proof')
    loaded = selection.get('loaded_files', [])
    require(loaded and len({row['module'] for row in loaded}) == len(loaded),
            'Missing or duplicate actual loaded bindings')
    witnessed = {row['module']: row['sha256'] for row in column.get('package', {}).get('bindings', [])}
    require(witnessed and witnessed == {row['module']: row['sha256'] for row in loaded},
            'Column bindings differ from runtime receipt')
    for row in loaded:
        require(re.fullmatch('[0-9a-f]{64}', str(row.get('sha256', ''))), 'Invalid loaded hash')
        if role == 'baseline' and not row['module'].endswith('_host'):
            entry = files.get(row.get('file'))
            require(entry is not None and entry['sha256'] == row['sha256']
                    and entry.get('numeric_mode') == 'identical', 'Loaded bytes are not IDENTICAL baseline payload')
    if role == 'baseline':
        require(selection.get('manifest_sha256') == manifest_hash, 'Different baseline payload manifest')
        require(any(not row['module'].endswith('_host') for row in loaded), 'No GPU binding executed')
        runtime = selection.get('runtime_receipt', {})
        require(runtime.get('schema') == 'mojolearn.ptx-baseline-selection.v1'
                and runtime.get('source_commit') == manifest['source_commit']
                and runtime.get('manifest_sha256') == manifest_hash
                and runtime.get('requested') == 'ptx-baseline' and runtime.get('selected') == 'ptx-baseline'
                and runtime.get('native_fallback') is False, 'Missing forced-path runtime receipt')
        require({(row['file'], row['sha256']) for row in runtime.get('loaded_files', [])}
                == {(row['file'], row['sha256']) for row in loaded if not row['module'].endswith('_host')},
                'Runtime path witness differs from loaded payload')
    return (hardware['uuid'], tuple(capability), receipt['driver_version'])


def check(manifest_path, receipt_paths, *, prototype=False):
    manifest_path = Path(manifest_path)
    manifest = read(manifest_path)
    files = manifest_files(manifest)
    # Validate the actual transported artifact, not only self-reported hashes.
    for name, entry in files.items():
        require(sha(manifest_path.parent / name) == entry['sha256'], 'Payload bytes changed: ' + name)
    h, v = modules()
    scope = inventory(h)
    require(prototype or not scope['applicability_gaps'], 'Unresolved applicability metadata; full comparison refused')
    receipts = [(Path(p), read(p)) for p in receipt_paths]
    require(receipts, 'No receipts')
    lanes, fixtures = receipts[0][1].get('lanes', []), receipts[0][1].get('fixtures', [])
    require(lanes and fixtures and len(set(lanes)) == len(lanes) and len(set(fixtures)) == len(fixtures),
            'Empty or duplicate scope')
    require(set(lanes) <= set(scope['lanes']) and set(fixtures) <= set(scope['fixtures']),
            'Unknown or inapplicable scope')
    require(prototype or (set(lanes) == set(scope['lanes']) and set(fixtures) == set(scope['fixtures'])),
            'Missing full applicable coverage; use --prototype for explicitly limited evidence')
    witnesses = fixture_witnesses(h, fixtures)
    compared, baseline_configs, native_configs = None, set(), set()
    inputs = []
    for path, receipt in receipts:
        require(receipt.get('lanes') == lanes and receipt.get('fixtures') == fixtures, 'Different report scope')
        column_path = path.parent / receipt['column_file']
        require(sha(column_path) == receipt.get('column_sha256'), 'Retained column hash differs')
        column = read(column_path)
        config = validate_receipt(receipt, column, manifest, sha(manifest_path), files,
                                  harness_digest())
        values = column_values(column, h, v, lanes, fixtures, witnesses, manifest['source_commit'])
        require(compared is None or values == compared, 'Bitwise or structural result mismatch')
        compared = values
        (baseline_configs if receipt['role'] == 'baseline' else native_configs).add(config)
        inputs.append(dict(file=str(path), sha256=sha(path), column_sha256=sha(column_path)))
    require(baseline_configs and native_configs, 'Baseline and native reference runs are both required')
    require(prototype or len({c[1] for c in baseline_configs}) >= 2,
            'Full comparison requires distinct GPU compute capabilities')
    return dict(schema='mojolearn.nvidia-baseline-comparison.v1',
                status='PROTOTYPE_AGREEMENT' if prototype else 'OBSERVED_CONFIGURATION_AGREEMENT',
                source_commit=manifest['source_commit'], manifest_sha256=sha(manifest_path),
                lanes=lanes, fixtures=fixtures, compared_parts=len(compared),
                baseline_configurations=sorted(baseline_configs), native_configurations=sorted(native_configs),
                full_applicable_single_gpu_coverage=not prototype, excluded=scope['excluded'],
                universal_gpu_support=False, future_drivers_qualified=False,
                identical_qualified=False, release_qualified=False, inputs=inputs)


def collect(args):
    # Import the real package before inventory tools create a stub namespace.
    import mojolearn as ml
    from mojolearn import _backend
    from mojolearn._verify import binding_artifacts
    manifest_path = args.manifest.resolve()
    manifest = read(manifest_path)
    files = manifest_files(manifest)
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == manifest['source_commit'], 'Collector checkout is not the frozen payload source')
    require(subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', 'HEAD']).returncode == 0,
            'Collector checkout has tracked source changes')
    require(ml.vendor() == 'cuda' and ml.numeric_mode() == 'identical', 'CUDA IDENTICAL is required')
    expected = 'ptx-baseline' if args.role == 'baseline' else 'native'
    plugin = _backend.gpu_plugin() or {}
    require(plugin.get('code_format') == expected, 'Loader did not select requested code format')
    # An ambiguous physical-device mapping cannot become a qualification receipt.
    smi = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,compute_cap,driver_version',
                                  '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    require(len(smi) == 1, 'Collector currently requires exactly one physical NVIDIA GPU')
    uuid, name, cap, driver = [x.strip() for x in smi[0].split(',')]
    h, _ = modules()
    scope = inventory(h)
    lanes = args.lanes.split(',') if args.lanes else scope['lanes']
    fixtures = args.fixtures.split(',') if args.fixtures else scope['fixtures']
    require(set(lanes) <= set(scope['lanes']) and set(fixtures) <= set(scope['fixtures']),
            'Unknown or inapplicable requested scope')
    out = args.out.resolve()
    column = out.with_suffix('.column.json')
    require(not out.exists() and not column.exists(), 'Refusing to overwrite retained evidence')
    out.parent.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv
    try:
        sys.argv = ['identity_break.py', '--json', str(column), '--lanes', ','.join(lanes),
                    '--fixtures', ','.join(fixtures), '--repeats', '2', '--require-backend', 'cuda',
                    '--fail-on-refused']
        code = h.main()
    finally:
        sys.argv = old_argv
    require(code in (None, 0), 'Identity run failed; raw column retained')
    loaded = []
    runtime = None
    for row in binding_artifacts():
        item = dict(module=row['module'], sha256=row['sha256'], file=row['file'])
        if args.role == 'baseline' and not row['module'].endswith('_host'):
            relative = str(Path(row['file']).resolve().relative_to(manifest_path.parent))
            require(relative in files and files[relative]['sha256'] == row['sha256'],
                    'A loaded GPU binding is outside the baseline payload')
            item['file'] = relative
        loaded.append(item)
    if args.role == 'baseline':
        runtime = _backend.baseline_selection_receipt()
        require(runtime.get('schema') == 'mojolearn.ptx-baseline-selection.v1'
                and runtime.get('requested') == expected and runtime.get('selected') == expected
                and runtime.get('native_fallback') is False
                and runtime.get('manifest_sha256') == sha(manifest_path)
                and runtime.get('source_commit') == manifest['source_commit'],
                'Missing or inconsistent forced-baseline runtime receipt')
        require({(x['file'], x['sha256']) for x in runtime.get('loaded_files', [])}
                == {(x['file'], x['sha256']) for x in loaded if not x['module'].endswith('_host')},
                'Runtime selection receipt differs from actual loaded bindings')
    receipt = dict(schema=SCHEMA, collector='same-process-loaded-bindings-v1', role=args.role,
                   source_commit=manifest['source_commit'], harness_sha256=harness_digest(),
                   hardware=dict(uuid=uuid, name=name, compute_capability=[int(x) for x in cap.split('.')]),
                   driver_version=driver, exit_code=0, lanes=lanes, fixtures=fixtures,
                   column_file=column.name, column_sha256=sha(column),
                   selection=dict(requested=expected, selected=expected, native_fallback=False,
                                  manifest_sha256=sha(manifest_path), loaded_files=loaded,
                                  runtime_receipt=runtime))
    validate_receipt(receipt, read(column), manifest, sha(manifest_path), files, receipt['harness_sha256'])
    write(out, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('inventory')
    collect_parser = commands.add_parser('collect')
    collect_parser.add_argument('--manifest', type=Path, required=True)
    collect_parser.add_argument('--out', type=Path, required=True)
    collect_parser.add_argument('--role', choices=['baseline', 'native-reference'], default='baseline')
    collect_parser.add_argument('--lanes', help='comma-separated prototype scope; default all applicable')
    collect_parser.add_argument('--fixtures', help='comma-separated prototype scope; default all nine')
    checker = commands.add_parser('check')
    checker.add_argument('--manifest', type=Path, required=True)
    checker.add_argument('--prototype', action='store_true')
    checker.add_argument('--out', type=Path, required=True)
    checker.add_argument('receipts', nargs='+', type=Path)
    args = parser.parse_args()
    if args.command == 'inventory':
        result = inventory(modules()[0])
    elif args.command == 'collect':
        result = collect(args)
    else:
        result = check(args.manifest, args.receipts, prototype=args.prototype)
        write(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
