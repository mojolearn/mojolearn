#!/usr/bin/env python3
"""Collect and compare experimental PTX evidence without enabling IDENTICAL.

inventory derives one-GPU scope from the existing applicability registry.
collect runs identity_break in this process and witnesses the files it loaded.
check compares retained columns, including exact hashes and structural N/A.
Receipts are evidence from a trusted collector, not cryptographic attestation.
Even full agreement is limited to the observed hardware/driver combinations;
this tool never edits the runtime admission policy or qualifies a release.
Both roles require the installed core COMMIT witness. Native references also
require the installed release payload inventories and exact loaded-file hashes;
a new harness checkout cannot confer its source SHA on an old native wheel.

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
The --manifest path may point to an external retained build artifact; loaded
files are resolved against the installed loader's validated baseline root.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

# A staged copy of this checker (newer tooling than the frozen payload source)
# reads the harness of the checkout named here; collectors never set it.
ROOT = Path(os.environ.get('MOJOLEARN_QUALIFICATION_ROOT') or Path(__file__).resolve().parents[1]).resolve()
SCHEMA = 'mojolearn.nvidia-baseline-run.v1'
UNDECLARED = 'n/a:UNDECLARED'
# The harness declares no batch probe for these lanes (no `_batch_decl` entry
# in tools/identity_break.py's BATCH), so `_probe_batch` records UNDECLARED on
# every fixture and device. The canonical verifier (`_part_value`, with
# `_SKIPPED_NA`) reads that as no usable value, the same as a skipped part. Declaring the probes would
# change the harness digest bound by retained receipts and reference columns,
# so the checker pins the exclusion instead: each (lane, part) here MUST read
# exactly UNDECLARED in every compared column, is reported as excluded, and
# never enters the compared values. Any other undeclared part still fails.
UNDECLARED_EXCLUSIONS = (('gbdt-class-weights', 'batch'), ('gbdt-multiclass-offgrid', 'batch'))
# Native kernels ship for sm_89 and sm_90/sm_90a. A set runs on its own major
# at its minor or later, so a device below every floor of its major (an A100,
# capability 8.0) has no native payload: the only place the fallback triggers.
NATIVE_FLOORS = ((8, 9), (9, 0))
NATIVE_ABSENT_RULE = ('forced-PTX column equals every native reference column collected on the '
                      'natively supported devices from the same source, cell for cell')


def native_supported(capability):
    major, minor = capability
    return any(major == floor[0] and minor >= floor[1] for floor in NATIVE_FLOORS)


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


def document_sha(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def installed_source_evidence(package_dir, backend, role, source, loaded):
    """Read actual installed core/payload provenance, never infer it from Git."""
    package_dir = Path(package_dir).resolve()
    commit_text = (package_dir / 'identity_columns/COMMIT').read_text()
    require(commit_text.strip() == source, 'Installed core source differs from frozen source')
    evidence = dict(core_commit=commit_text,
                    core_commit_sha256=hashlib.sha256(commit_text.encode()).hexdigest(), inventories=[])
    if role == 'native-reference':
        plugin = backend.gpu_plugin() or {}
        for name in ['mojolearn', *plugin.get('payloads', [])]:
            dist = backend._find_distribution(name, [str(package_dir.parent)])
            require(dist is not None, 'Missing installed distribution: ' + name)
            raw = dist.read_text('LINUX_PAYLOAD.json')
            require(raw is not None, 'Native reference requires installed LINUX_PAYLOAD provenance: ' + name)
            doc = json.loads(raw)
            require(doc.get('schema') == 'mojolearn.linux-payload.v1' and doc.get('source_commit') == source,
                    'Native payload inventory source differs: ' + name)
            evidence['inventories'].append(dict(distribution=name, document=doc,
                document_sha256=document_sha(doc), installed_text_sha256=hashlib.sha256(raw.encode()).hexdigest()))
        for row in loaded:
            row['installed_member'] = Path(row['file']).resolve().relative_to(package_dir.parent).as_posix()
    validate_installed_evidence(evidence, role, source, loaded)
    return evidence


def validate_installed_evidence(evidence, role, source, loaded):
    text = evidence.get('core_commit', '')
    require(isinstance(text, str) and text.strip() == source
            and evidence.get('core_commit_sha256') == hashlib.sha256(text.encode()).hexdigest(),
            'Missing or stale installed core source witness')
    if role != 'native-reference':
        return
    docs = evidence.get('inventories', [])
    require(docs and len({r['distribution'] for r in docs}) == len(docs),
            'Missing or duplicate installed native inventories')
    owned = {}
    for row in docs:
        doc = row.get('document', {})
        require(doc.get('schema') == 'mojolearn.linux-payload.v1' and doc.get('source_commit') == source
                and row.get('document_sha256') == document_sha(doc)
                and re.fullmatch('[0-9a-f]{64}', str(row.get('installed_text_sha256', ''))),
                'Invalid installed native inventory source or document hash')
        split = doc.get('split', {})
        require(split.get('distribution') == row['distribution'], 'Native inventory ownership differs')
        hashes = dict(doc.get('extensions', {}))
        hashes.update({r['archive_path']: r['sha256'] for r in doc.get('host_native', {}).values()})
        for member in split.get('native_members', []):
            if member in hashes:
                require(member not in owned, 'Two native inventories own the same loaded path')
                owned[member] = hashes[member]
    for row in loaded:
        member = row.get('installed_member', '')
        require(member.startswith(('mojolearn/cuda_native/', 'mojolearn/host/'))
                and owned.get(member) == row['sha256'],
                'Loaded native bytes differ from installed payload inventory')


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


def undeclared_scope(lanes, fixtures):
    """Pinned exclusions inside this scope, in report form."""
    return [dict(lane=lane, part=part, fixtures=list(fixtures), value=UNDECLARED)
            for lane, part in UNDECLARED_EXCLUSIONS if lane in lanes]


def column_values(column, h, v, lanes, fixtures, witnesses, source, excluded=None):
    """Compared values of one column. Pinned undeclared parts go to `excluded`."""
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
                if (lane, part) in UNDECLARED_EXCLUSIONS:
                    # An excluded part is absent evidence, never an equal hash.
                    # A value here means the harness changed: review the pin.
                    # The canonical verifier yields no usable value for an
                    # undeclared part, so read the retained repeats directly.
                    require(value is None and cell.get(part) == [UNDECLARED] * column['repeats']
                            and cell.get(part + '_verdict') == 'N/A' and excluded is not None,
                            'Pinned undeclared exclusion carries another value: ' + key + '/' + part)
                    excluded.add(key + '/' + part)
                    continue
                require(value is not None and not value.startswith(UNDECLARED),
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
            # Aggregate hashes do not replace retained constituent witnesses.
            # Compare these for every fixture, including fixtures absent from
            # the canonical three-fixture Apple/AMD columns.
            components = cell.get('parts')
            require(isinstance(components, list) and len(components) == column['repeats']
                    and components and all(item == components[0] for item in components),
                    'Missing or unstable components: ' + key)
            component = components[0]
            require(isinstance(component, dict) and component,
                    'Empty component evidence: ' + key)
            for name, digest in component.items():
                require(isinstance(name, str) and name and isinstance(digest, str)
                        and re.fullmatch('[0-9a-f]{16}', digest),
                        'Invalid component hash: ' + key + '/' + str(name))
                values[key + '/parts/' + name] = digest
            reloads = cell.get('reload')
            if reloads is None:
                model = values[key + '/model']
                require(model.startswith('n/a:'), 'Missing reload for saved model: ' + key)
                values[key + '/reload'] = model
            else:
                require(isinstance(reloads, list) and len(reloads) == column['repeats']
                        and reloads and all(item == reloads[0] for item in reloads)
                        and isinstance(reloads[0], str) and re.fullmatch('[0-9a-f]{16}', reloads[0]),
                        'Missing or unstable reload hash: ' + key)
                require(reloads[0] == values[key + '/infer'], 'Reload differs from inference: ' + key)
                values[key + '/reload'] = reloads[0]
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
    validate_installed_evidence(receipt.get('installed_source', {}), role, manifest['source_commit'], loaded)
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
    undeclared = undeclared_scope(lanes, fixtures)
    expected_excluded = {row['lane'] + '/' + fixture + '/' + row['part']
                         for row in undeclared for fixture in fixtures}
    compared, baseline_configs, native_configs, absent_configs = None, set(), set(), set()
    inputs = []
    for path, receipt in receipts:
        require(receipt.get('lanes') == lanes and receipt.get('fixtures') == fixtures, 'Different report scope')
        column_path = path.parent / receipt['column_file']
        require(sha(column_path) == receipt.get('column_sha256'), 'Retained column hash differs')
        column = read(column_path)
        config = validate_receipt(receipt, column, manifest, sha(manifest_path), files,
                                  harness_digest())
        excluded = set()
        values = column_values(column, h, v, lanes, fixtures, witnesses, manifest['source_commit'], excluded)
        require(excluded == expected_excluded and not excluded & set(values),
                'Undeclared exclusions differ from the pinned list')
        if compared is not None and values != compared:
            differing = sorted(key for key in compared.keys() | values.keys()
                               if compared.get(key) != values.get(key))
            # Bound console output while retaining the complete input columns.
            # Include the exact lane/fixture/part so numerical triage does not
            # require printing thousands of successful cells.
            examples = [dict(part=key, reference=compared.get(key), actual=values.get(key))
                        for key in differing[:8]]
            raise ValueError('Bitwise or structural result mismatch: '
                             + json.dumps(dict(receipt=str(path), count=len(differing),
                                               examples=examples), sort_keys=True))
        compared = values
        if receipt['role'] != 'baseline':
            require(native_supported(config[1]), 'Native reference from a device without a native payload')
            native_configs.add(config)
        elif native_supported(config[1]):
            baseline_configs.add(config)
        else:
            # No native column can exist on this device. Its PTX column has
            # just been compared with every other column, native ones included.
            absent_configs.add(config)
        inputs.append(dict(file=str(path), sha256=sha(path), column_sha256=sha(column_path)))
    require((baseline_configs or absent_configs) and native_configs,
            'Baseline and native reference runs are both required')
    # Only natively supported devices count here; a native-absent device never
    # stands in for one of the two capabilities.
    require(prototype or len({c[1] for c in baseline_configs}) >= 2,
            'Full comparison requires distinct GPU compute capabilities')
    native_capabilities = sorted({c[1] for c in native_configs})
    require(prototype or not absent_configs
            or (len(native_capabilities) >= 2 and {c[1] for c in baseline_configs} <= set(native_capabilities)),
            'Native-absent comparison requires a native reference on every natively supported capability')
    return dict(schema='mojolearn.nvidia-baseline-comparison.v1',
                status='PROTOTYPE_AGREEMENT' if prototype else 'OBSERVED_CONFIGURATION_AGREEMENT',
                source_commit=manifest['source_commit'], manifest_sha256=sha(manifest_path),
                lanes=lanes, fixtures=fixtures, compared_parts=len(compared),
                undeclared_exclusions=undeclared, excluded_parts=len(expected_excluded),
                baseline_configurations=sorted(baseline_configs | absent_configs),
                native_configurations=sorted(native_configs),
                native_absent_configurations=sorted(absent_configs),
                native_absent_rule=NATIVE_ABSENT_RULE if absent_configs else None,
                native_reference_capabilities=native_capabilities,
                full_applicable_single_gpu_coverage=not prototype, excluded=scope['excluded'],
                universal_gpu_support=False, future_drivers_qualified=False,
                identical_qualified=False, release_qualified=False, inputs=inputs)


def collect(args):
    # Import the real package before inventory tools create a stub namespace.
    import mojolearn as ml
    from mojolearn import _backend
    from mojolearn._verify import binding_artifacts
    require(not os.environ.get('MOJOLEARN_QUALIFICATION_ROOT'),
            'Collectors run from the frozen payload checkout only')
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
    installed_source_evidence(Path(ml.__file__).resolve().parent, _backend,
                              args.role, manifest['source_commit'], [])
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
            # --manifest can be the external build artifact. Resolve actual
            # imports against the root validated by the installed loader.
            baseline_root = Path(_backend._BASELINE_ROOT).resolve()
            relative = str(Path(row['file']).resolve().relative_to(baseline_root))
            require(relative in files and files[relative]['sha256'] == row['sha256'],
                    'A loaded GPU binding is outside the baseline payload')
            item['file'] = relative
        loaded.append(item)
    installation = installed_source_evidence(Path(ml.__file__).resolve().parent, _backend,
                                            args.role, manifest['source_commit'], loaded)
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
                   installed_source=installation,
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
