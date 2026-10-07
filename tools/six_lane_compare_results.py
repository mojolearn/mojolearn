#!/usr/bin/env python3
"""Compare retained scored receipts; emit pending inputs for the existing board.

This is metadata-only orchestration. It never imports an estimator, opens tensor
payloads, compiles, measures, promotes a default, or invokes a board writer.
Each IDENTICAL arm is compared across routes separately; A need not equal B.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from six_lane_timing import scored  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'mojolearn.six-lane-comparison-input/1'
COLUMNS = {'nvidia-native': 'nvidia', 'nvidia-ptx': 'nvidia',
           'amd': 'amd', 'apple': 'apple', 'host': 'host'}
# IDENTICAL identity is decided by the two GPU vendors (owner, 2026-10-07):
# NVIDIA native and AMD must agree. PTX, Apple and host columns are compared
# when supplied but are never required and never make a case INCOMPLETE.
REQUIRED_IDENTICAL = ('nvidia-native', 'amd')
SCOPE = ('source_sha', 'workload_id', 'dataset_sha256', 'dataset_version',
         'dataset_split', 'seed', 'mode', 'dimensions', 'estimator_settings',
         'harness_sha256', 'timed_boundary')
# Kernel/runtime scope for --accept-kernel-equivalent-sources: a column whose
# attempt ran a different commit is accepted when `git diff --name-only` between
# the pinned commit and that commit touches none of these paths, so the compiled
# kernels and the runtime are the same source (the rule lq uses for ID checks).
# The full changed-file list is recorded in the report either way.
KERNEL_SCOPE = ('*.mojo', 'bindings/', 'python/mojolearn/')
ACCEPT_KERNEL_EQUIVALENT = False
# These choose a backend/code container, never an experiment's arithmetic.
# Exclusions require explicit per-column values in the comparison manifest.
TRANSPORT_ENV = {'MOJOLEARN_VENDOR', 'MOJOLEARN_CUDA_CODE_FORMAT'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def string_list(value):
    return (isinstance(value, list) and bool(value) and all(nonempty(v) for v in value)
            and len(value) == len(set(value)))


def read(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    canonical(value)  # Refuse NaN/Infinity, including in nested metadata.
    return value, hashlib.sha256(raw).hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def configuration(value, transport=None):
    require(isinstance(value, dict) and set(value) == {'defines', 'environment', 'runtime'},
            'configuration requires exactly defines/environment/runtime')
    defines = value['defines']
    require(isinstance(defines, list) and all(nonempty(v) for v in defines), 'invalid defines')
    require(len({v.split('=', 1)[0] for v in defines}) == len(defines), 'duplicate defines')
    env = value['environment']
    require(isinstance(env, dict) and all(nonempty(k) and isinstance(v, str) for k, v in env.items()),
            'invalid configuration environment')
    require(isinstance(value['runtime'], dict), 'invalid runtime controls')
    transport = transport or {}
    require(isinstance(transport, dict) and set(transport) <= TRANSPORT_ENV,
            'only explicit backend/code-format environment values may be excluded')
    require(all(k in env and env[k] == v for k, v in transport.items()),
            'declared transport environment differs from receipt')
    return dict(defines=sorted(defines), environment={k: v for k, v in env.items() if k not in transport},
                runtime=value['runtime'])


def expected_scope(case):
    expected = dict(case.get('expected', {}))
    for key in ('source_sha', 'workload_id', 'mode'):
        require(key in case, 'case missing ' + key)
        require(key not in expected or expected[key] == case[key], 'conflicting expected ' + key)
        expected[key] = case[key]
    require(all(key in expected for key in SCOPE), 'incomplete pinned workload scope')
    require(sha(expected['source_sha'], 40), 'source_sha must be a full commit')
    for key in ('dataset_sha256', 'harness_sha256'):
        require(sha(expected[key]), 'invalid ' + key)
    for key in ('workload_id', 'dataset_version', 'timed_boundary'):
        require(nonempty(expected[key]), 'missing ' + key)
    split = expected['dataset_split']
    require(nonempty(split) or (isinstance(split, dict) and bool(split)),
            'missing dataset_split label or structured split declaration')
    require(expected['mode'] in ('identical', 'fast'), 'unsupported mode')
    require(isinstance(expected['dimensions'], dict) and bool(expected['dimensions']), 'missing shapes')
    require(all(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape)
                for shape in expected['dimensions'].values()), 'invalid shapes')
    require(isinstance(expected['estimator_settings'], dict) and bool(expected['estimator_settings']),
            'missing estimator settings')
    configs = expected.get('configurations', {})
    require(isinstance(configs, dict) and set(configs) == {'A', 'B'}, 'pin both logical configurations')
    expected['configurations'] = {arm: configuration(configs[arm]) for arm in ('A', 'B')}
    expected.setdefault('repeated_operations', 0)
    require(type(expected['repeated_operations']) is int and expected['repeated_operations'] >= 0,
            'invalid repeated operation count')
    paths = expected.get('capture_paths', {})
    require(isinstance(paths, dict) and string_list(paths.get('outputs')),
            'pin nonempty complete output path set')
    # Model state is compared when both columns captured it completely; the
    # output bits alone decide identity when a complete typed state export is
    # absent (owner, 2026-10-07). An inconsistent declaration is still refused.
    require(paths.get('model_state') is None or string_list(paths.get('model_state')),
            'model-state path set must be nonempty when declared')
    return expected


def capture_signature(capture, paths, label):
    require(isinstance(capture, dict), label + ': missing capture')
    require(capture.get('status') == 'CAPTURED', label + ': not CAPTURED')
    require(capture.get('completeness') == 'complete_declared_scope', label + ': unqualified scope')
    require(capture.get('missing_state') == [], label + ': missing/undeclared state')
    require(sha(capture.get('sha256')), label + ': invalid hash')
    require(nonempty(capture.get('scope')) and nonempty(capture.get('encoding')), label + ': missing encoding/scope')
    manifest = capture.get('manifest')
    require(isinstance(manifest, list) and bool(manifest), label + ': missing typed manifest')
    normalized = []
    for row in manifest:
        require(isinstance(row, dict), label + ': invalid manifest row')
        require(all(nonempty(row.get(k)) for k in ('path', 'dtype', 'encoding')), label + ': untyped leaf')
        require(isinstance(row.get('shape'), list) and all(type(n) is int and n >= 0 for n in row['shape']),
                label + ': invalid leaf shape')
        require(row['dtype'] != 'object' and not re.fullmatch(r'[|<>=]?O[0-9]*', row['dtype']),
                label + ': object state cannot establish identity')
        leaf = {k: row[k] for k in ('path', 'dtype', 'encoding', 'shape')}
        if 'bytes' in row:
            require(type(row['bytes']) is int and row['bytes'] >= 0, label + ': invalid byte count')
            leaf['bytes'] = row['bytes']
        normalized.append(leaf)
    actual = [row['path'] for row in normalized]
    require(len(set(actual)) == len(actual) and set(actual) == set(paths), label + ': captured path set differs')
    return dict(sha256=capture['sha256'], scope=capture['scope'], encoding=capture['encoding'],
                manifest=sorted(normalized, key=lambda row: row['path']))


def observed_outputs(run, receipt, case, expected, column, transport):
    """Report literal scored output hashes; never substitute for full identity."""
    data = run.get('result', {})
    try:
        require(receipt.get('status') == 'MEASURED_FULL' and not receipt.get('error'), 'attempt incomplete')
        require(type(run.get('returncode')) is int and run['returncode'] == 0
                and not run.get('error') and run.get('excluded') is False, 'scored process failed/excluded')
        require(data.get('schema') == 'mojolearn.full-ab-result/1' and data.get('status') == 'PASS'
                and not data.get('error') and not data.get('errors'), 'scored result failed')
        require(data.get('phase') == 'scored' and data.get('arm') == run['arm']
                and data.get('vendor') == COLUMNS[column], 'arm/phase/vendor differs')
        require(data.get('sample_counts') == {'excluded_warmups': 0, 'scored': 1}
                and data.get('full_dataset_coverage') is True, 'sample/full scope differs')
        require(receipt.get('source_sha') == data.get('source_sha') and sha(data.get('source_sha'), 40),
                'original source attribution differs')
        require(receipt.get('workload', {}).get('master_selection', {}).get('id') == case['configuration_id']
                and data.get('implementation_ids') == case['implementation_ids'], 'configuration attribution differs')
        for key in SCOPE:
            if key not in ('source_sha', 'harness_sha256'):
                require(key in data and canonical(data[key]) == canonical(expected[key]), 'scope differs: ' + key)
        require(configuration(data.get('configuration'), transport) == expected['configurations'][run['arm']],
                'logical configuration differs')
        outputs = capture_signature(data.get('outputs'), expected['capture_paths']['outputs'], 'outputs')
        require(data.get('output_sha256') == outputs['sha256'], 'output aliases differ')
        return dict(status='OBSERVED_ONLY', outputs=outputs, source_sha=data['source_sha'],
                    harness_sha256=data.get('harness_sha256'), qualified_identity=False)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return dict(status='INCOMPLETE', issues=[str(exc)], qualified_identity=False)


def check_scored(run, receipt, case, expected, column, transport, equivalent_source=None):
    issues = []
    data = run.get('result')
    if not isinstance(data, dict):
        return dict(status='INCOMPLETE', issues=['missing embedded scored result'])
    try:
        require(receipt.get('status') == 'MEASURED_FULL', 'attempt failed or is unfinished')
        require(not receipt.get('error'), 'receipt records failure')
        require(type(run.get('returncode')) is int and run['returncode'] == 0 and not run.get('error'),
                'scored process failed or validation failed')
        require(run.get('excluded') is False, 'scored sample is excluded or exclusion status missing')
        require(data.get('schema') == 'mojolearn.full-ab-result/1' and data.get('status') == 'PASS',
                'unsupported or failed scored result')
        require(not data.get('error') and not data.get('errors'), 'scored result records failure')
        require(data.get('phase') == 'scored' and data.get('arm') == run['arm'], 'arm/phase mismatch')
        require(data.get('vendor') == COLUMNS[column], 'column vendor mismatch')
        require(data.get('sample_counts') == {'excluded_warmups': 0, 'scored': 1}, 'scored sample counts differ')
        require(data.get('full_dataset_coverage') is True, 'full dataset coverage missing')
        require(data.get('hashing_outside_timing') is True, 'hash/report timing exclusion missing')
        source = equivalent_source or expected['source_sha']
        require(receipt.get('source_sha') == source, 'attempt source differs')
        require(receipt.get('mode') == expected['mode'], 'attempt mode differs')
        job = receipt.get('workload', {})
        require(isinstance(job, dict), 'missing frozen job')
        require(job.get('master_selection', {}).get('id') == case['configuration_id'],
                'frozen configuration ID differs')
        for key in SCOPE:
            require(key in data and canonical(data[key]) == canonical(source if key == 'source_sha' else expected[key]), 'scope differs: ' + key)
        require(data.get('implementation_ids') == case['implementation_ids'], 'implementation attribution differs')
        require(configuration(data.get('configuration'), transport) == expected['configurations'][run['arm']],
                'logical arm configuration differs')
        for key in ('hardware', 'compiler', 'resource_policy'):
            require(bool(data.get(key)), 'missing provenance: ' + key)
        require(isinstance(data.get('thread_environment'), dict), 'missing worker thread environment')
        require(positive(data.get('timings', {}).get('full_operation_seconds')), 'missing full-operation timing')
        artifacts = data.get('loaded_artifacts')
        require(isinstance(artifacts, dict) and bool(artifacts)
                and all(nonempty(k) and sha(v) for k, v in artifacts.items()), 'missing loaded artifacts')
        declared = job.get('artifact_provenance', {}).get(run['arm'])
        require(isinstance(declared, list) and bool(declared), 'missing artifact source/target provenance')
        for artifact in declared:
            require(isinstance(artifact, dict) and sha(artifact.get('numerical_source_sha'), 40)
                    and artifact.get('compiler') and artifact.get('target')
                    and isinstance(artifact.get('defines'), list) and nonempty(artifact.get('path'))
                    and sha(artifact.get('sha256')), 'incomplete artifact source/target provenance')
        require(len({a['path'] for a in declared}) == len(declared)
                and {a['path']: a['sha256'] for a in declared} == artifacts,
                'loaded artifact declarations differ')
        names = ('outputs',)
        if case['mode'] == 'identical' and state_declared(data, expected):
            names = ('outputs', 'model_state')
        for name in names:
            capture_signature(data.get(name), expected['capture_paths'][name], name)
        require(data.get('output_sha256') == data['outputs']['sha256'], 'output hash aliases differ')
        repeated = data.get('repeated_use')
        require(isinstance(repeated, list) and len(repeated) == expected['repeated_operations'],
                'repeated operation scope differs or is undeclared')
        for index, item in enumerate(repeated):
            require(isinstance(item, dict) and type(item.get('index')) is int and item['index'] == index,
                    'repeated operation index differs')
            for name in names:
                capture_signature(item.get(name), expected['capture_paths'][name], 'repeated ' + name)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        issues.append(str(exc))
    return dict(status='READY' if not issues else 'INCOMPLETE', issues=issues,
                result=data, log=run.get('log'), output=run.get('output'),
                result_sha256=run.get('result_sha256'), returncode=run.get('returncode'),
                observed_outputs=observed_outputs(run, receipt, case, expected, column, transport))


def receipt_summary(value):
    if not isinstance(value, dict):
        return dict(status='INCOMPLETE', error='receipt is not an object')
    return {key: value.get(key) for key in ('status', 'source_sha', 'mode', 'key', 'receipt',
                                           'previous_receipt', 'error', 'started', 'finished')}


def read_column(spec, base, case, expected, column):
    if isinstance(spec, str):
        spec = {'receipt': spec}
    report = dict(status='INCOMPLETE', arms={}, history=[])
    try:
        require(isinstance(spec, dict) and nonempty(spec.get('receipt')), 'missing column receipt')
        path = (base / spec['receipt']).resolve()
        report['path'] = str(path)
        receipt, report['sha256'] = read(path)
        require(isinstance(receipt, dict), 'receipt is not an object')
        report['attempt'] = receipt_summary(receipt)
        job = receipt.get('workload', {})
        require(isinstance(job, dict), 'invalid workload object')
        report['artifact_provenance'] = job.get('artifact_provenance')
        require(isinstance(receipt.get('runs'), list), 'missing attempt runs')
        report['runs'] = [{k: r.get(k) for k in ('phase', 'arm', 'excluded', 'returncode', 'error', 'log', 'output')}
                          for r in receipt.get('runs', []) if isinstance(r, dict)]
        transport = spec.get('transport_environment', {})
        require(isinstance(transport, dict) and set(transport) <= {'A', 'B'}, 'invalid transport mapping')
        equivalent_source = None
        attempt_source = receipt.get('source_sha')
        if (ACCEPT_KERNEL_EQUIVALENT and sha(attempt_source, 40) and attempt_source != expected['source_sha']
                and column in REQUIRED_IDENTICAL):
            report['source_equivalence'] = kernel_equivalence(expected['source_sha'], attempt_source)
            if report['source_equivalence']['status'] == 'KERNEL_EQUIVALENT':
                equivalent_source = attempt_source
        if case.get('freeze_equivalence') and column in ('nvidia-native', 'amd'):
            try:
                from six_lane_freeze_equivalence import validate
                report['freeze_equivalence'] = validate(case['freeze_equivalence'], base, case, expected,
                                                       column, report['sha256'])
                equivalent_source = report['freeze_equivalence']['original_source_sha']
            except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as exc:
                report['freeze_equivalence_error'] = str(exc)
        for arm in ('A', 'B'):
            runs = [r for r in receipt.get('runs', []) if isinstance(r, dict)
                    and r.get('phase') == 'scored' and r.get('arm') == arm]
            if len(runs) != 1:
                report['arms'][arm] = dict(status='INCOMPLETE', issues=['expected exactly one scored ' + arm])
            else:
                report['arms'][arm] = check_scored(runs[0], receipt, case, expected, column, transport.get(arm), equivalent_source)
        if report.get('freeze_equivalence_error'):
            for arm in report['arms'].values():
                arm['status'] = 'INCOMPLETE'
                arm.setdefault('issues', []).append(report['freeze_equivalence_error'])
        report['status'] = 'READY' if all(r['status'] == 'READY' for r in report['arms'].values()) else 'INCOMPLETE'
        require(isinstance(spec.get('history', []), list), 'invalid history list')
        for old in spec.get('history', []):
            old_path = (base / old).resolve()
            try:
                value, digest = read(old_path)
                report['history'].append(dict(path=str(old_path), sha256=digest, attempt=receipt_summary(value)))
            except (OSError, ValueError, TypeError) as exc:
                report['history'].append(dict(path=str(old_path), error=str(exc)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        report['error'] = str(exc)
        report['status'] = 'INCOMPLETE'
        for arm in report['arms'].values():
            arm['status'] = 'INCOMPLETE'
            arm.setdefault('issues', []).append(str(exc))
    return report


def state_declared(data, expected):
    """True when the pinned scope declares model state and this result captured it completely."""
    if not expected['capture_paths'].get('model_state'):
        return False
    state = data.get('model_state')
    return (isinstance(state, dict) and state.get('status') == 'CAPTURED'
            and state.get('completeness') == 'complete_declared_scope' and state.get('missing_state') == [])


def compare_arm(columns, arm, expected):
    pairs = []
    present = [c for c in COLUMNS if c in columns]
    for left, right in itertools.combinations(present, 2):
        a = columns.get(left, {}).get('arms', {}).get(arm, {})
        b = columns.get(right, {}).get('arms', {}).get(arm, {})
        row = dict(left=left, right=right, status='INCOMPLETE', parts={})
        if a.get('status') == b.get('status') == 'READY':
            both_state = state_declared(a['result'], expected) and state_declared(b['result'], expected)
            names = ('outputs', 'model_state') if both_state else ('outputs',)
            if not both_state:
                row['parts']['model_state'] = 'NOT_CAPTURED'
            for name in names:
                x = capture_signature(a['result'][name], expected['capture_paths'][name], name)
                y = capture_signature(b['result'][name], expected['capture_paths'][name], name)
                row['parts'][name] = 'MATCH' if x == y else 'MISMATCH'
            for index in range(expected['repeated_operations']):
                for name in names:
                    x = capture_signature(a['result']['repeated_use'][index][name], expected['capture_paths'][name], name)
                    y = capture_signature(b['result']['repeated_use'][index][name], expected['capture_paths'][name], name)
                    row['parts'][f'repeated.{index}.{name}'] = 'MATCH' if x == y else 'MISMATCH'
            row['status'] = 'MATCH' if all(s in ('MATCH', 'NOT_CAPTURED') for s in row['parts'].values()) else 'MISMATCH'
        else:
            row['issues'] = {name: value.get('issues', ['missing column/capture'])
                             for name, value in ((left, a), (right, b)) if value.get('status') != 'READY'}
        pairs.append(row)
    states = {r['status'] for r in pairs}
    decisive = [r for r in pairs if {r['left'], r['right']} == set(REQUIRED_IDENTICAL)]
    status = 'INCOMPLETE'
    if 'MISMATCH' in states:
        status = 'MISMATCH'
    elif decisive and decisive[0]['status'] == 'MATCH':
        status = 'MATCH'  # the two GPU vendors agree; other supplied columns matched too
    return dict(status=status, decided_by=list(REQUIRED_IDENTICAL), pairs=pairs)


def kernel_equivalence(anchor, attempt):
    """Record whether two commits share every kernel/runtime source file (metadata only; nothing is built)."""
    def names(*pathspec):
        out = subprocess.run(['git', '-C', str(ROOT), 'diff', '--name-only', anchor, attempt, '--', *pathspec],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise ValueError('git diff failed: ' + out.stderr.strip()[:200])
        return sorted(line for line in out.stdout.split('\n') if line and not line.startswith('experiments/'))
    try:
        kernel = names(*KERNEL_SCOPE)
        every = names()
    except (OSError, ValueError) as exc:
        return dict(status='UNRESOLVED', anchor_source_sha=anchor, attempt_source_sha=attempt, error=str(exc))
    return dict(status='KERNEL_EQUIVALENT' if not kernel else 'KERNEL_SOURCE_DIFFERS',
                anchor_source_sha=anchor, attempt_source_sha=attempt, kernel_scope=list(KERNEL_SCOPE),
                kernel_files_changed=kernel, other_files_changed=every,
                rule='Accepted only when no *.mojo, bindings/ or python/mojolearn/ file differs between the two commits.')


def compare_cases(manifest, base):
    require(isinstance(manifest, dict) and manifest.get('schema') == SCHEMA, 'unsupported comparison manifest')
    cases = manifest.get('cases')
    require(isinstance(cases, list) and bool(cases), 'no cases declared')
    require(all(isinstance(c, dict) and nonempty(c.get('id')) for c in cases), 'invalid case identifiers')
    require(len({c['id'] for c in cases}) == len(cases), 'duplicate case identifier')
    output = []
    for case in cases:
        report = {k: case.get(k) for k in ('id', 'configuration_id', 'implementation_ids', 'source_sha', 'workload_id', 'mode')}
        report.update(status='INCOMPLETE', receipt_status='INCOMPLETE', columns={}, arms={}, accepted=False, promoted=False)
        try:
            require(nonempty(case.get('configuration_id')) and string_list(case.get('implementation_ids')), 'missing implementation/configuration IDs')
            expected = expected_scope(case)
            report['expected'] = expected
            specs = case.get('columns', {})
            require(isinstance(specs, dict) and set(specs) <= set(COLUMNS), 'unknown column')
            require(case['mode'] != 'fast' or set(specs) <= {'apple'}, 'Apple FAST only supports Apple here')
            report['columns'] = {column: read_column(spec, base, case, expected, column) for column, spec in specs.items()}
            # NVIDIA has two routes with the same vendor field. One copied
            # receipt cannot stand in for two independently retained columns.
            for left, right in itertools.combinations(report['columns'].values(), 2):
                if left.get('sha256') and left.get('sha256') == right.get('sha256'):
                    for value in (left, right):
                        value['status'] = 'INCOMPLETE'
                        for arm in value.get('arms', {}).values():
                            arm['status'] = 'INCOMPLETE'
                            arm.setdefault('issues', []).append('same receipt reused for distinct columns')
                            arm['observed_outputs'] = dict(status='INCOMPLETE', qualified_identity=False)
            required = REQUIRED_IDENTICAL if case['mode'] == 'identical' else ('apple',)
            if all(report['columns'].get(name, {}).get('status') == 'READY' for name in required):
                report['receipt_status'] = 'COMPLETE'
            if case['mode'] == 'fast':
                report['status'] = 'NOT_REQUIRED'
                report['policy'] = 'Apple FAST uses retained task-quality gates; no cross-vendor equality requirement.'
            else:
                report['observed_output_bits'] = {}
                for arm in ('A', 'B'):
                    observations = []
                    for left, right in itertools.combinations([c for c in COLUMNS if c in specs], 2):
                        a = report['columns'].get(left, {}).get('arms', {}).get(arm, {}).get('observed_outputs', {})
                        b = report['columns'].get(right, {}).get('arms', {}).get(arm, {}).get('observed_outputs', {})
                        status = 'INCOMPLETE'
                        if a.get('status') == b.get('status') == 'OBSERVED_ONLY':
                            status = 'AGREE' if a['outputs'] == b['outputs'] else 'DIFFER'
                        observations.append(dict(left=left, right=right, status=status, qualified_identity=False,
                                                 observations={left: a, right: b}))
                    report['observed_output_bits'][arm] = observations
                report['arms'] = {arm: compare_arm(report['columns'], arm, expected) for arm in ('A', 'B')}
                states = {r['status'] for r in report['arms'].values()}
                report['status'] = 'MISMATCH' if 'MISMATCH' in states else ('MATCH' if states == {'MATCH'} else 'INCOMPLETE')
                report['missing_columns'] = sorted(set(REQUIRED_IDENTICAL) - set(specs))
                report['optional_columns_absent'] = sorted(set(COLUMNS) - set(REQUIRED_IDENTICAL) - set(specs))
        except (ValueError, KeyError, TypeError) as exc:
            report['error'] = str(exc)
        output.append(report)
    counts = {status: sum(c['status'] == status for c in output) for status in ('MATCH', 'MISMATCH', 'INCOMPLETE', 'NOT_REQUIRED')}
    return dict(schema='mojolearn.six-lane-comparison/1', cases=output, counts=counts,
                incomplete_receipt_cases=sum(c['receipt_status'] != 'COMPLETE' for c in output),
                required_identical_columns=list(REQUIRED_IDENTICAL), optional_columns=sorted(set(COLUMNS) - set(REQUIRED_IDENTICAL)),
                promotion_voters=['nvidia', 'amd'],
                apple_timing_votes=False, accepted=False, promoted=False,
                kernel_equivalent_sources_accepted=ACCEPT_KERNEL_EQUIVALENT,
                route_attribution='Explicit manifest columns; compiler/target provenance is retained, not inferred from hardware or recompiled.',
                policy='Scored A across routes and scored B across routes separately. MATCH covers declared captures only; no quality, admission or promotion inferred.',
                history_policy='Explicit history summaries and all previous_receipt/log references retained; history never substitutes for selected current evidence.')


VOTING_ROUTES = {'nvidia': 'nvidia-native', 'amd': 'amd', 'apple': 'apple'}


def timing_verdict(case, floors):
    """Scored-clock A/B verdict for one case against A/A floors (six_lane_timing)."""
    from six_lane_timing import PHASES, VOTERS, judge, log_ratios
    table = (floors or {}).get('floors', {})
    per_vendor = {}
    for vendor, route in VOTING_ROUTES.items():
        arms = case.get('columns', {}).get(route, {}).get('arms', {})
        a, b = (arms.get(x, {}).get('result') for x in ('A', 'B'))
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        ratios = log_ratios(a, b)
        if ratios is None:
            continue
        floor = table.get(str(case.get('workload_id')) + '|' + vendor)
        per_vendor[vendor] = dict(log_ratio={p: ratios[p] for p in PHASES}, timing_source=ratios['source'],
                                  floor=floor['floor'] if floor else None,
                                  floor_evidence=floor['evidence'] if floor else None)
    voters = VOTERS.get(case.get('mode'), VOTERS['identical'])
    phases = {p: judge(per_vendor, voters, p) for p in PHASES}
    return dict(verdict=phases['scored']['verdict'], phases=phases, vendors=per_vendor, voters=list(voters))


def board_inputs(report, catalog, previous_inventory=None, previous_index=None, floors=None):
    cards = {r['id']: dict(id=r['id'], title=r['title'], mode=r['mode'], vendors=r['vendors'])
             for r in catalog['entries']}
    inventory = copy.deepcopy(previous_inventory) if previous_inventory else dict(campaign='six-lane-integration', candidates=[])
    index = copy.deepcopy(previous_index) if previous_index else dict(cells=[], notes=[], decisions=[])
    known = {c['id']: c for c in inventory['candidates']}
    for case in report['cases']:
        case_timing = timing_verdict(case, floors)
        for ident in case.get('implementation_ids') or []:
            require(ident in cards, 'unknown implementation ID: ' + ident)
            require(cards[ident]['mode'] == case['mode'], 'catalog mode differs: ' + ident)
            if ident not in known:
                inventory['candidates'].append(cards[ident])
                known[ident] = cards[ident]
            for column, value in case['columns'].items():
                vendor = COLUMNS[column]
                require(vendor in known[ident]['vendors'], 'unsupported catalog vendor: ' + ident + '/' + vendor)
                arms = value.get('arms', {})
                ready = set(arms) == {'A', 'B'} and all(a['status'] == 'READY' for a in arms.values())
                cell = dict(id=ident, vendor=vendor, route=column, case=case['id'], scope='full_workload',
                            configuration_id=case['configuration_id'],
                            source_sha=value.get('attempt', {}).get('source_sha', case['source_sha']),
                            comparison_anchor_source_sha=case['source_sha'],
                            freeze_equivalence=value.get('freeze_equivalence'),
                            observed_output_bits=case.get('observed_output_bits'),
                            status='PENDING_ADMISSION' if ready else 'FAILED_OR_INCOMPLETE',
                            evidence=value.get('path'), receipt_sha256=value.get('sha256'),
                            identity=case['status'], accepted=False, promoted=False,
                            identity_arms=case['arms'], capture_issues={a: v.get('issues', []) for a, v in arms.items()},
                            task_quality={a: v.get('result', {}).get('task_quality', {'status': 'PENDING'}) for a, v in arms.items()},
                            retained_timings={a: v.get('result', {}).get('timings') for a, v in arms.items()},
                            scored_timings={a: scored(v.get('result', {}).get('timings')) for a, v in arms.items()},
                            timing_verdict=case_timing,
                            attempt=value.get('attempt'), history=value.get('history'), runs=value.get('runs'),
                            promotion_vote=case['mode'] == 'fast' or vendor in ('nvidia', 'amd'))
                if case['status'] == 'MISMATCH':
                    cell['status'] = 'IDENTITY_MISMATCH'
                if any(q.get('status') == 'FAIL' for q in cell['task_quality'].values() if isinstance(q, dict)):
                    cell['status'] = 'QUALITY_FAILED'
                if cell not in index['cells']:
                    index['cells'].append(cell)
    inventory['identity_policy'] = report['policy']
    inventory['evidence_policy'] = 'Retained full-workload receipts; quality and identity remain explicit. No automatic admission, opponent ratios or default promotion.'
    inventory['timing_policy'] = ('Scored clock = fit + transform/predict only (full_operation_seconds kept as a record). '
                                  'A timing verdict needs |log(A/B)| above the workload A/A floor on every voting vendor, all agreeing in direction.')
    note = 'Comparison inputs retain pending evidence; use the existing board tool. Historical cells are preserved; no new MEASURED rows or opponent ratios are synthesized.'
    if note not in index.setdefault('notes', []):
        index['notes'].append(note)
    return inventory, index


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='new output directory; existing evidence is never overwritten')
    parser.add_argument('--accept-kernel-equivalent-sources', action='store_true',
                        help='accept a required column whose commit differs from the pinned one only outside *.mojo, bindings/ and python/mojolearn/ (recorded per column)')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'experiments/six_lane_integration/catalog.json')
    parser.add_argument('--previous-inventory', type=Path)
    parser.add_argument('--previous-index', type=Path)
    parser.add_argument('--aa-floors', type=Path, help='six_lane_timing floors output; without it every timing verdict is NO_VERDICT')
    args = parser.parse_args(argv)
    global ACCEPT_KERNEL_EQUIVALENT
    ACCEPT_KERNEL_EQUIVALENT = bool(args.accept_kernel_equivalent_sources)
    require(bool(args.previous_inventory) == bool(args.previous_index), 'supply both previous board inputs together')
    manifest, manifest_hash = read(args.manifest)
    report = compare_cases(manifest, args.manifest.resolve().parent)
    report['inputs'] = dict(manifest=str(args.manifest.resolve()), manifest_sha256=manifest_hash,
                           tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    catalog, catalog_hash = read(args.catalog)
    report['inputs'].update(catalog=str(args.catalog.resolve()), catalog_sha256=catalog_hash)
    old_inventory = old_index = None
    if args.previous_inventory:
        old_inventory, ih = read(args.previous_inventory)
        old_index, xh = read(args.previous_index)
        report['inputs']['previous_board'] = dict(inventory=str(args.previous_inventory.resolve()), inventory_sha256=ih,
                                                 index=str(args.previous_index.resolve()), index_sha256=xh)
    floors = None
    if args.aa_floors:
        floors, fh = read(args.aa_floors)
        report['inputs'].update(aa_floors=str(args.aa_floors.resolve()), aa_floors_sha256=fh)
    inventory, index = board_inputs(report, catalog, old_inventory, old_index, floors)
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'report.json', report)
    write(args.out / 'inventory.json', inventory)
    write(args.out / 'index.json', index)
    write(args.out / 'future_command.json', dict(argv=[sys.executable, str(ROOT / 'tools/performance_measurement_board.py'),
          '--inventory', str((args.out / 'inventory.json').resolve()), '--index', str((args.out / 'index.json').resolve()),
          '--out', str((args.out / 'board').resolve())], execution='NOT RUN'))
    print(json.dumps(dict(counts=report['counts'], output=str(args.out), board_execution='NOT RUN')))
    return 1 if report['counts']['MISMATCH'] else (2 if report['incomplete_receipt_cases'] else 0)


if __name__ == '__main__':
    raise SystemExit(main())
