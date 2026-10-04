# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Fail-closed runtime validation of offline PTX IDENTICAL admission records.

This validates a release decision, not its underlying experimental evidence.
Only the offline evidence checker may establish complete applicable coverage.
No supported configuration or admission record is shipped by this module.
"""
import re

ADMISSION_FILE = 'PTX_IDENTITY_ADMISSION.json'
SCHEMA = 'mojolearn.ptx-identity-admission.v1'
COVERAGE_CONTRACT = 'mojolearn.cross-vendor-identical.v1'
SHARED_FIXTURES = ('base', 'denormal', 'odd')
NVIDIA_FIXTURES = ('base', 'ties', 'hashed', 'wide', 'denormal', 'denormal_ftz', 'dupes', 'odd', 'negative')
SHARED_PARTS = ('train', 'infer', 'model', 'batchgrad', 'batchscale', 'ragged', 'stepfull')
NVIDIA_PARTS = SHARED_PARTS + ('batch', 'rlpair')


def _require(condition, message):
    if not condition:
        raise ValueError('PTX identity admission: ' + message)


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _names(value):
    return (isinstance(value, list) and bool(value)
            and all(isinstance(v, str) and v.strip() == v and bool(v) for v in value)  # glue: validate metadata identifier strings
            and len(value) == len(set(value)))


def _configuration(value):
    _require(isinstance(value, dict) and set(value) == {
        'device_name', 'compute_capability', 'driver_version', 'cuda_driver_version'},
        'configuration fields are missing or unknown')
    cap = value['compute_capability']
    _require(isinstance(cap, list) and len(cap) == 2
             and all(type(v) is int and v >= 0 for v in cap) and cap >= [8, 0],  # glue: validate architecture metadata integers
             'unsupported compute capability')
    _require(isinstance(value['device_name'], str) and bool(value['device_name'].strip()), 'missing device name')
    _require(isinstance(value['driver_version'], str)
             and re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', value['driver_version']) is not None,
             'unknown driver version')
    _require(type(value['cuda_driver_version']) is int and value['cuda_driver_version'] > 0,
             'missing CUDA driver API version')
    return (value['device_name'], tuple(cap), value['driver_version'], value['cuda_driver_version'])


def validate_admission(doc, *, source_commit, manifest_sha256, configuration=None):
    """Validate immutable release metadata; optionally match this exact runtime.

    Source and manifest come from actual installed files. Packaging supplies no
    configuration, while the loader must supply independently queried hardware.
    """
    _require(isinstance(doc, dict), 'record must be an object')
    _require(doc.get('schema') == SCHEMA and doc.get('qualified') is True
             and doc.get('numeric_mode') == 'identical'
             and doc.get('coverage_contract') == COVERAGE_CONTRACT, 'unqualified record or wrong contract')
    _require(isinstance(source_commit, str) and re.fullmatch('[0-9a-f]{40}', source_commit) is not None
             and doc.get('source_commit') == source_commit, 'source mismatch')
    _require(_digest(manifest_sha256) and doc.get('manifest_sha256') == manifest_sha256, 'manifest mismatch')
    coverage = doc.get('coverage')
    _require(isinstance(coverage, dict) and _digest(coverage.get('inventory_sha256'))
             and _digest(coverage.get('harness_sha256')), 'missing coverage provenance')
    shared, native = coverage.get('shared'), coverage.get('nvidia')
    for name, row, fixtures, parts in (('shared', shared, SHARED_FIXTURES, SHARED_PARTS),  # glue: validate two release coverage metadata records
                                      ('nvidia', native, NVIDIA_FIXTURES, NVIDIA_PARTS)):
        _require(isinstance(row, dict) and _names(row.get('lanes')), name + ' lanes missing or duplicated')
        _require(_names(row.get('fixtures')) and set(row['fixtures']) == set(fixtures), name + ' fixtures incomplete')
        _require(_names(row.get('parts')) and set(row['parts']) == set(parts), name + ' parts incomplete')
        _require(_digest(row.get('comparison_sha256')), name + ' comparison missing')
    _require(set(shared['lanes']) == set(native['lanes']), 'shared and NVIDIA lane scopes differ')
    # Undeclared parts are named exclusions of the NVIDIA scope, never coverage.
    # The shared scope admits none.
    _require('undeclared_exclusions' not in shared, 'shared scope cannot exclude undeclared parts')
    excluded = native.get('undeclared_exclusions', [])
    _require(isinstance(excluded, list), 'undeclared exclusions malformed')
    seen = set()
    for row in excluded:  # glue: validate excluded lane/part metadata records
        _require(isinstance(row, dict) and set(row) == {'lane', 'part'}
                 and row['lane'] in native['lanes'] and row['part'] in native['parts']
                 and (row['lane'], row['part']) not in seen, 'undeclared exclusion unknown or duplicated')
        seen.add((row['lane'], row['part']))
    _require(_names(shared.get('vendors')) and set(shared['vendors']) == {'cuda', 'hip', 'metal'},
             'cross-vendor references incomplete')
    configs = doc.get('configurations')
    _require(isinstance(configs, list) and bool(configs), 'no admitted configurations')
    keys = [_configuration(row) for row in configs]  # glue: validate release hardware metadata entries
    _require(len(keys) == len(set(keys)), 'duplicate configurations')
    absent = coverage.get('native_absent')
    if absent is not None:
        # Devices with no native payload: nine fixtures against other devices'
        # native references, three shared fixtures against Apple and AMD.
        _require(isinstance(absent, dict) and isinstance(absent.get('configurations'), list)
                 and bool(absent['configurations']), 'native-absent scope has no configurations')
        absent_keys = [_configuration(row) for row in absent['configurations']]  # glue: validate release hardware metadata entries
        _require(len(absent_keys) == len(set(absent_keys)) and set(absent_keys) <= set(keys),
                 'native-absent configuration is not an admitted configuration')
        for name, fixtures, parts in (('nvidia', NVIDIA_FIXTURES, NVIDIA_PARTS),  # glue: validate two release coverage metadata records
                                      ('shared', SHARED_FIXTURES, SHARED_PARTS)):
            row = absent.get(name)
            _require(isinstance(row, dict) and _names(row.get('fixtures')) and set(row['fixtures']) == set(fixtures)
                     and _names(row.get('parts')) and set(row['parts']) == set(parts)
                     and _digest(row.get('comparison_sha256')), 'native-absent ' + name + ' scope incomplete')
        caps = absent['nvidia'].get('native_reference_capabilities')
        _require(isinstance(caps, list) and len(caps) >= 2
                 and all(isinstance(cap, list) and len(cap) == 2 and all(type(v) is int for v in cap) for cap in caps)  # glue: validate architecture metadata integers
                 and len({tuple(cap) for cap in caps}) == len(caps)  # glue: compare architecture metadata pairs
                 and not {tuple(cap) for cap in caps} & {key[1] for key in absent_keys},  # glue: compare architecture metadata pairs
                 'native-absent scope lacks two distinct native reference capabilities')
        _require(_names(absent['shared'].get('vendors')) and set(absent['shared']['vendors']) == {'hip', 'metal'},
                 'native-absent cross-vendor references incomplete')
    if configuration is not None:
        _require(_configuration(configuration) in keys, 'this device/driver configuration is not admitted')
    return doc


# --------------------------------------------------------------------------
# LOCAL QUALIFICATION (`python -m mojolearn verify --qualify-gpu`)
# --------------------------------------------------------------------------
# A release admission names the few device/driver configurations that were
# measured before publication. Every other configuration can qualify ITSELF:
# the command runs the shipped identity suite through the PTX payload on the
# user's device and compares every part with the reference table the wheel
# ships. A complete, clean comparison is recorded in a separate per-user file
# with its own schema. It is this user's evidence about this exact wheel,
# device and driver; it is never a release claim and never widens to another
# configuration. This module only validates records and judges finished
# verifier reports. It runs nothing and computes no numeric result.

LOCAL_SCHEMA = 'mojolearn.ptx-local-identity-admission.v1'
LOCAL_SCOPE = 'this-wheel-device-driver-only'
LOCAL_DIR_ENV = 'MOJOLEARN_PTX_ADMISSION_DIR'
REPORT_FORMAT = 'mojolearn.verify-all-report.v1'
#: The two verifier scopes that together select every lane the harness defines.
QUALIFY_PROFILES = ('routine', 'neural-training')
#: The only lane states a qualification accepts. NOT APPLICABLE is the par-*
#: two-device drivers, a structural exclusion that stays named in the record.
#: OWED, HELD, NOT RUN and UNDECLARED do not gate `verify`; they gate this.
LANE_VERIFIED = 'VERIFIED'
LANE_NOT_APPLICABLE = 'NOT APPLICABLE'


def local_admission_dir():
    """The per-user directory local admissions live in."""
    import os
    override = os.environ.get(LOCAL_DIR_ENV, '').strip()
    if override:
        return override
    state = os.environ.get('XDG_STATE_HOME', '').strip() or os.path.join(os.path.expanduser('~'), '.local', 'state')
    return os.path.join(state, 'mojolearn', 'ptx-admissions')


def _file_sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):  # glue: hash file bytes
            digest.update(block)
    return digest.hexdigest()


def reference_hashes(pkg_dir):
    """Hashes of the reference table and identity harness this install ships."""
    import os
    table = os.path.join(str(pkg_dir), 'verify_reference', 'table.json')
    harness = os.path.join(str(pkg_dir), '_identity_break.py')
    for label, path in (('reference table', table), ('identity harness', harness)):  # glue: check two shipped files exist
        if not os.path.isfile(path):
            raise ValueError(f'PTX identity admission: this install ships no {label} ({path})')
    return dict(table_sha256=_file_sha256(table), harness_sha256=_file_sha256(harness))


def _reference(value):
    _require(isinstance(value, dict) and set(value) == {'table_sha256', 'harness_sha256'}
             and all(_digest(v) for v in value.values()), 'reference hashes missing or unknown')  # glue: check two SHA256 metadata fields
    return (value['table_sha256'], value['harness_sha256'])


def local_admission_key(source_commit, manifest_sha256, configuration, reference):
    """One digest over everything a local admission is bound to."""
    import hashlib
    import json
    name, cap, driver, api = _configuration(configuration)
    table, harness = _reference(reference)
    _require(isinstance(source_commit, str) and re.fullmatch('[0-9a-f]{40}', source_commit) is not None,
             'source commit missing')
    _require(_digest(manifest_sha256), 'manifest hash missing')
    bound = dict(schema=LOCAL_SCHEMA, source_commit=source_commit, manifest_sha256=manifest_sha256,
                 device_name=name, compute_capability=list(cap), driver_version=driver,
                 cuda_driver_version=api, table_sha256=table, harness_sha256=harness)
    return hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def local_admission_path(directory, key):
    import os
    _require(_digest(key), 'invalid local admission key')
    return os.path.join(directory, f'ptx-identity-{key[:32]}.json')


# --------------------------------------------------------------------------
# THE QUALIFICATION TOKEN
# --------------------------------------------------------------------------
# While `verify --qualify-gpu` runs, its verifier processes must load the
# IDENTICAL PTX set on a configuration no admission covers yet, or there would
# be nothing to compare. That load is not available to any other process. The
# command writes a token file for one run, names it in QUALIFY_TOKEN_ENV for
# its children, and removes it when the run ends. A process is a qualification
# process only if the variable names such a file AND the file is bound to this
# exact wheel, device, driver and reference AND the live process the token
# names is an ancestor of this one AND that ancestor is the qualification
# command itself. A variable exported by hand in an ordinary process meets
# none of those, so IDENTICAL still refuses there.

QUALIFY_TOKEN_SCHEMA = 'mojolearn.ptx-qualification-token.v1'
QUALIFY_TOKEN_ENV = 'MOJOLEARN_PTX_QUALIFYING'
_ANCESTOR_HOPS = 64


def is_qualify_argv(argv):
    """True when `sys.argv`, as seen while the package is imported, is
    `python -m mojolearn verify --qualify-gpu` (`['-m', 'verify', ...]`) or the
    same through `.../mojolearn/__main__.py`."""
    import os
    argv = list(argv or [])
    return (bool(argv) and all(isinstance(a, str) for a in argv)  # glue: check command-line strings
            and os.path.basename(argv[0]) in ('-m', '__main__.py')
            and argv[1:2] == ['verify'] and '--qualify-gpu' in argv[2:])


def is_qualify_command_line(command):
    """True when a full process command line is the qualification command:
    `[python, <interpreter options>, '-m', 'mojolearn', 'verify', ..., '--qualify-gpu']`
    or the same through `.../mojolearn/__main__.py`. Anything that is not an
    interpreter option before the launcher (a script name) fails."""
    import os
    command = list(command or [])
    if len(command) < 2 or not all(isinstance(a, str) for a in command):  # glue: check command-line strings
        return False
    for i in range(1, len(command)):  # glue: scan command-line words for the launcher
        if command[i] == '-m':
            return command[i + 1:i + 2] == ['mojolearn'] and is_qualify_argv(['-m'] + command[i + 2:])
        if os.path.basename(command[i]) == '__main__.py':
            return is_qualify_argv(command[i:])
        if not command[i].startswith('-'):
            return False
    return False


def _process_parent(pid):
    """Parent pid of `pid` from /proc, or None where that cannot be read."""
    try:
        with open(f'/proc/{int(pid)}/stat', 'rb') as stream:
            text = stream.read().decode('ascii', 'replace')
        return int(text[text.rindex(')') + 1:].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _process_command(pid):
    """Command line of `pid` from /proc, or None where that cannot be read."""
    try:
        with open(f'/proc/{int(pid)}/cmdline', 'rb') as stream:
            return [part.decode('utf-8', 'replace') for part in stream.read().split(b'\0') if part]  # glue: split a process command line
    except (OSError, ValueError):
        return None


def qualification_token(key, pid, nonce):
    """The token document of one qualification run."""
    _require(_digest(key) and _digest(nonce) and type(pid) is int and pid > 1, 'qualification token fields invalid')
    return dict(schema=QUALIFY_TOKEN_SCHEMA, key=key, pid=pid, nonce=nonce)


def validate_qualification_token(path, key):
    """Accept `path` only as the live token of a qualification run for `key`
    (see `local_admission_key`) that this process descends from. Raises
    ValueError naming the first reason it is not."""
    import json
    import os
    import stat
    _require(isinstance(path, str) and os.path.isabs(path), 'qualification token is not an absolute file path')
    try:
        info = os.lstat(path)
        _require(stat.S_ISREG(info.st_mode) and info.st_size <= 4096, 'qualification token is not a small regular file')
        _require(info.st_uid == os.getuid() and not info.st_mode & 0o022,
                 'qualification token is not a private file of this user')
        with open(path, 'rb') as stream:
            doc = json.loads(stream.read(4097))
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith('PTX identity admission: '):
            raise
        raise ValueError(f'PTX identity admission: qualification token unreadable: {exc}') from exc
    _require(isinstance(doc, dict) and set(doc) == {'schema', 'key', 'pid', 'nonce'}
             and doc['schema'] == QUALIFY_TOKEN_SCHEMA and _digest(doc['nonce'])
             and type(doc['pid']) is int and doc['pid'] > 1, 'qualification token malformed')
    _require(_digest(key) and doc['key'] == key,
             'qualification token is for another wheel, device, driver or reference')
    pid, hops = os.getppid(), 0
    while pid is not None and pid > 1 and pid != doc['pid'] and hops < _ANCESTOR_HOPS:
        pid, hops = _process_parent(pid), hops + 1
    _require(pid == doc['pid'], 'the qualification run this token names is not an ancestor of this process')
    _require(is_qualify_command_line(_process_command(doc['pid'])),
             'the process this token names is not the qualification command')
    return doc


def _clean_profile(row, profile):
    """One profile's recorded comparison summary must itself read complete and clean."""
    _require(isinstance(row, dict) and row.get('profile') == profile, 'comparison profile missing')
    counts = row.get('counts')
    _require(row.get('verdict') == 'VERIFIED' and isinstance(counts, dict)
             and all(type(counts.get(k)) is int for k in ('IDENTICAL', 'DIVERGENT', 'REFUSED', 'OWED'))  # glue: validate recorded state counts
             and counts['IDENTICAL'] > 0 and counts['DIVERGENT'] == 0 and counts['REFUSED'] == 0
             and counts['OWED'] == 0, profile + ' comparison is not complete and identical')
    _require(type(row.get('lanes_verified')) is int and row['lanes_verified'] > 0
             and type(row.get('lanes_in_scope')) is int
             and isinstance(row.get('lanes_not_applicable'), list)
             and row['lanes_verified'] + len(row['lanes_not_applicable']) == row['lanes_in_scope'],
             profile + ' comparison does not account for every lane in scope')
    _require(_names(row.get('fixtures')) and _digest(row.get('report_sha256')),
             profile + ' comparison evidence missing')


def validate_local_admission(doc, *, source_commit, manifest_sha256, configuration, reference):
    """Accept a local admission for exactly this wheel, device, driver and reference."""
    _require(isinstance(doc, dict), 'record must be an object')
    _require(doc.get('schema') == LOCAL_SCHEMA and doc.get('qualified') is True
             and doc.get('numeric_mode') == 'identical' and doc.get('scope') == LOCAL_SCOPE,
             'not a qualified local admission')
    _require(doc.get('source_commit') == source_commit, 'local admission is for another source commit')
    _require(doc.get('manifest_sha256') == manifest_sha256, 'local admission is for another PTX payload')
    _require(_configuration(doc.get('configuration')) == _configuration(configuration),
             'local admission is for another device or driver')
    _require(_reference(doc.get('reference')) == _reference(reference),
             'local admission is for another reference table or harness')
    _require(doc.get('key') == local_admission_key(source_commit, manifest_sha256, configuration, reference),
             'local admission key mismatch')
    comparison = doc.get('comparison')
    _require(isinstance(comparison, dict) and isinstance(comparison.get('profiles'), list)
             and [row.get('profile') if isinstance(row, dict) else None
                  for row in comparison['profiles']] == list(QUALIFY_PROFILES),  # glue: check recorded profile names
             'local admission does not cover every verification profile')
    for row, profile in zip(comparison['profiles'], QUALIFY_PROFILES):  # glue: validate two recorded summaries
        _clean_profile(row, profile)
    return doc


def judge_report(report, *, profile, reference, table_fixtures, ptx_identical_sha256, report_sha256):
    """Judge one finished `verify --all` report for local qualification.

    Returns dict(summary, differing, missing, incomplete). `differing` is wrong
    answers, `missing` is reference data the wheel does not ship for something
    this device ran or should have run, `incomplete` is anything that did not
    run or cannot be attributed to the PTX payload. Qualification needs all
    three empty; nothing here weakens a state `verify` itself reported."""
    differing, missing, incomplete = [], [], []
    if not isinstance(report, dict) or report.get('format') != REPORT_FORMAT or 'cells' not in report:
        detail = report.get('detail') if isinstance(report, dict) else None
        return dict(summary=None, differing=[], missing=[],
                    incomplete=[f'{profile}: the verifier produced no complete report'
                                + (f' ({detail})' if detail else '')])
    selected = ((report.get('selection') or {}).get('profile') or {}).get('name')
    if selected != profile:
        incomplete.append(f'{profile}: report is for profile {selected!r}')
    fixtures = list(report.get('fixtures') or [])
    if report.get('depth') != 'full' or set(fixtures) != set(table_fixtures) or not fixtures:
        incomplete.append(f'{profile}: not a full-depth run over every reference fixture '
                          f'(depth {report.get("depth")!r}, fixtures {fixtures})')
    device = report.get('device') or {}
    if device.get('vendor') != 'cuda' or device.get('numeric_mode') != 'identical':
        incomplete.append(f'{profile}: ran {device.get("vendor")!r} in {device.get("numeric_mode")!r} mode, '
                          'not cuda in identical mode')
    table, harness = report.get('table') or {}, report.get('harness') or {}
    if table.get('sha256') != reference['table_sha256'] or harness.get('sha256') != reference['harness_sha256']:
        incomplete.append(f'{profile}: the run did not use the reference table and harness this install ships')
    if harness.get('matches_table') is not True:
        missing.append(f'{profile}: the shipped reference table was not generated by the shipped harness')
    execution = report.get('execution') or {}
    if (execution.get('interrupted') or not execution.get('total_cells')
            or execution.get('completed_cells') != execution.get('total_cells')):
        incomplete.append(f'{profile}: stopped after {execution.get("completed_cells")} of '
                          f'{execution.get("total_cells")} cells: {execution.get("interrupted")}')
    if (report.get('self_test') or {}).get('passed') is not True:
        incomplete.append(f'{profile}: the comparator self-test did not pass')
    parts = set()
    for row in report['cells']:  # glue: sort judged verifier rows by their recorded state
        where = f'{row.get("lane")}/{row.get("fixture")}/{row.get("part")}'
        state = row.get('state')
        if state == 'IDENTICAL':
            parts.add(row.get('part'))
        elif state == 'DIVERGENT':
            differing.append(f'{where}: {row.get("detail")}')
        elif state == 'OWED':
            missing.append(f'{where}: {row.get("detail")}')
        elif state == 'REFUSED':
            incomplete.append(f'{where}: {str(row.get("detail"))[:300]}')
        elif state != 'N/A':
            incomplete.append(f'{where}: unknown state {state!r}')
    accounting = report.get('lane_accounting') or {}
    scope = list(accounting.get('verdict_scope') or [])
    lanes = accounting.get('lanes') or {}
    verified, not_applicable = [], []
    for lane in scope:  # glue: sort lane accounting rows by their recorded state
        entry = lanes.get(lane) or {}
        state = entry.get('state')
        if state == LANE_VERIFIED:
            verified.append(lane)
        elif state == LANE_NOT_APPLICABLE:
            not_applicable.append(lane)
        elif state == 'DIVERGENT':
            differing.append(f'{lane}: lane DIVERGENT: {entry.get("reason")}')
        elif state == 'REFUSED':
            incomplete.append(f'{lane}: lane REFUSED: {str(entry.get("reason"))[:300]}')
        else:
            missing.append(f'{lane}: lane {state or "not accounted for"}: {entry.get("reason")}')
    if not scope or accounting.get('total') is None:
        incomplete.append(f'{profile}: the report carries no lane accounting')
    if not verified:
        incomplete.append(f'{profile}: no lane was verified')
    gpu_bindings = [b for b in report.get('bindings') or [] if isinstance(b, dict) and b.get('kind') != 'host']  # glue: select GPU binding provenance rows
    foreign = sorted({str(b.get('module')) for b in gpu_bindings if b.get('sha256') not in ptx_identical_sha256})  # glue: match binding digests to the PTX manifest
    if not gpu_bindings or foreign:
        incomplete.append(f'{profile}: GPU bindings not loaded from the IDENTICAL PTX payload: '
                          f'{foreign or "none recorded"}')
    cross = report.get('cross_check') or {}
    if profile == 'routine':
        if cross.get('differ'):
            differing.append(f'{profile}: GPU and CPU inference differ in {cross.get("differ")} cells')
        if cross.get('ran') is not True or cross.get('passed') is not True:
            incomplete.append(f'{profile}: the GPU/CPU inference comparison did not run to a pass')
    counts = dict(report.get('counts') or {})
    if report.get('exit') != 0 or report.get('verdict') != 'VERIFIED':
        if not (differing or missing or incomplete):
            incomplete.append(f'{profile}: verifier verdict {report.get("verdict")!r}, exit {report.get("exit")}')
    summary = dict(profile=profile, verdict=report.get('verdict'), counts=counts,
                   lanes_in_scope=len(scope), lanes_verified=len(verified),
                   lanes_not_applicable=sorted(not_applicable),  # glue: canonicalize lane names
                   fixtures=fixtures,
                   parts_identical=sorted(p for p in parts if isinstance(p, str)),  # glue: canonicalize part names
                   cells=len(report['cells']),
                   cross_check=dict(ran=cross.get('ran'), passed=cross.get('passed'),
                                    compared=cross.get('compared'), agree=cross.get('agree')),
                   report_sha256=report_sha256)
    return dict(summary=summary, differing=differing, missing=missing, incomplete=incomplete)


def build_local_admission(*, source_commit, manifest_sha256, configuration, reference, summaries,
                          core_version, created_utc):
    """The local admission document, validated before it is returned."""
    doc = dict(schema=LOCAL_SCHEMA, qualified=True, numeric_mode='identical', scope=LOCAL_SCOPE,
               key=local_admission_key(source_commit, manifest_sha256, configuration, reference),
               source_commit=source_commit, manifest_sha256=manifest_sha256, core_version=core_version,
               configuration=dict(configuration), reference=dict(reference),
               comparison=dict(reference='mojolearn/verify_reference/table.json', profiles=list(summaries)),
               created_utc=created_utc)
    return validate_local_admission(doc, source_commit=source_commit, manifest_sha256=manifest_sha256,
                                    configuration=configuration, reference=reference)
