"""Admit unchanged release components plus exercised, byte-bound repairs.

Version and provenance changes alone do not invalidate runtime evidence. This
adapter supports the qualified forest input-admission and split-wheel verifier
repairs, plus the two caught-and-restored fixture corrections. It rejects other
runtime edits.
"""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import release_evidence as evidence
import release_gate_reuse as reuse

SCHEMA = 'mojolearn.release.incremental-gate.v1'
HOST = 'mojolearn/host/_mojolearn_native_tokenizer_host.so'
VERIFIER = 'mojolearn/_verify_distributed.py'
FIXTURE_FILES = {
    'mojolearn/_identity_break.py': 'tools/identity_break.py',
    'mojolearn/_identity_lane_neighbors.py': 'tools/identity_lanes/neighbors.py',
}
FIXTURE_ARMS = {
    'coverage-precise-x_neighbors-p2m_transpose_i': 'knn-clf/base',
    'coverage-precise-x_neighbors-p2m_const_cols': 'x-neighbors-nearest-centroid/base',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def contents(ref, version):
    with zipfile.ZipFile(io.BytesIO(evidence.read_bytes(reuse.checked_ref(ref)))) as z:
        require(len(z.namelist()) == len(set(z.namelist())), 'duplicate wheel member')
        return {n.replace('-' + version + '.dist-info/', '-VERSION.dist-info/'): z.read(n)
                for n in z.namelist() if not n.endswith('/')}


def delta(before, after, old_version, new_version, qualified):
    left, right = contents(before, old_version), contents(after, new_version)
    require(set(left) == set(right), 'wheel member coverage changed')
    changes = []
    for name in sorted(left):
        if left[name] == right[name]:
            continue
        changes.append(name)
        if name in qualified:
            require(hashlib.sha256(right[name]).hexdigest() == qualified[name],
                    'qualified component bytes differ: ' + name)
        elif name.endswith(('.dist-info/RECORD', '/LINUX_PAYLOAD.json', '/gpu_plugin.json', '/PTX_BASELINE.json')):
            # The candidate installer and retained assembly validators check
            # these structured manifests independently; they carry no runtime.
            continue
        elif name == 'mojolearn/identity_columns/COMMIT':
            continue
        elif name.endswith('.dist-info/portable-math.json'):
            # This audit is written before stripping/signing. Its provenance
            # hashes change on repack; the native members are compared above.
            a, b = json.loads(left[name]), json.loads(right[name])
            policy = lambda d: {k: v for k, v in d.items() if k not in ('binaries', 'runtime_changes')}
            imports = lambda d: {r['file']: {k: v for k, v in r.items() if k != 'sha256'}
                                 for r in d['binaries']}
            require(policy(a) == policy(b) and imports(a) == imports(b),
                    'portable-math policy or import coverage changed')
        elif name.endswith(('.dist-info/METADATA', '.dist-info/gpu_plugins.json')) or name in (
                'mojolearn/_version.py', 'mojolearn/CITATION.cff'):
            require(left[name].replace(old_version.encode(), new_version.encode()) == right[name],
                    'change beyond version metadata: ' + name)
        else:
            raise ValueError('unqualified runtime/package change: ' + name)
    return changes


def host_boundary(document, identity):
    boundary = reuse.read(document['boundary'])
    require(boundary.get('schema') == 'mojolearn.release.host-boundary.v1'
            and boundary.get('passed') is True and boundary.get('exit_code') == 0
            and boundary.get('cases') == 4, 'host boundary did not pass all four cases')
    require(boundary['source_commit'] == identity['source_commit']
            and boundary['version'] == identity['version'], 'host boundary source/version differs')
    wanted = {name.replace('-', '_') + '-' + identity['version'] + '-': value
              for name, value in identity['wheels'].items()}
    for prefix, value in wanted.items():
        matches = [h for n, h in boundary['wheels'].items() if n.startswith(prefix)]
        require(matches == [value], 'host boundary did not install exact candidate wheels')
    require(set(boundary['checks']) == {'venv', 'install', 'dependencies', 'self-test', 'forest'},
            'missing installed host boundary checks')
    for name, check in boundary['checks'].items():
        require(check['exit_code'] == 0 and check['sha256'] == document['boundary_logs'][name]['sha256'],
                'failed or changed host boundary log')
        reuse.checked_ref(document['boundary_logs'][name])
    require(boundary['junit']['sha256'] == document['boundary_junit']['sha256'], 'JUnit identity differs')
    cases = list(ET.fromstring(evidence.read_bytes(reuse.checked_ref(document['boundary_junit']))).iter('testcase'))
    require(len(cases) == 4 and not any(c.find(t) is not None
                for c in cases for t in ('failure', 'error', 'skipped')), 'incomplete forest runtime coverage')
    return boundary


def retained_host_groups(before, after):
    """Map separately built host bytes only when every compile input matches.

    Both assembly proofs must also pass the original producer validators. A
    different original compilation is provenance, not an input invalidation.
    """
    import retained_release_composition as composition
    old_context, new_context = before['admitted_inputs'], after['admitted_inputs']
    old = composition.compilation_bindings(old_context, composition._host_keys(old_context))
    new = composition.compilation_bindings(new_context, composition._host_keys(new_context))
    old_index, = [r['files'] for n, r in before['merged_indexes'].items() if '/host/' in n]
    new_index, = [r['files'] for n, r in after['merged_indexes'].items() if '/host/' in n]
    require(old_index == new_index, 'host physical/logical group membership changed')
    result = {}
    for filename, members in new_index.items():
        keys = ['linux-64/host/-/host/' + name for name in members]
        if not all(k in old and k in new and old[k] == new[k] for k in keys):
            continue
        before_hash, = [h for p, h in before['host_extension'].items() if p.endswith('/' + filename)]
        after_hash, = [h for p, h in after['host_extension'].items() if p.endswith('/' + filename)]
        result['mojolearn/host/' + filename] = dict(before_sha256=before_hash,
            after_sha256=after_hash, members=members, compilation_inputs_equal=True)
    return result


def verifier_update(reference):
    """Admit only the two exercised fixture repairs; never native changes."""
    import coverage_sabotage
    update = reuse.read(reference)
    require(update.get('schema') == 'mojolearn.release.verifier-fixture-update.v1',
            'unknown verifier update')
    require(set(update['files']) == set(FIXTURE_FILES)
            and set(update['proofs']) == set(FIXTURE_ARMS), 'verifier update coverage differs')
    qualified = {}
    for name, source in FIXTURE_FILES.items():
        reuse.checked_ref(update['files'][name])
        qualified[name] = update['files'][name]['sha256']
    for arm, cell in FIXTURE_ARMS.items():
        proof = update['proofs'][arm]
        snapshot = reuse.read(proof['source'])
        require(all(snapshot[source] == qualified[name] for name, source in FIXTURE_FILES.items()),
                'fault proof used different verifier bytes')
        # The original checker reads ordinary files. Materialize its inputs
        # through the evidence resolver so the same proof works in the private
        # publication bundle without rewriting historical paths or records.
        with tempfile.TemporaryDirectory(prefix='release-fixture-proof-') as directory:
            paths = []
            for key in ('clean', 'sabotaged', 'restored', 'review'):
                path = Path(directory) / (key + '.json')
                path.write_bytes(evidence.read_bytes(reuse.checked_ref(proof[key])))
                paths.append(path)
            # This validator also runs inside JSON-producing publication CLIs.
            # The checker is a human-facing CLI; retain its diagnostic text on
            # failure without corrupting the caller's machine-readable stdout.
            diagnostic = io.StringIO()
            with redirect_stdout(diagnostic):
                failures = coverage_sabotage.check(coverage_sabotage.arms()[arm], *paths,
                                                   batch_bindings=proof['batch_bindings'])
        require(not failures, 'verifier fault proof failed: ' + diagnostic.getvalue())
        previous, clean = reuse.read(proof['previous_clean']), reuse.read(proof['clean'])
        require(previous['cells'][cell]['hashes'] == clean['cells'][cell]['hashes'],
                'fixture repair changed reference outputs')
    apple = reuse.read(update['apple'])
    require(apple.get('complete') is True and set(apple['cells']) == set(FIXTURE_ARMS.values())
            and all(c['verdict'] == 'STABLE' and not c.get('oracle_errors')
                    for c in apple['cells'].values()), 'changed Apple fixtures did not pass')
    return qualified


def fixture_repack(document, identity):
    """Carry exact prior execution forward across a tested verifier-only repack."""
    prior = reuse.read(document['retained_gate'])
    require(prior.get('schema') == SCHEMA and 'verifier_update' not in prior,
            'fixture repack needs the original qualified incremental gate')
    validate(prior, prior['candidate'])
    for name in ('version', 'vendor', 'release_modes'):
        require(identity[name] == prior['candidate'][name], 'fixture repack changes ' + name)
    for field in ('correctness', 'autograd_counts', 'autograd_artifacts', 'adapters'):
        require(document[field] == prior[field], 'fixture repack rewrote retained execution')
    qualified = verifier_update(document['verifier_update'])
    require(set(document['wheels']) == set(identity['wheels']) == set(prior['candidate']['wheels']),
            'fixture repack wheel coverage differs')
    for package, row in document['wheels'].items():
        require(row['before']['sha256'] == prior['candidate']['wheels'][package]
                and row['after']['sha256'] == identity['wheels'][package], 'fixture repack wheel identity differs')
        require(delta(row['before'], row['after'], identity['version'], identity['version'], qualified)
                == row['changed'], 'fixture repack changes differ')
    return True


def validate(document, identity):
    require(document.get('schema') == SCHEMA and document.get('candidate') == identity
            and document.get('verdict') == 'PASS' and document.get('profile') == 'release-light-v1',
            'incremental gate does not identify exact candidate')
    if 'verifier_update' in document:
        return fixture_repack(document, identity)
    old = reuse.read(document['retained_gate'])
    require(old.get('schema') == reuse.SCHEMA, 'expected original component gate')
    reuse.validate(old, old['candidate'])
    require(old['candidate']['vendor'] == identity['vendor'], 'retained vendor differs')
    for field in ('correctness', 'autograd_counts', 'autograd_artifacts', 'adapters'):
        require(document[field] == old[field], 'retained execution record was rewritten: ' + field)
    boundary = host_boundary(document, identity)
    repair = reuse.read(document['host_component'])
    require(repair.get('schema') == 'mojolearn.targeted-host-component.v1'
            and repair.get('passed') is True and repair.get('exit_code') == 0
            and repair.get('cases') == 4, 'missing separately compiled host qualification')
    require(repair['native_sha256'] == boundary['host_native_sha256'], 'installed host differs from tested component')
    require(repair['test_sha256'] == boundary['test_sha256'], 'host boundary tested different public operations')
    import retained_release_composition as composition
    import release_host_composition as hc
    assembly = reuse.read(document['assembly'])
    composition.validate(assembly)
    prior_assembly = reuse.read(old['assembly'])
    composition.validate(prior_assembly)
    retained = retained_host_groups(prior_assembly, assembly)
    require(document['retained_host_groups'] == retained, 'retained host input comparison differs')
    require(assembly['source_commit'] == identity['source_commit'], 'assembly source differs')
    host = hc.decoded(assembly['original_host']['proof'])
    require(host['source_commit'] == repair['native_source_commit']
            and document['host_build_proof_sha256'] == repair['host_proof_sha256']
            and assembly['original_host']['proof']['sha256'] == repair['host_proof_sha256'],
            'host qualification belongs to another build')
    producer = reuse.read(document['verifier_producer'])
    capture = reuse.read(document['verifier_capture'])
    require(any(r.get('path') == 'capture/report.json'
                and r.get('sha256') == document['verifier_capture']['sha256']
                for r in producer['evidence']), 'verifier capture is not bound to the producer')
    require(producer.get('status') == 'passed' and producer.get('exit_code') == 0
            and producer.get('vendor') == identity['vendor'], 'new verifier producer failed')
    require([r['expected'] for r in producer['coverage']] == [44, 8, 17]
            and all(r['status'] == 'passed' and r['exit_code'] == 0
                    and r['completed'] == r['expected'] and r['failed'] == r['skipped'] == 0
                    for r in producer['coverage']), 'new verifier coverage incomplete')
    require(capture['status'] == 'NUMERICS_AND_PLACEMENT_PASS'
            and capture['source']['source_sha256']['_verify_distributed.py'] == boundary['verifier_sha256'],
            'new verifier bytes were not qualified')
    qualified = {name: r['after_sha256'] for name, r in retained.items()}
    qualified.update({HOST: boundary['host_native_sha256'], VERIFIER: boundary['verifier_sha256']})
    require(set(document['wheels']) == set(identity['wheels']) == set(old['candidate']['wheels']),
            'candidate wheel coverage differs')
    for package, row in document['wheels'].items():
        require(row['before']['sha256'] == old['candidate']['wheels'][package]
                and row['after']['sha256'] == identity['wheels'][package], 'candidate wheel identity differs')
        require(delta(row['before'], row['after'], old['candidate']['version'], identity['version'], qualified)
                == row['changed'], 'candidate wheel changes differ')
        if package == 'mojolearn-core':
            original = contents(row['before'], old['candidate']['version'])
            for name, item in retained.items():
                require(hashlib.sha256(original[name]).hexdigest() == item['before_sha256'],
                        'old host wheel differs from its admitted producer')
    return True


def errors(document, identity):
    try:
        validate(document, identity)
        return []
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AssertionError) as error:
        return [str(error)]


def apple_adapters(document, wheel_sha256):
    """Retain the installed adapter result when its whole runtime is unchanged."""
    import adapter_gate
    require(document.get('schema') == 'mojolearn.release.apple-adapter-reuse.v1'
            and document['after']['sha256'] == wheel_sha256, 'Apple reuse identifies another wheel')
    if 'verifier_update' in document:
        prior = reuse.read(document['retained_gate'])
        require('verifier_update' not in prior and prior['after'] == document['before'],
                'Apple fixture repack needs the original qualified wheel')
        apple_adapters(prior, document['before']['sha256'])
        qualified = verifier_update(document['verifier_update'])
        require(delta(document['before'], document['after'], document['version'],
                      document['version'], qualified) == document['changed'],
                'Apple fixture repack changes differ')
        return True
    stamp = reuse.read(document['original_wheel_stamp'])
    require(stamp['wheel_sha256'] == document['before']['sha256'], 'original adapter wheel differs')
    report = reuse.read(document['original_adapter_report'])
    _, passed, reason = adapter_gate.judge(report)
    require(passed, reason)
    boundary = reuse.read(document['boundary'])
    require(boundary.get('schema') == 'mojolearn.release.apple-host-boundary.v1'
            and boundary.get('passed') is True and boundary['wheel_sha256'] == wheel_sha256,
            'missing installed Apple host boundary')
    require([c['vendor'] for c in boundary['checks']] == ['cpu', 'apple'], 'missing CPU/Apple boundary')
    for check in boundary['checks']:
        require(check['exit_code'] == 0 and check['cases'] == 4, 'failed Apple host boundary')
        reuse.checked_ref(check['log'])
        cases = list(ET.fromstring(evidence.read_bytes(reuse.checked_ref(check['junit']))).iter('testcase'))
        require(len(cases) == 4 and not any(c.find(t) is not None
                for c in cases for t in ('failure', 'error', 'skipped')), 'incomplete Apple host boundary')
    qualified = {'mojolearn/host/_mojolearn_dev_tokenizer_host.so': boundary['host_native_sha256'],
                 VERIFIER: boundary['verifier_sha256']}
    require(delta(document['before'], document['after'], document['old_version'],
                  document['version'], qualified) == document['changed'], 'Apple wheel changes differ')
    return True
