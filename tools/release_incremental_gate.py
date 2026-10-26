"""Admit unchanged release components plus exercised, byte-bound repairs.

Version and provenance changes alone do not invalidate runtime evidence. This
adapter deliberately supports only the forest input-admission and split-wheel
verifier repairs qualified by the retained producers, not arbitrary native edits.
"""
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile

import release_evidence as evidence
import release_gate_reuse as reuse

SCHEMA = 'mojolearn.release.incremental-gate.v1'
HOST = 'mojolearn/host/_mojolearn_native_tokenizer_host.so'
VERIFIER = 'mojolearn/_verify_distributed.py'


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
        elif name.endswith('.dist-info/METADATA') or name in (
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


def validate(document, identity):
    require(document.get('schema') == SCHEMA and document.get('candidate') == identity
            and document.get('verdict') == 'PASS' and document.get('profile') == 'release-light-v1',
            'incremental gate does not identify exact candidate')
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
    import retained_release_composition as composition
    import release_host_composition as hc
    assembly = reuse.read(document['assembly'])
    composition.validate(assembly)
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
    qualified = {HOST: boundary['host_native_sha256'], VERIFIER: boundary['verifier_sha256']}
    require(set(document['wheels']) == set(identity['wheels']) == set(old['candidate']['wheels']),
            'candidate wheel coverage differs')
    for package, row in document['wheels'].items():
        require(row['before']['sha256'] == old['candidate']['wheels'][package]
                and row['after']['sha256'] == identity['wheels'][package], 'candidate wheel identity differs')
        require(delta(row['before'], row['after'], old['candidate']['version'], identity['version'], qualified)
                == row['changed'], 'candidate wheel changes differ')
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
