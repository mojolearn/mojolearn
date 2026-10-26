"""Reuse unchanged installed checks and run changed verifier/adapter boundaries.

Receipts explicitly distinguish old executions from new executions. They never
rewrite the source, wheel hashes, failures, or elapsed time of an original run.
"""
import hashlib
import json
from pathlib import Path
import zipfile

import adapter_gate
import release_gate_profiles as profiles
import release_host_composition as hc

SCHEMA = 'mojolearn.release.component-gate.v1'
VERIFIER = {'mojolearn/' + name for name in (
    '__main__.py', '_identity.py', '_identity_break.py', '_verify_reference.py',
    'verify_reference/table.json', 'identity_columns/COMMIT')}


def inventory(path):
    with zipfile.ZipFile(path) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}


def wheel_delta(before, after):
    left, right = inventory(before), inventory(after)
    changed = {name for name in set(left) | set(right) if left.get(name) != right.get(name)}
    unexpected = []
    for name in sorted(changed):
        # RECORD is derived from all package members. These two provenance
        # manifests change when the same compiled outputs are reassembled.
        allowed = (name in VERIFIER or name.endswith('.dist-info/RECORD')
                   or name.endswith('/LINUX_PAYLOAD.json') or name.endswith('/gpu_plugin.json')
                   or name.endswith('/PTX_BASELINE.json')
                   or name.endswith('/_adapters/adapters.json')
                   or name.endswith('/_adapters/libmojolearn_adapter.so'))
        if not allowed or name not in left or name not in right:
            unexpected.append(name)
    if unexpected:
        raise ValueError('changed outside verifier/adapter/assembly boundary: ' + ', '.join(unexpected))
    return sorted(changed)


def checked_ref(record, relocations=None):
    p = Path(record['path'])
    if not p.is_file():
        for old, new in (relocations or {}).items():
            if p.is_relative_to(old):
                p = Path(new) / p.relative_to(old)
                break
    if hc.file_sha(p) != record['sha256']:
        raise ValueError('evidence bytes changed: ' + str(p))
    return p


def read(record):
    return json.loads(checked_ref(record).read_text())


def errors(document, identity):
    try:
        validate(document, identity)
        return []
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AssertionError) as error:
        return [str(error)]


def validate(document, identity):
    from release_requirement_receipts import candidate_autograd_errors
    def require(ok, reason):
        if not ok:
            raise ValueError(reason)
    require(document.get('schema') == SCHEMA and document.get('candidate') == identity,
            'component gate does not identify exact final candidate')
    require(document.get('profile') == profiles.LIGHT and document.get('verdict') == 'PASS',
            'component gate is incomplete')
    original = read(document['retained_gate'])
    old = original['candidate']
    require(old['vendor'] == identity['vendor'] and old['version'] == identity['version'],
            'retained gate belongs to another vendor/version')
    require(original.get('correctness_rc') == 0 and original.get('autograd_rc') == 0,
            'retained execution did not succeed')
    reasons = profiles.correctness_errors(original['correctness'], old['source_commit'], old['version'], old['vendor'])
    reasons += candidate_autograd_errors(original, old)
    require(not reasons, '; '.join(reasons))
    require(document['correctness'] == original['correctness']
            and document['autograd_counts'] == original['autograd_counts'], 'retained records were rewritten')
    autograd = read(document['retained_autograd'])
    require(len(autograd['rows']) == sum(original['autograd_counts'].values())
            and all(row['status'] == 'ok' and row['rc'] == 0 for row in autograd['rows']),
            'retained autograd row coverage failed')
    require(set(document['wheels']) == set(identity['wheels']) == set(old['wheels']), 'wheel coverage differs')
    for package, row in document['wheels'].items():
        before, after = checked_ref(row['before']), checked_ref(row['after'])
        require(row['before']['sha256'] == old['wheels'][package]
                and row['after']['sha256'] == identity['wheels'][package], 'wheel identity differs')
        require(wheel_delta(before, after) == row['changed'], 'wheel byte comparison changed')
    boundary = read(document['boundary'])
    require(boundary.get('candidate') == identity and boundary.get('complete') is True,
            'changed boundary did not run on exact candidate')
    require(set(boundary['checks']) == {'install', 'dependencies', 'self-test', 'identity-cli', 'adapters'},
            'missing changed-boundary checks')
    for name, check in boundary['checks'].items():
        require(check['exit_code'] == 0 and not check['timed_out'], 'failed changed boundary: ' + name)
        checked_ref(check['log'], document.get('relocations'))
    adapters = read(document['adapter_report'])
    status, passed, reason = adapter_gate.judge(adapters)
    require(passed, reason)
    require(document['adapters']['phases'] == {name: 0 for name in adapter_gate.PHASES}, 'adapter phase coverage differs')
    # Retain the assembly proof as well as the wheel diff: source closures,
    # original compiler provenance and separately rebuilt native adapter bytes.
    import retained_release_composition as rc
    proof = read(document['assembly'])
    rc.validate(proof)
    require(proof['source_commit'] == identity['source_commit'] and proof.get('native_adapter_refresh'),
            'missing separate native adapter compilation admission')
    require(proof['original_device']['leg'] == {'nvidia':'cuda/sm_89','amd':'hip/gfx942'}[identity['vendor']],
            'assembly belongs to another device')
    package = 'mojolearn-' + identity['vendor']
    with zipfile.ZipFile(checked_ref(document['wheels'][package]['after'])) as wheel:
        prefix = 'mojolearn/' + {'nvidia':'cuda_native/sm_89','amd':'hip_native/gfx942'}[identity['vendor']] + '/_adapters/'
        for name, value in next(iter(proof['adapters'].values()))['files'].items():
            require(hashlib.sha256(wheel.read(prefix + name)).hexdigest() == value,
                    'shipped adapter differs from admitted producer: ' + name)
    return True
