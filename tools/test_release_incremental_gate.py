"""Tampering controls for cross-version reuse and installed boundary evidence."""
import hashlib
import json
import zipfile
import copy

import pytest

import release_incremental_gate as gate


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def wheel(path, files):
    with zipfile.ZipFile(path, 'w') as z:
        for name, value in files.items():
            z.writestr(name, value)
    return ref(path)


def test_version_only_change_keeps_runtime_bytes(tmp_path):
    before = wheel(tmp_path / 'old.whl', {'pkg-1.dist-info/METADATA': b'Version: 1\nRequires-Dist: core==1\n',
                                       'mojolearn/host/a.so': b'native'})
    after = wheel(tmp_path / 'new.whl', {'pkg-2.dist-info/METADATA': b'Version: 2\nRequires-Dist: core==2\n',
                                      'mojolearn/host/a.so': b'native'})
    assert gate.delta(before, after, '1', '2', {}) == ['pkg-VERSION.dist-info/METADATA']


@pytest.mark.parametrize('name', ['mojolearn/adapters/torch.py', 'mojolearn/.libs/lib.so',
                                 'mojolearn/host/other.so', 'mojolearn/_backend.py'])
def test_unqualified_runtime_change_is_refused(tmp_path, name):
    before = wheel(tmp_path / 'old.whl', {name: b'original'})
    after = wheel(tmp_path / 'new.whl', {name: b'changed'})
    with pytest.raises(ValueError, match='unqualified'):
        gate.delta(before, after, '1', '2', {})


def test_component_requires_exact_tested_bytes(tmp_path):
    before = wheel(tmp_path / 'old.whl', {gate.HOST: b'old'})
    after = wheel(tmp_path / 'new.whl', {gate.HOST: b'new'})
    with pytest.raises(ValueError, match='component bytes'):
        gate.delta(before, after, '1', '2', {gate.HOST: hashlib.sha256(b'other build').hexdigest()})
    assert gate.delta(before, after, '1', '2', {gate.HOST: hashlib.sha256(b'new').hexdigest()}) == [gate.HOST]


def test_metadata_cannot_hide_dependency_change(tmp_path):
    before = wheel(tmp_path / 'old.whl', {'pkg-1.dist-info/METADATA': b'Version: 1\nRequires-Dist: numpy==3\n'})
    after = wheel(tmp_path / 'new.whl', {'pkg-2.dist-info/METADATA': b'Version: 2\nRequires-Dist: numpy==4\n'})
    with pytest.raises(ValueError, match='beyond version'):
        gate.delta(before, after, '1', '2', {})


def test_gpu_plugin_registry_only_allows_version_change(tmp_path):
    name = 'pkg-1.dist-info/gpu_plugins.json'
    before = wheel(tmp_path / 'old.whl', {name: '{"version":"1","distribution":"core"}'})
    after = wheel(tmp_path / 'new.whl', {name.replace('-1.', '-2.'): '{"version":"2","distribution":"core"}'})
    assert gate.delta(before, after, '1', '2', {})
    bad = wheel(tmp_path / 'bad.whl', {name.replace('-1.', '-2.'): '{"version":"2","distribution":"other"}'})
    with pytest.raises(ValueError, match='beyond version'):
        gate.delta(before, bad, '1', '2', {})


def test_fixture_update_cannot_qualify_arbitrary_runtime_file(tmp_path):
    path = tmp_path / 'update.json'
    path.write_text(json.dumps(dict(schema='mojolearn.release.verifier-fixture-update.v1',
                                    files={'mojolearn/_backend.py': {}}, proofs={})))
    with pytest.raises(ValueError, match='coverage differs'):
        gate.verifier_update(ref(path))


def test_fixture_repack_requires_a_qualified_original_gate(tmp_path):
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(dict(schema=gate.SCHEMA, verifier_update={})))
    with pytest.raises(ValueError, match='original qualified'):
        gate.fixture_repack({'retained_gate': ref(path)}, {})


def test_removed_member_is_refused(tmp_path):
    before = wheel(tmp_path / 'old.whl', {gate.HOST: b'old'})
    after = wheel(tmp_path / 'new.whl', {})
    with pytest.raises(ValueError, match='coverage'):
        gate.delta(before, after, '1', '2', {})


def test_portable_math_report_only_reuses_unchanged_policy(tmp_path):
    old = dict(platform_math_free=True, binaries=[dict(file='mojolearn/a.so', sha256='before', math_imports=[])])
    new = dict(platform_math_free=True, binaries=[dict(file='mojolearn/a.so', sha256='after', math_imports=[])])
    before = wheel(tmp_path / 'old.whl', {'pkg-1.dist-info/portable-math.json': json.dumps(old)})
    after = wheel(tmp_path / 'new.whl', {'pkg-2.dist-info/portable-math.json': json.dumps(new)})
    assert gate.delta(before, after, '1', '2', {})
    new['platform_math_free'] = False
    after = wheel(tmp_path / 'bad.whl', {'pkg-2.dist-info/portable-math.json': json.dumps(new)})
    with pytest.raises(ValueError, match='portable-math policy'):
        gate.delta(before, after, '1', '2', {})


@pytest.fixture
def boundary(tmp_path):
    identity = dict(version='2', source_commit='freeze', wheels={'mojolearn-core': 'wheelhash'})
    checks, logs = {}, {}
    for name in ('venv', 'install', 'dependencies', 'self-test', 'forest'):
        path = tmp_path / (name + '.log')
        path.write_text(name + ' completed')
        logs[name] = ref(path)
        checks[name] = dict(exit_code=0, sha256=logs[name]['sha256'])
    junit = tmp_path / 'tests.xml'
    junit.write_text('<testsuite>' + '<testcase/>' * 4 + '</testsuite>')
    report = dict(schema='mojolearn.release.host-boundary.v1', passed=True, exit_code=0, cases=4,
                  version='2', source_commit='freeze', wheels={'mojolearn_core-2-py3.whl': 'wheelhash'},
                  checks=checks, junit=ref(junit))
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(report))
    doc = dict(boundary=ref(path), boundary_logs=logs, boundary_junit=ref(junit))
    return doc, identity, report, path


def test_complete_installed_boundary(boundary):
    doc, identity, _, _ = boundary
    assert gate.host_boundary(doc, identity)['cases'] == 4


def test_log_tampering_is_refused(boundary):
    doc, identity, _, _ = boundary
    from pathlib import Path
    Path(doc['boundary_logs']['forest']['path']).write_text('different run')
    with pytest.raises(ValueError, match='bytes changed'):
        gate.host_boundary(doc, identity)


def test_skipped_case_cannot_be_counted_as_pass(boundary):
    doc, identity, report, path = boundary
    from pathlib import Path
    junit = Path(doc['boundary_junit']['path'])
    junit.write_text('<testsuite><testcase><skipped/></testcase>' + '<testcase/>' * 3 + '</testsuite>')
    doc['boundary_junit'] = ref(junit)
    report['junit'] = ref(junit)
    path.write_text(json.dumps(report))
    doc['boundary'] = ref(path)
    with pytest.raises(ValueError, match='incomplete forest'):
        gate.host_boundary(doc, identity)


def test_other_candidate_wheel_is_refused(boundary):
    doc, identity, _, _ = boundary
    identity['wheels']['mojolearn-core'] = 'different wheel'
    with pytest.raises(ValueError, match='exact candidate'):
        gate.host_boundary(doc, identity)


@pytest.fixture
def apple_reuse(tmp_path):
    import adapter_gate
    def document(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return ref(path)
    old = wheel(tmp_path / 'old.whl', {'mojolearn/_adapters/a.so': b'adapter'})
    new = wheel(tmp_path / 'new.whl', {'mojolearn/_adapters/a.so': b'adapter'})
    checks = []
    for vendor in ['cpu', 'apple']:
        log, xml = tmp_path / (vendor + '.log'), tmp_path / (vendor + '.xml')
        log.write_text('completed')
        xml.write_text('<testsuite>' + '<testcase/>' * 4 + '</testsuite>')
        checks.append(dict(vendor=vendor, exit_code=0, cases=4, log=ref(log), junit=ref(xml)))
    return dict(schema='mojolearn.release.apple-adapter-reuse.v1', before=old, after=new,
                original_wheel_stamp=document('stamp.json', dict(wheel_sha256=old['sha256'])),
                original_adapter_report=document('adapters.json', dict(status='PASSED',
                    phases={name: dict(exit=0) for name in adapter_gate.PHASES})),
                boundary=document('boundary.json', dict(schema='mojolearn.release.apple-host-boundary.v1',
                    passed=True, wheel_sha256=new['sha256'], checks=checks,
                    host_native_sha256='host', verifier_sha256='verifier')),
                old_version='1', version='2', changed=[])


def test_unchanged_apple_adapter_is_reused(apple_reuse):
    assert gate.apple_adapters(apple_reuse, apple_reuse['after']['sha256'])


def test_failed_original_adapter_cannot_be_reused(apple_reuse):
    from pathlib import Path
    path = Path(apple_reuse['original_adapter_report']['path'])
    path.write_text(json.dumps(dict(status='NO_FRAMEWORK')))
    apple_reuse['original_adapter_report'] = ref(path)
    with pytest.raises(ValueError, match='nothing ran'):
        gate.apple_adapters(apple_reuse, apple_reuse['after']['sha256'])


def test_apple_reuse_for_other_wheel_is_refused(apple_reuse):
    with pytest.raises(ValueError, match='another wheel'):
        gate.apple_adapters(apple_reuse, 'other wheel')


@pytest.fixture
def host_builds():
    name = '_mojolearn_native_host.so'
    before = dict(admitted_inputs=dict(binding_identities={
        'linux-64/host/-/host/_mojolearn_core_host': dict(source='closure', compiler='compiler', flags='flags')}),
        merged_indexes={'mojolearn/cuda/sm_89/host/_mojolearn_merged.json':
                        dict(files={name: ['_mojolearn_core_host']})},
        host_extension={'mojolearn/cuda/sm_89/host/' + name: 'original-binary'})
    after = copy.deepcopy(before)
    after['host_extension']['mojolearn/cuda/sm_89/host/' + name] = 'other-original-compilation'
    return before, after


def test_separate_compilation_with_same_inputs_reuses_evidence(host_builds):
    result = gate.retained_host_groups(*host_builds)
    assert result['mojolearn/host/_mojolearn_native_host.so']['compilation_inputs_equal']


@pytest.mark.parametrize('field', ['source', 'compiler', 'flags'])
def test_changed_compile_input_cannot_reuse_host_group(host_builds, field):
    before, after = host_builds
    after['admitted_inputs']['binding_identities']['linux-64/host/-/host/_mojolearn_core_host'][field] = 'changed'
    assert gate.retained_host_groups(before, after) == {}


def test_changed_group_membership_cannot_reuse_host_group(host_builds):
    before, after = host_builds
    next(iter(after['merged_indexes'].values()))['files']['_mojolearn_native_host.so'].append('_another')
    with pytest.raises(ValueError, match='membership'):
        gate.retained_host_groups(before, after)
