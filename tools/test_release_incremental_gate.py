"""Tampering controls for cross-version reuse and installed boundary evidence."""
import hashlib
import json
import zipfile

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


def test_removed_member_is_refused(tmp_path):
    before = wheel(tmp_path / 'old.whl', {gate.HOST: b'old'})
    after = wheel(tmp_path / 'new.whl', {})
    with pytest.raises(ValueError, match='coverage'):
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
