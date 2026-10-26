"""Byte-boundary controls for incremental installed qualification."""
import zipfile

import pytest

import release_gate_reuse as reuse


def wheel(path, files):
    with zipfile.ZipFile(path, 'w') as z:
        for name, value in files.items():
            z.writestr(name, value)
    return path


@pytest.mark.parametrize('name', ['mojolearn/host/model.so', 'mojolearn/.libs/libAsyncRT.so',
    'mojolearn/adapters/torch.py', 'mojolearn-0.8.59.dist-info/METADATA',
    'mojolearn/cuda_native/sm_89/_adapters/mojolearn_torch.so'])
def test_refuses_changes_outside_qualified_boundary(tmp_path, name):
    before = wheel(tmp_path/'before.whl', {name: 'before'})
    after = wheel(tmp_path/'after.whl', {name: 'after'})
    with pytest.raises(ValueError, match='outside'):
        reuse.wheel_delta(before, after)


def test_verifier_and_native_change_do_not_imply_other_native_change(tmp_path):
    files = {'mojolearn/_identity.py':'old', 'mojolearn/cuda_native/sm_89/_adapters/libmojolearn_adapter.so':'old',
             'mojolearn/host/model.so':'same'}
    before = wheel(tmp_path/'before.whl', files)
    changed = {key: 'new' if value == 'old' else value for key,value in files.items()}
    after = wheel(tmp_path/'after.whl', changed)
    assert len(reuse.wheel_delta(before, after)) == 2


def test_removed_verifier_not_admitted(tmp_path):
    before = wheel(tmp_path/'before.whl', {'mojolearn/_identity.py':'old'})
    after = wheel(tmp_path/'after.whl', {})
    with pytest.raises(ValueError, match='outside'):
        reuse.wheel_delta(before, after)


def test_admission_bundle_reads_retained_bytes_without_original_paths(tmp_path):
    import hashlib
    import release_evidence
    raw = b'{"retained": true}'
    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path/digest).write_bytes(raw)
    original = '/absent/release/retained.json'
    record = dict(path=original, sha256=digest)
    with release_evidence.evidence_context(objects=tmp_path, mapping={original:digest}):
        assert reuse.read(record) == {'retained':True}
