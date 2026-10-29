"""Mac merged-root admission proves canonical exports; no builds/runtime."""
import copy
import hashlib
import json

import pytest

import merged_bindings as merged
import release_retained_compile_evidence as common

NAME = '_mojolearn_training'
ENTRY = f'bindings/entry/entry_{NAME}.mojo'


class Tree:
    def text(self, path):
        if path == 'tools/merged_bindings.py':
            return '\n'.join(k + ' = ' + repr(getattr(merged, k))
                             for k in ('ENTRY', 'ROOT_HEAD', 'ROOT_MEMBER', 'ROOT_TAIL'))
        if path == ENTRY:
            return merged.ENTRY.format(name=NAME)
        return 'substantive source\n'

    def sha256(self, path):
        return hashlib.sha256(self.text(path).encode()).hexdigest()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tree = Tree()
    stem = '_mojolearn_dev_training'
    fields = dict(stem=stem, members=[NAME],
                  root_sha256=hashlib.sha256(merged.root_text(stem, [NAME]).encode()).hexdigest())
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps(dict(fields=fields)))
    group = dict(members=[NAME], mode='identical', toolchain_locked={}, host_toolchain={},
                 environment={'MOJOLEARN_COMPILE_JOBS': '10'}, source_files={},
                 receipt=common.reference(receipt), binary={})
    key = f'osx-arm64/metal/apple/identical/{NAME}'
    wanted = dict(inputs=dict(binding=dict(target='osx-arm64', vendor='metal', arch='apple',
        tier='identical', name=NAME, toolchain={}, host_toolchain={}, flags={'compile_jobs':'10'}),
        unit={'build_flags':{}}),coverage=['compile',key])
    facts = dict(groups={stem:group},source_root='/source',source_revision='a'*40,result={})
    unit = dict(group=stem,binding_key=key)
    monkeypatch.setattr(common, 'supports_requirement', lambda w: True)
    monkeypatch.setattr(common.Tree, 'at', lambda *a: tree)
    monkeypatch.setattr(common, 'require_closure', lambda *a: [ENTRY])
    return tree,receipt,wanted,facts,unit


def check(data):
    _,_,wanted,facts,unit=data
    return common._dev_check('compile-apple/'+unit['binding_key'],wanted,facts,unit)


def test_mac_merged_root_proves_exact_public_entry(setup):
    rows,_=check(setup)
    assert len(rows)==2


def test_changed_recorded_root_is_refused(setup):
    p=setup[1];d=json.loads(p.read_text());d['fields']['root_sha256']='0'*64;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='generated development root differs'):
        check(setup)


def test_changed_standalone_entry_is_refused(setup,monkeypatch):
    tree=setup[0];original=tree.text
    monkeypatch.setattr(tree,'text',lambda p:original(p)+('\n# changed' if p==ENTRY else ''))
    with pytest.raises(ValueError,match='not the tested canonical wrapper'):
        check(setup)


def test_missing_substantive_source_is_never_excused(setup,monkeypatch):
    monkeypatch.setattr(common,'require_closure',lambda *a:[ENTRY,'algorithm.mojo'])
    with pytest.raises(ValueError,match='not compiled at its tested bytes: algorithm.mojo'):
        check(setup)


def test_wrong_logical_binding_is_refused(setup):
    setup[4]['binding_key']='osx-arm64/metal/apple/identical/_mojolearn_other'
    with pytest.raises(ValueError,match='admitted unit names another binding'):
        check(setup)


def test_present_entry_with_wrong_bytes_does_not_use_wrapper_proof(setup):
    setup[3]['groups'][setup[4]['group']]['source_files'][ENTRY]='0'*64
    with pytest.raises(ValueError,match='not compiled at its tested bytes'):
        check(setup)
