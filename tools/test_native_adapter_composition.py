"""Reject native substitutions that invalidate retained frontend evidence."""
import copy
import hashlib
import json

import pytest

import native_adapter_composition as n


def witness(value):
    raw = json.dumps(value)
    return dict(utf8=raw, sha256=hashlib.sha256(raw.encode()).hexdigest())


@pytest.fixture
def case():
    command = ['pixi', 'run', 'mojo', 'build', '-j', '2', '--emit', 'shared-lib',
               '--target-cpu', 'x86-64-v3', '--target-accelerator', 'sm_89',
               '-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1', '-D', 'MOJOLEARN_COLUMN_NVIDIA',
               '-I', '.', '-I', 'bindings', 'adapters/native/c_api.mojo', '-o', '/old/native.so']
    old = dict(source_commit='a' * 40, sources={'native': 'old', 'abi': 'same'},
               commands=dict(native=command), files={role:dict(name=role, bytes=1, sha256=role)
                   for role in ('native', 'torch', 'max')}, unstaged=dict(native={}))
    new = copy.deepcopy(old)
    new.update(source_commit='b' * 40, source_commit_scope='package', artifact_origins={},
               sources={'native': 'new', 'abi': 'same'})
    new['commands']['native'][-1] = '/new/native.so'
    new['files']['native']['sha256'] = 'staged'
    new['unstaged']['native'] = dict(name='libmojolearn_adapter.so', sha256='raw', bytes=2)
    before = dict(adapter_sources=old['sources'], adapter_builders={'builder': 'same'})
    current = dict(adapter_sources=new['sources'], adapter_builders={'builder': 'same'})
    refresh = dict(schema='mojolearn.linux.native-adapter-composition.v1',
        build=dict(component='adapter-native', state='built', vendor='nvidia', target='sm_89', sha256='raw'),
        compiler_sha256='compiler', captured_inputs=new['sources'], frontend_inputs={'torch': {'abi': 'same'}},
        original_manifest=witness(old), manifest=witness(new), command=new['commands']['native'],
        staged_sha256='staged', unstaged_bytes=2)
    adapters = {'mojolearn/cuda/sm_89/_adapters':dict(files={'adapters.json': refresh['original_manifest']['sha256'],
        'libmojolearn_adapter.so':'old', 'mojolearn_torch.so':'torch', 'mojolearn_ops.mojoc':'max'},
        manifest_sha256=refresh['original_manifest']['sha256'])}
    return refresh, before, current, 'compiler', 'cuda/sm_89', adapters


def test_admits_only_native_and_manifest(case):
    result = n.validate(*case)['mojolearn/cuda/sm_89/_adapters']['files']
    assert result['libmojolearn_adapter.so'] == 'staged'
    assert result['mojolearn_torch.so'] == 'torch'
    assert result['mojolearn_ops.mojoc'] == 'max'


@pytest.mark.parametrize('change', ['compiler', 'target', 'frontend', 'closure', 'flags', 'manifest', 'bytes'])
def test_refuses_invalidated_evidence(case, change):
    refresh, before, current, *_ = case
    if change == 'compiler': refresh['compiler_sha256'] = 'other'
    elif change == 'target': refresh['build']['target'] = 'sm_80'
    elif change == 'frontend':
        before['adapter_sources']['abi'] = 'changed'
    elif change == 'closure': refresh['captured_inputs'] = {'unrelated': 'source'}
    elif change == 'flags': refresh['command'] = refresh['command'] + ['-D', 'UNPROVEN=1']
    elif change == 'manifest':
        doc = json.loads(refresh['manifest']['utf8']); doc['files']['torch']['sha256'] = 'changed'
        refresh['manifest'] = witness(doc)
    elif change == 'bytes': refresh['staged_sha256'] = 'wrong'
    with pytest.raises((ValueError, RuntimeError, SystemExit)):
        n.validate(*case)
