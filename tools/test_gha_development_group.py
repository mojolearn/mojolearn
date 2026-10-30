"""Selected GHA compile wrapper contract; no network, rentals or compilation."""
import copy
import hashlib
import json
from pathlib import Path
import os
import subprocess

import pytest

import gha_development_group as wrapper


def test_actual_allocation_overrides_inherited_small_caps():
    env=wrapper.workers_environment({'MOJOLEARN_COMPILE_JOBS':'2'},'sm_89',4)
    assert env['MOJOLEARN_COMPILE_JOBS']=='4'
    assert env['MOJOLEARN_CUDA_CODE_FORMAT']=='native'
    assert env['MOJOLEARN_GPU_ARCHS']=='sm_89'


def test_ptx_and_host_routing_are_explicit():
    env=wrapper.workers_environment({},'sm_80',8)
    assert env['MOJOLEARN_CUDA_CODE_FORMAT']=='ptx'
    host=wrapper.workers_environment(env,'host',4)
    assert host['MOJOLEARN_TARGET_COLUMN']=='cpu'
    assert 'MOJOLEARN_GPU_ARCHS' not in host and 'MOJOLEARN_CUDA_CODE_FORMAT' not in host


@pytest.mark.parametrize('arch', ['gfx942', 'gfx950'])
def test_amd_routing_discards_cuda_state_and_uses_actual_allocation(arch):
    inherited = {'MOJOLEARN_TARGET_COLUMN': 'nvidia',
                 'MOJOLEARN_CUDA_CODE_FORMAT': 'ptx', 'MOJOLEARN_COMPILE_JOBS': '2'}
    env = wrapper.workers_environment(inherited, arch, 8)
    assert env['MOJOLEARN_GPU_ARCHS'] == arch
    assert env['MOJOLEARN_TARGET_COLUMN'] == 'amd'
    assert env['MOJOLEARN_COMPILE_JOBS'] == '8'
    assert 'MOJOLEARN_CUDA_CODE_FORMAT' not in env
    assert inherited['MOJOLEARN_CUDA_CODE_FORMAT'] == 'ptx'


@pytest.mark.parametrize('mode,expected', [
    ('development', 'MOJOLEARN_COMMIT must be a full SHA'),
    ('shard', 'unknown arch gfx950'),
])
def test_gfx950_box_selection_is_limited_to_development(mode, expected):
    # Fail at source admission before any directories, network or compiler
    # work. This exercises the actual shell dispatcher, not a copied parser.
    root = Path(wrapper.__file__).resolve().parents[1]
    result = subprocess.run(['bash', str(root / 'tools/gha_release_box.sh'), mode, 'gfx950'],
                            env=dict(os.environ, MOJOLEARN_COMMIT='not-a-commit'),
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert expected in result.stderr


@pytest.mark.parametrize('cores',[0,-1,True,1.5,'4'])
def test_invalid_allocation_refused(cores):
    with pytest.raises(ValueError):wrapper.workers_environment({},'host',cores)


@pytest.fixture
def result(tmp_path):
    group='_mojolearn_dev_core_host';key='a'*64
    directory=tmp_path/'artifacts'/key;directory.mkdir(parents=True)
    binary=directory/(group+'.so');binary.write_bytes(b'actual retained bytes')
    fields={'recorded':'inputs'}
    receipt=dict(fields=fields,exit_code=0,sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (directory/'receipt.json').write_text(json.dumps(receipt))
    report=dict(schema='mojolearn.development-build.v1',state='incomplete',missing=['host:other'],
                plan=dict(groups=[dict(stem=group,key=key,fields=fields)]),
                results=[dict(key=key,exit_code=0)])
    return report,group,tmp_path,binary


def test_selected_success_does_not_require_unrelated_groups(result):
    report,group,cache,binary=result
    directory,row=wrapper.selected_result(report,group,cache)
    assert directory/binary.name==binary
    assert report['state']=='incomplete' and report['missing']==['host:other']


@pytest.mark.parametrize('change',['failed-group','failed-support','missing-result','failed-producer','wrong-bytes','wrong-fields','duplicate-group'])
def test_failed_or_unbound_evidence_refused(result,change):
    report,group,cache,binary=result
    if change=='failed-group':report['results'][0]['exit_code']=1
    if change=='failed-support':report['results'].append(dict(key='support',exit_code=1))
    if change=='missing-result':report['results']=[]
    if change=='failed-producer':report['state']='failed'
    if change=='wrong-bytes':binary.write_bytes(b'changed')
    if change=='wrong-fields':report['plan']['groups'][0]['fields']={'changed':'inputs'}
    if change=='duplicate-group':report['plan']['groups'].append(copy.deepcopy(report['plan']['groups'][0]))
    with pytest.raises(ValueError):wrapper.selected_result(report,group,cache)


def test_manual_workflow_retains_complete_artifacts_without_rental():
    root=Path(__file__).resolve().parents[1]
    text=(root/'.github/workflows/development-group-build.yml').read_text()
    assert 'workflow_dispatch:' in text and '\n  push:' not in text and '\n  schedule:' not in text
    assert 'fail-fast: false' in text and 'if: always()' in text
    assert 'strategy.job-index' in text and 'GHA_BUILD_JOBS=1' in text
    assert 'container.log' in text and 'container.exit' in text
    assert 'timeout-minutes: 50' in text and 'timeout-minutes: 60' in text
    assert 'runpod' not in text.lower()
    assert 'DEVELOPMENT_GROUP_SELECTORS:' in text and '\n          GROUPS:' not in text


@pytest.fixture
def input_bytes(tmp_path):
    root=tmp_path/'source';cache=tmp_path/'cache';out=tmp_path/'out'
    out.mkdir();(root/'tokenizer/impl').mkdir(parents=True)
    generated=root/'tokenizer/impl/unicode_table_generated.mojo'
    generated.write_text('generated pinned Unicode bytes\n')
    helper=root/'python/mojolearn/.libs/libMojolearnMath.so'
    helper.parent.mkdir(parents=True);helper.write_bytes(b'actual runtime bytes')
    key='b'*64;roots=cache/'work'/key/'roots';roots.mkdir(parents=True)
    (roots/'group.mojo').write_text('actual generated entry\n')
    row={'key':key,'fields':{'source':{'files':{
        str(generated.relative_to(root)):hashlib.sha256(generated.read_bytes()).hexdigest()}}}}
    return root,cache,row,out,generated,helper,roots


def test_generated_source_roots_and_runtime_bytes_are_retained(input_bytes):
    root,cache,row,out,generated,helper,roots=input_bytes
    retained=wrapper.retain_inputs(root,cache,row,out)
    assert len(retained)==3
    assert (out/'consumed-source/tokenizer/impl/unicode_table_generated.mojo').read_bytes()==generated.read_bytes()
    assert (out/'generated-roots/group.mojo').read_bytes()==(roots/'group.mojo').read_bytes()
    assert (out/'bootstrap-runtime/libMojolearnMath.so').read_bytes()==helper.read_bytes()
    assert json.loads((out/'retained-inputs.json').read_text())==retained


@pytest.mark.parametrize('change',['changed-source','missing-source','missing-root','missing-runtime','outside-source'])
def test_unbound_or_missing_generated_inputs_refused(input_bytes,change):
    root,cache,row,out,generated,helper,roots=input_bytes
    if change=='changed-source':generated.write_text('changed')
    if change=='missing-source':generated.unlink()
    if change=='missing-root':(roots/'group.mojo').unlink()
    if change=='missing-runtime':helper.unlink()
    if change=='outside-source':row['fields']['source']['files']={'../outside':'invalid'}
    with pytest.raises((ValueError,FileNotFoundError)):
        wrapper.retain_inputs(root,cache,row,out)


def test_failed_group_keeps_completed_support_and_entry_bytes_on_output_mount(tmp_path, monkeypatch):
    import argparse
    import subprocess
    import sys
    import types

    root=tmp_path/'source';out=tmp_path/'mounted-output'
    (root/'tools').mkdir(parents=True)
    (root/'tools/development_build.py').write_text('frozen producer')
    helper=root/'python/mojolearn/.libs/libMojolearnMath.so'
    helper.parent.mkdir(parents=True);helper.write_bytes(b'bootstrap runtime')
    commit='c'*40
    monkeypatch.setitem(sys.modules,'build_sizing',types.SimpleNamespace(box_resources=lambda:(4,16<<30)))
    monkeypatch.setitem(sys.modules,'development_build',types.SimpleNamespace(selected_group=lambda *args:True))
    monkeypatch.setitem(sys.modules,'merged_bindings',types.SimpleNamespace(
        development_members=lambda *args:{'_mojolearn_dev_example':['example']}))
    monkeypatch.setattr(wrapper.os,'sched_getaffinity',lambda pid:{0,1,2,3},raising=False)
    monkeypatch.setattr(wrapper.subprocess,'check_output',lambda cmd,**kw:commit+'\n' if 'rev-parse' in cmd else '')

    def build(cmd, **kw):
        if 'archive' in cmd:
            kw['stdout'].write(b'actual frozen source archive')
            return subprocess.CompletedProcess(cmd,0)
        cache=Path(cmd[cmd.index('--cache')+1])
        assert cache==out/'cache'
        support=cache/'artifacts'/'completed-support';support.mkdir(parents=True)
        (support/'libMojolearnMath.so').write_bytes(b'completed support binary')
        entry=cache/'work'/'failed-selected'/'roots';entry.mkdir(parents=True)
        (entry/'selected.mojo').write_text('actual failed compiler entry')
        assert (out/'bootstrap-runtime/libMojolearnMath.so').read_bytes()==b'bootstrap runtime'
        return subprocess.CompletedProcess(cmd,1)

    monkeypatch.setattr(wrapper.subprocess,'run',build)
    assert wrapper.run(argparse.Namespace(root=root,out=out,commit=commit,
                                        group='identical:example',arch='sm_80'))==1
    receipt=json.loads((out/'request.json').read_text())
    assert receipt['status']=='failed' and receipt['exit_code']==1
    assert (out/'cache/artifacts/completed-support/libMojolearnMath.so').read_bytes()==b'completed support binary'
    assert (out/'cache/work/failed-selected/roots/selected.mojo').read_text()=='actual failed compiler entry'
    assert 'retained_key' not in receipt


def test_timeout_collection_retains_generated_checkout_input():
    root=Path(__file__).resolve().parents[1]
    text=(root/'.github/workflows/development-group-build.yml').read_text()
    start=text.index('      - name: Retain generated source')
    stop=text.index('      - uses: actions/upload-artifact',start)
    step=text[start:stop]
    assert 'if: always()' in step
    assert 'source/tokenizer/impl/unicode_table_generated.mojo' in step
    assert 'development-out/generated-source' in step
