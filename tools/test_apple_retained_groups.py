"""Retained Apple admission controls; no compilation, signing or native import."""
import copy
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

import apple_retained_groups as a
import bincache
import binding_stamps
import release_host_composition as w

OLD='b'*40
NEW='c'*40
COMPILER='d'*64


def raw(doc):
    text=json.dumps(doc,sort_keys=True)
    return dict(utf8=text,sha256=w.hashlib.sha256(text.encode()).hexdigest())


@pytest.fixture
def admitted(tmp_path,monkeypatch):
    root=tmp_path/'source';root.mkdir()
    dependencies={}
    for name in binding_stamps.MERGED_INPUTS:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# unchanged compiler input\n')
        dependencies[name]=w.file_sha(p)
    inventory=raw(dependencies)
    identities={'device':dict(closure={'digest':'e'*64})}
    capture=dict(source_commit=OLD,compiler_sha256=COMPILER,host_toolchain={'metal':'same'},
        binding_identities=identities,source_inventory_sha256=inventory['sha256'],source_archive_sha256='f'*64)
    groups={}
    for stem in a.GROUPS:
        relative='python/mojolearn/identical/'+stem+'.so'
        fields=dict(schema=bincache.SCHEMA,script='bindings/merged_build.sh',args=['gpu','--group',stem],
            source={'digest':'a'*64,'scope':'tree'},toolchain={'packages':['same']},numeric_mode='identical',
            gpu_archs='',target_column='',build_env={'MOJOLEARN_NUMERIC_MODE':'identical'},
            repo_path=str(root),image='local',os={'id':'darwin'},device_arch='none',
            local={'outputs':[relative],'shell':'sh','host_toolchain':capture['host_toolchain'],'inputs':{'digest':'e'*64}})
        key=bincache.key_of(fields)
        artifact=tmp_path/(stem+'.so');artifact.write_bytes(('compiled:'+stem).encode())
        outcome='time\tbindings/merged_build.sh\tmiss+built-cached\t'+key+'\t1.0\t'+relative+'='+w.file_sha(artifact)+'\n'
        receipt=dict(complete=True,build_exit=0,stamp_exit=0,source_commit=OLD,group=stem,compiler_sha256=COMPILER,
            key=key,output_sha256=w.file_sha(artifact),cache_outcome=outcome)
        groups[stem]=dict(receipt=raw(receipt),before_build_key=raw(dict(key=key,fields=fields)),
            actual_cache_key=raw(fields),artifact=a.file_ref(artifact),original_source_commit=OLD,original_key=key,
            cache_outcome={'utf8':outcome,'sha256':w.hashlib.sha256(outcome.encode()).hexdigest()})
    old=dict(before_build=raw(capture),source_inventory=inventory,source_archive={'path':'retained.tar.gz','sha256':'f'*64},
        dependency_files=dependencies,merged_build_inputs=dependencies,groups=groups)
    document=dict(schema=a.SCHEMA,action='admit-original-device-groups',complete=True,
        admitted_source_commit=NEW,source_root=str(root),compiler_sha256=COMPILER,host_toolchain=capture['host_toolchain'],
        admitted_dependency_files=dependencies,binding_identities=identities,original=old)
    admission=tmp_path/'admission.json';a.write(admission,document)
    monkeypatch.setattr(a,'current_revision',lambda p:NEW)
    monkeypatch.setattr(a,'actual_compiler',lambda p:COMPILER)
    monkeypatch.setattr(a.sys,'platform','darwin')
    return root,admission,document


def invocation(admitted,tmp_path,stem=a.GROUPS[0]):
    root,admission,document=admitted
    fields=copy.deepcopy(w.decoded(document['original']['groups'][stem]['before_build_key'])['fields'])
    fields['source']['digest']='9'*64  # A host-only edit changed the all-tree key.
    fields['build_env'].update(MOJOLEARN_REUSE_PLAN='',MOJOLEARN_REUSE_DIR='')
    env={a.SWITCH:str(admission),a.PLACEMENTS:str(tmp_path/'placements'),
         'MOJOLEARN_BINCACHE_OUTPUTS':fields['local']['outputs'][0],
         'MOJOLEARN_BINCACHE_SHELL':'sh','MOJOLEARN_BINCACHE_OUT':str(tmp_path/'cache-proof')}
    return root,fields,env


def test_admitted_device_keeps_origin_and_never_creates_a_new_archive(admitted,tmp_path,monkeypatch):
    root,fields,env=invocation(admitted,tmp_path)
    monkeypatch.setattr(bincache,'local_fields',lambda *args:fields)
    monkeypatch.setattr(bincache,'run_tee',lambda *args:pytest.fail('must not compile an admitted device'))
    monkeypatch.setattr(bincache.subprocess,'call',lambda *args,**kw:pytest.fail('must not fall back to compilation'))
    cache=tmp_path/'cache'
    assert bincache.cmd_build_local(str(root/'bindings/merged_build.sh'),fields['args'],env,cache)==0
    assert not list(cache.glob('*.tar.gz'))
    receipt=json.loads((Path(env[a.PLACEMENTS])/(a.GROUPS[0]+'.json')).read_text())
    assert receipt['origin']=='retained-admitted' and receipt['original_source_commit']==OLD
    assert receipt['admitted_source_commit']==NEW and receipt['original_key']!=receipt['requested_current_key']
    assert receipt['new_compilation'] is False and receipt['new_cache_archive'] is False
    assert (root/receipt['output']).read_bytes()==('compiled:'+a.GROUPS[0]).encode()
    placement=json.loads((Path(env['MOJOLEARN_BINCACHE_OUT'])/'placements.jsonl').read_text())
    assert placement['archive_source_commit']==OLD


@pytest.mark.parametrize('broken',['flags','target','toolchain','dependency','compiler','missing-output','key-error','policy-refusal'])
def test_explicit_admission_failure_never_restarts_gpu_compilation(admitted,tmp_path,monkeypatch,broken):
    root,fields,env=invocation(admitted,tmp_path)
    if broken=='flags':fields['build_env']['MOJOLEARN_COMPILE_JOBS']='99'
    elif broken=='target':fields['gpu_archs']='sm_89'
    elif broken=='toolchain':fields['toolchain']={'packages':['changed']}
    elif broken=='dependency':(root/binding_stamps.MERGED_INPUTS[0]).write_text('changed flags\n')
    elif broken=='compiler':monkeypatch.setattr(a,'actual_compiler',lambda p:'f'*64)
    elif broken=='missing-output':env.pop('MOJOLEARN_BINCACHE_OUTPUTS')
    def get_fields(*args):
        if broken=='key-error':raise ValueError('no usable current cache key')
        return fields
    monkeypatch.setattr(bincache,'local_fields',get_fields)
    if broken=='policy-refusal':monkeypatch.setattr(bincache,'refusal',lambda *args:'changed-policy')
    monkeypatch.setattr(bincache,'run_tee',lambda *args:pytest.fail('must not compile after failed admission'))
    monkeypatch.setattr(bincache.subprocess,'call',lambda *args,**kw:pytest.fail('must not bypass failed admission'))
    assert bincache.cmd_build_local(str(root/'bindings/merged_build.sh'),fields['args'],env,tmp_path/'cache')==2
    assert not (root/fields['local']['outputs'][0]).exists()


@pytest.mark.parametrize('broken',['raw-bytes','receipt-source','receipt-output','original-key','invented-build-exit','failed-original'])
def test_changed_original_proof_or_artifact_is_refused(admitted,tmp_path,broken):
    root,fields,env=invocation(admitted,tmp_path)
    document=json.loads(admitted[1].read_text());group=document['original']['groups'][a.GROUPS[0]]
    if broken=='raw-bytes':Path(group['artifact']['path']).write_bytes(b'changed')
    elif broken=='original-key':group['original_key']='e'*64
    elif broken=='invented-build-exit':document['build_exit']=0
    else:
        receipt=w.decoded(group['receipt'])
        if broken=='receipt-source':receipt['source_commit']='e'*40
        elif broken=='receipt-output':receipt['output_sha256']='e'*64
        else:receipt['build_exit']=1
        group['receipt']=raw(receipt)
    a.write(admitted[1],document)
    with pytest.raises(ValueError):
        a.try_place(root,str(root/'bindings/merged_build.sh'),fields['args'],env,fields['local']['outputs'],
                    fields,bincache.key_of(fields),bincache.Record(tmp_path/'proof'),0)


def test_host_command_is_left_for_normal_production_builder(admitted,tmp_path):
    root,fields,env=invocation(admitted,tmp_path)
    assert a.try_place(root,'bindings/merged_build.sh',['host'],env,['host.so'],fields,'key',None,0) is None


def test_final_wheel_proof_preserves_raw_staged_and_packed_identity(admitted,tmp_path,monkeypatch):
    root,admission,document=admitted
    def production(argv,cwd,env,**kwargs):
        assert argv==['sh','packaging/macos/build_release_wheel.sh','--build-only']
        for stem in a.GROUPS:
            _,fields,_=invocation(admitted,tmp_path,stem)
            a.try_place(root,'bindings/merged_build.sh',fields['args'],env,fields['local']['outputs'],
                fields,bincache.key_of(fields),bincache.Record(tmp_path/'proof'),0)
            (root/fields['local']['outputs'][0]).write_bytes(('staged:'+stem).encode())
        dist=root/'python/dist';dist.mkdir(parents=True)
        with zipfile.ZipFile(dist/'mojolearn-0.8.57-py3-none-macosx_11_0_arm64.whl','w') as archive:
            for stem in a.GROUPS:archive.writestr('mojolearn/identical/'+stem+'.so',('packed:'+stem).encode())
        return subprocess.CompletedProcess(argv,0)
    monkeypatch.setattr(a.subprocess,'run',production)
    report=a.pack(admission,root,tmp_path/'pack-evidence',tmp_path/'cache')
    assert report['pack_exit']==0 and 'build_exit' not in report
    assert len(report['artifact_origins'])==len(a.GROUPS) and report['admission']==w.witness(admission)
    for row in report['artifact_origins'].values():
        assert row['origin']=='retained-admitted' and row['original_source_commit']==OLD
        assert len({row['raw_compiler_sha256'],row['staged_tree_sha256'],row['wheel_sha256']})==3
    assert a.read_ref(report['wheel']).is_file()
