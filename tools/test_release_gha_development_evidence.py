"""Producer proof only: no native compilation or runtime."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import release_gha_development_evidence as gha
import release_linux_development_evidence as linux
from test_release_linux_development_evidence import producer, put, Tree


@pytest.fixture
def gha_producer(producer, tmp_path, monkeypatch):
    original, binary, receipt_path, stage = producer
    base = tmp_path/'gha/development';base.mkdir(parents=True)
    result = json.loads(Path(original['result']).read_text())
    result['source_commit']='a'*40
    row=result['plan']['groups'][0];key=row['key']
    shutil.copytree(Path(original['result']).parent,base/'evidence')
    put(base/'evidence/result.json',result)
    shutil.copytree(receipt_path.parent,base/'retained'/key)
    source=put(base/'source.tar.gz','retained source archive')
    root=put(base/'generated-roots'/f"{row['stem']}.mojo",'generated root')
    # Use the same actual canonical root text bound by the existing entry proof.
    import merged_bindings
    root.write_text(merged_bindings.root_text(row['stem'],row['members']))
    consumed=put(base/'consumed-source/a.mojo','source')
    bootstrap=put(base/'bootstrap-runtime/libMojolearnMath.so','runtime bytes')
    inputs={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root,consumed,bootstrap)}
    put(base/'retained-inputs.json',inputs)
    tooling={}
    for name in ('tools/gha_development_group.py','tools/gha_release_box.sh','.github/workflows/development-group-build.yml'):
        p=put(base/'tooling'/name,'tooling bytes '+name);tooling[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    old=Tree.sha256
    monkeypatch.setattr(Tree,'sha256',lambda self,name: 'b'*64 if name=='tools/development_build.py' else old(self,name))
    request=dict(schema=gha.SCHEMA,source_commit='a'*40,dirty_tracked='',status='selected-group-compiled',
        source_builder_sha256='b'*64,source_archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tooling_files=tooling,tooling_script_sha256=tooling['tools/gha_development_group.py'],
        exit_code=3,group=row['stem'],retained_key=key,members=row['members'],requested_group='identical:training',
        compiler_workers=32,build_jobs=1,arch='sm_89',environment=dict(row['fields']['environment'],MOJOLEARN_COMMIT='a'*40),
        retained_input_count=len(inputs),bootstrap_inputs={str(bootstrap.relative_to(base)):inputs[str(bootstrap.relative_to(base))]})
    put(base/'request.json',request)
    return dict(request=str(base/'request.json')),base


def test_selected_success_is_raw_compile_only(gha_producer):
    p,base=gha_producer;r=gha.collect('/source',{},p)
    assert len(r['units'])==1 and r['facts']['artifact_scope']=='raw-selected-compilation-only'
    assert 'leg' not in r['facts'] and 'stage' not in r['facts']
    assert 'staged_binary' not in next(iter(r['facts']['groups'].values()))
    assert r['configuration']['dependency_context']['selected_binding_compile_jobs']=={
        'linux-64/cuda/sm_89/identical/_mojolearn_training':'32'}


@pytest.mark.parametrize('damage',['dirty','source','builder','selector','member','target','workers','missing_workers','failed','exit','missing_input','root','archive','binary','execution','tooling'])
def test_bad_original_witness_refused(gha_producer,damage):
    p,b=gha_producer;r=json.loads((b/'request.json').read_text())
    if damage=='dirty':r['dirty_tracked']=' M a.mojo'
    elif damage=='source':r['source_commit']='f'*40
    elif damage=='builder':r['source_builder_sha256']='c'*64
    elif damage=='selector':r['requested_group']='identical:gbdt'
    elif damage=='member':r['members']=['_mojolearn_other']
    elif damage=='target':r['arch']='sm_80'
    elif damage=='workers':r['compiler_workers']=2
    elif damage=='missing_workers':r.pop('compiler_workers')
    elif damage=='failed':r['status']='failed'
    elif damage=='exit':r['exit_code']=1
    elif damage=='missing_input':(b/'consumed-source/a.mojo').unlink()
    elif damage=='root':next((b/'generated-roots').glob('*.mojo')).write_text('changed')
    elif damage=='archive':(b/'source.tar.gz').write_text('changed')
    elif damage=='binary':next((b/'retained').glob('*/*.so')).write_text('changed')
    elif damage=='execution':(b/'evidence/build.log.execution.json').write_text('{}')
    elif damage=='tooling':(b/'tooling/tools/gha_development_group.py').write_text('changed')
    put(b/'request.json',r)
    with pytest.raises((ValueError,OSError)):gha.collect('/source',{},p)


def test_self_consistent_changed_generated_bytes_still_refused(gha_producer):
    p,b=gha_producer;root=next((b/'generated-roots').glob('*.mojo'));root.write_text('other root')
    inputs=json.loads((b/'retained-inputs.json').read_text());inputs[str(root.relative_to(b))]=hashlib.sha256(root.read_bytes()).hexdigest()
    put(b/'retained-inputs.json',inputs)
    with pytest.raises(ValueError,match='actual generated root'):gha.collect('/source',{},p)


def test_explicit_retained_tooling_location(gha_producer,tmp_path):
    p,b=gha_producer;other=tmp_path/'before-dispatch-tooling';shutil.move(b/'tooling',other)
    with pytest.raises(OSError):gha.collect('/source',{},p)
    p['tooling']=str(other)
    assert len(gha.collect('/source',{},p)['units'])==1


def test_unit_checks_flags_workers_sources_and_retained_references(gha_producer,monkeypatch):
    from test_release_linux_development_evidence import wanted
    import release_retained_compile_evidence as common
    p,_=gha_producer;found=gha.collect('/source',{},p);w=wanted(found)
    monkeypatch.setattr(common,'require_closure',lambda *args:['a.mojo'])
    rid,unit=next(iter(found['units'].items()))
    coverage,refs=gha.check_unit(rid,w,found['facts'],unit)
    assert coverage and refs
    for key,value in [('compile_jobs','2'),('gpu_archs','sm_80')]:
        bad=copy.deepcopy(w);bad['inputs']['binding']['flags'][key]=value
        with pytest.raises(ValueError,match='compiled flag differs'):
            gha.check_unit(rid,bad,found['facts'],unit)
    bad=copy.deepcopy(found['facts']);bad['groups'][unit['group']]['source_files']['a.mojo']='0'*64
    with pytest.raises(ValueError,match='required source was not compiled'):
        gha.check_unit(rid,w,bad,unit)
