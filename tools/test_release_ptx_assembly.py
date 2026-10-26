"""PTX shard provenance/admission tests; no compiler, GPU or native load."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

import bincache
import release_ptx_assembly as pa
import release_host_build as hb
import release_host_composition as hc
import retained_release_composition as rc
from test_retained_release_composition import fixture, OLD, NEW, COMPILER
from test_release_host_build import host_only

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'packaging/linux'))
import ptx_baseline

PAYLOAD=b'ELF-placeholder\0.version 8.1\n.target sm_80\n.address_size 64\n.visible .entry k() {\nret;\n}\n\0'
RECIPE=dict(source=dict(scope='tree',digest='d'*64,files=7),declared=dict(digest='e'*64,files=['bindings/merged_build.sh']),merged='f'*64)


def witness(value):
    raw=json.dumps(value) if not isinstance(value,str) else value
    return dict(utf8=raw,sha256=hashlib.sha256(raw.encode()).hexdigest())


def compilation(before, image='fixture'):
    toolchain=next(iter(rc._device_keys(pa.LEG,before).values()))['toolchain']
    groups={}
    for filename in pa.device_plan():
        rel='python/mojolearn/identical/'+filename
        fields=dict(schema=bincache.SCHEMA,script='bindings/merged_build.sh',args=['gpu','--group',filename[:-3]],
            source=RECIPE['source'],toolchain=toolchain,numeric_mode='identical',gpu_archs='sm_80',target_column='nvidia',
            device_arch='none',repo_path='/root/mojolearn',os=dict(machine='x86_64'),image=image,
            declared=dict(inputs=RECIPE['declared'],outputs=[rel],shell='bash'),
            build_env=dict(MODULAR_HOME='/root/mojolearn/.pixi/envs/default/share/max',MOJOLEARN_BUILD_PIXI_ENV='default',
                MOJOLEARN_BUILD_TIERS='identical',MOJOLEARN_COMPILE_JOBS='2',MOJOLEARN_CUDA_CODE_FORMAT='ptx',
                MOJOLEARN_GPU_ARCHS='sm_80',MOJOLEARN_LINUX_CPU='x86-64-v3',MOJOLEARN_MERGED_KEY=RECIPE['merged'],
                MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_PACKAGE_BYTE_LM='1',MOJOLEARN_TARGET_COLUMN='nvidia'))
        key=bincache.key_of(fields);raw_sha=hashlib.sha256(PAYLOAD).hexdigest();outcome='miss+built-not-uploaded:no-slot'
        manifest=dict(schema=bincache.SCHEMA,key=key,fields=fields,source_commit=OLD,installed_toolchain=toolchain['packages'],
            files=[dict(path=rel,sha256=raw_sha,size=len(PAYLOAD),mode=448)],build_log_sha256='1'*64)
        groups[filename]=dict(manifest=witness(manifest),key=witness(fields),raw_sha256=raw_sha,archive_sha256='2'*64,
            shard=witness('identical:merged_build.sh:'+filename[:-3]+'\tok\t'+outcome+'\t1.0\t'+key+'\n'),
            provenance=witness('date\tbindings/merged_build.sh\t'+outcome+'\t'+key+'\t1.0\t'+rel+'='+raw_sha+'\n'),
            archive_hashes=witness(key+'\t'+'2'*64+'\n'),
            build_log=dict(path='retained.log',sha256='3'*64,compiler_stdout_sha256='1'*64,scope='build_sets-wrapper-v1'),
            box=witness('Mojo 1.0.0 (fixture)\n'))
    return dict(schema='mojolearn.ptx-compiled-shards.v1',source_commit=OLD,groups=groups,source_inputs=before,source_recipe=RECIPE,
        compiler_evidence=dict(basis='locked and actually installed original compiler packages plus original Mojo version',observed_original_executable_sha256=None))


def staged(fixture,tmp_path,monkeypatch):
    source_root,_old,_gpu,before,current=fixture
    hostdir=host_only(fixture,tmp_path)
    # Legacy synthetic donor keys omit the real production image field.
    donor_source=json.loads((hostdir/'runtime-source.json').read_text())
    keys=hc.decoded(donor_source['host_cache_key']);dp=hc.decoded(donor_source['proof'])
    for filename,key in keys.items():
        key['image']='fixture-image'
        dp['bincache']['mojolearn/'+donor_source['leg']+'/host/'+filename]['key']=hc.digest(key)
    donor_source['host_cache_key']=witness(keys);donor_source['proof']=witness(dp)
    hb.write(hostdir/'runtime-source.json',donor_source)
    hs=json.loads((hostdir/'stage.json').read_text());hs['runtime_origin_proof_sha256']=donor_source['proof']['sha256'];hb.write(hostdir/'stage.json',hs)
    hp=json.loads((hostdir/'host-build-proof.json').read_text());hp['stage_sha256']=hc.file_sha(hostdir/'stage.json');hb.write(hostdir/'host-build-proof.json',hp)
    host,hostset=hb.inspect(hostdir);contract,_,_,_=hb.record(host,COMPILER)
    donor,_=hc._record(host['runtime_source'],COMPILER)
    image=next(iter(donor['host_cache_inputs'].values()))['image']
    compiled=compilation(before,image)
    monkeypatch.setattr(pa,'source_recipe',lambda root,revision:copy.deepcopy(RECIPE))
    out=tmp_path/'device';target=out/'set';(target/'identical').mkdir(parents=True)
    write=hb.write
    for filename in pa.device_plan():(target/'identical'/filename).write_bytes(PAYLOAD)
    write(target/'identical'/hc.INDEX,dict(format=1,files=pa.device_plan()))
    shutil.copytree(hostset/'.libs',target/'.libs')
    write(out/'runtime-host.json',host);write(out/'source-inventory.json',current['source_inventory'])
    capture=dict(phase='before-assembly',source_commit=NEW,compiler_sha256=COMPILER,compiler_observation_scope='assembly-only',
        installed_packages=contract['installed_packages'],source_inputs=current,binding_identities=rc._device_keys(pa.LEG,current),
        original_compiler_evidence=compiled['compiler_evidence'])
    write(out/'assembly-inputs.json',capture)
    stage=dict(device_outputs={f:dict(raw_sha256=hashlib.sha256(PAYLOAD).hexdigest(),sha256=hc.file_sha(target/'identical'/f),
        needed=['libRuntime.so'],runpath=pa.RUNPATH,rpath=None,architectures=['sm_80']) for f in pa.device_plan()},
        runtime_libraries=contract['runtime_libraries'],runtime_byte_changes=[],runtime_origin_proof_sha256=host['proof']['sha256'],
        dependency_check_exit=0,ptx_patch_exit=0,ptx_audit_exit=0)
    write(out/'stage.json',stage)
    members=[dict(name=n,vendor='cuda') for names in pa.device_plan().values() for n in names]
    write(out/'readback.json',dict(members=members,groups=pa.device_plan(),host_architectures={f:[] for f in hc.host_plan()}))
    hostrows=hc.decoded(host['readback'])['members']
    (target/'readback.txt').write_text(''.join('identical '+r['name']+' cuda\n' for r in members)+''.join('host '+r['name']+' cpu\n' for r in hostrows))
    (target/'arch_readback.txt').write_text(''.join('identical '+r['name']+' sm_80\n' for r in members)+''.join('host '+r['name']+' NONE-BY-DESIGN\n' for r in hostrows))
    write(out/'isa.json',dict(baseline='x86-64-v3',failing=0,report={path:{} for path in [*('identical/'+f for f in pa.device_plan()),*('host/'+f for f in hc.host_plan()),*('.libs/'+f for f in contract['runtime_libraries'])]}))
    write(target/'PTX_BASELINE.json',ptx_baseline.audit_tree(target,NEW,'Mojo 1.0.0 (fixture)'))
    hs=hc.decoded(host['stage'])
    write(target/'manifest.json',dict(extensions=[dict(path='identical/'+f,sha256=r['sha256']) for f,r in stage['device_outputs'].items()]
        +[dict(path='host/'+f,sha256=sha) for f,sha in hs['staged_sha256'].items()],
        portable_math=hc.decoded(host['runtime_source']['manifest']).get('portable_math'),
        staged_libs=[dict(name=name,sha256=sha) for name,sha in contract['runtime_libraries'].items()]))
    proof=dict(schema=pa.SCHEMA,action='stage-compiled-ptx-shards',complete=True,binding_layout='merged',release_modes=['identical'],
        source_commit=NEW,source_inventory=current['source_inventory'],source_sha256=rc.sha_inventory(current['source_inventory']),
        original_compilation=compiled,capture_sha256=hc.file_sha(out/'assembly-inputs.json'),
        extensions={pa.PREFIX+'identical/'+f:r['sha256'] for f,r in stage['device_outputs'].items()},host_extension={},adapters={},
        merged_indexes={pa.PREFIX+'identical/'+hc.INDEX:dict(sha256=hc.file_sha(target/'identical'/hc.INDEX),files=pa.device_plan())},
        bincache={pa.PREFIX+'identical/'+f:dict(compiled_from_commit=OLD) for f in pa.device_plan()},
        **{name+'_sha256':hc.file_sha(path) for name,path in dict(manifest=out/'stage.json',readback=out/'readback.json',isa=out/'isa.json',
            ptx=target/'PTX_BASELINE.json',set_manifest=target/'manifest.json',vendor_text=target/'readback.txt',arch_text=target/'arch_readback.txt').items()})
    write(out/'device-stage-proof.json',proof)
    return out,hostdir,source_root


def test_four_original_compiles_can_be_staged_and_composed_without_new_compilation(fixture,tmp_path,monkeypatch):
    out,hostdir,source_root=staged(fixture,tmp_path,monkeypatch)
    source,directory=pa.inspect(out)
    pa.record(source,COMPILER,source_root)
    assert pa.validate_bytes(source,directory,COMPILER)
    proof=rc.prepare(out,hostdir,source_root,COMPILER,tmp_path/'composed')
    assert rc.validate(proof,source_root) and rc.validate_bytes(proof,tmp_path/'composed/set')
    from staged_native_admission import complete_native_proof
    assert complete_native_proof(proof,source_root)
    assert 'build_exit' not in proof
    assert {row['source_commit'] for path,row in proof['artifact_origins'].items() if '/identical/' in path}=={OLD}
    doc=json.loads((tmp_path/'composed/set/PTX_BASELINE.json').read_text())
    assert rc.validate_ptx(doc,tmp_path/'composed/set')
    assert hc.decoded(source['capture'])['compiler_observation_scope']=='assembly-only'
    # Exercise the actual packer's set reader, including PTX admission and all
    # vendor/architecture/host coverage witnesses (no ELF/native execution).
    from pack_wheel import load_set
    monkeypatch.setenv('MOJOLEARN_RELEASE_MODES','identical')
    packed=tmp_path/'sets/cuda/sm_80';shutil.copytree(tmp_path/'composed/set',packed)
    sets=load_set(packed.parent,include_byte_lm=True)
    assert len(sets)==1 and sets[0].vendor=='cuda' and sets[0].arch=='sm_80'


@pytest.mark.parametrize('change',['flags','source','toolchain','installed','target','command','manifest-source','original-log','output','success','archive','coverage'])
def test_compilation_admission_rejects_missing_or_changed_inputs(fixture,change):
    compiled=compilation(fixture[3]);filename=next(iter(compiled['groups']));row=compiled['groups'][filename]
    fields=hc.decoded(row['key']);manifest=hc.decoded(row['manifest'])
    if change=='flags':fields['build_env']['MOJOLEARN_EXTRA']='1'
    elif change=='source':fields['source']=dict(fields['source'],digest='9'*64)
    elif change=='toolchain':fields['toolchain']={'packages':['wrong']}
    elif change=='target':fields['gpu_archs']='sm_89'
    elif change=='command':fields['args']=['host','--group',filename[:-3]]
    elif change=='installed':manifest['installed_toolchain']=['wrong']
    elif change=='manifest-source':manifest['source_commit']=NEW
    elif change=='original-log':row['build_log']['compiler_stdout_sha256']='9'*64
    elif change=='output':row['raw_sha256']='9'*64
    elif change=='success':row['shard']=witness(hc.raw(row['shard']).decode().replace('\tok\t','\tFAILED\t'))
    elif change=='archive':row['archive_sha256']='9'*64
    else:compiled['groups'].pop(filename)
    row['key']=witness(fields);row['manifest']=witness(manifest)
    with pytest.raises(ValueError):pa.validate_compilation(compiled)


@pytest.mark.parametrize('change',['invented-compile','compiler-scope','missing-readback','wrong-architecture','failed-isa','failed-ptx','changed-runtime'])
def test_complete_stage_rejects_unperformed_or_failed_work(fixture,tmp_path,monkeypatch,change):
    out,_,_=staged(fixture,tmp_path,monkeypatch);source,_=pa.inspect(out)
    key='proof'
    doc=hc.decoded(source[key])
    if change=='invented-compile':doc['build_exit']=0
    else:
        key={'compiler-scope':'capture','missing-readback':'readback','wrong-architecture':'arch_text','failed-isa':'isa','failed-ptx':'ptx','changed-runtime':'manifest'}[change]
        if key=='arch_text':source[key]=witness(hc.raw(source[key]).decode().replace('sm_80','sm_89'))
        else:
            data=hc.decoded(source[key])
            if change=='compiler-scope':data['compiler_observation_scope']='original-compilation'
            elif change=='missing-readback':data['members'].pop()
            elif change=='failed-isa':data['failing']=1
            elif change=='failed-ptx':data['errors']=['unpinned arithmetic']
            else:data['runtime_byte_changes']=['libRuntime.so']
            source[key]=witness(data)
        doc['capture_sha256' if key=='capture' else key+'_sha256']=source[key]['sha256']
    source['proof']=witness(doc)
    with pytest.raises(ValueError):pa.record(source,COMPILER)
