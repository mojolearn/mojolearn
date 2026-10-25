"""Host-only provenance controls; no compiler or native module is executed."""
import copy
import json
import shutil

import pytest

import release_host_build as hb
import release_host_composition as hc
import retained_release_composition as rc
from test_retained_release_composition import fixture, OLD, NEW, COMPILER
import binding_stamps


def host_only(fixture, tmp_path):
    source, old, _, before, current = fixture
    # The older fixture did not need complete source stamp fields; these inputs
    # describe all host closures independently of any device output.
    identities = {k:v for k,v in current['binding_identities'].items() if '/host/' in k}
    for identity in identities.values():
        identity['closure']['scope'] = 'closure'
        identity['script'] = binding_stamps.member_script(identity['name'])
    out = tmp_path / 'independent-host'
    staged = out / 'set'
    (staged / 'host').mkdir(parents=True)
    (out / 'raw').mkdir()
    (out / 'raw' / hc.HOST_FILE).write_bytes(b'raw-new-host')
    (staged / hb.HOST_PATH).write_bytes(b'staged-new-host')
    donor, donor_set = hc.inspect_build(old[0])
    donor_contract, _ = hc._record(donor, COMPILER)
    shutil.copytree(donor_set / '.libs', staged / '.libs')
    index = dict(format=1,files={hc.HOST_FILE:sorted(hc.expected_names())})
    hb.write(staged / hb.HOST_INDEX,index)
    inventory = current['source_inventory']
    for name in binding_stamps.MERGED_INPUTS:
        if name not in dict(inventory):
            inventory.append([name, 'e' * 64])
    merged_inputs = {name:dict(inventory)[name] for name in binding_stamps.MERGED_INPUTS}
    before['merged_build_inputs'] = current['merged_build_inputs'] = merged_inputs
    hb.write(out / 'source-inventory.json',inventory)
    hb.write(out / 'runtime-source.json',donor)
    capture = dict(schema=hb.CAPTURE_SCHEMA,phase='before-build',source_commit=NEW,
        source_inventory_sha256=hc.file_sha(out / 'source-inventory.json'),compiler_sha256=COMPILER,
        installed_packages=donor_contract['installed_packages'],toolchain=donor_contract['toolchain'],
        binding_identities=identities,runtime_identity=current['runtime_identity'],
        merged_build_inputs=merged_inputs,
        build_environment=hb.FLAGS,pixi_environment='gbmbench',command=hb.command('gbmbench'))
    hb.write(out / 'compile-inputs.json',capture)
    stamp = dict(schema=2,binding=hb.HOST_PATH,script='merged_build.sh',commit=NEW,
        members={i['name']:dict(script=i['script'],scope='closure',digest=i['closure']['digest']) for i in identities.values()})
    merged_hash = hc.hashlib.sha256()
    for name in binding_stamps.MERGED_INPUTS:
        merged_hash.update(name.encode()+b'\0'+merged_inputs[name].encode()+b'\n')
    stamp['merged_inputs'] = merged_hash.hexdigest()
    hb.write(out / 'binding-stamp.json',stamp)
    stage = dict(schema='mojolearn.host-staging.v1',action='stage-host-only',
        runtime_origin_proof_sha256=donor['proof']['sha256'],runtime_libraries=donor_contract['runtime_libraries'],
        runtime_byte_changes=[],raw_sha256=hc.file_sha(out / 'raw' / hc.HOST_FILE),
        staged_sha256=hc.file_sha(staged / hb.HOST_PATH),runpath=hb.RUNPATH,rpath=None,needed=['libRuntime.so'])
    hb.write(out / 'stage.json',stage)
    hb.write(out / 'readback.json',dict(schema='mojolearn.host-readback.v1',source_commit=NEW,
        members=[dict(name=name,vendor='cpu',column='cpu',numeric_mode=1) for name in sorted(hc.expected_names())]))
    proof = dict(schema=hb.SCHEMA,action='build-host-only',complete=True,build_exit=0,stamp_exit=0,
        stage_exit=0,readback_exit=0,source_commit=NEW,compiler_sha256=COMPILER,binding_layout='merged',
        release_modes=['identical'],compile_inputs_sha256=hc.file_sha(out / 'compile-inputs.json'),
        source_inventory=inventory,source_sha256=hb.inventory_sha(inventory),command=hb.command('gbmbench'),
        raw_host_sha256=stage['raw_sha256'],host_extension={hb.PREFIX+hb.HOST_PATH:stage['staged_sha256']},
        merged_indexes={hb.PREFIX+hb.HOST_INDEX:dict(sha256=hc.file_sha(staged / hb.HOST_INDEX),files=index['files'])},
        stamp_sha256=hc.file_sha(out / 'binding-stamp.json'),stage_sha256=hc.file_sha(out / 'stage.json'),
        readback_sha256=hc.file_sha(out / 'readback.json'),runtime_identity=current['runtime_identity'])
    hb.write(out / 'host-build-proof.json',proof)
    return out


def test_actual_host_only_proof_composes_without_gpu_host_producer(fixture, tmp_path):
    host = host_only(fixture,tmp_path)
    record, directory = hb.inspect(host)
    assert hb.validate_bytes(record,directory,COMPILER)
    proof = hc.decoded(record['proof'])
    assert 'extensions' not in proof and 'vendor' not in hc.decoded(record['capture'])
    assert hb.command('gbmbench')[-3:] == ['bash','bindings/merged_build.sh','host']
    source, old, _, _, _ = fixture
    result = rc.prepare(old[0],host,source,COMPILER,tmp_path/'assembled')
    assert rc.validate(result,source) and rc.validate_bytes(result,tmp_path/'assembled/set')
    assert result['original_host']['kind'] == hb.KIND
    assert result['original_device']['leg'] == 'cuda/sm_89'
    assert {r['source_commit'] for r in result['artifact_origins'].values()} == {OLD,NEW}
    assert result['host_extension']['mojolearn/cuda/sm_89/'+hb.HOST_PATH] == proof['host_extension'][hb.PREFIX+hb.HOST_PATH]
    assert not (directory/'identical').exists()
    assert 'build_exit' not in result


def change_witness(source,key,operation):
    doc = hc.decoded(source[key]);operation(doc)
    raw=json.dumps(doc)
    source[key]={'utf8':raw,'sha256':hc.hashlib.sha256(raw.encode()).hexdigest()}
    return doc


@pytest.mark.parametrize('changed', ['failed-build','boolean-build-exit','failed-readback','fake-gpu-output',
    'compiler','flags','gpu-command','host-closure','host-index','readback-coverage','readback-column',
    'runtime-bytes','runtime-producer','stage-runpath'])
def test_host_only_proof_cannot_claim_missing_or_different_work(fixture,tmp_path,changed):
    host=host_only(fixture,tmp_path);source,_=hb.inspect(host)
    proof=hc.decoded(source['proof'])
    target='proof'
    if changed=='failed-build': proof['build_exit']=1
    elif changed=='boolean-build-exit': proof['build_exit']=False
    elif changed=='failed-readback': proof['readback_exit']=1
    elif changed=='fake-gpu-output': proof['extensions']={}
    elif changed in ('compiler','flags','gpu-command','host-closure'):
        def mutate(d):
            if changed=='compiler': d['compiler_sha256']='f'*64
            elif changed=='flags': d['build_environment']['MOJOLEARN_GPU_ARCHS']='sm_89'
            elif changed=='gpu-command': d['command'][-1]='gpu'
            else: next(iter(d['binding_identities'].values()))['closure']['digest']='f'*64
        change_witness(source,'capture',mutate)
        proof['compile_inputs_sha256']=source['capture']['sha256']
    elif changed=='host-index':
        change_witness(source,'index',lambda d:d['files'][hc.HOST_FILE].pop())
        proof['merged_indexes'][hb.PREFIX+hb.HOST_INDEX]['sha256']=source['index']['sha256']
    elif changed.startswith('readback-'):
        def mutate(d):
            if changed=='readback-coverage': d['members'].pop()
            else: d['members'][0]['column']='nvidia'
        change_witness(source,'readback',mutate);proof['readback_sha256']=source['readback']['sha256']
    else:
        def mutate(d):
            if changed=='runtime-bytes': d['runtime_libraries']['libRuntime.so']='f'*64
            elif changed=='runtime-producer': d['runtime_origin_proof_sha256']='f'*64
            else: d['runpath']='$ORIGIN/build-box-only'
        change_witness(source,'stage',mutate);proof['stage_sha256']=source['stage']['sha256']
    raw=json.dumps(proof);source['proof']={'utf8':raw,'sha256':hc.hashlib.sha256(raw.encode()).hexdigest()}
    with pytest.raises(ValueError): hb.record(source,COMPILER)


def test_missing_original_compiler_bytes_and_changed_staged_host_are_refused(fixture,tmp_path):
    host=host_only(fixture,tmp_path);source,staged=hb.inspect(host)
    (staged/hb.HOST_PATH).write_bytes(b'changed')
    with pytest.raises(ValueError,match='staged host bytes'): hb.validate_bytes(source,staged,COMPILER)
    (host/'raw'/hc.HOST_FILE).unlink()
    with pytest.raises(FileNotFoundError): hb.inspect(host)
