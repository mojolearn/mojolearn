#!/usr/bin/env python3
"""Assemble already compiled sm_80 shards, without invoking a compiler.

Original archives, cache keys and compilation sources remain unchanged. The
result is a retained-composition admission, never a fabricated fresh build.
The original shard format records locked/installed compiler packages, not an
observed executable SHA; the assembly observation is explicitly separate.
"""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

import bincache
import release_host_build as hb
import release_host_composition as hc
import retained_release_composition as rc

SCHEMA = 'mojolearn.linux.ptx-shard-stage.v1'
KIND = 'compiled-ptx-shard-stage'
LEG = 'cuda/sm_80'
PREFIX = 'mojolearn/' + LEG + '/'
RUNPATH = '$ORIGIN/../.libs:$ORIGIN/../../.libs:$ORIGIN/../../../.libs'
# The original workflow's source export, retained in source_recipe(). Unknown
# changes to that export fail closed rather than rewriting a source-tree key.
EXCLUDES = ':!bench/results :!mamba/corpus :!bench/oracle_* :!bench/minentropy_oracle.txt'


def require(ok, why):
    hc.require(ok, 'PTX assembly: ' + why)


def write(path, value):
    return hb.write(path, value)


def device_plan():
    import merged_bindings
    return {stem+'.so': members for stem, _flags, members in merged_bindings.plan('identical', 'linux', True)}


def excluded(name):
    return name.startswith(('bench/results/', 'mamba/corpus/', 'bench/oracle_')) or name == 'bench/minentropy_oracle.txt'


@contextmanager
def source_tree(root, revision):
    """Reproduce committed, curated source bytes; no checkout or execution."""
    workflow = subprocess.check_output(['git','-C',str(root),'show',revision+':.github/workflows/release-linux-build.yml'], text=True)
    export = re.search(r'^  EXCLUDES: "([^"]+)"$', workflow, re.M)
    require(export is not None and export.group(1) == EXCLUDES,
            'unrecognized original source export recipe')
    with tempfile.TemporaryDirectory(prefix='mojolearn-ptx-inputs-') as temporary:
        process = subprocess.Popen(['git','-C',str(root),'archive',revision], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                name = member.name
                wanted = name.endswith('.mojo') or name.startswith(('bindings/','tokenizer/','tools/')) or name in ('pixi.toml','pixi.lock')
                if member.isfile() and wanted and not excluded(name):
                    hc.relative(name)
                    archive.extract(member, temporary, filter='data')
        error = process.stderr.read().decode(); process.stdout.close(); process.stderr.close()
        require(process.wait() == 0, 'source archive failed: '+error)
        yield Path(temporary)


@lru_cache(maxsize=8)
def source_recipe(root, revision):
    with source_tree(root, revision) as tree:
        script=tree/'bindings/merged_build.sh'
        source, _ = bincache.source_digest(tree, script, ['gpu'])
        declared = bincache.local_inputs(tree, script)
        merged = hashlib.sha256((tree/'bindings/merged_groups.json').read_bytes() + (tree/'tools/merged_bindings.py').read_bytes()).hexdigest()
    return dict(source=source, declared=declared, merged=merged)


def _text(path):
    return hc.witness(path)


def _rows(witness):
    return [line.split('\t') for line in hc.raw(witness).decode().splitlines() if line.strip()]


def validate_compilation(compilation, source_root=None):
    """Validate original positive compilation, archive content and input scopes."""
    require(compilation.get('schema') == 'mojolearn.ptx-compiled-shards.v1', 'unknown shard inventory')
    require(compilation.get('compiler_evidence') == dict(
        basis='locked and actually installed original compiler packages plus original Mojo version',
        observed_original_executable_sha256=None), 'original compiler observation scope differs')
    groups = compilation.get('groups', {})
    plan=device_plan()
    require(set(groups) == set(plan), 'incomplete or unexpected device groups')
    context = compilation['source_inputs']
    revision = compilation['source_commit']
    require(re.fullmatch('[0-9a-f]{40}', revision), 'missing original compilation source')
    recipes = compilation['source_recipe']
    identities = rc._device_keys(LEG, context)
    require({i['name'] for i in identities.values()} == {n for names in plan.values() for n in names}, 'device identity coverage differs')
    toolchain = next(iter(identities.values()))['toolchain']
    require(all(i['toolchain'] == toolchain for i in identities.values()), 'mixed device toolchains')
    for filename, row in groups.items():
        manifest=hc.decoded(row['manifest']); fields=hc.decoded(row['key'])
        key=bincache.key_of(fields)
        require(manifest.get('schema') == bincache.SCHEMA and manifest.get('key') == key
                and manifest.get('fields') == fields and manifest.get('source_commit') == revision,
                'archive manifest/key/original source differs')
        require(fields.get('schema') == bincache.SCHEMA and fields.get('script') == 'bindings/merged_build.sh'
                and fields.get('args') == ['gpu','--group',filename[:-3]]
                and fields.get('source') == recipes['source'] and fields.get('toolchain') == toolchain
                and fields.get('numeric_mode') == 'identical' and fields.get('gpu_archs') == 'sm_80'
                and fields.get('target_column') == 'nvidia' and fields.get('device_arch') == 'none'
                and 'variant' not in fields, 'wrong source, compiler, target, flags or shard command')
        wanted_env=dict(MODULAR_HOME='/root/mojolearn/.pixi/envs/default/share/max', MOJOLEARN_BUILD_PIXI_ENV='default',
            MOJOLEARN_BUILD_TIERS='identical', MOJOLEARN_COMPILE_JOBS='2', MOJOLEARN_CUDA_CODE_FORMAT='ptx',
            MOJOLEARN_GPU_ARCHS='sm_80', MOJOLEARN_LINUX_CPU='x86-64-v3', MOJOLEARN_MERGED_KEY=recipes['merged'],
            MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_PACKAGE_BYTE_LM='1', MOJOLEARN_TARGET_COLUMN='nvidia')
        require(fields.get('build_env') == wanted_env, 'unexpected original compile environment')
        rel='python/mojolearn/identical/'+filename
        require(fields.get('declared') == dict(inputs=recipes['declared'],outputs=[rel],shell='bash'), 'declared compiler inputs/outputs differ')
        require(fields.get('repo_path') == '/root/mojolearn' and fields.get('os', {}).get('machine') == 'x86_64', 'unsupported original host platform')
        require(manifest.get('installed_toolchain') == toolchain['packages'], 'actual original installed compiler packages differ')
        files=manifest.get('files', [])
        require(len(files)==1 and files[0].get('path')==rel and files[0].get('sha256')==row.get('raw_sha256')
                and isinstance(files[0].get('size'),int) and files[0]['size']>0, 'original compiler output differs')
        token='identical:merged_build.sh:'+filename[:-3]
        shards=[r for r in _rows(row['shard']) if r[0]==token]
        require(len(shards)==1 and len(shards[0])==5 and shards[0][1]=='ok' and shards[0][2].startswith('miss+built-')
                and shards[0][4]==key, 'no successful original shard compilation')
        provenance=[r for r in _rows(row['provenance']) if len(r)==6 and r[3]==key]
        require(len(provenance)==1 and provenance[0][1]=='bindings/merged_build.sh' and provenance[0][2]==shards[0][2]
                and provenance[0][5]==rel+'='+row['raw_sha256'], 'original compilation output receipt differs')
        archives=[r for r in _rows(row['archive_hashes']) if r[0]==key]
        require(len(archives)==1 and len(archives[0])==2 and archives[0][1]==row['archive_sha256']
                and re.fullmatch('[0-9a-f]{64}',row['archive_sha256']), 'original archive digest differs')
        require(manifest.get('build_log_sha256')==row['build_log'].get('compiler_stdout_sha256')
                and re.fullmatch('[0-9a-f]{64}',row['build_log'].get('sha256',''))
                and row['build_log'].get('scope')=='build_sets-wrapper-v1', 'original compiler log differs')
        require(re.search(r'^Mojo \S+ \([^)]+\)$', hc.raw(row['box']).decode(), re.M), 'original compiler version observation missing')
    if source_root is not None:
        require(rc.source_context(str(Path(source_root).resolve()),revision)==context, 'original device source inputs do not reproduce')
        require(source_recipe(str(Path(source_root).resolve()),revision)==recipes, 'original cache source/recipe does not reproduce')
    return toolchain


def collect(shards, source_root):
    shards=Path(shards).resolve(); groups={}; blobs={}; revisions=set()
    for report in sorted(shards.glob('shard-artifacts/*/shard/*/report')):
        rows=[line.split('\t') for line in (report/'shard.tsv').read_text().splitlines() if line.strip()]
        require(len(rows)==1 and len(rows[0])==5, 'ambiguous shard report')
        token, _status, _outcome, _seconds, key=rows[0]
        require(token.startswith('identical:merged_build.sh:'), 'non-device shard in input')
        filename=token.rsplit(':',1)[1]+'.so'; require(filename in device_plan() and filename not in groups, 'unknown/duplicate shard')
        fields=json.loads((report/'bincache/keys'/ (key+'.json')).read_text())
        archive=report.parent/'hot'/(key+'.tar.gz')
        manifest, payload=bincache.verify_archive(archive,key,fields)
        rel='python/mojolearn/identical/'+filename
        require(set(payload)=={rel}, 'archive contains unexpected compiler outputs')
        with tarfile.open(archive) as tar:
            raw=tar.extractfile('manifest.json').read()
        log=report/'build_logs'/('identical_merged_'+filename[:-3]+'.log')
        lines=log.read_bytes().splitlines(keepends=True)
        require(lines[0].startswith(b'start ') and lines[-2].startswith(b'BINCACHE ')
                and lines[-1].startswith(b'end ') and lines[-1].rstrip().endswith(b' OK'), 'unrecognized compiler log wrapper')
        stdout_sha=hashlib.sha256(b''.join(lines[1:-2])).hexdigest()
        groups[filename]=dict(manifest={'sha256':hashlib.sha256(raw).hexdigest(),'utf8':raw.decode()},
            key=_text(report/'bincache/keys'/(key+'.json')), shard=_text(report/'shard.tsv'),
            provenance=_text(report/'bincache/provenance.tsv'), archive_hashes=_text(report/'hot_sha256.tsv'),
            build_log=dict(path=str(log),sha256=hc.file_sha(log),compiler_stdout_sha256=stdout_sha,scope='build_sets-wrapper-v1'), box=_text(report/'box.txt'),
            archive_sha256=hc.file_sha(archive), raw_sha256=hashlib.sha256(payload[rel]).hexdigest())
        blobs[filename]=payload[rel]; revisions.add(manifest['source_commit'])
    require(len(revisions)==1, 'mixed or missing compilation sources')
    revision=revisions.pop(); root=str(Path(source_root).resolve())
    compilation=dict(schema='mojolearn.ptx-compiled-shards.v1',source_commit=revision,groups=groups,
        source_inputs=rc.source_context(root,revision),source_recipe=source_recipe(root,revision),
        compiler_evidence=dict(basis='locked and actually installed original compiler packages plus original Mojo version',
                               observed_original_executable_sha256=None))
    validate_compilation(compilation,source_root)
    return compilation, blobs


def record(source, compiler, source_root=None):
    require(source.get('kind')==KIND and source.get('leg')==LEG, 'wrong staged producer scope')
    proof,capture,inventory,stage,readback,isa,ptx = (hc.decoded(source[k]) for k in ('proof','capture','inventory','manifest','readback','isa','ptx'))
    require(proof.get('schema')==SCHEMA and proof.get('action')=='stage-compiled-ptx-shards' and proof.get('complete') is True
            and 'build_exit' not in proof, 'staging must not impersonate fresh compilation')
    require(proof.get('binding_layout')=='merged' and proof.get('release_modes')==['identical'], 'unsupported layout/mode')
    require(proof.get('source_inventory')==inventory and proof.get('source_sha256')==rc.sha_inventory(inventory)
            and proof.get('capture_sha256')==source['capture']['sha256'], 'source/capture linkage differs')
    require(capture.get('phase')=='before-assembly' and capture.get('source_commit')==proof.get('source_commit')
            and capture.get('compiler_sha256')==compiler and capture.get('compiler_observation_scope')=='assembly-only', 'assembly compiler observation differs')
    compilation=proof['original_compilation']; toolchain=validate_compilation(compilation,source_root)
    require(capture.get('original_compiler_evidence')==compilation['compiler_evidence'], 'original compiler evidence linkage differs')
    current=capture['source_inputs']; before=compilation['source_inputs']
    require(current['source_inventory']==inventory and capture['binding_identities']==rc._device_keys(LEG,current)
            and rc.compilation_bindings(before,rc._device_keys(LEG,before))==rc.compilation_bindings(current,rc._device_keys(LEG,current)),
            'original device compilation does not match admitted inputs')
    host_contract,_,_hp,_hc=hb.record(source['runtime_host'],compiler)
    require(toolchain==host_contract['toolchain'] and rc.runtime_inputs(before)==rc.runtime_inputs(current), 'runtime/compiler inputs differ')
    donor,_=hc._record(source['runtime_host']['runtime_source'],compiler)
    require(all(hc.decoded(row['key'])['image']==next(iter(donor['host_cache_inputs'].values()))['image'] for row in compilation['groups'].values()), 'original compiler image differs from runtime producer')
    require(capture.get('installed_packages')==host_contract['installed_packages'], 'assembly installed packages differ')
    plan=device_plan(); extensions={PREFIX+'identical/'+f:row['sha256'] for f,row in stage['device_outputs'].items()}
    require(set(stage['device_outputs'])==set(plan) and proof['extensions']==extensions, 'staged device coverage differs')
    index=hc.decoded(source['index'])
    require(index==dict(format=1,files=plan) and proof['merged_indexes']=={PREFIX+'identical/'+hc.INDEX:dict(sha256=source['index']['sha256'],files=plan)}, 'device index differs')
    require(proof.get('host_extension')=={} and proof.get('adapters')=={}, 'device stage invented host/adapter outputs')
    require(stage.get('runtime_libraries')==host_contract['runtime_libraries'] and stage.get('runtime_byte_changes')==[]
            and stage.get('runtime_origin_proof_sha256')==source['runtime_host']['proof']['sha256'], 'retained runtime bytes changed')
    for filename,row in stage['device_outputs'].items():
        require(row.get('raw_sha256')==compilation['groups'][filename]['raw_sha256'] and row.get('runpath')==RUNPATH
                and not row.get('rpath') and row.get('architectures')==['sm_80'] and re.fullmatch('[0-9a-f]{64}',row.get('sha256','')), 'staging input/output differs')
    expected={n for names in plan.values() for n in names}
    require(readback.get('members') and len(readback['members'])==len(expected)
            and {r['name'] for r in readback['members']}==expected
            and all(r['vendor']=='cuda' for r in readback['members'])
            and readback.get('groups')==plan
            and readback.get('host_architectures')=={f:[] for f in hc.host_plan()}, 'incomplete native device readback')
    host_rows=hc.decoded(source['runtime_host']['readback'])['members']
    expected_vendor={('identical',name,'cuda') for name in expected}|{('host',row['name'],'cpu') for row in host_rows}
    expected_arch={('identical',name,'sm_80') for name in expected}|{('host',row['name'],'NONE-BY-DESIGN') for row in host_rows}
    for name,wanted in (('vendor_text',expected_vendor),('arch_text',expected_arch)):
        lines=[tuple(line.split()) for line in hc.raw(source[name]).decode().splitlines()]
        require(len(lines)==len(wanted) and set(lines)==wanted,'packer native readback differs')
    package_manifest=hc.decoded(source['set_manifest'])
    require({row['path']:row['sha256'] for row in package_manifest.get('extensions',[])}
            == {**{'identical/'+f:r['sha256'] for f,r in stage['device_outputs'].items()},
                **{'host/'+f:sha for f,sha in hc.decoded(source['runtime_host']['stage'])['staged_sha256'].items()}},'packer manifest output differs')
    require(package_manifest.get('portable_math')==hc.decoded(source['runtime_host']['runtime_source']['manifest']).get('portable_math')
            and {row['name']:row['sha256'] for row in package_manifest.get('staged_libs',[])}==host_contract['runtime_libraries'],
            'packer runtime or portable math provenance differs')
    require(all(type(stage.get(key)) is int and stage[key]==0 for key in ('dependency_check_exit','ptx_patch_exit','ptx_audit_exit')),
            'failed or missing assembly phase')
    require(isa.get('baseline')=='x86-64-v3' and isa.get('failing')==0
            and ({'identical/'+f for f in plan}|{'host/'+f for f in hc.host_plan()}|{'.libs/'+name for name in host_contract['runtime_libraries']}) <= set(isa.get('report',{})), 'ISA audit failed/incomplete')
    require(ptx.get('source_commit')==proof['source_commit'] and ptx.get('source_dirty') is False
            and ptx.get('vendor')=='cuda' and ptx.get('target')=='sm_80' and ptx.get('code_format')=='ptx'
            and ptx.get('errors')==[] and {r['file']:r['sha256'] for r in ptx.get('files',[])}
            == {'identical/'+f:row['sha256'] for f,row in stage['device_outputs'].items()}, 'PTX final audit failed/incomplete')
    for name in ('manifest','readback','isa','ptx','set_manifest','vendor_text','arch_text'):
        require(proof.get(name+'_sha256')==source[name]['sha256'], name+' proof linkage differs')
    if source_root is not None:
        require(rc.source_context(str(Path(source_root).resolve()),proof['source_commit'])==current, 'admitted assembly inputs do not reproduce')
    contract=dict(runtime_libraries=host_contract['runtime_libraries'],toolchain=toolchain,installed_packages=host_contract['installed_packages'])
    return contract,{},proof,capture


def inspect(root):
    root=Path(root).resolve()
    source=dict(kind=KIND,leg=LEG,original_build_directory=str(root),**{key:_text(root/path) for key,path in {
        'proof':'device-stage-proof.json','capture':'assembly-inputs.json','inventory':'source-inventory.json',
        'manifest':'stage.json','readback':'readback.json','index':'set/identical/'+hc.INDEX,
        'isa':'isa.json','ptx':'set/PTX_BASELINE.json','set_manifest':'set/manifest.json',
        'vendor_text':'set/readback.txt','arch_text':'set/arch_readback.txt'}.items()})
    source['runtime_host']=json.loads((root/'runtime-host.json').read_text())
    return source,root/'set'


def validate_bytes(source,directory,compiler,composed=False):
    contract,_,proof,_=record(source,compiler); directory=Path(directory)
    require(all(hc.file_sha(directory/path[len(PREFIX):])==sha for path,sha in proof['extensions'].items()), 'staged device bytes differ')
    require(hc.file_sha(directory/'identical'/hc.INDEX)==source['index']['sha256'], 'staged device index differs')
    for key,name in (('set_manifest','manifest.json'),('vendor_text','readback.txt'),('arch_text','arch_readback.txt'),('ptx','PTX_BASELINE.json')):
        if composed and key=='ptx':
            continue  # Composition embeds its admission; original format audit still reproduces below.
        require(hc.file_sha(directory/name)==source[key]['sha256'],'staged packaging witness differs')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'packaging/linux'))
    import ptx_baseline
    ptx=hc.decoded(source['ptx'])
    require(ptx_baseline.audit_tree(directory,proof['source_commit'],ptx['mojo_version'],False)==ptx,'PTX audit does not reproduce from shipped bytes')
    require({p.name:hc.file_sha(p) for p in (directory/'.libs').iterdir() if p.is_file()}==contract['runtime_libraries'], 'runtime bytes differ')
    if not composed and (directory/'host').exists():
        host_proof=hc.decoded(source['runtime_host']['proof'])
        require(all(hc.file_sha(directory/path[len(hb.PREFIX):])==sha for path,sha in host_proof['host_extension'].items()),
                'retained host bytes changed in device staging')
    return True


def architectures(path):
    data=subprocess.check_output(['strings','-a',str(path)])
    return sorted({value.decode() for value in re.findall(rb'\b(?:sm_[0-9]+[a-z]*|compute_[0-9]+[a-z]*|gfx[0-9a-f]+)\b',data)})


def readback(root,binary,name,out,merged=False):
    registry='mojolearn._native_loaded_modules'
    spec=importlib.util.spec_from_file_location(registry,Path(root)/'python/mojolearn/_native_loaded_modules.py')
    module=importlib.util.module_from_spec(spec);sys.modules[registry]=module;spec.loader.exec_module(module)
    loader=importlib.machinery.ExtensionFileLoader(name,str(binary))
    module=importlib.util.module_from_spec(importlib.util.spec_from_loader(name,loader,origin=str(binary)));loader.exec_module(module)
    if merged:
        write(out,dict(members=str(module.merged_members()).split(',')))
    else:
        fn='mojolearn_vendor' if name=='_mojolearn' else name[len('_mojolearn_'):]+'_vendor'
        vendor=str(getattr(module,fn)());require(vendor=='cuda','native vendor readback failed')
        if name=='_mojolearn_byte_lm':
            require(module.byte_lm_numeric_mode()==1
                    and module.byte_lm_profile()=='mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1',
                    'byte LM native mode/profile differs')
        write(out,dict(name=name,vendor=vendor))


def assemble(root,shards,host_build,out,compiler_sha,pixi_env='gbmbench',patchelf=None):
    require(sys.platform=='linux','assembly runs on Linux without a GPU')
    root,out=Path(root).resolve(),Path(out).resolve()
    require(re.fullmatch('[A-Za-z0-9_-]+',pixi_env) and not out.exists(),'unsafe environment or existing output')
    revision,inventory,context=rc.current_source(root)
    host,host_set=hb.inspect(host_build);hb.validate_bytes(host,host_set,compiler_sha)
    host_contract,_,_,_=hb.record(host,compiler_sha)
    compilation,blobs=collect(shards,root)
    before=compilation['source_inputs']
    require(rc.compilation_bindings(before,rc._device_keys(LEG,before))==rc.compilation_bindings(context,rc._device_keys(LEG,context)), 'current device inputs changed')
    prefix=root/'.pixi/envs'/pixi_env; compiler=prefix/'bin/mojo'
    require(hc.file_sha(compiler)==compiler_sha,'assembly compiler bytes differ')
    packages=sorted(p.stem for p in (prefix/'conda-meta').glob('*.json'))
    require(packages==host_contract['installed_packages'],'assembly package environment differs')
    device=out/'device'; staged=device/'set'; (staged/'identical').mkdir(parents=True)
    write(device/'source-inventory.json',inventory);write(device/'runtime-host.json',host)
    capture=dict(phase='before-assembly',source_commit=revision,compiler_sha256=compiler_sha,
        compiler_observation_scope='assembly-only',assembly_tool_sha256=hc.file_sha(Path(__file__)),installed_packages=packages,source_inputs=context,
        binding_identities=rc._device_keys(LEG,context),original_compiler_evidence=compilation['compiler_evidence'])
    write(device/'assembly-inputs.json',capture)
    for filename,data in blobs.items():(staged/'identical'/filename).write_bytes(data)
    write(staged/'identical'/hc.INDEX,dict(format=1,files=device_plan()))
    shutil.copytree(host_set/'.libs',staged/'.libs')
    shutil.copytree(host_set/'host',staged/'host')  # Retained host proof owns these bytes; include them in the ISA audit.
    env=dict(os.environ);env['PYTHONPATH']=str(root/'python')
    python=prefix/'bin/python3'
    files=[str(staged/'identical'/f) for f in device_plan()]
    for action in ('patch','audit'):
        hb.run_logged([str(python),str(root/'packaging/linux/ptx_contract.py'),action,*files],root,env,device/('ptx-'+action+'.jsonl'))
    sys.path.insert(0,str(root/'packaging/linux'));import stage_libs
    patchelf=patchelf or shutil.which('patchelf');require(patchelf and Path(patchelf).is_file(),'patchelf unavailable')
    outputs={}
    for filename in device_plan():
        path=staged/'identical'/filename
        stage_libs.set_runpath(patchelf,path,RUNPATH)
        needed,runpath,rpath=stage_libs.elf_dynamic(path)
        outputs[filename]=dict(raw_sha256=compilation['groups'][filename]['raw_sha256'],sha256=hc.file_sha(path),needed=needed,runpath=runpath,rpath=rpath,architectures=architectures(path))
        require(outputs[filename]['architectures']==['sm_80'],'device architecture readback differs')
    staged_binaries=[*(staged/'identical').glob('*.so'),*(staged/'host').glob('*.so'),*(staged/'.libs').glob('*.so*')]
    stage_libs.verify_closed(staged_binaries,staged/'.libs',prefix/'lib')
    edges=[dict(source=path.relative_to(staged).as_posix(),needs=dep,kind=stage_libs.classify(dep,prefix/'lib'))
           for path in staged_binaries for dep in stage_libs.elf_dynamic(path)[0]]
    require({p.name:hc.file_sha(p) for p in (staged/'.libs').iterdir() if p.is_file()}==host_contract['runtime_libraries'],'runtime bytes changed during staging')
    host_stage=hc.decoded(host['stage'])
    donor_manifest=hc.decoded(host['runtime_source']['manifest'])
    manifest=dict(binding_layout='merged',release_modes=['identical'],
        extensions=[dict(path='identical/'+f,tier='identical',bytes=(staged/'identical'/f).stat().st_size,
                         sha256=row['sha256'],needed=row['needed'],runpath=row['runpath'],members=device_plan()[f]) for f,row in outputs.items()]
                   + [dict(path='host/'+f,tier='host',bytes=(host_set/'host'/f).stat().st_size,
                           sha256=host_stage['staged_sha256'][f],needed=host_stage['needed'][f],runpath=host_stage['runpath'][f],
                           members=hc.host_plan()[f]) for f in hc.host_plan()],
        staged_libs=[dict(name=p.name,bytes=p.stat().st_size,sha256=hc.file_sha(p),source='retained-host-runtime') for p in sorted((staged/'.libs').iterdir()) if p.is_file()],
        merged_indexes=[dict(path='identical/'+hc.INDEX,sha256=hc.file_sha(staged/'identical'/hc.INDEX),files=device_plan()),
                        dict(path=hb.HOST_INDEX,sha256=host['index']['sha256'],files=hc.host_plan())],
        portable_math=donor_manifest['portable_math'],driver_libs_not_staged=sorted({row['needs'] for row in edges if row['kind']=='driver'}),
        edges=edges,input_edges=edges,
        adapters=[],framework_libs_not_staged=[],
        optional_native={'_mojolearn_byte_lm':dict(included=True,supported_modes=['identical'],unsupported_modes=['fast'],
            profile='mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1')})
    manifest['bytes_extensions']=sum(r['bytes'] for r in manifest['extensions'])
    manifest['bytes_staged_libs']=sum(r['bytes'] for r in manifest['staged_libs'])
    manifest['bytes_total_uncompressed']=manifest['bytes_extensions']+manifest['bytes_staged_libs']
    write(staged/'manifest.json',manifest)
    write(device/'stage.json',dict(device_outputs=outputs,runtime_libraries=host_contract['runtime_libraries'],runtime_byte_changes=[],
        runtime_origin_proof_sha256=host['proof']['sha256'],patchelf_sha256=hc.file_sha(patchelf),
        dependency_check_exit=0,ptx_patch_exit=0,ptx_audit_exit=0,
        ptx_patch_sha256=hc.file_sha(device/'ptx-patch.jsonl'),ptx_audit_sha256=hc.file_sha(device/'ptx-audit.jsonl')))
    hb.run_logged([str(python),str(root/'packaging/linux/isa_baseline_linux.py'),str(staged),'--json',str(device/'isa.json')],root,env,device/'isa.log')
    rows=[];groups={};probes=device/'readback';probes.mkdir()
    for filename,names in device_plan().items():
        for name in [filename[:-3],*names]:
            result=probes/(name+'.json')
            argv=[str(python),str(Path(__file__).resolve()),'readback','--root',str(root),'--binary',str(staged/'identical'/filename),'--name',name,'--out',str(result)]
            if name==filename[:-3]:argv.append('--merged')
            hb.run_logged(argv,root,env,probes/(name+'.log'));row=json.loads(result.read_text())
            if name==filename[:-3]:
                require(row['members']==names,'merged native membership differs');groups[filename]=row['members']
            else:rows.append(row)
    host_arch={f:architectures(host_set/'host'/f) for f in hc.host_plan()}
    require(not any(host_arch.values()),'CPU host contains a GPU architecture')
    write(device/'readback.json',dict(members=rows,groups=groups,host_architectures=host_arch))
    host_rows=hc.decoded(host['readback'])['members']
    (staged/'readback.txt').write_text(''.join('identical '+row['name']+' '+row['vendor']+'\n' for row in rows)
        + ''.join('host '+row['name']+' '+row['vendor']+'\n' for row in host_rows))
    (staged/'arch_readback.txt').write_text(''.join('identical '+name+' '+','.join(outputs[f]['architectures'])+'\n' for f,names in groups.items() for name in names)
        + ''.join('host '+row['name']+' NONE-BY-DESIGN\n' for row in host_rows))
    version=subprocess.check_output([str(compiler),'--version'],cwd=root,env=env,text=True)
    hb.run_logged([str(python),str(root/'packaging/linux/ptx_baseline.py'),'audit',str(staged),'--repo',str(root),
        '--mojo-version',version,'--output',str(staged/'PTX_BASELINE.json')],root,env,device/'ptx-baseline.log')
    require(rc.current_source(root)[:2]==(revision,inventory),'source changed during assembly')
    proof=dict(schema=SCHEMA,action='stage-compiled-ptx-shards',complete=True,binding_layout='merged',release_modes=['identical'],
        source_commit=revision,source_inventory=inventory,source_sha256=rc.sha_inventory(inventory),original_compilation=compilation,
        capture_sha256=hc.file_sha(device/'assembly-inputs.json'),extensions={PREFIX+'identical/'+f:r['sha256'] for f,r in outputs.items()},
        host_extension={},adapters={},merged_indexes={PREFIX+'identical/'+hc.INDEX:dict(sha256=hc.file_sha(staged/'identical'/hc.INDEX),files=device_plan())},
        bincache={PREFIX+'identical/'+f:dict(compiled_from_commit=compilation['source_commit']) for f in outputs},
        qualification='Original successful compilation with pinned installed package evidence; assembly metadata/load/ISA/PTX audits only. No GPU execution, identity or installed-wheel qualification. Original compiler executable SHA was not captured.',
        **{name+'_sha256':hc.file_sha(path) for name,path in dict(manifest=device/'stage.json',readback=device/'readback.json',isa=device/'isa.json',ptx=staged/'PTX_BASELINE.json',set_manifest=staged/'manifest.json',vendor_text=staged/'readback.txt',arch_text=staged/'arch_readback.txt').items()})
    write(device/'device-stage-proof.json',proof)
    source,directory=inspect(device);record(source,compiler_sha,root);validate_bytes(source,directory,compiler_sha)
    result=rc.prepare(device,host_build,root,compiler_sha,out/'composed')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--root',required=True);plan.add_argument('--shards',required=True);plan.add_argument('--out',required=True)
    assembly=sub.add_parser('assemble')
    for name in ('root','shards','host-build','out','compiler-sha256'):assembly.add_argument('--'+name,required=True)
    assembly.add_argument('--pixi-env',default='gbmbench');assembly.add_argument('--patchelf')
    probe=sub.add_parser('readback')
    for name in ('root','binary','name','out'):probe.add_argument('--'+name,required=True)
    probe.add_argument('--merged',action='store_true')
    args=parser.parse_args(argv)
    if args.action=='plan':
        compilation,_=collect(args.shards,args.root);write(args.out,compilation)
        print(json.dumps(dict(groups=len(compilation['groups']),source_commit=compilation['source_commit'],qualification='compilation only; no assembly or runtime claim')))
    elif args.action=='readback':readback(args.root,args.binary,args.name,args.out,args.merged)
    else:
        result=assemble(args.root,args.shards,args.host_build,args.out,args.compiler_sha256,args.pixi_env,args.patchelf)
        print(json.dumps(dict(schema=result['schema'],action=result['action'],source_commit=result['source_commit'],out=args.out)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
