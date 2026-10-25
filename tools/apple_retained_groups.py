#!/usr/bin/env python3
"""Explicitly admit retained Apple device builds and preserve their origins.

The unchanged production Mac packer still owns host/adapters, staging, signing,
audits and wheel assembly. A cache hook places admitted raw device outputs;
it never creates a current-source cache archive or claims a new compilation.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import zipfile

import bincache
import binding_stamps
import release_host_composition as witness
import release_reuse as reuse

SCHEMA='mojolearn.apple.retained-device-admission.v1'
PACK_SCHEMA='mojolearn.apple.retained-pack.v1'
GROUPS=('_mojolearn_native','_mojolearn_native_m1')
SWITCH='MOJOLEARN_BINCACHE_RETAINED_APPLE'
PLACEMENTS='MOJOLEARN_BINCACHE_RETAINED_PLACEMENTS'


def require(ok, why):
    witness.require(ok,'Apple retained admission: '+why)


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def current_revision(root):
    revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    require(re.fullmatch('[0-9a-f]{40}',revision), 'missing committed source')
    require(not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip(),
            'tracked source is dirty')
    return revision


def file_ref(path):
    return dict(path=str(Path(path).resolve()),sha256=witness.file_sha(path))


def read_ref(ref):
    path=Path(ref['path'])
    require(witness.file_sha(path)==ref['sha256'],'retained file bytes changed: '+str(path))
    return path


def actual_compiler(root):
    path=Path(root)/'.pixi/envs/default/bin/mojo'
    require(path.is_file(),'default compiler is missing')
    return witness.file_sha(path)


def source_inputs(root,host_toolchain):
    facts=reuse.tree_facts(root)
    bindings=[b for b in reuse.bindings(reuse.MACOS,['identical']) if not b.host]
    identities={b.key:reuse.identity(b,facts,host_toolchain) for b in bindings}
    require(len(identities)==33 and all(not reuse.unreadable(i) and i['closure']['scope']=='closure'
                                      for i in identities.values()),'unresolved or incomplete device closures')
    files=set(binding_stamps.MERGED_INPUTS)|set(reuse.MACOS_BUILDERS)|{'pixi.toml','pixi.lock'}
    for b in bindings: files.update(facts['closures'][b.script]['sources'])
    return identities,{name:witness.file_sha(Path(root)/name) for name in sorted(files)}


def original(run):
    """Read actual original captures/results; no retrospective build is invented."""
    run=Path(run).resolve()
    before=witness.witness(run/'before-build.json');capture=witness.decoded(before)
    summary=witness.witness(run/'summary.json');completed=witness.decoded(summary)
    require((run/'controller.rc').read_text().strip()=='0' and completed.get('complete') is True
            and completed.get('expected_groups')==completed.get('completed_groups')==list(GROUPS),
            'original subset controller did not finish both device groups')
    revision=capture.get('source_commit')
    require(capture.get('schema')=='mojolearn.apple.device-subset-inputs.v1'
            and capture.get('phase')=='before-build' and re.fullmatch('[0-9a-f]{40}',revision or '')
            and completed.get('source_commit')==revision and capture.get('groups')==list(GROUPS)
            and capture.get('layout')=='release' and capture.get('numeric_modes')==['identical'],
            'unsupported original producer scope/source')
    inventory=witness.witness(run/'source-inventory.json')
    archive=file_ref(run/'source-inputs.tar.gz')
    require(inventory['sha256']==capture.get('source_inventory_sha256')
            and archive['sha256']==capture.get('source_archive_sha256'),'original source capture changed')
    members=witness.decoded(inventory)
    with tempfile.TemporaryDirectory(prefix='mojolearn-apple-original-') as temp:
        with tarfile.open(archive['path']) as source:
            rows=source.getmembers()
            require(len(rows)==len(members) and {r.name for r in rows}==set(members)
                    and all(r.isfile() for r in rows),'original source archive coverage differs')
            for row in rows:
                witness.relative(row.name)
                require(hashlib.sha256(source.extractfile(row).read()).hexdigest()==members[row.name],
                        'original source archive bytes differ')
            source.extractall(temp,filter='data')
        identities, dependency_files=source_inputs(temp,capture['host_toolchain'])
        require(identities==capture['binding_identities'],'original device source identities do not reproduce')
        merged={name:members[name] for name in binding_stamps.MERGED_INPUTS}
        merged_key=hashlib.sha256((Path(temp)/'bindings/merged_groups.json').read_bytes()
                                  +(Path(temp)/'tools/merged_bindings.py').read_bytes()).hexdigest()
        require(capture['build_environment']==dict(MOJOLEARN_COMPILE_JOBS='1',MOJOLEARN_MERGED_BINDINGS='1',
            MOJOLEARN_MERGED_KEY=merged_key,MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_PACKAGE_BYTE_LM='1'),
            'original production compiler flags or merged inputs differ')
        groups={}
        for stem in GROUPS:
            base=run/stem
            receipt=witness.witness(base/'receipt.json');result=witness.decoded(receipt)
            before_key=witness.witness(base/'before-build-key.json');key_record=witness.decoded(before_key)
            fields=key_record['fields'];key=key_record['key']
            raw=base/(stem+'.so')
            relative='python/mojolearn/identical/'+stem+'.so'
            require(result.get('schema')=='mojolearn.apple.device-subset-result.v1'
                    and result.get('source_commit')==revision and result.get('group')==stem
                    and result.get('complete') is True
                    and type(result.get('build_exit')) is int and result['build_exit']==0
                    and type(result.get('stamp_exit')) is int and result['stamp_exit']==0
                    and result.get('compiler_sha256')==capture['compiler_sha256']
                    and result.get('output')==relative and result.get('output_sha256')==witness.file_sha(raw),
                    'missing successful original device output: '+stem)
            require((base/'build.rc').read_text().strip()==(base/'stamp.rc').read_text().strip()=='0'
                    and result.get('log_sha256')==witness.file_sha(base/'build.log'),
                    'original phase logs/exit statuses differ')
            require(result.get('key')==key==bincache.key_of(fields)
                    and fields.get('args')==['gpu','--group',stem]
                    and fields.get('script')=='bindings/merged_build.sh'
                    and fields.get('numeric_mode')=='identical' and not fields.get('gpu_archs')
                    and not fields.get('target_column')
                    and fields.get('build_env')==capture['build_environment']
                    and fields.get('local',{}).get('outputs')==[relative]
                    and fields['local'].get('shell')=='sh'
                    and fields['local'].get('host_toolchain')==capture['host_toolchain'],
                    'original command/cache identity differs')
            source_digest,_=bincache.source_digest(Path(temp),str(Path(temp)/fields['script']),fields['args'])
            require(source_digest==fields['source']
                    and bincache.local_inputs(Path(temp),str(Path(temp)/fields['script']))==fields['local']['inputs'],
                    'original merged cache source inputs do not reproduce')
            actual_key=witness.witness(base/'cache-provenance/keys'/(key+'.json'))
            require(witness.decoded(actual_key)==fields,'actual compiler cache key differs from before-build capture')
            outcome=witness.witness(base/'cache-provenance/provenance.tsv')
            rows=[line.split('\t') for line in witness.raw(outcome).decode().splitlines()]
            require(len(rows)==1 and len(rows[0])==6 and rows[0][1]=='bindings/merged_build.sh'
                    and rows[0][2]=='miss+built-cached' and rows[0][3]==key
                    and rows[0][5]==relative+'='+result['output_sha256']
                    and result.get('cache_outcome')==witness.raw(outcome).decode(),
                    'original compile receipt is not an independently witnessed build')
            stamp=witness.witness(base/'binding-stamp.json');stamp_doc=witness.decoded(stamp)
            index=witness.witness(base/'_mojolearn_merged.json');index_doc=witness.decoded(index)
            group_names=index_doc.get('files',{}).get(stem+'.so')
            require(stamp_doc.get('schema')==2 and stamp_doc.get('commit')==revision
                    and stamp_doc.get('binding')=='identical/'+stem+'.so'
                    and group_names and set(stamp_doc.get('members',{}))==set(group_names),
                    'original group membership/stamp differs')
            by_name={i['name']:i for i in identities.values()}
            for name,row in stamp_doc['members'].items():
                require(row['scope']=='closure' and row['digest']==by_name[name]['closure']['digest']
                        and row['script']==by_name[name]['script'],'original member stamp source differs')
            stamp_hash=hashlib.sha256()
            for name in binding_stamps.MERGED_INPUTS:
                stamp_hash.update(name.encode()+b'\0'+merged[name].encode()+b'\n')
            require(stamp_doc.get('merged_inputs')==stamp_hash.hexdigest(),'original merged stamp inputs differ')
            groups[stem]=dict(receipt=receipt,before_build_key=before_key,actual_cache_key=actual_key,
                cache_outcome=outcome,stamp=stamp,index=index,artifact=file_ref(raw),members=group_names,
                original_source_commit=revision,original_key=key)
        require({r['group']:r for r in completed.get('receipts',[])}
                =={stem:witness.decoded(row['receipt']) for stem,row in groups.items()},
                'original controller summary disagrees with group results')
    return dict(before_build=before,summary=summary,source_inventory=inventory,source_archive=archive,
                dependency_files=dependency_files,merged_build_inputs=merged,groups=groups)


def prepare(run,root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    require(root==Path(reuse.ROOT).resolve(),'admission tools must belong to the target source checkout')
    require(not output.exists(),'admission output already exists')
    original_record=original(run);capture=witness.decoded(original_record['before_build'])
    revision=current_revision(root)
    host_toolchain=bincache.darwin_toolchain()
    require(host_toolchain==capture['host_toolchain'] and actual_compiler(root)==capture['compiler_sha256'],
            'compiler executable or Apple host toolchain changed')
    identities,dependency_files=source_inputs(root,host_toolchain)
    require(identities==capture['binding_identities'],'device source/compiler/flags/builders changed')
    require(dependency_files==original_record['dependency_files'],'device compiler dependency bytes changed')
    require({name:dependency_files[name] for name in binding_stamps.MERGED_INPUTS}
            ==original_record['merged_build_inputs'],'merged builder, generator, defines or manifest changed')
    document=dict(schema=SCHEMA,action='admit-original-device-groups',complete=True,
        admitted_source_commit=revision,source_root=str(root),compiler_sha256=capture['compiler_sha256'],
        host_toolchain=host_toolchain,admitted_dependency_files=dependency_files,
        binding_identities=identities,original=original_record,
        qualification='Original compiler evidence admitted by unchanged inputs; no new device compilation or runtime qualification')
    write(output,document)
    return document


def validate_admission(document,root):
    root=Path(root).resolve()
    require(document.get('schema')==SCHEMA and document.get('action')=='admit-original-device-groups'
            and document.get('complete') is True and 'build_exit' not in document,'invalid assembly admission')
    require(document.get('source_root')==str(root) and current_revision(root)==document.get('admitted_source_commit'),
            'admission does not match frozen target checkout')
    require(actual_compiler(root)==document['compiler_sha256'],'compiler executable changed after admission')
    for name,sha in document['admitted_dependency_files'].items():
        require(witness.file_sha(root/witness.relative(name))==sha,'admitted dependency changed: '+name)
    old=document['original'];capture=witness.decoded(old['before_build'])
    require(capture['binding_identities']==document['binding_identities']
            and capture['compiler_sha256']==document['compiler_sha256']
            and capture['host_toolchain']==document['host_toolchain']
            and set(old['groups'])==set(GROUPS),'original/admitted inputs differ')
    inventory=witness.decoded(old['source_inventory'])
    require(old['source_inventory']['sha256']==capture['source_inventory_sha256']
            and old['source_archive']['sha256']==capture['source_archive_sha256']
            and old['dependency_files']==document['admitted_dependency_files']
            and all(inventory.get(name)==sha for name,sha in old['dependency_files'].items()),
            'original source bytes do not match admission')
    for stem,group in old['groups'].items():
        receipt=witness.decoded(group['receipt']);key=witness.decoded(group['before_build_key'])
        require(receipt.get('complete') is True and type(receipt.get('build_exit')) is int and receipt['build_exit']==0
                and type(receipt.get('stamp_exit')) is int and receipt['stamp_exit']==0
                and receipt['source_commit']==group['original_source_commit']==capture['source_commit']
                and receipt['group']==stem and receipt['compiler_sha256']==capture['compiler_sha256']
                and receipt['key']==group['original_key']==key['key']==bincache.key_of(key['fields'])
                and receipt['output_sha256']==group['artifact']['sha256']
                and witness.decoded(group['actual_cache_key'])==key['fields'],
                'original device result/key/output differs')
        require(receipt['cache_outcome']==witness.raw(group['cache_outcome']).decode()
                and '\tmiss+built-cached\t'+key['key']+'\t' in receipt['cache_outcome'],
                'original device compilation outcome differs')
    require({name:document['admitted_dependency_files'][name] for name in binding_stamps.MERGED_INPUTS}
            ==old['merged_build_inputs'],'merged input admission differs')
    return True


def comparable_fields(fields):
    doc=copy.deepcopy(fields)
    doc.pop('source')  # Reconciled by exact device closures, not the broader all-Mojo-tree cache key.
    env=doc['build_env']
    # The unchanged Mac packer exports these as empty strings. Neither activates
    # a per-binding reuse path; original subset builds leave them unset.
    for name in ('MOJOLEARN_REUSE_PLAN','MOJOLEARN_REUSE_DIR'):
        if env.get(name)=='':env.pop(name)
    return doc


def try_place(repo,script,args,environ,outputs,fields,key,rec,started):
    """Called by local bincache before a hit/build; never write a cache archive."""
    admission_path=environ.get(SWITCH)
    if not admission_path or Path(script).name!='merged_build.sh' or args[:1]!=['gpu']:
        return None
    require(sys.platform=='darwin' and args[:2]==['gpu','--group'] and len(args)==3
            and args[2] in GROUPS,'unsupported retained device invocation')
    document=json.loads(Path(admission_path).read_text());validate_admission(document,repo)
    stem=args[2];original_group=document['original']['groups'][stem]
    original_fields=witness.decoded(original_group['before_build_key'])['fields']
    require(comparable_fields(fields)==comparable_fields(original_fields),
            'actual compiler flags/target/environment differ from original device invocation')
    relative='python/mojolearn/identical/'+stem+'.so'
    require(outputs==[relative] and not (repo/relative).exists(),'wrong declared or existing device output')
    original_file=read_ref(original_group['artifact'])
    destination=repo/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name('.retained-'+str(os.getpid())+'-'+destination.name)
    shutil.copyfile(original_file,temporary)
    os.chmod(temporary,0o755);temporary.replace(destination)
    require(witness.file_sha(destination)==original_group['artifact']['sha256'],'placed retained bytes differ')
    placed=[dict(path=relative,sha256=original_group['artifact']['sha256'])]
    receipt=dict(schema='mojolearn.apple.retained-placement.v1',origin='retained-admitted',group=stem,
        admitted_source_commit=document['admitted_source_commit'],original_source_commit=original_group['original_source_commit'],
        original_key=original_group['original_key'],requested_current_key=key,
        original_receipt_sha256=original_group['receipt']['sha256'],admission_sha256=witness.file_sha(admission_path),
        output=relative,sha256=placed[0]['sha256'],new_compilation=False,new_cache_archive=False)
    placement_dir=environ.get(PLACEMENTS)
    require(placement_dir,'explicit retained placement evidence directory is required')
    target=Path(placement_dir)/(stem+'.json')
    require(not target.exists(),'retained placement receipt already exists')
    write(target,receipt)
    rec.row('bindings/merged_build.sh','retained-admitted',key,time.time()-started,placed)
    rec.placement('bindings/merged_build.sh','retained-admitted',key,placed,
        dict(source_commit=original_group['original_source_commit'],builder=dict(original_key=original_group['original_key'],
             original_receipt_sha256=original_group['receipt']['sha256'])))
    print('admitted '+relative+' from original '+original_group['original_source_commit']+'; no compilation or new cache archive')
    return 0


def pack(admission,root,out,cache):
    """Run unchanged production packaging and bind its final wheel to admission."""
    root,out=Path(root).resolve(),Path(out).resolve()
    require(not out.exists(),'pack evidence output already exists')
    document=json.loads(Path(admission).read_text());validate_admission(document,root)
    out.mkdir(parents=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('MOJOLEARN_','MOJO_','MODULAR_'))}
    env.update(MOJOLEARN_RELEASE_MODES='identical',MOJOLEARN_COMPILE_JOBS='1',MOJOLEARN_BUILD_JOBS='1',
        MOJOLEARN_PACKAGE_BYTE_LM='1',MOJOLEARN_MERGED_BINDINGS='1',MOJOLEARN_BINCACHE_DIR=str(Path(cache).resolve()),
        MOJOLEARN_BINCACHE_OUT=str(out/'cache-provenance'))
    env[SWITCH]=str(Path(admission).resolve());env[PLACEMENTS]=str(out/'placements')
    argv=['sh','packaging/macos/build_release_wheel.sh','--build-only']
    launch=dict(schema='mojolearn.apple.retained-pack-launch.v1',source_commit=document['admitted_source_commit'],
        command=argv,admission=witness.witness(admission),compiler_sha256=document['compiler_sha256'],created_ns=time.time_ns())
    write(out/'launch.json',launch)
    with (out/'pack.log').open('w') as log:
        result=subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/'pack.rc').write_text(str(result.returncode)+'\n')
    require(result.returncode==0,'production pack failed; see '+str(out/'pack.log'))
    validate_admission(document,root)
    placements={stem:witness.witness(out/'placements'/(stem+'.json')) for stem in GROUPS}
    wheels=list((root/'python/dist').glob('mojolearn-*-macosx_*.whl'))
    require(len(wheels)==1,'expected exactly one final Mac wheel')
    wheel=wheels[0];artifacts={}
    with zipfile.ZipFile(wheel) as archive:
        for stem,raw_receipt in placements.items():
            receipt=witness.decoded(raw_receipt);original_group=document['original']['groups'][stem]
            require(receipt.get('origin')=='retained-admitted' and receipt.get('new_compilation') is False
                    and receipt.get('new_cache_archive') is False
                    and receipt.get('admission_sha256')==launch['admission']['sha256']
                    and receipt.get('original_receipt_sha256')==original_group['receipt']['sha256']
                    and receipt.get('sha256')==original_group['artifact']['sha256'], 'final placement provenance differs')
            member='mojolearn/identical/'+stem+'.so'
            artifacts[member]=dict(origin='retained-admitted',original_source_commit=receipt['original_source_commit'],
                admitted_source_commit=document['admitted_source_commit'],original_key=receipt['original_key'],
                original_receipt_sha256=receipt['original_receipt_sha256'],raw_compiler_sha256=receipt['sha256'],
                staged_tree_sha256=witness.file_sha(root/'python'/member),
                wheel_sha256=hashlib.sha256(archive.read(member)).hexdigest())
    shutil.copy2(wheel,out/wheel.name)
    report=dict(schema=PACK_SCHEMA,action='assemble-and-stage-retained-devices',complete=True,pack_exit=0,
        source_commit=document['admitted_source_commit'],launch=witness.witness(out/'launch.json'),
        admission=launch['admission'],placements=placements,wheel=file_ref(out/wheel.name),artifact_origins=artifacts,
        qualification='Unchanged production packaging passed; retained device compilation kept separate from new host/adapters; installed-wheel runtime gates remain required')
    write(out/'pack-proof.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--run',type=Path,required=True);prep.add_argument('--root',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True)
    assembly=sub.add_parser('pack')
    for name in ('admission','root','output','cache'):assembly.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    if args.action=='prepare':result=prepare(args.run,args.root,args.output)
    else:
        try:result=pack(args.admission,args.root,args.output,args.cache)
        except BaseException:
            if args.output.is_dir():(args.output/'failure.txt').write_text(traceback.format_exc())
            raise
    print(json.dumps(dict(schema=result['schema'],source_commit=result.get('source_commit',result.get('admitted_source_commit')))))
    return 0


if __name__=='__main__':raise SystemExit(main())
