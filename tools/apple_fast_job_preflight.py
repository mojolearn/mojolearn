#!/usr/bin/env python3
"""Read-only M3 metadata gate for FUTURE pinned jobs. SPEC.json -> READY JSON.
No queue changes, builds, worktree creation, native imports, or numerical runs.
Readiness is a snapshot of declared prerequisites, not ABI/caller certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

from apple_fast_job_policy import policy_for


class InfrastructureError(Exception):
    pass


def require(condition,message):
    if not condition:
        raise InfrastructureError(message)


def exact_sha(value,label='source'):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{40}',value),label+' must be a full commit SHA')
    return value


def safe_relative(value):
    require(isinstance(value,str) and value and not value.startswith('/') and '\\' not in value
            and all(p not in ('','.', '..') for p in value.split('/')),'invalid repo/artifact path: '+str(value))
    return value


def read_json(path):
    require(path.is_file(),'missing JSON prerequisite: '+str(path))
    try:
        return json.loads(path.read_text())
    except (ValueError,OSError) as exc:
        raise InfrastructureError('invalid JSON '+str(path)+': '+str(exc)) from exc


def digest(path):
    require(path.is_file(),'missing artifact/prerequisite: '+str(path))
    before=path.stat()
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    after=path.stat()
    require((before.st_size,before.st_mtime_ns,before.st_ino)==
            (after.st_size,after.st_mtime_ns,after.st_ino),'file changed during preflight: '+str(path))
    return h.hexdigest()


def expected_fields(actual,expected,label):
    require(isinstance(actual,dict) and isinstance(expected,dict),'expected object metadata: '+label)
    for key,value in expected.items():
        require(key in actual and actual[key]==value,label+' mismatch for '+key)


class GitMirror:
    def __init__(self,path):
        self.path=path
        require(path.is_dir(),'missing source mirror: '+str(path))

    def _git(self,*args):
        completed=subprocess.run(['git','--git-dir',str(self.path),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        require(completed.returncode==0,'mirror lookup failed: '+' '.join(args))
        return completed.stdout

    def commit(self,source):
        self._git('cat-file','-e',exact_sha(source)+'^{commit}')

    def blob(self,source,path):
        safe_relative(path)
        require(self._git('cat-file','-t',source+':'+path).strip()==b'blob','source path is not a file: '+path)
        return self._git('show',source+':'+path)


def no_duplicate(tag,mq):
    # Scan text and top-level names only. No recursive race-output scans.
    pattern=re.compile(r'(?<![A-Za-z0-9_.-])'+re.escape(tag)+r'(?![A-Za-z0-9_.-])')
    records={}
    for name in ('queue.txt','results.txt'):
        path=mq/name
        require(path.is_file(),'missing queue state: '+str(path))
        with path.open(errors='replace') as stream:
            for lineno,line in enumerate(stream,1):
                require(not pattern.search(line),'duplicate tag in '+name+':'+str(lineno))
        records[name]=dict(sha256=digest(path),size=path.stat().st_size)
    output=mq/'out'
    require(output.is_dir(),'missing output root: '+str(output))
    for path in output.iterdir():
        name=path.name
        require(not (name==tag or name.startswith(tag+'-') or name.startswith(tag+'.')),
                'tag output already exists: '+str(path))
    return records


def artifact_check(item,mq,repo):
    source=exact_sha(item.get('compiled_source'),'compiled_source')
    binding=item.get('binding')
    require(isinstance(binding,str) and re.fullmatch('[A-Za-z0-9_]+',binding),'invalid binding')
    repo.commit(source)
    directory=mq/'verified-arms'/source/binding
    manifest_path=directory/'manifest.json'
    manifest=read_json(manifest_path)
    expected_fields(manifest,dict(source_sha=source,binding=binding,
        numeric_mode=item['numeric_mode']),str(manifest_path))
    mode=item['numeric_mode']
    require(mode in ('fast','identical','deterministic'),'unknown numeric mode')
    hashes={}
    if item['kind']=='pair':
        expected_fields(manifest,dict(defines_A=item['defines_A'],defines_B=item['defines_B']),str(manifest_path))
        require(isinstance(manifest.get('hashes'),dict),'missing A/B hashes')
        for arm in ('A','B'):
            hashes[arm]=digest(directory/(arm+'.so'))
            require(hashes[arm]==manifest['hashes'].get(arm),'binary hash mismatch: '+binding+'/'+arm)
    elif item['kind']=='single':
        # Supports ibase single-artifact schema; no inferred install or import.
        filename=safe_relative(item['artifact'])
        require('/' not in filename,'single artifact must be a sibling basename')
        expected_fields(manifest,dict(artifact=filename,defines=item['defines']),str(manifest_path))
        hashes[filename]=digest(directory/filename)
        require(hashes[filename]==manifest.get('sha256'),'single binary hash mismatch: '+binding)
    else:
        raise InfrastructureError('artifact kind must be pair or single')
    expected_fields(manifest,item.get('manifest_equals',{}),str(manifest_path))
    return dict(id=item['id'],kind=item['kind'],compiled_source=source,binding=binding,
                manifest=str(manifest_path),manifest_sha256=digest(manifest_path),hashes=hashes)


def prerequisite_check(item,home):
    require(item.get('kind')=='file','prerequisite kind must be file')
    value=item.get('path')
    require(isinstance(value,str) and value,'prerequisite path required')
    path=home/value[2:] if value.startswith('~/') else Path(value)
    require(path.is_absolute(),'prerequisite path must be absolute or ~/')
    expected=item.get('sha256')
    require(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected),'prerequisite SHA256 required')
    actual=digest(path)
    require(actual==expected,'prerequisite hash mismatch: '+str(path))
    if 'json_equals' in item:
        expected_fields(read_json(path),item['json_equals'],str(path))
    return dict(id=item['id'],kind='file',path=str(path),sha256=actual)


def preflight(spec,home,brand,repo=None):
    require(brand.strip()=='Apple M3 Ultra','preflight requires Apple M3 Ultra metadata host')
    require(isinstance(spec,dict) and spec.get('version')==1,'expected version1 object spec')
    tag=spec.get('tag')
    require(isinstance(tag,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',tag),'invalid tag')
    source=exact_sha(spec.get('harness_source'),'harness_source')
    script=spec.get('script')
    args=spec.get('args')
    try:
        policy=policy_for(source,script,args)
    except ValueError as exc:
        raise InfrastructureError(str(exc)) from exc
    if policy!='reference':
        require(len(args)>=2 and args[1]==tag,'helper tag argument must equal submission tag')
    mq=home/'mq'
    state=no_duplicate(tag,mq)
    repo=repo or GitMirror(home/'mojolearn.git')
    repo.commit(source)
    script_hash=hashlib.sha256(repo.blob(source,script)).hexdigest()
    artifacts=spec.get('artifacts')
    prerequisites=spec.get('prerequisites')
    cases=spec.get('cases')
    require(isinstance(artifacts,list) and isinstance(prerequisites,list),'explicit artifacts and prerequisites lists required')
    require(isinstance(cases,list) and cases,'explicit nonempty case list required')
    dependencies={}
    for item in artifacts+prerequisites:
        require(isinstance(item,dict),'dependency must be an object')
        identifier=item.get('id')
        require(isinstance(identifier,str) and re.fullmatch('[A-Za-z0-9_.-]+',identifier),'dependency id required')
        require(identifier not in dependencies,'duplicate dependency id: '+identifier)
        try:
            dependencies[identifier]=(artifact_check(item,mq,repo) if item in artifacts
                                      else prerequisite_check(item,home))
        except KeyError as exc:
            raise InfrastructureError('missing dependency field '+str(exc)+' for '+identifier) from exc
    if policy!='reference':
        require(any(a.get('compiled_source')==args[0] for a in artifacts),
                'no declared artifact for helper compiled-source argument')
    checked=[]
    names=set()
    for case in cases:
        require(isinstance(case,dict),'case must be an object')
        name=case.get('name')
        require(isinstance(name,str) and name and name not in names,'unique case name required')
        names.add(name)
        required=case.get('requires')
        files=case.get('source_files')
        require(isinstance(required,list) and all(isinstance(x,str) for x in required),'case requires must be explicit string list: '+name)
        require(len(set(required))==len(required),'duplicate required dependency: '+name)
        require(isinstance(files,list) and all(isinstance(x,str) for x in files),'case source_files must be explicit string list: '+name)
        for identifier in required:
            require(identifier in dependencies,'missing declared dependency '+identifier+' for case '+name)
        if policy!='reference':
            artifact_ids={a['id'] for a in artifacts}
            require(bool(artifact_ids.intersection(required)),'native case lacks explicit artifact prerequisite: '+name)
        file_hashes={f:hashlib.sha256(repo.blob(source,f)).hexdigest() for f in files}
        checked.append(dict(name=name,requires=required,source_files=file_hashes))
    policy_path=Path(__file__).with_name('apple_fast_job_policy.py')
    return dict(status='READY',version=1,tag=tag,harness_source=source,script=script,args=args,
        script_sha256=script_hash,policy=policy,policy_sha256=digest(policy_path),
        cases=checked,dependencies=list(dependencies.values()),queue_snapshot=state,
        checked_at=datetime.now(timezone.utc).isoformat(),machine=brand.strip(),
        queue_changed=False,native_modules_loaded=False,symbols_validated=False,
        numerical_quality_validated=False,caller_reach_validated=False,
        requires_final_helper_gate=True,receipt_is_reservation=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    args=parser.parse_args()
    try:
        spec=read_json(Path(args.spec).expanduser())
        machine=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True)
        receipt=preflight(spec,Path.home(),machine)
    except (InfrastructureError,OSError,subprocess.SubprocessError,ValueError) as exc:
        print(json.dumps(dict(status='INFRASTRUCTURE_ERROR',error=str(exc),queue_changed=False),sort_keys=True))
        return 2
    print(json.dumps(receipt,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
