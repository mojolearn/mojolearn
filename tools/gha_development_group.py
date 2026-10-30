"""One selected, full-allocation development compile on a manual GHA runner.

The frozen source's builder, recipes and raw receipts are unmodified. This
wrapper retains its own provenance separately; no installed/runtime claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def workers_environment(environment, arch, cores):
    require(type(cores) is int and cores > 0, 'positive allocated CPU count required')
    require(arch in ('host', 'sm_89', 'sm_80', 'gfx942', 'gfx950'), 'unsupported target')
    env = dict(environment)
    env.update(MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_COMPILE_JOBS=str(cores),
               MOJOLEARN_LINUX_CPU='x86-64-v3', MOJOLEARN_PACKAGE_BYTE_LM='1')
    if arch == 'host':
        env.pop('MOJOLEARN_GPU_ARCHS', None)
        env.pop('MOJOLEARN_CUDA_CODE_FORMAT', None)
        env['MOJOLEARN_TARGET_COLUMN'] = 'cpu'
    elif arch.startswith('gfx'):
        # Use the supported Mojo target through the existing development
        # builder. This selects compilation only; hardware runtime and
        # cross-vendor admission still require their own retained evidence.
        env.pop('MOJOLEARN_CUDA_CODE_FORMAT', None)
        env.update(MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_TARGET_COLUMN='amd')
    else:
        env.update(MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_TARGET_COLUMN='nvidia',
                   MOJOLEARN_CUDA_CODE_FORMAT='ptx' if arch == 'sm_80' else 'native')
    return env


def selected_result(result, group, cache):
    require(result.get('schema') == 'mojolearn.development-build.v1'
            and result.get('state') in ('activated', 'incomplete'), 'producer is failed or unknown')
    require(all(r.get('exit_code') == 0 for r in result.get('results', [])),
            'a requested compile or support build failed')
    rows = [r for r in result['plan']['groups'] if r['stem'] == group]
    require(len(rows) == 1, 'selected physical group is not unique in result')
    row = rows[0]
    runs = [r for r in result['results'] if r.get('key') == row['key']]
    require(len(runs) == 1 and runs[0].get('exit_code') == 0,
            'selected group did not compile successfully')
    directory = cache/'artifacts'/row['key']
    receipt = json.loads((directory/'receipt.json').read_text())
    binary = directory/(group+'.so')
    require(receipt.get('fields') == row['fields'] and receipt.get('exit_code') == 0
            and receipt.get('sha256') == hashlib.sha256(binary.read_bytes()).hexdigest(),
            'selected raw binary/receipt differs')
    return directory, row


def retain_bootstrap(root, out):
    helper = root/'python/mojolearn/.libs/libMojolearnMath.so'
    require(helper.is_file(), 'bootstrap portable-math binary is missing')
    target = out/'bootstrap-runtime'/helper.name
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256(helper.read_bytes()).hexdigest()
    if target.exists():
        require(hashlib.sha256(target.read_bytes()).hexdigest() == expected,
                'bootstrap portable-math input changed during build')
    else:
        shutil.copy2(helper, target)
    return {str(target.relative_to(out)):expected}


def retain_inputs(root, cache, row, out):
    """Retain consumed bytes, including generated source absent from git archive."""
    retained = {}
    for relative, expected in row['fields']['source']['files'].items():
        name = Path(relative)
        require(not name.is_absolute() and '..' not in name.parts,
                'source input leaves frozen checkout')
        source = root/name
        require(hashlib.sha256(source.read_bytes()).hexdigest() == expected,
                'consumed source input changed: '+relative)
        target = out/'consumed-source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        retained[str(target.relative_to(out))] = expected
    roots = cache/'work'/row['key']/'roots'
    require(roots.is_dir() and any(roots.rglob('*.mojo')),
            'generated merged entry source is missing')
    shutil.copytree(roots, out/'generated-roots')
    for target in sorted((out/'generated-roots').rglob('*')):
        if target.is_file():
            retained[str(target.relative_to(out))] = hashlib.sha256(target.read_bytes()).hexdigest()
    retained.update(retain_bootstrap(root, out))
    (out/'retained-inputs.json').write_text(json.dumps(retained,indent=2,sort_keys=True)+'\n')
    return retained


def run(args):
    root,out=args.root.resolve(),args.out.resolve()
    require(re.fullmatch(r'[0-9a-f]{40}', args.commit), 'full source commit required')
    require(re.fullmatch(r'(?:host:|identical:)?[a-zA-Z0-9_]+', args.group), 'explicit group selector required')
    require(not out.exists() and not out.is_relative_to(root), 'new external output required')
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip()
    require(head == args.commit and not dirty, 'frozen source differs or is dirty')
    sys.path.insert(0,str(root/'tools'))
    import build_sizing
    import development_build as development
    import merged_bindings
    mode = 'host' if args.arch == 'host' else 'identical'
    matches = [stem for stem in merged_bindings.development_members(mode, 'linux', True)
               if development.selected_group(mode, stem, [args.group])]
    require(len(matches) == 1, 'selector must name exactly one existing physical group')
    requested_group, args.group = args.group, matches[0]
    cores,memory=build_sizing.box_resources()
    env=workers_environment(os.environ,args.arch,cores)
    out.mkdir(parents=True)
    # This is the small development artifact/work cache, NOT the compiler or
    # pixi cache. It lives on the bind-mounted output so timeout/failure cannot
    # discard successful support images or generated roots with docker --rm.
    cache=out/'cache'
    bootstrap_inputs=retain_bootstrap(root,out)
    command=[sys.executable,str(root/'tools/development_build.py'),'build',
             '--sets','host' if args.arch=='host' else 'identical','--group',args.group,
             '--jobs','1','--cache',str(cache),'--evidence',str(out/'evidence'),
             '--json',str(out/'evidence/result.json'),'--group-seconds','2700','--stall-seconds','600']
    record=dict(schema='mojolearn.gha-development-group.v1',source_commit=head,dirty_tracked=dirty,
                source_root=str(root),tooling_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_builder_sha256=hashlib.sha256((root/'tools/development_build.py').read_bytes()).hexdigest(),
                command=command,arch=args.arch,group=args.group,requested_group=requested_group,compiler_workers=cores,build_jobs=1,
                affinity=sorted(os.sched_getaffinity(0)),allocated_memory_bytes=memory,
                cgroup_cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
                environment={k:env[k] for k in sorted(env) if k.startswith('MOJOLEARN_') and not any(x in k for x in ('KEY','TOKEN','SECRET'))},
                bootstrap_inputs=bootstrap_inputs,status='running')
    with (out/'source.tar.gz').open('wb') as stream:
        subprocess.run(['git','-C',str(root),'archive','--format=tar.gz',head],stdout=stream,check=True)
    record['source_archive_sha256']=hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest()
    tooling = Path(__file__).resolve().parents[1]
    tooling_files = ['tools/gha_development_group.py', 'tools/gha_release_box.sh',
                     '.github/workflows/development-group-build.yml']
    record['tooling_files'] = {}
    for name in tooling_files:
        source = tooling/name
        target = out/'tooling'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        record['tooling_files'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'request.json').write_text(json.dumps(record,indent=2)+'\n')
    start=time.monotonic()
    with (out/'controller.log').open('w') as log:
        completed=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(exit_code=completed.returncode,elapsed_seconds=time.monotonic()-start)
    try:
        require(completed.returncode in (0,3), 'development builder failed; preserve complete logs')
        result=json.loads((out/'evidence/result.json').read_text())
        require(result.get('source_commit') == head, 'producer source differs')
        require((completed.returncode == 0 and result.get('state') == 'activated')
                or (completed.returncode == 3 and result.get('state') == 'incomplete'),
                'producer exit does not describe the recorded whole-set status')
        directory,row=selected_result(result,args.group,cache)
        shutil.copytree(directory,out/'retained'/row['key'])
        record['retained_input_count'] = len(retain_inputs(root,cache,row,out))
        record['retained_support_keys'] = []
        for support in result['plan'].get('support', []):
            source = cache/'artifacts'/support['key']
            require(source.is_dir(), 'support artifact missing: '+support['name'])
            shutil.copytree(source,out/'retained'/support['key'])
            record['retained_support_keys'].append(support['key'])
        record.update(status='selected-group-compiled',retained_key=row['key'],members=row['members'],
                      whole_set_state=result['state'],whole_set_missing=result.get('missing',[]),
                      scope='Only the requested group compiled; incomplete whole-set exit3 is retained, not relabeled whole-set success.')
    except Exception as error:
        record.update(status='failed',error=str(error))
    finally:
        (out/'request.json').write_text(json.dumps(record,indent=2)+'\n')
    return 0 if record['status']=='selected-group-compiled' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--group',required=True)
    p.add_argument('--arch',choices=('host','sm_89','sm_80','gfx942','gfx950'),required=True)
    return run(p.parse_args())


if __name__=='__main__':
    raise SystemExit(main())
