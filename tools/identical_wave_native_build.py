#!/usr/bin/env python3
"""Build one frozen source tree, sequential modules, fail closed import smoke.

Run only on NVIDIA/AMD Linux boxes. Never installs mojolearn or an opponent.
Failed modules remain FAILED while independent modules continue compiling.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

DEFAULT='svm,svm_host,estimators,estimators_host,x_decomp,x_decomp_host,x_prep,x_prep_host,x_trees,x_trees_host,x_sequence,x_sequence_host,gp,gp_host'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def command(argv,cwd,env,log,timeout):
    start=time.time()
    with log.open('wb') as stream:
        try:
            proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try: rc=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                rc=124
        except OSError as exc: stream.write(str(exc).encode()); rc=127
    return {'rc':rc,'elapsed_seconds':round(time.time()-start,3),'log':str(log)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sha',required=True)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--python',required=True,type=Path)
    p.add_argument('--vendor',required=True,choices=('nvidia','amd'))
    p.add_argument('--arm',choices=('on','off'),default='on')
    p.add_argument('--mode',choices=('identical','fast'),default='identical')
    p.add_argument('--builders',default=DEFAULT)
    p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args()
    if sys.platform!='linux': p.error('Linux GPU box required')
    if not re.fullmatch('[0-9a-f]{40}',a.sha): p.error('full immutable SHA required')
    builders=a.builders.split(',')
    if len(set(builders))!=len(builders) or any(not re.fullmatch('[a-z0-9_]+',b) for b in builders): p.error('unique binding names required')
    if a.mode=='fast' and any(b.endswith('_host') for b in builders): p.error('host twins support IDENTICAL only; omit host builders for FAST')
    a.repo=a.repo.resolve(); a.out=a.out.resolve(); a.python=a.python.resolve()
    a.out.mkdir(parents=True,exist_ok=False)
    source=a.out/'source'; arch={'nvidia':'sm_89','amd':'gfx942'}[a.vendor]
    env=dict(os.environ,PATH='/root/.pixi/bin:/opt/rocm/bin:'+os.environ.get('PATH',''),
             MOJOLEARN_NUMERIC_MODE=a.mode,MOJOLEARN_COMPILE_JOBS='1',MOJOLEARN_GPU_ARCHS=arch,
             MOJOLEARN_TARGET_COLUMN=a.vendor,PYTHONUNBUFFERED='1',MOJOLEARN_SKIP_BUILD_GATE='1',
             MOJOLEARN_MOJO_BUILD_FLAGS='-D MOJOLEARN_IDN_ALL_OFF=1' if a.arm=='off' else '',
             PYTHONPATH=str(source/'python'),LD_LIBRARY_PATH=str(source/'python/mojolearn/.libs')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    report={'sha':a.sha,'vendor':a.vendor,'arch':arch,'arm':a.arm,'mode':a.mode,'status':'RUNNING','expected_builders':builders,'modules':{},'bootstrap':[]}
    receipt=a.out/'native-build.json'; save(receipt,report)
    def boot(name,argv,timeout):
        row=command(argv,a.repo,env,a.out/(name+'.log'),timeout); report['bootstrap'].append(dict(id=name,**row)); save(receipt,report)
        if row['rc']: raise RuntimeError(name+' failed, see '+row['log'])
    try:
        exact=subprocess.check_output(['git','-C',str(a.repo),'rev-parse',a.sha+'^{commit}'],text=True).strip()
        if exact!=a.sha: raise RuntimeError('SHA not exact')
        boot('source-worktree',['git','worktree','add','--detach',str(source),a.sha],120)
        if not (a.repo/'.pixi/envs/default').is_dir(): raise RuntimeError('base pixi environment missing')
        (source/'.pixi').symlink_to(a.repo/'.pixi',target_is_directory=True)
        if (source/'pixi.lock').read_bytes()!=(a.repo/'pixi.lock').read_bytes(): raise RuntimeError('source pixi.lock differs from reused environment checkout')
        (source/'python/mojolearn/.libs').mkdir(parents=True,exist_ok=True)
        code="import pathlib,sys;sys.path.insert(0,"+repr(str(source/'packaging/portable_math'))+");import stage;stage.build(pathlib.Path("+repr(str(source/'python/mojolearn/.libs/libMojolearnMath.so'))+"))"
        boot('portable-math',[str(a.python),'-c',code],600)
        # These are native imports only: no estimator fit, quality run, or timing.
        for binding in builders:
            log=a.out/('build-'+binding+'.log'); build_env=dict(env)
            if binding.endswith('_host'):
                build_env.pop('MOJOLEARN_GPU_ARCHS',None); build_env['MOJOLEARN_TARGET_COLUMN']='cpu'
            row=command(['bash','bindings/build_'+binding+'.sh'],source,build_env,log,a.timeout)
            row['status']='BUILD_FAILED' if row['rc'] else 'BUILT'
            report['modules'][binding]=row; save(receipt,report)
            if row['rc']: continue
            filename='_mojolearn_'+binding+'.so'
            matches=list((source/'python/mojolearn').rglob(filename))
            matches=[m for m in matches if (m.parent.name=='host' if binding.endswith('_host') else (m.parent.name=='identical')==(a.mode=='identical'))]
            if len(matches)!=1:
                row.update(status='ARTIFACT_FAILED',artifacts=[str(x) for x in matches]); save(receipt,report); continue
            artifact=matches[0]
            expected_mode=1 if a.mode=='identical' else 0
            smoke="""import importlib.util,json,sys
path=sys.argv[1]; spec=importlib.util.spec_from_file_location(sys.argv[2],path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
names=[n for n in dir(m) if n.endswith('_numeric_mode')]
assert names,'numeric mode export absent'
modes={n:getattr(m,n)() for n in names}
assert all(v==int(sys.argv[3]) for v in modes.values()),modes
vendors={n:getattr(m,n)() for n in dir(m) if n.endswith('_vendor')}
print(json.dumps({'artifact':path,'numeric_modes':modes,'vendors':vendors}))
"""
            smoke_result=command([str(a.python),'-c',smoke,str(artifact),'_mojolearn_'+binding,str(expected_mode)],source,build_env,a.out/('import-'+binding+'.log'),120)
            row.update(status='PASS' if smoke_result['rc']==0 else 'IMPORT_FAILED',import_smoke=smoke_result,
                       artifact=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
            save(receipt,report)
            print('NATIVE_MODULE',binding,row['status'],flush=True)
        report['status']='PASS' if len(report['modules'])==len(builders) and all(v['status']=='PASS' for v in report['modules'].values()) else 'FAILED'
    except Exception as exc:
        report.update(status='FAILED',error=str(exc))
    save(receipt,report)
    print('NATIVE_BUILD',report['status'],'modules',len(report['modules']),'expected',len(builders),'receipt',receipt,flush=True)
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':sys.exit(main())
