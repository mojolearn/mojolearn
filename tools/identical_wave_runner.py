#!/usr/bin/env python3
"""Immutable SHA, fresh source ON/OFF builds; no wheels or opponent execution.

Plan schema is documented in identical_wave_plan.json. Run prepare, quality,
identity, then timing against ONE output directory. A failed/missing gate blocks
timing. Every subprocess writes full logs and preserved return codes under out.
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

from identical_wave_compare import validate_arch, validate_proof

ROOT=Path(__file__).resolve().parents[1]


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.new')
    temporary.write_text(json.dumps(obj,indent=2,default=str)+'\n')
    temporary.replace(path)



def binary_inventory(source):
    return {str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source/'python/mojolearn').rglob('*.so')}


def verify_products(source, recorded):
    if not recorded or binary_inventory(source) != recorded:
        raise ValueError('complete binary inventory changed (added, removed or modified .so)')


class ProgressSteps(list):
    def __init__(self, persist):
        super().__init__()
        self.persist = persist

    def append(self, step):
        super().append(step)
        self.persist()


def run(argv, cwd, env, log, timeout):
    log.parent.mkdir(parents=True,exist_ok=True)
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
    write(log.with_suffix('.rc.json'),{'returncode':rc,'argv':argv})
    return rc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','quality','identity','timing'))
    p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--sha',required=True)
    p.add_argument('--vendor',required=True,choices=('nvidia','amd'))
    p.add_argument('--gpu-arch',required=True,help='Actual build target, e.g. sm_120, sm_89 or gfx942; bound into every phase receipt')
    p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--python',required=True,type=Path,help='Existing Python with NumPy/SciPy; no mojolearn install used')
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--cross-vendor-proof',type=Path,help='PASS proof from identical_wave_compare.py; required for timing')
    p.add_argument('--semaphore',type=Path,default=Path('/root/mojolearn-evidence/compile_slot.sh'))
    a=p.parse_args()
    try: arch=validate_arch(a.vendor,a.gpu_arch)
    except ValueError as exc: p.error(str(exc))
    if sys.platform!='linux': p.error('remote Linux NVIDIA/AMD only')
    if not re.fullmatch('[0-9a-f]{40}',a.sha): p.error('full immutable 40-character SHA required')
    a.repo=a.repo.resolve(); a.out=a.out.resolve(); a.python=a.python.resolve(); a.data=a.data.resolve()
    plan=json.loads(a.plan.read_text()); plan_hash=hashlib.sha256(a.plan.read_bytes()).hexdigest()
    resolved=subprocess.check_output(['git','-C',str(a.repo),'rev-parse',a.sha+'^{commit}'],text=True).strip()
    if resolved!=a.sha: p.error('SHA does not resolve exactly')
    manifest=a.data.parent/'rows-small-files.sha256'
    if not manifest.is_file(): p.error('canonical rows-small file manifest missing')
    lines=manifest.read_text().splitlines()
    for line in lines:
        wanted,name=line.split(None,1); path=a.data.parent/name.strip()
        if not path.resolve().is_relative_to(a.data): p.error('data manifest path outside canonical rows-small')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=wanted: p.error('canonical data hash mismatch: '+name)
    identity={'sha':a.sha,'vendor':a.vendor,'gpu_arch':arch,'plan_sha256':plan_hash,'data_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    neural_hashes={}
    for case in plan['cases']:
        if case.get('driver')=='neural':
            fixture=a.data.parent/'wave-neural'/case['fixture']
            if not fixture.is_file(): p.error('canonical neural fixture missing: '+str(fixture)+'; run preparation once and transfer through R2')
            neural_hashes[case['fixture']]=hashlib.sha256(fixture.read_bytes()).hexdigest()
    identity['neural_fixture_sha256']=neural_hashes
    marker=a.out/'wave.json'
    if marker.exists() and json.loads(marker.read_text())!=identity: p.error('output belongs to different SHA/plan/vendor/data')
    write(marker,identity)
    backend={'nvidia':'cuda','amd':'hip'}[a.vendor]
    report={'identity':identity,'phase':a.phase,'status':'INCOMPLETE','arms':{},'opponents':'store only; never executed'}
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'MODULAR_MOJO_', 'MOJO_COMPILE_'))}
    envbase=dict(clean_env,MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_COMPILE_JOBS='1',PYTHONUNBUFFERED='1',
                 MOJOLEARN_GPU_ARCHS=arch,MOJOLEARN_TARGET_COLUMN=a.vendor,MOJOLEARN_VENDOR=backend,
                 MOJOLEARN_SKIP_BUILD_GATE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    envbase.pop('MOJOLEARN_BENCH_INSTALLED',None)
    envbase['PATH']='/root/.pixi/bin:/opt/rocm/bin:'+envbase.get('PATH','')
    required=set(plan['required_quality_gates'])
    if set(g['id'] for g in plan['quality_gates'])!=required: p.error('quality inventory does not match required gates')
    if a.phase in ('prepare','quality') and not a.semaphore.is_file(): p.error('required compile semaphore missing')
    if a.phase != 'prepare':
        prep=a.out/'prepare.json'
        if not prep.exists(): p.error('successful prepare receipt required')
        prepared=json.loads(prep.read_text())
        if prepared.get('status')!='PASS' or prepared.get('identity')!=identity:
            p.error('successful prepare receipt for this identity required')
        # Check BOTH arms before any phase executes, including unexpected .so files.
        for arm in ('on','off'):
            source=a.out/arm/'source'
            if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=a.sha:
                p.error('source HEAD changed')
            if subprocess.run(['git','diff','--quiet','HEAD','--'],cwd=source).returncode:
                p.error('tracked source changed after frozen checkout')
            try:
                verify_products(source,json.loads((a.out/arm/'build-products.json').read_text()))
            except (ValueError,OSError) as exc:
                p.error(str(exc))
    if a.phase=='timing':
        for phase in ('quality','identity'):
            receipt=a.out/(phase+'.json')
            if not receipt.exists(): p.error('timing blocked: '+phase+' receipt missing')
            doc=json.loads(receipt.read_text())
            if doc.get('identity')!=identity or doc.get('status')!='PASS': p.error('timing blocked: '+phase+' incomplete or failed')
        if a.cross_vendor_proof is None: p.error('timing blocked: --cross-vendor-proof required')
        try:
            proof_bytes=a.cross_vendor_proof.read_bytes()
            validate_proof(json.loads(proof_bytes),(a.out/'identity.json').read_bytes(),a.vendor,a.plan.read_bytes())
        except (ValueError,KeyError,TypeError,OSError) as exc:
            p.error('timing blocked: '+str(exc))
        report['cross_vendor_proof_sha256']=hashlib.sha256(proof_bytes).hexdigest()
        try:
            with (a.out/'timing.json').open('x') as reservation:
                reservation.write(json.dumps(report,indent=2)+'\n')
        except FileExistsError:
            p.error('one run per arm: timing receipt already exists')
        # The exclusive reservation survives interruption or competing runners.
    write(a.out/(a.phase+'.json'),report)
    for arm in ('on','off'):
        source=a.out/arm/'source'; folder=a.out/arm/a.phase
        env=dict(envbase,PYTHONPATH=str(source/'python'))
        flags='-D MOJOLEARN_IDN_ALL_OFF=1' if arm=='off' else ''
        env['MOJOLEARN_MOJO_BUILD_FLAGS']=flags
        if arm=='off': env['MOJOLEARN_IDN_ALL_OFF']='1'
        steps=ProgressSteps(lambda:write(a.out/(a.phase+'.json'),report)); report['arms'][arm]=steps
        if a.phase=='prepare':
            if source.exists(): p.error('fresh source directory required; existing '+str(source))
            rc=run(['git','worktree','add','--detach',str(source),a.sha],a.repo,env,folder/'worktree.log',120)
            steps.append({'id':'worktree','rc':rc});
            if rc: break
            pixi=a.repo/'.pixi'
            if not (pixi/'envs/default').exists(): p.error('locked base pixi environment absent')
            if (source/'pixi.lock').read_bytes()!=(a.repo/'pixi.lock').read_bytes(): p.error('source pixi.lock differs from environment checkout')
            (source/'.pixi').symlink_to(pixi,target_is_directory=True)
            # Portable math belongs to our source too. Never copy a published wheel.
            code="import pathlib,sys;sys.path.insert(0,'packaging/portable_math');import stage;stage.build(pathlib.Path('python/mojolearn/.libs/libMojolearnMath.so'))"
            (source/'python/mojolearn/.libs').mkdir(parents=True,exist_ok=True)
            rc=run([str(a.python),'-c',code],source,env,folder/'portable-math.log',600)
            steps.append({'id':'portable-math','rc':rc})
            if rc: break
            for builder in plan['builders']:
                if not re.fullmatch(r'build(?:_[a-z0-9_]+)?\.sh',builder): p.error('invalid builder name')
                build_env=dict(env)
                if builder.endswith('_host.sh'):
                    build_env.pop('MOJOLEARN_GPU_ARCHS',None); build_env['MOJOLEARN_TARGET_COLUMN']='cpu'
                rc=run(['bash',str(a.semaphore),'bash','bindings/'+builder],source,build_env,folder/(builder+'.log'),plan.get('build_timeout',3600))
                steps.append({'id':builder,'rc':rc})
                if rc: break
            if rc: break
            products=binary_inventory(source)
            if not products: p.error('no source-built binaries')
            write(a.out/arm/'build-products.json',products)
        else:
            if a.phase=='quality':
                for gate in plan['quality_gates']:
                    if arm not in gate.get('arms',['on','off']):
                        steps.append({'id':gate['id'],'status':'NOT_APPLICABLE','reason':'Gate plan explicitly excludes this arm'}); continue
                    if not gate.get('path'):
                        steps.append({'id':gate['id'],'status':'OWED','reason':gate['owed']}); continue
                    path=source/gate['path']; log=folder/(gate['id']+'.log')
                    if gate['kind']=='mojo':
                        cmd=['bash',str(a.semaphore),'pixi','run','mojo','run','-j','1','--target-accelerator',arch,'-D','MOJOLEARN_NUMERIC_IDENTICAL=1','-D','MOJOLEARN_COLUMN_'+a.vendor.upper(),'-I','.']
                        if arm=='off': cmd+=['-D','MOJOLEARN_IDN_ALL_OFF=1']
                        for define in gate.get('defines',[]): cmd+=['-D',define]
                        cmd+=[str(path)]
                    else:
                        cmd=[str(a.python),str(path)]
                    cmd += [v.replace('{report}',str(folder/(gate['id']+'.json'))).replace('{full_data}',str(a.data.parent/'rows-full')) for v in gate.get('args',[])]
                    rc=run(cmd,source,env,log,gate.get('timeout',3600))
                    step={'id':gate['id'],'rc':rc,'status':'PASS' if rc==0 else 'FAIL'}
                    if gate.get('owed'):
                        step.update(status='OWED' if rc==0 else 'FAIL',reason=gate['owed'])
                    steps.append(step)
            else:
                for case in plan['cases']:
                    tag=case['lane']+'--'+case['dataset']
                    if not re.fullmatch('[a-zA-Z0-9_-]+',tag): p.error('invalid case tag')
                    vendors=(backend,'cpu') if a.phase=='identity' else (backend,)
                    digests={}
                    for vendor in vendors:
                        out=folder/(tag+'--'+vendor)
                        cmd=[str(a.python),str(source/'tools/identical_wave_worker.py'),'--source',str(source),'--lane',case['lane'],'--dataset',case['dataset'],'--data',str(a.data),'--operation',a.phase,'--vendor',vendor,'--out',str(out),'--case-json',json.dumps(case),'--timing-contract',plan.get('timing_contract','board-warm-single-call')]
                        rc=run(cmd,source,dict(env,MOJOLEARN_VENDOR=vendor),folder/(tag+'--'+vendor+'.log'),case.get('timeout',1800))
                        steps.append({'id':tag+'--'+vendor,'rc':rc,'status':'PASS' if rc==0 else 'FAIL'})
                        if rc==0: digests[vendor]=json.loads((out/'result.json').read_text())['digest']
                    if a.phase=='identity': steps.append({'id':tag+'--bits','status':'PASS' if len(digests)==2 and len(set(digests.values()))==1 else 'FAIL','digests':digests})
    counts=[len(s) for s in report['arms'].values()]
    all_steps=[s for arm in report['arms'].values() for s in arm]
    good=len(report['arms'])==2 and all(counts) and all(s.get('status','PASS' if s.get('rc')==0 else 'FAIL')in ('PASS','NOT_APPLICABLE') for s in all_steps)
    if a.phase=='quality': good=good and all({s['id'] for s in steps}==required for steps in report['arms'].values())
    report['status']='PASS' if good else 'INCOMPLETE_OR_FAILED'
    write(a.out/(a.phase+'.json'),report)
    print('IDENTICAL_WAVE',a.phase,report['status'],'steps',len(all_steps),'receipt',a.out/(a.phase+'.json'))
    return 0 if good else 1

if __name__=='__main__':sys.exit(main())
