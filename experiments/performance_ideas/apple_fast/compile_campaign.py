#!/usr/bin/env python3
"""Plan/compile every independent Apple FAST card arm; never execute GPU code.

The existing M3 Ultra queue is the supported build host. Jobs deduplicate exact
(binding, tier, defines), retain all caller consumers, and attest frozen source
and artifact hashes. Compilation is not numerical/performance qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from support import ROOT


def matrix():
    jobs={};variants=[]
    def add(binding, mode, defines, consumer, source=None):
        defines=tuple(sorted(defines))
        key=(binding,mode,defines,source)
        if key not in jobs:
            jobs[key]=dict(id=f'job{len(jobs)+1:03d}',binding=binding,numeric_mode=mode,
                defines=list(defines),source=source,consumers=[])
        jobs[key]['consumers'].append(consumer)
        return jobs[key]['id']
    for n in range(1,21):
        idea=f'F{n:02d}';card=json.loads((ROOT/'experiments/performance_ideas'/idea/'manifest.json').read_text())
        for variant in sorted({'default'}|card.get('variants',{}).keys()):
            consumer=f'{idea}/{variant}';binding=card.get('variant_bindings',{}).get(variant,card['binding'])
            candidate=card.get('variants',{}).get(variant,card['candidate_defines'])
            baseline=card.get('variant_baseline_defines',{}).get(variant,card['baseline_defines'])
            arms={arm:add(binding,'fast',defines,consumer+'/'+arm) for arm,defines in [('A',baseline),('B',candidate)]}
            prerequisites=[add(dep,'fast',[],consumer+'/prerequisite') for dep in card.get('variant_prerequisite_bindings',{}).get(variant,card.get('prerequisite_bindings',['core']))]
            prerequisites.append(add('core','identical',[],consumer+'/input_transport_helpers'))
            checks=[]
            for check in card.get('native_checks',[]):
                for arm,defines in [('A',baseline),('B',candidate)]:
                    checks.append(add(check['name'],'fast',defines,consumer+'/native/'+arm,check['source']))
            variants.append(dict(idea=idea,variant=variant,arms=arms,prerequisites=prerequisites,native_checks=checks))
    return dict(schema=1,vendor='apple',target_column='apple',status='compiler_pending',qualification='none',
        jobs=list(jobs.values()),variants=variants)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--job',action='append',help='Execute selected job IDs; full plan remains exhaustive')
    args=parser.parse_args()
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if source!=args.source_sha:raise RuntimeError('frozen source SHA mismatch')
    if subprocess.run(['git','diff','--quiet','HEAD','--'],cwd=ROOT).returncode:raise RuntimeError('tracked source is dirty')
    plan=matrix();plan['source_sha']=source
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    selected=[job for job in plan['jobs'] if not args.job or job['id'] in args.job]
    if args.job and set(args.job)-{job['id'] for job in plan['jobs']}:raise RuntimeError('unknown job IDs')
    if not args.execute:
        print(f'APPLE_FAST_COMPILE_PLAN variants={len(plan["variants"])} jobs={len(plan["jobs"])} source={source} evidence={args.output}')
        return
    chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    if 'Apple M3 Ultra' not in chip:raise RuntimeError('compile campaign requires existing M3 Ultra queue')
    slot=Path.home()/'mojolearn-evidence/compile_slot.sh'
    if not slot.is_file():raise RuntimeError('existing compiler semaphore missing')
    results=[]
    for job in selected:
        directory=args.output/job['id'];directory.mkdir()
        binary=ROOT/'python/mojolearn'
        if job['numeric_mode']=='identical':binary=binary/'identical'
        binary=binary/('_mojolearn.so' if job['binding']=='core' else '_mojolearn_'+job['binding']+'.so')
        flags=' '.join('-D '+token for token in job['defines'])
        env=dict(os.environ,MOJOLEARN_NUMERIC_MODE=job['numeric_mode'],MOJOLEARN_VENDOR='apple',
            MOJOLEARN_TARGET_COLUMN='apple',MOJOLEARN_COMPILE_JOBS='1',MOJOLEARN_SKIP_BUILD_GATE='1',
            MOJOLEARN_MOJO_BUILD_FLAGS=flags,MOJOLEARN_BUILD_EXTRA_DEFINES='')
        if job['source']:
            binary=directory/'native-check'
            command=['pixi','run','mojo','build','-j','1','--target-cpu','apple-m1','--target-accelerator','metal:1','-I','.','-I','bindings']
            for token in job['defines']:command.extend(['-D',token])
            command.extend([job['source'],'-o',str(binary)])
        else:
            script='bindings/build.sh' if job['binding']=='core' else 'bindings/build_'+job['binding']+'.sh'
            command=['bash',script]
            if job['binding']=='byte_lm':
                env['MOJOLEARN_BYTE_LM_OUTDIR']=str(directory/'byte-lm-output')
                binary=Path(env['MOJOLEARN_BYTE_LM_OUTDIR'])/'_mojolearn_byte_lm.so'
        log=directory/'build.log'
        with log.open('x') as stream:
            rc=subprocess.run(['bash',str(slot)]+command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        artifact=None;digest=None
        if rc==0 and binary.is_file():
            artifact=directory/('native-check' if job['source'] else binary.name)
            if binary.resolve()!=artifact.resolve():shutil.copy2(binary,artifact)
            digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
        result=dict(**job,source_sha=source,vendor='apple',builder=chip,exit_code=rc,
            status='compile_pass' if rc==0 and artifact else 'compile_failed',log=str(log),
            artifact=str(artifact) if artifact else None,sha256=digest,device_quality='pending',performance='pending')
        results.append(result)
        (directory/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(f'APPLE_FAST_COMPILE job={job["id"]} status={result["status"]} rc={rc} log={log}',flush=True)
    failed=sum(result['status']!='compile_pass' for result in results)
    summary=dict(source_sha=source,total_jobs=len(plan['jobs']),selected_jobs=len(selected),compiled=len(results),failed=failed,
        complete_matrix=len(results)==len(plan['jobs']),device_quality='pending',performance='pending')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    raise SystemExit(1 if failed else 0)

if __name__=='__main__':main()
