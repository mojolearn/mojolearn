#!/usr/bin/env python3
"""Build one frozen source tree, sequential modules, fail closed import smoke.

Run only on NVIDIA/AMD Linux boxes. Never installs mojolearn or an opponent.
Failed modules remain FAILED while independent modules continue compiling.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

DEFAULT='base,base_host,svm,svm_host,estimators,estimators_host,x_decomp,x_decomp_host,x_prep,x_prep_host,x_trees,x_trees_host,x_sequence,x_sequence_host,gp,gp_host'


def explicit_defines(values):
    """Only compile definitions, never shell fragments or mode/vendor overrides."""
    result=[]; seen=set()
    for value in values:
        if not re.fullmatch(r'MOJOLEARN_[A-Z0-9_]+(?:=[1-9][0-9]*)?',value):
            raise ValueError('invalid explicit compile define: '+value)
        name=value.split('=',1)[0]
        if name=='MOJOLEARN_IDN_ALL_OFF' or name.startswith(('MOJOLEARN_NUMERIC_','MOJOLEARN_COLUMN_')):
            raise ValueError('use --arm/--mode/--vendor instead of overriding '+name)
        if name in seen: raise ValueError('duplicate explicit compile define: '+name)
        seen.add(name); result.append(value if '=' in value else value+'=1')
    return result


def candidate_recipe(repo,sha,name,role):
    """Read recipes from the numerical source commit, not a mutable checkout."""
    payload=subprocess.check_output(['git','-C',str(repo),'show',sha+':tools/identical_candidate_recipes.json'])
    document=json.loads(payload)
    matches=[r for r in document['recipes'] if r['id']==name]
    if len(matches)!=1: raise ValueError('unknown/ambiguous candidate recipe: '+name)
    row=matches[0]
    return row, row[role+'_defines'], hashlib.sha256(payload).hexdigest()


def neural_recipe(repo,sha,identities,variant_specs,role):
    """Resolve neural controls from the frozen source, including helper provenance.

    The current helper implementation is usable only when both helper files
    match the requested commit byte for byte. Lane records and the caller map
    always come from git-show; mutable working-tree metadata is never an arm.
    """
    hashes={}
    def frozen(path):
        payload=subprocess.check_output(['git','-C',str(repo),'show',sha+':'+path])
        hashes[path]=hashlib.sha256(payload).hexdigest()
        return payload

    helper_dir=Path(__file__).resolve().parent
    names=('neural_identical_ideas','neural_identical_integration')
    for name in names:
        path='tools/'+name+'.py'
        if (helper_dir/(name+'.py')).read_bytes()!=frozen(path):
            raise ValueError('neural selection helper differs from --sha: '+path+
                             '; use the frozen checkout tools')
    # Load by the exact compared path, not by a potentially different module
    # on PYTHONPATH. The integration helper imports this same registered ideas
    # module when it constructs the two common source plans.
    helpers={}
    for name in names:
        spec=importlib.util.spec_from_file_location(name,helper_dir/(name+'.py'))
        if spec is None or spec.loader is None:
            raise ValueError('cannot load frozen-matching neural helper: '+name)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
        helpers[name]=module
    ideas=helpers['neural_identical_ideas']
    integration=helpers['neural_identical_integration']
    variants=ideas.variant_selections(variant_specs)
    cards=[]; seen=set()
    directory=ideas.DIRECTORY.relative_to(ideas.ROOT).as_posix()
    for lane in ideas.LANES:
        document=json.loads(frozen(directory+'/'+lane+'.json'))
        for entry in document['experiments']:
            identity=entry['id']
            if identity in seen:
                raise ValueError('duplicate frozen neural lane record: '+identity)
            seen.add(identity)
            cards.append(dict(entry,implementation_lane=lane))
    pair=integration.selection(cards,identities,variants)
    integration.require_selectable(pair)
    mapping=json.loads(frozen(integration.MAPPING))
    bindings=set(); workloads=set()
    selected_ids=pair['A']['ids']
    for identity in selected_ids:
        row=mapping['ideas'][identity]
        bindings.update(row['binding_families'])
        workloads.update(row['workloads'])
    builders=[name for family in sorted(bindings)
              for name in ((family,) if family.endswith('_host') else (family,family+'_host'))]
    builders=list(dict.fromkeys(builders))
    if not builders:
        raise ValueError('neural selection has no recorded caller binding builders')
    selected_arm='A' if role=='candidate' else 'B'
    plan=pair[selected_arm]
    return {
        'schema':'mojolearn.neural-identical-native-build-selection/1',
        'source_sha':sha,'ids':selected_ids,'variants':variants,
        'recipe_role':role,'selected_arm':selected_arm,
        'source_selection':pair,
        'recommended_builders':builders,
        'runtime_environment':plan['runtime_environment'],
        'affected_workloads':{name:mapping['workloads'][name] for name in sorted(workloads)},
        'source_file_sha256':hashes,
        'helper_policy':'Current helper bytes match frozen commit before import; lane and integration metadata read with git show.',
    }, integration.compiler_defines(plan)


def define_source_locations(repo,sha,defines):
    locations={}
    for value in defines:
        name=value.split('=',1)[0]
        found=subprocess.run(['git','-C',str(repo),'grep','-n','-w','-F','-e',name,sha,'--','*.mojo'],
                             capture_output=True,text=True)
        if found.returncode not in (0,1): raise RuntimeError('source define lookup failed: '+name)
        references=[]
        for line in found.stdout.splitlines():
            parts=line.split(':',3)
            if len(parts)!=4: continue
            code=parts[3].split('#',1)[0]
            if '"'+name+'"' in code:
                references.append({'path':parts[1],'line':int(parts[2])})
        if not references: raise ValueError('define has no compiled source reference at pinned SHA: '+name)
        locations[name]=references
    return locations


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
    p.add_argument('--gpu-arch', help='Actual device target; required for new NVIDIA architectures such as sm_120')
    p.add_argument('--arm',choices=('on','off'),default='on')
    p.add_argument('--mode',choices=('identical','fast'),default='identical')
    p.add_argument('--define',action='append',default=[],help='Explicit MOJOLEARN_NAME[=positive_integer]; omit presence flags to disable, never use =0')
    p.add_argument('--candidate-recipe',help='ID from tools/identical_candidate_recipes.json at --sha')
    p.add_argument('--neural-idea',action='append',default=[],metavar='NIxx',
                   help='Neural idea from frozen lane records; repeat to combine ideas')
    p.add_argument('--neural-variant',action='append',default=[],metavar='NIxx=NAME',
                   help='Recorded variant for a selected --neural-idea; repeat for different ideas')
    p.add_argument('--recipe-role',choices=('baseline','candidate'),default=None)
    p.add_argument('--builders',help='Comma list; defaults to recipe dependencies or the standard batch')
    p.add_argument('--plan-only',action='store_true',help='Validate frozen source/recipe/flags and print receipt without creating a worktree or compiling')
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--semaphore',type=Path,default=Path('/root/mojolearn-evidence/compile_slot.sh'))
    a=p.parse_args()
    if sys.platform!='linux': p.error('Linux GPU box required')
    if not re.fullmatch('[0-9a-f]{40}',a.sha): p.error('full immutable SHA required')
    a.repo=a.repo.resolve(); a.out=a.out.resolve(); a.python=a.python.resolve()
    recipe=None; recipe_hash=None; neural=None
    try:
        values=list(a.define)
        if a.neural_idea:
            if a.mode!='identical': raise ValueError('neural ideas require --mode identical')
            if a.candidate_recipe or a.define or a.arm=='off':
                raise ValueError('--neural-idea cannot mix with --candidate-recipe, --define, or --arm off; each idea records its own A/B controls')
            neural,values=neural_recipe(a.repo,a.sha,a.neural_idea,a.neural_variant,a.recipe_role or 'candidate')
        elif a.neural_variant:
            raise ValueError('--neural-variant requires --neural-idea')
        elif a.candidate_recipe:
            if a.mode!='identical': raise ValueError('IDENTICAL recipes require --mode identical')
            recipe, recipe_values, recipe_hash=candidate_recipe(a.repo,a.sha,a.candidate_recipe,a.recipe_role or 'candidate')
            values=recipe_values+values
        elif a.recipe_role:
            raise ValueError('--recipe-role requires --candidate-recipe or --neural-idea')
        defines=explicit_defines(values)
        references=define_source_locations(a.repo,a.sha,defines)
    except (ValueError,RuntimeError,KeyError,OSError,subprocess.CalledProcessError) as exc:
        p.error(str(exc))
    recommended=(neural or recipe or {}).get('recommended_builders')
    builders=(a.builders or (','.join(recommended) if recommended else DEFAULT)).split(',')
    if len(set(builders))!=len(builders) or any(not re.fullmatch('[a-z0-9_]+',b) for b in builders): p.error('unique binding names required')
    if recipe and not set(recipe['recommended_builders']).issubset(builders):
        p.error('recipe dependency builders missing; use --define for an explicitly narrower diagnostic')
    if neural and not set(neural['recommended_builders']).issubset(builders):
        p.error('neural caller dependency builders missing from --builders')
    if a.mode=='fast' and any(b.endswith('_host') for b in builders): p.error('host twins support IDENTICAL only; omit host builders for FAST')
    if not a.semaphore.is_file(): p.error('required compile semaphore missing: '+str(a.semaphore))
    source=a.out/'source'; arch=a.gpu_arch or {'nvidia':'sm_89','amd':'gfx942'}[a.vendor]
    if not re.fullmatch(r'sm_[0-9]+[a-z]?' if a.vendor=='nvidia' else r'gfx[0-9a-f]+',arch):
        p.error('GPU architecture does not match vendor')
    clean_env={k:v for k,v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'MODULAR_MOJO_', 'MOJO_COMPILE_'))}
    env=dict(clean_env,PATH='/root/.pixi/bin:/opt/rocm/bin:'+os.environ.get('PATH',''),
             MOJOLEARN_NUMERIC_MODE=a.mode,MOJOLEARN_COMPILE_JOBS='1',MOJOLEARN_GPU_ARCHS=arch,
             MOJOLEARN_TARGET_COLUMN=a.vendor,PYTHONUNBUFFERED='1',MOJOLEARN_SKIP_BUILD_GATE='1',
             MOJOLEARN_MOJO_BUILD_FLAGS=' '.join('-D '+d for d in (['MOJOLEARN_IDN_ALL_OFF=1'] if a.arm=='off' else [])+defines),
             PYTHONPATH=str(source/'python'),LD_LIBRARY_PATH=str(source/'python/mojolearn/.libs')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    if neural:
        for key,value in neural['runtime_environment'].items():
            if key in env and env[key]!=value:
                p.error('neural runtime environment conflicts with managed build environment: '+key)
            env[key]=value
    report={'sha':a.sha,'vendor':a.vendor,'arch':arch,'arm':a.arm,'mode':a.mode,'status':'RUNNING','expected_builders':builders,'modules':{},'bootstrap':[]}
    report.update(explicit_defines=defines,define_source_locations=references,
                  effective_mojo_build_flags=env['MOJOLEARN_MOJO_BUILD_FLAGS'],
                  candidate_recipe=recipe,recipe_role=a.recipe_role or ('candidate' if recipe or neural else None),
                  candidate_recipe_file_sha256=recipe_hash,
                  candidate_qualification='BUILD_ONLY; fixture route/identity/quality/timing still required' if recipe or neural or defines else None)
    if neural:
        report.update(neural_selection=neural,
                      runtime_environment=neural['runtime_environment'],
                      candidate_qualification='BUILD_ONLY; same-version host/NVIDIA/AMD/Apple identity, full-workload quality and end-to-end A/B timing still required')
    if a.plan_only:
        report['status']='PLANNED_NOT_BUILT'
        print(json.dumps(report,indent=2))
        return 0
    a.out.mkdir(parents=True,exist_ok=False)
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
            builder={'base':'build.sh','base_host':'build_core_host.sh'}.get(binding,'build_'+binding+'.sh')
            row=command(['bash',str(a.semaphore),'bash','bindings/'+builder],source,build_env,log,a.timeout)
            row['status']='BUILD_FAILED' if row['rc'] else 'BUILT'
            report['modules'][binding]=row; save(receipt,report)
            if row['rc']: continue
            module={'base':'_mojolearn','base_host':'_mojolearn_core_host'}.get(binding,'_mojolearn_'+binding)
            filename=module+'.so'
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
assert vendors,'vendor export absent'
expected_vendor=sys.argv[4]
assert all(v==expected_vendor for v in vendors.values()),(vendors,expected_vendor)
print(json.dumps({'artifact':path,'numeric_modes':modes,'vendors':vendors,'expected_vendor':expected_vendor}))
"""
            smoke_result=command([str(a.python),'-c',smoke,str(artifact),module,str(expected_mode),'cpu' if binding.endswith('_host') else {'nvidia':'cuda','amd':'hip'}[a.vendor]],source,build_env,a.out/('import-'+binding+'.log'),120)
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
