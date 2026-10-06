#!/usr/bin/env python3
"""Master six-lane A/B planner and supported local compile-only orchestrator.

Discovery/build never import an estimator. Future execution uses the existing
full-operation queue and a concrete admitted recipe; planning is not execution.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from six_lane_catalog import ROOT, STORE, catalog_document, config, unique, read, source_graph

VENDORS=('nvidia','amd','apple','host')
THREAD_ENV=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','MOJOLEARN_BENCH_THREADS')
# These pairs own competing schedules/graphs, so neither may silently mask the
# other. More detailed lane guards remain in their source and retained catalogs.
CONFLICTS=[
 ('MOJOLEARN_NN20_BALANCED_SUMMARY_TREE','MOJOLEARN_IDN_ATTENTION_V2'),
 ('MOJOLEARN_NN25_RMS_SPLIT_SCALE','MOJOLEARN_IDN_RMS_ROW_BLOCK'),
 ('MOJOLEARN_NN27_QK_ROPE_PAIR','MOJOLEARN_IDN_ROPE_CACHE'),
 ('MOJOLEARN_NN36_SHARED_DECAY','MOJOLEARN_IDN_M2_YOFF_EXP_CACHE'),
 ('MOJOLEARN_NN43_WGRAD_FIXED128','MOJOLEARN_IDN_SEQ_WGRAD_LEAF256'),
 ('MOJOLEARN_NN54_LOSS_PROFILE','MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2'),
 ('MOJOLEARN_NN14_BOUNDED_IM2COL','MOJOLEARN_NI12_IMPLICIT_CONV'),
 ('MOJOLEARN_IDN_NEURAL_NN12','MOJOLEARN_NI01_TRAINING_WORKSPACE'),
 ('MOJOLEARN_NN48_CSR_TILES','MOJOLEARN_NI55_GRAPH_FEATURE4'),
 ('MOJOLEARN_NI59_DROPOUT_CHANNEL','MOJOLEARN_NI60_DROPOUT_APPLY4'),
]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def sha_value(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def catalog():
    return read('experiments/six_lane_integration/catalog.json')


def combine(configs):
    ds={};env={};runtime={};problems=[]
    for c in configs:
        for d in c['defines']:
            key,_,value=d.partition('=');value=value or '1'
            if key in ds and ds[key]!=value:problems.append('Conflicting define '+key)
            ds[key]=value
        for dest,key in ((env,'environment'),(runtime,'runtime')):
            for k,v in c[key].items():
                if k in dest and dest[k]!=v:problems.append('Conflicting '+key+' '+k)
                dest[k]=v
    keys=set(ds)
    for a,b in CONFLICTS:
        if a in keys and b in keys:problems.append('Mutually exclusive '+a+' / '+b)
    routes=[x for x in ['NN01','NN02','NN08','NN10'] if 'MOJOLEARN_IDN_NEURAL_'+x in keys]
    if 'MOJOLEARN_IDN_NEURAL_NN09' in keys or 'MOJOLEARN_IDN_NEURAL_NN15' in keys:routes.append('staging')
    if 'MOJOLEARN_IDN_NEURAL_NN11' in keys and 'MOJOLEARN_IDN_NEURAL_NN02' not in keys:routes.append('fold')
    if len(routes)>1:problems.append('Competing neural GEMM schedules: '+','.join(routes))
    if 'MOJOLEARN_IDN_ALL_OFF' in keys:problems.append('Blanket ALL_OFF cannot define an incumbent')
    return config([k+'='+v for k,v in sorted(ds.items())],env,runtime),problems


def configurations(doc):
    entries={e['id']:e for e in doc['entries']};rows=[]
    for e in doc['entries']:
        for arm in e['arms']:
            A,problems=combine([arm['A']]);rows.append(dict(arm,id=arm['id'],members=[e['id']],mode=e['mode'],vendors=e['vendors'],A=A,problems=problems,kind='candidate'))
    for e in doc['interactions']:
        if e.get('selection_only'):
            members=[];specs=[];missing=[]
            for mid in e['members']:
                base,_,variant=mid.partition(':');item=entries.get(base)
                if not item:missing.append('Missing scoped implementation '+mid);continue
                choices=item['arms'];chosen=next((a for a in choices if a['name']==variant),choices[0]) if choices else None
                if chosen:members.append(base);specs.append(chosen)
                else:missing.append('No selectable subarm '+mid)
            A,problems=combine([a['A'] for a in specs])
            workloads=[]
            for a in specs:
                for w in a['workloads']:
                    if w not in workloads:workloads.append(w)
            mode=entries[members[0]]['mode'] if members else 'identical'
            rows.append(dict(id=e['id'],members=members,mode=mode,vendors=['apple'] if mode=='fast' else list(VENDORS),A=A,B=config(),problems=missing+problems,workloads=workloads,kind=e['kind'],rationale=e['rationale'],source_gaps=[],parameters={}))
        else:
            for arm in e.get('arms',[]):
                A,problems=combine([arm['A']]);rows.append(dict(arm,members=e.get('members',[e['id']]),mode=e['mode'],vendors=e['vendors'],A=A,problems=problems,kind=e['kind']))
    return rows


def work_id(w):
    if isinstance(w,str):return w
    return w.get('id') or w.get('key') or w.get('lane') or sha_value(w)[:12]


def matrix(doc):
    configs=configurations(doc);cells=[]
    for c in configs:
        for vendor in c['vendors']:
            for w in c.get('workloads') or [dict(id='MISSING_WORKLOAD',status='no saved workload mapping')]:
                gaps=list(c['problems'])+c.get('source_gaps',[])
                gaps+=['Full dataset/version/hash, dimensions, settings, cap audit and accepted artifacts must be supplied from the frozen saved recipe.']
                if c['A']==c['B']:gaps.append('Reused incumbent/no distinct A configuration; historical comparison is not new work')
                if c['A']['runtime']:
                    gaps.append('Declared runtime operation/settings require a matching existing saved race; never change a race to reach this arm')
                key=sha_value([c['id'],vendor,w])[:20]
                cells.append(dict(key=key,configuration=c['id'],implementation_ids=c['members'],vendor=vendor,mode=c['mode'],workload=w,workload_id=work_id(w),status='INCOMPATIBLE' if c['problems'] else 'PENDING_COVERAGE',blockers=gaps,
                    promotion_vote=c['mode']=='fast' or vendor in ('nvidia','amd'),identity_group='same-arm-across-columns' if c['mode']=='identical' else 'task-quality',planned_excluded_warmups=1,planned_scored_samples=1,actual_samples=0))
    return dict(schema='mojolearn.six-lane-matrix/1',base_main=doc['base_main'],configurations=configs,cells=cells,execution='NOT EXECUTED',qualification=doc['qualification'])


BENCHMARK_FILES=['tools/bench_board.py','tools/bench_board_neural.py','tools/bench_board_algos.py','tools/bench_board_more.py','tools/classical_two_datasets.py','tools/classical_two_datasets_2.py','tools/speed_gbdt_arm.py','bench/speed/forest_speed_arm.py','tools/bench_neural_decode.py','tools/bench_board_state.py','tools/performance_measurement_board.py','experiments/performance_ideas/README.md','pixi.lock','pyproject.toml']


def freeze_benchmarks():
    base=read('experiments/six_lane_integration/inputs.json')['base_main'];files=[]
    for f in BENCHMARK_FILES:
        try:old=subprocess.check_output(['git','-C',str(ROOT),'show',base+':'+f],stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:continue
        now=(ROOT/f).read_bytes();record=dict(path=f,base_git_blob=git('rev-parse',base+':'+f),base_sha256=hashlib.sha256(old).hexdigest(),integrated_sha256=hashlib.sha256(now).hexdigest(),changed=old!=now)
        if f.endswith('.py'):
            def decls(data):
                tree=ast.parse(data);return {','.join(n.id for n in node.targets if isinstance(n,ast.Name)):ast.dump(node.value,include_attributes=False) for node in tree.body if isinstance(node,ast.Assign)}
            before=decls(old);after=decls(now);changed={k:dict(before=v,after=after.get(k)) for k,v in before.items() if after.get(k)!=v}
            record['changed_incumbent_module_assignments']=changed
            record['added_integration_assignments']=sorted(set(after)-set(before))
        files.append(record)
    return dict(schema='mojolearn.benchmark-freeze/1',base_main=base,files=files,policy='Existing datasets, splits, seeds, preprocessing, workload sizes, estimator settings, modes, quality rules, timing definitions and opponent roster are immutable. Added capture/selector APIs do not authorize alternate races.',drift_policy='Any mismatch at queue admission blocks. Changed incumbent assignments require semantic repair before freeze.',dataset_materialization='NOT RUN; source recipes are frozen, concrete data hashes remain pending',opponent_execution='NOT RUN; opponent source/dependencies/results untouched')


def check_benchmark():
    d=read('experiments/six_lane_integration/benchmark.json')
    for e in d['files']:
        if digest(ROOT/e['path'])!=e['integrated_sha256']:raise ValueError('Benchmark source drift: '+e['path'])


def binding_paths(e,mode):
    out=[]
    for name in e.get('bindings',[]):
        name=name.removeprefix('_mojolearn_').removesuffix('.mojo')
        path='bindings/_mojolearn'+('' if name=='core' else '_'+name)+'.mojo'
        if (ROOT/path).exists():out.append(path)
    # Source closure also finds transitive callers omitted by handoffs.
    out+=e.get('source_binding_reach',[])
    return unique(p for p in out if not p.endswith('_host.mojo') or mode=='identical')


def build_plan(doc,mat):
    entries={e['id']:e for e in doc['entries']};entries.update({e['id']:e for e in doc['interactions'] if 'arms' in e})
    jobs={};blocked=[]
    for c in mat['configurations']:
        if c['problems']:
            blocked.append(dict(configuration=c['id'],status='INCOMPATIBLE',reasons=c['problems']));continue
        paths=unique([p for mid in c['members'] if mid in entries for p in binding_paths(entries[mid],c['mode'])])
        if not paths:blocked.append(dict(configuration=c['id'],status='MISSING_BINDING_MAP'));continue
        for path in paths:
            vendor='host' if path.endswith('_host.mojo') else 'apple'
            if vendor not in c['vendors']:continue
            for arm in ('B','A'):
                defines=c[arm]['defines'];params=c.get('parameters',{})
                if any('{' in d for d in defines):
                    blocked.append(dict(configuration=c['id'],arm=arm,status='UNRESOLVED_COMPILE_PARAMETERS',parameters=params));continue
                if c['mode']=='identical':defines=defines+['MOJOLEARN_NUMERIC_IDENTICAL=1']
                defines=unique(sorted(defines))
                key=sha_value([path,vendor,c['mode'],defines])[:20]
                job=jobs.setdefault(key,dict(key=key,binding=path,vendor=vendor,mode=c['mode'],defines=defines,configurations=[],status='NOT_COMPILED',target='Apple arm64 CPU' if vendor=='host' else 'Apple Metal on local Apple silicon',runtime_reach='NOT_VERIFIED'))
                job['configurations'].append(dict(configuration=c['id'],arm=arm))
    return dict(schema='mojolearn.six-lane-build-plan/1',jobs=list(jobs.values()),blocked=blocked,unsupported=[dict(vendor=v,status='NOT_COMPILED',reason='No supported local NVIDIA/AMD target; no cross-compilation or remote jobs authorized') for v in ('nvidia','amd')],policy='Deduplicated by binding, mode, vendor and exact defines. Compile success is not runtime reach; no import, smoke, quality, timing or identity execution.')


def frozen():
    status=git('status','--porcelain','--untracked-files=all')
    if status:raise ValueError('Compile requires a clean committed integration freeze')
    if not git('branch','--show-current').startswith('integration/'):raise ValueError('Expected isolated integration branch')
    return git('rev-parse','HEAD')


def compile_jobs(args):
    if platform.system()!='Darwin' or platform.machine()!='arm64':raise ValueError('This local campaign supports Apple arm64 only')
    source=frozen();check_benchmark();plan=json.loads(args.plan.read_text());compiler=args.compiler.resolve()
    if not compiler.is_file():raise ValueError('Compiler missing')
    out=args.output.resolve()
    if out.is_relative_to(ROOT):raise ValueError('Retain binaries/logs outside the source freeze')
    out.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('MOJOLEARN_') and k not in THREAD_ENV and k!='MACOSX_DEPLOYMENT_TARGET'}
    env['PATH']=str(compiler.parent)+os.pathsep+env.get('PATH','')
    home=compiler.parent.parent/'share/max'
    if (home/'modular.cfg').exists():env['MODULAR_HOME']=str(home)
    with (out/'compiler-version.log').open('w') as log:
        rc=subprocess.run([str(compiler),'--version'],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:raise ValueError('Compiler version probe failed; see compiler-version.log')
    version=(out/'compiler-version.log').read_text().strip()
    sdk=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-version'],text=True).strip()
    hardware={k:subprocess.check_output(['sysctl','-n',k],text=True).strip() for k in ['machdep.cpu.brand_string','hw.ncpu','hw.memsize']}
    manifest=dict(schema='mojolearn.six-lane-build-campaign/1',source_sha=source,compiler=str(compiler),compiler_sha256=digest(compiler),compiler_version=version,hardware=hardware,build_plan_sha256=digest(args.plan),records=[],qualification='No runtime, identity, quality or performance execution')
    closures,_=source_graph()
    chosen=[j for j in plan['jobs'] if (not args.binding or j['binding'] in args.binding) and (not args.key or j['key'] in args.key)]
    if args.limit:chosen=chosen[:args.limit]
    for job in chosen:
        if frozen()!=source:raise ValueError('Source changed during freeze')
        directory=out/job['key'];directory.mkdir(exist_ok=True)
        receipt=directory/'receipt.json'
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old.get('source_sha')==source and old.get('status')=='COMPILED' and digest(old['artifact'])==old['artifact_sha256']:
                manifest['records'].append(old);continue
            raise ValueError('Evidence already exists; choose a new campaign directory for repairs')
        artifact=directory/(Path(job['binding']).stem+'.so');log=directory/'build.log'
        flags=['--target-cpu','apple-m1']
        family=Path(job['binding']).stem.removeprefix('_mojolearn').lstrip('_')
        wrapper=ROOT/'bindings'/('build_'+family+'.sh' if family else 'build.sh')
        # Match inspected shipped Darwin builders. Neural AOT builders deliberately
        # omit --target-accelerator; expansion bindings explicitly use metal:1.
        if job['vendor']=='apple' and wrapper.exists() and '--target-accelerator metal:1' in wrapper.read_text():flags+=['--target-accelerator','metal:1']
        flags+=['-D','MOJOLEARN_COLUMN_CPU' if job['vendor']=='host' else 'MOJOLEARN_COLUMN_APPLE']
        argv=[str(compiler),'build','-j','2','--emit','shared-lib',*flags,'-Xlinker','-platform_version','-Xlinker','macos','-Xlinker','11.0','-Xlinker',sdk,'-I',str(ROOT),'-I',str(ROOT/'bindings')]
        for define in job['defines']:argv+=['-D',define]
        argv += [str(ROOT/job['binding']),'-o',str(artifact)]
        dep_hash={p:digest(ROOT/p) for p in sorted(closures.get(job['binding'],{job['binding']}))}
        record=dict(job,source_sha=source,compiler=version,compiler_sha256=manifest['compiler_sha256'],hardware=hardware,argv=argv,source_closure_sha256=sha_value(dep_hash),source_files=dep_hash,artifact=str(artifact),log=str(log),wrapper=str(wrapper),wrapper_sha256=digest(wrapper) if wrapper.exists() else None)
        write(directory/'input.json',record)
        with log.open('x') as stream:proc=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record.update(returncode=proc.returncode,status='COMPILED' if proc.returncode==0 and artifact.exists() else 'FAILED',artifact_sha256=digest(artifact) if artifact.exists() else None)
        write(receipt,record);manifest['records'].append(record);write(out/'campaign.json',manifest)
        print(json.dumps(dict(key=job['key'],binding=job['binding'],status=record['status'],log=str(log))),flush=True)
        if record['status']=='FAILED' and not args.keep_going:break
    manifest['selected_jobs']=len(chosen);manifest['completed_jobs']=len(manifest['records']);manifest['planned_jobs']=len(plan['jobs']);manifest['incomplete_jobs']=len(plan['jobs'])-len(manifest['records']);write(out/'campaign.json',manifest)
    return 1 if any(r['status']=='FAILED' for r in manifest['records']) else 0


def queue(args):
    """Produce the existing queue schema from frozen concrete admitted recipes.

    No job execution here. Incomplete cells remain blocked; never fabricate data,
    adapter commands, accepted quality, loaded binaries or opponent ratios.
    """
    check_benchmark();mat=read('experiments/six_lane_integration/matrix.json');recipes=json.loads(args.recipes.read_text()) if args.recipes else {}
    configurations_by_id={c['id']:c for c in mat['configurations']};jobs=[]
    for cell in mat['cells']:
        if cell['vendor']!=args.vendor or (args.select and cell['configuration'] not in args.select):continue
        recipe=recipes.get(cell['key'],{});c=configurations_by_id[cell['configuration']]
        job=dict(key=cell['key'],mode=cell['mode'],implementation_ids=cell['implementation_ids'],master_selection=c,workload_id=cell['workload_id'],blocked=list(cell['blockers']),arms={a:dict(configuration=c[a], argv=[sys.executable, str(ROOT/'tools/six_lane_ab_worker.py'), '--recipe', 'PENDING_RESOLVED_RECIPE', '--arm', a, '--phase', '{phase}', '--output', '{output}'], environment=c[a]['environment']) for a in ('A','B')})
        if recipe:
            required=('dataset_sha256','dimensions','estimator_settings','timed_boundary','intrinsic_caps','full_dataset_coverage','artifact_provenance','arms','benchmark_sha256')
            absent=[k for k in required if k not in recipe]
            if absent:raise ValueError('Recipe missing '+','.join(absent))
            if recipe['benchmark_sha256']!=digest(STORE/'benchmark.json'):raise ValueError('Recipe benchmark specification drift')
            if recipe.get('changes_frozen_race') or recipe['full_dataset_coverage'] is not True:raise ValueError('Recipe changes frozen race or lacks full coverage')
            if c['problems']:raise ValueError('Incompatible selection')
            for arm in ('A','B'):
                if recipe['arms'][arm].get('configuration')!=c[arm]:raise ValueError('Recipe controls differ from master '+arm)
                argv=recipe['arms'][arm].get('argv',[])
                if not argv or any('--full-tree-workload'==a for a in argv):raise ValueError('Unsupported or changed workload command')
            job.update(recipe);job['blocked']=recipe.get('remaining_blockers',[])
        jobs.append(job)
    result=dict(schema='mojolearn.full-ab-queue/1',repo=str(ROOT),source_sha=git('rev-parse','HEAD'),vendor=args.vendor,jobs=jobs,environment={},master_policy=read('experiments/six_lane_integration/evidence_policy.json'))
    write(args.output,result)
    print(json.dumps(dict(jobs=len(jobs),blocked=sum(bool(j.get('blocked')) for j in jobs),output=str(args.output),execution='NOT RUN')))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    s.add_parser('refresh',help='regenerate source-only catalog, matrix and build plan')
    ls=s.add_parser('list');ls.add_argument('--lane')
    sh=s.add_parser('show');sh.add_argument('id')
    b=s.add_parser('compile',help='compile shared libraries only, never import or execute')
    b.add_argument('--plan',type=Path,default=STORE/'build_plan.json');b.add_argument('--compiler',type=Path,required=True);b.add_argument('--output',type=Path,required=True);b.add_argument('--binding',action='append');b.add_argument('--key',action='append');b.add_argument('--limit',type=int);b.add_argument('--keep-going',action='store_true')
    q=s.add_parser('queue',help='write future queue; incomplete cells stay blocked');q.add_argument('--vendor',choices=VENDORS,required=True);q.add_argument('--recipes',type=Path);q.add_argument('--select',action='append');q.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='refresh':
        doc=catalog_document();mat=matrix(doc);build=build_plan(doc,mat);bench=freeze_benchmarks()
        for name,value in [('catalog',doc),('matrix',mat),('build_plan',build),('benchmark',bench)]:write(STORE/(name+'.json'),value)
        print(json.dumps(dict(entries=len(doc['entries']),configurations=len(mat['configurations']),matrix_cells=len(mat['cells']),build_jobs=len(build['jobs']),blocked_builds=len(build['blocked']))))
    elif args.command=='list':
        for e in catalog()['entries']:
            if not args.lane or e['lane']==args.lane:print(e['id'],e['title'])
    elif args.command=='show':
        doc=catalog();rows=[e for e in doc['entries']+doc['interactions']+doc['aliases'] if e['id']==args.id]
        if not rows:raise ValueError('Unknown namespaced ID')
        print(json.dumps(rows,indent=2))
    elif args.command=='compile':return compile_jobs(args)
    else:queue(args)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
