#!/usr/bin/env python3
"""Master six-lane A/B planner and compile-only orchestrator.

Discovery/build never import an estimator. Future execution uses the existing
full-operation queue and a concrete admitted recipe; planning is not execution.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import functools
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from six_lane_matrix_io import read_matrix, write_matrix
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
 ('MOJOLEARN_IDN_NEURAL_NN02','MOJOLEARN_NI02_GEMM_STREAM_PARTIALS'),
 ('MOJOLEARN_IDN_NEURAL_NN03','MOJOLEARN_NI08_GEMM_LEAF_256'),
 ('MOJOLEARN_NN34_AFFINE_PREFIX','MOJOLEARN_IDN_M1_STATE_WINDOW'),
 ('MOJOLEARN_NN39_M2_GRAD_TREE','MOJOLEARN_IDN_M2_GRAD_LEAF128'),
 ('MOJOLEARN_NN53_HEAD_CHUNK512','MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2'),
 ('MOJOLEARN_FOREST_ORDERED_RESIDENT_OFF','MOJOLEARN_AFT_P02'),
 ('MOJOLEARN_AFT_P07','MOJOLEARN_SHAP_FAST_ROW_PAIR'),
 ('MOJOLEARN_AFCL_G01','MOJOLEARN_KNN_FAST_MMA_OFF'),
 ('MOJOLEARN_IDN_NEURAL_NN01','MOJOLEARN_IDN_NEURAL_NN03'),
 ('MOJOLEARN_IDN_NEURAL_NN01','MOJOLEARN_IDN_NEURAL_NN04'),
 ('MOJOLEARN_IDN_GEMM_FOLD_LEAF_64','MOJOLEARN_NI08_GEMM_LEAF_256'),
 ('MOJOLEARN_NN53_HEAD_CHUNK512','MOJOLEARN_NN53_HEAD_CHUNK2048'),
 ('MOJOLEARN_AFN26_MAMBA1_CHUNKS16','MOJOLEARN_AFN26_MAMBA1_CHUNKS64'),
 ('MOJOLEARN_AFN26_MAMBA3_THREADS64','MOJOLEARN_AFN26_MAMBA3_THREADS256'),
 ('MOJOLEARN_AFN26_EMB_THREADS64','MOJOLEARN_AFN26_EMB_THREADS128'),
 ('MOJOLEARN_AFN26_ATTN_NORM_TPB128','MOJOLEARN_AFN26_ATTN_NORM_TPB512'),
 ('MOJOLEARN_C52_PAIR_128','MOJOLEARN_C52_PAIR_512'),
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
    for key in keys:
        if key.endswith('_OFF') and key[:-4] in keys:problems.append('Enabled and disabled simultaneously: '+key[:-4])
    for a,b in CONFLICTS:
        if a in keys and b in keys:problems.append('Mutually exclusive '+a+' / '+b)
    routes=[x for x in ['NN01','NN02','NN08','NN10'] if 'MOJOLEARN_IDN_NEURAL_'+x in keys]
    if 'MOJOLEARN_IDN_NEURAL_NN09' in keys or 'MOJOLEARN_IDN_NEURAL_NN15' in keys:routes.append('staging')
    if 'MOJOLEARN_IDN_NEURAL_NN11' in keys and 'MOJOLEARN_IDN_NEURAL_NN02' not in keys:routes.append('fold')
    if len(routes)>1:problems.append('Competing neural GEMM schedules: '+','.join(routes))
    if 'MOJOLEARN_IDN_ALL_OFF' in keys:problems.append('Blanket ALL_OFF cannot define an incumbent')
    return config([k+'='+v for k,v in sorted(ds.items())],env,runtime),problems


def arm_problems(arms,combined):
    keys={d.split('=')[0] for d in combined['defines']};problems=[]
    for arm in arms:
        if not arm.get('source_selectable',True):problems.append('Source arm is not selectable: '+arm['id'])
        for conflict in arm.get('conflicts',[]):
            if isinstance(conflict,str) and conflict.startswith('MOJOLEARN_') and conflict.split('=')[0] in keys:problems.append(arm['id']+' excludes '+conflict)
    return unique(problems)


def configurations(doc):
    entries={e['id']:e for e in doc['entries']};rows=[]
    aliases={a['id']:a for a in doc['aliases'] if a['kind']=='equivalent_implementation'}
    for e in doc['entries']:
        for arm in e['arms']:
            A,problems=combine([arm['A']]);problems+=arm_problems([arm],A)
            rows.append(dict(arm,id=arm['id'],members=[e['id']],mode=e['mode'],vendors=e['vendors'],A=A,problems=problems,kind='candidate',campaign_role=e['campaign_role'],source_gaps=unique(arm['source_gaps']+e['gaps']),alias_of=aliases.get(arm['id'],{}).get('canonical')))
    for e in doc['interactions']:
        if e.get('selection_only'):
            members=[];specs=[];missing=[e['pending_reason']] if e.get('pending_reason') else [];excluded=[];compile_specs=[]
            ordered=sorted(e['members'],key=lambda mid:0 if mid=='I.N.NN02' else 1) if e['kind']=='complete_proposed' else e['members']
            for mid in ordered:
                base,_,variant=mid.partition(':');item=entries.get(base)
                if not item:missing.append('Missing scoped implementation '+mid);continue
                choices=item['arms'];chosen=next((a for a in choices if a['name']==variant),choices[0]) if choices else None
                if not chosen:missing.append('No selectable subarm '+mid);continue
                proposed=e['kind']=='complete_proposed'
                if proposed and (item['campaign_role']!='new_candidate' or chosen['id'] in aliases):
                    excluded.append(dict(id=chosen['id'],reason=item['campaign_role'] if chosen['id'] not in aliases else 'Equivalent '+aliases[chosen['id']]['canonical']));continue
                # Public operations belong to individual saved workloads. They
                # are retained per member, never overwritten by a global flag.
                cs=dict(chosen['A'],runtime={})
                trial,incompatible=combine(compile_specs+[cs]);incompatible+=arm_problems(specs+[chosen],trial)
                if proposed and incompatible:
                    excluded.append(dict(id=chosen['id'],reason=incompatible));continue
                members.append(base);specs.append(chosen);compile_specs.append(cs)
            A,problems=combine(compile_specs);problems+=arm_problems(specs,A)
            workloads=[]
            for arm in specs:
                for w in arm['workloads']:
                    if w not in workloads:workloads.append(w)
            mode=entries[members[0]]['mode'] if members else 'identical'
            rows.append(dict(id=e['id'],members=members,selected_subarms=[a['id'] for a in specs],excluded_alternatives=excluded,mode=mode,vendors=['apple'] if mode=='fast' else list(VENDORS),A=A,B=config(),problems=missing+problems,workloads=workloads,kind=e['kind'],campaign_role='new_interaction',rationale=e['rationale'],source_gaps=unique(g for a in specs for g in a['source_gaps']),parameters={},runtime_by_member={a['id']:a['A']['runtime'] for a in specs if a['A']['runtime']}))
        else:
            for arm in e.get('arms',[]):
                A,problems=combine([arm['A']]);rows.append(dict(arm,members=e.get('members',[e['id']]),mode=e['mode'],vendors=e['vendors'],A=A,problems=problems,kind=e['kind'],campaign_role='new_interaction'))
    return rows


def work_id(w):
    if isinstance(w,str):return w
    return w.get('id') or w.get('key') or w.get('lane') or sha_value(w)[:12]


def expand_workload(value):
    return _expand_workload_cached(json.dumps(value,sort_keys=True))


@functools.lru_cache(maxsize=None)
def _expand_workload_cached(encoded):
    value=json.loads(encoded)
    if isinstance(value,str):
        for prefix,source in [('classical:', 'tools/classical_two_datasets.py'),('classical/', 'tools/classical_two_datasets.py'),('more:', 'tools/bench_board_more.py'),('expanded:', 'tools/bench_board_algos.py'),('classical2/','tools/bench_board_more.py'),('algos/','tools/bench_board_algos.py')]:
            if value.startswith(prefix):
                return [dict(id=value+'@dataset='+dataset,source_workload=value,harness=source,dataset=dataset,recipe='Saved full '+dataset+' race; intrinsic caps and additional operation suffixes remain unresolved') for dataset in ('taxi','istella')]
    if isinstance(value,dict) and value.get('affected_models'):
        tree=ast.parse((ROOT/'tools/bench_board_neural.py').read_text())
        mapping=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODEL_OF' for t in n.targets))
        aliases={'LM':['lm'],'MLP':['mlp'],'Mamba/Samba':['mamba1','mamba2','mamba3','samba'],'transformer':['transformer'],'CNN':[]}
        models=unique(m for model in value['affected_models'] for m in aliases.get(model,[model.lower()]))
        rows=[dict(value,id='neural:'+lane,lane=lane,model=model,recipe_status='Saved full shape; column-specific dimensions/caps require audit') for lane,model in mapping.items() if model in models]
        if 'CNN' in value['affected_models']:rows.append(dict(value,id='neural:CNN',recipe_status='Resolve existing expanded CNN classifier/regressor races; no new race'))
        return rows or [value]
    return [value]


def matrix(doc):
    configs=configurations(doc);cells=[]
    for c in configs:
        for vendor in c['vendors']:
            workloads=[w for value in c.get('workloads',[]) for w in expand_workload(value)]
            for w in workloads or [dict(id='MISSING_WORKLOAD',status='no saved workload mapping')]:
                gaps=list(c['problems'])
                gaps+=['Full dataset/version/hash, dimensions, settings, cap audit and accepted artifacts must be supplied from the frozen saved recipe.']
                if c['A']==c['B']:gaps.append('Reused incumbent/no distinct A configuration; historical comparison is not new work')
                if runtime_requirements(c,work_id(w),{item['id']:item for item in configs}):
                    gaps.append('Declared runtime operation/settings require a matching existing saved race; never change a race to reach this arm')
                key=sha_value([c['id'],vendor,w])[:20]
                cells.append(dict(key=key,configuration=c['id'],implementation_ids=c['members'],vendor=vendor,mode=c['mode'],workload=w,workload_id=work_id(w),status='INCOMPATIBLE' if c['problems'] else 'ALIAS' if c.get('alias_of') else 'RETAINED_DEPENDENCY' if c['campaign_role']=='incumbent_dependency' else 'SOURCE_REJECTED' if c['campaign_role']=='source_rejected' else 'PENDING_COVERAGE',blockers=gaps,source_coverage_pending=c.get('source_gaps',[]),
                    campaign_role=c['campaign_role'],alias_of=c.get('alias_of'),promotion_vote=c['mode']=='fast' or vendor in ('nvidia','amd'),identity_group='same-arm-across-columns' if c['mode']=='identical' else 'task-quality',planned_excluded_warmups=1,planned_scored_samples=1,actual_samples=0))
    from six_lane_full_variants import append_registered_cells
    cells = append_registered_cells(cells)
    return dict(schema='mojolearn.six-lane-matrix/1',base_main=doc['base_main'],configurations=configs,cells=cells,execution='NOT EXECUTED',qualification=doc['qualification'])


BENCHMARK_FILES=['tools/bench_board.py','tools/bench_board_neural.py','tools/bench_board_algos.py','tools/bench_board_more.py','tools/classical_two_datasets.py','tools/classical_two_datasets_2.py','tools/speed_gbdt_arm.py','bench/speed/forest_speed_arm.py','tools/bench_neural_decode.py','tools/bench_board_state.py','tools/performance_measurement_board.py','experiments/performance_ideas/README.md','pixi.lock','pyproject.toml','tools/knn_datasets.py','tools/bench_board_params.py','tools/torch_lm_step_opponent.py','tools/speed_torch_seq.py']


def freeze_benchmarks(base=None):
    # Keep a reviewed benchmark reference across metadata refreshes. A merge
    # may explicitly advance it while retaining the previous freeze document.
    if base is None:
        spec=STORE/'benchmark.json'
        base=(json.loads(spec.read_text())['base_main'] if spec.exists()
              else read('experiments/six_lane_integration/inputs.json')['base_main'])
    files=[]
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
    return unique(p for p in out if ('probe' not in p and 'check' not in p) and (not p.endswith('_host.mojo') or mode=='identical'))


def build_plan(doc,mat):
    entries={e['id']:e for e in doc['entries']};entries.update({e['id']:e for e in doc['interactions'] if 'arms' in e})
    jobs={};blocked=[]
    for c in mat['configurations']:
        if c['campaign_role'] in ('incumbent_dependency','source_rejected'):
            blocked.append(dict(configuration=c['id'],status='RETAINED_DEPENDENCY' if c['campaign_role']=='incumbent_dependency' else 'SOURCE_REJECTED'));continue
        if c['problems']:
            blocked.append(dict(configuration=c['id'],status='INCOMPATIBLE',reasons=c['problems']));continue
        paths=unique([p for mid in c['members'] if mid in entries for p in binding_paths(entries[mid],c['mode'])])
        if not paths:blocked.append(dict(configuration=c['id'],status='MISSING_BINDING_MAP'));continue
        for path in paths:
            supported=['host'] if path.endswith('_host.mojo') else (['apple','nvidia','amd'] if c['mode']=='identical' else ['apple'])
            for vendor in supported:
              if vendor not in c['vendors']:continue
              for arm in ('B','A'):
                  defines=c[arm]['defines'];params=c.get('parameters',{})
                  if any('{' in d for d in defines):
                      blocked.append(dict(configuration=c['id'],arm=arm,status='UNRESOLVED_COMPILE_PARAMETERS',parameters=params));continue
                  if c['mode']=='identical':defines=defines+['MOJOLEARN_NUMERIC_IDENTICAL=1']
                  defines=unique(sorted(defines))
                  key=sha_value([path,vendor,c['mode'],defines])[:20]
                  job=jobs.setdefault(key,dict(key=key,binding=path,vendor=vendor,mode=c['mode'],defines=defines,configurations=[],status='NOT_COMPILED',target={'host':'native CPU host binding','apple':'Apple Metal on local Apple silicon','nvidia':'NVIDIA shipped/default or native architecture on authorized worker','amd':'native AMD gfx architecture on authorized worker'}[vendor],runtime_reach='NOT_VERIFIED'))
                  job['configurations'].append(dict(configuration=c['id'],arm=arm))
    return dict(schema='mojolearn.six-lane-build-plan/1',jobs=list(jobs.values()),blocked=blocked,unsupported=[],policy='Deduplicated by binding, mode, vendor and exact defines. Compile success is not runtime reach; no import, smoke, quality, timing or identity execution.')


def frozen(manifest=None):
    if manifest is not None:
        d=json.loads(Path(manifest).read_text())
        if d['branch']!='main' and not d['branch'].startswith('integration/'):
            raise ValueError('Archive is not a main or integration freeze')
        for p,h in d['files'].items():
            if digest(ROOT/p)!=h:raise ValueError('Archive source drift: '+p)
        return d['source_sha']
    status=git('status','--porcelain','--untracked-files=all')
    if status:raise ValueError('Compile requires a clean committed integration freeze')
    branch=git('branch','--show-current')
    if branch!='main' and not branch.startswith('integration/'):raise ValueError('Expected main or an integration branch')
    return git('rev-parse','HEAD')


def comparable_compile_argv(values, binding):
    """Ignore relocation only; callers still compare compiler and closure hashes.

    Source archives and Apple build/timing machines use different checkout
    roots. Only the known compiler, source-root includes, binding input and
    output locations are relocatable. Preserve all target/define/other flags.
    """
    values = list(values)
    if len(values) < 4 or values[-2] != '-o':
        return values
    source = Path(values[-3])
    suffix = Path(binding)
    if source.parts[-len(suffix.parts):] != suffix.parts:
        return values
    root = source
    for _ in suffix.parts:
        root = root.parent
    normalized = ['<compiler>', *values[1:-2]]
    normalized[-1] = '<source>/' + binding
    for index in range(1, len(normalized)-1):
        if normalized[index-1] == '-I':
            if normalized[index] == str(root):
                normalized[index] = '<source>'
            elif normalized[index] == str(root/'bindings'):
                normalized[index] = '<source>/bindings'
    return normalized


def compile_jobs(args):
    apple=platform.system()=='Darwin' and platform.machine()=='arm64'
    linux=platform.system()=='Linux' and platform.machine()=='x86_64'
    if not (apple or linux):raise ValueError('Supported compile hosts are Apple arm64 or Linux x86_64')
    if linux and args.vendor not in ('nvidia','amd','host'):raise ValueError('Linux compilation requires an explicit nvidia, amd or host selection')
    if apple and args.vendor in ('nvidia','amd'):raise ValueError('NVIDIA/AMD compilation requires an authorized Linux worker')
    if args.vendor=='amd' and not re.fullmatch(r'gfx[0-9a-f]+',args.accelerator or ''):raise ValueError('AMD requires an explicit supported native gfx architecture; no generic/portable target')
    if args.vendor!='amd' and args.accelerator:raise ValueError('--accelerator is only for the native AMD target')
    source=frozen(args.source_manifest);check_benchmark();plan=json.loads(args.plan.read_text());compiler=args.compiler.resolve()
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
    if apple:
        sdk=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-version'],text=True).strip()
        hardware={k:subprocess.check_output(['sysctl','-n',k],text=True).strip() for k in ['machdep.cpu.brand_string','hw.ncpu','hw.memsize']}
    else:
        hardware={'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0))}
        for name in ('cpu.max','memory.max','cpuset.cpus.effective'):
            path=Path('/sys/fs/cgroup')/name
            hardware[name]=path.read_text().strip() if path.exists() else 'unavailable'
        hardware['cpu_model']=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unavailable')
        if args.vendor=='nvidia':
            hardware['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,compute_cap','--format=csv,noheader'],text=True).strip()
            caps={line.rsplit(',',1)[1].strip().replace('.','') for line in hardware['gpu'].splitlines()}
            if len(caps)!=1:raise ValueError('Need one native NVIDIA architecture')
            accelerator='sm_'+caps.pop()
        if args.vendor=='amd':
            accelerator=args.accelerator
            hardware['requested_native_accelerator']=accelerator
    target_track=('nvidia-'+args.nvidia_target if args.vendor=='nvidia' else 'amd-'+args.accelerator if args.vendor=='amd' else args.vendor or 'apple-host')
    manifest=dict(target_track=target_track,schema='mojolearn.six-lane-build-campaign/1',source_sha=source,compiler=str(compiler),compiler_sha256=digest(compiler),compiler_version=version,hardware=hardware,build_plan_sha256=digest(args.plan),records=[],qualification='No runtime, identity, quality or performance execution')
    closures,_=source_graph()
    reusable=[]
    for root in args.reuse or []:
        for path in root.resolve().glob('**/receipt.json'):
            old=json.loads(path.read_text())
            if old.get('status')=='COMPILED':reusable.append((path,old))
    chosen=[j for j in plan['jobs'] if (j['vendor'] in (('apple','host') if apple else (args.vendor,))) and (not args.vendor or j['vendor']==args.vendor) and (not args.binding or j['binding'] in args.binding) and (not args.key or j['key'] in args.key)]
    if args.limit:chosen=chosen[:args.limit]
    for job in chosen:
        if frozen(args.source_manifest)!=source:raise ValueError('Source changed during freeze')
        directory=out/job['key'];directory.mkdir(exist_ok=True)
        receipt=directory/'receipt.json'
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old.get('source_sha')==source and old.get('target_track')==target_track and old.get('compiler_sha256')==manifest['compiler_sha256'] and old.get('status')=='COMPILED' and digest(old['artifact'])==old['artifact_sha256']:
                manifest['records'].append(old);continue
            raise ValueError('Evidence already exists; choose a new campaign directory for repairs')
        artifact=directory/(Path(job['binding']).stem+'.so');log=directory/'build.log'
        flags=['--target-cpu','apple-m1' if apple else 'x86-64-v3']
        family=Path(job['binding']).stem.removeprefix('_mojolearn').lstrip('_')
        wrapper=ROOT/'bindings'/('build_'+family+'.sh' if family else 'build.sh')
        # Match inspected shipped Darwin builders. Neural AOT builders deliberately
        # omit --target-accelerator; expansion bindings explicitly use metal:1.
        if apple and job['vendor']=='apple' and wrapper.exists() and '--target-accelerator metal:1' in wrapper.read_text():flags+=['--target-accelerator','metal:1']
        if linux and (job['vendor']=='amd' or (job['vendor']=='nvidia' and args.nvidia_target=='native')):flags+=['--target-accelerator',accelerator]
        flags+=['-D',{'host':'MOJOLEARN_COLUMN_CPU','apple':'MOJOLEARN_COLUMN_APPLE','nvidia':'MOJOLEARN_COLUMN_NVIDIA','amd':'MOJOLEARN_COLUMN_AMD'}[job['vendor']]]
        if apple:flags+=['-Xlinker','-platform_version','-Xlinker','macos','-Xlinker','11.0','-Xlinker',sdk]
        argv=[str(compiler),'build','-j',str(args.jobs),'--emit','shared-lib',*flags,'-I',str(ROOT),'-I',str(ROOT/'bindings')]
        for define in job['defines']:argv+=['-D',define]
        argv += [str(ROOT/job['binding']),'-o',str(artifact)]
        dep_hash={p:digest(ROOT/p) for p in sorted(closures.get(job['binding'],{job['binding']}))}
        record=dict(job,target_track=target_track,source_sha=source,compiler=version,compiler_sha256=manifest['compiler_sha256'],hardware=hardware,argv=argv,source_closure_sha256=sha_value(dep_hash),source_files=dep_hash,artifact=str(artifact),log=str(log),wrapper=str(wrapper),wrapper_sha256=digest(wrapper) if wrapper.exists() else None)
        write(directory/'input.json',record)
        match=next(((path,old) for path,old in reusable if old.get('key')==job['key']
            and old.get('source_closure_sha256')==record['source_closure_sha256']
            and old.get('compiler_sha256')==record['compiler_sha256']
            and old.get('compiler')==version
            and comparable_compile_argv(old.get('argv',[]),job['binding'])==comparable_compile_argv(argv,job['binding'])
            and Path(old.get('artifact','')).is_file()
            and digest(old['artifact'])==old.get('artifact_sha256')),None)
        if match:
            old_path,old=match
            reused=dict(old,qualification_source_sha=source,reused_receipt=str(old_path),reuse_basis='Identical conservative source closure, compiler, target flags, mode/defines and artifact hash; original source SHA retained')
            write(receipt,reused);manifest['records'].append(reused);write(out/'campaign.json',manifest)
            print(json.dumps(dict(key=job['key'],binding=job['binding'],status='REUSED_COMPILED',receipt=str(old_path))),flush=True)
            continue
        with log.open('x') as stream:proc=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if any(digest(ROOT/p)!=h for p,h in dep_hash.items()):raise ValueError('Numerical source changed during compilation; artifact not admitted')
        record.update(returncode=proc.returncode,status='COMPILED' if proc.returncode==0 and artifact.exists() else 'FAILED',artifact_sha256=digest(artifact) if artifact.exists() else None)
        write(receipt,record);manifest['records'].append(record);write(out/'campaign.json',manifest)
        print(json.dumps(dict(key=job['key'],binding=job['binding'],status=record['status'],log=str(log))),flush=True)
        if record['status']=='FAILED' and not args.keep_going:break
    manifest['selected_jobs']=len(chosen);manifest['completed_jobs']=len(manifest['records']);manifest['planned_jobs']=len(plan['jobs']);manifest['incomplete_jobs']=len(plan['jobs'])-len(manifest['records']);write(out/'campaign.json',manifest)
    return 1 if any(r['status']=='FAILED' for r in manifest['records']) else 0


def runtime_requirements(configuration, workload_id, configurations):
    """Only controls declared for this workload can require a runtime adapter.

    Combined candidates include controls for unrelated estimators. Keep unknown
    member mappings blocked, and never invent an operation to reach a control.
    """
    from six_lane_full_variants import original_id
    registered_workload_id = workload_id
    workload_id = original_id(workload_id)
    pending = {}
    targets = [w for value in configuration.get('workloads', []) for w in expand_workload(value) if work_id(w)==workload_id]
    # These four saved estimator recipes were absent from the original matrix.
    # Resolve only the reviewed exact registration; unknown workloads stay blocked.
    if not targets and registered_workload_id.endswith('@input=mlp-full-v1'):
        from six_lane_mlp_variants import contracts
        targets = [dict(harness=r['harness'], lane=r['lane']) for r in contracts()['rows']
                   if r['configuration'] == configuration['id']
                   and r['variant_workload_id'] == registered_workload_id]
    target_harnesses = {w['harness'] for w in targets if isinstance(w, dict) and w.get('harness')}
    if configuration['A']['runtime']:
        pending[configuration['id']] = configuration['A']['runtime']
    for member, controls in configuration.get('runtime_by_member', {}).items():
        selected = configurations.get(member)
        if selected is None or not selected.get('workloads'):
            pending[member] = controls
            continue
        affected = [w for value in selected['workloads'] for w in expand_workload(value)]
        # A declared neural-only caller cannot require a control in a classical
        # PCA race. Unmapped callers within the same harness remain ambiguous.
        declared_harnesses = set()
        all_scoped = True
        for value in selected['workloads']:
            if isinstance(value, dict) and value.get('harness'):
                declared_harnesses.add(value['harness'])
            elif isinstance(value, dict) and value.get('board_lanes') and len(set(value.get('recipe_paths', [])) & {'tools/bench_board_algos.py', 'tools/bench_board_neural.py'}) == 1:
                declared_harnesses.update(set(value['recipe_paths']) & {'tools/bench_board_algos.py', 'tools/bench_board_neural.py'})
            else:
                all_scoped = False
        if all_scoped and target_harnesses and declared_harnesses.isdisjoint(target_harnesses):
            continue
        # Some neural controls share bench_board_algos with linear regressors.
        # Explicit board_lanes narrow that shared harness; unknown or mixed
        # declarations remain blocked rather than manufacturing an adapter.
        lane_scopes = []
        for value in selected['workloads']:
            if not isinstance(value, dict) or not value.get('board_lanes'):
                break
            harnesses = set(value.get('recipe_paths', [])) & {'tools/bench_board_algos.py', 'tools/bench_board_neural.py'}
            if len(harnesses) != 1:
                break
            lane_scopes.extend((harness, lane) for harness in harnesses for lane in value['board_lanes'])
        else:
            target_scopes = {(w['harness'], str(w.get('source_workload', w.get('lane', ''))).split('@')[0].rsplit(':', 1)[-1].rsplit('/', 1)[-1]) for w in targets if isinstance(w, dict) and w.get('harness')}
            if lane_scopes and target_scopes and all(lane for _, lane in target_scopes) and target_scopes.isdisjoint(lane_scopes):
                continue
        ambiguous = any(isinstance(w, dict) and not (w.get('id') or w.get('key') or w.get('lane')) for w in affected)
        if ambiguous or workload_id in {work_id(w) for w in affected}:
            pending[member] = controls
    return pending


def queue(args):
    """Write the existing queue contract; never launch a worker or a build."""
    check_benchmark();mat=read_matrix(args.matrix)
    recipes=json.loads(args.recipes.read_text()) if args.recipes else {}
    configs={c['id']:c for c in mat['configurations']};jobs=[];source=git('rev-parse','HEAD')
    for cell in mat['cells']:
        if cell['vendor']!=args.vendor or (args.select and cell['configuration'] not in args.select):continue
        c=configs[cell['configuration']];recipe=recipes.get(cell['key'])
        job=dict(key=cell['key'],mode=cell['mode'],implementation_ids=cell['implementation_ids'],master_selection=c,workload_id=cell['workload_id'],matrix_status=cell['status'],source_coverage_pending=cell.get('source_coverage_pending',[]),blocked=list(cell['blockers']),arms={a:dict(configuration=c[a],argv=[],environment=c[a]['environment']) for a in ('A','B')})
        if cell['status']!='PENDING_COVERAGE':job['blocked'].append('Not an independent new executable candidate: '+cell['status'])
        if recipe:
            required=('dataset_sha256','dimensions','estimator_settings','timed_boundary','intrinsic_caps','full_dataset_coverage','artifact_provenance','benchmark_sha256','workload','packages','coverage_resolutions','resource_policy')
            absent=[k for k in required if k not in recipe]
            if absent:raise ValueError('Recipe missing '+','.join(absent))
            if recipe['benchmark_sha256']!=digest(STORE/'benchmark.json'):raise ValueError('Recipe benchmark specification drift')
            if recipe.get('source_sha')!=source:raise ValueError('Recipe names a different source freeze')
            from six_lane_full_variants import validate_variant
            validate_variant(recipe, cell)
            if recipe['full_dataset_coverage'] is not True:raise ValueError('Recipe lacks full coverage')
            if c['problems'] or cell['status']!='PENDING_COVERAGE':raise ValueError('Incompatible, historical, rejected or alias-only selection')
            if runtime_requirements(c,cell['workload_id'],configs):raise ValueError('This source API needs a saved matching race; the master never alters one')
            resolutions=recipe['coverage_resolutions']
            if set(resolutions)!=set(job['blocked']):raise ValueError('Resolve each recorded coverage gap explicitly; removing blockers is not evidence')
            for reason,evidence in resolutions.items():
                if not evidence.get('conclusion') or digest(evidence['path'])!=evidence['sha256']:raise ValueError('Missing/changed gap evidence: '+reason)
            work=recipe['workload']
            if work['actual_shapes']!=recipe['dimensions'] or work['estimator_settings_record']!=recipe['estimator_settings']:raise ValueError('Recipe dimensions/settings differ from worker admission facts')
            for field in ('dataset_sha256','dimensions','estimator_settings','timed_boundary','intrinsic_caps','full_dataset_coverage','artifact_provenance'):job[field]=recipe[field]
            if recipe.get('registered_input_variant'):
                job.update(registered_input_variant=recipe['registered_input_variant'], changes_frozen_race=True)
            job['blocked']=[]
            worker=dict(recipe,job=job,source_sha=source,vendor=args.vendor,execution_authorized=False)
            worker_path=args.output.resolve().parent/(args.output.stem+'-workers')/(cell['key']+'.json')
            for arm in ('A','B'):
                job['arms'][arm]['argv']=[sys.executable,str(ROOT/'tools/six_lane_ab_worker.py'),'--recipe',str(worker_path),'--arm',arm,'--phase','{phase}','--output','{output}']
            write(worker_path,worker)
        jobs.append(job)
    result=dict(schema='mojolearn.full-ab-queue/1',repo=str(ROOT),source_sha=source,vendor=args.vendor,jobs=jobs,environment={},execution_authorized=False,master_policy=read('experiments/six_lane_integration/evidence_policy.json'))
    write(args.output,result)
    print(json.dumps(dict(jobs=len(jobs),blocked=sum(bool(j.get('blocked')) for j in jobs),output=str(args.output),execution='NOT RUN; later authorization must be recorded in queue and workers')))


def board_plan(args):
    from six_lane_evidence import board_inputs
    inventory,index=board_inputs(catalog(),[])
    write(args.output/'inventory.json',inventory);write(args.output/'index.json',index)
    write(args.output/'future_command.json',dict(argv=[sys.executable,str(ROOT/'tools/performance_measurement_board.py'),'--inventory',str(args.output/'inventory.json'),'--index',str(args.output/'index.json'),'--out',str(args.output/'board')],execution='NOT RUN',note='Future own-only board inputs; no historical opponent rows, admitted measurements, or default changes.'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    s.add_parser('refresh',help='regenerate source-only catalog, matrix and build plan')
    ls=s.add_parser('list');ls.add_argument('--lane')
    sh=s.add_parser('show');sh.add_argument('id')
    b=s.add_parser('compile',help='compile shared libraries only, never import or execute')
    b.add_argument('--plan',type=Path,default=STORE/'build_plan.json');b.add_argument('--compiler',type=Path,required=True);b.add_argument('--output',type=Path,required=True);b.add_argument('--binding',action='append');b.add_argument('--key',action='append');b.add_argument('--limit',type=int);b.add_argument('--keep-going',action='store_true');b.add_argument('--reuse',type=Path,action='append',help='Prior compile evidence roots; exact source-closure/toolchain matches only')
    b.add_argument('--vendor',choices=('apple','host','nvidia','amd'));b.add_argument('--accelerator',help='AMD native gfx target, e.g. gfx942; portable/generic targets forbidden');b.add_argument('--nvidia-target',choices=('native','default'),default='native',help='native pins the observed sm target; default preserves shipped compiler target selection, recorded separately');b.add_argument('--jobs',type=int,default=2);b.add_argument('--source-manifest',type=Path,help='Exact committed-source manifest for a verified archive on the authorized NVIDIA worker')
    q=s.add_parser('queue',help='write future queue; incomplete cells stay blocked');q.add_argument('--vendor',choices=VENDORS,required=True);q.add_argument('--recipes',type=Path);q.add_argument('--matrix',type=Path,default=STORE/'matrix.json.gz');q.add_argument('--select',action='append');q.add_argument('--output',type=Path,required=True)
    bp=s.add_parser('board-plan',help='write inputs and command for the existing board tool, without invoking it');bp.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='refresh':
        doc=catalog_document();mat=matrix(doc);build=build_plan(doc,mat);bench=freeze_benchmarks()
        for name,value in [('catalog',doc),('build_plan',build),('benchmark',bench)]:write(STORE/(name+'.json'),value)
        storage=write_matrix(STORE/'matrix.json.gz',mat)
        write(STORE/'matrix-storage.json',dict(schema='mojolearn.matrix-storage/1',
              compressed_path='experiments/six_lane_integration/matrix.json.gz',
              generation='six_lane_ab refresh',generation_source_commit=git('rev-parse','HEAD'),
              logical_bytes=storage['logical_bytes'],logical_sha256=storage['logical_sha256'],
              compressed_bytes=storage['stored_bytes'],compressed_sha256=storage['stored_sha256'],
              encoding='gzip; UTF-8 JSON bytes; filename empty; mtime zero',
              digest_policy='Logical digests hash decompressed bytes; artifact digests hash physical bytes.'))
        print(json.dumps(dict(entries=len(doc['entries']),configurations=len(mat['configurations']),matrix_cells=len(mat['cells']),build_jobs=len(build['jobs']),blocked_builds=len(build['blocked']))))
    elif args.command=='list':
        for e in catalog()['entries']:
            if not args.lane or e['lane']==args.lane:print(e['id'],e['title'])
    elif args.command=='show':
        doc=catalog();rows=[e for e in doc['entries']+doc['interactions']+doc['aliases'] if e['id']==args.id]
        if not rows:raise ValueError('Unknown namespaced ID')
        print(json.dumps(rows,indent=2))
    elif args.command=='compile':return compile_jobs(args)
    elif args.command=='board-plan':board_plan(args)
    else:queue(args)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
