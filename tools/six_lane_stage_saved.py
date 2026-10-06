#!/usr/bin/env python3
"""Metadata-only deployment/materialization for explicitly selected saved full-workload cells.

Requires the retained workload-facts/audits/compile-receipt payload generated
from saved full regression inputs. This stages authorized queue documents only;
it never imports an estimator, runs a fit, compiles, or launches measurements.
A caller launches the emitted script after the device and data-transfer leases
are free. The complete source and package freeze is mandatory.

The saved GMM recipe is full fit plus fitted-parameter consumption. Its original
held-out likelihood/BIC calculation remains untimed quality, not a claim of
measured native inference. Other selected regressors consume every prediction."""
import argparse,copy,json,os,pathlib,subprocess,sys,time,shlex
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('/root/six-lane-full-ab-20261006'));p.add_argument('--source-sha',required=True);p.add_argument('--configuration',default='I.X.complete-proposed');p.add_argument('--inputs',type=pathlib.Path,required=True);p.add_argument('--state',type=pathlib.Path,required=True);p.add_argument('--results',type=pathlib.Path,required=True);a=p.parse_args()
repo=a.root/'source';state=a.state
if (state/'queue.json').exists() or (a.results.exists() and any(a.results.iterdir())):raise SystemExit('Use fresh state/results directories; preserve prior attempts')
state.mkdir(parents=True,exist_ok=True);a.results.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(repo/'tools'))
from six_lane_ab import matrix,catalog,digest,write,runtime_requirements
from six_lane_materialize import materialize
source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if source!=a.source_sha:raise SystemExit('Wrong source freeze: '+source)
if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all'],text=True).strip():raise SystemExit('Source is dirty')
raw=json.loads(a.inputs.read_text());mat=matrix(catalog());cfg=a.configuration;vendor='nvidia'
# This queue intentionally selects the explicitly supplied existing saved full races. The other
# campaign cells retain their original unresolved coverage, separately below.
selected_ids={wid.replace('classical/','classical:') for wid in raw['facts']}
selected=[c for c in mat['cells'] if c['configuration']==cfg and c['vendor']==vendor and c['workload_id'] in selected_ids]
if not selected_ids or len(selected)!=len(selected_ids):raise SystemExit('Every supplied full workload must map to exactly one selected cell')
order={key:index for index,key in enumerate(wid.replace('classical/','classical:') for wid in raw['facts'])}
selected.sort(key=lambda c:order[c['workload_id']])
full_gap='Full dataset/version/hash, dimensions, settings, cap audit and accepted artifacts must be supplied from the frozen saved recipe.'
if any(any(reason!=full_gap for reason in cell['blockers']) for cell in selected):raise SystemExit('Additional coverage barriers require explicit resolution; this stager only supplies full-recipe facts')
slim=dict(mat,cells=selected);write(state/'matrix.json',slim)
write(state/'remaining-matrix-coverage.json',dict(source_sha=source,planned=len(mat['cells']),selected_keys=[c['key'] for c in selected],remaining=[dict(key=c['key'],configuration=c['configuration'],vendor=c['vendor'],workload_id=c['workload_id'],status=c['status'],blockers=c['blockers'],source_coverage_pending=c.get('source_coverage_pending',[])) for c in mat['cells'] if c['key'] not in {r['key'] for r in selected}]))
facts={}
for cell in selected:
 wid=cell['workload_id'];old=wid.replace('classical:','classical/');fact=copy.deepcopy(raw['facts'][old]);work=fact['workload'];ds=work['dataset'];lane=work['lane'];work['data_directory']=str(a.root/'data');work['runtime_vendor']='cuda'
 for item in work['input_files']:item['path']=str(a.root/'data'/pathlib.Path(item['path']).name)
 audit_name=pathlib.Path(work['intrinsic_cap_audit']['evidence']).name
 audit=dict(raw['audits'][audit_name],native_relocation=dict(source_sha=source,source_facts_sha256=digest(a.inputs),data_files=work['input_files'],target='NVIDIA H100 sm_90',runtime_scope='Saved runtime-operation applicability is independently enforced by the materializer; no operation or setting is altered here.',source_coverage_pending=cell.get('source_coverage_pending',[]),scope='This combined configuration measurement does not complete other caller coverage or independently establish constituent switch reach.'))
 audit_path=state/audit_name;write(audit_path,audit);work['intrinsic_cap_audit']['evidence']=str(audit_path)
 fact['coverage_resolutions']={reason:dict(path=str(audit_path),sha256=digest(audit_path),conclusion='Retained full train and query split, identical saved constructor/settings and declared whole-operation boundary; exact runtime arrays/settings and binding hashes checked by the scored worker. Source coverage limits remain separate and unqualified.') for reason in cell['blockers']}
 fact['resource_policy']=dict(policy='full allocated NVIDIA worker; actual affinity/cgroup and effective pools captured in each worker; arms serial; no inherited diagnostic caps')
 facts[wid]=fact
write(state/'workload-facts.json',facts)
deployment=dict(configuration=cfg,vendor=vendor,target_track='nvidia-native',packages={arm:str(a.root/'packages'/arm) for arm in ('A','B')},artifacts={})
for arm in ('A','B'):
 deployment['artifacts'][arm]=[]
 for item in raw['receipts'][arm]:
  receipt=item['receipt'];rpath=state/'receipts'/arm/(receipt['key']+'.json');write(rpath,receipt)
  deployed=a.root/'packages'/arm/'mojolearn'/'identical'/(pathlib.Path(item['binding']).stem+'.so')
  deployment['artifacts'][arm].append(dict(path=str(deployed),receipt=str(rpath)))
deployments=[]
for wid in facts:
 if wid not in raw['required_bindings']:raise SystemExit('Explicit loaded binding dependencies missing for '+wid)
 entry=dict(deployment,workload_id=wid,artifacts={})
 for arm in ('A','B'):
  entry['artifacts'][arm]=[item for item in deployment['artifacts'][arm] if 'bindings/'+pathlib.Path(item['path']).stem+'.mojo' in raw['required_bindings'][wid]]
 deployments.append(entry)
write(state/'deployments.json',deployments)
args=argparse.Namespace(matrix=state/'matrix.json',workloads=state/'workload-facts.json',deployments=state/'deployments.json',vendor=vendor,target_track='nvidia-native',select=[cfg],output=state/'materialized')
materialize(args)
coverage=json.loads((state/'materialized/coverage.json').read_text())
if coverage['materialized']!=len(selected) or coverage['blocked']:raise SystemExit('Incomplete concrete admission; see coverage.json')
python=repo/'.pixi/envs/bench/bin/python'
subprocess.run([str(python),str(repo/'tools/six_lane_ab.py'),'queue','--matrix',str(state/'matrix.json'),'--vendor',vendor,'--select',cfg,'--recipes',str(state/'materialized/recipes.json'),'--output',str(state/'queue.json')],check=True)
queue=json.loads((state/'queue.json').read_text());queue.update(execution_authorized=True,authorization='Owner explicitly requested all full A/B runs using retained compatible artifacts; no compilation; A=candidate B=incumbent',environment=dict(PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(repo/'.pixi/envs/bench/lib')+':'+str(repo/'.pixi/envs/default/lib')))
for job in queue['jobs']:
 job['timeout_seconds']=86400
 for arm in ('A','B'):
  argv=job['arms'][arm]['argv'];argv[0]=str(python)
  worker_path=pathlib.Path(argv[argv.index('--recipe')+1]);worker=json.loads(worker_path.read_text());worker.update(execution_authorized=True,authorization=queue['authorization']);write(worker_path,worker)
write(state/'queue.json',queue)
launch=state/'launch.sh'
command=shlex.join([str(python),str(repo/'tools/performance_full_ab_queue.py'),'--config',str(state/'queue.json'),'--output',str(a.results)])
launch.write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexport LD_LIBRARY_PATH='+shlex.quote(queue['environment']['LD_LIBRARY_PATH'])+'\nexec 9>'+shlex.quote(str(a.root/'device-measurement.lock'))+'\nflock -n 9\nexec '+command+' "$@"\n');launch.chmod(0o755)
write(a.results/'status.json',dict(phase='READY_NOT_STARTED',source_sha=source,total=len(selected),completed=0,updated=time.time(),queue=str(state/'queue.json'),launch=str(launch),compilation='not run; retained native H100 artifacts'))
print(json.dumps(dict(status='READY_NOT_STARTED',cells=len(selected),source_sha=source,launch=str(launch),status_path=str(a.results/'status.json'))))
