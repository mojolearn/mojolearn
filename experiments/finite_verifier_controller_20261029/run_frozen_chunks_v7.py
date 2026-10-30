import argparse, collections, datetime, hashlib, json, os, pathlib, subprocess, sys, tarfile, time, shutil
from controller_process import run_process
from controller_lease_budget import lease_budget
ap=argparse.ArgumentParser();ap.add_argument('--configuration',type=pathlib.Path,required=True);ap.add_argument('--vendor',required=True,choices=['apple','cpu','nvidia','amd']);ap.add_argument('--phase',required=True,choices=['routine-all','training-all']);ap.add_argument('--exclusions',type=pathlib.Path);ap.add_argument('--execute',action='store_true');ap.add_argument('--lease-deadline-epoch',type=float);ap.add_argument('--unit-budget-seconds',type=float);ap.add_argument('--retention-reserve-seconds',type=float);ap.add_argument('--max-units',type=int);ap.add_argument('--keep-going-local-failures',action='store_true');a=ap.parse_args()
lease_budget(a.lease_deadline_epoch,a.unit_budget_seconds,a.retention_reserve_seconds,time.time())
if a.max_units is not None and a.max_units <= 0:raise ValueError('max-units must be positive')
CFG=json.loads(a.configuration.read_text())
B=pathlib.Path(CFG['output_root']);ROOT=pathlib.Path(CFG['source_root']);REV=CFG['source_revision']
TOOLS=pathlib.Path(CFG['tooling_root']);TOOLREV=CFG['tooling_revision']
LEDGER=pathlib.Path(CFG['ledger']);CHUNKS=pathlib.Path(CFG['chunks'])
RECORDS=pathlib.Path(CFG['records_root']);PY=pathlib.Path(CFG['tools_python']);INST=pathlib.Path(CFG['installed_python']);PKG=pathlib.Path(CFG['installed_package'])
from controller_preflight_v2 import frozen_inputs,exact_column_begin
actual_tools=subprocess.check_output(['git','-C',str(TOOLS),'rev-parse','HEAD'],text=True).strip()
if actual_tools!=TOOLREV or subprocess.check_output(['git','-C',str(TOOLS),'status','--porcelain','--untracked-files=no'],text=True).strip():raise RuntimeError('tooling checkout differs or has tracked changes')
sys.path.insert(0,str(TOOLS/'tools'))
import bincache, evidence_r2
from observation_classification import classify
from controller_offsite import defer_archive, checked_archive, pending_count
from controller_local_failures import LocalObservationFailure, validated_failure

def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');t.replace(p)
def call(args,log,env=None,timeout=None):
 log.parent.mkdir(parents=True,exist_ok=True)
 result=run_process(args,cwd=ROOT,log=log,env=env,timeout=timeout)
 save(log.with_suffix('.execution.json'),result)
 log.with_suffix('.exit').write_text(str(result['exit_code'])+'\n')
 return result['exit_code']

def retain(c,out):
 col=out/f"{c['vendor']}-{c['lane']}.identical.json"
 if not col.exists():
  rc=call([PY,TOOLS/'tools/identity_columns.py','from-verify',out/'verify.json','--harness',PKG/'_identity_break.py','--out',col,'--launch',out/'verifier-run.json'],out/'conversion.log')
  if rc:raise RuntimeError(f'conversion failed {rc}: {out}')
 rc=call([PY,TOOLS/'tools/retain_run_summary.py','--repo',RECORDS,'identity',col],out/'retention.log')
 if rc:raise RuntimeError(f'repository retention failed {rc}: {out}')
 if not (out/'r2-retention.json').exists():
  if CFG.get('offsite_policy') == 'deferred-local' and (out/'r2-pending.json').exists():
   checked_archive(json.loads((out/'r2-pending.json').read_text()))
   return
  target=B/'archives'/c['vendor']/c['phase']/f"{c['lane']}.tar.gz";target.parent.mkdir(parents=True,exist_ok=True)
  with tarfile.open(target,'w:gz') as t:t.add(out,arcname=c['lane'])
  raw=target.read_bytes();sha=hashlib.sha256(raw).hexdigest();key=f"verification/0.8.63/{REV[:10]}/{c['vendor']}/{c['phase']}/{c['lane']}/{sha}.tar.gz"
  if CFG.get('offsite_policy') == 'deferred-local':
   if sys.platform != 'darwin' or c['vendor'] not in ('apple','cpu'):raise RuntimeError('deferred retention authorized only for local Mac reference collection')
   defer_archive(B,out,target,raw,key,REV,c)
   return
  creds=evidence_r2._creds();status,_=bincache.r2_request(creds,'PUT',key,data=raw,timeout=1200)
  if status not in (200,201):raise RuntimeError(f'R2 PUT {status}')
  status,got=bincache.r2_request(creds,'GET',key,timeout=1200)
  if status!=200 or hashlib.sha256(got).hexdigest()!=sha:raise RuntimeError('R2 readback mismatch')
  save(out/'r2-retention.json',dict(key=key,sha256=sha,bytes=len(raw),readback_verified=True,archive=str(target)))

def retain_partial(c,out,producer_log,classification,exit_code):
 out.mkdir(parents=True,exist_ok=True)
 controls=out/'producer-control';controls.mkdir(exist_ok=True)
 for source in (producer_log,producer_log.with_suffix('.exit'),producer_log.with_suffix('.execution.json')):
  if source.exists():shutil.copy2(source,controls/source.name)
 shutil.copy2(pathlib.Path(CFG['launches'][c['vendor']]),controls/'original-launch.json')
 if (out/'controller-summary.json').exists() and not (controls/'original-controller-summary.json').exists():shutil.copy2(out/'controller-summary.json',controls/'original-controller-summary.json')
 save(out/'controller-summary.json',dict(chunk=c,local_complete=False,exit_code=exit_code,classification=classification,final_report_fabricated=False,identity_admission='not-admitted'))
 target=B/'archives'/c['vendor']/c['phase']/f"{c['lane']}.partial.tar.gz";target.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(target,'w:gz') as t:t.add(out,arcname=c['lane'])
 raw=target.read_bytes();sha=hashlib.sha256(raw).hexdigest();key=f"verification/0.8.63/{REV[:10]}/{c['vendor']}/{c['phase']}/{c['lane']}/{sha}.tar.gz"
 if CFG.get('offsite_policy') == 'deferred-local':
  if sys.platform != 'darwin' or c['vendor'] not in ('apple','cpu'):raise RuntimeError('deferred retention authorized only for local Mac reference collection')
  defer_archive(B,out,target,raw,key,REV,c)
  return
 creds=evidence_r2._creds();status,_=bincache.r2_request(creds,'PUT',key,data=raw,timeout=1200)
 if status not in (200,201):raise RuntimeError(f'partial R2 PUT {status}; raw files preserved locally')
 status,got=bincache.r2_request(creds,'GET',key,timeout=1200)
 if status!=200 or hashlib.sha256(got).hexdigest()!=sha:raise RuntimeError('partial R2 readback mismatch; raw files preserved locally')
 save(out/'r2-retention.json',dict(key=key,sha256=sha,bytes=len(raw),readback_verified=True,archive=str(target),partial=True))

def preserve_local_failure(c,out,producer_log,report,receipt):
 detail=validated_failure(report,receipt,hashlib.sha256((out/'verify.json').read_bytes()).hexdigest(),c['vendor'],c['lane'])
 if (out/'r2-pending.json').exists():
  checked_archive(json.loads((out/'r2-pending.json').read_text()))
 elif not (out/'r2-retention.json').exists():
  retain_partial(c,out,producer_log,'validated-local-algorithm-failure',receipt['exit_code'])
 save(out/'continued-local-failure.json',detail)
 raise LocalObservationFailure(detail)

def run(c):
 out=B/'runs'/c['vendor']/c['phase']/c['lane'];state=B/'controller-state.json'
 producer_log=B/'controller-logs'/c['vendor']/c['phase']/f"{c['lane']}.producer.log"
 save(state,dict(status='running',chunk=c,at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 if not out.exists():
  launch=pathlib.Path(CFG['launches'][c['vendor']])
  # The sole ledger owner begins units before delegating execution.
  # This executor never mutates the central ledger or creates missing launches.
  exact_column_begin(launch,c['vendor'],c['lane'],REV,frozen_inputs(ROOT,PKG,LEDGER,REV))
  lanes=B/'controller-inputs'/c['vendor']/c['phase']/f"{c['lane']}.lanes";lanes.parent.mkdir(parents=True,exist_ok=True);lanes.write_text(c['lane']+'\n')
  env=dict(os.environ)
  for k in ('MOJOLEARN_VENDOR','MOJOLEARN_VENDOR_FORCE','MOJOLEARN_GPU_ARCH'):env.pop(k,None)
  env.update(PYTHONUNBUFFERED='1',MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32='0',MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE='0')
  if c['vendor']=='apple':env['MOJOLEARN_APPLE_FULL_DIAGNOSTIC']='1'
  env['MOJOLEARN_VENDOR']={'apple':'metal','cpu':'cpu','nvidia':'cuda','amd':'hip'}[c['vendor']]
  producer_log=B/'controller-logs'/c['vendor']/c['phase']/f"{c['lane']}.producer.log"
  timeout=a.unit_budget_seconds
  if a.lease_deadline_epoch is not None:
   timeout=min(timeout,a.lease_deadline_epoch-time.time()-a.retention_reserve_seconds)
   if timeout<=0:
    save(B/'controller-state.json',dict(status='paused-lease-budget-before-producer',next_chunk=c))
    raise SystemExit(75)
  call([PY,TOOLS/'tools/identity_verify_launch.py','--root',ROOT,'--python',INST,'--out',out,'--lanes-file',lanes,'--profile',c['profile'],'--cpu-threads',str(CPU_THREADS)],producer_log,env,timeout)
  execution=json.loads(producer_log.with_suffix('.execution.json').read_text())
  if execution['timed_out']:
   retain_partial(c,out,producer_log,'producer-timeout',execution['exit_code'])
   raise RuntimeError(f'producer timeout; partial evidence retained without fabricated report: {out}')
 if (out/'controller-summary.json').exists():
  try:prior=json.loads((out/'controller-summary.json').read_text())
  except (ValueError,OSError):
   retain_partial(c,out,producer_log,'malformed-controller-summary',None)
   raise RuntimeError(f'malformed prior summary retained: {out}')
  if prior['local_complete']:
   retain(c,out)
   return
  if a.keep_going_local_failures:
   prior_report=json.loads((out/'verify.json').read_text());prior_receipt=json.loads((out/'verifier-run.json').read_text())
   preserve_local_failure(c,out,producer_log,prior_report,prior_receipt)
  raise RuntimeError(f'prior incomplete chunk preserved, no automatic rerun: {out}')
 execution_path=producer_log.with_suffix('.execution.json')
 try:execution=json.loads(execution_path.read_text()) if execution_path.exists() else {}
 except (ValueError,OSError):execution={}
 if not (out/'verify.json').exists():
  retain_partial(c,out,producer_log,'producer-missing-final-report',execution.get('exit_code'))
  raise RuntimeError(f'producer has no final report; raw evidence retained: {out}')
 try:
  d=json.loads((out/'verify.json').read_text());r=json.loads((out/'verifier-run.json').read_text())
  if not isinstance(d,dict) or not isinstance(r,dict):raise ValueError('report and receipt must be objects')
 except (ValueError,OSError):
  retain_partial(c,out,producer_log,'malformed-or-missing-report-receipt',execution.get('exit_code'))
  raise RuntimeError(f'malformed report/receipt; raw evidence retained: {out}')
 e=d.get('execution',{});ls=d.get('lanes_summary',{})
 if d.get('device',{}).get('device_class')!=c['vendor']:raise RuntimeError(f'wrong actual vendor retained in raw report: {out}')
 decision=classify(d,r);failures=decision['local_property_failures'];complete=decision['observation_complete']
 summary=dict(chunk=c,local_complete=complete,exit_code=r.get('exit_code'),counts=d.get('counts'),execution=e,lanes_summary=ls,native_provenance_complete=d.get('native_provenance_complete'),cross_vendor_admission='pending', historical_reference_differences=d.get('counts',{}).get('DIVERGENT',0), local_property_failures=failures, observation_classification=decision)
 save(out/'controller-summary.json',summary)
 if not complete:
  preserve_local_failure(c,out,producer_log,d,r)
 retain(c,out)
 print(json.dumps(dict(vendor=c['vendor'],lane=c['lane'],local_complete=complete,counts=d.get('counts'))),flush=True)
 if not complete:raise RuntimeError(f'local incomplete/failure retained: {out}')
 save(state,dict(status='chunk-complete',chunk=c))

chunks=[c for c in json.loads(CHUNKS.read_text())['chunks'] if c['vendor']==a.vendor and c['phase']==a.phase]
if not chunks:raise ValueError('configured phase has no registered units; no completion claim')
excluded=json.loads(a.exclusions.read_text()) if a.exclusions else {}
selected=[c for c in chunks if c['lane'] not in excluded]
frozen_inputs(ROOT,PKG,LEDGER,REV)
if not a.execute:
 print(json.dumps(dict(preview=True,source_revision=REV,chunks=len(selected),excluded=excluded,output_root=str(B))))
 raise SystemExit(0)
if not CFG.get('execution_authorized'):
 raise RuntimeError('runtime hold remains in configuration')
thread_policy=CFG.get('cpu_threads')
if thread_policy == 'allocation':
 from build_sizing import box_resources
 CPU_THREADS, available_memory = box_resources()
 resource_record=dict(policy='affinity-and-cgroup-allocation',cpu_threads=CPU_THREADS,affinity=sorted(os.sched_getaffinity(0)),available_memory_bytes=available_memory)
elif type(thread_policy) is int and thread_policy > 0:
 CPU_THREADS=thread_policy
 resource_record=dict(policy='explicit-bounded-verification',cpu_threads=CPU_THREADS)
else:
 raise RuntimeError('configuration requires positive cpu_threads or allocation policy')
save(B/'worker-resources.json',resource_record)
failed_local=[]
try:
 for index,c in enumerate(selected):
  if a.max_units is not None and index >= a.max_units:
   save(B/'controller-state.json',dict(status='paused-finite-batch',completed_units=index,next_chunk=c))
   raise SystemExit(75)
  budget=lease_budget(a.lease_deadline_epoch,a.unit_budget_seconds,a.retention_reserve_seconds,time.time())
  if budget is not None:
   B.mkdir(parents=True,exist_ok=True)
   with (B/'lease-admission.jsonl').open('a') as f:f.write(json.dumps(dict(chunk=c,**budget),sort_keys=True)+'\n')
   if not budget['allowed']:
    save(B/'controller-state.json',dict(status='paused-lease-budget',next_chunk=c,budget=budget))
    raise SystemExit(75)
  if (B/'pause-after-current-chunk').exists():
   save(B/'controller-state.json',dict(status='paused-between-chunks',next_chunk=c))
   raise SystemExit(75)
  try:
   run(c)
  except LocalObservationFailure as failure:
   if not a.keep_going_local_failures:raise
   failed_local.append(failure.detail)
   save(B/f'{a.vendor}-{a.phase}-local-failures.json',dict(failed=failed_local,qualification_claim=False))
   print(json.dumps(dict(lane=c['lane'],vendor=c['vendor'],status='failed-retained-continuing',detail=failure.detail)),flush=True)
 save(B/f'{a.vendor}-{a.phase}-controller.json',dict(status='partial-with-local-failures' if failed_local else ('partial-with-explicit-blockers' if excluded else ('local-reference-collection-complete-offsite-pending' if pending_count(B) else 'complete')),chunks=len(selected),excluded=excluded,failed_local=failed_local,offsite_pending=pending_count(B),release_qualification='pending-cross-vendor-admission'))
 if failed_local or excluded:raise SystemExit(1)
except Exception as ex:
 save(B/'controller-state.json',dict(status='stopped',error=str(ex)))
 raise
