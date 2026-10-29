import argparse, collections, datetime, hashlib, json, os, pathlib, subprocess, sys, tarfile, time, shutil
from controller_process import run_process
from controller_lease_budget import lease_budget
ap=argparse.ArgumentParser();ap.add_argument('--configuration',type=pathlib.Path,required=True);ap.add_argument('--vendor',required=True,choices=['apple','cpu','nvidia','amd']);ap.add_argument('--phase',required=True);ap.add_argument('--exclusions',type=pathlib.Path);ap.add_argument('--execute',action='store_true');ap.add_argument('--lease-deadline-epoch',type=float);ap.add_argument('--unit-budget-seconds',type=float);ap.add_argument('--retention-reserve-seconds',type=float);ap.add_argument('--max-units',type=int);a=ap.parse_args()
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
  target=B/'archives'/c['vendor']/c['phase']/f"{c['lane']}.tar.gz";target.parent.mkdir(parents=True,exist_ok=True)
  with tarfile.open(target,'w:gz') as t:t.add(out,arcname=c['lane'])
  raw=target.read_bytes();sha=hashlib.sha256(raw).hexdigest();key=f"verification/0.8.63/{REV[:10]}/{c['vendor']}/{c['phase']}/{c['lane']}/{sha}.tar.gz"
  creds=evidence_r2._creds();status,_=bincache.r2_request(creds,'PUT',key,data=raw)
  if status not in (200,201):raise RuntimeError(f'R2 PUT {status}')
  status,got=bincache.r2_request(creds,'GET',key)
  if status!=200 or hashlib.sha256(got).hexdigest()!=sha:raise RuntimeError('R2 readback mismatch')
  save(out/'r2-retention.json',dict(key=key,sha256=sha,bytes=len(raw),readback_verified=True,archive=str(target)))

def retain_partial(c,out,producer_log):
 out.mkdir(parents=True,exist_ok=True)
 controls=out/'producer-control';controls.mkdir(exist_ok=True)
 for source in (producer_log,producer_log.with_suffix('.exit'),producer_log.with_suffix('.execution.json')):
  if source.exists():shutil.copy2(source,controls/source.name)
 shutil.copy2(pathlib.Path(CFG['launches'][c['vendor']]),controls/'original-launch.json')
 save(out/'controller-summary.json',dict(chunk=c,local_complete=False,exit_code=124,classification='producer-timeout',final_report_fabricated=False,identity_admission='not-admitted'))
 target=B/'archives'/c['vendor']/c['phase']/f"{c['lane']}.timeout.tar.gz";target.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(target,'w:gz') as t:t.add(out,arcname=c['lane'])
 raw=target.read_bytes();sha=hashlib.sha256(raw).hexdigest();key=f"verification/0.8.63/{REV[:10]}/{c['vendor']}/{c['phase']}/{c['lane']}/{sha}.tar.gz"
 creds=evidence_r2._creds();status,_=bincache.r2_request(creds,'PUT',key,data=raw)
 if status not in (200,201):raise RuntimeError(f'partial R2 PUT {status}; raw files preserved locally')
 status,got=bincache.r2_request(creds,'GET',key)
 if status!=200 or hashlib.sha256(got).hexdigest()!=sha:raise RuntimeError('partial R2 readback mismatch; raw files preserved locally')
 save(out/'r2-retention.json',dict(key=key,sha256=sha,bytes=len(raw),readback_verified=True,archive=str(target),partial=True))

def run(c):
 out=B/'runs'/c['vendor']/c['phase']/c['lane'];state=B/'controller-state.json'
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
  call([PY,TOOLS/'tools/identity_verify_launch.py','--root',ROOT,'--python',INST,'--out',out,'--lanes-file',lanes,'--profile',c['profile'],'--cpu-threads','10'],producer_log,env,timeout)
  execution=json.loads(producer_log.with_suffix('.execution.json').read_text())
  if execution['timed_out']:
   retain_partial(c,out,producer_log)
   raise RuntimeError(f'producer timeout; partial evidence retained without fabricated report: {out}')
 if (out/'controller-summary.json').exists():
  prior=json.loads((out/'controller-summary.json').read_text())
  if prior['local_complete']:
   retain(c,out)
   return
  raise RuntimeError(f'prior incomplete chunk preserved, no automatic rerun: {out}')
 if not (out/'verify.json').exists():raise RuntimeError(f'producer has no final report: {out}')
 d=json.loads((out/'verify.json').read_text());r=json.loads((out/'verifier-run.json').read_text());e=d.get('execution',{});ls=d.get('lanes_summary',{})
 if d.get('device',{}).get('device_class')!=c['vendor']:raise RuntimeError(f'wrong actual vendor retained in raw report: {out}')
 decision=classify(d,r);failures=decision['local_property_failures'];complete=decision['observation_complete']
 summary=dict(chunk=c,local_complete=complete,exit_code=r.get('exit_code'),counts=d.get('counts'),execution=e,lanes_summary=ls,native_provenance_complete=d.get('native_provenance_complete'),cross_vendor_admission='pending', historical_reference_differences=d.get('counts',{}).get('DIVERGENT',0), local_property_failures=failures, observation_classification=decision)
 save(out/'controller-summary.json',summary)
 retain(c,out)
 print(json.dumps(dict(vendor=c['vendor'],lane=c['lane'],local_complete=complete,counts=d.get('counts'))),flush=True)
 if not complete:raise RuntimeError(f'local incomplete/failure retained: {out}')
 save(state,dict(status='chunk-complete',chunk=c))

chunks=[c for c in json.loads(CHUNKS.read_text())['chunks'] if c['vendor']==a.vendor and c['phase']==a.phase]
excluded=json.loads(a.exclusions.read_text()) if a.exclusions else {}
selected=[c for c in chunks if c['lane'] not in excluded]
frozen_inputs(ROOT,PKG,LEDGER,REV)
if not a.execute:
 print(json.dumps(dict(preview=True,source_revision=REV,chunks=len(selected),excluded=excluded,output_root=str(B))))
 raise SystemExit(0)
if not CFG.get('execution_authorized'):
 raise RuntimeError('runtime hold remains in configuration')
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
  run(c)
 save(B/f'{a.vendor}-{a.phase}-controller.json',dict(status='partial-with-explicit-blockers' if excluded else 'complete',chunks=len(selected),excluded=excluded))
except Exception as ex:
 save(B/'controller-state.json',dict(status='stopped',error=str(ex)))
 raise
