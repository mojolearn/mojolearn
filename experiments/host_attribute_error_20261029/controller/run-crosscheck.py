from pathlib import Path
import datetime,hashlib,json,os,subprocess
r=Path(__file__).parent
python=r/'venv-repaired/bin/python';package=r/'venv-repaired/lib/python3.13/site-packages/mojolearn'
original=Path.home()/'mojolearn-evidence/current-main-verification-20261029/full-bc11389/runs/apple/routine-all/agglomerative/verify.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((r/'focused-summary.json').read_text())['qualified']
env=dict(os.environ);env.pop('PYTHONPATH',None)
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32='0',MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE='0')
command=[str(python),'-m','mojolearn','verify','--cross-check','all','--lanes','agglomerative','--cpu-threads','10','--json']
launch={'command':command,'cwd':str(r),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage_receipt_sha256':sha(r/'stage-receipt.json'),'expected_fixtures':9,'expected_parts':18,'original_primary_report':str(original),'original_primary_report_sha256':sha(original),'source_files':{str(p):sha(p) for p in [package/'_verify_all.py',package/'_identity_break.py']},'environment':{k:v for k,v in env.items() if k.startswith('MOJOLEARN_') or k in ('PYTHONDONTWRITEBYTECODE','PYTHONFAULTHANDLER')},'scope':'Independent actual CPU cross-check component; does not synthesize or replace original failed whole-lane report'}
(r/'crosscheck-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
with (r/'crosscheck.json').open('w') as output,(r/'crosscheck.log').open('w') as errors:
 result=subprocess.run(command,cwd=r,env=env,stdout=output,stderr=errors)
(r/'crosscheck.exit').write_text(str(result.returncode)+'\n')
summary={'exit_code':result.returncode,'expected_fixtures':9,'expected_parts':18}
try:
 doc=json.loads((r/'crosscheck.json').read_text())
 summary.update({k:doc.get(k) for k in ('complete','passed','compared','agree','differ','vendor')})
 summary['fixtures']=len(doc.get('requested_fixtures',[]));summary['missing']=len(doc.get('missing',[]));summary['skipped']=len(doc.get('skipped',{}));summary['failed_workers']=[x for x in doc.get('jobs',[]) if x.get('exit')!=0]
except (ValueError,TypeError) as error:summary['parse_error']=str(error)
summary['qualified']=result.returncode==0 and summary.get('passed') is True and summary.get('complete') is True and summary.get('fixtures')==9 and summary.get('compared')==18 and summary.get('agree')==18 and summary.get('differ')==0 and summary.get('missing')==0 and summary.get('skipped')==0
(r/'crosscheck-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));raise SystemExit(0 if summary['qualified'] else 1)
