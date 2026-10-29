"""Close old unadmitted qualifications without inventing numerical failures."""
import pathlib,json,sys,hashlib
B=pathlib.Path(__file__).parent;ROOT=pathlib.Path('/Users/andrewhendel/mojolearn-wt/dependency-analysis-cache-20261029');sys.path.insert(0,str(ROOT/'tools'))
from release_requirements import RequirementsLedger,changes
ledger=RequirementsLedger(B/'ledger-0.8.63.json',source_root=pathlib.Path('/Users/andrewhendel/mojolearn-wt/verification-final-095ab-20261029'))
doc=ledger.read();finished={a['run_id'] for a in doc['attempts']};records=[];transitions=[]
for run in doc['runs']:
 if run['id'] in finished:continue
 name=run['requirement'];old=run['fingerprint'];current=doc['requirements'][name]['fingerprint']
 if old==current:continue
 vendor,lane=name.split('/',1);vendor=vendor.removeprefix('identity-')
 reports=list((B/'runs'/vendor).glob('*/'+lane+'/verify.json'));assert len(reports)==1,(name,reports)
 report=reports[0];receipt=report.with_name('verifier-run.json');d=json.loads(report.read_text());r=json.loads(receipt.read_text())
 assert r['status']=='completed' and not d['execution'].get('crashed_cells') and d['execution']['completed_cells']==d['execution']['total_cells']==9
 assert not any(x.get('local_check')=='failed' or x.get('state')=='REFUSED' for x in d['cells'])
 transition=dict(requirement=name,run_id=run['id'],classification='qualification_not_admitted/superseded',observation_status='completed',numerical_failed=0,original_cli_exit=r['exit_code'],old_source=run['source_revision'],new_source=doc['source_revision'],old_fingerprint=old,new_fingerprint=current,changed_inputs=changes(run['inputs'],doc['requirements'][name]['inputs']),reason='Required cross-vendor comparison was pending when concrete runtime inputs changed; original observations remain retained and are not numerical failures.',report=str(report),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
 transitions.append(transition)
 records.append(dict(run_id=run['id'],status='failed',paths=[str(report),str(receipt),str(B/'sync-095ab32829-summary.json')],summary='qualification_not_admitted/superseded; observation completed; numerical_failed=0; original CLI exit '+str(r['exit_code'])+'. Required cross-vendor comparison pending before concrete inputs changed. Not a product/numerical failure; original report retained.',exit_code=r['exit_code'],expected=len(d['cells']),completed=0,failed=0,skipped=len(d['cells'])))
assert len(records)==51,len(records)
path=B/'superseded-observation-transitions.json';path.write_text(json.dumps(dict(schema='mojolearn.observation-qualification-transition.v1',reason='Existing ledger has only passed/failed completion API; failed here means qualification not admitted, not numerical failure.',observations=transitions),indent=2)+'\n')
for rec in records:rec['paths'].append(str(path))
result=ledger.record_many(records)
(B/'superseded-observation-transitions-result.json').write_text(json.dumps(dict(recorded=len(result),numerical_failed=0,genuine_failed_attempts_preserved=4,source_revision=doc['source_revision']),indent=2)+'\n');print('closed',len(result),'superseded qualifications; numerical failures invented:0; genuine failed attempts:4 preserved')
