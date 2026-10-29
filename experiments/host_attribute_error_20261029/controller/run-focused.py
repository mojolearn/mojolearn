from pathlib import Path
import datetime,hashlib,json,os,subprocess,xml.etree.ElementTree as ET
r=Path(__file__).parent
python=r/'venv-repaired/bin/python';test=r/'tests/test_host_attribute_lifetime.py'
env=dict(os.environ);env.pop('PYTHONPATH',None)
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32='0',MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE='0')
command=[str(python),'-m','pytest','-x','-vv',str(test),'--junitxml='+str(r/'focused.xml'),'--basetemp='+str(r/'test-tmp')]
launch={'command':command,'cwd':str(r),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage_receipt_sha256':hashlib.sha256((r/'stage-receipt.json').read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(test.read_bytes()).hexdigest(),'expected_tests':11,'environment':{k:v for k,v in env.items() if k.startswith('MOJOLEARN_') or k in ('PYTHONDONTWRITEBYTECODE','PYTHONFAULTHANDLER')}}
(r/'focused-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
with (r/'focused.log').open('w') as output:
 result=subprocess.run(command,cwd=r,env=env,stdout=output,stderr=subprocess.STDOUT)
(r/'focused.exit').write_text(str(result.returncode)+'\n')
summary={'exit_code':result.returncode,'expected_tests':11}
if (r/'focused.xml').exists():
 tree=ET.parse(r/'focused.xml');cases=tree.findall('.//testcase');summary.update(tests=len(cases),failed=sum(bool(list(x.findall('failure'))+list(x.findall('error'))) for x in cases),skipped=sum(x.find('skipped') is not None for x in cases))
summary['qualified']=result.returncode==0 and summary.get('tests')==11 and summary.get('failed')==0 and summary.get('skipped')==0
(r/'focused-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));raise SystemExit(0 if summary['qualified'] else 1)
