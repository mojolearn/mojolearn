from pathlib import Path
import json,os,subprocess,time
base=Path('/Volumes/MojoPerfBuild20261005/six-lane-full-ab-20261006/apple/opponent-quality')
source=Path('/Volumes/MojoPerfBuild20261005/six-lane-full-ab-20261006/apple/source-47301d12')
sha='47301d12b14859e81cadc9ab6a0cd4f728d0e206'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==sha
assert not (base/'controller.pid').exists(),'Do not duplicate an existing launch'
assert not (base/'run-01/status.json').exists(),'Do not reset an existing attempt'
command=['/Users/ec2-user/mojolearn-full-f867b50e8/venv/bin/python','-u',str(source/'experiments/performance_ideas/apple_opponent_quality_20261006/run.py'),'--evidence',str(base/'run-01'),'--runtime','/Users/ec2-user/mojolearn-full-f867b50e8','--full-big','/Volumes/MojoPerfBuild20261005/full-ab-main-20261006/datasets/full-uncapped','--full-reg','/Volumes/MojoPerfBuild20261005/full-ab-main-20261006/apple-opponent-tail/data/rows-full','--expected-sha',sha]
(base/'command.json').write_text(json.dumps(command,indent=2)+'\n')
with (base/'controller.log').open('xb') as log:
 p=subprocess.Popen(command,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(base/'controller.pid').write_text(str(p.pid)+'\n')
time.sleep(1)
print(json.dumps({'pid':p.pid,'exit':p.poll(),'status_path':str(base/'run-01/status.json'),'source_sha':sha}))
if p.poll() is not None:raise SystemExit(p.returncode)
