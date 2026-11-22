import hashlib,json,os,shutil,subprocess
from pathlib import Path
R=Path('/Users/andrewhendel/mojolearn-wt/qualify-ordinary-23')
OLD=Path('/Users/andrewhendel/mojolearn-wt/release-087-final')
B=Path('/Users/andrewhendel/mojolearn-evidence/next-wheel-coverage/cpu-kernel-identity')
OUT=Path('/Users/andrewhendel/mojolearn-evidence/ordinary-23-2026-09-18')
P=OLD/'.pixi/envs/test/bin/python'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
stage=OUT/'gpu-package/mojolearn'
shutil.copytree(R/'python/mojolearn',stage,ignore=shutil.ignore_patterns('__pycache__','*.so','host','identical','.pytest_cache'))
(stage/'identical').mkdir()
for path in (OLD/'python/mojolearn/identical').iterdir():
 if path.name!='_mojolearn_kernel_methods.so':(stage/'identical'/path.name).symlink_to(path)
(stage/'identical/_mojolearn_kernel_methods.so').symlink_to(B/'apple/_mojolearn_kernel_methods.so')
if (OLD/'python/mojolearn/.dylibs').exists():(stage/'.dylibs').symlink_to(OLD/'python/mojolearn/.dylibs')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=dict(status='INCOMPLETE',source_commit=commit,scope='Apple saved-model recording and CPU replay using retained kernel-identity native bindings; not fresh main-native or wheel qualification',native={str(p):sha(p) for p in [B/'apple/_mojolearn_kernel_methods.so',B/'clean/_mojolearn_estimators_host.so']},stages=[])
def save():(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
save()
env={k:v for k,v in os.environ.items() if not k.startswith('MOJOLEARN_') and k not in ('PYTHONPATH','PYTHONHOME')}
env.update(MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_COMMIT=commit,MOJOLEARN_GATE_COMMIT=commit,MOJOLEARN_HOST_DIR=str(B/'clean'),PYTHONPATH=str(stage.parent)+os.pathsep+str(R/'tools'))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MOJOLEARN_CPU_THREADS'):env[k]='1'
for family in ('kernel-ridge','nystroem'):
 for kernel in ('poly','sigmoid','laplacian'):
  lane=family+'-'+kernel
  models=OUT/'saved-models'/lane
  for mode,args in [('record',['record',str(models),'--lanes',lane]),('replay',['check',str(models),'--gpu-column',str(B/('apple-'+lane+'.json')),'--report',str(OUT/(lane+'-replay.json'))])]:
   cmd=[str(P),str(R/'tools/classical_host_gate.py'),'--package-root',str(stage.parent),*args]
   with (OUT/(lane+'-'+mode+'.log')).open('w') as log:ret=subprocess.run(cmd,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=180)
   receipt['stages'].append(dict(lane=lane,mode=mode,exit=ret.returncode));save()
   if ret.returncode:raise RuntimeError((lane,mode,ret.returncode))
receipt['status']='APPLE_SAVED_MODEL_REPLAY_PASSED_UNQUALIFIED';save()
