import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path('/Users/andrewhendel/mojolearn-wt/cpu-kernel-identity')
OLD=Path('/Users/andrewhendel/mojolearn-wt/release-087-final')
B=Path('/Users/andrewhendel/mojolearn-evidence/next-wheel-coverage/cpu-kernel-identity')
P=OLD/'.pixi/envs/test/bin/python'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip()
receipt=dict(status='INCOMPLETE',source_commit=commit,scope='New kernel-family CPU and Apple source comparison; not installed-wheel or NVIDIA/AMD qualification',records=[],reused_support_bindings_source='aff968968f3ffc45844e36f3fd6260391eb829e5')
def save(): (B/'comparison-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
save()
env={k:v for k,v in os.environ.items() if not k.startswith('MOJOLEARN_') and k not in ('PYTHONPATH','PYTHONHOME')}
env.update(MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_COMMIT=commit,MOJOLEARN_GATE_COMMIT=commit)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MOJOLEARN_CPU_THREADS'):env[k]='1'
def run(name,args,e,timeout=600,allowed=(0,)):
 with (B/(name+'.log')).open('w') as log:
  p=subprocess.run(args,cwd=R,env=e,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 assert p.returncode in allowed,(name,p.returncode)
 print('PASSED',name,flush=True)
try:
 stage=B/'gpu-package/mojolearn'
 shutil.copytree(R/'python/mojolearn',stage,ignore=shutil.ignore_patterns('__pycache__','*.so','host','identical','.pytest_cache'))
 (stage/'identical').mkdir()
 for p in (OLD/'python/mojolearn/identical').iterdir():
  if p.name!='_mojolearn_kernel_methods.so':(stage/'identical'/p.name).symlink_to(p)
 (stage/'identical/_mojolearn_kernel_methods.so').symlink_to(B/'apple/_mojolearn_kernel_methods.so')
 if (OLD/'python/mojolearn/.dylibs').exists() and not (stage/'.dylibs').exists():(stage/'.dylibs').symlink_to(OLD/'python/mojolearn/.dylibs')
 lanes=[f'{f}-{k}' for f in ('kernel-ridge','nystroem') for k in ('poly','sigmoid','laplacian')]
 fixtures={'base','ties','hashed','wide','denormal','denormal_ftz','dupes','odd','negative'}
 parts=('train','infer','model','batch','stepfull','batchgrad','batchscale','ragged')
 columns={}
 for arm in ('cpu','apple','sabotage'):
  ae=env.copy();ae['PYTHONPATH']=str(stage.parent if arm=='apple' else R/'python')+os.pathsep+str(R/'tools')
  ae['MOJOLEARN_HOST_DIR']=str(B/('sabotage' if arm=='sabotage' else 'clean'))
  if arm=='sabotage':ae['MOJOLEARN_HOST_ALLOW_SABOTAGE']='1'
  for lane in lanes:
   name=arm+'-'+lane;target=B/(name+'.json')
   args=[str(P),str(R/'tools/identity_break.py'),'--require-backend','metal' if arm=='apple' else 'cpu','--vendor','apple-m4' if arm=='apple' else 'cpu-apple-m4','--lanes',lane,'--repeats','2','--batch-grad','--batch-scale','--ragged','--step-full','--json',str(target)]
   if arm!='sabotage':args.append('--fail-on-refused')
   run(name,args,ae)
   doc=json.loads(target.read_text());assert doc['complete'] and doc['repeats']==2
   assert set(doc['cells'])=={lane+'/'+f for f in fixtures}
   if arm!='sabotage':
    for cell,c in doc['cells'].items():
     for part in parts:
      vals=c.get('hashes' if part=='train' else part);verdict=c.get('verdict' if part=='train' else part+'_verdict')
      assert vals and len(vals)==2 and vals[0]==vals[1] and verdict in ('STABLE','N/A'),(arm,cell,part,verdict)
   columns[(arm,lane)]=doc
   receipt['records'].append(dict(arm=arm,lane=lane,sha256=hashlib.sha256(target.read_bytes()).hexdigest()));save()
 matched=0;negative=[]
 for lane in lanes:
  a=columns[('cpu',lane)]['cells'];b=columns[('apple',lane)]['cells'];s=columns[('sabotage',lane)]['cells']
  for cell,c in a.items():
   for part in parts:
    key='hashes' if part=='train' else part
    assert c[key]==b[cell][key],(cell,part,c[key],b[cell][key])
    if not c[key][0].startswith('n/a:'):matched+=1
   sc=s[cell];assert sc['verdict']=='STABLE' and len(set(sc['hashes']))==1,(cell,sc['verdict'])
   assert sc['hashes']!=c['hashes'],(cell,'sabotage did not move train hash')
   negative.append(cell)
 receipt.update(status='PASSED',matched_numerical_parts=matched,sabotage_changed_training_cells=negative);save()
except BaseException as e:
 receipt.update(status='FAILED',reason=repr(e));save();raise
