import hashlib,json,os,subprocess,sys,time,shutil
from pathlib import Path
base=Path(__file__).resolve().parent
root=Path('/Users/andrewhendel/mojolearn-evidence/cpu-verification-completion/control-repair-source-544605965')
out=base/'native-control-repairs';out.mkdir(exist_ok=False)
os.chdir(root)
sys.path.insert(0,str(root/'tools'))
from verify_cpu_batch import evaluate_pair,expected_oracle_failure
python='/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python'
mojo='/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo'
env=os.environ.copy()
env.update(PYTHONPATH=str(root/'python'),MOJOLEARN_NUMERIC_MODE='identical')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[key]='1'
for key in ('MOJOLEARN_GPU_ARCHS','MACOSX_DEPLOYMENT_TARGET','MOJOLEARN_HOST_DIR','MOJOLEARN_HOST_ALLOW_SABOTAGE'):env.pop(key,None)
sdk=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-version'],text=True).strip()
receipt={'source_commit':'544605965a721d4ffa1f426ee20ed21e603fa85b','arms':{}}
for arm in ('clean','sabotage'):
 host=out/arm;host.mkdir()
 # A raw fresh core build resolves the shared numerical runtime from the activated environment.
 shutil.copy2(base/'fresh-host-build-7f5b786ae/_mojolearn_core_host.so',host)
 receipt['arms'][arm]={'families':{}}
 for family in ('estimators','preprocessing'):
  dest=host/f'_mojolearn_{family}_host.so'
  command=[mojo,'build','-j','1','--emit','shared-lib','--target-cpu','apple-m1','-Xlinker','-platform_version','-Xlinker','macos','-Xlinker','11.0','-Xlinker',sdk,'-D','MOJOLEARN_NUMERIC_IDENTICAL=1','-D','MOJOLEARN_COLUMN_CPU','-I','.','-I','bindings']
  if arm=='sabotage':command+=['-D','MOJOLEARN_HOST_SABOTAGE=1']
  command+=[f'bindings/_mojolearn_{family}_host.mojo','-o',str(dest)]
  started=time.monotonic()
  with (out/(arm+'-'+family+'-build.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
  receipt['arms'][arm]['families'][family]={'command':command,'seconds':time.monotonic()-started,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
 arm_env=dict(env,MOJOLEARN_HOST_DIR=str(host))
 if arm=='sabotage':arm_env['MOJOLEARN_HOST_ALLOW_SABOTAGE']='1'
 with (out/(arm+'.log')).open('w') as log:
  p=subprocess.run([python,'tools/identity_break.py','--require-cpu','--lanes','dbscan,dbscan-brute-l1,dbscan-weighted,minmax-scaler,minmax-scaler-clip','--repeats','2','--step-full','--json',str(out/('cpu-'+arm+'.json'))],env=arm_env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 receipt['arms'][arm]['exit_code']=p.returncode
 (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(arm,'exit',p.returncode,flush=True)
clean=json.loads((out/'cpu-clean.json').read_text());bad=json.loads((out/'cpu-sabotage.json').read_text())
results={}
for lane in ('dbscan','dbscan-brute-l1','dbscan-weighted','minmax-scaler','minmax-scaler-clip'):
 a=dict(clean,cells={k:v for k,v in clean['cells'].items() if k.startswith(lane+'/')})
 b=dict(bad,cells={k:v for k,v in bad['cells'].items() if k.startswith(lane+'/')})
 result=evaluate_pair(a,b,lane,clean['fixtures'])
 result['passed'] &= receipt['arms']['clean']['exit_code']==0 and (receipt['arms']['sabotage']['exit_code']==0 or (receipt['arms']['sabotage']['exit_code']==1 and expected_oracle_failure(bad)))
 results[lane]=result
(out/'negative-controls.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(r['passed'] for r in results.values()),results
print('PASS: all five repaired native controls, all nine fixtures, two repeats',flush=True)
