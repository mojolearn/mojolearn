import os,json,subprocess,sys,hashlib,zipfile
from pathlib import Path
b=Path(__file__).resolve().parent
env={k:v for k,v in os.environ.items() if not k.startswith(('MOJOLEARN_','DYLD_')) and k not in ('PYTHONPATH','PYTHONHOME')}
env.update(MOJOLEARN_NUMERIC_MODE='identical',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
python=str(b/'installed-env/bin/python')
cases=[('coverage',['--coverage']),('models',['--models-only','--repeats','2']),('cpu-shards',['--include-pending','--lanes','par-arima,par-forest,par-forest-et,par-forest-et-clf,par-forest-reg,par-holtwinters,par-mlp,par-queries-kde,par-queries-knn,par-queries-nn,par-queries-radius,par-reference-knn,par-reference-knn-reg,par-samba,par-samba-clip,par-scaler,par-scaler-minmax','--fixtures','base','--repeats','2','--no-models']),('pending-properties',['--include-pending','--lanes','transformer','--fixtures','base','--repeats','2','--batch-checks','--no-models'])]
cases.append(('pending-all',['--include-pending','--lanes','gbdt-query-rmse,gmm-random-init-sample,gmm-sample,gp-normalize-y,gp-optimize,gp-optimize-restarts,gp-sample-y,gp-sample-y-normalize,gpc,gpc-multiclass,ivf-extend,mamba3,samba,samba-untied-dropout-accum,svc-poly,transformer,transformer-window','--fixtures','base','--repeats','2','--no-models']))
results={};problems=[]
for name,args in cases:
 p=subprocess.run([python,'-m','mojolearn','verify',*args,'--json'],cwd=b,env=env,capture_output=True,text=True,timeout=600)
 (b/(name+'.json')).write_text(p.stdout);(b/(name+'.log')).write_text(p.stderr)
 try:r=json.loads(p.stdout)
 except ValueError:problems.append((name,'invalid JSON',p.returncode));continue
 results[name]={'exit':p.returncode,'counts':r.get('counts'),'verdict':r.get('verdict')}
 print(name,results[name],flush=True)
 if name=='coverage':
  assert r['counts']['appendix_entries']==246
  for e in r['entries']:
   if e.get('api') in ('HostForest','HostGBDT'):assert e['installed_check']['status']=='available'
 elif name=='models':
  if p.returncode!=0:problems.append((name,r['counts']))
  assert r['models_checked']==4 and r['lanes']==[] and r['self_test']['ran'] is False
 else:
  if r['counts']['REFUSED'] or r['counts']['DIVERGENT']:problems.append((name,r['counts']))
  assert p.returncode==5 and r['scope_gaps'],(name,p.returncode)
  assert 'site-packages' in r['harness']['path']
  if name=='pending-properties':
   assert set(('batchgrad','batchscale','ragged','rlpair'))<=set(r['properties'])
   assert 'transformer' in r['lanes'] and r['counts']['OWED']>0
(b/'installed-summary.json').write_text(json.dumps({'results':results,'problems':problems},indent=2)+'\n')
wheel=next((b/'dist').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
 receipt={'wheel':wheel.name,'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'source_commit':z.read('mojolearn/identity_columns/COMMIT').decode().strip(),'scope':'Installed CPU development wheel; numerical native bindings reused from fresh 7f5b786ae build; no new hardware or release qualification','files':{n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith(('.so','.dylib'))}}
(b/'wheel-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert not problems,problems
