import gzip,hashlib,json,os,subprocess,sys,time,zipfile,shutil
from pathlib import Path
base=Path(__file__).resolve().parent
out=base/'fresh-cpu-wheel-replay';out.mkdir(exist_ok=False)
python=base/'fresh-installed-env/bin/python'
env=os.environ.copy()
for key in list(env):
 if key.startswith(('MOJOLEARN_','DYLD_')) or key in ('PYTHONPATH','PYTHONHOME'):env.pop(key)
env.update(MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_CPU_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
wheel=next((base/'fresh-dist').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
 receipt={'wheel':wheel.name,'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'source_commit':z.read('mojolearn/identity_columns/COMMIT').decode().strip(),'native_build_commit':'7f5b786ae','bindings':{n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith(('.so','.dylib'))},'scope':'All available CPU reference lanes, all nine fixtures, two repetitions; fresh native builds. Not GPU, multi-GPU, cross-interpreter or full release qualification.'}
(out/'wheel-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
archive=base/'wheel-artifacts'/receipt['sha256'];archive.mkdir(parents=True,exist_ok=True);shutil.copy2(wheel,archive/wheel.name)
result=subprocess.run([str(python),'-m','mojolearn','verify','--coverage','--json'],cwd=base,env=env,capture_output=True,text=True,timeout=120)
(out/'coverage.json.gz').write_bytes(gzip.compress(result.stdout.encode(),mtime=0));(out/'coverage.log').write_text(result.stderr)
result.check_returncode();coverage=json.loads(result.stdout)
assert coverage['evidence_snapshot_matches_harness']
lanes=sorted(lane for lane,row in coverage['lanes'].items() if row['status']=='available')
assert len(lanes)==150,(len(lanes),coverage['counts'])
summary={'complete':False,'lanes':lanes,'results':{},'wheel_sha256':receipt['sha256']}
def checkpoint(): (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
checkpoint()
for lane in lanes:
 command=[str(python),'-m','mojolearn','verify','--lanes',lane,'--repeats','2','--no-models','--json']
 started=time.monotonic()
 try:
  run=subprocess.run(command,cwd=base,env=env,capture_output=True,text=True,timeout=1200)
  stdout,stderr=run.stdout,run.stderr
  row={'exit_code':run.returncode,'seconds':time.monotonic()-started,'command':command}
  if stdout:
   try:
    report=json.loads(stdout);row['counts']=report['counts'];row['self_test']=report['self_test'];row['harness']=report['harness']
    row['passed']=run.returncode==0 and all(report['counts'][s]==0 for s in ('DIVERGENT','REFUSED','OWED')) and report['self_test']['passed'] and 'site-packages' in report['harness']['path']
   except (ValueError,KeyError) as e:row['error']=str(e);row['passed']=False
  else:row['passed']=False
 except subprocess.TimeoutExpired as exc:
  stdout,stderr=exc.stdout or '',exc.stderr or ''
  if isinstance(stdout,bytes):stdout=stdout.decode(errors='replace')
  if isinstance(stderr,bytes):stderr=stderr.decode(errors='replace')
  row={'exit_code':124,'seconds':time.monotonic()-started,'command':command,'passed':False,'error':'1200-second timeout; no incomplete record admitted'}
 (out/(lane+'.json.gz')).write_bytes(gzip.compress(stdout.encode(),mtime=0));(out/(lane+'.log')).write_text(stderr)
 summary['results'][lane]=row;checkpoint()
 print(f'{lane}: {"PASS" if row["passed"] else "FAIL"} {row["seconds"]:.1f}s {row.get("counts",{})}',flush=True)
summary['complete']=True;summary['passed']=all(r['passed'] for r in summary['results'].values());checkpoint()
print(f'Completed {len(lanes)} lanes, passed {sum(r["passed"] for r in summary["results"].values())}',flush=True)
raise SystemExit(0 if summary['passed'] else 1)
