import hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
base=Path(__file__).resolve().parent
lanes={'spectral','metrics-fowlkes-mallows','arima-exog','arima-exog-seasonal'}
env=os.environ.copy()
for key in ('PYTHONPATH','MOJOLEARN_HOST_DIR','MOJOLEARN_HOST_ALLOW_SABOTAGE','MOJOLEARN_IDENTITY_BREAK'):
    env.pop(key,None)
env.update(MOJOLEARN_NUMERIC_MODE='identical',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
def run(name,args):
    p=subprocess.run([str(base/'installed-env/bin/python'),'-m','mojolearn','verify',*args,'--json'],cwd=base,env=env,text=True,capture_output=True,timeout=300)
    (base/(name+'.json')).write_text(p.stdout)
    (base/(name+'.log')).write_text(p.stderr)
    assert p.returncode==0,(name,p.returncode,p.stderr[-1000:],p.stdout[-1000:])
    return json.loads(p.stdout)
r=run('coverage',['--coverage'])
assert all(r['lanes'][l]['status']=='available' for l in lanes)
assert r['evidence_snapshot_matches_harness']
for lane in lanes:
    controls=r['lanes'][lane]['historical_evidence']['negative_controls']
    assert len({c['fixture'] for c in controls if c['kind']=='build' and c['part']=='train' and '2026-09-17_cpu-public-promotion' in c['record']})==9
r=run('public-clean',['--lanes',','.join(sorted(lanes)),'--repeats','2','--no-models'])
assert set(r['lanes'])==lanes
assert r['counts']==dict(IDENTICAL=117,DIVERGENT=0,OWED=0,REFUSED=0,**{'N/A':63}),r['counts']
assert 'site-packages' in r['harness']['path']
# Exercise default selection without executing unrelated lanes.
code="from mojolearn import _verify_all as v; h=v.load_harness(); lanes,_=v.select_lanes(h,v.vref.load_table(),'cpu','full',[]); print(','.join(lanes))"
p=subprocess.run([str(base/'installed-env/bin/python'),'-c',code],cwd=base,env=env,text=True,capture_output=True,timeout=30)
assert p.returncode==0,p.stderr
assert lanes<=set(p.stdout.strip().split(','))
wheel=next((base/'dist').glob('*.whl'))
receipt=dict(wheel=wheel.name,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),lanes=sorted(lanes),counts=r['counts'],scope='Installed CPU development wheel with reused macOS bindings; fresh Linux source controls retained separately; not final release qualification')
(base/'wheel-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Installed wheel: coverage, 36 cells, and default CPU selection PASS',flush=True)
