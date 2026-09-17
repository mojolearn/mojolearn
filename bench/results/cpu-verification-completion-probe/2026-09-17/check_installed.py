import hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
base=Path(__file__).resolve().parent
lanes=sys.argv[1].split(',')
env=os.environ.copy()
for key in list(env):
    if key.startswith(('MOJOLEARN_', 'DYLD_')) or key in ('PYTHONPATH','PYTHONHOME'):
        env.pop(key)
env.update(MOJOLEARN_NUMERIC_MODE='identical',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
label=sys.argv[2]
def run(name,args):
    p=subprocess.run([str(base/'installed-env/bin/python'),'-m','mojolearn','verify',*args,'--json'],cwd=base,env=env,text=True,capture_output=True,timeout=300)
    (base/(name+'.json')).write_text(p.stdout);(base/(name+'.log')).write_text(p.stderr)
    assert p.returncode==0,(name,p.returncode,p.stderr[-1000:])
    return json.loads(p.stdout)
c=run(label+'-coverage',['--coverage'])
assert c['evidence_snapshot_matches_harness']
for lane in lanes:
    assert c['lanes'][lane]['status']=='available',lane
    if lane not in ('bpe-trainer', 'cross-val-folds'):
        assert len({x['fixture'] for x in c['lanes'][lane]['historical_evidence']['negative_controls'] if x['kind']=='build' and x['part']=='train'})==9,lane
r=run(label+'-clean',['--lanes',','.join(lanes),'--repeats','2','--no-models'])
assert all(r['counts'][state]==0 for state in ('DIVERGENT','OWED','REFUSED')),r['counts']
assert r['self_test']['passed'] and 'site-packages' in r['harness']['path']
code="from mojolearn import _verify_all as v; h=v.load_harness(); lanes,_=v.select_lanes(h,v.vref.load_table(),'cpu','full',[]); print(','.join(lanes))"
p=subprocess.run([str(base/'installed-env/bin/python'),'-c',code],cwd=base,env=env,text=True,capture_output=True,timeout=30)
assert p.returncode==0,p.stderr
assert set(lanes)<=set(p.stdout.strip().split(','))
wheel=next((base/'dist').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
    receipt=dict(wheel=wheel.name,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        source_commit=z.read('mojolearn/identity_columns/COMMIT').decode().strip(),lanes=lanes,counts=r['counts'],
        bindings={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith(('.so','.dylib'))},
        scope='Installed CPU development wheel, reused Mac native bindings, no path overrides; not final release qualification')
(base/(label+'-wheel-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(label,r['counts'],c['counts'],flush=True)
