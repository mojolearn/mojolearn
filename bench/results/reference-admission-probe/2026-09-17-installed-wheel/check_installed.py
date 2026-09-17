from pathlib import Path
import json, os, subprocess, sys, hashlib, zipfile
base=Path(__file__).resolve().parent
py=str(base/'installed-env/bin/python')
env=os.environ.copy()
for key in ('PYTHONPATH','MOJOLEARN_HOST_DIR','MOJOLEARN_HOST_ALLOW_SABOTAGE','MOJOLEARN_IDENTITY_BREAK','MOJOLEARN_IDENTITY_GBDT_CTR_MODELS'):
    env.pop(key,None)
env.update(MOJOLEARN_NUMERIC_MODE='identical',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
def run(name,args,want):
    p=subprocess.run([py,'-m','mojolearn','verify',*args,'--json'],cwd=base,env=env,text=True,capture_output=True,timeout=120)
    (base/f'{name}.json').write_text(p.stdout)
    (base/f'{name}.log').write_text(p.stderr)
    assert p.returncode==want,(name,p.returncode,p.stderr[-600:],p.stdout[-600:])
    result=json.loads(p.stdout)
    print(name,'PASS',flush=True)
    return result
r=run('coverage',['--coverage'],0)
assert len(r['entries'])==246 and r['evidence_snapshot_matches_harness']
assert r['reference_admission_policy']['status']=='legacy'
for lane in ('kpss','select-d','holtwinters','holtwinters-multiplicative'):
    rows=r['lanes'][lane]['historical_evidence']['negative_controls']
    controls=[c for c in rows if c['kind']=='build' and c['part']=='train' and '2026-09-17_tsa-negative-controls' in c['record']]
    assert len(controls)==9,(lane,len(controls))
assert r['evidence_provenance']['sources'] and not r['evidence_provenance']['release_qualified']
r=run('ctr',['--lanes','gbdt-categorical-ctr-tables,gbdt-tensor-ctr-tables','--repeats','2','--no-models'],0)
assert r['counts']==dict(IDENTICAL=54,DIVERGENT=0,OWED=0,REFUSED=0,**{'N/A':36}),r['counts']
assert 'site-packages' in r['harness']['path']
for name in ('ols-a','ols-b'):
    r=run(name,['--lanes','ols','--fixtures','base','--repeats','2','--no-models'],0)
    assert r['verification_contract']['fixtures']['base']['X']
run('compare',['--compare',str(base/'ols-a.json'),str(base/'ols-b.json')],0)
a=json.loads((base/'ols-a.json').read_text())
a['verification_contract']['heldout']['base']['X']='0'*16
(base/'changed-context.json').write_text(json.dumps(a))
r=run('compare-changed',['--compare',str(base/'ols-a.json'),str(base/'changed-context.json')],4)
assert r['verdict']=='INCOMPARABLE' and r['agree']==0
wheel=next((base/'dist').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
    members=[n for n in z.namelist() if n.startswith('mojolearn/verify_reference/ctr_models/')]
    assert len(members)==18
    for path in members:
        assert z.read(path)
receipt=dict(wheel=wheel.name,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),ctr_models=len(members),scope='CPU development wheel with reused candidate bindings; not release qualification')
(base/'wheel-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
