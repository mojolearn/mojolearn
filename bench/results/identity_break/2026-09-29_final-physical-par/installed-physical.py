import hashlib,json,os,pathlib,signal,subprocess,sys,time
if not __debug__: raise RuntimeError('Python -O is refused for physical qualification')
R=pathlib.Path('/root/final-artifact-9ab5d5269');O=R/'physical';O.mkdir(exist_ok=False)
import mojolearn as m
from mojolearn import _backend
import importlib
ib=importlib.import_module('mojolearn._identity_break')
assert m.vendor()=='cuda' and _backend.gpu_arch()=='sm_89'
P=pathlib.Path(m.__file__).parent
assert P.is_relative_to(pathlib.Path(sys.prefix)) and (P/'identity_columns/COMMIT').read_text().strip()=='9ab5d5269f60667efbe3da56307edc4f84261601'
sys.path.insert(0,str(R/'tools'));import algos_lane_check as a
h=P/'_identity_break.py';assert h.exists()
fixtures=('base','ties','hashed','wide','denormal','denormal_ftz','dupes','odd','negative')
parts=('infer','model','batch','stepfull','batchgrad','batchscale','ragged');rows=[]
provenance={'package':str(P),'source':'9ab5d5269f60667efbe3da56307edc4f84261601','native':'3dded3bd47b9db2928ed3b7919f502b20c576e9c','harness_sha256':hashlib.sha256(h.read_bytes()).hexdigest(),'architecture':_backend.gpu_arch(),'bindings':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*.so')}}
(O/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
for lane in ('par-gmm','par-graph-umap'):
 for fx in fixtures:
  row={'lane':lane,'fixture':fx,'passed':False}
  try:
   values=[]
   for kind in ('one','two'):
    path=O/f'{lane}.{fx}.{kind}.json';witness=path.with_suffix('.witness.json')
    args=['--lanes',lane,'--fixtures',fx,'--repeats','1','--fail-on-refused','--require-backend','cuda','--json',str(path)]
    cmd=[sys.executable,'-u',str(h),*args] if kind=='one' else [sys.executable,'-u',str(R/'installed-witness.py'),'--witness-json',str(witness),'--',*args]
    env={k:v for k,v in os.environ.items() if not k.startswith('MOJOLEARN_') and k not in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','PYTHONOPTIMIZE')};env.update(MOJOLEARN_COMMIT=provenance['source'],MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_PAR_DEVICES='0' if kind=='one' else '0,1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MOJOLEARN_CPU_THREADS='1')
    start=time.monotonic();code=124
    with path.with_suffix('.log').open('w') as log:
     p=subprocess.Popen(cmd,env=env,cwd=O,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     try:code=p.wait(timeout=120)
     except subprocess.TimeoutExpired:pass
     finally:
      try:os.killpg(p.pid,signal.SIGKILL)
      except ProcessLookupError:pass
      p.wait()
    assert code==0,(kind,code)
    doc=json.loads(path.read_text());assert doc['commit']==provenance['source'] and doc['commit_source']=='MOJOLEARN_COMMIT';assert doc['lane_revisions'][lane]==ib.LANE_REVISIONS[lane];key=f'{lane}/{fx}';assert set(doc['cells'])=={key};c=doc['cells'][key];assert c['verdict']=='STABLE' and len(c['hashes'])==1
    for part in parts:
     assert c[part+'_verdict'] in ('STABLE','N/A') and len(c[part])==1,(kind,part,c)
     assert 'UNDECLARED' not in c[part][0]
    values.append(c)
    if kind=='two':
     w=json.loads(witness.read_text());assert set(w['cells'])=={key};verdict,detail=a.par_witness_verdict(lane,'AGREE','strict exact8-part raw comparison',w,[fx]);assert verdict=='AGREE',(verdict,detail)
   for part in ('hashes',)+parts:assert values[0][part]==values[1][part],(part,values[0][part],values[1][part])
   row.update(passed=True,witness=detail)
  except Exception as e:row['error']=repr(e)
  rows.append(row);print(json.dumps(row),flush=True);(O/'summary.json').write_text(json.dumps({'source':provenance['source'],'lanes':['par-gmm','par-graph-umap'],'fixtures':list(fixtures),'rows':rows,'passed':len(rows)==18 and all(x['passed'] for x in rows)},indent=2)+'\n')
raise SystemExit(not all(x['passed'] for x in rows))
