"""Fail-closed evidence checks for actual full forest fits; no numeric runtime."""
import hashlib,json,math,pathlib,re

def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))
def full_log(path,lane,dataset):
 text=pathlib.Path(path).read_text();lines=text.splitlines()
 assert not any(x.startswith('FSPEED-REFUSED ') for x in lines),'refused full fit'
 headers=[fields(x) for x in lines if x.startswith('FSPEED-HEADER ')]
 rounds=[fields(x) for x in lines if x.startswith('FSPEED ')]
 warmups=[fields(x) for x in lines if x.startswith('FSPEED-WARMUP ')]
 assert len(headers)==len(rounds)==len(warmups)==1,'exact one own arm, warmup and sample required'
 h=headers[0];r=rounds[0];w=warmups[0]
 assert h['mode']=='IDENTICAL' and h['size']=='shipped' and h['rounds']=='1'
 assert all(x['arm']=='ours' and x['lane']==lane for x in [h,r,w])
 assert r['round']=='1' and r['shape']==w['shape'] and r['shape'].startswith(dataset)
 assert re.fullmatch('[0-9a-f]{16}',r['hash'])
 for row in [r,w]:assert math.isfinite(float(row['ms'])) and float(row['ms'])>0
 scales=[fields(x) for x in lines if x.startswith('FSPEED-SCALE ')]
 assert len(scales)==2 and scales[0]['stage']=='before' and scales[1]['stage']=='after'
 expected={'taxi':(4110786,16),'istella':(2043304,220),'istellareg':(2043304,220),'year':(463715,90)}[dataset]
 for s in scales:assert (int(s['rows']),int(s['features']))==expected,'full workload shape differs'
 return {'scored_ms':float(r['ms']),'warmup_ms':float(w['ms']),'prediction_digest':r['hash'],'shape':r['shape'],'rows':expected[0],'features':expected[1]}

def fixture(path,profile,expected_vendor,expected_bindings):
 d=json.loads(pathlib.Path(path).read_text());n=6 if profile.startswith('rf-') else 12
 assert d['profile']==profile and d['vendor']==expected_vendor and d['numeric_mode'].lower()=='identical'
 assert d['status']=='CAPTURE_COMPLETE_REQUIRES_COMPARISON' and len(d['cases'])==n
 assert len({x['name'] for x in d['cases']})==n
 for row in d['cases']:
  assert row['status']=='CAPTURED'
  for key in ['input_hash','model_state_hash','prediction_hash']:assert re.fullmatch('[0-9a-f]{64}',row[key])
 for rec in d['bindings'].values():assert expected_bindings.get(rec['file'])==rec['sha256'],'unexpected fixture binding'
 return d

def compare_fixtures(left,right):
 assert left['profile']==right['profile'] or {left['profile'],right['profile']}<= {'rf-k4','rf-k1','rf-k2','rf-k8'}
 a={x['name']:x for x in left['cases']};b={x['name']:x for x in right['cases']};assert a.keys()==b.keys()
 for key in a:
  for field in ['input_hash','model_state_hash','prediction_hash','params','state_members']:assert a[key][field]==b[key][field],(key,field)
 return {'status':'EXACT_MATCH','cases':len(a)}

def scored_state(folder,harness,lane,dataset):
 import sys,numpy as np
 sys.path.insert(0,str(pathlib.Path(harness)/'tools'))
 from identical_forest_state_fixture import arrays_hash
 folder=pathlib.Path(folder);d=json.loads((folder/'receipt.json').read_text());assert set(d)=={'ours'};r=d['ours']
 with np.load(folder/'ours/model.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files if k!='device'}
 with np.load(folder/'ours/predictions.npz',allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
 assert {'offsets','colid','quesval','left_child','leaves','meta'}<=state.keys()
 assert pred and all(np.isfinite(v).all() for v in pred.values())
 assert all(np.isfinite(v).all() for v in state.values() if v.dtype.kind in 'fc'),'nonfinite model state'
 assert arrays_hash(state)==r['model_state_hash'] and arrays_hash(pred)==r['prediction_hash']
 assert hashlib.sha256((folder/'ours/model.npz').read_bytes()).hexdigest()==r['model_file_sha256']
 assert r['metrics'] and all(math.isfinite(x['value']) for x in r['metrics'])
 shape={'taxi':[4110786,16],'istella':[2043304,220],'istellareg':[2043304,220],'year':[463715,90]}[dataset]
 assert r['inputs']['X_train']['shape']==shape and r['inputs']['y_train']['shape'][0]==shape[0]
 assert set(r['inputs'])=={'X_train','y_train','X_test','y_test'}
 assert r['model_params_record'] and r['model_native_config']
 for key in ['input_hash','model_state_hash','prediction_hash']:assert re.fullmatch('[0-9a-f]{64}',r[key])
 return r

if __name__=='__main__':
 import sys
 root=pathlib.Path(sys.argv[1]);harness=sys.argv[2];lane=sys.argv[3];dataset=sys.argv[4]
 r=full_log(root/'worker.log',lane,dataset);r['state']=scored_state(root/'scored',harness,lane,dataset)
 proof=json.loads((root/'worker-receipt.json').read_text());assert proof['provenance_verified'] is True and proof['driver_exit']==0
 r.update(status='FINITE_MEASURED_STATE_CAPTURED',qualification='PENDING_REQUIRED_CROSS_VENDOR_AND_REFERENCE_COMPARISON',provenance=proof['provenance'])
 (root/'validated.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
