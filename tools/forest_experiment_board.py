#!/usr/bin/env python3
"""Incremental, own-only RF/ET evidence board. Never flips runtime defaults.

Input cells are immutable snapshots with hashes recorded on the measurement box.
The main experiment board and vendor views distinguish captured measurements from
cross-vendor identity qualification. Historical opponent ratios are not imported.
"""
import argparse,hashlib,json,math,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from identical_forest_state_fixture import arrays_hash
import numpy as np
PROFILES={'rf':['rf-k4','rf-k1','rf-k2','rf-k8'],'et':['et-float','et-u16']}
DATASETS={'rf':['taxi','istella'],'et':['istellareg','year']}
SHAPES={'taxi':[4110786,16],'istella':[2043304,220],'istellareg':[2043304,220],'year':[463715,90]}
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def inspect_cell(entry):
 root=pathlib.Path(entry['root']);snapshot=read(root/'snapshot.json')
 assert snapshot['status']=='COMPLETED_CELL_SNAPSHOT'
 assert {'validated.json','worker-receipt.json','invocation.json','worker.log','scored/receipt.json','scored/ours/model.npz','scored/ours/predictions.npz'}<=snapshot['files'].keys()
 for name,h in snapshot['files'].items():
  p=(root/name).resolve();assert p.is_relative_to(root.resolve());assert sha(p)==h,(name,'snapshot hash')
 outcome=snapshot['outcome'];assert outcome['status']=='MEASURED_STATE_CAPTURED' and outcome['worker_exit']==outcome['audit_exit']==0
 d=read(root/'validated.json');w=read(root/'worker-receipt.json');i=read(root/'invocation.json');r=read(root/'scored/receipt.json')['ours']
 assert d['status']=='FINITE_MEASURED_STATE_CAPTURED' and d['state']==r
 assert w['provenance_verified'] is True and w['driver_exit']==0 and w['provenance']==d['provenance']
 assert i['settings']=={'warmups':1,'scored_samples':1,'rows':'full','opponents':False}
 assert [d['rows'],d['features']]==SHAPES[entry['dataset']]
 assert finite(d['scored_ms']) and d['scored_ms']>0 and finite(d['warmup_ms']) and d['warmup_ms']>0
 assert r['inputs']['X_train']['shape']==SHAPES[entry['dataset']]
 assert r['model_params_record'] and r['model_native_config'] and r['metrics']
 assert all(finite(x['value']) for x in r['metrics'])
 proof=d['provenance'];a=proof['artifact_identity'];packet=read(entry['packet']);assert sha(entry['packet'])==entry['packet_sha256']
 assert proof['status']=='verified' and proof['numeric_mode']==a['numeric_mode']=='identical'
 assert a['source_commit']==packet['source_sha']==snapshot['source_sha']
 assert snapshot['harness_sha']==packet['harness_sha']
 assert a['vendor']=={'nvidia':'cuda','amd':'hip'}[entry['vendor']]
 expected={k:v['sha256'] for k,v in packet['bindings'][entry['profile']].items()}
 assert a['files']==expected
 runtime={packet['remote_root']+'/'+k:h for k,h in packet['packet_files'].items() if k.startswith('runtime/')}
 assert runtime and a['runtime_files']==runtime,'Runtime differs from pinned packet'
 family='identical/_mojolearn_'+('rf' if entry['family']=='rf' else 'trees')+'.so'
 loaded=[x for x in proof['loaded_files'] if x['file']==family];assert loaded and all(x['sha256']==expected[family] for x in loaded)
 assert proof['hardware'] and 'N/A' not in proof['hardware']
 assert proof['loaded_runtime_files'] and all(a['runtime_files'].get(x['file'])==x['sha256'] for x in proof['loaded_runtime_files'])
 for file,key in [('model.npz','model_state_hash'),('predictions.npz','prediction_hash')]:
  with np.load(root/'scored/ours'/file,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files if k!='device'}
  assert arrays and all(np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in 'fc')
  assert arrays_hash(arrays)==r[key],key
 assert sha(root/'scored/ours/model.npz')==r['model_file_sha256']
 return dict(entry,status='CAPTURED_PENDING_COMPARISON',scored_ms=d['scored_ms'],warmup_ms=d['warmup_ms'],state=r,settings=i['settings'],source_sha=a['source_commit'],harness_sha=snapshot['harness_sha'],hardware=proof['hardware'],artifact_identity=a,snapshot_sha256=sha(root/'snapshot.json'),prediction_digest=d['prediction_digest'])
def compatible(a,b,same_device=False):
 for key in ['source_sha','harness_sha','settings','family','dataset']:assert a[key]==b[key],key
 for key in ['input_hash','inputs','model_params_record','model_native_config']:assert a['state'][key]==b['state'][key],key
 if same_device:assert a['hardware']==b['hardware'],'same-device A/B required'
def bits(a,b):return all(a['state'][k]==b['state'][k] for k in ['model_state_hash','prediction_hash'])
def evaluate(rows):
 idx={(r['vendor'],r['dataset'],r['profile']):r for r in rows if r['status']=='CAPTURED_PENDING_COMPARISON'};decisions=[]
 for family,profiles in PROFILES.items():
  baseline=profiles[0]
  for profile in profiles[1:]:
   result={'family':family,'profile':profile,'baseline':baseline,'status':'PENDING_REQUIRED_CELLS','promoted':False,'ratios':{},'checks':[]};needed=[(v,d,p) for v in ['nvidia','amd'] for d in DATASETS[family] for p in [baseline,profile]]
   result['missing']=[list(k) for k in needed if k not in idx]
   try:
    for dataset in DATASETS[family]:
     if any((v,dataset,p) not in idx for v in ['nvidia','amd'] for p in [baseline,profile]):continue
     for p in [baseline,profile]:
      a,b=idx['nvidia',dataset,p],idx['amd',dataset,p];compatible(a,b);assert bits(a,b),('cross-vendor bits',dataset,p)
     for vendor in ['nvidia','amd']:
      a,b=idx[vendor,dataset,baseline],idx[vendor,dataset,profile];compatible(a,b,True)
      if family=='rf':assert bits(a,b),('RF profile bits',vendor,dataset)
      result['ratios'][vendor+'/'+dataset]=b['scored_ms']/a['scored_ms']
      result['checks'].append({'vendor':vendor,'dataset':dataset,'baseline_metrics':a['state']['metrics'],'candidate_metrics':b['state']['metrics'],'baseline_candidate_bits_equal':bits(a,b)})
    if result['missing']:decisions.append(result);continue
    result['status']='CROSS_VENDOR_COMPARABLE_REQUIRES_DECISION';result['combined_ratio']=math.exp(sum(math.log(x) for x in result['ratios'].values())/len(result['ratios']))
    result['vendor_ratios']={v:math.exp(sum(math.log(x) for k,x in result['ratios'].items() if k.startswith(v+'/'))/len(DATASETS[family])) for v in ['nvidia','amd']}
    result['limitations']='One warmup/sample; Apple/host identity and ET quality decision must be linked before promotion; no automatic material-regression threshold.'
   except AssertionError as e:result.update(status='REFUSED_COMPARISON',error=str(e))
   decisions.append(result)
 return decisions

def build(index):
 rows=[]
 for entry in index['cells']:
  try:rows.append(inspect_cell(entry))
  except (AssertionError,KeyError,ValueError,OSError) as e:rows.append(dict(entry,status='REFUSED_EVIDENCE',error=str(e)))
 return {'schema':'mojolearn.forest-experiment-board/1','promotion':False,'rows':rows,'decisions':evaluate(rows),'failures':index.get('failures',[]),'pending':index.get('pending',[]),'historical_attempts':index.get('historical_attempts',[])}
def write(board,out):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
 for name,data in [('board.json',board)]+[(v+'.json',dict(board,rows=[r for r in board['rows'] if r['vendor']==v])) for v in ['nvidia','amd']]:
  p=out/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');t.replace(p)
 lines=['# RF / ExtraTrees experiment board','','Own-only full fits: one excluded warmup and one scored sample. Pending comparison is not IDENTICAL qualification. No opponent ratios or default changes.','','| Vendor | Dataset | Profile | Scored ms | Evidence |','|---|---|---|---:|---|']
 for r in board['rows']:lines.append('| '+ ' | '.join([r['vendor'],r['dataset'],r['profile'],str(r.get('scored_ms','—')),r['status']])+' |')
 lines+=['','## Candidate decisions','']+[f"- {d['profile']}: {d['status']}; ratios {d.get('ratios',{})}" for d in board['decisions']]
 lines+=['','## Pending and failed evidence','',f"Pending cells: {len(board['pending'])}; current failed cells: {len(board['failures'])}. Historical attempts are retained in board.json."]
 (out/'BOARD.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--out',required=True);args=p.parse_args();b=build(read(args.index));write(b,args.out);print(json.dumps({'rows':len(b['rows']),'refused':sum(r['status']=='REFUSED_EVIDENCE' for r in b['rows']),'decisions':{d['profile']:d['status'] for d in b['decisions']}}))
