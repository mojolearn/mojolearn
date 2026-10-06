"""Match newly retained CPU14 quality using saved input hashes/settings only."""
from pathlib import Path
import collections,datetime,hashlib,json,math,runpy,shutil,sys
BASE=Path('/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006')
REPO=Path('/Users/andrewhendel/CascadeProjects/mojolearn')
sys.path.insert(0,str(REPO/'tools'))
import af_quality
OUT=BASE/'quality-review'
semantic=runpy.run_path(str(OUT/'semantic-opponent-contracts.py'))
def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
facts={}
for p in (BASE/'apple/classical-workload-facts.json',BASE/'apple/resample-workload-facts.json',BASE/'apple/apple-reg-inputs.json',Path('/Users/andrewhendel/mojolearn-evidence/full-ab-main-20261006/gpu-six-admission/next-reg/native-next-reg-inputs.json')):
 x=load(p)
 for key,value in x.get('facts',x).items():
  if isinstance(value,dict) and 'dataset_sha256' in value:
   lane=key.split('@')[0].replace(':','/').rsplit('/',1)[-1]
   facts[(lane,key.split('dataset=')[-1],value['dataset_sha256'])]=value
plan=load(REPO/'experiments/performance_ideas/apple_opponent_quality_20261006/plan.json')
expected=collections.defaultdict(set)
for c in plan['cells']:expected[(c['race'].split('/')[1],c['dataset'])].add(c['arm'])
refs=collections.defaultdict(list)
capture=BASE/'apple/opponent-quality/captured'
for p in capture.glob('run-*/cells/*/result.json'):
 result=load(p)
 if result.get('status')!='MEASURED' or result.get('returncode')!=0:continue
 race=result['race'].split('/');lane,dataset=race[1:3];arm=result['arm']
 for q in (p.parent/'board/raw').rglob('*.json'):
  raw=load(q)
  if arm not in raw.get('quality',{}):continue
  refs[(lane,dataset)].append({'arm':arm,'result':result,'raw':raw,'path':str(q),'sha256':digest(q)})
updates=0
for target in (OUT/'classical-12-pair-quality-review.json',OUT/'next-reg-resample-quality-review.json'):
 review=load(target);changed=False
 for row in review['rows']:
  prior_assessment=row.get('quality_assessment','');prior_reason=row.get('reason')
  key=(row['lane'],row['dataset']);available=refs.get(key,[])
  if not available:continue
  fact=facts.get((*key,row.get('dataset_sha256')))
  comparisons=[];unqualified=[]
  for ref in available:
   raw=ref['raw'];input_receipt=ref['result']['input_receipt'];reason=[]
   ownfiles=(fact or {}).get('workload',{}).get('input_files',[])
   if not fact or not any(v.get('sha256')==input_receipt.get('npz_sha256') and str(v.get('path','')).endswith('.npz') for v in ownfiles):reason.append('No exact full NPZ match to this A/B workload fact')
   if raw.get('block',{}).get('arrays')!=input_receipt.get('arrays'):reason.append('Reference saved array metadata differs from full input receipt')
   ownparams=(fact or {}).get('estimator_settings',{}).get('params',{})
   oppparams=semantic['params'](raw.get('arms',{}).get(ref['arm'],{}).get('params_record',{}))
   if not ownparams or not oppparams:reason.append('Missing comparable constructor settings')
   original_differences={k:[v,oppparams.get(k)] for k,v in ownparams.items() if k not in oppparams or oppparams[k]!=v}
   mismatches,equivalences=semantic['apply'](row['lane'],ref['arm'],ownparams,oppparams,raw,row['source_sha'])
   if mismatches:reason.append('Constructor settings differ or are missing: '+json.dumps(mismatches,sort_keys=True))
   quality=raw.get('quality',{}).get(ref['arm'],{})
   if not quality or quality.get('finite') is False or any(isinstance(v,float) and not math.isfinite(v) for v in quality.values()):reason.append('Opponent task metrics missing/nonfinite; not an accepted quality reference')
   item={'arm':ref['arm'],'receipt':ref['path'],'receipt_sha256':ref['sha256'],'input_npz_sha256':input_receipt.get('npz_sha256'),'parameter_mismatches':mismatches,'recorded_parameter_differences':original_differences,'semantic_equivalences':equivalences,'parameter_record_schema':'nested get_params or direct declared function kwargs'}
   if reason:unqualified.append(dict(item,reasons=reason));continue
   quality=raw['quality'][ref['arm']]
   comparisons.append(dict(item,metrics=quality,candidate=af_quality.compare(row['metrics']['A'],quality),baseline=af_quality.compare(row['metrics']['B'],quality)))
  row['new_full_opponent_review']={'qualified':comparisons,'unqualified':unqualified,'expected_arms':sorted(expected[key]),'matched_arms':sorted({v['arm'] for v in comparisons}),'scope':'Saved task-quality only; no opponent ratios or default promotion'}
  # Preserve an observed candidate/baseline regression regardless of opponents.
  baseline=row.get('candidate_vs_baseline',{}).get('verdict')
  if baseline=='WORSE':row.update(quality_assessment='QUALITY_FAILED',reason='Candidate materially worse than baseline under existing quality rules; independent references do not override it.')
  elif any(v['candidate']['verdict']=='WORSE' for v in comparisons):
   row.update(quality_assessment='QUALITY_FAILED' if row['vendor']=='apple' else 'BASELINE_NONREGRESSION_WITH_OPPONENT_QUALITY_DEFICIT',reason='Saved candidate task metrics are worse than at least one exact-full-input/settings opponent under existing tolerance; no default admission.')
  elif baseline in ('SAME','BETTER') and not row.get('candidate_vs_baseline',{}).get('unknown') and expected[key] and expected[key]<={v['arm'] for v in comparisons} and all(v['candidate']['verdict'] in ('SAME','BETTER') and not v['candidate'].get('unknown') for v in comparisons):
   if prior_assessment=='QUALITY_FAILED' or prior_assessment.startswith('FAILED_'):row['superseded_quality_failure']={'assessment':prior_assessment,'reason':prior_reason,'superseded_by_exact_new_reference_receipts':[v['receipt_sha256'] for v in comparisons]}
   row.update(quality_assessment='TASK_METRIC_GATE_PASSED',reason='Saved candidate preserves baseline and meets all selected supported same-full-input/settings opponent task metrics under existing tolerances.')
  elif prior_assessment=='QUALITY_FAILED' or prior_assessment.startswith('FAILED_'):
   row.update(quality_assessment=prior_assessment,reason=prior_reason)
   row['new_full_opponent_review']['failure_preserved']='Unqualified or incomplete new references cannot erase an established quality failure.'
  else:row.update(quality_assessment='PENDING',reason='New full opponent output retained but comparable settings, complete opponent roster, or established metric directions remain unresolved.')
  changed=True;updates+=1
 if changed:
  review['new_opponent_review_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  review['counts']=dict(collections.Counter(v['quality_assessment'] for v in review['rows']))
  snap=OUT/'snapshots';snap.mkdir(exist_ok=True);shutil.copy2(target,snap/(target.stem+'-before-opponents-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'))
  tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(review,indent=2)+'\n');tmp.replace(target)
print(json.dumps({'new_reference_groups':len(refs),'reviewed_rows':updates,'no_numerical_work':True}))
