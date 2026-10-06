"""Review saved terminal task metrics only; no model loads, fits, or verification."""
from pathlib import Path
import collections, datetime, hashlib, json, math, shutil, sys
REPO=Path('/Users/andrewhendel/CascadeProjects/mojolearn')
BASE=Path('/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006')
OUT=BASE/'quality-review'
sys.path.insert(0,str(REPO/'tools'))
import af_quality
reference_path=Path('/Users/andrewhendel/mojolearn-evidence/full-ab-main-20261006/apple-repair-2h/captured/attempt-01/cells/cell-03/board/raw/classical2/rows-full/gmm-taxi.json')
reference=json.loads(reference_path.read_text())
audit_path=Path('/Users/andrewhendel/mojolearn-evidence/full-ab-main-20261006/gpu-six-admission/next-reg/gmm-taxi-full-audit.json')
audit=json.loads(audit_path.read_text())
expected_gmm_dataset_sha256=json.loads(Path('/Users/andrewhendel/mojolearn-evidence/full-ab-main-20261006/gpu-six-admission/next-reg/native-next-reg-inputs.json').read_text())['facts']['more:gmm@dataset=taxi']['dataset_sha256']
board=json.loads((REPO/'bench/results/bench_board/m3ultra-0834/board.json').read_text())
old_refs={}
for section in ('races','extra_races'):
 for rid,race in board.get(section,{}).items():
  if race.get('lane') not in ('ridge','lasso','elasticnet','linearsvr','gmm','resample','sgd-reg','pa-reg','bayesian-ridge','ridge-cv','lasso-cv','huber','lars','lasso-lars','enet-cv'):continue
  for cell in race.get('cells',[]):
   if cell.get('mode')=='opponent':old_refs.setdefault((race.get('lane'),race.get('dataset')),[]).append({'race':rid,'arm':cell.get('arm'),'status':cell.get('status'),'shape':cell.get('shape'),'quality':cell.get('quality')})
roots=[('nvidia',BASE/'nvidia-native/capture-attempt-02/artifacts'),('apple',BASE/'apple/captured'),('amd',BASE/'amd/capture-attempt-01/artifacts')]
previous_path=OUT/'next-reg-resample-quality-review.json'
previous_rows={r['receipt_sha256']:r for r in json.loads(previous_path.read_text()).get('rows',[])} if previous_path.exists() else {}
rows=[];unfinished=[];read_errors=[];seen=set()
for vendor,root in roots:
 for path in root.rglob('receipt.json'):
  try: raw=path.read_bytes();receipt=json.loads(raw)
  except (OSError,ValueError) as exc:read_errors.append({'path':str(path),'error':repr(exc)});continue
  workload=receipt.get('workload',{});wid=workload.get('workload_id','')
  # Namespace names differ across native/Apple controllers; derive the lane
  # generically and exclude only the already-reviewed original classical cells.
  lane=wid.split('@',1)[0].replace(':','/').rsplit('/',1)[-1]
  if vendor != 'amd' and lane in ('pca','ols','kmeans'):continue
  sha=hashlib.sha256(raw).hexdigest()
  if sha in seen:continue
  seen.add(sha);dataset=wid.partition('@dataset=')[2].split('@',1)[0]
  if receipt.get('status')!='MEASURED_FULL':unfinished.append({'vendor':vendor,'workload':wid,'receipt':str(path),'status':receipt.get('status','RUNNING')});continue
  scored={r['arm']:r for r in receipt.get('runs',[]) if r.get('phase')=='scored'}
  row={'vendor':vendor,'workload':wid,'mode':receipt.get('mode'),'lane':lane,'dataset':dataset,'source_sha':receipt.get('source_sha'),'receipt':str(path),'receipt_sha256':sha,'execution_status':receipt['status'],'returncodes':[r.get('returncode') for r in receipt.get('runs',[])],'dimensions':workload.get('dimensions'),'dataset_sha256':workload.get('dataset_sha256'),'promotion_authorized':False}
  metrics={a:next(iter(r.get('result',{}).get('task_quality',{}).get('metrics',{}).values()),{}) for a,r in scored.items()}
  row.update(metrics=metrics,output_sha256={a:r.get('output_sha256') for a,r in scored.items()},sample_counts={a:{'excluded_warmups':sum(v.get('arm')==a and v.get('phase')=='warmup' and v.get('excluded') is True and v.get('returncode')==0 for v in receipt.get('runs',[])),'scored':sum(v.get('arm')==a and v.get('phase')=='scored' and v.get('returncode')==0 for v in receipt.get('runs',[]))} for a in scored})
  row['source_coverage_pending']=workload.get('source_coverage_pending',[])
  row['declared_timed_boundary']=workload.get('timed_boundary')
  row['saved_task_quality_declarations']={a:{k:r.get('result',{}).get('task_quality',{}).get(k) for k in ('status','gate_source','reason')} for a,r in scored.items()}
  if 'svd' in lane.lower():
   row['output_scope_limitations']=['Saved metrics and declared consumed outputs only; no broader reconstruction/subspace/orthogonality quality or complete-model identity admission is inferred. Unknown metric directions and unmatched full opponents remain pending.']

  if set(metrics)!= {'A','B'} or not all(metrics.values()):
   row.update(quality_assessment='PENDING',reason='Missing scored A/B task quality')
   previous=previous_rows.get(sha,{})
   if previous.get('quality_assessment')=='QUALITY_FAILED' or previous.get('quality_assessment','').startswith('FAILED_'):
    row.update(quality_assessment=previous['quality_assessment'],reason=previous.get('reason'),established_quality_failure_preserved=True)
   rows.append(row);continue
  finite=all(not q.get('error') and q.get('finite') is not False and all(not isinstance(v,float) or math.isfinite(v) for v in q.values()) for q in metrics.values())
  comparison=af_quality.compare(metrics['A'],metrics['B']);row['candidate_vs_baseline']=comparison
  row['metric_directions']={k:af_quality.direction(k) for k in metrics['A']}
  row['historical_opponents']=old_refs.get((lane,dataset),[])
  if not finite or comparison['verdict']=='WORSE':
   row.update(quality_assessment='QUALITY_FAILED',reason='Saved scored metrics are nonfinite or candidate materially worse than baseline under existing af_quality rules.')
  elif comparison['verdict']=='UNKNOWN' or comparison.get('unknown'):row.update(quality_assessment='PENDING',reason='Saved metrics lack an established shared comparison; no quality threshold invented.')
  elif lane=='gmm' and dataset=='taxi':
   params=workload.get('estimator_settings',{}).get('params',{});refparams=reference['arms']['sklearn-cpu']['params_record']['params'];mismatch={k:[v,refparams[k]] for k,v in params.items() if k in refparams and v!=refparams[k]}
   missing_parameters=sorted(set(audit['estimator_settings']['params'])-set(params)); inputs_match=(audit['input_metadata']['arrays']==reference['block']['arrays'] and workload.get('dataset_sha256')==expected_gmm_dataset_sha256);row['reference_provenance']={'receipt':str(reference_path),'sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),'arrays_shape_dtype_sha256_match':inputs_match,'shared_constructor_parameter_mismatches':mismatch,'missing_constructor_parameters':missing_parameters,'matched_parameter_names':sorted(set(params)&set(refparams)),'workload_dataset_sha256':workload.get('dataset_sha256'),'runtime_reference_precision':'sklearn same-value float64 widening included in fit/inference clock; retained quality only','scope':'Independent task-quality comparison; no historical timing or ratio imported'}
   if not inputs_match or mismatch or missing_parameters:row.update(quality_assessment='PENDING',reason='Full GMM reference found but input/constructor comparability unresolved')
   else:
    row['opponent_metrics']=reference['quality']['sklearn-cpu'];opp=af_quality.compare(metrics['A'],row['opponent_metrics']);row['candidate_vs_opponent']=opp
    row['baseline_vs_opponent']=af_quality.compare(metrics['B'],row['opponent_metrics'])
    if opp['verdict']=='WORSE':row.update(quality_assessment='QUALITY_FAILED' if vendor=='apple' else 'BASELINE_NONREGRESSION_WITH_OPPONENT_QUALITY_DEFICIT',reason='Candidate preserves baseline but has lower task quality than retained same-input sklearn GMM; report all BIC/log-likelihood outcomes without default admission.')
    elif opp['verdict'] in ('SAME','BETTER'):row.update(quality_assessment='TASK_METRIC_GATE_PASSED',reason='Recorded candidate task metrics preserve baseline and meet same-input independent sklearn quality under existing tolerances.')
    else:row.update(quality_assessment='PENDING',reason='Independent GMM reference has no judged shared metric')
  else:row.update(quality_assessment='PENDING',reason='Baseline task metrics show no material regression, but no matched accepted full-workload opponent reference has been established; historical reduced or quality-failed references do not qualify.')
  if vendor in ('nvidia','amd'):row['remaining_requirements']=[('AMD' if vendor=='nvidia' else 'NVIDIA')+' timing and same-arm cross-column IDENTICAL identity','Complete affected estimator/configuration coverage']
  else:row['remaining_requirements']=['Independent same-full-workload opponent quality unless explicitly qualified above','Complete affected estimator/configuration coverage']
  previous=previous_rows.get(sha,{})
  prior=previous.get('quality_assessment','')
  if row.get('quality_assessment')=='PENDING' and (prior=='QUALITY_FAILED' or prior.startswith('FAILED_')):
   row.update(quality_assessment=prior,reason=previous.get('reason'),established_quality_failure_preserved=True)
   if previous.get('new_full_opponent_review'):row['previous_full_opponent_failure_evidence']=previous['new_full_opponent_review']
  rows.append(row)
result={'schema':'mojolearn.saved-quality-review/1','observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Every captured native/Apple terminal new receipt including TSVD and selectors; saved taskmetrics only; original12classicalpairs reviewed separately','discovery_roots':[{ 'vendor':v,'path':str(p),'recursive':True} for v,p in roots],'future_roots_covered':['nvidia-native/capture-attempt-02/artifacts/measurements-tsvd-full-v1','apple/captured/tsvd-full-v1/runs','apple/captured/selectors-full/runs'],'no_fits_builds_or_verification_reruns':True,'gate_source_sha256':hashlib.sha256((REPO/'tools/af_quality.py').read_bytes()).hexdigest(),'gate_sources':['tools/af_quality.py rel_tol=1e-3 abs_tol=1e-6 and recorded metric directions','CLAUDE.md FAST main and best-opponent quality rule','tools/af_board_apply.py:quality_gate','CONTRIBUTING.md and docs/PERFORMANCE_ACCEPTANCE.md numeric-contract requirements'],'sample_policy':'One excludedwarmup and one scored execution perarm; no extra samples demanded','counts':dict(collections.Counter(r['quality_assessment'] for r in rows)),'completed_pairs_reviewed':len(rows),'baseline_comparison_counts':dict(collections.Counter(r.get('candidate_vs_baseline',{}).get('verdict','UNKNOWN') for r in rows)),'unfinished_or_failed_receipts':unfinished,'read_errors':read_errors,'rows':rows,'default_promotion':False}
target=OUT/'next-reg-resample-quality-review.json'
if target.exists():
 history=OUT/'snapshots';history.mkdir(exist_ok=True);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');shutil.copy2(target,history/('next-reg-resample-'+stamp+'.json'))
tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(target)
print(json.dumps({k:result[k] for k in ('completed_pairs_reviewed','counts','baseline_comparison_counts','read_errors')}))
